/*
 * ClusterMMU Sv39 page tables, guard pages and deferred reclaim.
 *
 * Three properties of this hardware shape the whole file.
 *
 * A Cluster has one page-table walker base and one ASID.  There is no
 * per-Context page table and no per-Context ASID (only 8 ASIDs exist and all
 * ports share GLOBAL_ASID), so Contexts cannot be isolated by the MMU.  They
 * are isolated by never handing two Contexts overlapping virtual addresses on
 * the one shared table.  The VA allocator is a bump pointer that never
 * recycles, which is what makes that guarantee cheap to hold.
 *
 * A fault does not abort the access.  The MMU substitutes the whole master
 * address with a per-class redirect base and the access completes there.
 * There are no fault-status or fault-address registers, so the only way to
 * observe a fault is to point the redirect bases at pages filled with a known
 * pattern and inspect them afterwards.  That is what the guard pages are.
 *
 * Writing TLB_FLUSH while any AXI port still has traffic in flight hangs the
 * Cluster permanently, and there is no quiesce register to prevent it.  So
 * unmapping never flushes inline: it clears the PTEs, parks the backing
 * memory on a reclaim list, and the flush happens later from the one place
 * that knows the device is idle.  QEMU's flush is synchronous and cannot
 * reproduce the hang, so this ordering is not something a test will catch.
 *
 * The page tables live in device memory, which the Host cannot dereference.
 * Every table therefore keeps a Host-side shadow copy; writes go to the
 * shadow and are pushed to the device with npu_hal_copy_to_address().
 */

#include "service_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "platform_regs.h"

/* Sv39: three levels of 512 entries, 4 KiB pages. */
#define NPU_MMU_PTE_V           UINT64_C(0x001)
#define NPU_MMU_PTE_R           UINT64_C(0x002)
#define NPU_MMU_PTE_W           UINT64_C(0x004)
#define NPU_MMU_PTE_X           UINT64_C(0x008)
#define NPU_MMU_PTE_U           UINT64_C(0x010)
#define NPU_MMU_PTE_G           UINT64_C(0x020)
#define NPU_MMU_PTE_A           UINT64_C(0x040)
#define NPU_MMU_PTE_D           UINT64_C(0x080)

#define NPU_MMU_PTE_PPN_SHIFT   10U
#define NPU_MMU_PTE_PER_TABLE   512U
#define NPU_MMU_VPN_BITS        9U
#define NPU_MMU_LEVELS          3U
#define NPU_MMU_TABLE_BYTES     (NPU_MMU_PTE_PER_TABLE * sizeof(uint64_t))

/*
 * Virtual addresses handed to Contexts.  The window starts above the
 * identity-mapped control and DRAM apertures so an identity mapping and a
 * Context mapping can never collide, and ends at the Sv39 limit.
 */
#define NPU_MMU_VA_BASE         UINT64_C(0x0000000100000000)
#define NPU_MMU_VA_TOP          UINT64_C(0x0000004000000000)

/* Guard-page fill: recognisable, and encodes its own offset so a corrupted
 * word tells us which VA[11:0] the faulting access carried. */
#define NPU_MMU_POISON_BASE     UINT64_C(0xdead000000000000)

static uint64_t npu_mmu_poison_word(uint64_t offset)
{
    return NPU_MMU_POISON_BASE | offset;
}

static uint32_t npu_mmu_vpn(uint64_t va, unsigned level)
{
    return (uint32_t)((va >> (DAYAN_MMU_PAGE_SHIFT +
                              level * NPU_MMU_VPN_BITS)) &
                      (NPU_MMU_PTE_PER_TABLE - 1U));
}

/*
 * A and D are preset rather than left to the hardware.  The ClusterMMU does
 * not update them, so a leaf with A clear would fault forever.
 */
static uint64_t npu_mmu_pte_make(uint64_t pa, uint32_t prot, int leaf)
{
    uint64_t pte = NPU_MMU_PTE_V |
                   ((pa >> DAYAN_MMU_PAGE_SHIFT) << NPU_MMU_PTE_PPN_SHIFT);

    if (!leaf)
        return pte;
    pte |= NPU_MMU_PTE_A | NPU_MMU_PTE_D | NPU_MMU_PTE_U | NPU_MMU_PTE_G;
    if (prot & NPU_SERVICE_MMU_READ)
        pte |= NPU_MMU_PTE_R;
    if (prot & NPU_SERVICE_MMU_WRITE)
        pte |= NPU_MMU_PTE_W;
    if (prot & NPU_SERVICE_MMU_EXEC)
        pte |= NPU_MMU_PTE_X;
    return pte;
}

static uint64_t npu_mmu_pte_pa(uint64_t pte)
{
    return (pte >> NPU_MMU_PTE_PPN_SHIFT) << DAYAN_MMU_PAGE_SHIFT;
}

/* ---------------------------------------------------------------- */
/* Page tables                                                       */
/* ---------------------------------------------------------------- */

/*
 * Push one shadow entry to the device copy of the table.  Tables are written
 * a word at a time rather than wholesale: a full-table push would rewrite
 * entries the walker may be reading concurrently.
 */
static int npu_mmu_push_entry(npu_service_device_t *device,
                              uint32_t cluster,
                              const struct npu_mmu_table *table,
                              uint32_t index)
{
    return npu_hal_copy_to_address(device->hal, cluster,
                                   table->device_address +
                                   index * sizeof(uint64_t),
                                   &table->entries[index],
                                   sizeof(table->entries[index]));
}

/*
 * Reserve one page of Device memory for MMU use, taken from the top of the
 * Normal region and working downwards.
 *
 * Plain allocation cannot be used here.  npu_service_mmu_init() runs before
 * BL1 is loaded, and the firmware is reserved at fixed addresses starting at
 * the bottom of Normal memory (its ELF entry is 0x80000000).  An ordinary
 * allocation hands out the lowest free address, which is exactly where the
 * firmware has to go, so page tables placed that way make the later BL1
 * reservation fail with -EEXIST.
 */
static int npu_mmu_reserve_page(npu_service_device_t *device, uint32_t cluster,
                                uint64_t size,
                                npu_hal_allocation_t **allocation_out)
{
    struct npu_mmu *mmu = &device->mmu;
    npu_hal_memory_request_t request;
    int status = -ENOMEM;

    if (!mmu->pool_cursor) {
        /* First use: find the top of the Normal region that is visible to
         * every Cluster, and grow downwards from there. */
        uint32_t index;

        for (index = 0; index < device->hal_info.memory_region_count;
             ++index) {
            npu_hal_memory_region_t region;

            status = npu_hal_get_memory_region(device->hal, index, &region);
            if (status < 0)
                return status;
            if (region.memory_class != NPU_HAL_MEMORY_NORMAL ||
                region.scope != NPU_HAL_MEMORY_DEVICE)
                continue;
            mmu->pool_cursor = region.base + region.size;
            mmu->pool_limit = region.base;
            break;
        }
        if (!mmu->pool_cursor)
            return -ENODEV;
    }
    if (mmu->pool_cursor - mmu->pool_limit < size)
        return -ENOMEM;
    mmu->pool_cursor -= size;

    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_NORMAL;
    request.usage = NPU_HAL_MEMORY_SHARED;
    request.owner = UINT32_MAX;
    request.scope = NPU_HAL_MEMORY_DEVICE;
    request.location.cluster = NPU_HAL_ANY_CLUSTER;
    request.location.core = NPU_HAL_ANY_CORE;
    request.size = size;
    request.alignment = DAYAN_MMU_PAGE_SIZE;
    request.fixed_address = mmu->pool_cursor;
    (void)cluster;
    return npu_hal_memory_reserve(device->hal, &request, allocation_out);
}

static int npu_mmu_table_alloc(npu_service_device_t *device, uint32_t cluster,
                               struct npu_mmu_table **table_out)
{
    struct npu_mmu_table *table;
    npu_hal_allocation_t *allocation = 0;
    npu_hal_allocation_info_t info;
    int status;

    *table_out = 0;
    table = calloc(1, sizeof(*table));
    if (!table)
        return -ENOMEM;
    table->entries = calloc(NPU_MMU_PTE_PER_TABLE, sizeof(*table->entries));
    if (!table->entries) {
        free(table);
        return -ENOMEM;
    }
    table->children = calloc(NPU_MMU_PTE_PER_TABLE, sizeof(*table->children));
    if (!table->children) {
        free(table->entries);
        free(table);
        return -ENOMEM;
    }
    /* A page table must be page-aligned for its PPN to be exact. */
    status = npu_mmu_reserve_page(device, cluster, NPU_MMU_TABLE_BYTES,
                                  &allocation);
    if (status < 0)
        goto fail;
    status = npu_hal_memory_info(device->hal, allocation, &info);
    if (status < 0)
        goto fail_free;
    table->allocation = allocation;
    table->device_address = info.address;
    /* The walker must never see stale bytes as valid entries. */
    status = npu_hal_copy_to_address(device->hal, cluster,
                                     table->device_address, table->entries,
                                     NPU_MMU_TABLE_BYTES);
    if (status < 0)
        goto fail_free;
    *table_out = table;
    return 0;

fail_free:
    (void)npu_hal_memory_free(device->hal, allocation);
fail:
    free(table->children);
    free(table->entries);
    free(table);
    return status;
}

static void npu_mmu_table_free(npu_service_device_t *device,
                               struct npu_mmu_table *table)
{
    uint32_t index;

    if (!table)
        return;
    for (index = 0; index < NPU_MMU_PTE_PER_TABLE; ++index)
        npu_mmu_table_free(device, table->children[index]);
    if (table->allocation)
        (void)npu_hal_memory_free(device->hal, table->allocation);
    free(table->children);
    free(table->entries);
    free(table);
}

/*
 * Walk to the leaf slot for @va, optionally creating interior tables.
 * Returns the owning table and the leaf index rather than a pointer, because
 * every entry write has to be mirrored to device memory and so needs to know
 * which table it belongs to.
 */
static int npu_mmu_walk(npu_service_device_t *device, uint32_t cluster,
                        uint64_t va, int create,
                        struct npu_mmu_table **table_out, uint32_t *index_out)
{
    struct npu_mmu *mmu = &device->mmu;
    struct npu_mmu_table *table = mmu->root[cluster];
    unsigned level;

    for (level = NPU_MMU_LEVELS - 1U; level > 0U; --level) {
        uint32_t index = npu_mmu_vpn(va, level);
        struct npu_mmu_table *child = table->children[index];

        if (!child) {
            int status;

            if (!create)
                return -ENOENT;
            status = npu_mmu_table_alloc(device, cluster, &child);
            if (status < 0)
                return status;
            table->children[index] = child;
            /*
             * The child's contents are already in device memory (zeroed by
             * npu_mmu_table_alloc) before this entry makes it reachable, so
             * the walker can never follow this pointer into uninitialised
             * memory.
             */
            table->entries[index] = npu_mmu_pte_make(child->device_address,
                                                     0, 0);
            status = npu_mmu_push_entry(device, cluster, table, index);
            if (status < 0) {
                /* Retract the link: an entry the device never received must
                 * not be left claimed in the shadow. */
                table->entries[index] = 0;
                table->children[index] = 0;
                npu_mmu_table_free(device, child);
                return status;
            }
        }
        table = table->children[index];
    }
    *table_out = table;
    *index_out = npu_mmu_vpn(va, 0);
    return 0;
}

/* ---------------------------------------------------------------- */
/* Init and teardown                                                 */
/* ---------------------------------------------------------------- */

static int npu_mmu_guard_init(npu_service_device_t *device, uint32_t cluster);

/*
 * Identity-map the apertures whose addresses are baked into the firmware.
 *
 * device/linker.ld places every section at CONFIG_RAM_BASE, so every pointer
 * inside the firmware is that address; giving the image a non-identity
 * mapping would mean rewriting the linker script and the firmware's whole
 * address model.  The control window, URAM and the coherent window are
 * identity-mapped for the same reason - the firmware reaches them through
 * fixed constants in platform_regs.h and config.h.
 */
static int npu_mmu_map_fixed_apertures(npu_service_device_t *device,
                                       uint32_t cluster)
{
    uint32_t region_count = device->hal_info.memory_region_count;
    uint32_t index;
    int status;

    /*
     * Control space: mailboxes, IRQ, the MMU's own register page, the
     * coherent (atomic cache) window and the CLINT.  This stops below URAM
     * at 0x800000; URAM comes from the HAL region list below, and widening
     * this to cover it would make the two overlap and fail with -EEXIST.
     *
     * The BL0 ROM in the middle of it is the one place below URAM the Device
     * fetches from, and it does so after translation is on: the MMU is
     * activated before BL1 is loaded, so BL0's wake-up from wfi executes
     * against these tables.  Without X here that fetch is an exec-permission
     * fault redirected to a guard page.  (QEMU's CPU TLB caches the ROM page
     * before activation and is not flushed by MMU register writes, which is
     * the only reason the omission ever passed there.)
     */
    status = npu_service_mmu_map(device, cluster, 0, 0,
                                 NPU_SERVICE_MMU_ROM_BASE,
                                 NPU_SERVICE_MMU_READ |
                                 NPU_SERVICE_MMU_WRITE);
    if (status < 0)
        return status;
    status = npu_service_mmu_map(device, cluster, NPU_SERVICE_MMU_ROM_BASE,
                                 NPU_SERVICE_MMU_ROM_BASE,
                                 NPU_SERVICE_MMU_ROM_SIZE,
                                 NPU_SERVICE_MMU_READ |
                                 NPU_SERVICE_MMU_EXEC);
    if (status < 0)
        return status;
    status = npu_service_mmu_map(device, cluster,
                                 NPU_SERVICE_MMU_ROM_BASE +
                                 NPU_SERVICE_MMU_ROM_SIZE,
                                 NPU_SERVICE_MMU_ROM_BASE +
                                 NPU_SERVICE_MMU_ROM_SIZE,
                                 NPU_SERVICE_MMU_URAM_BASE -
                                 NPU_SERVICE_MMU_ROM_BASE -
                                 NPU_SERVICE_MMU_ROM_SIZE,
                                 NPU_SERVICE_MMU_READ |
                                 NPU_SERVICE_MMU_WRITE);
    if (status < 0)
        return status;

    /*
     * Everything else comes from the HAL's own region list rather than
     * hardcoded constants, because the Normal region's size is configurable
     * (--normal-memory-size-mib).  Hardcoding it would leave any allocation
     * past the assumed end unmapped, and an unmapped address does not fail
     * visibly here - it gets redirected at run time.
     */
    for (index = 0; index < region_count; ++index) {
        npu_hal_memory_region_t region;
        uint64_t base;
        uint64_t size;
        uint32_t prot = NPU_SERVICE_MMU_READ | NPU_SERVICE_MMU_WRITE;

        status = npu_hal_get_memory_region(device->hal, index, &region);
        if (status < 0)
            return status;
        /* Regions below URAM are already covered by the control window. */
        if (region.base < NPU_SERVICE_MMU_URAM_BASE)
            continue;
        /* A Cluster- or Core-scoped region belongs only to its own Cluster. */
        if (region.scope != NPU_HAL_MEMORY_DEVICE &&
            region.cluster != cluster)
            continue;
        base = region.base & ~(uint64_t)DAYAN_MMU_PAGE_MASK;
        size = (region.base + region.size + DAYAN_MMU_PAGE_MASK) &
               ~(uint64_t)DAYAN_MMU_PAGE_MASK;
        size -= base;
        /* The firmware image lives in Normal memory and is executed. */
        if (region.memory_class == NPU_HAL_MEMORY_NORMAL)
            prot |= NPU_SERVICE_MMU_EXEC;
        status = npu_service_mmu_map(device, cluster, base, base, size, prot);
        /*
         * Regions can repeat across Cores that share an aperture; an
         * existing identical mapping is not an error here.
         */
        if (status == -EEXIST)
            continue;
        if (status < 0)
            return status;
    }
    return 0;
}

/*
 * Bring one Cluster's MMU up.  The order is the safety argument and must not
 * be rearranged: the page table has to be complete and PTW_BASE live before
 * FixMap is switched off, or the first access after the switch walks an empty
 * table.  N_FIX_MAP_EN is therefore written last, by its own function.
 */
static int npu_mmu_enable_cluster(npu_service_device_t *device,
                                  uint32_t cluster)
{
    struct npu_mmu *mmu = &device->mmu;
    int status;

    status = npu_mmu_table_alloc(device, cluster, &mmu->root[cluster]);
    if (status < 0)
        return status;
    status = npu_hal_mmu_write64(device->hal, cluster, DAYAN_MMU_PTW_BASE_LO,
                                 mmu->root[cluster]->device_address);
    if (status < 0)
        return status;
    /* One shared ASID; Contexts are separated by VA range, not by ASID. */
    status = npu_hal_mmu_write_reg(device->hal, cluster,
                                   DAYAN_MMU_GLOBAL_ASID, 0);
    if (status < 0)
        return status;
    /*
     * Apertures before guard pages: the guard pages are allocated out of
     * DRAM, which these mappings cover, and npu_mmu_guard_init() relies on
     * that identity mapping already existing rather than adding its own.
     */
    status = npu_mmu_map_fixed_apertures(device, cluster);
    if (status < 0)
        return status;
    return npu_mmu_guard_init(device, cluster);
}

/* The single point where translation actually starts. */
static int npu_mmu_activate(npu_service_device_t *device, uint32_t cluster)
{
    return npu_hal_mmu_write_reg(device->hal, cluster,
                                 DAYAN_MMU_N_FIX_MAP_EN,
                                 DAYAN_MMU_N_FIX_MAP_EN_BIT);
}

static int npu_mmu_deactivate(npu_service_device_t *device, uint32_t cluster)
{
    return npu_hal_mmu_write_reg(device->hal, cluster,
                                 DAYAN_MMU_N_FIX_MAP_EN, 0);
}

int npu_service_mmu_init(npu_service_device_t *device)
{
    struct npu_mmu *mmu = &device->mmu;
    uint32_t clusters = device->hal_info.cluster_count;
    uint32_t cluster;
    int status;

    pthread_mutex_init(&mmu->lock, 0);
    mmu->cluster_count = clusters;
    mmu->va_next = NPU_MMU_VA_BASE;
    mmu->va_limit = NPU_MMU_VA_TOP;

    /*
     * Off by default.  FixMap's reset state is a full-range identity map, so
     * leaving the MMU alone keeps every existing address working unchanged;
     * that is what makes this opt-in rather than a flag day.
     */
    if (!device->options.enable_mmu)
        return 0;

    for (cluster = 0; cluster < clusters; ++cluster) {
        int present = 0;

        status = npu_hal_mmu_present(device->hal, cluster, &present);
        if (status < 0)
            return status;
        if (!present)
            /* Asked for, but not there: identity addressing stays. */
            return -ENODEV;
    }

    mmu->root = calloc(clusters, sizeof(*mmu->root));
    mmu->clusters = calloc(clusters, sizeof(*mmu->clusters));
    if (!mmu->root || !mmu->clusters) {
        status = -ENOMEM;
        goto fail;
    }
    /* Mapping helpers check mmu->enabled, so it must be set before the
     * aperture mappings below are built. */
    mmu->enabled = 1;

    for (cluster = 0; cluster < clusters; ++cluster) {
        status = npu_mmu_enable_cluster(device, cluster);
        if (status < 0)
            goto fail;
    }
    for (cluster = 0; cluster < clusters; ++cluster) {
        status = npu_mmu_activate(device, cluster);
        if (status < 0)
            goto fail;
    }
    return 0;

fail:
    npu_service_mmu_fini(device);
    return status;
}

/*
 * Teardown in the reverse order: return to identity addressing first, so
 * nothing is translating while the tables are dismantled.
 */
void npu_service_mmu_fini(npu_service_device_t *device)
{
    struct npu_mmu *mmu = &device->mmu;
    uint32_t cluster;

    if (!mmu->enabled)
        goto out;

    for (cluster = 0; cluster < mmu->cluster_count; ++cluster)
        (void)npu_mmu_deactivate(device, cluster);
    /* FixMap is back on, so no TLB entry can be consulted; the flush would be
     * redundant and this path may run with launches unaccounted for. */
    (void)npu_service_mmu_run_reclaim(device, 1);

    for (cluster = 0; cluster < mmu->cluster_count; ++cluster) {
        unsigned class_index;

        if (mmu->clusters) {
            struct npu_mmu_cluster *state = &mmu->clusters[cluster];

            for (class_index = 0; class_index < DAYAN_MMU_REDIR_CLASS_COUNT;
                 ++class_index)
                if (state->guard_allocation[class_index])
                    (void)npu_hal_memory_free(
                        device->hal, state->guard_allocation[class_index]);
        }
        if (mmu->root)
            npu_mmu_table_free(device, mmu->root[cluster]);
    }
    mmu->enabled = 0;

out:
    free(mmu->clusters);
    free(mmu->root);
    mmu->clusters = 0;
    mmu->root = 0;
    pthread_mutex_destroy(&mmu->lock);
}

/* ---------------------------------------------------------------- */
/* Virtual address space                                             */
/* ---------------------------------------------------------------- */

/*
 * Hand out a non-overlapping virtual range.  The allocator is a bump pointer
 * and ranges are never recycled.  That wastes a 39-bit space only in theory,
 * and it removes the entire class of bug where a Context is given a range
 * another Context still has live TLB entries for - which matters here because
 * one page table and one ASID are shared by everything.
 */
int npu_service_mmu_reserve_va(npu_service_device_t *device, uint64_t size,
                               uint64_t *base_out)
{
    struct npu_mmu *mmu;
    uint64_t aligned;
    int status = 0;

    if (!device || !size || !base_out)
        return -EINVAL;
    mmu = &device->mmu;
    if (!mmu->enabled) {
        /* Identity addressing: callers use physical addresses directly. */
        *base_out = 0;
        return -ENOTSUP;
    }
    aligned = (size + DAYAN_MMU_PAGE_MASK) & ~(uint64_t)DAYAN_MMU_PAGE_MASK;
    if (aligned < size)
        return -EOVERFLOW;
    pthread_mutex_lock(&mmu->lock);
    if (aligned > mmu->va_limit - mmu->va_next)
        status = -ENOSPC;
    else {
        *base_out = mmu->va_next;
        mmu->va_next += aligned;
    }
    pthread_mutex_unlock(&mmu->lock);
    return status;
}

/*
 * Deliberately does not return the range to the pool; see the comment on
 * npu_service_mmu_reserve_va().  Kept as a call site so the ownership
 * transfer is explicit at the point a Context goes away.
 */
void npu_service_mmu_release_va(npu_service_device_t *device, uint64_t base,
                                uint64_t size)
{
    (void)device;
    (void)base;
    (void)size;
}

/* ---------------------------------------------------------------- */
/* Mapping                                                           */
/* ---------------------------------------------------------------- */

int npu_service_mmu_map(npu_service_device_t *device, uint32_t cluster,
                        uint64_t va, uint64_t pa, uint64_t size, uint32_t prot)
{
    struct npu_mmu *mmu;
    uint64_t offset;
    int status = 0;

    if (!device || cluster >= device->hal_info.cluster_count)
        return -EINVAL;
    mmu = &device->mmu;
    if (!mmu->enabled)
        return 0;
    if ((va & DAYAN_MMU_PAGE_MASK) || (pa & DAYAN_MMU_PAGE_MASK) ||
        (size & DAYAN_MMU_PAGE_MASK) || !size)
        return -EINVAL;
    if (va > UINT64_MAX - size || pa > UINT64_MAX - size)
        return -EOVERFLOW;
    if ((pa + size - 1U) >> DAYAN_MMU_PA_BITS)
        return -EINVAL;

    pthread_mutex_lock(&mmu->lock);
    for (offset = 0; offset < size; offset += DAYAN_MMU_PAGE_SIZE) {
        struct npu_mmu_table *table = 0;
        uint32_t index = 0;

        status = npu_mmu_walk(device, cluster, va + offset, 1, &table, &index);
        if (status < 0)
            break;
        if (table->entries[index] & NPU_MMU_PTE_V) {
            status = -EEXIST;
            break;
        }
        table->entries[index] = npu_mmu_pte_make(pa + offset, prot, 1);
        status = npu_mmu_push_entry(device, cluster, table, index);
        if (status < 0) {
            table->entries[index] = 0;
            break;
        }
    }
    if (status < 0) {
        /*
         * Take back the pages this call did install.  The caller frees the
         * backing memory on failure, so a leaf left behind here would point
         * at memory that is about to belong to someone else.  Nothing can
         * have translated these pages yet, so no flush is needed.
         */
        uint64_t undo;

        for (undo = 0; undo < offset; undo += DAYAN_MMU_PAGE_SIZE) {
            struct npu_mmu_table *table = 0;
            uint32_t index = 0;

            if (npu_mmu_walk(device, cluster, va + undo, 0, &table, &index) < 0)
                continue;
            table->entries[index] = 0;
            (void)npu_mmu_push_entry(device, cluster, table, index);
        }
    }
    pthread_mutex_unlock(&mmu->lock);

    /*
     * No flush on map.  A newly valid entry cannot be stale in the TLB,
     * because nothing has translated that address before, and flushing is the
     * single most dangerous thing this driver does.
     */
    return status;
}

/*
 * Clear PTEs without flushing.  Callers must route the freeing of the backing
 * memory through npu_service_mmu_defer_reclaim(), because until a flush
 * happens the TLB may still translate these addresses - and the flush cannot
 * be issued here (see the file header).
 */
int npu_service_mmu_unmap(npu_service_device_t *device, uint32_t cluster,
                          uint64_t va, uint64_t size)
{
    struct npu_mmu *mmu;
    uint64_t offset;
    int status = 0;

    if (!device || cluster >= device->hal_info.cluster_count)
        return -EINVAL;
    mmu = &device->mmu;
    if (!mmu->enabled)
        return 0;
    if ((va & DAYAN_MMU_PAGE_MASK) || (size & DAYAN_MMU_PAGE_MASK))
        return -EINVAL;

    pthread_mutex_lock(&mmu->lock);
    for (offset = 0; offset < size; offset += DAYAN_MMU_PAGE_SIZE) {
        struct npu_mmu_table *table = 0;
        uint32_t index = 0;
        int walked = npu_mmu_walk(device, cluster, va + offset, 0, &table,
                                  &index);

        /* An absent interior table means nothing was mapped here. */
        if (walked == -ENOENT)
            continue;
        if (walked < 0) {
            status = walked;
            break;
        }
        if (!(table->entries[index] & NPU_MMU_PTE_V))
            continue;
        table->entries[index] = 0;
        status = npu_mmu_push_entry(device, cluster, table, index);
        if (status < 0)
            break;
    }
    pthread_mutex_unlock(&mmu->lock);
    return status;
}

/*
 * Narrowing permissions does need a flush, because a cached entry with the
 * old wider permissions would still be honoured.  The flush is the caller's,
 * for the same reason unmap does not do it: it may only be issued when the
 * device is idle.
 */
int npu_service_mmu_protect(npu_service_device_t *device, uint32_t cluster,
                            uint64_t va, uint64_t size, uint32_t prot)
{
    struct npu_mmu *mmu;
    uint64_t offset;
    int status = 0;

    if (!device || cluster >= device->hal_info.cluster_count)
        return -EINVAL;
    mmu = &device->mmu;
    if (!mmu->enabled)
        return 0;
    if ((va & DAYAN_MMU_PAGE_MASK) || (size & DAYAN_MMU_PAGE_MASK))
        return -EINVAL;

    pthread_mutex_lock(&mmu->lock);
    for (offset = 0; offset < size; offset += DAYAN_MMU_PAGE_SIZE) {
        struct npu_mmu_table *table = 0;
        uint32_t index = 0;
        uint64_t previous;

        status = npu_mmu_walk(device, cluster, va + offset, 0, &table, &index);
        if (status < 0)
            break;
        if (!(table->entries[index] & NPU_MMU_PTE_V)) {
            status = -ENOENT;
            break;
        }
        previous = table->entries[index];
        table->entries[index] =
            npu_mmu_pte_make(npu_mmu_pte_pa(previous), prot, 1);
        status = npu_mmu_push_entry(device, cluster, table, index);
        if (status < 0) {
            table->entries[index] = previous;
            break;
        }
    }
    pthread_mutex_unlock(&mmu->lock);
    return status;
}

/* ---------------------------------------------------------------- */
/* Deferred reclaim                                                  */
/* ---------------------------------------------------------------- */

/*
 * Two-phase release.  Phase one (here) has already cleared the PTEs; the
 * backing allocation is parked, not freed.  Phase two
 * (npu_service_mmu_run_reclaim) flushes the TLB once for the whole batch and
 * then frees, and it only runs when the device has no launches in flight.
 *
 * Parking rather than freeing is what makes the window safe: if the device
 * still holds a stale TLB entry for one of these pages, it reads the original
 * allocation, which nothing else owns yet.  Freeing first would let a later
 * allocation reuse that memory while a stale entry still pointed at it.
 */
int npu_service_mmu_defer_reclaim(npu_service_device_t *device,
                                  uint32_t cluster,
                                  npu_hal_allocation_t *allocation)
{
    struct npu_mmu *mmu;
    struct npu_mmu_reclaim *item;

    if (!device || !allocation)
        return -EINVAL;
    mmu = &device->mmu;
    if (!mmu->enabled)
        return npu_hal_memory_free(device->hal, allocation);
    item = calloc(1, sizeof(*item));
    if (!item)
        return -ENOMEM;
    item->allocation = allocation;
    item->cluster = cluster;
    pthread_mutex_lock(&mmu->lock);
    item->next = mmu->reclaim;
    mmu->reclaim = item;
    pthread_mutex_unlock(&mmu->lock);
    return 0;
}

/*
 * Drain the reclaim list.  Callers must hold no device lock and must have
 * established that no launch is in flight - see npu_hal_mmu_flush()'s
 * contract.  @force skips the flush entirely and is only for teardown paths
 * where the MMU is about to be switched off anyway.
 *
 * A flush failure leaves the batch parked rather than freeing it: freeing
 * memory whose TLB entries may still be live is the one outcome worse than
 * leaking it until the next attempt.
 */
int npu_service_mmu_run_reclaim(npu_service_device_t *device, int force)
{
    struct npu_mmu *mmu;
    struct npu_mmu_reclaim *batch;
    struct npu_mmu_reclaim *item;
    uint32_t cluster;
    int status = 0;

    if (!device)
        return -EINVAL;
    mmu = &device->mmu;
    if (!mmu->enabled)
        return 0;

    pthread_mutex_lock(&mmu->lock);
    batch = mmu->reclaim;
    mmu->reclaim = 0;
    pthread_mutex_unlock(&mmu->lock);
    if (!batch)
        return 0;

    if (!force) {
        /*
         * Every Cluster, not just the one recorded on the entry.  A Buffer's
         * mapping is installed in every Cluster's table, so a stale entry can
         * survive anywhere; flushing only the originating Cluster would let
         * another Cluster keep translating an address whose backing memory is
         * about to be handed to someone else.
         */
        for (cluster = 0; cluster < mmu->cluster_count; ++cluster) {
            status = npu_hal_mmu_flush(device->hal, cluster, 0, 0,
                                       device->options.default_timeout_ms);
            if (status < 0)
                break;
        }
    }

    if (status < 0) {
        /* Put the batch back; nothing has been freed. */
        pthread_mutex_lock(&mmu->lock);
        for (item = batch; item && item->next; item = item->next)
            ;
        if (item) {
            item->next = mmu->reclaim;
            mmu->reclaim = batch;
        }
        pthread_mutex_unlock(&mmu->lock);
        return status;
    }

    while (batch) {
        struct npu_mmu_reclaim *next = batch->next;
        int freed = npu_hal_memory_free(device->hal, batch->allocation);

        if (freed < 0 && status == 0)
            status = freed;
        free(batch);
        batch = next;
    }
    return status;
}

/* ---------------------------------------------------------------- */
/* Guard pages                                                       */
/* ---------------------------------------------------------------- */

/*
 * One page per fault class, each filled with a self-describing pattern and
 * pointed to by that class's redirect base.  ERROR_SLV_OFF_MODE is set so the
 * redirect keeps VA[11:0], which means a corrupted word's offset within the
 * page recovers the low 12 bits of the faulting address.
 *
 * This is sampling, not reporting: it cannot give the VA's upper bits and
 * cannot count occurrences.  The hardware has no fault-status register, so
 * this is the ceiling on what is observable.
 */
static int npu_mmu_guard_fill(npu_service_device_t *device, uint32_t cluster,
                              uint64_t address)
{
    uint64_t pattern[DAYAN_MMU_PAGE_SIZE / sizeof(uint64_t)];
    size_t word;

    for (word = 0; word < sizeof(pattern) / sizeof(pattern[0]); ++word)
        pattern[word] = npu_mmu_poison_word(word * sizeof(uint64_t));
    return npu_hal_copy_to_address(device->hal, cluster, address, pattern,
                                   sizeof(pattern));
}

static int npu_mmu_guard_init(npu_service_device_t *device, uint32_t cluster)
{
    struct npu_mmu *mmu = &device->mmu;
    struct npu_mmu_cluster *state = &mmu->clusters[cluster];
    unsigned class_index;
    int status;

    for (class_index = 0; class_index < DAYAN_MMU_REDIR_CLASS_COUNT;
         ++class_index) {
        npu_hal_allocation_t *allocation = 0;
        npu_hal_allocation_info_t info;

        /* Also from the top-down pool, for the same reason page tables are:
         * the firmware needs the bottom of Normal memory. */
        status = npu_mmu_reserve_page(device, cluster, DAYAN_MMU_PAGE_SIZE,
                                      &allocation);
        if (status < 0)
            return status;
        status = npu_hal_memory_info(device->hal, allocation, &info);
        if (status < 0) {
            (void)npu_hal_memory_free(device->hal, allocation);
            return status;
        }
        state->guard_allocation[class_index] = allocation;
        state->guard_address[class_index] = info.address;
        status = npu_mmu_guard_fill(device, cluster, info.address);
        if (status < 0)
            return status;
        /* Redirect bases must be 4 KiB aligned; the allocation above
         * guarantees it, and the HAL rechecks. */
        status = npu_hal_mmu_write64(device->hal, cluster,
                                     DAYAN_MMU_REDIR_LO(class_index),
                                     info.address);
        if (status < 0)
            return status;
        /*
         * No mapping is installed here.  Guard pages come out of DRAM, which
         * the fixed apertures already identity-map, so mapping them again
         * would collide with -EEXIST.  Redirects are physical addresses
         * substituted by the hardware and do not need a translation of their
         * own; the identity mapping only matters so the Host and Device agree
         * on where the page is.
         */
    }

    status = npu_hal_mmu_write_reg(device->hal, cluster,
                                   DAYAN_MMU_ERROR_SLV_OFF_MODE,
                                   DAYAN_MMU_ERROR_SLV_OFF_VA);
    if (status < 0)
        return status;
    /* The error slave reports its own misconfiguration here; a set bit means
     * the redirect programming above was not accepted. */
    {
        uint32_t cfg_err = 0;

        status = npu_hal_mmu_read_reg(device->hal, cluster,
                                      DAYAN_MMU_ERROR_SLV_CFG_ERR, &cfg_err);
        if (status < 0)
            return status;
        if (cfg_err & DAYAN_MMU_ERROR_SLV_CFG_ERR_BIT) {
            (void)npu_hal_mmu_write_reg(device->hal, cluster,
                                        DAYAN_MMU_ERROR_SLV_CFG_CLR,
                                        DAYAN_MMU_ERROR_SLV_CFG_ERR_BIT);
            return -EIO;
        }
    }
    return 0;
}

/*
 * Report whether any guard page has been disturbed.  @fault_class and
 * @fault_offset describe the first disturbance found; the offset is the
 * faulting access's VA[11:0].
 */
int npu_service_mmu_check_faults(npu_service_device_t *device,
                                 uint32_t cluster, uint32_t *fault_class,
                                 uint32_t *fault_offset)
{
    struct npu_mmu *mmu;
    struct npu_mmu_cluster *state;
    uint64_t observed[DAYAN_MMU_PAGE_SIZE / sizeof(uint64_t)];
    unsigned class_index;

    if (!device || cluster >= device->hal_info.cluster_count)
        return -EINVAL;
    mmu = &device->mmu;
    if (!mmu->enabled)
        return 0;
    state = &mmu->clusters[cluster];
    for (class_index = 0; class_index < DAYAN_MMU_REDIR_CLASS_COUNT;
         ++class_index) {
        size_t word;
        int status;

        if (!state->guard_allocation[class_index])
            continue;
        status = npu_hal_copy_from_address(device->hal, cluster, observed,
                                          state->guard_address[class_index],
                                          sizeof(observed));
        if (status < 0)
            return status;
        for (word = 0; word < sizeof(observed) / sizeof(observed[0]); ++word) {
            if (observed[word] == npu_mmu_poison_word(word * sizeof(uint64_t)))
                continue;
            if (fault_class)
                *fault_class = class_index;
            if (fault_offset)
                *fault_offset = (uint32_t)(word * sizeof(uint64_t));
            /* Restore so the next check starts from a known state. */
            (void)npu_mmu_guard_fill(device, cluster,
                                     state->guard_address[class_index]);
            return 1;
        }
    }
    return 0;
}
