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
 * Top of the virtual addresses handed to Contexts: the Sv39 limit.  The base is
 * not a constant - it sits just above the firmware window, whose size the HAL
 * decides at run time, so npu_service_mmu_init() computes it.
 */
#define NPU_MMU_VA_TOP          UINT64_C(0x0000004000000000)

/* One level-1 Sv39 entry spans 2 MiB; a level-1 leaf maps all of it. */
#define NPU_MMU_LARGE_LEVEL     1U

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
 * One page of Device memory for MMU use - page tables and guard pages - taken
 * from the top of the firmware pool and working downwards.
 *
 * Both halves of that matter.  It has to come from the firmware pool, because
 * that is the range the firmware window maps: a page allocated above the split
 * would be unreachable by the Device and could collide with a Buffer.  And it
 * has to come from the top of the pool, because the firmware's position
 * *within* the pool is fixed even though the pool's own base need not be:
 * BL1's segments are reserved at pool_base + (link address - CONFIG_RAM_BASE),
 * which starts at pool_base itself.  An ordinary first-fit allocation hands
 * out pool_base and makes the later BL1 reservation fail with -EEXIST.
 *
 * The linear offset removed the need to guess where the firmware pool sits.
 * It did not remove the firmware's fixed offset inside it, which is what this
 * reservation stays clear of.
 */
static int npu_mmu_reserve_page(npu_service_device_t *device, uint32_t cluster,
                                uint64_t size,
                                npu_hal_allocation_t **allocation_out)
{
    struct npu_mmu *mmu = &device->mmu;
    npu_hal_memory_request_t request;
    uint64_t address;
    int status;

    if (!mmu->fw_pool_size)
        return -ENODEV;
    if (!mmu->pool_cursor)
        mmu->pool_cursor = mmu->fw_pool_base + mmu->fw_pool_size;
    if (mmu->pool_cursor - mmu->fw_pool_base < size)
        return -ENOMEM;
    address = mmu->pool_cursor - size;

    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_NORMAL;
    request.usage = NPU_HAL_MEMORY_SHARED;
    request.owner = UINT32_MAX;
    request.scope = NPU_HAL_MEMORY_DEVICE;
    request.location.cluster = NPU_HAL_ANY_CLUSTER;
    request.location.core = NPU_HAL_ANY_CORE;
    request.size = size;
    request.alignment = DAYAN_MMU_PAGE_SIZE;
    request.fixed_address = address;
    (void)cluster;
    /* The cursor moves only once the range is actually ours.  Consuming it
     * first would retire a page on every failed attempt, so a caller that
     * retries would walk the cursor down to the firmware's own segments. */
    status = npu_hal_memory_reserve(device->hal, &request, allocation_out);
    if (status < 0)
        return status;
    mmu->pool_cursor = address;
    return 0;
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
 * Walk to the entry slot for @va at @stop_level, optionally creating interior
 * tables.  Returns the owning table and the index rather than a pointer,
 * because every entry write has to be mirrored to device memory and so needs
 * to know which table it belongs to.
 *
 * @stop_level 0 is a 4 KiB leaf, 1 a 2 MiB leaf.  A level-1 slot reached this
 * way must be empty before it is written as a leaf: an interior entry already
 * there means 4 KiB mappings exist underneath, and overwriting it would strand
 * them.  The callers check that (npu_mmu_map_firmware_window() refuses a slot
 * with V set), and a create-walk down *through* a level-1 leaf is refused here.
 */
static int npu_mmu_walk_level(npu_service_device_t *device, uint32_t cluster,
                              uint64_t va, int create, unsigned stop_level,
                              struct npu_mmu_table **table_out,
                              uint32_t *index_out)
{
    struct npu_mmu *mmu = &device->mmu;
    struct npu_mmu_table *table = mmu->root[cluster];
    unsigned level;

    for (level = NPU_MMU_LEVELS - 1U; level > stop_level; --level) {
        uint32_t index = npu_mmu_vpn(va, level);
        struct npu_mmu_table *child = table->children[index];

        if (!child) {
            int status;

            /*
             * A valid entry with no child is a leaf - a 2 MiB firmware-window
             * leaf, since those are the only ones installed above level 0.
             * Descending would mean replacing it with an interior entry and
             * unmapping the 2 MiB it covers, so refuse instead.  The firmware
             * window and the Buffer range are disjoint by construction, which
             * is why this is a guard and not a live path.
             */
            if (table->entries[index] & NPU_MMU_PTE_V)
                return -EEXIST;
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
    *index_out = npu_mmu_vpn(va, stop_level);
    return 0;
}

static int npu_mmu_walk(npu_service_device_t *device, uint32_t cluster,
                        uint64_t va, int create,
                        struct npu_mmu_table **table_out, uint32_t *index_out)
{
    return npu_mmu_walk_level(device, cluster, va, create, 0, table_out,
                              index_out);
}

/* ---------------------------------------------------------------- */
/* Init and teardown                                                 */
/* ---------------------------------------------------------------- */

static int npu_mmu_guard_init(npu_service_device_t *device, uint32_t cluster);

/*
 * Map the firmware window: one contiguous range at a fixed virtual base, whose
 * physical base the Host chose, with 2 MiB leaves.
 *
 *     pa = va - NPU_SERVICE_MMU_FW_VA_BASE + fw_pool_base
 *
 * The virtual base is not negotiable - device/linker.ld places every section
 * at CONFIG_RAM_BASE, so every pointer inside the firmware is that address and
 * changing it means rebuilding the firmware.  The physical base is free, and
 * that is what removes the ordering constraint an identity mapping had: the
 * whole window is installed here, at init, before BL1 exists, so where BL1's
 * segments land inside it never matters.
 *
 * Nothing below 2 GiB is mapped - not the control window, not the BL0 ROM, not
 * URAM.  The MMU bypasses that range before consulting a page table, so PTEs
 * there would describe translations the hardware never performs.  This is also
 * what retires the old exec-permission hazard on the ROM: BL0's wake-up from
 * wfi is a bypassed fetch, not a translated one.
 */
static int npu_mmu_map_firmware_window(npu_service_device_t *device,
                                       uint32_t cluster)
{
    struct npu_mmu *mmu = &device->mmu;
    uint64_t offset;
    int status = 0;

    if (!mmu->fw_pool_size)
        return -ENODEV;
    /* 2 MiB leaves need both ends 2 MiB aligned.  The HAL aligns the pool base
     * for exactly this; a size that is not a whole number of leaves would
     * leave a tail this loop cannot express, so it is rejected rather than
     * silently mapped short. */
    if ((mmu->fw_pool_base & (NPU_SERVICE_MMU_LARGE_SIZE - 1U)) ||
        (mmu->fw_pool_size & (NPU_SERVICE_MMU_LARGE_SIZE - 1U)))
        return -EINVAL;

    pthread_mutex_lock(&mmu->lock);
    for (offset = 0; offset < mmu->fw_pool_size;
         offset += NPU_SERVICE_MMU_LARGE_SIZE) {
        struct npu_mmu_table *table = 0;
        uint32_t index = 0;

        status = npu_mmu_walk_level(device, cluster,
                                    NPU_SERVICE_MMU_FW_VA_BASE + offset, 1,
                                    NPU_MMU_LARGE_LEVEL, &table, &index);
        if (status < 0)
            break;
        if (table->entries[index] & NPU_MMU_PTE_V) {
            status = -EEXIST;
            break;
        }
        table->entries[index] =
            npu_mmu_pte_make(mmu->fw_pool_base + offset,
                             NPU_SERVICE_MMU_READ | NPU_SERVICE_MMU_WRITE |
                             NPU_SERVICE_MMU_EXEC, 1);
        status = npu_mmu_push_entry(device, cluster, table, index);
        if (status < 0) {
            table->entries[index] = 0;
            break;
        }
    }
    pthread_mutex_unlock(&mmu->lock);
    /*
     * No undo on failure, unlike npu_service_mmu_map().  A partial firmware
     * window is not a leak to clean up: init fails, npu_service_mmu_fini()
     * runs, and the tables go away whole.  Leaving the installed leaves in
     * place also keeps the failure diagnosable.
     */
    return status;
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
     * The window before the guard pages: the guard pages are allocated out of
     * the firmware pool, and npu_mmu_guard_init() relies on this mapping
     * already covering them rather than adding one of its own.
     */
    status = npu_mmu_map_firmware_window(device, cluster);
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
    mmu->va_limit = NPU_MMU_VA_TOP;

    /*
     * Off by default.  FixMap's reset state is a full-range identity map, so
     * leaving the MMU alone keeps every existing address working unchanged;
     * that is what makes this opt-in rather than a flag day.
     */
    if (!device->options.enable_mmu)
        return 0;

    /*
     * The firmware pool decides both the size of the window and where the
     * Buffer virtual addresses can start, so it has to be known before any
     * mapping is built.
     *
     * -ENOENT is refused rather than tolerated.  It means the Normal region
     * could not be split, so the whole of it would be the firmware window -
     * which puts every Buffer's physical pages inside the window and gives them
     * a second, linear-offset virtual address.  That is precisely the aliasing
     * the split exists to prevent: an out-of-bounds Buffer access would land on
     * a legally mapped address instead of a guard page.  Refusing here turns an
     * unusably small --normal-memory-size-mib into a startup error instead of a
     * silently weaker memory model.
     */
    status = npu_hal_firmware_pool(device->hal, &mmu->fw_pool_base,
                                   &mmu->fw_pool_size);
    if (status < 0)
        return status;
    /*
     * Buffer virtual addresses begin one 2 MiB sentinel above the top of the
     * window, so an overrun off the end of the firmware window faults instead
     * of sliding into Buffer space.  Deriving the base from the pool size,
     * rather than fixing it at some round number, is what keeps the property
     * that every physical page has exactly one virtual address.
     */
    mmu->va_next = NPU_SERVICE_MMU_FW_VA_BASE + mmu->fw_pool_size +
                   NPU_SERVICE_MMU_VA_SENTINEL;
    if (mmu->va_next >= mmu->va_limit)
        return -ENOSPC;

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
    /*
     * Every allocation out of the pool is gone, so the cursor has to go back to
     * unset rather than keep pointing part-way down a pool nothing holds any
     * more.  Otherwise a second init on the same device (the failure path here
     * calls fini, and the tests reuse a device) would carry the old cursor and
     * hand out pages the fresh pool has already given to the firmware.
     */
    mmu->pool_cursor = 0;
    mmu->fw_pool_base = 0;
    mmu->fw_pool_size = 0;
    mmu->va_next = 0;
    pthread_mutex_destroy(&mmu->lock);
}

/* ---------------------------------------------------------------- */
/* Firmware window address conversion                                */
/* ---------------------------------------------------------------- */

/*
 * Convert a firmware virtual address to the physical address it translates to.
 *
 * The firmware's own addresses are fixed by device/linker.ld at
 * CONFIG_RAM_BASE, but the memory behind them is wherever the Host put the
 * firmware pool, so anything the Host does by physical address - reserving a
 * BL1 segment, copying an image in, reading a descriptor back - has to come
 * through here.
 *
 * Returns -ENOTSUP with the address unchanged when the MMU is off, so callers
 * can use the result unconditionally: with FixMap identity-mapping everything,
 * the virtual address already is the physical one.
 */
int npu_service_mmu_fw_pa(npu_service_device_t *device, uint64_t va,
                          uint64_t size, uint64_t *pa_out)
{
    struct npu_mmu *mmu;
    uint64_t offset;

    if (!device || !pa_out)
        return -EINVAL;
    mmu = &device->mmu;
    if (!mmu->enabled) {
        *pa_out = va;
        return -ENOTSUP;
    }
    if (va < NPU_SERVICE_MMU_FW_VA_BASE)
        return -EINVAL;
    offset = va - NPU_SERVICE_MMU_FW_VA_BASE;
    /* Refuse rather than clamp: a range that runs off the end of the window is
     * a firmware image too big for the pool, and truncating the reservation
     * would put the overflow somewhere the Device cannot reach at all. */
    if (offset >= mmu->fw_pool_size || size > mmu->fw_pool_size - offset)
        return -ERANGE;
    *pa_out = mmu->fw_pool_base + offset;
    return 0;
}

/*
 * The address a Device Core must use to reach @allocation.
 *
 * This is the inverse of npu_service_mmu_fw_pa() and the one place a physical
 * address becomes a Device-visible one.  It is deliberately not folded into
 * npu_hal_memory_address(): two of that function's own callers
 * (npu_hal_memory_write/read) hand its result to npu_hal_copy_to_address(),
 * which is XDMA - a separate bus master that does not traverse the MMU and so
 * must keep using physical addresses.  Converting inside the HAL would corrupt
 * every Host-side read and write.
 *
 * Buffers are not handled here.  Their virtual addresses come from
 * npu_service_mmu_reserve_va() and are recorded by the Server, because a
 * Buffer's physical pages are outside the firmware window and have no
 * arithmetic relation to its virtual address.
 */
int npu_service_mmu_device_address(npu_service_device_t *device,
                                   const npu_hal_allocation_t *allocation,
                                   npu_hal_core_t viewer,
                                   uint64_t *address_out)
{
    struct npu_mmu *mmu;
    npu_hal_allocation_info_t info;
    uint64_t offset;
    int status;

    if (!device || !allocation || !address_out)
        return -EINVAL;
    status = npu_hal_memory_address(device->hal, allocation, viewer,
                                   address_out);
    if (status < 0)
        return status;
    mmu = &device->mmu;
    if (!mmu->enabled)
        return 0;
    status = npu_hal_memory_info(device->hal, allocation, &info);
    if (status < 0)
        return status;
    /*
     * Only Normal memory is translated.  Coherent and URAM live below 2 GiB,
     * which the MMU bypasses outright, so their addresses are already what the
     * Device must use.
     */
    if (info.memory_class != NPU_HAL_MEMORY_NORMAL)
        return 0;
    /*
     * A Buffer is outside the window by construction (the HAL allocates it
     * from the other pool).  Reporting its physical address would be wrong,
     * but silently returning a bogus virtual one would be worse, so say so.
     */
    if (*address_out < mmu->fw_pool_base ||
        *address_out - mmu->fw_pool_base >= mmu->fw_pool_size)
        return -EFAULT;
    offset = *address_out - mmu->fw_pool_base;
    *address_out = NPU_SERVICE_MMU_FW_VA_BASE + offset;
    return 0;
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
    /*
     * Below 2 GiB the MMU bypasses the page table entirely, so a PTE there
     * would never be consulted - a caller asking for one has the wrong address
     * and would otherwise get a success it cannot use.
     */
    if (va < NPU_SERVICE_MMU_BYPASS_LIMIT)
        return -EINVAL;
    /*
     * The firmware window is owned by npu_mmu_map_firmware_window() and mapped
     * with 2 MiB leaves.  A 4 KiB mapping landing inside it would either
     * collide with a large leaf or, worse, silently give a second virtual
     * address to a physical page that already has one.
     */
    if (va < NPU_SERVICE_MMU_FW_VA_BASE + mmu->fw_pool_size &&
        va + size > NPU_SERVICE_MMU_FW_VA_BASE)
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

        /* From the firmware pool, like page tables: the Host has to be able
         * to refill the page after a fault, and the Device reaches the
         * redirect target through the same window. */
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
         * No mapping is installed here.  Guard pages come out of the firmware
         * pool, which the window already covers, so mapping them again would
         * collide with -EEXIST.  A redirect base is a physical address that
         * the hardware substitutes for the whole master address and needs no
         * translation of its own - what matters is that info.address is the
         * physical address, which is what the HAL reports.
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
