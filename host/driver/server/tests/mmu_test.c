/*
 * ClusterMMU page-table and reclaim tests.
 *
 * These run against a fake HAL backend rather than a device: the point is the
 * Service-side logic - PTE encoding, three-level walks, the bump VA
 * allocator, and above all the deferred-reclaim ordering.
 *
 * That ordering is the reason this file exists.  Writing TLB_FLUSH while the
 * Device has AXI traffic in flight hangs the Cluster permanently, and there
 * is no quiesce register to prevent it, so unmapping must never flush inline.
 * QEMU's flush is synchronous and cannot reproduce the hang, so nothing else
 * in the tree would notice if that ordering regressed.
 */

#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "service_internal.h"
#include "platform_regs.h"

/* Sv39 bits, duplicated so a change in mmu.c has to be deliberate. */
#define PTE_V  UINT64_C(0x001)
#define PTE_R  UINT64_C(0x002)
#define PTE_W  UINT64_C(0x004)
#define PTE_X  UINT64_C(0x008)
#define PTE_U  UINT64_C(0x010)
#define PTE_G  UINT64_C(0x020)
#define PTE_A  UINT64_C(0x040)
#define PTE_D  UINT64_C(0x080)

#define DEVICE_MEMORY_BASE UINT64_C(0x80000000)
#define DEVICE_MEMORY_SIZE UINT64_C(0x00800000)

/*
 * The firmware pool: 4 MiB, which is two 2 MiB leaves - enough for "more than
 * one leaf" to mean something - with the Buffer pool at a different physical
 * address so the disjointness tests can tell them apart.
 *
 * FW_POOL_OFFSET is the point of this arrangement.  In the default server
 * configuration the pool sits at the Normal region base, which is also the
 * firmware's link address, so the linear offset degenerates to an identity and
 * a virtual address used as a physical one still names the right memory.  That
 * makes every address-domain confusion invisible.  Here the pool is deliberately
 * placed 2 MiB above the region base, so pa != va throughout and any code that
 * hands a firmware VA to the HAL reads or writes the wrong page.
 */
#define FW_POOL_OFFSET     UINT64_C(0x00200000)
#define FW_POOL_SIZE       UINT64_C(0x00400000)
#define FW_POOL_BASE       (DEVICE_MEMORY_BASE + FW_POOL_OFFSET)
#define BUFFER_POOL_BASE   (FW_POOL_BASE + FW_POOL_SIZE)
#define BUFFER_POOL_LIMIT  (DEVICE_MEMORY_BASE + DEVICE_MEMORY_SIZE)
/* The firmware window's virtual base, and the first Buffer VA above it. */
#define FW_VA_BASE         NPU_SERVICE_MMU_FW_VA_BASE
#define BUFFER_VA_BASE     (FW_VA_BASE + FW_POOL_SIZE + \
                            NPU_SERVICE_MMU_VA_SENTINEL)
/* One Buffer mapping leaf: 2 MiB, the same size as a firmware-window leaf. */
#define LEAF               NPU_SERVICE_MMU_PAGE_SIZE

/* Fake device memory, so page tables can be read back and inspected. */
static unsigned char device_memory[DEVICE_MEMORY_SIZE];
static uint32_t flush_count;
static uint32_t flush_fail;

static int in_device_memory(uint64_t address, size_t size)
{
    return address >= DEVICE_MEMORY_BASE &&
           address - DEVICE_MEMORY_BASE <= DEVICE_MEMORY_SIZE - size;
}

/*
 * Stubs for everything server/mmu.c calls.  Linking these instead of the real
 * HAL keeps the test to the page-table logic and lets it observe exactly which
 * register writes and flushes happen, which is the part that matters.
 */

struct npu_hal_allocation {
    uint64_t address;
    uint64_t size;
    int freed;
};

static uint32_t fake_alloc_live;

/*
 * Occupancy list, so the fake reserve and alloc can reject overlaps the way
 * the real HAL does.  What it now guards is pool disjointness: MMU pages and
 * BL1 segments come out of the firmware pool, Buffers out of the pool below
 * it, and nothing may be handed out twice or from the wrong side.
 */
#define FAKE_RESERVED_MAX 64

static struct {
    uint64_t address;
    uint64_t size;
} fake_reserved[FAKE_RESERVED_MAX];
static unsigned fake_reserved_count;

static int fake_range_free(uint64_t address, uint64_t size)
{
    unsigned index;

    for (index = 0; index < fake_reserved_count; ++index) {
        uint64_t base = fake_reserved[index].address;

        if (!fake_reserved[index].size)
            continue;
        if (address < base + fake_reserved[index].size &&
            base < address + size)
            return 0;
    }
    return 1;
}

static int fake_reserve(uint64_t address, uint64_t size,
                        npu_hal_allocation_t **allocation_out)
{
    npu_hal_allocation_t *allocation;
    unsigned index;

    if (address < DEVICE_MEMORY_BASE ||
        size > DEVICE_MEMORY_BASE + DEVICE_MEMORY_SIZE - address)
        return -ERANGE;
    if (!fake_range_free(address, size))
        return -EEXIST;
    if (fake_reserved_count == FAKE_RESERVED_MAX)
        return -ENOMEM;
    allocation = calloc(1, sizeof(*allocation));
    if (!allocation)
        return -ENOMEM;
    allocation->address = address;
    allocation->size = size;
    index = fake_reserved_count++;
    fake_reserved[index].address = address;
    fake_reserved[index].size = size;
    ++fake_alloc_live;
    *allocation_out = allocation;
    return 0;
}

/*
 * First-fit inside one pool, mirroring the real HAL: the search window is the
 * pool the usage belongs to, and the walk skips anything already taken.
 */
static int fake_alloc_in(uint64_t base, uint64_t limit, uint64_t size,
                         uint64_t alignment,
                         npu_hal_allocation_t **allocation_out)
{
    uint64_t mask = alignment ? alignment - 1U : 0;
    uint64_t address;

    for (address = (base + mask) & ~mask; address + size <= limit;
         address += alignment ? alignment : size) {
        if (fake_range_free(address, size))
            return fake_reserve(address, size, allocation_out);
    }
    return -ENOMEM;
}

/*
 * Buffer allocation, used by the reclaim tests.  Buffers come from above the
 * split, which is what keeps their physical pages outside the firmware window
 * and so reachable through exactly one virtual address.
 */
int npu_service_alloc_visible(npu_service_device_t *device,
                              npu_hal_core_t viewer,
                              npu_hal_memory_class_t memory_class,
                              npu_hal_memory_usage_t usage,
                              uint32_t owner, uint64_t size,
                              uint64_t alignment,
                              npu_hal_allocation_t **allocation_out)
{
    (void)device;
    (void)viewer;
    (void)memory_class;
    (void)owner;
    if (usage == NPU_HAL_MEMORY_BUFFER)
        return fake_alloc_in(BUFFER_POOL_BASE, BUFFER_POOL_LIMIT, size,
                             alignment, allocation_out);
    return fake_alloc_in(FW_POOL_BASE, FW_POOL_BASE + FW_POOL_SIZE, size,
                         alignment, allocation_out);
}

/* What server/mmu.c uses for page tables and guard pages. */
int npu_hal_memory_alloc(npu_hal_t *hal,
                         const npu_hal_memory_request_t *request,
                         npu_hal_allocation_t **allocation_out)
{
    (void)hal;
    *allocation_out = 0;
    if (request->version != NPU_HAL_MEMORY_REQUEST_VERSION ||
        !request->size || !request->alignment || request->fixed_address)
        return -EINVAL;
    /* Not a Buffer, so the firmware pool - the same routing the real HAL
     * applies through usage_is_firmware_pool(). */
    return fake_alloc_in(FW_POOL_BASE, FW_POOL_BASE + FW_POOL_SIZE,
                         request->size, request->alignment, allocation_out);
}

int npu_hal_firmware_pool(npu_hal_t *hal, uint64_t *base_out,
                          uint64_t *size_out)
{
    (void)hal;
    *base_out = FW_POOL_BASE;
    *size_out = FW_POOL_SIZE;
    return 0;
}

/*
 * Stands in for the BL1 segment reservation done later by server/elf.c, which
 * now reserves the physical address the firmware window maps its virtual base
 * to rather than the virtual base itself.
 */
static int fake_bl1_reserve(void)
{
    npu_hal_allocation_t *allocation = 0;
    int status = fake_reserve(FW_POOL_BASE, UINT64_C(0x20000), &allocation);

    if (status == 0)
        free(allocation);
    return status;
}

void npu_hal_memory_request_init(npu_hal_memory_request_t *request)
{
    memset(request, 0, sizeof(*request));
    request->version = NPU_HAL_MEMORY_REQUEST_VERSION;
    request->owner = UINT32_MAX;
    request->location.cluster = NPU_HAL_ANY_CLUSTER;
    request->location.core = NPU_HAL_ANY_CORE;
    request->alignment = 1;
}

int npu_hal_memory_reserve(npu_hal_t *hal,
                           const npu_hal_memory_request_t *request,
                           npu_hal_allocation_t **allocation_out)
{
    (void)hal;
    *allocation_out = 0;
    if (request->version != NPU_HAL_MEMORY_REQUEST_VERSION || !request->size ||
        !request->alignment ||
        (request->fixed_address & (request->alignment - 1U)))
        return -EINVAL;
    return fake_reserve(request->fixed_address, request->size, allocation_out);
}

/*
 * The apertures are derived from this list rather than from constants,
 * because the Normal region's size is configurable at run time.  URAM is
 * listed per core the way the real HAL does, including the repeats that
 * make -EEXIST an expected outcome rather than a failure.
 */
static const npu_hal_memory_region_t fake_regions[] = {
    { DEVICE_MEMORY_BASE, DEVICE_MEMORY_SIZE, NPU_HAL_MEMORY_NORMAL,
      NPU_HAL_MEMORY_DEVICE, NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
    { UINT64_C(0x00400000), UINT64_C(0x00400000), NPU_HAL_MEMORY_COHERENT,
      NPU_HAL_MEMORY_DEVICE, NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
    { UINT64_C(0x00800000), UINT64_C(0x00180000), NPU_HAL_MEMORY_URAM,
      NPU_HAL_MEMORY_CORE, 0, 0 },
    { UINT64_C(0x00a00000), UINT64_C(0x00180000), NPU_HAL_MEMORY_URAM,
      NPU_HAL_MEMORY_CORE, 0, 1 },
    /* Cluster 1's URAM, which Cluster 0 must not map. */
    { UINT64_C(0x00800000), UINT64_C(0x00180000), NPU_HAL_MEMORY_URAM,
      NPU_HAL_MEMORY_CORE, 1, 0 },
};

int npu_hal_get_memory_region(npu_hal_t *hal, uint32_t index,
                              npu_hal_memory_region_t *region)
{
    (void)hal;
    if (index >= sizeof(fake_regions) / sizeof(fake_regions[0]))
        return -ERANGE;
    *region = fake_regions[index];
    return 0;
}

int npu_hal_memory_info(npu_hal_t *hal, const npu_hal_allocation_t *allocation,
                        npu_hal_allocation_info_t *info)
{
    (void)hal;
    memset(info, 0, sizeof(*info));
    info->address = allocation->address;
    info->size = allocation->size;
    /* Every allocation in this test is Normal memory, which is the only class
     * the MMU translates - Coherent and URAM are below the bypass limit.  The
     * class has to be reported, or address translation would take the
     * not-translated path and hand back a physical address. */
    info->memory_class = NPU_HAL_MEMORY_NORMAL;
    info->scope = NPU_HAL_MEMORY_DEVICE;
    return 0;
}

/*
 * The HAL reports physical addresses; that is the whole reason
 * npu_service_mmu_device_address() exists as a separate call.
 */
int npu_hal_memory_address(npu_hal_t *hal,
                           const npu_hal_allocation_t *allocation,
                           npu_hal_core_t viewer, uint64_t *address_out)
{
    (void)hal;
    (void)viewer;
    if (!allocation)
        return -EACCES;
    *address_out = allocation->address;
    return 0;
}

int npu_hal_memory_free(npu_hal_t *hal, npu_hal_allocation_t *allocation)
{
    unsigned index;

    (void)hal;
    /* A double free here would mean the reclaim list released something
     * twice, which is exactly the bug this test is guarding. */
    assert(allocation && !allocation->freed);
    for (index = 0; index < fake_reserved_count; ++index) {
        if (fake_reserved[index].size &&
            fake_reserved[index].address == allocation->address) {
            fake_reserved[index].size = 0;
            break;
        }
    }
    --fake_alloc_live;
    free(allocation);
    return 0;
}

int npu_hal_copy_to_address(npu_hal_t *hal, uint32_t cluster, uint64_t address,
                            const void *source, size_t size)
{
    (void)hal;
    (void)cluster;
    assert(in_device_memory(address, size));
    memcpy(device_memory + (address - DEVICE_MEMORY_BASE), source, size);
    return 0;
}

int npu_hal_copy_from_address(npu_hal_t *hal, uint32_t cluster,
                             void *destination, uint64_t address, size_t size)
{
    (void)hal;
    (void)cluster;
    assert(in_device_memory(address, size));
    memcpy(destination, device_memory + (address - DEVICE_MEMORY_BASE), size);
    return 0;
}

/* Recorded register image, so init ordering can be checked. */
static uint32_t mmu_registers[DAYAN_MMU_APB_SIZE / 4];
static uint32_t fix_map_size_writes;
static uint32_t ptw_base_written;

int npu_hal_mmu_write_reg(npu_hal_t *hal, uint32_t cluster, uint32_t offset,
                          uint32_t value)
{
    (void)hal;
    (void)cluster;
    assert(offset < DAYAN_MMU_APB_SIZE && !(offset & 3U));
    if (offset == DAYAN_MMU_FIX_MAP_SIZE_LO && value) {
        /*
         * FixMap must not be narrowed before the walker base is programmed;
         * otherwise the first high address walks an empty table.
         */
        assert(ptw_base_written);
        ++fix_map_size_writes;
    }
    mmu_registers[offset >> 2] = value;
    return 0;
}

int npu_hal_mmu_read_reg(npu_hal_t *hal, uint32_t cluster, uint32_t offset,
                         uint32_t *value)
{
    (void)hal;
    (void)cluster;
    assert(offset < DAYAN_MMU_APB_SIZE && !(offset & 3U));
    *value = mmu_registers[offset >> 2];
    return 0;
}

int npu_hal_mmu_write64(npu_hal_t *hal, uint32_t cluster, uint32_t offset_lo,
                        uint64_t value)
{
    if (value >> DAYAN_MMU_PA_BITS)
        return -EINVAL;
    if (offset_lo == DAYAN_MMU_PTW_BASE_LO)
        ptw_base_written = 1;
    (void)npu_hal_mmu_write_reg(hal, cluster, offset_lo,
                               (uint32_t)(value & UINT32_C(0xffffffff)));
    return npu_hal_mmu_write_reg(hal, cluster, offset_lo + 4U,
                                 (uint32_t)((value >> 32) &
                                            DAYAN_MMU_PA_HI_MASK));
}

int npu_hal_mmu_present(npu_hal_t *hal, uint32_t cluster, int *present)
{
    (void)hal;
    (void)cluster;
    *present = 1;
    return 0;
}

int npu_hal_mmu_flush(npu_hal_t *hal, uint32_t cluster, int asid_selective,
                      uint8_t asid, uint32_t timeout_ms)
{
    (void)hal;
    (void)cluster;
    (void)asid_selective;
    (void)asid;
    (void)timeout_ms;
    ++flush_count;
    return flush_fail ? -ETIMEDOUT : 0;
}

/* Read one PTE out of the fake device memory. */
static uint64_t read_pte(uint64_t table_address, uint32_t index)
{
    uint64_t pte;

    assert(in_device_memory(table_address + index * 8U, sizeof(pte)));
    memcpy(&pte, device_memory + (table_address - DEVICE_MEMORY_BASE) +
                 index * 8U, sizeof(pte));
    return pte;
}

static uint64_t pte_pa(uint64_t pte)
{
    return (pte >> 10) << DAYAN_MMU_PAGE_SHIFT;
}

static int pte_is_leaf(uint64_t pte)
{
    return (pte & (PTE_R | PTE_W | PTE_X)) != 0;
}

/*
 * Walk the device-resident tables the way the hardware would, so the test
 * validates what the walker will actually see rather than the shadow.
 *
 * Stops early on a leaf, which is how a 2 MiB mapping is found: the walker
 * treats a level-1 entry with any of R/W/X set as a leaf and never reads a
 * level-0 table under it.  @level_out reports where it stopped, so a test can
 * distinguish a 2 MiB leaf from a 4 KiB one instead of just seeing a valid
 * PTE either way.
 */
static uint64_t walk_device_level(uint64_t root, uint64_t va, int *level_out)
{
    uint64_t table = root;
    int level;

    if (level_out)
        *level_out = -1;
    for (level = 2; level > 0; --level) {
        uint32_t index = (uint32_t)((va >> (DAYAN_MMU_PAGE_SHIFT +
                                            level * 9)) & 0x1ff);
        uint64_t pte = read_pte(table, index);

        if (!(pte & PTE_V))
            return 0;
        if (pte_is_leaf(pte)) {
            if (level_out)
                *level_out = level;
            return pte;
        }
        table = pte_pa(pte);
    }
    if (level_out)
        *level_out = 0;
    return read_pte(table, (uint32_t)((va >> DAYAN_MMU_PAGE_SHIFT) & 0x1ff));
}

static uint64_t walk_device(uint64_t root, uint64_t va)
{
    return walk_device_level(root, va, 0);
}

static void device_reset(npu_service_device_t *device)
{
    memset(device, 0, sizeof(*device));
    device->hal_info.cluster_count = 1;
    device->hal_info.cores_per_cluster = 4;
    device->hal_info.memory_region_count =
        sizeof(fake_regions) / sizeof(fake_regions[0]);
    device->options.enable_mmu = 1;
    device->options.default_timeout_ms = 100;
    memset(device_memory, 0, sizeof(device_memory));
    memset(mmu_registers, 0, sizeof(mmu_registers));
    fake_reserved_count = 0;
    memset(fake_reserved, 0, sizeof(fake_reserved));
    fake_alloc_live = 0;
    flush_count = 0;
    flush_fail = 0;
    fix_map_size_writes = 0;
    ptw_base_written = 0;
}

/* The MMU stays out of the way entirely when it is not enabled. */
static void test_disabled_is_inert(void)
{
    npu_service_device_t device;
    uint64_t va = 0;

    device_reset(&device);
    device.options.enable_mmu = 0;
    assert(npu_service_mmu_init(&device) == 0);
    /* No hardware touched, no memory taken. */
    assert(fix_map_size_writes == 0);
    assert(fake_alloc_live == 0);
    /* Mapping calls succeed as no-ops so callers need not branch. */
    assert(npu_service_mmu_map(&device, 0, BUFFER_VA_BASE, DEVICE_MEMORY_BASE,
                               0x1000, NPU_SERVICE_MMU_READ) == 0);
    assert(npu_service_mmu_unmap(&device, 0, BUFFER_VA_BASE, 0x1000) == 0);
    assert(npu_service_mmu_protect(&device, 0, BUFFER_VA_BASE, 0x1000,
                                   NPU_SERVICE_MMU_READ) == 0);
    assert(npu_service_mmu_check_faults(&device, 0, 0, 0) == 0);
    /* reserve_va is the one exception: there is no VA to give out. */
    assert(npu_service_mmu_reserve_va(&device, 0x1000, &va) == -ENOTSUP);
    npu_service_mmu_fini(&device);
}

static void test_init_ordering(void)
{
    npu_service_device_t device;

    device_reset(&device);
    assert(npu_service_mmu_init(&device) == 0);
    /* FixMap was narrowed exactly once, after PTW_BASE became live. */
    assert(fix_map_size_writes == 1);
    assert(mmu_registers[DAYAN_MMU_FIX_MAP_BASE_LO >> 2] == 0);
    assert(mmu_registers[DAYAN_MMU_FIX_MAP_BASE_HI >> 2] == 0);
    assert(mmu_registers[DAYAN_MMU_FIX_MAP_SIZE_LO >> 2] ==
           DAYAN_MMU_LOW_2G_PAGES);
    assert(mmu_registers[DAYAN_MMU_FIX_MAP_SIZE_HI >> 2] == 0);
    assert(mmu_registers[DAYAN_MMU_N_FIX_MAP_EN >> 2] == 0);
    /* One shared ASID: Contexts are separated by VA range, not ASID. */
    assert(mmu_registers[DAYAN_MMU_GLOBAL_ASID >> 2] == 0);
    /* Redirects must point somewhere owned, and offset mode must be on so a
     * guard hit records VA[11:0]. */
    assert(mmu_registers[DAYAN_MMU_REDIR_PAGE_FAULT_LO >> 2] != 0);
    assert(mmu_registers[DAYAN_MMU_REDIR_DEFAULT_LO >> 2] != 0);
    assert(mmu_registers[DAYAN_MMU_ERROR_SLV_OFF_MODE >> 2] ==
           DAYAN_MMU_ERROR_SLV_OFF_VA);
    npu_service_mmu_fini(&device);
    /* Teardown restores the reset-time full-range identity FixMap. */
    assert(mmu_registers[DAYAN_MMU_N_FIX_MAP_EN >> 2] == 0);
    assert(mmu_registers[DAYAN_MMU_FIX_MAP_BASE_LO >> 2] == 0);
    assert(mmu_registers[DAYAN_MMU_FIX_MAP_BASE_HI >> 2] == 0);
    assert(mmu_registers[DAYAN_MMU_FIX_MAP_SIZE_LO >> 2] == 0);
    assert(mmu_registers[DAYAN_MMU_FIX_MAP_SIZE_HI >> 2] == 0);
    assert(fake_alloc_live == 0);
}

static void test_pte_encoding_and_walk(void)
{
    npu_service_device_t device;
    uint64_t root;
    uint64_t pte;

    device_reset(&device);
    assert(npu_service_mmu_init(&device) == 0);
    root = ((uint64_t)mmu_registers[DAYAN_MMU_PTW_BASE_HI >> 2] << 32) |
           mmu_registers[DAYAN_MMU_PTW_BASE_LO >> 2];
    assert(root >= DEVICE_MEMORY_BASE);

    /*
     * The firmware window, as the hardware will see it: a linear-offset
     * mapping at a fixed virtual base, on 2 MiB leaves.  Checking the level
     * matters as much as the address - a 4 KiB mapping of the same range would
     * satisfy every other assertion here.
     */
    {
        int level = -1;

        pte = walk_device_level(root, FW_VA_BASE, &level);
        assert(pte & PTE_V);
        assert(level == 1);
        assert(pte_pa(pte) == FW_POOL_BASE);
        /* The firmware executes from here, and writes its own data. */
        assert((pte & PTE_R) && (pte & PTE_W) && (pte & PTE_X));
        /* Hardware never sets these, so a leaf without them faults forever. */
        assert((pte & PTE_A) && (pte & PTE_D));

        /* The offset is constant across the window, not just at its base. */
        pte = walk_device_level(root, FW_VA_BASE + NPU_SERVICE_MMU_LARGE_SIZE,
                                &level);
        assert((pte & PTE_V) && level == 1);
        assert(pte_pa(pte) ==
               FW_POOL_BASE + NPU_SERVICE_MMU_LARGE_SIZE);
        /* An address inside a leaf resolves through that same leaf. */
        pte = walk_device_level(root, FW_VA_BASE + 0x1000, &level);
        assert((pte & PTE_V) && level == 1);
        assert(pte_pa(pte) == FW_POOL_BASE);

        /* Past the end of the window nothing is mapped: an overrun off the
         * top must fault rather than reach the Buffer VA range. */
        assert(walk_device(root, FW_VA_BASE + FW_POOL_SIZE) == 0);
    }

    /*
     * Nothing below 2 GiB has a PTE - not control space, not the BL0 ROM, not
     * URAM.  The MMU bypasses that range before it consults a page table, so a
     * mapping here would describe a translation the hardware never performs.
     * This is the assertion that would catch a revival of the old identity
     * apertures.
     */
    assert(walk_device(root, UINT64_C(0x2000)) == 0);      /* INTC */
    assert(walk_device(root, UINT64_C(0x8000)) == 0);      /* MMU APB */
    assert(walk_device(root, UINT64_C(0x10000)) == 0);     /* BL0 ROM */
    assert(walk_device(root, UINT64_C(0x20000)) == 0);     /* CLINT */
    assert(walk_device(root, UINT64_C(0x400000)) == 0);    /* atomic cache */
    assert(walk_device(root, UINT64_C(0x800000)) == 0);    /* URAM core 0 */
    assert(walk_device(root, UINT64_C(0xa00000)) == 0);    /* URAM core 1 */
    /*
     * The Buffer pool has no identity alias.  There is nothing to walk for
     * here - its physical addresses are simply not the target of any PTE, and
     * that is what test_pools_are_disjoint() establishes.  Note that virtual
     * DEVICE_MEMORY_BASE is not it: that address is the firmware window's
     * virtual base, checked above, and the two spaces only happen to share the
     * number.
     */

    /* A W|X mapping, to check every permission bit is placed correctly. */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x200000000),
                               DEVICE_MEMORY_BASE + LEAF, 2 * LEAF,
                               NPU_SERVICE_MMU_READ | NPU_SERVICE_MMU_WRITE |
                               NPU_SERVICE_MMU_EXEC) == 0);
    {
        int level = -1;

        /* A Buffer is a 2 MiB leaf at level 1, like the firmware window; a
         * level-0 leaf would pass every other assertion below. */
        pte = walk_device_level(root, UINT64_C(0x200000000), &level);
        assert(level == 1);
        /* Any address inside the leaf resolves through it. */
        assert(walk_device_level(root, UINT64_C(0x200000000) + LEAF - 0x1000,
                                 &level) == pte);
        assert(level == 1);
    }
    assert(pte & PTE_V);
    assert(pte & PTE_R);
    assert(pte & PTE_W);
    assert(pte & PTE_X);
    /* A and D are preset because the hardware never updates them; a leaf
     * with A clear would fault forever. */
    assert(pte & PTE_A);
    assert(pte & PTE_D);
    assert(pte & PTE_U);
    assert(pte & PTE_G);
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + LEAF);
    /* The second leaf of the range resolves independently. */
    pte = walk_device(root, UINT64_C(0x200000000) + LEAF);
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + 2 * LEAF);

    /* Read-only must not carry W, or a redirect would never trigger. */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x300000000),
                               DEVICE_MEMORY_BASE + 4 * LEAF, LEAF,
                               NPU_SERVICE_MMU_READ) == 0);
    pte = walk_device(root, UINT64_C(0x300000000));
    assert((pte & PTE_R) && !(pte & PTE_W) && !(pte & PTE_X));

    /* Double mapping the same VA is a bookkeeping bug, not something to
     * silently overwrite. */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x300000000),
                               DEVICE_MEMORY_BASE + 5 * LEAF, LEAF,
                               NPU_SERVICE_MMU_READ) == -EEXIST);
    /*
     * A range that fails part-way must leave nothing behind.  The caller
     * frees the backing memory on failure, so a leaf installed for the first
     * part would otherwise point at memory about to be reused.
     */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x300000000) - 2 * LEAF,
                               DEVICE_MEMORY_BASE + 6 * LEAF, 3 * LEAF,
                               NPU_SERVICE_MMU_READ) == -EEXIST);
    assert(walk_device(root, UINT64_C(0x300000000) - 2 * LEAF) == 0);
    assert(walk_device(root, UINT64_C(0x300000000) - LEAF) == 0);
    /* The colliding leaf itself is untouched. */
    pte = walk_device(root, UINT64_C(0x300000000));
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + 4 * LEAF);

    /* Misalignment and zero length are rejected rather than rounded.  4 KiB
     * alignment is no longer enough: a 2 MiB leaf with an unaligned PPN is
     * an ILLEGAL_PTE to the walker. */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x400000000) + 0x1000,
                               DEVICE_MEMORY_BASE, LEAF,
                               NPU_SERVICE_MMU_READ) == -EINVAL);
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x400000000),
                               DEVICE_MEMORY_BASE + 0x1000, LEAF,
                               NPU_SERVICE_MMU_READ) == -EINVAL);
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x400000000),
                               DEVICE_MEMORY_BASE, 0x1000,
                               NPU_SERVICE_MMU_READ) == -EINVAL);
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x400000000),
                               DEVICE_MEMORY_BASE, 0,
                               NPU_SERVICE_MMU_READ) == -EINVAL);
    /* A PA beyond 56 bits cannot be encoded. */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x400000000),
                               UINT64_C(0x0100000000000000), LEAF,
                               NPU_SERVICE_MMU_READ) == -EINVAL);

    /* protect() narrows in place and keeps the PA. */
    assert(npu_service_mmu_protect(&device, 0, UINT64_C(0x200000000), LEAF,
                                   NPU_SERVICE_MMU_READ) == 0);
    pte = walk_device(root, UINT64_C(0x200000000));
    assert((pte & PTE_R) && !(pte & PTE_W) && !(pte & PTE_X));
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + LEAF);
    /* Nothing mapped there means nothing to protect. */
    assert(npu_service_mmu_protect(&device, 0, UINT64_C(0x500000000), LEAF,
                                   NPU_SERVICE_MMU_READ) == -ENOENT);
    assert(npu_service_mmu_protect(&device, 0, UINT64_C(0x200000000), 0x1000,
                                   NPU_SERVICE_MMU_READ) == -EINVAL);

    /*
     * The firmware window's leaves sit at the same level as a Buffer's, so a
     * walk no longer trips over them on the way down.  unmap and protect must
     * refuse the range outright, or a stray call would pull the firmware's
     * own memory out from under it.
     */
    assert(npu_service_mmu_unmap(&device, 0, FW_VA_BASE, LEAF) == -EINVAL);
    assert(npu_service_mmu_protect(&device, 0, FW_VA_BASE + LEAF, LEAF,
                                   NPU_SERVICE_MMU_READ) == -EINVAL);
    pte = walk_device(root, FW_VA_BASE);
    assert(pte_pa(pte) == FW_POOL_BASE && (pte & PTE_X));
    pte = walk_device(root, FW_VA_BASE + LEAF);
    assert((pte & PTE_W) && (pte & PTE_X));

    /* unmap clears the leaf in device memory... */
    assert(npu_service_mmu_unmap(&device, 0, UINT64_C(0x200000000),
                                 2 * LEAF) == 0);
    assert(walk_device(root, UINT64_C(0x200000000)) == 0);
    assert(walk_device(root, UINT64_C(0x200000000) + LEAF) == 0);
    /* ...and does not flush: that is the caller's, once the device is idle. */
    assert(flush_count == 0);
    /* Unmapping an absent range is not an error; reclaim retries are
     * allowed to be sloppy. */
    assert(npu_service_mmu_unmap(&device, 0, UINT64_C(0x600000000),
                                 LEAF) == 0);

    npu_service_mmu_fini(&device);
}

static void test_va_allocator(void)
{
    npu_service_device_t device;
    uint64_t first = 0;
    uint64_t second = 0;
    uint64_t third = 0;

    device_reset(&device);
    assert(npu_service_mmu_init(&device) == 0);

    assert(npu_service_mmu_reserve_va(&device, 0x1000, &first) == 0);
    /*
     * The first Buffer VA sits one 2 MiB sentinel above the top of the
     * firmware window, not at some round constant.  Deriving it from the pool
     * size is what keeps the window and the Buffer range adjacent-but-disjoint
     * whatever size the Normal region is configured to.
     */
    assert(first == BUFFER_VA_BASE);
    /* Nothing occupies the sentinel gap: this map succeeds rather than
     * colliding, which is what makes an overrun off the top of the firmware
     * window fault instead of sliding into Buffer space. */
    assert(npu_service_mmu_map(&device, 0, FW_VA_BASE + FW_POOL_SIZE,
                               DEVICE_MEMORY_BASE, LEAF,
                               NPU_SERVICE_MMU_READ) == 0);
    assert(npu_service_mmu_unmap(&device, 0, FW_VA_BASE + FW_POOL_SIZE,
                                 LEAF) == 0);
    assert(npu_service_mmu_reserve_va(&device, 0x1000, &second) == 0);
    /* Ranges never overlap and never come back, so a new Context can never
     * inherit an address another Context still has TLB entries for. */
    assert(second >= first + LEAF);
    /* Sub-leaf requests are rounded up to a whole 2 MiB leaf, so every
     * range starts on a boundary a leaf can map. */
    assert(second == first + LEAF);
    assert(npu_service_mmu_reserve_va(&device, 1, &third) == 0);
    assert(third == second + LEAF);
    assert(!(third & (LEAF - 1U)));
    /* Releasing does not recycle: the next reservation still moves forward. */
    npu_service_mmu_release_va(&device, first, 0x1000);
    {
        uint64_t after = 0;

        assert(npu_service_mmu_reserve_va(&device, 0x1000, &after) == 0);
        assert(after > third);
    }
    assert(npu_service_mmu_reserve_va(&device, 0, &first) == -EINVAL);
    /* Exhaustion is reported, not wrapped. */
    assert(npu_service_mmu_reserve_va(&device, UINT64_C(0x4000000000),
                                      &first) == -ENOSPC);

    npu_service_mmu_fini(&device);
}

/*
 * The core safety property: a release parks memory, and the flush happens
 * once for the whole batch at a moment the caller declares idle.
 */
static void test_deferred_reclaim(void)
{
    npu_service_device_t device;
    npu_hal_allocation_t *first = 0;
    npu_hal_allocation_t *second = 0;
    npu_hal_core_t viewer = { 0, 0 };
    uint32_t live_before;

    device_reset(&device);
    assert(npu_service_mmu_init(&device) == 0);
    assert(npu_service_alloc_visible(&device, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_BUFFER, 0, 0x1000,
                                     0x1000, &first) == 0);
    assert(npu_service_alloc_visible(&device, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_BUFFER, 0, 0x1000,
                                     0x1000, &second) == 0);
    live_before = fake_alloc_live;

    /* Parking must not free, and must not flush. */
    assert(npu_service_mmu_defer_reclaim(&device, 0, first) == 0);
    assert(npu_service_mmu_defer_reclaim(&device, 0, second) == 0);
    assert(fake_alloc_live == live_before);
    assert(flush_count == 0);

    /* A failing flush leaves everything parked: freeing memory whose TLB
     * entries may still be live is worse than leaking it until the retry. */
    flush_fail = 1;
    assert(npu_service_mmu_run_reclaim(&device, 0) == -ETIMEDOUT);
    assert(fake_alloc_live == live_before);
    assert(flush_count == 1);

    /* On success one flush covers the batch, then the memory goes back. */
    flush_fail = 0;
    assert(npu_service_mmu_run_reclaim(&device, 0) == 0);
    assert(flush_count == 2);
    assert(fake_alloc_live == live_before - 2);
    /* Draining an empty list is free - no flush, so the idle hook can be
     * called on every launch completion without cost. */
    assert(npu_service_mmu_run_reclaim(&device, 0) == 0);
    assert(flush_count == 2);

    /* force skips the flush, for teardown after translation is already off. */
    assert(npu_service_alloc_visible(&device, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_BUFFER, 0, 0x1000,
                                     0x1000, &first) == 0);
    assert(npu_service_mmu_defer_reclaim(&device, 0, first) == 0);
    assert(npu_service_mmu_run_reclaim(&device, 1) == 0);
    assert(flush_count == 2);

    npu_service_mmu_fini(&device);
    assert(fake_alloc_live == 0);
}

/*
 * A Buffer's mapping goes into every Cluster's table, so reclaiming it has to
 * flush every Cluster.  Flushing only the Cluster recorded on the reclaim
 * entry would leave another Cluster translating an address whose backing
 * memory has been handed to someone else - the exact bug deferred reclaim
 * exists to prevent, and one that needs more than one Cluster to show up.
 */
static void test_reclaim_flushes_every_cluster(void)
{
    npu_service_device_t device;
    npu_hal_allocation_t *allocation = 0;
    npu_hal_core_t viewer = { 0, 0 };

    device_reset(&device);
    device.hal_info.cluster_count = 2;
    assert(npu_service_mmu_init(&device) == 0);
    assert(npu_service_alloc_visible(&device, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_BUFFER, 0, 0x1000,
                                     0x1000, &allocation) == 0);
    /* Recorded against Cluster 0, as the Buffer release path does. */
    assert(npu_service_mmu_defer_reclaim(&device, 0, allocation) == 0);
    flush_count = 0;
    assert(npu_service_mmu_run_reclaim(&device, 0) == 0);
    assert(flush_count == 2);
    npu_service_mmu_fini(&device);
}

/* Guard pages turn an unreportable fault into an observable one. */
static void test_guard_pages(void)
{
    npu_service_device_t device;
    uint32_t fault_class = UINT32_MAX;
    uint32_t fault_offset = UINT32_MAX;
    uint64_t guard;
    uint64_t corruption = 0;

    device_reset(&device);
    assert(npu_service_mmu_init(&device) == 0);
    /* Untouched guard pages mean no fault was observed. */
    assert(npu_service_mmu_check_faults(&device, 0, &fault_class,
                                        &fault_offset) == 0);

    /* Simulate a redirected write landing in the write-permission guard.
     * With ERROR_SLV_OFF_MODE set the landing offset is the faulting
     * access's VA[11:0], so the offset is recoverable. */
    guard = ((uint64_t)mmu_registers[(DAYAN_MMU_REDIR_WRITE_PERM_LO + 4) >> 2]
             << 32) | mmu_registers[DAYAN_MMU_REDIR_WRITE_PERM_LO >> 2];
    assert(guard >= DEVICE_MEMORY_BASE);
    assert(npu_hal_copy_to_address(0, 0, guard + 0x40, &corruption,
                                   sizeof(corruption)) == 0);
    assert(npu_service_mmu_check_faults(&device, 0, &fault_class,
                                        &fault_offset) == 1);
    /* Class 2 is write-permission, per the redirect base ordering. */
    assert(fault_class == 2);
    assert(fault_offset == 0x40);
    /* The page is restored, so the next check starts clean and a second
     * fault is distinguishable from the first. */
    assert(npu_service_mmu_check_faults(&device, 0, &fault_class,
                                        &fault_offset) == 0);

    npu_service_mmu_fini(&device);
}

/*
 * The two pools must not overlap, and MMU pages must come from the firmware
 * one.
 *
 * This is the property the whole design rests on: every physical page has
 * exactly one virtual address.  Page tables and guard pages inside the
 * firmware window are reachable by the Device through the linear offset;
 * Buffer pages outside it are reachable only through their own VA.  If an MMU
 * page were allocated from the Buffer pool it would be both unreachable by the
 * firmware and able to collide with a Buffer.
 *
 * It also covers what the old top-down pool test covered - a later BL1
 * reservation still succeeding - but for a different reason.  BL1 is now
 * reserved at pool_base rather than at the bottom of Normal memory, so what
 * must be true is that init left pool_base free, not that it stayed high.
 */
static void test_pools_are_disjoint(void)
{
    npu_service_device_t device;
    npu_hal_allocation_t *buffer = 0;
    npu_hal_core_t viewer = { 0, 0 };
    npu_hal_allocation_info_t info;
    unsigned index;

    device_reset(&device);
    device.hal_info.cluster_count = 2;
    assert(npu_service_mmu_init(&device) == 0);
    /* Something really was taken, otherwise the loop below proves nothing. */
    assert(fake_alloc_live > 0);

    /* Every page init took is inside the firmware pool. */
    for (index = 0; index < fake_reserved_count; ++index) {
        if (!fake_reserved[index].size)
            continue;
        assert(fake_reserved[index].address >= FW_POOL_BASE);
        assert(fake_reserved[index].address - FW_POOL_BASE +
               fake_reserved[index].size <= FW_POOL_SIZE);
    }

    /* A Buffer lands outside the window, so it has no identity alias. */
    assert(npu_service_alloc_visible(&device, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_BUFFER, 0, 0x1000,
                                     0x1000, &buffer) == 0);
    assert(npu_hal_memory_info(0, buffer, &info) == 0);
    assert(info.address >= FW_POOL_BASE + FW_POOL_SIZE);
    assert(npu_hal_memory_free(0, buffer) == 0);

    /* BL1's segments are reserved at the pool base; init must not have taken
     * it, or the Server would refuse to start. */
    assert(fake_bl1_reserve() == 0);

    npu_service_mmu_fini(&device);

    /*
     * Re-init on the same device - no device_reset(), because what is under
     * test is exactly the state fini is supposed to have cleared.  Every page
     * the first init took has been freed, so the second one must start at the
     * top of the pool again.  A pool cursor carried across fini would instead
     * resume below the first init's low-water mark, and enough retries (the
     * failure path in npu_service_mmu_init() calls fini itself) would walk it
     * down into the segments BL1 is reserved at.
     */
    assert(npu_service_mmu_init(&device) == 0);
    {
        /* One 512-entry Sv39 table: the first thing any init reserves, and it
         * comes off the very top of the pool. */
        uint64_t top = FW_POOL_BASE + FW_POOL_SIZE - 0x1000;
        int found_top = 0;

        for (index = 0; index < fake_reserved_count; ++index) {
            if (!fake_reserved[index].size)
                continue;
            if (fake_reserved[index].address == top)
                found_top = 1;
        }
        assert(found_top);
    }
    npu_service_mmu_fini(&device);
}

/*
 * Host-side and Device-side addresses for the same firmware memory are related
 * by the pool offset, in both directions.  Everything the Host hands to the
 * Device goes through this conversion, so an error in it is not a fault the
 * hardware can report - the Device just reads the wrong memory.
 */
static void test_firmware_address_conversion(void)
{
    npu_service_device_t device;
    uint64_t pa = 0;

    device_reset(&device);
    assert(npu_service_mmu_init(&device) == 0);

    /*
     * The premise of every assertion below: the pool is not at the virtual
     * base, so a virtual address used as a physical one names the wrong memory.
     * With the default identity offset none of these would distinguish a
     * correct conversion from no conversion at all.
     */
    assert(FW_POOL_BASE != FW_VA_BASE);

    assert(npu_service_mmu_fw_pa(&device, FW_VA_BASE, 0x1000, &pa) == 0);
    assert(pa == FW_POOL_BASE);
    assert(npu_service_mmu_fw_pa(&device, FW_VA_BASE + 0x2000, 0x1000,
                                 &pa) == 0);
    assert(pa == FW_POOL_BASE + 0x2000);
    /* The last byte of the window is still inside it. */
    assert(npu_service_mmu_fw_pa(&device, FW_VA_BASE + FW_POOL_SIZE - 0x1000,
                                 0x1000, &pa) == 0);
    assert(pa == FW_POOL_BASE + FW_POOL_SIZE - 0x1000);
    /*
     * A range running off the end is refused, not clamped.  That case is a
     * firmware image too large for the pool, and truncating the reservation
     * would put the overflow where the Device cannot reach it at all.
     */
    assert(npu_service_mmu_fw_pa(&device, FW_VA_BASE + FW_POOL_SIZE - 0x1000,
                                 0x2000, &pa) == -ERANGE);
    assert(npu_service_mmu_fw_pa(&device, FW_VA_BASE + FW_POOL_SIZE, 0x1000,
                                 &pa) == -ERANGE);
    /* Below the window is not an offset into it. */
    assert(npu_service_mmu_fw_pa(&device, UINT64_C(0x10000), 0x1000,
                                 &pa) == -EINVAL);

    npu_service_mmu_fini(&device);

    /*
     * With the MMU off the two address spaces are the same, and the call
     * reports that rather than failing - callers use the result either way.
     */
    device_reset(&device);
    device.options.enable_mmu = 0;
    assert(npu_service_mmu_init(&device) == 0);
    assert(npu_service_mmu_fw_pa(&device, FW_VA_BASE + 0x3000, 0x1000,
                                 &pa) == -ENOTSUP);
    assert(pa == FW_VA_BASE + 0x3000);
    npu_service_mmu_fini(&device);
}

/*
 * The addresses handed to the Device.  Firmware-pool allocations are reported
 * as window virtual addresses; a Buffer is refused, because its virtual
 * address comes from the bump allocator and cannot be derived arithmetically.
 * Silently returning the physical address for a Buffer is the failure this
 * guards - it would work with the MMU off and read the wrong memory with it
 * on.
 */
static void test_device_address_translation(void)
{
    npu_service_device_t device;
    npu_hal_allocation_t *firmware = 0;
    npu_hal_allocation_t *buffer = 0;
    npu_hal_core_t viewer = { 0, 0 };
    npu_hal_allocation_info_t info;
    uint64_t address = 0;

    device_reset(&device);
    assert(npu_service_mmu_init(&device) == 0);

    assert(npu_service_alloc_visible(&device, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_LAUNCH, 0, 0x1000,
                                     0x1000, &firmware) == 0);
    assert(npu_hal_memory_info(0, firmware, &info) == 0);
    assert(npu_service_mmu_device_address(&device, firmware, viewer,
                                          &address) == 0);
    assert(address == FW_VA_BASE + (info.address - FW_POOL_BASE));
    /* Round-trips back to the physical address the Host uses. */
    {
        uint64_t pa = 0;

        assert(npu_service_mmu_fw_pa(&device, address, 0x1000, &pa) == 0);
        assert(pa == info.address);
    }

    assert(npu_service_alloc_visible(&device, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_BUFFER, 0, 0x1000,
                                     0x1000, &buffer) == 0);
    assert(npu_service_mmu_device_address(&device, buffer, viewer,
                                          &address) == -EFAULT);

    assert(npu_hal_memory_free(0, buffer) == 0);
    assert(npu_hal_memory_free(0, firmware) == 0);
    npu_service_mmu_fini(&device);

    /* With the MMU off the physical address is what the Device uses. */
    device_reset(&device);
    device.options.enable_mmu = 0;
    assert(npu_service_mmu_init(&device) == 0);
    assert(npu_service_alloc_visible(&device, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_BUFFER, 0, 0x1000,
                                     0x1000, &buffer) == 0);
    assert(npu_hal_memory_info(0, buffer, &info) == 0);
    assert(npu_service_mmu_device_address(&device, buffer, viewer,
                                          &address) == 0);
    assert(address == info.address);
    assert(npu_hal_memory_free(0, buffer) == 0);
    npu_service_mmu_fini(&device);
}

int main(void)
{
    test_disabled_is_inert();
    test_init_ordering();
    test_pte_encoding_and_walk();
    test_va_allocator();
    test_deferred_reclaim();
    test_reclaim_flushes_every_cluster();
    test_guard_pages();
    test_pools_are_disjoint();
    test_firmware_address_conversion();
    test_device_address_translation();
    puts("mmu_test: page tables, VA allocator, deferred reclaim, guard pages "
         "and firmware coexistence OK");
    return 0;
}
