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
#define DEVICE_MEMORY_SIZE UINT64_C(0x00400000)

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
 * Occupancy list, so the fake reserve can reject overlaps the way the real HAL
 * does.  That rejection is the point: npu_service_mmu_init() runs before the
 * firmware is loaded, and the firmware is reserved at fixed addresses starting
 * at the bottom of Normal memory.  A page-table pool that grew upwards from
 * the region base would take 0x80000000 and make the later BL1 reservation
 * fail with -EEXIST, which is what fake_bl1_reserve() below reproduces.
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
 * Ordinary Buffer allocation, used by the reclaim tests.  It bumps upwards
 * from the bottom of the region and skips anything already taken, mirroring
 * the real HAL's first-fit and giving out exactly the low addresses the MMU
 * pool must stay clear of.
 */
int npu_service_alloc_visible(npu_service_device_t *device,
                              npu_hal_core_t viewer,
                              npu_hal_memory_class_t memory_class,
                              npu_hal_memory_usage_t usage,
                              uint32_t owner, uint64_t size,
                              uint64_t alignment,
                              npu_hal_allocation_t **allocation_out)
{
    uint64_t address;
    uint64_t mask = alignment ? alignment - 1U : 0;

    (void)device;
    (void)viewer;
    (void)memory_class;
    (void)usage;
    (void)owner;
    for (address = (DEVICE_MEMORY_BASE + mask) & ~mask;
         address + size <= DEVICE_MEMORY_BASE + DEVICE_MEMORY_SIZE;
         address += alignment ? alignment : size) {
        if (fake_range_free(address, size))
            return fake_reserve(address, size, allocation_out);
    }
    return -ENOMEM;
}

/* Stands in for the BL1 segment reservation done later by server/elf.c. */
static int fake_bl1_reserve(void)
{
    npu_hal_allocation_t *allocation = 0;
    int status = fake_reserve(DEVICE_MEMORY_BASE, UINT64_C(0x20000),
                              &allocation);

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
static uint32_t n_fix_map_en_writes;
static uint32_t ptw_base_written;

int npu_hal_mmu_write_reg(npu_hal_t *hal, uint32_t cluster, uint32_t offset,
                          uint32_t value)
{
    (void)hal;
    (void)cluster;
    assert(offset < DAYAN_MMU_APB_SIZE && !(offset & 3U));
    if (offset == DAYAN_MMU_N_FIX_MAP_EN && value) {
        /*
         * Translation must not start before the walker base is programmed;
         * otherwise the first access after FixMap goes off walks an empty
         * table.
         */
        assert(ptw_base_written);
        ++n_fix_map_en_writes;
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

/* Walk the device-resident tables the way the hardware would, so the test
 * validates what the walker will actually see rather than the shadow. */
static uint64_t walk_device(uint64_t root, uint64_t va)
{
    uint64_t table = root;
    int level;

    for (level = 2; level > 0; --level) {
        uint32_t index = (uint32_t)((va >> (DAYAN_MMU_PAGE_SHIFT +
                                            level * 9)) & 0x1ff);
        uint64_t pte = read_pte(table, index);

        if (!(pte & PTE_V))
            return 0;
        table = pte_pa(pte);
    }
    return read_pte(table, (uint32_t)((va >> DAYAN_MMU_PAGE_SHIFT) & 0x1ff));
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
    n_fix_map_en_writes = 0;
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
    assert(n_fix_map_en_writes == 0);
    assert(fake_alloc_live == 0);
    /* Mapping calls succeed as no-ops so callers need not branch. */
    assert(npu_service_mmu_map(&device, 0, 0x1000, 0x1000, 0x1000,
                               NPU_SERVICE_MMU_READ) == 0);
    assert(npu_service_mmu_unmap(&device, 0, 0x1000, 0x1000) == 0);
    assert(npu_service_mmu_protect(&device, 0, 0x1000, 0x1000,
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
    /* Translation was switched on exactly once, and the stub asserted that
     * PTW_BASE preceded it. */
    assert(n_fix_map_en_writes == 1);
    assert(mmu_registers[DAYAN_MMU_N_FIX_MAP_EN >> 2] ==
           DAYAN_MMU_N_FIX_MAP_EN_BIT);
    /* One shared ASID: Contexts are separated by VA range, not ASID. */
    assert(mmu_registers[DAYAN_MMU_GLOBAL_ASID >> 2] == 0);
    /* Redirects must point somewhere owned, and offset mode must be on so a
     * guard hit records VA[11:0]. */
    assert(mmu_registers[DAYAN_MMU_REDIR_PAGE_FAULT_LO >> 2] != 0);
    assert(mmu_registers[DAYAN_MMU_REDIR_DEFAULT_LO >> 2] != 0);
    assert(mmu_registers[DAYAN_MMU_ERROR_SLV_OFF_MODE >> 2] ==
           DAYAN_MMU_ERROR_SLV_OFF_VA);
    npu_service_mmu_fini(&device);
    /* Teardown returns to identity addressing and releases everything. */
    assert(mmu_registers[DAYAN_MMU_N_FIX_MAP_EN >> 2] == 0);
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

    /* The identity mappings built at init must be walkable by the hardware
     * path, not just present in the Host shadow. */
    pte = walk_device(root, DEVICE_MEMORY_BASE);
    assert(pte & PTE_V);
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE);
    /* DRAM holds the firmware image and is executed. */
    assert(pte & PTE_X);
    /* Control space is identity-mapped but never executable... */
    pte = walk_device(root, UINT64_C(0x2000));
    assert((pte & PTE_V) && !(pte & PTE_X));
    assert(pte_pa(pte) == UINT64_C(0x2000));
    pte = walk_device(root, UINT64_C(0x20000));
    assert((pte & PTE_V) && !(pte & PTE_X));
    /*
     * ...except the BL0 ROM, which the Device fetches from after translation
     * is on (the MMU is activated before BL1 is loaded).  Without X here the
     * wake-up from wfi is an exec-permission fault redirected to a guard page.
     */
    pte = walk_device(root, NPU_SERVICE_MMU_ROM_BASE);
    assert((pte & PTE_V) && (pte & PTE_X) && !(pte & PTE_W));
    assert(pte_pa(pte) == NPU_SERVICE_MMU_ROM_BASE);
    pte = walk_device(root, NPU_SERVICE_MMU_ROM_BASE +
                            NPU_SERVICE_MMU_ROM_SIZE - 0x1000);
    assert((pte & PTE_V) && (pte & PTE_X));
    pte = walk_device(root, NPU_SERVICE_MMU_ROM_BASE + NPU_SERVICE_MMU_ROM_SIZE);
    assert((pte & PTE_V) && !(pte & PTE_X));
    /* The coherent window and this Cluster's URAM are mapped... */
    pte = walk_device(root, UINT64_C(0x400000));
    assert(pte & PTE_V);
    pte = walk_device(root, UINT64_C(0x800000));
    assert((pte & PTE_V) && !(pte & PTE_X));
    pte = walk_device(root, UINT64_C(0xa00000));
    assert(pte & PTE_V);
    /*
     * ...but the region list's last entry is Cluster 1's URAM, which must
     * not appear in Cluster 0's table.  It shares Cluster 0's URAM address,
     * so if the scope filter were missing this would instead collide and
     * fail init outright.
     */
    pte = walk_device(root, UINT64_C(0xc00000));
    assert(pte == 0);
    /* Nothing above DRAM is mapped. */
    assert(walk_device(root, DEVICE_MEMORY_BASE + DEVICE_MEMORY_SIZE) == 0);

    /* A W|X mapping, to check every permission bit is placed correctly. */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x200000000),
                               DEVICE_MEMORY_BASE + 0x10000, 0x2000,
                               NPU_SERVICE_MMU_READ | NPU_SERVICE_MMU_WRITE |
                               NPU_SERVICE_MMU_EXEC) == 0);
    pte = walk_device(root, UINT64_C(0x200000000));
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
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + 0x10000);
    /* The second page of the range resolves independently. */
    pte = walk_device(root, UINT64_C(0x200000000) + 0x1000);
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + 0x11000);

    /* Read-only must not carry W, or a redirect would never trigger. */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x300000000),
                               DEVICE_MEMORY_BASE + 0x20000, 0x1000,
                               NPU_SERVICE_MMU_READ) == 0);
    pte = walk_device(root, UINT64_C(0x300000000));
    assert((pte & PTE_R) && !(pte & PTE_W) && !(pte & PTE_X));

    /* Double mapping the same VA is a bookkeeping bug, not something to
     * silently overwrite. */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x300000000),
                               DEVICE_MEMORY_BASE + 0x21000, 0x1000,
                               NPU_SERVICE_MMU_READ) == -EEXIST);
    /*
     * A range that fails part-way must leave nothing behind.  The caller
     * frees the backing memory on failure, so a leaf installed for the first
     * page would otherwise point at memory about to be reused.
     */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x2ffffe000),
                               DEVICE_MEMORY_BASE + 0x30000, 0x3000,
                               NPU_SERVICE_MMU_READ) == -EEXIST);
    assert(walk_device(root, UINT64_C(0x2ffffe000)) == 0);
    assert(walk_device(root, UINT64_C(0x2fffff000)) == 0);
    /* The colliding page itself is untouched. */
    pte = walk_device(root, UINT64_C(0x300000000));
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + 0x20000);

    /* Misalignment and zero length are rejected rather than rounded. */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x400000000) + 1,
                               DEVICE_MEMORY_BASE, 0x1000,
                               NPU_SERVICE_MMU_READ) == -EINVAL);
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x400000000),
                               DEVICE_MEMORY_BASE + 1, 0x1000,
                               NPU_SERVICE_MMU_READ) == -EINVAL);
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x400000000),
                               DEVICE_MEMORY_BASE, 0x800,
                               NPU_SERVICE_MMU_READ) == -EINVAL);
    /* A PA beyond 56 bits cannot be encoded. */
    assert(npu_service_mmu_map(&device, 0, UINT64_C(0x400000000),
                               UINT64_C(0x0100000000000000), 0x1000,
                               NPU_SERVICE_MMU_READ) == -EINVAL);

    /* protect() narrows in place and keeps the PA. */
    assert(npu_service_mmu_protect(&device, 0, UINT64_C(0x200000000), 0x1000,
                                   NPU_SERVICE_MMU_READ) == 0);
    pte = walk_device(root, UINT64_C(0x200000000));
    assert((pte & PTE_R) && !(pte & PTE_W) && !(pte & PTE_X));
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + 0x10000);
    /* Nothing mapped there means nothing to protect. */
    assert(npu_service_mmu_protect(&device, 0, UINT64_C(0x500000000), 0x1000,
                                   NPU_SERVICE_MMU_READ) == -ENOENT);

    /* unmap clears the leaf in device memory... */
    assert(npu_service_mmu_unmap(&device, 0, UINT64_C(0x200000000),
                                 0x2000) == 0);
    assert(walk_device(root, UINT64_C(0x200000000)) == 0);
    /* ...and does not flush: that is the caller's, once the device is idle. */
    assert(flush_count == 0);
    /* Unmapping an absent range is not an error; reclaim retries are
     * allowed to be sloppy. */
    assert(npu_service_mmu_unmap(&device, 0, UINT64_C(0x600000000),
                                 0x1000) == 0);

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
    assert(npu_service_mmu_reserve_va(&device, 0x1000, &second) == 0);
    /* Ranges never overlap and never come back, so a new Context can never
     * inherit an address another Context still has TLB entries for. */
    assert(second >= first + 0x1000);
    /* Sub-page requests are rounded up to a whole page. */
    assert(npu_service_mmu_reserve_va(&device, 1, &third) == 0);
    assert(third >= second + 0x1000);
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
 * MMU init must leave the bottom of Normal memory alone.  It runs before the
 * firmware is loaded, and the firmware is reserved at fixed addresses starting
 * at the region base (BL1's ELF entry is 0x80000000).  Taking page tables from
 * the bottom made that later reservation fail with -EEXIST and the Server
 * refuse to start; the pool grows downwards from the top instead.
 */
static void test_pool_leaves_bl1_room(void)
{
    npu_service_device_t device;

    device_reset(&device);
    device.hal_info.cluster_count = 2;
    /* BL1 first: on a clean device it must succeed, so a later failure below
     * can only mean the MMU took the address. */
    assert(fake_bl1_reserve() == 0);
    device_reset(&device);

    assert(npu_service_mmu_init(&device) == 0);
    /* Something really was taken, otherwise this proves nothing. */
    assert(fake_alloc_live > 0);
    assert(fake_bl1_reserve() == 0);
    /* Every MMU page sits above the firmware window it just cleared. */
    {
        unsigned index;

        for (index = 0; index < fake_reserved_count; ++index) {
            if (!fake_reserved[index].size)
                continue;
            assert(fake_reserved[index].address >=
                   DEVICE_MEMORY_BASE + UINT64_C(0x20000) ||
                   fake_reserved[index].address == DEVICE_MEMORY_BASE);
        }
    }
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
    test_pool_leaves_bl1_room();
    puts("mmu_test: page tables, VA allocator, deferred reclaim, guard pages "
         "and firmware coexistence OK");
    return 0;
}
