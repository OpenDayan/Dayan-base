/*
 * ClusterMMU Service policy tests (server/mmu.c).
 *
 * server/mmu.c is linked together with the real HAL page-table code
 * (hal/mmu_pt.c) on top of a fake HAL (hal/tests/mmu_fake_hal.c), so what is
 * checked here is the policy as the hardware would see it: the firmware
 * window, the bump VA allocator, address conversion, pool disjointness and
 * above all the deferred-reclaim ordering.  The page-table mechanism itself is
 * covered by hal/tests/mmu_pt_test.c.
 *
 * The reclaim ordering is the reason this file exists.  Writing TLB_FLUSH
 * while the Device has AXI traffic in flight hangs the Cluster permanently,
 * and there is no quiesce register to prevent it, so unmapping must never
 * flush inline.  QEMU's flush is synchronous and cannot reproduce the hang,
 * so nothing else in the tree would notice if that ordering regressed.
 */

#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "service_internal.h"
#include "mmu_fake_hal.h"

/* The firmware window's virtual base, and the first Buffer VA above it. */
#define FW_VA_BASE         NPU_SERVICE_MMU_FW_VA_BASE
#define BUFFER_VA_BASE     (FW_VA_BASE + FW_POOL_SIZE + \
                            NPU_SERVICE_MMU_VA_SENTINEL)

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

static void device_reset(npu_service_device_t *device)
{
    memset(device, 0, sizeof(*device));
    device->hal = FAKE_HAL;
    device->hal_info.cluster_count = 1;
    device->hal_info.cores_per_cluster = 4;
    device->options.enable_mmu = 1;
    device->options.default_timeout_ms = 100;
    fake_hal_reset();
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
    assert(!ptw_base_written);
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

/*
 * Init end to end: tables built, the firmware window installed, and only then
 * FixMap narrowed.  The fake asserts that FixMap is never narrowed before
 * PTW_BASE is live.
 */
static void test_init_and_teardown(void)
{
    npu_service_device_t device;

    device_reset(&device);
    assert(npu_service_mmu_init(&device) == 0);
    assert(device.mmu.enabled);
    assert(fix_map_size_writes == 1);
    assert(fake_reg64(DAYAN_MMU_FIX_MAP_SIZE_LO) == DAYAN_MMU_LOW_2G_PAGES);
    /* The window was in place before activation, not added after it. */
    assert(walk_device(fake_ptw_base(0), FW_VA_BASE) != 0);
    npu_service_mmu_fini(&device);
    /* Teardown restores the reset-time full-range identity FixMap and gives
     * every page back. */
    assert(fake_reg64(DAYAN_MMU_FIX_MAP_SIZE_LO) == 0);
    assert(!device.mmu.enabled);
    assert(fake_alloc_live == 0);

    /* Asked for, but not there: init fails and identity addressing stays. */
    device_reset(&device);
    fake_mmu_absent_cluster = 0;
    assert(npu_service_mmu_init(&device) == -ENODEV);
    assert(!device.mmu.enabled);
    assert(fix_map_size_writes == 0);
    assert(fake_alloc_live == 0);
    npu_service_mmu_fini(&device);
}

/*
 * The firmware window, as the hardware will see it: a linear-offset mapping
 * at a fixed virtual base, on 2 MiB leaves, in every Cluster's table.
 */
static void test_firmware_window(void)
{
    npu_service_device_t device;
    uint32_t cluster;

    device_reset(&device);
    device.hal_info.cluster_count = 2;
    assert(npu_service_mmu_init(&device) == 0);
    for (cluster = 0; cluster < 2; ++cluster) {
        uint64_t root = fake_ptw_base(cluster);
        uint64_t pte;
        int level = -1;

        pte = walk_device_level(root, FW_VA_BASE, &level);
        assert((pte & PTE_V) && level == 1);
        assert(pte_pa(pte) == FW_POOL_BASE);
        /* The firmware executes from here, and writes its own data. */
        assert((pte & PTE_R) && (pte & PTE_W) && (pte & PTE_X));
        /* The offset is constant across the window, not just at its base. */
        pte = walk_device_level(root, FW_VA_BASE + LEAF, &level);
        assert((pte & PTE_V) && level == 1);
        assert(pte_pa(pte) == FW_POOL_BASE + LEAF);
        /* Past the end of the window nothing is mapped: an overrun off the
         * top must fault rather than reach the Buffer VA range. */
        assert(walk_device(root, FW_VA_BASE + FW_POOL_SIZE) == 0);
        /*
         * Nothing below 2 GiB has a PTE - not control space, not the BL0
         * ROM, not URAM.  The MMU bypasses that range before it consults a
         * page table.
         */
        assert(walk_device(root, UINT64_C(0x2000)) == 0);      /* INTC */
        assert(walk_device(root, UINT64_C(0x10000)) == 0);     /* BL0 ROM */
        assert(walk_device(root, UINT64_C(0x800000)) == 0);    /* URAM */
    }

    /*
     * The window's leaves sit at the same level as a Buffer's, so the HAL
     * would happily rewrite them.  The Service refuses any range touching
     * the window: map would alias a firmware page, unmap or protect would
     * pull the firmware's own memory out from under it.
     */
    assert(npu_service_mmu_map(&device, 0, FW_VA_BASE + LEAF,
                               DEVICE_MEMORY_BASE, LEAF,
                               NPU_SERVICE_MMU_READ) == -EINVAL);
    assert(npu_service_mmu_map(&device, 0, FW_VA_BASE - LEAF,
                               DEVICE_MEMORY_BASE, 2 * LEAF,
                               NPU_SERVICE_MMU_READ) == -EINVAL);
    assert(npu_service_mmu_unmap(&device, 0, FW_VA_BASE, LEAF) == -EINVAL);
    assert(npu_service_mmu_protect(&device, 0, FW_VA_BASE + LEAF, LEAF,
                                   NPU_SERVICE_MMU_READ) == -EINVAL);
    {
        uint64_t pte = walk_device(fake_ptw_base(0), FW_VA_BASE);

        assert(pte_pa(pte) == FW_POOL_BASE && (pte & PTE_W) &&
               (pte & PTE_X));
    }
    /* Out-of-range Cluster. */
    assert(npu_service_mmu_map(&device, 2, BUFFER_VA_BASE, DEVICE_MEMORY_BASE,
                               LEAF, NPU_SERVICE_MMU_READ) == -EINVAL);

    npu_service_mmu_fini(&device);
    assert(fake_alloc_live == 0);
}

/* A Buffer mapping through the Service lands in the HAL tables as a 2 MiB
 * leaf; the rules on it are the HAL's and tested there. */
static void test_buffer_mapping(void)
{
    npu_service_device_t device;
    uint64_t va = 0;
    uint64_t pte;
    int level = -1;

    device_reset(&device);
    assert(npu_service_mmu_init(&device) == 0);
    assert(npu_service_mmu_reserve_va(&device, 1, &va) == 0);
    assert(npu_service_mmu_map(&device, 0, va, BUFFER_POOL_BASE, LEAF,
                               NPU_SERVICE_MMU_READ |
                               NPU_SERVICE_MMU_WRITE) == 0);
    pte = walk_device_level(fake_ptw_base(0), va, &level);
    assert(level == 1 && pte_pa(pte) == BUFFER_POOL_BASE);
    assert((pte & PTE_R) && (pte & PTE_W) && !(pte & PTE_X));
    /* Sub-leaf sizes are the caller's bug, not something to round. */
    assert(npu_service_mmu_map(&device, 0, va + LEAF, BUFFER_POOL_BASE,
                               0x1000, NPU_SERVICE_MMU_READ) == -EINVAL);
    assert(npu_service_mmu_unmap(&device, 0, va, LEAF) == 0);
    assert(walk_device(fake_ptw_base(0), va) == 0);
    /* Unmapping does not flush; that belongs to deferred reclaim. */
    assert(flush_count == 0);
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
     * inherit an address another Context still has TLB entries for.
     * Sub-leaf requests are rounded up to a whole 2 MiB leaf, so every range
     * starts on a boundary a leaf can map. */
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

    /* fini drains whatever is still parked, without a flush. */
    assert(npu_service_alloc_visible(&device, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_BUFFER, 0, 0x1000,
                                     0x1000, &first) == 0);
    assert(npu_service_mmu_defer_reclaim(&device, 0, first) == 0);
    npu_service_mmu_fini(&device);
    assert(flush_count == 2);
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

/* Guard-page sampling reaches the caller through the Service unchanged. */
static void test_guard_faults(void)
{
    npu_service_device_t device;
    uint32_t fault_class = UINT32_MAX;
    uint32_t fault_offset = UINT32_MAX;
    uint64_t corruption = 0;
    uint64_t guard;

    device_reset(&device);
    assert(npu_service_mmu_init(&device) == 0);
    assert(npu_service_mmu_check_faults(&device, 0, &fault_class,
                                        &fault_offset) == 0);
    guard = fake_reg64(DAYAN_MMU_REDIR_READ_PERM_LO);
    assert(npu_hal_copy_to_address(FAKE_HAL, 0, guard + 0x18, &corruption,
                                   sizeof(corruption)) == 0);
    assert(npu_service_mmu_check_faults(&device, 0, &fault_class,
                                        &fault_offset) == 1);
    assert(fault_class == 1 && fault_offset == 0x18);
    assert(npu_service_mmu_check_faults(&device, 1, 0, 0) == -EINVAL);
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
 * BL1 is reserved at pool_base, so init must also have left pool_base free.
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
    assert(npu_hal_memory_info(FAKE_HAL, buffer, &info) == 0);
    assert(info.address >= FW_POOL_BASE + FW_POOL_SIZE);
    assert(npu_hal_memory_free(FAKE_HAL, buffer) == 0);

    /* BL1's segments are reserved at the pool base; init must not have taken
     * it, or the Server would refuse to start. */
    assert(fake_bl1_reserve() == 0);

    npu_service_mmu_fini(&device);

    /*
     * Re-init on the same device - no device_reset(), because what is under
     * test is exactly the state fini is supposed to have cleared.  Every page
     * the first init took has been freed, so the second one must start at the
     * top of the pool again rather than below the first init's low-water mark.
     */
    assert(npu_service_mmu_init(&device) == 0);
    assert(fake_ptw_base(0) == FW_POOL_BASE + FW_POOL_SIZE - 0x1000);
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
    assert(npu_hal_memory_info(FAKE_HAL, firmware, &info) == 0);
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

    assert(npu_hal_memory_free(FAKE_HAL, buffer) == 0);
    assert(npu_hal_memory_free(FAKE_HAL, firmware) == 0);
    npu_service_mmu_fini(&device);

    /* With the MMU off the physical address is what the Device uses. */
    device_reset(&device);
    device.options.enable_mmu = 0;
    assert(npu_service_mmu_init(&device) == 0);
    assert(npu_service_alloc_visible(&device, viewer, NPU_HAL_MEMORY_NORMAL,
                                     NPU_HAL_MEMORY_BUFFER, 0, 0x1000,
                                     0x1000, &buffer) == 0);
    assert(npu_hal_memory_info(FAKE_HAL, buffer, &info) == 0);
    assert(npu_service_mmu_device_address(&device, buffer, viewer,
                                          &address) == 0);
    assert(address == info.address);
    assert(npu_hal_memory_free(FAKE_HAL, buffer) == 0);
    npu_service_mmu_fini(&device);
}

int main(void)
{
    test_disabled_is_inert();
    test_init_and_teardown();
    test_firmware_window();
    test_buffer_mapping();
    test_va_allocator();
    test_deferred_reclaim();
    test_reclaim_flushes_every_cluster();
    test_guard_faults();
    test_pools_are_disjoint();
    test_firmware_address_conversion();
    test_device_address_translation();
    puts("mmu_test: firmware window, VA allocator, deferred reclaim, address "
         "conversion and pool disjointness OK");
    return 0;
}
