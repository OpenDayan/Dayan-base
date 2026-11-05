/*
 * ClusterMMU policy: what gets mapped where, and when it is safe to flush.
 *
 * The page tables and guard pages themselves are the HAL's (hal/mmu_pt.c),
 * which knows the hardware's table format and nothing about Contexts.  This
 * file decides everything that depends on the rest of the Service.
 *
 * A Cluster has one page-table walker base and one ASID.  There is no
 * per-Context page table and no per-Context ASID (only 8 ASIDs exist and all
 * ports share GLOBAL_ASID), so Contexts cannot be isolated by the MMU.  They
 * are isolated by never handing two Contexts overlapping virtual addresses on
 * the one shared table.  The VA allocator is a bump pointer that never
 * recycles, which is what makes that guarantee cheap to hold.
 *
 * The firmware window is a linear-offset mapping whose virtual base is fixed
 * by the firmware's linker script and whose physical base is the HAL's
 * firmware pool.  It is installed once at init and nothing else may touch it.
 *
 * Writing TLB_FLUSH while any AXI port still has traffic in flight hangs the
 * Cluster permanently, and there is no quiesce register to prevent it.  So
 * unmapping never flushes inline: it clears the PTEs, parks the backing
 * memory on a reclaim list, and the flush happens later from the one place
 * that knows the device is idle.  QEMU's flush is synchronous and cannot
 * reproduce the hang, so this ordering is not something a test will catch.
 */

#include "service_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- */
/* Init and teardown                                                 */
/* ---------------------------------------------------------------- */

/*
 * Map the firmware window on every Cluster:
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
 * The HAL refuses a pool that is not a whole number of 2 MiB leaves rather
 * than mapping it short.  The page tables and guard pages are carved from the
 * top of this same pool, so the window covers them too.
 */
static int npu_mmu_map_firmware_window(npu_service_device_t *device)
{
    struct npu_mmu *mmu = &device->mmu;
    uint32_t cluster;
    int status;

    for (cluster = 0; cluster < mmu->cluster_count; ++cluster) {
        status = npu_hal_mmu_map(mmu->tables, cluster,
                                 NPU_SERVICE_MMU_FW_VA_BASE,
                                 mmu->fw_pool_base, mmu->fw_pool_size,
                                 NPU_SERVICE_MMU_READ | NPU_SERVICE_MMU_WRITE |
                                 NPU_SERVICE_MMU_EXEC);
        if (status < 0)
            return status;
    }
    return 0;
}

int npu_service_mmu_init(npu_service_device_t *device)
{
    struct npu_mmu *mmu = &device->mmu;
    int status;

    pthread_mutex_init(&mmu->lock, 0);
    mmu->cluster_count = device->hal_info.cluster_count;
    mmu->va_limit = NPU_HAL_MMU_VA_LIMIT;

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

    /*
     * Tables and guard pages come from the top of the firmware pool.  It has
     * to be that pool, because it is the range the Device can reach and the
     * one Buffers never come from; and it has to be the top, because BL1's
     * segments are reserved at pool_base + (link address - CONFIG_RAM_BASE),
     * which starts at the pool base itself.  -ENODEV means the MMU was asked
     * for but is not there, and identity addressing stays.
     */
    status = npu_hal_mmu_create(device->hal, mmu->cluster_count,
                                mmu->fw_pool_base, mmu->fw_pool_size,
                                &mmu->tables);
    if (status < 0)
        return status;
    /* Mapping helpers check mmu->enabled, so it must be set before the
     * firmware window below is built. */
    mmu->enabled = 1;

    /*
     * The order is the safety argument: every Cluster's table is complete
     * before FixMap is narrowed, or the first access above 2 GiB walks an
     * empty table.
     */
    status = npu_mmu_map_firmware_window(device);
    if (status == 0)
        status = npu_hal_mmu_activate(mmu->tables);
    if (status < 0)
        npu_service_mmu_fini(device);
    return status;
}

/*
 * Teardown in the reverse order: return to identity addressing first, so
 * nothing is translating while the reclaim list drains and the tables go.
 */
void npu_service_mmu_fini(npu_service_device_t *device)
{
    struct npu_mmu *mmu = &device->mmu;

    if (mmu->enabled) {
        (void)npu_hal_mmu_deactivate(mmu->tables);
        /* FixMap is back on, so no TLB entry can be consulted; the flush
         * would be redundant and this path may run with launches unaccounted
         * for. */
        (void)npu_service_mmu_run_reclaim(device, 1);
        mmu->enabled = 0;
    }
    npu_hal_mmu_destroy(mmu->tables);
    mmu->tables = 0;
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
    aligned = (size + NPU_SERVICE_MMU_PAGE_MASK) &
              ~(uint64_t)NPU_SERVICE_MMU_PAGE_MASK;
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

/*
 * Whether [va, va + size) touches the firmware window.  Its leaves are the
 * same 2 MiB level-1 entries a Buffer uses, so nothing but the window's own
 * setup may write them: map would alias a firmware page, and unmap or protect
 * would pull the firmware's own memory out from under it.
 */
static int npu_mmu_check_range(const npu_service_device_t *device,
                               uint32_t cluster, uint64_t va, uint64_t size)
{
    const struct npu_mmu *mmu = &device->mmu;

    if (cluster >= device->hal_info.cluster_count)
        return -EINVAL;
    if (va > UINT64_MAX - size)
        return -EOVERFLOW;
    if (va < NPU_SERVICE_MMU_FW_VA_BASE + mmu->fw_pool_size &&
        va + size > NPU_SERVICE_MMU_FW_VA_BASE)
        return -EINVAL;
    return 0;
}

/*
 * No flush on map: a newly valid entry cannot be stale in the TLB, because
 * nothing has translated that address before.
 */
int npu_service_mmu_map(npu_service_device_t *device, uint32_t cluster,
                        uint64_t va, uint64_t pa, uint64_t size, uint32_t prot)
{
    int status;

    if (!device || cluster >= device->hal_info.cluster_count)
        return -EINVAL;
    if (!device->mmu.enabled)
        return 0;
    status = npu_mmu_check_range(device, cluster, va, size);
    if (status < 0)
        return status;
    return npu_hal_mmu_map(device->mmu.tables, cluster, va, pa, size, prot);
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
    int status;

    if (!device || cluster >= device->hal_info.cluster_count)
        return -EINVAL;
    if (!device->mmu.enabled)
        return 0;
    status = npu_mmu_check_range(device, cluster, va, size);
    if (status < 0)
        return status;
    return npu_hal_mmu_unmap(device->mmu.tables, cluster, va, size);
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
    int status;

    if (!device || cluster >= device->hal_info.cluster_count)
        return -EINVAL;
    if (!device->mmu.enabled)
        return 0;
    status = npu_mmu_check_range(device, cluster, va, size);
    if (status < 0)
        return status;
    return npu_hal_mmu_protect(device->mmu.tables, cluster, va, size, prot);
}

/* Sampling only; see npu_hal_mmu_check_faults(). */
int npu_service_mmu_check_faults(npu_service_device_t *device,
                                 uint32_t cluster, uint32_t *fault_class,
                                 uint32_t *fault_offset)
{
    if (!device || cluster >= device->hal_info.cluster_count)
        return -EINVAL;
    if (!device->mmu.enabled)
        return 0;
    return npu_hal_mmu_check_faults(device->mmu.tables, cluster, fault_class,
                                    fault_offset);
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
