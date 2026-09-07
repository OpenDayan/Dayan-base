/*
 * ClusterMMU APB4 register access.
 *
 * The MMU register page is reached through the raw address path rather than
 * npu_hal_register_*, because the register window those functions address is
 * anchored at DAYAN_IRQ_STATUS and does not extend far enough to cover the
 * MMU page at either the manual's 0x005000 or QEMU's 0x008000.  The h2c/c2h
 * path has no such window, so copy_to_address/copy_from_address reach it.
 *
 * Two hardware behaviours shape everything here.
 *
 * A 56-bit address is written as a LO/HI pair, and a LO half bound to a
 * translation window must be 4 KiB aligned.  A misaligned write is silently
 * ignored and does not raise an APB error, so a caller would otherwise see
 * success while the register kept its old value.  Every such write is
 * checked here first.
 *
 * Writing TLB_FLUSH while any AXI port still has traffic in flight hangs the
 * Cluster: the drain never finishes and FLUSH_BUSY never releases.  There is
 * no quiesce register on this hardware - the manual (section 2.2.8 clause 2)
 * says software must stop the ports from the NPU core side - so
 * npu_hal_mmu_flush() cannot make itself safe.  Its contract, stated in
 * hal.h, is that the caller has already established that the Cluster is
 * idle.  Getting that wrong does not return an error, it wedges the machine.
 */

#include "hal_internal.h"

#include <errno.h>
#include <string.h>
#include <time.h>

#include "platform_regs.h"

#define NPU_HAL_MMU_POLL_US      100U

static int mmu_offset_valid(uint32_t offset)
{
    return offset < DAYAN_MMU_APB_SIZE && !(offset & 3U);
}

/*
 * Whether a LO half at this offset feeds a translation window and therefore
 * carries the 4 KiB alignment constraint.  PTW_BASE is excluded: the walker
 * base is a page-table pointer, and page tables are page-aligned by
 * construction, so the hardware imposes no separate rule on it.
 */
static int mmu_offset_needs_page_align(uint32_t offset)
{
    if (offset == DAYAN_MMU_FIX_MAP_BASE_LO ||
        offset == DAYAN_MMU_ERROR_SLV_MASK_LO)
        return 1;
    /* The six redirect bases are LO/HI pairs on an 8-byte stride, so the LO
     * halves are exactly the offsets where the stride bit is clear. */
    return offset >= DAYAN_MMU_REDIR_PAGE_FAULT_LO &&
           offset <= DAYAN_MMU_REDIR_DEFAULT_LO &&
           ((offset - DAYAN_MMU_REDIR_PAGE_FAULT_LO) %
            DAYAN_MMU_REDIR_CLASS_STRIDE) == 0;
}

static void mmu_delay_us(uint32_t delay_us)
{
    struct timespec request;

    request.tv_sec = (time_t)(delay_us / 1000000U);
    request.tv_nsec = (long)(delay_us % 1000000U) * 1000L;
    while (nanosleep(&request, &request) != 0) {
        /* Retry on EINTR; any other failure means the delay is over. */
        if (errno != EINTR)
            break;
    }
}

static int mmu_monotonic_ms(uint64_t *value)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return -errno;
    *value = (uint64_t)now.tv_sec * 1000U + (uint64_t)now.tv_nsec / 1000000U;
    return 0;
}

int npu_hal_mmu_read_reg(npu_hal_t *hal, uint32_t cluster, uint32_t offset,
                         uint32_t *value)
{
    if (!hal || !value || !mmu_offset_valid(offset))
        return -EINVAL;
    return npu_hal_copy_from_address(hal, cluster, value,
                                     (uint64_t)DAYAN_MMU_APB_BASE + offset,
                                     sizeof(*value));
}

int npu_hal_mmu_write_reg(npu_hal_t *hal, uint32_t cluster, uint32_t offset,
                          uint32_t value)
{
    if (!hal || !mmu_offset_valid(offset))
        return -EINVAL;
    /*
     * Reject what the hardware would silently drop.  Returning -EINVAL is
     * the whole point: without it the write "succeeds" and the register
     * quietly keeps its previous value.
     */
    if (mmu_offset_needs_page_align(offset) &&
        (value & DAYAN_MMU_PAGE_MASK))
        return -EINVAL;
    return npu_hal_copy_to_address(hal, cluster,
                                   (uint64_t)DAYAN_MMU_APB_BASE + offset,
                                   &value, sizeof(value));
}

int npu_hal_mmu_write64(npu_hal_t *hal, uint32_t cluster, uint32_t offset_lo,
                        uint64_t value)
{
    uint32_t low = (uint32_t)(value & UINT32_C(0xffffffff));
    uint32_t high = (uint32_t)((value >> 32) & DAYAN_MMU_PA_HI_MASK);
    int status;

    if (!hal || !mmu_offset_valid(offset_lo) ||
        !mmu_offset_valid(offset_lo + 4U))
        return -EINVAL;
    /* Anything above bit 55 cannot be expressed and must not be dropped
     * silently, or a caller would map to an address it never asked for. */
    if (value >> DAYAN_MMU_PA_BITS)
        return -EINVAL;
    /* LO first, then HI: the alignment check on LO happens before either
     * half lands, so a rejected write leaves the pair untouched. */
    status = npu_hal_mmu_write_reg(hal, cluster, offset_lo, low);
    if (status < 0)
        return status;
    return npu_hal_mmu_write_reg(hal, cluster, offset_lo + 4U, high);
}

int npu_hal_mmu_read64(npu_hal_t *hal, uint32_t cluster, uint32_t offset_lo,
                       uint64_t *value)
{
    uint32_t low = 0;
    uint32_t high = 0;
    int status;

    if (!hal || !value)
        return -EINVAL;
    status = npu_hal_mmu_read_reg(hal, cluster, offset_lo, &low);
    if (status < 0)
        return status;
    status = npu_hal_mmu_read_reg(hal, cluster, offset_lo + 4U, &high);
    if (status < 0)
        return status;
    *value = ((uint64_t)(high & DAYAN_MMU_PA_HI_MASK) << 32) | low;
    return 0;
}

/*
 * Probe with a read only.  Writing FIX_MAP_SIZE changes address routing before
 * PTW_BASE and the page tables are ready, which can make the following Host
 * read time out on real hardware.  An absent MMU page is reported by the
 * backend read itself.
 */
int npu_hal_mmu_present(npu_hal_t *hal, uint32_t cluster, int *present)
{
    uint32_t value = 0;
    int status;

    if (!hal || !present)
        return -EINVAL;
    *present = 0;
    status = npu_hal_mmu_read_reg(hal, cluster, DAYAN_MMU_FIX_MAP_SIZE_LO,
                                  &value);
    if (status < 0)
        return status;
    *present = 1;
    return 0;
}

int npu_hal_mmu_flush(npu_hal_t *hal, uint32_t cluster, int asid_selective,
                      uint8_t asid, uint32_t timeout_ms)
{
    uint32_t command = DAYAN_MMU_TLB_FLUSH_GO;
    uint32_t status_word = 0;
    uint64_t deadline = 0;
    uint64_t now = 0;
    int status;

    if (!hal)
        return -EINVAL;
    if (asid_selective) {
        if (asid >= DAYAN_MMU_ASID_COUNT)
            return -EINVAL;
        command |= DAYAN_MMU_TLB_FLUSH_ASID_SEL |
                   ((uint32_t)asid << DAYAN_MMU_TLB_FLUSH_ASID_SHIFT);
    }

    status = mmu_monotonic_ms(&now);
    if (status < 0)
        return status;
    deadline = now + timeout_ms;

    /*
     * From here on the Cluster is committed: if the caller was wrong about
     * the ports being idle, the drain below never completes and no amount of
     * error handling here recovers it.  The timeout exists so the caller can
     * escalate to a reset rather than block forever.
     */
    status = npu_hal_mmu_write_reg(hal, cluster, DAYAN_MMU_TLB_FLUSH,
                                   command);
    if (status < 0)
        return status;

    for (;;) {
        status = npu_hal_mmu_read_reg(hal, cluster, DAYAN_MMU_FLUSH_BUSY,
                                      &status_word);
        if (status < 0)
            return status;
        if (!(status_word & DAYAN_MMU_FLUSH_BUSY_BIT))
            break;
        status = mmu_monotonic_ms(&now);
        if (status < 0)
            return status;
        if (now >= deadline)
            return -ETIMEDOUT;
        mmu_delay_us(NPU_HAL_MMU_POLL_US);
    }

    status = npu_hal_mmu_read_reg(hal, cluster, DAYAN_MMU_FLUSH_ERR_CODE,
                                  &status_word);
    if (status < 0)
        return status;
    if ((status_word & DAYAN_MMU_FLUSH_ERR_MASK) ==
        DAYAN_MMU_FLUSH_ERR_BUSY_REJECT)
        /* The flush was refused, not performed: a previous one was still
         * running.  Nothing was invalidated, so the caller must retry. */
        return -EBUSY;
    if (status_word & DAYAN_MMU_FLUSH_ERR_MASK)
        return -EIO;

    status = npu_hal_mmu_read_reg(hal, cluster, DAYAN_MMU_FLUSH_DONE,
                                  &status_word);
    if (status < 0)
        return status;
    if (!(status_word & DAYAN_MMU_FLUSH_DONE_BIT))
        return -EIO;

    /* DONE is sticky; leaving it set would make the next flush's completion
     * check pass before that flush had actually finished. */
    status = npu_hal_mmu_write_reg(hal, cluster, DAYAN_MMU_FLUSH_ACK,
                                   DAYAN_MMU_FLUSH_ACK_BIT);
    if (status < 0)
        return status;
    status = npu_hal_mmu_read_reg(hal, cluster, DAYAN_MMU_FLUSH_DONE,
                                  &status_word);
    if (status < 0)
        return status;
    return (status_word & DAYAN_MMU_FLUSH_DONE_BIT) ? -EIO : 0;
}
