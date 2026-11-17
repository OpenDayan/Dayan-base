// ReduceMean for NCHW spatial axes {2,3}.
// Each logical task owns complete channels; LSU reads exact 64-byte requests
// plus a short tail and MACRO performs the FP32 reduction without padding.
#include <dayan_kernel.h>
#include <stdint.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace {
enum : uint64_t {
    kRequestElements = 16,
    kTileElements = 256,
    kTileBytes = kTileElements * sizeof(float),
    kUramBytes = 32 * 1024,
};

static long fp32_bits(float value)
{
    union {
        float f;
        uint32_t u;
    } bits = {value};
    return (long)bits.u;
}

static void configure_lsu(uint64_t elements)
{
    const uint64_t requests =
        (elements + kRequestElements - 1U) / kRequestElements;
    const uint64_t request_elements =
        elements < kRequestElements ? elements : kRequestElements;
    __tx_lsu_tcsr_write<LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<LSU_DIM0>(requests - 1U);
    __tx_lsu_tcsr_write<LSU_DIM1>(0);
    __tx_lsu_tcsr_write<LSU_DIM2>(0);
    __tx_lsu_tcsr_write<LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(kTileBytes);
    __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM1STRIDE>(kTileBytes);
    __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTPADDING>(
        request_elements * sizeof(float) - 1U);
    __tx_lsu_tcsr_write<LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(
        request_elements * sizeof(float) - 1U);
    __tx_lsu_tcsr_write<LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<LSU_TABLESTRIDE>(64);
}

static void configure_macro_scalar()
{
    __tx_macro_tcsr_write<MACRO_DIM0>(0);
    __tx_macro_tcsr_write<MACRO_DIM1>(0);
    __tx_macro_tcsr_write<MACRO_DIM2>(0);
    __tx_macro_tcsr_write<MACRO_DIM3>(0);
    __tx_macro_tcsr_write<MACRO_SRC0DIM1STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC0DIM2STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC0DIM3STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC1DIM1STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC1DIM2STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC1DIM3STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_DSTDIM1STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_DSTDIM2STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_DSTDIM3STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<MACRO_DSTSWIZZLEMBS>(0);
}

static void configure_macro_reduce(uint64_t elements)
{
    const uint64_t stride = elements * sizeof(float);
    __tx_macro_tcsr_write<MACRO_DIM0>(elements - 1U);
    __tx_macro_tcsr_write<MACRO_DIM1>(0);
    __tx_macro_tcsr_write<MACRO_DIM2>(0);
    __tx_macro_tcsr_write<MACRO_DIM3>(0);
    __tx_macro_tcsr_write<MACRO_SRC0DIM1STRIDE>(stride);
    __tx_macro_tcsr_write<MACRO_SRC0DIM2STRIDE>(stride);
    __tx_macro_tcsr_write<MACRO_SRC0DIM3STRIDE>(stride);
    __tx_macro_tcsr_write<MACRO_SRC1DIM1STRIDE>(sizeof(float));
    __tx_macro_tcsr_write<MACRO_SRC1DIM2STRIDE>(sizeof(float));
    __tx_macro_tcsr_write<MACRO_SRC1DIM3STRIDE>(sizeof(float));
    __tx_macro_tcsr_write<MACRO_DSTDIM1STRIDE>(sizeof(float));
    __tx_macro_tcsr_write<MACRO_DSTDIM2STRIDE>(sizeof(float));
    __tx_macro_tcsr_write<MACRO_DSTDIM3STRIDE>(sizeof(float));
    __tx_macro_tcsr_write<MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<MACRO_DSTSWIZZLEMBS>(0);
}
} // namespace
#endif

__kernel__ void dayan_reduce_mean_nchw_fp32(const float *input, float *output,
                                            uint64_t channels, uint64_t hw)
{
#if defined(__DAYAN_DEVICE__)
    if (!input || !output || !channels || !hw ||
        !dayan_task_dim(DAYAN_TASK_X))
        dayan_thread_exit(-22);
    dayan_cache_invalidate(input, channels * hw * sizeof(float));
    dayan_cache_invalidate(output, channels * sizeof(float));
    const uint64_t task = dayan_task_id(DAYAN_TASK_X);
    const uint64_t tasks = dayan_task_dim(DAYAN_TASK_X);
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    if (!uram || dayan_uram_size() < kUramBytes)
        dayan_thread_exit(-12);
    const long tile = (long)uram;
    const long partial = (long)(uram + kTileBytes);
    const long sum = (long)(uram + 2U * kTileBytes);

    for (uint64_t channel = task; channel < channels; channel += tasks) {
        const uint64_t plane = channel * hw;
        *(volatile float *)(uintptr_t)sum = 0.0f;
        for (uint64_t offset = 0; offset < hw;
             offset += kRequestElements) {
            const uint64_t remaining = hw - offset;
            const uint64_t count = remaining < kRequestElements
                                       ? remaining
                                       : kRequestElements;
            configure_lsu(count);
            __tx_lsu_dma_d2u<0>(
                tile, (long)((uintptr_t)input +
                             (plane + offset) * sizeof(float)));
            dayan_wait_lsu();
            configure_macro_reduce(count);
            __tx_macro_reduce_add_fp32<0>(partial, tile);
            dayan_wait_macro();
            configure_macro_scalar();
            __tx_macro_elemwise_add_fp32<0, 0>(sum, sum, partial);
            dayan_wait_macro();
        }
        configure_macro_scalar();
        __tx_macro_elemwise_mul_fp32<0, 2>(
            sum, fp32_bits(1.0f / (float)hw), sum);
        dayan_wait_macro();
        configure_lsu(1);
        __tx_lsu_dma_u2d<0>(
            (long)((uintptr_t)output + channel * sizeof(float)), sum);
        dayan_wait_lsu();
    }
#endif
}

// One resident reduction Proc per Core avoids unsafe same-Hart MACRO reduce
// context sharing while the four-Core launch still partitions channels.
DAYAN_KERNEL_TASK_DESC(dayan_reduce_mean_nchw_fp32, 1, 0, 0, 32768U,
                       1, 0, 0, 1);
