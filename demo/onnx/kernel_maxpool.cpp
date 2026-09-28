// MaxPool2D(NCHW)：按输出行分片，每个输出行按 kOutputTile 个输出点一块处理。
// 每块把 kernel_h 行输入经 LSU 搬进 URAM，在 URAM 里拼出
// [output][padded_window] 窗口矩阵，再用一条 MACRO reduce_max 同时算出这一块
// 所有输出点（DIM0 = 窗口，DIM1 = 输出点数）。越界位置补 -inf。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

namespace {
#if defined(__DAYAN_DEVICE__)

__device__ constexpr uint32_t kOutputTile = 16;
__device__ constexpr uint32_t kMaxKernel = 16;
__device__ constexpr uint32_t kMaxStride = 16;
__device__ constexpr uint32_t kUramBytes = 64U * 1024U;
__device__ constexpr long kLsuGroup = 0;
__device__ constexpr long kMacroGroup = 0;

__device__ static inline float negative_infinity() {
    union FloatBits {
        uint32_t bits;
        float value;
    } value = {0xff800000U};
    return value.value;
}

__device__ static inline uint32_t align_up_64(uint32_t value) {
    return (value + 63U) & ~63U;
}

__device__ static inline void configure_lsu(uint32_t bytes,
                                            uint32_t request_bytes) {
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM0>(bytes / 64U - 1U);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INDIM1STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INDIM2STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(
        request_bytes - 1U);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(
        request_bytes - 1U);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

// 从 64B 对齐的 DRAM 地址搬 bytes 字节到 URAM：整块交给 LSU，尾部零头单独一次。
__device__ static void load_bytes(uint64_t destination, uintptr_t source,
                                  uint32_t bytes) {
    const uint32_t full_bytes = bytes & ~63U;
    const uint32_t remaining_bytes = bytes & 63U;
    if (full_bytes != 0) {
        configure_lsu(full_bytes, 64);
        __tx_lsu_dma_d2u<kLsuGroup>((long)destination, (long)source);
        dayan_wait_lsu();
    }
    if (remaining_bytes != 0) {
        configure_lsu(64, remaining_bytes);
        __tx_lsu_dma_d2u<kLsuGroup>((long)(destination + full_bytes),
                                    (long)(source + full_bytes));
        dayan_wait_lsu();
    }
}

// 窗口归约的 MACRO 配置：DIM0 = 窗口长度，DIM1 = 输出点数。
__device__ static inline void configure_reduce(uint32_t window_elements,
                                               uint32_t outputs) {
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM0>(
        window_elements - 1U);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM1>(outputs - 1U);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM2>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM3>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0DIM1STRIDE>(
        window_elements * (uint32_t)sizeof(float));
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0DIM2STRIDE>(
        window_elements * outputs * (uint32_t)sizeof(float));
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0DIM3STRIDE>(
        window_elements * outputs * (uint32_t)sizeof(float));
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1DIM1STRIDE>(
        (uint32_t)sizeof(float));
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1DIM2STRIDE>(
        outputs * (uint32_t)sizeof(float));
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1DIM3STRIDE>(
        outputs * (uint32_t)sizeof(float));
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTDIM1STRIDE>(
        (uint32_t)sizeof(float));
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTDIM2STRIDE>(
        outputs * (uint32_t)sizeof(float));
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTDIM3STRIDE>(
        outputs * (uint32_t)sizeof(float));
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTSWIZZLEMBS>(0);
}

__device__ static inline uint32_t shard_begin(uint32_t total) {
    const uint32_t tasks = dayan_task_dim(0);
    if (tasks < 2U)
        return 0;
    return (uint32_t)((uint64_t)total * dayan_task_id(0) / tasks);
}

__device__ static inline uint32_t shard_end(uint32_t total) {
    const uint32_t tasks = dayan_task_dim(0);
    if (tasks < 2U)
        return total;
    return (uint32_t)((uint64_t)total * (dayan_task_id(0) + 1U) / tasks);
}

#endif
} // namespace

__kernel__ void rvnpu_maxpool2d_nchw_fp32(
    const float *input, float *output, uint32_t batch, uint32_t channels,
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w,
    uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w,
    uint32_t pad_h, uint32_t pad_w) {
    if (batch == 0 || channels == 0 || output_h == 0 || output_w == 0)
        return;
    if (input == nullptr || output == nullptr || input_h == 0 || input_w == 0 ||
        kernel_h == 0 || kernel_h > kMaxKernel || kernel_w == 0 ||
        kernel_w > kMaxKernel || stride_h == 0 || stride_h > kMaxStride ||
        stride_w == 0 || stride_w > kMaxStride) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    dayan_cache_invalidate(input, (uint64_t)batch*channels*input_h*input_w*sizeof(float));
    dayan_cache_invalidate(output, (uint64_t)batch*channels*output_h*output_w*sizeof(float));
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    if (uram == 0 || dayan_uram_size() < kUramBytes)
        dayan_thread_exit(-2);

    const uint32_t window_elements = kernel_h * kernel_w;
    const uint32_t padded_window = (window_elements + 15U) & ~15U;
    // 一块输出最多覆盖这么宽的输入：其中每行还要多读最多 63 字节的对齐前缀。
    uint32_t widest = (kOutputTile - 1U) * stride_w + kernel_w;
    if (widest > input_w)
        widest = input_w;
    const uint32_t row_stride = align_up_64(63U + widest * sizeof(float));
    const uint64_t windows_at = uram + (uint64_t)kernel_h * row_stride;
    const uint64_t results_at =
        windows_at + (uint64_t)kOutputTile * padded_window * sizeof(float);
    if ((uint64_t)kernel_h * row_stride +
            (uint64_t)kOutputTile * padded_window * sizeof(float) +
            (uint64_t)kOutputTile * sizeof(float) >
        kUramBytes)
        dayan_thread_exit(-2);

    volatile float *windows = (volatile float *)(uintptr_t)windows_at;
    volatile float *results = (volatile float *)(uintptr_t)results_at;

    int32_t valid_width_begin[kMaxKernel];
    uint32_t row_prefix[kMaxKernel];
    bool row_loaded[kMaxKernel];

    const uint32_t planes = batch * channels;
    for (uint32_t plane = shard_begin(planes); plane < shard_end(planes);
         ++plane) {
        const uint64_t plane_offset =
            (uint64_t)plane * input_h * input_w;
        for (uint32_t oh = 0; oh < output_h; ++oh) {
            for (uint32_t output_begin = 0; output_begin < output_w;
                 output_begin += kOutputTile) {
                const uint32_t output_count =
                    output_w - output_begin > kOutputTile
                        ? kOutputTile
                        : output_w - output_begin;
                const int64_t width_begin =
                    (int64_t)output_begin * stride_w - pad_w;
                const int64_t width_end =
                    (int64_t)(output_begin + output_count - 1) * stride_w -
                    pad_w + kernel_w;
                const int32_t clamped_begin =
                    width_begin < 0 ? 0 : (int32_t)width_begin;
                const int32_t clamped_end =
                    width_end < 0 ? 0
                                  : (width_end > input_w ? (int32_t)input_w
                                                         : (int32_t)width_end);

                for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                    const int64_t ih =
                        (int64_t)oh * stride_h + kh - pad_h;
                    row_loaded[kh] =
                        ih >= 0 && ih < input_h && clamped_begin < clamped_end;
                    if (!row_loaded[kh])
                        continue;
                    const uint64_t index =
                        plane_offset + (uint64_t)ih * input_w + clamped_begin;
                    const uintptr_t source = (uintptr_t)(input + index);
                    const uintptr_t aligned = source & ~(uintptr_t)63;
                    const uint32_t prefix = (uint32_t)(source - aligned);
                    const uint32_t span =
                        (uint32_t)(clamped_end - clamped_begin);
                    valid_width_begin[kh] = clamped_begin;
                    row_prefix[kh] = prefix / sizeof(float);
                    load_bytes(uram + (uint64_t)kh * row_stride, aligned,
                               prefix + span * sizeof(float));
                }

                for (uint32_t output_index = 0; output_index < output_count;
                     ++output_index) {
                    const uint32_t ow = output_begin + output_index;
                    uint32_t window_index = 0;
                    for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                        const volatile float *row =
                            (const volatile float *)(uintptr_t)(
                                uram + (uint64_t)kh * row_stride);
                        for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                            const int64_t iw =
                                (int64_t)ow * stride_w + kw - pad_w;
                            float value = negative_infinity();
                            if (row_loaded[kh] && iw >= 0 && iw < input_w) {
                                const uint32_t column =
                                    row_prefix[kh] +
                                    (uint32_t)(iw - valid_width_begin[kh]);
                                value = row[column];
                            }
                            windows[(uint64_t)output_index * padded_window +
                                    window_index++] = value;
                        }
                    }
                    while (window_index < padded_window)
                        windows[(uint64_t)output_index * padded_window +
                                window_index++] = negative_infinity();
                }

                configure_reduce(padded_window, output_count);
                __tx_macro_reduce_max_fp32<kMacroGroup, 0>(
                    (long)results_at, (long)windows_at, 0);
                dayan_wait_macro();

                const uint64_t output_offset =
                    ((uint64_t)plane * output_h + oh) * output_w + output_begin;
                for (uint32_t index = 0; index < output_count; ++index)
                    output[output_offset + index] = results[index];
            }
        }
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(rvnpu_maxpool2d_nchw_fp32, 4, 0, 0, 65536U, 1, 0, 0, 1);
