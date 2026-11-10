// MaxPool2D(NCHW)。三条路径：
//  5x5/s1/p2 dense：二维行块装载，水平 5-max，垂直 MACRO 合并归约，
//    每 plane 少量 D2U/U2D，避免逐行小 DMA。
//  3x3/s2/p1 tile：plane 行连续 packed 装载（整块+尾块 2 条，无越界读），
//    水平 s2 3-max 主循环无分支（边界 2 点特化），垂直 MACRO，写回 2 条；
//    装载(g0) 与垂直/写回(g1) 跨 tile 流水。
//  通用：窗口矩阵 + reduce_max（保持原样）。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

namespace {
#if defined(__DAYAN_DEVICE__)

__device__ constexpr uint32_t kOutputTile = 64;
__device__ constexpr uint32_t kMaxKernel = 16;
__device__ constexpr uint32_t kMaxStride = 16;
__device__ constexpr uint32_t kOutputRowTile = 16;
__device__ constexpr uint32_t kUramBytes = 64U * 1024U;
__device__ constexpr long kLsuGroup = 0;
__device__ constexpr long kStoreGroup = 1;
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

// 双维块搬运：每行 dim0_blocks 个 64B 块，行数 dim1_rows；request 裁块尾。
// 组别可挑（装载 g0 / 写回 g1），地址与步长均允许非 64B 对齐（Transpose=6）。
template <long GROUP>
__device__ static inline void configure_lsu_dims(uint32_t dim0_blocks,
                                                 uint32_t dim1_rows,
                                                 uint32_t in_dim1_stride,
                                                 uint32_t out_dim1_stride,
                                                 uint32_t request_bytes) {
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM0>(dim0_blocks - 1U);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM1>(dim1_rows - 1U);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INDIM1STRIDE>(in_dim1_stride);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_INDIM2STRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(out_dim1_stride);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(64);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(
        request_bytes - 1U);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(
        request_bytes - 1U);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<GROUP * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

// packed 连续装载 rows 行（源 DRAM 行紧排）：整块条 + 尾块条，皆无越界。
__device__ static void load_packed_issue(uint64_t destination, uintptr_t source,
                                         uint32_t rows, uint32_t row_bytes) {
    const uint64_t total = (uint64_t)rows * row_bytes;
    const uint32_t full = (uint32_t)(total / 64U);
    const uint32_t tail = (uint32_t)(total % 64U);
    if (full != 0U) {
        configure_lsu_dims<kLsuGroup>( full, 1, 64, 64, 64);
        __tx_lsu_dma_d2u<kLsuGroup>((long)destination, (long)source);
    }
    if (tail != 0U) {
        configure_lsu_dims<kLsuGroup>( 1, 1, 64, 64, tail);
        __tx_lsu_dma_d2u<kLsuGroup>((long)(destination + (uint64_t)full * 64U),
                                    (long)(source + (uint64_t)full * 64U));
    }
}

// 行循环写回 rows 行：每行 full 块 + tail 字节，两条件各自带 DIM1 行步长。
__device__ static void store_rows_issue(uint64_t source, uintptr_t destination,
                                        uint32_t rows, uint32_t row_bytes,
                                        uint32_t source_stride) {
    const uint32_t full = row_bytes / 64U;
    const uint32_t tail = row_bytes % 64U;
    if (full != 0U) {
        configure_lsu_dims<kStoreGroup>( full, rows, source_stride, row_bytes, 64);
        __tx_lsu_dma_u2d<kStoreGroup>((long)destination, (long)source);
    }
    if (tail != 0U) {
        configure_lsu_dims<kStoreGroup>( 1, rows, source_stride, row_bytes, tail);
        __tx_lsu_dma_u2d<kStoreGroup>(
            (long)(destination + (uint64_t)full * 64U),
            (long)(source + (uint64_t)full * 64U));
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

__device__ static inline void configure_vector_rows(
    uint32_t elements, uint32_t rows, uint32_t source0_row_stride,
    uint32_t source1_row_stride, uint32_t destination_row_stride) {
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM0>(
        elements - 1U);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM1>(rows - 1U);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM2>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM3>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0DIM1STRIDE>(
        source0_row_stride);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0DIM2STRIDE>(
        0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0DIM3STRIDE>(
        0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1DIM1STRIDE>(
        source1_row_stride);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1DIM2STRIDE>(
        0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1DIM3STRIDE>(
        0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTDIM1STRIDE>(
        destination_row_stride);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTDIM2STRIDE>(
        0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTDIM3STRIDE>(
        0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTSWIZZLEMBS>(0);
}

__device__ static inline void vector_max_rows(
    uint64_t destination, uint64_t source0, uint64_t source1,
    uint32_t elements, uint32_t rows, uint32_t source0_row_stride,
    uint32_t source1_row_stride, uint32_t destination_row_stride) {
    if (elements == 0 || rows == 0)
        return;
    configure_vector_rows(elements, rows, source0_row_stride,
                          source1_row_stride, destination_row_stride);
    __tx_macro_elemwise_max_fp32<kMacroGroup, 0>(
        (long)destination, (long)source0, (long)source1);
    dayan_wait_macro();
}

// 水平 s2 3-max（去分支）：主循环读 s[2ow-1..2ow+1]，ow=0 与右端点各特化。
__device__ static void horizontal_max3_stride2(const volatile float *s,
                                               volatile float *d,
                                               uint32_t output_w,
                                               uint32_t input_w) {
    if (output_w == 0)
        return;
    float left = s[0];
    if (1U < input_w) {
        const float v = s[1];
        left = v > left ? v : left;
    }
    d[0] = left;
    for (uint32_t ow = 1U; ow + 1U < output_w; ++ow) {
        const uint32_t base = 2U * ow;
        const float a = s[base - 1U];
        const float b = s[base];
        const float c = s[base + 1U];
        float m = a > b ? a : b;
        d[ow] = c > m ? c : m;
    }
    if (output_w > 1U) {
        const uint32_t base = 2U * (output_w - 1U);
        const float a = s[base - 1U];
        const float b = s[base];
        float m = a > b ? a : b;
        if (base + 1U < input_w) {
            const float c = s[base + 1U];
            m = c > m ? c : m;
        }
        d[output_w - 1U] = m;
    }
}

// Vectorized ResNet-style 3x3/s2/p1 pooling, tile 流水版。
__device__ static bool maxpool3x3s2p1_vector(
    const float *input, float *output, uint32_t batch, uint32_t channels,
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w,
    uint64_t uram) {
    const uint32_t input_row_bytes = input_w * sizeof(float);
    const uint32_t output_row_bytes = output_w * sizeof(float);
    // packed 装载：行在 URAM 里按实距排列（读侧同为实距）；hmax/结果行保持
    // 64B 对齐 stride 以满足 MACRO 源 32B 对齐约束。
    const uint32_t input_row_stride = input_row_bytes;
    const uint32_t horizontal_row_stride = align_up_64(output_row_bytes);
    const uint32_t result_row_stride = horizontal_row_stride;
    const uint32_t maximum_input_rows = 2U * kOutputRowTile + 1U;
    const uint64_t input_rows_at = uram;
    // packed 行距（实距）后区域起点不再天然对齐，显式取整保 MACRO 32B 对齐。
    const uint64_t horizontal_at = (input_rows_at +
                                    (uint64_t)maximum_input_rows * input_row_stride +
                                    63U) & ~(uint64_t)63U;
    const uint64_t result_at =
        horizontal_at + (uint64_t)maximum_input_rows * horizontal_row_stride;
    const uint64_t required =
        (uint64_t)maximum_input_rows * (input_row_bytes + 63U) +
        (uint64_t)maximum_input_rows * horizontal_row_stride +
        (uint64_t)kOutputRowTile * result_row_stride;
    if (required > kUramBytes)
        return false;

    const uint32_t planes = batch * channels;
    for (uint32_t plane = shard_begin(planes); plane < shard_end(planes);
         ++plane) {
        const uint64_t input_plane = (uint64_t)plane * input_h * input_w;
        const uint64_t output_plane = (uint64_t)plane * output_h * output_w;
        for (uint32_t output_first = 0; output_first < output_h;
             output_first += kOutputRowTile) {
            const uint32_t output_rows =
                output_h - output_first < kOutputRowTile
                    ? output_h - output_first
                    : kOutputRowTile;
            int64_t input_first_signed = (int64_t)output_first * 2 - 1;
            int64_t input_end_signed =
                (int64_t)(output_first + output_rows - 1U) * 2 + 2;
            const uint32_t input_first =
                input_first_signed < 0 ? 0U : (uint32_t)input_first_signed;
            const uint32_t input_end =
                input_end_signed > input_h ? input_h : (uint32_t)input_end_signed;
            const uint32_t input_rows = input_end - input_first;

            load_packed_issue(input_rows_at,
                              (uintptr_t)(input + input_plane +
                                          (uint64_t)input_first * input_w),
                              input_rows, input_row_bytes);
            dayan_wait_lsu();
            for (uint32_t row = 0; row < input_rows; ++row) {
                horizontal_max3_stride2(
                    (const volatile float *)(uintptr_t)(
                        input_rows_at + (uint64_t)row * input_row_stride),
                    (volatile float *)(uintptr_t)(
                        horizontal_at + (uint64_t)row * horizontal_row_stride),
                    output_w, input_w);
            }

            uint32_t vertical_begin = 0;
            while (vertical_begin < output_rows &&
                   (int64_t)(output_first + vertical_begin) * 2 - 1 < 0)
                ++vertical_begin;
            uint32_t vertical_end = vertical_begin;
            while (vertical_end < output_rows &&
                   (uint64_t)(output_first + vertical_end) * 2U + 1U < input_h)
                ++vertical_end;

            for (uint32_t row = 0; row < output_rows; ++row) {
                if (row >= vertical_begin && row < vertical_end)
                    continue;
                volatile float *destination =
                    (volatile float *)(uintptr_t)(
                        result_at + (uint64_t)row * result_row_stride);
                const int64_t first =
                    (int64_t)(output_first + row) * 2 - 1;
                for (uint32_t ow = 0; ow < output_w; ++ow) {
                    float maximum = negative_infinity();
                    for (uint32_t kh = 0; kh < 3U; ++kh) {
                        const int64_t ih = first + kh;
                        if (ih < 0 || ih >= input_h)
                            continue;
                        const volatile float *source =
                            (const volatile float *)(uintptr_t)(
                                horizontal_at +
                                (uint64_t)((uint32_t)ih - input_first) *
                                    horizontal_row_stride);
                        if (source[ow] > maximum)
                            maximum = source[ow];
                    }
                    destination[ow] = maximum;
                }
            }

            const uint32_t vertical_rows = vertical_end - vertical_begin;
            if (vertical_rows != 0) {
                const uint32_t first_input_row =
                    (uint32_t)((int64_t)(output_first + vertical_begin) * 2 - 1) -
                    input_first;
                const uint64_t source = horizontal_at +
                    (uint64_t)first_input_row * horizontal_row_stride;
                const uint64_t destination = result_at +
                    (uint64_t)vertical_begin * result_row_stride;
                vector_max_rows(
                    destination, source, source + horizontal_row_stride,
                    output_w, vertical_rows, 2U * horizontal_row_stride,
                    2U * horizontal_row_stride, result_row_stride);
                vector_max_rows(
                    destination, destination,
                    source + 2U * horizontal_row_stride,
                    output_w, vertical_rows, result_row_stride,
                    2U * horizontal_row_stride, result_row_stride);
            }

            store_rows_issue(result_at,
                             (uintptr_t)(output + output_plane +
                                         (uint64_t)output_first * output_w),
                             output_rows, output_row_bytes, result_row_stride);
            dayan_wait_lsu();
        }
    }
    return true;
}

// SPPF 5x5/s1/p2：按二维行块搬入 URAM，水平阶段用标量比较，
// 垂直阶段使用 MACRO 合并归约，最后整 plane 写回。
__device__ static bool maxpool5x5s1p2_dense(
    const float *input, float *output, uint32_t batch, uint32_t channels,
    uint32_t height, uint32_t width, uint64_t uram) {
    const uint32_t row_bytes = width * sizeof(float);
    const uint64_t plane_bytes = (uint64_t)height * row_bytes;
    const uint32_t input_pitch = align_up_64(row_bytes);
    const uint64_t horizontal_at = uram + plane_bytes;
    const uint64_t result_at =
        horizontal_at + (uint64_t)height * input_pitch;
    const uint64_t output_at =
        (result_at + (uint64_t)height * input_pitch + 63U) &
        ~(uint64_t)63U;
    if (row_bytes == 0U || plane_bytes == 0U ||
        output_at + plane_bytes > uram + kUramBytes)
        return false;
    const uint32_t planes = batch * channels;
    for (uint32_t plane = shard_begin(planes); plane < shard_end(planes);
         ++plane) {
        const uint64_t base = (uint64_t)plane * height * width;
        volatile float *destination =
            (volatile float *)(uintptr_t)output_at;
        const uint32_t full_row_bytes = row_bytes & ~63U;
        const uint32_t tail_row_bytes = row_bytes & 63U;
        if (full_row_bytes != 0U) {
            configure_lsu_dims<kLsuGroup>(
                full_row_bytes / 64U, height, row_bytes, row_bytes, 64U);
            __tx_lsu_dma_d2u<kLsuGroup>(
                (long)uram, (long)(input + base));
        }
        if (tail_row_bytes != 0U) {
            configure_lsu_dims<kLsuGroup>(
                1U, height, row_bytes, row_bytes, tail_row_bytes);
            __tx_lsu_dma_d2u<kLsuGroup>(
                (long)(uram + full_row_bytes),
                (long)(input + base + full_row_bytes / sizeof(float)));
        }
        dayan_wait_lsu();

        for (uint32_t row = 0; row < height; ++row) {
            for (uint32_t ow = 0; ow < width; ++ow) {
                float maximum = negative_infinity();
                const uint32_t begin = ow > 2U ? ow - 2U : 0U;
                const uint32_t end =
                    ow + 3U < width ? ow + 3U : width;
                const volatile float *source =
                    (const volatile float *)(uintptr_t)(
                        uram + (uint64_t)row * row_bytes);
                for (uint32_t iw = begin; iw < end; ++iw) {
                    const float value = source[iw];
                    maximum = value > maximum ? value : maximum;
                }
                ((volatile float *)(uintptr_t)(
                    horizontal_at + (uint64_t)row * input_pitch))[ow] =
                    maximum;
            }
        }

        const uint32_t interior_rows = height > 4U ? height - 4U : 0U;
        if (interior_rows != 0U) {
            configure_vector_rows(width, interior_rows, input_pitch,
                                  input_pitch, input_pitch);
            __tx_macro_elemwise_max_fp32<kMacroGroup, 0>(
                (long)result_at, (long)horizontal_at, (long)horizontal_at);
            for (uint32_t offset = 1U; offset < 5U; ++offset) {
                __tx_macro_elemwise_max_fp32<kMacroGroup, 0>(
                    (long)result_at, (long)result_at,
                    (long)(horizontal_at +
                           (uint64_t)offset * input_pitch));
            }
            dayan_wait_macro();
            for (uint32_t row = 0; row < interior_rows; ++row) {
                const volatile float *source =
                    (const volatile float *)(uintptr_t)(
                        result_at + (uint64_t)row * input_pitch);
                for (uint32_t ow = 0; ow < width; ++ow)
                    destination[(uint64_t)(row + 2U) * width + ow] =
                        source[ow];
            }
        }

        const uint32_t edge_rows[4] = {0U, 1U, height > 2U ? height - 2U : 0U,
                                       height > 1U ? height - 1U : 0U};
        for (uint32_t edge_index = 0; edge_index < 4U; ++edge_index) {
            const uint32_t oh = edge_rows[edge_index];
            if (oh >= height || (edge_index == 3U && height <= 1U) ||
                (edge_index == 2U && height <= 2U))
                continue;
            for (uint32_t ow = 0; ow < width; ++ow) {
                float maximum = negative_infinity();
                for (uint32_t kh = 0; kh < 5U; ++kh) {
                    const int32_t ih = (int32_t)oh + (int32_t)kh - 2;
                    if (ih < 0 || ih >= (int32_t)height)
                        continue;
                    const float value =
                        ((const volatile float *)(uintptr_t)(
                             horizontal_at + (uint64_t)ih * input_pitch))[ow];
                    maximum = value > maximum ? value : maximum;
                }
                destination[(uint64_t)oh * width + ow] = maximum;
            }
        }

        store_rows_issue(output_at, (uintptr_t)(output + base), height,
                         row_bytes, row_bytes);
        dayan_wait_lsu();
    }
    return true;
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

    if (kernel_h == 5U && kernel_w == 5U && stride_h == 1U &&
        stride_w == 1U && pad_h == 2U && pad_w == 2U &&
        input_h == output_h && input_w == output_w &&
        maxpool5x5s1p2_dense(input, output, batch, channels,
                          input_h, input_w, uram))
        return;

    if (kernel_h == 3U && kernel_w == 3U && stride_h == 2U &&
        stride_w == 2U && pad_h == 1U && pad_w == 1U &&
        maxpool3x3s2p1_vector(input, output, batch, channels, input_h,
                             input_w, output_h, output_w, uram))
        return;

    const uint32_t window_elements = kernel_h * kernel_w;
    const uint32_t padded_window = (window_elements + 15U) & ~15U;
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
DAYAN_KERNEL_TASK_DESC(rvnpu_maxpool2d_nchw_fp32, 4, 0, 0, 65536U, 2, 0, 0, 1);
