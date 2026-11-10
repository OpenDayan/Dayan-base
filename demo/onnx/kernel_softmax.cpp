// Softmax(axis=1)：rank-4 [outer, axis, inner]，对 axis 维归一化。
// 一次取若干列搬进 URAM，每列的 axis 个值走 MACRO 链：
// reduce_max -> sub -> *(-log2e) -> exp2 -> reduce_add -> recip -> mul，
// 结果写回 URAM 后整段写回 DRAM。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

namespace {
#if defined(__DAYAN_DEVICE__)

__device__ constexpr uint32_t kUramBytes = 32U * 1024U;
__device__ constexpr uint32_t kStageBytes = 16U * 1024U;
__device__ constexpr uint32_t kVectorStride = 256U;
__device__ constexpr float kInverseLn2 = 1.4426950408889634f;
__device__ constexpr long kLsuGroup = 0;
__device__ constexpr long kMacroGroup = 0;

__device__ static inline uint32_t f32_bits(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

__device__ static inline void configure_lsu_linear(uint32_t blocks,
                                                   uint32_t request_bytes) {
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM0>(blocks - 1U);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(
        request_bytes - 1U);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(
        request_bytes - 1U);
}

// 一维 MACRO 配置：DIM0 为元素个数，其余维退化。
template <long GROUP>
__device__ static inline void configure_macro_1d(uint32_t elements) {
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DIM0>(elements - 1U);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DIM1>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DIM2>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DIM3>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC0DIM1STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC0DIM2STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC0DIM3STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC1DIM1STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC1DIM2STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC1DIM3STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DSTDIM1STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DSTDIM2STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DSTDIM3STRIDE>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<GROUP * MACRO_GROUPSZ + MACRO_DSTSWIZZLEMBS>(0);
}

// 立即数乘法：destination = value * source。
__device__ static inline void macro_immediate_mul(uint64_t destination,
                                                  float value,
                                                  uint64_t source) {
    __tx_macro_elemwise_mul_fp32<kMacroGroup, 2>(
        (long)destination, (long)f32_bits(value), (long)source);
}

// 一段连续数据 DRAM -> URAM，地址非 64B 对齐时由 Proc 兜底。
__device__ static void load_run(const float *source, uint64_t destination,
                                uint32_t elements) {
    if (elements == 0)
        return;
    if (((uintptr_t)source & 63U) != 0) {
        volatile float *uram = (volatile float *)(uintptr_t)destination;
        for (uint32_t index = 0; index < elements; ++index)
            uram[index] = source[index];
        return;
    }
    const uint32_t bytes = elements * sizeof(float);
    const uint32_t blocks = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    if (blocks != 0) {
        configure_lsu_linear(blocks, 64);
        __tx_lsu_dma_d2u<kLsuGroup>((long)destination, (long)source);
        dayan_wait_lsu();
    }
    if (tail != 0) {
        configure_lsu_linear(1, tail);
        __tx_lsu_dma_d2u<kLsuGroup>(
            (long)(destination + (uint64_t)blocks * 64U),
            (long)((const char *)source + (uint64_t)blocks * 64U));
        dayan_wait_lsu();
    }
}

// 一段连续数据 URAM -> DRAM，地址非 64B 对齐时由 Proc 兜底。
__device__ static void store_run(uint64_t source, float *destination,
                                 uint32_t elements, bool scalar_rows) {
    if (elements == 0)
        return;
    // Prefix/tail passes visit different axis rows. If rows share cache
    // lines, a later partial DMA could otherwise be overwritten by a dirty
    // scalar-store cache line from an adjacent row.
    if (scalar_rows || ((uintptr_t)destination & 63U) != 0) {
        const volatile float *uram = (const volatile float *)(uintptr_t)source;
        for (uint32_t index = 0; index < elements; ++index)
            destination[index] = uram[index];
        return;
    }
    const uint32_t bytes = elements * sizeof(float);
    const uint32_t blocks = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    if (blocks != 0) {
        configure_lsu_linear(blocks, 64);
        __tx_lsu_dma_u2d<kLsuGroup>((long)destination, (long)source);
        dayan_wait_lsu();
    }
    if (tail != 0) {
        configure_lsu_linear(1, tail);
        __tx_lsu_dma_u2d<kLsuGroup>(
            (long)((char *)destination + (uint64_t)blocks * 64U),
            (long)(source + (uint64_t)blocks * 64U));
        dayan_wait_lsu();
    }
}

__device__ static inline uint64_t shard_begin(uint64_t total) {
    const uint32_t tasks = dayan_task_dim(0);
    if (tasks < 2U)
        return 0;
    return total * dayan_task_id(0) / tasks;
}

__device__ static inline uint64_t shard_end(uint64_t total) {
    const uint32_t tasks = dayan_task_dim(0);
    if (tasks < 2U)
        return total;
    return total * (dayan_task_id(0) + 1U) / tasks;
}

// 处理 [axis, row_stride] 平面上从 begin 开始的 count 列。
// input/output 已指向该平面的起始位置。
__device__ static void softmax_chunk(const float *input, float *output,
                                     uint32_t axis, uint32_t row_stride,
                                     uint32_t count, uint64_t uram) {
    const uint64_t stage = uram;
    const uint64_t values = uram + kStageBytes;
    const uint64_t centered = values + kVectorStride;
    const uint64_t probabilities = centered + kVectorStride;
    const uint64_t maximum = probabilities + kVectorStride;
    const uint64_t sum = maximum + kVectorStride;

    for (uint32_t index = 0; index < axis; ++index)
        load_run(input + (uint64_t)index * row_stride,
                 stage + (uint64_t)index * count * sizeof(float), count);

    // YOLO DFL: SIMD lanes are independent spatial positions, not the
    // (usually 16-element) distribution axis. Keep the original layout.
    // Three column vectors fit after the stage for axis >= 4. Aligned
    // rows are required by MACRO; small/unaligned tails retain the old path.
    if (axis >= 4U && count >= 16U && (count & 15U) == 0) {
        const uint64_t column_bytes = (uint64_t)count * sizeof(float);
        const uint64_t vmax = uram + kStageBytes;
        const uint64_t vsum = vmax + column_bytes;
        const uint64_t work = vsum + column_bytes;
        configure_macro_1d<kMacroGroup>(count);
        __tx_macro_elemwise_max_fp32<kMacroGroup, 0>(
            (long)vmax, (long)stage, (long)stage);
        for (uint32_t a = 1; a < axis; ++a)
            __tx_macro_elemwise_max_fp32<kMacroGroup, 0>(
                (long)vmax, (long)vmax, (long)(stage + a * column_bytes));
        for (uint32_t a = 0; a < axis; ++a) {
            const uint64_t row = stage + a * column_bytes;
            __tx_macro_elemwise_sub_fp32<kMacroGroup, 0>(
                (long)work, (long)row, (long)vmax);
            macro_immediate_mul(work, kInverseLn2, work);
            __tx_macro_sfu_exp2_fp32<kMacroGroup>((long)row, (long)work);
            if (a == 0)
                __tx_macro_elemwise_mul_fp32<kMacroGroup, 2>(
                    (long)vsum, (long)f32_bits(1.0f), (long)row);
            else
                __tx_macro_elemwise_add_fp32<kMacroGroup, 0>(
                    (long)vsum, (long)vsum, (long)row);
        }
        __tx_macro_sfu_recip_fp32<kMacroGroup>((long)vsum, (long)vsum);
        for (uint32_t a = 0; a < axis; ++a) {
            const uint64_t row = stage + a * column_bytes;
            __tx_macro_elemwise_mul_fp32<kMacroGroup, 0>(
                (long)row, (long)row, (long)vsum);
        }
        dayan_wait_macro();
        for (uint32_t a = 0; a < axis; ++a)
            store_run(stage + a * column_bytes,
                      output + (uint64_t)a * row_stride, count,
                      (row_stride & 15U) != 0);
        return;
    }

    // wait 让出执行权后同 Hart 其他 Proc 可能污染 TCSR，宏链前重配。
    configure_macro_1d<kMacroGroup>(axis);

    const volatile float *staged = (const volatile float *)(uintptr_t)stage;
    volatile float *staged_write = (volatile float *)(uintptr_t)stage;
    const volatile float *result =
        (const volatile float *)(uintptr_t)probabilities;
    volatile float *tile = (volatile float *)(uintptr_t)values;
    for (uint32_t column = 0; column < count; ++column) {
        for (uint32_t index = 0; index < axis; ++index)
            tile[index] = staged[(uint64_t)index * count + column];

        configure_macro_1d<kMacroGroup>(axis);
        __tx_macro_reduce_max_fp32<kMacroGroup, 0>((long)maximum, 0,
                                                   (long)values);
        __tx_macro_elemwise_sub_fp32<kMacroGroup, 1>(
            (long)centered, (long)maximum, (long)values);
        macro_immediate_mul(centered, -kInverseLn2, centered);
        __tx_macro_sfu_exp2_fp32<kMacroGroup>((long)probabilities,
                                              (long)centered);
        __tx_macro_reduce_add_fp32<kMacroGroup>((long)sum, (long)probabilities);
        __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM0>(0);
        __tx_macro_sfu_recip_fp32<kMacroGroup>((long)sum, (long)sum);
        __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM0>(
            axis - 1U);
        __tx_macro_elemwise_mul_fp32<kMacroGroup, 1>(
            (long)probabilities, (long)sum, (long)probabilities);
        dayan_wait_macro();

        for (uint32_t index = 0; index < axis; ++index)
            staged_write[(uint64_t)index * count + column] = result[index];
    }

    for (uint32_t index = 0; index < axis; ++index)
        store_run(stage + (uint64_t)index * count * sizeof(float),
                  output + (uint64_t)index * row_stride, count,
                  (row_stride & 15U) != 0);
}

#endif
} // namespace

__kernel__ void rvnpu_softmax_axis1_fp32(const float *input, float *output,
                                         uint32_t outer, uint32_t axis,
                                         uint32_t inner) {
    // 空张量直接返回，不算参数错误。
    if (outer == 0 || axis == 0 || inner == 0)
        return;
    if (input == nullptr || output == nullptr || axis > 64) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    const uint64_t total_bytes =
        (uint64_t)outer * axis * inner * sizeof(float);
    const uintptr_t alignment = (uintptr_t)input | (uintptr_t)output |
        (uintptr_t)(inner * sizeof(float));
    if ((alignment & 63U) != 0) {
        dayan_cache_invalidate(input, total_bytes);
        dayan_cache_invalidate(output, total_bytes);
    }
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    if (uram == 0 || dayan_uram_size() < kUramBytes)
        dayan_thread_exit(-2);
    const uint32_t chunk = kStageBytes / (axis * (uint32_t)sizeof(float));
    if (chunk == 0)
        dayan_thread_exit(-2);
    if (kStageBytes + 6U * kVectorStride > kUramBytes)
        dayan_thread_exit(-2);

    const uint64_t chunks = (inner + chunk - 1U) / chunk;
    const uint64_t units = (uint64_t)outer * chunks;
    for (uint64_t unit = shard_begin(units); unit < shard_end(units); ++unit) {
        const uint32_t slice = (uint32_t)(unit / chunks);
        const uint32_t begin = (uint32_t)(unit % chunks) * chunk;
        const uint32_t count = inner - begin < chunk ? inner - begin : chunk;
        const uint64_t base = (uint64_t)slice * axis * inner + begin;
        // e.g. YOLO64 has 84 spatial positions: vectorize 80 columns and
        // retain the exact scalar-column path for the remaining four.
        // Unaligned DRAM row strides are handled by load_run/store_run.
        const uint32_t vector_count = axis >= 4U ? count & ~15U : 0U;
        if (vector_count != 0 && vector_count != count) {
            softmax_chunk(input + base, output + base, axis, inner,
                          vector_count, uram);
            softmax_chunk(input + base + vector_count,
                          output + base + vector_count, axis, inner,
                          count - vector_count, uram);
        } else {
            softmax_chunk(input + base, output + base, axis, inner, count, uram);
        }
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(rvnpu_softmax_axis1_fp32, 4, 0, 0, 32768U, 1, 0, 0, 1);
