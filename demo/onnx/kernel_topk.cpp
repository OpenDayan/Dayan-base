// TopK(最后一维)：对每一行逐次选出最大值与下标。
// 行宽放得进 URAM 时整行缓存，每选出一个就把该位置改成 -inf 再选下一个；
// 行太宽时按 tile 流式扫描，已选下标记录在 URAM 的 selection 表里并跳过。
// 归约走 MACRO reduce_max（带下标输出）。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

namespace {
#if defined(__DAYAN_DEVICE__)

__device__ constexpr uint32_t kTileElements = 4096;
__device__ constexpr uint32_t kUramBytes = 64U * 1024U;
__device__ constexpr uint32_t kReservedBytes = 256U;
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

// 一段数据搬进 URAM；DRAM 侧非 64B 对齐时由 Proc 兜底。
__device__ static void cache_run(const float *source, uint64_t destination,
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
        __tx_lsu_dma_d2u<kLsuGroup>((long)(destination + (uint64_t)blocks * 64U),
                                    (long)((const char *)source + blocks * 64U));
        dayan_wait_lsu();
    }
}

__device__ static inline void configure_reduce(uint32_t elements) {
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM0>(
        elements - 1U);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM1>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM2>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DIM3>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0DIM1STRIDE>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0DIM2STRIDE>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC0DIM3STRIDE>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1DIM1STRIDE>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1DIM2STRIDE>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_SRC1DIM3STRIDE>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTDIM1STRIDE>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTDIM2STRIDE>(0);
    __tx_macro_tcsr_write<kMacroGroup * MACRO_GROUPSZ + MACRO_DSTDIM3STRIDE>(0);
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

// 在已缓存的整行里找最大值；tile_value / tile_index 是归约结果的暂存位置。
__device__ static bool reduce_best(uint64_t scratch_at, uint32_t columns,
                                   uint64_t tile_value, uint64_t tile_index,
                                   float &best_value, uint32_t &best_index) {
    bool found = false;
    const volatile float *scratch = (const volatile float *)(uintptr_t)scratch_at;
    for (uint32_t offset = 0; offset < columns; offset += kTileElements) {
        const uint32_t remaining = columns - offset;
        const uint32_t length =
            remaining < kTileElements ? remaining : kTileElements;
        configure_reduce(length);
        __tx_macro_reduce_max_fp32<kMacroGroup, 1>(
            (long)tile_value, (long)(scratch_at + (uint64_t)offset * sizeof(float)),
            (long)tile_index);
        dayan_wait_macro();
        const float candidate = *(volatile float *)(uintptr_t)tile_value;
        const uint32_t candidate_index =
            offset + *(volatile uint32_t *)(uintptr_t)tile_index;
        (void)scratch;
        if (!found || candidate > best_value) {
            found = true;
            best_value = candidate;
            best_index = candidate_index;
        }
    }
    return found;
}

// 流式版本：逐 tile 现搬现算，已选中的位置在搬进来的 tile 里改成 -inf。
__device__ static bool reduce_best_streamed(
    const float *input, uint32_t columns, const uint64_t selected_at,
    uint32_t selected_count, uint64_t scratch_at, uint64_t tile_value,
    uint64_t tile_index, float &best_value, uint32_t &best_index) {
    bool found = false;
    for (uint32_t offset = 0; offset < columns; offset += kTileElements) {
        const uint32_t remaining = columns - offset;
        const uint32_t length =
            remaining < kTileElements ? remaining : kTileElements;
        cache_run(input + offset, scratch_at, length);
        volatile float *scratch = (volatile float *)(uintptr_t)scratch_at;
        const volatile uint32_t *selected =
            (const volatile uint32_t *)(uintptr_t)selected_at;
        for (uint32_t previous = 0; previous < selected_count; ++previous) {
            const uint32_t index = selected[previous];
            if (index >= offset && index - offset < length)
                scratch[index - offset] = negative_infinity();
        }
        configure_reduce(length);
        __tx_macro_reduce_max_fp32<kMacroGroup, 1>((long)tile_value,
                                                   (long)scratch_at,
                                                   (long)tile_index);
        dayan_wait_macro();
        const float candidate = *(volatile float *)(uintptr_t)tile_value;
        const uint32_t candidate_index =
            offset + *(volatile uint32_t *)(uintptr_t)tile_index;
        if (!found || candidate > best_value) {
            found = true;
            best_value = candidate;
            best_index = candidate_index;
        }
    }
    return found;
}

// 全行都是 -inf 时的兜底：取第一个还没被选走的下标。
__device__ static bool first_unselected(uint64_t selected_at,
                                        uint32_t selected_count,
                                        uint32_t columns, uint32_t &index) {
    const volatile uint32_t *selected =
        (const volatile uint32_t *)(uintptr_t)selected_at;
    for (uint32_t column = 0; column < columns; ++column) {
        bool taken = false;
        for (uint32_t previous = 0; previous < selected_count; ++previous) {
            if (selected[previous] == column) {
                taken = true;
                break;
            }
        }
        if (!taken) {
            index = column;
            return true;
        }
    }
    return false;
}

#endif
} // namespace

__kernel__ void rvnpu_topk_lastdim_fp32(const float *input, float *values,
                                        int64_t *indices, uint32_t rows,
                                        uint32_t columns, uint32_t k) {
    if (rows == 0 || k == 0)
        return;
    if (input == nullptr || values == nullptr || indices == nullptr ||
        columns == 0 || k > columns) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    dayan_cache_invalidate(input, (uint64_t)rows * columns * sizeof(float));
    dayan_cache_invalidate(values, (uint64_t)rows * k * sizeof(float));
    dayan_cache_invalidate(indices, (uint64_t)rows * k * sizeof(int64_t));
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    if (uram == 0 || dayan_uram_size() < kUramBytes)
        dayan_thread_exit(-2);

    // 整行缓存需要 columns 个 float；否则只留一个 tile 的位置。
    uint32_t cache_elements = kTileElements;
    const bool cache_whole_row =
        (uint64_t)columns * sizeof(float) + kReservedBytes +
                (uint64_t)k * sizeof(uint32_t) + 128U <=
            kUramBytes;
    if (cache_whole_row)
        cache_elements = columns;
    const uint32_t scratch_bytes =
        align_up_64(cache_elements * (uint32_t)sizeof(float));
    const uint64_t scratch_at = uram;
    const uint64_t tile_value = uram + scratch_bytes;
    const uint64_t tile_index = uram + scratch_bytes + 64U;
    const uint64_t selected_at = uram + scratch_bytes + 128U;
    const uint64_t selection_bytes = align_up_64(k * (uint32_t)sizeof(uint32_t));
    if (scratch_bytes + 128U + selection_bytes > kUramBytes)
        dayan_thread_exit(-2);
    volatile uint32_t *selected = (volatile uint32_t *)(uintptr_t)selected_at;

    for (uint32_t row = shard_begin(rows); row < shard_end(rows); ++row) {
        const uint64_t input_base = (uint64_t)row * columns;
        const uint64_t output_base = (uint64_t)row * k;
        if (cache_whole_row)
            cache_run(input + input_base, scratch_at, columns);
        for (uint32_t output_index = 0; output_index < k; ++output_index) {
            float best_value = negative_infinity();
            uint32_t best_index = 0;
            const bool found =
                cache_whole_row
                    ? reduce_best(scratch_at, columns, tile_value, tile_index,
                                  best_value, best_index)
                    : reduce_best_streamed(input + input_base, columns,
                                           selected_at, output_index,
                                           scratch_at, tile_value, tile_index,
                                           best_value, best_index);
            if (!found)
                dayan_thread_exit(-2);

            // 全 -inf 的行：显式跳过已选下标，保证下标不重复。
            if (best_value == negative_infinity()) {
                if (!first_unselected(selected_at, output_index, columns,
                                      best_index))
                    dayan_thread_exit(-2);
                best_value = input[input_base + best_index];
            }

            selected[output_index] = best_index;
            values[output_base + output_index] = best_value;
            indices[output_base + output_index] = (int64_t)best_index;
            if (cache_whole_row)
                ((volatile float *)(uintptr_t)scratch_at)[best_index] =
                    negative_infinity();
        }
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(rvnpu_topk_lastdim_fp32, 4, 0, 0, 65536U, 1, 0, 0, 1);
