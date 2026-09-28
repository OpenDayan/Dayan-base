// ReduceMax：沿最后一维求最大值。
// Proc 按逻辑分片认领若干行，逐 tile 执行 Load -> MACRO reduce_max -> 归约，
// 尾 tile 与非对齐行由 Proc 兜底。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

namespace {
#if defined(__DAYAN_DEVICE__)

__device__ constexpr uint32_t kTileElements = 4096;
__device__ constexpr uint32_t kUramBytes = 32U * 1024U;
__device__ constexpr long kLsuGroup = 0;
__device__ constexpr long kMacroGroup = 0;

__device__ static inline uint32_t align_up_64(uint32_t value) {
    return (value + 63U) & ~63U;
}

__device__ static inline float negative_infinity() {
    union FloatBits {
        uint32_t bits;
        float value;
    } value = {0xff800000U};
    return value.value;
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

// 搬一个 tile 进 URAM：源地址 64B 对齐时交给 LSU，否则 Proc 逐元素兜底。
__device__ static void load_tile(const float *source, uint64_t destination,
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
        __tx_lsu_dma_d2u<kLsuGroup>((long)(destination + blocks * 64U),
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

#endif
} // namespace

__kernel__ void rvnpu_reduce_max_lastdim_fp32(const float *input, float *output,
                                              uint32_t rows, uint32_t columns) {
    // 空张量直接返回，不算参数错误。
    if (rows == 0 || columns == 0)
        return;
    if (input == nullptr || output == nullptr) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    dayan_cache_invalidate(input, (uint64_t)rows*columns*sizeof(float));
    dayan_cache_invalidate(output, (uint64_t)rows*sizeof(float));
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    const uint32_t scratch_bytes =
        align_up_64(kTileElements * (uint32_t)sizeof(float));
    if (uram == 0 || scratch_bytes + 128U > kUramBytes ||
        dayan_uram_size() < kUramBytes)
        dayan_thread_exit(-2);
    const uint64_t tile_value = uram + scratch_bytes;
    const uint64_t tile_index = uram + scratch_bytes + 64U;

    for (uint32_t row = shard_begin(rows); row < shard_end(rows); ++row) {
        float best = negative_infinity();
        for (uint32_t begin = 0; begin < columns; begin += kTileElements) {
            const uint32_t remaining = columns - begin;
            const uint32_t count =
                remaining < kTileElements ? remaining : kTileElements;
            load_tile(input + (uint64_t)row * columns + begin, uram, count);
            configure_reduce(count);
            __tx_macro_reduce_max_fp32<kMacroGroup, 1>((long)tile_value,
                                                       (long)uram,
                                                       (long)tile_index);
            dayan_wait_macro();
            const float candidate = *(volatile float *)(uintptr_t)tile_value;
            if (candidate > best)
                best = candidate;
        }
        output[row] = best;
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(rvnpu_reduce_max_lastdim_fp32, 4, 0, 0, 32768U, 1, 0, 0, 1);
