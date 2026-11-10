// Transpose：支持 EP 允许的三种置换，转置中转区放在 URAM。
//   mode 0：rank-2 [d0, d1]          -> [d1, d0]        （perm 1,0）
//   mode 1：rank-3 [d0, d1, d2]      -> [d0, d2, d1]    （perm 0,2,1）
//   mode 2：rank-4 [d0, d1, d2, d3]  -> [d0, d2, d1, d3]（perm 0,2,1,3）
// 连续段搬运交给 LSU，只有 64B 非对齐的段才回退到 Proc 逐元素拷贝。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

namespace {
#if defined(__DAYAN_DEVICE__)

__device__ constexpr long kLsuGroup = 0;
__device__ constexpr uint32_t kUramBytes = 256U * 1024U;
// mode 1 的列分块宽度：d1 <= 128 时两块中转区合计 256KiB。
__device__ constexpr uint32_t kTileColumns = 256;

__device__ static inline void configure_copy(uint32_t blocks,
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

// 三段通用搬运：DRAM->DRAM / DRAM->URAM / URAM->DRAM。
// 参与搬运的 DRAM 侧地址必须 64B 对齐，否则 Proc 逐元素兜底。
template <typename T>
__device__ static void copy_run(const T *source, T *destination,
                                uint32_t elements) {
    if (elements == 0)
        return;
    if ((((uintptr_t)source | (uintptr_t)destination) & 63U) != 0) {
        dayan_cache_invalidate(source, elements * (uint32_t)sizeof(T));
        dayan_cache_invalidate(destination, elements * (uint32_t)sizeof(T));
        for (uint32_t index = 0; index < elements; ++index)
            destination[index] = source[index];
        return;
    }
    const uint32_t bytes = elements * (uint32_t)sizeof(T);
    const uint32_t blocks = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    if (blocks != 0) {
        configure_copy(blocks, 64);
        __tx_lsu_dma_d2d<kLsuGroup>((long)destination, (long)source);
        dayan_wait_lsu();
    }
    if (tail != 0) {
        configure_copy(1, tail);
        __tx_lsu_dma_d2d<kLsuGroup>(
            (long)((char *)destination + blocks * 64U),
            (long)((const char *)source + blocks * 64U));
        dayan_wait_lsu();
    }
}

// DRAM -> URAM，地址不满足 64B 对齐时由 Proc 兜底。
__device__ static void load_run(const float *source, uint64_t uram,
                               uint32_t elements) {
    if (elements == 0)
        return;
    if (((uintptr_t)source & 63U) != 0) {
        dayan_cache_invalidate(source, elements * (uint32_t)sizeof(float));
        volatile float *destination = (volatile float *)(uintptr_t)uram;
        for (uint32_t index = 0; index < elements; ++index)
            destination[index] = source[index];
        return;
    }
    const uint32_t bytes = elements * sizeof(float);
    const uint32_t blocks = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    if (blocks != 0) {
        configure_copy(blocks, 64);
        __tx_lsu_dma_d2u<kLsuGroup>((long)uram, (long)source);
        dayan_wait_lsu();
    }
    if (tail != 0) {
        configure_copy(1, tail);
        __tx_lsu_dma_d2u<kLsuGroup>(
            (long)(uram + (uint64_t)blocks * 64U),
            (long)((const char *)source + (uint64_t)blocks * 64U));
        dayan_wait_lsu();
    }
}

// URAM -> DRAM，地址不满足 64B 对齐时由 Proc 兜底。
__device__ static void store_run(uint64_t uram, float *destination,
                                 uint32_t elements) {
    if (elements == 0)
        return;
    if (((uintptr_t)destination & 63U) != 0) {
        dayan_cache_invalidate(destination, elements * (uint32_t)sizeof(float));
        const volatile float *source = (const volatile float *)(uintptr_t)uram;
        for (uint32_t index = 0; index < elements; ++index)
            destination[index] = source[index];
        return;
    }
    const uint32_t bytes = elements * sizeof(float);
    const uint32_t blocks = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    if (blocks != 0) {
        configure_copy(blocks, 64);
        __tx_lsu_dma_u2d<kLsuGroup>((long)destination, (long)uram);
        dayan_wait_lsu();
    }
    if (tail != 0) {
        configure_copy(1, tail);
        __tx_lsu_dma_u2d<kLsuGroup>(
            (long)((char *)destination + (uint64_t)blocks * 64U),
            (long)(uram + (uint64_t)blocks * 64U));
        dayan_wait_lsu();
    }
}

// rank-3 [d0, d1, d2] -> [d0, d2, d1]。
// 每次取 d1 x tile 列的一段搬进 URAM，在 URAM 内完成 d1 x count 的转置，
// 再整段写回 [count, d1] 排布的输出。
__device__ static void transpose_021(const float *input, float *output,
                                     uint32_t d0, uint32_t d1, uint32_t d2,
                                     uint64_t uram, uint32_t tile) {
    const uint64_t input_tile = uram;
    const uint64_t output_tile = uram + (uint64_t)d1 * tile * sizeof(float);
    const uint32_t tile_count = (d2 + tile - 1U) / tile;
    const uint64_t units = (uint64_t)d0 * tile_count;
    for (uint64_t unit = shard_begin(units); unit < shard_end(units); ++unit) {
        const uint32_t i0 = (uint32_t)(unit / tile_count);
        const uint32_t begin = (uint32_t)(unit % tile_count) * tile;
        const uint32_t count = d2 - begin < tile ? d2 - begin : tile;
        for (uint32_t i1 = 0; i1 < d1; ++i1) {
            const uint64_t source = ((uint64_t)i0 * d1 + i1) * d2 + begin;
            load_run(input + source,
                     input_tile + (uint64_t)i1 * count * sizeof(float), count);
        }
        const volatile float *source_view =
            (const volatile float *)(uintptr_t)input_tile;
        volatile float *destination_view =
            (volatile float *)(uintptr_t)output_tile;
        for (uint32_t index = 0; index < count; ++index)
            for (uint32_t i1 = 0; i1 < d1; ++i1)
                destination_view[index * d1 + i1] =
                    source_view[i1 * count + index];
        const uint64_t destination = ((uint64_t)i0 * d2 + begin) * d1;
        store_run(output_tile, output + destination, count * d1);
    }
}

// rank-4 [d0, d1, d2, d3] -> [d0, d2, d1, d3]。
// 保持最内层 d3 连续，逐段搬运，批量段交给 LSU。
__device__ static void transpose_0213(const float *input, float *output,
                                      uint32_t d0, uint32_t d1, uint32_t d2,
                                      uint32_t d3) {
    const uint64_t units = (uint64_t)d0 * d1 * d2;
    for (uint64_t unit = shard_begin(units); unit < shard_end(units); ++unit) {
        const uint32_t i2 = (uint32_t)(unit % d2);
        const uint32_t i1 = (uint32_t)(unit / d2 % d1);
        const uint32_t i0 = (uint32_t)(unit / ((uint64_t)d1 * d2));
        const uint64_t source = (((uint64_t)i0 * d1 + i1) * d2 + i2) * d3;
        const uint64_t destination = (((uint64_t)i0 * d2 + i2) * d1 + i1) * d3;
        copy_run(input + source, output + destination, d3);
    }
}

// rank-2 [d0, d1] -> [d1, d0]。
// 每次取 tr 行搬进 URAM，在 URAM 内完成 tr x d1 的转置，再整段写回。
__device__ static void transpose_10(const float *input, float *output,
                                    uint32_t d0, uint32_t d1, uint64_t uram,
                                    uint32_t tile_rows) {
    const uint64_t input_tile = uram;
    const uint64_t output_tile = uram + (uint64_t)tile_rows * d1 * sizeof(float);
    const uint32_t tile_count = (d0 + tile_rows - 1U) / tile_rows;
    for (uint64_t tile = shard_begin(tile_count); tile < shard_end(tile_count);
         ++tile) {
        const uint32_t begin = (uint32_t)tile * tile_rows;
        const uint32_t count =
            d0 - begin < tile_rows ? d0 - begin : tile_rows;
        for (uint32_t row = 0; row < count; ++row)
            load_run(input + (uint64_t)(begin + row) * d1,
                     input_tile + (uint64_t)row * d1 * sizeof(float), d1);
        const volatile float *source_view =
            (const volatile float *)(uintptr_t)input_tile;
        volatile float *destination_view =
            (volatile float *)(uintptr_t)output_tile;
        for (uint32_t column = 0; column < d1; ++column)
            for (uint32_t row = 0; row < count; ++row)
                destination_view[(uint64_t)column * count + row] =
                    source_view[(uint64_t)row * d1 + column];
        for (uint32_t column = 0; column < d1; ++column)
            store_run(output_tile + (uint64_t)column * count * sizeof(float),
                      output + (uint64_t)column * d0 + begin, count);
    }
}

// 选择 rank-3 路径的列分块宽度，使两块 URAM 中转区放得下。
__device__ static uint32_t transpose_021_tile(uint32_t d1, uint64_t uram_size) {
    if (d1 == 0)
        return 0;
    const uint64_t budget = uram_size / (2U * (uint64_t)sizeof(float));
    uint32_t tile = kTileColumns;
    const uint64_t capacity = budget / d1;
    if (capacity < tile)
        tile = (uint32_t)capacity;
    return tile & ~15U;
}

#endif
} // namespace

__kernel__ void rvnpu_transpose_fp32(const float *input, float *output,
                                     uint32_t d0, uint32_t d1, uint32_t d2,
                                     uint32_t d3, uint32_t mode) {
    if (mode > 2) {
        dayan_thread_exit(-22);
        return;
    }
    // 空张量直接返回，不算参数错误。
    if (d0 == 0 || d1 == 0 || (mode != 0 && d2 == 0) || (mode == 2 && d3 == 0))
        return;
    if (input == nullptr || output == nullptr) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    const uint64_t uram_size = dayan_uram_size();
    if (uram == 0 || uram_size < kUramBytes)
        dayan_thread_exit(-2);

    if (mode == 0) {
        // rank-2 没有列宽约束，中转区按行分块大小自适应。
        const uint64_t budget = uram_size / (2U * (uint64_t)sizeof(float));
        if (d1 <= budget) {
            uint32_t tile_rows = (uint32_t)(budget / d1);
            if (tile_rows > d0)
                tile_rows = d0;
            if (tile_rows == 0)
                tile_rows = 1;
            transpose_10(input, output, d0, d1, uram, tile_rows);
        } else {
            // 单行都放不进 URAM，退化为 Proc 逐元素转置。
            dayan_cache_invalidate(input, (uint64_t)d0 * d1 * sizeof(float));
            dayan_cache_invalidate(output, (uint64_t)d0 * d1 * sizeof(float));
            const uint64_t units = (uint64_t)d0 * d1;
            for (uint64_t unit = shard_begin(units); unit < shard_end(units);
                 ++unit) {
                const uint32_t i0 = (uint32_t)(unit / d1);
                const uint32_t i1 = (uint32_t)(unit % d1);
                output[(uint64_t)i1 * d0 + i0] =
                    input[(uint64_t)i0 * d1 + i1];
            }
        }
    } else if (mode == 1) {
        const uint32_t tile = transpose_021_tile(d1, uram_size);
        if (tile == 0) {
            dayan_cache_invalidate(input, (uint64_t)d0 * d1 * d2 * sizeof(float));
            dayan_cache_invalidate(output, (uint64_t)d0 * d1 * d2 * sizeof(float));
            const uint64_t elements = (uint64_t)d0 * d1 * d2;
            for (uint64_t index = shard_begin(elements);
                 index < shard_end(elements); ++index) {
                const uint32_t i2 = (uint32_t)(index / d1 % d2);
                const uint32_t i1 = (uint32_t)(index % d1);
                const uint32_t i0 = (uint32_t)(index / ((uint64_t)d1 * d2));
                output[((uint64_t)i0 * d2 + i2) * d1 + i1] =
                    input[((uint64_t)i0 * d1 + i1) * d2 + i2];
            }
        } else {
            transpose_021(input, output, d0, d1, d2, uram, tile);
        }
    } else {
        transpose_0213(input, output, d0, d1, d2, d3);
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(rvnpu_transpose_fp32, 4, 0, 0, 262144U, 1, 0, 0, 0);

__kernel__ void rvnpu_transpose_10_int64(const int64_t *input, int64_t *output,
                                        uint32_t d0, uint32_t d1) {
    // 空张量（例如 NonZero 得到 0 列后再转置）直接返回。
    if (d0 == 0 || d1 == 0)
        return;
    if (input == nullptr || output == nullptr) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    dayan_cache_invalidate(input, (uint64_t)d0*d1*sizeof(int64_t));
    dayan_cache_invalidate(output, (uint64_t)d0*d1*sizeof(int64_t));
    const uint64_t units = (uint64_t)d0 * d1;
    for (uint64_t unit = shard_begin(units); unit < shard_end(units); ++unit) {
        const uint32_t i0 = (uint32_t)(unit / d1);
        const uint32_t i1 = (uint32_t)(unit % d1);
        output[(uint64_t)i1 * d0 + i0] = input[(uint64_t)i0 * d1 + i1];
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(rvnpu_transpose_10_int64, 4, 0, 0, 0, 0, 0, 0, 0);
