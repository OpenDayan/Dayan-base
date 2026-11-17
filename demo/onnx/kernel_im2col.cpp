// Im2Col(NCHW)：把每个输出点的卷积窗口拉成一行 [channels * kernel_h * kernel_w]，
// 输出布局与 MatMul 的 A 矩阵一致（紧凑 K，不补零）。
// 每个输出点一行：先在 URAM 里拼好这一行，再用 LSU 整行写回 DRAM；
// 行内元素直接按 NCHW 下标读取，越界补 0。Proc 按行分片。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

namespace {
#if defined(__DAYAN_DEVICE__)

__device__ constexpr uint32_t kUramBytes = 64U * 1024U;
__device__ constexpr long kLsuGroup = 0;

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

// URAM 里的一整行写回 DRAM；整块交给 LSU，尾部零头单独一次。
// DRAM 侧地址非 64B 对齐时由 Proc 兜底。
__device__ static void store_row(uint64_t source, float *destination,
                                 uint32_t elements) {
    if (elements == 0)
        return;
    if (((uintptr_t)destination & 63U) != 0) {
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

__device__ static void configure_lsu_transpose(uint32_t dim0_bytes,
                                                   uint32_t dim1_rows,
                                                   uint32_t in_stride,
                                                   uint32_t out_stride) {
    constexpr uint32_t kSub = 16U;
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCK>(kSub - 1U);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM0>(dim0_bytes / 64U - 1U);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM1>(
        dim1_rows / kSub - 1U);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(in_stride);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_INDIM1STRIDE>(
        in_stride * kSub);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(
        out_stride);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(
        out_stride * kSub);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(64);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(2);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_BLOCKPADDING>(kSub - 1U);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
}

__device__ static void transpose_columns(const float *input, float *columns,
                                         uint32_t channels,
                                         uint64_t positions) {
    constexpr uint32_t kBatchBytes = 32U * 1024U;
    uint64_t batch = kBatchBytes / ((uint64_t)channels * 4U) / 16U * 16U;
    if (batch == 0)
        batch = 16U;
    for (uint64_t first = 0; first < positions; first += batch) {
        const uint64_t rows =
            positions - first < batch ? positions - first : batch;
        configure_lsu_transpose((uint32_t)rows * 4U, channels,
                                (uint32_t)positions * 4U, channels * 4U);
        __tx_lsu_dma_d2d<kLsuGroup>((long)(columns + first * channels),
                                    (long)(input + first));
        dayan_wait_lsu();
    }
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

__kernel__ void dayan_im2col_nchw_fp32(const float *input, float *columns,
                                       uint32_t channels, uint32_t input_h,
                                       uint32_t input_w, uint32_t output_h,
                                       uint32_t output_w, uint32_t kernel_h,
                                       uint32_t kernel_w, uint32_t stride_h,
                                       uint32_t stride_w, uint32_t pad_h,
                                       uint32_t pad_w) {
    if (output_h == 0 || output_w == 0)
        return;
    if (input == nullptr || columns == nullptr || channels == 0 ||
        input_h == 0 || input_w == 0 || kernel_h == 0 || kernel_w == 0 ||
        stride_h == 0 || stride_w == 0) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    const uint32_t kernel_size = channels * kernel_h * kernel_w;
    const uint64_t positions = (uint64_t)output_h * output_w;
    const uint32_t row_bytes = kernel_size * (uint32_t)sizeof(float);
    const uint64_t input_bytes =
        (uint64_t)channels * input_h * input_w * sizeof(float);
    const uint64_t columns_bytes = positions * kernel_size * sizeof(float);
    dayan_cache_invalidate(input, input_bytes);
    dayan_cache_invalidate(columns, columns_bytes);
    if (kernel_h == 1 && kernel_w == 1 && stride_h == 1 && stride_w == 1 &&
        pad_h == 0 && pad_w == 0 && (channels % 16U) == 0U &&
        (positions % 16U) == 0U) {
        transpose_columns(input, columns, channels, positions);
        return;
    }
    const uint64_t staging = (uint64_t)(uintptr_t)dayan_uram_base();
    const bool stage_in_uram = align_up_64(row_bytes) <= kUramBytes;
    if (stage_in_uram && (staging == 0 || dayan_uram_size() < kUramBytes))
        dayan_thread_exit(-2);
    const uint32_t batch_capacity =
        stage_in_uram ? kUramBytes / row_bytes : 1U;
    uint64_t batch_first = 0;
    uint32_t batch_count = 0;

    for (uint64_t row = shard_begin(positions); row < shard_end(positions);
         ++row) {
        const uint32_t output_y = (uint32_t)(row / output_w);
        const uint32_t output_x = (uint32_t)(row % output_w);
        float *destination = columns + row * kernel_size;
        volatile float *staged =
            stage_in_uram
                ? (volatile float *)(uintptr_t)staging +
                      (uint64_t)batch_count * kernel_size
                : nullptr;
        float *direct = stage_in_uram ? nullptr : destination;
        uint32_t index = 0;
        for (uint32_t channel = 0; channel < channels; ++channel) {
            const uint64_t plane =
                (uint64_t)channel * input_h * input_w;
            for (uint32_t kernel_y = 0; kernel_y < kernel_h; ++kernel_y) {
                const int64_t input_y =
                    (int64_t)output_y * stride_h + kernel_y - pad_h;
                const bool row_valid =
                    input_y >= 0 && input_y < (int64_t)input_h;
                for (uint32_t kernel_x = 0; kernel_x < kernel_w; ++kernel_x) {
                    const int64_t input_x =
                        (int64_t)output_x * stride_w + kernel_x - pad_w;
                    float value = 0.0f;
                    if (row_valid && input_x >= 0 &&
                        input_x < (int64_t)input_w)
                        value = input[plane + (uint64_t)input_y * input_w +
                                      (uint64_t)input_x];
                    if (stage_in_uram)
                        staged[index] = value;
                    else
                        direct[index] = value;
                    ++index;
                }
            }
        }
        if (!stage_in_uram)
            continue;
        if (batch_count == 0)
            batch_first = row;
        ++batch_count;
        const uint64_t shard_stop = shard_end(positions);
        if (batch_count < batch_capacity && row + 1 < shard_stop)
            continue;
        float *batch_destination = columns + batch_first * kernel_size;
        const uint32_t batch_elements = batch_count * kernel_size;
        if (((uintptr_t)batch_destination & 63U) == 0) {
            store_row(staging, batch_destination, batch_elements);
        } else {
            const volatile float *source =
                (const volatile float *)(uintptr_t)staging;
            for (uint32_t index = 0; index < batch_elements; ++index)
                batch_destination[index] = source[index];
        }
        batch_count = 0;
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(dayan_im2col_nchw_fp32, 4, 0, 0, 65536U, 1, 0, 0, 0);
