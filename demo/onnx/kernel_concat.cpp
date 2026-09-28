// Concat：沿某一维把多段输入拼进同一个输出张量。
// 每个 (outer 切片, 输入轴行) 对应一段连续拷贝，Proc 按逻辑分片顺序执行
// Load/Store，批量搬运交给 LSU。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

namespace {
#if defined(__DAYAN_DEVICE__)

__device__ constexpr long kLsuGroup = 0;

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

// 一个 Task 坐标负责的输出坐标区间 [begin, end)。
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

// 连续数据搬运：两侧 64B 对齐时由 LSU 分块完成，否则 Proc 逐元素兜底。
__device__ static void copy_run(const float *source, float *destination,
                                uint32_t elements) {
    if (elements == 0)
        return;
    if ((((uintptr_t)source | (uintptr_t)destination) & 63U) != 0) {
        for (uint32_t index = 0; index < elements; ++index)
            destination[index] = source[index];
        return;
    }
    const uint32_t bytes = elements * sizeof(float);
    const uint32_t blocks = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    if (blocks != 0) {
        configure_lsu_linear(blocks, 64);
        __tx_lsu_dma_d2d<kLsuGroup>((long)destination, (long)source);
        dayan_wait_lsu();
    }
    if (tail != 0) {
        configure_lsu_linear(1, tail);
        __tx_lsu_dma_d2d<kLsuGroup>(
            (long)((char *)destination + blocks * 64U),
            (long)((const char *)source + blocks * 64U));
        dayan_wait_lsu();
    }
}

#endif
} // namespace

__kernel__ void rvnpu_concat_fp32(const float *input, float *output,
                                  uint32_t outer, uint32_t input_axis,
                                  uint32_t output_axis, uint32_t inner,
                                  uint32_t axis_offset) {
    if (output_axis < input_axis || axis_offset > output_axis - input_axis) {
        dayan_thread_exit(-22);
        return;
    }
    // 空张量不参与拼接：直接返回，不能算参数错误。
    if (outer == 0 || input_axis == 0 || inner == 0)
        return;
    if (input == nullptr || output == nullptr) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    dayan_cache_invalidate(input, (uint64_t)outer*input_axis*inner*sizeof(float));
    dayan_cache_invalidate(output, (uint64_t)outer*output_axis*inner*sizeof(float));
    // 输入侧共有 outer * input_axis 行连续数据，每行 inner 个元素。
    const uint64_t rows = (uint64_t)outer * input_axis;
    for (uint64_t row = shard_begin(rows); row < shard_end(rows); ++row) {
        const uint32_t surface = (uint32_t)(row / input_axis);
        const uint32_t axis_index = (uint32_t)(row % input_axis);
        const uint64_t destination_row =
            (uint64_t)surface * output_axis + axis_offset + axis_index;
        copy_run(input + row * inner, output + destination_row * inner, inner);
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(rvnpu_concat_fp32, 4, 0, 0, 0, 1, 0, 0, 0);
