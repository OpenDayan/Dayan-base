// Slice：沿某一维取一段（step = +1 或 -1），元素类型为 FP32 / INT64。
// 每个 (outer 切片, 输出轴行) 对应一段连续拷贝，Proc 按逻辑分片顺序执行，
// 批量搬运交给 LSU。
//
// 关于 step 的设备端语义（实测 dycc 会误编译若干写法，务必按下面的方式写）：
//   * step 以 uint32 位模式传递：正序为 1，倒序为 0xFFFFFFFF（-1 的补码）。
//   * 设备端把 32 位运行时参数零扩展成 64 位保存，而把最高位为 1 的常量做符号
//     扩展，所以 `step != 0xFFFFFFFFU` 恒为真、(int32_t)step 的符号不可信；
//     用 `step >> 31` 推方向、或在条件表达式里用比较结果做选择，也会被算错。
//   * 可靠写法只有纯位运算与无符号回绕算术：
//       reverse   = (step & 0x80000000U) >> 31   —— 0 正序 / 1 倒序
//       delta     = 1U - (reverse << 1)          —— 1 或 0xFFFFFFFF
//       source    = start + index * delta        —— 回绕乘法即 start - index
//     以及比较两个运行时值（== / != / >=），例如下面的参数校验。
//     注意：`reverse * 大常量` 这类乘法也会被算错（实测 1 + reverse*0x7FFFFFFE
//     得到 1 而不是 0x7FFFFFFF），所以期望值一律从 delta 用位运算推出来。
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

// 单位元是元素而不是字节：FP32 为 4B，INT64 为 8B。
template <typename T>
__device__ static void copy_run(const T *source, T *destination,
                                uint32_t elements) {
    if (elements == 0)
        return;
    if ((((uintptr_t)source | (uintptr_t)destination) & 63U) != 0) {
        for (uint32_t index = 0; index < elements; ++index)
            destination[index] = source[index];
        return;
    }
    const uint32_t bytes = elements * (uint32_t)sizeof(T);
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

// delta 为 1（正序）或 0xFFFFFFFF（倒序）：
//   source_axis = start + axis_index * delta
// 倒序时 axis_index * 0xFFFFFFFF 在 32 位无符号回绕下正好是 -axis_index，
// 而 axis_index <= length - 1 <= start，所以结果落在 [0, start] 内，不会出界。
template <typename T>
__device__ static void slice_last_axis(const T *input, T *output,
                                       uint32_t outer, uint32_t input_axis,
                                       uint32_t start, uint32_t length,
                                       uint32_t delta, uint32_t inner) {
    const uint64_t rows = (uint64_t)outer * length;
    for (uint64_t row = shard_begin(rows); row < shard_end(rows); ++row) {
        const uint32_t surface = (uint32_t)(row / length);
        const uint32_t axis_index = (uint32_t)(row % length);
        const uint32_t source_axis = start + axis_index * delta;
        const uint64_t source_row =
            (uint64_t)surface * input_axis + source_axis;
        copy_run(input + source_row * inner, output + row * inner, inner);
    }
}

#endif
} // namespace

__kernel__ void rvnpu_slice_fp32(const float *input, float *output,
                                 uint32_t outer, uint32_t input_axis,
                                 uint32_t start, uint32_t length,
                                 uint32_t step, uint32_t inner) {
    const uint32_t reverse = (step & 0x80000000U) >> 31;
    const uint32_t delta = 1U - (reverse << 1);
    // step 只能是 +1 / -1：剥掉符号位后正序为 1、倒序为 0x7FFFFFFF，
    // 正好等于 delta 去掉符号位（只做位运算，避开设备端的乘法问题）。
    const uint32_t expected = delta & 0x7FFFFFFFU;
    if ((step & 0x7FFFFFFFU) != expected) {
        dayan_thread_exit(-22);
        return;
    }
    // 空输出（outer / length / inner 任一为 0）：无事可做，直接返回。
    if (outer == 0 || length == 0 || inner == 0)
        return;
    if (input == nullptr || output == nullptr || input_axis == 0) {
        dayan_thread_exit(-22);
        return;
    }
    // 边界检查沿用 npu-kernels 的 slice()：正序要求 start + length <= input_axis，
    // 倒序要求 start < input_axis 且 length <= start + 1。
    // 统一写成「最后一个输出下标」：正序 last = start + length - 1，
    // 倒序 last = start - length + 1；倒序越界时回绕成极大值，同样被拦住。
    const uint32_t last = start + (length - 1U) * delta;
    if (start >= input_axis || last >= input_axis) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    dayan_cache_invalidate(input, (uint64_t)outer*input_axis*inner*sizeof(float));
    dayan_cache_invalidate(output, (uint64_t)outer*length*inner*sizeof(float));
    slice_last_axis(input, output, outer, input_axis, start, length, delta,
                    inner);
#endif
}
DAYAN_KERNEL_TASK_DESC(rvnpu_slice_fp32, 4, 0, 0, 0, 1, 0, 0, 0);

__kernel__ void rvnpu_slice_int64(const int64_t *input, int64_t *output,
                                  uint32_t outer, uint32_t input_axis,
                                  uint32_t start, uint32_t length,
                                  uint32_t step, uint32_t inner) {
    const uint32_t reverse = (step & 0x80000000U) >> 31;
    const uint32_t delta = 1U - (reverse << 1);
    // step 只能是 +1 / -1：剥掉符号位后正序为 1、倒序为 0x7FFFFFFF，
    // 正好等于 delta 去掉符号位（只做位运算，避开设备端的乘法问题）。
    const uint32_t expected = delta & 0x7FFFFFFFU;
    if ((step & 0x7FFFFFFFU) != expected) {
        dayan_thread_exit(-22);
        return;
    }
    // 空输出（outer / length / inner 任一为 0）：无事可做，直接返回。
    if (outer == 0 || length == 0 || inner == 0)
        return;
    if (input == nullptr || output == nullptr || input_axis == 0) {
        dayan_thread_exit(-22);
        return;
    }
    // 边界检查沿用 npu-kernels 的 slice()：正序要求 start + length <= input_axis，
    // 倒序要求 start < input_axis 且 length <= start + 1。
    // 统一写成「最后一个输出下标」：正序 last = start + length - 1，
    // 倒序 last = start - length + 1；倒序越界时回绕成极大值，同样被拦住。
    const uint32_t last = start + (length - 1U) * delta;
    if (start >= input_axis || last >= input_axis) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    dayan_cache_invalidate(input, (uint64_t)outer*input_axis*inner*sizeof(int64_t));
    dayan_cache_invalidate(output, (uint64_t)outer*length*inner*sizeof(int64_t));
    slice_last_axis(input, output, outer, input_axis, start, length, delta,
                    inner);
#endif
}
DAYAN_KERNEL_TASK_DESC(rvnpu_slice_int64, 4, 0, 0, 0, 1, 0, 0, 0);
