// Concat：沿某一维把多段输入拼进同一个输出张量。
// 每个 (outer 切片, 输入轴行) 对应一段连续拷贝，Proc 按逻辑分片顺序执行
// Load/Store，批量搬运交给 LSU。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>
// 内嵌共享流水实现：本算子独立保留 config/shape/partition/loader/run 结构。
// Concat/Split共用：同一surface内连续行合并成DMA，不越过任务或surface边界。
struct CopyRowsConfig {
    enum : uint32_t { kMaxElements = 64U * 1024U, kElementsPerBeat = 16U };
    enum : long { kLsuGroup = 0 };
};

struct CopyRowsShape {
    uint32_t outer, rows_per_surface, source_axis, destination_axis, inner;
    uint32_t source_offset, destination_offset;
};

#if defined(__DAYAN_DEVICE__)
namespace copy_rows_detail {
using Config = CopyRowsConfig;

__device__ static inline void configure(uint32_t blocks) {
    constexpr long g = Config::kLsuGroup * LSU_GROUPSZ;
    __tx_lsu_tcsr_write<g + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<g + LSU_DIM0>(blocks - 1U);
    __tx_lsu_tcsr_write<g + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<g + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<g + LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<g + LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<g + LSU_INDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<g + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<g + LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<g + LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<g + LSU_OUTDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<g + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<g + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<g + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<g + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<g + LSU_REQUESTWRITESIZE>(63);
}

__device__ static bool run(const float *input, float *output, const CopyRowsShape &shape) {
    if ((((uintptr_t)input | (uintptr_t)output) & 63U) != 0 ||
        shape.inner % Config::kElementsPerBeat != 0)
        return false;
    if (!shape.rows_per_surface || !shape.inner || !shape.outer)
        return true;
    const uint64_t rows = (uint64_t)shape.outer * shape.rows_per_surface;
    const uint32_t tasks = dayan_task_dim(0), task = dayan_task_id(0);
    const uint64_t begin = tasks > 1 ? rows * task / tasks : 0;
    const uint64_t end = tasks > 1 ? rows * (task + 1U) / tasks : rows;
    uint32_t configured_blocks = 0;
    for (uint64_t row = begin; row < end;) {
        const uint64_t surface = row / shape.rows_per_surface;
        const uint32_t axis_row = (uint32_t)(row % shape.rows_per_surface);
        const uint64_t available = shape.rows_per_surface - axis_row;
        const uint64_t count = end - row < available ? end - row : available;
        const float *source = input +
            (surface * shape.source_axis + shape.source_offset + axis_row) * shape.inner;
        float *destination = output +
            (surface * shape.destination_axis + shape.destination_offset + axis_row) * shape.inner;
        const uint64_t elements = count * shape.inner;
        for (uint64_t offset = 0; offset < elements;) {
            const uint64_t remaining = elements - offset;
            const uint32_t chunk = remaining < Config::kMaxElements ?
                (uint32_t)remaining : Config::kMaxElements;
            const uint32_t blocks = chunk / Config::kElementsPerBeat;
            if (!configured_blocks)
                configure(blocks);
            else if (blocks != configured_blocks)
                __tx_lsu_tcsr_write<Config::kLsuGroup * LSU_GROUPSZ + LSU_DIM0>(blocks - 1U);
            configured_blocks = blocks;
            __tx_lsu_dma_d2d<Config::kLsuGroup>((long)(destination + offset),
                                              (long)(source + offset));
            dayan_wait_lsu();
            offset += chunk;
        }
        row += count;
    }
    return true;
}
} // namespace copy_rows_detail
#endif


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
    if (copy_rows_detail::run(input, output,
            {outer, input_axis, input_axis, output_axis, inner, 0, axis_offset}))
        return;
    // 非对齐行会走Proc写回，由一个核拥有输出，避免共享脏cache line。
    if (dayan_task_id(0) != 0)
        return;
    const uintptr_t alignment = (uintptr_t)input | (uintptr_t)output |
        (uintptr_t)(inner * sizeof(float));
    if ((alignment & 63U) != 0) {
        dayan_cache_invalidate(input, (uint64_t)outer*input_axis*inner*sizeof(float));
        dayan_cache_invalidate(output, (uint64_t)outer*output_axis*inner*sizeof(float));
    }
    // 输入侧共有 outer * input_axis 行连续数据，每行 inner 个元素。
    const uint64_t rows = (uint64_t)outer * input_axis;
    for (uint64_t row = 0; row < rows; ++row) {
        const uint32_t surface = (uint32_t)(row / input_axis);
        const uint32_t axis_index = (uint32_t)(row % input_axis);
        const uint64_t destination_row =
            (uint64_t)surface * output_axis + axis_offset + axis_index;
        copy_run(input + row * inner, output + destination_row * inner, inner);
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(rvnpu_concat_fp32, 4, 0, 0, 0, 1, 0, 0, 0);
