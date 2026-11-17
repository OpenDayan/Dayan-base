// ArgMax：沿最后一维取最大值下标，输出 INT64。
// Proc 按逻辑分片认领若干行，逐 tile 执行 Load -> MACRO reduce_max(带下标)
// -> 记录最优 (值, 下标)，尾 tile 与非对齐行由 Proc 兜底。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>
// 内嵌共享流水实现：本算子独立保留 config/shape/partition/loader/run 结构。
// ReduceMax/ArgMax 共用的连续短行路径；不改变长行及非对齐行的原回退。
// 双 tile 路径在能够容纳两块 tile 时重叠下一批 D2U 和当前批 MACRO 归约。
struct ReductionRowsConfig {
    enum : uint32_t {
        kTileElements = 4096, kMaxRows = 32, kUramBytes = 32U * 1024U,
    };
    enum : long { kLsuGroup = 0, kMacroGroup = 0 };
};

struct ReductionRowsShape {
    uint32_t begin, end, batch_rows, columns;
};

#if defined(__DAYAN_DEVICE__)
namespace reduction_rows_detail {

using Config = ReductionRowsConfig;

static_assert(Config::kTileElements * sizeof(float) +
                  2 * Config::kMaxRows * sizeof(uint32_t) <= Config::kUramBytes,
              "短行归约缓冲必须落在原有32KiB资源内");

template <typename Output>
__device__ static ReductionRowsShape partition(Output *output, uint32_t rows,
                                               uint32_t columns) {
    const uint32_t tasks = dayan_task_dim(0);
    const uint32_t task = dayan_task_id(0);
    if (tasks < 2)
        return {0, rows, 0, columns};
    // Proc写回按cache line拥有输出，不能让不同核更新同一条脏缓存行。
    // 非对齐输出保留正确性，由task 0串行处理。
    if (((uintptr_t)output & 63U) != 0)
        return {task == 0 ? 0U : rows, rows, 0, columns};
    constexpr uint32_t per_line = 64U / sizeof(Output);
    const uint64_t groups = ((uint64_t)rows + per_line - 1U) / per_line;
    const uint64_t begin = groups * task / tasks * per_line;
    const uint64_t end = groups * (task + 1U) / tasks * per_line;
    return {(uint32_t)(begin < rows ? begin : rows),
            (uint32_t)(end < rows ? end : rows), 0, columns};
}

__device__ static void configure_load(uint32_t elements) {
    constexpr long g = Config::kLsuGroup * LSU_GROUPSZ;
    __tx_lsu_tcsr_write<g + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<g + LSU_DIM0>(elements / 16U - 1U);
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

__device__ static void start_load(const float *input, uint64_t destination,
                                  uint32_t elements) {
    configure_load(elements);
    __tx_lsu_dma_d2u<Config::kLsuGroup>(
        (long)destination, (long)input);
}

__device__ static void configure_reduce(uint32_t columns, uint32_t rows) {
    constexpr long g = Config::kMacroGroup * MACRO_GROUPSZ;
    __tx_macro_tcsr_write<g + MACRO_DIM0>(columns - 1U);
    __tx_macro_tcsr_write<g + MACRO_DIM1>(rows - 1U);
    __tx_macro_tcsr_write<g + MACRO_DIM2>(0);
    __tx_macro_tcsr_write<g + MACRO_DIM3>(0);
    __tx_macro_tcsr_write<g + MACRO_SRC0DIM1STRIDE>(columns * sizeof(float));
    __tx_macro_tcsr_write<g + MACRO_SRC0DIM2STRIDE>(0);
    __tx_macro_tcsr_write<g + MACRO_SRC0DIM3STRIDE>(0);
    __tx_macro_tcsr_write<g + MACRO_SRC1DIM1STRIDE>(sizeof(uint32_t));
    __tx_macro_tcsr_write<g + MACRO_SRC1DIM2STRIDE>(0);
    __tx_macro_tcsr_write<g + MACRO_SRC1DIM3STRIDE>(0);
    __tx_macro_tcsr_write<g + MACRO_DSTDIM1STRIDE>(sizeof(float));
    __tx_macro_tcsr_write<g + MACRO_DSTDIM2STRIDE>(0);
    __tx_macro_tcsr_write<g + MACRO_DSTDIM3STRIDE>(0);
    __tx_macro_tcsr_write<g + MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<g + MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<g + MACRO_DSTSWIZZLEMBS>(0);
}

template <bool WithIndex, typename Output>
__device__ static bool run(const float *input, Output *output,
                          uint32_t rows, uint32_t columns) {
    if (columns == 0 || columns > Config::kTileElements ||
        columns % 16U != 0 || ((uintptr_t)input & 63U) != 0)
        return false;
    const uint32_t capacity = Config::kTileElements / columns;
    ReductionRowsShape shape = partition(output, rows, columns);
    shape.batch_rows = capacity < Config::kMaxRows ? capacity : Config::kMaxRows;
    if (shape.begin == shape.end)
        return true;
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    if (uram == 0 || dayan_uram_size() < Config::kUramBytes) {
        dayan_thread_exit(-2);
        return true;
    }
    const uint32_t batch_elements = shape.batch_rows * columns;
    const uint32_t tile_span =
        (batch_elements * sizeof(float) + 63U) & ~63U;
    const uint32_t scratch_bytes =
        2U * Config::kMaxRows * (uint32_t)sizeof(uint32_t);
    const bool pipelined =
        2U * tile_span + scratch_bytes <= Config::kUramBytes;
    const uint64_t tile[2] = {
        uram,
        uram + (pipelined ? tile_span : 0U),
    };
    const uint64_t values = uram + (pipelined
        ? 2U * tile_span
        : Config::kTileElements * sizeof(float));
    const uint64_t indices = values + Config::kMaxRows * sizeof(float);
    if (!pipelined && tile_span + scratch_bytes > Config::kUramBytes)
        return false;

    uint32_t begin = shape.begin;
    uint32_t count = (shape.end - begin) < shape.batch_rows
        ? shape.end - begin : shape.batch_rows;
    uint32_t slot = 0;
    uint32_t configured_load = 0;
    uint32_t configured_rows = 0;
    start_load(input + (uint64_t)begin * shape.columns, tile[slot],
               count * shape.columns);
    configured_load = count;
    dayan_wait_lsu();

    for (;;) {
        // 当前批的 MACRO 配置必须在发射前完成；下一批 LSU 配置可以
        // 在当前 MACRO 运行期间提前完成。
        if (count != configured_rows) {
            configure_reduce(shape.columns, count);
            configured_rows = count;
        }
        __tx_macro_reduce_max_fp32<Config::kMacroGroup, WithIndex ? 1 : 0>(
            (long)values, (long)tile[slot], WithIndex ? (long)indices : 0);

        const uint32_t next_begin = begin + count;
        const bool has_next = next_begin < shape.end;
        uint32_t next_count = 0;
        uint32_t next_slot = slot;
        if (has_next) {
            const uint32_t remaining = shape.end - next_begin;
            next_count = remaining < shape.batch_rows
                ? remaining : shape.batch_rows;
            next_slot = pipelined ? 1U - slot : slot;
            if (pipelined) {
                if (next_count != configured_load) {
                    configure_load(next_count * shape.columns);
                    configured_load = next_count;
                }
                __tx_lsu_dma_d2u<Config::kLsuGroup>(
                    (long)tile[next_slot],
                    (long)(input + (uint64_t)next_begin * shape.columns));
            }
        }

        dayan_wait_macro();
        for (uint32_t row = 0; row < count; ++row) {
            const float value = ((volatile float *)(uintptr_t)values)[row];
            if constexpr (WithIndex) {
                // 硬件并列值可能返回最后下标；保持ONNX首个下标语义。
                uint32_t index = ((volatile uint32_t *)(uintptr_t)indices)[row];
                const volatile float *source =
                    (volatile float *)(uintptr_t)tile[slot] +
                    row * shape.columns;
                for (uint32_t k = 0; k < shape.columns; ++k) {
                    if (source[k] == value) {
                        index = k;
                        break;
                    }
                }
                output[begin + row] = (Output)index;
            } else {
                output[begin + row] = (Output)value;
            }
        }
        if (!has_next)
            break;
        if (!pipelined) {
            if (next_count != configured_load) {
                configure_load(next_count * shape.columns);
                configured_load = next_count;
            }
            __tx_lsu_dma_d2u<Config::kLsuGroup>(
                (long)tile[next_slot],
                (long)(input + (uint64_t)next_begin * shape.columns));
        }
        dayan_wait_lsu();
        begin = next_begin;
        count = next_count;
        slot = next_slot;
    }
    return true;
}

} // namespace reduction_rows_detail
#endif


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

#endif
} // namespace

__kernel__ void dayan_argmax_lastdim_fp32(const float *input, int64_t *output,
                                          uint32_t rows, uint32_t columns,
                                          uint32_t extra) {
    // 空张量直接返回，不算参数错误。
    if (rows == 0 || columns == 0)
        return;
    if (input == nullptr || output == nullptr || extra != 0) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    const auto shape = reduction_rows_detail::partition(output, rows, columns);
    if (shape.begin == shape.end)
        return;
    dayan_cache_invalidate(output + shape.begin,
        (uint64_t)(shape.end - shape.begin) * sizeof(int64_t));
    if (reduction_rows_detail::run<true>(input, output, rows, columns))
        return;
    dayan_cache_invalidate(input + (uint64_t)shape.begin * columns,
        (uint64_t)(shape.end - shape.begin) * columns * sizeof(float));
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    const uint32_t scratch_bytes =
        align_up_64(kTileElements * (uint32_t)sizeof(float));
    if (uram == 0 || scratch_bytes + 128U > kUramBytes ||
        dayan_uram_size() < kUramBytes)
        dayan_thread_exit(-2);
    const uint64_t tile_value = uram + scratch_bytes;
    const uint64_t tile_index = uram + scratch_bytes + 64U;

    for (uint32_t row = shape.begin; row < shape.end; ++row) {
        float best_value = negative_infinity();
        uint32_t best_index = 0;
        bool found = false;
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
            uint32_t local = *(volatile uint32_t *)(uintptr_t)tile_index;
            // RTL 并列返回最后下标，ONNX 要第一个。
            const volatile float *tile =
                (const volatile float *)(uintptr_t)uram;
            for (uint32_t k = 0; k < count; ++k) {
                if (tile[k] == candidate) {
                    local = k;
                    break;
                }
            }
            if (!found || candidate > best_value) {
                found = true;
                best_value = candidate;
                best_index = begin + local;
            }
        }
        output[row] = (int64_t)best_index;
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(dayan_argmax_lastdim_fp32, 4, 0, 0, 32768U, 1, 0, 0, 1);
