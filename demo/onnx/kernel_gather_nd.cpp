// GatherND：按 INT64 [rows, index_depth] 索引从数据张量中取切片。
// index_depth = 1 取整块 [d1, d2]，= 2 取一行 d2，= 3 取单个元素。
// 每行是一段连续拷贝，批量段交给 LSU。
#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stdint.h>

namespace {
#if defined(__DAYAN_DEVICE__)

__device__ constexpr long kLsuGroup = 0;

__device__ static inline void configure_lsu_linear(uint32_t blocks) {
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
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<kLsuGroup * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
}

// 一整块连续数据：两侧 64B 对齐且长度是 64B 整数倍时交给 LSU，
// 否则 Proc 逐元素兜底（GatherND 的每一行宽度经常不是 64B 的整数倍）。
template <typename T>
__device__ static void copy_segment(const T *source, T *destination,
                                    uint32_t elements) {
    if (elements == 0)
        return;
    const uint32_t bytes = elements * (uint32_t)sizeof(T);
    if (bytes % 64U == 0U &&
        ((((uintptr_t)source | (uintptr_t)destination) & 63U) == 0U)) {
        configure_lsu_linear(bytes / 64U);
        __tx_lsu_dma_d2d<kLsuGroup>((long)destination, (long)source);
        dayan_wait_lsu();
        return;
    }
    for (uint32_t index = 0; index < elements; ++index)
        destination[index] = source[index];
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

__device__ static void gather_nd(const float *data, const int64_t *indices,
                                 float *output, uint32_t rows,
                                 uint32_t index_depth, uint32_t d0, uint32_t d1,
                                 uint32_t d2) {
    const uint32_t width = index_depth == 1 ? d1 * d2
                           : index_depth == 2 ? d2
                                              : 1U;
    for (uint32_t row = shard_begin(rows); row < shard_end(rows); ++row) {
        const uint64_t offset = (uint64_t)row * index_depth;
        const int64_t i0 = indices[offset];
        if (i0 < 0 || i0 >= (int64_t)d0)
            dayan_thread_exit(-2);
        uint64_t source = (uint64_t)i0 * d1 * d2;
        if (index_depth >= 2U) {
            const int64_t i1 = indices[offset + 1U];
            if (i1 < 0 || i1 >= (int64_t)d1)
                dayan_thread_exit(-2);
            source += (uint64_t)i1 * d2;
        }
        if (index_depth >= 3U) {
            const int64_t i2 = indices[offset + 2U];
            if (i2 < 0 || i2 >= (int64_t)d2)
                dayan_thread_exit(-2);
            source += (uint64_t)i2;
        }
        copy_segment(data + source, output + (uint64_t)row * width, width);
    }
}

#endif
} // namespace

__kernel__ void dayan_gather_nd_fp32(const float *data, const int64_t *indices,
                                     float *output, uint32_t rows,
                                     uint32_t index_depth, uint32_t d0,
                                     uint32_t d1, uint32_t d2) {
    if (rows == 0)
        return;
    if (data == nullptr || indices == nullptr || output == nullptr ||
        index_depth == 0 || index_depth > 3 || d0 == 0) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    const uint64_t data_bytes = (uint64_t)d0 * d1 * d2 * sizeof(float);
    const uint64_t output_bytes =
        (uint64_t)rows * (index_depth == 1 ? d1 * d2 :
                          index_depth == 2 ? d2 : 1U) * sizeof(float);
    dayan_cache_invalidate(data, data_bytes);
    dayan_cache_invalidate(indices,
                           (uint64_t)rows * index_depth * sizeof(int64_t));
    dayan_cache_invalidate(output, output_bytes);
    gather_nd(data, indices, output, rows, index_depth, d0, d1, d2);
    dayan_cache_flush(output, output_bytes);
#endif
}
DAYAN_KERNEL_TASK_DESC(dayan_gather_nd_fp32, 4, 0, 0, 0, 1, 0, 0, 0);

__kernel__ void dayan_gather_nd_int64(const int64_t *data,
                                      const int64_t *indices, int64_t *output,
                                      uint32_t rows, uint32_t index_depth,
                                      uint32_t d0, uint32_t d1, uint32_t d2) {
    if (rows == 0)
        return;
    if (data == nullptr || indices == nullptr || output == nullptr ||
        index_depth == 0 || index_depth > 3 || d0 == 0) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    const uint32_t width = index_depth == 1 ? d1 * d2
                           : index_depth == 2 ? d2
                                              : 1U;
    const uint64_t data_bytes = (uint64_t)d0 * d1 * d2 * sizeof(int64_t);
    const uint64_t output_bytes = (uint64_t)rows * width * sizeof(int64_t);
    dayan_cache_invalidate(data, data_bytes);
    dayan_cache_invalidate(indices,
                           (uint64_t)rows * index_depth * sizeof(int64_t));
    dayan_cache_invalidate(output, output_bytes);
    for (uint32_t row = shard_begin(rows); row < shard_end(rows); ++row) {
        const uint64_t offset = (uint64_t)row * index_depth;
        const int64_t i0 = indices[offset];
        if (i0 < 0 || i0 >= (int64_t)d0)
            dayan_thread_exit(-2);
        uint64_t source = (uint64_t)i0 * d1 * d2;
        if (index_depth >= 2U) {
            const int64_t i1 = indices[offset + 1U];
            if (i1 < 0 || i1 >= (int64_t)d1)
                dayan_thread_exit(-2);
            source += (uint64_t)i1 * d2;
        }
        if (index_depth >= 3U) {
            const int64_t i2 = indices[offset + 2U];
            if (i2 < 0 || i2 >= (int64_t)d2)
                dayan_thread_exit(-2);
            source += (uint64_t)i2;
        }
        copy_segment(data + source, output + (uint64_t)row * width, width);
    }
    dayan_cache_flush(output, output_bytes);
#endif
}
DAYAN_KERNEL_TASK_DESC(dayan_gather_nd_int64, 4, 0, 0, 0, 1, 0, 0, 0);
