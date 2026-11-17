// Resize(NCHW, nearest)：输出行分片。
// 每个输出行按 floor(oh * input_h / output_h) 找到对应输入行——整数比例时与
// 逐倍放大完全一致——输入行搬进 URAM 缓存（同一输入行只搬一次），
// 在 URAM 里展开成整行输出，再用 LSU 整行写回 DRAM。
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

// 一段数据 DRAM -> URAM；DRAM 侧非 64B 对齐时由 Proc 兜底。
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
        __tx_lsu_dma_d2u<kLsuGroup>((long)(destination + (uint64_t)blocks * 64U),
                                    (long)((const char *)source + blocks * 64U));
        dayan_wait_lsu();
    }
}

// 一段数据 URAM -> DRAM；DRAM 侧非 64B 对齐时由 Proc 兜底。
__device__ static void store_run(uint64_t source, float *destination,
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

#endif
} // namespace

__kernel__ void dayan_resize_nearest_nchw_fp32(const float *input,
                                               float *output, uint32_t batch,
                                               uint32_t channels,
                                               uint32_t input_h,
                                               uint32_t input_w,
                                               uint32_t output_h,
                                               uint32_t output_w) {
    if (batch == 0 || channels == 0 || output_h == 0 || output_w == 0)
        return;
    if (input == nullptr || output == nullptr || input_h == 0 || input_w == 0) {
        dayan_thread_exit(-22);
        return;
    }
#if defined(__DAYAN_DEVICE__)
    dayan_cache_invalidate(input, (uint64_t)batch*channels*input_h*input_w*sizeof(float));
    dayan_cache_invalidate(output, (uint64_t)batch*channels*output_h*output_w*sizeof(float));
    const uint32_t input_row_bytes = align_up_64(input_w * sizeof(float));
    const uint32_t output_row_bytes = align_up_64(output_w * sizeof(float));
    // 两行都放得下才用 URAM，否则直接按索引从 DRAM 取。
    const bool stage_in_uram =
        (uint64_t)input_row_bytes + output_row_bytes + 128U <= kUramBytes;
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    if (stage_in_uram && (uram == 0 || dayan_uram_size() < kUramBytes))
        dayan_thread_exit(-2);
    const uint64_t cached_row = uram;
    const uint64_t staged_row = uram + input_row_bytes;

    const uint64_t planes = (uint64_t)batch * channels;
    const uint64_t units = planes * output_h;
    uint64_t previous_row = ~(uint64_t)0;
    for (uint64_t unit = shard_begin(units); unit < shard_end(units); ++unit) {
        const uint32_t plane = (uint32_t)(unit / output_h);
        const uint32_t oh = (uint32_t)(unit % output_h);
        const uint64_t plane_base = (uint64_t)plane * input_h * input_w;
        const uint32_t ih = (uint32_t)(((uint64_t)oh * input_h) / output_h);
        if (stage_in_uram) {
            const uint64_t row_key = plane_base + (uint64_t)ih * input_w;
            if (row_key != previous_row) {
                load_run(input + row_key, cached_row, input_w);
                const volatile float *cached =
                    (const volatile float *)(uintptr_t)cached_row;
                volatile float *staged =
                    (volatile float *)(uintptr_t)staged_row;
                if ((uint64_t)output_w == (uint64_t)input_w * 2U) {
                    // YOLO nearest 2x: no per-output integer division, one
                    // read per source. Reuse the expanded row vertically.
                    for (uint32_t iw = 0; iw < input_w; ++iw) {
                        const float value = cached[iw];
                        staged[2U * iw] = value;
                        staged[2U * iw + 1U] = value;
                    }
                } else {
                    for (uint32_t ow = 0; ow < output_w; ++ow)
                        staged[ow] =
                            cached[(uint32_t)(((uint64_t)ow * input_w) / output_w)];
                }
                previous_row = row_key;
            }
            store_run(staged_row,
                      output + (plane * output_h + oh) * output_w, output_w);
        } else {
            float *destination =
                output + (plane * output_h + oh) * output_w;
            const float *row = input + plane_base + (uint64_t)ih * input_w;
            for (uint32_t ow = 0; ow < output_w; ++ow)
                destination[ow] =
                    row[(uint32_t)(((uint64_t)ow * input_w) / output_w)];
        }
    }
#endif
}
DAYAN_KERNEL_TASK_DESC(dayan_resize_nearest_nchw_fp32, 4, 0, 0, 65536U, 1, 0, 0, 0);
