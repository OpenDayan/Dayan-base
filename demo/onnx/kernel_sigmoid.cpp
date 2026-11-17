#include <dayan_kernel.h>

#include <stddef.h>
#include <stdint.h>

struct SigmoidConfig {
    const float *input;
    float *output;
    uint64_t elements;
};

#ifndef RVNPU_SIGMOID_BATCH_ELEMS
#define RVNPU_SIGMOID_BATCH_ELEMS 4096U
#endif
struct SigmoidShape {
    enum : uint32_t {
        kTileElements = 1024,
        kBatchElements = RVNPU_SIGMOID_BATCH_ELEMS,
        kBatchBytes = sizeof(float) * kBatchElements,
        kBankBytes = 2 * kBatchBytes,
        kUramBytes = (2 * kBankBytes + 32767U) & ~32767U,
    };
};
static_assert(SigmoidShape::kBatchElements >= SigmoidShape::kTileElements &&
                  SigmoidShape::kBatchElements % SigmoidShape::kTileElements == 0,
              "Sigmoid batch must contain whole padded blocks");

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace sigmoid_detail
{

static void invalid_arguments(void) { dayan_thread_exit(-22); }
static inline uint32_t f32_bits(float v)
{
    uint32_t b;
    __builtin_memcpy(&b, &v, 4);
    return b;
}
static inline void configure_contiguous_lsu(uint64_t bytes)
{
    __tx_lsu_tcsr_write<LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<LSU_DIM0>((bytes + 63) / 64 - 1);
    __tx_lsu_tcsr_write<LSU_DIM1>(0);
    __tx_lsu_tcsr_write<LSU_DIM2>(0);
    __tx_lsu_tcsr_write<LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(bytes);
    __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM1STRIDE>(bytes);
    __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<LSU_TABLESTRIDE>(64);
}
static inline void configure_macro_1d(uint32_t n)
{
    __tx_macro_tcsr_write<MACRO_DIM0>(n - 1);
    __tx_macro_tcsr_write<MACRO_DIM1>(0);
    __tx_macro_tcsr_write<MACRO_DIM2>(0);
    __tx_macro_tcsr_write<MACRO_DIM3>(0);
    __tx_macro_tcsr_write<MACRO_SRC0DIM1STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC0DIM2STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC0DIM3STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC1DIM1STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC1DIM2STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC1DIM3STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_DSTDIM1STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_DSTDIM2STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_DSTDIM3STRIDE>(0);
    __tx_macro_tcsr_write<MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<MACRO_DSTSWIZZLEMBS>(0);
}
static inline uint64_t vector_resources(uint64_t n)
{
    const uint64_t u = (uint64_t)(uintptr_t)dayan_uram_base();
    if (!u || !n || (n & (SigmoidShape::kTileElements - 1)) ||
        dayan_uram_size() < SigmoidShape::kUramBytes)
        invalid_arguments();
    return u;
}
template <long G, typename T, typename U>
static inline void macro_iv_add_fp32(T r, float v, U s)
{
    __tx_macro_elemwise_add_fp32<G, 2>(r, (long)f32_bits(v), (long)s);
}
template <long G, typename T, typename U>
static inline void macro_iv_mul_fp32(T r, float v, U s)
{
    __tx_macro_elemwise_mul_fp32<G, 2>(r, (long)f32_bits(v), (long)s);
}

static void compute_batch(uint64_t values)
{
    const uint64_t work = values + SigmoidShape::kBatchBytes;
    macro_iv_mul_fp32<0>(work, -1.4426950408889634f, values);
    __tx_macro_sfu_exp2_fp32<0>(values, work);
    macro_iv_add_fp32<0>(work, 1.0f, values);
    __tx_macro_sfu_recip_fp32<0>(values, work);
}

// One proc, two disjoint URAM banks. Prefetch overlaps current compute;
// current writeback overlaps next compute. No child-thread scheduling/join.
static void sigmoid_pipeline(const SigmoidConfig &cfg, uint64_t uram)
{
    const float *input = cfg.input;
    float *output = cfg.output;
    const uint64_t n = cfg.elements;
    uint64_t offset = 0;
    uint32_t count =
        n < SigmoidShape::kBatchElements ? (uint32_t)n : SigmoidShape::kBatchElements;
    uint32_t bank = 0;
    // This proc exclusively owns group 0. Initialize invariant fields once;
    // only DIM0 changes for a short final batch (DIM1/2 remain zero).
    configure_contiguous_lsu(count * sizeof(float));
    configure_macro_1d(count);
    uint32_t lsu_count = count;
    __tx_lsu_dma_d2u<0>((long)uram, (long)input);
    dayan_wait_lsu();
    compute_batch(uram);
    while (true) {
        const uint64_t next_offset = offset + count;
        const uint64_t here = uram + (uint64_t)bank * SigmoidShape::kBankBytes;
        const uint64_t next = uram + (uint64_t)(bank ^ 1U) * SigmoidShape::kBankBytes;
        uint32_t next_count = 0;
        if (next_offset < n) {
            next_count = n - next_offset < SigmoidShape::kBatchElements
                             ? (uint32_t)(n - next_offset)
                             : SigmoidShape::kBatchElements;
            if (lsu_count != next_count) {
                __tx_lsu_tcsr_write<LSU_DIM0>(next_count * sizeof(float) / 64 - 1);
                lsu_count = next_count;
            }
            __tx_lsu_dma_d2u<0>((long)next, (long)(input + next_offset));
            dayan_wait_lsu();
        }
        dayan_wait_macro();
        if (lsu_count != count) {
            __tx_lsu_tcsr_write<LSU_DIM0>(count * sizeof(float) / 64 - 1);
            lsu_count = count;
        }
        __tx_lsu_dma_u2d<0>((long)(output + offset), (long)here);
        if (next_count) {
            // Previous MACRO work has completed; never mutate a live shape.
            if (next_count != count)
                __tx_macro_tcsr_write<MACRO_DIM0>(next_count - 1);
            compute_batch(next);
        }
        dayan_wait_lsu(); // bank 'here' is reusable only after its store.
        if (!next_count)
            break;
        offset = next_offset;
        count = next_count;
        bank ^= 1U;
    }
}
} // namespace sigmoid_detail
#endif

__kernel__ void dayan_sigmoid_f32(const float *x, float *y, uint32_t d0, uint32_t d1,
                                  uint32_t d2, uint32_t d3)
{
#if defined(__DAYAN_DEVICE__)
    using namespace sigmoid_detail;
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    if (!((dayan_task_mode() == 1 && tasks == 1U) ||
          (dayan_task_mode() == 4 && tasks == 4U)) ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U || dayan_task_dim(DAYAN_TASK_Z) != 1U)
        invalid_arguments();
    const uint64_t n = (uint64_t)d0 * d1 * d2 * d3;
    const uint64_t uram = vector_resources(n);
    // Keep 1024-element partition alignment. Distribute whole tiles
    // across cores, so no partial DMA/cache line is shared between tasks.
    const uint64_t blocks = n / SigmoidShape::kTileElements;
    const uint64_t first = blocks * dayan_task_id(DAYAN_TASK_X) / tasks;
    const uint64_t last = blocks * (dayan_task_id(DAYAN_TASK_X) + 1U) / tasks;
    if (first == last)
        return;
    const SigmoidConfig cfg{x + first * SigmoidShape::kTileElements,
                            y + first * SigmoidShape::kTileElements,
                            (last - first) * SigmoidShape::kTileElements};
    sigmoid_pipeline(cfg, uram);
#endif
}
DAYAN_KERNEL_TASK_DESC(dayan_sigmoid_f32, 4, 0, 0, SigmoidShape::kUramBytes, 1, 0, 0,
                       1);
