#include <dayan_kernel.h>

#include <stddef.h>
#include <stdint.h>

struct ElementwiseConfig {
    const float *a;
    const float *b;
    float *output;
    uint64_t elements;
};

struct ElementwiseShape {
    enum : uint32_t {
        kRequestElements = 16,
        kBinaryTileElements = 2048,
        kBinaryTileBytes = kBinaryTileElements * sizeof(float),
        kUramBytes = 32768,
    };
};
static_assert(3 * ElementwiseShape::kBinaryTileBytes <= ElementwiseShape::kUramBytes);
static_assert(ElementwiseShape::kBinaryTileElements %
                  ElementwiseShape::kRequestElements ==
              0);

#define DAYAN_SCALAR_DESC(name) DAYAN_KERNEL_TASK_DESC(name, 4, 0, 0, 0, 0, 0, 0, 0)
#define DAYAN_VECTOR_DESC(name) DAYAN_KERNEL_TASK_DESC(name, 4, 0, 0, 32768, 1, 0, 0, 1)
#define DAYAN_VECTOR_PIPELINE_DESC(name)                                               \
    DAYAN_KERNEL_TASK_DESC(name, 4, 1, 0, 32768, 2, 0, 0, 2)

#define ELEMENTWISE_TILE_ELEMS 1024U
#define ELEMENTWISE_TILE_BYTES 4096U
#define ELEMENTWISE_LANE_BYTES (3U * ELEMENTWISE_TILE_BYTES)

#define LEGACY_REQUEST_ELEMS 16U
#define LEGACY_TILE_ELEMS 256U
#define LEGACY_TILE_BYTES (LEGACY_TILE_ELEMS * sizeof(float))

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace elementwise_detail
{

static void invalid_arguments(void) { dayan_thread_exit(-22); }
static inline void check_result(int result)
{
    if (result < 0)
        dayan_thread_exit(result);
}
static inline long fp32_bits(float value)
{
    union {
        float f;
        uint32_t u;
    } bits = {value};
    return (long)bits.u;
}
static bool task_range(uint64_t elements, uint64_t *begin_out, uint64_t *end_out)
{
    const uint64_t task = (uint64_t)dayan_task_id(DAYAN_TASK_X);
    const uint64_t tasks = (uint64_t)dayan_task_dim(DAYAN_TASK_X);
    if (!tasks || task >= tasks) {
        *begin_out = 0;
        *end_out = 0;
        return false;
    }
    const uint64_t requests =
        (elements + LEGACY_REQUEST_ELEMS - 1U) / LEGACY_REQUEST_ELEMS;
    const uint64_t requests_per_task = requests / tasks;
    const uint64_t extra_requests = requests % tasks;
    const uint64_t task_requests =
        requests_per_task + (task < extra_requests ? 1U : 0U);
    const uint64_t first_request =
        task * requests_per_task + (task < extra_requests ? task : extra_requests);
    const uint64_t begin = first_request * LEGACY_REQUEST_ELEMS;
    uint64_t end = (first_request + task_requests) * LEGACY_REQUEST_ELEMS;
    if (end > elements)
        end = elements;
    *begin_out = begin;
    *end_out = end;
    return task_requests != 0 && begin < end;
}
static inline uint64_t tile_len(uint64_t end, uint64_t base)
{
    if (base >= end)
        return 0;
    const uint64_t remaining = end - base;
    return remaining < LEGACY_TILE_ELEMS ? remaining : LEGACY_TILE_ELEMS;
}
static inline uint64_t segment_len(uint64_t remaining)
{
    if (remaining < LEGACY_REQUEST_ELEMS)
        return remaining;
    return remaining - remaining % LEGACY_REQUEST_ELEMS;
}
static inline void configure_macro_1d(uint32_t n);
static inline void configure_legacy_lsu_macro(uint64_t len)
{
    const uint64_t requests = (len + LEGACY_REQUEST_ELEMS - 1U) / LEGACY_REQUEST_ELEMS;
    const uint64_t request_elements =
        len < LEGACY_REQUEST_ELEMS ? len : LEGACY_REQUEST_ELEMS;
    __tx_lsu_tcsr_write<LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<LSU_DIM0>(requests - 1U);
    __tx_lsu_tcsr_write<LSU_DIM1>(0);
    __tx_lsu_tcsr_write<LSU_DIM2>(0);
    __tx_lsu_tcsr_write<LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(LEGACY_TILE_BYTES);
    __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM1STRIDE>(LEGACY_TILE_BYTES);
    __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTPADDING>(request_elements * sizeof(float) - 1U);
    __tx_lsu_tcsr_write<LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(request_elements * sizeof(float) - 1U);
    __tx_lsu_tcsr_write<LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<LSU_TABLESTRIDE>(64);
    configure_macro_1d((uint32_t)len);
}
static inline uint64_t legacy_resources(uint64_t n)
{
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    const uint32_t mode = dayan_task_mode();
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    if (!uram || !n || (mode != 1U && mode != 4U) ||
        (mode == 4U && tasks != 4U) || dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U || dayan_uram_size() < 32768)
        invalid_arguments();
    return uram;
}
static inline void store_legacy(long device, long uram)
{
    __tx_lsu_dma_u2d<0>(device, uram);
    dayan_wait_lsu();
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
    if (!u || !n || (n & 1023) || dayan_uram_size() < 32768)
        invalid_arguments();
    configure_contiguous_lsu(ELEMENTWISE_TILE_BYTES);
    configure_macro_1d(ELEMENTWISE_TILE_ELEMS);
    return u;
}

enum class ElementwiseOp { Add, Mul, Sub };

template <ElementwiseOp Op>
static inline void elementwise_inst(uint64_t dst, uint64_t lhs, uint64_t rhs)
{
    if (Op == ElementwiseOp::Add)
        __tx_macro_elemwise_add_fp32<0, 0>((long)dst, (long)lhs, (long)rhs);
    else if (Op == ElementwiseOp::Mul)
        __tx_macro_elemwise_mul_fp32<0, 0>((long)dst, (long)lhs, (long)rhs);
    else
        __tx_macro_elemwise_sub_fp32<0, 0>((long)dst, (long)lhs, (long)rhs);
}

template <ElementwiseOp Op> static void binary_entry(const ElementwiseConfig &cfg)
{
    const float *a = cfg.a;
    const float *b = cfg.b;
    float *c = cfg.output;
    const uint64_t n = cfg.elements;
    if (!a || !b || !c)
        invalid_arguments();
    uint64_t begin = 0, end = 0;
    if (!task_range(n, &begin, &end))
        return;
    const uint64_t uram = legacy_resources(n);
    const long a_uram = (long)uram;
    const long b_uram = (long)(uram + ElementwiseShape::kBinaryTileBytes);
    const long c_uram = (long)(uram + 2U * ElementwiseShape::kBinaryTileBytes);
    uint64_t configured = 0;
    // One proc owns the group. Full batches reuse the configuration; an
    // unaligned tail is split into whole 64-byte requests and an exact final
    // request, so no DMA crosses the logical tensor or task boundary.
    for (uint64_t base = begin; base < end;
         base += ElementwiseShape::kBinaryTileElements) {
        const uint64_t remaining = end - base;
        const uint64_t len = remaining < ElementwiseShape::kBinaryTileElements
                                 ? remaining
                                 : ElementwiseShape::kBinaryTileElements;
        for (uint64_t offset = 0; offset < len;) {
            const uint64_t segment = segment_len(len - offset);
            const uint64_t index = base + offset;
            if (configured != segment) {
                configure_legacy_lsu_macro(segment);
                configured = segment;
            }
            __tx_lsu_dma_d2u<0>(a_uram, (long)((uintptr_t)a + index * sizeof(float)));
            dayan_wait_lsu();
            __tx_lsu_dma_d2u<0>(b_uram, (long)((uintptr_t)b + index * sizeof(float)));
            dayan_wait_lsu();
            elementwise_inst<Op>(c_uram, a_uram, b_uram);
            dayan_wait_macro();
            store_legacy((long)((uintptr_t)c + index * sizeof(float)), c_uram);
            offset += segment;
        }
    }
}

static void legacy_relu_entry(const float *input, float *output, uint64_t n)
{
    if (!input || !output)
        invalid_arguments();
    uint64_t begin = 0, end = 0;
    if (!task_range(n, &begin, &end))
        return;
    const uint64_t uram = legacy_resources(n);
    const long input_uram = (long)uram;
    const long output_uram = (long)(uram + LEGACY_TILE_BYTES);
    const long zero_uram = (long)(uram + 2U * LEGACY_TILE_BYTES);
    *(float *)zero_uram = 0.0f;
    for (uint64_t base = begin; base < end; base += LEGACY_TILE_ELEMS) {
        const uint64_t len = tile_len(end, base);
        for (uint64_t offset = 0; offset < len;) {
            const uint64_t segment = segment_len(len - offset);
            const uint64_t index = base + offset;
            configure_legacy_lsu_macro(segment);
            __tx_lsu_dma_d2u<0>(input_uram,
                                (long)((uintptr_t)input + index * sizeof(float)));
            dayan_wait_lsu();
            __tx_macro_elemwise_max_fp32<0, 1>(output_uram, zero_uram, input_uram);
            dayan_wait_macro();
            store_legacy((long)((uintptr_t)output + index * sizeof(float)),
                         output_uram);
            offset += segment;
        }
    }
}

static void legacy_hardsigmoid_entry(const float *input, float *output, uint64_t n,
                                     uint64_t coeff_bits)
{
    if (!input || !output)
        invalid_arguments();
    uint64_t begin = 0, end = 0;
    if (!task_range(n, &begin, &end))
        return;
    const uint64_t uram = legacy_resources(n);
    const long input_uram = (long)uram;
    const long middle_uram = (long)(uram + LEGACY_TILE_BYTES);
    const long output_uram = (long)(uram + 2U * LEGACY_TILE_BYTES);
    const uint32_t alpha_bits = (uint32_t)coeff_bits;
    const uint32_t beta_bits = (uint32_t)(coeff_bits >> 32);
    for (uint64_t base = begin; base < end; base += LEGACY_TILE_ELEMS) {
        const uint64_t len = tile_len(end, base);
        for (uint64_t offset = 0; offset < len;) {
            const uint64_t segment = segment_len(len - offset);
            const uint64_t index = base + offset;
            configure_legacy_lsu_macro(segment);
            __tx_lsu_dma_d2u<0>(input_uram,
                                (long)((uintptr_t)input + index * sizeof(float)));
            dayan_wait_lsu();
            __tx_macro_elemwise_mul_fp32<0, 2>(middle_uram, (long)alpha_bits,
                                               input_uram);
            dayan_wait_macro();
            __tx_macro_elemwise_add_fp32<0, 2>(middle_uram, (long)beta_bits,
                                               middle_uram);
            dayan_wait_macro();
            __tx_macro_elemwise_max_fp32<0, 2>(output_uram, fp32_bits(0.0f),
                                               middle_uram);
            dayan_wait_macro();
            __tx_macro_elemwise_min_fp32<0, 2>(output_uram, fp32_bits(1.0f),
                                               output_uram);
            dayan_wait_macro();
            store_legacy((long)((uintptr_t)output + index * sizeof(float)),
                         output_uram);
            offset += segment;
        }
    }
}

static void legacy_hardswish_entry(const float *input, float *output, uint64_t n)
{
    if (!input || !output)
        invalid_arguments();
    uint64_t begin = 0, end = 0;
    if (!task_range(n, &begin, &end))
        return;
    const uint64_t uram = legacy_resources(n);
    const long input_uram = (long)uram;
    const long middle_uram = (long)(uram + LEGACY_TILE_BYTES);
    const long output_uram = (long)(uram + 2U * LEGACY_TILE_BYTES);
    for (uint64_t base = begin; base < end; base += LEGACY_TILE_ELEMS) {
        const uint64_t len = tile_len(end, base);
        for (uint64_t offset = 0; offset < len;) {
            const uint64_t segment = segment_len(len - offset);
            const uint64_t index = base + offset;
            configure_legacy_lsu_macro(segment);
            __tx_lsu_dma_d2u<0>(input_uram,
                                (long)((uintptr_t)input + index * sizeof(float)));
            dayan_wait_lsu();
            __tx_macro_elemwise_add_fp32<0, 2>(middle_uram, fp32_bits(3.0f),
                                               input_uram);
            dayan_wait_macro();
            __tx_macro_elemwise_max_fp32<0, 2>(middle_uram, fp32_bits(0.0f),
                                               middle_uram);
            dayan_wait_macro();
            __tx_macro_elemwise_min_fp32<0, 2>(middle_uram, fp32_bits(6.0f),
                                               middle_uram);
            dayan_wait_macro();
            __tx_macro_elemwise_mul_fp32<0, 0>(output_uram, input_uram, middle_uram);
            dayan_wait_macro();
            __tx_macro_elemwise_mul_fp32<0, 2>(output_uram, fp32_bits(1.0f / 6.0f),
                                               output_uram);
            dayan_wait_macro();
            store_legacy((long)((uintptr_t)output + index * sizeof(float)),
                         output_uram);
            offset += segment;
        }
    }
}

struct State {
    const float *a;
    const float *b;
    float *c;
    uint64_t blocks;
    uint64_t uram;
};

struct LoadArgs {
    State *state;
    int lane;
};

template <ElementwiseOp Op> static void elementwise_lane(void *argument)
{
    const LoadArgs *args = (LoadArgs *)argument;
    const State *s = args->state;
    const uint64_t a_uram = s->uram + (uint64_t)args->lane * ELEMENTWISE_LANE_BYTES;
    const uint64_t b_uram = a_uram + ELEMENTWISE_TILE_BYTES;
    const uint64_t c_uram = b_uram + ELEMENTWISE_TILE_BYTES;
    configure_contiguous_lsu(ELEMENTWISE_TILE_BYTES);
    configure_macro_1d(ELEMENTWISE_TILE_ELEMS);
    for (uint64_t q = args->lane; q < s->blocks; q += 2) {
        const uint64_t offset = q * ELEMENTWISE_TILE_ELEMS;
        __tx_lsu_dma_d2u<0>((long)a_uram, (long)(s->a + offset));
        __tx_lsu_dma_d2u<0>((long)b_uram, (long)(s->b + offset));
        dayan_wait_lsu();
        elementwise_inst<Op>(c_uram, a_uram, b_uram);
        dayan_wait_macro();
        __tx_lsu_dma_u2d<0>((long)(s->c + offset), (long)c_uram);
        dayan_wait_lsu();
    }
}

template <ElementwiseOp Op>
static void elementwise_entry(const float *a, const float *b, float *c, uint64_t n)
{
    if (!a || !b || !c || dayan_task_mode() != 1 ||
        dayan_task_dim(DAYAN_TASK_X) != 1U || dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U)
        invalid_arguments();
    const uint64_t uram = vector_resources(n);
    const uint64_t blocks = n / ELEMENTWISE_TILE_ELEMS;
    State s = {a, b, c, blocks, uram};
    LoadArgs lanes[2] = {{&s, 0}, {&s, 1}};
    dayan_thread_resources_t resources = {};
    resources.lsu_tcsr_count = 1;
    resources.macro_tcsr_count = 1;
    check_result(dayan_create_thread("elementwise", elementwise_lane<Op>, &lanes[1],
                                     &resources));
    elementwise_lane<Op>(&lanes[0]);
    check_result(dayan_thread_join_all());
}
} // namespace elementwise_detail
#endif

__kernel__ void dayan_add_fp32(const float *a, const float *b, float *c, uint64_t n)
{
#if defined(__DAYAN_DEVICE__)
    using namespace elementwise_detail;
    binary_entry<ElementwiseOp::Add>({a, b, c, n});
#endif
}

DAYAN_VECTOR_DESC(dayan_add_fp32);

__kernel__ void dayan_mul_fp32(const float *a, const float *b, float *c, uint64_t n)
{
#if defined(__DAYAN_DEVICE__)
    using namespace elementwise_detail;
    binary_entry<ElementwiseOp::Mul>({a, b, c, n});
#endif
}

DAYAN_VECTOR_DESC(dayan_mul_fp32);

__kernel__ void dayan_sub_fp32(const float *a, const float *b, float *c, uint64_t n)
{
#if defined(__DAYAN_DEVICE__)
    using namespace elementwise_detail;
    elementwise_entry<ElementwiseOp::Sub>(a, b, c, n);
#endif
}

DAYAN_VECTOR_PIPELINE_DESC(dayan_sub_fp32);

__kernel__ void dayan_relu_fp32(const float *input, float *output, uint64_t n)
{
#if defined(__DAYAN_DEVICE__)
    using namespace elementwise_detail;
    legacy_relu_entry(input, output, n);
#endif
}
DAYAN_VECTOR_DESC(dayan_relu_fp32);

__kernel__ void dayan_hardsigmoid_fp32(const float *input, float *output, uint64_t n,
                                       uint64_t coeff_bits)
{
#if defined(__DAYAN_DEVICE__)
    using namespace elementwise_detail;
    legacy_hardsigmoid_entry(input, output, n, coeff_bits);
#endif
}
DAYAN_VECTOR_DESC(dayan_hardsigmoid_fp32);

__kernel__ void dayan_hardswish_fp32(const float *input, float *output, uint64_t n)
{
#if defined(__DAYAN_DEVICE__)
    using namespace elementwise_detail;
    legacy_hardswish_entry(input, output, n);
#endif
}
DAYAN_VECTOR_DESC(dayan_hardswish_fp32);

__kernel__ void dayan_div_fp32(const float *a, const float *b, float *c, uint64_t n)
{
#if defined(__DAYAN_DEVICE__)
    using namespace elementwise_detail;
    const uint64_t uram = vector_resources(n);
    const uint64_t blocks = n / ELEMENTWISE_TILE_ELEMS;
    for (uint64_t q = 0; q < blocks; ++q) {
        const uint64_t au = uram;
        const uint64_t bu = uram + ELEMENTWISE_TILE_BYTES;
        const uint64_t reciprocal = uram + 2U * ELEMENTWISE_TILE_BYTES;
        const uint64_t cu = uram + 3U * ELEMENTWISE_TILE_BYTES;
        __tx_lsu_dma_d2u<0>((long)au, (long)(a + q * 1024U));
        dayan_wait_lsu();
        __tx_lsu_dma_d2u<0>((long)bu, (long)(b + q * 1024U));
        dayan_wait_lsu();
        __tx_macro_sfu_recip_fp32<0>(reciprocal, bu);
        __tx_macro_elemwise_mul_fp32<0, 0>(cu, au, reciprocal);
        dayan_wait_macro();
        __tx_lsu_dma_u2d<0>((long)(c + q * 1024U), (long)cu);
        dayan_wait_lsu();
    }
#endif
}
DAYAN_VECTOR_DESC(dayan_div_fp32);

__kernel__ void dayan_binary_broadcast_fp32(const float *a, const float *b, float *c,
                                            uint32_t a0, uint32_t a1, uint32_t a2,
                                            uint32_t a3, uint32_t b0, uint32_t b1,
                                            uint32_t b2, uint32_t b3, uint32_t op)
{
#if defined(__DAYAN_DEVICE__)
    using namespace elementwise_detail;
    if (!a || !b || !c || op > 3)
        invalid_arguments();
    const uint32_t ad[4] = {a0, a1, a2, a3};
    const uint32_t bd[4] = {b0, b1, b2, b3};
    uint32_t od[4];
    bool a_is_output = true;
    for (int d = 0; d < 4; ++d) {
        od[d] = ad[d] > bd[d] ? ad[d] : bd[d];
        a_is_output = a_is_output && ad[d] == od[d];
    }
    const uint64_t output_elements =
        (uint64_t)od[0] * od[1] * od[2] * od[3];
    const uint64_t total = output_elements;
    if (!total)
        invalid_arguments();
    // Dimensions are fastest-first. Equal inner dimensions form a contiguous
    // binary span; only the outer coordinates need broadcast index arithmetic.
    // Reuse exactly the same batching, tail handling and arithmetic as Add/Mul.
    uint64_t contiguous = 1;
    for (int d = 0; d < 4 && ad[d] == bd[d]; ++d)
        contiguous *= ad[d];
    // Keep every row start cache-line aligned; irregular inner spans retain
    // the scalar fallback until their DMA alignment has been RTL-validated.
    if (op <= 1 && a_is_output &&
        contiguous >= ElementwiseShape::kRequestElements &&
        contiguous % ElementwiseShape::kRequestElements == 0) {
        for (uint64_t base = 0; base < total; base += contiguous) {
            uint64_t remaining = base, b_index = 0, b_stride = 1;
            for (int d = 0; d < 4; ++d) {
                const uint64_t coordinate = remaining % ad[d];
                remaining /= ad[d];
                if (bd[d] != 1)
                    b_index += coordinate * b_stride;
                b_stride *= bd[d];
            }
            const ElementwiseConfig cfg{a + base, b + b_index, c + base, contiguous};
            if (op == 0)
                binary_entry<ElementwiseOp::Add>(cfg);
            else
                binary_entry<ElementwiseOp::Mul>(cfg);
        }
        return;
    }
    dayan_cache_invalidate(
        a, (uint64_t)a0 * a1 * a2 * a3 * sizeof(float));
    dayan_cache_invalidate(b, (uint64_t)b0 * b1 * b2 * b3 * sizeof(float));
    dayan_cache_invalidate(c, total * sizeof(float));
    uint64_t a_stride[4] = {1U, 1U, 1U, 1U};
    uint64_t b_stride[4] = {1U, 1U, 1U, 1U};
    for (int d = 1; d < 4; ++d) {
        a_stride[d] = a_stride[d - 1] * (uint64_t)ad[d - 1];
        b_stride[d] = b_stride[d - 1] * (uint64_t)bd[d - 1];
    }
    for (uint64_t i = 0; i < total; ++i) {
        uint64_t r = i, ai = 0, bi = 0;
        for (int d = 0; d < 4; ++d) {
            const uint32_t x = (uint32_t)(r % od[d]);
            r /= od[d];
            if (ad[d] != 1U)
                ai += (uint64_t)x * a_stride[d];
            if (bd[d] != 1U)
                bi += (uint64_t)x * b_stride[d];
        }
        float v = op == 0   ? a[ai] + b[bi]
                  : op == 1 ? a[ai] * b[bi]
                  : op == 2 ? a[ai] - b[bi]
                            : a[ai] / b[bi];
        c[i] = v;
    }
#endif
}

DAYAN_VECTOR_DESC(dayan_binary_broadcast_fp32);
