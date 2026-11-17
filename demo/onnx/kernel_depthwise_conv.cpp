// Four-Core FP32 depthwise Conv for MobileNet-style NCHW tensors.
//
// Each Core owns complete channel planes. Input rows are moved to URAM once
// per output tile, scalar gathers form exact 1..16-wide vectors, and MACRO
// performs the multiply/add accumulation. No im2col tensor, channel padding,
// output-width padding, or invalid output positions are materialized.
#include <dayan_kernel.h>
#include <stdint.h>

struct DepthwiseConv2dConfig {
    uint64_t channels;
    uint64_t input_h, input_w;
    uint64_t output_h, output_w;
    uint64_t kernel_h, kernel_w;
    uint64_t stride_h, stride_w;
    uint64_t pad_h, pad_w;
    uint64_t has_bias;
};

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>

namespace {
enum : uint32_t {
    kVectorElements = 16,
    kVectorBytes = kVectorElements * sizeof(float),
    kRowBytes = 256,
    kUramBytes = 32 * 1024,
};

static long fp32_bits(float value)
{
    union {
        float f;
        uint32_t u;
    } bits = {value};
    return (long)bits.u;
}

static void configure_lsu(uint32_t transfer_bytes, uint32_t request_bytes)
{
    __tx_lsu_tcsr_write<LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<LSU_DIM0>(transfer_bytes / 64U - 1U);
    __tx_lsu_tcsr_write<LSU_DIM1>(0);
    __tx_lsu_tcsr_write<LSU_DIM2>(0);
    __tx_lsu_tcsr_write<LSU_INBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM1STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_INDIM2STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTBLOCKSTRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM0STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM1STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_OUTDIM2STRIDE>(64);
    __tx_lsu_tcsr_write<LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTREADSIZE>(request_bytes - 1U);
    __tx_lsu_tcsr_write<LSU_REQUESTPADDING>(request_bytes - 1U);
    __tx_lsu_tcsr_write<LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<LSU_REQUESTWRITESIZE>(request_bytes - 1U);
    __tx_lsu_tcsr_write<LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<LSU_TABLESTRIDE>(64);
}

static void load_bytes(uint64_t destination, uintptr_t source, uint32_t bytes)
{
    const uint32_t full_bytes = bytes & ~63U;
    const uint32_t remaining_bytes = bytes & 63U;
    if (full_bytes != 0) {
        configure_lsu(full_bytes, 64);
        __tx_lsu_dma_d2u<0>((long)destination, (long)source);
        dayan_wait_lsu();
    }
    if (remaining_bytes != 0) {
        configure_lsu(64, remaining_bytes);
        __tx_lsu_dma_d2u<0>((long)(destination + full_bytes),
                            (long)(source + full_bytes));
        dayan_wait_lsu();
    }
}

static void store_vector(uintptr_t destination, uint64_t source,
                         uint32_t elements)
{
    configure_lsu(64, elements * sizeof(float));
    __tx_lsu_dma_u2d<0>((long)destination, (long)source);
    dayan_wait_lsu();
}

static void configure_macro(uint32_t elements)
{
    __tx_macro_tcsr_write<MACRO_DIM0>(elements - 1U);
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

static uint32_t shard_begin(uint32_t total)
{
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    return (uint32_t)((uint64_t)total * dayan_task_id(DAYAN_TASK_X) / tasks);
}

static uint32_t shard_end(uint32_t total)
{
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    return (uint32_t)((uint64_t)total *
                      (dayan_task_id(DAYAN_TASK_X) + 1U) / tasks);
}

static bool valid_config(const DepthwiseConv2dConfig *config)
{
    if (!config || !config->channels || !config->input_h ||
        !config->input_w || !config->output_h || !config->output_w ||
        (config->kernel_h != 3U && config->kernel_h != 5U) ||
        config->kernel_w != config->kernel_h ||
        (config->stride_h != 1U && config->stride_h != 2U) ||
        (config->stride_w != 1U && config->stride_w != 2U) ||
        config->has_bias > 1U)
        return false;
    if (config->input_h + 2U * config->pad_h < config->kernel_h ||
        config->input_w + 2U * config->pad_w < config->kernel_w)
        return false;
    return config->output_h ==
               (config->input_h + 2U * config->pad_h - config->kernel_h) /
                       config->stride_h +
                   1U &&
           config->output_w ==
               (config->input_w + 2U * config->pad_w - config->kernel_w) /
                       config->stride_w +
                   1U;
}
} // namespace
#endif

__kernel__ void dayan_depthwise_conv2d_nchw_fp32(
    const DepthwiseConv2dConfig *config, const float *input,
    const float *weight, const float *bias, float *output)
{
#if defined(__DAYAN_DEVICE__)
    dayan_cache_invalidate(config, sizeof(*config));
    if (!valid_config(config) || !input || !weight || !bias || !output ||
        dayan_task_dim(DAYAN_TASK_X) == 0U ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U ||
        (dayan_task_mode() != 1U && dayan_task_mode() != 4U))
        dayan_thread_exit(-22);

    const uint64_t input_elements =
        config->channels * config->input_h * config->input_w;
    const uint64_t weight_elements =
        config->channels * config->kernel_h * config->kernel_w;
    const uint64_t output_elements =
        config->channels * config->output_h * config->output_w;
    dayan_cache_invalidate(input, input_elements * sizeof(float));
    dayan_cache_invalidate(weight, weight_elements * sizeof(float));
    if (config->has_bias)
        dayan_cache_invalidate(bias, config->channels * sizeof(float));
    dayan_cache_invalidate(output, output_elements * sizeof(float));

    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    if (!uram || dayan_uram_size() < kUramBytes)
        dayan_thread_exit(-12);
    const uint64_t row_at = uram;
    const uint64_t sample_at = row_at + kRowBytes;
    const uint64_t product_at = sample_at + kVectorBytes;
    const uint64_t accumulator_at = product_at + kVectorBytes;
    volatile float *sample =
        (volatile float *)(uintptr_t)sample_at;
    volatile float *accumulator =
        (volatile float *)(uintptr_t)accumulator_at;

    const uint32_t channels = (uint32_t)config->channels;
    const uint32_t input_h = (uint32_t)config->input_h;
    const uint32_t input_w = (uint32_t)config->input_w;
    const uint32_t output_h = (uint32_t)config->output_h;
    const uint32_t output_w = (uint32_t)config->output_w;
    const uint32_t kernel_h = (uint32_t)config->kernel_h;
    const uint32_t kernel_w = (uint32_t)config->kernel_w;
    const uint32_t stride_h = (uint32_t)config->stride_h;
    const uint32_t stride_w = (uint32_t)config->stride_w;
    const uint32_t pad_h = (uint32_t)config->pad_h;
    const uint32_t pad_w = (uint32_t)config->pad_w;
    const uint64_t input_plane = (uint64_t)input_h * input_w;
    const uint64_t output_plane = (uint64_t)output_h * output_w;
    const uint64_t kernel_plane = (uint64_t)kernel_h * kernel_w;

    for (uint32_t channel = shard_begin(channels);
         channel < shard_end(channels); ++channel) {
        const float initial = config->has_bias ? bias[channel] : 0.0f;
        for (uint32_t oh = 0; oh < output_h; ++oh) {
            for (uint32_t output_begin = 0; output_begin < output_w;
                 output_begin += kVectorElements) {
                const uint32_t output_count =
                    output_w - output_begin < kVectorElements
                        ? output_w - output_begin
                        : kVectorElements;
                for (uint32_t index = 0; index < output_count; ++index)
                    accumulator[index] = initial;

                const int64_t width_begin =
                    (int64_t)output_begin * stride_w - pad_w;
                const int64_t width_end =
                    (int64_t)(output_begin + output_count - 1U) * stride_w -
                    pad_w + kernel_w;
                const int32_t clamped_begin =
                    width_begin < 0 ? 0 : (int32_t)width_begin;
                const int32_t clamped_end =
                    width_end > input_w ? (int32_t)input_w
                                        : (int32_t)width_end;

                for (uint32_t kh = 0; kh < kernel_h; ++kh) {
                    const int64_t ih =
                        (int64_t)oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= input_h ||
                        clamped_begin >= clamped_end)
                        continue;
                    const uint64_t input_index =
                        (uint64_t)channel * input_plane +
                        (uint64_t)ih * input_w + (uint32_t)clamped_begin;
                    const uintptr_t source =
                        (uintptr_t)(input + input_index);
                    const uintptr_t aligned = source & ~(uintptr_t)63;
                    const uint32_t prefix = (uint32_t)(source - aligned);
                    const uint32_t span =
                        (uint32_t)(clamped_end - clamped_begin);
                    const uint32_t bytes = prefix + span * sizeof(float);
                    if (bytes > kRowBytes)
                        dayan_thread_exit(-12);
                    load_bytes(row_at, aligned, bytes);
                    const volatile float *row =
                        (const volatile float *)(uintptr_t)row_at;
                    const uint32_t prefix_elements = prefix / sizeof(float);

                    for (uint32_t kw = 0; kw < kernel_w; ++kw) {
                        for (uint32_t index = 0; index < output_count;
                             ++index) {
                            const int64_t iw =
                                (int64_t)(output_begin + index) * stride_w +
                                kw - pad_w;
                            sample[index] =
                                iw >= 0 && iw < input_w
                                    ? row[prefix_elements +
                                          (uint32_t)(iw - clamped_begin)]
                                    : 0.0f;
                        }
                        const float coefficient =
                            weight[(uint64_t)channel * kernel_plane +
                                   (uint64_t)kh * kernel_w + kw];
                        configure_macro(output_count);
                        __tx_macro_elemwise_mul_fp32<0, 2>(
                            (long)product_at, fp32_bits(coefficient),
                            (long)sample_at);
                        dayan_wait_macro();
                        __tx_macro_elemwise_add_fp32<0, 0>(
                            (long)accumulator_at, (long)accumulator_at,
                            (long)product_at);
                        dayan_wait_macro();
                    }
                }

                const uint64_t output_index =
                    (uint64_t)channel * output_plane +
                    (uint64_t)oh * output_w + output_begin;
                store_vector((uintptr_t)(output + output_index),
                             accumulator_at, output_count);
            }
        }
    }
#endif
}

// One resident Proc per Core keeps MACRO/LSU state private to each channel
// shard while the fixed four-Core launch provides model-level parallelism.
DAYAN_KERNEL_TASK_DESC(dayan_depthwise_conv2d_nchw_fp32, 1, 0, 0,
                       32768U, 1, 0, 0, 1);
