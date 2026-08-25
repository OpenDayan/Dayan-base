#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stddef.h>
#include <stdint.h>

enum : uint32_t {
    kElements = 64,
    kTaskStride = 256,
    kLoadedOffset = 0,
    kSquareOffset = 64,
    kScaledOffset = 128,
    kScalarOffset = 192,
};

#if defined(__DAYAN_DEVICE__)

template <long Group>
static inline void configure_macro(uint32_t elements)
{
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM0>(
        elements - 1U);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM1>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM2>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM3>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0DIM1STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0DIM2STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0DIM3STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1DIM1STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1DIM2STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1DIM3STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTDIM1STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTDIM2STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTDIM3STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTSWIZZLEMBS>(0);
}

static inline uint32_t float_bits(float value)
{
    union {
        float f;
        uint32_t u;
    } bits;

    bits.f = value;
    return bits.u;
}

#endif

__kernel__ void normalization_probe(const float *input, float *trace,
                                    uint64_t publish_mode)
{
    const uint32_t task = dayan_task_id(DAYAN_TASK_X);
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    const uint64_t uram = (uint64_t)(uintptr_t)dayan_uram_base();
    const uint64_t input_uram = uram;
    const uint64_t square_uram = uram + 256;
    const uint64_t scaled_uram = uram + 512;
    float *scalar = (float *)(uintptr_t)(uram + 768);
    float *task_trace;
    long checksum = 0;

    if (!input || !trace || publish_mode > 4U || !uram ||
        dayan_uram_size() < 1024 ||
        dayan_task_mode() != 1U || !tasks || tasks > 4U || task >= tasks)
        dayan_thread_exit(-22);

    task_trace = trace + (uint64_t)task * kTaskStride;
    set_lsu_matrix<0>(checksum, 1, kElements * sizeof(float),
                      kElements * sizeof(float),
                      kElements * sizeof(float));

    __tx_lsu_dma_d2u<0>((long)input_uram,
                        (long)(input + (uint64_t)task * kElements));
    dayan_wait_lsu();
    __tx_lsu_dma_u2d<0>((long)(task_trace + kLoadedOffset),
                        (long)input_uram);
    dayan_wait_lsu();

    configure_macro<0>(kElements);
    __tx_macro_elemwise_mul_fp32<0, 0>((long)square_uram,
                                      (long)input_uram,
                                      (long)input_uram);
    dayan_wait_macro();
    __tx_lsu_dma_u2d<0>((long)(task_trace + kSquareOffset),
                        (long)square_uram);
    dayan_wait_lsu();

    configure_macro<0>(kElements);
    __tx_macro_reduce_add_fp32<0>((long)scalar, (long)square_uram);
    dayan_wait_macro();
    dayan_cache_invalidate(scalar, 64);
    const float sum = scalar[0];
    volatile float mean = sum;
    if (publish_mode != 4U)
        mean = sum / (float)kElements + 0.000001f;
    if (publish_mode == 2U) {
        task_trace[kScalarOffset + 0] = sum;
        task_trace[kScalarOffset + 1] = mean;
        dayan_cache_flush(task_trace + kScalarOffset, 64);
        set_lsu_matrix<0>(checksum, 1, 64, 64, 64);
        __tx_lsu_dma_d2u<0>((long)scalar,
                            (long)(task_trace + kScalarOffset));
        dayan_wait_lsu();
    } else if (publish_mode < 3U) {
        scalar[1] = mean;
    }
    if (publish_mode == 0U)
        dayan_cache_flush(scalar, 64);
    else if (publish_mode == 1U)
        __asm__ volatile("fence rw, rw" ::: "memory");

    long mean_address = (long)(scalar + 1);
    long inverse_address = (long)(scalar + 2);
    if (publish_mode >= 3U) {
        configure_macro<0>(1);
        __tx_macro_elemwise_sub_fp32<0, 0>((long)square_uram,
                                          (long)square_uram,
                                          (long)square_uram);
        __tx_macro_elemwise_add_fp32<0, 2>((long)square_uram,
                                          (long)float_bits(mean),
                                          (long)square_uram);
        dayan_wait_macro();
        mean_address = (long)square_uram;
        inverse_address = (long)scalar;
    }

    configure_macro<0>(1);
    __tx_macro_sfu_rsqrt_fp32<0>(inverse_address, mean_address);
    dayan_wait_macro();
    dayan_cache_invalidate(scalar, 64);
    const float inverse = publish_mode >= 3U ? scalar[0] : scalar[2];

    configure_macro<0>(kElements);
    __tx_macro_elemwise_mul_fp32<0, 1>((long)scaled_uram,
                                      inverse_address,
                                      (long)input_uram);
    dayan_wait_macro();
    set_lsu_matrix<0>(checksum, 1, kElements * sizeof(float),
                      kElements * sizeof(float),
                      kElements * sizeof(float));
    __tx_lsu_dma_u2d<0>((long)(task_trace + kScaledOffset),
                        (long)scaled_uram);
    dayan_wait_lsu();

    task_trace[kScalarOffset + 0] = sum;
    task_trace[kScalarOffset + 1] = mean;
    task_trace[kScalarOffset + 2] = inverse;
    task_trace[kScalarOffset + 3] = (float)(uintptr_t)dayan_uram_base();
    dayan_cache_flush(task_trace + kScalarOffset, 64);
}

DAYAN_KERNEL_LOCKED_TASK_DESC(normalization_probe, 0, 0, 32 * 1024,
                              1, 0, 0, 1);

#if !defined(__DAYAN_DEVICE__)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

extern "C" __host__ char *getenv(const char *name);

constexpr uint32_t kMaxTasks = 4;
constexpr uint32_t kTimeoutMs = 300000;

__host__ static void preserve_error(int candidate, int *status)
{
    if (*status == 0 && candidate < 0)
        *status = candidate;
}

__host__ static int wait_and_destroy(npu_event_t *event)
{
    npu_event_info_t info;
    int status;
    int destroy_status;

    if (!event)
        return -1;
    status = npu_event_wait(event, kTimeoutMs, &info);
    destroy_status = npu_event_destroy(event);
    if (status < 0)
        return status;
    if (info.status < std::numeric_limits<int>::min() ||
        info.status > std::numeric_limits<int>::max())
        return -1;
    if (info.status < 0)
        return (int)info.status;
    return destroy_status;
}

__host__ static int copy_h2d(npu_stream_t *stream, npu_buffer_t *buffer,
                             const void *source, size_t size)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_h2d_async(stream, buffer, 0, source, size,
                                      &event);

    return status < 0 ? status : wait_and_destroy(event);
}

__host__ static int copy_d2h(npu_stream_t *stream, void *destination,
                             npu_buffer_t *buffer, size_t size)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_d2h_async(stream, destination, buffer, 0, size,
                                      &event);

    return status < 0 ? status : wait_and_destroy(event);
}

__host__ static float input_value(uint32_t task, uint32_t element)
{
    return 0.03125f * (float)(1U + task * kElements + element);
}

__host__ static bool close_enough(float actual, float expected,
                                  float tolerance)
{
    return std::isfinite(actual) &&
           std::fabs(actual - expected) <=
               tolerance * (1.0f + std::fabs(expected));
}

__host__ static int validate(const float *trace, uint32_t tasks,
                             uint64_t publish_mode)
{
    int status = 0;

    for (uint32_t task = 0; task < tasks; ++task) {
        const float *record = trace + (uint64_t)task * kTaskStride;
        float sum = 0.0f;
        float loaded_error = 0.0f;
        float square_error = 0.0f;
        float scaled_error = 0.0f;

        for (uint32_t element = 0; element < kElements; ++element) {
            const float value = input_value(task, element);
            sum += value * value;
        }
        const float mean = publish_mode == 4U ? sum :
            sum / (float)kElements + 0.000001f;
        const float inverse = 1.0f / std::sqrt(mean);
        for (uint32_t element = 0; element < kElements; ++element) {
            const float value = input_value(task, element);
            loaded_error = std::fmax(
                loaded_error,
                std::fabs(record[kLoadedOffset + element] - value));
            square_error = std::fmax(
                square_error,
                std::fabs(record[kSquareOffset + element] - value * value));
            scaled_error = std::fmax(
                scaled_error,
                std::fabs(record[kScaledOffset + element] -
                          value * inverse));
        }

        std::printf("[host] task=%u load_err=%g square_err=%g "
                    "sum=%g/%g mean=%g/%g inv=%g/%g scaled_err=%g\n",
                    task, loaded_error, square_error,
                    record[kScalarOffset + 0], sum,
                    record[kScalarOffset + 1], mean,
                    record[kScalarOffset + 2], inverse, scaled_error);
        if (loaded_error > 1e-6f || square_error > 1e-5f ||
            !close_enough(record[kScalarOffset + 0], sum, 1e-5f) ||
            !close_enough(record[kScalarOffset + 1], mean, 1e-5f) ||
            !close_enough(record[kScalarOffset + 2], inverse, 2e-3f) ||
            scaled_error > 3e-3f)
            status = -1;
    }
    return status;
}

__host__ int main(int argc, char **argv)
{
    float input[kMaxTasks * kElements] = {};
    float trace[kMaxTasks * kTaskStride] = {};
    npu_runtime_options_t options;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *input_buffer = nullptr;
    npu_buffer_t *trace_buffer = nullptr;
    const char *socket_path;
    uint32_t tasks = 4;
    int status = 0;

    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("usage: %s [1|4] [flush|fence|lsu|macro|l2]\n", argv[0]);
        return 0;
    }
    uint64_t publish_mode = 0;

    if (argc >= 2 && std::strcmp(argv[1], "1") == 0)
        tasks = 1;
    else if (argc >= 2 && std::strcmp(argv[1], "4") != 0)
        return 2;
    if (argc == 3 && std::strcmp(argv[2], "fence") == 0)
        publish_mode = 1;
    else if (argc == 3 && std::strcmp(argv[2], "lsu") == 0)
        publish_mode = 2;
    else if (argc == 3 && std::strcmp(argv[2], "macro") == 0)
        publish_mode = 3;
    else if (argc == 3 && std::strcmp(argv[2], "l2") == 0)
        publish_mode = 4;
    else if (argc == 3 && std::strcmp(argv[2], "flush") != 0)
        return 2;
    else if (argc != 1 && argc != 2 && argc != 3)
        return 2;

    for (uint32_t task = 0; task < tasks; ++task)
        for (uint32_t element = 0; element < kElements; ++element)
            input[task * kElements + element] = input_value(task, element);

    npu_runtime_options_init(&options);
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        options.server_socket = socket_path;
    status = npu_runtime_open(&runtime, &options);
    if (status == 0)
        status = npu_context_create(runtime, nullptr, &context);
    if (status == 0)
        status = npu_stream_create(context, nullptr, &stream);
    if (status == 0) {
        npu_buffer_desc_t desc;
        npu_buffer_desc_init(&desc, sizeof(input));
        desc.alignment = 64;
        status = npu_buffer_create(context, &desc, &input_buffer);
    }
    if (status == 0) {
        npu_buffer_desc_t desc;
        npu_buffer_desc_init(&desc, sizeof(trace));
        desc.alignment = 64;
        status = npu_buffer_create(context, &desc, &trace_buffer);
    }
    if (status == 0)
        status = copy_h2d(stream, input_buffer, input, sizeof(input));
    if (status == 0)
        status = copy_h2d(stream, trace_buffer, trace, sizeof(trace));
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        npu_kernel_launch_attrs_init(&attrs, context, stream, "norm-probe");
        attrs.config.task_dim = NPUTaskDim3{tasks, 1, 1};
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms = kTimeoutMs;
        status = normalization_probe(input_buffer, trace_buffer,
                                     publish_mode, &attrs);
    }
    if (status == 0)
        status = copy_d2h(stream, trace, trace_buffer, sizeof(trace));
    if (status == 0)
        status = validate(trace, tasks, publish_mode);

    if (status == 0)
        std::printf("[host] PASS: normalization stages tasks=%u publish=%s\n",
                    tasks, publish_mode == 0 ? "flush" :
                           publish_mode == 1 ? "fence" :
                           publish_mode == 2 ? "lsu" :
                           publish_mode == 3 ? "macro" : "l2");
    else
        std::printf("[host] FAIL: normalization stages tasks=%u publish=%s "
                    "status=%d\n", tasks,
                    publish_mode == 0 ? "flush" :
                    publish_mode == 1 ? "fence" :
                    publish_mode == 2 ? "lsu" :
                    publish_mode == 3 ? "macro" : "l2", status);

    if (trace_buffer)
        preserve_error(npu_buffer_destroy(trace_buffer), &status);
    if (input_buffer)
        preserve_error(npu_buffer_destroy(input_buffer), &status);
    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status < 0 ? 1 : 0;
}

#endif
