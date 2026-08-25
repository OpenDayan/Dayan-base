#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stddef.h>
#include <stdint.h>

struct alignas(64) TensorInterruptResult {
    uint32_t before;
    uint32_t after;
    uint64_t reserved[6];
};

static_assert(sizeof(TensorInterruptResult) == 64,
              "tensor interrupt result ABI");

__kernel__ void tensor_interrupt_probe(TensorInterruptResult *result)
{
    if (!result)
        dayan_thread_exit(-22);

    result->before = 0x13579bdfU;
    dayan_cache_flush(result, sizeof(*result));

    /*
     * The Proc owns one logical LSU group, so group 1 is deliberately out of
     * range. The access must raise source 12 and remain resumable.
     */
    __tx_lsu_tcsr_write<LSU_GROUPSZ + LSU_DIM0>(0x2aU);

    result->after = 0x2468ace0U;
    dayan_cache_flush(result, sizeof(*result));
}

DAYAN_KERNEL_DESC(tensor_interrupt_probe, 0, 0, 0, 1, 0, 0, 0);

#if !defined(__DAYAN_DEVICE__)

#include <cstdio>
#include <cstring>
#include <limits>

extern "C" __host__ char *getenv(const char *name);

namespace {

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

__host__ static int download(npu_stream_t *stream, void *destination,
                             npu_buffer_t *buffer, size_t size)
{
    npu_event_t *event = nullptr;
    const int status = npu_memcpy_d2h_async(
        stream, destination, buffer, 0, size, &event);

    return status < 0 ? status : wait_and_destroy(event);
}

} // namespace

__host__ int main(int argc, char **argv)
{
    TensorInterruptResult result = {};
    npu_runtime_options_t options;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *buffer = nullptr;
    int status = 0;

    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("usage: %s\n", argv[0]);
        return 0;
    }
    if (argc != 1)
        return 2;

    npu_runtime_options_init(&options);
    const char *socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        options.server_socket = socket_path;
    status = npu_runtime_open(&runtime, &options);
    if (status == 0)
        status = npu_context_create(runtime, nullptr, &context);
    if (status == 0)
        status = npu_stream_create(context, nullptr, &stream);
    if (status == 0) {
        npu_buffer_desc_t desc;

        npu_buffer_desc_init(&desc, sizeof(result));
        desc.alignment = 64;
        status = npu_buffer_create(context, &desc, &buffer);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;

        npu_kernel_launch_attrs_init(&attrs, context, stream,
                                     "interrupt-probe");
        attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms = kTimeoutMs;
        status = tensor_interrupt_probe(buffer, &attrs);
    }
    if (status == 0)
        status = download(stream, &result, buffer, sizeof(result));
    if (status == 0 &&
        (result.before != 0x13579bdfU || result.after != 0x2468ace0U))
        status = -1;

    std::printf("[host] interrupt probe before=0x%08x after=0x%08x\n",
                result.before, result.after);
    if (status == 0)
        std::printf("[host] PASS: tensor interrupt resumes the faulting Proc\n");
    else
        std::printf("[host] FAIL: tensor interrupt probe status=%d\n", status);

    if (buffer)
        preserve_error(npu_buffer_destroy(buffer), &status);
    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status < 0 ? 1 : 0;
}

#endif
