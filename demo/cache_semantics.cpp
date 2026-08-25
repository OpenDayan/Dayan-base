#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>
#endif

#include <stdint.h>

struct alignas(64) CacheSemanticsState {
    volatile uint64_t value;
    uint64_t value_padding[7];
    volatile uint64_t ready;
    uint64_t ready_padding[7];
    volatile uint64_t far_value;
    uint64_t far_value_padding[7];
    volatile uint64_t release;
    uint64_t release_padding[7];
    volatile uint64_t observed;
    uint64_t observed_padding[7];
};

static_assert(sizeof(CacheSemanticsState) == 320, "cache test state ABI");

__kernel__ void cache_semantics(CacheSemanticsState *state, uint64_t operation)
{
    if (!state || operation > 3U)
        dayan_thread_exit(-22);

    volatile uint64_t *target = operation >= 2U ? &state->far_value :
                                                  &state->value;
    const uint64_t initial = *target;

    *target = 0x55aa33cc77ee1199ULL;
    if (operation == 0U)
        dayan_cache_flush((const void *)&state->value, 64);
    else if (operation == 1U)
        dayan_cache_invalidate((const void *)&state->value, 64);
    else if (operation == 2U)
        dayan_cache_flush((const void *)state, 224);
    else
        dayan_cache_invalidate((const void *)state, 224);

    state->ready = initial;
    dayan_cache_flush((const void *)&state->ready, 64);
    for (;;) {
        dayan_cache_invalidate((const void *)&state->release, 64);
        if (state->release)
            break;
    }

    target = (volatile uint64_t *)((uintptr_t)state + state->release);
    state->observed = *target;
    dayan_cache_flush((const void *)&state->observed, 64);
}

DAYAN_KERNEL_DESC(cache_semantics, 0, 0, 0, 0, 0, 0, 0);

#if !defined(__DAYAN_DEVICE__)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

extern "C" __host__ char *getenv(const char *name);

constexpr uint64_t kInitial = UINT64_C(0x1122334455667788);
constexpr uint64_t kDeviceDirty = UINT64_C(0x55aa33cc77ee1199);
constexpr uint64_t kUpdated = UINT64_C(0x8877665544332211);
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

__host__ static int upload(npu_stream_t *stream, npu_buffer_t *buffer,
                           uint64_t offset, const void *source, size_t size)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_h2d_async(stream, buffer, offset, source, size,
                                      &event);

    return status < 0 ? status : wait_and_destroy(event);
}

__host__ static int download(npu_stream_t *stream, void *destination,
                             npu_buffer_t *buffer, uint64_t offset,
                             size_t size)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_d2h_async(stream, destination, buffer, offset,
                                      size, &event);

    return status < 0 ? status : wait_and_destroy(event);
}

__host__ static int run_case(uint64_t operation, npu_context_t *context,
                             npu_stream_t *kernel_stream,
                             npu_stream_t *transfer_stream,
                             npu_buffer_t *buffer)
{
    CacheSemanticsState state = {};
    npu_kernel_launch_attrs_t attrs;
    npu_event_t *kernel_event = nullptr;
    uint64_t ready = 0;
    uint64_t written_back = 0;
    uint64_t release;
    const uint64_t target_offset = operation >= 2U ?
        offsetof(CacheSemanticsState, far_value) :
        offsetof(CacheSemanticsState, value);
    int status;

    state.value = kInitial;
    state.far_value = kInitial;
    release = target_offset;
    status = upload(transfer_stream, buffer, 0, &state, sizeof(state));
    if (status < 0)
        return status;

    npu_kernel_launch_attrs_init(&attrs, context, kernel_stream,
                                 operation == 0U ? "cache-flush" :
                                                   "cache-inval");
    attrs.timeout_ms = kTimeoutMs;
    attrs.flags = NPU_KERNEL_LAUNCH_ASYNC;
    attrs.event_out = &kernel_event;
    status = cache_semantics(buffer, operation, &attrs);
    if (status < 0)
        return status;

    while (!ready && status == 0)
        status = download(transfer_stream, &ready, buffer,
                          offsetof(CacheSemanticsState, ready),
                          sizeof(ready));
    if (status == 0 && ready != kInitial)
        status = -1;
    if (status == 0 && (operation == 0U || operation == 2U))
        status = download(transfer_stream, &written_back, buffer,
                          target_offset, sizeof(written_back));
    if (status == 0 && (operation == 0U || operation == 2U) &&
        written_back != kDeviceDirty) {
        std::printf("[host] FAIL: flush did not write back the dirty line: "
                    "0x%016llx\n", (unsigned long long)written_back);
        status = -1;
    }
    if (status == 0)
        status = upload(transfer_stream, buffer,
                        target_offset, &kUpdated, sizeof(kUpdated));
    if (status == 0)
        status = upload(transfer_stream, buffer,
                        offsetof(CacheSemanticsState, release), &release,
                        sizeof(release));
    if (status == 0)
        status = wait_and_destroy(kernel_event);
    else if (kernel_event)
        (void)npu_event_destroy(kernel_event);
    kernel_event = nullptr;
    if (status == 0)
        status = download(transfer_stream, &state, buffer, 0, sizeof(state));
    if (status < 0)
        return status;

    std::printf("[host] %s initial=0x%016llx updated=0x%016llx "
                "observed=0x%016llx\n",
                operation == 0U ? "flush" :
                operation == 1U ? "invalidate" :
                operation == 2U ? "flush-224" : "invalidate-224",
                (unsigned long long)kInitial,
                (unsigned long long)kUpdated,
                (unsigned long long)state.observed);
    if (state.observed != kUpdated) {
        std::printf("[host] FAIL: %s retained the old cache line\n",
                    operation == 0U ? "flush" :
                    operation == 1U ? "invalidate" :
                    operation == 2U ? "flush-224" : "invalidate-224");
        return -1;
    }
    std::printf("[host] PASS: %s invalidated the cache line\n",
                operation == 0U ? "flush" :
                operation == 1U ? "invalidate" :
                operation == 2U ? "flush-224" : "invalidate-224");
    return 0;
}

__host__ int main(int argc, char **argv)
{
    npu_runtime_options_t options;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *kernel_stream = nullptr;
    npu_stream_t *transfer_stream = nullptr;
    npu_buffer_t *buffer = nullptr;
    const char *socket_path;
    uint64_t operation = 0;
    const char *stage = "runtime open";
    int status = 0;

    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("usage: %s [flush|invalidate|flush-224|invalidate-224]\n",
                    argv[0]);
        return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "invalidate") == 0)
        operation = 1;
    else if (argc == 2 && std::strcmp(argv[1], "flush-224") == 0)
        operation = 2;
    else if (argc == 2 && std::strcmp(argv[1], "invalidate-224") == 0)
        operation = 3;
    else if (argc == 2 && std::strcmp(argv[1], "flush") != 0)
        return 2;
    else if (argc != 1 && argc != 2)
        return 2;

    npu_runtime_options_init(&options);
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        options.server_socket = socket_path;
    status = npu_runtime_open(&runtime, &options);
    if (status == 0) {
        stage = "context create";
        status = npu_context_create(runtime, nullptr, &context);
    }
    if (status == 0) {
        stage = "kernel stream create";
        status = npu_stream_create(context, nullptr, &kernel_stream);
    }
    if (status == 0) {
        stage = "transfer stream create";
        status = npu_stream_create(context, nullptr, &transfer_stream);
    }
    if (status == 0) {
        npu_buffer_desc_t desc;

        stage = "buffer create";
        npu_buffer_desc_init(&desc, sizeof(CacheSemanticsState));
        desc.alignment = 64;
        status = npu_buffer_create(context, &desc, &buffer);
    }
    if (status == 0) {
        stage = operation == 0U ? "flush test" :
                operation == 1U ? "invalidate test" :
                operation == 2U ? "flush-224 test" :
                                  "invalidate-224 test";
        status = run_case(operation, context, kernel_stream, transfer_stream,
                          buffer);
    }

    if (status < 0)
        std::printf("[host] cache semantics failed at %s: %d\n", stage,
                    status);

    if (buffer)
        preserve_error(npu_buffer_destroy(buffer), &status);
    if (transfer_stream)
        preserve_error(npu_stream_destroy(transfer_stream), &status);
    if (kernel_stream)
        preserve_error(npu_stream_destroy(kernel_stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status < 0 ? 1 : 0;
}

#endif
