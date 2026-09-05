#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_intrinsics.h>
#include <dayan_sys.h>
#endif

#include <stddef.h>
#include <stdint.h>

struct alignas(64) ContextThreadResult {
    uint32_t observed_tcsr[2];
    uint32_t reused_tcsr;
    int32_t exhausted_status;
    int32_t join_status;
    uint32_t uram_match_mask;
    uint32_t uram_size;
    uint32_t completion_mask;
    uint64_t root_uram;
    uint64_t reserved[2];
};

static_assert(sizeof(ContextThreadResult) == 64,
              "Context Thread result ABI");

#if defined(__DAYAN_DEVICE__)

namespace context_thread_probe_detail {

enum : uint32_t {
    kFirstMarker = 17,
    kSecondMarker = 29,
    kReuseMarker = 41,
};

struct ProbeState {
    ContextThreadResult *result;
    void *root_uram;
    size_t root_uram_size;
    volatile uint32_t ready_mask;
};

struct ProbeThreadArg {
    ProbeState *state;
    uint32_t index;
    uint32_t marker;
};

static void record_uram(ProbeState *state, uint32_t bit)
{
    if (dayan_uram_base() == state->root_uram &&
        dayan_uram_size() == state->root_uram_size)
        state->result->uram_match_mask |= bit;
}

static void isolated_tcsr_thread(void *argument)
{
    ProbeThreadArg *arg = (ProbeThreadArg *)argument;
    ProbeState *state = arg->state;
    long observed = 0;

    record_uram(state, 1U << arg->index);
    __tx_lsu_tcsr_write<LSU_DIM0>(arg->marker);
    state->ready_mask |= 1U << arg->index;
    while (state->ready_mask != 3U)
        dayan_yield();
    dayan_yield();
    __tx_lsu_tcsr_read<LSU_DIM0>(observed);
    state->result->observed_tcsr[arg->index] = (uint32_t)observed;
}

static void uram_only_thread(void *argument)
{
    ProbeState *state = (ProbeState *)argument;

    record_uram(state, 1U << 2);
}

static void reused_tcsr_thread(void *argument)
{
    ProbeState *state = (ProbeState *)argument;
    long observed = 0;

    record_uram(state, 1U << 3);
    __tx_lsu_tcsr_write<LSU_DIM0>(kReuseMarker);
    dayan_yield();
    __tx_lsu_tcsr_read<LSU_DIM0>(observed);
    state->result->reused_tcsr = (uint32_t)observed;
}

static int test_deferred_resource_transitions(ContextThreadResult *result)
{
    const dayan_resource_t local_resource = DAYAN_LOCAL_RESOURCE(0);
    const dayan_resource_t shared_resource = DAYAN_SHARED_RESOURCE(0);
    int status;

    status = dayan_resource_wait_free(local_resource);
    if (status == 0)
        status = dayan_resource_produce_lsu(local_resource, 1U, 0);
    if (status == 0)
        status = dayan_resource_wait_ready(local_resource);
    if (status == 0)
        status = dayan_resource_consume_mat(
            DAYAN_RESOURCE_MASK(local_resource));
    if (status < 0)
        return status;
    dayan_wait_mat();
    result->completion_mask |= 1U;

    status = dayan_resource_wait_free(shared_resource);
    if (status == 0)
        status = dayan_resource_produce_vec(shared_resource, 1U, 0);
    if (status == 0)
        status = dayan_resource_wait_ready(shared_resource);
    if (status == 0)
        status = dayan_resource_consume_macro(
            DAYAN_RESOURCE_MASK(shared_resource));
    if (status < 0)
        return status;
    dayan_wait_macro();
    result->completion_mask |= 2U;
    return 0;
}

} // namespace context_thread_probe_detail

#endif

__kernel__ void context_thread_probe(ContextThreadResult *result)
{
    using namespace context_thread_probe_detail;

    if (!result || dayan_task_mode() != 1U ||
        dayan_task_dim(DAYAN_TASK_X) != 1U ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U ||
        !dayan_uram_base() || dayan_uram_size() != 32U * 1024U)
        dayan_thread_exit(-22);

    ContextThreadResult output = {};
    output.root_uram = (uint64_t)(uintptr_t)dayan_uram_base();
    output.uram_size = (uint32_t)dayan_uram_size();
    ProbeState state = {
        &output,
        dayan_uram_base(),
        dayan_uram_size(),
        0,
    };
    ProbeThreadArg args[2] = {
        {&state, 0U, kFirstMarker},
        {&state, 1U, kSecondMarker},
    };
    dayan_thread_resources_t resources;
    dayan_thread_resources_init(&resources);
    resources.lsu_tcsr_count = 1;

    int status = dayan_create_thread("tcsr-0", isolated_tcsr_thread,
                                     &args[0], &resources);
    if (status >= 0)
        status = dayan_create_thread("tcsr-1", isolated_tcsr_thread,
                                     &args[1], &resources);
    if (status < 0)
        dayan_thread_exit(status);

    output.exhausted_status = dayan_create_thread(
        "tcsr-full", reused_tcsr_thread, &state, &resources);
    status = dayan_create_thread("uram-only", uram_only_thread, &state,
                                 nullptr);
    if (status < 0)
        dayan_thread_exit(status);
    output.join_status = dayan_thread_join_all();
    if (output.join_status < 0)
        dayan_thread_exit(output.join_status);

    const int reused = dayan_create_thread("tcsr-reuse", reused_tcsr_thread,
                                           &state, &resources);
    if (reused < 0)
        dayan_thread_exit(reused);
    output.join_status = dayan_thread_join((uint32_t)reused, nullptr);
    if (output.join_status < 0)
        dayan_thread_exit(output.join_status);
    status = test_deferred_resource_transitions(&output);
    if (status < 0)
        dayan_thread_exit(status);

    *result = output;
    dayan_cache_flush(result, sizeof(*result));
}

DAYAN_KERNEL_NAMED_RESOURCE_DESC(
    context_thread_probe, 3, 64, 32 * 1024, 2, 0, 0, 0, 1, 1);

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
    ContextThreadResult result = {};
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

        npu_kernel_launch_attrs_init(&attrs, context, stream, "context-probe");
        attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms = kTimeoutMs;
        status = context_thread_probe(buffer, &attrs);
    }
    if (status == 0)
        status = download(stream, &result, buffer, sizeof(result));
    if (status == 0 &&
        (result.observed_tcsr[0] != 17U ||
         result.observed_tcsr[1] != 29U ||
         result.reused_tcsr != 41U || result.exhausted_status != -12 ||
         result.join_status != 0 || result.uram_match_mask != 0xfU ||
         result.uram_size != 32U * 1024U ||
         result.completion_mask != 3U || !result.root_uram))
        status = -1;

    std::printf("[host] tcsr=(%u,%u) reused=%u exhausted=%d "
                "uram_mask=0x%x uram_size=%u completion_mask=0x%x\n",
                result.observed_tcsr[0], result.observed_tcsr[1],
                result.reused_tcsr, result.exhausted_status,
                result.uram_match_mask, result.uram_size,
                result.completion_mask);
    if (status == 0)
        std::printf("[host] PASS: Thread Contexts isolate logical TCSR 0 and inherit URAM\n");
    else
        std::printf("[host] FAIL: Context Thread probe status=%d\n", status);

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
