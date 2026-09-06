#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum : uint64_t {
    kClusterCores = 4,
    kDefaultIterations = 128,
    kWarmupIterations = 64,
    kResultMagic = 0x5343484544594c44ULL,
};

/* Each Hart owns a cache line; result writes cannot overwrite another Hart. */
struct alignas(64) SchedulerYieldResults {
    uint64_t magic;
    uint64_t iterations;
    uint64_t completed;
};

#if defined(__DAYAN_DEVICE__)

namespace scheduler_yield_bench_detail {

enum : dayan_resource_t {
    kNotify = DAYAN_SHARED_RESOURCE(0),
    kDone = DAYAN_SHARED_RESOURCE(1),
};

struct alignas(64) ClusterShared {
    dayan_cluster_barrier_t finish;
};

static inline void require_status(int status)
{
    if (status < 0)
        dayan_thread_exit(status);
}

} // namespace scheduler_yield_bench_detail

#endif

__kernel__ void scheduler_yield_bench(uint64_t iterations,
                                      SchedulerYieldResults *results)
{
    using namespace scheduler_yield_bench_detail;

    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    if (!iterations || iterations > UINT64_MAX - kWarmupIterations ||
        !results || !shared || dayan_task_mode() != kClusterCores ||
        dayan_task_dim(DAYAN_TASK_X) != kClusterCores ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U)
        dayan_thread_exit(-22);

    const uint64_t total = kWarmupIterations + iterations;
    uint64_t completed = 0;
    if (lane == 0U) {
        for (; completed < total; ++completed) {
            require_status(dayan_resource_wait_free(kNotify));
            require_status(dayan_resource_produce_lsu(kNotify, 1U, 0));
        }
    } else if (lane == 1U) {
        require_status(dayan_resource_wait_free(kDone));
        for (; completed < total; ++completed) {
            require_status(dayan_resource_wait_ready(kNotify));
            require_status(dayan_resource_consume_lsu(
                DAYAN_RESOURCE_MASK(kNotify)));
        }
    }

    /* Harts 2/3 sleep on hardware tokens throughout the notification loop. */
    if (lane == 1U)
        require_status(dayan_resource_produce_lsu(kDone, 3U, 0));
    else {
        require_status(dayan_resource_wait_ready(kDone));
        require_status(dayan_resource_consume_lsu(DAYAN_RESOURCE_MASK(kDone)));
    }
    dayan_wait_lsu();
    results[lane].magic = kResultMagic;
    results[lane].iterations = iterations;
    results[lane].completed = completed;
    require_status(dayan_cluster_barrier(
        kClusterCores, &((ClusterShared *)shared)->finish));
}

DAYAN_KERNEL_NAMED_RESOURCE_DESC(
    scheduler_yield_bench, 0, 64, 0, 0, 0, 0, 0, 0, 2);

#if !defined(__DAYAN_DEVICE__)

#include <limits>
#include <stdlib.h>

extern "C" __host__ char *getenv(const char *name);

__host__ static void usage(const char *program)
{
    printf("usage: %s [ITERATIONS]\n", program);
    printf("default iterations: %llu\n",
           (unsigned long long)kDefaultIterations);
}

__host__ static int parse_positive(const char *text, uint64_t *value_out)
{
    char *end = nullptr;
    unsigned long long value;

    if (!text || !*text || !value_out)
        return -1;
    value = strtoull(text, &end, 0);
    if (!end || *end || !value || value > UINT64_MAX - kWarmupIterations)
        return -1;
    *value_out = (uint64_t)value;
    return 0;
}

__host__ static void preserve_error(int candidate, int *status)
{
    if (*status == 0 && candidate < 0)
        *status = candidate;
}

__host__ static int wait_and_destroy(npu_event_t *event, uint32_t timeout_ms)
{
    npu_event_info_t info;
    int status;
    int destroy_status;

    if (!event)
        return -1;
    status = npu_event_wait(event, timeout_ms, &info);
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
                           const void *source, size_t size,
                           uint32_t timeout_ms)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_h2d_async(stream, buffer, 0, source, size,
                                      &event);

    return status < 0 ? status : wait_and_destroy(event, timeout_ms);
}

__host__ static int download(npu_stream_t *stream, void *destination,
                             npu_buffer_t *buffer, size_t size,
                             uint32_t timeout_ms)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_d2h_async(stream, destination, buffer, 0, size,
                                      &event);

    return status < 0 ? status : wait_and_destroy(event, timeout_ms);
}

__host__ int main(int argc, char **argv)
{
    uint64_t iterations = kDefaultIterations;
    npu_runtime_options_t runtime_options;
    npu_kernel_launch_attrs_t launch_attrs;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *results_buffer = nullptr;
    SchedulerYieldResults results[kClusterCores] = {};
    const char *socket_path;
    constexpr uint32_t timeout_ms = 60U * 60U * 1000U;
    int status = 0;

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (argc > 2 ||
        (argc == 2 && parse_positive(argv[1], &iterations) < 0)) {
        usage(argv[0]);
        return 2;
    }

    npu_runtime_options_init(&runtime_options);
    runtime_options.default_timeout_ms = timeout_ms;
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        runtime_options.server_socket = socket_path;

    status = npu_runtime_open(&runtime, &runtime_options);
    if (status == 0)
        status = npu_context_create(runtime, nullptr, &context);
    if (status == 0)
        status = npu_stream_create(context, nullptr, &stream);
    if (status == 0) {
        npu_buffer_desc_t desc;

        npu_buffer_desc_init(&desc, sizeof(results));
        status = npu_buffer_create(context, &desc, &results_buffer);
    }
    if (status == 0)
        status = upload(stream, results_buffer, &results, sizeof(results),
                        timeout_ms);
    if (status == 0) {
        npu_kernel_launch_attrs_init(&launch_attrs, context, stream,
                                     "notify-bench");
        launch_attrs.config.task_dim = NPUTaskDim3{kClusterCores, 1, 1};
        launch_attrs.config.task_mode = NPUTaskMode::kFourCore;
        launch_attrs.timeout_ms = timeout_ms;
        status = scheduler_yield_bench(iterations, results_buffer,
                                       &launch_attrs);
    }
    if (status == 0)
        status = download(stream, &results, results_buffer, sizeof(results),
                          timeout_ms);
    for (uint32_t lane = 0; status == 0 && lane < kClusterCores; ++lane) {
        const uint64_t expected = lane < 2U ? kWarmupIterations + iterations : 0;

        if (results[lane].magic != kResultMagic ||
            results[lane].iterations != iterations ||
            results[lane].completed != expected)
            status = -1;
    }

    if (status == 0) {
        printf("[host] shared-notify-bench iterations=%llu warmup=%llu "
               "sender_completed=%llu receiver_completed=%llu\n",
               (unsigned long long)iterations,
               (unsigned long long)kWarmupIterations,
               (unsigned long long)results[0].completed,
               (unsigned long long)results[1].completed);
        printf("[host] PASS: shared resource notification benchmark completed\n");
    } else {
        printf("[host] FAIL: shared resource notification status=%d\n", status);
    }

    if (results_buffer)
        preserve_error(npu_buffer_destroy(results_buffer), &status);
    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status < 0 ? 1 : 0;
}

#endif
