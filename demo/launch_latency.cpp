#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum : uint64_t {
    kLoopIterations = 450000,
    kWarmupLaunches = 5,
    kMaximumLaunches = 100000,
    kLanes = 4,
};

struct alignas(64) LaunchSample {
    uint64_t begin;
    uint64_t end;
    uint64_t completed;
    uint64_t launch_id;
};

__kernel__ void launch_latency_kernel(uint64_t launch_id, LaunchSample *samples)
{
    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    uint64_t completed = 0;
    const uint64_t begin = dayan_get_time();
    for (uint32_t iteration = 0; iteration < kLoopIterations; ++iteration) {
        // Keep the counter in a register and prevent loop removal.
        __asm__ volatile("" : "+r"(completed));
        ++completed;
    }
    const uint64_t end = dayan_get_time();
    LaunchSample *sample = &samples[launch_id * kLanes + lane];
    sample->begin = begin;
    sample->end = end;
    sample->completed = completed;
    sample->launch_id = launch_id;
    dayan_cache_flush(sample, sizeof(*sample));
}

DAYAN_KERNEL_TASK_DESC(launch_latency_kernel, 1, 0, 0, 0, 0, 0, 0, 0);

#if !defined(__DAYAN_DEVICE__)

#include <algorithm>
#include <chrono>
#include <errno.h>
#include <stdlib.h>
#include <vector>

struct HostSample {
    uint64_t begin;
    uint64_t end;
};

__host__ static uint64_t now_ns()
{
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

__host__ static void usage(const char *program)
{
    printf("usage: %s [LAUNCHES [CORES [CLOCK_HZ]]]\n", program);
    printf("defaults: 1000 launches, 4 cores, cycles only (CLOCK_HZ=0)\n");
    printf("LAUNCHES: 1..100000; CORES: 1 or 4; CLOCK_HZ: actual device frequency\n");
    printf("Each synchronous launch runs 450000 loop iterations per core.\n");
    printf("Five warmup launches are excluded from the statistics.\n");
}

__host__ static int parse_u64(const char *text, uint64_t *value)
{
    char *end;

    if (!text || text[0] < '0' || text[0] > '9')
        return -1;
    errno = 0;
    *value = strtoull(text, &end, 10);
    return errno || *end ? -1 : 0;
}

__host__ static void preserve_error(int candidate, int *status)
{
    if (*status == 0 && candidate < 0)
        *status = candidate;
}

__host__ static int wait_and_destroy(npu_event_t *event, uint32_t timeout_ms)
{
    npu_event_info_t info = {};
    int status = npu_event_wait(event, timeout_ms, &info);

    if (status == 0 && info.status != 0)
        status = -EIO;
    preserve_error(npu_event_destroy(event), &status);
    return status;
}

__host__ static void print_stats(const char *name, std::vector<uint64_t> values,
                                 double scale, const char *unit)
{
    if (values.empty())
        return;
    std::sort(values.begin(), values.end());
    long double sum = 0;
    for (uint64_t value : values)
        sum += value;
    printf("[host] %-19s n=%zu min=%.3f mean=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f %s\n",
           name, values.size(), values.front() * scale,
           (double)(sum / values.size()) * scale,
           values[(values.size() * 50 + 99) / 100 - 1] * scale,
           values[(values.size() * 95 + 99) / 100 - 1] * scale,
           values[(values.size() * 99 + 99) / 100 - 1] * scale,
           values.back() * scale, unit);
}

__host__ int main(int argc, char **argv)
{
    uint64_t arguments[3] = {1000, 4, 0};
    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (argc > 4) {
        usage(argv[0]);
        return 2;
    }
    for (int index = 1; index < argc; ++index) {
        if (parse_u64(argv[index], &arguments[index - 1]) < 0) {
            usage(argv[0]);
            return 2;
        }
    }
    const uint64_t launches = arguments[0];
    const uint64_t cores = arguments[1];
    const uint64_t clock_hz = arguments[2];
    if (!launches || launches > kMaximumLaunches || (cores != 1 && cores != 4)) {
        usage(argv[0]);
        return 2;
    }

    const uint64_t total = kWarmupLaunches + launches;
    constexpr uint32_t timeout_ms = 300000;
    std::vector<LaunchSample> samples(total * kLanes);
    std::vector<HostSample> host(total);
    npu_runtime_options_t options;
    npu_runtime_topology_t topology = {};
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *buffer = nullptr;
    npu_event_t *event = nullptr;
    npu_kernel_launch_attrs_t attrs;
    const size_t bytes = samples.size() * sizeof(samples[0]);
    int status;

    npu_runtime_options_init(&options);
    options.default_timeout_ms = timeout_ms;
    const char *socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        options.server_socket = socket_path;
    status = npu_runtime_open(&runtime, &options);
    if (status == 0)
        status = npu_runtime_get_topology(runtime, &topology);
    if (status == 0 && (__builtin_popcount(topology.enabled_cluster_mask) != 1 ||
        (cores == 4 && topology.enabled_core_mask != 0xf))) {
        fprintf(stderr, "launch_latency requires one enabled Cluster and four enabled Cores in four-core mode\n");
        status = -EINVAL;
    }
    const bool same_hart = cores == 4 || __builtin_popcount(topology.enabled_core_mask) == 1;
    if (status == 0)
        status = npu_context_create(runtime, nullptr, &context);
    if (status == 0)
        status = npu_stream_create(context, nullptr, &stream);
    if (status == 0) {
        npu_buffer_desc_t desc;
        npu_buffer_desc_init(&desc, bytes);
        desc.alignment = 64;
        status = npu_buffer_create(context, &desc, &buffer);
    }
    if (status == 0) {
        status = npu_memcpy_h2d_async(stream, buffer, 0, samples.data(), bytes, &event);
        if (status == 0)
            status = wait_and_destroy(event, timeout_ms);
    }
    if (status == 0) {
        npu_kernel_launch_attrs_init(&attrs, context, stream, "launch-loop");
        attrs.config = {NPUTaskDim3{(uint32_t)cores, 1, 1},
                        cores == 4 ? NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore};
        attrs.timeout_ms = timeout_ms;
        printf("[host] launches=%llu warmup=%llu cores=%llu iterations=%llu clock_hz=%llu mode=sync\n",
               (unsigned long long)launches, (unsigned long long)kWarmupLaunches,
               (unsigned long long)cores, (unsigned long long)kLoopIterations,
               (unsigned long long)clock_hz);
        fflush(stdout);
        for (uint64_t index = 0; index < total; ++index) {
            host[index].begin = now_ns();
            status = launch_latency_kernel(index, buffer, &attrs);
            host[index].end = now_ns();
            if (status < 0) {
                fprintf(stderr, "launch %llu failed: %d\n", (unsigned long long)index, status);
                break;
            }
        }
    }
    if (status == 0) {
        status = npu_memcpy_d2h_async(stream, samples.data(), buffer, 0, bytes, &event);
        if (status == 0)
            status = wait_and_destroy(event, timeout_ms);
    }
    for (uint64_t index = 0; status == 0 && index < total; ++index) {
        for (uint64_t lane = 0; lane < cores; ++lane) {
            const auto &sample = samples[index * kLanes + lane];
            if (sample.launch_id != index || sample.completed != kLoopIterations || sample.end <= sample.begin ||
                (same_hart && index && sample.begin <= samples[(index - 1) * kLanes + lane].end)) {
                fprintf(stderr, "invalid sample: launch=%llu lane=%llu completed=%llu begin=%llu end=%llu\n",
                        (unsigned long long)index, (unsigned long long)lane,
                        (unsigned long long)sample.completed, (unsigned long long)sample.begin,
                        (unsigned long long)sample.end);
                status = -EIO;
                break;
            }
        }
    }
    if (status == 0) {
        std::vector<uint64_t> calls, host_gaps, loops, device_gaps;
        for (uint64_t index = kWarmupLaunches; index < total; ++index) {
            calls.push_back(host[index].end - host[index].begin);
            if (index > kWarmupLaunches)
                host_gaps.push_back(host[index].begin - host[index - 1].end);
            for (uint64_t lane = 0; lane < cores; ++lane) {
                const auto &sample = samples[index * kLanes + lane];
                loops.push_back(sample.end - sample.begin);
                if (same_hart && index > kWarmupLaunches)
                    device_gaps.push_back(sample.begin - samples[(index - 1) * kLanes + lane].end);
            }
        }
        printf("[host] first_call=%.3f us (includes cold module loading)\n", (host[0].end - host[0].begin) / 1000.0);
        print_stats("host_call", calls, 0.001, "us");
        print_stats("host_between_calls", host_gaps, 0.001, "us");
        print_stats("device_loop", loops, 1, "cycles");
        print_stats("device_gap_per_lane", device_gaps, 1, "cycles");
        if (clock_hz) {
            print_stats("device_loop", loops, 1.0e6 / clock_hz, "us");
            print_stats("device_gap_per_lane", device_gaps, 1.0e6 / clock_hz, "us");
        }
        if (!same_hart)
            printf("[host] device gap omitted: single-core launches may move between Harts; use a one-bit Server core mask to measure it\n");
    }

    if (buffer)
        preserve_error(npu_buffer_destroy(buffer), &status);
    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    if (status == 0)
        printf("[host] PASS: consecutive launches completed and loop counts verified\n");
    else
        fprintf(stderr, "[host] FAIL: launch latency status=%d\n", status);
    return status < 0 ? 1 : 0;
}

#endif
