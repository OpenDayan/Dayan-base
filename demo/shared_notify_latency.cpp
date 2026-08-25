#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum : uint64_t {
    kClusterCores = 4,
    kSenderLane = 0,
    kReceiverLane = 1,
    kDefaultSamples = 64,
    kMaximumSamples = 512,
    kWarmupSamples = 16,
    kResultMagic = 0x5348524e4f544946ULL,
};

struct alignas(64) SharedNotifyResults {
    uint64_t magic;
    uint64_t sample_count;
    uint64_t warmup_count;
    uint64_t sender_lane;
    uint64_t receiver_lane;
    uint64_t reserved[3];
    alignas(64) uint64_t send_cycles[kMaximumSamples];
    alignas(64) uint64_t produce_return_cycles[kMaximumSamples];
    alignas(64) uint64_t receive_cycles[kMaximumSamples];
    alignas(64) uint64_t ack_cycles[kMaximumSamples];
};

#if defined(__DAYAN_DEVICE__)

namespace shared_notify_latency_detail {

struct alignas(64) BarrierLine {
    dayan_cluster_barrier_t barrier;
    uint8_t padding[64 - sizeof(dayan_cluster_barrier_t)];
};

struct alignas(64) ResourceLine {
    dayan_shared_resource_t resource;
    uint8_t padding[64 - sizeof(dayan_shared_resource_t)];
};

struct alignas(64) StatusLine {
    int32_t status;
    uint8_t padding[64 - sizeof(int32_t)];
};

struct alignas(64) ClusterShared {
    BarrierLine initialize;
    BarrierLine finish;
    ResourceLine armed;
    ResourceLine notify;
    ResourceLine ack;
    StatusLine initialize_status;
};

static_assert(sizeof(ClusterShared) <= 512,
              "shared-notify state exceeds its Atomic Cache allocation");
static_assert(sizeof(SharedNotifyResults::send_cycles) ==
                  kMaximumSamples * sizeof(uint64_t),
              "cycle sample layout changed unexpectedly");

static inline uint64_t read_mcycle()
{
    uint64_t cycle;

    __asm__ volatile("csrr %0, mcycle" : "=r"(cycle) : : "memory");
    return cycle;
}

static inline void delay_cycles(uint64_t cycles)
{
    const uint64_t begin = read_mcycle();

    while (read_mcycle() - begin < cycles)
        __asm__ volatile("" ::: "memory");
}

static inline void require_status(int status)
{
    if (status < 0)
        dayan_thread_exit(status);
}

static int initialize_shared(ClusterShared *cluster, uint32_t lane)
{
    if (lane == kSenderLane) {
        int status = dayan_shared_resource_init(&cluster->armed.resource);

        if (status == 0)
            status = dayan_shared_resource_init(&cluster->notify.resource);
        if (status == 0)
            status = dayan_shared_resource_init(&cluster->ack.resource);
        cluster->initialize_status.status = status;
    }

    const int barrier_status = dayan_cluster_barrier(
        kClusterCores, &cluster->initialize.barrier);
    if (barrier_status < 0)
        return barrier_status;
    return cluster->initialize_status.status;
}

static void run_receiver(ClusterShared *cluster, uint64_t sample_count,
                         SharedNotifyResults *results)
{
    const uint64_t total = kWarmupSamples + sample_count;

    for (uint64_t iteration = 0; iteration < total; ++iteration) {
        require_status(dayan_shared_resource_wait_free(
            &cluster->armed.resource));
        require_status(dayan_shared_resource_alloc(
            &cluster->armed.resource, 0, 1));

        require_status(dayan_shared_resource_wait_ready(
            &cluster->notify.resource));
        const uint64_t receive_cycle = read_mcycle();
        require_status(dayan_shared_resource_consume(
            &cluster->notify.resource));

        require_status(dayan_shared_resource_wait_free(
            &cluster->ack.resource));
        require_status(dayan_shared_resource_alloc(
            &cluster->ack.resource, 1, 1));
        require_status(dayan_shared_resource_produce(
            &cluster->ack.resource));

        if (iteration >= kWarmupSamples)
            results->receive_cycles[iteration - kWarmupSamples] =
                receive_cycle;
    }

    results->magic = kResultMagic;
    results->sample_count = sample_count;
    results->warmup_count = kWarmupSamples;
    results->sender_lane = kSenderLane;
    results->receiver_lane = kReceiverLane;
}

static void run_sender(ClusterShared *cluster, uint64_t sample_count,
                       SharedNotifyResults *results)
{
    const uint64_t total = kWarmupSamples + sample_count;

    for (uint64_t iteration = 0; iteration < total; ++iteration) {
        require_status(dayan_shared_resource_wait_ready(
            &cluster->armed.resource));
        require_status(dayan_shared_resource_consume(
            &cluster->armed.resource));

        require_status(dayan_shared_resource_wait_free(
            &cluster->notify.resource));
        require_status(dayan_shared_resource_alloc(
            &cluster->notify.resource, 1, 1));

        const uint64_t send_cycle = read_mcycle();
        require_status(dayan_shared_resource_produce(
            &cluster->notify.resource));
        const uint64_t produce_return_cycle = read_mcycle();

        require_status(dayan_shared_resource_wait_ready(
            &cluster->ack.resource));
        const uint64_t ack_cycle = read_mcycle();
        require_status(dayan_shared_resource_consume(
            &cluster->ack.resource));

        if (iteration >= kWarmupSamples) {
            const uint64_t sample = iteration - kWarmupSamples;

            results->send_cycles[sample] = send_cycle;
            results->produce_return_cycles[sample] = produce_return_cycle;
            results->ack_cycles[sample] = ack_cycle;
        }
    }
}

} // namespace shared_notify_latency_detail

#endif /* __DAYAN_DEVICE__ */

__kernel__ void shared_notify_latency(uint64_t sample_count,
                                      uint64_t start_delay_cycles,
                                      SharedNotifyResults *results)
{
    using namespace shared_notify_latency_detail;

    const uint32_t lane = dayan_task_id(DAYAN_TASK_X);
    if (!sample_count || sample_count > kMaximumSamples || !results ||
        !shared || lane >= kClusterCores || dayan_task_mode() != 4U ||
        dayan_task_dim(DAYAN_TASK_X) != kClusterCores ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U)
        dayan_thread_exit(-22);

    ClusterShared *cluster = (ClusterShared *)shared;
    require_status(initialize_shared(cluster, lane));

    if (lane == kSenderLane) {
        if (start_delay_cycles) {
            printf("[device] shared-notify waveform marker: delay=%llu "
                   "cycles\n",
                   (unsigned long long)start_delay_cycles);
            delay_cycles(start_delay_cycles);
        }
        run_sender(cluster, sample_count, results);
    } else if (lane == kReceiverLane) {
        run_receiver(cluster, sample_count, results);
    }

    require_status(dayan_cluster_barrier(kClusterCores,
                                         &cluster->finish.barrier));
}

DAYAN_KERNEL_DESC(shared_notify_latency, 0, 512, 0, 0, 0, 0, 0);

#if !defined(__DAYAN_DEVICE__)

#include <algorithm>
#include <limits>
#include <stdlib.h>
#include <vector>

extern "C" __host__ char *getenv(const char *name);

struct LatencyStats {
    uint64_t minimum;
    uint64_t p50;
    uint64_t p90;
    uint64_t p99;
    uint64_t maximum;
    double average;
};

__host__ static void usage(const char *program)
{
    printf("usage: %s [SAMPLES [START_DELAY_CYCLES]]\n", program);
    printf("default samples: %llu, maximum: %llu, warmup: %llu\n",
           (unsigned long long)kDefaultSamples,
           (unsigned long long)kMaximumSamples,
           (unsigned long long)kWarmupSamples);
}

__host__ static int parse_sample_count(const char *text,
                                       uint64_t *sample_count)
{
    char *end = nullptr;

    if (!text || !*text || !sample_count)
        return -1;
    const unsigned long long value = strtoull(text, &end, 0);
    if (!end || *end || !value || value > kMaximumSamples)
        return -1;
    *sample_count = (uint64_t)value;
    return 0;
}

__host__ static int parse_delay_cycles(const char *text, uint64_t *cycles)
{
    char *end = nullptr;

    if (!text || !*text || !cycles)
        return -1;
    const unsigned long long value = strtoull(text, &end, 0);
    if (!end || *end || !value)
        return -1;
    *cycles = (uint64_t)value;
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
    const int status = npu_memcpy_h2d_async(stream, buffer, 0, source, size,
                                            &event);

    return status < 0 ? status : wait_and_destroy(event, timeout_ms);
}

__host__ static int download(npu_stream_t *stream, void *destination,
                             npu_buffer_t *buffer, size_t size,
                             uint32_t timeout_ms)
{
    npu_event_t *event = nullptr;
    const int status = npu_memcpy_d2h_async(stream, destination, buffer, 0,
                                            size, &event);

    return status < 0 ? status : wait_and_destroy(event, timeout_ms);
}

__host__ static uint64_t percentile(const std::vector<uint64_t> &sorted,
                                    uint64_t percent)
{
    const size_t rank =
        (size_t)((sorted.size() * percent + 99U) / 100U);

    return sorted[rank ? rank - 1U : 0U];
}

__host__ static LatencyStats summarize(std::vector<uint64_t> values)
{
    std::sort(values.begin(), values.end());
    long double sum = 0.0;
    for (uint64_t value : values)
        sum += value;

    LatencyStats stats = {};
    stats.minimum = values.front();
    stats.p50 = percentile(values, 50);
    stats.p90 = percentile(values, 90);
    stats.p99 = percentile(values, 99);
    stats.maximum = values.back();
    stats.average = (double)(sum / values.size());
    return stats;
}

__host__ static void print_stats(const char *name, const LatencyStats &stats)
{
    printf("[host] %-22s min=%llu avg=%.2f p50=%llu p90=%llu "
           "p99=%llu max=%llu cycles\n",
           name, (unsigned long long)stats.minimum, stats.average,
           (unsigned long long)stats.p50,
           (unsigned long long)stats.p90,
           (unsigned long long)stats.p99,
           (unsigned long long)stats.maximum);
}

__host__ static int validate_and_print(const SharedNotifyResults &results,
                                       uint64_t sample_count)
{
    if (results.magic != kResultMagic ||
        results.sample_count != sample_count ||
        results.warmup_count != kWarmupSamples ||
        results.sender_lane != kSenderLane ||
        results.receiver_lane != kReceiverLane) {
        printf("[host] FAIL: shared-notify result header is invalid\n");
        return -1;
    }

    std::vector<uint64_t> produce;
    std::vector<uint64_t> one_way;
    std::vector<uint64_t> round_trip;
    produce.reserve(sample_count);
    one_way.reserve(sample_count);
    round_trip.reserve(sample_count);

    uint64_t clock_order_violations = 0;
    for (uint64_t sample = 0; sample < sample_count; ++sample) {
        const uint64_t send = results.send_cycles[sample];
        const uint64_t produced = results.produce_return_cycles[sample];
        const uint64_t received = results.receive_cycles[sample];
        const uint64_t ack = results.ack_cycles[sample];

        if (!send || !produced || !received || !ack || produced < send ||
            ack < send) {
            printf("[host] FAIL: incomplete timestamp at sample %llu\n",
                   (unsigned long long)sample);
            return -1;
        }

        produce.push_back(produced - send);
        round_trip.push_back(ack - send);
        if (received < send || received > ack) {
            ++clock_order_violations;
        } else {
            one_way.push_back(received - send);
        }
    }

    printf("[host] core %llu -> core %llu, samples=%llu, warmup=%llu\n",
           (unsigned long long)kSenderLane,
           (unsigned long long)kReceiverLane,
           (unsigned long long)sample_count,
           (unsigned long long)kWarmupSamples);
    print_stats("producer API", summarize(produce));
    if (!clock_order_violations) {
        print_stats("notify -> B resumes", summarize(one_way));
    } else {
        printf("[host] WARN: %llu cross-core timestamps fall outside the "
               "sender RTT; one-way latency is not trustworthy\n",
               (unsigned long long)clock_order_violations);
    }
    const LatencyStats rtt = summarize(round_trip);
    print_stats("notify + immediate ACK", rtt);
    printf("[host] RTT/2 estimate         avg=%.2f p50=%.2f cycles\n",
           rtt.average / 2.0, (double)rtt.p50 / 2.0);
    printf("[host] PASS: shared-resource notification benchmark completed\n");
    return 0;
}

__host__ int main(int argc, char **argv)
{
    uint64_t sample_count = kDefaultSamples;
    uint64_t start_delay_cycles = 0;

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (argc > 3 ||
        (argc >= 2 && parse_sample_count(argv[1], &sample_count) < 0) ||
        (argc == 3 &&
         parse_delay_cycles(argv[2], &start_delay_cycles) < 0)) {
        usage(argv[0]);
        return 2;
    }

    constexpr uint32_t timeout_ms = 60U * 60U * 1000U;
    SharedNotifyResults results = {};
    npu_runtime_options_t runtime_options;
    npu_kernel_launch_attrs_t launch_attrs;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *results_buffer = nullptr;
    const char *socket_path;
    int status = 0;

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
                                     "notify-lat");
        launch_attrs.config.task_dim = NPUTaskDim3{4, 1, 1};
        launch_attrs.config.task_mode = NPUTaskMode::kFourCore;
        launch_attrs.timeout_ms = timeout_ms;
        printf("[host] measuring dayan_shared_resource core 0 -> core 1 "
               "notification latency\n");
        fflush(stdout);
        status = shared_notify_latency(sample_count, start_delay_cycles,
                                       results_buffer,
                                       &launch_attrs);
    }
    if (status == 0)
        status = download(stream, &results, results_buffer, sizeof(results),
                          timeout_ms);
    if (status == 0)
        status = validate_and_print(results, sample_count);
    if (status < 0)
        printf("[host] FAIL: shared notification latency status=%d\n",
               status);

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

#endif /* !__DAYAN_DEVICE__ */
