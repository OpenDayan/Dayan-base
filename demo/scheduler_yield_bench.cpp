#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum : uint64_t {
    kDefaultIterations = 128,
    kWarmupIterations = 64,
    kTargetHandoffCycles = 1000,
    kResultMagic = 0x5343484544594c44ULL,
};

struct SchedulerYieldResults {
    uint64_t magic;
    uint64_t iterations;
    uint64_t worker_cycles[2];
    uint64_t round_trip_cycles;
    uint64_t handoff_cycles;
};

#if defined(__DAYAN_DEVICE__)

namespace scheduler_yield_bench_detail {

struct BenchState {
    volatile uint32_t ready;
    volatile uint32_t start;
    volatile uint32_t measured;
    uint64_t iterations;
    uint64_t start_delay_cycles;
    uint64_t cycles[2];
};

struct WorkerArgument {
    BenchState *state;
    uint32_t index;
};

static inline uint64_t read_cycle()
{
    uint64_t cycle;

    __asm__ volatile("csrr %0, mcycle" : "=r"(cycle) : : "memory");
    return cycle;
}

static inline void delay_cycles(uint64_t cycles)
{
    const uint64_t begin = read_cycle();

    while (read_cycle() - begin < cycles)
        __asm__ volatile("" ::: "memory");
}

static void yield_worker(void *opaque)
{
    WorkerArgument *argument = (WorkerArgument *)opaque;
    BenchState *state = argument->state;

    ++state->ready;
    while (state->ready != 2U)
        dayan_yield();

    if (argument->index == 0U) {
        if (state->start_delay_cycles) {
            printf("[device] scheduler-yield waveform marker: delay=%llu "
                   "cycles\n",
                   (unsigned long long)state->start_delay_cycles);
            delay_cycles(state->start_delay_cycles);
        }
        state->start = 1U;
    } else {
        while (!state->start)
            dayan_yield();
    }

    for (uint64_t i = 0; i < kWarmupIterations; ++i)
        dayan_yield();

    const uint64_t begin = read_cycle();
    for (uint64_t i = 0; i < state->iterations; ++i)
        dayan_yield();
    state->cycles[argument->index] = read_cycle() - begin;

    ++state->measured;
    while (state->measured != 2U)
        dayan_yield();
}

} // namespace scheduler_yield_bench_detail

#endif

__kernel__ void scheduler_yield_bench(uint64_t iterations,
                                      uint64_t start_delay_cycles,
                                      SchedulerYieldResults *results)
{
    using namespace scheduler_yield_bench_detail;

    if (!iterations || !results)
        dayan_proc_exit(-22);

    BenchState state = {};
    WorkerArgument arguments[2] = {
        {&state, 0},
        {&state, 1},
    };
    state.iterations = iterations;
    state.start_delay_cycles = start_delay_cycles;

    int status = dayan_create_proc("yield-0", yield_worker, &arguments[0]);
    if (status >= 0)
        status = dayan_create_proc("yield-1", yield_worker, &arguments[1]);
    if (status >= 0)
        status = dayan_join_all();
    if (status < 0)
        dayan_proc_exit(status);

    const uint64_t total_cycles = state.cycles[0] + state.cycles[1];
    const uint64_t round_trip_cycles = total_cycles / (2U * iterations);
    const uint64_t handoff_cycles = total_cycles / (4U * iterations);

    results->magic = kResultMagic;
    results->iterations = iterations;
    results->worker_cycles[0] = state.cycles[0];
    results->worker_cycles[1] = state.cycles[1];
    results->round_trip_cycles = round_trip_cycles;
    results->handoff_cycles = handoff_cycles;
}

DAYAN_KERNEL_DESC(scheduler_yield_bench, 3, 0, 0);

#if !defined(__DAYAN_DEVICE__)

#include <limits>
#include <stdlib.h>

extern "C" __host__ char *getenv(const char *name);

__host__ static void usage(const char *program)
{
    printf("usage: %s [ITERATIONS [START_DELAY_CYCLES]]\n", program);
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
    if (!end || *end || !value)
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
    uint64_t start_delay_cycles = 0;
    npu_runtime_options_t runtime_options;
    npu_kernel_launch_attrs_t launch_attrs;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *results_buffer = nullptr;
    SchedulerYieldResults results = {};
    const char *socket_path;
    constexpr uint32_t timeout_ms = 60U * 60U * 1000U;
    int status = 0;

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (argc > 3 ||
        (argc >= 2 && parse_positive(argv[1], &iterations) < 0) ||
        (argc == 3 &&
         parse_positive(argv[2], &start_delay_cycles) < 0)) {
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
                                     "yield-bench");
        launch_attrs.config.task_dim = NPUTaskDim3{1, 1, 1};
        launch_attrs.config.task_mode = NPUTaskMode::kSingleCore;
        launch_attrs.timeout_ms = timeout_ms;
        if (start_delay_cycles) {
            printf("[host] scheduler-yield waveform launch marker: "
                   "delay=%llu cycles\n",
                   (unsigned long long)start_delay_cycles);
            fflush(stdout);
        }
        status = scheduler_yield_bench(iterations, start_delay_cycles,
                                       results_buffer,
                                       &launch_attrs);
    }
    if (status == 0)
        status = download(stream, &results, results_buffer, sizeof(results),
                          timeout_ms);
    if (status == 0 &&
        (results.magic != kResultMagic || results.iterations != iterations ||
         !results.worker_cycles[0] || !results.worker_cycles[1]))
        status = -1;

    if (status == 0) {
        printf("[host] scheduler-yield-bench iterations=%llu "
               "worker0_cycles=%llu worker1_cycles=%llu "
               "round_trip_cycles=%llu handoff_cycles=%llu\n",
               (unsigned long long)results.iterations,
               (unsigned long long)results.worker_cycles[0],
               (unsigned long long)results.worker_cycles[1],
               (unsigned long long)results.round_trip_cycles,
               (unsigned long long)results.handoff_cycles);
        if (results.handoff_cycles < kTargetHandoffCycles) {
            printf("[host] PASS: scheduler yield handoff is below %llu cycles\n",
                   (unsigned long long)kTargetHandoffCycles);
        } else {
            printf("[host] FAIL: scheduler yield handoff must be below %llu "
                   "cycles\n",
                   (unsigned long long)kTargetHandoffCycles);
            status = -1;
        }
    } else {
        printf("[host] FAIL: scheduler yield benchmark status=%d\n", status);
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
