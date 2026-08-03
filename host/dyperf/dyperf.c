#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "npu_runtime.h"

#define DYPERF_DEFAULT_DURATION_MS 1000U
#define DYPERF_READ_INTERVAL_MS    100U
#define DYPERF_DRAIN_TIMEOUT_MS    5000U
#define DYPERF_EVENT_ID_COUNT      65536U

struct trace_target {
    struct trace_target *next;
    uint64_t epoch;
    uint32_t last_cycle;
    uint32_t cluster;
    uint32_t hart;
    int have_cycle;
    unsigned char *proc_seen;
    unsigned char *state_valid;
    uint16_t *proc_state;
};

struct trace_writer {
    FILE *file;
    struct trace_target *targets;
    int first;
    int failed;
};

static volatile sig_atomic_t interrupted;

static void handle_signal(int signal_number)
{
    (void)signal_number;
    interrupted = 1;
}

static uint64_t monotonic_ms(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) < 0)
        return 0;
    return (uint64_t)now.tv_sec * 1000U +
           (uint64_t)now.tv_nsec / 1000000U;
}

static int parse_u64(const char *text, uint64_t *value_out)
{
    char *end = 0;
    unsigned long long value;
    uint64_t multiplier = 1;

    if (!text || !*text || text[0] == '-')
        return -EINVAL;
    errno = 0;
    value = strtoull(text, &end, 0);
    if (errno || end == text)
        return -EINVAL;
    if (*end) {
        if (end[1])
            return -EINVAL;
        if (*end == 'k' || *end == 'K')
            multiplier = 1024U;
        else if (*end == 'm' || *end == 'M')
            multiplier = 1024U * 1024U;
        else if (*end == 'g' || *end == 'G')
            multiplier = 1024U * 1024U * 1024U;
        else
            return -EINVAL;
    }
    if ((uint64_t)value > UINT64_MAX / multiplier)
        return -ERANGE;
    *value_out = (uint64_t)value * multiplier;
    return 0;
}

static void usage(FILE *file, const char *program)
{
    fprintf(file,
            "Usage: %s [OPTIONS]\n"
            "Capture Dayan proc state trace as Chrome Trace JSON.\n\n"
            "  -s, --socket PATH       Dayan Server socket (default: %s)\n"
            "  -o, --output FILE       output JSON (default: dayan-trace.json)\n"
            "  -d, --duration-ms MS    capture duration; 0 waits for Ctrl-C\n"
            "  -b, --buffer-size SIZE  Device buffer bytes; K/M/G accepted\n"
            "  -h, --help              show this help\n",
            program, NPU_DEFAULT_SERVER_SOCKET);
}

static int trace_item_begin(struct trace_writer *writer)
{
    if (!writer->first && fputc(',', writer->file) == EOF)
        return -EIO;
    writer->first = 0;
    return 0;
}

static uint64_t trace_pid(uint32_t cluster, uint32_t hart)
{
    return UINT64_C(1) + ((uint64_t)cluster << 32) + hart;
}

static struct trace_target *trace_target_get(struct trace_writer *writer,
                                              uint32_t cluster,
                                              uint32_t hart)
{
    struct trace_target *target;

    for (target = writer->targets; target; target = target->next) {
        if (target->cluster == cluster && target->hart == hart)
            return target;
    }
    target = calloc(1, sizeof(*target));
    if (!target)
        return 0;
    target->proc_seen = calloc(DYPERF_EVENT_ID_COUNT, 1);
    target->state_valid = calloc(DYPERF_EVENT_ID_COUNT, 1);
    target->proc_state = calloc(DYPERF_EVENT_ID_COUNT,
                                sizeof(*target->proc_state));
    if (!target->proc_seen || !target->state_valid || !target->proc_state) {
        free(target->proc_state);
        free(target->state_valid);
        free(target->proc_seen);
        free(target);
        return 0;
    }
    target->cluster = cluster;
    target->hart = hart;
    target->next = writer->targets;
    writer->targets = target;
    if (trace_item_begin(writer) < 0 ||
        fprintf(writer->file,
                "{\"name\":\"process_name\",\"ph\":\"M\","
                "\"pid\":%" PRIu64 ",\"tid\":0,\"args\":{"
                "\"name\":\"cluster %u hart %u\"}}",
                trace_pid(cluster, hart), cluster, hart) < 0) {
        writer->failed = 1;
        return 0;
    }
    return target;
}

static int trace_proc_metadata(struct trace_writer *writer,
                               struct trace_target *target,
                               uint16_t event_id)
{
    if (target->proc_seen[event_id])
        return 0;
    target->proc_seen[event_id] = 1;
    if (trace_item_begin(writer) < 0 ||
        fprintf(writer->file,
                "{\"name\":\"thread_name\",\"ph\":\"M\","
                "\"pid\":%" PRIu64 ",\"tid\":%u,\"args\":{"
                "\"name\":\"proc %u\"}}",
                trace_pid(target->cluster, target->hart), event_id,
                event_id) < 0)
        return -EIO;
    return 0;
}

static uint64_t extend_cycle(struct trace_target *target, uint32_t cycle)
{
    if (target->have_cycle && cycle < target->last_cycle)
        target->epoch += UINT64_C(1) << 32;
    target->last_cycle = cycle;
    target->have_cycle = 1;
    return target->epoch + cycle;
}

static const char *trace_state_name(uint16_t state)
{
    switch (state) {
    case NPU_PROFILE_PROC_RUNNING:
        return "RUNNING";
    case NPU_PROFILE_PROC_READY:
        return "READY";
    case NPU_PROFILE_PROC_WAIT_POLL:
        return "WAIT_POLL";
    case NPU_PROFILE_PROC_WAIT_EVENT:
        return "WAIT_EVENT";
    case NPU_PROFILE_PROC_WAIT_START:
        return "WAIT_START";
    case NPU_PROFILE_PROC_WAIT_EXIT:
        return "WAIT_EXIT";
    case NPU_PROFILE_PROC_WAIT_PRODUCE:
        return "WAIT_PRODUCE";
    case NPU_PROFILE_PROC_KILL_WAIT_PRODUCE:
        return "KILL_WAIT_PRODUCE";
    default:
        return 0;
    }
}

static int trace_state_event(struct trace_writer *writer,
                             const struct trace_target *target,
                             uint16_t event_id, uint16_t state,
                             const char *phase, uint64_t cycle,
                             uint32_t raw_cycle)
{
    const char *name = trace_state_name(state);

    if (!name)
        return -EPROTO;
    if (trace_item_begin(writer) < 0 ||
        fprintf(writer->file,
                "{\"name\":\"%s\",\"cat\":\"proc_state\","
                "\"ph\":\"%s\",\"pid\":%" PRIu64 ","
                "\"tid\":%u,\"ts\":%" PRIu64 ",\"args\":{"
                "\"cycle\":%u,\"state\":%u}}",
                name, phase, trace_pid(target->cluster, target->hart),
                event_id, cycle, raw_cycle, state) < 0)
        return -EIO;
    return 0;
}

static int trace_write_batch(struct trace_writer *writer,
                             const npu_profile_batch_info_t *batch,
                             const npu_profile_event_t *events)
{
    struct trace_target *target;
    uint32_t index;

    if (!batch->count)
        return 0;
    target = trace_target_get(writer, batch->cluster, batch->hart);
    if (!target)
        return writer->failed ? -EIO : -ENOMEM;
    for (index = 0; index < batch->count; ++index) {
        const npu_profile_event_t *event = &events[index];
        uint64_t cycle;

        if (!trace_state_name(event->state))
            return -EPROTO;
        if (trace_proc_metadata(writer, target, event->event_id) < 0)
            return -EIO;
        cycle = extend_cycle(target, event->cycle);
        if (target->state_valid[event->event_id] &&
            trace_state_event(writer, target, event->event_id,
                              target->proc_state[event->event_id], "E",
                              cycle, event->cycle) < 0)
            return -EIO;
        if (trace_state_event(writer, target, event->event_id,
                              event->state, "B", cycle,
                              event->cycle) < 0)
            return -EIO;
        target->proc_state[event->event_id] = event->state;
        target->state_valid[event->event_id] = 1;
    }
    return 0;
}

static int trace_close_states(struct trace_writer *writer,
                              const struct trace_target *target)
{
    uint64_t cycle = target->epoch + target->last_cycle;
    uint32_t event_id;

    if (!target->have_cycle)
        return 0;
    for (event_id = 0; event_id < DYPERF_EVENT_ID_COUNT; ++event_id) {
        if (target->state_valid[event_id] &&
            trace_state_event(writer, target, (uint16_t)event_id,
                              target->proc_state[event_id], "E", cycle,
                              target->last_cycle) < 0)
            return -EIO;
    }
    return 0;
}

static int trace_writer_open(struct trace_writer *writer, const char *path)
{
    memset(writer, 0, sizeof(*writer));
    writer->file = fopen(path, "w");
    writer->first = 1;
    if (!writer->file)
        return -errno;
    if (fputs("{\"traceEvents\":[", writer->file) == EOF) {
        (void)fclose(writer->file);
        writer->file = 0;
        return -EIO;
    }
    return 0;
}

static int trace_writer_close(struct trace_writer *writer)
{
    struct trace_target *target = writer->targets;
    int status = writer->failed ? -EIO : 0;

    if (writer->file && status == 0) {
        struct trace_target *closing;

        for (closing = writer->targets; closing; closing = closing->next) {
            if (trace_close_states(writer, closing) < 0) {
                status = -EIO;
                break;
            }
        }
    }
    if (writer->file &&
        fputs("],\"metadata\":{\"time_unit\":\"device_cycle\"}}\n",
              writer->file) == EOF)
        status = -EIO;
    if (writer->file && fclose(writer->file) < 0)
        status = -errno;
    while (target) {
        struct trace_target *next = target->next;

        free(target->proc_state);
        free(target->state_valid);
        free(target->proc_seen);
        free(target);
        target = next;
    }
    writer->file = 0;
    writer->targets = 0;
    return status;
}

static int read_one(npu_profile_t *profile, struct trace_writer *writer,
                    uint32_t timeout_ms, int *end_out)
{
    npu_profile_event_t events[NPU_PROFILE_MAX_BATCH_EVENTS];
    npu_profile_batch_info_t batch;
    int status;

    memset(&batch, 0, sizeof(batch));
    status = npu_profile_read(profile, events,
                              NPU_PROFILE_MAX_BATCH_EVENTS,
                              timeout_ms, &batch);
    if (status < 0)
        return status;
    status = trace_write_batch(writer, &batch, events);
    if (status < 0)
        return status;
    *end_out = !!(batch.flags & NPU_PROFILE_BATCH_END);
    return 0;
}

int main(int argc, char **argv)
{
    static const struct option long_options[] = {
        { "socket", required_argument, 0, 's' },
        { "output", required_argument, 0, 'o' },
        { "duration-ms", required_argument, 0, 'd' },
        { "buffer-size", required_argument, 0, 'b' },
        { "help", no_argument, 0, 'h' },
        { 0, 0, 0, 0 },
    };
    const char *socket_path = NPU_DEFAULT_SERVER_SOCKET;
    const char *output_path = "dayan-trace.json";
    uint64_t buffer_size = NPU_PROFILE_DEFAULT_BUFFER_SIZE;
    uint64_t duration_value = DYPERF_DEFAULT_DURATION_MS;
    npu_runtime_options_t runtime_options;
    npu_profile_options_t profile_options;
    struct trace_writer writer;
    npu_runtime_t *runtime = 0;
    npu_profile_t *profile = 0;
    uint64_t start_ms;
    int writer_open = 0;
    int profile_started = 0;
    int status = 0;
    int option;

    while ((option = getopt_long(argc, argv, "s:o:d:b:h", long_options,
                                 0)) != -1) {
        uint64_t value;

        switch (option) {
        case 's':
            socket_path = optarg;
            break;
        case 'o':
            output_path = optarg;
            break;
        case 'd':
            status = parse_u64(optarg, &value);
            if (status < 0 || value > UINT32_MAX) {
                fprintf(stderr, "dyperf: invalid duration: %s\n", optarg);
                return 2;
            }
            duration_value = value;
            break;
        case 'b':
            status = parse_u64(optarg, &value);
            if (status < 0 || !value) {
                fprintf(stderr, "dyperf: invalid buffer size: %s\n", optarg);
                return 2;
            }
            buffer_size = value;
            break;
        case 'h':
            usage(stdout, argv[0]);
            return 0;
        default:
            usage(stderr, argv[0]);
            return 2;
        }
    }
    if (optind != argc || !*socket_path || !*output_path) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (signal(SIGINT, handle_signal) == SIG_ERR ||
        signal(SIGTERM, handle_signal) == SIG_ERR) {
        fprintf(stderr, "dyperf: cannot install signal handler\n");
        return 1;
    }
    status = trace_writer_open(&writer, output_path);
    if (status < 0) {
        fprintf(stderr, "dyperf: cannot open %s: %s\n", output_path,
                strerror(-status));
        return 1;
    }
    writer_open = 1;
    npu_runtime_options_init(&runtime_options);
    runtime_options.server_socket = socket_path;
    status = npu_runtime_open(&runtime, &runtime_options);
    if (status < 0) {
        fprintf(stderr, "dyperf: Runtime open failed: %s\n",
                strerror(-status));
        goto out;
    }
    npu_profile_options_init(&profile_options);
    profile_options.buffer_size = buffer_size;
    status = npu_profile_start(runtime, &profile_options, &profile);
    if (status < 0) {
        fprintf(stderr, "dyperf: profile start failed: %s\n",
                strerror(-status));
        goto out;
    }
    profile_started = 1;
    start_ms = monotonic_ms();
    while (!interrupted) {
        uint64_t elapsed = monotonic_ms() - start_ms;
        uint32_t timeout = DYPERF_READ_INTERVAL_MS;
        int end = 0;

        if (duration_value && elapsed >= duration_value)
            break;
        if (duration_value && duration_value - elapsed < timeout)
            timeout = (uint32_t)(duration_value - elapsed);
        status = read_one(profile, &writer, timeout, &end);
        if (status == -ETIMEDOUT || status == -EAGAIN) {
            status = 0;
            continue;
        }
        if (status < 0 || end)
            break;
    }
    {
        int stop_status = npu_profile_stop(profile);

        if (stop_status < 0 && status == 0)
            status = stop_status;
        if (stop_status < 0)
            fprintf(stderr, "dyperf: profile stop failed: %s\n",
                    strerror(-stop_status));
        else
            profile_started = 0;
    }
    if (status == 0) {
        int end = 0;

        while (!end) {
            status = read_one(profile, &writer, DYPERF_DRAIN_TIMEOUT_MS,
                              &end);
            if (status < 0)
                break;
        }
    }

out:
    if (profile) {
        int destroy_status;

        if (profile_started)
            (void)npu_profile_stop(profile);
        destroy_status = npu_profile_destroy(profile);
        if (status == 0 && destroy_status < 0)
            status = destroy_status;
    }
    if (runtime) {
        int close_status = npu_runtime_close(runtime);

        if (status == 0 && close_status < 0)
            status = close_status;
    }
    if (writer_open) {
        int close_status = trace_writer_close(&writer);

        if (status == 0 && close_status < 0)
            status = close_status;
    }
    if (status < 0) {
        fprintf(stderr, "dyperf: capture failed: %s\n", strerror(-status));
        return 1;
    }
    fprintf(stderr, "dyperf: wrote %s\n", output_path);
    return 0;
}
