#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <getopt.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "npu_runtime.h"

#define DYPERF_DEFAULT_DURATION_MS 1000U
#define DYPERF_READ_INTERVAL_MS    100U
#define DYPERF_DRAIN_TIMEOUT_MS    5000U
#define DYPERF_TRACE_PID_BASE      UINT64_C(0x10000)
#define DYPERF_PENDING_BYTES       (NPU_PROFILE_SLOT_DEFINE_SIZE - 1U)
#define DYPERF_PCU_TID             (UINT64_C(1) << 40)
#define DYPERF_DEFAULT_CLOCK_HZ    1000000000.0
#define DYPERF_DEFAULT_PCU_INTERVAL_MS 100U
#define DYPERF_DEFAULT_LSU_BEAT_BYTES 64U

enum {
    OPTION_PCU = 256,
    OPTION_PCU_INTERVAL,
    OPTION_CLOCK_HZ,
    OPTION_LSU_BEAT_BYTES,
};

static const char *const pcu_metric_names[] = {
    "URAM read mean (%)", "URAM read variance (%^2)",
    "URAM read stddev (%)", "URAM write mean (%)",
    "URAM write variance (%^2)", "URAM write stddev (%)",
    "MAT utilization (%)", "LSU read (GB/s)", "LSU write (GB/s)",
    "VEC group 0 IPC", "VEC group 1 IPC", "VEC total IPC", "Scalar IPC",
};
#define DYPERF_PCU_METRIC_COUNT \
    (sizeof(pcu_metric_names) / sizeof(pcu_metric_names[0]))

struct trace_thread_order;

struct trace_slot {
    uint32_t generation;
    uint8_t state;
    unsigned char defined;
    unsigned char state_valid;
    char name[NPU_PROC_NAME_SIZE];
};

struct trace_target {
    struct trace_target *next;
    uint64_t epoch;
    uint64_t last_counter;
    uint32_t cluster;
    uint32_t hart;
    size_t pending_size;
    int have_cycle;
    int have_execution_state;
    int scheduler_running;
    int have_pcu;
    int have_pcu_interval;
    uint64_t pcu_first_cycle;
    npu_pcu_core_sample_t pcu_previous;
    size_t thread_count;
    size_t thread_capacity;
    struct trace_thread_order *threads;
    struct trace_slot slots[NPU_PROFILE_SLOT_COUNT];
    unsigned char pending[DYPERF_PENDING_BYTES];
};

struct trace_thread_order {
    char name[NPU_PROC_NAME_SIZE];
    uint8_t slot;
    uint32_t generation;
};

struct trace_writer {
    FILE *file;
    struct trace_target *targets;
    int first;
    int failed;
    double clock_hz;
    uint32_t lsu_beat_bytes;
    uint32_t pcu_interval_ms;
    int pcu_enabled;
};

static volatile sig_atomic_t interrupted;
static volatile sig_atomic_t interrupted_signal;

static void handle_signal(int signal_number)
{
    interrupted = 1;
    interrupted_signal = signal_number;
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

static int parse_real(const char *text, int allow_seconds_suffix,
                      double *value_out)
{
    char *end;
    double value;

    if (!text || !*text || text[0] == '-')
        return -EINVAL;
    errno = 0;
    value = strtod(text, &end);
    if (errno || end == text || !isfinite(value) || value < 0 ||
        (*end && !(allow_seconds_suffix && !strcmp(end, "s"))))
        return -EINVAL;
    *value_out = value;
    return 0;
}

static void usage(FILE *file, const char *program)
{
    fprintf(file,
            "Usage: %s [OPTIONS] [-- COMMAND [ARG...]]\n"
            "Capture Dayan proc state trace as Chrome Trace JSON.\n\n"
            "  -s, --socket PATH       Server socket (env DAYAN_SERVER_SOCKET, else %s)\n"
            "  -o, --output FILE       output JSON (default: dayan-trace.json)\n"
            "  -d, --duration-ms MS    capture duration; 0 waits for Ctrl-C or COMMAND\n"
            "  -f, --follow            continuously capture until Ctrl-C or COMMAND exits\n"
            "  -b, --buffer-size SIZE  Device buffer bytes; K/M/G accepted\n"
            "      --pcu              include PCU metrics (default interval: 0.1 s)\n"
            "      --pcu-interval SEC enable PCU; delay after each round, 0 = continuous\n"
            "      --clock-hz HZ      Device cycle frequency (default: 1e9)\n"
            "      --lsu-beat-bytes N LSU bus bytes per beat (default: 64)\n"
            "  -- COMMAND [ARG...]     profile while COMMAND uses the same Server\n"
            "  -h, --help              show this help\n",
            program, NPU_DEFAULT_SERVER_SOCKET);
}

static int command_status_code(int wait_status)
{
    if (WIFEXITED(wait_status))
        return WEXITSTATUS(wait_status);
    if (WIFSIGNALED(wait_status))
        return 128 + WTERMSIG(wait_status);
    return 1;
}

static int command_poll(pid_t command_pid, int *done_out,
                        int *status_out)
{
    int wait_status;
    pid_t result;

    *done_out = 0;
    do {
        result = waitpid(command_pid, &wait_status, WNOHANG);
    } while (result < 0 && errno == EINTR && !interrupted);
    if (result < 0)
        return -errno;
    if (!result)
        return 0;
    *done_out = 1;
    *status_out = command_status_code(wait_status);
    return 0;
}

static int command_wait(pid_t command_pid, int forward_signal,
                        int *status_out)
{
    int wait_status;
    pid_t result;

    if (forward_signal > 0)
        (void)kill(command_pid, forward_signal);
    do {
        result = waitpid(command_pid, &wait_status, 0);
    } while (result < 0 && errno == EINTR);
    if (result < 0)
        return -errno;
    *status_out = command_status_code(wait_status);
    return 0;
}

static int command_start(char *const command[], const char *socket_path,
                         pid_t *pid_out)
{
    pid_t pid = fork();

    if (pid < 0)
        return -errno;
    if (!pid) {
        (void)signal(SIGINT, SIG_DFL);
        (void)signal(SIGTERM, SIG_DFL);
        if (setenv("DAYAN_SERVER_SOCKET", socket_path, 1) < 0) {
            fprintf(stderr, "dyperf: cannot set DAYAN_SERVER_SOCKET: %s\n",
                    strerror(errno));
            _exit(127);
        }
        execvp(command[0], command);
        fprintf(stderr, "dyperf: cannot execute %s: %s\n", command[0],
                strerror(errno));
        _exit(127);
    }
    *pid_out = pid;
    return 0;
}

static int trace_item_begin(struct trace_writer *writer)
{
    if (!writer->first && fputc(',', writer->file) == EOF)
        return -EIO;
    writer->first = 0;
    return 0;
}

static int trace_json_string(FILE *file, const char *text, size_t capacity)
{
    size_t index;

    if (fputc('"', file) == EOF)
        return -EIO;
    for (index = 0; index < capacity && text[index]; ++index) {
        unsigned char value = (unsigned char)text[index];

        if (value == '"' || value == '\\') {
            if (fputc('\\', file) == EOF || fputc(value, file) == EOF)
                return -EIO;
        } else if (value == '\b' || value == '\f' || value == '\n' ||
                   value == '\r' || value == '\t') {
            const char escaped = value == '\b' ? 'b' :
                                 value == '\f' ? 'f' :
                                 value == '\n' ? 'n' :
                                 value == '\r' ? 'r' : 't';

            if (fputc('\\', file) == EOF || fputc(escaped, file) == EOF)
                return -EIO;
        } else if (value < 0x20U) {
            if (fprintf(file, "\\u%04x", value) < 0)
                return -EIO;
        } else if (fputc(value, file) == EOF) {
            return -EIO;
        }
    }
    return fputc('"', file) == EOF ? -EIO : 0;
}

static uint64_t trace_pid(uint32_t cluster, uint32_t hart)
{
    /* Keep process IDs outside the uint16_t Device event-ID namespace. */
    return DYPERF_TRACE_PID_BASE + ((uint64_t)cluster << 32) + hart;
}

static uint64_t trace_process_sort_index(uint32_t cluster, uint32_t hart)
{
    return ((uint64_t)cluster << 32) + hart;
}

static uint64_t trace_thread_id(const struct trace_target *target,
                                uint8_t slot, uint32_t generation)
{
    uint64_t local_id;

    /* Perfetto identifies tid == pid as the process main thread. */
    if (slot == NPU_PROFILE_SLOT_SCHEDULER)
        return trace_pid(target->cluster, target->hart);
    local_id = ((uint64_t)generation - 1U) * UINT8_MAX + slot;
    return local_id;
}

static int trace_proc_metadata(struct trace_writer *writer,
                               struct trace_target *target,
                               uint8_t slot, uint32_t generation,
                               const char *name)
{
    size_t name_length = strnlen(name, NPU_PROC_NAME_SIZE);

    if (!name_length || name_length == NPU_PROC_NAME_SIZE)
        return -EPROTO;
    if (trace_item_begin(writer) < 0 ||
        fprintf(writer->file,
                "{\"name\":\"thread_name\",\"ph\":\"M\","
                "\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"args\":{"
                "\"name\":",
                trace_pid(target->cluster, target->hart),
                trace_thread_id(target, slot, generation)) < 0 ||
        trace_json_string(writer->file, name, name_length) < 0 ||
        fputs("}}", writer->file) == EOF)
        return -EIO;
    return 0;
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
    if (trace_item_begin(writer) < 0 ||
        fprintf(writer->file,
                "{\"name\":\"process_sort_index\",\"ph\":\"M\","
                "\"pid\":%" PRIu64 ",\"tid\":0,\"args\":{"
                "\"sort_index\":%" PRIu64 "}}",
                trace_pid(cluster, hart),
                trace_process_sort_index(cluster, hart)) < 0 ||
        trace_proc_metadata(writer, target, NPU_PROFILE_SLOT_SCHEDULER,
                            0, "scheduler") < 0) {
        writer->failed = 1;
        return 0;
    }
    return target;
}

static uint64_t extend_counter(struct trace_target *target, uint64_t counter)
{
    if (!target->have_cycle && target->have_pcu) {
        uint64_t low = target->pcu_first_cycle & NPU_PROFILE_COUNTER_MASK;
        uint64_t period = UINT64_C(1) << 48;

        target->epoch = target->pcu_first_cycle & ~NPU_PROFILE_COUNTER_MASK;
        if (counter > low && counter - low > period / 2 &&
            target->epoch >= period)
            target->epoch -= period;
        else if (low > counter && low - counter > period / 2)
            target->epoch += period;
    }
    if (target->have_cycle && counter < target->last_counter)
        target->epoch += UINT64_C(1) << 48;
    target->last_counter = counter;
    target->have_cycle = 1;
    return target->epoch + counter;
}

static long double trace_time_us(const struct trace_writer *writer,
                                 uint64_t cycle)
{
    return (long double)cycle * 1000000.0L / writer->clock_hz;
}

static void pcu_metrics(const npu_pcu_core_sample_t *previous,
                        const npu_pcu_core_sample_t *sample,
                        double clock_hz, uint32_t beat_bytes,
                        double values[DYPERF_PCU_METRIC_COUNT])
{
    double cycles = (double)(sample->cycle - previous->cycle);
    uint64_t delta[NPU_PCU_COUNTER_COUNT];
    unsigned int i;

    for (i = 0; i < NPU_PCU_COUNTER_COUNT; ++i)
        delta[i] = sample->counters[i] - previous->counters[i];
    for (i = 0; i < 2; ++i) {
        double mean = 0;
        double variance = 0;
        unsigned int bank;
        unsigned int base = i ? NPU_PCU_URAM_WRITE(0) : NPU_PCU_URAM_READ(0);

        for (bank = 0; bank < NPU_PCU_URAM_BANK_COUNT; ++bank)
            mean += 100.0 * (double)delta[base + bank] / cycles;
        mean /= NPU_PCU_URAM_BANK_COUNT;
        for (bank = 0; bank < NPU_PCU_URAM_BANK_COUNT; ++bank) {
            double deviation = 100.0 * (double)delta[base + bank] / cycles - mean;

            variance += deviation * deviation;
        }
        variance /= NPU_PCU_URAM_BANK_COUNT;
        values[i * 3] = mean;
        values[i * 3 + 1] = variance;
        values[i * 3 + 2] = sqrt(variance);
    }
    values[6] = 100.0 * (double)delta[NPU_PCU_MAT_EXEC] / cycles;
    values[7] = (double)delta[NPU_PCU_LSU_READ] / cycles *
                beat_bytes * (clock_hz / 1e9);
    values[8] = (double)delta[NPU_PCU_LSU_WRITE] / cycles *
                beat_bytes * (clock_hz / 1e9);
    values[9] = (double)delta[NPU_PCU_VEC_ISSUE(0)] / cycles;
    values[10] = (double)delta[NPU_PCU_VEC_ISSUE(1)] / cycles;
    values[11] = values[9] + values[10];
    values[12] = (double)delta[NPU_PCU_SCALAR_COMMIT] / cycles;
}

static int trace_pcu_counters(struct trace_writer *writer,
                              const struct trace_target *target,
                              uint64_t cycle, const double *values)
{
    size_t i;

    for (i = 0; i < DYPERF_PCU_METRIC_COUNT; ++i) {
        if (trace_item_begin(writer) < 0 ||
            fprintf(writer->file,
                    "{\"name\":\"%s\",\"cat\":\"pcu\",\"ph\":\"C\","
                    "\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ","
                    "\"ts\":%.9Lf,\"args\":{\"value\":%.12g}}",
                    pcu_metric_names[i], trace_pid(target->cluster, target->hart),
                    DYPERF_PCU_TID, trace_time_us(writer, cycle),
                    values ? values[i] : 0.0) < 0)
            return -EIO;
    }
    return 0;
}

static int trace_pcu_sample(struct trace_writer *writer, uint32_t cluster,
                            uint32_t hart, const npu_pcu_core_sample_t *sample)
{
    struct trace_target *target = trace_target_get(writer, cluster, hart);
    uint64_t pid = trace_pid(cluster, hart);
    unsigned int i;

    if (!target)
        return writer->failed ? -EIO : -ENOMEM;
    if (target->have_pcu && sample->cycle < target->pcu_previous.cycle)
        return -ERANGE;
    if (target->have_pcu && sample->cycle == target->pcu_previous.cycle)
        return memcmp(sample, &target->pcu_previous, sizeof(*sample)) ?
               -EPROTO : 0;
    if (!sample->cycle)
        return 0;
    if (!target->have_pcu) {
        if (trace_item_begin(writer) < 0 ||
            fprintf(writer->file,
                    "{\"name\":\"thread_name\",\"ph\":\"M\","
                    "\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ","
                    "\"args\":{\"name\":\"PCU intervals\"}}",
                    pid, DYPERF_PCU_TID) < 0)
            return -EIO;
        target->pcu_first_cycle = sample->cycle;
    } else {
        const npu_pcu_core_sample_t *previous = &target->pcu_previous;
        double values[DYPERF_PCU_METRIC_COUNT];

        pcu_metrics(previous, sample, writer->clock_hz,
                     writer->lsu_beat_bytes, values);
        if (trace_pcu_counters(writer, target, previous->cycle, values) < 0 ||
            trace_item_begin(writer) < 0 ||
            fprintf(writer->file,
                    "{\"name\":\"PCU interval\",\"cat\":\"pcu\",\"ph\":\"X\","
                    "\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ","
                    "\"ts\":%.9Lf,\"dur\":%.9Lf,\"args\":{"
                    "\"start_cycle\":%" PRIu64 ",\"end_cycle\":%" PRIu64 ","
                    "\"cycles\":%" PRIu64,
                    pid, DYPERF_PCU_TID, trace_time_us(writer, previous->cycle),
                    trace_time_us(writer, sample->cycle - previous->cycle),
                    previous->cycle, sample->cycle,
                    sample->cycle - previous->cycle) < 0)
            return -EIO;
        for (i = 0; i < DYPERF_PCU_METRIC_COUNT; ++i) {
            if (fprintf(writer->file, ",\"%s\":%.12g", pcu_metric_names[i],
                        values[i]) < 0)
                return -EIO;
        }
        if (fputs("}}", writer->file) == EOF)
            return -EIO;
        target->have_pcu_interval = 1;
    }
    if (trace_item_begin(writer) < 0 ||
        fprintf(writer->file,
                "{\"name\":\"PCU snapshot\",\"cat\":\"pcu\",\"ph\":\"i\","
                "\"s\":\"t\",\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ","
                "\"ts\":%.9Lf,\"args\":{\"cycle\":%" PRIu64 ",\"counters\":[",
                pid, DYPERF_PCU_TID, trace_time_us(writer, sample->cycle),
                sample->cycle) < 0)
        return -EIO;
    for (i = 0; i < NPU_PCU_COUNTER_COUNT; ++i) {
        if (fprintf(writer->file, "%s%" PRIu64, i ? "," : "",
                    sample->counters[i]) < 0)
            return -EIO;
    }
    if (fputs("]}}", writer->file) == EOF)
        return -EIO;
    target->pcu_previous = *sample;
    target->have_pcu = 1;
    return 0;
}

static int read_pcu_round(npu_profile_t *profile, struct trace_writer *writer)
{
    npu_pcu_batch_t batch;
    int status;

    do {
        uint32_t core;

        status = npu_profile_pcu_read(profile, &batch);
        if (status < 0)
            break;
        for (core = 0; core < batch.core_count; ++core) {
            status = trace_pcu_sample(writer, batch.cluster, core,
                                       &batch.cores[core]);
            if (status < 0)
                break;
        }
    } while (status == 0 && !(batch.flags & NPU_PCU_ROUND_END));
    if (status == -ENODATA)
        fprintf(stderr, "dyperf: PCU cycle registers are zero; snapshot "
                "timestamps require RTL support (reserved per-core slot 1022)\n");
    else if (status < 0)
        fprintf(stderr, "dyperf: PCU snapshot failed: %s\n", strerror(-status));
    return status;
}

static const char *trace_state_name(uint8_t state)
{
    switch (state) {
    case NPU_PROFILE_PROC_RUNNING:
        return "RUNNING";
    case NPU_PROFILE_PROC_READY:
        return "READY";
    case NPU_PROFILE_PROC_POLLING:
        return "POLLING";
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
    case NPU_PROFILE_PROC_WAIT_LSU:
        return "WAIT_LSU";
    case NPU_PROFILE_PROC_WAIT_MAT:
        return "WAIT_MAT";
    case NPU_PROFILE_PROC_WAIT_VEC:
        return "WAIT_VEC";
    case NPU_PROFILE_PROC_WAIT_MACRO:
        return "WAIT_MACRO";
    case NPU_PROFILE_PROC_YIELD:
        return "YIELD";
    case NPU_PROFILE_PROC_WAIT_JOIN:
        return "WAIT_JOIN";
    case NPU_PROFILE_PROC_WAIT_RESOURCE:
        return "WAIT_RESOURCE";
    default:
        return 0;
    }
}

static int trace_state_event(struct trace_writer *writer,
                             const struct trace_target *target,
                             uint8_t slot, uint32_t generation,
                             uint8_t state,
                             const char *phase, uint64_t cycle,
                             uint64_t raw_counter)
{
    const char *name = trace_state_name(state);
    char timestamp[48];
    int length;

    if (!name)
        return -EPROTO;
    length = snprintf(timestamp, sizeof(timestamp), "%.9Lf",
                      trace_time_us(writer, cycle));
    if (length < 0 || (size_t)length >= sizeof(timestamp))
        return -EOVERFLOW;
    if (trace_item_begin(writer) < 0 ||
        fprintf(writer->file,
                "{\"name\":\"%s\",\"cat\":\"proc_state\","
                "\"ph\":\"%s\",\"pid\":%" PRIu64 ","
                "\"tid\":%" PRIu64 ",\"ts\":%s,\"args\":{"
                "\"counter\":%" PRIu64 ",\"state\":%u}}",
                name, phase, trace_pid(target->cluster, target->hart),
                trace_thread_id(target, slot, generation), timestamp,
                raw_counter, state) < 0)
        return -EIO;
    return 0;
}

static int trace_scheduler_update(struct trace_writer *writer,
                                  struct trace_target *target,
                                  uint8_t old_state, int old_state_valid,
                                  uint8_t new_state,
                                  uint64_t cycle, uint64_t raw_counter)
{
    int leaves_cpu = old_state_valid &&
                     old_state == NPU_PROFILE_PROC_RUNNING &&
                     new_state != NPU_PROFILE_PROC_RUNNING;
    int enters_cpu = new_state == NPU_PROFILE_PROC_RUNNING;

    if (!target->have_execution_state) {
        target->have_execution_state = 1;
        if (!enters_cpu) {
            if (trace_state_event(writer, target,
                                  NPU_PROFILE_SLOT_SCHEDULER, 0,
                                  NPU_PROFILE_PROC_RUNNING, "B", cycle,
                                  raw_counter) < 0)
                return -EIO;
            target->scheduler_running = 1;
        }
    } else if (leaves_cpu && !target->scheduler_running) {
        if (trace_state_event(writer, target,
                              NPU_PROFILE_SLOT_SCHEDULER, 0,
                              NPU_PROFILE_PROC_RUNNING, "B", cycle,
                              raw_counter) < 0)
            return -EIO;
        target->scheduler_running = 1;
    }
    if (enters_cpu && target->scheduler_running) {
        if (trace_state_event(writer, target,
                              NPU_PROFILE_SLOT_SCHEDULER, 0,
                              NPU_PROFILE_PROC_RUNNING, "E", cycle,
                              raw_counter) < 0)
            return -EIO;
        target->scheduler_running = 0;
    }
    return 0;
}

static int trace_slot_define(struct trace_writer *writer,
                             struct trace_target *target,
                             uint8_t slot, uint64_t raw_counter,
                             const char *name)
{
    struct trace_slot *entry;
    struct trace_thread_order *thread;
    size_t name_length;
    uint64_t counter;

    if (!slot || slot >= NPU_PROFILE_SLOT_COUNT)
        return -EPROTO;
    name_length = strnlen(name, NPU_PROC_NAME_SIZE);
    if (!name_length || name_length == NPU_PROC_NAME_SIZE)
        return -EPROTO;
    entry = &target->slots[slot];
    counter = extend_counter(target, raw_counter);
    if (entry->state_valid) {
        if (trace_state_event(writer, target, slot, entry->generation,
                              entry->state, "E", counter,
                              raw_counter) < 0)
            return -EIO;
        if (entry->state == NPU_PROFILE_PROC_RUNNING &&
            !target->scheduler_running) {
            if (trace_state_event(writer, target,
                                  NPU_PROFILE_SLOT_SCHEDULER, 0,
                                  NPU_PROFILE_PROC_RUNNING, "B", counter,
                                  raw_counter) < 0)
                return -EIO;
            target->scheduler_running = 1;
        }
    }
    ++entry->generation;
    if (!entry->generation)
        return -EOVERFLOW;
    entry->defined = 1;
    entry->state_valid = 0;
    memset(entry->name, 0, sizeof(entry->name));
    memcpy(entry->name, name, name_length);
    if (target->thread_count == target->thread_capacity) {
        size_t capacity = target->thread_capacity ?
            target->thread_capacity * 2U : NPU_PROFILE_SLOT_COUNT;
        void *threads;

        if (capacity < target->thread_capacity ||
            capacity > SIZE_MAX / sizeof(*target->threads))
            return -EOVERFLOW;
        threads = realloc(target->threads,
                          capacity * sizeof(*target->threads));
        if (!threads)
            return -ENOMEM;
        target->threads = threads;
        target->thread_capacity = capacity;
    }
    thread = &target->threads[target->thread_count++];
    memset(thread, 0, sizeof(*thread));
    memcpy(thread->name, name, name_length);
    thread->slot = slot;
    thread->generation = entry->generation;
    return trace_proc_metadata(writer, target, slot, entry->generation,
                               entry->name);
}

static int trace_state_record(struct trace_writer *writer,
                              struct trace_target *target,
                              uint64_t raw)
{
    uint64_t raw_counter = npu_profile_record_counter(raw);
    uint8_t slot = npu_profile_record_slot(raw);
    uint8_t state = npu_profile_record_state(raw);
    struct trace_slot *entry;
    uint64_t cycle;

    if (!slot || slot >= NPU_PROFILE_SLOT_COUNT || !trace_state_name(state))
        return -EPROTO;
    entry = &target->slots[slot];
    if (!entry->defined)
        return -EPROTO;
    cycle = extend_counter(target, raw_counter);
    if (entry->state_valid &&
        trace_state_event(writer, target, slot, entry->generation,
                          entry->state, "E", cycle, raw_counter) < 0)
        return -EIO;
    if (trace_scheduler_update(writer, target, entry->state,
                               entry->state_valid, state, cycle,
                               raw_counter) < 0)
        return -EIO;
    if (trace_state_event(writer, target, slot, entry->generation,
                          state, "B", cycle, raw_counter) < 0)
        return -EIO;
    entry->state = state;
    entry->state_valid = 1;
    return 0;
}

static int trace_decode(struct trace_writer *writer,
                        struct trace_target *target,
                        const unsigned char *data, size_t size,
                        size_t *consumed_out)
{
    size_t offset = 0;

    while (size - offset >= NPU_PROFILE_RECORD_SIZE) {
        uint64_t raw;
        uint8_t state;
        size_t record_size;
        int status;

        memcpy(&raw, data + offset, sizeof(raw));
        state = npu_profile_record_state(raw);
        record_size = npu_profile_record_size(raw);
        if (!record_size)
            return -EPROTO;
        if (size - offset < record_size)
            break;
        if (state == NPU_PROFILE_RECORD_SLOT_DEFINE) {
            status = trace_slot_define(
                writer, target, npu_profile_record_slot(raw),
                npu_profile_record_counter(raw),
                (const char *)data + offset + NPU_PROFILE_RECORD_SIZE);
        } else {
            status = trace_state_record(writer, target, raw);
        }
        if (status < 0)
            return status;
        offset += record_size;
    }
    *consumed_out = offset;
    return 0;
}

static int trace_write_batch(struct trace_writer *writer,
                             const npu_profile_batch_info_t *batch,
                             const void *data)
{
    const unsigned char *bytes = data;
    struct trace_target *target;
    size_t consumed;
    size_t combined_size;
    unsigned char combined[NPU_PROFILE_MAX_BATCH_BYTES +
                           DYPERF_PENDING_BYTES];
    int status;

    if (!batch->byte_count)
        return 0;
    target = trace_target_get(writer, batch->cluster, batch->hart);
    if (!target)
        return writer->failed ? -EIO : -ENOMEM;
    if (batch->byte_count > NPU_PROFILE_MAX_BATCH_BYTES ||
        target->pending_size > sizeof(target->pending))
        return -EPROTO;
    combined_size = target->pending_size + batch->byte_count;
    memcpy(combined, target->pending, target->pending_size);
    memcpy(combined + target->pending_size, bytes, batch->byte_count);
    status = trace_decode(writer, target, combined, combined_size, &consumed);
    if (status < 0)
        return status;
    target->pending_size = combined_size - consumed;
    if (target->pending_size > sizeof(target->pending))
        return -EPROTO;
    memcpy(target->pending, combined + consumed, target->pending_size);
    return 0;
}

static int trace_close_states(struct trace_writer *writer,
                              const struct trace_target *target)
{
    uint64_t cycle = target->epoch + target->last_counter;
    uint32_t slot;

    if (target->pending_size)
        return -EPROTO;
    if (!target->have_cycle)
        return 0;
    for (slot = 1; slot < NPU_PROFILE_SLOT_COUNT; ++slot) {
        const struct trace_slot *entry = &target->slots[slot];

        if (entry->state_valid &&
            trace_state_event(writer, target, (uint8_t)slot,
                              entry->generation, entry->state, "E", cycle,
                              target->last_counter) < 0)
            return -EIO;
    }
    if (target->scheduler_running &&
        trace_state_event(writer, target, NPU_PROFILE_SLOT_SCHEDULER, 0,
                          NPU_PROFILE_PROC_RUNNING, "E", cycle,
                          target->last_counter) < 0)
        return -EIO;
    return 0;
}

static int trace_thread_order_compare(const void *left, const void *right)
{
    const struct trace_thread_order *a = left;
    const struct trace_thread_order *b = right;
    int order = strcmp(a->name, b->name);

    if (order)
        return order;
    if (a->slot != b->slot)
        return a->slot < b->slot ? -1 : 1;
    return a->generation < b->generation ? -1 :
           a->generation != b->generation;
}

static int trace_thread_sort_metadata(struct trace_writer *writer,
                                      const struct trace_target *target)
{
    struct trace_thread_order *threads;
    size_t count = target->thread_count + 1U;
    size_t index = 0;

    threads = calloc(count, sizeof(*threads));
    if (!threads)
        return -ENOMEM;
    memcpy(threads[index].name, "scheduler", sizeof("scheduler"));
    threads[index].slot = NPU_PROFILE_SLOT_SCHEDULER;
    threads[index].generation = 0;
    ++index;
    memcpy(threads + index, target->threads,
           target->thread_count * sizeof(*threads));
    qsort(threads, count, sizeof(*threads), trace_thread_order_compare);
    for (index = 0; index < count; ++index) {
        if (trace_item_begin(writer) < 0 ||
            fprintf(writer->file,
                    "{\"name\":\"thread_sort_index\",\"ph\":\"M\","
                    "\"pid\":%" PRIu64 ",\"tid\":%" PRIu64 ",\"args\":{"
                    "\"sort_index\":%zu}}",
                    trace_pid(target->cluster, target->hart),
                    trace_thread_id(target, threads[index].slot,
                                    threads[index].generation),
                    index) < 0) {
            free(threads);
            return -EIO;
        }
    }
    free(threads);
    return 0;
}

static int trace_writer_open(struct trace_writer *writer, const char *path)
{
    memset(writer, 0, sizeof(*writer));
    writer->file = fopen(path, "w");
    writer->first = 1;
    writer->clock_hz = DYPERF_DEFAULT_CLOCK_HZ;
    writer->lsu_beat_bytes = DYPERF_DEFAULT_LSU_BEAT_BYTES;
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
        for (closing = writer->targets; status == 0 && closing;
             closing = closing->next) {
            status = trace_thread_sort_metadata(writer, closing);
            if (status == 0 && closing->have_pcu_interval)
                status = trace_pcu_counters(writer, closing,
                                             closing->pcu_previous.cycle, 0);
        }
    }
    if (writer->file &&
        fprintf(writer->file,
                "],\"displayTimeUnit\":\"ns\",\"metadata\":{"
                "\"time_unit\":\"ns\",\"cycle_ns\":%.12g,"
                "\"clock_hz\":%.12g,\"pcu_enabled\":%s,"
                "\"pcu_interval_ms\":%u,\"lsu_beat_bytes\":%u}}\n",
                1e9 / writer->clock_hz, writer->clock_hz,
                writer->pcu_enabled ? "true" : "false",
                writer->pcu_interval_ms, writer->lsu_beat_bytes) < 0)
        status = -EIO;
    if (writer->file && fclose(writer->file) < 0)
        status = -errno;
    while (target) {
        struct trace_target *next = target->next;

        free(target->threads);
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
    unsigned char data[NPU_PROFILE_MAX_BATCH_BYTES];
    npu_profile_batch_info_t batch;
    int status;

    memset(&batch, 0, sizeof(batch));
    status = npu_profile_read(profile, data,
                              NPU_PROFILE_MAX_BATCH_BYTES,
                              timeout_ms, &batch);
    if (status < 0)
        return status;
    status = trace_write_batch(writer, &batch, data);
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
        { "follow", no_argument, 0, 'f' },
        { "buffer-size", required_argument, 0, 'b' },
        { "pcu", no_argument, 0, OPTION_PCU },
        { "pcu-interval", required_argument, 0, OPTION_PCU_INTERVAL },
        { "clock-hz", required_argument, 0, OPTION_CLOCK_HZ },
        { "lsu-beat-bytes", required_argument, 0, OPTION_LSU_BEAT_BYTES },
        { "help", no_argument, 0, 'h' },
        { 0, 0, 0, 0 },
    };
    const char *socket_path = getenv("DAYAN_SERVER_SOCKET");
    const char *output_path = "dayan-trace.json";
    uint64_t buffer_size = NPU_PROFILE_DEFAULT_BUFFER_SIZE;
    uint64_t duration_value = DYPERF_DEFAULT_DURATION_MS;
    uint32_t pcu_interval_ms = DYPERF_DEFAULT_PCU_INTERVAL_MS;
    uint32_t lsu_beat_bytes = DYPERF_DEFAULT_LSU_BEAT_BYTES;
    double clock_hz = DYPERF_DEFAULT_CLOCK_HZ;
    uint64_t next_pcu_ms = 0;
    int pcu_enabled = 0;
    int pcu_ready = 0;
    char **command = 0;
    npu_runtime_options_t runtime_options;
    npu_profile_options_t profile_options;
    struct trace_writer writer;
    npu_runtime_t *runtime = 0;
    npu_profile_t *profile = 0;
    uint64_t start_ms;
    pid_t command_pid = -1;
    int command_exit_status = 0;
    int command_started = 0;
    int duration_set = 0;
    int follow_set = 0;
    int writer_open = 0;
    int profile_started = 0;
    int status = 0;
    int option;

    if (!socket_path || !*socket_path)
        socket_path = NPU_DEFAULT_SERVER_SOCKET;
    while ((option = getopt_long(argc, argv, "+s:o:d:b:fh", long_options,
                                 0)) != -1) {
        uint64_t value;
        double real;

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
            duration_set = 1;
            break;
        case 'f':
            duration_value = 0;
            follow_set = 1;
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
        case OPTION_PCU:
            pcu_enabled = 1;
            break;
        case OPTION_PCU_INTERVAL:
            if (parse_real(optarg, 1, &real) < 0 ||
                real > (double)UINT32_MAX / 1000.0) {
                fprintf(stderr, "dyperf: invalid PCU interval: %s\n", optarg);
                return 2;
            }
            pcu_interval_ms = (uint32_t)ceil(real * 1000.0);
            pcu_enabled = 1;
            break;
        case OPTION_CLOCK_HZ:
            if (parse_real(optarg, 0, &clock_hz) < 0 ||
                clock_hz < 1.0 || clock_hz > 1e12) {
                fprintf(stderr, "dyperf: invalid clock frequency: %s\n", optarg);
                return 2;
            }
            break;
        case OPTION_LSU_BEAT_BYTES:
            if (parse_u64(optarg, &value) < 0 || !value ||
                value > 4096 || (value & (value - 1))) {
                fprintf(stderr, "dyperf: invalid LSU beat size: %s\n", optarg);
                return 2;
            }
            lsu_beat_bytes = (uint32_t)value;
            break;
        default:
            usage(stderr, argv[0]);
            return 2;
        }
    }
    if (!*socket_path || !*output_path) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (duration_set && follow_set) {
        fprintf(stderr,
                "dyperf: --duration-ms and --follow are mutually exclusive\n");
        return 2;
    }
    if (optind < argc) {
        command = &argv[optind];
        if (!duration_set && !follow_set)
            duration_value = 0;
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
    writer.clock_hz = clock_hz;
    writer.lsu_beat_bytes = lsu_beat_bytes;
    writer.pcu_interval_ms = pcu_interval_ms;
    writer.pcu_enabled = pcu_enabled;
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
    if (pcu_enabled) {
        status = read_pcu_round(profile, &writer);
        if (status < 0)
            goto stop;
        pcu_ready = 1;
        next_pcu_ms = monotonic_ms() + pcu_interval_ms;
    }
    if (command) {
        status = command_start(command, socket_path, &command_pid);
        if (status < 0) {
            fprintf(stderr, "dyperf: cannot start %s: %s\n", command[0],
                    strerror(-status));
            goto stop;
        }
        command_started = 1;
    }
    start_ms = monotonic_ms();
    while (!interrupted) {
        uint64_t elapsed = monotonic_ms() - start_ms;
        uint32_t timeout = DYPERF_READ_INTERVAL_MS;
        int end = 0;

        if (command_pid > 0) {
            int command_done;

            status = command_poll(command_pid, &command_done,
                                  &command_exit_status);
            if (status < 0)
                break;
            if (command_done) {
                command_pid = -1;
                break;
            }
        }
        if (duration_value && elapsed >= duration_value)
            break;
        if (pcu_enabled) {
            uint64_t now = monotonic_ms();

            if (now >= next_pcu_ms) {
                status = read_pcu_round(profile, &writer);
                if (status < 0)
                    break;
                now = monotonic_ms();
                next_pcu_ms = now + pcu_interval_ms;
            }
            if (next_pcu_ms - now < timeout)
                timeout = (uint32_t)(next_pcu_ms - now);
            elapsed = now - start_ms;
            if (duration_value && elapsed >= duration_value)
                break;
        }
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
stop:
    if (pcu_ready && status == 0)
        status = read_pcu_round(profile, &writer);
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
    if (command_pid > 0) {
        int wait_status = command_wait(
            command_pid, interrupted ? (int)interrupted_signal : 0,
            &command_exit_status);

        if (status == 0 && wait_status < 0)
            status = wait_status;
    }
    if (status < 0) {
        fprintf(stderr, "dyperf: capture failed: %s\n", strerror(-status));
        return 1;
    }
    fprintf(stderr, "dyperf: wrote %s\n", output_path);
    return command_started ? command_exit_status : 0;
}
