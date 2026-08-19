#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "npu_package.h"
#include "npu_runtime.h"
#include "server_internal.h"

#define LOCK_TEST_CHILD_PROCS 2U
#define LOCK_TEST_STARTED_MASK ((UINT64_C(1) << LOCK_TEST_CHILD_PROCS) - 1U)
#define CONSECUTIVE_LAUNCHES_PER_CORE 4U
#define MAX_CONSECUTIVE_LAUNCHES 16U

struct lock_test_results {
    uint64_t release;
    uint64_t release_padding[7];
    uint64_t root_begin;
    uint64_t child_begin[LOCK_TEST_CHILD_PROCS];
    uint64_t child_end[LOCK_TEST_CHILD_PROCS];
    uint64_t started_mask;
    uint64_t family_end;
    uint64_t normal_begin;
};

_Static_assert(sizeof(struct lock_test_results) == 128,
               "locked-task cmodel result ABI");

struct run_state {
    dayan_server_t *server;
    int status;
};

struct proc_exit_capture {
    unsigned count;
    unsigned returned;
    unsigned cancelled;
    unsigned bad;
};

static void capture_proc_exit(void *context, uint32_t cluster,
                              uint32_t core, uint32_t proc,
                              int64_t result, uint64_t cookie)
{
    struct proc_exit_capture *capture = context;

    (void)cluster;
    (void)core;
    (void)cookie;
    if (!proc)
        __atomic_add_fetch(&capture->bad, 1U, __ATOMIC_RELAXED);
    if (result == -ECANCELED) {
        __atomic_add_fetch(&capture->cancelled, 1U, __ATOMIC_RELAXED);
    } else if (result > INT32_MAX) {
        __atomic_add_fetch(&capture->returned, 1U, __ATOMIC_RELAXED);
    } else {
        __atomic_add_fetch(&capture->bad, 1U, __ATOMIC_RELAXED);
    }
    __atomic_add_fetch(&capture->count, 1U, __ATOMIC_RELAXED);
}

static int run_kill_test(dayan_server_t *server, npu_stream_t *stream,
                         npu_kernel_t *kernel, npu_buffer_t *buffer)
{
    npu_service_kill_result_t killed;
    npu_service_proc_snapshot_t snapshot;
    npu_launch_desc_t launch;
    npu_event_info_t info;
    npu_event_t *event = 0;
    npu_hal_core_t target = { 0, 0 };
    uint64_t value = UINT64_C(0x9876543210);
    unsigned attempt;
    int found = 0;
    int status;

    npu_launch_desc_init(&launch, "kill-test");
    launch.config.task_dim = (NPUTaskDim3){ 1, 1, 1 };
    launch.config.task_mode = (NPUTaskMode)1;
    launch.arg_count = 2;
    npu_kernel_arg_buffer(&launch.args[0], buffer, 0,
                          NPU_KERNEL_ARG_WRITE);
    npu_kernel_arg_value(&launch.args[1], value, sizeof(value));
    status = npu_kernel_launch(stream, kernel, &launch, &event);
    if (status < 0) {
        fprintf(stderr, "kill test launch failed: %d\n", status);
        return status;
    }
    for (attempt = 0; attempt < 5000U && !found; ++attempt) {
        uint32_t cluster;

        for (cluster = 0; cluster < server->hal_info.cluster_count &&
                          !found; ++cluster) {
            uint32_t core;

            for (core = 0; core < server->hal_info.cores_per_cluster;
                 ++core) {
                uint32_t record;

                target.cluster = cluster;
                target.core = core;
                status = npu_service_rpc_ps(server->device, target,
                                            &snapshot);
                if (status < 0)
                    continue;
                for (record = 0; record < snapshot.count; ++record) {
                    if (snapshot.records[record].flags &
                        NPU_SERVICE_PROC_HOST_ROOT) {
                        if (snapshot.records[record].host_proc_id !=
                                snapshot.records[record].proc_id ||
                            snapshot.records[record].parent_proc_id ||
                            strcmp(snapshot.records[record].name,
                                   "kill-test") != 0 ||
                            snapshot.reserved_slots != 1U ||
                            snapshot.global_free_slots != 15U)
                            return -EIO;
                        found = 1;
                        break;
                    }
                }
                if (found)
                    break;
            }
        }
        if (!found) {
            const struct timespec delay = { 0, 1000000L };

            (void)nanosleep(&delay, 0);
        }
    }
    if (!found) {
        fprintf(stderr, "kill test proc was not visible through rpc_ps\n");
        return -ETIMEDOUT;
    }
    status = npu_service_rpc_kill_all(server->device, target, &killed);
    if (status < 0) {
        fprintf(stderr, "kill test rpc_kill_all failed: %d\n", status);
        return status;
    }
    if (killed.killed_roots != 1U || killed.killed_children ||
        killed.released_slots != 1U || killed.waiting_produce)
        return -EIO;
    status = npu_service_rpc_ps(server->device, target, &snapshot);
    if (status < 0 || snapshot.count != 3U ||
        snapshot.global_free_slots != 16U || snapshot.reserved_slots)
        return status < 0 ? status : -EIO;
    status = npu_event_wait(event, 10000, &info);
    if (status < 0) {
        fprintf(stderr, "kill test event wait failed: %d\n", status);
        return status;
    }
    if (info.status != -ECANCELED)
        return -EIO;
    return npu_event_destroy(event);
}

static void *run_server(void *argument)
{
    struct run_state *state = argument;

    state->status = dayan_server_run(state->server, 0);
    return 0;
}

static int read_file(const char *path, void **image_out, size_t *size_out)
{
    long length;
    void *image;
    FILE *file = fopen(path, "rb");

    if (!file)
        return -errno;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) <= 0 || (uint64_t)length > SIZE_MAX ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -EIO;
    }
    image = malloc((size_t)length);
    if (!image) {
        fclose(file);
        return -ENOMEM;
    }
    if (fread(image, 1, (size_t)length, file) != (size_t)length ||
        fclose(file) != 0) {
        free(image);
        return -EIO;
    }
    *image_out = image;
    *size_out = (size_t)length;
    return 0;
}

static int wait_and_destroy(npu_event_t *event)
{
    npu_event_info_t info;
    int status = npu_event_wait(event, 10000, &info);

    if (status < 0)
        return status;
    if (info.status < 0)
        return info.status < INT32_MIN ? -EIO : (int)info.status;
    return npu_event_destroy(event);
}

static int run_grid(npu_stream_t *stream, npu_kernel_t *kernel,
                    npu_buffer_t *buffer, NPUTaskDim3 dim,
                    uint32_t mode, uint64_t base, uint64_t *output,
                    size_t count)
{
    uint64_t zeros[8] = { 0 };
    npu_launch_desc_t launch;
    npu_event_t *upload = 0;
    npu_event_t *execution = 0;
    npu_event_t *download = 0;
    size_t index;
    int status;

    if (count > sizeof(zeros) / sizeof(zeros[0]))
        return -EINVAL;
    status = npu_memcpy_h2d_async(stream, buffer, 0, zeros,
                                  count * sizeof(zeros[0]), &upload);
    if (status < 0) {
        fprintf(stderr, "grid upload submit failed: %d\n", status);
        return status;
    }
    npu_launch_desc_init(&launch, "grid-kernel");
    launch.config.task_dim = dim;
    launch.config.task_mode = (NPUTaskMode)mode;
    launch.arg_count = 2;
    npu_kernel_arg_buffer(&launch.args[0], buffer, 0,
                          NPU_KERNEL_ARG_WRITE);
    npu_kernel_arg_value(&launch.args[1], base, sizeof(base));
    status = npu_kernel_launch(stream, kernel, &launch, &execution);
    if (status < 0) {
        fprintf(stderr, "grid kernel submit failed: %d\n", status);
        return status;
    }
    memset(output, 0, count * sizeof(output[0]));
    status = npu_memcpy_d2h_async(stream, output, buffer, 0,
                                  count * sizeof(output[0]), &download);
    if (status < 0) {
        fprintf(stderr, "grid download submit failed: %d\n", status);
        return status;
    }
    status = wait_and_destroy(upload);
    if (status < 0)
        fprintf(stderr, "grid upload completion failed: %d\n", status);
    if (status == 0) {
        status = wait_and_destroy(execution);
        if (status < 0)
            fprintf(stderr, "grid execution failed: %d\n", status);
    }
    if (status == 0) {
        status = wait_and_destroy(download);
        if (status < 0)
            fprintf(stderr, "grid download completion failed: %d\n", status);
    }
    if (status < 0)
        return status;
    for (index = 0; index < count; ++index) {
        uint64_t expected = base + (uint64_t)mode * 100U + index;

        if (output[index] != expected) {
            fprintf(stderr,
                    "grid mismatch at %zu: expected 0x%llx, got 0x%llx\n",
                    index, (unsigned long long)expected,
                    (unsigned long long)output[index]);
            return -EIO;
        }
    }
    return 0;
}

static int run_consecutive_launches(npu_context_t *context,
                                    npu_kernel_t *kernel,
                                    unsigned launch_count)
{
    npu_stream_t *streams[MAX_CONSECUTIVE_LAUNCHES] = { 0 };
    npu_buffer_t *buffers[MAX_CONSECUTIVE_LAUNCHES] = { 0 };
    npu_event_t *executions[MAX_CONSECUTIVE_LAUNCHES] = { 0 };
    npu_event_t *downloads[MAX_CONSECUTIVE_LAUNCHES] = { 0 };
    uint64_t outputs[MAX_CONSECUTIVE_LAUNCHES] = { 0 };
    const uint64_t base = UINT64_C(0x3456789000005000);
    unsigned index;
    int status = 0;

    if (!launch_count || launch_count > MAX_CONSECUTIVE_LAUNCHES)
        return -EINVAL;
    for (index = 0; index < launch_count; ++index) {
        npu_buffer_desc_t buffer_desc;

        status = npu_stream_create(context, 0, &streams[index]);
        if (status < 0)
            goto out;
        npu_buffer_desc_init(&buffer_desc, sizeof(outputs[index]));
        status = npu_buffer_create(context, &buffer_desc, &buffers[index]);
        if (status < 0)
            goto out;
    }
    for (index = 0; index < launch_count; ++index) {
        npu_launch_desc_t launch;
        uint64_t value = base + index;

        npu_launch_desc_init(&launch, "sequence-kernel");
        launch.arg_count = 2;
        npu_kernel_arg_buffer(&launch.args[0], buffers[index], 0,
                              NPU_KERNEL_ARG_WRITE);
        npu_kernel_arg_value(&launch.args[1], value, sizeof(value));
        status = npu_kernel_launch(streams[index], kernel, &launch,
                                   &executions[index]);
        if (status < 0) {
            fprintf(stderr, "sequence launch %u submit failed: %d\n",
                    index, status);
            goto out;
        }
    }
    for (index = 0; index < launch_count; ++index) {
        status = npu_memcpy_d2h_async(streams[index], &outputs[index],
                                      buffers[index], 0,
                                      sizeof(outputs[index]),
                                      &downloads[index]);
        if (status < 0)
            goto out;
    }
    for (index = 0; index < launch_count; ++index) {
        status = wait_and_destroy(executions[index]);
        if (status < 0)
            goto out;
        executions[index] = 0;
        status = wait_and_destroy(downloads[index]);
        if (status < 0)
            goto out;
        downloads[index] = 0;

        if (outputs[index] != base + index) {
            fprintf(stderr,
                    "sequence launch %u output mismatch: expected=0x%llx "
                    "actual=0x%llx\n",
                    index, (unsigned long long)(base + index),
                    (unsigned long long)outputs[index]);
            status = -EIO;
            goto out;
        }
    }

out:
    for (index = 0; index < launch_count; ++index) {
        if (executions[index])
            (void)wait_and_destroy(executions[index]);
        if (downloads[index])
            (void)wait_and_destroy(downloads[index]);
        if (streams[index])
            (void)npu_stream_destroy(streams[index]);
        if (buffers[index])
            (void)npu_buffer_destroy(buffers[index]);
    }
    return status;
}

static int copy_lock_results(npu_stream_t *stream, npu_buffer_t *buffer,
                             struct lock_test_results *results)
{
    npu_event_t *event = 0;
    int status = npu_memcpy_d2h_async(stream, results, buffer, 0,
                                      sizeof(*results), &event);

    return status < 0 ? status : wait_and_destroy(event);
}

static int release_locked_children(npu_stream_t *stream,
                                   npu_buffer_t *buffer)
{
    const uint64_t release = 1;
    npu_event_t *event = 0;
    int status = npu_memcpy_h2d_async(
        stream, buffer, offsetof(struct lock_test_results, release),
        &release, sizeof(release), &event);

    return status < 0 ? status : wait_and_destroy(event);
}

static int run_locked_task_test(npu_stream_t *locked_stream,
                                npu_stream_t *normal_stream,
                                npu_stream_t *transfer_stream,
                                npu_kernel_t *locked_kernel,
                                npu_kernel_t *normal_kernel,
                                npu_buffer_t *buffer)
{
    struct lock_test_results zeros = { 0 };
    struct lock_test_results results = { 0 };
    npu_launch_desc_t launch;
    npu_event_info_t locked_info;
    npu_event_info_t normal_info;
    npu_event_t *upload = 0;
    npu_event_t *locked_event = 0;
    npu_event_t *normal_event = 0;
    npu_event_t *unexpected = 0;
    unsigned attempt;
    int released = 0;
    int status;

    npu_launch_desc_init(&launch, "locked-invalid");
    launch.config.task_dim = (NPUTaskDim3){ 4, 1, 1 };
    launch.config.task_mode = (NPUTaskMode)4;
    launch.arg_count = 1;
    npu_kernel_arg_buffer(&launch.args[0], buffer, 0,
                          NPU_KERNEL_ARG_READ | NPU_KERNEL_ARG_WRITE);
    if (npu_kernel_launch(locked_stream, locked_kernel, &launch,
                          &unexpected) != -EINVAL || unexpected)
        return -EIO;

    status = npu_memcpy_h2d_async(transfer_stream, buffer, 0, &zeros,
                                  sizeof(zeros), &upload);
    if (status == 0)
        status = wait_and_destroy(upload);
    if (status < 0)
        return status;

    npu_launch_desc_init(&launch, "locked-family");
    launch.arg_count = 1;
    npu_kernel_arg_buffer(&launch.args[0], buffer, 0,
                          NPU_KERNEL_ARG_READ | NPU_KERNEL_ARG_WRITE);
    status = npu_kernel_launch(locked_stream, locked_kernel, &launch,
                               &locked_event);
    if (status < 0)
        return status;
    for (attempt = 0; attempt < 5000U; ++attempt) {
        status = npu_event_query(locked_event, &locked_info);
        if (status < 0 || locked_info.state == NPU_EVENT_RUNNING)
            break;
        if (locked_info.state != NPU_EVENT_QUEUED) {
            status = -EIO;
            break;
        }
        {
            const struct timespec delay = { 0, 1000000L };

            (void)nanosleep(&delay, 0);
        }
    }
    if (status == 0 && (attempt == 5000U ||
                        locked_info.state != NPU_EVENT_RUNNING))
        status = -ETIMEDOUT;
    if (status < 0)
        goto out;

    npu_launch_desc_init(&launch, "normal-task");
    launch.arg_count = 1;
    npu_kernel_arg_buffer(&launch.args[0], buffer, 0,
                          NPU_KERNEL_ARG_READ | NPU_KERNEL_ARG_WRITE);
    status = npu_kernel_launch(normal_stream, normal_kernel, &launch,
                               &normal_event);
    if (status < 0)
        goto out;

    for (attempt = 0; attempt < 5000U; ++attempt) {
        status = npu_event_query(locked_event, &locked_info);
        if (status < 0)
            break;
        status = npu_event_query(normal_event, &normal_info);
        if (status < 0)
            break;
        if (locked_info.state != NPU_EVENT_RUNNING ||
            normal_info.state != NPU_EVENT_QUEUED ||
            normal_info.status != -EINPROGRESS) {
            status = -EIO;
            break;
        }
        status = copy_lock_results(transfer_stream, buffer, &results);
        if (status < 0 ||
            results.started_mask == LOCK_TEST_STARTED_MASK)
            break;
        {
            const struct timespec delay = { 0, 1000000L };

            (void)nanosleep(&delay, 0);
        }
    }
    if (status == 0 && (attempt == 5000U ||
                        results.started_mask != LOCK_TEST_STARTED_MASK))
        status = -ETIMEDOUT;
    if (status < 0)
        goto out;

    status = release_locked_children(transfer_stream, buffer);
    if (status < 0)
        goto out;
    released = 1;
    status = wait_and_destroy(locked_event);
    if (status < 0)
        goto out;
    locked_event = 0;
    status = wait_and_destroy(normal_event);
    if (status < 0)
        goto out;
    normal_event = 0;
    status = copy_lock_results(transfer_stream, buffer, &results);
    if (status < 0)
        goto out;
    if (!results.root_begin ||
        results.started_mask != LOCK_TEST_STARTED_MASK ||
        !results.family_end || !results.normal_begin ||
        results.normal_begin < results.family_end)
        status = -EIO;
    for (attempt = 0; status == 0 && attempt < LOCK_TEST_CHILD_PROCS;
         ++attempt) {
        if (results.child_begin[attempt] < results.root_begin ||
            results.child_end[attempt] < results.child_begin[attempt] ||
            results.family_end < results.child_end[attempt])
            status = -EIO;
    }

out:
    if (locked_event && !released) {
        int release_status = release_locked_children(transfer_stream, buffer);

        if (status == 0 && release_status < 0)
            status = release_status;
    }
    if (locked_event) {
        int wait_status = wait_and_destroy(locked_event);

        if (status == 0 && wait_status < 0)
            status = wait_status;
    }
    if (normal_event) {
        int wait_status = wait_and_destroy(normal_event);

        if (status == 0 && wait_status < 0)
            status = wait_status;
    }
    return status;
}

static int finish_profile(npu_profile_t *profile, unsigned core_count)
{
    unsigned char data[NPU_PROFILE_MAX_BATCH_BYTES];
    char slot_names[32][NPU_PROFILE_SLOT_COUNT][NPU_PROC_NAME_SIZE] = {
        { { 0 } },
    };
    uint64_t record_count = 0;
    uint32_t first_hart_mask = 0;
    unsigned first_hart_count = 0;
    int saw_running = 0;
    int saw_ready = 0;
    int saw_wait = 0;
    int saw_exit = 0;
    int saw_full = 0;
    int saw_named_root = 0;
    int saw_named_child = 0;
    int end = 0;
    int status;

    status = npu_profile_stop(profile);
    if (status < 0)
        return status;
    while (!end) {
        npu_profile_batch_info_t batch;
        size_t offset = 0;

        status = npu_profile_read(profile, data,
                                  NPU_PROFILE_MAX_BATCH_BYTES, 10000,
                                  &batch);
        if (status < 0)
            return status;
        if (batch.cluster || batch.hart >= core_count ||
            (batch.flags & ~NPU_PROFILE_BATCH_END))
            return -EIO;
        if (batch.byte_count >=
            1024U - NPU_PROFILE_SLOT_DEFINE_SIZE)
            saw_full = 1;
        if (batch.byte_count &&
            !(first_hart_mask & (UINT32_C(1) << batch.hart))) {
            if (batch.byte_count < 96U)
                return -EIO;
            first_hart_mask |= UINT32_C(1) << batch.hart;
            ++first_hart_count;
        }
        while (offset < batch.byte_count) {
            uint64_t raw;
            uint8_t slot;
            uint8_t state;

            if (batch.byte_count - offset < NPU_PROFILE_RECORD_SIZE)
                return -EIO;
            memcpy(&raw, data + offset, sizeof(raw));
            slot = npu_profile_record_slot(raw);
            state = npu_profile_record_state(raw);
            if (!slot || slot >= NPU_PROFILE_SLOT_COUNT)
                return -EIO;
            if (state == NPU_PROFILE_RECORD_SLOT_DEFINE) {
                const char *name;

                if (batch.byte_count - offset <
                    NPU_PROFILE_SLOT_DEFINE_SIZE)
                    return -EIO;
                name = (const char *)data + offset +
                       NPU_PROFILE_RECORD_SIZE;
                if (!name[0] || strnlen(name, NPU_PROC_NAME_SIZE) ==
                                NPU_PROC_NAME_SIZE)
                    return -EIO;
                memcpy(slot_names[batch.hart][slot], name,
                       NPU_PROC_NAME_SIZE);
                if (strcmp(name, "grid-kernel") == 0 ||
                    strcmp(name, "kill-test") == 0)
                    saw_named_root = 1;
                if (strcmp(name, "child-write") == 0 ||
                    strcmp(name, "nested-write") == 0)
                    saw_named_child = 1;
                offset += NPU_PROFILE_SLOT_DEFINE_SIZE;
                continue;
            }
            if (!slot_names[batch.hart][slot][0])
                return -EIO;
            switch (state) {
            case NPU_PROFILE_PROC_RUNNING:
                saw_running = 1;
                break;
            case NPU_PROFILE_PROC_READY:
                saw_ready = 1;
                break;
            case NPU_PROFILE_PROC_WAIT_POLL:
            case NPU_PROFILE_PROC_WAIT_EVENT:
            case NPU_PROFILE_PROC_WAIT_START:
            case NPU_PROFILE_PROC_WAIT_PRODUCE:
            case NPU_PROFILE_PROC_KILL_WAIT_PRODUCE:
                saw_wait = 1;
                break;
            case NPU_PROFILE_PROC_WAIT_EXIT:
                saw_exit = 1;
                break;
            default:
                return -EIO;
            }
            offset += NPU_PROFILE_RECORD_SIZE;
            ++record_count;
        }
        end = !!(batch.flags & NPU_PROFILE_BATCH_END);
    }
    if (!record_count || first_hart_count != core_count || !saw_running ||
        !saw_ready || !saw_wait || !saw_exit || !saw_full ||
        !saw_named_root || !saw_named_child)
        return -EIO;
    return npu_profile_destroy(profile);
}

int main(int argc, char **argv)
{
    npu_package_argument_t arguments[3];
    npu_package_kernel_desc_t kernel_desc[5];
    dayan_server_options_t server_options;
    npu_runtime_options_t runtime_options;
    npu_context_t *context = 0;
    npu_buffer_t *buffer = 0;
    npu_stream_t *stream = 0;
    npu_stream_t *normal_stream = 0;
    npu_stream_t *transfer_stream = 0;
    npu_module_t *module = 0;
    npu_kernel_t *kernel = 0;
    npu_kernel_t *killable_kernel = 0;
    npu_kernel_t *locked_kernel = 0;
    npu_kernel_t *normal_kernel = 0;
    npu_kernel_t *sequence_kernel = 0;
    npu_runtime_t *runtime = 0;
    npu_profile_t *profile = 0;
    dayan_server_t *server = 0;
    struct run_state run;
    struct proc_exit_capture exits = { 0, 0, 0, 0 };
    pthread_t server_thread;
    void *object = 0;
    void *package = 0;
    size_t object_size = 0;
    size_t package_size = 0;
    uint64_t output[8];
    unsigned core_count;
    unsigned consecutive_launches;
    char socket_path[128];
    const char *device_prefix;
    const char *stage = "runtime open";
    int status;

    if (argc != 5) {
        fprintf(stderr,
                "usage: %s BL1.elf API.elf KERNEL.o CORE_COUNT\n", argv[0]);
        return 2;
    }
    core_count = (unsigned)strtoul(argv[4], 0, 0);
    if (!core_count || core_count > 4)
        return 2;
    consecutive_launches = core_count * CONSECUTIVE_LAUNCHES_PER_CORE;
    device_prefix = getenv("NPU_DEVICE_PREFIX");
    if (!device_prefix || !*device_prefix)
        device_prefix = "/tmp/virtual_npu";
    status = read_file(argv[3], &object, &object_size);
    if (status < 0)
        return 1;
    memset(arguments, 0, sizeof(arguments));
    arguments[0].kind = NPU_PACKAGE_ARG_BUFFER;
    arguments[0].size = 8;
    arguments[0].alignment = 8;
    arguments[0].access = NPU_PACKAGE_ARG_WRITE;
    arguments[1].kind = NPU_PACKAGE_ARG_VALUE;
    arguments[1].size = 8;
    arguments[2] = arguments[0];
    arguments[2].access = NPU_PACKAGE_ARG_READ | NPU_PACKAGE_ARG_WRITE;
    memset(kernel_desc, 0, sizeof(kernel_desc));
    kernel_desc[0].name = "grid_kernel";
    kernel_desc[0].entry_symbol = "grid_kernel";
    kernel_desc[0].supported_task_modes = NPU_PACKAGE_TASK_MODE_ALL;
    kernel_desc[0].max_tasks_per_core = 1;
    kernel_desc[0].max_child_procs = NPU_PACKAGE_MAX_CHILD_PROCS;
    kernel_desc[0].atomic_cache_size = 64;
    kernel_desc[0].uram_size = 0x180000;
    kernel_desc[0].arguments = arguments;
    kernel_desc[0].argument_count = 2;
    kernel_desc[1] = kernel_desc[0];
    kernel_desc[1].name = "kill_kernel";
    kernel_desc[1].entry_symbol = "kill_kernel";
    kernel_desc[1].max_child_procs = 0;
    kernel_desc[2].name = "locked_kernel";
    kernel_desc[2].entry_symbol = "locked_kernel";
    kernel_desc[2].supported_task_modes =
        NPU_PACKAGE_TASK_MODE_SINGLE_CORE;
    kernel_desc[2].max_tasks_per_core = 1;
    kernel_desc[2].max_child_procs = LOCK_TEST_CHILD_PROCS;
    kernel_desc[2].flags = NPU_PACKAGE_KERNEL_LOCKED_TASK;
    kernel_desc[2].arguments = &arguments[2];
    kernel_desc[2].argument_count = 1;
    kernel_desc[3] = kernel_desc[2];
    kernel_desc[3].name = "normal_kernel";
    kernel_desc[3].entry_symbol = "normal_kernel";
    kernel_desc[3].max_tasks_per_core = 4;
    kernel_desc[3].max_child_procs = 0;
    kernel_desc[3].flags = 0;
    kernel_desc[4].name = "sequence_kernel";
    kernel_desc[4].entry_symbol = "sequence_kernel";
    kernel_desc[4].supported_task_modes =
        NPU_PACKAGE_TASK_MODE_SINGLE_CORE;
    kernel_desc[4].max_tasks_per_core = 4;
    kernel_desc[4].arguments = arguments;
    kernel_desc[4].argument_count = 2;
    status = npu_package_build(object, object_size, kernel_desc, 5, 0,
                               &package, &package_size);
    free(object);
    if (status < 0)
        return 1;
    assert(snprintf(socket_path, sizeof(socket_path),
                    "/tmp/dayan-cmodel-%ld.sock", (long)getpid()) > 0);
    dayan_server_options_init(&server_options);
    server_options.socket_path = socket_path;
    server_options.firmware_path = argv[1];
    server_options.api_elf_path = argv[2];
    server_options.enabled_core_mask = (UINT32_C(1) << core_count) - 1U;
    server_options.hal.device_prefix = device_prefix;
    server_options.service.proc_exit = capture_proc_exit;
    server_options.service.callback_context = &exits;
    status = dayan_server_create(&server, &server_options);
    if (status < 0) {
        fprintf(stderr, "server create failed: %d\n", status);
        return 1;
    }
    {
        npu_service_kill_result_t killed;
        npu_service_proc_snapshot_t snapshot;
        npu_hal_core_t target = { 0, 0 };

        status = npu_service_rpc_ps(server->device, target, &snapshot);
        if (status == 0 &&
            (snapshot.count != 3U || snapshot.global_free_slots != 16U ||
             snapshot.reserved_slots ||
             strcmp(snapshot.records[0].name, "rpc-receive") != 0 ||
             strcmp(snapshot.records[1].name, "rpc-send") != 0 ||
             strcmp(snapshot.records[2].name, "profile-flush") != 0))
            status = -EIO;
        if (status == 0)
            status = npu_service_rpc_kill_all(server->device, target,
                                              &killed);
        if (status == 0 &&
            (killed.killed_roots || killed.killed_children ||
             killed.released_slots || killed.waiting_produce))
            status = -EIO;
        if (status < 0) {
            fprintf(stderr, "admin RPC smoke failed: %d\n", status);
            dayan_server_destroy(server);
            npu_package_free(package);
            return 1;
        }
    }
    run.server = server;
    run.status = -1;
    assert(pthread_create(&server_thread, 0, run_server, &run) == 0);
    npu_runtime_options_init(&runtime_options);
    runtime_options.server_socket = socket_path;
    status = npu_runtime_open(&runtime, &runtime_options);
    if (status == 0) {
        npu_profile_options_t options;

        stage = "profile start";
        npu_profile_options_init(&options);
        /* Exactly 1024 bytes/Hart fits the snapshot and forces a flush. */
        options.buffer_size = (uint64_t)core_count * 1024U;
        status = npu_profile_start(runtime, &options, &profile);
    }
    if (status == 0) {
        stage = "context create";
        status = npu_context_create(runtime, 0, &context);
    }
    if (status == 0) {
        npu_buffer_desc_t desc;

        stage = "buffer create";
        npu_buffer_desc_init(&desc, sizeof(struct lock_test_results));
        status = npu_buffer_create(context, &desc, &buffer);
    }
    if (status == 0) {
        stage = "stream create";
        status = npu_stream_create(context, 0, &stream);
    }
    if (status == 0) {
        stage = "normal stream create";
        status = npu_stream_create(context, 0, &normal_stream);
    }
    if (status == 0) {
        stage = "transfer stream create";
        status = npu_stream_create(context, 0, &transfer_stream);
    }
    if (status == 0) {
        stage = "module load";
        status = npu_module_load(context, package, package_size, &module);
    }
    if (status == 0) {
        stage = "kernel lookup";
        status = npu_module_get_kernel(module, "grid_kernel", &kernel);
    }
    if (status == 0) {
        stage = "kill kernel lookup";
        status = npu_module_get_kernel(module, "kill_kernel",
                                       &killable_kernel);
    }
    if (status == 0) {
        stage = "locked kernel lookup";
        status = npu_module_get_kernel(module, "locked_kernel",
                                       &locked_kernel);
    }
    if (status == 0) {
        stage = "normal kernel lookup";
        status = npu_module_get_kernel(module, "normal_kernel",
                                       &normal_kernel);
    }
    if (status == 0) {
        stage = "sequence kernel lookup";
        status = npu_module_get_kernel(module, "sequence_kernel",
                                       &sequence_kernel);
    }
    if (status == 0) {
        npu_launch_desc_t invalid;
        npu_event_t *unexpected = 0;

        stage = "four-core dimension validation";
        npu_launch_desc_init(&invalid, "invalid-grid");
        invalid.config.task_dim = (NPUTaskDim3){ 3, 1, 1 };
        invalid.config.task_mode = (NPUTaskMode)4;
        if (npu_kernel_launch(stream, kernel, &invalid, &unexpected) !=
                -EINVAL || unexpected)
            status = -EIO;
    }
    if (status == 0) {
        stage = "single-core grid";
        status = run_grid(stream, kernel, buffer,
                          (NPUTaskDim3){ 2, 2, 2 }, 1,
                          UINT64_C(0x1234567800001000), output, 8);
    }
    if (status == 0 && core_count == 1) {
        stage = "locked-task Hart exclusion";
        status = run_locked_task_test(stream, normal_stream, transfer_stream,
                                      locked_kernel, normal_kernel, buffer);
    }
    if (status == 0 && core_count == 4) {
        stage = "four-core grid";
        status = run_grid(stream, kernel, buffer,
                          (NPUTaskDim3){ 4, 2, 1 }, 4,
                          UINT64_C(0x2345678900002000), output, 8);
    }
    if (status == 0) {
        stage = "consecutive asynchronous launches";
        status = run_consecutive_launches(context, sequence_kernel,
                                          consecutive_launches);
    }
    if (status == 0) {
        stage = "rpc kill all";
        status = run_kill_test(server, stream, killable_kernel, buffer);
    }
    if (status == 0) {
        stage = "profile stop and drain";
        status = finish_profile(profile, core_count);
        if (status == 0)
            profile = 0;
    }
    if (status == 0 &&
        (__atomic_load_n(&exits.bad, __ATOMIC_RELAXED) ||
         __atomic_load_n(&exits.count, __ATOMIC_RELAXED) !=
                 9U + (core_count == 1 ? 2U : 0U) +
                     (core_count == 4 ? 8U : 0U) + consecutive_launches ||
         __atomic_load_n(&exits.returned, __ATOMIC_RELAXED) !=
                 8U + (core_count == 1 ? 2U : 0U) +
                     (core_count == 4 ? 8U : 0U) + consecutive_launches ||
         __atomic_load_n(&exits.cancelled, __ATOMIC_RELAXED) != 1U)) {
        stage = "root-only proc exit notification";
        status = -EIO;
    }
    if (status < 0)
        fprintf(stderr, "cmodel runtime test failed at %s: %d\n", stage,
                status);
    if (kernel)
        (void)npu_kernel_release(kernel);
    if (killable_kernel)
        (void)npu_kernel_release(killable_kernel);
    if (locked_kernel)
        (void)npu_kernel_release(locked_kernel);
    if (normal_kernel)
        (void)npu_kernel_release(normal_kernel);
    if (sequence_kernel)
        (void)npu_kernel_release(sequence_kernel);
    if (module)
        (void)npu_module_unload(module);
    if (stream)
        (void)npu_stream_destroy(stream);
    if (normal_stream)
        (void)npu_stream_destroy(normal_stream);
    if (transfer_stream)
        (void)npu_stream_destroy(transfer_stream);
    if (buffer)
        (void)npu_buffer_destroy(buffer);
    if (context)
        (void)npu_context_destroy(context);
    if (runtime)
        (void)npu_runtime_close(runtime);
    npu_package_free(package);
    dayan_server_request_stop(server);
    (void)pthread_join(server_thread, 0);
    dayan_server_destroy(server);
    if (status < 0 || run.status < 0)
        return 1;
    puts("cmodel smoke: BL1, logical grid, locked tasks and consecutive "
         "asynchronous launches passed");
    return 0;
}
