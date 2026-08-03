#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "npu_package.h"
#include "npu_runtime.h"
#include "server_internal.h"

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

    npu_launch_desc_init(&launch);
    launch.config.task_dim = (NPUTaskDim3){ 1, 1, 1 };
    launch.config.task_mode = (NPUTaskMode)1;
    launch.arg_count = 2;
    npu_kernel_arg_buffer(&launch.args[0], buffer, 0,
                          NPU_KERNEL_ARG_WRITE);
    npu_kernel_arg_value(&launch.args[1], value, sizeof(value));
    status = npu_kernel_launch(stream, kernel, &launch, &event);
    if (status < 0)
        return status;
    for (attempt = 0; attempt < 500U && !found; ++attempt) {
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
    if (!found)
        return -ETIMEDOUT;
    status = npu_service_rpc_kill_all(server->device, target, &killed);
    if (status < 0)
        return status;
    if (killed.killed_roots != 1U || killed.killed_children ||
        killed.released_slots != 1U || killed.waiting_produce)
        return -EIO;
    status = npu_service_rpc_ps(server->device, target, &snapshot);
    if (status < 0 || snapshot.count != 3U ||
        snapshot.global_free_slots != 16U || snapshot.reserved_slots)
        return status < 0 ? status : -EIO;
    status = npu_event_wait(event, 10000, &info);
    if (status < 0)
        return status;
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
    npu_launch_desc_init(&launch);
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

static int finish_profile(npu_profile_t *profile, unsigned core_count)
{
    npu_profile_event_t events[NPU_PROFILE_MAX_BATCH_EVENTS];
    uint64_t event_count = 0;
    uint32_t first_hart_mask = 0;
    unsigned first_hart_count = 0;
    int saw_running = 0;
    int saw_ready = 0;
    int saw_wait = 0;
    int saw_exit = 0;
    int saw_full = 0;
    int end = 0;
    int status;

    status = npu_profile_stop(profile);
    if (status < 0)
        return status;
    while (!end) {
        npu_profile_batch_info_t batch;
        uint32_t index;

        status = npu_profile_read(profile, events,
                                  NPU_PROFILE_MAX_BATCH_EVENTS, 10000,
                                  &batch);
        if (status < 0)
            return status;
        if (batch.cluster || batch.hart >= core_count ||
            (batch.flags & ~NPU_PROFILE_BATCH_END))
            return -EIO;
        if (batch.count == 128U)
            saw_full = 1;
        if (batch.count && !(first_hart_mask & (UINT32_C(1) << batch.hart))) {
            if (batch.count < 3U || events[0].event_id != 1U ||
                events[0].state != NPU_PROFILE_PROC_RUNNING ||
                events[1].event_id != 2U || events[2].event_id != 3U)
                return -EIO;
            first_hart_mask |= UINT32_C(1) << batch.hart;
            ++first_hart_count;
        }
        for (index = 0; index < batch.count; ++index) {
            switch (events[index].state) {
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
        }
        event_count += batch.count;
        end = !!(batch.flags & NPU_PROFILE_BATCH_END);
    }
    if (!event_count || first_hart_count != core_count || !saw_running ||
        !saw_ready || !saw_wait || !saw_exit || !saw_full)
        return -EIO;
    return npu_profile_destroy(profile);
}

int main(int argc, char **argv)
{
    npu_package_argument_t arguments[2];
    npu_package_kernel_desc_t kernel_desc[2];
    dayan_server_options_t server_options;
    npu_runtime_options_t runtime_options;
    npu_context_t *context = 0;
    npu_buffer_t *buffer = 0;
    npu_stream_t *stream = 0;
    npu_module_t *module = 0;
    npu_kernel_t *kernel = 0;
    npu_kernel_t *killable_kernel = 0;
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
    status = npu_package_build(object, object_size, kernel_desc, 2, 0,
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
             snapshot.reserved_slots))
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
        /* 1 KiB/Hart fits the snapshot and still forces a full flush. */
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
        npu_buffer_desc_init(&desc, sizeof(output));
        status = npu_buffer_create(context, &desc, &buffer);
    }
    if (status == 0) {
        stage = "stream create";
        status = npu_stream_create(context, 0, &stream);
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
        npu_launch_desc_t invalid;
        npu_event_t *unexpected = 0;

        stage = "four-core dimension validation";
        npu_launch_desc_init(&invalid);
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
    if (status == 0 && core_count == 4) {
        stage = "four-core grid";
        status = run_grid(stream, kernel, buffer,
                          (NPUTaskDim3){ 4, 2, 1 }, 4,
                          UINT64_C(0x2345678900002000), output, 8);
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
                 (core_count == 4 ? 17U : 9U) ||
         __atomic_load_n(&exits.returned, __ATOMIC_RELAXED) !=
                 (core_count == 4 ? 16U : 8U) ||
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
    if (module)
        (void)npu_module_unload(module);
    if (stream)
        (void)npu_stream_destroy(stream);
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
    puts("cmodel smoke: BL1, socket/token memcpy and logical grid passed");
    return 0;
}
