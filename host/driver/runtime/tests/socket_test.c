#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "npu_package.h"
#include "npu_runtime.h"
#include "platform_regs.h"
#include "rpc.h"
#include "runtime_internal.h"
#include "server_internal.h"
#include "service_internal.h"

struct run_state {
    dayan_server_t *server;
    int status;
};

struct output_capture {
    char data[128];
    size_t size;
    uint32_t cluster;
    uint32_t core;
    uint32_t proc;
    uint32_t flags;
    unsigned calls;
};

struct profile_read_state {
    npu_runtime_t *runtime;
    uint64_t profile_id;
    int status;
};

struct rpc_call_state {
    npu_service_device_t *device;
    npu_hal_core_t target;
    uint16_t opcode;
    uint32_t timeout_ms;
    uint64_t argument;
    uint64_t response;
    int status;
    int started;
    int finished;
};

static void capture_output(void *context, uint32_t cluster, uint32_t core,
                           uint32_t proc, const void *data, size_t size,
                           uint32_t flags)
{
    struct output_capture *capture = context;

    assert(capture);
    assert(size <= sizeof(capture->data) - capture->size);
    if (size)
        memcpy(capture->data + capture->size, data, size);
    capture->size += size;
    capture->cluster = cluster;
    capture->core = core;
    capture->proc = proc;
    capture->flags = flags;
    ++capture->calls;
}

static void create_file(const char *directory, const char *name, size_t size)
{
    char path[512];
    int fd;

    assert(snprintf(path, sizeof(path), "%s/%s", directory, name) > 0);
    fd = open(path, O_CREAT | O_EXCL | O_RDWR, 0600);
    assert(fd >= 0);
    assert(ftruncate(fd, (off_t)size) == 0);
    close(fd);
}

static int open_file(const char *directory, const char *name, int flags)
{
    char path[512];

    assert(snprintf(path, sizeof(path), "%s/%s", directory, name) > 0);
    return open(path, flags);
}

static void *run_server(void *argument)
{
    struct run_state *state = argument;

    state->status = dayan_server_run(state->server, 0);
    return 0;
}

static void *run_blocking_profile_read(void *argument)
{
    struct profile_read_state *state = argument;
    npu_ipc_profile_read_t request;
    size_t response_size = 0;
    void *response = 0;

    memset(&request, 0, sizeof(request));
    request.max_bytes = NPU_PROFILE_RECORD_SIZE;
    request.timeout_ms = 3000;
    state->status = runtime_control_call_variable(
        state->runtime, NPU_IPC_PROFILE_READ, state->profile_id, &request,
        sizeof(request), &response, &response_size);
    free(response);
    return 0;
}

static void *run_blocking_rpc(void *argument)
{
    struct rpc_call_state *state = argument;

    __atomic_store_n(&state->started, 1, __ATOMIC_RELEASE);
    state->status = npu_service_rpc_call(
        state->device, state->target, state->opcode, state->argument,
        &state->response, state->timeout_ms);
    __atomic_store_n(&state->finished, 1, __ATOMIC_RELEASE);
    return 0;
}

static struct server_session *runtime_session_locked(
    dayan_server_t *server, const npu_runtime_t *runtime)
{
    struct server_session *session;

    for (session = server->sessions; session; session = session->next) {
        if (session->id == runtime->session_id)
            return session;
    }
    return 0;
}

static void make_package_object(unsigned char *object, size_t size)
{
    Elf64_Ehdr header;
    Elf64_Shdr section;

    assert(size == sizeof(header) + sizeof(section));
    memset(&header, 0, sizeof(header));
    memcpy(header.e_ident, ELFMAG, SELFMAG);
    header.e_ident[EI_CLASS] = ELFCLASS64;
    header.e_ident[EI_DATA] = ELFDATA2LSB;
    header.e_ident[EI_VERSION] = EV_CURRENT;
    header.e_type = ET_REL;
    header.e_machine = EM_RISCV;
    header.e_version = EV_CURRENT;
    header.e_ehsize = sizeof(header);
    header.e_shoff = sizeof(header);
    header.e_shentsize = sizeof(section);
    header.e_shnum = 1;
    memset(&section, 0, sizeof(section));
    memcpy(object, &header, sizeof(header));
    memcpy(object + sizeof(header), &section, sizeof(section));
}

static void assert_package_cache(dayan_server_t *server, size_t entries,
                                 size_t immutable_entries,
                                 size_t total_references)
{
    const struct server_package_cache *package;
    size_t observed_entries = 0;
    size_t observed_immutable = 0;
    size_t observed_references = 0;

    pthread_mutex_lock(&server->mutex);
    for (package = server->packages; package; package = package->next) {
        ++observed_entries;
        observed_immutable += !!package->immutable;
        observed_references += package->references;
    }
    pthread_mutex_unlock(&server->mutex);
    assert(observed_entries == entries);
    assert(observed_immutable == immutable_entries);
    assert(observed_references == total_references);
}

int main(void)
{
    static const npu_hal_memory_region_t regions[] = {
        { 0x1000, 0x10000, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_DEVICE,
          NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
        { 0x20000, 0x10000, NPU_HAL_MEMORY_COHERENT,
          NPU_HAL_MEMORY_DEVICE, NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
    };
    char directory[] = "/tmp/dayan-socket-test-XXXXXX";
    char socket_path[512];
    static const uint64_t input[] = {
        UINT64_C(0x1122334455667788), UINT64_C(0xaabbccddeeff0011),
    };
    uint64_t output[2] = { 0, 0 };
    struct output_capture output_capture = { { 0 }, 0, 0, 0, 0, 0, 0 };
    dayan_server_options_t server_options;
    npu_runtime_options_t runtime_options;
    npu_context_t *context = 0;
    npu_context_t *foreign_context = 0;
    npu_buffer_t *buffer = 0;
    npu_stream_t *stream = 0;
    npu_stream_t *parallel_stream = 0;
    npu_stream_t *foreign_stream = 0;
    npu_event_t *event = 0;
    npu_event_info_t event_info;
    npu_runtime_t *runtime = 0;
    dayan_server_t *server = 0;
    struct run_state run;
    pthread_t thread;
    int fd;

    assert(mkdtemp(directory));
    create_file(directory, "xdma0_h2c_0", 0x40000);
    create_file(directory, "xdma0_c2h_0", 0x40000);
    create_file(directory, "xdma0_user", 0x400);
    assert(snprintf(socket_path, sizeof(socket_path), "%s/server.sock",
                    directory) > 0);

    dayan_server_options_init(&server_options);
    assert(server_options.max_procs_per_hart == NPU_PACKAGE_HART_MAX_PROCS);
    server_options.max_procs_per_hart = NPU_PACKAGE_HART_MAX_PROCS + 1U;
    assert(dayan_server_create(&server, &server_options) == -EINVAL);
    assert(!server);
    server_options.max_procs_per_hart = NPU_PACKAGE_HART_MAX_PROCS - 1U;
    server_options.socket_path = socket_path;
    server_options.hal.device_prefix = directory;
    server_options.hal.memory_regions = regions;
    server_options.hal.memory_region_count =
        sizeof(regions) / sizeof(regions[0]);
    assert(dayan_server_create(&server, &server_options) == 0);
    run.server = server;
    run.status = -1;
    assert(pthread_create(&thread, 0, run_server, &run) == 0);

    npu_runtime_options_init(&runtime_options);
    runtime_options.server_socket = socket_path;
    runtime_options.output = capture_output;
    runtime_options.output_context = &output_capture;
    assert(npu_runtime_open(&runtime, &runtime_options) == 0);
    {
        struct profile_read_state profile_read;
        struct server_session *profile_session;
        npu_runtime_t *concurrent_runtime = 0;
        npu_context_t *concurrent_context = 0;
        pthread_t profile_thread;

        memset(&profile_read, 0, sizeof(profile_read));
        profile_read.runtime = runtime;
        pthread_mutex_lock(&server->mutex);
        profile_session = runtime_session_locked(server, runtime);
        assert(profile_session);
        assert(server->next_id != UINT64_MAX);
        profile_read.profile_id = ++server->next_id;
        server->profile_owner = profile_session;
        server->profile_id = profile_read.profile_id;
        server->profile_active = 1;
        server->profile_stopping = 0;
        server->profile_stopped = 0;
        pthread_mutex_unlock(&server->mutex);

        assert(pthread_create(&profile_thread, 0, run_blocking_profile_read,
                              &profile_read) == 0);
        pthread_mutex_lock(&server->mutex);
        while (!server->profile_waiters)
            assert(pthread_cond_wait(&server->condition,
                                     &server->mutex) == 0);
        pthread_mutex_unlock(&server->mutex);

        assert(npu_runtime_open(&concurrent_runtime, &runtime_options) == 0);
        assert(npu_context_create(concurrent_runtime, 0,
                                  &concurrent_context) == 0);
        pthread_mutex_lock(&server->mutex);
        assert(server->profile_waiters == 1);
        server->profile_stopped = 1;
        pthread_cond_broadcast(&server->condition);
        pthread_mutex_unlock(&server->mutex);

        assert(pthread_join(profile_thread, 0) == 0);
        assert(profile_read.status == 0);
        assert(npu_context_destroy(concurrent_context) == 0);
        assert(npu_runtime_close(concurrent_runtime) == 0);

        pthread_mutex_lock(&server->mutex);
        assert(server->profile_owner == profile_session);
        assert(server->profile_id == profile_read.profile_id);
        server->profile_owner = 0;
        server->profile_id = 0;
        server->profile_active = 0;
        server->profile_stopping = 0;
        server->profile_stopped = 0;
        pthread_cond_broadcast(&server->condition);
        pthread_mutex_unlock(&server->mutex);
    }
    {
        npu_profile_options_t profile_options;
        npu_profile_t *profile = 0;

        npu_profile_options_init(&profile_options);
        assert(profile_options.version == NPU_PROFILE_OPTIONS_VERSION);
        assert(profile_options.buffer_size ==
               NPU_PROFILE_DEFAULT_BUFFER_SIZE);
        assert(npu_profile_start(runtime, &profile_options, &profile) ==
               -ENODEV);
        assert(!profile);
    }
    assert(npu_context_create(runtime, 0, &context) == 0);
    {
        npu_buffer_desc_t desc;
        npu_buffer_desc_init(&desc, sizeof(input));
        assert(npu_buffer_create(context, &desc, &buffer) == 0);
    }
    assert(npu_stream_create(context, 0, &stream) == 0);
    {
        static const char message[] = "Device output belongs to Runtime\n";
        struct server_session *session;

        pthread_mutex_lock(&server->mutex);
        session = server->sessions;
        assert(session && !session->next);
        assert(server_queue_output_locked(server, session, 1, 2, 3,
                                          message, sizeof(message) - 1,
                                          0) == 0);
        pthread_mutex_unlock(&server->mutex);
        assert(npu_stream_synchronize(stream, 5000) == 0);
        assert(output_capture.calls == 1);
        assert(output_capture.cluster == 1 && output_capture.core == 2 &&
               output_capture.proc == 3 && !output_capture.flags);
        assert(output_capture.size == sizeof(message) - 1);
        assert(memcmp(output_capture.data, message, sizeof(message) - 1) == 0);
    }

    assert(npu_memcpy_h2d_async(stream, buffer, 0, input, sizeof(input),
                                &event) == 0);
    assert(npu_event_wait(event, 5000, &event_info) == 0);
    assert(event_info.state == NPU_EVENT_COMPLETE && event_info.status == 0);
    assert(npu_event_destroy(event) == 0);
    fd = open_file(directory, "xdma0_h2c_0", O_RDONLY);
    assert(fd >= 0);
    assert(pread(fd, output, sizeof(output), 0x1000) ==
           (ssize_t)sizeof(output));
    close(fd);
    assert(memcmp(output, input, sizeof(input)) == 0);

    fd = open_file(directory, "xdma0_c2h_0", O_WRONLY);
    assert(fd >= 0);
    assert(pwrite(fd, input, sizeof(input), 0x1000) ==
           (ssize_t)sizeof(input));
    close(fd);
    memset(output, 0, sizeof(output));
    assert(npu_memcpy_d2h_async(stream, output, buffer, 0, sizeof(output),
                                &event) == 0);
    assert(npu_event_wait(event, 5000, &event_info) == 0);
    assert(event_info.state == NPU_EVENT_COMPLETE && event_info.status == 0);
    assert(memcmp(output, input, sizeof(input)) == 0);
    assert(npu_event_destroy(event) == 0);
    assert(npu_stream_synchronize(stream, 5000) == 0);

    /* Cross-Stream ordering is explicit; cross-Context object use is illegal. */
    assert(npu_stream_create(context, 0, &parallel_stream) == 0);
    assert(npu_context_create(runtime, 0, &foreign_context) == 0);
    assert(npu_stream_create(foreign_context, 0, &foreign_stream) == 0);
    {
        unsigned char object[sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr)];
        npu_package_kernel_desc_t kernel;
        npu_module_t *immutable_a = 0;
        npu_module_t *immutable_b = 0;
        npu_module_t *private_a = 0;
        npu_module_t *private_b = 0;
        void *immutable_package = 0;
        void *private_package = 0;
        size_t immutable_size = 0;
        size_t private_size = 0;

        make_package_object(object, sizeof(object));
        memset(&kernel, 0, sizeof(kernel));
        kernel.name = "cache_kernel";
        kernel.entry_symbol = "cache_kernel";
        kernel.supported_task_modes = NPU_PACKAGE_TASK_MODE_ALL;
        kernel.max_tasks_per_core = 1;
        assert(npu_package_build(object, sizeof(object), &kernel, 1,
                                 NPU_PACKAGE_IMMUTABLE_MODULE,
                                 &immutable_package, &immutable_size) == 0);
        assert(npu_package_build(object, sizeof(object), &kernel, 1, 0,
                                 &private_package, &private_size) == 0);

        assert(npu_module_load(context, immutable_package, immutable_size,
                               &immutable_a) == 0);
        assert(npu_module_load(foreign_context, immutable_package,
                               immutable_size, &immutable_b) == 0);
        assert_package_cache(server, 1, 1, 2);

        assert(npu_module_load(context, private_package, private_size,
                               &private_a) == 0);
        assert(npu_module_load(foreign_context, private_package, private_size,
                               &private_b) == 0);
        assert_package_cache(server, 3, 1, 4);

        assert(npu_module_unload(private_b) == 0);
        assert(npu_module_unload(private_a) == 0);
        assert_package_cache(server, 1, 1, 2);
        assert(npu_module_unload(immutable_b) == 0);
        assert(npu_module_unload(immutable_a) == 0);
        assert_package_cache(server, 0, 0, 0);
        npu_package_free(private_package);
        npu_package_free(immutable_package);
    }
    {
        unsigned char object[sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr)];
        npu_package_kernel_desc_t kernel;
        npu_launch_desc_t launch;
        npu_module_t *module = 0;
        npu_kernel_t *resource_kernel = 0;
        npu_event_t *unexpected = 0;
        void *package = 0;
        size_t package_size = 0;

        make_package_object(object, sizeof(object));
        memset(&kernel, 0, sizeof(kernel));
        kernel.name = "resource_kernel";
        kernel.entry_symbol = "resource_kernel";
        kernel.supported_task_modes = NPU_PACKAGE_TASK_MODE_ALL;
        kernel.max_tasks_per_core = 1;
        kernel.max_child_procs = NPU_PACKAGE_MAX_CHILD_PROCS;
        assert(npu_package_build(object, sizeof(object), &kernel, 1, 0,
                                 &package, &package_size) == 0);
        assert(npu_module_load(context, package, package_size, &module) == 0);
        assert(npu_module_get_kernel(module, "resource_kernel",
                                     &resource_kernel) == 0);
        npu_launch_desc_init(&launch, "");
        assert(npu_kernel_launch(stream, resource_kernel, &launch,
                                 &unexpected) == -EINVAL);
        assert(!unexpected);
        npu_launch_desc_init(&launch, "task-name-is-far-too-long");
        assert(npu_kernel_launch(stream, resource_kernel, &launch,
                                 &unexpected) == -EINVAL);
        assert(!unexpected);
        npu_launch_desc_init(&launch, "resource-test");
        assert(npu_kernel_launch(stream, resource_kernel, &launch,
                                 &unexpected) == -ENOSPC);
        assert(!unexpected);
        assert(npu_kernel_release(resource_kernel) == 0);
        assert(npu_module_unload(module) == 0);
        npu_package_free(package);

        resource_kernel = 0;
        module = 0;
        package = 0;
        package_size = 0;
        kernel.max_child_procs = 0;
        kernel.uram_size = 1024;
        assert(npu_package_build(object, sizeof(object), &kernel, 1, 0,
                                 &package, &package_size) == 0);
        assert(npu_module_load(context, package, package_size, &module) == 0);
        assert(npu_module_get_kernel(module, "resource_kernel",
                                     &resource_kernel) == 0);
        npu_launch_desc_init(&launch, "resource-test");
        assert(npu_kernel_launch(stream, resource_kernel, &launch,
                                 &unexpected) == -ENOSPC);
        assert(!unexpected);
        assert(npu_kernel_release(resource_kernel) == 0);
        assert(npu_module_unload(module) == 0);
        npu_package_free(package);
    }
    assert(npu_memcpy_h2d_async(stream, buffer, 0, input, sizeof(input),
                                &event) == 0);
    assert(npu_stream_wait_event(foreign_stream, event) == -EINVAL);
    {
        npu_event_t *unexpected = 0;

        assert(npu_memcpy_h2d_async(foreign_stream, buffer, 0, input,
                                    sizeof(input), &unexpected) == -EINVAL);
        assert(!unexpected);
    }
    assert(npu_stream_wait_event(parallel_stream, event) == 0);
    {
        npu_event_t *dependent = 0;
        npu_event_info_t source_info;

        assert(npu_event_record(parallel_stream, &dependent) == 0);
        assert(npu_event_wait(dependent, 5000, &event_info) == 0);
        assert(event_info.state == NPU_EVENT_COMPLETE &&
               event_info.status == 0);
        assert(npu_event_wait(event, 5000, &source_info) == 0);
        assert(source_info.state == NPU_EVENT_COMPLETE &&
               source_info.status == 0 &&
               event_info.sequence > source_info.sequence);
        assert(npu_event_destroy(dependent) == 0);
    }
    assert(npu_event_destroy(event) == 0);
    assert(npu_stream_destroy(foreign_stream) == 0);
    assert(npu_context_destroy(foreign_context) == 0);
    foreign_stream = 0;
    foreign_context = 0;
    assert(npu_stream_destroy(parallel_stream) == 0);
    parallel_stream = 0;

    assert(npu_stream_destroy(stream) == 0);
    assert(npu_buffer_destroy(buffer) == 0);
    assert(npu_context_destroy(context) == 0);
    context = 0;
    buffer = 0;
    stream = 0;

    /* Cancelling a Context must drain granted direct-HAL tokens first. */
    assert(npu_context_create(runtime, 0, &context) == 0);
    {
        npu_buffer_desc_t desc;
        unsigned char *cancel_input = malloc(0x8000);
        unsigned index;

        assert(cancel_input);
        memset(cancel_input, 0x5a, 0x8000);
        npu_buffer_desc_init(&desc, 0x8000);
        assert(npu_buffer_create(context, &desc, &buffer) == 0);
        assert(npu_stream_create(context, 0, &stream) == 0);
        for (index = 0; index < 32; ++index) {
            npu_event_t *pending_event = 0;

            assert(npu_memcpy_h2d_async(stream, buffer, 0, cancel_input,
                                        0x8000, &pending_event) == 0);
            assert(pending_event);
        }
        assert(npu_context_destroy(context) == 0);
        free(cancel_input);
    }
    assert(npu_runtime_close(runtime) == 0);
    dayan_server_request_stop(server);
    assert(pthread_join(thread, 0) == 0);
    assert(run.status == 0);
    dayan_server_destroy(server);

    /*
     * The shared RPC FIFO must preserve per-Hart order, allow independent
     * Harts to run concurrently, and block producers instead of returning
     * EBUSY when the FIFO is full.  Service abort must wake every stage.
     */
    server = 0;
    assert(dayan_server_create(&server, &server_options) == 0);
    {
        struct rpc_call_state rpc_calls[2];
        pthread_t rpc_threads[2];
        unsigned attempt;

        memset(rpc_calls, 0, sizeof(rpc_calls));
        rpc_calls[0].device = server->device;
        rpc_calls[0].opcode = RPC_ICACHE_FLUSH;
        rpc_calls[0].argument = 0x31;
        rpc_calls[0].timeout_ms = 60000;
        rpc_calls[0].status = -EINPROGRESS;
        rpc_calls[1] = rpc_calls[0];
        rpc_calls[1].argument = 0x32;
        assert(pthread_create(&rpc_threads[0], 0, run_blocking_rpc,
                              &rpc_calls[0]) == 0);
        for (attempt = 0; attempt < 1000; ++attempt) {
            int active;

            pthread_mutex_lock(&server->device->mutex);
            active = server->device->calls[0].active &&
                     server->device->calls[0].pending;
            pthread_mutex_unlock(&server->device->mutex);
            if (active)
                break;
            npu_delay_us(1000);
        }
        assert(attempt < 1000);
        assert(pthread_create(&rpc_threads[1], 0, run_blocking_rpc,
                              &rpc_calls[1]) == 0);
        for (attempt = 0; attempt < 1000; ++attempt) {
            size_t queued;

            pthread_mutex_lock(&server->device->mutex);
            queued = server->device->rpc_queue_count;
            pthread_mutex_unlock(&server->device->mutex);
            if (queued == 1)
                break;
            npu_delay_us(1000);
        }
        assert(attempt < 1000);
        pthread_mutex_lock(&server->device->mutex);
        assert(server->device->calls[0].active);
        assert(server->device->calls[0].pending);
        assert(server->device->calls[0].expected_argument ==
               rpc_calls[0].argument);
        server->device->calls[0].response = rpc_command_pack(
            server->device->calls[0].expected_opcode,
            server->device->calls[0].expected_argument);
        server->device->calls[0].status = 0;
        server->device->calls[0].pending = 0;
        assert(server->device->calls[0].active);
        pthread_cond_broadcast(&server->device->condition);
        pthread_mutex_unlock(&server->device->mutex);
        assert(pthread_join(rpc_threads[0], 0) == 0);
        assert(rpc_calls[0].status == 0);
        assert(rpc_calls[0].response == rpc_command_pack(
            RPC_ICACHE_FLUSH_RESPONSE, rpc_calls[0].argument));
        for (attempt = 0; attempt < 1000; ++attempt) {
            int second_pending;

            pthread_mutex_lock(&server->device->mutex);
            second_pending = server->device->calls[0].active &&
                server->device->calls[0].pending &&
                server->device->calls[0].expected_argument ==
                    rpc_calls[1].argument;
            pthread_mutex_unlock(&server->device->mutex);
            if (second_pending)
                break;
            npu_delay_us(1000);
        }
        assert(attempt < 1000);
        pthread_mutex_lock(&server->device->mutex);
        server->device->calls[0].response = rpc_command_pack(
            server->device->calls[0].expected_opcode,
            server->device->calls[0].expected_argument);
        server->device->calls[0].status = 0;
        server->device->calls[0].pending = 0;
        pthread_cond_broadcast(&server->device->condition);
        pthread_mutex_unlock(&server->device->mutex);
        assert(pthread_join(rpc_threads[1], 0) == 0);
        assert(rpc_calls[1].status == 0);
        assert(rpc_calls[1].response == rpc_command_pack(
            RPC_ICACHE_FLUSH_RESPONSE, rpc_calls[1].argument));
        pthread_mutex_lock(&server->device->mutex);
        assert(!server->device->calls[0].active);
        assert(!server->device->calls[0].pending);
        assert(server->device->rpc_queue_count == 0);
        pthread_mutex_unlock(&server->device->mutex);
    }
    {
        struct rpc_call_state rpc_calls[5];
        pthread_t rpc_threads[5];
        uint32_t msip = 0;
        size_t index;
        unsigned attempt;

        fd = open_file(directory, "xdma0_c2h_0", O_RDWR);
        assert(fd >= 0);
        assert(pwrite(fd, &msip, sizeof(msip), DAYAN_CLINT_MSIP(0)) ==
               (ssize_t)sizeof(msip));
        msip = 1;
        assert(pwrite(fd, &msip, sizeof(msip), DAYAN_CLINT_MSIP(1)) ==
               (ssize_t)sizeof(msip));
        pthread_mutex_lock(&server->device->mutex);
        assert(server->device->rpc_queue_count == 0);
        assert(server->device->rpc_queue_capacity ==
               NPU_SERVICE_RPC_QUEUE_CAPACITY);
        server->device->rpc_queue_capacity = 2;
        pthread_mutex_unlock(&server->device->mutex);
        memset(rpc_calls, 0, sizeof(rpc_calls));
        for (index = 0; index < sizeof(rpc_calls) / sizeof(rpc_calls[0]);
             ++index) {
            rpc_calls[index].device = server->device;
            rpc_calls[index].target.cluster = 0;
            rpc_calls[index].target.core = index == 1 ? 1 : 0;
            rpc_calls[index].opcode = RPC_ICACHE_FLUSH;
            rpc_calls[index].argument = 0x10U + index;
            rpc_calls[index].timeout_ms = 60000;
            rpc_calls[index].status = -EINPROGRESS;
        }
        assert(pthread_create(&rpc_threads[0], 0, run_blocking_rpc,
                              &rpc_calls[0]) == 0);
        assert(pthread_create(&rpc_threads[1], 0, run_blocking_rpc,
                              &rpc_calls[1]) == 0);
        for (attempt = 0; attempt < 1000; ++attempt) {
            int pending;

            pthread_mutex_lock(&server->device->mutex);
            pending = server->device->calls[0].pending &&
                      server->device->calls[1].pending;
            pthread_mutex_unlock(&server->device->mutex);
            if (pending)
                break;
            npu_delay_us(1000);
        }
        assert(attempt < 1000);

        assert(pthread_create(&rpc_threads[2], 0, run_blocking_rpc,
                              &rpc_calls[2]) == 0);
        for (attempt = 0; attempt < 1000; ++attempt) {
            size_t queued;

            pthread_mutex_lock(&server->device->mutex);
            queued = server->device->rpc_queue_count;
            pthread_mutex_unlock(&server->device->mutex);
            if (queued == 1)
                break;
            npu_delay_us(1000);
        }
        assert(attempt < 1000);
        assert(pthread_create(&rpc_threads[3], 0, run_blocking_rpc,
                              &rpc_calls[3]) == 0);
        for (attempt = 0; attempt < 1000; ++attempt) {
            size_t queued;

            pthread_mutex_lock(&server->device->mutex);
            queued = server->device->rpc_queue_count;
            pthread_mutex_unlock(&server->device->mutex);
            if (queued == 2)
                break;
            npu_delay_us(1000);
        }
        assert(attempt < 1000);
        pthread_mutex_lock(&server->device->mutex);
        assert(server->device->rpc_queue_head);
        assert(server->device->rpc_queue_head->argument ==
               rpc_calls[2].argument);
        assert(server->device->rpc_queue_head->next);
        assert(server->device->rpc_queue_head->next->argument ==
               rpc_calls[3].argument);
        assert(!server->device->rpc_queue_head->next->next);
        pthread_mutex_unlock(&server->device->mutex);

        assert(pthread_create(&rpc_threads[4], 0, run_blocking_rpc,
                              &rpc_calls[4]) == 0);
        for (attempt = 0; attempt < 1000; ++attempt) {
            size_t waiters;

            pthread_mutex_lock(&server->device->mutex);
            waiters = server->device->rpc_queue_waiters;
            pthread_mutex_unlock(&server->device->mutex);
            if (waiters == 1)
                break;
            npu_delay_us(1000);
        }
        assert(attempt < 1000);
        assert(__atomic_load_n(&rpc_calls[4].started, __ATOMIC_ACQUIRE));
        assert(!__atomic_load_n(&rpc_calls[4].finished, __ATOMIC_ACQUIRE));
        pthread_mutex_lock(&server->device->mutex);
        assert(server->device->calls[0].active);
        assert(server->device->calls[0].pending);
        assert(server->device->calls[0].expected_argument ==
               rpc_calls[0].argument);
        server->device->calls[0].response = rpc_command_pack(
            server->device->calls[0].expected_opcode,
            server->device->calls[0].expected_argument);
        server->device->calls[0].status = 0;
        server->device->calls[0].pending = 0;
        pthread_cond_broadcast(&server->device->condition);
        pthread_mutex_unlock(&server->device->mutex);
        assert(pthread_join(rpc_threads[0], 0) == 0);
        assert(rpc_calls[0].status == 0);
        for (attempt = 0; attempt < 1000; ++attempt) {
            int space_reused;

            pthread_mutex_lock(&server->device->mutex);
            space_reused = server->device->calls[0].active &&
                server->device->calls[0].pending &&
                server->device->calls[0].expected_argument ==
                    rpc_calls[2].argument &&
                server->device->rpc_queue_count == 2 &&
                server->device->rpc_queue_waiters == 0;
            pthread_mutex_unlock(&server->device->mutex);
            if (space_reused)
                break;
            npu_delay_us(1000);
        }
        assert(attempt < 1000);
        pthread_mutex_lock(&server->device->mutex);
        assert(server->device->rpc_queue_head);
        assert(server->device->rpc_queue_head->argument ==
               rpc_calls[3].argument);
        assert(server->device->rpc_queue_head->next);
        assert(server->device->rpc_queue_head->next->argument ==
               rpc_calls[4].argument);
        assert(!server->device->rpc_queue_head->next->next);
        pthread_mutex_unlock(&server->device->mutex);
        npu_service_abort(server->device, -ESHUTDOWN);
        for (index = 1; index < sizeof(rpc_calls) / sizeof(rpc_calls[0]);
             ++index) {
            assert(pthread_join(rpc_threads[index], 0) == 0);
            assert(rpc_calls[index].status == -ESHUTDOWN);
        }
        pthread_mutex_lock(&server->device->mutex);
        assert(server->device->rpc_queue_count == 0);
        assert(server->device->rpc_queue_waiters == 0);
        assert(!server->device->rpc_queue_head);
        assert(!server->device->rpc_queue_tail);
        pthread_mutex_unlock(&server->device->mutex);
        msip = 0;
        assert(pwrite(fd, &msip, sizeof(msip), DAYAN_CLINT_MSIP(0)) ==
               (ssize_t)sizeof(msip));
        assert(pwrite(fd, &msip, sizeof(msip), DAYAN_CLINT_MSIP(1)) ==
               (ssize_t)sizeof(msip));
        close(fd);
    }
    dayan_server_destroy(server);

    /* A lost XDMA/FUSE endpoint must stop the Server without a signal. */
    server = 0;
    assert(dayan_server_create(&server, &server_options) == 0);
    run.server = server;
    run.status = 0;
    assert(pthread_create(&thread, 0, run_server, &run) == 0);
    fd = open_file(directory, "xdma0_user", O_WRONLY);
    assert(fd >= 0);
    assert(ftruncate(fd, 0) == 0);
    close(fd);
    assert(pthread_join(thread, 0) == 0);
    assert(run.status == -EIO);
    dayan_server_destroy(server);
    return 0;
}
