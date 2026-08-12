#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "server_internal.h"

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define DAYAN_SERVER_BACKLOG 64U
#define DAYAN_SERVER_DEFAULT_RESIDENT (64U * 1024U * 1024U)
#define DAYAN_SERVER_DEFAULT_PROCS_PER_HART 16U
#define DAYAN_SERVER_MAX_OUTPUT_BYTES (16U * 1024U * 1024U)
#define DAYAN_SERVER_MAX_PROFILE_BYTES (64U * 1024U * 1024U)
#define DAYAN_SERVER_SHUTDOWN_RPC_TIMEOUT_MS 1000U

struct server_connection {
    dayan_server_t *server;
    int fd;
};

struct server_shutdown_kill {
    npu_service_device_t *device;
    npu_hal_core_t target;
    pthread_t thread;
    npu_service_kill_result_t result;
    int status;
    uint32_t started;
};

static void server_log(const char *category, const char *format, ...)
{
    va_list arguments;

    flockfile(stderr);
    fprintf(stderr, "dayan_server: [%s] ", category);
    va_start(arguments, format);
    vfprintf(stderr, format, arguments);
    va_end(arguments);
    fputc('\n', stderr);
    fflush(stderr);
    funlockfile(stderr);
}

static const char *server_status_string(int status)
{
    return strerror(status < 0 ? -status : status);
}

static const char *connection_role_name(uint32_t role)
{
    return role == NPU_IPC_ROLE_CONTROL ? "control" :
           role == NPU_IPC_ROLE_DATA ? "data" : "unknown";
}

static void *shutdown_kill_main(void *argument)
{
    struct server_shutdown_kill *request = argument;

    request->status = npu_service_rpc_kill_all_timeout(
        request->device, request->target,
        DAYAN_SERVER_SHUTDOWN_RPC_TIMEOUT_MS, &request->result);
    return 0;
}

static void shutdown_device_processes(dayan_server_t *server)
{
    struct server_shutdown_kill *requests;
    size_t request_count = 0;
    size_t successful = 0;
    size_t unavailable = 0;
    size_t failed = 0;
    size_t index;

    for (index = 0; index < server->slot_count; ++index) {
        if (server->slots[index].enabled)
            ++request_count;
    }
    if (!request_count)
        return;
    requests = calloc(request_count, sizeof(*requests));
    if (!requests) {
        server_log("shutdown", "cannot allocate kill_all requests; "
                   "forcing Host-side cleanup");
        return;
    }
    request_count = 0;
    for (index = 0; index < server->slot_count; ++index) {
        struct server_shutdown_kill *request;
        int status;

        if (!server->slots[index].enabled)
            continue;
        request = &requests[request_count++];
        request->device = server->device;
        request->target = server->slots[index].target;
        request->status = -ECANCELED;
        status = pthread_create(&request->thread, 0, shutdown_kill_main,
                                request);
        if (status)
            request->status = -status;
        else
            request->started = 1;
    }
    for (index = 0; index < request_count; ++index) {
        struct server_shutdown_kill *request = &requests[index];

        if (request->started)
            (void)pthread_join(request->thread, 0);
        if (request->status == 0) {
            ++successful;
        } else if (request->status == -ENODEV) {
            ++unavailable;
        } else {
            ++failed;
            server_log("shutdown",
                       "kill_all Cluster %u Hart %u failed: %s (%d)",
                       request->target.cluster, request->target.core,
                       server_status_string(request->status),
                       request->status);
        }
    }
    server_log("shutdown",
               "kill_all finished: accepted=%zu unavailable=%zu failed=%zu",
               successful, unavailable, failed);
    free(requests);
}

static uint32_t all_mask(uint32_t count)
{
    return count == 32U ? UINT32_MAX : (UINT32_C(1) << count) - 1U;
}

static int task_name_valid(const char name[NPU_IPC_PROC_NAME_SIZE])
{
    _Static_assert(NPU_PROC_NAME_SIZE == NPU_IPC_PROC_NAME_SIZE,
                   "Runtime and IPC proc name sizes");
    return name && name[0] &&
           strnlen(name, NPU_IPC_PROC_NAME_SIZE) < NPU_IPC_PROC_NAME_SIZE;
}

static size_t output_charge(const struct server_output *output)
{
    return sizeof(*output) + output->size;
}

static struct server_output *output_create(
    uint32_t cluster, uint32_t core, uint32_t proc,
    const void *data, size_t size, uint32_t flags)
{
    struct server_output *output;

    if ((size && !data) || size > NPU_IPC_MAX_PAYLOAD - sizeof(*output) ||
        (flags & ~NPU_IPC_OUTPUT_CLEAR))
        return 0;
    output = malloc(sizeof(*output) + size);
    if (!output)
        return 0;
    output->next = 0;
    output->size = size;
    output->offset = 0;
    output->cluster = cluster;
    output->core = core;
    output->proc = proc;
    output->flags = flags;
    if (size)
        memcpy(output->data, data, size);
    return output;
}

static void output_list_free(struct server_output *output)
{
    while (output) {
        struct server_output *next = output->next;

        free(output);
        output = next;
    }
}

static size_t profile_chunk_charge(const struct server_profile_chunk *chunk)
{
    return sizeof(*chunk) + chunk->size;
}

static void profile_chunk_list_free(struct server_profile_chunk *chunk)
{
    while (chunk) {
        struct server_profile_chunk *next = chunk->next;

        free(chunk);
        chunk = next;
    }
}

static void profile_clear_locked(dayan_server_t *server)
{
    profile_chunk_list_free(server->profile_head);
    server->profile_head = 0;
    server->profile_tail = 0;
    server->profile_bytes = 0;
    server->profile_owner = 0;
    server->profile_id = 0;
    server->profile_active = 0;
    server->profile_stopping = 0;
    server->profile_stopped = 0;
}

static int session_append_output_locked(struct server_session *session,
                                        struct server_output *output)
{
    size_t charge = output_charge(output);

    if (session->closing ||
        charge > DAYAN_SERVER_MAX_OUTPUT_BYTES - session->output_bytes)
        return -ENOSPC;
    if (session->output_tail)
        session->output_tail->next = output;
    else
        session->output_head = output;
    session->output_tail = output;
    session->output_bytes += charge;
    return 0;
}

int server_queue_output_locked(dayan_server_t *server,
                               struct server_session *session,
                               uint32_t cluster, uint32_t core,
                               uint32_t proc, const void *data, size_t size,
                               uint32_t flags)
{
    struct server_output *output;
    int status;

    if (!server || !session)
        return -EINVAL;
    output = output_create(cluster, core, proc, data, size, flags);
    if (!output)
        return -ENOMEM;
    status = session_append_output_locked(session, output);
    if (status < 0)
        free(output);
    return status;
}

static struct server_session *output_session_locked(
    dayan_server_t *server, uint32_t cluster, uint32_t core, uint32_t proc)
{
    struct server_session *session;

    for (session = server->sessions; session; session = session->next) {
        struct server_context *context;

        for (context = session->contexts; context; context = context->next) {
            struct server_stream *stream;

            for (stream = context->streams; stream; stream = stream->next) {
                struct server_command *command;

                for (command = stream->head; command;
                     command = command->next) {
                    struct server_invocation *invocation;

                    if (command->kind != SERVER_COMMAND_LAUNCH)
                        continue;
                    for (invocation = command->data.launch.invocations;
                         invocation; invocation = invocation->next) {
                        npu_hal_core_t target =
                            server->slots[invocation->slot_index].target;

                        if (invocation->proc_id == proc &&
                            target.cluster == cluster &&
                            target.core == core)
                            return session;
                    }
                }
            }
        }
    }
    return 0;
}

void server_route_pending_outputs_locked(dayan_server_t *server)
{
    struct server_output **link = &server->pending_outputs;
    struct server_output *tail = 0;

    while (*link) {
        struct server_output *output = *link;
        struct server_session *session = output_session_locked(
            server, output->cluster, output->core, output->proc);

        if (!session) {
            link = &output->next;
            continue;
        }
        *link = output->next;
        server->pending_output_bytes -= output_charge(output);
        output->next = 0;
        if (session_append_output_locked(session, output) < 0)
            free(output);
    }
    for (tail = server->pending_outputs; tail && tail->next;
         tail = tail->next)
        ;
    server->pending_output_tail = tail;
}

static void server_device_output(void *context, uint32_t cluster,
                                 uint32_t core, uint32_t proc,
                                 const void *data, size_t size,
                                 uint32_t flags)
{
    dayan_server_t *server = context;
    struct server_output *output;
    struct server_session *session;
    size_t charge;

    _Static_assert(NPU_SERVICE_OUTPUT_CLEAR == NPU_IPC_OUTPUT_CLEAR,
                   "Service and socket output flags");
    if (!server)
        return;
    output = output_create(cluster, core, proc, data, size, flags);
    if (!output)
        return;
    charge = output_charge(output);
    pthread_mutex_lock(&server->mutex);
    session = output_session_locked(server, cluster, core, proc);
    if (session) {
        if (session_append_output_locked(session, output) < 0)
            free(output);
    } else if (charge <= DAYAN_SERVER_MAX_OUTPUT_BYTES -
                         server->pending_output_bytes) {
        if (server->pending_output_tail)
            server->pending_output_tail->next = output;
        else
            server->pending_outputs = output;
        server->pending_output_tail = output;
        server->pending_output_bytes += charge;
    } else {
        free(output);
    }
    pthread_mutex_unlock(&server->mutex);
}

static void server_device_trap(void *context, uint32_t cluster,
                               uint32_t core,
                               const struct trap_record *record)
{
    dayan_server_t *server = context;

    if (server && server->service_trap) {
        server->service_trap(server->service_callback_context, cluster, core,
                             record);
    }
}

static void server_device_proc_exit(void *context, uint32_t cluster,
                                    uint32_t core, uint32_t proc,
                                    int64_t result, uint64_t cookie)
{
    dayan_server_t *server = context;

    if (!server)
        return;
    pthread_mutex_lock(&server->mutex);
    pthread_cond_broadcast(&server->condition);
    pthread_mutex_unlock(&server->mutex);
    if (server->service_proc_exit)
        server->service_proc_exit(server->service_callback_context,
                                  cluster, core, proc, result, cookie);
}

static int server_device_profile(void *context, uint32_t cluster,
                                 uint32_t core,
                                 const void *data, size_t size)
{
    dayan_server_t *server = context;
    struct server_profile_chunk *chunk;
    size_t charge;
    int status = 0;
    int callback_status = 0;

    if (!server || !data || !size || size > SIZE_MAX - sizeof(*chunk))
        return -EINVAL;
    chunk = malloc(sizeof(*chunk) + size);
    if (!chunk)
        return -ENOMEM;
    memset(chunk, 0, sizeof(*chunk));
    chunk->size = size;
    chunk->cluster = cluster;
    chunk->hart = core;
    memcpy(chunk->data, data, size);
    charge = profile_chunk_charge(chunk);

    pthread_mutex_lock(&server->mutex);
    if (!server->profile_owner || !server->profile_active ||
        server->profile_owner->closing) {
        status = -ECANCELED;
    } else if (charge > DAYAN_SERVER_MAX_PROFILE_BYTES -
                        server->profile_bytes) {
        status = -ENOSPC;
    } else {
        if (server->profile_tail)
            server->profile_tail->next = chunk;
        else
            server->profile_head = chunk;
        server->profile_tail = chunk;
        server->profile_bytes += charge;
        chunk = 0;
        pthread_cond_broadcast(&server->condition);
    }
    pthread_mutex_unlock(&server->mutex);
    free(chunk);
    if (server->service_profile)
        callback_status = server->service_profile(
            server->service_callback_context, cluster, core, data, size);
    return status < 0 ? status : callback_status;
}

void dayan_server_options_init(dayan_server_options_t *options)
{
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->version = DAYAN_SERVER_OPTIONS_VERSION;
    options->socket_path = NPU_DEFAULT_SERVER_SOCKET;
    options->resident_limit_per_core = DAYAN_SERVER_DEFAULT_RESIDENT;
    options->max_procs_per_hart = DAYAN_SERVER_DEFAULT_PROCS_PER_HART;
    npu_hal_options_init(&options->hal);
    npu_service_options_init(&options->service);
}

static int package_release_locked(dayan_server_t *server,
                                  struct server_package_cache *package)
{
    struct server_package_cache **link;

    if (!package || !package->references)
        return -EINVAL;
    if (--package->references)
        return 0;
    link = &server->packages;
    while (*link && *link != package)
        link = &(*link)->next;
    if (*link)
        *link = package->next;
    {
        int status = npu_service_kernel_forget(
            server->device, package->view.object,
            (size_t)package->view.header->object_size);

        if (status < 0 && status != -ENOENT) {
            ++package->references;
            package->next = server->packages;
            server->packages = package;
            return status;
        }
    }
    free(package->image);
    free(package);
    return 0;
}

static int context_collectable(const struct server_context *context)
{
    const struct server_stream *stream;
    const struct server_event *event;

    if (!context->closing || context->pending_commands)
        return 0;
    for (stream = context->streams; stream; stream = stream->next) {
        if (stream->head || stream->references)
            return 0;
    }
    for (event = context->events; event; event = event->next) {
        if (event->references)
            return 0;
    }
    return 1;
}

static void free_context_locked(dayan_server_t *server,
                                struct server_context *context)
{
    struct server_kernel *kernel = context->kernels;
    struct server_module *module = context->modules;
    struct server_buffer *buffer = context->buffers;
    struct server_stream *stream = context->streams;
    struct server_event *event = context->events;

    while (kernel) {
        struct server_kernel *next = kernel->next;

        if (kernel->module->kernel_handles)
            --kernel->module->kernel_handles;
        free(kernel->active_per_core);
        free(kernel);
        kernel = next;
    }
    while (module) {
        struct server_module *next = module->next;

        (void)package_release_locked(server, module->package);
        free(module);
        module = next;
    }
    while (buffer) {
        struct server_buffer *next = buffer->next;

        (void)npu_hal_memory_free(server->hal,
                                  buffer->allocation);
        free(buffer);
        buffer = next;
    }
    while (stream) {
        struct server_stream *next = stream->next;

        free(stream);
        stream = next;
    }
    while (event) {
        struct server_event *next = event->next;

        free(event);
        event = next;
    }
    free(context);
}

void server_collect_contexts_locked(dayan_server_t *server)
{
    struct server_session **session_link = &server->sessions;

    while (*session_link) {
        struct server_session *session = *session_link;
        struct server_context **context_link = &session->contexts;

        while (*context_link) {
            struct server_context *context = *context_link;

            if (!context_collectable(context)) {
                context_link = &context->next;
                continue;
            }
            *context_link = context->next;
            free_context_locked(server, context);
        }
        if (session->closing && !session->contexts &&
            server->profile_owner != session &&
            session->control_fd < 0 && session->data_fd < 0) {
            *session_link = session->next;
            output_list_free(session->output_head);
            free(session);
        } else {
            session_link = &session->next;
        }
    }
}

static void cancel_all_locked(dayan_server_t *server)
{
    struct server_session *session;

    for (session = server->sessions; session; session = session->next) {
        struct server_context *context;

        session->closing = 1;
        for (context = session->contexts; context; context = context->next)
            server_cancel_context_locked(server, context);
    }
}

static int initialize_slots(dayan_server_t *server)
{
    uint32_t cluster_mask = server->options.enabled_cluster_mask;
    uint32_t core_mask = server->options.enabled_core_mask;
    size_t index;

    if (!cluster_mask)
        cluster_mask = all_mask(server->hal_info.cluster_count);
    if (!core_mask)
        core_mask = all_mask(server->hal_info.cores_per_cluster);
    if ((cluster_mask & ~all_mask(server->hal_info.cluster_count)) ||
        (core_mask & ~all_mask(server->hal_info.cores_per_cluster)))
        return -EINVAL;
    server->options.enabled_cluster_mask = cluster_mask;
    server->options.enabled_core_mask = core_mask;
    server->slot_count = (size_t)server->hal_info.cluster_count *
                         server->hal_info.cores_per_cluster;
    server->slots = calloc(server->slot_count, sizeof(*server->slots));
    if (!server->slots)
        return -ENOMEM;
    for (index = 0; index < server->slot_count; ++index) {
        uint32_t cluster = (uint32_t)(
            index / server->hal_info.cores_per_cluster);
        uint32_t core = (uint32_t)(
            index % server->hal_info.cores_per_cluster);
        int status;

        server->slots[index].target.cluster = cluster;
        server->slots[index].target.core = core;
        status = npu_hal_memory_capacity(
            server->hal, NPU_HAL_MEMORY_COHERENT,
            server->slots[index].target,
            &server->slots[index].atomic_cache_capacity);
        if (status < 0)
            return status;
        status = npu_hal_memory_capacity(
            server->hal, NPU_HAL_MEMORY_URAM,
            server->slots[index].target,
            &server->slots[index].uram_capacity);
        if (status < 0)
            return status;
        server->slots[index].enabled =
            !!(cluster_mask & (UINT32_C(1) << cluster)) &&
            !!(core_mask & (UINT32_C(1) << core));
    }
    return 0;
}

int dayan_server_create(dayan_server_t **server_out,
                        const dayan_server_options_t *options)
{
    dayan_server_options_t defaults;
    dayan_server_t *server;
    uint32_t cluster;
    const char *stage = "validating options";
    int mutex_initialized = 0;
    int condition_initialized = 0;
    int status;

    if (!server_out)
        return -EINVAL;
    *server_out = 0;
    if (!options) {
        dayan_server_options_init(&defaults);
        options = &defaults;
    }
    if (options->version != DAYAN_SERVER_OPTIONS_VERSION ||
        !options->max_procs_per_hart ||
        options->max_procs_per_hart > NPU_PACKAGE_HART_MAX_PROCS ||
        !options->socket_path ||
        !*options->socket_path)
        return -EINVAL;
    server = calloc(1, sizeof(*server));
    if (!server)
        return -ENOMEM;
    server->listen_fd = -1;
    server->options = *options;
    server->service_trap = options->service.trap;
    server->service_proc_exit = options->service.proc_exit;
    server->service_profile = options->service.profile;
    server->service_callback_context = options->service.callback_context;
    server->options.service.output = server_device_output;
    server->options.service.trap = server_device_trap;
    server->options.service.proc_exit = server_device_proc_exit;
    server->options.service.profile = server_device_profile;
    server->options.service.callback_context = server;
    server_log("startup",
               "configuration: socket=%s backend=%u device_prefix=%s "
               "device_index=%u requested_clusters=%u cluster_stride=%u "
               "normal_memory_mib=%u",
               options->socket_path, options->hal.backend,
               options->hal.device_prefix ? options->hal.device_prefix :
                                            "<default>",
               options->hal.device_index, options->hal.cluster_count,
               options->hal.cluster_device_stride,
               options->hal.normal_memory_size_mib);
    server_log("startup",
               "requested masks: clusters=0x%08x cores=0x%08x; "
               "firmware=%s; api_elf=%s",
               options->enabled_cluster_mask, options->enabled_core_mask,
               options->firmware_path ? options->firmware_path : "<none>",
               options->api_elf_path ? options->api_elf_path : "<none>");
    stage = "copying the socket path";
    server->socket_path = strdup(options->socket_path);
    if (!server->socket_path) {
        status = -ENOMEM;
        goto fail;
    }
    server->options.socket_path = server->socket_path;
    stage = "initializing the server mutex";
    status = pthread_mutex_init(&server->mutex, 0);
    if (status) {
        status = -status;
        goto fail;
    }
    mutex_initialized = 1;
    stage = "initializing the server condition variable";
    status = pthread_cond_init(&server->condition, 0);
    if (status) {
        status = -status;
        goto fail;
    }
    condition_initialized = 1;
    stage = "opening the HAL";
    server_log("startup", "opening HAL");
    status = npu_hal_open(&server->hal, &options->hal);
    if (status < 0)
        goto fail;
    server_log("startup", "HAL opened");
    stage = "starting the Device RPC service";
    server_log("startup", "starting Device RPC service");
    status = npu_service_open(&server->device, server->hal,
                              &server->options.service);
    if (status < 0)
        goto fail;
    server_log("startup", "Device RPC service started");
    stage = "reading HAL topology";
    server_log("startup", "reading HAL topology");
    status = npu_hal_get_info(server->hal, &server->hal_info);
    if (status < 0)
        goto fail;
    server_log("startup",
               "topology: clusters=%u cores_per_cluster=%u "
               "memory_regions=%u max_transfer=%llu bytes",
               server->hal_info.cluster_count,
               server->hal_info.cores_per_cluster,
               server->hal_info.memory_region_count,
               (unsigned long long)server->hal_info.max_transfer_size);
    /* NPUTaskMode::kFourCore is defined by the current 4-core Cluster ABI. */
    if (server->hal_info.cores_per_cluster != 4U) {
        status = -ENOTSUP;
        goto fail;
    }
    stage = "initializing Cluster/Core slots";
    server_log("startup", "initializing Cluster/Core resource slots");
    status = initialize_slots(server);
    if (status < 0)
        goto fail;
    server_log("startup",
               "resource slots ready: total=%zu enabled_clusters=0x%08x "
               "enabled_cores=0x%08x max_procs_per_hart=%u",
               server->slot_count, server->options.enabled_cluster_mask,
               server->options.enabled_core_mask,
               server->options.max_procs_per_hart);
    if (options->api_elf_path) {
        stage = "loading the Device API ELF";
        server_log("startup", "loading Device API ELF: %s",
                   options->api_elf_path);
        status = npu_service_set_api_elf_file(server->device,
                                               options->api_elf_path);
        if (status < 0)
            goto fail;
        server_log("startup", "Device API ELF registered");
    }
    if (options->firmware_path) {
        for (cluster = 0; cluster < server->hal_info.cluster_count;
             ++cluster) {
            if (!(server->options.enabled_cluster_mask &
                  (UINT32_C(1) << cluster)))
                continue;
            stage = "loading BL1 firmware";
            server_log("startup",
                       "Cluster %u: loading BL1 firmware on core mask "
                       "0x%08x",
                       cluster, server->options.enabled_core_mask);
            status = npu_service_load_bl1_file(
                server->device, cluster, options->firmware_path,
                server->options.enabled_core_mask);
            if (status < 0)
                goto fail;
            server_log("startup", "Cluster %u: BL1 is ready", cluster);
        }
    }
    stage = "creating the Runtime Unix socket";
    server_log("startup", "creating Runtime socket: %s",
               server->socket_path);
    server->listen_fd = npu_ipc_listen(server->socket_path,
                                       DAYAN_SERVER_BACKLOG);
    if (server->listen_fd < 0) {
        status = server->listen_fd;
        server->listen_fd = -1;
        goto fail;
    }
    server_log("startup", "Runtime socket is listening");
    stage = "starting the scheduler";
    server_log("startup", "starting command scheduler");
    status = pthread_create(&server->scheduler, 0, dayan_scheduler_main,
                            server);
    if (status) {
        status = -status;
        goto fail;
    }
    server->scheduler_started = 1;
    server_log("startup", "command scheduler started");
    *server_out = server;
    return 0;

fail:
    server_log("error", "startup failed while %s: %s (%d)", stage,
               server_status_string(status), status);
    if (server->listen_fd >= 0)
        close(server->listen_fd);
    if (server->socket_path)
        (void)unlink(server->socket_path);
    free(server->slots);
    if (server->device)
        (void)npu_service_close(server->device);
    npu_hal_close(server->hal);
    if (condition_initialized)
        pthread_cond_destroy(&server->condition);
    if (mutex_initialized)
        pthread_mutex_destroy(&server->mutex);
    free(server->socket_path);
    free(server);
    return status;
}

static struct server_context *find_context(struct server_session *session,
                                           uint64_t id)
{
    struct server_context *context;

    for (context = session->contexts; context; context = context->next) {
        if (context->id == id)
            return context;
    }
    return 0;
}

static struct server_buffer *find_buffer(struct server_session *session,
                                         uint64_t id)
{
    struct server_context *context;

    for (context = session->contexts; context; context = context->next) {
        struct server_buffer *buffer;

        for (buffer = context->buffers; buffer; buffer = buffer->next) {
            if (buffer->id == id)
                return buffer;
        }
    }
    return 0;
}

static struct server_stream *find_stream(struct server_session *session,
                                         uint64_t id)
{
    struct server_context *context;

    for (context = session->contexts; context; context = context->next) {
        struct server_stream *stream;

        for (stream = context->streams; stream; stream = stream->next) {
            if (stream->id == id)
                return stream;
        }
    }
    return 0;
}

static struct server_module *find_module(struct server_session *session,
                                         uint64_t id)
{
    struct server_context *context;

    for (context = session->contexts; context; context = context->next) {
        struct server_module *module;

        for (module = context->modules; module; module = module->next) {
            if (module->id == id)
                return module;
        }
    }
    return 0;
}

static struct server_kernel *find_kernel(struct server_session *session,
                                         uint64_t id)
{
    struct server_context *context;

    for (context = session->contexts; context; context = context->next) {
        struct server_kernel *kernel;

        for (kernel = context->kernels; kernel; kernel = kernel->next) {
            if (kernel->id == id)
                return kernel;
        }
    }
    return 0;
}

static struct server_event *find_event(struct server_session *session,
                                       uint64_t id)
{
    struct server_context *context;

    for (context = session->contexts; context; context = context->next) {
        struct server_event *event;

        for (event = context->events; event; event = event->next) {
            if (event->id == id)
                return event;
        }
    }
    return 0;
}

static int next_id_locked(dayan_server_t *server, uint64_t *id_out)
{
    if (server->next_id == UINT64_MAX)
        return -EOVERFLOW;
    *id_out = ++server->next_id;
    return 0;
}

static int memory_request_for_buffer(dayan_server_t *server,
                                     struct server_context *context,
                                     const npu_ipc_buffer_create_t *request,
                                     npu_hal_allocation_t **allocation_out)
{
    npu_hal_memory_request_t memory;
    npu_hal_memory_scope_t scopes[2];
    uint32_t start_cluster;
    size_t pass;
    int status = -ENOMEM;

    npu_hal_memory_request_init(&memory);
    memory.memory_class = (request->flags & NPU_BUFFER_COHERENT) ?
        NPU_HAL_MEMORY_COHERENT : NPU_HAL_MEMORY_NORMAL;
    memory.usage = NPU_HAL_MEMORY_BUFFER;
    memory.owner = context->owner;
    memory.size = request->size;
    memory.alignment = request->alignment;
    if (request->flags & NPU_BUFFER_DEVICE_LOCAL) {
        scopes[0] = NPU_HAL_MEMORY_CLUSTER;
        scopes[1] = NPU_HAL_MEMORY_DEVICE;
    } else {
        scopes[0] = NPU_HAL_MEMORY_DEVICE;
        scopes[1] = NPU_HAL_MEMORY_CLUSTER;
    }
    start_cluster = (uint32_t)(server->next_slot /
                               server->hal_info.cores_per_cluster);
    for (pass = 0; pass < 2; ++pass) {
        if (scopes[pass] == NPU_HAL_MEMORY_DEVICE) {
            memory.scope = NPU_HAL_MEMORY_DEVICE;
            memory.location.cluster = NPU_HAL_ANY_CLUSTER;
            memory.location.core = NPU_HAL_ANY_CORE;
            status = npu_hal_memory_alloc(server->hal,
                                          &memory, allocation_out);
            if (status == 0 || status != -ENOMEM)
                return status;
        } else {
            uint32_t offset;

            for (offset = 0; offset < server->hal_info.cluster_count;
                 ++offset) {
                uint32_t cluster = (start_cluster + offset) %
                                   server->hal_info.cluster_count;

                if (!(server->options.enabled_cluster_mask &
                      (UINT32_C(1) << cluster)))
                    continue;
                memory.scope = NPU_HAL_MEMORY_CLUSTER;
                memory.location.cluster = cluster;
                memory.location.core = NPU_HAL_ANY_CORE;
                status = npu_hal_memory_alloc(
                    server->hal, &memory,
                    allocation_out);
                if (status == 0) {
                    server->next_slot = (size_t)(cluster + 1U) %
                        server->hal_info.cluster_count *
                        server->hal_info.cores_per_cluster;
                    return 0;
                }
                if (status != -ENOMEM)
                    return status;
            }
        }
    }
    return status;
}

static int context_create_locked(dayan_server_t *server,
                                 struct server_session *session,
                                 const npu_ipc_context_create_t *request,
                                 uint64_t *id_out)
{
    struct server_context *context;
    int status;

    if (request->flags || session->closing)
        return -EINVAL;
    context = calloc(1, sizeof(*context));
    if (!context)
        return -ENOMEM;
    status = next_id_locked(server, &context->id);
    if (status < 0) {
        free(context);
        return status;
    }
    if (!++server->next_owner) {
        free(context);
        return -EOVERFLOW;
    }
    context->owner = server->next_owner;
    context->memory_quota = request->memory_quota;
    context->max_pending_commands = request->max_pending_commands;
    context->session = session;
    context->next = session->contexts;
    session->contexts = context;
    *id_out = context->id;
    return 0;
}

static int buffer_create_locked(dayan_server_t *server,
                                struct server_session *session,
                                const npu_ipc_buffer_create_t *request,
                                uint64_t *id_out)
{
    const uint32_t valid_flags = NPU_BUFFER_DEVICE_LOCAL |
        NPU_BUFFER_COHERENT | NPU_BUFFER_READ | NPU_BUFFER_WRITE;
    struct server_context *context = find_context(session,
                                                   request->context_id);
    struct server_buffer *buffer;
    int status;

    if (!context || context->closing || !request->size ||
        !request->alignment ||
        (request->alignment & (request->alignment - 1U)) ||
        (request->flags & ~valid_flags) ||
        !(request->flags & (NPU_BUFFER_READ | NPU_BUFFER_WRITE)) ||
        request->reserved ||
        request->size > UINT64_MAX - context->memory_used ||
        (context->memory_quota &&
         request->size > context->memory_quota - context->memory_used))
        return -EINVAL;
    buffer = calloc(1, sizeof(*buffer));
    if (!buffer)
        return -ENOMEM;
    status = memory_request_for_buffer(server, context, request,
                                       &buffer->allocation);
    if (status < 0) {
        free(buffer);
        return status;
    }
    status = next_id_locked(server, &buffer->id);
    if (status < 0) {
        (void)npu_hal_memory_free(server->hal,
                                  buffer->allocation);
        free(buffer);
        return status;
    }
    buffer->size = request->size;
    buffer->flags = request->flags;
    buffer->context = context;
    buffer->next = context->buffers;
    context->buffers = buffer;
    context->memory_used += request->size;
    *id_out = buffer->id;
    return 0;
}

static int buffer_destroy_locked(dayan_server_t *server,
                                 struct server_session *session,
                                 uint64_t id)
{
    struct server_buffer *buffer = find_buffer(session, id);
    struct server_buffer **link;
    int status;

    if (!buffer)
        return -ENOENT;
    if (buffer->references)
        return -EBUSY;
    status = npu_hal_memory_free(server->hal,
                                 buffer->allocation);
    if (status < 0)
        return status;
    link = &buffer->context->buffers;
    while (*link && *link != buffer)
        link = &(*link)->next;
    if (*link)
        *link = buffer->next;
    if (buffer->context->memory_used >= buffer->size)
        buffer->context->memory_used -= buffer->size;
    free(buffer);
    return 0;
}

static int stream_create_locked(dayan_server_t *server,
                                struct server_session *session,
                                const npu_ipc_stream_create_t *request,
                                uint64_t *id_out)
{
    struct server_context *context = find_context(session,
                                                   request->context_id);
    struct server_stream *stream;
    int status;

    if (!context || context->closing || request->flags)
        return -EINVAL;
    stream = calloc(1, sizeof(*stream));
    if (!stream)
        return -ENOMEM;
    status = next_id_locked(server, &stream->id);
    if (status < 0) {
        free(stream);
        return status;
    }
    stream->context = context;
    stream->next = context->streams;
    context->streams = stream;
    *id_out = stream->id;
    return 0;
}

static int stream_destroy_locked(struct server_session *session, uint64_t id)
{
    struct server_stream *stream = find_stream(session, id);
    struct server_stream **link;

    if (!stream)
        return -ENOENT;
    if (stream->head || stream->references)
        return -EBUSY;
    link = &stream->context->streams;
    while (*link && *link != stream)
        link = &(*link)->next;
    if (*link)
        *link = stream->next;
    free(stream);
    return 0;
}

static int transfer_submit_locked(dayan_server_t *server,
                                  struct server_session *session,
                                  const npu_ipc_transfer_submit_t *request,
                                  uint64_t *event_id_out)
{
    struct server_context *context = find_context(session,
                                                   request->context_id);
    struct server_stream *stream = find_stream(session, request->stream_id);
    struct server_buffer *destination = request->destination_buffer ?
        find_buffer(session, request->destination_buffer) : 0;
    struct server_buffer *source = request->source_buffer ?
        find_buffer(session, request->source_buffer) : 0;
    struct server_command *command;
    struct server_event *event;
    int status;

    if (!context || !stream || stream->context != context ||
        context->closing || !request->size || request->size > SIZE_MAX ||
        request->reserved)
        return -EINVAL;
    switch (request->direction) {
    case NPU_IPC_TRANSFER_H2D:
        if (!destination || source || destination->context != context ||
            !(destination->flags & NPU_BUFFER_WRITE) ||
            request->destination_offset > destination->size ||
            request->size > destination->size - request->destination_offset)
            return -EINVAL;
        break;
    case NPU_IPC_TRANSFER_D2H:
        if (!source || destination || source->context != context ||
            !(source->flags & NPU_BUFFER_READ) ||
            request->source_offset > source->size ||
            request->size > source->size - request->source_offset)
            return -EINVAL;
        break;
    case NPU_IPC_TRANSFER_D2D:
        if (!source || !destination || source->context != context ||
            destination->context != context ||
            !(source->flags & NPU_BUFFER_READ) ||
            !(destination->flags & NPU_BUFFER_WRITE) ||
            request->source_offset > source->size ||
            request->size > source->size - request->source_offset ||
            request->destination_offset > destination->size ||
            request->size > destination->size -
                                request->destination_offset)
            return -EINVAL;
        break;
    default:
        return -EINVAL;
    }
    command = calloc(1, sizeof(*command));
    if (!command)
        return -ENOMEM;
    command->kind = request->direction == NPU_IPC_TRANSFER_H2D ?
        SERVER_COMMAND_H2D : request->direction == NPU_IPC_TRANSFER_D2H ?
        SERVER_COMMAND_D2H : SERVER_COMMAND_D2D;
    command->data.transfer.destination = destination;
    command->data.transfer.source = source;
    command->data.transfer.destination_offset = request->destination_offset;
    command->data.transfer.source_offset = request->source_offset;
    command->data.transfer.size = request->size;
    if (destination)
        ++destination->references;
    if (source)
        ++source->references;
    status = server_enqueue_locked(server, context, stream, command, 1,
                                   &event);
    if (status < 0) {
        if (destination)
            --destination->references;
        if (source)
            --source->references;
        free(command);
        return status;
    }
    command->data.transfer.transfer_id = event->id;
    *event_id_out = event->id;
    return 0;
}

static int event_record_locked(dayan_server_t *server,
                               struct server_session *session,
                               uint64_t stream_id, uint64_t *event_id_out)
{
    struct server_stream *stream = find_stream(session, stream_id);
    struct server_command *command;
    struct server_event *event;
    int status;

    if (!stream || stream->context->closing)
        return -EINVAL;
    command = calloc(1, sizeof(*command));
    if (!command)
        return -ENOMEM;
    command->kind = SERVER_COMMAND_EVENT;
    status = server_enqueue_locked(server, stream->context, stream, command,
                                   1, &event);
    if (status < 0) {
        free(command);
        return status;
    }
    *event_id_out = event->id;
    return 0;
}

static int stream_wait_event_locked(dayan_server_t *server,
                                    struct server_session *session,
                                    const npu_ipc_stream_wait_t *request)
{
    struct server_stream *stream = find_stream(session, request->stream_id);
    struct server_event *dependency = find_event(session, request->event_id);
    struct server_command *command;
    struct server_event *internal_event;
    int status;

    if (!stream || !dependency || request->reserved ||
        stream->context != dependency->context || stream->context->closing)
        return -EINVAL;
    command = calloc(1, sizeof(*command));
    if (!command)
        return -ENOMEM;
    command->kind = SERVER_COMMAND_WAIT;
    command->data.wait.dependency = dependency;
    ++dependency->references;
    status = server_enqueue_locked(server, stream->context, stream, command,
                                   0, &internal_event);
    if (status < 0) {
        --dependency->references;
        free(command);
    }
    return status;
}

static int package_get_locked(dayan_server_t *server,
                              const void *image, size_t size,
                              struct server_package_cache **package_out)
{
    struct server_package_cache *package;
    struct npu_package_view view;
    unsigned char *copy;
    uint64_t hash;
    int immutable;
    int status;

    copy = malloc(size);
    if (!copy)
        return -ENOMEM;
    memcpy(copy, image, size);
    status = npu_package_view_init(&view, copy, size);
    if (status < 0) {
        free(copy);
        return status;
    }
    hash = npu_package_hash(copy, size);
    immutable = !!(view.header->flags & NPU_PACKAGE_IMMUTABLE_MODULE);
    if (immutable) {
        for (package = server->packages; package; package = package->next) {
            if (package->immutable && package->hash == hash &&
                package->size == size && memcmp(package->image, copy, size) == 0) {
                ++package->references;
                free(copy);
                *package_out = package;
                return 0;
            }
        }
    }
    package = calloc(1, sizeof(*package));
    if (!package) {
        free(copy);
        return -ENOMEM;
    }
    package->image = copy;
    package->size = size;
    package->hash = hash;
    package->references = 1;
    package->immutable = (uint32_t)immutable;
    status = npu_package_view_init(&package->view, copy, size);
    if (status < 0) {
        free(copy);
        free(package);
        return status;
    }
    package->next = server->packages;
    server->packages = package;
    *package_out = package;
    return 0;
}

static int module_load_locked(dayan_server_t *server,
                              struct server_session *session,
                              const npu_ipc_module_load_t *request,
                              const unsigned char *image, size_t image_size,
                              uint64_t *id_out)
{
    struct server_context *context = find_context(session,
                                                   request->context_id);
    struct server_module *module;
    int status;

    if (!context || context->closing || request->package_size != image_size ||
        !image_size)
        return -EINVAL;
    module = calloc(1, sizeof(*module));
    if (!module)
        return -ENOMEM;
    status = package_get_locked(server, image, image_size, &module->package);
    if (status < 0) {
        free(module);
        return status;
    }
    status = next_id_locked(server, &module->id);
    if (status < 0) {
        (void)package_release_locked(server, module->package);
        free(module);
        return status;
    }
    module->context = context;
    module->next = context->modules;
    context->modules = module;
    *id_out = module->id;
    return 0;
}

static int module_unload_locked(dayan_server_t *server,
                                struct server_session *session, uint64_t id)
{
    struct server_module *module = find_module(session, id);
    struct server_module **link;
    int status;

    if (!module)
        return -ENOENT;
    if (module->kernel_handles || module->active_commands)
        return -EBUSY;
    status = package_release_locked(server, module->package);
    if (status < 0)
        return status;
    link = &module->context->modules;
    while (*link && *link != module)
        link = &(*link)->next;
    if (*link)
        *link = module->next;
    free(module);
    return 0;
}

static int kernel_get_locked(dayan_server_t *server,
                             struct server_session *session,
                             const npu_ipc_kernel_get_t *request,
                             const char *name, uint64_t *id_out)
{
    struct server_module *module = find_module(session, request->module_id);
    const npu_package_kernel_t *metadata;
    struct server_kernel *kernel;
    int status;

    if (!module || module->context->closing || !request->name_size ||
        request->reserved || name[request->name_size - 1U] != '\0' ||
        !memchr(name, '\0', request->name_size))
        return -EINVAL;
    metadata = npu_package_find_kernel(&module->package->view, name);
    if (!metadata)
        return -ENOENT;
    kernel = calloc(1, sizeof(*kernel));
    if (!kernel)
        return -ENOMEM;
    kernel->active_per_core = calloc(server->slot_count,
                                     sizeof(*kernel->active_per_core));
    if (!kernel->active_per_core) {
        free(kernel);
        return -ENOMEM;
    }
    status = next_id_locked(server, &kernel->id);
    if (status < 0) {
        free(kernel->active_per_core);
        free(kernel);
        return status;
    }
    kernel->module = module;
    kernel->metadata = metadata;
    kernel->next = module->context->kernels;
    module->context->kernels = kernel;
    ++module->kernel_handles;
    *id_out = kernel->id;
    return 0;
}

static int kernel_release_locked(struct server_session *session, uint64_t id)
{
    struct server_kernel *kernel = find_kernel(session, id);
    struct server_kernel **link;

    if (!kernel)
        return -ENOENT;
    if (kernel->references)
        return -EBUSY;
    link = &kernel->module->context->kernels;
    while (*link && *link != kernel)
        link = &(*link)->next;
    if (*link)
        *link = kernel->next;
    if (kernel->module->kernel_handles)
        --kernel->module->kernel_handles;
    free(kernel->active_per_core);
    free(kernel);
    return 0;
}

static uint32_t package_mode_bit(uint32_t mode)
{
    return mode == 1U ? NPU_PACKAGE_TASK_MODE_SINGLE_CORE :
           mode == 4U ? NPU_PACKAGE_TASK_MODE_FOUR_CORE : 0;
}

static int launch_total_units(const npu_ipc_kernel_launch_t *request,
                              uint64_t *total_out)
{
    uint64_t total;

    if (!request->task_x || !request->task_y || !request->task_z ||
        (request->task_mode != 1U && request->task_mode != 4U) ||
        (request->task_mode == 4U && request->task_x != 4U))
        return -EINVAL;
    total = request->task_mode == 4U ? request->task_y : request->task_x;
    if (request->task_mode == 1U) {
        if (request->task_y > UINT64_MAX / total)
            return -EOVERFLOW;
        total *= request->task_y;
    }
    if (request->task_z > UINT64_MAX / total)
        return -EOVERFLOW;
    *total_out = total * request->task_z;
    return 0;
}

static int kernel_launch_locked(dayan_server_t *server,
                                struct server_session *session,
                                const npu_ipc_kernel_launch_t *request,
                                uint64_t *event_id_out)
{
    struct server_context *context = find_context(session,
                                                   request->context_id);
    struct server_stream *stream = find_stream(session, request->stream_id);
    struct server_kernel *kernel = find_kernel(session, request->kernel_id);
    const npu_package_kernel_t *metadata;
    const struct npu_package_view *view;
    struct server_command *command;
    struct server_event *event;
    uint64_t total_units;
    size_t index;
    int status;

    if (!context || !stream || !kernel || context->closing ||
        stream->context != context || kernel->module->context != context ||
        request->flags || request->reserved ||
        !task_name_valid(request->task_name) ||
        request->argument_count > NPU_KERNEL_MAX_ARGS)
        return -EINVAL;
    status = launch_total_units(request, &total_units);
    if (status < 0)
        return status;
    metadata = kernel->metadata;
    view = &kernel->module->package->view;
    if (!(metadata->supported_task_modes &
          package_mode_bit(request->task_mode)) ||
        request->argument_count != metadata->argument_count)
        return -EINVAL;
    if (request->task_mode == 4U &&
        (server->options.enabled_core_mask & UINT32_C(0xf)) !=
            UINT32_C(0xf))
        return -ENODEV;
    command = calloc(1, sizeof(*command));
    if (!command)
        return -ENOMEM;
    command->kind = SERVER_COMMAND_LAUNCH;
    command->data.launch.kernel = kernel;
    command->data.launch.argument_count = request->argument_count;
    command->data.launch.config.task_dim.x = request->task_x;
    command->data.launch.config.task_dim.y = request->task_y;
    command->data.launch.config.task_dim.z = request->task_z;
    command->data.launch.config.task_mode = (NPUTaskMode)request->task_mode;
    command->data.launch.total_units = total_units;
    command->data.launch.cookie = request->cookie;
    memcpy(command->data.launch.task_name, request->task_name,
           sizeof(command->data.launch.task_name));
    for (index = 0; index < request->argument_count; ++index) {
        const npu_package_argument_t *expected =
            &view->arguments[metadata->first_argument + index];
        const npu_ipc_launch_argument_t *argument = &request->arguments[index];

        if (argument->kind != expected->kind ||
            argument->size != expected->size ||
            argument->access != expected->access || argument->reserved) {
            status = -EINVAL;
            goto fail;
        }
        command->data.launch.arguments[index] = *argument;
        if (argument->kind == NPU_KERNEL_ARG_BUFFER) {
            struct server_buffer *buffer = find_buffer(
                session, argument->value_or_buffer);
            npu_hal_allocation_info_t allocation_info;

            if (!buffer || buffer->context != context ||
                argument->offset >= buffer->size ||
                ((argument->access & NPU_KERNEL_ARG_READ) &&
                 !(buffer->flags & NPU_BUFFER_READ)) ||
                ((argument->access & NPU_KERNEL_ARG_WRITE) &&
                 !(buffer->flags & NPU_BUFFER_WRITE))) {
                status = -EINVAL;
                goto fail;
            }
            status = npu_hal_memory_info(server->hal,
                                         buffer->allocation,
                                         &allocation_info);
            if (status < 0 ||
                allocation_info.address > UINT64_MAX - argument->offset ||
                ((allocation_info.address + argument->offset) &
                 (expected->alignment - 1U))) {
                status = status < 0 ? status : -EINVAL;
                goto fail;
            }
            command->data.launch.buffers[index] = buffer;
            ++buffer->references;
        }
    }
    status = server_launch_resources_valid(server, command);
    if (status < 0)
        goto fail;
    ++kernel->references;
    ++kernel->module->active_commands;
    status = server_enqueue_locked(server, context, stream, command, 1,
                                   &event);
    if (status < 0) {
        --kernel->references;
        --kernel->module->active_commands;
        goto fail;
    }
    event->cookie = request->cookie;
    *event_id_out = event->id;
    return 0;

fail:
    for (index = 0; index < request->argument_count; ++index) {
        if (command->data.launch.buffers[index] &&
            command->data.launch.buffers[index]->references)
            --command->data.launch.buffers[index]->references;
    }
    free(command);
    return status;
}

static void fill_event_info(const struct server_event *event,
                            npu_ipc_event_info_t *info)
{
    info->state = event->state;
    info->command_type = event->command_type;
    info->status = event->status;
    info->sequence = event->sequence;
    info->cookie = event->cookie;
}

static int wait_with_timeout(dayan_server_t *server, uint32_t timeout_ms,
                             int (*ready)(void *), void *object)
{
    struct timespec deadline;
    int status;

    if (timeout_ms == NPU_TIMEOUT_INFINITE) {
        while (!ready(object) && !server->stop)
            (void)pthread_cond_wait(&server->condition, &server->mutex);
        return ready(object) ? 0 : -ECANCELED;
    }
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0)
        return -EIO;
    deadline.tv_sec += timeout_ms / 1000U;
    deadline.tv_nsec += (long)(timeout_ms % 1000U) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        ++deadline.tv_sec;
        deadline.tv_nsec -= 1000000000L;
    }
    while (!ready(object) && !server->stop) {
        status = pthread_cond_timedwait(&server->condition, &server->mutex,
                                        &deadline);
        if (status == ETIMEDOUT)
            return -ETIMEDOUT;
        if (status)
            return -status;
    }
    return ready(object) ? 0 : -ECANCELED;
}

static int stream_ready(void *object)
{
    return ((struct server_stream *)object)->head == 0;
}

static int event_ready(void *object)
{
    struct server_event *event = object;

    return event->state == NPU_EVENT_COMPLETE ||
           event->state == NPU_EVENT_FAILED ||
           event->state == NPU_EVENT_CANCELLED;
}

static int event_destroy_locked(struct server_session *session, uint64_t id)
{
    struct server_event *event = find_event(session, id);
    struct server_event **link;

    if (!event)
        return -ENOENT;
    if (!event_ready(event) || event->command || event->references)
        return -EBUSY;
    link = &event->context->events;
    while (*link && *link != event)
        link = &(*link)->next;
    if (*link)
        *link = event->next;
    free(event);
    return 0;
}

static int payload_exact(const npu_ipc_header_t *header, size_t expected)
{
    return header->payload_size == expected;
}

static int output_read_locked(struct server_session *session,
                              npu_ipc_output_t *response,
                              size_t *response_size)
{
    struct server_output *output = session->output_head;
    size_t remaining;
    size_t chunk;

    if (!output)
        return -EAGAIN;
    remaining = output->size - output->offset;
    chunk = remaining < NPU_IPC_OUTPUT_CHUNK_SIZE ?
                remaining : NPU_IPC_OUTPUT_CHUNK_SIZE;
    memset(response, 0, sizeof(*response));
    response->cluster = output->cluster;
    response->core = output->core;
    response->proc = output->proc;
    response->flags = output->offset ? 0 : output->flags;
    response->data_size = (uint32_t)chunk;
    if (chunk)
        memcpy(response + 1, output->data + output->offset, chunk);
    output->offset += chunk;
    if (output->offset < output->size) {
        response->flags |= NPU_IPC_OUTPUT_CONTINUED;
    } else {
        size_t charge = output_charge(output);

        session->output_head = output->next;
        if (!session->output_head)
            session->output_tail = 0;
        session->output_bytes -= charge;
        free(output);
    }
    *response_size = sizeof(*response) + chunk;
    return 0;
}

static int profile_read_ready(void *object)
{
    const dayan_server_t *server = object;

    return server->profile_head || server->profile_stopped ||
           !server->profile_owner;
}

static int profile_start_locked(dayan_server_t *server,
                                struct server_session *session,
                                const npu_ipc_profile_start_t *request,
                                uint64_t *object_out)
{
    uint64_t profile_id;
    int status;

    if (!request->buffer_size || request->flags || request->reserved ||
        session->closing)
        return -EINVAL;
    if (server->profile_owner)
        return -EBUSY;
    status = next_id_locked(server, &profile_id);
    if (status < 0)
        return status;
    server->profile_owner = session;
    server->profile_id = profile_id;
    server->profile_active = 1;
    server->profile_stopping = 0;
    server->profile_stopped = 0;
    pthread_mutex_unlock(&server->mutex);
    status = npu_service_profile_start(server->device,
                                       request->buffer_size);
    if (status < 0) {
        int cleanup_status = npu_service_profile_stop(server->device);

        if (cleanup_status < 0 && cleanup_status != -ENOENT)
            (void)npu_service_profile_stop(server->device);
    }
    pthread_mutex_lock(&server->mutex);
    if (status < 0) {
        if (server->profile_owner == session &&
            server->profile_id == profile_id)
            profile_clear_locked(server);
        pthread_cond_broadcast(&server->condition);
        return status;
    }
    *object_out = profile_id;
    return 0;
}

static int profile_stop_locked(dayan_server_t *server,
                               struct server_session *session,
                               uint64_t profile_id)
{
    int status;

    if (server->profile_owner != session ||
        server->profile_id != profile_id)
        return -ENOENT;
    if (server->profile_stopped)
        return 0;
    if (!server->profile_active || server->profile_stopping)
        return -EBUSY;
    server->profile_stopping = 1;
    pthread_mutex_unlock(&server->mutex);
    status = npu_service_profile_stop(server->device);
    pthread_mutex_lock(&server->mutex);
    if (server->profile_owner != session ||
        server->profile_id != profile_id)
        return status < 0 ? status : -ECANCELED;
    server->profile_stopping = 0;
    if (status == 0 || status == -ENOENT) {
        server->profile_active = 0;
        server->profile_stopped = 1;
        status = 0;
    }
    pthread_cond_broadcast(&server->condition);
    return status;
}

static int profile_read_locked(dayan_server_t *server,
                               struct server_session *session,
                               uint64_t profile_id,
                               const npu_ipc_profile_read_t *request,
                               npu_ipc_profile_batch_t *response,
                               size_t *response_size)
{
    struct server_profile_chunk *chunk;
    size_t remaining;
    size_t count;
    int status;

    if (!request->max_bytes ||
        request->max_bytes > NPU_IPC_PROFILE_CHUNK_BYTES)
        return -EINVAL;
    if (server->profile_owner != session ||
        server->profile_id != profile_id)
        return -ENOENT;
    if (!server->profile_head && !server->profile_stopped) {
        ++server->profile_waiters;
        pthread_cond_broadcast(&server->condition);
        status = wait_with_timeout(server, request->timeout_ms,
                                   profile_read_ready, server);
        --server->profile_waiters;
        pthread_cond_broadcast(&server->condition);
        if (status < 0)
            return status;
        if (server->profile_owner != session ||
            server->profile_id != profile_id)
            return -ECANCELED;
    }
    memset(response, 0, sizeof(*response));
    chunk = server->profile_head;
    if (!chunk) {
        if (!server->profile_stopped)
            return -EAGAIN;
        response->flags = NPU_IPC_PROFILE_END;
        *response_size = sizeof(*response);
        return 0;
    }
    remaining = chunk->size - chunk->offset;
    count = remaining < request->max_bytes ? remaining :
                                              request->max_bytes;
    response->cluster = chunk->cluster;
    response->hart = chunk->hart;
    response->byte_count = (uint32_t)count;
    memcpy(response + 1, chunk->data + chunk->offset, count);
    chunk->offset += count;
    if (chunk->offset == chunk->size) {
        size_t charge = profile_chunk_charge(chunk);

        server->profile_head = chunk->next;
        if (!server->profile_head)
            server->profile_tail = 0;
        server->profile_bytes -= charge;
        free(chunk);
    }
    *response_size = sizeof(*response) + count;
    return 0;
}

static int profile_destroy_locked(dayan_server_t *server,
                                  struct server_session *session,
                                  uint64_t profile_id)
{
    if (server->profile_owner != session ||
        server->profile_id != profile_id)
        return -ENOENT;
    if (server->profile_active || server->profile_stopping ||
        !server->profile_stopped)
        return -EBUSY;
    profile_clear_locked(server);
    pthread_cond_broadcast(&server->condition);
    return 0;
}

static int dispatch_control(dayan_server_t *server,
                            struct server_session *session,
                            const npu_ipc_header_t *header,
                            const void *payload, uint64_t *object_out,
                            void *response, size_t *response_size)
{
    int status = -ENOSYS;

    *object_out = 0;
    *response_size = 0;
    pthread_mutex_lock(&server->mutex);
    switch (header->opcode) {
    case NPU_IPC_CONTEXT_CREATE:
        if (payload_exact(header, sizeof(npu_ipc_context_create_t)))
            status = context_create_locked(server, session, payload,
                                           object_out);
        else
            status = -EMSGSIZE;
        break;
    case NPU_IPC_CONTEXT_DESTROY: {
        struct server_context *context = find_context(session,
                                                       header->object_id);

        if (!payload_exact(header, 0))
            status = -EMSGSIZE;
        else if (!context)
            status = -ENOENT;
        else {
            server_cancel_context_locked(server, context);
            status = 0;
        }
        break;
    }
    case NPU_IPC_BUFFER_CREATE:
        if (payload_exact(header, sizeof(npu_ipc_buffer_create_t)))
            status = buffer_create_locked(server, session, payload,
                                          object_out);
        else
            status = -EMSGSIZE;
        break;
    case NPU_IPC_BUFFER_DESTROY:
        status = payload_exact(header, 0) ?
            buffer_destroy_locked(server, session, header->object_id) :
            -EMSGSIZE;
        break;
    case NPU_IPC_STREAM_CREATE:
        if (payload_exact(header, sizeof(npu_ipc_stream_create_t)))
            status = stream_create_locked(server, session, payload,
                                          object_out);
        else
            status = -EMSGSIZE;
        break;
    case NPU_IPC_STREAM_DESTROY:
        status = payload_exact(header, 0) ?
            stream_destroy_locked(session, header->object_id) : -EMSGSIZE;
        break;
    case NPU_IPC_STREAM_SYNCHRONIZE: {
        const npu_ipc_event_wait_t *request = payload;
        struct server_stream *stream = find_stream(session,
                                                    header->object_id);

        if (!payload_exact(header, sizeof(*request)))
            status = -EMSGSIZE;
        else if (!stream || request->event_id)
            status = -EINVAL;
        else {
            ++stream->references;
            status = wait_with_timeout(server, request->timeout_ms,
                                       stream_ready, stream);
            --stream->references;
        }
        break;
    }
    case NPU_IPC_STREAM_WAIT_EVENT:
        if (payload_exact(header, sizeof(npu_ipc_stream_wait_t)))
            status = stream_wait_event_locked(server, session, payload);
        else
            status = -EMSGSIZE;
        break;
    case NPU_IPC_TRANSFER_SUBMIT:
        if (payload_exact(header, sizeof(npu_ipc_transfer_submit_t)))
            status = transfer_submit_locked(server, session, payload,
                                            object_out);
        else
            status = -EMSGSIZE;
        break;
    case NPU_IPC_MODULE_LOAD: {
        const npu_ipc_module_load_t *request = payload;

        if (header->payload_size < sizeof(*request)) {
            status = -EMSGSIZE;
        } else {
            const unsigned char *image =
                (const unsigned char *)payload + sizeof(*request);
            size_t image_size = header->payload_size - sizeof(*request);

            status = module_load_locked(server, session, request, image,
                                        image_size, object_out);
        }
        break;
    }
    case NPU_IPC_MODULE_UNLOAD:
        status = payload_exact(header, 0) ?
            module_unload_locked(server, session, header->object_id) :
            -EMSGSIZE;
        break;
    case NPU_IPC_KERNEL_GET: {
        const npu_ipc_kernel_get_t *request = payload;

        if (header->payload_size < sizeof(*request) ||
            request->name_size != header->payload_size - sizeof(*request)) {
            status = -EMSGSIZE;
        } else {
            const char *name = (const char *)payload + sizeof(*request);

            status = kernel_get_locked(server, session, request, name,
                                       object_out);
        }
        break;
    }
    case NPU_IPC_KERNEL_INFO: {
        struct server_kernel *kernel = find_kernel(session,
                                                    header->object_id);
        npu_ipc_kernel_info_t *info = response;

        if (!payload_exact(header, 0))
            status = -EMSGSIZE;
        else if (!kernel)
            status = -ENOENT;
        else {
            info->supported_task_modes =
                kernel->metadata->supported_task_modes;
            info->max_tasks_per_core = kernel->metadata->max_tasks_per_core;
            info->max_child_procs = kernel->metadata->max_child_procs;
            info->argument_count = kernel->metadata->argument_count;
            info->sync_resource_count = kernel->metadata->sync_count;
            info->stack_size = kernel->metadata->stack_size;
            info->atomic_cache_size = kernel->metadata->atomic_cache_size;
            info->resident_memory = kernel->metadata->resident_memory;
            info->uram_size = kernel->metadata->uram_size;
            *response_size = sizeof(*info);
            status = 0;
        }
        break;
    }
    case NPU_IPC_KERNEL_RELEASE:
        status = payload_exact(header, 0) ?
            kernel_release_locked(session, header->object_id) : -EMSGSIZE;
        break;
    case NPU_IPC_KERNEL_LAUNCH:
        if (payload_exact(header, sizeof(npu_ipc_kernel_launch_t)))
            status = kernel_launch_locked(server, session, payload,
                                          object_out);
        else
            status = -EMSGSIZE;
        break;
    case NPU_IPC_EVENT_RECORD:
        if (payload_exact(header, 0))
            status = event_record_locked(server, session, header->object_id,
                                         object_out);
        else
            status = -EMSGSIZE;
        break;
    case NPU_IPC_EVENT_QUERY: {
        struct server_event *event = find_event(session, header->object_id);

        if (!payload_exact(header, 0))
            status = -EMSGSIZE;
        else if (!event)
            status = -ENOENT;
        else {
            fill_event_info(event, response);
            *response_size = sizeof(npu_ipc_event_info_t);
            status = 0;
        }
        break;
    }
    case NPU_IPC_EVENT_WAIT: {
        const npu_ipc_event_wait_t *request = payload;
        struct server_event *event = find_event(session, header->object_id);

        if (!payload_exact(header, sizeof(*request)))
            status = -EMSGSIZE;
        else if (!event || request->reserved)
            status = -EINVAL;
        else {
            ++event->references;
            status = wait_with_timeout(server, request->timeout_ms,
                                       event_ready, event);
            if (status == 0) {
                fill_event_info(event, response);
                *response_size = sizeof(npu_ipc_event_info_t);
            }
            --event->references;
        }
        break;
    }
    case NPU_IPC_EVENT_DESTROY:
        status = payload_exact(header, 0) ?
            event_destroy_locked(session, header->object_id) : -EMSGSIZE;
        break;
    case NPU_IPC_OUTPUT_READ:
        if (!payload_exact(header, 0) || header->object_id)
            status = -EMSGSIZE;
        else
            status = output_read_locked(session, response, response_size);
        break;
    case NPU_IPC_PROFILE_START:
        if (payload_exact(header, sizeof(npu_ipc_profile_start_t)))
            status = profile_start_locked(server, session, payload,
                                          object_out);
        else
            status = -EMSGSIZE;
        break;
    case NPU_IPC_PROFILE_STOP:
        status = payload_exact(header, 0) ?
            profile_stop_locked(server, session, header->object_id) :
            -EMSGSIZE;
        break;
    case NPU_IPC_PROFILE_READ:
        if (payload_exact(header, sizeof(npu_ipc_profile_read_t)))
            status = profile_read_locked(server, session, header->object_id,
                                         payload, response, response_size);
        else
            status = -EMSGSIZE;
        break;
    case NPU_IPC_PROFILE_DESTROY:
        status = payload_exact(header, 0) ?
            profile_destroy_locked(server, session, header->object_id) :
            -EMSGSIZE;
        break;
    default:
        status = -ENOSYS;
        break;
    }
    server_collect_contexts_locked(server);
    pthread_mutex_unlock(&server->mutex);
    return status;
}

static int transfer_grant_ready(void *object)
{
    struct server_event *event = object;
    struct server_command *command = event->command;

    return event_ready(event) ||
           (command && (command->kind == SERVER_COMMAND_H2D ||
                        command->kind == SERVER_COMMAND_D2H) &&
            command->data.transfer.token_ready);
}

static int dispatch_data(dayan_server_t *server,
                         struct server_session *session,
                         const npu_ipc_header_t *header,
                         const void *payload, void *response,
                         size_t *response_size)
{
    int status = -ENOSYS;

    *response_size = 0;
    pthread_mutex_lock(&server->mutex);
    if (header->opcode == NPU_IPC_TRANSFER_ACQUIRE) {
        struct server_event *event = find_event(session, header->object_id);

        if (!payload_exact(header, 0)) {
            status = -EMSGSIZE;
        } else if (!event || !event->command ||
                   (event->command->kind != SERVER_COMMAND_H2D &&
                    event->command->kind != SERVER_COMMAND_D2H)) {
            status = -ENOENT;
        } else {
            struct server_command *command;
            npu_ipc_transfer_grant_t *grant = response;

            ++event->references;
            status = wait_with_timeout(server, NPU_TIMEOUT_INFINITE,
                                       transfer_grant_ready, event);
            command = event->command;
            if (status == 0 && event_ready(event))
                status = event->status < INT32_MIN ||
                         event->status > INT32_MAX ? -EIO :
                                                   (int)event->status;
            if (status == 0 && (!command ||
                command->data.transfer.granted))
                status = -EALREADY;
            if (status == 0) {
                command->data.transfer.granted = 1;
                grant->event_id = event->id;
                grant->transfer_id = command->data.transfer.transfer_id;
                grant->token = command->data.transfer.token;
                *response_size = sizeof(*grant);
            }
            --event->references;
        }
    } else if (header->opcode == NPU_IPC_TRANSFER_COMPLETE) {
        const npu_ipc_transfer_complete_t *complete = payload;
        struct server_event *event;
        struct server_command *command;

        if (!payload_exact(header, sizeof(*complete))) {
            status = -EMSGSIZE;
        } else if (!(event = find_event(session, complete->event_id)) ||
                   !(command = event->command) ||
                   (command->kind != SERVER_COMMAND_H2D &&
                    command->kind != SERVER_COMMAND_D2H)) {
            status = -ENOENT;
        } else if (!command->data.transfer.granted ||
                   command->data.transfer.completed ||
                   command->data.transfer.transfer_id !=
                       complete->transfer_id || complete->reserved ||
                   memcmp(&command->data.transfer.token, &complete->token,
                          sizeof(complete->token)) != 0) {
            status = -EINVAL;
        } else {
            status = npu_hal_finish_transfer(
                server->hal, &complete->token,
                complete->transfer_status);
            command->data.transfer.completed = 1;
            command->data.transfer.transfer_status = status < 0 ? status :
                complete->transfer_status;
            pthread_cond_broadcast(&server->condition);
            status = 0;
        }
    }
    pthread_mutex_unlock(&server->mutex);
    return status;
}

static int send_reply(int fd, const npu_ipc_header_t *request, int status,
                      uint64_t object_id, const void *payload,
                      size_t payload_size, const int *handles,
                      size_t handle_count)
{
    npu_ipc_header_t response;

    if (payload_size > UINT32_MAX)
        return -EOVERFLOW;
    npu_ipc_header_init(&response, request->opcode, request->request_id,
                        object_id, (uint32_t)payload_size);
    response.flags = NPU_IPC_RESPONSE;
    response.status = status;
    return npu_ipc_send(fd, &response, payload, handles, handle_count);
}

static int peer_credentials(int fd, uid_t *uid_out, gid_t *gid_out)
{
    struct ucred credentials;
    socklen_t size = sizeof(credentials);

    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials, &size) < 0)
        return -errno;
    if (size != sizeof(credentials))
        return -EPROTO;
    *uid_out = credentials.uid;
    *gid_out = credentials.gid;
    return 0;
}

static struct server_session *create_control_session_locked(
    dayan_server_t *server, int fd, uid_t uid, gid_t gid)
{
    struct server_session *session = calloc(1, sizeof(*session));

    if (!session || next_id_locked(server, &session->id) < 0) {
        free(session);
        return 0;
    }
    session->uid = uid;
    session->gid = gid;
    session->control_fd = fd;
    session->data_fd = -1;
    session->next = server->sessions;
    server->sessions = session;
    return session;
}

static struct server_session *find_session_locked(dayan_server_t *server,
                                                  uint64_t id, uid_t uid,
                                                  gid_t gid)
{
    struct server_session *session;

    for (session = server->sessions; session; session = session->next) {
        if (session->id == id && session->uid == uid && session->gid == gid)
            return session;
    }
    return 0;
}

static int connection_hello(struct server_connection *connection,
                            struct server_session **session_out,
                            uint32_t *role_out)
{
    dayan_server_t *server = connection->server;
    npu_ipc_hello_response_t response;
    npu_ipc_hello_request_t *hello;
    npu_ipc_header_t header;
    int handles[NPU_HAL_DATA_MAX_HANDLES];
    size_t handle_count = 0;
    void *payload = 0;
    uid_t uid = 0;
    gid_t gid = 0;
    size_t received_handles = 0;
    struct server_session *session = 0;
    int status;

    status = npu_ipc_receive(connection->fd, &header, &payload, handles,
                             &received_handles);
    if (status < 0)
        return status;
    if (received_handles || header.opcode != NPU_IPC_HELLO ||
        header.flags || header.payload_size != sizeof(*hello)) {
        free(payload);
        return -EPROTO;
    }
    hello = payload;
    if (hello->runtime_version != NPU_RUNTIME_OPTIONS_VERSION) {
        status = -EPROTONOSUPPORT;
        goto reply;
    }
    status = peer_credentials(connection->fd, &uid, &gid);
    if (status < 0)
        goto reply;
    memset(&response, 0, sizeof(response));
    response.service_version = NPU_IPC_VERSION;
    response.capabilities = NPU_IPC_CAP_DIRECT_TRANSFER |
                            NPU_IPC_CAP_DEVICE_OUTPUT |
                            NPU_IPC_CAP_PROFILE;
    pthread_mutex_lock(&server->mutex);
    if (hello->role == NPU_IPC_ROLE_CONTROL && !hello->session_id) {
        session = create_control_session_locked(server, connection->fd,
                                                uid, gid);
        status = session ? 0 : -ENOMEM;
    } else if (hello->role == NPU_IPC_ROLE_DATA && hello->session_id) {
        session = find_session_locked(server, hello->session_id, uid, gid);
        if (!session || session->closing || session->data_fd >= 0) {
            status = -EACCES;
            session = 0;
        } else {
            session->data_fd = connection->fd;
            status = 0;
        }
    } else {
        status = -EINVAL;
    }
    if (session)
        response.session_id = session->id;
    pthread_mutex_unlock(&server->mutex);
    if (status == 0 && hello->role == NPU_IPC_ROLE_DATA) {
        status = npu_hal_export_data_plane(server->hal,
                                           &response.data_plane, handles);
        if (status == 0)
            handle_count = response.data_plane.handle_count;
    }

reply:
    if (send_reply(connection->fd, &header, status,
                   session ? session->id : 0,
                   status == 0 ? &response : 0,
                   status == 0 ? sizeof(response) : 0,
                   handles, status == 0 ? handle_count : 0) < 0 &&
        status == 0)
        status = -EPIPE;
    while (handle_count)
        close(handles[--handle_count]);
    if (status == 0) {
        *session_out = session;
        *role_out = hello->role;
    } else if (session) {
        pthread_mutex_lock(&server->mutex);
        if (hello->role == NPU_IPC_ROLE_CONTROL &&
            session->control_fd == connection->fd) {
            session->control_fd = -1;
            session->closing = 1;
        } else if (hello->role == NPU_IPC_ROLE_DATA &&
                   session->data_fd == connection->fd) {
            session->data_fd = -1;
        }
        server_collect_contexts_locked(server);
        pthread_mutex_unlock(&server->mutex);
    }
    free(payload);
    return status;
}

static void connection_closed(dayan_server_t *server,
                              struct server_session *session,
                              uint32_t role, int fd)
{
    int stop_profile = 0;

    pthread_mutex_lock(&server->mutex);
    if (role == NPU_IPC_ROLE_CONTROL && session->control_fd == fd) {
        struct server_context *context;

        session->control_fd = -1;
        session->closing = 1;
        for (context = session->contexts; context; context = context->next)
            server_cancel_context_locked(server, context);
        if (server->profile_owner == session) {
            if (server->stop) {
                profile_clear_locked(server);
            } else if (server->profile_active &&
                       !server->profile_stopping) {
                server->profile_stopping = 1;
                stop_profile = 1;
            } else if (!server->profile_active) {
                profile_clear_locked(server);
            }
        }
    } else if (role == NPU_IPC_ROLE_DATA && session->data_fd == fd) {
        struct server_context *context;

        session->data_fd = -1;
        for (context = session->contexts; context; context = context->next) {
            struct server_stream *stream;

            for (stream = context->streams; stream; stream = stream->next) {
                struct server_command *command;

                for (command = stream->head; command; command = command->next) {
                    if ((command->kind == SERVER_COMMAND_H2D ||
                         command->kind == SERVER_COMMAND_D2H) &&
                        command->data.transfer.granted &&
                        !command->data.transfer.completed) {
                        (void)npu_hal_finish_transfer(
                            server->hal,
                            &command->data.transfer.token, -ECONNRESET);
                        command->data.transfer.completed = 1;
                        command->data.transfer.transfer_status =
                            -ECONNRESET;
                    }
                }
            }
        }
    }
    if (session->control_fd < 0 && session->data_fd < 0)
        session->closing = 1;
    server_collect_contexts_locked(server);
    pthread_cond_broadcast(&server->condition);
    pthread_mutex_unlock(&server->mutex);
    if (stop_profile) {
        int status = npu_service_profile_stop(server->device);

        if (status < 0 && status != -ENOENT)
            (void)npu_service_profile_stop(server->device);
        pthread_mutex_lock(&server->mutex);
        if (server->profile_owner == session)
            profile_clear_locked(server);
        server_collect_contexts_locked(server);
        pthread_cond_broadcast(&server->condition);
        pthread_mutex_unlock(&server->mutex);
    }
}

static void *connection_main(void *argument)
{
    struct server_connection *connection = argument;
    dayan_server_t *server = connection->server;
    struct server_session *session = 0;
    uint32_t role = 0;
    int hello_status;

    hello_status = connection_hello(connection, &session, &role);
    if (hello_status == 0) {
        uint64_t session_id = session->id;

        server_log("connection", "session=%llu role=%s connected",
                   (unsigned long long)session_id,
                   connection_role_name(role));
        for (;;) {
            union {
                npu_ipc_event_info_t event;
                npu_ipc_kernel_info_t kernel;
                npu_ipc_transfer_grant_t grant;
                struct {
                    npu_ipc_output_t header;
                    unsigned char data[NPU_IPC_OUTPUT_CHUNK_SIZE];
                } output;
                struct {
                    npu_ipc_profile_batch_t header;
                    unsigned char data[NPU_IPC_PROFILE_CHUNK_BYTES];
                } profile;
            } response;
            int handles[NPU_HAL_DATA_MAX_HANDLES];
            npu_ipc_header_t header;
            size_t handle_count = 0;
            size_t response_size = 0;
            uint64_t object_id = 0;
            void *payload = 0;
            int status = npu_ipc_receive(connection->fd, &header, &payload,
                                         handles, &handle_count);

            while (handle_count)
                close(handles[--handle_count]);
            if (status < 0)
                break;
            if (header.flags || header.opcode == NPU_IPC_HELLO) {
                status = -EPROTO;
            } else if (role == NPU_IPC_ROLE_CONTROL) {
                status = dispatch_control(server, session, &header, payload,
                                          &object_id, &response,
                                          &response_size);
            } else {
                status = dispatch_data(server, session, &header, payload,
                                       &response, &response_size);
            }
            free(payload);
            if (send_reply(connection->fd, &header, status, object_id,
                           status == 0 ? &response : 0,
                           status == 0 ? response_size : 0, 0, 0) < 0)
                break;
        }
        connection_closed(server, session, role, connection->fd);
        server_log("connection", "session=%llu role=%s disconnected",
                   (unsigned long long)session_id,
                   connection_role_name(role));
    } else {
        server_log("connection", "handshake rejected: %s (%d)",
                   server_status_string(hello_status), hello_status);
    }
    close(connection->fd);
    pthread_mutex_lock(&server->mutex);
    if (server->connection_count)
        --server->connection_count;
    pthread_cond_broadcast(&server->condition);
    pthread_mutex_unlock(&server->mutex);
    free(connection);
    return 0;
}

int dayan_server_run(dayan_server_t *server,
                     const volatile int *stop_requested)
{
    if (!server)
        return -EINVAL;
    for (;;) {
        struct pollfd poll_fd;
        int device_status;
        int result;

        device_status = npu_hal_device_status(server->hal);
        if (device_status < 0) {
            server_log("error", "Device access failed: %s (%d); "
                       "stopping Server",
                       server_status_string(device_status), device_status);
            dayan_server_request_stop(server);
            return device_status;
        }
        pthread_mutex_lock(&server->mutex);
        if (server->stop || (stop_requested && *stop_requested)) {
            pthread_mutex_unlock(&server->mutex);
            break;
        }
        poll_fd.fd = server->listen_fd;
        poll_fd.events = POLLIN;
        poll_fd.revents = 0;
        pthread_mutex_unlock(&server->mutex);
        result = poll(&poll_fd, 1, 100);
        if (result < 0 && errno == EINTR)
            continue;
        if (result < 0)
            return -errno;
        if (!result)
            continue;
        if (poll_fd.revents & POLLIN) {
            struct server_connection *connection;
            pthread_attr_t attributes;
            pthread_t thread;
            int fd = accept4(server->listen_fd, 0, 0, SOCK_CLOEXEC);
            int status;

            if (fd < 0) {
                if (errno == EINTR || errno == EAGAIN)
                    continue;
                pthread_mutex_lock(&server->mutex);
                status = server->stop ? 0 : -errno;
                pthread_mutex_unlock(&server->mutex);
                if (!status)
                    break;
                return status;
            }
            connection = calloc(1, sizeof(*connection));
            if (!connection) {
                close(fd);
                return -ENOMEM;
            }
            connection->server = server;
            connection->fd = fd;
            pthread_mutex_lock(&server->mutex);
            ++server->connection_count;
            pthread_mutex_unlock(&server->mutex);
            int attributes_initialized = 0;

            status = pthread_attr_init(&attributes);
            if (!status) {
                attributes_initialized = 1;
                status = pthread_attr_setdetachstate(
                    &attributes, PTHREAD_CREATE_DETACHED);
            }
            if (!status)
                status = pthread_create(&thread, &attributes,
                                        connection_main, connection);
            if (attributes_initialized)
                (void)pthread_attr_destroy(&attributes);
            if (status) {
                pthread_mutex_lock(&server->mutex);
                --server->connection_count;
                pthread_mutex_unlock(&server->mutex);
                close(fd);
                free(connection);
                return -status;
            }
        }
    }
    dayan_server_request_stop(server);
    return 0;
}

void dayan_server_request_stop(dayan_server_t *server)
{
    if (!server)
        return;
    pthread_mutex_lock(&server->mutex);
    if (!server->stop) {
        server->stop = 1;
        cancel_all_locked(server);
        if (server->listen_fd >= 0)
            (void)shutdown(server->listen_fd, SHUT_RDWR);
        {
            struct server_session *session;

            for (session = server->sessions; session; session = session->next) {
                if (session->control_fd >= 0)
                    (void)shutdown(session->control_fd, SHUT_RDWR);
                if (session->data_fd >= 0)
                    (void)shutdown(session->data_fd, SHUT_RDWR);
            }
        }
        pthread_cond_broadcast(&server->condition);
    }
    pthread_mutex_unlock(&server->mutex);
}

void dayan_server_destroy(dayan_server_t *server)
{
    struct server_package_cache *package;
    int device_status;
    int service_status;

    if (!server)
        return;
    dayan_server_request_stop(server);
    device_status = npu_hal_device_status(server->hal);
    if (device_status < 0) {
        server_log("shutdown", "Device is unavailable; skipping Device-side "
                   "process cleanup");
    } else {
        server_log("shutdown", "requesting process cleanup on enabled Harts");
        shutdown_device_processes(server);
        device_status = npu_hal_device_status(server->hal);
    }
    npu_service_abort(server->device,
                      device_status < 0 ? device_status : -ESHUTDOWN);
    pthread_mutex_lock(&server->mutex);
    pthread_cond_broadcast(&server->condition);
    pthread_mutex_unlock(&server->mutex);
    if (server->scheduler_started)
        (void)pthread_join(server->scheduler, 0);
    pthread_mutex_lock(&server->mutex);
    while (server->connection_count)
        (void)pthread_cond_wait(&server->condition, &server->mutex);
    server_collect_contexts_locked(server);
    pthread_mutex_unlock(&server->mutex);
    if (server->listen_fd >= 0)
        close(server->listen_fd);
    (void)unlink(server->socket_path);
    package = server->packages;
    while (package) {
        struct server_package_cache *next = package->next;

        free(package->image);
        free(package);
        package = next;
    }
    output_list_free(server->pending_outputs);
    profile_chunk_list_free(server->profile_head);
    service_status = npu_service_close(server->device);
    if (service_status < 0)
        server_log("shutdown", "Device service cleanup failed: %s (%d)",
                   server_status_string(service_status), service_status);
    npu_hal_close(server->hal);
    free(server->slots);
    pthread_cond_destroy(&server->condition);
    pthread_mutex_destroy(&server->mutex);
    free(server->socket_path);
    free(server);
}
