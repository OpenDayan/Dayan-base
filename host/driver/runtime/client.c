#define _POSIX_C_SOURCE 200809L

#include "runtime_internal.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define NPU_RUNTIME_DEFAULT_CONNECT_TIMEOUT 5000U
#define NPU_RUNTIME_DEFAULT_TIMEOUT         5000U
#define NPU_RUNTIME_MAX_OUTPUT_READS        4096U

struct runtime_output {
    struct runtime_output *next;
    uint32_t cluster;
    uint32_t core;
    uint32_t proc;
    uint32_t flags;
    size_t size;
    unsigned char data[];
};

_Static_assert(NPU_RUNTIME_OUTPUT_CLEAR == NPU_IPC_OUTPUT_CLEAR,
               "Runtime and socket output flags");
_Static_assert(NPU_RUNTIME_OUTPUT_CONTINUED == NPU_IPC_OUTPUT_CONTINUED,
               "Runtime and socket continuation flags");

static void default_output(void *context, uint32_t cluster, uint32_t core,
                           uint32_t proc, const void *data, size_t size,
                           uint32_t flags)
{
    (void)context;
    (void)cluster;
    (void)core;
    (void)proc;
    if (flags & NPU_RUNTIME_OUTPUT_CLEAR)
        fflush(stdout);
    if (data && size) {
        (void)fwrite(data, 1, size, stdout);
        fflush(stdout);
    }
}

static void close_handles(int *handles, size_t count)
{
    size_t index;

    for (index = 0; index < count; ++index) {
        if (handles[index] >= 0)
            close(handles[index]);
        handles[index] = -1;
    }
}

static int ipc_call_locked(npu_runtime_t *runtime, int fd, uint16_t opcode,
                           uint64_t object_id, const void *request,
                           size_t request_size, void *response,
                           size_t expected_response_size,
                           uint64_t *response_object_id)
{
    int handles[NPU_HAL_DATA_MAX_HANDLES];
    npu_ipc_header_t request_header;
    npu_ipc_header_t response_header;
    uint64_t request_id;
    size_t handle_count = 0;
    void *payload = 0;
    int status;

    if (request_size > UINT32_MAX ||
        (request_size && !request) ||
        (expected_response_size && !response))
        return -EINVAL;
    request_id = __atomic_add_fetch(&runtime->next_request_id, 1,
                                    __ATOMIC_RELAXED);
    if (!request_id)
        return -EOVERFLOW;
    npu_ipc_header_init(&request_header, opcode, request_id, object_id,
                        (uint32_t)request_size);
    status = npu_ipc_send(fd, &request_header, request, 0, 0);
    if (status < 0)
        return status;
    status = npu_ipc_receive(fd, &response_header, &payload, handles,
                             &handle_count);
    if (status < 0)
        return status;
    if (handle_count || response_header.opcode != opcode ||
        response_header.flags != NPU_IPC_RESPONSE ||
        response_header.request_id != request_id) {
        status = -EPROTO;
        goto out;
    }
    if (response_header.status < 0) {
        if (response_header.payload_size)
            status = -EPROTO;
        else
            status = response_header.status;
        goto out;
    }
    if (response_header.status ||
        response_header.payload_size != expected_response_size) {
        status = -EPROTO;
        goto out;
    }
    if (expected_response_size)
        memcpy(response, payload, expected_response_size);
    if (response_object_id)
        *response_object_id = response_header.object_id;
    status = 0;

out:
    close_handles(handles, handle_count);
    free(payload);
    return status;
}

static int output_drain_locked(npu_runtime_t *runtime,
                               struct runtime_output **head_out)
{
    struct runtime_output *head = 0;
    struct runtime_output *tail = 0;
    unsigned reads;
    int status = 0;

    *head_out = 0;
    for (reads = 0; reads < NPU_RUNTIME_MAX_OUTPUT_READS; ++reads) {
        int handles[NPU_HAL_DATA_MAX_HANDLES];
        npu_ipc_header_t request;
        npu_ipc_header_t response;
        npu_ipc_output_t *output;
        struct runtime_output *record;
        uint64_t request_id;
        size_t handle_count = 0;
        void *payload = 0;

        request_id = __atomic_add_fetch(&runtime->next_request_id, 1,
                                        __ATOMIC_RELAXED);
        if (!request_id) {
            status = -EOVERFLOW;
            break;
        }
        npu_ipc_header_init(&request, NPU_IPC_OUTPUT_READ, request_id, 0, 0);
        status = npu_ipc_send(runtime->control_fd, &request, 0, 0, 0);
        if (status < 0)
            break;
        status = npu_ipc_receive(runtime->control_fd, &response, &payload,
                                 handles, &handle_count);
        if (status < 0)
            break;
        if (handle_count || response.opcode != NPU_IPC_OUTPUT_READ ||
            response.flags != NPU_IPC_RESPONSE ||
            response.request_id != request_id) {
            status = -EPROTO;
            close_handles(handles, handle_count);
            free(payload);
            break;
        }
        close_handles(handles, handle_count);
        if (response.status == -EAGAIN) {
            status = response.payload_size ? -EPROTO : 0;
            free(payload);
            break;
        }
        if (response.status ||
            response.payload_size < sizeof(npu_ipc_output_t)) {
            status = response.status < 0 ? response.status : -EPROTO;
            free(payload);
            break;
        }
        output = payload;
        if (output->reserved ||
            (output->flags & ~(NPU_IPC_OUTPUT_CLEAR |
                               NPU_IPC_OUTPUT_CONTINUED)) ||
            output->data_size > NPU_IPC_OUTPUT_CHUNK_SIZE ||
            response.payload_size != sizeof(*output) + output->data_size) {
            status = -EPROTO;
            free(payload);
            break;
        }
        record = malloc(sizeof(*record) + output->data_size);
        if (!record) {
            status = -ENOMEM;
            free(payload);
            break;
        }
        record->next = 0;
        record->cluster = output->cluster;
        record->core = output->core;
        record->proc = output->proc;
        record->flags = output->flags;
        record->size = output->data_size;
        if (record->size)
            memcpy(record->data, output + 1, record->size);
        if (tail)
            tail->next = record;
        else
            head = record;
        tail = record;
        free(payload);
        status = 0;
    }
    *head_out = head;
    return status;
}

static void output_deliver(npu_runtime_t *runtime,
                           struct runtime_output *output)
{
    while (output) {
        struct runtime_output *next = output->next;

        if (runtime->output) {
            runtime->output(runtime->output_context, output->cluster,
                            output->core, output->proc, output->data,
                            output->size, output->flags);
        }
        free(output);
        output = next;
    }
}

static int opcode_drains_output(uint16_t opcode)
{
    return opcode == NPU_IPC_CONTEXT_DESTROY ||
           opcode == NPU_IPC_STREAM_DESTROY ||
           opcode == NPU_IPC_STREAM_SYNCHRONIZE ||
           opcode == NPU_IPC_EVENT_QUERY ||
           opcode == NPU_IPC_EVENT_WAIT ||
           opcode == NPU_IPC_EVENT_DESTROY;
}

int runtime_control_call(npu_runtime_t *runtime, uint16_t opcode,
                         uint64_t object_id, const void *request,
                         size_t request_size, void *response,
                         size_t expected_response_size,
                         uint64_t *response_object_id)
{
    struct runtime_output *output = 0;
    int status;
    int output_status = 0;

    if (!runtime || runtime->control_fd < 0)
        return -EINVAL;
    pthread_mutex_lock(&runtime->control_mutex);
    status = ipc_call_locked(runtime, runtime->control_fd, opcode, object_id,
                             request, request_size, response,
                             expected_response_size, response_object_id);
    if ((status == 0 || status == -ETIMEDOUT) &&
        opcode_drains_output(opcode))
        output_status = output_drain_locked(runtime, &output);
    pthread_mutex_unlock(&runtime->control_mutex);
    output_deliver(runtime, output);
    return status == 0 && output_status < 0 ? output_status : status;
}

int runtime_control_call_variable(npu_runtime_t *runtime, uint16_t opcode,
                                  uint64_t object_id, const void *request,
                                  size_t request_size, void **response_out,
                                  size_t *response_size_out)
{
    int handles[NPU_HAL_DATA_MAX_HANDLES];
    npu_ipc_header_t request_header;
    npu_ipc_header_t response_header;
    uint64_t request_id;
    size_t handle_count = 0;
    void *payload = 0;
    int status;

    if (!runtime || runtime->control_fd < 0 || !response_out ||
        !response_size_out || request_size > UINT32_MAX ||
        (request_size && !request))
        return -EINVAL;
    *response_out = 0;
    *response_size_out = 0;
    pthread_mutex_lock(&runtime->control_mutex);
    request_id = __atomic_add_fetch(&runtime->next_request_id, 1,
                                    __ATOMIC_RELAXED);
    if (!request_id) {
        status = -EOVERFLOW;
        goto out;
    }
    npu_ipc_header_init(&request_header, opcode, request_id, object_id,
                        (uint32_t)request_size);
    status = npu_ipc_send(runtime->control_fd, &request_header, request,
                          0, 0);
    if (status < 0)
        goto out;
    status = npu_ipc_receive(runtime->control_fd, &response_header, &payload,
                             handles, &handle_count);
    if (status < 0)
        goto out;
    if (handle_count || response_header.opcode != opcode ||
        response_header.flags != NPU_IPC_RESPONSE ||
        response_header.request_id != request_id) {
        status = -EPROTO;
        goto out;
    }
    if (response_header.status < 0) {
        status = response_header.payload_size ? -EPROTO :
                                                response_header.status;
        goto out;
    }
    if (response_header.status) {
        status = -EPROTO;
        goto out;
    }
    *response_out = payload;
    *response_size_out = response_header.payload_size;
    payload = 0;
    status = 0;

out:
    close_handles(handles, handle_count);
    free(payload);
    pthread_mutex_unlock(&runtime->control_mutex);
    return status;
}

int runtime_data_call(npu_runtime_t *runtime, uint16_t opcode,
                      uint64_t object_id, const void *request,
                      size_t request_size, void *response,
                      size_t expected_response_size)
{
    int status;

    if (!runtime || runtime->data_fd < 0)
        return -EINVAL;
    pthread_mutex_lock(&runtime->data_mutex);
    status = ipc_call_locked(runtime, runtime->data_fd, opcode, object_id,
                             request, request_size, response,
                             expected_response_size, 0);
    pthread_mutex_unlock(&runtime->data_mutex);
    return status;
}

static int hello(int fd, npu_runtime_t *runtime, uint32_t role,
                 uint64_t session_id, npu_ipc_hello_response_t *hello_out,
                 int handles[NPU_HAL_DATA_MAX_HANDLES],
                 size_t *handle_count_out)
{
    npu_ipc_hello_request_t request;
    npu_ipc_header_t request_header;
    npu_ipc_header_t response_header;
    void *payload = 0;
    uint64_t request_id;
    size_t handle_count = 0;
    int status;

    memset(&request, 0, sizeof(request));
    request.role = role;
    request.runtime_version = NPU_RUNTIME_OPTIONS_VERSION;
    request.session_id = session_id;
    request_id = __atomic_add_fetch(&runtime->next_request_id, 1,
                                    __ATOMIC_RELAXED);
    npu_ipc_header_init(&request_header, NPU_IPC_HELLO, request_id, 0,
                        sizeof(request));
    status = npu_ipc_send(fd, &request_header, &request, 0, 0);
    if (status < 0)
        return status;
    status = npu_ipc_receive(fd, &response_header, &payload, handles,
                             &handle_count);
    if (status < 0)
        return status;
    if (response_header.opcode != NPU_IPC_HELLO ||
        response_header.flags != NPU_IPC_RESPONSE ||
        response_header.request_id != request_id ||
        response_header.status < 0 ||
        response_header.payload_size != sizeof(*hello_out)) {
        status = response_header.status < 0 ? response_header.status :
                                              -EPROTO;
        goto fail;
    }
    memcpy(hello_out, payload, sizeof(*hello_out));
    if (!hello_out->session_id ||
        (session_id && hello_out->session_id != session_id) ||
        hello_out->service_version != NPU_IPC_VERSION ||
        (hello_out->capabilities & (NPU_IPC_CAP_DIRECT_TRANSFER |
                                    NPU_IPC_CAP_DEVICE_OUTPUT |
                                    NPU_IPC_CAP_PROFILE)) !=
            (NPU_IPC_CAP_DIRECT_TRANSFER | NPU_IPC_CAP_DEVICE_OUTPUT |
             NPU_IPC_CAP_PROFILE)) {
        status = -EPROTO;
        goto fail;
    }
    if ((role == NPU_IPC_ROLE_CONTROL && handle_count) ||
        (role == NPU_IPC_ROLE_DATA &&
         handle_count != hello_out->data_plane.handle_count)) {
        status = -EPROTO;
        goto fail;
    }
    *handle_count_out = handle_count;
    free(payload);
    return 0;

fail:
    close_handles(handles, handle_count);
    free(payload);
    return status;
}

void npu_runtime_options_init(npu_runtime_options_t *options)
{
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->version = NPU_RUNTIME_OPTIONS_VERSION;
    options->server_socket = NPU_DEFAULT_SERVER_SOCKET;
    options->connect_timeout_ms = NPU_RUNTIME_DEFAULT_CONNECT_TIMEOUT;
    options->default_timeout_ms = NPU_RUNTIME_DEFAULT_TIMEOUT;
    options->output = default_output;
}

int npu_runtime_open(npu_runtime_t **runtime_out,
                     const npu_runtime_options_t *options)
{
    npu_runtime_options_t defaults;
    npu_ipc_hello_response_t control_hello;
    npu_ipc_hello_response_t data_hello;
    int handles[NPU_HAL_DATA_MAX_HANDLES];
    npu_runtime_t *runtime;
    const char *socket_path;
    size_t handle_count = 0;
    int control_mutex_initialized = 0;
    int data_mutex_initialized = 0;
    int mutex_initialized = 0;
    int condition_initialized = 0;
    int status;

    if (!runtime_out)
        return -EINVAL;
    *runtime_out = 0;
    if (!options) {
        npu_runtime_options_init(&defaults);
        options = &defaults;
    }
    if (options->version != NPU_RUNTIME_OPTIONS_VERSION || options->flags ||
        !options->server_socket || !*options->server_socket ||
        !options->connect_timeout_ms || !options->default_timeout_ms)
        return -EINVAL;
    runtime = calloc(1, sizeof(*runtime));
    if (!runtime)
        return -ENOMEM;
    runtime->control_fd = -1;
    runtime->data_fd = -1;
    runtime->default_timeout_ms = options->default_timeout_ms;
    runtime->output = options->output;
    runtime->output_context = options->output_context;
    status = pthread_mutex_init(&runtime->control_mutex, 0);
    if (status) {
        status = -status;
        goto fail;
    }
    control_mutex_initialized = 1;
    status = pthread_mutex_init(&runtime->data_mutex, 0);
    if (status) {
        status = -status;
        goto fail;
    }
    data_mutex_initialized = 1;
    status = pthread_mutex_init(&runtime->mutex, 0);
    if (status) {
        status = -status;
        goto fail;
    }
    mutex_initialized = 1;
    status = pthread_cond_init(&runtime->condition, 0);
    if (status) {
        status = -status;
        goto fail;
    }
    condition_initialized = 1;
    socket_path = options->server_socket;
    runtime->control_fd = npu_ipc_connect(socket_path,
                                           options->connect_timeout_ms);
    if (runtime->control_fd < 0) {
        status = runtime->control_fd;
        runtime->control_fd = -1;
        goto fail;
    }
    status = hello(runtime->control_fd, runtime, NPU_IPC_ROLE_CONTROL, 0,
                   &control_hello, handles, &handle_count);
    if (status < 0)
        goto fail;
    runtime->session_id = control_hello.session_id;
    runtime->data_fd = npu_ipc_connect(socket_path,
                                       options->connect_timeout_ms);
    if (runtime->data_fd < 0) {
        status = runtime->data_fd;
        runtime->data_fd = -1;
        goto fail;
    }
    status = hello(runtime->data_fd, runtime, NPU_IPC_ROLE_DATA,
                   runtime->session_id, &data_hello, handles, &handle_count);
    if (status < 0)
        goto fail;
    status = npu_hal_data_plane_import(&runtime->data_plane,
                                       &data_hello.data_plane, handles,
                                       handle_count);
    close_handles(handles, handle_count);
    handle_count = 0;
    if (status < 0)
        goto fail;
    status = pthread_create(&runtime->data_worker, 0, runtime_data_worker,
                            runtime);
    if (status) {
        status = -status;
        goto fail;
    }
    runtime->data_worker_started = 1;
    *runtime_out = runtime;
    return 0;

fail:
    close_handles(handles, handle_count);
    if (runtime->data_worker_started) {
        pthread_mutex_lock(&runtime->mutex);
        runtime->data_worker_stop = 1;
        pthread_cond_broadcast(&runtime->condition);
        pthread_mutex_unlock(&runtime->mutex);
        (void)pthread_join(runtime->data_worker, 0);
    }
    npu_hal_data_plane_close(runtime->data_plane);
    if (runtime->data_fd >= 0)
        close(runtime->data_fd);
    if (runtime->control_fd >= 0)
        close(runtime->control_fd);
    if (condition_initialized)
        pthread_cond_destroy(&runtime->condition);
    if (mutex_initialized)
        pthread_mutex_destroy(&runtime->mutex);
    if (data_mutex_initialized)
        pthread_mutex_destroy(&runtime->data_mutex);
    if (control_mutex_initialized)
        pthread_mutex_destroy(&runtime->control_mutex);
    free(runtime);
    return status;
}

int runtime_context_valid(const struct npu_context *context)
{
    return context && context->runtime && !context->closing &&
           !context->runtime->closing;
}

int runtime_stream_valid(const struct npu_stream *stream)
{
    return stream && runtime_context_valid(stream->context);
}

int runtime_event_adopt(struct npu_context *context, uint64_t id,
                        struct npu_event *event,
                        struct npu_event **event_out)
{
    npu_runtime_t *runtime;

    if (!context || !id || !event || !event_out)
        return -EINVAL;
    runtime = context->runtime;
    pthread_mutex_lock(&runtime->mutex);
    if (!runtime_context_valid(context)) {
        pthread_mutex_unlock(&runtime->mutex);
        return -ECANCELED;
    }
    event->context = context;
    event->id = id;
    event->cached.state = NPU_EVENT_QUEUED;
    event->next = context->events;
    context->events = event;
    pthread_mutex_unlock(&runtime->mutex);
    *event_out = event;
    return 0;
}

void runtime_event_remove(struct npu_event *event)
{
    struct npu_event **link;
    npu_runtime_t *runtime;

    if (!event || !event->context)
        return;
    runtime = event->context->runtime;
    pthread_mutex_lock(&runtime->mutex);
    link = &event->context->events;
    while (*link && *link != event)
        link = &(*link)->next;
    if (*link)
        *link = event->next;
    pthread_mutex_unlock(&runtime->mutex);
    free(event);
}
