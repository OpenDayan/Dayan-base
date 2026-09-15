#include "runtime_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define RUNTIME_OUTPUT_POLL_MS 100U

int npu_runtime_get_topology(npu_runtime_t *runtime,
                             npu_runtime_topology_t *topology)
{
    if (!runtime || !topology)
        return -EINVAL;
    return runtime_control_call(runtime, NPU_IPC_TOPOLOGY, 0,
                                0, 0, topology, sizeof(*topology), 0);
}

void npu_context_desc_init(npu_context_desc_t *desc)
{
    if (!desc)
        return;
    memset(desc, 0, sizeof(*desc));
    desc->version = NPU_CONTEXT_DESC_VERSION;
}

void npu_buffer_desc_init(npu_buffer_desc_t *desc, uint64_t size)
{
    if (!desc)
        return;
    memset(desc, 0, sizeof(*desc));
    desc->version = NPU_BUFFER_DESC_VERSION;
    desc->flags = NPU_BUFFER_READ | NPU_BUFFER_WRITE;
    desc->size = size;
    desc->alignment = 64;
}

void npu_stream_desc_init(npu_stream_desc_t *desc)
{
    if (!desc)
        return;
    memset(desc, 0, sizeof(*desc));
    desc->version = NPU_STREAM_DESC_VERSION;
}

void npu_launch_config_init(NPULaunchConfig *config)
{
    if (!config)
        return;
    memset(config, 0, sizeof(*config));
    config->task_dim.x = 1;
    config->task_dim.y = 1;
    config->task_dim.z = 1;
#ifdef __cplusplus
    config->task_mode = NPUTaskMode::kSingleCore;
#else
    config->task_mode = NPU_TASK_MODE_SINGLE_CORE;
#endif
}

void npu_launch_desc_init(npu_launch_desc_t *desc, const char *task_name)
{
    size_t length;

    if (!desc)
        return;
    memset(desc, 0, sizeof(*desc));
    desc->version = NPU_LAUNCH_DESC_VERSION;
    npu_launch_config_init(&desc->config);
    if (!task_name || !task_name[0])
        return;
    length = strnlen(task_name, sizeof(desc->task_name));
    if (length == sizeof(desc->task_name))
        return;
    memcpy(desc->task_name, task_name, length + 1U);
}

void npu_kernel_arg_value(npu_kernel_arg_t *arg, uint64_t value,
                          uint32_t size)
{
    if (!arg)
        return;
    memset(arg, 0, sizeof(*arg));
    arg->kind = NPU_KERNEL_ARG_VALUE;
    arg->size = size;
    arg->data.value = value;
}

void npu_kernel_arg_buffer(npu_kernel_arg_t *arg, npu_buffer_t *buffer,
                           uint64_t offset, uint32_t access)
{
    if (!arg)
        return;
    memset(arg, 0, sizeof(*arg));
    arg->kind = NPU_KERNEL_ARG_BUFFER;
    arg->access = access;
    arg->size = sizeof(uint64_t);
    arg->data.buffer.buffer = buffer;
    arg->data.buffer.offset = offset;
}

int npu_context_create(npu_runtime_t *runtime,
                       const npu_context_desc_t *desc,
                       npu_context_t **context_out)
{
    npu_context_desc_t defaults;
    npu_ipc_context_create_t request;
    struct npu_context *context;
    uint64_t id;
    int status;

    if (!runtime || !context_out)
        return -EINVAL;
    *context_out = 0;
    if (!desc) {
        npu_context_desc_init(&defaults);
        desc = &defaults;
    }
    if (desc->version != NPU_CONTEXT_DESC_VERSION || desc->flags ||
        desc->reserved)
        return -EINVAL;
    context = calloc(1, sizeof(*context));
    if (!context)
        return -ENOMEM;
    memset(&request, 0, sizeof(request));
    request.memory_quota = desc->memory_quota;
    request.max_pending_commands = desc->max_pending_commands;
    request.flags = desc->flags;
    status = runtime_control_call(runtime, NPU_IPC_CONTEXT_CREATE, 0,
                                  &request, sizeof(request), 0, 0, &id);
    if (status < 0) {
        free(context);
        return status;
    }
    context->runtime = runtime;
    context->id = id;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->closing) {
        pthread_mutex_unlock(&runtime->mutex);
        (void)runtime_control_call(runtime, NPU_IPC_CONTEXT_DESTROY, id,
                                   0, 0, 0, 0, 0);
        free(context);
        return -ECANCELED;
    }
    context->next = runtime->contexts;
    runtime->contexts = context;
    pthread_mutex_unlock(&runtime->mutex);
    *context_out = context;
    return 0;
}

static void free_context_objects(struct npu_context *context)
{
    struct npu_buffer *buffer = context->buffers;
    struct npu_stream *stream = context->streams;
    struct npu_module *module = context->modules;
    struct npu_event *event = context->events;

    while (buffer) {
        struct npu_buffer *next = buffer->next;
        free(buffer);
        buffer = next;
    }
    while (stream) {
        struct npu_stream *next = stream->next;
        free(stream);
        stream = next;
    }
    while (module) {
        struct npu_module *next = module->next;
        struct npu_kernel *kernel = module->kernels;

        while (kernel) {
            struct npu_kernel *kernel_next = kernel->next;
            free(kernel);
            kernel = kernel_next;
        }
        free(module);
        module = next;
    }
    while (event) {
        struct npu_event *next = event->next;
        free(event);
        event = next;
    }
}

static int context_destroy(npu_context_t *context, int runtime_is_closing)
{
    npu_runtime_t *runtime;
    struct npu_context **link;
    int status;

    if (!context || !context->runtime)
        return -EINVAL;
    runtime = context->runtime;
    pthread_mutex_lock(&runtime->mutex);
    if (context->closing || (!runtime_is_closing && runtime->closing)) {
        pthread_mutex_unlock(&runtime->mutex);
        return -EINVAL;
    }
    context->closing = 1;
    pthread_mutex_unlock(&runtime->mutex);
    status = runtime_control_call(runtime, NPU_IPC_CONTEXT_DESTROY,
                                  context->id, 0, 0, 0, 0, 0);
    if (status < 0 && runtime->data_fd >= 0)
        (void)shutdown(runtime->data_fd, SHUT_RDWR);
    pthread_mutex_lock(&runtime->mutex);
    while (context->transfer_jobs)
        (void)pthread_cond_wait(&runtime->condition, &runtime->mutex);
    link = &runtime->contexts;
    while (*link && *link != context)
        link = &(*link)->next;
    if (*link)
        *link = context->next;
    pthread_mutex_unlock(&runtime->mutex);
    runtime_kernel_descriptors_forget_context(context);
    free_context_objects(context);
    free(context);
    return status;
}

int npu_context_destroy(npu_context_t *context)
{
    return context_destroy(context, 0);
}

int npu_runtime_close(npu_runtime_t *runtime)
{
    int first_status = 0;

    if (!runtime)
        return -EINVAL;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->closing) {
        pthread_mutex_unlock(&runtime->mutex);
        return -EINVAL;
    }
    runtime->closing = 1;
    pthread_mutex_unlock(&runtime->mutex);
    {
        int status = runtime_profile_close_all(runtime);

        if (status < 0)
            first_status = status;
    }
    for (;;) {
        struct npu_context *context;
        int status;

        pthread_mutex_lock(&runtime->mutex);
        context = runtime->contexts;
        pthread_mutex_unlock(&runtime->mutex);
        if (!context)
            break;
        status = context_destroy(context, 1);
        if (status < 0 && !first_status)
            first_status = status;
    }
    pthread_mutex_lock(&runtime->mutex);
    runtime->data_worker_stop = 1;
    pthread_cond_broadcast(&runtime->condition);
    pthread_mutex_unlock(&runtime->mutex);
    if (runtime->data_worker_started)
        (void)pthread_join(runtime->data_worker, 0);
    npu_hal_data_plane_close(runtime->data_plane);
    if (runtime->data_fd >= 0)
        close(runtime->data_fd);
    if (runtime->control_fd >= 0)
        close(runtime->control_fd);
    pthread_cond_destroy(&runtime->condition);
    pthread_mutex_destroy(&runtime->mutex);
    pthread_mutex_destroy(&runtime->data_mutex);
    pthread_mutex_destroy(&runtime->control_mutex);
    free(runtime);
    return first_status;
}

int npu_buffer_create(npu_context_t *context, const npu_buffer_desc_t *desc,
                      npu_buffer_t **buffer_out)
{
    npu_buffer_desc_t defaults;
    npu_ipc_buffer_create_t request;
    struct npu_buffer *buffer;
    npu_runtime_t *runtime;
    uint64_t id;
    int status;

    if (!context || !buffer_out)
        return -EINVAL;
    *buffer_out = 0;
    if (!desc) {
        npu_buffer_desc_init(&defaults, 0);
        desc = &defaults;
    }
    if (!runtime_context_valid(context) ||
        desc->version != NPU_BUFFER_DESC_VERSION || !desc->size ||
        !desc->alignment || (desc->alignment & (desc->alignment - 1U)))
        return -EINVAL;
    buffer = calloc(1, sizeof(*buffer));
    if (!buffer)
        return -ENOMEM;
    memset(&request, 0, sizeof(request));
    request.context_id = context->id;
    request.size = desc->size;
    request.alignment = desc->alignment;
    request.flags = desc->flags;
    runtime = context->runtime;
    status = runtime_control_call(runtime, NPU_IPC_BUFFER_CREATE, 0,
                                  &request, sizeof(request), 0, 0, &id);
    if (status < 0) {
        free(buffer);
        return status;
    }
    buffer->context = context;
    buffer->id = id;
    buffer->size = desc->size;
    pthread_mutex_lock(&runtime->mutex);
    if (context->closing) {
        pthread_mutex_unlock(&runtime->mutex);
        (void)runtime_control_call(runtime, NPU_IPC_BUFFER_DESTROY, id,
                                   0, 0, 0, 0, 0);
        free(buffer);
        return -ECANCELED;
    }
    buffer->next = context->buffers;
    context->buffers = buffer;
    pthread_mutex_unlock(&runtime->mutex);
    *buffer_out = buffer;
    return 0;
}

int npu_buffer_create_view(npu_buffer_t *buffer, uint64_t offset,
                           uint64_t size, npu_buffer_t **view_out)
{
    struct npu_buffer *view;

    if (!buffer || !view_out || !runtime_context_valid(buffer->context) ||
        !size || offset > buffer->size || size > buffer->size - offset)
        return -EINVAL;
    *view_out = 0;
    view = calloc(1, sizeof(*view));
    if (!view)
        return -ENOMEM;
    view->context = buffer->context;
    /*
     * @buffer is either the root or a view of it, and a view already carries
     * the root's Server id and an absolute @base_offset, so both cases flatten
     * the same way.  Nothing walks to the root: it may already have been
     * destroyed, which is exactly what made the old parent pointer unsafe.
     */
    view->id = buffer->id;
    view->size = size;
    view->base_offset = buffer->base_offset + offset;
    if (view->base_offset < buffer->base_offset) {
        free(view);
        return -EOVERFLOW;
    }
    view->is_view = 1;
    *view_out = view;
    return 0;
}

int npu_buffer_destroy(npu_buffer_t *buffer)
{
    struct npu_buffer **link;
    npu_runtime_t *runtime;
    int status;

    if (!buffer || !runtime_context_valid(buffer->context))
        return -EINVAL;
    if (buffer->is_view) {
        free(buffer);
        return 0;
    }
    runtime = buffer->context->runtime;
    status = runtime_control_call(runtime, NPU_IPC_BUFFER_DESTROY, buffer->id,
                                  0, 0, 0, 0, 0);
    if (status < 0)
        return status;
    pthread_mutex_lock(&runtime->mutex);
    link = &buffer->context->buffers;
    while (*link && *link != buffer)
        link = &(*link)->next;
    if (*link)
        *link = buffer->next;
    pthread_mutex_unlock(&runtime->mutex);
    free(buffer);
    return 0;
}

int npu_buffer_get_size(const npu_buffer_t *buffer, uint64_t *size_out)
{
    if (!buffer || !size_out || !runtime_context_valid(buffer->context))
        return -EINVAL;
    *size_out = buffer->size;
    return 0;
}

int npu_stream_create(npu_context_t *context, const npu_stream_desc_t *desc,
                      npu_stream_t **stream_out)
{
    npu_stream_desc_t defaults;
    npu_ipc_stream_create_t request;
    struct npu_stream *stream;
    npu_runtime_t *runtime;
    uint64_t id;
    int status;

    if (!context || !stream_out)
        return -EINVAL;
    *stream_out = 0;
    if (!desc) {
        npu_stream_desc_init(&defaults);
        desc = &defaults;
    }
    if (!runtime_context_valid(context) ||
        desc->version != NPU_STREAM_DESC_VERSION || desc->flags ||
        desc->reserved)
        return -EINVAL;
    stream = calloc(1, sizeof(*stream));
    if (!stream)
        return -ENOMEM;
    request.context_id = context->id;
    request.priority = desc->priority;
    request.flags = desc->flags;
    runtime = context->runtime;
    status = runtime_control_call(runtime, NPU_IPC_STREAM_CREATE, 0,
                                  &request, sizeof(request), 0, 0, &id);
    if (status < 0) {
        free(stream);
        return status;
    }
    stream->context = context;
    stream->id = id;
    pthread_mutex_lock(&runtime->mutex);
    if (context->closing) {
        pthread_mutex_unlock(&runtime->mutex);
        (void)runtime_control_call(runtime, NPU_IPC_STREAM_DESTROY, id,
                                   0, 0, 0, 0, 0);
        free(stream);
        return -ECANCELED;
    }
    stream->next = context->streams;
    context->streams = stream;
    pthread_mutex_unlock(&runtime->mutex);
    *stream_out = stream;
    return 0;
}

int npu_stream_destroy(npu_stream_t *stream)
{
    struct npu_stream **link;
    npu_runtime_t *runtime;
    int status;

    if (!runtime_stream_valid(stream))
        return -EINVAL;
    runtime = stream->context->runtime;
    status = runtime_control_call(runtime, NPU_IPC_STREAM_DESTROY, stream->id,
                                  0, 0, 0, 0, 0);
    if (status < 0)
        return status;
    pthread_mutex_lock(&runtime->mutex);
    link = &stream->context->streams;
    while (*link && *link != stream)
        link = &(*link)->next;
    if (*link)
        *link = stream->next;
    pthread_mutex_unlock(&runtime->mutex);
    free(stream);
    return 0;
}

static uint64_t monotonic_milliseconds(void)
{
    struct timespec now;

    (void)clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000U + (uint64_t)now.tv_nsec / 1000000U;
}

static int runtime_wait(npu_runtime_t *runtime, uint16_t opcode,
                        uint64_t object_id, uint64_t event_id,
                        uint32_t timeout_ms, void *response,
                        size_t response_size)
{
    npu_ipc_event_wait_t request;
    uint64_t deadline = timeout_ms == NPU_TIMEOUT_INFINITE ? 0 :
                        monotonic_milliseconds() + timeout_ms;

    for (;;) {
        uint32_t slice = timeout_ms;
        int status;

        if (timeout_ms == NPU_TIMEOUT_INFINITE) {
            slice = RUNTIME_OUTPUT_POLL_MS;
        } else if (timeout_ms) {
            uint64_t now = monotonic_milliseconds();
            uint64_t remaining;

            if (now >= deadline)
                return -ETIMEDOUT;
            remaining = deadline - now;
            slice = remaining < RUNTIME_OUTPUT_POLL_MS ?
                    (uint32_t)remaining : RUNTIME_OUTPUT_POLL_MS;
        }
        memset(&request, 0, sizeof(request));
        request.event_id = event_id;
        request.timeout_ms = slice;
        status = runtime_control_call(runtime, opcode, object_id,
                                      &request, sizeof(request), response,
                                      response_size, 0);
        if (status != -ETIMEDOUT || !timeout_ms)
            return status;
        if (timeout_ms != NPU_TIMEOUT_INFINITE &&
            monotonic_milliseconds() >= deadline)
            return -ETIMEDOUT;
    }
}

int npu_stream_synchronize(npu_stream_t *stream, uint32_t timeout_ms)
{
    if (!runtime_stream_valid(stream))
        return -EINVAL;
    return runtime_wait(stream->context->runtime,
                        NPU_IPC_STREAM_SYNCHRONIZE, stream->id, 0,
                        timeout_ms, 0, 0);
}

int npu_stream_wait_event(npu_stream_t *stream, npu_event_t *event)
{
    npu_ipc_stream_wait_t request;

    if (!runtime_stream_valid(stream) || !event ||
        event->context != stream->context)
        return -EINVAL;
    memset(&request, 0, sizeof(request));
    request.stream_id = stream->id;
    request.event_id = event->id;
    return runtime_control_call(stream->context->runtime,
                                NPU_IPC_STREAM_WAIT_EVENT, 0, &request,
                                sizeof(request), 0, 0, 0);
}

int npu_event_record(npu_stream_t *stream, npu_event_t **event_out)
{
    struct npu_event *event;
    uint64_t id;
    int status;

    if (!runtime_stream_valid(stream) || !event_out)
        return -EINVAL;
    *event_out = 0;
    event = calloc(1, sizeof(*event));
    if (!event)
        return -ENOMEM;
    status = runtime_control_call(stream->context->runtime,
                                  NPU_IPC_EVENT_RECORD, stream->id,
                                  0, 0, 0, 0, &id);
    if (status < 0) {
        free(event);
        return status;
    }
    status = runtime_event_adopt(stream->context, id, event, event_out);
    if (status < 0)
        free(event);
    return status;
}

static void copy_event_info(npu_event_info_t *destination,
                            const npu_ipc_event_info_t *source)
{
    destination->state = source->state;
    destination->command_type = source->command_type;
    destination->status = source->status;
    destination->sequence = source->sequence;
    destination->cookie = source->cookie;
}

int npu_event_query(npu_event_t *event, npu_event_info_t *info)
{
    npu_ipc_event_info_t response;
    int status;

    if (!event || !info || !runtime_context_valid(event->context))
        return -EINVAL;
    status = runtime_control_call(event->context->runtime,
                                  NPU_IPC_EVENT_QUERY, event->id,
                                  0, 0, &response, sizeof(response), 0);
    if (status < 0)
        return status;
    copy_event_info(&event->cached, &response);
    *info = event->cached;
    return 0;
}

int npu_event_wait(npu_event_t *event, uint32_t timeout_ms,
                   npu_event_info_t *info)
{
    npu_ipc_event_info_t response;
    int status;

    if (!event || !runtime_context_valid(event->context))
        return -EINVAL;
    status = runtime_wait(event->context->runtime, NPU_IPC_EVENT_WAIT,
                          event->id, event->id, timeout_ms, &response,
                          sizeof(response));
    if (status < 0)
        return status;
    copy_event_info(&event->cached, &response);
    if (info)
        *info = event->cached;
    return 0;
}

int npu_event_destroy(npu_event_t *event)
{
    int status;

    if (!event || !runtime_context_valid(event->context))
        return -EINVAL;
    status = runtime_control_call(event->context->runtime,
                                  NPU_IPC_EVENT_DESTROY, event->id,
                                  0, 0, 0, 0, 0);
    if (status < 0)
        return status;
    runtime_event_remove(event);
    return 0;
}
