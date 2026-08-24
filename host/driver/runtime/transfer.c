#include "runtime_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

int runtime_transfer_enqueue(struct runtime_transfer_job *job)
{
    npu_runtime_t *runtime;

    if (!job || !job->context || !job->event || !job->size)
        return -EINVAL;
    runtime = job->context->runtime;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->data_worker_stop) {
        pthread_mutex_unlock(&runtime->mutex);
        return -ECANCELED;
    }
    job->next = 0;
    if (runtime->job_tail)
        runtime->job_tail->next = job;
    else
        runtime->job_head = job;
    runtime->job_tail = job;
    ++job->context->transfer_jobs;
    pthread_cond_broadcast(&runtime->condition);
    pthread_mutex_unlock(&runtime->mutex);
    return 0;
}

void *runtime_data_worker(void *argument)
{
    npu_runtime_t *runtime = argument;

    for (;;) {
        struct runtime_transfer_job *job;
        npu_ipc_transfer_grant_t grant;
        npu_ipc_transfer_complete_t complete;
        int status;

        pthread_mutex_lock(&runtime->mutex);
        while (!runtime->job_head && !runtime->data_worker_stop)
            (void)pthread_cond_wait(&runtime->condition, &runtime->mutex);
        if (!runtime->job_head && runtime->data_worker_stop) {
            pthread_mutex_unlock(&runtime->mutex);
            break;
        }
        job = runtime->job_head;
        runtime->job_head = job->next;
        if (!runtime->job_head)
            runtime->job_tail = 0;
        pthread_mutex_unlock(&runtime->mutex);

        memset(&grant, 0, sizeof(grant));
        status = runtime_data_call(runtime, NPU_IPC_TRANSFER_ACQUIRE,
                                   job->event->id, 0, 0, &grant,
                                   sizeof(grant));
        if (status == 0) {
            status = npu_hal_data_plane_transfer(runtime->data_plane,
                                                 &grant.token,
                                                 job->host_buffer,
                                                 job->size);
            memset(&complete, 0, sizeof(complete));
            complete.event_id = grant.event_id;
            complete.transfer_id = grant.transfer_id;
            complete.transfer_status = status;
            complete.token = grant.token;
            {
                int complete_status = runtime_data_call(
                    runtime, NPU_IPC_TRANSFER_COMPLETE, 0, &complete,
                    sizeof(complete), 0, 0);

                if (status == 0 && complete_status < 0)
                    status = complete_status;
            }
        }
        (void)status;
        pthread_mutex_lock(&runtime->mutex);
        if (job->context->transfer_jobs)
            --job->context->transfer_jobs;
        pthread_cond_broadcast(&runtime->condition);
        pthread_mutex_unlock(&runtime->mutex);
        free(job);
    }
    return 0;
}

static int submit_host_transfer(npu_stream_t *stream,
                                npu_buffer_t *destination,
                                uint64_t destination_offset,
                                npu_buffer_t *source,
                                uint64_t source_offset,
                                void *host_buffer, size_t size,
                                uint32_t direction,
                                npu_event_t **event_out)
{
    npu_ipc_transfer_submit_t request;
    struct runtime_transfer_job *job;
    struct npu_event *event;
    uint64_t id;
    int status;

    if (!runtime_stream_valid(stream) || !event_out || !host_buffer || !size ||
        (destination && destination->context != stream->context) ||
        (source && source->context != stream->context) ||
        (destination && (destination_offset > destination->size ||
                         size > destination->size - destination_offset)) ||
        (source && (source_offset > source->size ||
                    size > source->size - source_offset)))
        return -EINVAL;
    *event_out = 0;
    event = calloc(1, sizeof(*event));
    job = calloc(1, sizeof(*job));
    if (!event || !job) {
        free(event);
        free(job);
        return -ENOMEM;
    }
    memset(&request, 0, sizeof(request));
    request.context_id = stream->context->id;
    request.stream_id = stream->id;
    request.destination_buffer = destination ? destination->id : 0;
    request.destination_offset = destination ? destination->base_offset +
                                                destination_offset : 0;
    request.source_buffer = source ? source->id : 0;
    request.source_offset = source ? source->base_offset + source_offset : 0;
    request.size = size;
    request.direction = direction;
    status = runtime_control_call(stream->context->runtime,
                                  NPU_IPC_TRANSFER_SUBMIT, 0, &request,
                                  sizeof(request), 0, 0, &id);
    if (status < 0)
        goto fail;
    status = runtime_event_adopt(stream->context, id, event, event_out);
    if (status < 0)
        goto fail_submitted;
    job->context = stream->context;
    job->event = event;
    job->host_buffer = host_buffer;
    job->size = size;
    status = runtime_transfer_enqueue(job);
    if (status < 0) {
        runtime_event_remove(event);
        event = 0;
        *event_out = 0;
        goto fail_submitted;
    }
    return 0;

fail_submitted:
    /* Context cancellation is the only safe way to revoke an unowned event. */
    (void)runtime_control_call(stream->context->runtime,
                               NPU_IPC_CONTEXT_DESTROY,
                               stream->context->id, 0, 0, 0, 0, 0);
fail:
    free(event);
    free(job);
    return status;
}

int npu_memcpy_h2d_async(npu_stream_t *stream, npu_buffer_t *destination,
                         uint64_t destination_offset, const void *source,
                         size_t size, npu_event_t **event_out)
{
    if (!destination || !source)
        return -EINVAL;
    return submit_host_transfer(stream, destination, destination_offset, 0, 0,
                                (void *)source, size, NPU_IPC_TRANSFER_H2D,
                                event_out);
}

int npu_memcpy_d2h_async(npu_stream_t *stream, void *destination,
                         npu_buffer_t *source, uint64_t source_offset,
                         size_t size, npu_event_t **event_out)
{
    if (!destination || !source)
        return -EINVAL;
    return submit_host_transfer(stream, 0, 0, source, source_offset,
                                destination, size, NPU_IPC_TRANSFER_D2H,
                                event_out);
}

int npu_memcpy_d2d_async(npu_stream_t *stream, npu_buffer_t *destination,
                         uint64_t destination_offset, npu_buffer_t *source,
                         uint64_t source_offset, size_t size,
                         npu_event_t **event_out)
{
    npu_ipc_transfer_submit_t request;
    struct npu_event *event;
    uint64_t id;
    int status;

    if (!runtime_stream_valid(stream) || !destination || !source || !size ||
        !event_out || destination->context != stream->context ||
        source->context != stream->context ||
        destination_offset > destination->size ||
        size > destination->size - destination_offset ||
        source_offset > source->size || size > source->size - source_offset)
        return -EINVAL;
    *event_out = 0;
    event = calloc(1, sizeof(*event));
    if (!event)
        return -ENOMEM;
    memset(&request, 0, sizeof(request));
    request.context_id = stream->context->id;
    request.stream_id = stream->id;
    request.destination_buffer = destination->id;
    request.destination_offset = destination->base_offset +
                                 destination_offset;
    request.source_buffer = source->id;
    request.source_offset = source->base_offset + source_offset;
    request.size = size;
    request.direction = NPU_IPC_TRANSFER_D2D;
    status = runtime_control_call(stream->context->runtime,
                                  NPU_IPC_TRANSFER_SUBMIT, 0, &request,
                                  sizeof(request), 0, 0, &id);
    if (status < 0) {
        free(event);
        return status;
    }
    status = runtime_event_adopt(stream->context, id, event, event_out);
    if (status < 0) {
        free(event);
        (void)runtime_control_call(stream->context->runtime,
                                   NPU_IPC_CONTEXT_DESTROY,
                                   stream->context->id, 0, 0, 0, 0, 0);
    }
    return status;
}
