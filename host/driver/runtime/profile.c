#include "runtime_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(NPU_PROFILE_BATCH_END == NPU_IPC_PROFILE_END,
               "Runtime and IPC profile batch flags");
_Static_assert(NPU_PROFILE_MAX_BATCH_BYTES ==
               NPU_IPC_PROFILE_CHUNK_BYTES,
               "Runtime and IPC profile batch capacity");

void npu_profile_options_init(npu_profile_options_t *options)
{
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->version = NPU_PROFILE_OPTIONS_VERSION;
    options->buffer_size = NPU_PROFILE_DEFAULT_BUFFER_SIZE;
}

static int profile_linked_locked(const npu_profile_t *profile)
{
    const npu_profile_t *current;

    for (current = profile->runtime->profiles; current;
         current = current->next) {
        if (current == profile)
            return 1;
    }
    return 0;
}

int npu_profile_start(npu_runtime_t *runtime,
                      const npu_profile_options_t *options,
                      npu_profile_t **profile_out)
{
    npu_profile_options_t defaults;
    npu_ipc_profile_start_t request;
    npu_profile_t *profile;
    uint64_t id = 0;
    int status;

    if (!runtime || !profile_out)
        return -EINVAL;
    *profile_out = 0;
    if (!options) {
        npu_profile_options_init(&defaults);
        options = &defaults;
    }
    if (options->version != NPU_PROFILE_OPTIONS_VERSION || options->flags ||
        !options->buffer_size)
        return -EINVAL;
    profile = calloc(1, sizeof(*profile));
    if (!profile)
        return -ENOMEM;
    memset(&request, 0, sizeof(request));
    request.buffer_size = options->buffer_size;
    request.flags = options->flags;
    status = runtime_control_call(runtime, NPU_IPC_PROFILE_START, 0,
                                  &request, sizeof(request), 0, 0, &id);
    if (status < 0) {
        free(profile);
        return status;
    }
    if (!id) {
        free(profile);
        return -EPROTO;
    }
    profile->runtime = runtime;
    profile->id = id;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->closing) {
        pthread_mutex_unlock(&runtime->mutex);
        (void)runtime_control_call(runtime, NPU_IPC_PROFILE_STOP, id,
                                   0, 0, 0, 0, 0);
        (void)runtime_control_call(runtime, NPU_IPC_PROFILE_DESTROY, id,
                                   0, 0, 0, 0, 0);
        free(profile);
        return -ECANCELED;
    }
    profile->next = runtime->profiles;
    runtime->profiles = profile;
    pthread_mutex_unlock(&runtime->mutex);
    *profile_out = profile;
    return 0;
}

static int profile_stop(npu_profile_t *profile, int runtime_closing)
{
    npu_runtime_t *runtime;
    int status;

    if (!profile || !profile->runtime)
        return -EINVAL;
    runtime = profile->runtime;
    pthread_mutex_lock(&runtime->mutex);
    if (!profile_linked_locked(profile) || profile->destroying ||
        profile->stopping || (!runtime_closing && runtime->closing)) {
        pthread_mutex_unlock(&runtime->mutex);
        return -EINVAL;
    }
    if (profile->stopped) {
        pthread_mutex_unlock(&runtime->mutex);
        return 0;
    }
    profile->stopping = 1;
    pthread_mutex_unlock(&runtime->mutex);
    status = runtime_control_call(runtime, NPU_IPC_PROFILE_STOP,
                                  profile->id, 0, 0, 0, 0, 0);
    pthread_mutex_lock(&runtime->mutex);
    profile->stopping = 0;
    if (status == 0)
        profile->stopped = 1;
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

int npu_profile_stop(npu_profile_t *profile)
{
    return profile_stop(profile, 0);
}

int npu_profile_read(npu_profile_t *profile, void *data,
                     uint32_t capacity_bytes, uint32_t timeout_ms,
                     npu_profile_batch_info_t *batch_out)
{
    npu_ipc_profile_read_t request;
    const npu_ipc_profile_batch_t *batch;
    npu_runtime_t *runtime;
    void *payload = 0;
    size_t payload_size = 0;
    size_t expected;
    int status;

    if (!profile || !profile->runtime || !data || !capacity_bytes ||
        !batch_out || capacity_bytes > NPU_PROFILE_MAX_BATCH_BYTES)
        return -EINVAL;
    runtime = profile->runtime;
    pthread_mutex_lock(&runtime->mutex);
    if (!profile_linked_locked(profile) || profile->destroying ||
        runtime->closing) {
        pthread_mutex_unlock(&runtime->mutex);
        return -EINVAL;
    }
    pthread_mutex_unlock(&runtime->mutex);
    request.max_bytes = capacity_bytes;
    request.timeout_ms = timeout_ms;
    status = runtime_control_call_variable(
        runtime, NPU_IPC_PROFILE_READ, profile->id, &request,
        sizeof(request), &payload, &payload_size);
    if (status < 0)
        return status;
    if (payload_size < sizeof(*batch)) {
        status = -EPROTO;
        goto out;
    }
    batch = payload;
    expected = sizeof(*batch) + batch->byte_count;
    if (batch->byte_count > capacity_bytes ||
        (batch->flags & ~NPU_IPC_PROFILE_END) ||
        ((batch->flags & NPU_IPC_PROFILE_END) && batch->byte_count) ||
        payload_size != expected) {
        status = -EPROTO;
        goto out;
    }
    batch_out->cluster = batch->cluster;
    batch_out->hart = batch->hart;
    batch_out->byte_count = batch->byte_count;
    batch_out->flags = batch->flags;
    if (batch->byte_count)
        memcpy(data, batch + 1, batch->byte_count);
    status = 0;

out:
    free(payload);
    return status;
}

static int profile_destroy(npu_profile_t *profile, int runtime_closing)
{
    npu_runtime_t *runtime;
    npu_profile_t **link;
    int stop_status;
    int status;

    if (!profile || !profile->runtime)
        return -EINVAL;
    runtime = profile->runtime;
    stop_status = profile_stop(profile, runtime_closing);
    if (stop_status < 0 && !runtime_closing)
        return stop_status;
    pthread_mutex_lock(&runtime->mutex);
    if (!profile_linked_locked(profile) || profile->destroying) {
        pthread_mutex_unlock(&runtime->mutex);
        return -EINVAL;
    }
    profile->destroying = 1;
    pthread_mutex_unlock(&runtime->mutex);
    status = runtime_control_call(runtime, NPU_IPC_PROFILE_DESTROY,
                                  profile->id, 0, 0, 0, 0, 0);
    if (status < 0 && !runtime_closing) {
        pthread_mutex_lock(&runtime->mutex);
        profile->destroying = 0;
        pthread_mutex_unlock(&runtime->mutex);
        return status;
    }
    pthread_mutex_lock(&runtime->mutex);
    link = &runtime->profiles;
    while (*link && *link != profile)
        link = &(*link)->next;
    if (*link)
        *link = profile->next;
    pthread_mutex_unlock(&runtime->mutex);
    free(profile);
    if (stop_status < 0)
        return stop_status;
    return status;
}

int npu_profile_destroy(npu_profile_t *profile)
{
    return profile_destroy(profile, 0);
}

int runtime_profile_close_all(npu_runtime_t *runtime)
{
    int first_status = 0;

    for (;;) {
        npu_profile_t *profile;
        int status;

        pthread_mutex_lock(&runtime->mutex);
        profile = runtime->profiles;
        pthread_mutex_unlock(&runtime->mutex);
        if (!profile)
            break;
        status = profile_destroy(profile, 1);
        if (status < 0 && !first_status)
            first_status = status;
    }
    return first_status;
}
