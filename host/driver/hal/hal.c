#define _POSIX_C_SOURCE 200809L

#include "hal_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "platform_regs.h"
#include "rpc.h"

#define NPU_HAL_COPY_CHUNK       (64U * 1024U)
#define NPU_IRQ_STATUS_OFFSET    0x00U
#define NPU_IRQ_CLEAR_OFFSET     0x0cU
#define NPU_IRQ_MAILBOX_OFFSET   0x20U

void npu_hal_options_init(npu_hal_options_t *options)
{
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->version = NPU_HAL_OPTIONS_VERSION;
    options->backend = NPU_HAL_BACKEND_FUSE_XDMA;
    options->cluster_count = 1;
    options->cores_per_cluster = 4;
    options->cluster_device_stride = 1;
}

void npu_hal_memory_request_init(npu_hal_memory_request_t *request)
{
    if (!request)
        return;
    memset(request, 0, sizeof(*request));
    request->version = NPU_HAL_MEMORY_REQUEST_VERSION;
    request->memory_class = NPU_HAL_MEMORY_NORMAL;
    request->scope = NPU_HAL_MEMORY_DEVICE;
    request->usage = NPU_HAL_MEMORY_BUFFER;
    request->location.cluster = NPU_HAL_ANY_CLUSTER;
    request->location.core = NPU_HAL_ANY_CORE;
    request->alignment = 64;
}

static int options_valid(const npu_hal_options_t *options)
{
    return options && options->version == NPU_HAL_OPTIONS_VERSION &&
           options->cluster_count &&
           options->cluster_count <= NPU_HAL_MAX_CLUSTERS &&
           options->cores_per_cluster &&
           options->cores_per_cluster <= NPU_HAL_MAX_CORES_PER_CLUSTER &&
           options->cluster_device_stride && !options->reserved &&
           (!options->memory_region_count || options->memory_regions);
}

int npu_hal_open(npu_hal_t **hal_out, const npu_hal_options_t *options)
{
    npu_hal_options_t defaults;
    npu_hal_t *hal;
    int mutex_initialized = 0;
    int status;

    if (!hal_out)
        return -EINVAL;
    *hal_out = 0;
    if (!options) {
        npu_hal_options_init(&defaults);
        options = &defaults;
    }
    if (!options_valid(options))
        return -EINVAL;
    hal = calloc(1, sizeof(*hal));
    if (!hal)
        return -ENOMEM;
    hal->backend = options->backend;
    hal->cluster_count = options->cluster_count;
    hal->cores_per_cluster = options->cores_per_cluster;
    status = pthread_mutex_init(&hal->memory_mutex, 0);
    if (status != 0) {
        status = -status;
        goto fail;
    }
    mutex_initialized = 1;
    status = npu_hal_memory_manager_init(hal, options);
    if (status < 0)
        goto fail;
    switch (options->backend) {
    case NPU_HAL_BACKEND_FUSE_XDMA:
        status = npu_hal_fuse_open(options, &hal->ops, &hal->state);
        break;
    default:
        status = -ENOTSUP;
        break;
    }
    if (status < 0)
        goto fail;
    *hal_out = hal;
    return 0;

fail:
    npu_hal_memory_manager_destroy(hal);
    if (mutex_initialized)
        pthread_mutex_destroy(&hal->memory_mutex);
    free(hal);
    return status;
}

void npu_hal_close(npu_hal_t *hal)
{
    if (!hal)
        return;
    if (hal->ops && hal->ops->close)
        hal->ops->close(hal->state);
    npu_hal_memory_manager_destroy(hal);
    pthread_mutex_destroy(&hal->memory_mutex);
    free(hal);
}

int npu_hal_get_info(npu_hal_t *hal, npu_hal_info_t *info)
{
    int status;

    if (!hal || !info || !hal->ops->get_info)
        return -EINVAL;
    memset(info, 0, sizeof(*info));
    status = hal->ops->get_info(hal->state, info);
    if (status < 0)
        return status;
    if (!info->data_provider ||
        info->data_handle_count > NPU_HAL_DATA_MAX_HANDLES ||
        !info->max_transfer_size || !info->transfer_alignment ||
        (info->transfer_alignment & (info->transfer_alignment - 1U))) {
        memset(info, 0, sizeof(*info));
        return -EPROTO;
    }
    info->abi_version = NPU_HAL_ABI_VERSION;
    info->backend = hal->backend;
    info->cluster_count = hal->cluster_count;
    info->cores_per_cluster = hal->cores_per_cluster;
    info->memory_region_count = hal->arena_count;
    return 0;
}

int npu_hal_get_memory_region(npu_hal_t *hal, uint32_t index,
                              npu_hal_memory_region_t *region)
{
    if (!hal || !region || index >= hal->arena_count)
        return -EINVAL;
    *region = hal->arenas[index].region;
    return 0;
}

int npu_hal_core_valid(npu_hal_t *hal, npu_hal_core_t core)
{
    return hal && core.cluster < hal->cluster_count &&
           core.core < hal->cores_per_cluster;
}

int npu_hal_copy_to_address(npu_hal_t *hal, uint32_t cluster,
                            uint64_t address, const void *source, size_t size)
{
    if (!hal || cluster >= hal->cluster_count || (!source && size) ||
        !hal->ops->copy_to_address)
        return -EINVAL;
    return size ? hal->ops->copy_to_address(hal->state, cluster, address,
                                             source, size) : 0;
}

int npu_hal_copy_from_address(npu_hal_t *hal, uint32_t cluster,
                              void *destination, uint64_t address,
                              size_t size)
{
    if (!hal || cluster >= hal->cluster_count || (!destination && size) ||
        !hal->ops->copy_from_address)
        return -EINVAL;
    return size ? hal->ops->copy_from_address(hal->state, cluster,
                                               destination, address, size) : 0;
}

int npu_hal_copy_address(npu_hal_t *hal, uint32_t cluster,
                         uint64_t destination, uint64_t source, size_t size)
{
    unsigned char *buffer;
    size_t done = 0;
    int status = 0;

    if (!hal || cluster >= hal->cluster_count)
        return -EINVAL;
    if (!size || destination == source)
        return 0;
    buffer = malloc(size < NPU_HAL_COPY_CHUNK ? size : NPU_HAL_COPY_CHUNK);
    if (!buffer)
        return -ENOMEM;
    while (done < size) {
        size_t chunk = size - done;

        if (chunk > NPU_HAL_COPY_CHUNK)
            chunk = NPU_HAL_COPY_CHUNK;
        status = npu_hal_copy_from_address(hal, cluster, buffer,
                                           source + done, chunk);
        if (status < 0)
            break;
        status = npu_hal_copy_to_address(hal, cluster, destination + done,
                                         buffer, chunk);
        if (status < 0)
            break;
        done += chunk;
    }
    free(buffer);
    return status;
}

int npu_hal_register_read(npu_hal_t *hal, uint32_t cluster, uint64_t offset,
                          void *value, size_t size)
{
    if (!hal || cluster >= hal->cluster_count || !value || !size ||
        !hal->ops->register_read)
        return -EINVAL;
    return hal->ops->register_read(hal->state, cluster, offset, value, size);
}

int npu_hal_register_write(npu_hal_t *hal, uint32_t cluster, uint64_t offset,
                           const void *value, size_t size)
{
    if (!hal || cluster >= hal->cluster_count || !value || !size ||
        !hal->ops->register_write)
        return -EINVAL;
    return hal->ops->register_write(hal->state, cluster, offset, value, size);
}

int npu_hal_irq_status(npu_hal_t *hal, uint32_t cluster, uint32_t *status)
{
    return npu_hal_register_read(hal, cluster, NPU_IRQ_STATUS_OFFSET,
                                 status, sizeof(*status));
}

int npu_hal_irq_ack(npu_hal_t *hal, uint32_t cluster, uint32_t mask)
{
    return npu_hal_register_write(hal, cluster, NPU_IRQ_CLEAR_OFFSET,
                                  &mask, sizeof(mask));
}

int npu_hal_irq_mailbox(npu_hal_t *hal, npu_hal_core_t core,
                        uint64_t *command)
{
    if (!npu_hal_core_valid(hal, core) || !command)
        return -EINVAL;
    return npu_hal_register_read(hal, core.cluster,
                                 NPU_IRQ_MAILBOX_OFFSET + core.core * 8U,
                                 command, sizeof(*command));
}

int npu_hal_wait_command_consumed(npu_hal_t *hal, npu_hal_core_t core,
                                  uint32_t timeout_ms)
{
    struct timespec delay = { 0, 1000000L };
    struct timespec now;
    uint64_t deadline;

    if (!npu_hal_core_valid(hal, core) ||
        clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return -EINVAL;
    deadline = (uint64_t)now.tv_sec * 1000U +
               (uint64_t)now.tv_nsec / 1000000U + timeout_ms;
    for (;;) {
        uint32_t pending = 0;
        uint64_t current;
        int status = npu_hal_copy_from_address(
            hal, core.cluster, &pending, DAYAN_CLINT_MSIP(core.core),
            sizeof(pending));

        if (status < 0)
            return status;
        if (!pending)
            return 0;
        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
            return -EIO;
        current = (uint64_t)now.tv_sec * 1000U +
                  (uint64_t)now.tv_nsec / 1000000U;
        if (current >= deadline)
            return -ETIMEDOUT;
        (void)nanosleep(&delay, 0);
    }
}

int npu_hal_command_submit(npu_hal_t *hal, npu_hal_core_t core,
                           uint64_t command)
{
    const uint32_t pending = 1;
    int status;

    if (!npu_hal_core_valid(hal, core))
        return -EINVAL;
    status = npu_hal_copy_to_address(hal, core.cluster,
                                     DAYAN_DEVICE_MAILBOX(core.core),
                                     &command, sizeof(command));
    if (status < 0)
        return status;
    return npu_hal_copy_to_address(hal, core.cluster,
                                   DAYAN_CLINT_MSIP(core.core),
                                   &pending, sizeof(pending));
}

int npu_hal_boot_core(npu_hal_t *hal, npu_hal_core_t core, uint64_t entry,
                      uint32_t timeout_ms)
{
    int status;

    if (!npu_hal_core_valid(hal, core) ||
        (entry & ~RPC_COMMAND_ARG_MASK))
        return -EINVAL;
    status = npu_hal_wait_command_consumed(hal, core, timeout_ms);
    if (status < 0)
        return status;
    status = npu_hal_command_submit(hal, core, entry);
    if (status < 0)
        return status;
    return npu_hal_wait_command_consumed(hal, core, timeout_ms);
}

int npu_hal_export_data_plane(npu_hal_t *hal,
                              npu_hal_data_export_t *exported,
                              int handles[NPU_HAL_DATA_MAX_HANDLES])
{
    npu_hal_info_t info;
    uint32_t index;
    int status;

    if (!hal || !exported || !handles || !hal->ops->export_data_plane)
        return -EINVAL;
    memset(exported, 0, sizeof(*exported));
    for (index = 0; index < NPU_HAL_DATA_MAX_HANDLES; ++index)
        handles[index] = -1;
    status = npu_hal_get_info(hal, &info);
    if (status == 0)
        status = hal->ops->export_data_plane(hal->state, exported, handles);
    if (status == 0 &&
        (exported->version != NPU_HAL_DATA_EXPORT_VERSION ||
         exported->provider != info.data_provider || exported->flags ||
         exported->handle_count != info.data_handle_count ||
         exported->handle_count > NPU_HAL_DATA_MAX_HANDLES))
        status = -EPROTO;
    if (status == 0) {
        for (index = 0; index < NPU_HAL_DATA_MAX_HANDLES; ++index) {
            if ((index < exported->handle_count) != (handles[index] >= 0)) {
                status = -EPROTO;
                break;
            }
        }
    }
    if (status < 0) {
        for (index = 0; index < NPU_HAL_DATA_MAX_HANDLES; ++index) {
            if (handles[index] >= 0) {
                close(handles[index]);
                handles[index] = -1;
            }
        }
        memset(exported, 0, sizeof(*exported));
    }
    return status;
}

int npu_hal_prepare_transfer(npu_hal_t *hal,
                             const npu_hal_allocation_t *allocation,
                             npu_hal_core_t viewer, uint64_t offset,
                             size_t size,
                             npu_hal_data_direction_t direction,
                             npu_hal_data_token_t *token)
{
    npu_hal_info_t info;
    uint64_t address;
    int status;

    if (!hal || !allocation || !size || !token ||
        !hal->ops->prepare_transfer || offset > allocation->size ||
        size > allocation->size - offset ||
        (direction != NPU_HAL_DATA_HOST_TO_DEVICE &&
         direction != NPU_HAL_DATA_DEVICE_TO_HOST))
        return -EINVAL;
    status = npu_hal_memory_address(hal, allocation, viewer, &address);
    if (status < 0 || address > UINT64_MAX - offset)
        return status < 0 ? status : -EOVERFLOW;
    address += offset;
    status = npu_hal_get_info(hal, &info);
    if (status < 0)
        return status;
    if (size > info.max_transfer_size)
        return -EMSGSIZE;
    if ((address & (info.transfer_alignment - 1U)) ||
        (size & (info.transfer_alignment - 1U)))
        return -EINVAL;
    memset(token, 0, sizeof(*token));
    status = hal->ops->prepare_transfer(hal->state, viewer.cluster, address,
                                        size, direction, token);
    if (status < 0)
        return status;
    if (token->version != NPU_HAL_DATA_TOKEN_VERSION ||
        token->provider != info.data_provider ||
        token->direction != (uint32_t)direction || token->flags ||
        token->size != size) {
        memset(token, 0, sizeof(*token));
        return -EPROTO;
    }
    return 0;
}

int npu_hal_finish_transfer(npu_hal_t *hal,
                            const npu_hal_data_token_t *token,
                            int transfer_status)
{
    npu_hal_info_t info;
    int status;

    if (!hal || !token || !hal->ops->finish_transfer || transfer_status > 0)
        return -EINVAL;
    status = npu_hal_get_info(hal, &info);
    if (status < 0)
        return status;
    if (token->version != NPU_HAL_DATA_TOKEN_VERSION ||
        token->provider != info.data_provider || token->flags || !token->size)
        return -EINVAL;
    return hal->ops->finish_transfer(hal->state, token, transfer_status);
}
