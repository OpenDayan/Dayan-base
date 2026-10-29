#include "hal_internal.h"
#include "io.h"
#include "platform_regs.h"

#include <errno.h>
#include <string.h>
#include <unistd.h>

static int inprocess_open(const npu_hal_options_t *options, void **state_out)
{
    const npu_hal_inprocess_options_t *backend =
        (const npu_hal_inprocess_options_t *)options->backend_options;
    struct npu_inprocess_device *device;
    int status;

    if (!backend || backend->base.backend != NPU_HAL_BACKEND_INPROCESS ||
        backend->base.version != NPU_HAL_INPROCESS_OPTIONS_VERSION)
        return -EINVAL;
    status = npu_inprocess_acquire(backend->device_id, (uint64_t)getpid(),
                                  &device);
    if (status < 0)
        return status;
    if (npu_inprocess_clusters(device) != options->cluster_count) {
        npu_inprocess_release(device);
        return -EINVAL;
    }
    *state_out = device;
    return 0;
}

static void inprocess_close(void *state)
{
    npu_inprocess_release(state);
}

static int inprocess_info(void *state, npu_hal_info_t *info)
{
    (void)state;
    info->data_provider = NPU_HAL_DATA_PROVIDER_INPROCESS;
    info->data_handle_count = 0;
    info->max_transfer_size = SIZE_MAX;
    info->transfer_alignment = 1;
    return 0;
}

static int inprocess_read(void *state, uint32_t cluster, void *destination,
                          uint64_t address, size_t size)
{
    return npu_inprocess_read(state, cluster, address, destination, size);
}

static int inprocess_write(void *state, uint32_t cluster, uint64_t address,
                           const void *source, size_t size)
{
    return npu_inprocess_write(state, cluster, address, source, size);
}

static int register_read(void *state, uint32_t cluster, uint64_t offset,
                         void *value, size_t size)
{
    if (offset > UINT64_MAX - DAYAN_IRQ_STATUS)
        return -EOVERFLOW;
    return npu_inprocess_read(state, cluster, DAYAN_IRQ_STATUS + offset,
                              value, size);
}

static int register_write(void *state, uint32_t cluster, uint64_t offset,
                          const void *value, size_t size)
{
    if (offset > UINT64_MAX - DAYAN_IRQ_STATUS)
        return -EOVERFLOW;
    return npu_inprocess_write(state, cluster, DAYAN_IRQ_STATUS + offset,
                               value, size);
}

static int export_data(void *state, npu_hal_data_export_t *exported,
                       int handles[NPU_HAL_DATA_MAX_HANDLES])
{
    (void)handles;
    memset(exported, 0, sizeof(*exported));
    exported->version = NPU_HAL_DATA_EXPORT_VERSION;
    exported->provider = NPU_HAL_DATA_PROVIDER_INPROCESS;
    exported->opaque[0] = npu_inprocess_id(state);
    exported->opaque[1] = (uint64_t)getpid();
    return 0;
}

static int prepare(void *state, uint32_t cluster, uint64_t address,
                    size_t size, npu_hal_data_direction_t direction,
                    npu_hal_data_token_t *token)
{
    if (cluster >= npu_inprocess_clusters(state) ||
        size > UINT64_MAX - address)
        return -EINVAL;
    memset(token, 0, sizeof(*token));
    token->version = NPU_HAL_DATA_TOKEN_VERSION;
    token->provider = NPU_HAL_DATA_PROVIDER_INPROCESS;
    token->direction = direction;
    token->size = size;
    token->opaque[0] = address;
    token->opaque[1] = cluster;
    token->opaque[2] = npu_inprocess_id(state);
    return 0;
}

static int finish(void *state, const npu_hal_data_token_t *token, int status)
{
    if (token->version != NPU_HAL_DATA_TOKEN_VERSION ||
        token->provider != NPU_HAL_DATA_PROVIDER_INPROCESS || token->flags ||
        token->opaque[1] >= npu_inprocess_clusters(state) ||
        token->opaque[2] != npu_inprocess_id(state) || token->opaque[3] ||
        (token->direction != NPU_HAL_DATA_HOST_TO_DEVICE &&
         token->direction != NPU_HAL_DATA_DEVICE_TO_HOST))
        return -EINVAL;
    return status;
}

const struct npu_hal_backend_ops npu_hal_inprocess_backend_ops = {
    .backend = NPU_HAL_BACKEND_INPROCESS,
    .open = inprocess_open,
    .close = inprocess_close,
    .get_info = inprocess_info,
    .copy_to_address = inprocess_write,
    .copy_from_address = inprocess_read,
    .register_read = register_read,
    .register_write = register_write,
    /* Generic HAL IRQ polling uses register_read; no event file is needed. */
    .export_data_plane = export_data,
    .prepare_transfer = prepare,
    .finish_transfer = finish,
};
