#include "dataplane_internal.h"

#include <errno.h>
#include <stdlib.h>

struct npu_hal_data_plane {
    const struct npu_hal_data_plane_ops *ops;
    void *state;
};

static const struct npu_hal_data_plane_ops *const providers[] = {
    &npu_hal_xilinx_dma_data_plane_ops,
    &npu_hal_soc_mmap_data_plane_ops,
    &npu_hal_inprocess_data_plane_ops,
};

static const struct npu_hal_data_plane_ops *find_provider(uint32_t provider)
{
    size_t index;

    for (index = 0; index < sizeof(providers) / sizeof(providers[0]); ++index) {
        if (providers[index]->provider == provider)
            return providers[index];
    }
    return 0;
}

int npu_hal_data_plane_import(npu_hal_data_plane_t **data_plane_out,
                              const npu_hal_data_export_t *exported,
                              int *handles, size_t handle_count)
{
    const struct npu_hal_data_plane_ops *ops;
    npu_hal_data_plane_t *data_plane;
    int status;

    if (!data_plane_out || !exported || (!handles && handle_count) ||
        exported->version != NPU_HAL_DATA_EXPORT_VERSION ||
        exported->handle_count != handle_count ||
        handle_count > NPU_HAL_DATA_MAX_HANDLES || exported->flags)
        return -EINVAL;
    *data_plane_out = 0;
    ops = find_provider(exported->provider);
    if (!ops)
        return -ENOTSUP;
    data_plane = calloc(1, sizeof(*data_plane));
    if (!data_plane)
        return -ENOMEM;
    status = ops->import(exported, handles, handle_count, &data_plane->state);
    if (status < 0) {
        free(data_plane);
        return status;
    }
    data_plane->ops = ops;
    *data_plane_out = data_plane;
    return 0;
}

void npu_hal_data_plane_close(npu_hal_data_plane_t *data_plane)
{
    if (!data_plane)
        return;
    data_plane->ops->close(data_plane->state);
    free(data_plane);
}

int npu_hal_data_plane_transfer(npu_hal_data_plane_t *data_plane,
                                const npu_hal_data_token_t *token,
                                void *host_buffer, size_t size)
{
    if (!data_plane || !token || (!host_buffer && size) || !size ||
        token->version != NPU_HAL_DATA_TOKEN_VERSION ||
        token->provider != data_plane->ops->provider || token->size != size ||
        token->flags)
        return -EINVAL;
    return data_plane->ops->transfer(data_plane->state, token, host_buffer,
                                     size);
}
