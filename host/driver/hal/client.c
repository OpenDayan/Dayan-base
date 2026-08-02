#include "client_internal.h"

#include <errno.h>
#include <stdlib.h>

struct npu_hal_data_client {
    const struct npu_hal_data_provider_ops *ops;
    void *state;
};

static const struct npu_hal_data_provider_ops *const providers[] = {
    &npu_hal_fd_offset_provider,
};

static const struct npu_hal_data_provider_ops *find_provider(uint32_t provider)
{
    size_t index;

    for (index = 0; index < sizeof(providers) / sizeof(providers[0]); ++index) {
        if (providers[index]->provider == provider)
            return providers[index];
    }
    return 0;
}

int npu_hal_data_client_import(npu_hal_data_client_t **client_out,
                               const npu_hal_data_export_t *exported,
                               int *handles, size_t handle_count)
{
    const struct npu_hal_data_provider_ops *ops;
    npu_hal_data_client_t *client;
    int status;

    if (!client_out || !exported || (!handles && handle_count) ||
        exported->version != NPU_HAL_DATA_EXPORT_VERSION ||
        exported->handle_count != handle_count ||
        handle_count > NPU_HAL_DATA_MAX_HANDLES || exported->flags)
        return -EINVAL;
    *client_out = 0;
    ops = find_provider(exported->provider);
    if (!ops)
        return -ENOTSUP;
    client = calloc(1, sizeof(*client));
    if (!client)
        return -ENOMEM;
    status = ops->import(exported, handles, handle_count, &client->state);
    if (status < 0) {
        free(client);
        return status;
    }
    client->ops = ops;
    *client_out = client;
    return 0;
}

void npu_hal_data_client_close(npu_hal_data_client_t *client)
{
    if (!client)
        return;
    client->ops->close(client->state);
    free(client);
}

int npu_hal_data_client_transfer(npu_hal_data_client_t *client,
                                 const npu_hal_data_token_t *token,
                                 void *host_buffer, size_t size)
{
    if (!client || !token || (!host_buffer && size) || !size ||
        token->version != NPU_HAL_DATA_TOKEN_VERSION ||
        token->provider != client->ops->provider || token->size != size ||
        token->flags)
        return -EINVAL;
    return client->ops->transfer(client->state, token, host_buffer, size);
}
