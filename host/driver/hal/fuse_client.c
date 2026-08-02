#define _POSIX_C_SOURCE 200809L

#include "client_internal.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

struct fd_offset_endpoint {
    int h2d_fd;
    int d2h_fd;
};

struct fd_offset_client {
    uint32_t endpoint_count;
    struct fd_offset_endpoint endpoints[];
};

static int full_pread(int fd, uint64_t offset, void *data, size_t size)
{
    unsigned char *cursor = data;
    size_t done = 0;

    if (fd < 0 || (!data && size) || offset > (uint64_t)INT64_MAX ||
        size > (uint64_t)INT64_MAX - offset)
        return -EINVAL;
    while (done < size) {
        ssize_t result = pread(fd, cursor + done, size - done,
                               (off_t)(offset + done));

        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0)
            return result < 0 ? -errno : -EIO;
        done += (size_t)result;
    }
    return 0;
}

static int full_pwrite(int fd, uint64_t offset,
                       const void *data, size_t size)
{
    const unsigned char *cursor = data;
    size_t done = 0;

    if (fd < 0 || (!data && size) || offset > (uint64_t)INT64_MAX ||
        size > (uint64_t)INT64_MAX - offset)
        return -EINVAL;
    while (done < size) {
        ssize_t result = pwrite(fd, cursor + done, size - done,
                                (off_t)(offset + done));

        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0)
            return result < 0 ? -errno : -EIO;
        done += (size_t)result;
    }
    return 0;
}

static int fd_offset_import(const npu_hal_data_export_t *exported,
                            int *handles, size_t handle_count,
                            void **state_out)
{
    struct fd_offset_client *client;
    uint64_t endpoint_count = exported->opaque[0];
    size_t client_size;
    size_t index;

    if (!state_out || !endpoint_count ||
        endpoint_count > NPU_HAL_DATA_MAX_HANDLES / 2U ||
        handle_count != endpoint_count * 2U || exported->opaque[1] ||
        exported->opaque[2] || exported->opaque[3] ||
        endpoint_count > (SIZE_MAX - sizeof(*client)) /
                             sizeof(client->endpoints[0]))
        return -EPROTO;
    client_size = sizeof(*client) +
                  (size_t)endpoint_count * sizeof(client->endpoints[0]);
    client = calloc(1, client_size);
    if (!client)
        return -ENOMEM;
    client->endpoint_count = (uint32_t)endpoint_count;
    for (index = 0; index < endpoint_count; ++index) {
        if (handles[index * 2U] < 0 || handles[index * 2U + 1U] < 0) {
            while (index) {
                --index;
                close(client->endpoints[index].d2h_fd);
                close(client->endpoints[index].h2d_fd);
            }
            free(client);
            return -EBADF;
        }
        client->endpoints[index].h2d_fd = handles[index * 2U];
        client->endpoints[index].d2h_fd = handles[index * 2U + 1U];
        handles[index * 2U] = -1;
        handles[index * 2U + 1U] = -1;
    }
    *state_out = client;
    return 0;
}

static void fd_offset_close(void *opaque)
{
    struct fd_offset_client *client = opaque;
    uint32_t index;

    if (!client)
        return;
    for (index = 0; index < client->endpoint_count; ++index) {
        close(client->endpoints[index].d2h_fd);
        close(client->endpoints[index].h2d_fd);
    }
    free(client);
}

static int fd_offset_transfer(void *opaque,
                              const npu_hal_data_token_t *token,
                              void *host_buffer, size_t size)
{
    struct fd_offset_client *client = opaque;
    uint64_t offset = token->opaque[0];
    uint64_t endpoint = token->opaque[1];

    if (endpoint >= client->endpoint_count || token->opaque[2] ||
        token->opaque[3])
        return -EACCES;
    switch (token->direction) {
    case NPU_HAL_DATA_HOST_TO_DEVICE:
        return full_pwrite(client->endpoints[endpoint].h2d_fd, offset,
                           host_buffer, size);
    case NPU_HAL_DATA_DEVICE_TO_HOST:
        return full_pread(client->endpoints[endpoint].d2h_fd, offset,
                          host_buffer, size);
    default:
        return -EINVAL;
    }
}

const struct npu_hal_data_provider_ops npu_hal_fd_offset_provider = {
    .provider = NPU_HAL_DATA_PROVIDER_FD_OFFSET,
    .import = fd_offset_import,
    .close = fd_offset_close,
    .transfer = fd_offset_transfer,
};
