#define _POSIX_C_SOURCE 200809L

#include "hal_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "platform_regs.h"

#define NPU_FUSE_DEFAULT_PREFIX "/tmp/virtual_npu"
#define NPU_FUSE_CONTROL_BASE   DAYAN_IRQ_STATUS

struct npu_hal_fuse_endpoint {
    int h2c_fd;
    int c2h_fd;
    int user_fd;
    int event_fd;
};

struct npu_hal_fuse {
    uint32_t cluster_count;
    uint64_t memory_address_bias;
    struct npu_hal_fuse_endpoint endpoints[];
};

static int make_path(char *path, size_t capacity, const char *prefix,
                     uint32_t index, const char *suffix)
{
    size_t length = strlen(prefix);
    int result = snprintf(path, capacity, "%s%sxdma%u%s", prefix,
                          length && prefix[length - 1] == '/' ? "" : "/",
                          index, suffix);

    return result > 0 && (size_t)result < capacity ? 0 : -ENAMETOOLONG;
}

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

static void fuse_close(void *opaque)
{
    struct npu_hal_fuse *state = opaque;
    uint32_t cluster;

    if (!state)
        return;
    for (cluster = 0; cluster < state->cluster_count; ++cluster) {
        struct npu_hal_fuse_endpoint *endpoint = &state->endpoints[cluster];

        if (endpoint->event_fd >= 0)
            close(endpoint->event_fd);
        if (endpoint->user_fd >= 0)
            close(endpoint->user_fd);
        if (endpoint->c2h_fd >= 0)
            close(endpoint->c2h_fd);
        if (endpoint->h2c_fd >= 0)
            close(endpoint->h2c_fd);
    }
    free(state);
}

static int fuse_get_info(void *opaque, npu_hal_info_t *info)
{
    const struct npu_hal_fuse *state = opaque;

    if (!state || !info)
        return -EINVAL;
    info->data_provider = NPU_HAL_DATA_PROVIDER_FD_OFFSET;
    info->data_handle_count = state->cluster_count * 2U;
    info->max_transfer_size = (uint64_t)INT64_MAX;
    info->transfer_alignment = 1;
    return 0;
}

static int fuse_copy_to_address(void *opaque, uint32_t cluster,
                                uint64_t address, const void *source,
                                size_t size)
{
    struct npu_hal_fuse *state = opaque;

    if (cluster >= state->cluster_count ||
        address > UINT64_MAX - state->memory_address_bias)
        return -EOVERFLOW;
    return full_pwrite(state->endpoints[cluster].h2c_fd,
                       address + state->memory_address_bias, source, size);
}

static int fuse_copy_from_address(void *opaque, uint32_t cluster,
                                  void *destination, uint64_t address,
                                  size_t size)
{
    struct npu_hal_fuse *state = opaque;

    if (cluster >= state->cluster_count ||
        address > UINT64_MAX - state->memory_address_bias)
        return -EOVERFLOW;
    return full_pread(state->endpoints[cluster].c2h_fd,
                      address + state->memory_address_bias,
                      destination, size);
}

static int fuse_register_read(void *opaque, uint32_t cluster,
                              uint64_t offset, void *value, size_t size)
{
    struct npu_hal_fuse *state = opaque;
    struct npu_hal_fuse_endpoint *endpoint;

    if (cluster >= state->cluster_count)
        return -EINVAL;
    endpoint = &state->endpoints[cluster];
    if (endpoint->user_fd >= 0)
        return full_pread(endpoint->user_fd, offset, value, size);
    return fuse_copy_from_address(state, cluster, value,
                                  NPU_FUSE_CONTROL_BASE + offset, size);
}

static int fuse_register_write(void *opaque, uint32_t cluster,
                               uint64_t offset, const void *value,
                               size_t size)
{
    struct npu_hal_fuse *state = opaque;
    struct npu_hal_fuse_endpoint *endpoint;

    if (cluster >= state->cluster_count)
        return -EINVAL;
    endpoint = &state->endpoints[cluster];
    if (endpoint->user_fd >= 0)
        return full_pwrite(endpoint->user_fd, offset, value, size);
    return fuse_copy_to_address(state, cluster,
                                NPU_FUSE_CONTROL_BASE + offset, value, size);
}

static int fuse_export_data_plane(void *opaque,
                                  npu_hal_data_export_t *exported,
                                  int handles[NPU_HAL_DATA_MAX_HANDLES])
{
    struct npu_hal_fuse *state = opaque;
    uint32_t count = 0;
    uint32_t cluster;

    for (cluster = 0; cluster < state->cluster_count; ++cluster) {
        int h2c = fcntl(state->endpoints[cluster].h2c_fd,
                        F_DUPFD_CLOEXEC, 0);
        int c2h;

        if (h2c < 0)
            goto fail;
        handles[count++] = h2c;
        c2h = fcntl(state->endpoints[cluster].c2h_fd,
                    F_DUPFD_CLOEXEC, 0);
        if (c2h < 0)
            goto fail;
        handles[count++] = c2h;
    }
    memset(exported, 0, sizeof(*exported));
    exported->version = NPU_HAL_DATA_EXPORT_VERSION;
    exported->provider = NPU_HAL_DATA_PROVIDER_FD_OFFSET;
    exported->handle_count = count;
    exported->opaque[0] = state->cluster_count;
    return 0;

fail:
    {
        int status = -errno;

        /* The HAL front end owns and closes every partially exported handle. */
        return status;
    }
}

static int fuse_prepare_transfer(void *opaque, uint32_t cluster,
                                 uint64_t address, size_t size,
                                 npu_hal_data_direction_t direction,
                                 npu_hal_data_token_t *token)
{
    struct npu_hal_fuse *state = opaque;

    if (cluster >= state->cluster_count ||
        address > UINT64_MAX - state->memory_address_bias ||
        address + state->memory_address_bias > (uint64_t)INT64_MAX ||
        size > (uint64_t)INT64_MAX - address - state->memory_address_bias)
        return -EOVERFLOW;
    memset(token, 0, sizeof(*token));
    token->version = NPU_HAL_DATA_TOKEN_VERSION;
    token->provider = NPU_HAL_DATA_PROVIDER_FD_OFFSET;
    token->direction = (uint32_t)direction;
    token->size = size;
    token->opaque[0] = address + state->memory_address_bias;
    token->opaque[1] = cluster;
    return 0;
}

static int fuse_finish_transfer(void *opaque,
                                const npu_hal_data_token_t *token,
                                int transfer_status)
{
    struct npu_hal_fuse *state = opaque;

    if (token->version != NPU_HAL_DATA_TOKEN_VERSION ||
        token->provider != NPU_HAL_DATA_PROVIDER_FD_OFFSET ||
        token->opaque[1] >= state->cluster_count)
        return -EINVAL;
    return transfer_status;
}

static const struct npu_hal_ops fuse_ops = {
    .close = fuse_close,
    .get_info = fuse_get_info,
    .copy_to_address = fuse_copy_to_address,
    .copy_from_address = fuse_copy_from_address,
    .register_read = fuse_register_read,
    .register_write = fuse_register_write,
    .export_data_plane = fuse_export_data_plane,
    .prepare_transfer = fuse_prepare_transfer,
    .finish_transfer = fuse_finish_transfer,
};

static int open_endpoint(struct npu_hal_fuse_endpoint *endpoint,
                         const char *prefix, uint32_t index)
{
    char path[PATH_MAX];
    int status;

    status = make_path(path, sizeof(path), prefix, index, "_h2c_0");
    if (status < 0)
        return status;
    endpoint->h2c_fd = open(path, O_WRONLY | O_CLOEXEC);
    if (endpoint->h2c_fd < 0)
        return -errno;
    status = make_path(path, sizeof(path), prefix, index, "_c2h_0");
    if (status < 0)
        return status;
    endpoint->c2h_fd = open(path, O_RDONLY | O_CLOEXEC);
    if (endpoint->c2h_fd < 0)
        return -errno;
    status = make_path(path, sizeof(path), prefix, index, "_user");
    if (status < 0)
        return status;
    endpoint->user_fd = open(path, O_RDWR | O_CLOEXEC);
    if (endpoint->user_fd < 0 && errno != ENOENT)
        return -errno;
    status = make_path(path, sizeof(path), prefix, index, "_events_0");
    if (status < 0)
        return status;
    endpoint->event_fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    return 0;
}

int npu_hal_fuse_open(const npu_hal_options_t *options,
                      const struct npu_hal_ops **ops_out, void **state_out)
{
    struct npu_hal_fuse *state;
    const char *prefix;
    size_t state_size;
    uint32_t cluster;
    int status = 0;

    if (!options || !ops_out || !state_out || !options->cluster_count ||
        options->cluster_count > NPU_HAL_MAX_CLUSTERS)
        return -EINVAL;
    state_size = sizeof(*state) +
                 options->cluster_count * sizeof(state->endpoints[0]);
    state = calloc(1, state_size);
    if (!state)
        return -ENOMEM;
    state->cluster_count = options->cluster_count;
    state->memory_address_bias = options->memory_address_bias;
    for (cluster = 0; cluster < state->cluster_count; ++cluster) {
        state->endpoints[cluster].h2c_fd = -1;
        state->endpoints[cluster].c2h_fd = -1;
        state->endpoints[cluster].user_fd = -1;
        state->endpoints[cluster].event_fd = -1;
    }
    prefix = options->device_prefix;
    if (!prefix || !*prefix)
        prefix = getenv("NPU_DEVICE_PREFIX");
    if (!prefix || !*prefix)
        prefix = NPU_FUSE_DEFAULT_PREFIX;
    for (cluster = 0; cluster < state->cluster_count; ++cluster) {
        uint64_t index = options->device_index +
                         (uint64_t)cluster * options->cluster_device_stride;

        if (index > UINT32_MAX) {
            status = -EOVERFLOW;
            break;
        }
        status = open_endpoint(&state->endpoints[cluster], prefix,
                               (uint32_t)index);
        if (status < 0)
            break;
    }
    if (status < 0) {
        fuse_close(state);
        return status;
    }
    *ops_out = &fuse_ops;
    *state_out = state;
    return 0;
}
