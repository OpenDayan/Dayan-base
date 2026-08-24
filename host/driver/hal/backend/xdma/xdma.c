#define _POSIX_C_SOURCE 200809L

#include "hal_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "platform_regs.h"
#include "io.h"
#include "npu_hal_xdma.h"

#define NPU_XDMA_CONTROL_BASE DAYAN_IRQ_STATUS

struct npu_hal_xdma_endpoint {
    int h2c_fd;
    int c2h_fd;
    int user_fd;
    int event_fd;
};

struct npu_hal_xdma {
    uint32_t cluster_count;
    uint32_t event_wait_disabled;
    uint64_t mmio_address_offset;
    struct npu_hal_xdma_endpoint endpoints[];
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

typedef int (*xdma_io_fn)(int fd, uint64_t offset, void *data, size_t size);

static int xdma_read(int fd, uint64_t offset, void *data, size_t size)
{
    return npu_hal_xdma_pread(fd, offset, data, size);
}

static int xdma_write(int fd, uint64_t offset, void *data, size_t size)
{
    return npu_hal_xdma_pwrite(fd, offset, data, size);
}

static int xdma_address_io(const struct npu_hal_xdma *state, int fd,
                           uint64_t address, void *data, size_t size,
                           xdma_io_fn io)
{
    unsigned char *cursor = data;
    size_t done = 0;

    if (!state || !io || (!data && size))
        return -EINVAL;
    while (done < size) {
        uint64_t current = address + done;
        uint64_t translated = current;
        size_t chunk = size - done;
        int status;

        if (current < address)
            return -EOVERFLOW;
        if (current < NPU_HAL_XDMA_MMIO_LIMIT) {
            uint64_t low_remaining = NPU_HAL_XDMA_MMIO_LIMIT - current;

            if (chunk > low_remaining)
                chunk = (size_t)low_remaining;
            if (current > UINT64_MAX - state->mmio_address_offset)
                return -EOVERFLOW;
            translated += state->mmio_address_offset;
        }
        status = io(fd, translated, cursor + done, chunk);
        if (status < 0)
            return status;
        done += chunk;
    }
    return 0;
}

static int xdma_translate_range(const struct npu_hal_xdma *state,
                                uint64_t address, size_t size,
                                uint64_t *translated_out)
{
    uint64_t translated = address;

    if (!state || !size || !translated_out ||
        (address < NPU_HAL_XDMA_MMIO_LIMIT &&
         size > NPU_HAL_XDMA_MMIO_LIMIT - address))
        return -EINVAL;
    if (address < NPU_HAL_XDMA_MMIO_LIMIT) {
        if (address > UINT64_MAX - state->mmio_address_offset)
            return -EOVERFLOW;
        translated += state->mmio_address_offset;
    }
    if (translated > (uint64_t)INT64_MAX ||
        size > (uint64_t)INT64_MAX - translated)
        return -EOVERFLOW;
    *translated_out = translated;
    return 0;
}

static void xdma_close(void *opaque)
{
    struct npu_hal_xdma *state = opaque;
    uint32_t cluster;

    if (!state)
        return;
    for (cluster = 0; cluster < state->cluster_count; ++cluster) {
        struct npu_hal_xdma_endpoint *endpoint = &state->endpoints[cluster];

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

static int xdma_get_info(void *opaque, npu_hal_info_t *info)
{
    const struct npu_hal_xdma *state = opaque;

    if (!state || !info)
        return -EINVAL;
    info->data_provider = NPU_HAL_DATA_PROVIDER_XDMA;
    info->data_handle_count = state->cluster_count * 2U;
    info->max_transfer_size = (uint64_t)INT64_MAX;
    info->transfer_alignment = 1;
    return 0;
}

static int xdma_copy_to_address(void *opaque, uint32_t cluster,
                                uint64_t address, const void *source,
                                size_t size)
{
    struct npu_hal_xdma *state = opaque;

    if (cluster >= state->cluster_count)
        return -EINVAL;
    return xdma_address_io(state, state->endpoints[cluster].h2c_fd,
                           address, (void *)source, size, xdma_write);
}

static int xdma_copy_from_address(void *opaque, uint32_t cluster,
                                  void *destination, uint64_t address,
                                  size_t size)
{
    struct npu_hal_xdma *state = opaque;

    if (cluster >= state->cluster_count)
        return -EINVAL;
    return xdma_address_io(state, state->endpoints[cluster].c2h_fd,
                           address, destination, size, xdma_read);
}

static int xdma_register_read(void *opaque, uint32_t cluster,
                              uint64_t offset, void *value, size_t size)
{
    struct npu_hal_xdma *state = opaque;
    struct npu_hal_xdma_endpoint *endpoint;

    if (cluster >= state->cluster_count)
        return -EINVAL;
    endpoint = &state->endpoints[cluster];
    if (endpoint->user_fd >= 0)
        return npu_hal_xdma_pread(endpoint->user_fd, offset, value, size);
    return xdma_copy_from_address(state, cluster, value,
                                  NPU_XDMA_CONTROL_BASE + offset, size);
}

static int xdma_register_write(void *opaque, uint32_t cluster,
                               uint64_t offset, const void *value,
                               size_t size)
{
    struct npu_hal_xdma *state = opaque;
    struct npu_hal_xdma_endpoint *endpoint;

    if (cluster >= state->cluster_count)
        return -EINVAL;
    endpoint = &state->endpoints[cluster];
    if (endpoint->user_fd >= 0)
        return npu_hal_xdma_pwrite(endpoint->user_fd, offset, value, size);
    return xdma_copy_to_address(state, cluster,
                                NPU_XDMA_CONTROL_BASE + offset, value, size);
}

static int monotonic_ms(uint64_t *value)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return -errno;
    *value = (uint64_t)now.tv_sec * 1000U +
             (uint64_t)now.tv_nsec / 1000000U;
    return 0;
}

static int xdma_pending_irq(struct npu_hal_xdma *state,
                            uint32_t *cluster_out, uint32_t *status_out)
{
    uint32_t cluster;

    for (cluster = 0; cluster < state->cluster_count; ++cluster) {
        uint32_t pending = 0;
        int status = xdma_register_read(state, cluster, 0, &pending,
                                        sizeof(pending));

        if (status < 0)
            return status;
        if (pending) {
            *cluster_out = cluster;
            *status_out = pending;
            return 0;
        }
    }
    return -EAGAIN;
}

static int xdma_irq_wait(void *opaque, uint32_t timeout_ms,
                         uint32_t *cluster_out, uint32_t *status_out)
{
    struct npu_hal_xdma *state = opaque;
    struct pollfd events[NPU_HAL_MAX_CLUSTERS];
    uint64_t deadline = 0;
    uint32_t cluster;
    int status;

    status = xdma_pending_irq(state, cluster_out, status_out);
    if (status != -EAGAIN)
        return status;
    if (state->event_wait_disabled)
        return -ENOTSUP;
    for (cluster = 0; cluster < state->cluster_count; ++cluster) {
        if (state->endpoints[cluster].event_fd < 0)
            return -ENOTSUP;
        events[cluster].fd = state->endpoints[cluster].event_fd;
        events[cluster].events = POLLIN;
        events[cluster].revents = 0;
    }
    if (timeout_ms != NPU_HAL_TIMEOUT_INFINITE) {
        status = monotonic_ms(&deadline);
        if (status < 0)
            return status;
        deadline += timeout_ms;
    }
    for (;;) {
        int poll_timeout = -1;
        int result;

        if (timeout_ms != NPU_HAL_TIMEOUT_INFINITE) {
            uint64_t now = 0;
            uint64_t remaining;

            status = monotonic_ms(&now);
            if (status < 0)
                return status;
            if (now >= deadline)
                return -ETIMEDOUT;
            remaining = deadline - now;
            poll_timeout = remaining > INT_MAX ? INT_MAX : (int)remaining;
        }
        result = poll(events, state->cluster_count, poll_timeout);
        if (result < 0) {
            if (errno == EINTR)
                continue;
            return -errno;
        }
        if (!result)
            return -ETIMEDOUT;
        for (cluster = 0; cluster < state->cluster_count; ++cluster) {
            uint32_t event_count;
            ssize_t count;

            if (!(events[cluster].revents & POLLIN)) {
                if (events[cluster].revents &
                    (POLLERR | POLLHUP | POLLNVAL)) {
                    state->event_wait_disabled = 1;
                    return -ENOTSUP;
                }
                continue;
            }
            do {
                count = read(events[cluster].fd, &event_count,
                             sizeof(event_count));
            } while (count < 0 && errno == EINTR);
            if (count != (ssize_t)sizeof(event_count)) {
                state->event_wait_disabled = 1;
                return -ENOTSUP;
            }
            status = xdma_register_read(state, cluster, 0, status_out,
                                        sizeof(*status_out));
            if (status < 0)
                return status;
            if (*status_out) {
                *cluster_out = cluster;
                return 0;
            }
            state->event_wait_disabled = 1;
            return -ENOTSUP;
        }
    }
}

static int xdma_export_data_plane(void *opaque,
                                  npu_hal_data_export_t *exported,
                                  int handles[NPU_HAL_DATA_MAX_HANDLES])
{
    struct npu_hal_xdma *state = opaque;
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
    exported->provider = NPU_HAL_DATA_PROVIDER_XDMA;
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

static int xdma_prepare_transfer(void *opaque, uint32_t cluster,
                                 uint64_t address, size_t size,
                                 npu_hal_data_direction_t direction,
                                 npu_hal_data_token_t *token)
{
    struct npu_hal_xdma *state = opaque;
    uint64_t translated;
    int status;

    if (cluster >= state->cluster_count)
        return -EINVAL;
    status = xdma_translate_range(state, address, size, &translated);
    if (status < 0)
        return status;
    memset(token, 0, sizeof(*token));
    token->version = NPU_HAL_DATA_TOKEN_VERSION;
    token->provider = NPU_HAL_DATA_PROVIDER_XDMA;
    token->direction = (uint32_t)direction;
    token->size = size;
    token->opaque[0] = translated;
    token->opaque[1] = cluster;
    return 0;
}

static int xdma_finish_transfer(void *opaque,
                                const npu_hal_data_token_t *token,
                                int transfer_status)
{
    struct npu_hal_xdma *state = opaque;

    if (token->version != NPU_HAL_DATA_TOKEN_VERSION ||
        token->provider != NPU_HAL_DATA_PROVIDER_XDMA ||
        token->opaque[1] >= state->cluster_count)
        return -EINVAL;
    return transfer_status;
}

static int xdma_open(const npu_hal_options_t *options, void **state_out);

const struct npu_hal_backend_ops npu_hal_xdma_backend_ops = {
    .backend = NPU_HAL_BACKEND_XDMA,
    .open = xdma_open,
    .close = xdma_close,
    .get_info = xdma_get_info,
    .copy_to_address = xdma_copy_to_address,
    .copy_from_address = xdma_copy_from_address,
    .register_read = xdma_register_read,
    .register_write = xdma_register_write,
    .irq_wait = xdma_irq_wait,
    .export_data_plane = xdma_export_data_plane,
    .prepare_transfer = xdma_prepare_transfer,
    .finish_transfer = xdma_finish_transfer,
};

static int open_endpoint(struct npu_hal_xdma_endpoint *endpoint,
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

static int xdma_open(const npu_hal_options_t *options, void **state_out)
{
    struct npu_hal_xdma *state;
    npu_hal_xdma_options_t defaults;
    const npu_hal_xdma_options_t *xdma;
    size_t state_size;
    uint32_t cluster;
    int status = 0;

    if (!options || !state_out || !options->cluster_count ||
        options->cluster_count > NPU_HAL_MAX_CLUSTERS)
        return -EINVAL;
    if (!options->backend_options) {
        status = npu_hal_xdma_options_init(
            &defaults, NPU_HAL_XDMA_PROFILE_SIMULATOR);
        if (status < 0)
            return status;
        xdma = &defaults;
    } else {
        if (options->backend_options->backend != NPU_HAL_BACKEND_XDMA ||
            options->backend_options->version !=
                NPU_HAL_XDMA_OPTIONS_VERSION)
            return -EINVAL;
        xdma = (const npu_hal_xdma_options_t *)options->backend_options;
    }
    if ((xdma->profile != NPU_HAL_XDMA_PROFILE_SIMULATOR &&
         xdma->profile != NPU_HAL_XDMA_PROFILE_FPGA) ||
        !xdma->cluster_device_stride || !xdma->device_directory ||
        !*xdma->device_directory || xdma->reserved)
        return -EINVAL;
    state_size = sizeof(*state) +
                 options->cluster_count * sizeof(state->endpoints[0]);
    state = calloc(1, state_size);
    if (!state)
        return -ENOMEM;
    state->cluster_count = options->cluster_count;
    state->mmio_address_offset = xdma->mmio_address_offset;
    for (cluster = 0; cluster < state->cluster_count; ++cluster) {
        state->endpoints[cluster].h2c_fd = -1;
        state->endpoints[cluster].c2h_fd = -1;
        state->endpoints[cluster].user_fd = -1;
        state->endpoints[cluster].event_fd = -1;
    }
    for (cluster = 0; cluster < state->cluster_count; ++cluster) {
        uint64_t index = xdma->device_index +
                         (uint64_t)cluster * xdma->cluster_device_stride;

        if (index > UINT32_MAX) {
            status = -EOVERFLOW;
            break;
        }
        status = open_endpoint(&state->endpoints[cluster],
                               xdma->device_directory,
                               (uint32_t)index);
        if (status < 0)
            break;
    }
    if (status < 0) {
        xdma_close(state);
        return status;
    }
    *state_out = state;
    return 0;
}
