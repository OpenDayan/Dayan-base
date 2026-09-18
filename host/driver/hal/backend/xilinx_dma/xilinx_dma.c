#define _POSIX_C_SOURCE 200809L
#include "hal_internal.h"
#include "npu_hal_xilinx_dma.h"
#include "platform_regs.h"
#include "io.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

struct dma_endpoint { int h2c, c2h, control, event; };
struct dma_state {
    uint32_t count, completion;
    uint64_t offset;
    struct dma_endpoint endpoints[];
};

static void dma_close(void *opaque)
{
    struct dma_state *s = opaque;
    uint32_t i;
    if (!s) return;
    for (i = 0; i < s->count; ++i) {
        struct dma_endpoint *e = &s->endpoints[i];
        if (e->event >= 0) close(e->event);
        if (e->control >= 0) close(e->control);
        if (e->c2h >= 0) close(e->c2h);
        if (e->h2c >= 0) close(e->h2c);
    }
    free(s);
}

static int device_path(char *path, size_t capacity, const char *directory,
                        const char *name, const char *suffix, uint32_t index)
{
    int n = snprintf(path, capacity, "%s/%s%s%u", directory, name, suffix, index);
    return n > 0 && (size_t)n < capacity ? 0 : -ENAMETOOLONG;
}

static int open_endpoint(struct dma_endpoint *e,
                         const npu_hal_xilinx_dma_options_t *o, uint32_t cluster)
{
    char name[128], path[PATH_MAX];
    int status, n;
    int qdma = o->driver == NPU_HAL_XILINX_DMA_DRIVER_QDMA;
    uint64_t index = o->device_index + (uint64_t)cluster * o->cluster_device_stride;
    if (index > UINT32_MAX) return -EOVERFLOW;
    n = o->device_name ? snprintf(name, sizeof(name), "%s", o->device_name) :
        qdma ? snprintf(name, sizeof(name), "qdma01000") :
               snprintf(name, sizeof(name), "xdma%u", (uint32_t)index);
    if (n <= 0 || (size_t)n >= sizeof(name)) return -ENAMETOOLONG;
    status = device_path(path, sizeof(path), o->device_directory, name,
                         qdma ? "-MM-" : "_h2c_", qdma ? o->queue : o->channel);
    if (status < 0) return status;
    e->h2c = open(path, O_WRONLY | O_CLOEXEC);
    if (e->h2c < 0) return -errno;
    status = device_path(path, sizeof(path), o->device_directory, name,
                         qdma ? "-MM-" : "_c2h_", qdma ? o->queue : o->channel);
    if (status < 0) return status;
    e->c2h = open(path, O_RDONLY | O_CLOEXEC);
    if (e->c2h < 0) return -errno;
    if (!qdma) {
        n = snprintf(path, sizeof(path), "%s/%s_user", o->device_directory, name);
        if (n <= 0 || (size_t)n >= sizeof(path)) return -ENAMETOOLONG;
        e->control = open(path, O_RDWR | O_CLOEXEC);
        if (e->control < 0 && errno != ENOENT) return -errno;
    }
    if (o->completion == NPU_HAL_XILINX_DMA_COMPLETION_POLLING) return 0;
    status = device_path(path, sizeof(path), o->device_directory, name,
                         qdma ? "-user_irq" : "_events_", qdma ? o->user_irq : o->event);
    if (status < 0) return status;
    e->event = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (e->event < 0 && !(errno == ENOENT &&
        o->completion == NPU_HAL_XILINX_DMA_COMPLETION_LEGACY)) return -errno;
    return 0;
}

static int dma_open(const npu_hal_options_t *options, void **out)
{
    npu_hal_xilinx_dma_options_t defaults;
    const npu_hal_xilinx_dma_options_t *o;
    struct dma_state *s;
    uint32_t i;
    int status;
    if (!options || !out || !options->cluster_count ||
        options->cluster_count > NPU_HAL_MAX_CLUSTERS) return -EINVAL;
    *out = NULL;
    if (!options->backend_options) {
        npu_hal_xilinx_dma_options_init(&defaults, NPU_HAL_XILINX_DMA_PROFILE_SIMULATOR);
        o = &defaults;
    } else {
        if (options->backend_options->backend != NPU_HAL_BACKEND_XILINX_DMA ||
            options->backend_options->version != NPU_HAL_XILINX_DMA_OPTIONS_VERSION)
            return -EINVAL;
        o = (const void *)options->backend_options;
    }
    if ((o->driver != NPU_HAL_XILINX_DMA_DRIVER_XDMA && o->driver != NPU_HAL_XILINX_DMA_DRIVER_QDMA) ||
        (o->profile != NPU_HAL_XILINX_DMA_PROFILE_SIMULATOR && o->profile != NPU_HAL_XILINX_DMA_PROFILE_FPGA) ||
        o->completion < NPU_HAL_XILINX_DMA_COMPLETION_IRQ ||
        o->completion > NPU_HAL_XILINX_DMA_COMPLETION_LEGACY ||
        !o->device_directory || !*o->device_directory || !o->cluster_device_stride ||
        (o->device_name && (!*o->device_name || strchr(o->device_name, '/')))) return -EINVAL;
    /* A named PCI function is not implicitly mapped to several clusters. */
    if (options->cluster_count != 1 && (o->device_name ||
        o->driver == NPU_HAL_XILINX_DMA_DRIVER_QDMA)) return -ENOTSUP;
    s = calloc(1, sizeof(*s) + options->cluster_count * sizeof(s->endpoints[0]));
    if (!s) return -ENOMEM;
    s->count = options->cluster_count;
    s->offset = o->mmio_address_offset;
    s->completion = o->completion;
    for (i = 0; i < s->count; ++i)
        s->endpoints[i] = (struct dma_endpoint){-1, -1, -1, -1};
    for (i = 0; i < s->count; ++i) {
        status = open_endpoint(&s->endpoints[i], o, i);
        if (status < 0) { dma_close(s); return status; }
    }
    *out = s;
    return 0;
}

static int dma_info(void *opaque, npu_hal_info_t *info)
{
    struct dma_state *s = opaque;
    if (!s || !info) return -EINVAL;
    info->data_provider = NPU_HAL_DATA_PROVIDER_XILINX_DMA;
    info->data_handle_count = s->count * 2U;
    info->max_transfer_size = INT64_MAX;
    info->transfer_alignment = 1;
    return 0;
}

static int translate(struct dma_state *s, uint64_t address, size_t size, uint64_t *out)
{
    if (!s || !size) return -EINVAL;
    if (address < NPU_HAL_XILINX_DMA_MMIO_LIMIT) {
        if (size > NPU_HAL_XILINX_DMA_MMIO_LIMIT - address) return -EINVAL;
        if (address > UINT64_MAX - s->offset) return -EOVERFLOW;
        address += s->offset;
    }
    if (address > INT64_MAX || size > (uint64_t)INT64_MAX - address) return -EOVERFLOW;
    *out = address;
    return 0;
}

static int dma_copy(void *opaque, uint32_t cluster, uint64_t address, void *data, size_t size, int write)
{
    struct dma_state *s = opaque;
    unsigned char *cursor = data;
    if (!s || cluster >= s->count || (!data && size)) return -EINVAL;
    if (size > UINT64_MAX - address) return -EOVERFLOW;
    while (size) {
        uint64_t offset;
        size_t chunk = size;
        int status;
        if (address < NPU_HAL_XILINX_DMA_MMIO_LIMIT && chunk > NPU_HAL_XILINX_DMA_MMIO_LIMIT - address)
            chunk = (size_t)(NPU_HAL_XILINX_DMA_MMIO_LIMIT - address);
        status = translate(s, address, chunk, &offset);
        if (status < 0) return status;
        status = write ? npu_hal_xilinx_dma_pwrite(s->endpoints[cluster].h2c, offset, cursor, chunk) :
                         npu_hal_xilinx_dma_pread(s->endpoints[cluster].c2h, offset, cursor, chunk);
        if (status < 0) return status;
        cursor += chunk; address += chunk; size -= chunk;
    }
    return 0;
}
static int dma_write(void *s, uint32_t c, uint64_t a, const void *v, size_t n)
{ return dma_copy(s, c, a, (void *)v, n, 1); }
static int dma_read(void *s, uint32_t c, void *v, uint64_t a, size_t n)
{ return dma_copy(s, c, a, v, n, 0); }
static int dma_reg_read(void *opaque, uint32_t c, uint64_t a, void *v, size_t n)
{
    struct dma_state *s = opaque;
    if (!s || c >= s->count) return -EINVAL;
    if (s->endpoints[c].control >= 0)
        return npu_hal_xilinx_dma_pread(s->endpoints[c].control, a, v, n);
    if (a > UINT64_MAX - DAYAN_IRQ_STATUS) return -EOVERFLOW;
    return dma_read(s, c, v, DAYAN_IRQ_STATUS + a, n);
}
static int dma_reg_write(void *opaque, uint32_t c, uint64_t a, const void *v, size_t n)
{
    struct dma_state *s = opaque;
    if (!s || c >= s->count) return -EINVAL;
    if (s->endpoints[c].control >= 0)
        return npu_hal_xilinx_dma_pwrite(s->endpoints[c].control, a, v, n);
    if (a > UINT64_MAX - DAYAN_IRQ_STATUS) return -EOVERFLOW;
    return dma_write(s, c, DAYAN_IRQ_STATUS + a, v, n);
}
static int now_ms(uint64_t *out)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) return -errno;
    *out = (uint64_t)ts.tv_sec * 1000U + (uint64_t)ts.tv_nsec / 1000000U;
    return 0;
}
static int dma_irq(void *opaque, uint32_t timeout, uint32_t *cluster, uint32_t *pending)
{
    struct dma_state *s = opaque;
    struct pollfd events[NPU_HAL_MAX_CLUSTERS];
    uint64_t deadline = 0;
    uint32_t i;
    int status;
    if (!s || !cluster || !pending) return -EINVAL;
    if (s->completion == NPU_HAL_XILINX_DMA_COMPLETION_POLLING) return -ENOTSUP;
    for (i = 0; i < s->count; ++i) {
        if (s->endpoints[i].event < 0) return -ENOTSUP;
        events[i] = (struct pollfd){s->endpoints[i].event, POLLIN, 0};
    }
    if (timeout != NPU_HAL_TIMEOUT_INFINITE) {
        status = now_ms(&deadline);
        if (status < 0) return status;
        deadline += timeout;
    }
    for (;;) {
        int wait_ms = -1;
        for (i = 0; i < s->count; ++i) {
            status = dma_reg_read(s, i, 0, pending, sizeof(*pending));
            if (status < 0) return status;
            if (*pending) { *cluster = i; return 0; }
        }
        if (timeout != NPU_HAL_TIMEOUT_INFINITE) {
            uint64_t now = 0;
            status = now_ms(&now);
            if (status < 0) return status;
            if (now >= deadline) return -ETIMEDOUT;
            wait_ms = deadline - now > INT_MAX ? INT_MAX : (int)(deadline - now);
        }
        status = poll(events, s->count, wait_ms);
        if (status < 0) { if (errno == EINTR) continue; return -errno; }
        for (i = 0; i < s->count; ++i) {
            uint32_t count;
            ssize_t bytes;
            if (events[i].revents & (POLLERR | POLLHUP | POLLNVAL))
                return s->completion == NPU_HAL_XILINX_DMA_COMPLETION_LEGACY ? -ENOTSUP : -EIO;
            if (!(events[i].revents & POLLIN)) continue;
            do { bytes = read(events[i].fd, &count, sizeof(count)); } while (bytes < 0 && errno == EINTR);
            if (bytes < 0 && errno == EAGAIN) continue;
            if (bytes != sizeof(count))
                return s->completion == NPU_HAL_XILINX_DMA_COMPLETION_LEGACY ? -ENOTSUP :
                       bytes < 0 ? -errno : -EIO;
        }
    }
}
static int dma_export(void *opaque, npu_hal_data_export_t *e, int h[NPU_HAL_DATA_MAX_HANDLES])
{
    struct dma_state *s = opaque;
    uint32_t i;
    for (i = 0; i < s->count; ++i) {
        h[i * 2] = fcntl(s->endpoints[i].h2c, F_DUPFD_CLOEXEC, 0);
        if (h[i * 2] < 0) return -errno;
        h[i * 2 + 1] = fcntl(s->endpoints[i].c2h, F_DUPFD_CLOEXEC, 0);
        if (h[i * 2 + 1] < 0) return -errno;
    }
    memset(e, 0, sizeof(*e));
    e->version = NPU_HAL_DATA_EXPORT_VERSION;
    e->provider = NPU_HAL_DATA_PROVIDER_XILINX_DMA;
    e->handle_count = s->count * 2;
    e->opaque[0] = s->count;
    return 0;
}
static int dma_prepare(void *opaque, uint32_t c, uint64_t a, size_t n,
                         npu_hal_data_direction_t d, npu_hal_data_token_t *t)
{
    struct dma_state *s = opaque;
    uint64_t offset;
    int status;
    if (!s || !t || c >= s->count) return -EINVAL;
    status = translate(s, a, n, &offset);
    if (status < 0) return status;
    memset(t, 0, sizeof(*t));
    t->version = NPU_HAL_DATA_TOKEN_VERSION;
    t->provider = NPU_HAL_DATA_PROVIDER_XILINX_DMA;
    t->direction = d; t->size = n; t->opaque[0] = offset; t->opaque[1] = c;
    return 0;
}
static int dma_finish(void *opaque, const npu_hal_data_token_t *t, int status)
{
    struct dma_state *s = opaque;
    if (!s || !t || t->version != NPU_HAL_DATA_TOKEN_VERSION ||
        t->provider != NPU_HAL_DATA_PROVIDER_XILINX_DMA || t->opaque[1] >= s->count)
        return -EINVAL;
    return status;
}
const struct npu_hal_backend_ops npu_hal_xilinx_dma_backend_ops = {
    .backend = NPU_HAL_BACKEND_XILINX_DMA, .open = dma_open, .close = dma_close,
    .get_info = dma_info, .copy_to_address = dma_write, .copy_from_address = dma_read,
    .register_read = dma_reg_read, .register_write = dma_reg_write, .irq_wait = dma_irq,
    .export_data_plane = dma_export, .prepare_transfer = dma_prepare, .finish_transfer = dma_finish,
};
