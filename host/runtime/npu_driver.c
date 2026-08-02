#define _POSIX_C_SOURCE 200809L

#include "npu_driver.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "platform_regs.h"

#define NPU_DEFAULT_PREFIX       "/tmp/virtual_npu"
#define NPU_IRQ_USER_BASE        DAYAN_IRQ_STATUS
#define NPU_IRQ_STATUS_OFFSET    0x00U
#define NPU_IRQ_CLEAR_OFFSET     0x0cU
#define NPU_IRQ_ENABLE_OFFSET    0x04U
#define NPU_IRQ_MAILBOX_OFFSET   0x20U
#define NPU_POLL_NS              1000000L

static uint64_t now_ms(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0;
    return (uint64_t)now.tv_sec * 1000U +
           (uint64_t)now.tv_nsec / 1000000U;
}

static void poll_delay(void)
{
    const struct timespec delay = { 0, NPU_POLL_NS };

    (void)nanosleep(&delay, 0);
}

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

void npu_driver_options_init(npu_driver_options_t *options)
{
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->version = NPU_DRIVER_OPTIONS_VERSION;
}

int npu_device_open(npu_device_t *device,
                    const npu_driver_options_t *options)
{
    npu_driver_options_t defaults;
    const char *prefix;
    char path[PATH_MAX];
    int status;

    if (!device)
        return -EINVAL;
    if (!options) {
        npu_driver_options_init(&defaults);
        options = &defaults;
    }
    if (options->version != NPU_DRIVER_OPTIONS_VERSION)
        return -EINVAL;
    memset(device, 0, sizeof(*device));
    device->h2c_fd = -1;
    device->c2h_fd = -1;
    device->user_fd = -1;
    device->event_fd = -1;
    device->device_index = options->device_index;
    device->memory_address_bias = options->memory_address_bias;
    prefix = options->device_prefix;
    if (!prefix || !*prefix)
        prefix = getenv("NPU_DEVICE_PREFIX");
    if (!prefix || !*prefix)
        prefix = NPU_DEFAULT_PREFIX;

    status = make_path(path, sizeof(path), prefix, options->device_index,
                       "_h2c_0");
    if (status < 0)
        goto fail;
    device->h2c_fd = open(path, O_WRONLY | O_CLOEXEC);
    if (device->h2c_fd < 0) {
        status = -errno;
        goto fail;
    }
    status = make_path(path, sizeof(path), prefix, options->device_index,
                       "_c2h_0");
    if (status < 0)
        goto fail;
    device->c2h_fd = open(path, O_RDONLY | O_CLOEXEC);
    if (device->c2h_fd < 0) {
        status = -errno;
        goto fail;
    }
    status = make_path(path, sizeof(path), prefix, options->device_index,
                       "_user");
    if (status < 0)
        goto fail;
    device->user_fd = open(path, O_RDWR | O_CLOEXEC);
    if (device->user_fd < 0 && errno != ENOENT) {
        status = -errno;
        goto fail;
    }
    status = make_path(path, sizeof(path), prefix, options->device_index,
                       "_events_0");
    if (status < 0)
        goto fail;
    device->event_fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    device->opened = 1;
    return 0;

fail:
    npu_device_close(device);
    return status;
}

void npu_device_close(npu_device_t *device)
{
    if (!device)
        return;
    if (device->event_fd >= 0)
        close(device->event_fd);
    if (device->user_fd >= 0)
        close(device->user_fd);
    if (device->c2h_fd >= 0)
        close(device->c2h_fd);
    if (device->h2c_fd >= 0)
        close(device->h2c_fd);
    device->event_fd = -1;
    device->user_fd = -1;
    device->c2h_fd = -1;
    device->h2c_fd = -1;
    device->opened = 0;
}

int npu_device_read(npu_device_t *device, uint64_t address,
                    void *data, size_t size)
{
    if (!device || !device->opened ||
        address > UINT64_MAX - device->memory_address_bias)
        return -EINVAL;
    return full_pread(device->c2h_fd,
                      address + device->memory_address_bias, data, size);
}

int npu_device_write(npu_device_t *device, uint64_t address,
                     const void *data, size_t size)
{
    if (!device || !device->opened ||
        address > UINT64_MAX - device->memory_address_bias)
        return -EINVAL;
    return full_pwrite(device->h2c_fd,
                       address + device->memory_address_bias, data, size);
}

int npu_device_read_control(npu_device_t *device, uint64_t offset,
                            void *data, size_t size)
{
    if (!device || !device->opened)
        return -EINVAL;
    if (device->user_fd >= 0)
        return full_pread(device->user_fd, offset, data, size);
    return npu_device_read(device, NPU_IRQ_USER_BASE + offset, data, size);
}

int npu_device_write_control(npu_device_t *device, uint64_t offset,
                             const void *data, size_t size)
{
    if (!device || !device->opened)
        return -EINVAL;
    if (device->user_fd >= 0)
        return full_pwrite(device->user_fd, offset, data, size);
    return npu_device_write(device, NPU_IRQ_USER_BASE + offset, data, size);
}

int npu_device_read_irq_status(npu_device_t *device, uint32_t *status)
{
    return npu_device_read_control(device, NPU_IRQ_STATUS_OFFSET,
                                   status, sizeof(*status));
}

int npu_device_clear_irq(npu_device_t *device, uint32_t mask)
{
    return npu_device_write_control(device, NPU_IRQ_CLEAR_OFFSET,
                                    &mask, sizeof(mask));
}

int npu_device_read_irq_mailbox(npu_device_t *device, uint32_t hart,
                                uint64_t *command)
{
    if (hart >= NPU_DRIVER_MAX_HARTS || !command)
        return -EINVAL;
    return npu_device_read_control(device,
                                   NPU_IRQ_MAILBOX_OFFSET + hart * 8U,
                                   command, sizeof(*command));
}

int npu_device_wait_msip_clear(npu_device_t *device, uint32_t hart,
                               uint32_t timeout_ms)
{
    uint64_t deadline = now_ms() + timeout_ms;

    if (hart >= NPU_DRIVER_MAX_HARTS)
        return -EINVAL;
    for (;;) {
        uint32_t pending = 0;
        int status = npu_device_read(device, DAYAN_CLINT_MSIP(hart),
                                     &pending, sizeof(pending));

        if (status < 0 || !(pending & 1U))
            return status;
        if (now_ms() >= deadline)
            return -ETIMEDOUT;
        poll_delay();
    }
}

int npu_device_submit(npu_device_t *device, uint32_t hart,
                      uint64_t command)
{
    const uint32_t one = 1;
    int status;

    if (hart >= NPU_DRIVER_MAX_HARTS)
        return -EINVAL;
    status = npu_device_write(device, DAYAN_DEVICE_MAILBOX(hart),
                              &command, sizeof(command));
    if (status < 0)
        return status;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return npu_device_write(device, DAYAN_CLINT_MSIP(hart),
                            &one, sizeof(one));
}

int npu_device_boot(npu_device_t *device, uint32_t hart_mask,
                    uint32_t hart_count, uint64_t entry,
                    uint32_t timeout_ms)
{
    uint32_t hart;
    int status;

    if (!device || !device->opened || !hart_mask || !hart_count ||
        hart_count > NPU_DRIVER_MAX_HARTS ||
        (hart_count < 32U && (hart_mask >> hart_count)))
        return -EINVAL;
    for (hart = 0; hart < hart_count; ++hart) {
        if (!(hart_mask & (1U << hart)))
            continue;
        status = npu_device_wait_msip_clear(device, hart, timeout_ms);
        if (status < 0)
            return status;
        status = npu_device_submit(device, hart, entry);
        if (status < 0)
            return status;
    }
    for (hart = 0; hart < hart_count; ++hart) {
        if (!(hart_mask & (1U << hart)))
            continue;
        status = npu_device_wait_msip_clear(device, hart, timeout_ms);
        if (status < 0)
            return status;
    }
    return 0;
}
