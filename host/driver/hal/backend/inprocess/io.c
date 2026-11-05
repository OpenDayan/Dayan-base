#include "io.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct npu_inprocess_device {
    npu_hal_inprocess_device_desc_t desc;
    uint64_t id;
    uint64_t process;
    size_t references;
    struct npu_inprocess_device *next;
};

static pthread_mutex_t registry_mutex = PTHREAD_MUTEX_INITIALIZER;
static struct npu_inprocess_device *devices;
static uint64_t next_id = 1;

void npu_hal_inprocess_options_init(npu_hal_inprocess_options_t *options,
                                    uint64_t device_id)
{
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->base.version = NPU_HAL_INPROCESS_OPTIONS_VERSION;
    options->base.backend = NPU_HAL_BACKEND_INPROCESS;
    options->device_id = device_id;
}

int npu_hal_inprocess_device_register(
    const npu_hal_inprocess_device_desc_t *desc, uint64_t *device_id)
{
    struct npu_inprocess_device *device;

    if (!device_id)
        return -EINVAL;
    *device_id = 0;
    if (!desc || !desc->read || !desc->write || !desc->cluster_count ||
        desc->cluster_count > NPU_HAL_MAX_CLUSTERS)
        return -EINVAL;
    device = calloc(1, sizeof(*device));
    if (!device)
        return -ENOMEM;
    device->desc = *desc;
    device->process = (uint64_t)getpid();
    pthread_mutex_lock(&registry_mutex);
    if (!next_id) {
        pthread_mutex_unlock(&registry_mutex);
        free(device);
        return -EOVERFLOW;
    }
    device->id = next_id++;
    device->next = devices;
    devices = device;
    *device_id = device->id;
    pthread_mutex_unlock(&registry_mutex);
    return 0;
}

int npu_hal_inprocess_device_unregister(uint64_t device_id)
{
    struct npu_inprocess_device **cursor;
    struct npu_inprocess_device *device;
    int status = -ENODEV;

    pthread_mutex_lock(&registry_mutex);
    for (cursor = &devices; (device = *cursor); cursor = &device->next) {
        if (device->id != device_id)
            continue;
        if (device->process != (uint64_t)getpid()) {
            status = -EXDEV;
        } else if (device->references) {
            status = -EBUSY;
        } else {
            *cursor = device->next;
            free(device);
            status = 0;
        }
        break;
    }
    pthread_mutex_unlock(&registry_mutex);
    return status;
}

int npu_inprocess_acquire(uint64_t id, uint64_t process,
                         struct npu_inprocess_device **device_out)
{
    struct npu_inprocess_device *device;
    int status = -ENODEV;

    *device_out = 0;
    if (process != (uint64_t)getpid())
        return -EXDEV;
    pthread_mutex_lock(&registry_mutex);
    for (device = devices; device; device = device->next) {
        if (device->id == id && device->process == process) {
            ++device->references;
            *device_out = device;
            status = 0;
            break;
        }
    }
    pthread_mutex_unlock(&registry_mutex);
    return status;
}

void npu_inprocess_release(struct npu_inprocess_device *device)
{
    pthread_mutex_lock(&registry_mutex);
    --device->references;
    pthread_mutex_unlock(&registry_mutex);
}

uint64_t npu_inprocess_id(const struct npu_inprocess_device *device)
{
    return device->id;
}

uint32_t npu_inprocess_clusters(const struct npu_inprocess_device *device)
{
    return device->desc.cluster_count;
}

static int range_valid(const struct npu_inprocess_device *device,
                       uint32_t cluster, uint64_t address,
                       const void *data, size_t size)
{
    return device->process == (uint64_t)getpid() &&
           cluster < device->desc.cluster_count && (data || !size) &&
           size <= UINT64_MAX - address;
}

int npu_inprocess_read(struct npu_inprocess_device *device, uint32_t cluster,
                      uint64_t address, void *destination, size_t size)
{
    int status;

    if (!range_valid(device, cluster, address, destination, size))
        return -EINVAL;
    if (!size)
        return 0;
    status = device->desc.read(device->desc.context, cluster, address,
                                destination, size);
    return status > 0 ? -EIO : status;
}

int npu_inprocess_write(struct npu_inprocess_device *device, uint32_t cluster,
                       uint64_t address, const void *source, size_t size)
{
    int status;

    if (!range_valid(device, cluster, address, source, size))
        return -EINVAL;
    if (!size)
        return 0;
    status = device->desc.write(device->desc.context, cluster, address,
                                 source, size);
    return status > 0 ? -EIO : status;
}
