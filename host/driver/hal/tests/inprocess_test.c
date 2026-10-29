#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "hal.h"
#include "backend/inprocess/npu_hal_inprocess.h"

struct fake_device {
    pthread_mutex_t mutex;
    unsigned char memory[0x40000];
    int error;
};

static const npu_hal_memory_region_t regions[] = {
    { 0x10000, 0x10000, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_DEVICE,
      NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
    { 0x20000, 0x10000, NPU_HAL_MEMORY_COHERENT, NPU_HAL_MEMORY_DEVICE,
      NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
};

static int fake_io(struct fake_device *device, uint64_t address,
                    void *data, size_t size, int write)
{
    int status;

    pthread_mutex_lock(&device->mutex);
    status = device->error;
    if (!status && (address > sizeof(device->memory) ||
        size > sizeof(device->memory) - address))
        status = -EFAULT;
    if (!status) {
        if (write)
            memcpy(device->memory + address, data, size);
        else
            memcpy(data, device->memory + address, size);
    }
    pthread_mutex_unlock(&device->mutex);
    return status;
}

static int fake_read(void *context, uint32_t cluster, uint64_t address,
                      void *data, size_t size)
{
    assert(cluster == 0);
    return fake_io(context, address, data, size, 0);
}

static int fake_write(void *context, uint32_t cluster, uint64_t address,
                       const void *data, size_t size)
{
    assert(cluster == 0);
    return fake_io(context, address, (void *)data, size, 1);
}

static void direct_hal_test(uint64_t id, struct fake_device *device)
{
    npu_hal_options_t options;
    npu_hal_inprocess_options_t backend;
    npu_hal_t *hal = 0;
    npu_hal_data_export_t exported, invalid;
    npu_hal_data_plane_t *plane = 0;
    npu_hal_allocation_t *allocation = 0;
    npu_hal_memory_request_t request;
    npu_hal_data_token_t token, wrong;
    npu_hal_info_t info;
    npu_hal_core_t core = { 0, 0 };
    int handles[NPU_HAL_DATA_MAX_HANDLES];
    uint64_t input = UINT64_C(0x1234567812345678), output = 0;
    uint32_t pending, cluster;
    pid_t child;
    int child_status;

    npu_hal_options_init(&options);
    npu_hal_inprocess_options_init(&backend, id);
    options.backend_options = &backend.base;
    options.memory_regions = regions;
    options.memory_region_count = 2;
    backend.base.version++;
    assert(npu_hal_open(&hal, &options) == -EINVAL);
    backend.base.version--;
    options.cluster_count = 2;
    assert(npu_hal_open(&hal, &options) == -EINVAL);
    options.cluster_count = 1;
    assert(npu_hal_open(&hal, &options) == 0);
    assert(npu_hal_get_info(hal, &info) == 0);
    assert(info.backend == NPU_HAL_BACKEND_INPROCESS);
    assert(info.data_provider == NPU_HAL_DATA_PROVIDER_INPROCESS);
    assert(info.data_handle_count == 0);
    assert(npu_hal_inprocess_device_unregister(id) == -EBUSY);
    assert(npu_hal_data_plane_export(hal, &exported, handles) == 0);
    assert(exported.handle_count == 0);
    assert(npu_hal_data_plane_import(&plane, &exported, 0, 0) == 0);
    invalid = exported;
    invalid.opaque[1]++;
    {
        npu_hal_data_plane_t *other = 0;
        assert(npu_hal_data_plane_import(&other, &invalid, 0, 0) == -EXDEV);
    }
    child = fork();
    assert(child >= 0);
    if (!child) {
        npu_hal_data_plane_t *other = 0;
        _exit(npu_hal_data_plane_import(&other, &exported, 0, 0) == -EXDEV ? 0 : 1);
    }
    assert(waitpid(child, &child_status, 0) == child);
    assert(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0);

    npu_hal_memory_request_init(&request);
    request.size = 64;
    assert(npu_hal_memory_alloc(hal, &request, &allocation) == 0);
    assert(npu_hal_transfer_prepare(hal, allocation, core, 3, sizeof(input),
                                    NPU_HAL_DATA_HOST_TO_DEVICE, &token) == 0);
    assert(npu_hal_data_plane_transfer(plane, &token, &input, sizeof(input)) == 0);
    assert(npu_hal_transfer_finish(hal, &token, 0) == 0);
    assert(npu_hal_memory_read(hal, allocation, core, &output, 3, sizeof(output)) == 0);
    assert(output == input);
    assert(npu_hal_transfer_prepare(hal, allocation, core, 3, sizeof(output),
                                    NPU_HAL_DATA_DEVICE_TO_HOST, &token) == 0);
    output = 0;
    assert(npu_hal_data_plane_transfer(plane, &token, &output, sizeof(output)) == 0);
    assert(output == input);
    wrong = token;
    wrong.opaque[2]++;
    assert(npu_hal_data_plane_transfer(plane, &wrong, &output, sizeof(output)) == -EACCES);
    wrong = token;
    wrong.opaque[0] = UINT64_MAX;
    assert(npu_hal_data_plane_transfer(plane, &wrong, &output, sizeof(output)) == -EINVAL);
    assert(npu_hal_transfer_prepare(hal, allocation, core, 60, 8,
                                    NPU_HAL_DATA_HOST_TO_DEVICE, &token) == -EINVAL);
    assert(npu_hal_copy_from_address(hal, 0, &output, 0x40000, 8) == -EFAULT);

    pending = 1;
    assert(npu_hal_register_write(hal, 0, 0, &pending, sizeof(pending)) == 0);
    assert(npu_hal_irq_wait(hal, 20, &cluster, &pending) == 0);
    assert(cluster == 0 && pending == 1);
    pending = 0;
    assert(npu_hal_register_write(hal, 0, 0, &pending, sizeof(pending)) == 0);
    assert(npu_hal_irq_wait(hal, 1, &cluster, &pending) == -ETIMEDOUT);
    assert(npu_hal_memory_free(hal, allocation) == 0);
    device->error = -EIO;
    assert(npu_hal_copy_from_address(hal, 0, &output, 0x10000, 8) == -EIO);
    assert(npu_hal_device_status(hal) == -EIO);
    device->error = 0;
    npu_hal_close(hal);
    assert(npu_hal_inprocess_device_unregister(id) == -EBUSY);
    npu_hal_data_plane_close(plane);
    assert(npu_hal_inprocess_device_unregister(id) == 0);
    plane = 0;
    assert(npu_hal_data_plane_import(&plane, &exported, 0, 0) == -ENODEV);
}

int main(void)
{
    struct fake_device device = { .mutex = PTHREAD_MUTEX_INITIALIZER };
    npu_hal_inprocess_device_desc_t desc = { 1, &device, fake_read, fake_write };
    uint64_t id, previous;

    assert(npu_hal_inprocess_device_register(0, &id) == -EINVAL && id == 0);
    assert(npu_hal_inprocess_device_register(&desc, &id) == 0);
    previous = id;
    direct_hal_test(id, &device);
    assert(npu_hal_inprocess_device_register(&desc, &id) == 0 && id != previous);
    assert(npu_hal_inprocess_device_unregister(id) == 0);
    assert(npu_hal_inprocess_device_unregister(id) == -ENODEV);
    pthread_mutex_destroy(&device.mutex);
    puts("inprocess HAL / provider: PASS");
    return 0;
}
