#include "service_internal.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(RPC_PROFILE_RECORD_SIZE == 8U,
               "Host and Device profile record ABI");

static int profile_config_call(npu_service_device_t *device,
                               struct npu_service_profile_core *core,
                               uint32_t command, uint64_t generation,
                               uint32_t *count_out, int *enabled_out)
{
    struct rpc_profile_config config;
    npu_hal_allocation_t *allocation = 0;
    uint64_t descriptor_address;
    int descriptor_may_be_live = 0;
    int status;

    if (count_out)
        *count_out = 0;
    if (enabled_out)
        *enabled_out = -1;
    status = npu_service_alloc_rpc_descriptor(
        device, core->target, sizeof(config), &allocation);
    if (status < 0)
        return status;
    status = npu_hal_memory_address(device->hal, allocation, core->target,
                                    &descriptor_address);
    if (status < 0 || descriptor_address > RPC_COMMAND_ARG_MASK) {
        status = status < 0 ? status : -EOVERFLOW;
        goto out;
    }
    memset(&config, 0, sizeof(config));
    config.version = RPC_PROFILE_VERSION;
    config.size = sizeof(config);
    config.command = command;
    config.hart_id = core->target.core;
    config.capacity = core->capacity;
    config.buffer_addr = core->address;
    config.generation = generation;
    config.status = -EIO;
    status = npu_hal_memory_write(device->hal, allocation, core->target, 0,
                                  &config, sizeof(config));
    if (status < 0)
        goto out;
    status = npu_service_rpc_call(device, core->target, RPC_PROFILE_CONFIG,
                                  descriptor_address,
                                  device->options.default_timeout_ms,
                                  &descriptor_may_be_live);
    if (status < 0)
        goto out;
    status = npu_hal_memory_read(device->hal, allocation, core->target,
                                 &config, 0, sizeof(config));
    if (status < 0)
        goto out;
    if (config.version != RPC_PROFILE_VERSION ||
        config.size != sizeof(config) || config.command != command ||
        config.flags || config.hart_id != core->target.core ||
        config.capacity != core->capacity ||
        config.buffer_addr != core->address ||
        config.generation != generation || config.reserved ||
        config.enabled > 1U || config.count > core->capacity) {
        status = -EPROTO;
        goto out;
    }
    if (enabled_out)
        *enabled_out = (int)config.enabled;
    if (config.status < 0) {
        status = config.status >= INT_MIN && config.status <= INT_MAX ?
            (int)config.status : -EIO;
        goto out;
    }
    if ((command == RPC_PROFILE_START &&
         (!config.enabled || config.count)) ||
        (command == RPC_PROFILE_STOP && config.enabled)) {
        status = -EPROTO;
        goto out;
    }
    if (count_out)
        *count_out = config.count;
    status = 0;

out:
    if (allocation && !descriptor_may_be_live) {
        int free_status = npu_hal_memory_free(device->hal, allocation);

        if (status == 0 && free_status < 0)
            status = free_status;
    }
    return status;
}

static int profile_core_matches(const npu_service_device_t *device,
                                const struct npu_service_profile_core *core,
                                npu_hal_core_t target, uint64_t generation,
                                uint64_t address, uint32_t capacity)
{
    return core->configured && device->profile_allocation &&
           device->profile_generation == generation &&
           core->target.cluster == target.cluster &&
           core->target.core == target.core && core->address == address &&
           core->capacity == capacity;
}

int npu_service_profile_copy(npu_service_device_t *device,
                             npu_hal_core_t target, uint64_t generation,
                             uint64_t address, uint32_t capacity,
                             uint32_t count, int *continue_out)
{
    struct npu_service_profile_core *core;
    unsigned char *data;
    npu_hal_allocation_t *allocation;
    size_t index;
    int callback_status;
    int status;

    if (!device || !continue_out || !count || count > capacity ||
        !npu_hal_core_valid(device->hal, target))
        return -EINVAL;
    index = npu_service_core_index(device, target);
    pthread_mutex_lock(&device->mutex);
    if (index >= device->profile_core_count ||
        !profile_core_matches(device, &device->profile_cores[index], target,
                              generation, address, capacity)) {
        pthread_mutex_unlock(&device->mutex);
        return -EPROTO;
    }
    core = &device->profile_cores[index];
    allocation = device->profile_allocation;
    ++device->profile_readers;
    pthread_mutex_unlock(&device->mutex);

    data = malloc(count);
    if (!data) {
        status = -ENOMEM;
        callback_status = status;
    } else {
        status = npu_hal_memory_read(device->hal, allocation, target, data,
                                     core->offset, count);
        callback_status = status < 0 ? status : device->options.profile(
            device->options.callback_context, target.cluster, target.core,
            data, count);
    }
    free(data);

    pthread_mutex_lock(&device->mutex);
    *continue_out = callback_status == 0 && device->profile_active &&
        index < device->profile_core_count &&
        profile_core_matches(device, &device->profile_cores[index], target,
                             generation, address, capacity);
    if (device->profile_readers)
        --device->profile_readers;
    pthread_cond_broadcast(&device->condition);
    pthread_mutex_unlock(&device->mutex);
    return callback_status;
}

static int allocate_profile(npu_service_device_t *device,
                            uint64_t buffer_size,
                            npu_hal_allocation_t **allocation_out,
                            struct npu_service_profile_core **cores_out,
                            size_t *core_slots_out, uint64_t *usable_out)
{
    npu_hal_memory_request_t request;
    struct npu_service_profile_core *cores;
    uint32_t *booted_cores;
    size_t core_slots;
    size_t booted_count = 0;
    uint64_t slice_size;
    uint64_t usable_size;
    uint32_t cluster;
    size_t ordinal = 0;
    int status;

    core_slots = (size_t)device->hal_info.cluster_count *
                 device->hal_info.cores_per_cluster;
    booted_cores = calloc(device->hal_info.cluster_count,
                          sizeof(*booted_cores));
    if (!booted_cores)
        return -ENOMEM;
    pthread_mutex_lock(&device->mutex);
    for (cluster = 0; cluster < device->hal_info.cluster_count; ++cluster)
        booted_cores[cluster] = device->clusters[cluster].booted_cores;
    pthread_mutex_unlock(&device->mutex);
    for (cluster = 0; cluster < device->hal_info.cluster_count; ++cluster) {
        uint32_t core;

        for (core = 0; core < device->hal_info.cores_per_cluster; ++core) {
            if (booted_cores[cluster] & (UINT32_C(1) << core))
                ++booted_count;
        }
    }
    if (!booted_count) {
        free(booted_cores);
        return -ENODEV;
    }
    slice_size = (buffer_size / booted_count) & ~UINT64_C(7);
    if (slice_size < RPC_PROFILE_SNAPSHOT_MAX_BYTES ||
        slice_size > UINT32_MAX) {
        status = -EINVAL;
        goto fail_booted;
    }
    usable_size = slice_size * booted_count;
    cores = calloc(core_slots, sizeof(*cores));
    if (!cores) {
        status = -ENOMEM;
        goto fail_booted;
    }
    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_NORMAL;
    request.scope = NPU_HAL_MEMORY_DEVICE;
    request.usage = NPU_HAL_MEMORY_PROFILE;
    request.location.cluster = NPU_HAL_ANY_CLUSTER;
    request.location.core = NPU_HAL_ANY_CORE;
    request.size = usable_size;
    request.alignment = 64;
    status = npu_hal_memory_alloc(device->hal, &request, allocation_out);
    if (status < 0) {
        free(cores);
        goto fail_booted;
    }
    for (cluster = 0; cluster < device->hal_info.cluster_count; ++cluster) {
        uint32_t core_id;

        for (core_id = 0; core_id < device->hal_info.cores_per_cluster;
             ++core_id) {
            npu_hal_core_t target = { cluster, core_id };
            size_t index = npu_service_core_index(device, target);
            uint64_t base;

            if (!(booted_cores[cluster] & (UINT32_C(1) << core_id)))
                continue;
            status = npu_hal_memory_address(device->hal, *allocation_out,
                                            target, &base);
            if (status < 0 || base > UINT64_MAX - ordinal * slice_size) {
                status = status < 0 ? status : -EOVERFLOW;
                goto fail;
            }
            cores[index].target = target;
            cores[index].offset = ordinal * slice_size;
            cores[index].address = base + cores[index].offset;
            cores[index].capacity = (uint32_t)slice_size;
            ++ordinal;
        }
    }
    *cores_out = cores;
    *core_slots_out = core_slots;
    *usable_out = usable_size;
    free(booted_cores);
    return 0;

fail:
    (void)npu_hal_memory_free(device->hal, *allocation_out);
    *allocation_out = 0;
    free(cores);
fail_booted:
    free(booted_cores);
    return status;
}

static void clear_profile_locked(npu_service_device_t *device)
{
    device->profile_allocation = 0;
    device->profile_cores = 0;
    device->profile_core_count = 0;
    device->profile_generation = 0;
    device->profile_active = 0;
    device->profile_transition = 0;
}

void npu_service_profile_discard(npu_service_device_t *device)
{
    npu_hal_allocation_t *allocation;
    struct npu_service_profile_core *cores;

    if (!device)
        return;
    pthread_mutex_lock(&device->mutex);
    while (device->profile_readers)
        (void)pthread_cond_wait(&device->condition, &device->mutex);
    allocation = device->profile_allocation;
    cores = device->profile_cores;
    clear_profile_locked(device);
    pthread_mutex_unlock(&device->mutex);
    if (allocation)
        (void)npu_hal_memory_free(device->hal, allocation);
    free(cores);
}

int npu_service_profile_start(npu_service_device_t *device,
                              uint64_t buffer_size)
{
    npu_hal_allocation_t *allocation = 0;
    struct npu_service_profile_core *cores = 0;
    size_t core_slots = 0;
    uint64_t usable_size = 0;
    uint64_t generation;
    size_t index;
    int cleanup_remaining = 0;
    int status;

    if (!device || !buffer_size)
        return -EINVAL;
    pthread_mutex_lock(&device->mutex);
    if (device->profile_allocation || device->profile_transition) {
        pthread_mutex_unlock(&device->mutex);
        return -EBUSY;
    }
    device->profile_transition = 1;
    generation = ++device->profile_next_generation;
    if (!generation)
        generation = ++device->profile_next_generation;
    pthread_mutex_unlock(&device->mutex);

    status = allocate_profile(device, buffer_size, &allocation, &cores,
                              &core_slots, &usable_size);
    if (status < 0)
        goto fail_uninstalled;
    pthread_mutex_lock(&device->mutex);
    device->profile_allocation = allocation;
    device->profile_cores = cores;
    device->profile_core_count = core_slots;
    device->profile_generation = generation;
    device->profile_active = 1;
    pthread_mutex_unlock(&device->mutex);

    for (index = 0; index < core_slots; ++index) {
        int enabled;

        if (!cores[index].capacity)
            continue;
        pthread_mutex_lock(&device->mutex);
        cores[index].configured = 1;
        pthread_mutex_unlock(&device->mutex);
        status = profile_config_call(device, &cores[index],
                                     RPC_PROFILE_START, generation, 0,
                                     &enabled);
        if (status < 0) {
            pthread_mutex_lock(&device->mutex);
            if (enabled == 0)
                cores[index].configured = 0;
            device->profile_active = 0;
            pthread_mutex_unlock(&device->mutex);
            goto fail_installed;
        }
    }
    pthread_mutex_lock(&device->mutex);
    device->profile_transition = 0;
    pthread_mutex_unlock(&device->mutex);
    return 0;

fail_installed:
    for (index = 0; index < core_slots; ++index) {
        if (cores[index].configured) {
            int enabled;
            int cleanup_status = profile_config_call(
                device, &cores[index], RPC_PROFILE_STOP, generation, 0,
                &enabled);

            pthread_mutex_lock(&device->mutex);
            if (cleanup_status == 0 || enabled == 0)
                cores[index].configured = 0;
            else
                cleanup_remaining = 1;
            pthread_mutex_unlock(&device->mutex);
        }
    }
    pthread_mutex_lock(&device->mutex);
    while (device->profile_readers)
        (void)pthread_cond_wait(&device->condition, &device->mutex);
    if (!cleanup_remaining)
        clear_profile_locked(device);
    else
        device->profile_transition = 0;
    pthread_mutex_unlock(&device->mutex);
    if (!cleanup_remaining) {
        (void)npu_hal_memory_free(device->hal, allocation);
        free(cores);
    }
    return status;

fail_uninstalled:
    pthread_mutex_lock(&device->mutex);
    device->profile_transition = 0;
    pthread_mutex_unlock(&device->mutex);
    return status;
}

int npu_service_profile_stop(npu_service_device_t *device)
{
    npu_hal_allocation_t *allocation;
    struct npu_service_profile_core *cores;
    size_t core_slots;
    uint64_t generation;
    size_t index;
    int first_status = 0;
    int remaining = 0;

    if (!device)
        return -EINVAL;
    pthread_mutex_lock(&device->mutex);
    if (!device->profile_allocation || device->profile_transition) {
        pthread_mutex_unlock(&device->mutex);
        return device->profile_transition ? -EBUSY : -ENOENT;
    }
    device->profile_active = 0;
    device->profile_transition = 1;
    allocation = device->profile_allocation;
    cores = device->profile_cores;
    core_slots = device->profile_core_count;
    generation = device->profile_generation;
    pthread_mutex_unlock(&device->mutex);

    for (index = 0; index < core_slots; ++index) {
        uint32_t count = 0;
        int config_status;
        int copy_status = 0;
        int enabled;

        if (!cores[index].configured)
            continue;
        config_status = profile_config_call(device, &cores[index],
                                            RPC_PROFILE_STOP, generation,
                                            &count, &enabled);
        if (config_status == 0 && count) {
            int keep;

            copy_status = npu_service_profile_copy(
                device, cores[index].target, generation,
                cores[index].address, cores[index].capacity, count, &keep);
        }
        if (config_status < 0) {
            if (!first_status)
                first_status = config_status;
            if (enabled == 0) {
                pthread_mutex_lock(&device->mutex);
                cores[index].configured = 0;
                pthread_mutex_unlock(&device->mutex);
            } else {
                remaining = 1;
            }
        } else {
            pthread_mutex_lock(&device->mutex);
            cores[index].configured = 0;
            pthread_mutex_unlock(&device->mutex);
            if (copy_status < 0 && !first_status)
                first_status = copy_status;
        }
    }
    pthread_mutex_lock(&device->mutex);
    while (device->profile_readers)
        (void)pthread_cond_wait(&device->condition, &device->mutex);
    if (!remaining)
        clear_profile_locked(device);
    else
        device->profile_transition = 0;
    pthread_mutex_unlock(&device->mutex);
    if (!remaining) {
        int status = npu_hal_memory_free(device->hal, allocation);

        free(cores);
        if (!first_status && status < 0)
            first_status = status;
    }
    return first_status;
}
