#define _POSIX_C_SOURCE 200809L

#include "service_internal.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "trap.h"

#define NPU_DEFAULT_RPC_POLL_US 1000U
#define NPU_DEFAULT_TIMEOUT_MS  5000U
#define NPU_DEFAULT_STACK_SIZE  UINT64_C(0x10000)
#define NPU_DEFAULT_PRINTF_SIZE UINT64_C(0x1000)

_Static_assert((int)NPU_SERVICE_LAUNCH_SUBMITTED == (int)RPC_PROC_SUBMITTED,
               "Host and Device launch state ABI");
_Static_assert((int)NPU_SERVICE_LAUNCH_RUNNING == (int)RPC_PROC_RUNNING,
               "Host and Device launch state ABI");
_Static_assert((int)NPU_SERVICE_LAUNCH_EXITED == (int)RPC_PROC_EXITED,
               "Host and Device launch state ABI");
_Static_assert((int)NPU_SERVICE_LAUNCH_FAILED == (int)RPC_PROC_FAILED,
               "Host and Device launch state ABI");
_Static_assert(NPU_SERVICE_PROC_MAX_RECORDS == RPC_PROC_PS_MAX_RECORDS,
               "Host and Device proc snapshot capacity ABI");
_Static_assert(sizeof(npu_service_proc_info_t) ==
               sizeof(struct rpc_proc_info),
               "Host and Device proc record ABI");

uint64_t npu_now_ms(void)
{
    struct timespec now;

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0;
    return (uint64_t)now.tv_sec * 1000U +
           (uint64_t)now.tv_nsec / 1000000U;
}

void npu_delay_us(uint32_t delay_us)
{
    struct timespec delay;

    delay.tv_sec = delay_us / 1000000U;
    delay.tv_nsec = (long)(delay_us % 1000000U) * 1000L;
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR)
        ;
}

size_t npu_service_core_index(const npu_service_device_t *device,
                              npu_hal_core_t core)
{
    return (size_t)core.cluster * device->hal_info.cores_per_cluster +
           core.core;
}

static void default_output(void *context, uint32_t cluster, uint32_t core,
                           uint32_t proc, const void *data, size_t size,
                           uint32_t flags)
{
    (void)context;
    (void)cluster;
    (void)core;
    (void)proc;
    if (flags & NPU_SERVICE_OUTPUT_CLEAR)
        fflush(stdout);
    if (data && size) {
        (void)fwrite(data, 1, size, stdout);
        fflush(stdout);
    }
}

static void default_trap(void *context, uint32_t cluster, uint32_t core,
                         const struct trap_record *record)
{
    (void)context;
    fprintf(stderr,
            "dayan_server: trap cluster=%u core=%u proc=%u cause=0x%llx "
            "epc=0x%llx value=0x%llx\n",
            cluster, core, record ? record->thread_id : 0,
            (unsigned long long)(record ? record->mcause : 0),
            (unsigned long long)(record ? record->mepc : 0),
            (unsigned long long)(record ? record->mtval : 0));
}

static void default_proc_exit(void *context, uint32_t cluster,
                              uint32_t core, uint32_t proc,
                              int64_t result, uint64_t cookie)
{
    (void)context;
    (void)cluster;
    (void)core;
    (void)proc;
    (void)result;
    (void)cookie;
}

void npu_service_options_init(npu_service_options_t *options)
{
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->version = NPU_SERVICE_OPTIONS_VERSION;
    options->rpc_poll_interval_us = NPU_DEFAULT_RPC_POLL_US;
    options->default_timeout_ms = NPU_DEFAULT_TIMEOUT_MS;
    options->output = default_output;
    options->trap = default_trap;
    options->proc_exit = default_proc_exit;
}

void npu_service_launch_desc_init(npu_service_launch_desc_t *desc)
{
    if (!desc)
        return;
    memset(desc, 0, sizeof(*desc));
    desc->version = NPU_SERVICE_LAUNCH_DESC_VERSION;
    desc->stack_size = NPU_DEFAULT_STACK_SIZE;
    desc->printf_buffer_size = NPU_DEFAULT_PRINTF_SIZE;
    desc->task_dim[0] = 1;
    desc->task_dim[1] = 1;
    desc->task_dim[2] = 1;
    desc->task_mode = 1;
}

static int options_valid(const npu_service_options_t *options)
{
    return options && options->version == NPU_SERVICE_OPTIONS_VERSION &&
           options->rpc_poll_interval_us && options->default_timeout_ms &&
           !options->reserved && options->output && options->trap &&
           options->proc_exit;
}

int npu_service_open(npu_service_device_t **device_out,
                     npu_hal_t *hal,
                     const npu_service_options_t *options)
{
    npu_service_options_t defaults;
    npu_service_device_t *device;
    size_t core_count;
    int mutex_initialized = 0;
    int kernel_mutex_initialized = 0;
    int condition_initialized = 0;
    int status;

    if (!device_out || !hal)
        return -EINVAL;
    *device_out = 0;
    if (!options) {
        npu_service_options_init(&defaults);
        options = &defaults;
    }
    if (!options_valid(options))
        return -EINVAL;
    device = calloc(1, sizeof(*device));
    if (!device)
        return -ENOMEM;
    device->options = *options;
    device->hal = hal;
    status = npu_hal_get_info(device->hal, &device->hal_info);
    if (status < 0)
        goto fail;
    core_count = (size_t)device->hal_info.cluster_count *
                 device->hal_info.cores_per_cluster;
    device->calls = calloc(core_count, sizeof(*device->calls));
    device->clusters = calloc(device->hal_info.cluster_count,
                              sizeof(*device->clusters));
    if (!device->calls || !device->clusters) {
        status = -ENOMEM;
        goto fail;
    }
    status = pthread_mutex_init(&device->mutex, 0);
    if (status) {
        status = -status;
        goto fail;
    }
    mutex_initialized = 1;
    status = pthread_mutex_init(&device->kernel_mutex, 0);
    if (status) {
        status = -status;
        goto fail;
    }
    kernel_mutex_initialized = 1;
    status = pthread_cond_init(&device->condition, 0);
    if (status) {
        status = -status;
        goto fail;
    }
    condition_initialized = 1;
    status = pthread_create(&device->worker, 0, npu_rpc_worker, device);
    if (status) {
        status = -status;
        goto fail;
    }
    device->worker_started = 1;
    *device_out = device;
    return 0;

fail:
    if (condition_initialized)
        pthread_cond_destroy(&device->condition);
    if (kernel_mutex_initialized)
        pthread_mutex_destroy(&device->kernel_mutex);
    if (mutex_initialized)
        pthread_mutex_destroy(&device->mutex);
    free(device->clusters);
    free(device->calls);
    free(device);
    return status;
}

static void free_heap_allocations(npu_service_device_t *device)
{
    struct npu_service_heap_allocation *item = device->heap_allocations;

    while (item) {
        struct npu_service_heap_allocation *next = item->next;

        (void)npu_hal_memory_free(device->hal, item->allocation);
        free(item);
        item = next;
    }
    device->heap_allocations = 0;
}

static void free_firmware_allocations(npu_service_device_t *device)
{
    struct npu_service_firmware_allocation *item =
        device->firmware_allocations;

    while (item) {
        struct npu_service_firmware_allocation *next = item->next;

        (void)npu_hal_memory_free(device->hal, item->allocation);
        free(item);
        item = next;
    }
    device->firmware_allocations = 0;
}

int npu_service_close(npu_service_device_t *device)
{
    if (!device)
        return -EINVAL;
    pthread_mutex_lock(&device->mutex);
    if (device->active_launches) {
        pthread_mutex_unlock(&device->mutex);
        return -EBUSY;
    }
    device->worker_stop = 1;
    pthread_cond_broadcast(&device->condition);
    pthread_mutex_unlock(&device->mutex);
    if (device->worker_started)
        (void)pthread_join(device->worker, 0);
    npu_service_kernel_state_destroy(device);
    free_heap_allocations(device);
    free_firmware_allocations(device);
    pthread_cond_destroy(&device->condition);
    pthread_mutex_destroy(&device->kernel_mutex);
    pthread_mutex_destroy(&device->mutex);
    free(device->clusters);
    free(device->calls);
    free(device);
    return 0;
}

const npu_service_options_t *npu_service_get_options(
    const npu_service_device_t *device)
{
    return device ? &device->options : 0;
}

int npu_service_alloc_visible(npu_service_device_t *device,
                              npu_hal_core_t viewer,
                              npu_hal_memory_class_t memory_class,
                              npu_hal_memory_usage_t usage,
                              uint32_t owner, uint64_t size,
                              uint64_t alignment,
                              npu_hal_allocation_t **allocation_out)
{
    static const npu_hal_memory_scope_t scopes[] = {
        NPU_HAL_MEMORY_CORE,
        NPU_HAL_MEMORY_CLUSTER,
        NPU_HAL_MEMORY_DEVICE,
    };
    npu_hal_memory_request_t request;
    size_t index;
    int status = -ENOMEM;

    if (!device || !allocation_out || !size ||
        !npu_hal_core_valid(device->hal, viewer))
        return -EINVAL;
    *allocation_out = 0;
    npu_hal_memory_request_init(&request);
    request.memory_class = memory_class;
    request.usage = usage;
    request.owner = owner;
    request.size = size;
    request.alignment = alignment;
    for (index = 0; index < sizeof(scopes) / sizeof(scopes[0]); ++index) {
        request.scope = scopes[index];
        request.location.cluster = scopes[index] == NPU_HAL_MEMORY_DEVICE ?
            NPU_HAL_ANY_CLUSTER : viewer.cluster;
        request.location.core = scopes[index] == NPU_HAL_MEMORY_CORE ?
            viewer.core : NPU_HAL_ANY_CORE;
        status = npu_hal_memory_alloc(device->hal, &request, allocation_out);
        if (status == 0 || status != -ENOMEM)
            return status;
    }
    return status;
}

static int read_file(const char *path, unsigned char **image_out,
                     size_t *size_out)
{
    unsigned char *image;
    long length;
    FILE *file;

    if (!path || !image_out || !size_out)
        return -EINVAL;
    *image_out = 0;
    *size_out = 0;
    file = fopen(path, "rb");
    if (!file)
        return -errno;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) <= 0 ||
        (uint64_t)length > SIZE_MAX ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -EIO;
    }
    image = malloc((size_t)length);
    if (!image) {
        fclose(file);
        return -ENOMEM;
    }
    if (fread(image, 1, (size_t)length, file) != (size_t)length) {
        free(image);
        fclose(file);
        return -EIO;
    }
    if (fclose(file) != 0) {
        free(image);
        return -EIO;
    }
    *image_out = image;
    *size_out = (size_t)length;
    return 0;
}

int npu_service_load_bl1(npu_service_device_t *device, uint32_t cluster,
                         const void *elf_image, size_t elf_size,
                         uint32_t core_mask)
{
    if (!device || !elf_image || !elf_size)
        return -EINVAL;
    return npu_service_load_bl1_elf(device, cluster, elf_image, elf_size,
                                    core_mask);
}

int npu_service_load_bl1_file(npu_service_device_t *device,
                              uint32_t cluster, const char *path,
                              uint32_t core_mask)
{
    unsigned char *image;
    size_t size;
    int status;

    if (!device)
        return -EINVAL;
    status = read_file(path, &image, &size);
    if (status < 0)
        return status;
    status = npu_service_load_bl1_elf(device, cluster, image, size,
                                      core_mask);
    free(image);
    return status;
}

static int task_valid(const npu_service_launch_desc_t *desc)
{
    size_t axis;

    if (desc->task_mode != 1U && desc->task_mode != 4U)
        return 0;
    if (desc->task_mode == 4U && desc->task_dim[0] != 4U)
        return 0;
    for (axis = 0; axis < 3; ++axis) {
        if (!desc->task_dim[axis] ||
            desc->task_id[axis] >= desc->task_dim[axis])
            return 0;
    }
    return 1;
}

static int shared_address(npu_service_device_t *device,
                          const npu_service_launch_desc_t *desc,
                          uint64_t *address_out)
{
    npu_hal_allocation_info_t info;
    uint64_t address;
    int status;

    if (!desc->shared_size) {
        if (desc->shared_allocation || desc->shared_offset)
            return -EINVAL;
        *address_out = 0;
        return 0;
    }
    if (!desc->shared_allocation)
        return -EINVAL;
    status = npu_hal_memory_info(device->hal, desc->shared_allocation, &info);
    if (status < 0 || info.memory_class != NPU_HAL_MEMORY_COHERENT ||
        desc->shared_offset > info.size ||
        desc->shared_size > info.size - desc->shared_offset)
        return status < 0 ? status : -EINVAL;
    status = npu_hal_memory_address(device->hal, desc->shared_allocation,
                                    desc->target, &address);
    if (status < 0 || address > UINT64_MAX - desc->shared_offset)
        return status < 0 ? status : -EOVERFLOW;
    address += desc->shared_offset;
    if (address & 15U)
        return -EINVAL;
    *address_out = address;
    return 0;
}

static int uram_address(npu_service_device_t *device,
                        const npu_service_launch_desc_t *desc,
                        uint64_t *address_out)
{
    npu_hal_allocation_info_t info;
    uint64_t address;
    int status;

    if (!desc->uram_size) {
        if (desc->uram_allocation || desc->uram_offset)
            return -EINVAL;
        *address_out = 0;
        return 0;
    }
    if (!desc->uram_allocation ||
        (desc->uram_size & (RPC_PROC_URAM_GRANULE - 1U)))
        return -EINVAL;
    status = npu_hal_memory_info(device->hal, desc->uram_allocation, &info);
    if (status < 0 || info.memory_class != NPU_HAL_MEMORY_URAM ||
        info.scope != NPU_HAL_MEMORY_CORE ||
        info.location.cluster != desc->target.cluster ||
        info.location.core != desc->target.core ||
        desc->uram_offset > info.size ||
        desc->uram_size > info.size - desc->uram_offset)
        return status < 0 ? status : -EINVAL;
    status = npu_hal_memory_address(device->hal, desc->uram_allocation,
                                    desc->target, &address);
    if (status < 0 || address > UINT64_MAX - desc->uram_offset)
        return status < 0 ? status : -EOVERFLOW;
    address += desc->uram_offset;
    if (address & (RPC_PROC_URAM_GRANULE - 1U))
        return -EINVAL;
    *address_out = address;
    return 0;
}

int npu_service_rpc_call(npu_service_device_t *device, npu_hal_core_t core,
                         uint16_t opcode, uint64_t argument,
                         uint64_t *response_out, uint32_t timeout_ms)
{
    struct npu_pending_call *call;
    struct timespec deadline;
    uint16_t expected;
    int status;

    if (!device || !response_out || !npu_hal_core_valid(device->hal, core) ||
        argument > RPC_COMMAND_ARG_MASK)
        return -EINVAL;
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0)
        return -EIO;
    deadline.tv_sec += timeout_ms / 1000U;
    deadline.tv_nsec += (long)(timeout_ms % 1000U) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        ++deadline.tv_sec;
        deadline.tv_nsec -= 1000000000L;
    }
    expected = (uint16_t)(opcode | UINT16_C(0x8000));
    call = &device->calls[npu_service_core_index(device, core)];
    pthread_mutex_lock(&device->mutex);
    if (call->pending) {
        pthread_mutex_unlock(&device->mutex);
        return -EBUSY;
    }
    call->expected_opcode = expected;
    call->expected_argument = argument;
    call->pending = 1;
    call->status = -EINPROGRESS;
    pthread_mutex_unlock(&device->mutex);

    status = npu_hal_wait_command_consumed(device->hal, core, timeout_ms);
    if (status == 0)
        status = npu_hal_command_submit(device->hal, core,
                                        rpc_command_pack(opcode, argument));
    if (status < 0) {
        pthread_mutex_lock(&device->mutex);
        call->pending = 0;
        call->status = status;
        pthread_mutex_unlock(&device->mutex);
        return status;
    }
    pthread_mutex_lock(&device->mutex);
    while (call->pending) {
        status = pthread_cond_timedwait(&device->condition, &device->mutex,
                                        &deadline);
        if (status == ETIMEDOUT) {
            call->pending = 0;
            call->status = -ETIMEDOUT;
            break;
        }
        if (status) {
            call->pending = 0;
            call->status = -status;
            break;
        }
    }
    status = call->status;
    *response_out = call->response;
    pthread_mutex_unlock(&device->mutex);
    return status;
}

int npu_service_launch(npu_service_device_t *device,
                       const npu_service_launch_desc_t *desc,
                       npu_service_launch_t **launch_out)
{
    struct rpc_proc_create_request request;
    npu_service_launch_t *launch;
    npu_hal_allocation_t *allocation = 0;
    uint64_t shared_base;
    uint64_t uram_base;
    uint64_t request_address;
    uint64_t response;
    uint32_t booted;
    size_t axis;
    int status;

    if (!device || !desc || !launch_out ||
        desc->version != NPU_SERVICE_LAUNCH_DESC_VERSION || desc->flags ||
        !npu_hal_core_valid(device->hal, desc->target) ||
        desc->argc > NPU_SERVICE_LAUNCH_MAX_ARGS || !desc->entry ||
        (desc->entry & 1U) || (desc->entry & ~RPC_COMMAND_ARG_MASK) ||
        desc->stack_size < RPC_PROC_MIN_STACK_SIZE ||
        desc->stack_size > RPC_PROC_MAX_STACK_SIZE ||
        desc->printf_buffer_size > RPC_PROC_MAX_PRINTF_SIZE ||
        desc->max_child_procs > RPC_PROC_MAX_CHILD_PROCS ||
        desc->resource_reserved ||
        !task_valid(desc) ||
        !npu_hal_address_valid(device->hal, desc->target, desc->entry, 2))
        return -EINVAL;
    *launch_out = 0;
    pthread_mutex_lock(&device->mutex);
    booted = device->clusters[desc->target.cluster].booted_cores;
    pthread_mutex_unlock(&device->mutex);
    if (!(booted & (UINT32_C(1) << desc->target.core)))
        return -ENODEV;
    status = shared_address(device, desc, &shared_base);
    if (status < 0)
        return status;
    status = uram_address(device, desc, &uram_base);
    if (status < 0)
        return status;
    launch = calloc(1, sizeof(*launch));
    if (!launch)
        return -ENOMEM;
    status = npu_service_alloc_visible(
        device, desc->target, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_LAUNCH,
        (uint32_t)npu_service_core_index(device, desc->target),
        sizeof(request), 64, &allocation);
    if (status < 0)
        goto fail;
    status = npu_hal_memory_address(device->hal, allocation, desc->target,
                                    &request_address);
    if (status < 0 || request_address > RPC_COMMAND_ARG_MASK) {
        status = status < 0 ? status : -EOVERFLOW;
        goto fail;
    }
    memset(&request, 0, sizeof(request));
    request.version = RPC_PROC_CREATE_VERSION;
    request.size = sizeof(request);
    request.state = RPC_PROC_SUBMITTED;
    request.entry = desc->entry;
    request.stack_size = desc->stack_size;
    request.shared_base = shared_base;
    request.shared_size = desc->shared_size;
    request.max_child_procs = desc->max_child_procs;
    request.uram_base = uram_base;
    request.uram_size = desc->uram_size;
    request.printf_buffer_size = desc->printf_buffer_size;
    request.argc = desc->argc;
    memcpy(request.args, desc->args, desc->argc * sizeof(desc->args[0]));
    request.cookie = desc->cookie;
    for (axis = 0; axis < 3; ++axis) {
        request.task_id[axis] = desc->task_id[axis];
        request.task_dim[axis] = desc->task_dim[axis];
    }
    request.task_mode = desc->task_mode;
    status = npu_hal_memory_write(device->hal, allocation, desc->target, 0,
                                  &request, sizeof(request));
    if (status < 0)
        goto fail;
    status = npu_service_rpc_call(device, desc->target, RPC_PROC_CREATE,
                                  request_address, &response,
                                  device->options.default_timeout_ms);
    if (status < 0)
        goto fail;
    if (rpc_command_opcode(response) != RPC_PROC_CREATE_RESPONSE ||
        rpc_command_arg(response) != request_address) {
        status = -EPROTO;
        goto fail;
    }
    status = npu_hal_memory_read(device->hal, allocation, desc->target,
                                 &request, 0, sizeof(request));
    if (status < 0)
        goto fail;
    if (request.version != RPC_PROC_CREATE_VERSION ||
        request.size != sizeof(request) || request.hart_id != desc->target.core ||
        request.cookie != desc->cookie ||
        (request.state != RPC_PROC_RUNNING &&
         request.state != RPC_PROC_EXITED &&
         request.state != RPC_PROC_FAILED)) {
        status = -EPROTO;
        goto fail;
    }
    if (request.state == RPC_PROC_FAILED) {
        status = request.status < 0 && request.status >= INT32_MIN ?
            (int)request.status : -EIO;
        goto fail;
    }
    launch->device = device;
    launch->target = desc->target;
    launch->request_allocation = allocation;
    launch->request_address = request_address;
    pthread_mutex_lock(&device->mutex);
    ++device->active_launches;
    pthread_mutex_unlock(&device->mutex);
    *launch_out = launch;
    return 0;

fail:
    if (allocation)
        (void)npu_hal_memory_free(device->hal, allocation);
    free(launch);
    return status;
}

int npu_service_launch_query(npu_service_launch_t *launch,
                             npu_service_launch_info_t *info)
{
    struct rpc_proc_create_request request;
    int status;

    if (!launch || !info || launch->released)
        return -EINVAL;
    status = npu_hal_memory_read(launch->device->hal,
                                 launch->request_allocation, launch->target,
                                 &request, 0, sizeof(request));
    if (status < 0)
        return status;
    if (request.version != RPC_PROC_CREATE_VERSION ||
        request.size != sizeof(request) ||
        request.hart_id != launch->target.core ||
        request.state > RPC_PROC_FAILED)
        return -EPROTO;
    memset(info, 0, sizeof(*info));
    info->state = request.state;
    info->target = launch->target;
    info->proc_id = request.proc_id;
    info->status = request.status;
    memcpy(info->task_id, request.task_id, sizeof(info->task_id));
    info->task_mode = request.task_mode;
    info->cookie = request.cookie;
    return 0;
}

int npu_service_launch_wait(npu_service_launch_t *launch,
                            uint32_t timeout_ms,
                            npu_service_launch_info_t *info)
{
    uint64_t deadline;
    int status;

    if (!launch || !info)
        return -EINVAL;
    deadline = timeout_ms == NPU_SERVICE_TIMEOUT_INFINITE ? UINT64_MAX :
               npu_now_ms() + timeout_ms;
    for (;;) {
        status = npu_service_launch_query(launch, info);
        if (status < 0)
            return status;
        if (info->state == NPU_SERVICE_LAUNCH_EXITED ||
            info->state == NPU_SERVICE_LAUNCH_FAILED)
            return 0;
        if (npu_now_ms() >= deadline)
            return -ETIMEDOUT;
        npu_delay_us(launch->device->options.rpc_poll_interval_us);
    }
}

int npu_service_launch_release(npu_service_launch_t *launch)
{
    npu_service_launch_info_t info;
    npu_service_device_t *device;
    int status;

    if (!launch || launch->released)
        return -EINVAL;
    status = npu_service_launch_query(launch, &info);
    if (status < 0)
        return status;
    if (info.state != NPU_SERVICE_LAUNCH_EXITED &&
        info.state != NPU_SERVICE_LAUNCH_FAILED)
        return -EBUSY;
    device = launch->device;
    status = npu_hal_memory_free(device->hal, launch->request_allocation);
    if (status < 0)
        return status;
    launch->released = 1;
    pthread_mutex_lock(&device->mutex);
    if (device->active_launches)
        --device->active_launches;
    pthread_mutex_unlock(&device->mutex);
    free(launch);
    return 0;
}

int npu_service_rpc_ps(npu_service_device_t *device, npu_hal_core_t target,
                       npu_service_proc_snapshot_t *snapshot_out)
{
    struct rpc_proc_ps_snapshot snapshot;
    npu_hal_allocation_t *allocation = 0;
    uint64_t address;
    uint64_t response;
    uint32_t booted;
    int status;

    if (!device || !snapshot_out ||
        !npu_hal_core_valid(device->hal, target))
        return -EINVAL;
    pthread_mutex_lock(&device->mutex);
    booted = device->clusters[target.cluster].booted_cores;
    pthread_mutex_unlock(&device->mutex);
    if (!(booted & (UINT32_C(1) << target.core)))
        return -ENODEV;
    status = npu_service_alloc_visible(
        device, target, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_LAUNCH,
        (uint32_t)npu_service_core_index(device, target),
        sizeof(snapshot), 64, &allocation);
    if (status < 0)
        return status;
    status = npu_hal_memory_address(device->hal, allocation, target,
                                    &address);
    if (status < 0 || address > RPC_COMMAND_ARG_MASK) {
        status = status < 0 ? status : -EOVERFLOW;
        goto out;
    }
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.version = RPC_PROC_PS_VERSION;
    snapshot.size = sizeof(snapshot);
    snapshot.capacity = RPC_PROC_PS_MAX_RECORDS;
    snapshot.status = -EIO;
    status = npu_hal_memory_write(device->hal, allocation, target, 0,
                                  &snapshot, sizeof(snapshot));
    if (status < 0)
        goto out;
    status = npu_service_rpc_call(device, target, RPC_PROC_PS, address,
                                  &response,
                                  device->options.default_timeout_ms);
    if (status < 0)
        goto out;
    if (rpc_command_opcode(response) != RPC_PROC_PS_RESPONSE ||
        rpc_command_arg(response) != address) {
        status = -EPROTO;
        goto out;
    }
    status = npu_hal_memory_read(device->hal, allocation, target, &snapshot,
                                 0, sizeof(snapshot));
    if (status < 0)
        goto out;
    if (snapshot.version != RPC_PROC_PS_VERSION ||
        snapshot.size != sizeof(snapshot) ||
        snapshot.capacity != RPC_PROC_PS_MAX_RECORDS || snapshot.flags ||
        snapshot.reserved || snapshot.hart_id != target.core ||
        snapshot.count > RPC_PROC_PS_MAX_RECORDS) {
        status = -EPROTO;
        goto out;
    }
    if (snapshot.status < 0) {
        status = snapshot.status >= INT32_MIN ? (int)snapshot.status : -EIO;
        goto out;
    }
    memset(snapshot_out, 0, sizeof(*snapshot_out));
    snapshot_out->target = target;
    snapshot_out->count = snapshot.count;
    snapshot_out->global_free_slots = snapshot.global_free_slots;
    snapshot_out->reserved_slots = snapshot.reserved_slots;
    memcpy(snapshot_out->records, snapshot.records,
           snapshot.count * sizeof(snapshot.records[0]));
    status = 0;

out:
    if (allocation) {
        int free_status = npu_hal_memory_free(device->hal, allocation);

        if (status == 0 && free_status < 0)
            status = free_status;
    }
    return status;
}

int npu_service_rpc_kill_all(npu_service_device_t *device,
                             npu_hal_core_t target,
                             npu_service_kill_result_t *result_out)
{
    struct rpc_proc_kill_result result;
    npu_hal_allocation_t *allocation = 0;
    uint64_t address;
    uint64_t response;
    uint32_t booted;
    int status;

    if (!device || !result_out ||
        !npu_hal_core_valid(device->hal, target))
        return -EINVAL;
    pthread_mutex_lock(&device->mutex);
    booted = device->clusters[target.cluster].booted_cores;
    pthread_mutex_unlock(&device->mutex);
    if (!(booted & (UINT32_C(1) << target.core)))
        return -ENODEV;
    status = npu_service_alloc_visible(
        device, target, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_LAUNCH,
        (uint32_t)npu_service_core_index(device, target),
        sizeof(result), 64, &allocation);
    if (status < 0)
        return status;
    status = npu_hal_memory_address(device->hal, allocation, target,
                                    &address);
    if (status < 0 || address > RPC_COMMAND_ARG_MASK) {
        status = status < 0 ? status : -EOVERFLOW;
        goto out;
    }
    memset(&result, 0, sizeof(result));
    result.version = RPC_PROC_KILL_ALL_VERSION;
    result.size = sizeof(result);
    result.status = -EIO;
    status = npu_hal_memory_write(device->hal, allocation, target, 0,
                                  &result, sizeof(result));
    if (status < 0)
        goto out;
    status = npu_service_rpc_call(device, target, RPC_PROC_KILL_ALL,
                                  address, &response,
                                  device->options.default_timeout_ms);
    if (status < 0)
        goto out;
    if (rpc_command_opcode(response) != RPC_PROC_KILL_ALL_RESPONSE ||
        rpc_command_arg(response) != address) {
        status = -EPROTO;
        goto out;
    }
    status = npu_hal_memory_read(device->hal, allocation, target, &result,
                                 0, sizeof(result));
    if (status < 0)
        goto out;
    if (result.version != RPC_PROC_KILL_ALL_VERSION ||
        result.size != sizeof(result) || result.hart_id != target.core ||
        result.flags || result.reserved) {
        status = -EPROTO;
        goto out;
    }
    if (result.status < 0) {
        status = result.status >= INT32_MIN ? (int)result.status : -EIO;
        goto out;
    }
    memset(result_out, 0, sizeof(*result_out));
    result_out->target = target;
    result_out->status = result.status;
    result_out->killed_roots = result.killed_roots;
    result_out->killed_children = result.killed_children;
    result_out->released_slots = result.released_slots;
    result_out->waiting_produce = result.waiting_produce;
    status = 0;

out:
    if (allocation) {
        int free_status = npu_hal_memory_free(device->hal, allocation);

        if (status == 0 && free_status < 0)
            status = free_status;
    }
    return status;
}
