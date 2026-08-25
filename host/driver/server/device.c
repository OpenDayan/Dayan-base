#define _POSIX_C_SOURCE 200809L

#include "service_internal.h"
#include "tensor_context.h"
#include "tensor_interrupt.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "trap.h"

#define NPU_DEFAULT_RPC_POLL_US 1000U
#define NPU_DEFAULT_TIMEOUT_MS  60000U
#define NPU_DEFAULT_STACK_SIZE  UINT64_C(0x10000)
#define NPU_DEFAULT_PRINTF_SIZE UINT64_C(0x1000)
#define NPU_RPC_CANCEL_POLL_MS  100U

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
_Static_assert(NPU_SERVICE_PROC_NAME_SIZE == RPC_PROC_NAME_SIZE,
               "Host and Device proc name size ABI");
_Static_assert(sizeof(npu_service_proc_info_t) ==
               sizeof(struct rpc_proc_info),
               "Host and Device proc record ABI");

static void *npu_launch_agent(void *arg);

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

static const char *tensor_interrupt_source_name(uint16_t source)
{
    static const char *const names[] = {
        "neomat_uram_a_read_range", "neomat_uram_a_read_align",
        "neomat_uram_b_read_range", "neomat_uram_b_read_align",
        "neomat_uram_scale_read_range", "neomat_uram_scale_read_align",
        "neomat_uram_c_read_range", "neomat_uram_c_read_align",
        "neomat_uram_c_write_range", "neomat_uram_c_write_align",
        "neomat_tcsr_access_error", "neomat_decode_error",
        "neodma_tcsr_access_error", "neodma_uram_table_range",
        "neodma_uram_table_align", "neodma_uram_read_range",
        "neodma_uram_read_align", "neodma_uram_write_range",
        "neodma_uram_write_align", "neodma_decode_error",
        "flexvec_decode_error", "flexvec_uram_group_0_range",
        "flexvec_uram_group_0_align", "flexvec_uram_group_1_range",
        "flexvec_uram_group_1_align", "neovec_uram_lda_range",
        "neovec_uram_lda_align", "neovec_uram_ldb_range",
        "neovec_uram_ldb_align", "neovec_uram_store_range",
        "neovec_uram_store_align", "neovec_tcsr_access_error",
        "neovec_decode_error", "instruction_decode_error",
    };

    return source < sizeof(names) / sizeof(names[0]) ? names[source] :
                                                       "unknown";
}

static void default_tensor_interrupt(
    void *context, uint32_t cluster, uint32_t core,
    const struct tensor_interrupt_record *record)
{
    uint64_t latest = record && record->recent_count ?
        record->recent[record->recent_count - 1U] : 0;
    uint16_t source = (uint16_t)(latest >> 32);

    (void)context;
    fprintf(stderr,
            "dayan_server: tensor interrupt cluster=%u core=%u proc=%u "
            "source=%u(%s) info=0x%08x context=0x%014llx count=%llu\n",
            cluster, core, record ? record->thread_id : 0, source,
            tensor_interrupt_source_name(source), (uint32_t)latest,
            (unsigned long long)(record ? record->tensor_context : 0),
            (unsigned long long)(record ? record->interrupt_count : 0));
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

static int default_profile(void *context, uint32_t cluster, uint32_t core,
                           const void *data, size_t size)
{
    (void)context;
    (void)cluster;
    (void)core;
    (void)data;
    (void)size;
    return 0;
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
    options->tensor_interrupt = default_tensor_interrupt;
    options->proc_exit = default_proc_exit;
    options->profile = default_profile;
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
           options->tensor_interrupt &&
           options->proc_exit && options->profile;
}

int npu_service_open(npu_service_device_t **device_out,
                     npu_hal_t *hal,
                     const npu_service_options_t *options)
{
    npu_service_options_t defaults;
    npu_service_device_t *device;
    size_t core_count;
    size_t index;
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
    device->rpc_queue_capacity = NPU_SERVICE_RPC_QUEUE_CAPACITY;
    status = npu_hal_get_info(device->hal, &device->hal_info);
    if (status < 0)
        goto fail;
    core_count = (size_t)device->hal_info.cluster_count *
                 device->hal_info.cores_per_cluster;
    device->calls = calloc(core_count, sizeof(*device->calls));
    device->launch_agents = calloc(core_count,
                                   sizeof(*device->launch_agents));
    device->clusters = calloc(device->hal_info.cluster_count,
                              sizeof(*device->clusters));
    if (!device->calls || !device->launch_agents || !device->clusters) {
        status = -ENOMEM;
        goto fail;
    }
    device->launch_agent_count = core_count;
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
    for (index = 0; index < core_count; ++index) {
        struct npu_service_launch_agent *agent =
            &device->launch_agents[index];

        agent->device = device;
        status = pthread_create(&agent->thread, 0, npu_launch_agent, agent);
        if (status) {
            status = -status;
            goto fail;
        }
        agent->started = 1;
    }
    *device_out = device;
    return 0;

fail:
    if (mutex_initialized && condition_initialized) {
        pthread_mutex_lock(&device->mutex);
        device->worker_stop = 1;
        pthread_cond_broadcast(&device->condition);
        pthread_mutex_unlock(&device->mutex);
    }
    if (device->launch_agents) {
        for (index = 0; index < device->launch_agent_count; ++index) {
            if (device->launch_agents[index].started)
                (void)pthread_join(device->launch_agents[index].thread, 0);
        }
    }
    if (device->worker_started)
        (void)pthread_join(device->worker, 0);
    if (condition_initialized)
        pthread_cond_destroy(&device->condition);
    if (kernel_mutex_initialized)
        pthread_mutex_destroy(&device->kernel_mutex);
    if (mutex_initialized)
        pthread_mutex_destroy(&device->mutex);
    free(device->clusters);
    free(device->launch_agents);
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

static void free_atomic_cache_backing(npu_service_device_t *device)
{
    uint32_t cluster;

    for (cluster = 0; cluster < device->hal_info.cluster_count; ++cluster) {
        struct npu_service_cluster *state = &device->clusters[cluster];

        if (state->atomic_cache_backing)
            (void)npu_hal_memory_free(device->hal,
                                      state->atomic_cache_backing);
        state->atomic_cache_backing = 0;
        state->atomic_cache_backing_address = 0;
    }
}

static void join_launch_agents(npu_service_device_t *device)
{
    size_t index;

    for (index = 0; index < device->launch_agent_count; ++index) {
        struct npu_service_launch_agent *agent =
            &device->launch_agents[index];

        if (!agent->started)
            continue;
        (void)pthread_join(agent->thread, 0);
        agent->started = 0;
    }
}

int npu_service_status(npu_service_device_t *device)
{
    int status;

    if (!device)
        return -EINVAL;
    pthread_mutex_lock(&device->mutex);
    status = device->abort_status;
    pthread_mutex_unlock(&device->mutex);
    return status;
}

void npu_service_abort(npu_service_device_t *device, int status)
{
    size_t core_count;
    size_t index;

    if (!device)
        return;
    if (status >= 0)
        status = -ESHUTDOWN;
    core_count = (size_t)device->hal_info.cluster_count *
                 device->hal_info.cores_per_cluster;
    pthread_mutex_lock(&device->mutex);
    if (!device->abort_status)
        device->abort_status = status;
    status = device->abort_status;
    device->worker_stop = 1;
    for (index = 0; index < core_count; ++index) {
        struct npu_pending_call *call = &device->calls[index];

        if (!call->pending)
            continue;
        call->pending = 0;
        call->status = status;
    }
    pthread_cond_broadcast(&device->condition);
    pthread_mutex_unlock(&device->mutex);
}

int npu_service_close(npu_service_device_t *device)
{
    int profile_status = 0;
    int aborted;
    int status;

    if (!device)
        return -EINVAL;
    aborted = npu_service_status(device) < 0;
    if (aborted) {
        npu_service_abort(device, -ESHUTDOWN);
        join_launch_agents(device);
        if (device->worker_started) {
            (void)pthread_join(device->worker, 0);
            device->worker_started = 0;
        }
        npu_service_profile_discard(device);
    }
    pthread_mutex_lock(&device->mutex);
    status = !!device->profile_allocation;
    pthread_mutex_unlock(&device->mutex);
    if (status) {
        profile_status = npu_service_profile_stop(device);
        pthread_mutex_lock(&device->mutex);
        status = !!device->profile_allocation;
        pthread_mutex_unlock(&device->mutex);
        if (profile_status < 0 && status)
            return profile_status;
    }
    pthread_mutex_lock(&device->mutex);
    if (device->active_launches) {
        pthread_mutex_unlock(&device->mutex);
        return -EBUSY;
    }
    device->worker_stop = 1;
    pthread_cond_broadcast(&device->condition);
    pthread_mutex_unlock(&device->mutex);
    join_launch_agents(device);
    if (device->worker_started)
        (void)pthread_join(device->worker, 0);
    npu_service_kernel_state_destroy(device);
    free_heap_allocations(device);
    free_atomic_cache_backing(device);
    free_firmware_allocations(device);
    pthread_cond_destroy(&device->condition);
    pthread_mutex_destroy(&device->kernel_mutex);
    pthread_mutex_destroy(&device->mutex);
    free(device->clusters);
    free(device->launch_agents);
    free(device->calls);
    free(device);
    return profile_status;
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

int npu_service_alloc_rpc_descriptor(
    npu_service_device_t *device, npu_hal_core_t viewer,
    size_t descriptor_size, npu_hal_allocation_t **allocation_out)
{
    const uint64_t granule_mask = RPC_DESCRIPTOR_GRANULE - 1U;
    uint64_t allocation_size;

    if (!device || !allocation_out || !descriptor_size ||
        !npu_hal_core_valid(device->hal, viewer) ||
        descriptor_size > UINT64_MAX - granule_mask)
        return -EINVAL;
    allocation_size = ((uint64_t)descriptor_size + granule_mask) &
                      ~granule_mask;
    return npu_service_alloc_visible(
        device, viewer, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_LAUNCH,
        (uint32_t)npu_service_core_index(device, viewer), allocation_size,
        RPC_DESCRIPTOR_GRANULE, allocation_out);
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

static int tensor_context_valid(uint64_t value)
{
    struct dayan_context_fields fields = dayan_context_unpack(value);

    return !(value >> 56) && dayan_context_fields_valid(&fields);
}

static int proc_name_valid(const char *name)
{
    return name && name[0] &&
           strnlen(name, NPU_SERVICE_PROC_NAME_SIZE) <
               NPU_SERVICE_PROC_NAME_SIZE;
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

static int wait_command_consumed(npu_service_device_t *device,
                                 npu_hal_core_t core, uint64_t deadline)
{
    for (;;) {
        uint64_t now;
        uint32_t slice = NPU_RPC_CANCEL_POLL_MS;
        int status = npu_service_status(device);

        if (status < 0)
            return status;
        now = npu_now_ms();
        if (deadline != UINT64_MAX) {
            uint64_t remaining;

            if (now >= deadline)
                return -ETIMEDOUT;
            remaining = deadline - now;
            if (remaining < slice)
                slice = (uint32_t)remaining;
        }
        status = npu_hal_wait_command_consumed(device->hal, core, slice);
        if (status != -ETIMEDOUT)
            return status;
    }
}

static int rpc_shutdown_status_locked(const npu_service_device_t *device)
{
    if (device->abort_status)
        return device->abort_status;
    return device->worker_stop ? -ESHUTDOWN : 0;
}

static int rpc_condition_wait_locked(npu_service_device_t *device,
                                     uint32_t timeout_ms,
                                     const struct timespec *deadline)
{
    int status;

    if (timeout_ms == NPU_SERVICE_TIMEOUT_INFINITE)
        status = pthread_cond_wait(&device->condition, &device->mutex);
    else
        status = pthread_cond_timedwait(&device->condition, &device->mutex,
                                        deadline);
    if (!status)
        return 0;
    return status == ETIMEDOUT ? -ETIMEDOUT : -status;
}

static int rpc_queue_first_for_core_locked(
    const npu_service_device_t *device,
    const struct npu_rpc_queue_entry *entry)
{
    const struct npu_rpc_queue_entry *item;

    for (item = device->rpc_queue_head; item && item != entry;
         item = item->next) {
        if (item->core_index == entry->core_index)
            return 0;
    }
    return item == entry;
}

static void rpc_queue_push_locked(npu_service_device_t *device,
                                  struct npu_rpc_queue_entry *entry)
{
    entry->next = 0;
    entry->queued = 1;
    if (device->rpc_queue_tail)
        device->rpc_queue_tail->next = entry;
    else
        device->rpc_queue_head = entry;
    device->rpc_queue_tail = entry;
    ++device->rpc_queue_count;
}

static void rpc_queue_remove_locked(npu_service_device_t *device,
                                    struct npu_rpc_queue_entry *entry)
{
    struct npu_rpc_queue_entry *previous = 0;
    struct npu_rpc_queue_entry *item;

    if (!entry->queued)
        return;
    for (item = device->rpc_queue_head; item && item != entry;
         item = item->next)
        previous = item;
    if (!item)
        return;
    if (previous)
        previous->next = item->next;
    else
        device->rpc_queue_head = item->next;
    if (device->rpc_queue_tail == item)
        device->rpc_queue_tail = previous;
    item->next = 0;
    item->queued = 0;
    --device->rpc_queue_count;
}

int npu_service_rpc_call(npu_service_device_t *device, npu_hal_core_t core,
                         uint16_t opcode, uint64_t argument,
                         uint32_t timeout_ms,
                         int *argument_may_be_live_out)
{
    struct npu_rpc_queue_entry entry;
    struct npu_pending_call *call;
    struct timespec deadline;
    uint64_t monotonic_deadline;
    size_t core_index;
    uint16_t expected;
    int queue_space_waiter = 0;
    int status;

    if (!device || !npu_hal_core_valid(device->hal, core) ||
        argument > RPC_COMMAND_ARG_MASK)
        return -EINVAL;
    if (argument_may_be_live_out)
        *argument_may_be_live_out = 0;
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0)
        return -EIO;
    monotonic_deadline = UINT64_MAX;
    if (timeout_ms != NPU_SERVICE_TIMEOUT_INFINITE) {
        monotonic_deadline = npu_now_ms() + timeout_ms;
        deadline.tv_sec += timeout_ms / 1000U;
        deadline.tv_nsec += (long)(timeout_ms % 1000U) * 1000000L;
        if (deadline.tv_nsec >= 1000000000L) {
            ++deadline.tv_sec;
            deadline.tv_nsec -= 1000000000L;
        }
    }
    expected = (uint16_t)(opcode | UINT16_C(0x8000));
    core_index = npu_service_core_index(device, core);
    call = &device->calls[core_index];
    memset(&entry, 0, sizeof(entry));
    entry.core_index = core_index;
    pthread_mutex_lock(&device->mutex);
    if (device->rpc_queue_count >= device->rpc_queue_capacity) {
        ++device->rpc_queue_waiters;
        queue_space_waiter = 1;
    }
    for (;;) {
        status = rpc_shutdown_status_locked(device);
        if (status < 0)
            goto space_wait_done;
        if (device->rpc_queue_count < device->rpc_queue_capacity)
            break;
        status = rpc_condition_wait_locked(device, timeout_ms, &deadline);
        if (status < 0)
            goto space_wait_done;
    }
    if (queue_space_waiter)
        --device->rpc_queue_waiters;
    rpc_queue_push_locked(device, &entry);
    for (;;) {
        status = rpc_shutdown_status_locked(device);
        if (status < 0)
            goto queue_fail;
        if (!call->active &&
            rpc_queue_first_for_core_locked(device, &entry))
            break;
        status = rpc_condition_wait_locked(device, timeout_ms, &deadline);
        if (status < 0)
            goto queue_fail;
    }
    rpc_queue_remove_locked(device, &entry);
    call->expected_opcode = expected;
    call->expected_argument = argument;
    call->pending = 1;
    call->active = 1;
    call->status = -EINPROGRESS;
    pthread_cond_broadcast(&device->condition);
    pthread_mutex_unlock(&device->mutex);

    status = wait_command_consumed(device, core, monotonic_deadline);
    if (status == 0)
        status = npu_service_status(device);
    if (status == 0) {
        if (argument_may_be_live_out)
            *argument_may_be_live_out = 1;
        status = npu_hal_command_submit(device->hal, core,
                                        rpc_command_pack(opcode, argument));
    }
    if (status < 0) {
        pthread_mutex_lock(&device->mutex);
        call->pending = 0;
        call->active = 0;
        call->status = status;
        pthread_cond_broadcast(&device->condition);
        pthread_mutex_unlock(&device->mutex);
        return status;
    }
    pthread_mutex_lock(&device->mutex);
    while (call->pending) {
        status = rpc_shutdown_status_locked(device);
        if (!status)
            status = rpc_condition_wait_locked(device, timeout_ms,
                                               &deadline);
        if (status < 0) {
            call->pending = 0;
            call->status = status;
            pthread_cond_broadcast(&device->condition);
            break;
        }
    }
    status = call->status;
    call->active = 0;
    pthread_cond_broadcast(&device->condition);
    pthread_mutex_unlock(&device->mutex);
    if (status == 0 && argument_may_be_live_out)
        *argument_may_be_live_out = 0;
    return status;

space_wait_done:
    if (queue_space_waiter)
        --device->rpc_queue_waiters;
queue_fail:
    rpc_queue_remove_locked(device, &entry);
    pthread_cond_broadcast(&device->condition);
    pthread_mutex_unlock(&device->mutex);
    return status;
}

static int launch_request_valid(const npu_service_launch_t *launch,
                                const struct rpc_proc_create_request *request)
{
    return request->version == RPC_PROC_CREATE_VERSION &&
           request->size == sizeof(*request) &&
           request->hart_id == launch->target.core &&
           request->cookie == launch->cookie &&
           request->task_mode == launch->task_mode &&
           !memcmp(request->task_id, launch->task_id,
                   sizeof(request->task_id)) &&
           proc_name_valid(request->name) &&
           !memcmp(request->name, launch->name, sizeof(request->name)) &&
           (request->state == RPC_PROC_RUNNING ||
            request->state == RPC_PROC_EXITED ||
            request->state == RPC_PROC_FAILED);
}

static void launch_submission_complete(npu_service_launch_t *launch,
                                       int status, uint32_t accepted,
                                       uint32_t descriptor_may_be_live)
{
    npu_service_device_t *device = launch->device;
    uint32_t abandoned;

    pthread_mutex_lock(&device->mutex);
    launch->submit_status = status;
    launch->accepted = accepted;
    launch->descriptor_may_be_live = descriptor_may_be_live;
    launch->submit_done = 1;
    abandoned = launch->abandoned;
    pthread_cond_broadcast(&device->condition);
    pthread_mutex_unlock(&device->mutex);
    if (abandoned)
        free(launch);
}

static void submit_launch(npu_service_launch_t *launch)
{
    struct rpc_proc_create_request request;
    int descriptor_may_be_live = 0;
    int status = npu_service_rpc_call(
        launch->device, launch->target, RPC_PROC_CREATE,
        launch->request_address, launch->device->options.default_timeout_ms,
        &descriptor_may_be_live);

    if (status < 0) {
        launch_submission_complete(launch, status, 0,
                                   (uint32_t)descriptor_may_be_live);
        return;
    }
    status = npu_hal_memory_read(launch->device->hal,
                                 launch->request_allocation, launch->target,
                                 &request, 0, sizeof(request));
    if (status == 0 && !launch_request_valid(launch, &request)) {
        fprintf(stderr,
                "dayan_server: PROC_CREATE request mismatch: hart=%u "
                "address=0x%llx version=%u size=%u request_hart=%u "
                "cookie=%llu expected_cookie=%llu state=%u name='%.*s' "
                "expected_name='%s'\n",
                launch->target.core,
                (unsigned long long)launch->request_address,
                request.version, request.size, request.hart_id,
                (unsigned long long)request.cookie,
                (unsigned long long)launch->cookie, request.state,
                (int)sizeof(request.name), request.name, launch->name);
        status = -EPROTO;
    }
    launch_submission_complete(launch, status, status == 0, 1);
}

static void *npu_launch_agent(void *arg)
{
    struct npu_service_launch_agent *agent = arg;
    npu_service_device_t *device = agent->device;

    for (;;) {
        npu_service_launch_t *launch;
        int status;

        pthread_mutex_lock(&device->mutex);
        while (!agent->head && !device->worker_stop)
            (void)pthread_cond_wait(&device->condition, &device->mutex);
        if (!agent->head) {
            pthread_mutex_unlock(&device->mutex);
            return 0;
        }
        launch = agent->head;
        agent->head = launch->queue_next;
        if (!agent->head)
            agent->tail = 0;
        launch->queue_next = 0;
        launch->submit_started = 1;
        status = rpc_shutdown_status_locked(device);
        pthread_mutex_unlock(&device->mutex);
        if (status < 0)
            launch_submission_complete(launch, status, 0, 0);
        else
            submit_launch(launch);
    }
}

int npu_service_launch(npu_service_device_t *device,
                       const npu_service_launch_desc_t *desc,
                       npu_service_launch_t **launch_out)
{
    struct rpc_proc_create_request request;
    struct npu_service_launch_agent *agent;
    npu_service_launch_t *launch;
    npu_hal_allocation_t *allocation = 0;
    uint64_t shared_base;
    uint64_t uram_base;
    uint64_t request_address;
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
        desc->max_child_threads > RPC_PROC_MAX_CHILD_THREADS ||
        desc->resource_reserved || !tensor_context_valid(desc->tensor_context) ||
        !proc_name_valid(desc->name) ||
        !task_valid(desc) ||
        !npu_hal_address_valid(device->hal, desc->target, desc->entry, 2))
        return -EINVAL;
    *launch_out = 0;
    pthread_mutex_lock(&device->mutex);
    booted = device->clusters[desc->target.cluster].booted_cores;
    status = rpc_shutdown_status_locked(device);
    pthread_mutex_unlock(&device->mutex);
    if (status < 0)
        return status;
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
    status = npu_service_alloc_rpc_descriptor(
        device, desc->target, sizeof(request), &allocation);
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
    request.max_child_threads = desc->max_child_threads;
    request.uram_base = uram_base;
    request.uram_size = desc->uram_size;
    request.tensor_context = desc->tensor_context;
    request.printf_buffer_size = desc->printf_buffer_size;
    request.argc = desc->argc;
    memcpy(request.args, desc->args, desc->argc * sizeof(desc->args[0]));
    request.cookie = desc->cookie;
    for (axis = 0; axis < 3; ++axis) {
        request.task_id[axis] = desc->task_id[axis];
        request.task_dim[axis] = desc->task_dim[axis];
    }
    request.task_mode = desc->task_mode;
    memcpy(request.name, desc->name, strlen(desc->name) + 1U);
    status = npu_hal_memory_write(device->hal, allocation, desc->target, 0,
                                  &request, sizeof(request));
    if (status < 0)
        goto fail;
    launch->device = device;
    launch->target = desc->target;
    launch->request_allocation = allocation;
    launch->request_address = request_address;
    launch->cookie = desc->cookie;
    memcpy(launch->task_id, desc->task_id, sizeof(launch->task_id));
    launch->task_mode = desc->task_mode;
    launch->submit_status = -EINPROGRESS;
    memcpy(launch->name, request.name, sizeof(launch->name));
    agent = &device->launch_agents[npu_service_core_index(device,
                                                           desc->target)];
    pthread_mutex_lock(&device->mutex);
    status = rpc_shutdown_status_locked(device);
    if (status == 0) {
        if (agent->tail)
            agent->tail->queue_next = launch;
        else
            agent->head = launch;
        agent->tail = launch;
        ++device->active_launches;
        pthread_cond_broadcast(&device->condition);
    }
    pthread_mutex_unlock(&device->mutex);
    if (status < 0)
        goto fail;
    *launch_out = launch;
    return 0;

fail:
    if (allocation)
        (void)npu_hal_memory_free(device->hal, allocation);
    free(launch);
    return status;
}

static void launch_info_init(const npu_service_launch_t *launch,
                             npu_service_launch_info_t *info,
                             uint32_t state, int64_t status)
{
    memset(info, 0, sizeof(*info));
    info->state = state;
    info->target = launch->target;
    info->status = status;
    memcpy(info->task_id, launch->task_id, sizeof(info->task_id));
    info->task_mode = launch->task_mode;
    info->cookie = launch->cookie;
    memcpy(info->name, launch->name, sizeof(info->name));
}

static int launch_proc_present(npu_service_launch_t *launch,
                               uint32_t proc_id, int *present_out)
{
    npu_service_proc_snapshot_t snapshot;
    uint32_t index;
    int status;

    *present_out = 0;
    status = npu_service_rpc_ps(launch->device, launch->target, &snapshot);
    if (status < 0)
        return status;
    for (index = 0; index < snapshot.count; ++index) {
        const npu_service_proc_info_t *proc = &snapshot.records[index];

        if (proc->proc_id == proc_id &&
            proc->host_proc_id == proc_id &&
            proc->cookie == launch->cookie) {
            *present_out = 1;
            break;
        }
    }
    return 0;
}

int npu_service_launch_query(npu_service_launch_t *launch,
                             npu_service_launch_info_t *info)
{
    struct rpc_proc_create_request request;
    int present;
    int status;

    if (!launch || !info)
        return -EINVAL;
    pthread_mutex_lock(&launch->device->mutex);
    status = launch->device->abort_status;
    if (status < 0) {
        pthread_mutex_unlock(&launch->device->mutex);
        return status;
    }
    if (!launch->submit_done) {
        launch_info_init(launch, info, NPU_SERVICE_LAUNCH_SUBMITTED, 0);
        pthread_mutex_unlock(&launch->device->mutex);
        return 0;
    }
    if (!launch->accepted) {
        launch_info_init(launch, info, NPU_SERVICE_LAUNCH_FAILED,
                         launch->submit_status);
        pthread_mutex_unlock(&launch->device->mutex);
        return 0;
    }
    pthread_mutex_unlock(&launch->device->mutex);
    status = npu_hal_memory_read(launch->device->hal,
                                 launch->request_allocation, launch->target,
                                 &request, 0, sizeof(request));
    if (status < 0)
        return status;
    if (!launch_request_valid(launch, &request))
        return -EPROTO;
    if ((request.state == RPC_PROC_EXITED ||
         request.state == RPC_PROC_FAILED) && request.proc_id) {
        status = launch_proc_present(launch, request.proc_id, &present);
        if (status < 0)
            return status;
        if (present) {
            launch_info_init(launch, info, NPU_SERVICE_LAUNCH_RUNNING, 0);
            info->proc_id = request.proc_id;
            return 0;
        }
    }
    launch_info_init(launch, info, request.state, request.status);
    info->proc_id = request.proc_id;
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
    int reclaim_descriptor;
    int status;

    if (!launch)
        return -EINVAL;
    status = npu_service_launch_query(launch, &info);
    if (status < 0)
        return status;
    if (info.state != NPU_SERVICE_LAUNCH_EXITED &&
        info.state != NPU_SERVICE_LAUNCH_FAILED)
        return -EBUSY;
    device = launch->device;
    pthread_mutex_lock(&device->mutex);
    reclaim_descriptor = launch->accepted ||
                         !launch->descriptor_may_be_live;
    pthread_mutex_unlock(&device->mutex);
    if (reclaim_descriptor) {
        status = npu_hal_memory_free(device->hal,
                                     launch->request_allocation);
        if (status < 0)
            return status;
    }
    pthread_mutex_lock(&device->mutex);
    if (device->active_launches)
        --device->active_launches;
    pthread_mutex_unlock(&device->mutex);
    free(launch);
    return 0;
}

int npu_service_launch_abandon(npu_service_launch_t *launch)
{
    struct npu_service_launch_agent *agent;
    npu_service_launch_t *item;
    npu_service_launch_t *previous = 0;
    npu_service_device_t *device;
    int free_launch = 0;

    if (!launch)
        return -EINVAL;
    device = launch->device;
    /* Device completion is unknown; keep the descriptor reserved until HAL close. */
    pthread_mutex_lock(&device->mutex);
    agent = &device->launch_agents[npu_service_core_index(device,
                                                           launch->target)];
    if (!launch->submit_started) {
        for (item = agent->head; item && item != launch;
             item = item->queue_next)
            previous = item;
        if (item) {
            if (previous)
                previous->queue_next = item->queue_next;
            else
                agent->head = item->queue_next;
            if (agent->tail == item)
                agent->tail = previous;
            item->queue_next = 0;
            free_launch = 1;
        }
    }
    if (!free_launch) {
        if (launch->submit_done)
            free_launch = 1;
        else
            launch->abandoned = 1;
    }
    if (device->active_launches)
        --device->active_launches;
    pthread_cond_broadcast(&device->condition);
    pthread_mutex_unlock(&device->mutex);
    if (free_launch)
        free(launch);
    return 0;
}

int npu_service_rpc_ps(npu_service_device_t *device, npu_hal_core_t target,
                       npu_service_proc_snapshot_t *snapshot_out)
{
    struct rpc_proc_ps_snapshot snapshot;
    npu_hal_allocation_t *allocation = 0;
    uint64_t address;
    uint32_t booted;
    uint32_t record;
    int descriptor_may_be_live = 0;
    int status;

    if (!device || !snapshot_out ||
        !npu_hal_core_valid(device->hal, target))
        return -EINVAL;
    pthread_mutex_lock(&device->mutex);
    booted = device->clusters[target.cluster].booted_cores;
    pthread_mutex_unlock(&device->mutex);
    if (!(booted & (UINT32_C(1) << target.core)))
        return -ENODEV;
    status = npu_service_alloc_rpc_descriptor(
        device, target, sizeof(snapshot), &allocation);
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
                                  device->options.default_timeout_ms,
                                  &descriptor_may_be_live);
    if (status < 0)
        goto out;
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
    for (record = 0; record < snapshot.count; ++record) {
        if (!proc_name_valid(snapshot.records[record].name)) {
            status = -EPROTO;
            goto out;
        }
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
    if (allocation && !descriptor_may_be_live) {
        int free_status = npu_hal_memory_free(device->hal, allocation);

        if (status == 0 && free_status < 0)
            status = free_status;
    }
    return status;
}

int npu_service_rpc_kill_all_timeout(npu_service_device_t *device,
                                     npu_hal_core_t target,
                                     uint32_t timeout_ms,
                                     npu_service_kill_result_t *result_out)
{
    struct rpc_proc_kill_result result;
    npu_hal_allocation_t *allocation = 0;
    uint64_t address;
    uint32_t booted;
    int descriptor_may_be_live = 0;
    int status;

    if (!device || !result_out || !timeout_ms ||
        !npu_hal_core_valid(device->hal, target))
        return -EINVAL;
    pthread_mutex_lock(&device->mutex);
    booted = device->clusters[target.cluster].booted_cores;
    pthread_mutex_unlock(&device->mutex);
    if (!(booted & (UINT32_C(1) << target.core)))
        return -ENODEV;
    status = npu_service_alloc_rpc_descriptor(
        device, target, sizeof(result), &allocation);
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
                                  address, timeout_ms,
                                  &descriptor_may_be_live);
    if (status < 0)
        goto out;
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
    if (allocation && !descriptor_may_be_live) {
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
    if (!device)
        return -EINVAL;
    return npu_service_rpc_kill_all_timeout(
        device, target, device->options.default_timeout_ms, result_out);
}
