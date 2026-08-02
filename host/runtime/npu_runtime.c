#include "runtime_internal.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "trap.h"

#define NPU_DEFAULT_RAM_BASE          UINT64_C(0x80000000)
#define NPU_DEFAULT_RAM_SIZE          UINT64_C(0x04000000)
#define NPU_DEFAULT_COHERENT_BASE     UINT64_C(0x00400000)
#define NPU_DEFAULT_COHERENT_SIZE     UINT64_C(0x00400000)
#define NPU_DEFAULT_RPC_POLL_US       1000U
#define NPU_DEFAULT_TIMEOUT_MS        5000U
#define NPU_DEFAULT_STACK_SIZE        UINT64_C(0x10000)
#define NPU_DEFAULT_PRINTF_SIZE       UINT64_C(0x1000)
#define NPU_COPY_CHUNK                (64U * 1024U)

_Static_assert((int)NPU_LAUNCH_SUBMITTED == (int)RPC_PROC_SUBMITTED,
               "Host and Device launch state ABI");
_Static_assert((int)NPU_LAUNCH_RUNNING == (int)RPC_PROC_RUNNING,
               "Host and Device launch state ABI");
_Static_assert((int)NPU_LAUNCH_EXITED == (int)RPC_PROC_EXITED,
               "Host and Device launch state ABI");
_Static_assert((int)NPU_LAUNCH_FAILED == (int)RPC_PROC_FAILED,
               "Host and Device launch state ABI");

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

static void default_output(void *context, uint32_t hart, uint32_t proc,
                           const void *data, size_t size, uint32_t flags)
{
    (void)context;
    (void)hart;
    (void)proc;
    if (flags & NPU_OUTPUT_CLEAR)
        fflush(stdout);
    if (data && size) {
        (void)fwrite(data, 1, size, stdout);
        fflush(stdout);
    }
}

static void default_trap(void *context, const struct trap_record *record)
{
    (void)context;
    fprintf(stderr,
            "NPU trap: hart=%u proc=%u mcause=0x%llx mepc=0x%llx "
            "mtval=0x%llx\n",
            record->hart_id, record->thread_id,
            (unsigned long long)record->mcause,
            (unsigned long long)record->mepc,
            (unsigned long long)record->mtval);
}

void npu_runtime_options_init(npu_runtime_options_t *options)
{
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->version = NPU_RUNTIME_OPTIONS_VERSION;
    options->hart_count = 4;
    options->ram_base = NPU_DEFAULT_RAM_BASE;
    options->ram_size = NPU_DEFAULT_RAM_SIZE;
    options->coherent_base = NPU_DEFAULT_COHERENT_BASE;
    options->coherent_size = NPU_DEFAULT_COHERENT_SIZE;
    options->rpc_poll_interval_us = NPU_DEFAULT_RPC_POLL_US;
    options->default_timeout_ms = NPU_DEFAULT_TIMEOUT_MS;
    npu_driver_options_init(&options->driver);
}

void npu_launch_desc_init(npu_launch_desc_t *desc)
{
    if (!desc)
        return;
    memset(desc, 0, sizeof(*desc));
    desc->version = NPU_LAUNCH_DESC_VERSION;
    desc->stack_size = NPU_DEFAULT_STACK_SIZE;
    desc->printf_buffer_size = NPU_DEFAULT_PRINTF_SIZE;
}

static int runtime_options_valid(const npu_runtime_options_t *options)
{
    return options && options->version == NPU_RUNTIME_OPTIONS_VERSION &&
           options->hart_count &&
           options->hart_count <= NPU_DRIVER_MAX_HARTS &&
           options->ram_size && options->coherent_size &&
           options->ram_base <= UINT64_MAX - options->ram_size &&
           options->coherent_base <= UINT64_MAX - options->coherent_size &&
           options->ram_base + options->ram_size <=
               RPC_COMMAND_ARG_MASK + 1U &&
           options->coherent_base + options->coherent_size <=
               RPC_COMMAND_ARG_MASK + 1U &&
           options->rpc_poll_interval_us && options->default_timeout_ms;
}

int npu_runtime_open(npu_runtime_t **runtime_out,
                     const npu_runtime_options_t *options)
{
    npu_runtime_options_t defaults;
    pthread_condattr_t condition_attr;
    npu_runtime_t *runtime;
    int mutex_initialized = 0;
    int kernel_mutex_initialized = 0;
    int condition_initialized = 0;
    int status;

    if (!runtime_out)
        return -EINVAL;
    *runtime_out = 0;
    if (!options) {
        npu_runtime_options_init(&defaults);
        options = &defaults;
    }
    if (!runtime_options_valid(options))
        return -EINVAL;
    runtime = calloc(1, sizeof(*runtime));
    if (!runtime)
        return -ENOMEM;
    runtime->device.h2c_fd = -1;
    runtime->device.c2h_fd = -1;
    runtime->device.user_fd = -1;
    runtime->device.event_fd = -1;
    runtime->options = *options;
    if (!runtime->options.output)
        runtime->options.output = default_output;
    if (!runtime->options.trap)
        runtime->options.trap = default_trap;
    npu_arena_init(&runtime->normal, options->ram_base, options->ram_size);
    npu_arena_init(&runtime->coherent, options->coherent_base,
                   options->coherent_size);

    status = pthread_mutex_init(&runtime->mutex, 0);
    if (status != 0) {
        status = -status;
        goto fail;
    }
    mutex_initialized = 1;
    status = pthread_mutex_init(&runtime->kernel_mutex, 0);
    if (status != 0) {
        status = -status;
        goto fail;
    }
    kernel_mutex_initialized = 1;
    status = pthread_condattr_init(&condition_attr);
    if (status != 0) {
        status = -status;
        goto fail;
    }
    (void)pthread_condattr_setclock(&condition_attr, CLOCK_MONOTONIC);
    status = pthread_cond_init(&runtime->condition, &condition_attr);
    pthread_condattr_destroy(&condition_attr);
    if (status != 0) {
        status = -status;
        goto fail;
    }
    condition_initialized = 1;
    status = npu_device_open(&runtime->device, &options->driver);
    if (status < 0)
        goto fail;
    status = pthread_create(&runtime->worker, 0, npu_rpc_worker, runtime);
    if (status != 0) {
        status = -status;
        goto fail;
    }
    runtime->worker_started = 1;
    *runtime_out = runtime;
    return 0;

fail:
    npu_device_close(&runtime->device);
    if (condition_initialized)
        pthread_cond_destroy(&runtime->condition);
    if (kernel_mutex_initialized)
        pthread_mutex_destroy(&runtime->kernel_mutex);
    if (mutex_initialized)
        pthread_mutex_destroy(&runtime->mutex);
    npu_arena_destroy(&runtime->coherent);
    npu_arena_destroy(&runtime->normal);
    free(runtime);
    return status;
}

int npu_runtime_close(npu_runtime_t *runtime)
{
    if (!runtime)
        return -EINVAL;
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->active_launches) {
        pthread_mutex_unlock(&runtime->mutex);
        return -EBUSY;
    }
    runtime->worker_stop = 1;
    pthread_cond_broadcast(&runtime->condition);
    pthread_mutex_unlock(&runtime->mutex);
    if (runtime->worker_started)
        pthread_join(runtime->worker, 0);
    npu_device_close(&runtime->device);
    npu_kernel_state_destroy(runtime);
    npu_arena_destroy(&runtime->coherent);
    npu_arena_destroy(&runtime->normal);
    pthread_cond_destroy(&runtime->condition);
    pthread_mutex_destroy(&runtime->kernel_mutex);
    pthread_mutex_destroy(&runtime->mutex);
    free(runtime);
    return 0;
}

int npu_runtime_load_bl1(npu_runtime_t *runtime,
                         const void *elf_image, size_t elf_size,
                         uint32_t hart_mask)
{
    if (!runtime || !elf_image || !elf_size)
        return -EINVAL;
    return npu_runtime_load_bl1_elf(runtime, elf_image, elf_size, hart_mask);
}

int npu_runtime_load_bl1_file(npu_runtime_t *runtime,
                              const char *path, uint32_t hart_mask)
{
    unsigned char *image = 0;
    long length;
    FILE *file;
    int status;

    if (!runtime || !path)
        return -EINVAL;
    file = fopen(path, "rb");
    if (!file)
        return -errno;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        status = -EIO;
        goto out;
    }
    if (!length || (uint64_t)length > SIZE_MAX) {
        status = -EFBIG;
        goto out;
    }
    image = malloc((size_t)length);
    if (!image) {
        status = -ENOMEM;
        goto out;
    }
    if (fread(image, 1, (size_t)length, file) != (size_t)length) {
        status = -EIO;
        goto out;
    }
    status = npu_runtime_load_bl1(runtime, image, (size_t)length, hart_mask);
out:
    free(image);
    fclose(file);
    return status;
}

static int runtime_arena_alloc(npu_runtime_t *runtime,
                               struct npu_arena *arena,
                               size_t size, size_t alignment,
                               uint32_t kind, uint64_t *address_out)
{
    int status;

    if (!runtime || !address_out || !size || !npu_is_power_of_two(alignment))
        return -EINVAL;
    pthread_mutex_lock(&runtime->mutex);
    status = npu_arena_alloc(arena, size, alignment, kind, UINT32_MAX,
                             address_out);
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

int npu_runtime_alloc(npu_runtime_t *runtime, size_t size,
                      size_t alignment, uint64_t *address_out)
{
    return runtime_arena_alloc(runtime, runtime ? &runtime->normal : 0,
                               size, alignment, NPU_ALLOC_USER, address_out);
}

int npu_runtime_free(npu_runtime_t *runtime, uint64_t address)
{
    int status;

    if (!runtime)
        return -EINVAL;
    pthread_mutex_lock(&runtime->mutex);
    status = npu_arena_free(&runtime->normal, address, NPU_ALLOC_USER,
                            UINT32_MAX);
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

int npu_runtime_coherent_alloc(npu_runtime_t *runtime, size_t size,
                               size_t alignment, uint64_t *address_out)
{
    return runtime_arena_alloc(runtime, runtime ? &runtime->coherent : 0,
                               size, alignment, NPU_ALLOC_COHERENT,
                               address_out);
}

int npu_runtime_coherent_free(npu_runtime_t *runtime, uint64_t address)
{
    int status;

    if (!runtime)
        return -EINVAL;
    pthread_mutex_lock(&runtime->mutex);
    status = npu_arena_free(&runtime->coherent, address,
                            NPU_ALLOC_COHERENT, UINT32_MAX);
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

int npu_runtime_copy_to_device(npu_runtime_t *runtime, uint64_t destination,
                               const void *source, size_t size)
{
    if (!runtime || (!source && size))
        return -EINVAL;
    return size ? npu_device_write(&runtime->device, destination,
                                   source, size) : 0;
}

int npu_runtime_copy_from_device(npu_runtime_t *runtime, void *destination,
                                 uint64_t source, size_t size)
{
    if (!runtime || (!destination && size))
        return -EINVAL;
    return size ? npu_device_read(&runtime->device, source,
                                  destination, size) : 0;
}

int npu_runtime_copy_device(npu_runtime_t *runtime, uint64_t destination,
                            uint64_t source, size_t size)
{
    unsigned char *buffer;
    size_t done = 0;
    int status = 0;

    if (!runtime)
        return -EINVAL;
    if (!size || destination == source)
        return 0;
    buffer = malloc(size < NPU_COPY_CHUNK ? size : NPU_COPY_CHUNK);
    if (!buffer)
        return -ENOMEM;
    while (done < size) {
        size_t chunk = size - done;

        if (chunk > NPU_COPY_CHUNK)
            chunk = NPU_COPY_CHUNK;
        status = npu_device_read(&runtime->device, source + done,
                                 buffer, chunk);
        if (status < 0)
            break;
        status = npu_device_write(&runtime->device, destination + done,
                                  buffer, chunk);
        if (status < 0)
            break;
        done += chunk;
    }
    free(buffer);
    return status;
}

static void call_cancel(npu_runtime_t *runtime, uint32_t hart, int status)
{
    pthread_mutex_lock(&runtime->mutex);
    runtime->calls[hart].pending = 0;
    runtime->calls[hart].status = status;
    pthread_cond_broadcast(&runtime->condition);
    pthread_mutex_unlock(&runtime->mutex);
}

static int condition_wait(npu_runtime_t *runtime, uint32_t hart,
                          uint32_t timeout_ms)
{
    int status = 0;

    pthread_mutex_lock(&runtime->mutex);
    if (timeout_ms == NPU_TIMEOUT_INFINITE) {
        while (runtime->calls[hart].pending && !runtime->worker_stop)
            pthread_cond_wait(&runtime->condition, &runtime->mutex);
    } else {
        struct timespec deadline;

        clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += timeout_ms / 1000U;
        deadline.tv_nsec += (long)(timeout_ms % 1000U) * 1000000L;
        if (deadline.tv_nsec >= 1000000000L) {
            ++deadline.tv_sec;
            deadline.tv_nsec -= 1000000000L;
        }
        while (runtime->calls[hart].pending && !runtime->worker_stop) {
            int wait_status = pthread_cond_timedwait(
                &runtime->condition, &runtime->mutex, &deadline);

            if (wait_status == ETIMEDOUT) {
                runtime->calls[hart].pending = 0;
                status = -ETIMEDOUT;
                break;
            }
            if (wait_status != 0) {
                runtime->calls[hart].pending = 0;
                status = -wait_status;
                break;
            }
        }
    }
    if (!status)
        status = runtime->worker_stop ? -ECANCELED :
                 runtime->calls[hart].status;
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

int npu_runtime_rpc_call(npu_runtime_t *runtime, uint32_t hart,
                         uint16_t opcode, uint64_t argument,
                         uint64_t *response_out, uint32_t timeout_ms)
{
    uint16_t response_opcode;
    int status;

    if (!runtime || !response_out || hart >= runtime->options.hart_count ||
        (opcode & 0x8000U) || !opcode ||
        (argument & ~RPC_COMMAND_ARG_MASK))
        return -EINVAL;
    pthread_mutex_lock(&runtime->mutex);
    if (!(runtime->booted_harts & (1U << hart))) {
        pthread_mutex_unlock(&runtime->mutex);
        return -ENODEV;
    }
    if (runtime->calls[hart].pending) {
        pthread_mutex_unlock(&runtime->mutex);
        return -EBUSY;
    }
    response_opcode = (uint16_t)(opcode | 0x8000U);
    runtime->calls[hart].expected_opcode = response_opcode;
    runtime->calls[hart].expected_argument = argument;
    runtime->calls[hart].response = 0;
    runtime->calls[hart].status = -EINPROGRESS;
    runtime->calls[hart].pending = 1;
    pthread_mutex_unlock(&runtime->mutex);

    status = npu_device_wait_msip_clear(
        &runtime->device, hart,
        timeout_ms == NPU_TIMEOUT_INFINITE ?
            runtime->options.default_timeout_ms : timeout_ms);
    if (status < 0) {
        call_cancel(runtime, hart, status);
        return status;
    }
    status = npu_device_submit(&runtime->device, hart,
                               rpc_command_pack(opcode, argument));
    if (status < 0) {
        call_cancel(runtime, hart, status);
        return status;
    }
    status = condition_wait(runtime, hart, timeout_ms);
    if (status < 0)
        return status;
    pthread_mutex_lock(&runtime->mutex);
    *response_out = runtime->calls[hart].response;
    pthread_mutex_unlock(&runtime->mutex);
    return 0;
}

static int launch_desc_valid(npu_runtime_t *runtime,
                             const npu_launch_desc_t *desc)
{
    uint64_t ram_limit = runtime->options.ram_base +
                         runtime->options.ram_size;
    uint64_t coherent_limit = runtime->options.coherent_base +
                              runtime->options.coherent_size;

    if (!desc || desc->version != NPU_LAUNCH_DESC_VERSION || desc->flags ||
        desc->hart >= runtime->options.hart_count ||
        desc->argc > NPU_LAUNCH_MAX_ARGS || (desc->entry & 1U) ||
        desc->entry < runtime->options.ram_base || desc->entry >= ram_limit ||
        desc->stack_size < RPC_PROC_MIN_STACK_SIZE ||
        desc->stack_size > RPC_PROC_MAX_STACK_SIZE ||
        desc->printf_buffer_size > RPC_PROC_MAX_PRINTF_SIZE)
        return 0;
    if (!desc->shared_base)
        return desc->shared_size <= runtime->options.coherent_size;
    return desc->shared_size && !(desc->shared_base & 15U) &&
           desc->shared_base >= runtime->options.coherent_base &&
           desc->shared_base < coherent_limit &&
           desc->shared_size <= coherent_limit - desc->shared_base;
}

static void launch_discard(npu_launch_t *launch)
{
    npu_runtime_t *runtime = launch->runtime;

    pthread_mutex_lock(&runtime->mutex);
    (void)npu_arena_free(&runtime->normal, launch->request_address,
                         NPU_ALLOC_LAUNCH, UINT32_MAX);
    if (launch->owned_shared)
        (void)npu_arena_free(&runtime->coherent, launch->owned_shared,
                             NPU_ALLOC_COHERENT, UINT32_MAX);
    if (runtime->active_launches)
        --runtime->active_launches;
    pthread_mutex_unlock(&runtime->mutex);
    free(launch);
}

int npu_runtime_launch(npu_runtime_t *runtime,
                       const npu_launch_desc_t *desc,
                       npu_launch_t **launch_out)
{
    struct rpc_proc_create_request request;
    npu_launch_t *launch;
    uint64_t response;
    uint64_t request_address;
    uint64_t shared_base = 0;
    uint32_t timeout;
    int status;

    if (!runtime || !launch_out || !launch_desc_valid(runtime, desc))
        return -EINVAL;
    *launch_out = 0;
    pthread_mutex_lock(&runtime->mutex);
    if (!runtime->firmware_loaded ||
        !(runtime->booted_harts & (1U << desc->hart))) {
        pthread_mutex_unlock(&runtime->mutex);
        return -ENODEV;
    }
    if (desc->shared_base) {
        shared_base = desc->shared_base;
    } else if (desc->shared_size) {
        status = npu_arena_alloc(&runtime->coherent, desc->shared_size, 64,
                                 NPU_ALLOC_COHERENT, UINT32_MAX,
                                 &shared_base);
        if (status < 0) {
            pthread_mutex_unlock(&runtime->mutex);
            return status;
        }
    }
    status = npu_arena_alloc(&runtime->normal, sizeof(request), 64,
                             NPU_ALLOC_LAUNCH, desc->hart,
                             &request_address);
    if (status < 0) {
        if (shared_base && !desc->shared_base)
            (void)npu_arena_free(&runtime->coherent, shared_base,
                                 NPU_ALLOC_COHERENT, UINT32_MAX);
        pthread_mutex_unlock(&runtime->mutex);
        return status;
    }
    ++runtime->active_launches;
    pthread_mutex_unlock(&runtime->mutex);

    launch = calloc(1, sizeof(*launch));
    if (!launch) {
        pthread_mutex_lock(&runtime->mutex);
        (void)npu_arena_free(&runtime->normal, request_address,
                             NPU_ALLOC_LAUNCH, desc->hart);
        if (shared_base && !desc->shared_base)
            (void)npu_arena_free(&runtime->coherent, shared_base,
                                 NPU_ALLOC_COHERENT, UINT32_MAX);
        --runtime->active_launches;
        pthread_mutex_unlock(&runtime->mutex);
        return -ENOMEM;
    }
    launch->runtime = runtime;
    launch->request_address = request_address;
    launch->owned_shared = desc->shared_base ? 0 : shared_base;
    launch->shared_base = shared_base;
    launch->shared_size = desc->shared_size;

    memset(&request, 0, sizeof(request));
    request.version = RPC_PROC_CREATE_VERSION;
    request.size = sizeof(request);
    request.state = RPC_PROC_SUBMITTED;
    request.entry = desc->entry;
    request.stack_size = desc->stack_size;
    request.shared_base = shared_base;
    request.shared_size = desc->shared_size;
    request.printf_buffer_size = desc->printf_buffer_size;
    request.argc = desc->argc;
    memcpy(request.args, desc->args, sizeof(request.args));
    request.status = -EINPROGRESS;
    request.cookie = desc->cookie;
    status = npu_device_write(&runtime->device, request_address,
                              &request, sizeof(request));
    if (status < 0) {
        launch_discard(launch);
        return status;
    }
    timeout = runtime->options.default_timeout_ms;
    status = npu_runtime_rpc_call(runtime, desc->hart, RPC_PROC_CREATE,
                                  request_address, &response, timeout);
    if (status < 0) {
        /* Submission is ambiguous after ringing MSIP; retain a recoverable
         * handle so the caller can query/wait instead of reusing storage. */
        *launch_out = launch;
        return status;
    }
    if (rpc_command_opcode(response) != RPC_PROC_CREATE_RESPONSE ||
        rpc_command_arg(response) != request_address) {
        *launch_out = launch;
        return -EPROTO;
    }
    status = npu_device_read(&runtime->device, request_address,
                             &request, sizeof(request));
    if (status < 0) {
        *launch_out = launch;
        return status;
    }
    if (request.state == RPC_PROC_FAILED) {
        if (request.proc_id) {
            *launch_out = launch;
            return 0;
        }
        status = request.status < 0 && request.status >= INT32_MIN ?
            (int)request.status : -EIO;
        launch_discard(launch);
        return status;
    }
    if (request.state != RPC_PROC_RUNNING &&
        request.state != RPC_PROC_EXITED) {
        *launch_out = launch;
        return -EPROTO;
    }
    *launch_out = launch;
    return 0;
}

int npu_launch_query(npu_launch_t *launch, npu_launch_info_t *info)
{
    struct rpc_proc_create_request request;
    int status;

    if (!launch || !info || launch->released)
        return -EINVAL;
    status = npu_device_read(&launch->runtime->device,
                             launch->request_address,
                             &request, sizeof(request));
    if (status < 0)
        return status;
    if (request.version != RPC_PROC_CREATE_VERSION ||
        request.size != sizeof(request))
        return -EPROTO;
    memset(info, 0, sizeof(*info));
    info->state = request.state;
    info->hart = request.hart_id;
    info->proc_id = request.proc_id;
    info->status = request.status;
    info->shared_base = launch->shared_base;
    info->shared_size = launch->shared_size;
    info->cookie = request.cookie;
    return 0;
}

int npu_launch_wait(npu_launch_t *launch, uint32_t timeout_ms,
                    npu_launch_info_t *info)
{
    uint64_t deadline;

    if (!launch || launch->released)
        return -EINVAL;
    deadline = timeout_ms == NPU_TIMEOUT_INFINITE ? UINT64_MAX :
               npu_now_ms() + timeout_ms;
    for (;;) {
        npu_launch_info_t current;
        int status = npu_launch_query(launch, &current);

        if (status < 0)
            return status;
        if (current.state == RPC_PROC_EXITED ||
            current.state == RPC_PROC_FAILED) {
            if (info)
                *info = current;
            return 0;
        }
        if (current.state != RPC_PROC_SUBMITTED &&
            current.state != RPC_PROC_RUNNING)
            return -EPROTO;
        if (npu_now_ms() >= deadline)
            return -ETIMEDOUT;
        npu_delay_us(launch->runtime->options.rpc_poll_interval_us);
    }
}

int npu_launch_release(npu_launch_t *launch)
{
    npu_launch_info_t info;
    int status;

    if (!launch || launch->released)
        return -EINVAL;
    status = npu_launch_query(launch, &info);
    if (status < 0)
        return status;
    if (info.state != RPC_PROC_EXITED && info.state != RPC_PROC_FAILED)
        return -EBUSY;
    launch->released = 1;
    launch_discard(launch);
    return 0;
}
