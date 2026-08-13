#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "errno.h"
#include "platform.h"
#include "profile.h"
#include "rpc.h"
#include "sche.h"
#include "stdlib.h"
#include "string.h"

_Static_assert(RPC_DESCRIPTOR_GRANULE == CONFIG_CACHE_LINE_SIZE,
               "RPC descriptors must occupy whole cache lines");

static int range_inside(uint64_t address, uint64_t size,
                        uint64_t base, uint64_t capacity)
{
    return size <= capacity && address >= base &&
           address - base <= capacity - size;
}

static int host_normal_range_valid(uint64_t address, uint64_t size)
{
    return size && address >= CONFIG_RAM_BASE &&
           size <= UINT64_MAX - address;
}

static int rpc_descriptor_address_valid(uint64_t address, size_t size)
{
    return !(address & (RPC_DESCRIPTOR_GRANULE - 1U)) &&
           host_normal_range_valid(address, size);
}

static int dcache_flush_request_valid(
    const struct rpc_dcache_flush_request *request)
{
    return request->version == RPC_DCACHE_FLUSH_VERSION &&
           request->size == sizeof(*request) &&
           request->hart_id == platform_hart_id() && !request->flags &&
           !request->reserved && request->length &&
           host_normal_range_valid(request->address, request->length);
}

static int proc_request_valid(const struct rpc_proc_create_request *request)
{
    uint64_t coherent_base = CONFIG_COHERENT_BASE;
    uint64_t coherent_size = CONFIG_COHERENT_SIZE;
    uint64_t uram_window = CONFIG_URAM_BASE +
        (uint64_t)platform_hart_id() * CONFIG_URAM_STRIDE;
    int shared_valid;
    int uram_valid;

    if (request->version != RPC_PROC_CREATE_VERSION ||
        request->size != sizeof(*request) || request->flags ||
        request->reserved || request->task_reserved ||
        request->resource_reserved ||
        request->state != RPC_PROC_SUBMITTED ||
        request->argc > RPC_PROC_MAX_ARGS ||
        !request->name[0] ||
        strnlen(request->name, sizeof(request->name)) ==
            sizeof(request->name) ||
        request->max_child_procs > RPC_PROC_MAX_CHILD_PROCS ||
        request->stack_size < RPC_PROC_MIN_STACK_SIZE ||
        request->stack_size > RPC_PROC_MAX_STACK_SIZE ||
        request->printf_buffer_size > RPC_PROC_MAX_PRINTF_SIZE ||
        (request->entry & 1U) ||
        !host_normal_range_valid(request->entry, 2) ||
        !request->task_dim[0] || !request->task_dim[1] ||
        !request->task_dim[2] ||
        request->task_id[0] >= request->task_dim[0] ||
        request->task_id[1] >= request->task_dim[1] ||
        request->task_id[2] >= request->task_dim[2] ||
        (request->task_mode != 1U && request->task_mode != 4U) ||
        (request->task_mode == 4U && request->task_dim[0] != 4U))
        return 0;
    shared_valid = (!request->shared_base && !request->shared_size) ||
        (request->shared_base && request->shared_size &&
         !(request->shared_base & 15U) &&
         range_inside(request->shared_base, request->shared_size,
                      coherent_base, coherent_size));
    uram_valid = (!request->uram_base && !request->uram_size) ||
        (request->uram_base && request->uram_size &&
         !(request->uram_base & (RPC_PROC_URAM_GRANULE - 1U)) &&
         !(request->uram_size & (RPC_PROC_URAM_GRANULE - 1U)) &&
         range_inside(request->uram_base, request->uram_size,
                      uram_window, CONFIG_URAM_SIZE));
    return shared_valid && uram_valid;
}

static void managed_proc_cleanup(struct proc *proc, void *arg)
{
    struct rpc_proc_create_request *request = arg;
    int64_t status = proc->exit_status;

    request->status = status;
    request->state = status < 0 ? RPC_PROC_FAILED : RPC_PROC_EXITED;
    platform_cache_clean(request, sizeof(*request));
}

static int rpc_proc_create(uint64_t arg, uint64_t *response_command)
{
    struct rpc_proc_create_request *request;
    struct proc *proc = 0;
    struct proc_family *family = 0;
    int status = -EINVAL;

    *response_command = rpc_command_pack(RPC_PROC_CREATE_RESPONSE, arg);
    if (!rpc_descriptor_address_valid(arg, sizeof(*request)))
        return 1;

    request = (struct rpc_proc_create_request *)(uintptr_t)arg;
    platform_cache_invalidate(request, sizeof(*request));
    if (!proc_request_valid(request))
        goto complete;

    status = proc_slot_acquire((size_t)request->stack_size,
                               (size_t)request->printf_buffer_size, &proc);
    if (status < 0)
        goto complete;
    status = proc_family_create(proc, request->max_child_procs,
                                (size_t)request->stack_size,
                                (size_t)request->printf_buffer_size,
                                &family);
    if (status < 0)
        goto complete;

    status = proc_slot_init_args(proc, request->name,
                                 (proc_fn)(uintptr_t)request->entry,
                                 (void *)(uintptr_t)request->shared_base,
                                 request->argc, request->args);
    if (status < 0)
        goto complete;
    status = proc_set_resources(proc, family,
                                (size_t)request->shared_size,
                                (void *)(uintptr_t)request->uram_base,
                                (size_t)request->uram_size);
    if (status < 0)
        platform_halt();
    proc->cookie = request->cookie;
    proc->host_proc_id = proc->thread.id;
    status = proc_set_cleanup(proc, managed_proc_cleanup, request);
    if (status < 0)
        platform_halt();
    status = proc_set_task(proc, request->task_id,
                           request->task_dim, request->task_mode);
    if (status < 0)
        platform_halt();

    request->status = 0;
    request->proc_id = proc->thread.id;
    request->hart_id = (uint32_t)platform_hart_id();
    request->state = RPC_PROC_RUNNING;
    platform_cache_clean(request, sizeof(*request));
    return 1;

complete:
    if (family)
        proc_family_destroy(family);
    if (proc)
        proc_slot_release(proc);
    request->status = status;
    request->proc_id = 0;
    request->hart_id = (uint32_t)platform_hart_id();
    request->state = RPC_PROC_FAILED;
    platform_cache_clean(request, sizeof(*request));
    return 1;
}

static int rpc_icache_flush(uint64_t token, uint64_t *response_command)
{
    platform_instruction_fence();
    *response_command = rpc_command_pack(RPC_ICACHE_FLUSH_RESPONSE, token);
    return 1;
}

static int rpc_dcache_flush(uint64_t arg, uint64_t *response_command)
{
    struct rpc_dcache_flush_request *request;
    uint64_t address;
    uint64_t length;
    int status = -EINVAL;

    *response_command = rpc_command_pack(RPC_DCACHE_FLUSH_RESPONSE, arg);
    if (!rpc_descriptor_address_valid(arg, sizeof(*request)))
        return 1;
    request = (struct rpc_dcache_flush_request *)(uintptr_t)arg;
    platform_cache_invalidate(request, sizeof(*request));
    if (dcache_flush_request_valid(request)) {
        address = request->address;
        length = request->length;
        platform_cache_invalidate((void *)(uintptr_t)address,
                                  (size_t)length);
        status = 0;
    }
    request->hart_id = (uint32_t)platform_hart_id();
    request->status = status;
    platform_cache_clean(request, sizeof(*request));
    return 1;
}

static int rpc_proc_ps(uint64_t arg, uint64_t *response_command)
{
    struct rpc_proc_ps_snapshot *snapshot;
    int status;

    *response_command = rpc_command_pack(RPC_PROC_PS_RESPONSE, arg);
    if (!rpc_descriptor_address_valid(arg, sizeof(*snapshot)))
        return 1;
    snapshot = (struct rpc_proc_ps_snapshot *)(uintptr_t)arg;
    platform_cache_invalidate(snapshot, sizeof(*snapshot));
    status = proc_snapshot(snapshot);
    if (status < 0) {
        snapshot->hart_id = (uint32_t)platform_hart_id();
        snapshot->count = 0;
        snapshot->status = status;
    }
    platform_cache_clean(snapshot, sizeof(*snapshot));
    return 1;
}

static int rpc_proc_kill_all(uint64_t arg, uint64_t *response_command)
{
    struct rpc_proc_kill_result *result;
    int status;

    *response_command = rpc_command_pack(RPC_PROC_KILL_ALL_RESPONSE, arg);
    if (!rpc_descriptor_address_valid(arg, sizeof(*result)))
        return 1;
    result = (struct rpc_proc_kill_result *)(uintptr_t)arg;
    platform_cache_invalidate(result, sizeof(*result));
    status = proc_kill_all(result);
    if (status < 0) {
        result->hart_id = (uint32_t)platform_hart_id();
        result->status = status;
        result->killed_roots = 0;
        result->killed_children = 0;
        result->released_slots = 0;
        result->waiting_produce = 0;
    }
    platform_cache_clean(result, sizeof(*result));
    return 1;
}

static int rpc_profile_config(uint64_t arg, uint64_t *response_command)
{
    struct rpc_profile_config *config;
    int status;

    *response_command = rpc_command_pack(RPC_PROFILE_CONFIG_RESPONSE, arg);
    if (!rpc_descriptor_address_valid(arg, sizeof(*config)))
        return 1;
    config = (struct rpc_profile_config *)(uintptr_t)arg;
    platform_cache_invalidate(config, sizeof(*config));
    status = profile_configure(config);
    if (status < 0) {
        config->hart_id = (uint32_t)platform_hart_id();
        config->status = status;
    }
    platform_cache_clean(config, sizeof(*config));
    return 1;
}

int rpc_receive(uint16_t opcode, uint64_t arg, uint64_t *response_command)
{
    if (!response_command)
        return -EINVAL;
    if (opcode == RPC_PROC_CREATE)
        return rpc_proc_create(arg, response_command);
    if (opcode == RPC_ICACHE_FLUSH)
        return rpc_icache_flush(arg, response_command);
    if (opcode == RPC_DCACHE_FLUSH)
        return rpc_dcache_flush(arg, response_command);
    if (opcode == RPC_PROC_PS)
        return rpc_proc_ps(arg, response_command);
    if (opcode == RPC_PROC_KILL_ALL)
        return rpc_proc_kill_all(arg, response_command);
    if (opcode == RPC_PROFILE_CONFIG)
        return rpc_profile_config(arg, response_command);
    return -ENOSYS;
}
