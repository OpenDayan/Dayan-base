#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "profile.h"
#include "rpc.h"
#include "sche.h"
#include "stdlib.h"
#include "string.h"
#include "tensor_context.h"

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
    return request->hart_id == platform_hart_id() && !request->flags &&
           !request->reserved && request->length &&
           host_normal_range_valid(request->address, request->length);
}

static int proc_request_valid(const struct rpc_proc_create_request *request)
{
    struct dayan_context_fields tensor_context =
        dayan_context_unpack(request->tensor_context);
    struct dayan_resource_context_fields resource_context =
        dayan_resource_context_unpack(request->resource_context);
    uint64_t coherent_base = CONFIG_COHERENT_BASE;
    uint64_t coherent_size = CONFIG_COHERENT_SIZE;
    uint64_t uram_window = CONFIG_URAM_BASE +
        (uint64_t)platform_hart_id() * CONFIG_URAM_STRIDE;
    int shared_valid;
    int uram_valid;
    int context_uram_valid;

    if (request->flags || request->reserved || request->task_reserved ||
        request->state != RPC_PROC_SUBMITTED ||
        request->argc > RPC_PROC_MAX_ARGS ||
        !request->name[0] ||
        strnlen(request->name, sizeof(request->name)) ==
            sizeof(request->name) ||
        request->max_child_threads > RPC_PROC_MAX_CHILD_THREADS ||
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
        (request->task_mode == 4U && request->task_dim[0] != 4U) ||
        (request->resource_context >> 22) ||
        !dayan_resource_context_fields_valid(&resource_context) ||
        (request->tensor_context >> 56) ||
        !dayan_context_fields_valid(&tensor_context))
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
    context_uram_valid =
        tensor_context.uram_lo ==
            ((request->uram_base & (DAYAN_CONTEXT_URAM_WINDOW - 1U)) /
             DAYAN_CONTEXT_URAM_GRANULE) &&
        tensor_context.uram_hi ==
            (((request->uram_base & (DAYAN_CONTEXT_URAM_WINDOW - 1U)) +
              request->uram_size) / DAYAN_CONTEXT_URAM_GRANULE);
    return shared_valid && uram_valid && context_uram_valid;
}

static void proc_request_publish_state(
    struct rpc_proc_create_request *request, uint32_t state)
{
    platform_cache_flush((unsigned char *)request +
                         offsetof(struct rpc_proc_create_request, status),
                         CONFIG_CACHE_LINE_SIZE);
    request->state = state;
    platform_cache_flush(request, CONFIG_CACHE_LINE_SIZE);
}

void rpc_proc_cleanup(struct thread *thread, void *arg)
{
    struct rpc_proc_create_request *request = arg;
    int64_t status = thread_result(thread);

    request->status = status;
    proc_request_publish_state(
        request, status < 0 ? RPC_PROC_FAILED : RPC_PROC_EXITED);
}

static int rpc_proc_create(uint64_t arg, uint64_t *response_command)
{
    struct rpc_proc_create_request *request;
    struct task_proc *task = 0;
    struct task_descriptor descriptor;
    uintptr_t args[RPC_PROC_MAX_ARGS];
    size_t index;
    int status = -EINVAL;

    if (cpu_self()->rpc.pending_start)
        platform_halt();

    *response_command = rpc_command_pack(RPC_PROC_CREATE_RESPONSE, arg);
    if (!rpc_descriptor_address_valid(arg, sizeof(*request)))
        return 1;

    request = (struct rpc_proc_create_request *)(uintptr_t)arg;
    platform_cache_invalidate(request, sizeof(*request));
    if (!proc_request_valid(request))
        goto complete;
    for (index = 0; index < request->argc; ++index)
        args[index] = ((const volatile uint64_t *)request->args)[index];

    memset(&descriptor, 0, sizeof(descriptor));
    memcpy(descriptor.task_id, request->task_id, sizeof(descriptor.task_id));
    memcpy(descriptor.task_dim, request->task_dim,
           sizeof(descriptor.task_dim));
    descriptor.task_mode = (uint8_t)request->task_mode;
    descriptor.max_child_threads = (uint8_t)request->max_child_threads;
    descriptor.child_stack_size_m1 = (uint16_t)(request->stack_size - 1U);
    descriptor.child_printf_size = (uint16_t)request->printf_buffer_size;
    descriptor.shared_offset = request->shared_base ?
        (uint32_t)(request->shared_base - CONFIG_COHERENT_BASE) : 0;
    descriptor.shared_size = (uint32_t)request->shared_size;
    descriptor.cookie = request->cookie;

    status = task_proc_slot_acquire(&task);
    if (status < 0)
        goto complete;
    status = task_proc_set_descriptor(task, &descriptor,
                                      request->tensor_context,
                                      request->resource_context);
    if (status < 0)
        goto complete;
    status = task_proc_reserve(task);
    if (status < 0)
        goto complete;
    status = task_proc_start(task, request->name,
                             (thread_fn)(uintptr_t)request->entry,
                             request->argc, args);
    if (status < 0)
        goto complete;
    status = task_proc_set_cleanup_arg(task, request);
    if (status < 0)
        platform_halt();

    request->status = 0;
    request->proc_id = task->thread.id;
    request->hart_id = (uint32_t)platform_hart_id();
    proc_request_publish_state(request, RPC_PROC_RUNNING);
    cpu_self()->rpc.pending_start = task;
    return 1;

complete:
    if (task) {
        task_proc_unreserve(task);
        task_proc_slot_release(task);
    }
    request->status = status;
    request->proc_id = 0;
    request->hart_id = (uint32_t)platform_hart_id();
    proc_request_publish_state(request, RPC_PROC_FAILED);
    return 1;
}

void rpc_receive_reply_queued(uint16_t opcode)
{
    struct task_proc *task = cpu_self()->rpc.pending_start;

    if (opcode != RPC_PROC_CREATE) {
        if (task)
            platform_halt();
        return;
    }
    if (!task)
        return;
    cpu_self()->rpc.pending_start = 0;
    if (task_proc_schedule(task) < 0)
        platform_halt();
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
    platform_cache_flush(request, sizeof(*request));
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
    platform_cache_flush(snapshot, sizeof(*snapshot));
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
    platform_cache_flush(result, sizeof(*result));
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
    platform_cache_flush(config, sizeof(*config));
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
