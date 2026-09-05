#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "errno.h"
#include "platform.h"
#include "profile.h"
#include "rpc.h"
#include "sche.h"

_Static_assert(RPC_DESCRIPTOR_GRANULE == CONFIG_CACHE_LINE_SIZE,
               "RPC descriptors must occupy whole cache lines");

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

static int rpc_proc_create(uint64_t arg, uint64_t *response_command)
{
    struct rpc_proc_create_request *request;
    struct task_proc *task = 0;
    int status = -EINVAL;

    *response_command = rpc_command_pack(RPC_PROC_CREATE_RESPONSE, arg);
    if (!rpc_descriptor_address_valid(arg, sizeof(*request)))
        return 1;

    request = (struct rpc_proc_create_request *)(uintptr_t)arg;
    platform_cache_invalidate(request, sizeof(*request));
    /* Host validates the ABI; only this image's static capacity is local. */
    if (request->stack_size <= CONFIG_PROC_STACK_SIZE &&
        request->printf_buffer_size <= CONFIG_PROC_PRINTF_SIZE) {
        task = task_proc_create(request);
        status = task ? 0 : -ENOMEM;
    }
    request->status = status;
    request->proc_id = task ? task->thread.id : 0;
    request->hart_id = (uint32_t)platform_hart_id();
    request->state = task ? RPC_PROC_RUNNING : RPC_PROC_FAILED;
    /* Return every descriptor cache line to Host before replying. */
    platform_cache_flush(request, sizeof(*request));
    if (task) {
        platform_instruction_fence();
        /* The blocking receiver will reply and ack before its next switch. */
        thread_schedule(&task->thread);
    }
    return 1;
}

static int rpc_dcache_flush(uint64_t arg, uint64_t *response_command)
{
    struct rpc_dcache_flush_request *request;
    int status = -EINVAL;

    *response_command = rpc_command_pack(RPC_DCACHE_FLUSH_RESPONSE, arg);
    if (!rpc_descriptor_address_valid(arg, sizeof(*request)))
        return 1;
    request = (struct rpc_dcache_flush_request *)(uintptr_t)arg;
    platform_cache_invalidate(request, sizeof(*request));
    if (request->hart_id == platform_hart_id() && !request->flags &&
        !request->reserved &&
        host_normal_range_valid(request->address, request->length)) {
        platform_cache_invalidate((void *)(uintptr_t)request->address,
                                  (size_t)request->length);
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
    if (opcode == RPC_PROC_CREATE)
        return rpc_proc_create(arg, response_command);
    if (opcode == RPC_ICACHE_FLUSH) {
        platform_instruction_fence();
        *response_command = rpc_command_pack(RPC_ICACHE_FLUSH_RESPONSE, arg);
        return 1;
    }
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
