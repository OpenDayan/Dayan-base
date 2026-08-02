#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "errno.h"
#include "platform.h"
#include "rpc.h"
#include "sche.h"
#include "stdlib.h"

struct rpc_managed_proc {
    struct proc proc;
    struct rpc_proc_create_request *request;
    struct proc_family *family;
    void *stack;
    void *printf_buffer;
};

static int range_inside(uint64_t address, uint64_t size,
                        uint64_t base, uint64_t capacity)
{
    return size <= capacity && address >= base &&
           address - base <= capacity - size;
}

static int proc_request_address_valid(uint64_t address)
{
    return !(address & 7U) &&
           range_inside(address, sizeof(struct rpc_proc_create_request),
                        CONFIG_RAM_BASE, CONFIG_RAM_SIZE);
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
        request->max_child_procs > RPC_PROC_MAX_CHILD_PROCS ||
        request->stack_size < RPC_PROC_MIN_STACK_SIZE ||
        request->stack_size > RPC_PROC_MAX_STACK_SIZE ||
        request->printf_buffer_size > RPC_PROC_MAX_PRINTF_SIZE ||
        (request->entry & 1U) ||
        !range_inside(request->entry, 2, CONFIG_RAM_BASE, CONFIG_RAM_SIZE) ||
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
    struct rpc_managed_proc *managed = arg;
    struct rpc_proc_create_request *request = managed->request;
    int status = proc->exit_status;

    request->status = status;
    request->state = status < 0 ? RPC_PROC_FAILED : RPC_PROC_EXITED;
    platform_cache_clean(request, sizeof(*request));

    proc_family_destroy(managed->family);
    free(managed->printf_buffer);
    free(managed->stack);
    free(managed);
}

static int rpc_proc_create(uint64_t arg, uint64_t *response_command)
{
    struct rpc_proc_create_request *request;
    struct rpc_managed_proc *managed = 0;
    int status = -EINVAL;

    *response_command = rpc_command_pack(RPC_PROC_CREATE_RESPONSE, arg);
    if (!proc_request_address_valid(arg))
        return 1;

    request = (struct rpc_proc_create_request *)(uintptr_t)arg;
    platform_cache_invalidate(request, sizeof(*request));
    if (!proc_request_valid(request))
        goto complete;

    managed = calloc(1, sizeof(*managed));
    if (!managed) {
        status = -ENOMEM;
        goto complete;
    }
    managed->request = request;
    managed->stack = malloc((size_t)request->stack_size);
    managed->printf_buffer = request->printf_buffer_size ?
        malloc((size_t)request->printf_buffer_size) : 0;
    if (!managed->stack ||
        (request->printf_buffer_size && !managed->printf_buffer)) {
        status = -ENOMEM;
        goto complete;
    }
    status = proc_family_create(request->max_child_procs,
                                (size_t)request->stack_size,
                                (size_t)request->printf_buffer_size,
                                &managed->family);
    if (status < 0)
        goto complete;

    status = proc_init_args(&managed->proc,
                            (proc_fn)(uintptr_t)request->entry,
                            managed->stack, (size_t)request->stack_size,
                            (void *)(uintptr_t)request->shared_base,
                            managed->printf_buffer,
                            (size_t)request->printf_buffer_size,
                            request->argc, request->args);
    if (status < 0)
        goto complete;
    status = proc_set_resources(&managed->proc, managed->family,
                                (size_t)request->shared_size,
                                (void *)(uintptr_t)request->uram_base,
                                (size_t)request->uram_size);
    if (status < 0)
        platform_halt();
    status = proc_set_cleanup(&managed->proc, managed_proc_cleanup, managed);
    if (status < 0)
        platform_halt();
    status = proc_set_task(&managed->proc, request->task_id,
                           request->task_dim, request->task_mode);
    if (status < 0)
        platform_halt();

    request->status = 0;
    request->proc_id = managed->proc.thread.id;
    request->hart_id = (uint32_t)platform_hart_id();
    request->state = RPC_PROC_RUNNING;
    platform_cache_clean(request, sizeof(*request));
    return 1;

complete:
    if (managed) {
        proc_family_destroy(managed->family);
        free(managed->printf_buffer);
        free(managed->stack);
        free(managed);
    }
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

int rpc_receive(uint16_t opcode, uint64_t arg, uint64_t *response_command)
{
    if (!response_command)
        return -EINVAL;
    if (opcode == RPC_PROC_CREATE)
        return rpc_proc_create(arg, response_command);
    if (opcode == RPC_ICACHE_FLUSH)
        return rpc_icache_flush(arg, response_command);
    return -ENOSYS;
}
