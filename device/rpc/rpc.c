#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "rpc.h"
#include "rpc_internal.h"
#include "string.h"

extern unsigned char __rpc_stack_start[];

struct rpc_call_context {
    struct rpc_transaction transaction;
    struct rpc_packet packet;
    void *response;
    size_t response_size;
};

static void rpc_transaction_init(struct rpc_transaction *transaction,
                                 uint64_t command,
                                 rpc_cleanup_fn cleanup, void *arg)
{
    INIT_LIST_HEAD(&transaction->node);
    INIT_LIST_HEAD(&transaction->wait);
    transaction->command = command;
    transaction->cleanup = cleanup;
    transaction->cleanup_arg = arg;
    transaction->completed = 0;
    transaction->status = 0;
}

static int rpc_transaction_submit(struct rpc_transaction *transaction)
{
    struct rpc_state *rpc = &cpu_self()->rpc;

    if (!rpc->initialized)
        return -ENOSYS;
    if (!transaction || !list_empty(&transaction->node))
        return -EBUSY;
    list_add_tail(&transaction->node, &rpc->send_queue);
    if (proc_event_wake_one(&rpc->send_event)) {
        /* Publish RPC responses before a newly-created proc can block. */
        list_move(&rpc->send.node, &rpc->send.thread.cpu->sche.ready);
    }
    return 0;
}

static void rpc_transaction_wait(struct rpc_transaction *transaction)
{
    struct proc *proc = proc_self();

    if (proc) {
        if (proc->system_wait)
            platform_halt();
        proc->system_wait = transaction;
    }
    while (!transaction->completed)
        proc_wait_event(&transaction->wait);
    if (proc)
        proc->system_wait = 0;
}

int rpc_proc_wait_cancel(struct proc *proc)
{
    struct rpc_transaction *transaction;

    if (!proc || proc->thread.cpu != cpu_self())
        return -EINVAL;
    transaction = proc->system_wait;
    if (!transaction)
        return 0;
    if (transaction->completed) {
        proc->system_wait = 0;
        return 0;
    }
    if (!list_empty(&transaction->node)) {
        list_del_init(&transaction->node);
        transaction->status = -ECANCELED;
        transaction->completed = 1;
        proc->system_wait = 0;
        return 0;
    }
    return 1;
}

static int rpc_receive_poll(struct proc *proc, void *arg)
{
    (void)proc;
    (void)arg;
    return platform_rpc_receive_pending();
}

static int rpc_send_poll(struct proc *proc, void *arg)
{
    (void)proc;
    (void)arg;
    return platform_rpc_send_ready();
}

static void rpc_send_main(void)
{
    struct rpc_state *rpc = &cpu_self()->rpc;

    for (;;) {
        struct rpc_transaction *transaction;

        while (list_empty(&rpc->send_queue))
            proc_wait_event(&rpc->send_event);

        transaction = list_first_entry(&rpc->send_queue,
                                       struct rpc_transaction, node);
        list_del_init(&transaction->node);

        while (!platform_rpc_send_ready())
            proc_wait_poll(rpc_send_poll, 0);
        platform_rpc_send(transaction->command);
        while (!platform_rpc_send_ready())
            proc_wait_poll(rpc_send_poll, 0);

        if (transaction->cleanup)
            transaction->cleanup(transaction, transaction->cleanup_arg);
        transaction->completed = 1;
        (void)proc_event_wake_all(&transaction->wait);
    }
}

__attribute__((weak))
int rpc_receive(uint16_t opcode, uint64_t arg, uint64_t *response_command)
{
    (void)opcode;
    (void)arg;
    (void)response_command;
    return 0;
}

static void rpc_receive_main(void)
{
    for (;;) {
        struct rpc_transaction response;
        uint64_t response_command = 0;
        uint64_t command;
        int reply;
        int status;

        while (!platform_rpc_receive_pending())
            proc_wait_poll(rpc_receive_poll, 0);

        command = platform_rpc_receive();
        reply = rpc_receive(rpc_command_opcode(command),
                            rpc_command_arg(command), &response_command);
        if (reply > 0) {
            rpc_transaction_init(&response, response_command, 0, 0);
            status = rpc_transaction_submit(&response);
            if (status < 0)
                platform_halt();
        }

        /* Host may reuse its mailbox only after this acknowledgement. */
        platform_rpc_receive_ack();

        if (reply > 0)
            rpc_transaction_wait(&response);
    }
}

int rpc_init(void)
{
    struct rpc_state *rpc = &cpu_self()->rpc;
    uintptr_t hart = platform_hart_id();
    uintptr_t base = (uintptr_t)__rpc_stack_start +
                     hart * 2U * CONFIG_RPC_STACK_SIZE;
    int status;

    rpc->initialized = 0;
    rpc->trap_pending = 0;
    INIT_LIST_HEAD(&rpc->send_queue);
    INIT_LIST_HEAD(&rpc->send_event);
    rpc_transaction_init(&rpc->trap, 0, 0, 0);
    platform_rpc_init();

    status = proc_init(&rpc->receive, "rpc-receive", rpc_receive_main,
                       (void *)base, CONFIG_RPC_STACK_SIZE, 0,
                       0, 0, 0);
    if (status < 0)
        return status;
    base += CONFIG_RPC_STACK_SIZE;
    status = proc_init(&rpc->send, "rpc-send", rpc_send_main,
                       (void *)base, CONFIG_RPC_STACK_SIZE, 0,
                       0, 0, 0);
    if (status < 0)
        return status;
    rpc->initialized = 1;
    return 0;
}

static void rpc_call_cleanup(struct rpc_transaction *transaction, void *arg)
{
    struct rpc_call_context *call = arg;

    platform_cache_invalidate(&call->packet, sizeof(call->packet));
    if (call->packet.status >= 0 && call->response_size)
        platform_cache_invalidate(call->response, call->response_size);
    transaction->status = (long)call->packet.status;
}

long rpc_call(uint32_t service,
              const void *request, size_t request_size,
              void *response, size_t response_size)
{
    struct rpc_call_context call;
    struct rpc_state *rpc;
    uintptr_t packet_addr;
    int status;

    if ((!request && request_size) || (!response && response_size) ||
        !service || service > UINT16_MAX)
        return -EINVAL;
    if (!current || current->type != THREAD_TYPE_PROC)
        return -EPERM;
    rpc = &cpu_self()->rpc;
    if (current == &rpc->send.thread)
        return -EPERM;

    call.packet.request_addr = (uintptr_t)request;
    call.packet.request_size = request_size;
    call.packet.response_addr = (uintptr_t)response;
    call.packet.response_size = response_size;
    call.packet.status = -EIO;
    call.response = response;
    call.response_size = response_size;
    packet_addr = (uintptr_t)&call.packet;
    if ((uint64_t)packet_addr & ~RPC_COMMAND_ARG_MASK)
        return -EOVERFLOW;

    if (request_size)
        platform_cache_clean(request, request_size);
    if (response_size)
        platform_cache_clean(response, response_size);
    platform_cache_clean(&call.packet, sizeof(call.packet));

    rpc_transaction_init(&call.transaction,
                         rpc_command_pack((uint16_t)service, packet_addr),
                         rpc_call_cleanup, &call);
    status = rpc_transaction_submit(&call.transaction);
    if (status < 0)
        return status;
    rpc_transaction_wait(&call.transaction);
    return call.transaction.status;
}

long rpc_write_proc(uint32_t proc_id, const void *data, size_t size)
{
    struct rpc_print_request request;

    if (!size)
        return 0;
    if (!proc_id || !data)
        return -EINVAL;
    request.hart_id = (uint32_t)platform_hart_id();
    request.proc_id = proc_id;
    request.data_addr = (uintptr_t)data;
    request.data_size = size;
    platform_cache_clean(data, size);
    return rpc_call(RPC_WRITE, &request, sizeof(request), 0, 0);
}

long rpc_write(const void *data, size_t size)
{
    struct proc *proc = proc_self();

    return proc ? rpc_write_proc(proc->host_proc_id ? proc->host_proc_id :
                                                      proc->thread.id,
                                 data, size) : -EPERM;
}

long rpc_clear_proc(uint32_t proc_id)
{
    struct rpc_print_request request;

    if (!proc_id)
        return -EINVAL;
    request.hart_id = (uint32_t)platform_hart_id();
    request.proc_id = proc_id;
    request.data_addr = 0;
    request.data_size = 0;
    return rpc_call(RPC_CLEAR, &request, sizeof(request), 0, 0);
}

long rpc_clear(void)
{
    struct proc *proc = proc_self();

    return proc ? rpc_clear_proc(proc->host_proc_id ? proc->host_proc_id :
                                                      proc->thread.id) :
                  -EPERM;
}

long rpc_heap_acquire(size_t size, struct rpc_heap_response *response)
{
    struct rpc_heap_request request;

    if (!size || !response)
        return -EINVAL;
    request.size = size;
    response->addr = 0;
    response->size = 0;
    return rpc_call(RPC_HEAP_ACQUIRE, &request, sizeof(request),
                    response, sizeof(*response));
}

long rpc_heap_release(uintptr_t addr, size_t size)
{
    struct rpc_heap_release release;

    if (!addr || !size)
        return -EINVAL;
    release.addr = addr;
    release.size = size;
    return rpc_call(RPC_HEAP_RELEASE, &release, sizeof(release), 0, 0);
}

long rpc_proc_exit(uint32_t proc_id, uint32_t parent_proc_id,
                   int64_t result, uint64_t cookie, uint32_t flags)
{
    struct rpc_proc_exit_request request;

    if (!proc_id || flags != RPC_PROC_EXIT_ROOT || parent_proc_id)
        return -EINVAL;
    request.version = RPC_PROC_EXIT_VERSION;
    request.size = sizeof(request);
    request.hart_id = (uint32_t)platform_hart_id();
    request.proc_id = proc_id;
    request.parent_proc_id = parent_proc_id;
    request.flags = flags;
    request.result = result;
    request.cookie = cookie;
    return rpc_call(RPC_PROC_EXIT, &request, sizeof(request), 0, 0);
}

long rpc_profile_flush(const struct rpc_profile_flush_request *request,
                       struct rpc_profile_flush_response *response)
{
    if (!request || !response)
        return -EINVAL;
    return rpc_call(RPC_PROFILE_FLUSH, request, sizeof(*request),
                    response, sizeof(*response));
}

static void rpc_trap_cleanup(struct rpc_transaction *transaction, void *arg)
{
    struct rpc_state *rpc = arg;

    (void)transaction;
    rpc->trap_pending = 0;
}

int rpc_report_trap(const void *record, size_t size)
{
    struct rpc_state *rpc = &cpu_self()->rpc;
    uintptr_t addr;
    int status;

    if (!record || !size)
        return -EINVAL;
    if (!rpc->initialized)
        return -ENOSYS;
    if (rpc->trap_pending || current == &rpc->send.thread)
        return -EBUSY;
    if (size > sizeof(rpc->trap_record))
        return -EOVERFLOW;
    memcpy(&rpc->trap_record, record, size);
    addr = (uintptr_t)&rpc->trap_record;
    if ((uint64_t)addr & ~RPC_COMMAND_ARG_MASK)
        return -EOVERFLOW;

    platform_cache_clean(&rpc->trap_record, size);
    rpc_transaction_init(&rpc->trap,
                         rpc_command_pack(RPC_TRAP_REPORT, addr),
                         rpc_trap_cleanup, rpc);
    rpc->trap_pending = 1;
    status = rpc_transaction_submit(&rpc->trap);
    if (status < 0)
        rpc->trap_pending = 0;
    return status;
}
