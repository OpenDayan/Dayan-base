#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "platform_regs.h"
#include "resource_scheduler.h"
#include "rpc.h"
#include "rpc_internal.h"
#include "string.h"

extern unsigned char __rpc_stack_start[];

struct rpc_call_context {
    struct rpc_transaction transaction;
    void *response;
    size_t response_size;
    _Alignas(RPC_DESCRIPTOR_GRANULE) struct rpc_packet packet;
    _Alignas(RPC_DESCRIPTOR_GRANULE) unsigned char request[RPC_DESCRIPTOR_GRANULE];
    _Alignas(RPC_DESCRIPTOR_GRANULE) unsigned char reply[RPC_DESCRIPTOR_GRANULE];
};

static void rpc_transaction_init(struct rpc_transaction *transaction,
                                 uint64_t command,
                                 rpc_cleanup_fn cleanup)
{
    thread_list_init(&transaction->wait);
    transaction->command = command;
    transaction->cleanup = cleanup;
    transaction->completed = 0;
    transaction->status = 0;
}

static void rpc_transaction_send(struct rpc_transaction *transaction)
{
    rpc_send_command(transaction->command);
    transaction->completed = 1;
    transaction->cleanup(transaction);
    (void)thread_event_wake_all(&transaction->wait);
}

void rpc_send_command(uint64_t command)
{
    cpu_self()->rpc.send(command);
}

static void rpc_transaction_submit(struct rpc_transaction *transaction)
{
    struct rpc_state *rpc = &cpu_self()->rpc;

    if (current->type == THREAD_TYPE_SYSTEM) {
        rpc_transaction_send(transaction);
        return;
    }
    list_add_tail(&transaction->node, &rpc->send_queue);
    (void)thread_event_wake_one(&rpc->send_event);
}

/* Sender and terminal callers finish queued reports before publishing exit. */
static void rpc_send_pending(void)
{
    struct rpc_state *rpc = &cpu_self()->rpc;

    tensor_interrupt_drain();
    while (!list_empty(&rpc->send_queue)) {
        struct rpc_transaction *transaction = list_first_entry(
            &rpc->send_queue, struct rpc_transaction, node);

        list_del_init(&transaction->node);
        rpc_transaction_send(transaction);
    }
}

static void rpc_send_main(void)
{
    struct rpc_state *rpc = &cpu_self()->rpc;

    for (;;) {
        uintptr_t interrupt_state = resource_scheduler_interrupt_disable();

        if (list_empty(&rpc->send_queue) &&
            !cpu_self()->tensor_interrupt.pending) {
            thread_prepare_wait(&rpc->send_event);
            resource_scheduler_interrupt_restore(interrupt_state);
            switch_context();
        } else {
            resource_scheduler_interrupt_restore(interrupt_state);
        }
        rpc_send_pending();
    }
}

void rpc_send_notify(void)
{
    struct rpc_state *rpc = &cpu_self()->rpc;

    (void)thread_event_wake_one(&rpc->send_event);
}

static void rpc_receive_main(void)
{
    for (;;) {
        uint64_t response_command = 0;
        uint64_t command = platform_rpc_receive();
        uint16_t opcode = rpc_command_opcode(command);
        int reply;

        if (!opcode && rpc_command_arg(command) == DAYAN_BOOTROM_RESET_VECTOR)
            platform_rpc_halt();
        reply = rpc_receive(opcode, rpc_command_arg(command), &response_command);
        if (reply > 0)
            platform_rpc_send(response_command);
        platform_rpc_receive_ack();
        thread_wait_rpc();
    }
}

static void rpc_start(thread_fn receiver, void (*send)(uint64_t), int local)
{
    struct rpc_state *rpc = &cpu_self()->rpc;
    uintptr_t hart = platform_hart_id();
    uintptr_t base = (uintptr_t)__rpc_stack_start +
                     hart * 2U * CONFIG_RPC_STACK_SIZE;

    INIT_LIST_HEAD(&rpc->send_queue);
    thread_list_init(&rpc->send_event);
    rpc->send = send;
    if (!local)
        platform_rpc_init();

    thread_start(thread_slot_get(RPC_PROFILE_SLOT_RPC_RECEIVE),
                  RPC_PROFILE_SLOT_RPC_RECEIVE, THREAD_TYPE_SYSTEM,
                  local ? "test-launch" : "rpc-receive", receiver,
                  (void *)base, CONFIG_RPC_STACK_SIZE, 0, 0, 0);
    base += CONFIG_RPC_STACK_SIZE;
    thread_start(thread_slot_get(RPC_PROFILE_SLOT_RPC_SEND),
                  RPC_PROFILE_SLOT_RPC_SEND, THREAD_TYPE_SYSTEM,
                  local ? "test-rpc" : "rpc-send", rpc_send_main,
                  (void *)base, CONFIG_RPC_STACK_SIZE, 0, 0, 0);
    if (local) {
        cpu_self()->rpc_receive_thread = 0;
        thread_schedule(thread_slot_get(RPC_PROFILE_SLOT_RPC_RECEIVE));
    }
}

void rpc_init(void)
{
    rpc_start(rpc_receive_main, platform_rpc_send, 0);
}

void rpc_start_local(thread_fn launcher, void (*send)(uint64_t command))
{
    rpc_start(launcher, send, 1);
}

static void rpc_call_cleanup(struct rpc_transaction *transaction)
{
    struct rpc_call_context *call = (struct rpc_call_context *)transaction;
    struct rpc_packet *packet = &call->packet;

    platform_cache_invalidate(packet, sizeof(*packet));
    if (packet->status >= 0 && call->response_size) {
        platform_cache_invalidate(call->reply, call->response_size);
        memcpy(call->response, call->reply, call->response_size);
    }
    transaction->status = (long)packet->status;
}

/* Firmware callers use fixed request/response types that fit one cache line. */
static void rpc_call_init(struct rpc_call_context *call, uint16_t service,
                         const void *request, size_t request_size,
                         void *response, size_t response_size)
{
    struct rpc_packet *packet = &call->packet;

    if (request_size)
        memcpy(call->request, request, request_size);
    if (response_size)
        memcpy(call->reply, response, response_size);
    packet->request_addr = request_size ? (uintptr_t)call->request : 0;
    packet->request_size = request_size;
    packet->response_addr = response_size ? (uintptr_t)call->reply : 0;
    packet->response_size = response_size;
    packet->status = -EIO;
    call->response = response;
    call->response_size = response_size;
    if (request_size)
        platform_cache_flush(call->request, request_size);
    if (response_size)
        platform_cache_flush(call->reply, response_size);
    platform_cache_flush(packet, sizeof(*packet));

    rpc_transaction_init(&call->transaction,
                         rpc_command_pack(service, (uintptr_t)packet),
                         rpc_call_cleanup);
}

long rpc_call(uint16_t service,
              const void *request, size_t request_size,
              void *response, size_t response_size)
{
    struct rpc_call_context call;

    rpc_call_init(&call, service, request, request_size,
                  response, response_size);
    rpc_transaction_submit(&call.transaction);
    if (!call.transaction.completed) {
        thread_system_wait_set(current, &call.transaction);
        thread_wait_event(&call.transaction.wait);
        thread_system_wait_set(current, 0);
    }
    return call.transaction.status;
}

long rpc_write_proc(uint32_t proc_id, const void *data, size_t size)
{
    struct rpc_print_request request;

    if (!size)
        return 0;
    request.hart_id = (uint32_t)platform_hart_id();
    request.proc_id = proc_id;
    request.data_addr = (uintptr_t)data;
    request.data_size = size;
    platform_cache_flush(data, size);
    return rpc_call(RPC_WRITE, &request, sizeof(request), 0, 0);
}

long rpc_heap_acquire(size_t size, struct rpc_heap_response *response)
{
    struct rpc_heap_request request;

    request.size = size;
    response->addr = 0;
    response->size = 0;
    return rpc_call(RPC_HEAP_ACQUIRE, &request, sizeof(request),
                    response, sizeof(*response));
}

long rpc_heap_release(uintptr_t addr, size_t size)
{
    struct rpc_heap_release release;

    release.addr = addr;
    release.size = size;
    return rpc_call(RPC_HEAP_RELEASE, &release, sizeof(release), 0, 0);
}

long rpc_proc_exit(struct task_proc *task, int64_t result)
{
    struct rpc_proc_exit_request request;
    struct rpc_call_context call;

    request.hart_id = (uint32_t)platform_hart_id();
    request.proc_id = thread_id(&task->thread);
    request.parent_proc_id = 0;
    request.flags = RPC_PROC_EXIT_ROOT;
    request.result = result;
    request.cookie = task->descriptor.cookie;
    /*
     * Terminal callers must finish on this stack without being rescheduled.
     * Senders never yield in flight; interrupts only queue work, so this
     * blocking exchange owns the mailbox until the Host acknowledges it.
     */
    rpc_call_init(&call, RPC_PROC_EXIT, &request, sizeof(request), 0, 0);
    rpc_send_pending();
    rpc_transaction_send(&call.transaction);
    return call.transaction.status;
}

static void rpc_trap_cleanup(struct rpc_transaction *transaction)
{
    (void)transaction;
    trap_report_complete();
}

int rpc_report_trap(const struct trap_record *record)
{
    struct rpc_state *rpc = &cpu_self()->rpc;

    /* A fault inside the sender cannot enqueue work for that same sender. */
    if (current == thread_slot_get(RPC_PROFILE_SLOT_RPC_SEND))
        return -EBUSY;

    platform_cache_flush(record, sizeof(*record));
    rpc_transaction_init(&rpc->trap,
                         rpc_command_pack(RPC_TRAP_REPORT, (uintptr_t)record),
                         rpc_trap_cleanup);
    rpc_transaction_submit(&rpc->trap);
    return 0;
}
