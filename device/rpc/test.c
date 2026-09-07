#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "stdio.h"
#include "test.h"

extern unsigned char __test_heap_end[];

long test_heap_acquire(size_t size, struct rpc_heap_response *response)
{
    struct test_state *state = &cpu_self()->test;
    uintptr_t end = (uintptr_t)__test_heap_end;
    const size_t mask = CONFIG_CACHE_LINE_SIZE - 1U;

    response->addr = 0;
    response->size = 0;
    if (!size)
        return -EINVAL;
    if (size > SIZE_MAX - mask)
        return -ENOMEM;
    size = (size + mask) & ~mask;
    if (state->heap_next > end || size > end - state->heap_next)
        return -ENOMEM;
    response->addr = state->heap_next;
    response->size = size;
    state->heap_next += size;
    return 0;
}

static long test_rpc_call(uint16_t service, struct rpc_packet *packet)
{
    const void *request = (const void *)(uintptr_t)packet->request_addr;
    void *response = (void *)(uintptr_t)packet->response_addr;

    switch (service) {
    case RPC_WRITE: {
        const struct rpc_print_request *print = request;

        if (!print || packet->request_size != sizeof(*print))
            return -EINVAL;
        return platform_print((const void *)(uintptr_t)print->data_addr,
                              print->data_size);
    }
    case RPC_HEAP_ACQUIRE: {
        const struct rpc_heap_request *heap = request;

        if (!heap || packet->request_size != sizeof(*heap) || !response ||
            packet->response_size != sizeof(struct rpc_heap_response))
            return -EINVAL;
        return test_heap_acquire(heap->size, response);
    }
    case RPC_HEAP_RELEASE:
        /* The backing allocator only advances; released regions are retired. */
        return request && packet->request_size == sizeof(struct rpc_heap_release)
                   ? 0 : -EINVAL;
    case RPC_PROC_EXIT: {
        const struct rpc_proc_exit_request *exit = request;
        struct test_state *state = &cpu_self()->test;

        if (!exit || packet->request_size != sizeof(*exit) ||
            exit->proc_id != state->proc_id ||
            exit->flags != RPC_PROC_EXIT_ROOT)
            return -EINVAL;
        if (exit->result || !state->result)
            state->result = exit->result;
        state->completed = 1;
        (void)thread_event_wake_all(&state->exit_event);
        return 0;
    }
    default:
        return -ENOSYS;
    }
}

void test_rpc_send(uint64_t command)
{
    uint16_t service = rpc_command_opcode(command);
    uintptr_t arg = rpc_command_arg(command);

    if (service == RPC_TRAP_REPORT || service == RPC_TENSOR_INTERRUPT_REPORT) {
        const struct trap_record *trap = (const void *)arg;
        char text[128];
        int size;

        cpu_self()->test.result = -EIO;
        if (service == RPC_TRAP_REPORT)
            size = snprintf(text, sizeof(text),
                            "[test] trap: hart=%u pc=%llx cause=%llx value=%llx\n",
                            trap->hart_id, (unsigned long long)trap->mepc,
                            (unsigned long long)trap->mcause,
                            (unsigned long long)trap->mtval);
        else
            size = snprintf(text, sizeof(text), "[test] tensor interrupt\n");
        if (size > 0)
            (void)platform_print(text, (size_t)size < sizeof(text) ?
                                 (size_t)size : sizeof(text) - 1U);
        return;
    }

    struct rpc_packet *packet = (void *)arg;

    packet->status = test_rpc_call(service, packet);
    /* The common RPC completion path invalidates the response before copying. */
    if (packet->status >= 0 && packet->response_size)
        platform_cache_flush((void *)(uintptr_t)packet->response_addr,
                             packet->response_size);
    platform_cache_flush(packet, sizeof(*packet));
}
