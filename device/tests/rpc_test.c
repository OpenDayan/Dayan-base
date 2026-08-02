#include <assert.h>
#include <stdint.h>

#include "platform_regs.h"
#include "rpc.h"
#include "trap.h"

int main(void)
{
    uint64_t command = rpc_command_pack(0x1234, 0xabcdef012345ULL);

    assert(rpc_command_opcode(command) == 0x1234);
    assert(rpc_command_arg(command) == 0xabcdef012345ULL);
    assert(sizeof(struct rpc_packet) == 40);
    assert(sizeof(struct rpc_print_request) == 24);
    assert(sizeof(struct rpc_heap_request) == 8);
    assert(sizeof(struct rpc_heap_response) == 16);
    assert(sizeof(struct rpc_heap_release) == 16);
    assert(sizeof(struct rpc_proc_create_request) == 208);
    assert(offsetof(struct rpc_proc_create_request, args) == 64);
    assert(offsetof(struct rpc_proc_create_request, status) == 128);
    assert(offsetof(struct rpc_proc_create_request, task_id) == 152);
    assert(offsetof(struct rpc_proc_create_request, task_dim) == 164);
    assert(offsetof(struct rpc_proc_create_request, task_mode) == 176);
    assert(offsetof(struct rpc_proc_create_request, max_child_procs) == 184);
    assert(offsetof(struct rpc_proc_create_request, uram_base) == 192);
    assert(RPC_PROC_CREATE == 0x0100);
    assert(RPC_PROC_CREATE_RESPONSE == (RPC_PROC_CREATE | 0x8000));
    assert(RPC_ICACHE_FLUSH == 0x0101);
    assert(RPC_ICACHE_FLUSH_RESPONSE == (RPC_ICACHE_FLUSH | 0x8000));
    assert(sizeof(struct trap_record) == 320);
    assert(DAYAN_DEVICE_MAILBOX(3) == 0x1018U);
    assert(DAYAN_IRQ_MAILBOX(3) == 0x2038U);
    assert(DAYAN_CLINT_MSIP(3) == 0x2000cU);
    return 0;
}
