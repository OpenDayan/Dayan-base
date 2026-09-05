#include <assert.h>
#include <stdint.h>

#include "platform_regs.h"
#include "rpc.h"
#include "tensor_interrupt.h"
#include "trap.h"

int main(void)
{
    uint64_t command = rpc_command_pack(0x1234, 0xabcdef012345ULL);

    assert(rpc_command_opcode(command) == 0x1234);
    assert(rpc_command_arg(command) == 0xabcdef012345ULL);
    assert(RPC_DESCRIPTOR_GRANULE == 64U);
    assert(sizeof(struct rpc_packet) == 40);
    assert(sizeof(struct rpc_dcache_flush_request) == 40);
    assert(sizeof(struct rpc_print_request) == 24);
    assert(sizeof(struct rpc_heap_request) == 8);
    assert(sizeof(struct rpc_heap_response) == 16);
    assert(sizeof(struct rpc_heap_release) == 16);
    assert(sizeof(struct rpc_proc_exit_request) == 32);
    assert(sizeof(struct rpc_proc_create_request) == 224);
    assert(sizeof(struct rpc_proc_info) == 96);
    assert(sizeof(struct rpc_proc_ps_snapshot) == 1864);
    assert(sizeof(struct rpc_proc_kill_result) == 40);
    assert(RPC_PROC_KILL_FAMILY == 1U);
    assert(sizeof(struct rpc_profile_record) == 8);
    assert(sizeof(struct rpc_profile_slot_define) == 24);
    assert(sizeof(struct rpc_profile_config) == 56);
    assert(sizeof(struct rpc_profile_flush_request) == 32);
    assert(sizeof(struct rpc_profile_flush_response) == 24);
    assert(offsetof(struct rpc_proc_create_request, args) == 56);
    assert(offsetof(struct rpc_proc_create_request, status) == 128);
    assert(offsetof(struct rpc_proc_create_request, task_id) == 152);
    assert(offsetof(struct rpc_proc_create_request, task_dim) == 164);
    assert(offsetof(struct rpc_proc_create_request, task_mode) == 120);
    assert(offsetof(struct rpc_proc_create_request, max_child_threads) == 124);
    assert(offsetof(struct rpc_proc_create_request, resource_context) == 180);
    assert(offsetof(struct rpc_proc_create_request, uram_base) == 184);
    assert(offsetof(struct rpc_proc_create_request, tensor_context) == 200);
    assert(offsetof(struct rpc_proc_create_request, name) == 208);
    assert(offsetof(struct rpc_proc_info, name) == 80);
    assert(offsetof(struct rpc_profile_slot_define, name) == 8);
    assert(RPC_PROC_SLOTS_PER_HART == 16);
    assert(RPC_PROC_NAME_SIZE == 16);
    assert(RPC_PROC_MAX_CHILD_THREADS == 15);
    assert(RPC_PROC_CREATE == 0x0100);
    assert(RPC_PROC_CREATE_RESPONSE == (RPC_PROC_CREATE | 0x8000));
    assert(RPC_ICACHE_FLUSH == 0x0101);
    assert(RPC_ICACHE_FLUSH_RESPONSE == (RPC_ICACHE_FLUSH | 0x8000));
    assert(RPC_PROC_PS == 0x0102);
    assert(RPC_PROC_PS_RESPONSE == (RPC_PROC_PS | 0x8000));
    assert(RPC_PROC_KILL_ALL == 0x0103);
    assert(RPC_PROC_KILL_ALL_RESPONSE ==
           (RPC_PROC_KILL_ALL | 0x8000));
    assert(RPC_TENSOR_INTERRUPT_REPORT == 6);
    assert(RPC_PROC_EXIT == 7);
    assert(RPC_PROFILE_FLUSH == 8);
    assert(RPC_PROFILE_CONFIG == 0x0104);
    assert(RPC_PROFILE_CONFIG_RESPONSE ==
           (RPC_PROFILE_CONFIG | 0x8000));
    assert(RPC_DCACHE_FLUSH == 0x0105);
    assert(RPC_DCACHE_FLUSH_RESPONSE ==
           (RPC_DCACHE_FLUSH | 0x8000));
    assert(RPC_PROFILE_PROC_RUNNING == 0U);
    assert(RPC_PROFILE_PROC_KILL_WAIT_PRODUCE == 7U);
    assert(RPC_PROFILE_RECORD_SLOT_DEFINE == 0x80U);
    assert(RPC_PROFILE_SLOT_COUNT == 20U);
    assert(sizeof(struct trap_record) == 328);
    assert(sizeof(struct tensor_interrupt_record) == 472);
    assert(DAYAN_BOOTROM_RESET_VECTOR == 0x10000U);
    assert(DAYAN_DEVICE_MAILBOX(3) == 0x1018U);
    assert(DAYAN_IRQ_MAILBOX(3) == 0x2038U);
    assert(DAYAN_IRQ_ENABLE_CLR == 0x2014U);
    assert(DAYAN_CLINT_MSIP(3) == 0x2000cU);
    return 0;
}
