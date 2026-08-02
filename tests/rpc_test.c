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
    assert(sizeof(struct trap_record) == 320);
    assert(DAYAN_DEVICE_MAILBOX(3) == 0x1018U);
    assert(DAYAN_IRQ_MAILBOX(3) == 0x2038U);
    assert(DAYAN_CLINT_MSIP(3) == 0x2000cU);
    return 0;
}
