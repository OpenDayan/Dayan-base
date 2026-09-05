#include <stdint.h>

#include "cpu.h"
#include "platform.h"
#include "rpc_internal.h"
#include "string.h"
#include "tensor_interrupt.h"
#include "trap.h"

struct tensor_interrupt_record_buffer {
    _Alignas(CONFIG_CACHE_LINE_SIZE) struct tensor_interrupt_record record;
};

static struct tensor_interrupt_record_buffer
    tensor_interrupt_records[CONFIG_HART_COUNT][2]
    __attribute__((section(".tensor_interrupt_records")));

static struct tensor_interrupt_record *tensor_interrupt_record_get(uint8_t id)
{
    uintptr_t hart = platform_hart_id();

    return &tensor_interrupt_records[hart][id - 1U].record;
}

void tensor_interrupt_handle(const struct trap_frame *frame)
{
    struct tensor_interrupt_state *state;
    struct tensor_interrupt_record *record;
    uint64_t control;
    size_t i;

    if (!current)
        platform_halt();
    __builtin_riscv_tx_base_tcsr_read(0, 1);
    __builtin_riscv_tx_base_tcsr_read(0, 2);
    __builtin_riscv_tx_base_tcsr_read(0, 3);
    __builtin_riscv_tx_base_tcsr_read(0, 4);
    state = &cpu_self()->tensor_interrupt;
    control = __builtin_riscv_tx_base_tcsr_read(5, 0);
    if (state->pending) {
        record = tensor_interrupt_record_get(state->pending);
        ++record->dropped_reports;
        __builtin_riscv_tx_base_tcsr_write(1, 5, 0);
        return;
    }
    state->pending = state->outbound == 1U ? 2U : 1U;
    record = tensor_interrupt_record_get(state->pending);
    memset(record, 0, sizeof(*record));
    record->hart_id = (uint32_t)platform_hart_id();
    record->thread_id = thread_id(current);
    record->thread_type = current->type;
    record->thread_state = current->state;
    record->recent_count = (uint32_t)((control >> 9) & 0x1fU);
    record->tensor_context = current->tensor_context;
    record->control = control;
    record->interrupt_count = __builtin_riscv_tx_base_tcsr_read(5, 33);
    record->recent[0] = __builtin_riscv_tx_base_tcsr_read(5, 17);
    record->recent[1] = __builtin_riscv_tx_base_tcsr_read(5, 18);
    record->recent[2] = __builtin_riscv_tx_base_tcsr_read(5, 19);
    record->recent[3] = __builtin_riscv_tx_base_tcsr_read(5, 20);
    record->recent[4] = __builtin_riscv_tx_base_tcsr_read(5, 21);
    record->recent[5] = __builtin_riscv_tx_base_tcsr_read(5, 22);
    record->recent[6] = __builtin_riscv_tx_base_tcsr_read(5, 23);
    record->recent[7] = __builtin_riscv_tx_base_tcsr_read(5, 24);
    record->recent[8] = __builtin_riscv_tx_base_tcsr_read(5, 25);
    record->recent[9] = __builtin_riscv_tx_base_tcsr_read(5, 26);
    record->recent[10] = __builtin_riscv_tx_base_tcsr_read(5, 27);
    record->recent[11] = __builtin_riscv_tx_base_tcsr_read(5, 28);
    record->recent[12] = __builtin_riscv_tx_base_tcsr_read(5, 29);
    record->recent[13] = __builtin_riscv_tx_base_tcsr_read(5, 30);
    record->recent[14] = __builtin_riscv_tx_base_tcsr_read(5, 31);
    record->recent[15] = __builtin_riscv_tx_base_tcsr_read(5, 32);
    record->mstatus = frame->mstatus;
    record->mepc = frame->mepc;
    record->mcause = frame->mcause;
    record->mtval = frame->mtval;
    record->fcsr = frame->fcsr;
    for (i = 0; i < 32; ++i)
        record->x[i] = frame->x[i];
    __builtin_riscv_tx_base_tcsr_write(1, 5, 0);
    rpc_send_notify();
}

void tensor_interrupt_drain(void)
{
    struct tensor_interrupt_state *state = &cpu_self()->tensor_interrupt;
    struct tensor_interrupt_record *record;

    if (!state->pending)
        return;
    state->outbound = state->pending;
    state->pending = 0;
    record = tensor_interrupt_record_get(state->outbound);
    platform_cache_flush(record, sizeof(*record));
    platform_rpc_send(rpc_command_pack(RPC_TENSOR_INTERRUPT_REPORT,
                                       (uintptr_t)record));
    state->outbound = 0;
}
