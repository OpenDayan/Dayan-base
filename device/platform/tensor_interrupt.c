/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include <stdint.h>

#include "cpu.h"
#include "platform.h"
#include "rpc_internal.h"
#include "string.h"
#include "tensor_interrupt.h"
#include "trap.h"

/*
 * Interrupt TCSRs: 0 control, 1-16 first records, 17-32 recent records,
 * 33 interrupt count.  The register list places them in the Global unit at
 * 256 + n.  Earlier RTL (and the early Dayan-emu) decoded them as a separate
 * unit 5 at n; on RTL that follows the list, a unit 5 read never retires and
 * stalls the Hart inside this handler.
 */
#define TENSOR_INTERRUPT_TCSR(n) (256U + (n))

void tensor_interrupt_handle(const struct trap_frame *frame)
{
    struct tensor_interrupt_state *state;
    struct tensor_interrupt_record *record;
    uint64_t control;
    size_t i;

    /* A failed unit may never retire its queue. Capture the error without
     * waiting for unit barriers, so reporting and Host recovery can proceed. */
    state = &cpu_self()->tensor_interrupt;
    control = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(0));
    if (state->pending) {
        record = &state->records[state->pending - 1U].record;
        ++record->dropped_reports;
        __builtin_riscv_tx_base_tcsr_write(1, 0, TENSOR_INTERRUPT_TCSR(0));
        return;
    }
    state->pending = state->outbound == 1U ? 2U : 1U;
    record = &state->records[state->pending - 1U].record;
    memset(record, 0, sizeof(*record));
    record->hart_id = (uint32_t)platform_hart_id();
    /* A terminal thread may have abandoned tp while dispatch waits for work. */
    if (current) {
        record->thread_id = thread_id(current);
        record->thread_type = current->type;
        record->thread_state = current->state;
        record->tensor_context = current->tensor_context;
    }
    record->recent_count = (uint32_t)((control >> 9) & 0x1fU);
    record->control = control;
    record->interrupt_count = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(33));
    record->recent[0] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(17));
    record->recent[1] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(18));
    record->recent[2] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(19));
    record->recent[3] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(20));
    record->recent[4] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(21));
    record->recent[5] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(22));
    record->recent[6] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(23));
    record->recent[7] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(24));
    record->recent[8] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(25));
    record->recent[9] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(26));
    record->recent[10] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(27));
    record->recent[11] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(28));
    record->recent[12] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(29));
    record->recent[13] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(30));
    record->recent[14] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(31));
    record->recent[15] = __builtin_riscv_tx_base_tcsr_read(0, TENSOR_INTERRUPT_TCSR(32));
    record->mstatus = frame->mstatus;
    record->mepc = frame->mepc;
    record->mcause = frame->mcause;
    record->mtval = frame->mtval;
    record->fcsr = frame->fcsr;
    for (i = 0; i < 32; ++i)
        record->x[i] = frame->x[i];
    __builtin_riscv_tx_base_tcsr_write(1, 0, TENSOR_INTERRUPT_TCSR(0));
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
    record = &state->records[state->outbound - 1U].record;
    platform_cache_flush(record, sizeof(*record));
    rpc_send_command(rpc_command_pack(RPC_TENSOR_INTERRUPT_REPORT,
                                       (uintptr_t)record));
    state->outbound = 0;
}
