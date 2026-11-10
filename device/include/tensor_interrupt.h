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

#ifndef NPU_TENSOR_INTERRUPT_H
#define NPU_TENSOR_INTERRUPT_H

#include <stddef.h>
#include <stdint.h>

#include "config.h"

#define TENSOR_INTERRUPT_RECENT_COUNT   16U

struct tensor_interrupt_record {
    uint32_t hart_id;
    uint32_t thread_id;
    uint32_t thread_type;
    uint32_t thread_state;
    uint32_t recent_count;
    uint32_t dropped_reports;
    uint64_t tensor_context;
    uint64_t control;
    uint64_t interrupt_count;
    uint64_t recent[TENSOR_INTERRUPT_RECENT_COUNT];
    uint64_t mstatus;
    uint64_t mepc;
    uint64_t mcause;
    uint64_t mtval;
    uint64_t fcsr;
    uint64_t x[32];
};

struct tensor_interrupt_state {
    /* Each Host-visible report owns whole cache lines, including its tail. */
    struct {
        _Alignas(CONFIG_CACHE_LINE_SIZE) struct tensor_interrupt_record record;
    } records[2];
    /* Zero means none; 1 and 2 select the per-Hart record buffers. */
    volatile uint8_t pending;
    volatile uint8_t outbound;
};

_Static_assert(sizeof(struct tensor_interrupt_record) == 472,
               "tensor interrupt report ABI");
_Static_assert(offsetof(struct tensor_interrupt_record, mstatus) == 176,
               "tensor interrupt CSR offset");
_Static_assert(offsetof(struct tensor_interrupt_record, x) == 216,
               "tensor interrupt GPR offset");

struct trap_frame;

void tensor_interrupt_handle(const struct trap_frame *frame);
void tensor_interrupt_drain(void);

#endif
