#ifndef NPU_TENSOR_INTERRUPT_H
#define NPU_TENSOR_INTERRUPT_H

#include <stddef.h>
#include <stdint.h>

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
};

struct tensor_interrupt_state {
    volatile uint32_t pending;
    uint32_t dropped_reports;
    struct tensor_interrupt_record record;
};

_Static_assert(sizeof(struct tensor_interrupt_record) == 176,
               "tensor interrupt report ABI");

void tensor_interrupt_handle(void);
void tensor_interrupt_drain(void);

#endif
