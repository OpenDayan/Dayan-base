#ifndef NPU_RESOURCE_SCHEDULER_H
#define NPU_RESOURCE_SCHEDULER_H

#include <stdint.h>

#define RESOURCE_WAIT_FREE       0x030U
#define RESOURCE_WAIT_READY      0x031U
#define RESOURCE_WAIT_QOS        0x032U
#define RESOURCE_WAIT            0x033U
#define RESOURCE_CONSUME_READY   0x034U
#define LOCAL_RESOURCE_INIT      0x035U
#define RESOURCE_TOKEN_EXT       0x036U
#define RESOURCE_WAIT_PHASE      0x037U
#define RESOURCE_PRODUCE_BASE    0x020U
#define PRODUCE_BAR_GLOBAL       0x000U

/* Scheduling is cooperative; tensor IRQs indicate unrecoverable failures. */
static inline void resource_scheduler_submit(uint64_t wait_free,
                                             uint64_t wait_ready,
                                             uint64_t phase,
                                             int8_t qos,
                                             uintptr_t token)
{
    __builtin_riscv_tx_base_tcsr_write(wait_free, 0, RESOURCE_WAIT_FREE);
    __builtin_riscv_tx_base_tcsr_write(wait_ready, 0, RESOURCE_WAIT_READY);
    __builtin_riscv_tx_base_tcsr_write(phase, 0, RESOURCE_WAIT_PHASE);
    __builtin_riscv_tx_base_tcsr_write((uint8_t)qos & 7U, 0, RESOURCE_WAIT_QOS);
    __builtin_riscv_tx_base_tcsr_write(token, 0, RESOURCE_WAIT);
}

static __attribute__((always_inline)) inline void
resource_scheduler_local_init(uint32_t mask)
{
    __builtin_riscv_tx_base_tcsr_write(mask, 0, LOCAL_RESOURCE_INIT);
}

#endif
