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
#define RESOURCE_PRODUCE_BASE    0x020U
#define PRODUCE_BAR_GLOBAL       0x000U

#define RESOURCE_TCSR_ENCODING(address, read, reg) \
    (0x0bU | ((uint32_t)(reg) << 7) | ((uint32_t)(read) << 12) | \
     ((uint32_t)(address) << 13))

static inline uintptr_t resource_scheduler_interrupt_disable(void)
{
    uintptr_t previous;
    uintptr_t mask = 8U;

    __asm__ volatile("csrrc %0, mstatus, %1"
                     : "=r"(previous)
                     : "r"(mask)
                     : "memory");
    return previous;
}

static inline void resource_scheduler_interrupt_restore(uintptr_t previous)
{
    if (previous & 8U)
        __asm__ volatile("csrsi mstatus, 8" ::: "memory");
}

static inline void resource_scheduler_write_a0(uint32_t address,
                                               uint64_t value)
{
    register uint64_t a0 __asm__("a0") = value;

    switch (address) {
    case RESOURCE_WAIT_FREE:
        __asm__ volatile(".insn 4, %1" : : "r"(a0),
                         "i"(RESOURCE_TCSR_ENCODING(
                             RESOURCE_WAIT_FREE, 0, 10)) : "memory");
        break;
    case RESOURCE_WAIT_READY:
        __asm__ volatile(".insn 4, %1" : : "r"(a0),
                         "i"(RESOURCE_TCSR_ENCODING(
                             RESOURCE_WAIT_READY, 0, 10)) : "memory");
        break;
    case RESOURCE_WAIT_QOS:
        __asm__ volatile(".insn 4, %1" : : "r"(a0),
                         "i"(RESOURCE_TCSR_ENCODING(
                             RESOURCE_WAIT_QOS, 0, 10)) : "memory");
        break;
    case RESOURCE_WAIT:
        __asm__ volatile(".insn 4, %1" : : "r"(a0),
                         "i"(RESOURCE_TCSR_ENCODING(
                             RESOURCE_WAIT, 0, 10)) : "memory");
        break;
    case RESOURCE_CONSUME_READY:
        __asm__ volatile(".insn 4, %1" : : "r"(a0),
                         "i"(RESOURCE_TCSR_ENCODING(
                             RESOURCE_CONSUME_READY, 0, 10)) : "memory");
        break;
    case LOCAL_RESOURCE_INIT:
        __asm__ volatile(".insn 4, %1" : : "r"(a0),
                         "i"(RESOURCE_TCSR_ENCODING(
                             LOCAL_RESOURCE_INIT, 0, 10)) : "memory");
        break;
    default:
        __builtin_trap();
    }
}

static inline uintptr_t resource_scheduler_pop(void)
{
    register uintptr_t a0 __asm__("a0");

    __asm__ volatile(".insn 4, %1"
                     : "=r"(a0)
                     : "i"(RESOURCE_TCSR_ENCODING(
                           PRODUCE_BAR_GLOBAL, 1, 10))
                     : "memory");
    return a0;
}

static inline void resource_scheduler_drain(void)
{
    while (resource_scheduler_pop())
        ;
}

static inline void resource_scheduler_submit(uint64_t wait_free,
                                             uint64_t wait_ready,
                                             int8_t qos,
                                             uintptr_t token)
{
    uintptr_t interrupt_state = resource_scheduler_interrupt_disable();

    resource_scheduler_write_a0(RESOURCE_WAIT_FREE, wait_free);
    resource_scheduler_write_a0(RESOURCE_WAIT_READY, wait_ready);
    resource_scheduler_write_a0(RESOURCE_WAIT_QOS, (uint8_t)qos & 7U);
    resource_scheduler_write_a0(RESOURCE_WAIT, token);
    resource_scheduler_interrupt_restore(interrupt_state);
}

static __attribute__((always_inline)) inline void
resource_scheduler_local_init(void)
{
    uintptr_t interrupt_state = resource_scheduler_interrupt_disable();

    resource_scheduler_write_a0(LOCAL_RESOURCE_INIT, 0);
    resource_scheduler_interrupt_restore(interrupt_state);
}

#endif
