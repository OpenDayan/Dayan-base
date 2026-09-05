#ifndef NPU_CPU_H
#define NPU_CPU_H

#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "list.h"
#include "rbtree.h"
#include "rpc_internal.h"
#include "thread.h"
#include "tensor_interrupt.h"
#include "trap.h"

struct mem_state {
    struct rb_root free;
    struct list_head regions;
    size_t capacity;
    size_t available;
};

struct cpu {
    uint32_t profile_enabled;
    struct mem_state mem;
    struct rpc_state rpc;
    struct trap_state trap;
    struct tensor_interrupt_state tensor_interrupt;
};

extern uint8_t __cpu_start[];

_Static_assert(sizeof(struct cpu) <= CONFIG_CPU_LOCAL_SIZE,
               "cpu state exceeds its linker reservation");

static inline struct cpu *cpu_self(void)
{
    uintptr_t hart;
    uintptr_t base;

    __asm__ volatile("csrr %0, mhartid\n"
                     "la %1, __cpu_start"
                     : "=r"(hart), "=r"(base));
    return (struct cpu *)(base + hart * CONFIG_CPU_LOCAL_SIZE);
}

#endif
