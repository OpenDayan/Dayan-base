#ifndef NPU_CPU_H
#define NPU_CPU_H

#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "context.h"
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
    struct profile_state *profiler;
    struct mem_state mem;
    struct rpc_state rpc;
    struct trap_state trap;
    struct tensor_interrupt_state tensor_interrupt;
};

extern uint8_t __cpu_start[];

_Static_assert(CONFIG_CPU_LOCAL_SIZE > 0 &&
                   !(CONFIG_CPU_LOCAL_SIZE & (CONFIG_CPU_LOCAL_SIZE - 1)),
               "CPU local size must be a power of two");
_Static_assert(offsetof(struct cpu, profiler) == CPU_PROFILER_OFFSET,
               "assembly profiler pointer offset");
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
