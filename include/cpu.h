#ifndef NPU_CPU_H
#define NPU_CPU_H

#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "thread.h"

struct mem_region;
struct proc;
struct rb_node;

struct mem_state {
    struct rb_node *free;
    struct mem_region *regions;
    size_t capacity;
    size_t available;
};

struct sche_state {
    struct thread init;
    struct proc *head;
    struct proc *tail;
    uint32_t next_id;
};

struct cpu {
    struct sche_state sche;
    struct mem_state mem;
};

_Static_assert(offsetof(struct cpu, sche) == 0,
               "bl1 expects scheduler state at CPU-local offset 0");
_Static_assert(offsetof(struct sche_state, init) == 0,
               "bl1 expects the init thread at CPU-local offset 0");
_Static_assert(sizeof(struct cpu) <= CONFIG_CPU_LOCAL_SIZE,
               "cpu state exceeds its linker reservation");

static inline struct cpu *cpu_self(void)
{
    return thread_self()->cpu;
}

#endif
