#ifndef NPU_CPU_H
#define NPU_CPU_H

#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "list.h"
#include "rbtree.h"
#include "rpc_internal.h"
#include "thread.h"
#include "trap.h"

struct mem_state {
    struct rb_root free;
    struct list_head regions;
    size_t capacity;
    size_t available;
};

struct sche_state {
    struct thread init;
    struct list_head ready;
    struct list_head wait_poll;
    struct list_head wait_event;
    size_t event_waiters;
    uint32_t next_id;
};

struct cpu {
    struct sche_state sche;
    struct mem_state mem;
    struct rpc_state rpc;
    struct trap_state trap;
};

_Static_assert(offsetof(struct cpu, sche) == 0,
               "bl1 expects scheduler state at CPU-local offset 0");
_Static_assert(offsetof(struct sche_state, init) == 0,
               "bl1 expects the init thread at CPU-local offset 0");
_Static_assert(sizeof(struct cpu) <= CONFIG_CPU_LOCAL_SIZE,
               "cpu state exceeds its linker reservation");

static inline struct cpu *cpu_self(void)
{
    return current->cpu;
}

#endif
