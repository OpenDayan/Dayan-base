#ifndef NPU_THREAD_H
#define NPU_THREAD_H

#include <stddef.h>
#include <stdint.h>

#include "context.h"

struct cpu;

enum thread_state {
    THREAD_RUNNING = 0,
    THREAD_READY = 1,
    THREAD_WAIT_EVENT = 2,
    THREAD_WAIT_START = 3,
    THREAD_WAIT_EXIT = 4,
};

enum thread_type {
    THREAD_TYPE_BASE = 0,
    THREAD_TYPE_PROC = 1,
};

_Static_assert(THREAD_RUNNING == 0, "running state ABI");
_Static_assert(THREAD_READY == 1, "ready state ABI");
_Static_assert(THREAD_WAIT_EVENT >= 2, "wait state ABI");
_Static_assert(THREAD_TYPE_BASE == 0, "base type ABI");

/* The fixed prefix addressed by tp for every execution context. */
struct thread {
    uint32_t id;
    uint32_t state;
    struct cpu *cpu;
    uintptr_t sp;
    uintptr_t pc;
    uint32_t type;
};

_Static_assert(offsetof(struct thread, id) == 0, "thread.id offset");
_Static_assert(offsetof(struct thread, state) == 4, "thread.state offset");
_Static_assert(offsetof(struct thread, cpu) == 8, "thread.cpu offset");
_Static_assert(offsetof(struct thread, sp) == THREAD_SP_OFFSET,
               "thread.sp offset");
_Static_assert(offsetof(struct thread, pc) == THREAD_PC_OFFSET,
               "thread.pc offset");
_Static_assert(offsetof(struct thread, type) == 32, "thread.type offset");
_Static_assert(sizeof(struct thread) == 40, "thread layout must stay fixed");

int thread_init(struct thread *thread, uint32_t type);
void switch_context(struct thread *prev, struct thread *next);

static inline struct thread *thread_self(void)
{
    struct thread *thread;

    __asm__ volatile("mv %0, tp" : "=r"(thread));
    return thread;
}

#endif
