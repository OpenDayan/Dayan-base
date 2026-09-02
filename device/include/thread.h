#ifndef NPU_THREAD_H
#define NPU_THREAD_H

#include <stdint.h>

enum thread_state {
    THREAD_RUNNING = 0,
    THREAD_READY = 1,
    THREAD_WAIT_POLL = 2,
    THREAD_WAIT_EVENT = 3,
    THREAD_WAIT_START = 4,
    THREAD_WAIT_EXIT = 5,
    THREAD_WAIT_PRODUCE = 6,
    THREAD_KILL_WAIT_PRODUCE = 7,
};

enum thread_type {
    THREAD_TYPE_BASE = 0,
    THREAD_TYPE_PROC = 1,
};

/* The fixed prefix addressed by tp for every execution context. */
struct thread {
    uint32_t id;
    uint32_t state;
    uintptr_t sp;
    uintptr_t pc;
    uint64_t tensor_context;
    uint32_t type;
    uint8_t profile_slot;
    uint8_t profile_defined;
};

int thread_init(struct thread *thread, uint32_t type);
void switch_context(struct thread *prev, struct thread *next);

/* U-Boot-style global register variable: current has no memory object. */
register struct thread *current __asm__("tp");

#endif
