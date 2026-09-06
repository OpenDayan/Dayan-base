#ifndef NPU_THREAD_H
#define NPU_THREAD_H

#include <stdint.h>

#include "api/dayan_list.h"

#define THREAD_NAME_SIZE 16U
#define THREAD_ID_SLOT_BITS 8U
#define THREAD_ID_SLOT_MASK UINT8_MAX

enum thread_state {
    THREAD_RUNNING = 0,
    THREAD_READY = 1,
    THREAD_POLLING = 2,
    /* Compatibility spelling for the pre-Profiler state name. */
    THREAD_WAIT_POLL = THREAD_POLLING,
    THREAD_WAIT_EVENT = 3,
    THREAD_WAIT_START = 4,
    THREAD_WAIT_EXIT = 5,
    THREAD_WAIT_PRODUCE = 6,
    THREAD_KILL_WAIT_PRODUCE = 7,
    THREAD_WAIT_LSU = 8,
    THREAD_WAIT_MAT = 9,
    THREAD_WAIT_VEC = 10,
    THREAD_WAIT_MACRO = 11,
    THREAD_YIELD = 12,
    THREAD_WAIT_JOIN = 13,
    THREAD_WAIT_RESOURCE = 14,
    THREAD_STATE_MAX = THREAD_WAIT_RESOURCE,
};

enum thread_type {
    THREAD_TYPE_SYSTEM = 1,
    THREAD_TYPE_TASK_THREAD = 2,
    THREAD_TYPE_TASK_PROC = 3,
};

/* The fixed prefix addressed by tp for every execution context. */
struct thread {
    /* Little-endian halfword: slot[7:0], state[15:8] for profiling. */
    uint8_t slot;
    uint8_t state;
    uint8_t generation;
    uint8_t type;
    uint8_t parent;
    struct thread_list_node node;
    uintptr_t sp;
    uintptr_t pc;
    uint64_t tensor_context;
    uint32_t resource_context;
    uint8_t token_ext;
    char name[THREAD_NAME_SIZE];
    uint8_t profile_defined;
};

static inline uint16_t thread_id(const struct thread *thread)
{
    return (uint16_t)((thread->generation << THREAD_ID_SLOT_BITS) | thread->slot);
}

void switch_context(void);

/* U-Boot-style global register variable: current has no memory object. */
register struct thread *current __asm__("tp");

#endif
