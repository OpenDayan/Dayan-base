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
    THREAD_WAIT_POLL = 2,
    THREAD_WAIT_EVENT = 3,
    THREAD_WAIT_START = 4,
    THREAD_WAIT_EXIT = 5,
    THREAD_WAIT_PRODUCE = 6,
    THREAD_KILL_WAIT_PRODUCE = 7,
};

enum thread_type {
    THREAD_TYPE_SYSTEM = 1,
    THREAD_TYPE_TASK_THREAD = 2,
    THREAD_TYPE_TASK_PROC = 3,
};

/* The fixed prefix addressed by tp for every execution context. */
struct thread {
    uint8_t type;
    uint8_t state;
    uint16_t id;
    uint8_t parent;
    struct thread_list_node node;
    uintptr_t sp;
    uintptr_t pc;
    uint64_t tensor_context;
    uint32_t resource_context;
    uint8_t token_ext;
    uint8_t reap_deferred;
    char name[THREAD_NAME_SIZE];
};

void switch_context(void);

/* U-Boot-style global register variable: current has no memory object. */
register struct thread *current __asm__("tp");

#endif
