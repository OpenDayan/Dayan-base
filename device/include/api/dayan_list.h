#ifndef DAYAN_LIST_H
#define DAYAN_LIST_H

#include <stdint.h>

/* Used by internal pointer-based queues such as RPC transactions. */
struct list_head {
    struct list_head *next;
    struct list_head *prev;
};

#define THREAD_SLOT_NONE     UINT8_MAX
#define THREAD_SLOT_UNLINKED (UINT8_MAX - 1U)

struct thread_list_node {
    uint8_t prev;
    uint8_t next;
};

struct thread_list {
    uint8_t head;
    uint8_t tail;
};

#endif
