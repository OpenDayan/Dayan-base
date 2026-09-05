#ifndef NPU_PROFILE_H
#define NPU_PROFILE_H

#define PROFILE_NEXT_OFFSET          0
#define PROFILE_END_OFFSET           8
#define PROFILE_WAKE_OFFSET         32
#define PROFILE_FULL_PENDING_OFFSET 34

#ifndef __ASSEMBLER__

#include <stddef.h>
#include <stdint.h>

#include "api/dayan_list.h"

struct thread;
struct rpc_profile_config;

struct profile_state {
    unsigned char *next;
    unsigned char *end;
    unsigned char *buffer;
    uint64_t generation;
    struct thread_list wake;
    uint8_t full_pending;
} __attribute__((aligned(64)));

_Static_assert(offsetof(struct profile_state, next) == PROFILE_NEXT_OFFSET,
               "assembly profile cursor offset");
_Static_assert(offsetof(struct profile_state, end) == PROFILE_END_OFFSET,
               "assembly profile end offset");
_Static_assert(offsetof(struct profile_state, wake) == PROFILE_WAKE_OFFSET,
               "assembly profile wake offset");
_Static_assert(offsetof(struct profile_state, full_pending) ==
                   PROFILE_FULL_PENDING_OFFSET,
               "assembly profile full flag offset");

void profile_init(void);
int profile_configure(struct rpc_profile_config *config);
void profile_record_thread_start(struct thread *thread);
void profile_record_thread_state(struct thread *thread);

#endif
#endif
