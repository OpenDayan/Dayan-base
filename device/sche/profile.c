#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "profile.h"
#include "resource_scheduler.h"
#include "rpc.h"
#include "sche.h"
#include "string.h"
#include "thread.h"

extern unsigned char __profile_stack_start[];

_Static_assert((uint32_t)THREAD_RUNNING == RPC_PROFILE_PROC_RUNNING,
               "profile running state ABI");
_Static_assert((uint32_t)THREAD_READY == RPC_PROFILE_PROC_READY,
               "profile ready state ABI");
_Static_assert((uint32_t)THREAD_WAIT_POLL == RPC_PROFILE_PROC_WAIT_POLL,
               "profile poll state ABI");
_Static_assert((uint32_t)THREAD_WAIT_EVENT == RPC_PROFILE_PROC_WAIT_EVENT,
               "profile proc state ABI");
_Static_assert((uint32_t)THREAD_WAIT_START == RPC_PROFILE_PROC_WAIT_START,
               "profile start state ABI");
_Static_assert((uint32_t)THREAD_WAIT_EXIT == RPC_PROFILE_PROC_WAIT_EXIT,
               "profile exit state ABI");
_Static_assert((uint32_t)THREAD_WAIT_PRODUCE == RPC_PROFILE_PROC_WAIT_PRODUCE,
               "profile produce state ABI");
_Static_assert((uint32_t)THREAD_KILL_WAIT_PRODUCE ==
                   RPC_PROFILE_PROC_KILL_WAIT_PRODUCE,
               "profile kill-wait-produce state ABI");

struct profile_state {
    unsigned char *buffer;
    uint64_t generation;
    uint32_t capacity;
    uint32_t count;
    struct thread_list wake;
    uint8_t enabled;
    uint8_t full_pending;
    uint8_t defined_generation[RPC_PROFILE_SLOT_COUNT];
} __attribute__((aligned(64)));

static struct profile_state profile_states[CONFIG_HART_COUNT]
    __attribute__((section(".profile_state"), aligned(64)));

static inline struct profile_state *profile_self(void)
{
    return &profile_states[platform_hart_id()];
}

static inline uint64_t profile_counter(void)
{
    uint64_t cycle;

    __asm__ volatile("csrr %0, cycle" : "=r"(cycle));
    return cycle & RPC_PROFILE_COUNTER_MASK;
}

static void profile_queue_record(struct profile_state *state,
                                 uint64_t counter, uint8_t slot,
                                 uint8_t record_state)
{
    uint64_t value = rpc_profile_record_pack(counter, slot, record_state);

    memcpy(state->buffer + state->count, &value, sizeof(value));
    state->count += sizeof(value);
}

static void profile_mark_full(struct profile_state *state)
{
    state->full_pending = 1;
    cpu_self()->profile_enabled = 0;
    (void)thread_event_wake_one(&state->wake);
}

static int profile_reserve(struct profile_state *state, uint32_t required)
{
    if (state->full_pending)
        return 0;
    if (required > state->capacity - state->count) {
        profile_mark_full(state);
        return 0;
    }
    return 1;
}

static void profile_finish_record(struct profile_state *state)
{
    if (state->capacity - state->count <
        RPC_PROFILE_SLOT_DEFINE_SIZE + RPC_PROFILE_RECORD_SIZE)
        profile_mark_full(state);
}

void profile_record_thread_start(struct thread *thread)
{
    struct profile_state *state = profile_self();
    uint8_t slot = thread_slot_id(thread);
    uint64_t counter;

    state->defined_generation[slot] = 0;
    if (!profile_reserve(state, RPC_PROFILE_SLOT_DEFINE_SIZE +
                                RPC_PROFILE_RECORD_SIZE))
        return;
    counter = profile_counter();
    profile_queue_record(state, counter, slot, RPC_PROFILE_RECORD_SLOT_DEFINE);
    memcpy(state->buffer + state->count, thread->name, sizeof(thread->name));
    state->count += sizeof(thread->name);
    profile_queue_record(state, counter, slot, thread->state);
    state->defined_generation[slot] = (uint8_t)(thread->id >>
                                                THREAD_ID_SLOT_BITS);
    profile_finish_record(state);
}

void profile_record_thread_state(struct thread *thread)
{
    struct profile_state *state = profile_self();
    uint8_t slot = thread_slot_id(thread);

    if (state->defined_generation[slot] !=
        (uint8_t)(thread->id >> THREAD_ID_SLOT_BITS)) {
        profile_record_thread_start(thread);
        return;
    }
    if (!profile_reserve(state, RPC_PROFILE_RECORD_SIZE))
        return;
    profile_queue_record(state, profile_counter(), slot,
                         thread->state);
    profile_finish_record(state);
}

static void profile_main(void)
{
    struct profile_state *state = profile_self();

    for (;;) {
        struct rpc_profile_flush_request request;
        struct rpc_profile_flush_response response;
        long status;
        uintptr_t interrupt_state = resource_scheduler_interrupt_disable();

        if (!state->full_pending) {
            thread_prepare_wait(&state->wake);
            resource_scheduler_interrupt_restore(interrupt_state);
            switch_context();
        } else {
            resource_scheduler_interrupt_restore(interrupt_state);
        }
        /* STOP may have consumed the buffer after this thread was woken. */
        if (!state->enabled) {
            state->full_pending = 0;
            continue;
        }
        platform_cache_flush(state->buffer, state->count);
        memset(&request, 0, sizeof(request));
        request.hart_id = (uint32_t)platform_hart_id();
        request.buffer_addr = (uintptr_t)state->buffer;
        request.count = state->count;
        request.capacity = state->capacity;
        request.generation = state->generation;
        memset(&response, 0, sizeof(response));
        response.generation = state->generation;
        response.status = -EIO;
        status = rpc_call(RPC_PROFILE_FLUSH, &request, sizeof(request),
                           &response, sizeof(response));
        state->count = 0;
        state->full_pending = 0;
        if (status < 0 || response.flags || response.status ||
            response.generation != state->generation ||
            response.continue_profile != 1U) {
            state->enabled = 0;
            cpu_self()->profile_enabled = 0;
        } else {
            cpu_self()->profile_enabled = 1;
        }
    }
}

int profile_configure(struct rpc_profile_config *config)
{
    struct profile_state *state = profile_self();
    int status = -EINVAL;

    if (config->flags || config->hart_id != platform_hart_id() ||
        config->reserved)
        return -EINVAL;
    config->count = 0;
    switch (config->command) {
    case RPC_PROFILE_START:
        if (state->buffer) {
            status = -EBUSY;
            break;
        }
        if (!config->generation || (config->buffer_addr & 7U) ||
            (config->capacity & 7U) ||
            config->capacity < RPC_PROFILE_SNAPSHOT_MAX_BYTES ||
            config->buffer_addr < CONFIG_RAM_BASE ||
            config->capacity > UINT64_MAX - config->buffer_addr)
            break;
        state->buffer = (unsigned char *)(uintptr_t)config->buffer_addr;
        state->capacity = config->capacity;
        state->count = 0;
        state->generation = config->generation;
        state->full_pending = 0;
        state->enabled = 1;
        cpu_self()->profile_enabled = 1;
        proc_profile_snapshot();
        status = 0;
        break;
    case RPC_PROFILE_STOP:
        if (!state->buffer || config->generation != state->generation ||
            config->buffer_addr != (uintptr_t)state->buffer ||
            config->capacity != state->capacity)
            break;
        state->enabled = 0;
        cpu_self()->profile_enabled = 0;
        platform_cache_flush(state->buffer, state->count);
        config->count = state->count;
        state->buffer = 0;
        state->capacity = 0;
        state->count = 0;
        state->generation = 0;
        state->full_pending = 0;
        status = 0;
        break;
    }
    config->enabled = state->enabled;
    config->status = status;
    return status;
}

void profile_init(void)
{
    uintptr_t hart = platform_hart_id();
    struct profile_state *state = profile_self();
    void *stack;

    memset(state, 0, sizeof(*state));
    thread_list_init(&state->wake);
    stack = __profile_stack_start + hart * CONFIG_PROFILE_STACK_SIZE;
    thread_start(thread_slot_get(RPC_PROFILE_SLOT_FLUSH), RPC_PROFILE_SLOT_FLUSH,
                  THREAD_TYPE_SYSTEM, "profile-flush", profile_main,
                  stack, CONFIG_PROFILE_STACK_SIZE, 0, 0, 0);
}
