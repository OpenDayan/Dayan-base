#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "profile.h"
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
    struct thread_list wake;
    struct thread_list flush_done;
    unsigned char *buffer;
    uint32_t capacity;
    uint32_t count;
    uint32_t enabled;
    uint32_t full_pending;
    uint32_t flushing;
    uint64_t generation;
    uint16_t defined_id[RPC_PROFILE_SLOT_COUNT];
};

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

static void profile_queue_slot_define(struct profile_state *state,
                                      const struct thread *thread,
                                      uint64_t counter)
{
    uint8_t slot = thread_slot_id(thread);

    profile_queue_record(state, counter, slot,
                         RPC_PROFILE_RECORD_SLOT_DEFINE);
    memcpy(state->buffer + state->count, thread->name, sizeof(thread->name));
    state->count += sizeof(thread->name);
}

static void profile_mark_full(struct profile_state *state)
{
    state->full_pending = 1;
    cpu_self()->sche.profile_enabled = 0;
    (void)thread_event_wake_one(&state->wake);
}

static uint8_t profile_thread_slot(const struct thread *thread)
{
    uint8_t slot = thread_slot_id(thread);

    if (!slot || slot >= RPC_PROFILE_SLOT_COUNT || !thread->id ||
        (thread->id & THREAD_ID_SLOT_MASK) != slot ||
        thread->state > RPC_PROFILE_PROC_KILL_WAIT_PRODUCE)
        platform_halt();
    return slot;
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
    uint8_t slot = profile_thread_slot(thread);
    uint64_t counter;

    state->defined_id[slot] = 0;
    if (!profile_reserve(state, RPC_PROFILE_SLOT_DEFINE_SIZE +
                                RPC_PROFILE_RECORD_SIZE))
        return;
    counter = profile_counter();
    profile_queue_slot_define(state, thread, counter);
    profile_queue_record(state, counter, slot, thread->state);
    state->defined_id[slot] = thread->id;
    profile_finish_record(state);
}

void profile_record_thread_state(struct thread *thread)
{
    struct profile_state *state = profile_self();
    uint8_t slot = profile_thread_slot(thread);

    if (state->defined_id[slot] != thread->id) {
        profile_record_thread_start(thread);
        return;
    }
    if (!profile_reserve(state, RPC_PROFILE_RECORD_SIZE))
        return;
    profile_queue_record(state, profile_counter(), slot,
                         thread->state);
    profile_finish_record(state);
}

static int profile_response_valid(
    const struct profile_state *state,
    const struct rpc_profile_flush_response *response)
{
    return !response->flags &&
           response->status == 0 &&
           response->generation == state->generation &&
           response->continue_profile <= 1U;
}

static void profile_main(void)
{
    struct profile_state *state = profile_self();

    for (;;) {
        struct rpc_profile_flush_request request;
        struct rpc_profile_flush_response response;
        long status;

        while (!state->full_pending) {
            thread_wait_event(&state->wake);
        }
        if (!state->enabled || !state->buffer || !state->count) {
            state->full_pending = 0;
            continue;
        }
        state->flushing = 1;
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
        status = rpc_profile_flush(&request, &response);
        state->count = 0;
        state->full_pending = 0;
        state->flushing = 0;
        if (!state->enabled || status < 0 ||
            !profile_response_valid(state, &response) ||
            !response.continue_profile) {
            state->enabled = 0;
            cpu_self()->sche.profile_enabled = 0;
        } else {
            cpu_self()->sche.profile_enabled = 1;
        }
        (void)thread_event_wake_all(&state->flush_done);
    }
}

static int profile_buffer_valid(uint64_t address, uint32_t capacity)
{
    return !(address & 7U) && !(capacity & 7U) &&
           capacity >= RPC_PROFILE_SNAPSHOT_MAX_BYTES &&
           address >= CONFIG_RAM_BASE && capacity <= UINT64_MAX - address;
}

static int profile_start(struct profile_state *state,
                         struct rpc_profile_config *config)
{
    if (state->buffer || state->flushing || !config->generation ||
        !profile_buffer_valid(config->buffer_addr, config->capacity))
        return state->buffer || state->flushing ? -EBUSY : -EINVAL;
    state->buffer = (unsigned char *)(uintptr_t)config->buffer_addr;
    state->capacity = config->capacity;
    state->count = 0;
    state->generation = config->generation;
    state->full_pending = 0;
    state->enabled = 1;
    cpu_self()->sche.profile_enabled = 1;
    proc_profile_snapshot();
    return 0;
}

static int profile_stop(struct profile_state *state,
                        struct rpc_profile_config *config)
{
    if (!state->buffer || config->generation != state->generation ||
        config->buffer_addr != (uintptr_t)state->buffer ||
        config->capacity != state->capacity)
        return -EINVAL;
    state->enabled = 0;
    cpu_self()->sche.profile_enabled = 0;
    if (!state->flushing)
        state->full_pending = 0;
    while (state->flushing) {
        thread_wait_event(&state->flush_done);
    }
    if (state->count)
        platform_cache_flush(state->buffer, state->count);
    config->count = state->count;
    state->buffer = 0;
    state->capacity = 0;
    state->count = 0;
    state->generation = 0;
    state->full_pending = 0;
    return 0;
}

int profile_configure(struct rpc_profile_config *config)
{
    struct profile_state *state = profile_self();
    int status;

    if (!config || config->flags || config->hart_id != platform_hart_id() ||
        config->reserved)
        return -EINVAL;
    config->count = 0;
    if (config->command == RPC_PROFILE_START)
        status = profile_start(state, config);
    else if (config->command == RPC_PROFILE_STOP)
        status = profile_stop(state, config);
    else
        status = -EINVAL;
    config->enabled = state->enabled;
    config->status = status;
    return status;
}

struct thread *profile_thread(void)
{
    return thread_slot_get(RPC_PROFILE_SLOT_FLUSH);
}

int profile_init(void)
{
    uintptr_t hart = platform_hart_id();
    struct profile_state *state = profile_self();
    void *stack;

    memset(state, 0, sizeof(*state));
    thread_list_init(&state->wake);
    thread_list_init(&state->flush_done);
    stack = __profile_stack_start + hart * CONFIG_PROFILE_STACK_SIZE;
    return thread_start(profile_thread(), RPC_PROFILE_SLOT_FLUSH,
                        THREAD_TYPE_SYSTEM, "profile-flush", profile_main,
                        stack, CONFIG_PROFILE_STACK_SIZE, 0, 0, 0, 0, 0);
}
