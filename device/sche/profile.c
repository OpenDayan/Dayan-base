#include <stddef.h>
#include <stdint.h>

#include "config.h"
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
               "profile event state ABI");
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
    struct proc worker;
    struct list_head wake;
    struct list_head flush_done;
    struct rpc_profile_event *buffer;
    uint32_t capacity;
    uint32_t count;
    uint32_t enabled;
    uint32_t full_pending;
    uint32_t flushing;
    uint32_t reserved;
    uint64_t generation;
};

static struct profile_state profile_states[CONFIG_HART_COUNT]
    __attribute__((section(".profile_state"), aligned(64)));

static struct profile_state *profile_self(void)
{
    uintptr_t hart = platform_hart_id();

    return hart < CONFIG_HART_COUNT ? &profile_states[hart] : 0;
}

static uint32_t profile_cycle(void)
{
    uintptr_t cycle;

    __asm__ volatile("csrr %0, cycle" : "=r"(cycle));
    return (uint32_t)cycle;
}

static void profile_queue_event(struct profile_state *state,
                                const struct proc *proc)
{
    struct rpc_profile_event *event = &state->buffer[state->count++];

    event->cycle = profile_cycle();
    event->event_id = (uint16_t)proc->thread.id;
    event->state = (uint16_t)proc->thread.state;
}

void profile_record_proc_state(const struct proc *proc)
{
    struct profile_state *state = profile_self();

    if (!state || !proc || proc->thread.type != THREAD_TYPE_PROC ||
        proc->thread.state > THREAD_KILL_WAIT_PRODUCE || !state->enabled ||
        state->full_pending || state->flushing || !state->buffer)
        return;
    if (state->count >= state->capacity) {
        state->full_pending = 1;
        (void)proc_event_wake_one(&state->wake);
        return;
    }
    profile_queue_event(state, proc);
    if (state->count == state->capacity) {
        state->full_pending = 1;
        (void)proc_event_wake_one(&state->wake);
    }
}

static int profile_response_valid(
    const struct profile_state *state,
    const struct rpc_profile_flush_response *response)
{
    return response->version == RPC_PROFILE_VERSION &&
           response->size == sizeof(*response) && !response->flags &&
           response->status == 0 &&
           response->generation == state->generation &&
           response->continue_profile <= 1U;
}

static void profile_main(void)
{
    struct profile_state *state = profile_self();

    if (!state)
        platform_halt();
    for (;;) {
        struct rpc_profile_flush_request request;
        struct rpc_profile_flush_response response;
        long status;

        while (!state->full_pending) {
            if (proc_wait_event(&state->wake) < 0)
                platform_halt();
        }
        if (!state->enabled || !state->buffer || !state->count) {
            state->full_pending = 0;
            continue;
        }
        state->flushing = 1;
        platform_cache_clean(state->buffer,
                             (size_t)state->count * sizeof(*state->buffer));
        memset(&request, 0, sizeof(request));
        request.version = RPC_PROFILE_VERSION;
        request.size = sizeof(request);
        request.hart_id = (uint32_t)platform_hart_id();
        request.buffer_addr = (uintptr_t)state->buffer;
        request.count = state->count;
        request.capacity = state->capacity;
        request.generation = state->generation;
        memset(&response, 0, sizeof(response));
        response.version = RPC_PROFILE_VERSION;
        response.size = sizeof(response);
        response.generation = state->generation;
        response.status = -EIO;
        status = rpc_profile_flush(&request, &response);
        state->count = 0;
        state->full_pending = 0;
        state->flushing = 0;
        if (status < 0 || !profile_response_valid(state, &response) ||
            !response.continue_profile)
            state->enabled = 0;
        (void)proc_event_wake_all(&state->flush_done);
    }
}

static int profile_buffer_valid(uint64_t address, uint32_t capacity)
{
    uint64_t bytes = (uint64_t)capacity * sizeof(struct rpc_profile_event);

    return !(address & 7U) && capacity >= RPC_PROC_PS_MAX_RECORDS &&
           bytes <= CONFIG_RAM_SIZE && address >= CONFIG_RAM_BASE &&
           address - CONFIG_RAM_BASE <= CONFIG_RAM_SIZE - bytes;
}

static int profile_start(struct profile_state *state,
                         struct rpc_profile_config *config)
{
    if (state->buffer || state->flushing || !config->generation ||
        !profile_buffer_valid(config->buffer_addr, config->capacity))
        return state->buffer || state->flushing ? -EBUSY : -EINVAL;
    state->buffer = (struct rpc_profile_event *)(uintptr_t)config->buffer_addr;
    state->capacity = config->capacity;
    state->count = 0;
    state->generation = config->generation;
    state->full_pending = 0;
    state->enabled = 1;
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
    if (!state->flushing)
        state->full_pending = 0;
    while (state->flushing) {
        if (proc_wait_event(&state->flush_done) < 0)
            platform_halt();
    }
    if (state->count)
        platform_cache_clean(state->buffer,
                             (size_t)state->count * sizeof(*state->buffer));
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

    if (!state || !config || config->version != RPC_PROFILE_VERSION ||
        config->size != sizeof(*config) || config->flags ||
        config->hart_id != platform_hart_id() || config->reserved)
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

struct proc *profile_proc(void)
{
    struct profile_state *state = profile_self();

    return state ? &state->worker : 0;
}

int profile_init(void)
{
    uintptr_t hart = platform_hart_id();
    struct profile_state *state = profile_self();
    void *stack;

    if (!state)
        return -EINVAL;
    memset(state, 0, sizeof(*state));
    INIT_LIST_HEAD(&state->wake);
    INIT_LIST_HEAD(&state->flush_done);
    stack = __profile_stack_start + hart * CONFIG_PROFILE_STACK_SIZE;
    return proc_init(&state->worker, profile_main,
                     stack, CONFIG_PROFILE_STACK_SIZE, 0, 0, 0, 0);
}
