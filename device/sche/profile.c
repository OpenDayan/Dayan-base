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

_Static_assert(RPC_PROFILE_RECORD_SIZE == 8 &&
                   RPC_PROFILE_SLOT_DEFINE_SIZE == 24 &&
                   RPC_PROFILE_COUNTER_BITS == 48 &&
                   RPC_PROFILE_SLOT_SHIFT == 48 &&
                   RPC_PROFILE_STATE_SHIFT == 56 &&
                   RPC_PROFILE_RECORD_SLOT_DEFINE == 0x80,
               "assembly profile record encoding");

void profile_record_thread_start(struct thread *thread)
{
    struct profile_state *state = cpu_self()->profiler;
    uint64_t cycle;

    if (state) {
        uint8_t slot = thread_slot_id(thread);
        uint8_t root = thread->type == THREAD_TYPE_SYSTEM ? 0 :
            cpu_self()->threads.slot_owner[slot - RPC_PROFILE_SLOT_USER_BASE];

        __asm__ volatile("rdcycle %0" : "=r"(cycle));
        *(uint64_t *)state->next = rpc_profile_record_pack(
            cycle, slot, RPC_PROFILE_RECORD_SLOT_DEFINE | root);
        memcpy(state->next + RPC_PROFILE_RECORD_SIZE,
               thread->name, sizeof(thread->name));
        state->next += RPC_PROFILE_SLOT_DEFINE_SIZE;
        thread->profile_defined = 1;
        profile_record_thread_state(thread);
    }
}

static void profile_main(void)
{
    struct profile_state *state = &cpu_self()->profile;

    for (;;) {
        struct rpc_profile_flush_request request;
        struct rpc_profile_flush_response response;
        long status;

        if (!state->full_pending) {
            thread_wait_event(&state->wake);
            continue;
        }
        platform_cache_flush(state->buffer, state->next - state->buffer);
        memset(&request, 0, sizeof(request));
        request.hart_id = (uint32_t)platform_hart_id();
        request.buffer_addr = (uintptr_t)state->buffer;
        request.count = state->next - state->buffer;
        request.capacity = state->end - state->buffer;
        request.generation = state->generation;
        memset(&response, 0, sizeof(response));
        response.generation = state->generation;
        response.status = -EIO;
        status = rpc_call(RPC_PROFILE_FLUSH, &request, sizeof(request),
                           &response, sizeof(response));
        state->next = state->buffer;
        state->full_pending = 0;
        if (status >= 0 && !response.flags && !response.status &&
            response.generation == state->generation &&
            response.continue_profile == 1U) {
            cpu_self()->profiler = state;
            proc_profile_snapshot(0);
        }
    }
}

int profile_configure(struct rpc_profile_config *config)
{
    struct profile_state *state = &cpu_self()->profile;
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
        if (!config->generation ||
            (config->buffer_addr & (CONFIG_CACHE_LINE_SIZE - 1U)) ||
            (config->capacity & (CONFIG_CACHE_LINE_SIZE - 1U)) ||
            config->capacity < RPC_PROFILE_SNAPSHOT_MAX_BYTES ||
            config->buffer_addr < CONFIG_RAM_BASE ||
            config->capacity > UINT64_MAX - config->buffer_addr)
            break;
        state->buffer = (unsigned char *)(uintptr_t)config->buffer_addr;
        state->next = state->buffer;
        state->end = state->buffer + config->capacity;
        state->generation = config->generation;
        state->full_pending = 0;
        cpu_self()->profiler = state;
        proc_profile_snapshot(1);
        status = 0;
        break;
    case RPC_PROFILE_STOP:
        if (!state->buffer || config->generation != state->generation ||
            config->buffer_addr != (uintptr_t)state->buffer ||
            config->capacity != (uintptr_t)(state->end - state->buffer))
            break;
        cpu_self()->profiler = 0;
        platform_cache_flush(state->buffer, state->next - state->buffer);
        config->count = state->next - state->buffer;
        state->buffer = 0;
        state->next = 0;
        state->end = 0;
        state->generation = 0;
        state->full_pending = 0;
        status = 0;
        break;
    }
    config->enabled = cpu_self()->profiler != 0 || state->full_pending;
    config->status = status;
    return status;
}

void profile_init(void)
{
    uintptr_t hart = platform_hart_id();
    void *stack;

    thread_list_init(&cpu_self()->profile.wake);
    stack = __profile_stack_start + hart * CONFIG_PROFILE_STACK_SIZE;
    thread_start(thread_slot_get(RPC_PROFILE_SLOT_FLUSH), RPC_PROFILE_SLOT_FLUSH,
                  THREAD_TYPE_SYSTEM, "profile-flush", profile_main,
                  stack, CONFIG_PROFILE_STACK_SIZE, 0, 0, 0);
}
