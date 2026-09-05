#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "cpu.h"
#include "profile.h"
#include "resource_scheduler.h"
#include "rpc.h"

_Alignas(64) uint8_t __cpu_start[CONFIG_CPU_LOCAL_SIZE];
static struct profile_state profiler;
static struct thread thread;
static unsigned wakes;
static struct {
    _Alignas(64) uint64_t records[16];
    uint64_t guard;
} buffer;

_Static_assert(sizeof(struct thread) == 56, "profiling must not grow threads");

static uintptr_t interrupt_enabled(void)
{
    uintptr_t status;

    __asm__ volatile("csrr %0, mstatus" : "=r"(status));
    return status & 8U;
}

uint8_t thread_slot_id(const struct thread *target)
{
    return target->slot;
}

size_t thread_event_wake_one(struct thread_list *event)
{
    unsigned char *next = profiler.next;

    assert(event == &profiler.wake);
    assert(!cpu_self()->profiler && profiler.full_pending);
    assert(!interrupt_enabled());
    ++wakes;
    /* Waking a worker can itself record READY; the full buffer is disabled. */
    profile_record_thread_state(&thread);
    assert(profiler.next == next);
    return 1;
}

int main(void)
{
    const uint64_t untouched = 0xa5a5a5a5a5a5a5a5ULL;
    unsigned char *base = (unsigned char *)buffer.records;

    memset(&buffer, 0xa5, sizeof(buffer));
    profiler.buffer = profiler.next = base;
    profiler.end = base + sizeof(buffer.records);
    thread.slot = 1;
    thread.generation = 1;
    thread.type = THREAD_TYPE_TASK_THREAD;
    thread.state = THREAD_READY;
    assert(thread_id(&thread) == 0x101);
    memcpy(thread.name, "first", sizeof("first"));

    profile_record_thread_start(&thread);
    assert(profiler.next == base && !thread.profile_defined);
    assert(!interrupt_enabled());

    cpu_self()->profiler = &profiler;
    resource_scheduler_interrupt_restore(8);
    profile_record_thread_start(&thread);
    assert(interrupt_enabled());
    assert(thread.profile_defined && profiler.next == base + 32);
    assert(rpc_profile_record_slot(buffer.records[0]) == 1);
    assert(rpc_profile_record_state(buffer.records[0]) ==
           RPC_PROFILE_RECORD_SLOT_DEFINE);
    assert(!memcmp(&buffer.records[1], thread.name, sizeof(thread.name)));
    assert(rpc_profile_record_slot(buffer.records[3]) == 1);
    assert(rpc_profile_record_state(buffer.records[3]) == THREAD_READY);
    assert(rpc_profile_record_counter(buffer.records[0]) <=
           rpc_profile_record_counter(buffer.records[3]));

    thread.state = THREAD_RUNNING;
    profile_record_thread_state(&thread);
    assert(interrupt_enabled() && profiler.next == base + 40);
    assert(rpc_profile_record_state(buffer.records[4]) == THREAD_RUNNING);
    assert(rpc_profile_record_slot(buffer.records[4]) == 1);

    /* Slot reuse gets a new definition without consulting a generation table. */
    memset(&thread, 0, sizeof(thread));
    thread.slot = 1;
    thread.generation = 2;
    thread.state = THREAD_WAIT_START;
    assert(thread_id(&thread) == 0x201);
    memcpy(thread.name, "reused", sizeof("reused"));
    profile_record_thread_start(&thread);
    assert(profiler.next == base + 72);
    assert(rpc_profile_record_state(buffer.records[5]) ==
           RPC_PROFILE_RECORD_SLOT_DEFINE);
    assert(!memcmp(&buffer.records[6], thread.name, sizeof(thread.name)));
    assert(rpc_profile_record_state(buffer.records[8]) == THREAD_WAIT_START);

    thread.state = THREAD_WAIT_EVENT;
    while (profiler.next != base + 96)
        profile_record_thread_state(&thread);
    assert(cpu_self()->profiler == &profiler && !wakes);
    profile_record_thread_state(&thread);
    assert(interrupt_enabled() && wakes == 1);
    assert(!cpu_self()->profiler && profiler.next == base + 104);
    assert(rpc_profile_record_state(buffer.records[12]) == THREAD_WAIT_EVENT);
    for (unsigned i = 13; i < 16; ++i)
        assert(buffer.records[i] == untouched);
    assert(buffer.guard == untouched);
    profile_record_thread_state(&thread);
    assert(interrupt_enabled() && wakes == 1 && profiler.next == base + 104);

    /* Resume keeps definitions; a new session resets them in its snapshot. */
    (void)resource_scheduler_interrupt_disable();
    profiler.next = base;
    profiler.full_pending = 0;
    cpu_self()->profiler = &profiler;
    profile_record_thread_state(&thread);
    assert(!interrupt_enabled() && profiler.next == base + 8);
    thread.profile_defined = 0;
    profile_record_thread_start(&thread);
    assert(profiler.next == base + 40 && !interrupt_enabled());
    assert(rpc_profile_record_state(buffer.records[1]) ==
           RPC_PROFILE_RECORD_SLOT_DEFINE);
    assert(!memcmp(&buffer.records[2], thread.name, sizeof(thread.name)));
    profiler.next = profiler.end - 32;
    profile_record_thread_start(&thread);
    assert(profiler.next == profiler.end && !cpu_self()->profiler);
    assert(wakes == 2 && !interrupt_enabled());
    assert(buffer.guard == untouched);
    return 0;
}
