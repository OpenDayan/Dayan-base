/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "cpu.h"
#include "profile.h"
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

uint8_t thread_slot_id(const struct thread *target)
{
    return target->slot;
}

size_t thread_event_wake_one(struct thread_list *event)
{
    unsigned char *next = profiler.next;

    assert(event == &profiler.wake);
    assert(!cpu_self()->profiler && profiler.full_pending);
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
    thread.type = THREAD_TYPE_TASK_PROC;
    cpu_self()->threads.slot_owner[0] = 1;
    thread.state = THREAD_READY;
    assert(thread_id(&thread) == 0x101);
    memcpy(thread.name, "first", sizeof("first"));

    profile_record_thread_start(&thread);
    assert(profiler.next == base && !thread.profile_defined);

    cpu_self()->profiler = &profiler;
    profile_record_thread_start(&thread);
    assert(thread.profile_defined && profiler.next == base + 32);
    assert(rpc_profile_record_slot(buffer.records[0]) == 1);
    assert(rpc_profile_record_state(buffer.records[0]) ==
           (RPC_PROFILE_RECORD_SLOT_DEFINE | 1));
    assert(!memcmp(&buffer.records[1], thread.name, sizeof(thread.name)));
    assert(rpc_profile_record_slot(buffer.records[3]) == 1);
    assert(rpc_profile_record_state(buffer.records[3]) == THREAD_READY);
    assert(rpc_profile_record_counter(buffer.records[0]) <=
           rpc_profile_record_counter(buffer.records[3]));

    thread.state = THREAD_RUNNING;
    profile_record_thread_state(&thread);
    assert(profiler.next == base + 40);
    assert(rpc_profile_record_state(buffer.records[4]) == THREAD_RUNNING);
    assert(rpc_profile_record_slot(buffer.records[4]) == 1);

    /* Slot reuse gets a new definition without consulting a generation table. */
    memset(&thread, 0, sizeof(thread));
    thread.slot = 1;
    thread.generation = 2;
    thread.type = THREAD_TYPE_TASK_PROC;
    thread.state = THREAD_WAIT_START;
    assert(thread_id(&thread) == 0x201);
    memcpy(thread.name, "reused", sizeof("reused"));
    profile_record_thread_start(&thread);
    assert(profiler.next == base + 72);
    assert(rpc_profile_record_state(buffer.records[5]) ==
           (RPC_PROFILE_RECORD_SLOT_DEFINE | 1));
    assert(!memcmp(&buffer.records[6], thread.name, sizeof(thread.name)));
    assert(rpc_profile_record_state(buffer.records[8]) == THREAD_WAIT_START);

    thread.state = THREAD_WAIT_EVENT;
    while (profiler.next != base + 96)
        profile_record_thread_state(&thread);
    assert(cpu_self()->profiler == &profiler && !wakes);
    profile_record_thread_state(&thread);
    assert(wakes == 1);
    assert(!cpu_self()->profiler && profiler.next == base + 104);
    assert(rpc_profile_record_state(buffer.records[12]) == THREAD_WAIT_EVENT);
    for (unsigned i = 13; i < 16; ++i)
        assert(buffer.records[i] == untouched);
    assert(buffer.guard == untouched);
    profile_record_thread_state(&thread);
    assert(wakes == 1 && profiler.next == base + 104);

    /* Resume keeps definitions; a new session resets them in its snapshot. */
    profiler.next = base;
    profiler.full_pending = 0;
    cpu_self()->profiler = &profiler;
    profile_record_thread_state(&thread);
    assert(profiler.next == base + 8);
    thread.profile_defined = 0;
    profile_record_thread_start(&thread);
    assert(profiler.next == base + 40);
    assert(rpc_profile_record_state(buffer.records[1]) ==
           (RPC_PROFILE_RECORD_SLOT_DEFINE | 1));
    assert(!memcmp(&buffer.records[2], thread.name, sizeof(thread.name)));
    profiler.next = profiler.end - 32;
    profile_record_thread_start(&thread);
    assert(profiler.next == profiler.end && !cpu_self()->profiler);
    assert(wakes == 2);
    assert(buffer.guard == untouched);

    /* A nested child records its family root, not its immediate parent. */
    profiler.next = base;
    profiler.full_pending = 0;
    cpu_self()->profiler = &profiler;
    thread.slot = 7;
    thread.parent = 4;
    thread.type = THREAD_TYPE_TASK_THREAD;
    cpu_self()->threads.slot_owner[6] = 1;
    profile_record_thread_start(&thread);
    assert(profiler.next == base + 32);
    assert(rpc_profile_record_slot(buffer.records[0]) == 7);
    assert(rpc_profile_record_state(buffer.records[0]) ==
           (RPC_PROFILE_RECORD_SLOT_DEFINE | 1));
    thread.slot = RPC_PROFILE_SLOT_RPC_RECEIVE;
    thread.type = THREAD_TYPE_SYSTEM;
    profile_record_thread_start(&thread);
    assert(profiler.next == base + 64);
    assert(rpc_profile_record_state(buffer.records[4]) ==
           RPC_PROFILE_RECORD_SLOT_DEFINE);
    assert(buffer.guard == untouched);
    return 0;
}
