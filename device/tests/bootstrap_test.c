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
#include "errno.h"
#include "rpc.h"
#include "sche.h"

_Alignas(64) uint8_t __cpu_start[CONFIG_CPU_LOCAL_SIZE];
static _Alignas(16) unsigned char stack[1024];
static uintptr_t observed[32];
static uintptr_t observed_sp, observed_gp;
static _Alignas(16) unsigned char proc_stacks[CONFIG_PROC_SLOT_COUNT][CONFIG_PROC_STACK_SIZE];

extern void run_bootstrap(uintptr_t sp, uintptr_t pc);

/* Exit RPCs are unreachable: the trampoline supplies a test return address. */
long rpc_write_proc(uint32_t proc_id, const void *data, size_t size)
{
    (void)proc_id; (void)data; (void)size;
    assert(0);
    return -1;
}

long rpc_proc_exit(struct task_proc *task, int64_t result)
{
    (void)task; (void)result;
    assert(0);
    return -1;
}

static void entry0(void)
{
    __asm__ volatile("mv %0, sp\n\tmv %1, gp"
                     : "=r"(observed_sp), "=r"(observed_gp));
}

static void entry1(uintptr_t a0)
{
    __asm__ volatile("mv %0, sp\n\tmv %1, gp"
                     : "=r"(observed_sp), "=r"(observed_gp));
    observed[0] = a0;
}

static void entry8(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7)
{
    __asm__ volatile("mv %0, sp\n\tmv %1, gp"
                     : "=r"(observed_sp), "=r"(observed_gp));
    observed[0] = a0;
    observed[1] = a1;
    observed[2] = a2;
    observed[3] = a3;
    observed[4] = a4;
    observed[5] = a5;
    observed[6] = a6;
    observed[7] = a7;
}

static void entry9(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7, uintptr_t a8)
{
    __asm__ volatile("mv %0, sp\n\tmv %1, gp"
                     : "=r"(observed_sp), "=r"(observed_gp));
    observed[0] = a0;
    observed[1] = a1;
    observed[2] = a2;
    observed[3] = a3;
    observed[4] = a4;
    observed[5] = a5;
    observed[6] = a6;
    observed[7] = a7;
    observed[8] = a8;
}

static void entry32(uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7, uintptr_t a8, uintptr_t a9, uintptr_t a10, uintptr_t a11, uintptr_t a12, uintptr_t a13, uintptr_t a14, uintptr_t a15, uintptr_t a16, uintptr_t a17, uintptr_t a18, uintptr_t a19, uintptr_t a20, uintptr_t a21, uintptr_t a22, uintptr_t a23, uintptr_t a24, uintptr_t a25, uintptr_t a26, uintptr_t a27, uintptr_t a28, uintptr_t a29, uintptr_t a30, uintptr_t a31)
{
    __asm__ volatile("mv %0, sp\n\tmv %1, gp"
                     : "=r"(observed_sp), "=r"(observed_gp));
    observed[0] = a0;
    observed[1] = a1;
    observed[2] = a2;
    observed[3] = a3;
    observed[4] = a4;
    observed[5] = a5;
    observed[6] = a6;
    observed[7] = a7;
    observed[8] = a8;
    observed[9] = a9;
    observed[10] = a10;
    observed[11] = a11;
    observed[12] = a12;
    observed[13] = a13;
    observed[14] = a14;
    observed[15] = a15;
    observed[16] = a16;
    observed[17] = a17;
    observed[18] = a18;
    observed[19] = a19;
    observed[20] = a20;
    observed[21] = a21;
    observed[22] = a22;
    observed[23] = a23;
    observed[24] = a24;
    observed[25] = a25;
    observed[26] = a26;
    observed[27] = a27;
    observed[28] = a28;
    observed[29] = a29;
    observed[30] = a30;
    observed[31] = a31;
}

static void test_proc_create(void)
{
    struct rpc_proc_create_request request = {0};
    struct thread_runtime *runtime = &cpu_self()->threads;

    request.entry = (uintptr_t)entry32;
    request.cookie = 0xfedcba9876543210ULL;
    request.shared_offset = 0x100;
    request.shared_size = 0x200;
    request.task_mode = 1;
    request.printf_size = 17;
    request.max_child_threads = 3;
    request.tensor_context = 0x123456789abcdef0ULL;
    request.resource_context = 3U << 16; /* Shared resources; no local HW init. */
    strcpy(request.name, "0123456789abcde");
    for (unsigned axis = 0; axis < 3; ++axis) {
        request.task_id[axis] = 0x87654320U + axis;
        request.task_dim[axis] = 0xfedcba90U + axis;
    }
    for (unsigned argc = 0; argc <= RPC_PROC_MAX_ARGS; ++argc) {
        size_t stack_size = argc & 1 ? CONFIG_PROC_STACK_SIZE : 1024;
        request.stack_size_m1 = stack_size - 1;
        request.argc = argc;
        for (unsigned i = 0; i < RPC_PROC_MAX_ARGS; ++i)
            request.args[i] = i < argc ? 0xfedcba9876543200ULL + i : 0;
        memset(runtime, 0, sizeof(*runtime));
        memset(proc_stacks[0], 0xa5, sizeof(proc_stacks[0]));
        runtime->stacks = proc_stacks;
        struct task_proc *task = task_proc_create(&request);
        assert(task && task->descriptor.last_slot == 4);
        assert(task->descriptor.cookie == request.cookie);
        assert(task->descriptor.shared == request.shared);
        assert(task->descriptor.launch == (request.launch | (4ULL << 56)));
        assert(!memcmp(task->descriptor.task_words, request.task_words, 24));
        assert(!memcmp(task->thread.name, request.name, 16));
        assert(task->thread.tensor_context == request.tensor_context);
        assert(task->thread.resource_state == request.resource_context);
        assert(task->thread.parent == THREAD_SLOT_NONE);
        assert(!task->thread.profile_defined && !task->thread.node.next && !task->thread.node.prev);
        struct proc_frame *frame = (struct proc_frame *)task->thread.sp;
        uintptr_t *stack_args = (uintptr_t *)(frame + 1);
        assert(!(task->thread.sp & 15));
        assert(frame->entry == request.entry && !frame->reserved);
        assert(frame->shared == CONFIG_COHERENT_BASE + request.shared_offset);
        for (unsigned i = 0; i < argc; ++i)
            assert((i < 8 ? frame->args[i] : stack_args[i - 8]) == request.args[i]);
        assert(proc_stacks[0][0] == 0xa5);
        if (stack_size < CONFIG_PROC_STACK_SIZE)
            assert(proc_stacks[0][stack_size] == 0xa5);
        if (argc == 32) {
            memset(observed, 0, sizeof(observed));
            run_bootstrap(task->thread.sp, task->thread.pc);
            for (unsigned i = 0; i < argc; ++i)
                assert(observed[i] == request.args[i]);
        }
    }
}

static void test_proc_name(void)
{
    struct thread *root = thread_slot_get(1);
    struct thread child = {
        .slot = 2, .type = THREAD_TYPE_TASK_THREAD, .parent = 1,
    };
    uint64_t records[16];
    struct profile_state profiler = {
        .buffer = (unsigned char *)records,
        .next = (unsigned char *)records,
        .end = (unsigned char *)(records + 16),
    };
    uint16_t id = thread_id(root);

    thread_list_init(&profiler.wake);
    current = 0;
    assert(thread_set_name("mega") == -EPERM);
    current = thread_slot_get(RPC_PROFILE_SLOT_RPC_RECEIVE);
    current->type = THREAD_TYPE_SYSTEM;
    assert(thread_set_name("mega") == -EPERM);
    current = root;
    root->state = THREAD_RUNNING;
    assert(!thread_set_name("0123456789abcde"));
    assert(!strcmp(root->name, "0123456789abcde"));

    cpu_self()->profiler = &profiler;
    assert(!thread_set_name("mega-load"));
    assert(!strcmp(root->name, "mega-load") && !root->name[15]);
    assert(root->profile_defined && profiler.next == profiler.buffer + 32);
    assert(thread_set_name(0) == -EINVAL);
    assert(thread_set_name("") == -EINVAL);
    assert(thread_set_name("0123456789abcdef") == -ENAMETOOLONG);
    assert(!strcmp(root->name, "mega-load") && root->profile_defined);
    assert(profiler.next == profiler.buffer + 32);
    assert(!thread_set_name("mega-mat"));
    assert(thread_id(root) == id && root->state == THREAD_RUNNING);
    for (unsigned i = 0; i < 8; i += 4) {
        assert(rpc_profile_record_slot(records[i]) == 1);
        assert(rpc_profile_record_state(records[i]) ==
               (RPC_PROFILE_RECORD_SLOT_DEFINE | 1));
        assert(rpc_profile_record_state(records[i + 3]) == THREAD_RUNNING);
    }
    assert(!strcmp((char *)&records[1], "mega-load"));
    assert(!strcmp((char *)&records[5], "mega-mat"));

    /* A buffer flush resumes with a snapshot of names changed while paused. */
    cpu_self()->profiler = 0;
    assert(!thread_set_name("mega-store") && !root->profile_defined);
    assert(profiler.next == profiler.buffer + 64);
    for (uint8_t slot = RPC_PROFILE_SLOT_RPC_RECEIVE;
         slot < RPC_PROFILE_SLOT_COUNT; ++slot)
        thread_slot_get(slot)->profile_defined = 1;
    cpu_self()->profiler = &profiler;
    proc_profile_snapshot(0);
    assert(root->profile_defined && profiler.next == profiler.buffer + 96);
    assert(!strcmp((char *)&records[9], "mega-store"));

    current = &child;
    assert(!thread_set_name("child-stage"));
    assert(!strcmp(child.name, "child-stage"));
    assert(!strcmp(root->name, "mega-store"));
    assert(rpc_profile_record_slot(records[12]) == 2);
    assert(rpc_profile_record_state(records[12]) ==
           (RPC_PROFILE_RECORD_SLOT_DEFINE | 1));
    assert(!strcmp((char *)&records[13], "child-stage"));
    current = 0;
    cpu_self()->profiler = 0;
}

int main(void)
{
    static const size_t counts[] = {0, 1, 8, 9, 32};
    static const thread_fn entries[] = {
        (thread_fn)entry0, (thread_fn)entry1, (thread_fn)entry8,
        (thread_fn)entry9, (thread_fn)entry32,
    };
    uintptr_t args[32];
    struct thread thread;

    for (size_t i = 0; i < 32; ++i)
        args[i] = 0xfedcba9876543200ULL + i;
    for (size_t test = 0; test < sizeof(counts) / sizeof(counts[0]); ++test) {
        size_t argc = counts[test];
        memset(stack, 0xa5, sizeof(stack));
        memset(&thread, 0xa5, sizeof(thread));
        memset(observed, 0, sizeof(observed));
        thread_start(&thread, RPC_PROFILE_SLOT_USER_BASE, THREAD_TYPE_TASK_PROC,
                     "args", entries[test], stack, sizeof(stack),
                     (void *)0x12345000, argc, argc ? args : 0);
        struct proc_frame *frame = (struct proc_frame *)thread.sp;
        uintptr_t *stack_args = (uintptr_t *)(frame + 1);
        assert(!(thread.sp & 15));
        assert(thread.generation == test + 1);
        assert(!thread.tensor_context && !thread.resource_context);
        assert(!thread.token_ext && !thread.profile_defined &&
               thread.parent == THREAD_SLOT_NONE);
        assert(!thread.node.prev && !thread.node.next);
        assert(!strcmp(thread.name, "args") && !thread.name[15]);
        for (size_t i = 0; i < argc; ++i)
            assert((i < 8 ? frame->args[i] : stack_args[i - 8]) == args[i]);
        if (argc <= 8)
            assert((uintptr_t)stack_args == (uintptr_t)(stack + sizeof(stack)));
        assert(stack[0] == 0xa5);
        run_bootstrap(thread.sp, thread.pc);
        assert(!(observed_sp & 15) && observed_gp == 0x12345000);
        for (size_t i = 0; i < argc; ++i)
            assert(observed[i] == args[i]);
        for (size_t i = argc; i < 32; ++i)
            assert(!observed[i]);
        assert(stack[0] == 0xa5);
    }
    test_proc_create();
    test_proc_name();
    return 0;
}
