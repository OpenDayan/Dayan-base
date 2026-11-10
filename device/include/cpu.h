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

#ifndef NPU_CPU_H
#define NPU_CPU_H

#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "context.h"
#include "list.h"
#include "profile.h"
#include "rbtree.h"
#include "rpc_internal.h"
#include "thread.h"
#include "tensor_interrupt.h"
#include "trap.h"
#include "test.h"

struct mem_state {
    struct rb_root free;
    struct list_head regions;
    size_t capacity;
    size_t available;
};

/* The only owner of persistent mutable firmware state on one Hart. */
struct cpu {
    struct profile_state *profiler;
    struct thread *rpc_receive_thread;
    struct mem_state mem;
    struct rpc_state rpc;
    struct thread_runtime threads;
    struct profile_state profile;
    struct trap_state trap;
    struct tensor_interrupt_state tensor_interrupt;
    struct test_state test;
} __attribute__((aligned(CONFIG_CACHE_LINE_SIZE)));

extern uint8_t __cpu_start[];

_Static_assert(CONFIG_CPU_LOCAL_SIZE > 0 &&
                   !(CONFIG_CPU_LOCAL_SIZE & (CONFIG_CPU_LOCAL_SIZE - 1)),
               "CPU local size must be a power of two");
_Static_assert(offsetof(struct cpu, profiler) == CPU_PROFILER_OFFSET,
               "assembly profiler pointer offset");
_Static_assert(offsetof(struct cpu, rpc_receive_thread) == CPU_RPC_RECEIVE_OFFSET,
               "assembly RPC receiver pointer offset");
_Static_assert(CONFIG_CPU_LOCAL_SIZE % CONFIG_CACHE_LINE_SIZE == 0,
               "CPU local reservations must not share cache lines");
_Static_assert(sizeof(struct cpu) <= CONFIG_CPU_LOCAL_SIZE,
               "cpu state exceeds its linker reservation");

static inline struct cpu *cpu_self(void)
{
    uintptr_t hart;
    uintptr_t base;

    __asm__ volatile("csrr %0, mhartid\n"
                     "la %1, __cpu_start"
                     : "=r"(hart), "=r"(base));
    return (struct cpu *)(base + hart * CONFIG_CPU_LOCAL_SIZE);
}

#endif
