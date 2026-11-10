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

#include "list.h"
#include "rbtree.h"

struct mem_state {
    struct rb_root free;
    struct list_head regions;
    size_t capacity;
    size_t available;
};

struct cpu {
    struct mem_state mem;
};

extern struct cpu test_cpu;

static inline struct cpu *cpu_self(void)
{
    return &test_cpu;
}

#endif
