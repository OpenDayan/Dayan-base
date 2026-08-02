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
