#ifndef NPU_CPU_H
#define NPU_CPU_H

#include <stddef.h>

struct mem_region;
struct rb_node;

struct mem_state {
    struct rb_node *free;
    struct mem_region *regions;
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
