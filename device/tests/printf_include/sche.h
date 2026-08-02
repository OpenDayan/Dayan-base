#ifndef TEST_SCHE_H
#define TEST_SCHE_H

#include <stddef.h>
#include <stdint.h>

struct thread {
    uint32_t id;
};

struct proc {
    struct thread thread;
    char *printf_buffer;
    size_t printf_capacity;
    size_t printf_size;
};

struct proc *proc_self(void);

#endif
