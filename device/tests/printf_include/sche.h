#ifndef TEST_SCHE_H
#define TEST_SCHE_H

#include <stddef.h>
#include <stdint.h>

struct thread {
    uint16_t id;
};

struct thread *thread_self(void);
int thread_output_enabled(const struct thread *thread);
int thread_output_put(struct thread *thread, char ch);
int thread_output_flush(struct thread *thread);

#endif
