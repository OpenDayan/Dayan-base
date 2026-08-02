#ifndef NPU_SCHE_H
#define NPU_SCHE_H

#include "list.h"
#include "thread.h"

struct proc;
typedef void (*proc_fn)(struct proc *self, void *arg);

struct proc {
    struct thread thread;
    struct list_head runq;
};

_Static_assert(offsetof(struct proc, thread) == 0,
               "proc must inherit thread as its first member");

void sche_init(void);
int proc_init(struct proc *proc, proc_fn run, void *arg,
              void *stack, size_t stack_size);
int proc_start(struct proc *proc);
int proc_wake(struct proc *proc);
void proc_yield(void);
void proc_wait(void);
_Noreturn void proc_exit(void);
_Noreturn void scheduler(void);

#endif
