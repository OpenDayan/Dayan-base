#ifndef NPU_SCHE_H
#define NPU_SCHE_H

#include "thread.h"

struct proc;
typedef void (*proc_fn)(struct proc *self, void *arg);

struct proc {
    struct thread thread;
    struct proc *next;
    proc_fn run;
    void *arg;
};

_Static_assert(offsetof(struct proc, thread) == 0,
               "proc must inherit thread as its first member");

void sche_init(void);
int proc_init(struct proc *proc, proc_fn run, void *arg);
int proc_start(struct proc *proc);
int proc_wake(struct proc *proc);
void proc_wait(void);
void proc_exit(void);
_Noreturn void scheduler(void);

#endif
