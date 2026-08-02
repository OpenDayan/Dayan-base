#ifndef NPU_SCHE_H
#define NPU_SCHE_H

#include <stdarg.h>

#include "list.h"
#include "thread.h"

struct proc;

typedef void (*proc_fn)(void);
/* Runs in scheduler context and must not block. */
typedef int (*proc_poll_fn)(struct proc *proc, void *arg);

struct proc {
    struct thread thread;
    struct list_head node;
    proc_poll_fn poll;
    void *poll_arg;
};

_Static_assert(offsetof(struct proc, thread) == 0,
               "proc must inherit thread as its first member");

void sche_init(void);
/* shared_base is Host-owned; every variadic argument is one uintptr_t word. */
int proc_init(struct proc *proc, proc_fn entry,
              void *stack, size_t stack_size, void *shared_base,
              size_t argc, ...);
/* Consumes args; copy it first if the caller needs to read it again. */
int proc_vinit(struct proc *proc, proc_fn entry,
               void *stack, size_t stack_size, void *shared_base,
               size_t argc, va_list args);
/* Moves an event waiter back to this CPU's ready queue. */
int proc_wake(struct proc *proc);
size_t proc_event_wake_one(struct list_head *event);
size_t proc_event_wake_all(struct list_head *event);
void proc_yield(void);
int proc_wait_poll(proc_poll_fn poll, void *arg);
/* A null event selects the scheduler's default externally-driven queue. */
int proc_wait_event(struct list_head *event);
void proc_scan_produce(void);
_Noreturn void proc_exit(void);
_Noreturn void scheduler(void);

#endif
