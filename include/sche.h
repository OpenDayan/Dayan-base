#ifndef NPU_SCHE_H
#define NPU_SCHE_H

#include <stdarg.h>

#include "list.h"
#include "thread.h"

struct proc;

typedef void (*proc_fn)(void);
/* Runs in scheduler context and must not block. */
typedef int (*proc_ready_fn)(struct proc *proc);

struct proc {
    struct thread thread;
    struct list_head node;
    proc_ready_fn ready;
};

_Static_assert(offsetof(struct proc, thread) == 0,
               "proc must inherit thread as its first member");

void sche_init(void);
/* Every variadic argument must be passed as one uintptr_t ABI word. */
int proc_init(struct proc *proc, proc_fn entry,
              void *stack, size_t stack_size, size_t argc, ...);
/* Consumes args; copy it first if the caller needs to read it again. */
int proc_vinit(struct proc *proc, proc_fn entry,
               void *stack, size_t stack_size, size_t argc, va_list args);
/* Moves an external-event waiter back to this CPU's ready queue. */
int proc_wake(struct proc *proc);
void proc_yield(void);
void proc_wait_internal(proc_ready_fn ready);
void proc_wait_external(void);
_Noreturn void proc_exit(void);
_Noreturn void scheduler(void);

#endif
