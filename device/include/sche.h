#ifndef NPU_SCHE_H
#define NPU_SCHE_H

#include <stdarg.h>

#include "list.h"
#include "thread.h"

struct proc;
struct proc_family;

typedef void (*proc_fn)(void);
/* Runs in scheduler context and must not block. */
typedef int (*proc_poll_fn)(struct proc *proc, void *arg);
/* Runs on the scheduler stack after a proc has stopped using its own stack. */
typedef void (*proc_cleanup_fn)(struct proc *proc, void *arg);

struct proc {
    struct thread thread;
    struct list_head node;
    proc_poll_fn poll;
    void *poll_arg;
    char *printf_buffer;
    size_t printf_capacity;
    size_t printf_size;
    proc_cleanup_fn cleanup;
    void *cleanup_arg;
    int exit_status;
    uint32_t task_id[3];
    uint32_t task_dim[3];
    uint32_t task_mode;
    struct proc *parent;
    struct proc_family *family;
    struct list_head wait_children;
    uint32_t live_children;
    void *shared_base;
    size_t shared_size;
    void *uram_base;
    size_t uram_size;
};

_Static_assert(offsetof(struct proc, thread) == 0,
               "proc must inherit thread as its first member");

void sche_init(void);
/* shared_base is Host-owned; printf_buffer remains owned by the caller. */
int proc_init(struct proc *proc, proc_fn entry,
              void *stack, size_t stack_size, void *shared_base,
              void *printf_buffer, size_t printf_capacity,
              size_t argc, ...);
/* Consumes args; copy it first if the caller needs to read it again. */
int proc_vinit(struct proc *proc, proc_fn entry,
               void *stack, size_t stack_size, void *shared_base,
               void *printf_buffer, size_t printf_capacity,
               size_t argc, va_list args);
int proc_init_args(struct proc *proc, proc_fn entry,
                   void *stack, size_t stack_size, void *shared_base,
                   void *printf_buffer, size_t printf_capacity,
                   size_t argc, const uintptr_t *args);
int proc_set_cleanup(struct proc *proc, proc_cleanup_fn cleanup, void *arg);
int proc_set_task(struct proc *proc, const uint32_t task_id[3],
                  const uint32_t task_dim[3], uint32_t task_mode);
int proc_family_create(uint32_t max_child_procs, size_t child_stack_size,
                       size_t child_printf_size,
                       struct proc_family **family_out);
void proc_family_destroy(struct proc_family *family);
int proc_set_resources(struct proc *proc, struct proc_family *family,
                       size_t shared_size, void *uram_base,
                       size_t uram_size);
int proc_create_child(struct proc *parent, proc_fn entry, void *argument,
                      uint32_t *proc_id_out);
struct proc *proc_self(void);
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
_Noreturn void proc_exit_status(int status);
_Noreturn void scheduler(void);

#endif
