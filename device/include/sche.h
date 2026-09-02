#ifndef NPU_SCHE_H
#define NPU_SCHE_H

#include <stdarg.h>
#include <stddef.h>

#include "list.h"
#include "thread.h"

struct proc;
struct proc_family;
struct dayan_thread_resources;
struct rpc_proc_ps_snapshot;
struct rpc_proc_kill_result;

#define PROC_NAME_SIZE 16U

typedef void (*proc_fn)(void);
/* Runs in scheduler context and must not block. */
typedef int (*proc_poll_fn)(struct proc *proc, void *arg);
/* Runs on the scheduler stack after a proc has stopped using its own stack. */
typedef void (*proc_cleanup_fn)(struct proc *proc, void *arg);

struct proc {
    struct thread thread;
    char name[PROC_NAME_SIZE];
    struct list_head node;
    proc_poll_fn poll;
    void *poll_arg;
    char *printf_buffer;
    size_t printf_capacity;
    size_t printf_size;
    proc_cleanup_fn cleanup;
    void *cleanup_arg;
    void *system_wait;
    int64_t exit_status;
    uint32_t task_id[3];
    uint32_t task_dim[3];
    uint32_t task_mode;
    struct proc *parent;
    uint32_t parent_id;
    struct proc_family *family;
    struct list_head wait_children;
    uint32_t live_children;
    uint32_t join_kind;
    uint32_t join_thread_id;
    uint32_t host_proc_id;
    uint64_t cookie;
    void *shared_base;
    size_t shared_size;
    void *uram_base;
    size_t uram_size;
};

_Static_assert(offsetof(struct proc, thread) == 0,
               "proc must inherit thread as its first member");

void sche_init(void);
/* shared_base is Host-owned; printf_buffer remains owned by the caller. */
int proc_init(struct proc *proc, const char *name, proc_fn entry,
              void *stack, size_t stack_size, void *shared_base,
              void *printf_buffer, size_t printf_capacity,
              size_t argc, ...);
/* Consumes args; copy it first if the caller needs to read it again. */
int proc_vinit(struct proc *proc, const char *name, proc_fn entry,
               void *stack, size_t stack_size, void *shared_base,
               void *printf_buffer, size_t printf_capacity,
               size_t argc, va_list args);
int proc_init_args(struct proc *proc, const char *name, proc_fn entry,
                   void *stack, size_t stack_size, void *shared_base,
                   void *printf_buffer, size_t printf_capacity,
                   size_t argc, const uintptr_t *args);
int proc_set_cleanup(struct proc *proc, proc_cleanup_fn cleanup, void *arg);
int proc_set_task(struct proc *proc, const uint32_t task_id[3],
                  const uint32_t task_dim[3], uint32_t task_mode);
int proc_slot_acquire(size_t stack_size, size_t printf_size,
                      struct proc **proc_out);
int proc_slot_init_args(struct proc *proc, const char *name, proc_fn entry,
                        void *shared_base, size_t argc,
                        const uintptr_t *args);
void proc_slot_release(struct proc *proc);
int proc_family_create(struct proc *owner, uint32_t max_child_threads,
                       size_t child_stack_size,
                       size_t child_printf_size,
                       struct proc_family **family_out);
void proc_family_destroy(struct proc_family *family);
int proc_set_resources(struct proc *proc, struct proc_family *family,
                       size_t shared_size, void *uram_base,
                       size_t uram_size, uint64_t tensor_context);
int thread_create_child(struct proc *parent, const char *name, proc_fn entry,
                        void *argument,
                        const struct dayan_thread_resources *resources,
                        uint32_t *thread_id_out);
int thread_join(uint32_t thread_id, int64_t *result);
int thread_join_all(void);
int proc_snapshot(struct rpc_proc_ps_snapshot *snapshot);
/* Emits a slot definition and initial state for every existing proc. */
void proc_profile_snapshot(void);
int proc_kill_all(struct rpc_proc_kill_result *result);
struct proc *proc_self(void);
/* Moves an event waiter back to this CPU's ready queue. */
int proc_wake(struct proc *proc);
size_t proc_event_wake_one(struct list_head *event);
size_t proc_event_wake_all(struct list_head *event);
void proc_yield(void);
void proc_wait_poll(proc_poll_fn poll, void *arg);
/* A null event selects the scheduler's default externally-driven queue. */
void proc_wait_event(struct list_head *event);
void proc_wait_produce(void);
void scheduler_scan_produce(void);
_Noreturn void proc_exit(void);
_Noreturn void proc_exit_status(int64_t status);
_Noreturn void scheduler(void);

#endif
