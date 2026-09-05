#ifndef NPU_SCHE_H
#define NPU_SCHE_H

#include <stddef.h>

#include "thread.h"

struct dayan_thread_resources;
struct rpc_proc_ps_snapshot;
struct rpc_proc_kill_result;

typedef void (*thread_fn)(void);
/* Runs in scheduler context and must not block. */
typedef int (*thread_poll_fn)(struct thread *thread, void *arg);

struct task_descriptor {
    uint64_t cookie;
    uint32_t task_id[3];
    uint32_t task_dim[3];
    uint32_t shared_offset;
    uint32_t shared_size;
    uint16_t child_stack_size_m1;
    uint16_t child_printf_size;
    uint8_t task_mode;
    uint8_t max_child_threads;
};

struct task_proc {
    struct thread thread;
    struct task_descriptor descriptor;
};

union thread_slot {
    struct thread thread;
    struct task_proc task_proc;
};

void sche_init(void);
struct thread *thread_slot_get(uint8_t slot);
uint8_t thread_slot_id(const struct thread *thread);
struct thread *thread_self(void);
struct task_proc *task_proc_self(void);
const struct task_descriptor *task_descriptor_self(void);
void *task_shared_base(const struct task_proc *task);
size_t task_shared_size(const struct task_proc *task);
size_t task_child_stack_size(const struct task_proc *task);
void *task_uram_base(const struct task_proc *task);
size_t task_uram_size(const struct task_proc *task);

int thread_start(struct thread *thread, uint8_t slot, uint8_t type,
                 const char *name, thread_fn entry,
                 void *stack, size_t stack_size, void *shared_base,
                 size_t argc, const uintptr_t *args);
int task_proc_slot_acquire(struct task_proc **task_out);
int task_proc_start(struct task_proc *task, const char *name,
                    thread_fn entry, size_t argc, const uintptr_t *args);
int task_proc_schedule(struct task_proc *task);
void task_proc_slot_release(struct task_proc *task);
int task_proc_reserve(struct task_proc *task);
void task_proc_unreserve(struct task_proc *task);
int task_proc_set_descriptor(struct task_proc *task,
                             const struct task_descriptor *descriptor,
                             uint64_t tensor_context,
                             uint32_t resource_context);
int task_proc_set_cleanup_arg(struct task_proc *task, void *arg);
int thread_create_child(struct thread *parent, const char *name,
                        thread_fn entry,
                        void *argument,
                        const struct dayan_thread_resources *resources,
                        uint32_t *thread_id_out);
int thread_join(uint32_t thread_id, int64_t *result);
int thread_join_all(void);
int proc_snapshot(struct rpc_proc_ps_snapshot *snapshot);
/* Emits a slot definition and initial state for every existing proc. */
void proc_profile_snapshot(void);
int proc_kill_all(struct rpc_proc_kill_result *result);
int64_t thread_result(const struct thread *thread);
void *thread_system_wait(const struct thread *thread);
void thread_system_wait_set(struct thread *thread, void *wait);
int thread_output_enabled(const struct thread *thread);
int thread_output_put(struct thread *thread, char ch);
int thread_output_flush(struct thread *thread);

int thread_wake(struct thread *thread);
size_t thread_event_wake_one(struct thread_list *event);
size_t thread_event_wake_all(struct thread_list *event);
void thread_list_init(struct thread_list *list);
void thread_yield(void);
void thread_wait_poll(thread_poll_fn poll, void *arg);
void thread_wait_event(struct thread_list *event);
void thread_wait_produce(void);
uint32_t thread_wait_resources(uint64_t wait_free, uint64_t wait_ready,
                               int8_t qos);
void thread_wait_rpc(void);
void scheduler_notify(void);
void scheduler_reap_thread(struct thread *thread);
void scheduler_kill_token(struct thread *thread);
_Noreturn void thread_exit_defer(void *wait);
void thread_exit_deferred_complete(struct thread *thread, void *wait);
_Noreturn void thread_exit(void);
_Noreturn void thread_exit_status(int64_t status);
_Noreturn void scheduler(void);

#endif
