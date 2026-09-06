#ifndef NPU_SCHE_H
#define NPU_SCHE_H

#include <stddef.h>

#include "config.h"
#include "rpc.h"
#include "thread.h"

struct dayan_thread_resources;
struct rpc_proc_ps_snapshot;
struct rpc_proc_kill_result;
struct rpc_proc_create_request;

typedef void (*thread_fn)(void);

struct task_descriptor {
    uint64_t cookie;
    uint32_t task_id[3];
    uint32_t task_dim[3];
    uint32_t shared_offset;
    uint32_t shared_size;
    uint16_t child_stack_size_m1;
    uint16_t child_printf_size;
    uint8_t task_mode;
};

struct task_proc {
    struct thread thread;
    struct task_descriptor descriptor;
};

union thread_slot {
    struct thread thread;
    struct task_proc task_proc;
};

/* Owned by the current Hart's struct cpu; threads never migrate between Harts. */
struct thread_runtime {
    union thread_slot slots[RPC_PROFILE_SLOT_COUNT - 1U];
    void *system_wait[RPC_PROFILE_SLOT_COUNT - 1U];
    int64_t result[CONFIG_PROC_SLOT_COUNT];
    struct thread_list *list_owner[RPC_PROFILE_SLOT_COUNT];
    struct thread_list wait_exit[CONFIG_PROC_SLOT_COUNT];
    uint16_t output_size[CONFIG_PROC_SLOT_COUNT];
    uint8_t slot_state[RPC_PROFILE_SLOT_COUNT];
    uint8_t slot_owner[CONFIG_PROC_SLOT_COUNT];
    uint8_t generation[RPC_PROFILE_SLOT_COUNT];
    unsigned char (*stacks)[CONFIG_PROC_STACK_SIZE];
    unsigned char (*output)[CONFIG_PROC_PRINTF_SIZE];
};

void sche_init(void);
struct thread *thread_slot_get(uint8_t slot);
uint8_t thread_slot_id(const struct thread *thread);
struct thread *thread_self(void);
struct task_proc *task_proc_self(void);
const struct task_descriptor *task_descriptor_self(void);
void *task_shared_base(const struct task_proc *task);

/* Internal caller supplies a valid name, entry, stack and argument count. */
void thread_start(struct thread *thread, uint8_t slot, uint8_t type,
                 const char *name, thread_fn entry,
                 void *stack, size_t stack_size, void *shared_base,
                 size_t argc, const uintptr_t *args);
/* Host-validated request is copied into the slot and startup frame. */
struct task_proc *task_proc_create(const struct rpc_proc_create_request *request);
void thread_schedule(struct thread *thread);
int thread_create_child(const char *name, thread_fn entry, void *argument,
                        const struct dayan_thread_resources *resources);
int thread_join(uint32_t id, int64_t *result);
int thread_join_all(void);
int proc_snapshot(struct rpc_proc_ps_snapshot *snapshot);
/* Emits a slot definition and initial state for every existing proc. */
/* START and resume hold IRQs disabled until definitions are published. */
void proc_profile_snapshot(int reset);
int proc_kill_all(struct rpc_proc_kill_result *result);
void thread_system_wait_set(struct thread *thread, void *wait);
int thread_output_enabled(const struct thread *thread);
int thread_output_put(struct thread *thread, char ch);
int thread_output_flush(struct thread *thread);

size_t thread_event_wake_one(struct thread_list *event);
size_t thread_event_wake_all(struct thread_list *event);
void thread_list_init(struct thread_list *list);
void thread_yield(void);
/* Register under IRQ exclusion, restore IRQs, then call switch_context. */
void thread_prepare_wait(struct thread_list *event);
void thread_wait_event(struct thread_list *event);
void thread_wait_join(struct thread_list *event);
void thread_wait_produce(void);
void thread_wait_unit(enum thread_state state);
uint32_t thread_wait_resources(uint64_t wait_free, uint64_t wait_ready,
                               int8_t qos);
void thread_wait_rpc(void);
void thread_busy_yield(void);
_Noreturn void thread_exit(void);
_Noreturn void thread_exit_status(int64_t status);

#endif
