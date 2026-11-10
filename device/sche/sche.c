/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include <stddef.h>
#include <stdint.h>

#include <dayan_sys.h>

#include "config.h"
#include "context.h"
#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "profile.h"
#include "resource_scheduler.h"
#include "rpc.h"
#include "rpc_internal.h"
#include "sche.h"
#include "string.h"
#include "tensor_context.h"

extern void proc_bootstrap(void);
extern unsigned char __proc_stack_start[];
extern unsigned char __proc_printf_start[];

#define THREAD_SLOT_COUNT       RPC_PROFILE_SLOT_COUNT
#define TASK_SLOT_FIRST         RPC_PROFILE_SLOT_USER_BASE
#define TASK_SLOT_LAST          CONFIG_PROC_SLOT_COUNT
#define SYSTEM_SLOT_FIRST       RPC_PROFILE_SLOT_RPC_RECEIVE

enum slot_state {
    SLOT_FREE = 0,
    SLOT_ROOT,
    SLOT_RESERVED,
    SLOT_CHILD,
    SLOT_ZOMBIE,
    SLOT_KILLED, /* Token pending; cleared once in thread_kill_resume. */
};

_Static_assert(PROC_MAX_ARGS == RPC_PROC_MAX_ARGS, "Proc argument ABI");
_Static_assert(PROC_FRAME_SIZE + (PROC_MAX_ARGS - PROC_REGISTER_ARGS) *
                   sizeof(uintptr_t) <= RPC_PROC_MIN_STACK_SIZE,
               "Proc arguments fit the minimum stack");
_Static_assert(THREAD_RUNNING == 0, "assembly stores zero for RUNNING");
_Static_assert(offsetof(struct thread, state) == THREAD_STATE_OFFSET,
               "assembly thread state offset");
_Static_assert(offsetof(struct thread, slot) == THREAD_SLOT_OFFSET &&
                   THREAD_STATE_OFFSET == THREAD_SLOT_OFFSET + 1 &&
                   !(THREAD_SLOT_OFFSET & 1),
               "assembly profile slot/state halfword");
_Static_assert(offsetof(struct thread, sp) == THREAD_SP_OFFSET,
               "assembly thread sp offset");
_Static_assert(offsetof(struct thread, pc) == THREAD_PC_OFFSET,
               "assembly thread pc offset");
_Static_assert(offsetof(struct thread, tensor_context) ==
                   THREAD_TENSOR_CONTEXT_OFFSET,
               "assembly thread Context1 offset");
_Static_assert(offsetof(struct thread, resource_context) ==
                   THREAD_RESOURCE_CONTEXT_OFFSET,
               "assembly thread Context2 offset");
_Static_assert(offsetof(struct thread, token_ext) == THREAD_TOKEN_EXT_OFFSET,
               "assembly thread token extension offset");
_Static_assert(sizeof(struct thread) == 56 &&
                   offsetof(struct thread, name) == 40 &&
                   offsetof(struct thread, profile_defined) == 7,
               "thread machine-word initialization layout");
_Static_assert(sizeof(struct task_descriptor) == 48 &&
                   offsetof(struct task_descriptor, task_words) == 24,
               "task descriptor word copy layout");

#define TASK_INDEX(slot)   ((slot) - TASK_SLOT_FIRST)
#define WAIT_INDEX(slot)   ((slot) - 1U)

struct thread *thread_slot_get(uint8_t slot)
{
    return &cpu_self()->threads.slots[slot - 1U].thread;
}

uint8_t thread_slot_id(const struct thread *thread)
{
    return thread->slot;
}

static enum slot_state slot_state_get(uint8_t slot)
{
    return (enum slot_state)cpu_self()->threads.slot_state[slot];
}

void thread_list_init(struct thread_list *list)
{
    list->head = THREAD_SLOT_NONE;
    list->tail = THREAD_SLOT_NONE;
}

static void thread_list_remove(struct thread *thread)
{
    uint8_t slot = thread_slot_id(thread);
    struct thread_list *list = cpu_self()->threads.list_owner[slot];
    uint8_t previous = thread->node.prev;
    uint8_t next = thread->node.next;

    if (previous == THREAD_SLOT_NONE)
        list->head = next;
    else
        thread_slot_get(previous)->node.next = next;
    if (next == THREAD_SLOT_NONE)
        list->tail = previous;
    else
        thread_slot_get(next)->node.prev = previous;
}

static struct task_proc *task_proc_for(struct thread *thread)
{
    uint8_t slot;

    if (!thread || (thread->type != THREAD_TYPE_TASK_PROC &&
                    thread->type != THREAD_TYPE_TASK_THREAD))
        return 0;
    slot = thread_slot_id(thread);
    return (struct task_proc *)thread_slot_get(
        cpu_self()->threads.slot_owner[TASK_INDEX(slot)]);
}

struct thread *thread_self(void)
{
    return current;
}

int thread_set_name(const char *name)
{
    struct thread *thread = current;

    if (!thread || (thread->type != THREAD_TYPE_TASK_PROC &&
                    thread->type != THREAD_TYPE_TASK_THREAD))
        return -EPERM;
    if (!name || !name[0])
        return -EINVAL;
    if (strnlen(name, sizeof(thread->name)) == sizeof(thread->name))
        return -ENAMETOOLONG;
    strncpy(thread->name, name, sizeof(thread->name));
    /* A paused profiler must define the new name when its buffer resumes. */
    thread->profile_defined = 0;
    profile_record_thread_start(thread);
    return 0;
}

struct task_proc *task_proc_self(void)
{
    return task_proc_for(current);
}

const struct task_descriptor *task_descriptor_self(void)
{
    struct task_proc *task = task_proc_self();

    return task ? &task->descriptor : 0;
}

void *task_shared_base(const struct task_proc *task)
{
    if (!task->descriptor.shared_size)
        return 0;
    return (void *)((uintptr_t)CONFIG_COHERENT_BASE +
                    task->descriptor.shared_offset);
}

static size_t task_child_stack_size(const struct task_proc *task)
{
    return (size_t)task->descriptor.child_stack_size_m1 + 1U;
}

static void thread_state_set(struct thread *thread, uint8_t state)
{
    thread->state = state;
    profile_record_thread_state(thread);
}

void thread_schedule(struct thread *thread)
{
    thread_state_set(thread, THREAD_READY);
    /* WAIT clears its staging registers after every accepted submission. */
    if (thread->type == THREAD_TYPE_SYSTEM)
        __builtin_riscv_tx_base_tcsr_write(3, 0, RESOURCE_WAIT_QOS);
    __builtin_riscv_tx_base_tcsr_write((uintptr_t)thread, 0, RESOURCE_WAIT);
}

static inline __attribute__((always_inline))
void startup_words(uintptr_t *out, const uintptr_t *in, size_t count)
{
    while (count >= 8) {
        out[0] = in[0];
        if (count > 8)
            platform_prefetch_read(in + 8);
        out[1] = in[1];
        out[2] = in[2];
        out[3] = in[3];
        out[4] = in[4];
        out[5] = in[5];
        out[6] = in[6];
        out[7] = in[7];
        in += 8;
        out += 8;
        count -= 8;
    }
    while (count--)
        *out++ = *in++;
}

static inline __attribute__((always_inline))
void thread_start_internal(struct thread *thread, uint8_t slot, uint8_t type,
                 const char *name, thread_fn entry,
                 void *stack, size_t stack_size, void *shared_base,
                 size_t argc, const uintptr_t *args, int rpc_args,
                 uint64_t tensor_context, uint64_t resource_state,
                 uint8_t parent)
{
    struct thread_runtime *runtime = &cpu_self()->threads;
    uintptr_t top = ((uintptr_t)stack + stack_size) & ~(uintptr_t)0xf;
    size_t register_argc = argc < PROC_REGISTER_ARGS ? argc : PROC_REGISTER_ARGS;
    size_t stack_argc = argc - register_argc;
    top -= (stack_argc * sizeof(uintptr_t) + 15U) & ~(uintptr_t)0xf;
    struct proc_frame *frame = (struct proc_frame *)(top - sizeof(*frame));
    uintptr_t *stack_args = (uintptr_t *)top;

    /* Initialize live fields once; do not byte-clear them before overwriting. */
    thread->tensor_context = tensor_context;
    thread->resource_state = resource_state;
    if (rpc_args) {
        typedef uint64_t name_word_t __attribute__((__may_alias__));
        const name_word_t *words = (const name_word_t *)name;
        thread->name_words[0] = words[0];
        thread->name_words[1] = words[1];
    } else {
        strncpy(thread->name, name, sizeof(thread->name));
    }
    if (!++runtime->generation[slot])
        ++runtime->generation[slot];
    thread->identity = (uint64_t)slot | ((uint64_t)THREAD_WAIT_START << 8) |
        ((uint64_t)runtime->generation[slot] << 16) | ((uint64_t)type << 24) |
        ((uint64_t)parent << 32);
    frame->entry = (uintptr_t)entry;
    frame->ra = (uintptr_t)thread_exit;
    if (rpc_args) {
        /* The base always contains five padded args, even when argc is zero. */
        frame->args[0] = args[0];
        if (argc > 5)
            platform_prefetch_read(args + 5);
        frame->args[1] = args[1];
        frame->args[2] = args[2];
        frame->args[3] = args[3];
        frame->args[4] = args[4];
        if (argc > 5) {
            /* Extra cache lines start at args 5, 13, 21, 29. */
            frame->args[5] = args[5];
            if (argc > 13)
                platform_prefetch_read(args + 13);
            frame->args[6] = args[6];
            frame->args[7] = args[7];
            size_t first = stack_argc < 5 ? stack_argc : 5;
            startup_words(stack_args, args + PROC_REGISTER_ARGS, first);
            if (argc > 13)
                startup_words(stack_args + 5, args + 13, argc - 13);
        }
    } else {
        startup_words(frame->args, args, register_argc);
        if (stack_argc)
            startup_words(stack_args, args + PROC_REGISTER_ARGS, stack_argc);
    }
    frame->shared = (uintptr_t)shared_base;
    frame->reserved = 0;
    thread->sp = (uintptr_t)frame;
    thread->pc = (uintptr_t)proc_bootstrap + PROC_BOOTSTRAP_PC_OFFSET(argc);
    runtime->system_wait[WAIT_INDEX(slot)] = 0;
    if (type != THREAD_TYPE_SYSTEM) {
        runtime->result[TASK_INDEX(slot)] = 0;
        runtime->output_size[TASK_INDEX(slot)] = 0;
        thread_list_init(&runtime->wait_exit[TASK_INDEX(slot)]);
    }
    profile_record_thread_start(thread);
    if (slot == RPC_PROFILE_SLOT_RPC_RECEIVE)
        thread_state_set(thread, THREAD_WAIT_POLL);
    else if (type == THREAD_TYPE_SYSTEM)
        thread_schedule(thread);
}

void thread_start(struct thread *thread, uint8_t slot, uint8_t type,
                 const char *name, thread_fn entry,
                 void *stack, size_t stack_size, void *shared_base,
                 size_t argc, const uintptr_t *args)
{
    thread_start_internal(thread, slot, type, name, entry, stack, stack_size,
                          shared_base, argc, args, 0, 0, 0, THREAD_SLOT_NONE);
}

/* No family tokens may remain. A terminal caller must next abandon its stack. */
static void task_proc_release(uint8_t root)
{
    struct thread_runtime *runtime = &cpu_self()->threads;

    for (uint8_t slot = root; slot <= TASK_SLOT_LAST; ++slot) {
        if (runtime->slot_owner[TASK_INDEX(slot)] != root)
            continue;
        runtime->slot_owner[TASK_INDEX(slot)] = THREAD_SLOT_NONE;
        runtime->slot_state[slot] = SLOT_FREE;
    }
}

struct task_proc *task_proc_create(const struct rpc_proc_create_request *request)
{
    struct thread_runtime *runtime = &cpu_self()->threads;
    struct task_proc *task;
    uint8_t root = THREAD_SLOT_NONE;
    uint8_t last = THREAD_SLOT_NONE;
    unsigned remaining = 1U + request->max_child_threads;

    for (uint8_t slot = TASK_SLOT_FIRST;
         slot <= TASK_SLOT_LAST && remaining; ++slot) {
        if (runtime->slot_state[slot] != SLOT_FREE)
            continue;
        if (root == THREAD_SLOT_NONE)
            root = slot;
        runtime->slot_owner[TASK_INDEX(slot)] = root;
        runtime->slot_state[slot] = SLOT_RESERVED;
        last = slot;
        --remaining;
    }
    if (remaining) {
        if (root != THREAD_SLOT_NONE)
            task_proc_release(root);
        return 0;
    }

    task = (struct task_proc *)thread_slot_get(root);
    runtime->slot_state[root] = SLOT_ROOT;
    task->descriptor.cookie = request->cookie;
    task->descriptor.shared = request->shared;
    task->descriptor.launch = request->launch | ((uint64_t)last << 56);
    task->descriptor.task_words[0] = request->task_words[0];
    task->descriptor.task_words[1] = request->task_words[1];
    task->descriptor.task_words[2] = request->task_words[2];
    thread_start_internal(&task->thread, root, THREAD_TYPE_TASK_PROC, request->name,
                  (thread_fn)(uintptr_t)request->entry,
                  runtime->stacks[TASK_INDEX(root)],
                  task_child_stack_size(task), task_shared_base(task),
                  request->argc, (const uintptr_t *)request->args, 1,
                  request->tensor_context, request->resource_context,
                  THREAD_SLOT_NONE);
    const struct dayan_resource_context_fields resources =
        dayan_resource_context_unpack(request->resource_context);
    if (resources.local.count)
        resource_scheduler_local_init((uint32_t)(
            ((1ULL << resources.local.count) - 1U) << resources.local.base));
    return task;
}

static int child_tcsr_reserve(uint8_t used, struct dayan_tcsr_range parent,
                              uint8_t count, struct dayan_tcsr_range *child)
{
    *child = (struct dayan_tcsr_range){0, 0};
    if (!count)
        return 0;
    if (count > parent.count)
        return -ENOMEM;
    for (uint8_t base = 0; base <= parent.count - count; ++base) {
        uint8_t mask = (uint8_t)(((1U << count) - 1U) << base);

        if (used & mask)
            continue;
        child->base = (uint8_t)(parent.base + base);
        child->count = count;
        return 0;
    }
    return -ENOMEM;
}

static uint8_t child_tcsr_mask(struct dayan_tcsr_range parent,
                               struct dayan_tcsr_range child)
{
    return child.count ?
        (uint8_t)(((1U << child.count) - 1U) << (child.base - parent.base)) : 0;
}

int thread_create_child(const char *name, thread_fn entry, void *argument,
                        const struct dayan_thread_resources *resources)
{
    struct thread_runtime *runtime = &cpu_self()->threads;
    struct task_proc *task = task_proc_self();
    struct thread *child;
    struct dayan_context_fields root_context, child_context;
    struct dayan_resource_context_fields root_resources, child_resources;
    const dayan_thread_resources_t empty_resources = {0};
    uintptr_t arg = (uintptr_t)argument;
    uint8_t root, child_slot = THREAD_SLOT_NONE;
    uint8_t used_lsu = 0, used_mat = 0, used_vec = 0, used_macro = 0;
    int needs_tcsr;

    if (!task)
        return -EPERM;
    if (!name || !name[0] || !entry)
        return -EINVAL;
    if (strnlen(name, THREAD_NAME_SIZE) == THREAD_NAME_SIZE)
        return -ENAMETOOLONG;
    if (!resources)
        resources = &empty_resources;
    if (resources->reserved ||
        resources->lsu_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->mat_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->vec_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->macro_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->local_resource_count > DAYAN_RESOURCE_LOGICAL_COUNT ||
        resources->shared_resource_count > DAYAN_RESOURCE_LOGICAL_COUNT)
        return -EINVAL;

    root = thread_slot_id(&task->thread);
    root_context = dayan_context_unpack(task->thread.tensor_context);
    root_resources = dayan_resource_context_unpack(task->thread.resource_context);
    needs_tcsr = resources->lsu_tcsr_count | resources->mat_tcsr_count |
                 resources->vec_tcsr_count | resources->macro_tcsr_count;
    /* Reservations are made in ascending order; no child lies below root. */
    for (uint8_t slot = root + 1; slot <= task->descriptor.last_slot; ++slot) {
        struct dayan_context_fields context;

        if (runtime->slot_owner[TASK_INDEX(slot)] != root)
            continue;
        if (runtime->slot_state[slot] == SLOT_RESERVED &&
            child_slot == THREAD_SLOT_NONE) {
            child_slot = slot;
            if (!needs_tcsr)
                break;
        }
        if (!needs_tcsr || runtime->slot_state[slot] != SLOT_CHILD)
            continue;
        context = dayan_context_unpack(thread_slot_get(slot)->tensor_context);
        used_lsu |= child_tcsr_mask(root_context.lsu, context.lsu);
        used_mat |= child_tcsr_mask(root_context.mat, context.mat);
        used_vec |= child_tcsr_mask(root_context.vec, context.vec);
        used_macro |= child_tcsr_mask(root_context.macro, context.macro);
    }
    if (child_slot == THREAD_SLOT_NONE ||
        resources->local_resource_count > root_resources.local.count ||
        resources->shared_resource_count > root_resources.shared.count)
        return -ENOMEM;

    child_context = root_context;
    child_resources = root_resources;
    child_resources.local.count = resources->local_resource_count;
    child_resources.shared.count = resources->shared_resource_count;
    if (child_tcsr_reserve(used_lsu, root_context.lsu,
                           resources->lsu_tcsr_count, &child_context.lsu) ||
        child_tcsr_reserve(used_mat, root_context.mat,
                           resources->mat_tcsr_count, &child_context.mat) ||
        child_tcsr_reserve(used_vec, root_context.vec,
                           resources->vec_tcsr_count, &child_context.vec) ||
        child_tcsr_reserve(used_macro, root_context.macro,
                           resources->macro_tcsr_count, &child_context.macro))
        return -ENOMEM;

    child = thread_slot_get(child_slot);
    /* Host validated root; reservation bounded every child range by root. */
    thread_start_internal(child, child_slot, THREAD_TYPE_TASK_THREAD, name,
                  entry, runtime->stacks[TASK_INDEX(child_slot)],
                  task_child_stack_size(task), task_shared_base(task), 1, &arg, 0,
                  dayan_context_pack_validated(&child_context),
                  dayan_resource_context_pack_validated(&child_resources),
                  thread_slot_id(current));
    runtime->slot_state[child_slot] = SLOT_CHILD;
    thread_schedule(child);
    return thread_id(child);
}

int thread_join(uint32_t id, int64_t *result)
{
    struct thread_runtime *runtime = &cpu_self()->threads;
    uint8_t slot = id & THREAD_ID_SLOT_MASK;
    uint8_t parent_slot = thread_slot_id(current);
    struct thread *child;

    if (!task_proc_self())
        return -EPERM;
    if (slot < TASK_SLOT_FIRST || slot > TASK_SLOT_LAST ||
        (runtime->slot_state[slot] != SLOT_CHILD &&
         runtime->slot_state[slot] != SLOT_ZOMBIE))
        return -ECHILD;
    child = thread_slot_get(slot);
    if (thread_id(child) != id || child->parent != parent_slot)
        return -ECHILD;
    if (runtime->slot_state[slot] != SLOT_ZOMBIE)
        thread_wait_join(&runtime->wait_exit[TASK_INDEX(slot)]);
    if (result)
        *result = runtime->result[TASK_INDEX(slot)];
    runtime->slot_state[slot] = SLOT_RESERVED;
    return 0;
}

int thread_join_all(void)
{
    struct thread_runtime *runtime = &cpu_self()->threads;
    uint8_t parent_slot = thread_slot_id(current);

    if (!task_proc_self())
        return -EPERM;
    for (uint8_t slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        struct thread *child = thread_slot_get(slot);

        if ((runtime->slot_state[slot] == SLOT_CHILD ||
             runtime->slot_state[slot] == SLOT_ZOMBIE) &&
            child->parent == parent_slot)
            (void)thread_join(thread_id(child), 0);
    }
    return 0;
}

void thread_system_wait_set(struct thread *thread, void *wait)
{
    uint8_t slot = thread_slot_id(thread);

    cpu_self()->threads.system_wait[WAIT_INDEX(slot)] = wait;
}

int thread_output_enabled(const struct thread *thread)
{
    struct task_proc *task = task_proc_for((struct thread *)thread);

    return task && task->descriptor.child_printf_size;
}

int thread_output_flush(struct thread *thread)
{
    struct thread_runtime *runtime = &cpu_self()->threads;
    struct task_proc *task = task_proc_for(thread);
    uint8_t slot;
    uint16_t *size;

    if (!task)
        return -1;
    slot = thread_slot_id(thread);
    size = &runtime->output_size[TASK_INDEX(slot)];
    if (!*size)
        return 0;
    if (rpc_write_proc(thread_id(&task->thread),
                        runtime->output[TASK_INDEX(slot)], *size) < 0)
        return -1;
    *size = 0;
    return 0;
}

int thread_output_put(struct thread *thread, char ch)
{
    struct thread_runtime *runtime = &cpu_self()->threads;
    struct task_proc *task = task_proc_for(thread);
    uint8_t slot;
    uint16_t *size;

    if (!task || !task->descriptor.child_printf_size)
        return -1;
    slot = thread_slot_id(thread);
    size = &runtime->output_size[TASK_INDEX(slot)];
    if (*size == task->descriptor.child_printf_size &&
        thread_output_flush(thread) < 0)
        return -1;
    runtime->output[TASK_INDEX(slot)][(*size)++] = ch;
    return *size == task->descriptor.child_printf_size ?
        thread_output_flush(thread) : 0;
}

static void snapshot_thread(struct rpc_proc_info *record,
                            struct thread *thread, uint32_t flags,
                            uint32_t slot_index)
{
    struct task_proc *task = task_proc_for(thread);
    struct thread *parent = thread->parent == THREAD_SLOT_NONE ? 0 :
                            thread_slot_get(thread->parent);

    memset(record, 0, sizeof(*record));
    record->proc_id = thread_id(thread);
    record->parent_proc_id = parent ? thread_id(parent) : 0;
    record->state = thread->state;
    record->flags = flags;
    for (uint8_t slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot)
        record->live_children += (slot_state_get(slot) == SLOT_CHILD ||
                                  slot_state_get(slot) == SLOT_KILLED) &&
            thread_slot_get(slot)->parent == thread_slot_id(thread);
    record->slot_index = slot_index;
    record->result = task ? cpu_self()->threads.result[
                               TASK_INDEX(thread_slot_id(thread))] : 0;
    record->pc = thread->pc;
    memcpy(record->name, thread->name, sizeof(record->name));
    if (task) {
        record->task_mode = task->descriptor.task_mode;
        record->host_proc_id = thread_id(&task->thread);
        memcpy(record->task_id, task->descriptor.task_id,
               sizeof(record->task_id));
        memcpy(record->task_dim, task->descriptor.task_dim,
               sizeof(record->task_dim));
        record->cookie = task->descriptor.cookie;
    }
}

static uint32_t snapshot_slot_flags(uint8_t slot)
{
    enum slot_state state = slot_state_get(slot);
    uint32_t flags;

    if (state == SLOT_KILLED) {
        return RPC_PROC_INFO_KILL_WAIT_PRODUCE |
            (thread_slot_get(slot)->type == THREAD_TYPE_TASK_PROC ?
                 RPC_PROC_INFO_HOST_ROOT : RPC_PROC_INFO_CHILD);
    } else if (state == SLOT_ROOT) {
        flags = RPC_PROC_INFO_HOST_ROOT;
    } else if (state == SLOT_CHILD) {
        flags = RPC_PROC_INFO_CHILD;
    } else if (state == SLOT_ZOMBIE) {
        flags = RPC_PROC_INFO_CHILD | RPC_PROC_INFO_ZOMBIE;
    } else {
        return 0;
    }
    return flags;
}

int proc_snapshot(struct rpc_proc_ps_snapshot *snapshot)
{
    uint32_t count = 0;
    uint32_t free_slots = 0;
    uint8_t slot;

    if (snapshot->flags ||
        snapshot->capacity != RPC_PROC_PS_MAX_RECORDS || snapshot->reserved)
        return -EINVAL;
    memset(snapshot->records, 0, sizeof(snapshot->records));
    for (slot = SYSTEM_SLOT_FIRST; slot < THREAD_SLOT_COUNT; ++slot) {
        struct thread *thread = thread_slot_get(slot);

        snapshot_thread(&snapshot->records[count++], thread,
                        RPC_PROC_INFO_SYSTEM, UINT32_MAX);
    }
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        uint32_t flags = snapshot_slot_flags(slot);

        free_slots += slot_state_get(slot) == SLOT_FREE;
        if (!flags)
            continue;
        snapshot_thread(&snapshot->records[count++], thread_slot_get(slot),
                        flags, slot - TASK_SLOT_FIRST);
    }
    snapshot->hart_id = (uint32_t)platform_hart_id();
    snapshot->count = count;
    snapshot->global_free_slots = free_slots;
    snapshot->reserved_slots = CONFIG_PROC_SLOT_COUNT - free_slots;
    snapshot->status = 0;
    return 0;
}

void proc_profile_snapshot(int reset)
{
    uint8_t slot;

    for (slot = SYSTEM_SLOT_FIRST; slot < THREAD_SLOT_COUNT; ++slot) {
        struct thread *thread = thread_slot_get(slot);

        if (reset || !thread->profile_defined)
            profile_record_thread_start(thread);
    }
    for (slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
        if (snapshot_slot_flags(slot)) {
            struct thread *thread = thread_slot_get(slot);

            if (reset || !thread->profile_defined)
                profile_record_thread_start(thread);
        }
    }
}

/*
 * The redirected PC enters once on a fresh stack. Earlier members leave their
 * slots reserved; the last member reports completion and releases the family.
 * No yield occurs between clearing SLOT_KILLED and abandoning this stack.
 */
static _Noreturn void thread_kill_resume(void)
{
    struct thread_runtime *runtime = &cpu_self()->threads;
    struct thread *thread = current;
    uint8_t slot = thread_slot_id(thread);
    uint8_t root = runtime->slot_owner[TASK_INDEX(slot)];

    thread_state_set(thread, THREAD_WAIT_EXIT);
    runtime->slot_state[slot] = slot == root ? SLOT_ROOT : SLOT_ZOMBIE;
    for (uint8_t member = TASK_SLOT_FIRST; member <= TASK_SLOT_LAST; ++member) {
        if (runtime->slot_owner[TASK_INDEX(member)] == root &&
            runtime->slot_state[member] == SLOT_KILLED)
            goto leave;
    }
    (void)rpc_proc_exit((struct task_proc *)thread_slot_get(root),
                       runtime->result[TASK_INDEX(root)]);
    task_proc_release(root);
leave:
    /* No allocation/switch can reuse this stack before the next handoff. */
    current = 0;
    switch_context();
    __builtin_unreachable();
}

static int thread_state_waits_token(uint8_t state)
{
    switch (state) {
    case THREAD_READY:
    case THREAD_POLLING:
    case THREAD_WAIT_PRODUCE:
    case THREAD_WAIT_LSU:
    case THREAD_WAIT_MAT:
    case THREAD_WAIT_VEC:
    case THREAD_WAIT_MACRO:
    case THREAD_YIELD:
    case THREAD_WAIT_RESOURCE:
        return 1;
    default:
        return 0;
    }
}

static int thread_state_waits_event(uint8_t state)
{
    return state == THREAD_WAIT_EVENT || state == THREAD_WAIT_JOIN;
}

static void kill_mark_thread(struct thread *thread, int64_t status)
{
    struct thread_runtime *runtime = &cpu_self()->threads;
    uint8_t slot = thread_slot_id(thread);
    int waiting_token = thread_state_waits_token(thread->state);

    runtime->result[TASK_INDEX(slot)] = status;
    if (thread_state_waits_event(thread->state))
        thread_list_remove(thread);
    runtime->slot_state[slot] = SLOT_KILLED;
    thread->sp = ((uintptr_t)runtime->stacks[TASK_INDEX(slot)] +
                   task_child_stack_size(task_proc_for(thread))) &
                  ~(uintptr_t)0xf;
    thread->pc = (uintptr_t)thread_kill_resume;
    /* Event/join waiters have no token; submit one dependency-free cleanup. */
    if (!waiting_token)
        thread_schedule(thread);
    thread_state_set(thread, THREAD_KILL_WAIT_PRODUCE);
}

int proc_kill_all(struct rpc_proc_kill_result *result)
{
    uintptr_t hart = platform_hart_id();
    struct thread_runtime *runtime = &cpu_self()->threads;
    uint64_t selector;
    int64_t exit_status;
    uint16_t selected_roots = 0;

    if (result->flags & ~RPC_PROC_KILL_FAMILY)
        return -EINVAL;
    selector = result->reserved;
    if (!!(result->flags & RPC_PROC_KILL_FAMILY) != !!selector ||
        selector > UINT16_MAX)
        return -EINVAL;
    exit_status = result->flags & RPC_PROC_KILL_FAMILY ? result->status :
                                                        -ECANCELED;
    if (exit_status >= 0)
        return -EINVAL;
    result->hart_id = (uint32_t)hart;
    result->status = 0;
    result->killed_roots = 0;
    result->killed_children = 0;
    result->released_slots = 0;
    result->waiting_produce = 0;

    for (uint8_t root = TASK_SLOT_FIRST; root <= TASK_SLOT_LAST; ++root) {
        struct thread *thread = thread_slot_get(root);

        if (runtime->slot_state[root] != SLOT_ROOT ||
            thread->state == THREAD_WAIT_EXIT ||
            (selector && thread_id(thread) != selector))
            continue;
        for (uint8_t slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
            if (slot != root && runtime->slot_owner[TASK_INDEX(slot)] == root &&
                (runtime->slot_state[slot] == SLOT_CHILD ||
                 runtime->slot_state[slot] == SLOT_ZOMBIE)) {
                ++result->killed_children;
                kill_mark_thread(thread_slot_get(slot), exit_status);
            }
        }
        ++result->killed_roots;
        selected_roots |= (uint16_t)(1U << TASK_INDEX(root));
        kill_mark_thread(thread, exit_status);
    }
    if (selector && !result->killed_roots)
        return -ESRCH;

    for (uint8_t root = TASK_SLOT_FIRST; root <= TASK_SLOT_LAST; ++root) {
        if (!(selected_roots & (1U << TASK_INDEX(root))))
            continue;
        for (uint8_t slot = TASK_SLOT_FIRST; slot <= TASK_SLOT_LAST; ++slot) {
            struct thread *thread = thread_slot_get(slot);
            struct rpc_transaction *transaction;

            if (runtime->slot_owner[TASK_INDEX(slot)] != root ||
                runtime->slot_state[slot] == SLOT_RESERVED)
                continue;
            transaction = runtime->system_wait[WAIT_INDEX(slot)];
            if (transaction) {
                /* The blocking sender cannot leave an in-flight RPC here. */
                if (!transaction->completed) {
                    list_del_init(&transaction->node);
                    transaction->status = -ECANCELED;
                    transaction->completed = 1;
                } else if (transaction->status >= 0 &&
                           rpc_command_opcode(transaction->command) == RPC_WRITE) {
                    /* The sender got the ack before this print waiter resumed. */
                    runtime->output_size[TASK_INDEX(slot)] = 0;
                }
                thread_system_wait_set(thread, 0);
            }
            (void)thread_output_flush(thread);
            ++result->waiting_produce;
        }
    }
    return 0;
}

size_t thread_event_wake_one(struct thread_list *event)
{
    struct thread *thread;

    if (event->head == THREAD_SLOT_NONE)
        return 0;
    thread = thread_slot_get(event->head);
    thread_list_remove(thread);
    thread_schedule(thread);
    return 1;
}

size_t thread_event_wake_all(struct thread_list *event)
{
    size_t count = 0;

    while (thread_event_wake_one(event))
        ++count;
    return count;
}

static void thread_prepare_wait_state(struct thread_list *event,
                                      uint8_t state)
{
    struct thread *thread = current;
    uint8_t slot = thread_slot_id(thread);
    uint8_t tail = event->tail;

    thread->node.prev = tail;
    thread->node.next = THREAD_SLOT_NONE;
    if (tail == THREAD_SLOT_NONE)
        event->head = slot;
    else
        thread_slot_get(tail)->node.next = slot;
    event->tail = slot;
    cpu_self()->threads.list_owner[slot] = event;
    thread_state_set(thread, state);
}

void thread_wait_event(struct thread_list *event)
{
    thread_prepare_wait_state(event, THREAD_WAIT_EVENT);
    switch_context();
}

void thread_wait_join(struct thread_list *event)
{
    thread_prepare_wait_state(event, THREAD_WAIT_JOIN);
    switch_context();
}

void thread_wait_produce(void)
{
    thread_state_set(current, THREAD_WAIT_PRODUCE);
    switch_context();
}

void thread_wait_unit(enum thread_state state)
{
    thread_state_set(current, (uint8_t)state);
    switch_context();
}

uint32_t thread_wait_resources(uint64_t wait_free, uint64_t wait_ready,
                               uint64_t phase,
                               int8_t qos)
{
    struct thread *thread = current;

    thread_state_set(thread, THREAD_WAIT_RESOURCE);
    resource_scheduler_submit(wait_free, wait_ready, phase,
                              qos, (uintptr_t)thread);
    switch_context();
    return thread->token_ext;
}

void thread_wait_rpc(void)
{
    thread_state_set(current, THREAD_POLLING);
    switch_context();
}

void thread_yield(void)
{
    thread_state_set(current, THREAD_YIELD);
    __builtin_riscv_tx_base_tcsr_write((uintptr_t)current, 0, RESOURCE_WAIT);
    switch_context();
}

void thread_busy_yield(void)
{
    thread_state_set(current, THREAD_POLLING);
    resource_scheduler_submit(0, 0, 0, -4, (uintptr_t)current);
    switch_context();
}

_Noreturn void thread_exit_status(int64_t status)
{
    struct thread_runtime *runtime = &cpu_self()->threads;
    struct thread *thread = current;
    struct task_proc *task = task_proc_for(thread);
    uint8_t slot = thread_slot_id(thread);
    int64_t *result = &runtime->result[TASK_INDEX(slot)];

    if (status < 0 || *result >= 0)
        *result = status;
    (void)thread_join_all();
    (void)thread_output_flush(thread);
    thread_state_set(thread, THREAD_WAIT_EXIT);
    if (thread->type == THREAD_TYPE_TASK_PROC) {
        (void)rpc_proc_exit(task, *result);
        task_proc_release(slot);
    } else {
        int64_t *parent_result = &runtime->result[TASK_INDEX(thread->parent)];

        runtime->slot_state[slot] = SLOT_ZOMBIE;
        if (*result < 0 && *parent_result == 0)
            *parent_result = *result;
        (void)thread_event_wake_all(&runtime->wait_exit[TASK_INDEX(slot)]);
    }
    /* Join/PROC_EXIT owns reclamation; never save or resume this context. */
    current = 0;
    switch_context();
    __builtin_unreachable();
}

_Noreturn void thread_exit(void)
{
    thread_exit_status(0);
}

void sche_init(void)
{
    uintptr_t hart = platform_hart_id();
    struct cpu *cpu = cpu_self();
    struct thread_runtime *runtime = &cpu->threads;

    /* BL1 already zeroed CPU Local, including all slots and runtime metadata. */
    cpu->rpc_receive_thread = thread_slot_get(RPC_PROFILE_SLOT_RPC_RECEIVE);
    runtime->stacks = (void *)(__proc_stack_start +
        hart * CONFIG_PROC_SLOT_COUNT * CONFIG_PROC_STACK_SIZE);
    runtime->output = (void *)(__proc_printf_start +
        hart * CONFIG_PROC_SLOT_COUNT * CONFIG_PROC_PRINTF_SIZE);
    resource_scheduler_local_init(UINT32_MAX);
    memset(runtime->slot_owner, THREAD_SLOT_NONE, sizeof(runtime->slot_owner));
}
