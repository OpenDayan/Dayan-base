#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include <dayan_sys.h>

#include "cpu.h"
#include "context.h"
#include "errno.h"
#include "platform.h"
#include "rpc.h"
#include "sche.h"
#include "stdlib.h"

extern void proc_bootstrap(void);

struct proc_slot {
    struct proc proc;
    struct proc_slot *all_next;
    struct proc_slot *free_next;
    void *stack;
    void *printf_buffer;
};

struct proc_family {
    struct proc *root;
    struct proc_slot *slots;
    struct proc_slot *free_slots;
    uint32_t max_child_procs;
    uint32_t active_child_procs;
    size_t child_stack_size;
    size_t child_printf_size;
};

struct proc *proc_self(void)
{
    struct thread *thread = current;

    if (!thread || thread->type != THREAD_TYPE_PROC)
        return 0;
    return container_of(thread, struct proc, thread);
}

static void switch_to_scheduler(struct thread *thread)
{
    struct thread *init = &thread->cpu->sche.init;

    init->state = THREAD_RUNNING;
    switch_context(thread, init);
}

void sche_init(void)
{
    struct thread *init = current;
    struct sche_state *sche = &init->cpu->sche;

    init->id = 0;
    init->state = THREAD_RUNNING;
    init->pc = 0;
    init->type = THREAD_TYPE_BASE;
    INIT_LIST_HEAD(&sche->ready);
    INIT_LIST_HEAD(&sche->wait_poll);
    INIT_LIST_HEAD(&sche->wait_event);
    sche->event_waiters = 0;
    sche->next_id = 1;
}

int thread_init(struct thread *thread, uint32_t type)
{
    struct cpu *cpu = cpu_self();
    struct sche_state *sche = &cpu->sche;

    if (!thread || type == THREAD_TYPE_BASE)
        return -EINVAL;
    if (!sche->next_id)
        return -EOVERFLOW;

    thread->id = sche->next_id++;
    thread->state = THREAD_WAIT_START;
    thread->cpu = cpu;
    thread->sp = 0;
    thread->pc = 0;
    thread->type = type;
    return 0;
}

int proc_init_args(struct proc *proc, proc_fn entry,
                   void *stack, size_t stack_size, void *shared_base,
                   void *printf_buffer, size_t printf_capacity,
                   size_t argc, const uintptr_t *args)
{
    struct proc_frame *frame;
    uintptr_t base;
    uintptr_t top;
    size_t i;
    int status;

    if (!proc || !entry || !stack || argc > PROC_MAX_ARGS ||
        (argc && !args) ||
        (!printf_buffer != !printf_capacity) ||
        (printf_buffer && printf_capacity >
         UINTPTR_MAX - (uintptr_t)printf_buffer))
        return -EINVAL;

    base = (uintptr_t)stack;
    if (stack_size > UINTPTR_MAX - base)
        return -EINVAL;
    top = (base + stack_size) & ~(uintptr_t)0xf;
    if (top < base || top - base < sizeof(*frame))
        return -EINVAL;

    status = thread_init(&proc->thread, THREAD_TYPE_PROC);
    if (status < 0)
        return status;

    frame = (struct proc_frame *)(top - sizeof(*frame));
    frame->entry = (uintptr_t)entry;
    frame->ra = (uintptr_t)proc_exit;
    for (i = 0; i < argc; ++i)
        frame->args[i] = args[i];
    frame->shared = (uintptr_t)shared_base;
    frame->reserved = 0;

    proc->thread.sp = (uintptr_t)frame;
    proc->thread.pc = (uintptr_t)proc_bootstrap +
                      PROC_BOOTSTRAP_PC_OFFSET(argc);
    INIT_LIST_HEAD(&proc->node);
    proc->poll = 0;
    proc->poll_arg = 0;
    proc->printf_buffer = printf_buffer;
    proc->printf_capacity = printf_capacity;
    proc->printf_size = 0;
    proc->cleanup = 0;
    proc->cleanup_arg = 0;
    proc->exit_status = 0;
    proc->task_id[0] = 0;
    proc->task_id[1] = 0;
    proc->task_id[2] = 0;
    proc->task_dim[0] = 1;
    proc->task_dim[1] = 1;
    proc->task_dim[2] = 1;
    proc->task_mode = 1;
    proc->parent = 0;
    proc->family = 0;
    INIT_LIST_HEAD(&proc->wait_children);
    proc->live_children = 0;
    proc->shared_base = shared_base;
    proc->shared_size = 0;
    proc->uram_base = 0;
    proc->uram_size = 0;
    proc->thread.state = THREAD_READY;
    list_add_tail(&proc->node, &proc->thread.cpu->sche.ready);
    return 0;
}

__attribute__((noinline))
int proc_vinit(struct proc *proc, proc_fn entry,
               void *stack, size_t stack_size, void *shared_base,
               void *printf_buffer, size_t printf_capacity,
               size_t argc, va_list args)
{
    uintptr_t words[PROC_MAX_ARGS];
    size_t i;

    if (argc > PROC_MAX_ARGS)
        return -EINVAL;
    for (i = 0; i < argc; ++i)
        words[i] = va_arg(args, uintptr_t);
    return proc_init_args(proc, entry, stack, stack_size, shared_base,
                          printf_buffer, printf_capacity, argc, words);
}

int proc_init(struct proc *proc, proc_fn entry,
              void *stack, size_t stack_size, void *shared_base,
              void *printf_buffer, size_t printf_capacity,
              size_t argc, ...)
{
    va_list args;
    int status;

    va_start(args, argc);
    status = proc_vinit(proc, entry, stack, stack_size, shared_base,
                        printf_buffer, printf_capacity, argc, args);
    va_end(args);
    return status;
}

int proc_set_cleanup(struct proc *proc, proc_cleanup_fn cleanup, void *arg)
{
    if (!proc || proc->thread.type != THREAD_TYPE_PROC)
        return -EINVAL;
    if (proc->thread.cpu != cpu_self())
        return -EPERM;
    if (proc->thread.state != THREAD_READY || proc->cleanup)
        return -EBUSY;
    proc->cleanup = cleanup;
    proc->cleanup_arg = arg;
    return 0;
}

int proc_set_task(struct proc *proc, const uint32_t task_id[3],
                  const uint32_t task_dim[3], uint32_t task_mode)
{
    size_t axis;

    if (!proc || !task_id || !task_dim ||
        (task_mode != 1U && task_mode != 4U) ||
        (task_mode == 4U && task_dim[0] != 4U))
        return -EINVAL;
    if (proc->thread.cpu != cpu_self())
        return -EPERM;
    if (proc->thread.type != THREAD_TYPE_PROC ||
        proc->thread.state != THREAD_READY)
        return -EBUSY;
    for (axis = 0; axis < 3; ++axis) {
        if (!task_dim[axis] || task_id[axis] >= task_dim[axis])
            return -EINVAL;
    }
    for (axis = 0; axis < 3; ++axis) {
        proc->task_id[axis] = task_id[axis];
        proc->task_dim[axis] = task_dim[axis];
    }
    proc->task_mode = task_mode;
    return 0;
}

int proc_family_create(uint32_t max_child_procs, size_t child_stack_size,
                       size_t child_printf_size,
                       struct proc_family **family_out)
{
    struct proc_family *family;
    uint32_t index;

    if (!family_out || max_child_procs > DAYAN_KERNEL_MAX_CHILD_PROCS ||
        child_stack_size < sizeof(struct proc_frame))
        return -EINVAL;
    *family_out = 0;
    family = calloc(1, sizeof(*family));
    if (!family)
        return -ENOMEM;
    family->max_child_procs = max_child_procs;
    family->child_stack_size = child_stack_size;
    family->child_printf_size = child_printf_size;
    for (index = 0; index < max_child_procs; ++index) {
        struct proc_slot *slot = calloc(1, sizeof(*slot));

        if (slot)
            slot->stack = malloc(child_stack_size);
        if (slot && child_printf_size)
            slot->printf_buffer = malloc(child_printf_size);
        if (!slot || !slot->stack ||
            (child_printf_size && !slot->printf_buffer)) {
            if (slot) {
                free(slot->printf_buffer);
                free(slot->stack);
                free(slot);
            }
            proc_family_destroy(family);
            return -ENOMEM;
        }
        slot->all_next = family->slots;
        family->slots = slot;
        slot->free_next = family->free_slots;
        family->free_slots = slot;
    }
    *family_out = family;
    return 0;
}

void proc_family_destroy(struct proc_family *family)
{
    struct proc_slot *slot;

    if (!family)
        return;
    if (family->active_child_procs)
        platform_halt();
    slot = family->slots;
    while (slot) {
        struct proc_slot *next = slot->all_next;

        free(slot->printf_buffer);
        free(slot->stack);
        free(slot);
        slot = next;
    }
    free(family);
}

int proc_set_resources(struct proc *proc, struct proc_family *family,
                       size_t shared_size, void *uram_base,
                       size_t uram_size)
{
    uintptr_t uram_address = (uintptr_t)uram_base;

    if (!proc || !family || proc->thread.cpu != cpu_self() ||
        proc->thread.type != THREAD_TYPE_PROC ||
        proc->thread.state != THREAD_READY || proc->family || family->root ||
        (!proc->shared_base != !shared_size) ||
        (!uram_base != !uram_size) ||
        (uram_size &&
         ((uram_address & (DAYAN_URAM_ALLOCATION_GRANULE - 1U)) ||
          (uram_size & (DAYAN_URAM_ALLOCATION_GRANULE - 1U)) ||
          uram_size > UINTPTR_MAX - uram_address)))
        return -EINVAL;
    proc->family = family;
    family->root = proc;
    proc->shared_size = shared_size;
    proc->uram_base = uram_base;
    proc->uram_size = uram_size;
    return 0;
}

static void child_proc_cleanup(struct proc *proc, void *arg)
{
    struct proc_slot *slot = arg;
    struct proc_family *family = proc->family;
    struct proc *parent = proc->parent;

    if (!slot || !family || !parent || !family->active_child_procs ||
        !parent->live_children)
        platform_halt();
    --family->active_child_procs;
    --parent->live_children;
    if (proc->exit_status < 0 && parent->exit_status == 0)
        parent->exit_status = proc->exit_status;
    if (!parent->live_children)
        (void)proc_event_wake_all(&parent->wait_children);
    slot->free_next = family->free_slots;
    family->free_slots = slot;
}

int proc_create_child(struct proc *parent, proc_fn entry, void *argument,
                      uint32_t *proc_id_out)
{
    struct proc_family *family;
    struct proc_slot *slot;
    uintptr_t args[1];
    int status;

    if (!parent || parent != proc_self() || !entry || !proc_id_out)
        return -EINVAL;
    family = parent->family;
    if (!family || family->root->thread.cpu != cpu_self())
        return -EPERM;
    if (family->active_child_procs >= family->max_child_procs ||
        !family->free_slots)
        return -ENOMEM;
    slot = family->free_slots;
    family->free_slots = slot->free_next;
    args[0] = (uintptr_t)argument;
    status = proc_init_args(&slot->proc, entry, slot->stack,
                            family->child_stack_size, parent->shared_base,
                            slot->printf_buffer, family->child_printf_size,
                            1, args);
    if (status < 0) {
        slot->free_next = family->free_slots;
        family->free_slots = slot;
        return status;
    }
    slot->proc.parent = parent;
    slot->proc.family = family;
    slot->proc.shared_size = parent->shared_size;
    slot->proc.uram_base = parent->uram_base;
    slot->proc.uram_size = parent->uram_size;
    status = proc_set_task(&slot->proc, parent->task_id, parent->task_dim,
                           parent->task_mode);
    if (status < 0)
        platform_halt();
    status = proc_set_cleanup(&slot->proc, child_proc_cleanup, slot);
    if (status < 0)
        platform_halt();
    ++parent->live_children;
    ++family->active_child_procs;
    *proc_id_out = slot->proc.thread.id;
    return 0;
}

int proc_wake(struct proc *proc)
{
    struct cpu *cpu = cpu_self();
    struct sche_state *sche = &cpu->sche;

    if (!proc)
        return -EINVAL;
    if (proc->thread.cpu != cpu)
        return -EPERM;
    if (proc->thread.type != THREAD_TYPE_PROC ||
        proc->thread.state != THREAD_WAIT_EVENT)
        return -EBUSY;
    if (!sche->event_waiters)
        return -EINVAL;
    proc->poll = 0;
    proc->poll_arg = 0;
    proc->thread.state = THREAD_READY;
    --sche->event_waiters;
    list_move_tail(&proc->node, &sche->ready);
    return 0;
}

size_t proc_event_wake_one(struct list_head *event)
{
    struct proc *proc;

    if (!event || list_empty(event))
        return 0;
    proc = list_first_entry(event, struct proc, node);
    return proc_wake(proc) == 0;
}

size_t proc_event_wake_all(struct list_head *event)
{
    size_t count = 0;

    while (proc_event_wake_one(event))
        ++count;
    return count;
}

int proc_wait_poll(proc_poll_fn poll, void *arg)
{
    struct proc *proc = proc_self();

    if (!poll)
        return -EINVAL;
    if (!proc)
        return -EPERM;
    proc->poll = poll;
    proc->poll_arg = arg;
    proc->thread.state = THREAD_WAIT_POLL;
    switch_to_scheduler(&proc->thread);
    return 0;
}

int proc_wait_event(struct list_head *event)
{
    struct proc *proc = proc_self();
    struct sche_state *sche;

    if (!proc)
        return -EPERM;
    if (!list_empty(&proc->node))
        return -EBUSY;
    sche = &proc->thread.cpu->sche;
    if (!event)
        event = &sche->wait_event;
    proc->poll = 0;
    proc->poll_arg = 0;
    proc->thread.state = THREAD_WAIT_EVENT;
    ++sche->event_waiters;
    list_add_tail(&proc->node, event);
    switch_to_scheduler(&proc->thread);
    return 0;
}

void proc_yield(void)
{
    struct proc *proc = proc_self();

    if (!proc)
        return;
    proc->poll = 0;
    proc->poll_arg = 0;
    proc->thread.state = THREAD_READY;
    switch_to_scheduler(&proc->thread);
}

_Noreturn void proc_exit_status(int status)
{
    struct proc *proc = proc_self();

    if (proc) {
        if (proc->printf_size &&
            rpc_write(proc->printf_buffer, proc->printf_size) >= 0)
            proc->printf_size = 0;
        /* Preserve a failure already propagated by a completed child. */
        if (status < 0 || proc->exit_status >= 0)
            proc->exit_status = status;
        while (proc->live_children)
            (void)proc_wait_event(&proc->wait_children);
        proc->thread.state = THREAD_WAIT_EXIT;
        switch_to_scheduler(&proc->thread);
    }
    platform_halt();
}

_Noreturn void proc_exit(void)
{
    proc_exit_status(0);
}

_Noreturn void scheduler(void)
{
    struct thread *init = current;
    struct sche_state *sche = &init->cpu->sche;

    for (;;) {
        struct list_head *node;
        struct list_head *next;
        struct proc *proc;

        proc_scan_produce();

        list_for_each_safe(node, next, &sche->wait_poll) {
            struct proc *waiting = list_entry(node, struct proc, node);

            if (!waiting->poll(waiting, waiting->poll_arg))
                continue;
            waiting->poll = 0;
            waiting->poll_arg = 0;
            waiting->thread.state = THREAD_READY;
            list_move_tail(node, &sche->ready);
        }

        proc = list_first_entry_or_null(&sche->ready, struct proc, node);

        if (!proc) {
            if (list_empty(&sche->wait_poll) && !sche->event_waiters)
                platform_idle();
            continue;
        }
        list_del_init(&proc->node);

        init->state = THREAD_READY;
        proc->thread.state = THREAD_RUNNING;
        switch_context(init, &proc->thread);

        if (proc->thread.state == THREAD_READY) {
            list_add_tail(&proc->node, &sche->ready);
        } else if (proc->thread.state == THREAD_WAIT_POLL) {
            list_add_tail(&proc->node, &sche->wait_poll);
        } else if (proc->thread.state == THREAD_WAIT_EXIT &&
                   proc->cleanup) {
            proc_cleanup_fn cleanup = proc->cleanup;
            void *cleanup_arg = proc->cleanup_arg;

            proc->cleanup = 0;
            proc->cleanup_arg = 0;
            cleanup(proc, cleanup_arg);
        }
    }
}
