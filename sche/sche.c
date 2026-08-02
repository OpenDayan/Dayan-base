#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include "cpu.h"
#include "context.h"
#include "errno.h"
#include "platform.h"
#include "rpc.h"
#include "sche.h"

extern void proc_bootstrap(void);

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
        proc->exit_status = status;
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
