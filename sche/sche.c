#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#include "cpu.h"
#include "context.h"
#include "errno.h"
#include "platform.h"
#include "sche.h"

extern void proc_bootstrap(void);

static struct proc *current_proc(void)
{
    struct thread *thread = current;

    if (thread->type != THREAD_TYPE_PROC)
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
    INIT_LIST_HEAD(&sche->wait_internal_event);
    INIT_LIST_HEAD(&sche->wait_external_event);
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

__attribute__((noinline))
int proc_vinit(struct proc *proc, proc_fn entry,
               void *stack, size_t stack_size, void *shared_base,
               size_t argc, va_list args)
{
    struct proc_frame *frame;
    uintptr_t base;
    uintptr_t top;
    size_t i;
    int status;

    if (!proc || !entry || !stack || argc > PROC_MAX_ARGS)
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
        frame->args[i] = va_arg(args, uintptr_t);
    frame->shared = (uintptr_t)shared_base;
    frame->reserved = 0;

    proc->thread.sp = (uintptr_t)frame;
    proc->thread.pc = (uintptr_t)proc_bootstrap +
                      PROC_BOOTSTRAP_PC_OFFSET(argc);
    INIT_LIST_HEAD(&proc->node);
    proc->ready = 0;
    proc->ready_arg = 0;
    proc->thread.state = THREAD_READY;
    list_add_tail(&proc->node, &proc->thread.cpu->sche.ready);
    return 0;
}

int proc_init(struct proc *proc, proc_fn entry,
              void *stack, size_t stack_size, void *shared_base,
              size_t argc, ...)
{
    va_list args;
    int status;

    va_start(args, argc);
    status = proc_vinit(proc, entry, stack, stack_size, shared_base,
                        argc, args);
    va_end(args);
    return status;
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
        proc->thread.state != THREAD_WAIT_EXTERNAL_EVENT)
        return -EBUSY;
    proc->ready = 0;
    proc->ready_arg = 0;
    proc->thread.state = THREAD_READY;
    list_move_tail(&proc->node, &sche->ready);
    return 0;
}

int proc_wait_internal(proc_ready_fn ready, void *arg)
{
    struct proc *proc = current_proc();

    if (!ready)
        return -EINVAL;
    if (!proc)
        return -EPERM;
    proc->ready = ready;
    proc->ready_arg = arg;
    proc->thread.state = THREAD_WAIT_INTERNAL_EVENT;
    switch_to_scheduler(&proc->thread);
    return 0;
}

void proc_wait_external(void)
{
    struct proc *proc = current_proc();

    if (!proc)
        return;
    proc->ready = 0;
    proc->ready_arg = 0;
    proc->thread.state = THREAD_WAIT_EXTERNAL_EVENT;
    switch_to_scheduler(&proc->thread);
}

void proc_yield(void)
{
    struct proc *proc = current_proc();

    if (!proc)
        return;
    proc->ready = 0;
    proc->ready_arg = 0;
    proc->thread.state = THREAD_READY;
    switch_to_scheduler(&proc->thread);
}

_Noreturn void proc_exit(void)
{
    struct proc *proc = current_proc();

    if (proc) {
        proc->thread.state = THREAD_WAIT_EXIT;
        switch_to_scheduler(&proc->thread);
    }
    platform_halt();
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

        list_for_each_safe(node, next, &sche->wait_internal_event) {
            struct proc *waiting = list_entry(node, struct proc, node);

            if (!waiting->ready(waiting, waiting->ready_arg))
                continue;
            waiting->ready = 0;
            waiting->ready_arg = 0;
            waiting->thread.state = THREAD_READY;
            list_move_tail(node, &sche->ready);
        }

        proc = list_first_entry_or_null(&sche->ready, struct proc, node);

        if (!proc) {
            if (list_empty(&sche->wait_internal_event) &&
                list_empty(&sche->wait_external_event))
                platform_idle();
            continue;
        }
        list_del_init(&proc->node);

        init->state = THREAD_READY;
        proc->thread.state = THREAD_RUNNING;
        switch_context(init, &proc->thread);

        if (proc->thread.state == THREAD_READY) {
            list_add_tail(&proc->node, &sche->ready);
        } else if (proc->thread.state == THREAD_WAIT_INTERNAL_EVENT) {
            list_add_tail(&proc->node, &sche->wait_internal_event);
        } else if (proc->thread.state == THREAD_WAIT_EXTERNAL_EVENT) {
            list_add_tail(&proc->node, &sche->wait_external_event);
        }
    }
}
