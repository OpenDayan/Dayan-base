#include <stddef.h>
#include <stdint.h>

#include "cpu.h"
#include "context.h"
#include "errno.h"
#include "platform.h"
#include "sche.h"
#include "string.h"

extern void proc_bootstrap(void);

static struct proc *current_proc(void)
{
    struct thread *thread = thread_self();

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
    struct thread *init = thread_self();
    struct sche_state *sche = &init->cpu->sche;

    init->id = 0;
    init->state = THREAD_RUNNING;
    init->pc = 0;
    init->type = THREAD_TYPE_BASE;
    INIT_LIST_HEAD(&sche->ready);
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

int proc_init(struct proc *proc, proc_fn run, void *arg,
              void *stack, size_t stack_size)
{
    struct proc_frame *frame;
    uintptr_t base;
    uintptr_t top;
    int status;

    if (!proc || !run || !stack)
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
    memset(frame, 0, sizeof(*frame));
    frame->entry = (uintptr_t)run;
    frame->ra = (uintptr_t)proc_exit;
    frame->a0 = (uintptr_t)proc;
    frame->a1 = (uintptr_t)arg;

    proc->thread.sp = (uintptr_t)frame;
    proc->thread.pc = (uintptr_t)proc_bootstrap;
    INIT_LIST_HEAD(&proc->runq);
    return 0;
}

int proc_start(struct proc *proc)
{
    struct cpu *cpu = cpu_self();
    struct sche_state *sche = &cpu->sche;

    if (!proc || !proc->thread.sp || !proc->thread.pc)
        return -EINVAL;
    if (proc->thread.type != THREAD_TYPE_PROC ||
        proc->thread.state != THREAD_WAIT_START ||
        !list_empty(&proc->runq))
        return -EBUSY;
    if (proc->thread.cpu != cpu)
        return -EPERM;
    proc->thread.state = THREAD_READY;
    list_add_tail(&proc->runq, &sche->ready);
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
        proc->thread.state != THREAD_WAIT_EVENT ||
        !list_empty(&proc->runq))
        return -EBUSY;
    proc->thread.state = THREAD_READY;
    list_add_tail(&proc->runq, &sche->ready);
    return 0;
}

void proc_wait(void)
{
    struct proc *proc = current_proc();

    if (!proc)
        return;
    proc->thread.state = THREAD_WAIT_EVENT;
    switch_to_scheduler(&proc->thread);
}

void proc_yield(void)
{
    struct proc *proc = current_proc();

    if (!proc)
        return;
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
    struct thread *init = thread_self();
    struct sche_state *sche = &init->cpu->sche;

    for (;;) {
        struct proc *proc = list_first_entry_or_null(&sche->ready,
                                                     struct proc, runq);

        if (!proc) {
            platform_idle();
            continue;
        }
        list_del_init(&proc->runq);

        init->state = THREAD_READY;
        proc->thread.state = THREAD_RUNNING;
        switch_context(init, &proc->thread);

        if (proc->thread.state == THREAD_READY)
            list_add_tail(&proc->runq, &sche->ready);
    }
}
