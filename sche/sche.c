#include <stddef.h>

#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "sche.h"

static void enqueue(struct sche_state *sche, struct proc *proc)
{
    proc->next = 0;
    if (sche->tail)
        sche->tail->next = proc;
    else
        sche->head = proc;
    sche->tail = proc;
}

static struct proc *dequeue(struct sche_state *sche)
{
    struct proc *proc = sche->head;

    if (!proc)
        return 0;
    sche->head = proc->next;
    if (!sche->head)
        sche->tail = 0;
    proc->next = 0;
    return proc;
}

static struct proc *current_proc(void)
{
    struct thread *thread = thread_self();

    if (thread->type != THREAD_TYPE_PROC)
        return 0;
    return (struct proc *)thread;
}

void sche_init(void)
{
    struct thread *init = thread_self();
    struct sche_state *sche = &init->cpu->sche;

    init->id = 0;
    init->state = THREAD_RUNNING;
    init->type = THREAD_TYPE_BASE;
    sche->head = 0;
    sche->tail = 0;
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
    thread->type = type;
    return 0;
}

int proc_init(struct proc *proc, proc_fn run, void *arg)
{
    int status;

    if (!proc || !run)
        return -EINVAL;
    status = thread_init(&proc->thread, THREAD_TYPE_PROC);
    if (status < 0)
        return status;
    proc->next = 0;
    proc->run = run;
    proc->arg = arg;
    return 0;
}

int proc_start(struct proc *proc)
{
    struct cpu *cpu = cpu_self();
    struct sche_state *sche = &cpu->sche;

    if (!proc || !proc->run)
        return -EINVAL;
    if (proc->thread.type != THREAD_TYPE_PROC ||
        proc->thread.state != THREAD_WAIT_START)
        return -EBUSY;
    if (proc->thread.cpu != cpu)
        return -EPERM;
    proc->thread.state = THREAD_READY;
    enqueue(sche, proc);
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
    proc->thread.state = THREAD_READY;
    enqueue(sche, proc);
    return 0;
}

void proc_wait(void)
{
    struct proc *proc = current_proc();

    if (proc)
        proc->thread.state = THREAD_WAIT_EVENT;
}

void proc_exit(void)
{
    struct proc *proc = current_proc();

    if (proc)
        proc->thread.state = THREAD_WAIT_EXIT;
}

_Noreturn void scheduler(void)
{
    struct thread *init = thread_self();
    struct sche_state *sche = &init->cpu->sche;

    for (;;) {
        struct proc *proc = dequeue(sche);

        if (!proc) {
            platform_idle();
            continue;
        }

        proc->thread.state = THREAD_RUNNING;
        thread_set(&proc->thread);
        proc->run(proc, proc->arg);
        thread_set(init);

        /* Returning while still running is the cooperative yield point. */
        if (proc->thread.state == THREAD_RUNNING) {
            proc->thread.state = THREAD_READY;
            enqueue(sche, proc);
        }
    }
}
