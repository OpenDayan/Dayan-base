#include <stddef.h>
#include <stdint.h>

#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "rpc_internal.h"
#include "sche.h"
#include "trap.h"

struct trap_record_buffer {
    _Alignas(CONFIG_CACHE_LINE_SIZE) struct trap_record record;
};

static struct trap_record_buffer trap_records[CONFIG_HART_COUNT]
    __attribute__((section(".trap_records")));

void trap_report_complete(void)
{
    cpu_self()->trap.active = 0;
}

void trap_handle(const struct trap_frame *frame)
{
    struct cpu *cpu;
    struct trap_record *record;
    uintptr_t hart;
    size_t i;

    if (!current)
        platform_halt();
    cpu = cpu_self();
    if (cpu->trap.active)
        platform_halt();
    cpu->trap.active = 1;
    hart = platform_hart_id();
    record = &trap_records[hart].record;
    record->hart_id = (uint32_t)hart;
    record->thread_id = thread_id(current);
    record->thread_type = current->type;
    record->thread_state = current->state;
    record->mstatus = frame->mstatus;
    record->mepc = frame->mepc;
    record->mcause = frame->mcause;
    record->mtval = frame->mtval;
    record->fcsr = frame->fcsr;
    for (i = 0; i < 32; ++i)
        record->x[i] = frame->x[i];
    record->host_proc_id = 0;
    record->parent_proc_id = 0;
    record->cookie = 0;
    if (current->type == THREAD_TYPE_TASK_PROC ||
        current->type == THREAD_TYPE_TASK_THREAD) {
        struct task_proc *task = task_proc_self();
        struct thread *parent = current->parent == THREAD_SLOT_NONE ? 0 :
                                thread_slot_get(current->parent);

        record->host_proc_id = thread_id(&task->thread);
        record->parent_proc_id = parent ? thread_id(parent) : 0;
        record->cookie = task->descriptor.cookie;
    }

}

_Noreturn void trap_exit(void)
{
    /* Reporting and joining may switch threads; use the faulting thread's stack. */
    if (rpc_report_trap(&trap_records[platform_hart_id()].record) < 0)
        platform_halt();
    if (current->type == THREAD_TYPE_TASK_PROC ||
        current->type == THREAD_TYPE_TASK_THREAD)
        thread_exit_status(-EIO);
    platform_halt();
}
