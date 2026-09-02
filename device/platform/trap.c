#include <stddef.h>
#include <stdint.h>

#include "cpu.h"
#include "errno.h"
#include "platform.h"
#include "rpc_internal.h"
#include "sche.h"
#include "trap.h"

_Noreturn void trap_handle(const struct trap_frame *frame)
{
    struct cpu *cpu;
    struct trap_record *record;
    size_t i;
    int status;

    if (!frame || !current)
        platform_halt();
    cpu = cpu_self();
    if (cpu->trap.active)
        platform_halt();
    cpu->trap.active = 1;
    record = &cpu->trap.record;
    record->hart_id = (uint32_t)platform_hart_id();
    record->thread_id = current->id;
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
    if (current->type == THREAD_TYPE_PROC) {
        struct proc *proc = proc_self();

        if (!proc)
            platform_halt();
        record->host_proc_id = proc->host_proc_id;
        record->parent_proc_id = proc->parent_id;
        record->cookie = proc->cookie;
    }

    status = rpc_report_trap(record, sizeof(*record));
    if (status < 0)
        platform_halt();
    cpu->trap.active = 0;
    if (current->type == THREAD_TYPE_PROC)
        proc_exit_status(-EIO);
    platform_halt();
}
