#include "context.h"
#include "profile.h"

.section .text.switch_context, "ax", @progbits
.align 2
.globl switch_context
.type switch_context, @function
switch_context:
.Lschedule:
    /* RPC MSIP has priority; otherwise fall through to the hardware queue. */
    csrr    t0, mip
    andi    t0, t0, 8
    bnez    t0, .Lrpc_receive

    /* BASE Produce/Bar[0] pop, result in t1 (x6). */
    .insn   4, 0x0000130b
    beqz    t1, .Lschedule
    /* TOKEN_EXT read, result in t2 (x7), before the next queue pop. */
    .insn   4, 0x0006d38b

.Lselected:
    sb      t2, THREAD_TOKEN_EXT_OFFSET(t1)
    li      t0, THREAD_RUNNING_VALUE
    sb      t0, THREAD_STATE_OFFSET(t1)
    csrr    a2, mhartid
    slli    a2, a2, cpu_local_shift
    la      t0, __cpu_start
    add     a2, a2, t0
    csrrci  t5, mstatus, 8
    ld      a1, CPU_PROFILER_OFFSET(a2)
    bnez    a1, .Lprofile_switch
.Lprofile_done:
    beq     t1, tp, .Lsame_thread
    mv      a3, t5
    beqz    tp, .Lfirst_thread

    /* next is known before any callee-saved register is spilled. */
    addi    t0, sp, -SWITCH_FRAME_SIZE
    sd      ra, SWITCH_FRAME_RA(t0)
    ld      t2, THREAD_SP_OFFSET(t1)
    sd      s0, SWITCH_FRAME_S0(t0)
    ld      t3, THREAD_TENSOR_CONTEXT_OFFSET(t1)
    sd      s1, SWITCH_FRAME_S1(t0)
    lwu     t4, THREAD_RESOURCE_CONTEXT_OFFSET(t1)
    sd      s2, SWITCH_FRAME_S2(t0)
    ld      t5, THREAD_PC_OFFSET(t1)
    sd      s3, SWITCH_FRAME_S3(t0)
    sd      s4, SWITCH_FRAME_S4(t0)
    sd      s5, SWITCH_FRAME_S5(t0)
    sd      s6, SWITCH_FRAME_S6(t0)
    sd      s7, SWITCH_FRAME_S7(t0)
    sd      s8, SWITCH_FRAME_S8(t0)
    sd      s9, SWITCH_FRAME_S9(t0)
    sd      s10, SWITCH_FRAME_S10(t0)
    sd      s11, SWITCH_FRAME_S11(t0)
    sd      gp, SWITCH_FRAME_GP(t0)
    csrr    t6, fcsr
    sd      t6, SWITCH_FRAME_FCSR(t0)
    sd      t0, THREAD_SP_OFFSET(tp)
    la      t6, switch_context_restore
    sd      t6, THREAD_PC_OFFSET(tp)
    lbu     t6, THREAD_STATE_OFFSET(tp)

.Lload_context:
    mv      sp, t2
    /* IRQs stay masked from profiling through the hardware context/tp update. */
    mv      t0, t3
    /* Context1: BASE unit, address 0x010, source t0 (x5). */
    .insn   4, 0x0002028b
    mv      t0, t4
    /* Context2: BASE unit, address 0x011, source t0 (x5). */
    .insn   4, 0x0002228b
    mv      a0, tp
    mv      tp, t1
    andi    a3, a3, 8
    csrs    mstatus, a3
    li      t0, THREAD_WAIT_EXIT_VALUE
    beq     t6, t0, .Lreap_thread
.Ljump_next:
    jr      t5

.Lrpc_receive:
    csrr    t0, mhartid
    la      t1, scheduler_rpc_receive_threads
    slli    t0, t0, 3
    add     t1, t1, t0
    ld      t1, 0(t1)
    li      t2, 0
    j       .Lselected

.Lreap_thread:
    lbu     t0, THREAD_REAP_DEFERRED_OFFSET(a0)
    bnez    t0, .Ljump_next

    /* Reap the old context only after moving to the next context's stack. */
    addi    sp, sp, -16
    sd      t5, 0(sp)
    call    scheduler_reap_thread
    ld      t5, 0(sp)
    addi    sp, sp, 16
    jr      t5

.Lsame_thread:
    andi    t5, t5, 8
    csrs    mstatus, t5
    ret

.Lfirst_thread:
    ld      t2, THREAD_SP_OFFSET(t1)
    ld      t3, THREAD_TENSOR_CONTEXT_OFFSET(t1)
    lwu     t4, THREAD_RESOURCE_CONTEXT_OFFSET(t1)
    ld      t5, THREAD_PC_OFFSET(t1)
    li      t6, THREAD_RUNNING_VALUE
    j       .Lload_context

.Lprofile_switch:
    la      a3, .Lprofile_done
    j       .Lprofile_record
.size switch_context, . - switch_context

.section .text.profile_record, "ax", @progbits
.align 2
.globl profile_record_thread_state
.type profile_record_thread_state, @function
/* C entry for READY/WAIT/EXIT records; restore the caller's interrupt state. */
profile_record_thread_state:
    mv      t1, a0
    csrr    a2, mhartid
    slli    a2, a2, cpu_local_shift
    la      t0, __cpu_start
    add     a2, a2, t0
    csrrci  t5, mstatus, 8
    ld      a1, CPU_PROFILER_OFFSET(a2)
    bnez    a1, .Lprofile_state
.Lprofile_return:
    andi    t5, t5, 8
    csrs    mstatus, t5
    ret

.Lprofile_state:
    la      a3, .Lprofile_return
/*
 * Append one timestamp48/slot8/state8 record. Enter with IRQs masked,
 * a non-null profiler and room for a record; names are defined in C.
 * t1 = thread, a1 = profiler, a2 = cpu, a3 = continuation, t5 = saved mstatus.
 * Preserve t1, t5 and ra; other caller-saved registers may be clobbered.
 * Return through a3 with IRQs still masked. No stack use on the normal path.
 * Pause below 32 free bytes, reserving room for the next name + state pair.
 */
.Lprofile_record:
    ld      t0, PROFILE_NEXT_OFFSET(a1)
    ld      t2, PROFILE_END_OFFSET(a1)
    rdcycle t3
    slli    t3, t3, 16
    srli    t3, t3, 16
    lhu     t4, THREAD_SLOT_OFFSET(t1)
    slli    t4, t4, 48
    or      t3, t3, t4
    sd      t3, 0(t0)
    addi    t0, t0, 8
    sd      t0, PROFILE_NEXT_OFFSET(a1)
    addi    t2, t2, -32
    bltu    t2, t0, .Lprofile_full
    jr      a3

.Lprofile_full:
    sd      zero, CPU_PROFILER_OFFSET(a2)
    li      t0, 1
    sb      t0, PROFILE_FULL_PENDING_OFFSET(a1)
    /* Disable recording before wake_one can recursively record READY. */
    addi    sp, sp, -32
    sd      ra, 24(sp)
    sd      t1, 16(sp)
    sd      t5, 8(sp)
    sd      a3, 0(sp)
    addi    a0, a1, PROFILE_WAKE_OFFSET
    call    thread_event_wake_one
    ld      a3, 0(sp)
    ld      t5, 8(sp)
    ld      t1, 16(sp)
    ld      ra, 24(sp)
    addi    sp, sp, 32
    jr      a3
.size profile_record_thread_state, . - profile_record_thread_state

.section .text.switch_context_restore, "ax", @progbits
.align 2
.globl switch_context_restore
.type switch_context_restore, @function
switch_context_restore:
    ld      ra, SWITCH_FRAME_RA(sp)
    ld      s0, SWITCH_FRAME_S0(sp)
    ld      s1, SWITCH_FRAME_S1(sp)
    ld      s2, SWITCH_FRAME_S2(sp)
    ld      s3, SWITCH_FRAME_S3(sp)
    ld      s4, SWITCH_FRAME_S4(sp)
    ld      s5, SWITCH_FRAME_S5(sp)
    ld      s6, SWITCH_FRAME_S6(sp)
    ld      s7, SWITCH_FRAME_S7(sp)
    ld      s8, SWITCH_FRAME_S8(sp)
    ld      s9, SWITCH_FRAME_S9(sp)
    ld      s10, SWITCH_FRAME_S10(sp)
    ld      s11, SWITCH_FRAME_S11(sp)
    ld      gp, SWITCH_FRAME_GP(sp)
    ld      t0, SWITCH_FRAME_FCSR(sp)
    csrw    fcsr, t0
    addi    sp, sp, SWITCH_FRAME_SIZE
    ret
.size switch_context_restore, . - switch_context_restore

.section .text.proc_bootstrap, "ax", @progbits
.option push
.option norvc
.align 2
.globl proc_bootstrap
.type proc_bootstrap, @function
proc_bootstrap:
    ld      a7, PROC_FRAME_A7(sp)
    ld      a6, PROC_FRAME_A6(sp)
    ld      a5, PROC_FRAME_A5(sp)
    ld      a4, PROC_FRAME_A4(sp)
    ld      a3, PROC_FRAME_A3(sp)
    ld      a2, PROC_FRAME_A2(sp)
    ld      a1, PROC_FRAME_A1(sp)
    ld      a0, PROC_FRAME_A0(sp)
.Lproc_bootstrap_args_end:
    .if (.Lproc_bootstrap_args_end - proc_bootstrap) != \
        (PROC_MAX_ARGS * PROC_BOOTSTRAP_ARG_INSN_SIZE)
    .error "proc bootstrap argument loads must remain fixed-width"
    .endif
    ld      gp, PROC_FRAME_SHARED(sp)
    csrw    fcsr, zero
    ld      t0, PROC_FRAME_ENTRY(sp)
    ld      ra, PROC_FRAME_RA(sp)
    addi    sp, sp, PROC_FRAME_SIZE
    jr      t0
.size proc_bootstrap, . - proc_bootstrap
.option pop

.section .note.GNU-stack, "", @progbits
