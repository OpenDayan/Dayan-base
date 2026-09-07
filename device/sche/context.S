#include "context.h"
#include "profile.h"

.section .text.switch_context, "ax", @progbits
.align 2
.globl switch_context
.type switch_context, @function
/*
 * Cooperative dispatch on the current Hart; there is no scheduler thread.
 *
 * Entry contract:
 *   tp identifies the outgoing thread, or is zero on boot/terminal exit.
 *   sp is a valid, 16-byte-aligned stack and ra is the caller's return address.
 *   The caller has already set its READY/WAIT/EXIT state and, where needed,
 *   submitted a hardware token or joined a software event/join wait list.
 *   Each queued token is a live thread pointer whose slot remains reserved
 *   until the token is consumed. A terminal caller clears tp after releasing
 *   its slot; its stack stays valid until handoff, with no intervening reuse.
 *
 * Selection samples RPC MSIP once on entry, then retries up to 128 empty
 * hardware pops before sampling it again. Sampling on entry also prevents
 * RPC starvation when runnable threads keep the hardware queue nonempty.
 * Event/join readiness is handled by explicit wakes, not by inspecting wait
 * conditions here. Selection uses only caller-saved registers and records
 * RUNNING before the same-thread test.
 *
 * For a different thread, save ra, s0-s11, gp and fcsr in a 128-byte frame,
 * publish the outgoing sp/restore PC, and install the target sp, Context1,
 * Context2 and tp. IRQs are masked from profile cursor access through this
 * handoff, then MIE is restored to its entry value before jumping to the
 * target's saved PC. Exit/join paths own slot reclamation;
 * dispatch does not inspect the outgoing state or run a reaper.
 *
 * Saved PCs select switch_context_restore, initial proc_bootstrap, or the
 * kill cleanup entry installed by RPC. The kill path therefore needs no
 * state check here. Restore/bootstrap finish installing gp and the software
 * frame; cleanup uses a fresh stack and never restores the abandoned frame.
 * A surviving caller returns only when its own thread is selected again.
 * The ABI is lp64: argument/temporary GPRs and FPRs are caller-saved.
 *
 * Register lifetimes (also see the shared profile block below):
 *   tp: outgoing thread until handoff; t1: selected thread until handoff.
 *   t0: selection scratch, then outgoing frame base.
 *   t2: token extension, then target sp; t3/t4: target Context1/Context2.
 *   t5: saved entry mstatus through profiling and handoff.
 *   t6: empty-pop budget, outgoing fcsr, then the outgoing restore PC.
 *   a1/a2: active profiler/current CPU; a3: profile continuation, then target PC.
 */
switch_context:
.Lschedule:
    li      t6, 128
    /* RPC MSIP has priority; otherwise fall through to the hardware queue. */
    csrr    t0, mip
    andi    t0, t0, 8
    bnez    t0, .Lrpc_receive

.Lpop_token:
    beqz    t6, .Lschedule
    addi    t6, t6, -1
    /* BASE Produce/Bar[0] pop returns a thread pointer in t1 (x6), or zero. */
    .insn   4, 0x0000130b
    beqz    t1, .Lpop_token
    /* TOKEN_EXT belongs to this pop; read it into t2 (x7) before another pop. */
    .insn   4, 0x0006d38b

.Lselected:
    csrr    a2, mhartid
    slli    a2, a2, cpu_local_shift
    la      t0, __cpu_start
    add     a2, a2, t0
.Lrun_thread:
    /* t1 is selected and a2 is CPU Local, including on the RPC entry path. */
    sb      t2, THREAD_TOKEN_EXT_OFFSET(t1)
    sb      zero, THREAD_STATE_OFFSET(t1)
    csrrci  t5, mstatus, 8
    ld      a1, CPU_PROFILER_OFFSET(a2)
    /* Null means disabled or paused on a full buffer; that path falls through. */
    bnez    a1, .Lprofile_switch
.Lprofile_done:
    /* A self-selection needs no switch frame or hardware context writes. */
    beq     t1, tp, .Lsame_thread
    beqz    tp, .Lfirst_thread

    /*
     * Keep sp on the outgoing stack while building its frame below sp.
     * Interleave target loads with stores of the outgoing callee-saved GPRs.
     * Context mappings come from thread metadata; gp/fcsr are saved live.
     */
    addi    t0, sp, -SWITCH_FRAME_SIZE
    sd      ra, SWITCH_FRAME_RA(t0)
    ld      t2, THREAD_SP_OFFSET(t1)
    sd      s0, SWITCH_FRAME_S0(t0)
    ld      t3, THREAD_TENSOR_CONTEXT_OFFSET(t1)
    sd      s1, SWITCH_FRAME_S1(t0)
    lwu     t4, THREAD_RESOURCE_CONTEXT_OFFSET(t1)
    sd      s2, SWITCH_FRAME_S2(t0)
    ld      a3, THREAD_PC_OFFSET(t1)
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
    /* A later selection resumes this caller by restoring the frame just saved. */
    sd      t0, THREAD_SP_OFFSET(tp)
    la      t6, switch_context_restore
    sd      t6, THREAD_PC_OFFSET(tp)

.Lload_context:
    /* t2/t3/t4/a3 hold the target sp/Context1/Context2/PC tuple. */
    mv      sp, t2
    /* Keep IRQs masked until the target stack, hardware mappings and tp agree. */
    /* Context1: BASE unit, address 0x010, source t3 (x28). */
    .insn   4, 0x00020e0b
    /* Context2: BASE unit, address 0x011, source t4 (x29). */
    .insn   4, 0x00022e8b
    mv      tp, t1
    andi    t5, t5, 8
    csrs    mstatus, t5
    /* The destination restores a saved frame, bootstraps, or handles RPC kill. */
    jr      a3

.Lrpc_receive:
    /* RPC has its own receiver thread and no queue token/extension to consume. */
    csrr    a2, mhartid
    slli    a2, a2, cpu_local_shift
    la      t0, __cpu_start
    add     a2, a2, t0
    ld      t1, CPU_RPC_RECEIVE_OFFSET(a2)
    /* Bare-metal mode has no Host receiver, even if MSIP is still asserted. */
    beqz    t1, .Lpop_token
    li      t2, 0
    j       .Lrun_thread

.Lsame_thread:
    /* Profiling left IRQs masked; restore entry MIE and return to the same caller. */
    andi    t5, t5, 8
    csrs    mstatus, t5
    ret

.Lfirst_thread:
    /* tp == 0: abandon the boot/exiting stack without saving a thread frame. */
    ld      t2, THREAD_SP_OFFSET(t1)
    ld      t3, THREAD_TENSOR_CONTEXT_OFFSET(t1)
    lwu     t4, THREAD_RESOURCE_CONTEXT_OFFSET(t1)
    ld      a3, THREAD_PC_OFFSET(t1)
    j       .Lload_context

.Lprofile_switch:
    /* Fall straight into the shared writer; a3 returns to dispatch without jal. */
    la      a3, .Lprofile_done
/*
 * Append one timestamp48/slot8/state8 record. Enter with IRQs masked,
 * a non-null profiler and at least eight free bytes; names are defined in C.
 * t1 = thread, a1 = profiler, a2 = cpu, a3 = continuation, t5 = saved mstatus.
 * Scratch: t0 = cursor, t2 = end threshold, t3 = record, t4 = slot/state halfword.
 * Preserve t1, t5 and ra; other caller-saved registers may be clobbered.
 * Return through a3 with IRQs still masked. No stack use on the normal path.
 * Slot and state occupy adjacent bytes, so one aligned lhu supplies both.
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
    /* Pause first, then notify the flush worker; no thread switch occurs here. */
    sd      zero, CPU_PROFILER_OFFSET(a2)
    li      t0, 1
    sb      t0, PROFILE_FULL_PENDING_OFFSET(a1)
    /* wake_one can itself record READY. Save only values needed by either entry. */
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
.size switch_context, . - switch_context

.section .text.profile_record, "ax", @progbits
.align 2
.globl profile_record_thread_state
.type profile_record_thread_state, @function
/*
 * C entry: a0 is the thread whose state changed outside dispatch. Reuse the
 * writer with the same IRQ exclusion, but return via the C caller's ra.
 */
profile_record_thread_state:
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
    mv      t1, a0
    la      a3, .Lprofile_return
    j       .Lprofile_record
.size profile_record_thread_state, . - profile_record_thread_state

.section .text.switch_context_restore, "ax", @progbits
.align 2
.globl switch_context_restore
.type switch_context_restore, @function
/*
 * Selected with sp at a saved switch frame and tp/hardware mappings installed.
 * Restore this thread's ABI state, pop the frame and return to its original
 * switch_context caller. The dispatch path has already restored entry MIE.
 */
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
/*
 * First entry uses a proc_frame instead of a switch_frame. thread_start chooses
 * a PC within these fixed-width loads to skip unused arguments. Install the
 * proc's shared-memory gp, clear fcsr, and use thread_exit as the kernel's ra.
 */
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
