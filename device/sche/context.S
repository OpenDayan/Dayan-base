#include "context.h"

.section .text.switch_context, "ax", @progbits
.align 2
.globl switch_context
.type switch_context, @function
switch_context:
    addi    sp, sp, -SWITCH_FRAME_SIZE
    sd      ra, SWITCH_FRAME_RA(sp)
    sd      s0, SWITCH_FRAME_S0(sp)
    sd      s1, SWITCH_FRAME_S1(sp)
    sd      s2, SWITCH_FRAME_S2(sp)
    sd      s3, SWITCH_FRAME_S3(sp)
    sd      s4, SWITCH_FRAME_S4(sp)
    sd      s5, SWITCH_FRAME_S5(sp)
    sd      s6, SWITCH_FRAME_S6(sp)
    sd      s7, SWITCH_FRAME_S7(sp)
    sd      s8, SWITCH_FRAME_S8(sp)
    sd      s9, SWITCH_FRAME_S9(sp)
    sd      s10, SWITCH_FRAME_S10(sp)
    sd      s11, SWITCH_FRAME_S11(sp)
    sd      gp, SWITCH_FRAME_GP(sp)
    csrr    t0, fcsr
    sd      t0, SWITCH_FRAME_FCSR(sp)

    sd      sp, THREAD_SP_OFFSET(a0)
    la      t0, switch_context_restore
    sd      t0, THREAD_PC_OFFSET(a0)

    ld      sp, THREAD_SP_OFFSET(a1)
    ld      t0, THREAD_TENSOR_CONTEXT_OFFSET(a1)
    /* BASE Context write: unit 0, address 0x10, source register t0 (x5). */
    .insn   4, 0x0002028b
    ld      t0, THREAD_PC_OFFSET(a1)
    mv      tp, a1
    jr      t0
.size switch_context, . - switch_context

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
