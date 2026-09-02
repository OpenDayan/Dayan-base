#include "trap.h"

.section .text.trap, "ax", @progbits
.align 2
.globl trap_entry
.type trap_entry, @function
trap_entry:
    csrrw   sp, mscratch, sp
    addi    sp, sp, -TRAP_FRAME_SIZE

    sd      zero, TRAP_FRAME_X0(sp)
    sd      ra, TRAP_FRAME_X1(sp)
    sd      gp, TRAP_FRAME_X3(sp)
    sd      tp, TRAP_FRAME_X4(sp)
    sd      t0, TRAP_FRAME_X5(sp)
    sd      t1, TRAP_FRAME_X6(sp)
    sd      t2, TRAP_FRAME_X7(sp)
    sd      s0, TRAP_FRAME_X8(sp)
    sd      s1, TRAP_FRAME_X9(sp)
    sd      a0, TRAP_FRAME_X10(sp)
    sd      a1, TRAP_FRAME_X11(sp)
    sd      a2, TRAP_FRAME_X12(sp)
    sd      a3, TRAP_FRAME_X13(sp)
    sd      a4, TRAP_FRAME_X14(sp)
    sd      a5, TRAP_FRAME_X15(sp)
    sd      a6, TRAP_FRAME_X16(sp)
    sd      a7, TRAP_FRAME_X17(sp)
    sd      s2, TRAP_FRAME_X18(sp)
    sd      s3, TRAP_FRAME_X19(sp)
    sd      s4, TRAP_FRAME_X20(sp)
    sd      s5, TRAP_FRAME_X21(sp)
    sd      s6, TRAP_FRAME_X22(sp)
    sd      s7, TRAP_FRAME_X23(sp)
    sd      s8, TRAP_FRAME_X24(sp)
    sd      s9, TRAP_FRAME_X25(sp)
    sd      s10, TRAP_FRAME_X26(sp)
    sd      s11, TRAP_FRAME_X27(sp)
    sd      t3, TRAP_FRAME_X28(sp)
    sd      t4, TRAP_FRAME_X29(sp)
    sd      t5, TRAP_FRAME_X30(sp)
    sd      t6, TRAP_FRAME_X31(sp)

    csrr    t0, mscratch
    sd      t0, TRAP_FRAME_X2(sp)
    csrr    t0, mstatus
    sd      t0, TRAP_FRAME_MSTATUS(sp)
    csrr    t0, mepc
    sd      t0, TRAP_FRAME_MEPC(sp)
    csrr    t0, mcause
    sd      t0, TRAP_FRAME_MCAUSE(sp)
    csrr    t0, mtval
    sd      t0, TRAP_FRAME_MTVAL(sp)
    csrr    t0, fcsr
    sd      t0, TRAP_FRAME_FCSR(sp)
    sd      zero, TRAP_FRAME_RESERVED(sp)

    /* RPC uses MSIP as a polled doorbell. Resume any stale latched IRQ. */
    ld      t0, TRAP_FRAME_MCAUSE(sp)
    li      t1, 0x8000000000000003
    beq     t0, t1, .Ltrap_resume_msip
    li      t1, 0x800000000000000b
    beq     t0, t1, .Ltrap_resume_tensor

    /* Rearm the dedicated stack before C switches away from this proc. */
    addi    t0, sp, TRAP_FRAME_SIZE
    csrw    mscratch, t0
    mv      a0, sp
    call    trap_handle

.Ltrap_halt:
    wfi
    j       .Ltrap_halt

.Ltrap_resume_msip:
    /* The receive thread already consumed this polled doorbell. */
    call    platform_rpc_receive_ack

    /* mret copies MPIE to MIE, so clear both bits before resuming. */
    li      t0, 0x88
    csrc    mstatus, t0
    j       .Ltrap_restore

.Ltrap_resume_tensor:
    mv      a0, sp
    call    tensor_interrupt_handle

.Ltrap_restore:
    ld      ra, TRAP_FRAME_X1(sp)
    ld      gp, TRAP_FRAME_X3(sp)
    ld      tp, TRAP_FRAME_X4(sp)
    ld      t2, TRAP_FRAME_X7(sp)
    ld      s0, TRAP_FRAME_X8(sp)
    ld      s1, TRAP_FRAME_X9(sp)
    ld      a0, TRAP_FRAME_X10(sp)
    ld      a1, TRAP_FRAME_X11(sp)
    ld      a2, TRAP_FRAME_X12(sp)
    ld      a3, TRAP_FRAME_X13(sp)
    ld      a4, TRAP_FRAME_X14(sp)
    ld      a5, TRAP_FRAME_X15(sp)
    ld      a6, TRAP_FRAME_X16(sp)
    ld      a7, TRAP_FRAME_X17(sp)
    ld      s2, TRAP_FRAME_X18(sp)
    ld      s3, TRAP_FRAME_X19(sp)
    ld      s4, TRAP_FRAME_X20(sp)
    ld      s5, TRAP_FRAME_X21(sp)
    ld      s6, TRAP_FRAME_X22(sp)
    ld      s7, TRAP_FRAME_X23(sp)
    ld      s8, TRAP_FRAME_X24(sp)
    ld      s9, TRAP_FRAME_X25(sp)
    ld      s10, TRAP_FRAME_X26(sp)
    ld      s11, TRAP_FRAME_X27(sp)
    ld      t3, TRAP_FRAME_X28(sp)
    ld      t4, TRAP_FRAME_X29(sp)
    ld      t5, TRAP_FRAME_X30(sp)
    ld      t6, TRAP_FRAME_X31(sp)
    ld      t0, TRAP_FRAME_X5(sp)
    ld      t1, TRAP_FRAME_X6(sp)
    addi    sp, sp, TRAP_FRAME_SIZE
    csrrw   sp, mscratch, sp
    mret
.size trap_entry, . - trap_entry

.section .note.GNU-stack, "", @progbits
