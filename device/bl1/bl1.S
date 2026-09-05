#include "config.h"

.section .text.bl1, "ax", @progbits
.globl _start
.type _start, @function
.align 2
_start:
    /* BL0 leaves the hart id in a0. Harts outside this image stay parked. */
    li      t0, CONFIG_HART_COUNT
    bgeu    a0, t0, .Lpark

    /* Each hart gets a disjoint stack. The RISC-V ABI requires 16-byte sp. */
    la      sp, __stack_start
    li      t0, CONFIG_STACK_SIZE
    addi    t1, a0, 1
    mul     t0, t0, t1
    add     sp, sp, t0
    andi    sp, sp, -16

    /* Clear CPU-local data; boot has no schedulable thread. */
    la      t2, __cpu_start
    li      t0, CONFIG_CPU_LOCAL_SIZE
    mul     t1, a0, t0
    add     t1, t2, t1
.Lzero_cpu:
    sd      zero, 0(t1)
    addi    t1, t1, 8
    addi    t0, t0, -8
    bnez    t0, .Lzero_cpu
    mv      tp, zero

    /* Boot owns no Host shared allocation. */
    mv      gp, zero

    /* Exceptions and tensor external interrupts use a per-Hart trap stack. */
    la      t0, __trap_stack_start
    li      t1, CONFIG_TRAP_STACK_SIZE
    addi    t2, a0, 1
    mul     t1, t1, t2
    add     t0, t0, t1
    csrw    mscratch, t0
    la      t0, trap_entry
    csrw    mtvec, t0
    /* Enable and reset the single-precision floating-point context. */
    li      t0, 0x6000
    csrs    mstatus, t0
    csrw    fcsr, zero

    call    init
    /* Publish readiness only after init has returned and released its stack. */
    call    platform_rpc_ready
    /* RPC MSIP is polled; enable only the tensor machine-external IRQ. */
    li      t0, 0x800
    csrw    mie, t0
    li      t0, 0x8
    csrs    mstatus, t0
    tail    switch_context

.Lpark:
    wfi
    j       .Lpark
.size _start, . - _start

.section .note.GNU-stack, "", @progbits
