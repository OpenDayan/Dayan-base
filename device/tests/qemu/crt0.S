.section .text.start, "ax", @progbits
.globl _start
.type _start, @function
.align 2
_start:
    csrr    t0, mhartid
    bnez    t0, .Lpark

    la      sp, __stack_top

    /* QEMU starts in M-mode; enable the single-precision FP context. */
    li      t0, 0x6000
    csrs    mstatus, t0
    csrw    fcsr, zero

    la      t0, __bss_start
    la      t1, __bss_end
.Lzero_bss:
    bgeu    t0, t1, .Lrun
    sb      zero, 0(t0)
    addi    t0, t0, 1
    j       .Lzero_bss

.Lrun:
    call    main
    call    dayan_test_exit

.Lpark:
    wfi
    j       .Lpark
.size _start, . - _start

.section .note.GNU-stack, "", @progbits
