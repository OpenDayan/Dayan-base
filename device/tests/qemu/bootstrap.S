/* Run the production bootstrap on a private stack, then restore the test. */
.section .text.run_bootstrap, "ax", @progbits
.globl run_bootstrap
.type run_bootstrap, @function
run_bootstrap:
    addi sp, sp, -32
    sd ra, 0(sp)
    sd s0, 8(sp)
    sd gp, 16(sp)
    mv s0, sp
    mv sp, a0
    la t0, .Lreturned
    sd t0, 8(sp)
    jr a1
.Lreturned:
    mv sp, s0
    ld ra, 0(sp)
    ld s0, 8(sp)
    ld gp, 16(sp)
    addi sp, sp, 32
    ret
.size run_bootstrap, . - run_bootstrap
.section .note.GNU-stack, "", @progbits
