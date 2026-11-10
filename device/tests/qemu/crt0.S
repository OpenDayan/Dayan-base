/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

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
