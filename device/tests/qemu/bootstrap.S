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
