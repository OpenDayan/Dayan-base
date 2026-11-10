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

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    ram (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

PHDRS
{
    text PT_LOAD FLAGS(5);
    data PT_LOAD FLAGS(6);
}

SECTIONS
{
    . = ORIGIN(ram);

    .text : ALIGN(16)
    {
        KEEP(*(.text.start))
        *(.text .text.*)
    } > ram :text

    .rodata : ALIGN(16)
    {
        *(.rodata .rodata.* .srodata .srodata.*)
    } > ram :text

    .data : ALIGN(16)
    {
        *(.data .data.* .sdata .sdata.*)
    } > ram :data

    .bss (NOLOAD) : ALIGN(16)
    {
        __bss_start = .;
        *(.bss .bss.* .sbss .sbss.* COMMON)
        __bss_end = .;
    } > ram :data

    .stack (NOLOAD) : ALIGN(16)
    {
        . += 64K;
        __stack_top = .;
    } > ram :data

    /DISCARD/ :
    {
        *(.eh_frame .eh_frame.*)
        *(.comment)
        *(.note .note.*)
    }
}
