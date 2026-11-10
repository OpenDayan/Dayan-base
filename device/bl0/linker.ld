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

SECTIONS
{
    ROM_BASE = 0x10000; /* ... but actually position independent */

    . = ROM_BASE;
    .text.hang : { *(.text.hang) }
    . = ROM_BASE + 0x40;
    .text.start : { *(.text.start) }
    . = ROM_BASE + 0x80;
    .text.hang80 : { *(.text.hang80) }
    . = ROM_BASE + 0x100;
    .text.hang100 : { *(.text.hang100) }
}