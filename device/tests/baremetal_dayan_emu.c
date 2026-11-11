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

#include <stddef.h>
#include <stdint.h>

/* rvtng-v0 has no UART model; only this test adapter uses QEMU semihosting. */
static long semihost(long operation, const void *argument)
{
    register long a0 __asm__("a0") = operation;
    register const void *a1 __asm__("a1") = argument;

    __asm__ volatile(".balign 16\n"
                     "slli zero, zero, 31\n"
                     "ebreak\n"
                     "srai zero, zero, 7"
                     : "+r"(a0) : "r"(a1) : "memory");
    return a0;
}

long platform_print(const void *data, size_t size)
{
    const char *text = data;
    char buffer[128];
    size_t offset = 0;

    while (offset < size) {
        size_t count = size - offset;

        if (count > sizeof(buffer) - 1U)
            count = sizeof(buffer) - 1U;
        for (size_t i = 0; i < count; ++i)
            buffer[i] = text[offset + i];
        buffer[count] = 0;
        (void)semihost(4, buffer);
        offset += count;
    }
    return (long)size;
}
