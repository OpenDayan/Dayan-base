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

#include "mem.h"
#include "platform.h"
#include "stdlib.h"
#include "string.h"

void *malloc(size_t size)
{
    return mem_alloc(size);
}

void *calloc(size_t count, size_t size)
{
    void *ptr;

    if (count && size > SIZE_MAX / count)
        return 0;
    size *= count;
    ptr = mem_alloc(size);
    if (ptr)
        memset(ptr, 0, size);
    return ptr;
}

void *realloc(void *ptr, size_t size)
{
    return mem_resize(ptr, size);
}

void free(void *ptr)
{
    mem_free(ptr);
}

_Noreturn void abort(void)
{
    platform_halt();
}

_Noreturn void exit(int status)
{
    (void)status;
    platform_halt();
}
