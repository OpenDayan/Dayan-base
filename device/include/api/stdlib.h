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

#ifndef DAYAN_STDLIB_H
#define DAYAN_STDLIB_H

#include <stddef.h>

#include "dayan_attrs.h"

#ifdef __cplusplus
extern "C" {
#endif

DAYAN_LIBC_API void *malloc(size_t size);
DAYAN_LIBC_API void *calloc(size_t count, size_t size);
DAYAN_LIBC_API void *realloc(void *ptr, size_t size);
DAYAN_LIBC_API void free(void *ptr);

DAYAN_LIBC_API DAYAN_NORETURN void abort(void);
DAYAN_LIBC_API DAYAN_NORETURN void exit(int status);

#ifdef __cplusplus
}
#endif

#endif
