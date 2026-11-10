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

#ifndef NPU_MEM_H
#define NPU_MEM_H

#include <stddef.h>

int mem_init(void *base, size_t size);
void *mem_alloc(size_t size);
/* Non-null pointers must name live allocations from this Hart's heap. */
void *mem_resize(void *ptr, size_t size);
void mem_free(void *ptr);
size_t mem_size(const void *ptr);
size_t mem_capacity(void);
size_t mem_available(void);

#endif
