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

#ifndef NPU_CONTAINER_OF_H
#define NPU_CONTAINER_OF_H

#include <stddef.h>

#define container_of(ptr, type, member) __extension__ ({ \
    const __typeof__(((type *)0)->member) *__member = (ptr); \
    (type *)((char *)__member - offsetof(type, member)); \
})

#define container_of_const(ptr, type, member) __extension__ ({ \
    const __typeof__(((type *)0)->member) *__member = (ptr); \
    (const type *)((const char *)__member - offsetof(type, member)); \
})

#endif
