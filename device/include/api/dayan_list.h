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

#ifndef DAYAN_LIST_H
#define DAYAN_LIST_H

#include <stdint.h>

/* Used by internal pointer-based queues such as RPC transactions. */
struct list_head {
    struct list_head *next;
    struct list_head *prev;
};

#define THREAD_SLOT_NONE     UINT8_MAX
#define THREAD_SLOT_UNLINKED (UINT8_MAX - 1U)

struct thread_list_node {
    uint8_t prev;
    uint8_t next;
};

struct thread_list {
    uint8_t head;
    uint8_t tail;
};

#endif
