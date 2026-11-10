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

#ifndef NPU_LIST_H
#define NPU_LIST_H

#include "api/dayan_list.h"
#include "container_of.h"

#define LIST_HEAD_INIT(name) { &(name), &(name) }
#define LIST_HEAD(name) struct list_head name = LIST_HEAD_INIT(name)

static inline void INIT_LIST_HEAD(struct list_head *head)
{
    head->next = head;
    head->prev = head;
}

static inline int list_empty(const struct list_head *head)
{
    return head->next == head;
}

static inline int list_is_singular(const struct list_head *head)
{
    return !list_empty(head) && head->next == head->prev;
}

static inline void __list_add(struct list_head *node,
                              struct list_head *prev,
                              struct list_head *next)
{
    next->prev = node;
    node->next = next;
    node->prev = prev;
    prev->next = node;
}

static inline void list_add(struct list_head *node, struct list_head *head)
{
    __list_add(node, head, head->next);
}

static inline void list_add_tail(struct list_head *node,
                                 struct list_head *head)
{
    __list_add(node, head->prev, head);
}

static inline void __list_del(struct list_head *prev, struct list_head *next)
{
    next->prev = prev;
    prev->next = next;
}

static inline void list_del(struct list_head *node)
{
    __list_del(node->prev, node->next);
    node->next = 0;
    node->prev = 0;
}

static inline void list_del_init(struct list_head *node)
{
    __list_del(node->prev, node->next);
    INIT_LIST_HEAD(node);
}

static inline void list_move(struct list_head *node, struct list_head *head)
{
    __list_del(node->prev, node->next);
    list_add(node, head);
}

static inline void list_move_tail(struct list_head *node,
                                  struct list_head *head)
{
    __list_del(node->prev, node->next);
    list_add_tail(node, head);
}

#define list_entry(ptr, type, member) container_of(ptr, type, member)
#define list_entry_const(ptr, type, member) \
    container_of_const(ptr, type, member)

#define list_first_entry(head, type, member) \
    list_entry((head)->next, type, member)

#define list_last_entry(head, type, member) \
    list_entry((head)->prev, type, member)

#define list_next_entry(pos, member) \
    list_entry((pos)->member.next, __typeof__(*(pos)), member)

#define list_prev_entry(pos, member) \
    list_entry((pos)->member.prev, __typeof__(*(pos)), member)

#define list_first_entry_or_null(head, type, member) \
    (list_empty(head) ? 0 : list_first_entry(head, type, member))

#define list_for_each(pos, head) \
    for ((pos) = (head)->next; (pos) != (head); (pos) = (pos)->next)

#define list_for_each_safe(pos, next, head) \
    for ((pos) = (head)->next, (next) = (pos)->next; \
         (pos) != (head); \
         (pos) = (next), (next) = (pos)->next)

#define list_for_each_entry(pos, head, member) \
    for (struct list_head *__node = (head)->next; \
         __node != (head) && \
         (((pos) = list_entry(__node, __typeof__(*(pos)), member)), 1); \
         __node = __node->next)

#define list_for_each_entry_safe(pos, tmp, head, member) \
    for (struct list_head *__node = (head)->next, *__next; \
         __node != (head) && \
         ((__next = __node->next), \
          ((pos) = list_entry(__node, __typeof__(*(pos)), member)), \
          ((tmp) = __next == (head) ? 0 : \
                   list_entry(__next, __typeof__(*(tmp)), member)), 1); \
         __node = __next)

#endif
