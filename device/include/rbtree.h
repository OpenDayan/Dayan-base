#ifndef NPU_RBTREE_H
#define NPU_RBTREE_H

#include "container_of.h"

struct rb_node {
    unsigned long __rb_parent_color;
    struct rb_node *rb_right;
    struct rb_node *rb_left;
};

struct rb_root {
    struct rb_node *rb_node;
};

_Static_assert(sizeof(unsigned long) == sizeof(void *),
               "rb parent/color packing requires an LP ABI");
_Static_assert(_Alignof(struct rb_node) >= 4,
               "rb parent/color packing requires aligned nodes");

#define RB_ROOT (struct rb_root) { 0 }

#define RB_RED          0UL
#define RB_BLACK        1UL
#define RB_PARENT_MASK  (~3UL)

#define rb_parent(node) \
    ((struct rb_node *)((node)->__rb_parent_color & RB_PARENT_MASK))
#define rb_color(node) ((node)->__rb_parent_color & 1UL)
#define rb_is_red(node) (!rb_color(node))
#define rb_is_black(node) rb_color(node)

#define RB_EMPTY_ROOT(root) ((root)->rb_node == 0)
#define RB_EMPTY_NODE(node) \
    ((node)->__rb_parent_color == (unsigned long)(node))
#define RB_CLEAR_NODE(node) \
    ((node)->__rb_parent_color = (unsigned long)(node))

#define rb_entry(ptr, type, member) container_of(ptr, type, member)
#define rb_entry_const(ptr, type, member) \
    container_of_const(ptr, type, member)

static inline void rb_set_parent(struct rb_node *node,
                                 struct rb_node *parent)
{
    node->__rb_parent_color =
        (node->__rb_parent_color & ~RB_PARENT_MASK) | (unsigned long)parent;
}

static inline void rb_set_color(struct rb_node *node, unsigned long color)
{
    node->__rb_parent_color =
        (node->__rb_parent_color & RB_PARENT_MASK) | color;
}

static inline void rb_set_parent_color(struct rb_node *node,
                                       struct rb_node *parent,
                                       unsigned long color)
{
    node->__rb_parent_color = (unsigned long)parent | color;
}

static inline void rb_set_red(struct rb_node *node)
{
    rb_set_color(node, RB_RED);
}

static inline void rb_set_black(struct rb_node *node)
{
    rb_set_color(node, RB_BLACK);
}

static inline void rb_link_node(struct rb_node *node,
                                struct rb_node *parent,
                                struct rb_node **link)
{
    node->__rb_parent_color = (unsigned long)parent;
    node->rb_left = 0;
    node->rb_right = 0;
    *link = node;
}

void rb_insert_color(struct rb_node *node, struct rb_root *root);
void rb_erase(struct rb_node *node, struct rb_root *root);

struct rb_node *rb_first(const struct rb_root *root);
struct rb_node *rb_last(const struct rb_root *root);
struct rb_node *rb_next(const struct rb_node *node);
struct rb_node *rb_prev(const struct rb_node *node);

#endif
