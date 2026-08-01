#ifndef NPU_RBTREE_H
#define NPU_RBTREE_H

struct rb_node {
    struct rb_node *left;
    struct rb_node *right;
    struct rb_node *parent;
    unsigned char red;
};

void rb_insert(struct rb_node **root, struct rb_node *node);
void rb_remove(struct rb_node **root, struct rb_node *node);

#endif
