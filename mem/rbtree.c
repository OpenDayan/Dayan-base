#include <stddef.h>

#include "rbtree.h"

static int is_red(const struct rb_node *node)
{
    return node && node->red;
}

static void rotate_left(struct rb_node **root, struct rb_node *node)
{
    struct rb_node *right = node->right;

    node->right = right->left;
    if (right->left)
        right->left->parent = node;
    right->parent = node->parent;
    if (!node->parent)
        *root = right;
    else if (node == node->parent->left)
        node->parent->left = right;
    else
        node->parent->right = right;
    right->left = node;
    node->parent = right;
}

static void rotate_right(struct rb_node **root, struct rb_node *node)
{
    struct rb_node *left = node->left;

    node->left = left->right;
    if (left->right)
        left->right->parent = node;
    left->parent = node->parent;
    if (!node->parent)
        *root = left;
    else if (node == node->parent->right)
        node->parent->right = left;
    else
        node->parent->left = left;
    left->right = node;
    node->parent = left;
}

void rb_insert(struct rb_node **root, struct rb_node *node)
{
    node->red = 1;

    while (node != *root && is_red(node->parent)) {
        struct rb_node *parent = node->parent;
        struct rb_node *grand = parent->parent;

        if (parent == grand->left) {
            struct rb_node *uncle = grand->right;

            if (is_red(uncle)) {
                parent->red = 0;
                uncle->red = 0;
                grand->red = 1;
                node = grand;
            } else {
                if (node == parent->right) {
                    node = parent;
                    rotate_left(root, node);
                    parent = node->parent;
                    grand = parent->parent;
                }
                parent->red = 0;
                grand->red = 1;
                rotate_right(root, grand);
            }
        } else {
            struct rb_node *uncle = grand->left;

            if (is_red(uncle)) {
                parent->red = 0;
                uncle->red = 0;
                grand->red = 1;
                node = grand;
            } else {
                if (node == parent->left) {
                    node = parent;
                    rotate_right(root, node);
                    parent = node->parent;
                    grand = parent->parent;
                }
                parent->red = 0;
                grand->red = 1;
                rotate_left(root, grand);
            }
        }
    }
    (*root)->red = 0;
}

static void replace(struct rb_node **root,
                    struct rb_node *old, struct rb_node *node)
{
    if (!old->parent)
        *root = node;
    else if (old == old->parent->left)
        old->parent->left = node;
    else
        old->parent->right = node;
    if (node)
        node->parent = old->parent;
}

static struct rb_node *minimum(struct rb_node *node)
{
    while (node->left)
        node = node->left;
    return node;
}

static void remove_fixup(struct rb_node **root,
                         struct rb_node *node, struct rb_node *parent)
{
    while (node != *root && !is_red(node)) {
        struct rb_node *sibling;

        if (!parent)
            break;
        if (node == parent->left) {
            sibling = parent->right;
            if (is_red(sibling)) {
                sibling->red = 0;
                parent->red = 1;
                rotate_left(root, parent);
                sibling = parent->right;
            }
            if (!sibling) {
                node = parent;
                parent = node->parent;
                continue;
            }
            if (!is_red(sibling->left) && !is_red(sibling->right)) {
                sibling->red = 1;
                node = parent;
                parent = node->parent;
            } else {
                if (!is_red(sibling->right)) {
                    if (sibling->left)
                        sibling->left->red = 0;
                    sibling->red = 1;
                    rotate_right(root, sibling);
                    sibling = parent->right;
                }
                sibling->red = parent->red;
                parent->red = 0;
                if (sibling->right)
                    sibling->right->red = 0;
                rotate_left(root, parent);
                node = *root;
                parent = 0;
            }
        } else {
            sibling = parent->left;
            if (is_red(sibling)) {
                sibling->red = 0;
                parent->red = 1;
                rotate_right(root, parent);
                sibling = parent->left;
            }
            if (!sibling) {
                node = parent;
                parent = node->parent;
                continue;
            }
            if (!is_red(sibling->left) && !is_red(sibling->right)) {
                sibling->red = 1;
                node = parent;
                parent = node->parent;
            } else {
                if (!is_red(sibling->left)) {
                    if (sibling->right)
                        sibling->right->red = 0;
                    sibling->red = 1;
                    rotate_left(root, sibling);
                    sibling = parent->left;
                }
                sibling->red = parent->red;
                parent->red = 0;
                if (sibling->left)
                    sibling->left->red = 0;
                rotate_right(root, parent);
                node = *root;
                parent = 0;
            }
        }
    }
    if (node)
        node->red = 0;
}

void rb_remove(struct rb_node **root, struct rb_node *node)
{
    struct rb_node *moved = node;
    struct rb_node *child;
    struct rb_node *parent;
    unsigned char was_red = moved->red;

    if (!node->left) {
        child = node->right;
        parent = node->parent;
        replace(root, node, child);
    } else if (!node->right) {
        child = node->left;
        parent = node->parent;
        replace(root, node, child);
    } else {
        moved = minimum(node->right);
        was_red = moved->red;
        child = moved->right;
        if (moved->parent == node) {
            parent = moved;
            if (child)
                child->parent = moved;
        } else {
            parent = moved->parent;
            replace(root, moved, child);
            moved->right = node->right;
            moved->right->parent = moved;
        }
        replace(root, node, moved);
        moved->left = node->left;
        moved->left->parent = moved;
        moved->red = node->red;
    }

    if (!was_red)
        remove_fixup(root, child, parent);

    node->left = 0;
    node->right = 0;
    node->parent = 0;
    node->red = 0;
}
