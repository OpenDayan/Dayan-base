#include <stddef.h>

#include "rbtree.h"

static int is_red(const struct rb_node *node)
{
    return node && rb_is_red(node);
}

static void rotate_left(struct rb_root *root, struct rb_node *node)
{
    struct rb_node *right = node->rb_right;
    struct rb_node *parent = rb_parent(node);

    node->rb_right = right->rb_left;
    if (right->rb_left)
        rb_set_parent(right->rb_left, node);
    rb_set_parent(right, parent);
    if (!parent)
        root->rb_node = right;
    else if (node == parent->rb_left)
        parent->rb_left = right;
    else
        parent->rb_right = right;
    right->rb_left = node;
    rb_set_parent(node, right);
}

static void rotate_right(struct rb_root *root, struct rb_node *node)
{
    struct rb_node *left = node->rb_left;
    struct rb_node *parent = rb_parent(node);

    node->rb_left = left->rb_right;
    if (left->rb_right)
        rb_set_parent(left->rb_right, node);
    rb_set_parent(left, parent);
    if (!parent)
        root->rb_node = left;
    else if (node == parent->rb_right)
        parent->rb_right = left;
    else
        parent->rb_left = left;
    left->rb_right = node;
    rb_set_parent(node, left);
}

void rb_insert_color(struct rb_node *node, struct rb_root *root)
{
    while (node != root->rb_node && is_red(rb_parent(node))) {
        struct rb_node *parent = rb_parent(node);
        struct rb_node *grand = rb_parent(parent);

        if (parent == grand->rb_left) {
            struct rb_node *uncle = grand->rb_right;

            if (is_red(uncle)) {
                rb_set_black(parent);
                rb_set_black(uncle);
                rb_set_red(grand);
                node = grand;
            } else {
                if (node == parent->rb_right) {
                    node = parent;
                    rotate_left(root, node);
                    parent = rb_parent(node);
                    grand = rb_parent(parent);
                }
                rb_set_black(parent);
                rb_set_red(grand);
                rotate_right(root, grand);
            }
        } else {
            struct rb_node *uncle = grand->rb_left;

            if (is_red(uncle)) {
                rb_set_black(parent);
                rb_set_black(uncle);
                rb_set_red(grand);
                node = grand;
            } else {
                if (node == parent->rb_left) {
                    node = parent;
                    rotate_right(root, node);
                    parent = rb_parent(node);
                    grand = rb_parent(parent);
                }
                rb_set_black(parent);
                rb_set_red(grand);
                rotate_left(root, grand);
            }
        }
    }
    rb_set_black(root->rb_node);
}

static void replace(struct rb_root *root,
                    struct rb_node *old, struct rb_node *node)
{
    struct rb_node *parent = rb_parent(old);

    if (!parent)
        root->rb_node = node;
    else if (old == parent->rb_left)
        parent->rb_left = node;
    else
        parent->rb_right = node;
    if (node)
        rb_set_parent(node, parent);
}

static struct rb_node *minimum(struct rb_node *node)
{
    while (node->rb_left)
        node = node->rb_left;
    return node;
}

static struct rb_node *maximum(struct rb_node *node)
{
    while (node->rb_right)
        node = node->rb_right;
    return node;
}

static void erase_fixup(struct rb_root *root,
                        struct rb_node *node, struct rb_node *parent)
{
    while (node != root->rb_node && !is_red(node)) {
        struct rb_node *sibling;

        if (!parent)
            break;
        if (node == parent->rb_left) {
            sibling = parent->rb_right;
            if (is_red(sibling)) {
                rb_set_black(sibling);
                rb_set_red(parent);
                rotate_left(root, parent);
                sibling = parent->rb_right;
            }
            if (!sibling) {
                node = parent;
                parent = rb_parent(node);
                continue;
            }
            if (!is_red(sibling->rb_left) &&
                !is_red(sibling->rb_right)) {
                rb_set_red(sibling);
                node = parent;
                parent = rb_parent(node);
            } else {
                if (!is_red(sibling->rb_right)) {
                    if (sibling->rb_left)
                        rb_set_black(sibling->rb_left);
                    rb_set_red(sibling);
                    rotate_right(root, sibling);
                    sibling = parent->rb_right;
                }
                rb_set_color(sibling, rb_color(parent));
                rb_set_black(parent);
                if (sibling->rb_right)
                    rb_set_black(sibling->rb_right);
                rotate_left(root, parent);
                node = root->rb_node;
                parent = 0;
            }
        } else {
            sibling = parent->rb_left;
            if (is_red(sibling)) {
                rb_set_black(sibling);
                rb_set_red(parent);
                rotate_right(root, parent);
                sibling = parent->rb_left;
            }
            if (!sibling) {
                node = parent;
                parent = rb_parent(node);
                continue;
            }
            if (!is_red(sibling->rb_left) &&
                !is_red(sibling->rb_right)) {
                rb_set_red(sibling);
                node = parent;
                parent = rb_parent(node);
            } else {
                if (!is_red(sibling->rb_left)) {
                    if (sibling->rb_right)
                        rb_set_black(sibling->rb_right);
                    rb_set_red(sibling);
                    rotate_left(root, sibling);
                    sibling = parent->rb_left;
                }
                rb_set_color(sibling, rb_color(parent));
                rb_set_black(parent);
                if (sibling->rb_left)
                    rb_set_black(sibling->rb_left);
                rotate_right(root, parent);
                node = root->rb_node;
                parent = 0;
            }
        }
    }
    if (node)
        rb_set_black(node);
}

void rb_erase(struct rb_node *node, struct rb_root *root)
{
    struct rb_node *moved = node;
    struct rb_node *child;
    struct rb_node *parent;
    unsigned long color = rb_color(moved);

    if (!node->rb_left) {
        child = node->rb_right;
        parent = rb_parent(node);
        replace(root, node, child);
    } else if (!node->rb_right) {
        child = node->rb_left;
        parent = rb_parent(node);
        replace(root, node, child);
    } else {
        moved = minimum(node->rb_right);
        color = rb_color(moved);
        child = moved->rb_right;
        if (rb_parent(moved) == node) {
            parent = moved;
            if (child)
                rb_set_parent(child, moved);
        } else {
            parent = rb_parent(moved);
            replace(root, moved, child);
            moved->rb_right = node->rb_right;
            rb_set_parent(moved->rb_right, moved);
        }
        replace(root, node, moved);
        moved->rb_left = node->rb_left;
        rb_set_parent(moved->rb_left, moved);
        rb_set_color(moved, rb_color(node));
    }

    if (color == RB_BLACK)
        erase_fixup(root, child, parent);

    node->rb_left = 0;
    node->rb_right = 0;
    RB_CLEAR_NODE(node);
}

struct rb_node *rb_first(const struct rb_root *root)
{
    return root->rb_node ? minimum(root->rb_node) : 0;
}

struct rb_node *rb_last(const struct rb_root *root)
{
    return root->rb_node ? maximum(root->rb_node) : 0;
}

struct rb_node *rb_next(const struct rb_node *node)
{
    struct rb_node *parent;

    if (node->rb_right)
        return minimum(node->rb_right);
    while ((parent = rb_parent(node)) && node == parent->rb_right)
        node = parent;
    return parent;
}

struct rb_node *rb_prev(const struct rb_node *node)
{
    struct rb_node *parent;

    if (node->rb_left)
        return maximum(node->rb_left);
    while ((parent = rb_parent(node)) && node == parent->rb_left)
        node = parent;
    return parent;
}

