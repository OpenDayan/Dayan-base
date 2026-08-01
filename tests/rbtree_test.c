#include <assert.h>
#include <stddef.h>

#include "rbtree.h"

#define NODE_COUNT 512

struct item {
    struct rb_node node;
    unsigned key;
    int present;
};

static struct item items[NODE_COUNT];
static struct rb_node *root;
static unsigned random_state = 1;

static unsigned next_random(void)
{
    random_state = random_state * 1664525U + 1013904223U;
    return random_state;
}

static struct item *item_from_node(struct rb_node *node)
{
    return (struct item *)((char *)node - offsetof(struct item, node));
}

static int check_node(struct rb_node *node, unsigned min, unsigned max,
                      struct rb_node *parent, unsigned *seen)
{
    struct item *item;
    int left_height;
    int right_height;

    if (!node)
        return 1;
    item = item_from_node(node);
    assert(item->present);
    assert(item->key >= min && item->key < max);
    assert(node->parent == parent);
    if (node->red) {
        assert(!node->left || !node->left->red);
        assert(!node->right || !node->right->red);
    }

    ++*seen;
    left_height = check_node(node->left, min, item->key, node, seen);
    right_height = check_node(node->right, item->key + 1, max, node, seen);
    assert(left_height == right_height);
    return left_height + !node->red;
}

static void check_tree(unsigned expected)
{
    unsigned seen = 0;

    assert(!root || !root->red);
    (void)check_node(root, 0, NODE_COUNT + 1, 0, &seen);
    assert(seen == expected);
}

static void insert(struct item *item)
{
    struct rb_node **link = &root;
    struct rb_node *parent = 0;

    while (*link) {
        parent = *link;
        if (item->key < item_from_node(parent)->key)
            link = &parent->left;
        else
            link = &parent->right;
    }
    item->node.left = 0;
    item->node.right = 0;
    item->node.parent = parent;
    item->present = 1;
    *link = &item->node;
    rb_insert(&root, &item->node);
}

static void shuffle(unsigned *order)
{
    unsigned i;

    for (i = NODE_COUNT - 1; i; --i) {
        unsigned index = next_random() % (i + 1);
        unsigned value = order[i];

        order[i] = order[index];
        order[index] = value;
    }
}

int main(void)
{
    unsigned order[NODE_COUNT];
    unsigned i;

    for (i = 0; i < NODE_COUNT; ++i) {
        items[i].key = i + 1;
        order[i] = i;
    }
    shuffle(order);
    for (i = 0; i < NODE_COUNT; ++i) {
        insert(&items[order[i]]);
        check_tree(i + 1);
    }

    shuffle(order);
    for (i = 0; i < NODE_COUNT; ++i) {
        struct item *item = &items[order[i]];

        rb_remove(&root, &item->node);
        item->present = 0;
        check_tree(NODE_COUNT - i - 1);
    }
    return 0;
}
