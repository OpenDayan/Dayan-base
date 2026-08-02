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
static struct rb_root root = RB_ROOT;
static unsigned random_state = 1;

static unsigned next_random(void)
{
    random_state = random_state * 1664525U + 1013904223U;
    return random_state;
}

static struct item *item_from_node(struct rb_node *node)
{
    return rb_entry(node, struct item, node);
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
    assert(rb_parent(node) == parent);
    if (rb_is_red(node)) {
        assert(!node->rb_left || rb_is_black(node->rb_left));
        assert(!node->rb_right || rb_is_black(node->rb_right));
    }

    ++*seen;
    left_height = check_node(node->rb_left, min, item->key, node, seen);
    right_height = check_node(node->rb_right,
                              item->key + 1, max, node, seen);
    assert(left_height == right_height);
    return left_height + rb_is_black(node);
}

static void check_tree(unsigned expected)
{
    struct rb_node *node;
    unsigned key = 0;
    unsigned ordered = 0;
    unsigned reverse = NODE_COUNT + 1;
    unsigned seen = 0;

    assert(RB_EMPTY_ROOT(&root) || rb_is_black(root.rb_node));
    (void)check_node(root.rb_node, 0, NODE_COUNT + 1, 0, &seen);
    assert(seen == expected);

    for (node = rb_first(&root); node; node = rb_next(node)) {
        assert(item_from_node(node)->key > key);
        key = item_from_node(node)->key;
        ++ordered;
    }
    assert(ordered == expected);

    ordered = 0;
    for (node = rb_last(&root); node; node = rb_prev(node)) {
        assert(item_from_node(node)->key < reverse);
        reverse = item_from_node(node)->key;
        ++ordered;
    }
    assert(ordered == expected);
}

static void insert(struct item *item)
{
    struct rb_node **link = &root.rb_node;
    struct rb_node *parent = 0;

    while (*link) {
        parent = *link;
        if (item->key < item_from_node(parent)->key)
            link = &parent->rb_left;
        else
            link = &parent->rb_right;
    }
    item->present = 1;
    rb_link_node(&item->node, parent, link);
    rb_insert_color(&item->node, &root);
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
    unsigned round;
    unsigned i;

    for (round = 0; round < 32; ++round) {
        for (i = 0; i < NODE_COUNT; ++i) {
            items[i].key = i + 1;
            items[i].present = 0;
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

            rb_erase(&item->node, &root);
            item->present = 0;
            assert(RB_EMPTY_NODE(&item->node));
            check_tree(NODE_COUNT - i - 1);
        }
        assert(RB_EMPTY_ROOT(&root));
    }
    return 0;
}
