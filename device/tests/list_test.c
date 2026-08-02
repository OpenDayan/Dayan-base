#include <assert.h>

#include "list.h"

struct item {
    unsigned value;
    struct list_head node;
};

int main(void)
{
    struct item items[4];
    struct item *item;
    struct item *tmp;
    LIST_HEAD(head);
    unsigned expected = 0;
    unsigned i;

    assert(list_empty(&head));
    for (i = 0; i < 4; ++i) {
        items[i].value = i;
        INIT_LIST_HEAD(&items[i].node);
        list_add_tail(&items[i].node, &head);
        if (!i)
            assert(list_is_singular(&head));
    }

    assert(!list_empty(&head));
    assert(!list_is_singular(&head));
    assert(list_first_entry(&head, struct item, node) == &items[0]);
    assert(list_last_entry(&head, struct item, node) == &items[3]);
    assert(list_entry_const((const struct list_head *)&items[0].node,
                            struct item, node) == &items[0]);

    list_for_each_entry(item, &head, node)
        assert(item->value == expected++);
    assert(expected == 4);

    list_move(&items[3].node, &head);
    assert(list_first_entry(&head, struct item, node) == &items[3]);
    list_move_tail(&items[3].node, &head);

    list_for_each_entry_safe(item, tmp, &head, node)
        list_del_init(&item->node);
    assert(list_empty(&head));
    for (i = 0; i < 4; ++i)
        assert(list_empty(&items[i].node));
    return 0;
}
