#ifndef DAYAN_LIST_H
#define DAYAN_LIST_H

/* Public only so per-hart DAPI resources can embed Linux-style wait queues. */
struct list_head {
    struct list_head *next;
    struct list_head *prev;
};

#endif
