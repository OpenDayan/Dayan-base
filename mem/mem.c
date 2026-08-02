#include <stddef.h>
#include <stdint.h>

#include "cpu.h"
#include "errno.h"
#include "mem.h"
#include "rpc.h"
#include "rbtree.h"
#include "string.h"

#define ALIGNMENT 16UL
#define USED      1UL

struct block {
    size_t size;
    size_t prev_size;
};

struct free_block {
    struct block block;
    struct rb_node node;
};

struct mem_region {
    struct list_head node;
    uintptr_t raw;
    size_t raw_size;
    struct block *first;
    unsigned char dynamic;
};

#define ALIGN_UP(value)   (((value) + ALIGNMENT - 1) & ~(ALIGNMENT - 1))
#define ALIGN_DOWN(value) ((value) & ~(ALIGNMENT - 1))
#define REGION_SIZE       ALIGN_UP(sizeof(struct mem_region))
#define MIN_BLOCK         ALIGN_UP(sizeof(struct free_block))

static size_t block_size(const struct block *block)
{
    return block->size & ~USED;
}

static int block_used(const struct block *block)
{
    return (block->size & USED) != 0;
}

static struct block *next_block(struct block *block)
{
    return (struct block *)((unsigned char *)block + block_size(block));
}

static struct rb_node *block_node(struct block *block)
{
    return &((struct free_block *)block)->node;
}

static struct block *node_block(struct rb_node *node)
{
    return &rb_entry(node, struct free_block, node)->block;
}

static int block_before(const struct block *left, const struct block *right)
{
    size_t left_size = block_size(left);
    size_t right_size = block_size(right);

    return left_size < right_size ||
           (left_size == right_size && (uintptr_t)left < (uintptr_t)right);
}

static void tree_add(struct mem_state *mem, struct block *block)
{
    struct rb_node **link = &mem->free.rb_node;
    struct rb_node *parent = 0;
    struct rb_node *node = block_node(block);

    while (*link) {
        struct block *other = node_block(*link);

        parent = *link;
        if (block_before(block, other))
            link = &parent->rb_left;
        else
            link = &parent->rb_right;
    }

    rb_link_node(node, parent, link);
    rb_insert_color(node, &mem->free);
}

static void tree_remove(struct mem_state *mem, struct block *block)
{
    rb_erase(block_node(block), &mem->free);
}

static struct block *tree_best(struct mem_state *mem, size_t size)
{
    struct rb_node *node = mem->free.rb_node;
    struct block *best = 0;

    while (node) {
        struct block *block = node_block(node);

        if (block_size(block) >= size) {
            best = block;
            node = node->rb_left;
        } else {
            node = node->rb_right;
        }
    }
    return best;
}

static int overlaps(const struct mem_region *region,
                    uintptr_t raw, size_t size)
{
    uintptr_t end = raw + size;
    uintptr_t other_end = region->raw + region->raw_size;

    return raw < other_end && region->raw < end;
}

static int region_add(struct mem_state *mem, uintptr_t raw, size_t raw_size,
                      int dynamic)
{
    uintptr_t start;
    uintptr_t end;
    struct mem_region *region;
    struct block *block;
    struct block *sentinel;
    struct mem_region *other;
    size_t size;

    if (!raw || raw > UINTPTR_MAX - (ALIGNMENT - 1) ||
        raw_size > UINTPTR_MAX - raw)
        return -EINVAL;
    start = ALIGN_UP(raw);
    end = ALIGN_DOWN(raw + raw_size);
    if (end <= start || end - start < REGION_SIZE + MIN_BLOCK + sizeof(*sentinel))
        return -EINVAL;

    list_for_each_entry(other, &mem->regions, node) {
        if (overlaps(other, raw, raw_size))
            return -EINVAL;
    }

    region = (struct mem_region *)start;
    block = (struct block *)(start + REGION_SIZE);
    sentinel = (struct block *)(end - sizeof(*sentinel));
    size = (size_t)((unsigned char *)sentinel - (unsigned char *)block);
    if (size < MIN_BLOCK)
        return -EINVAL;

    region->raw = raw;
    region->raw_size = raw_size;
    region->first = block;
    region->dynamic = (unsigned char)dynamic;

    block->size = size;
    block->prev_size = 0;
    sentinel->size = USED;
    sentinel->prev_size = size;

    list_add(&region->node, &mem->regions);
    mem->capacity += size;
    mem->available += size;
    tree_add(mem, block);
    return 0;
}

static int region_grow(struct mem_state *mem, size_t need)
{
    struct rpc_heap_request request;
    struct rpc_heap_response response;
    long status;

    if (need > SIZE_MAX - REGION_SIZE - sizeof(struct block) -
               2 * (ALIGNMENT - 1))
        return -EOVERFLOW;
    request.size = need + REGION_SIZE + sizeof(struct block) +
                   2 * (ALIGNMENT - 1);
    response.addr = 0;
    response.size = 0;
    status = rpc_call(RPC_HEAP_ACQUIRE,
                      &request, sizeof(request),
                      &response, sizeof(response));
    if (status < 0)
        return (int)status;
    return region_add(mem, response.addr, response.size, 1);
}

static struct mem_region *find_region(struct mem_state *mem, const void *ptr)
{
    struct mem_region *region;
    uintptr_t addr = (uintptr_t)ptr;

    list_for_each_entry(region, &mem->regions, node) {
        if (addr > (uintptr_t)region->first &&
            addr < region->raw + region->raw_size)
            return region;
    }
    return 0;
}

static int release_region(struct mem_state *mem,
                          struct mem_region *region, struct block *block)
{
    struct rpc_heap_release release;
    struct list_head *prev;
    struct list_head *next;
    size_t size;
    long status;

    if (!region->dynamic || block != region->first ||
        block_size(next_block(block)) != 0)
        return 0;

    release.addr = region->raw;
    release.size = region->raw_size;
    prev = region->node.prev;
    next = region->node.next;
    size = block_size(block);
    status = rpc_call(RPC_HEAP_RELEASE,
                      &release, sizeof(release), 0, 0);
    if (status < 0)
        return 0;

    /* The host may reclaim region immediately after the successful call. */
    __list_del(prev, next);
    mem->capacity -= size;
    mem->available -= size;
    return 1;
}

int mem_init(void *base, size_t size)
{
    struct mem_state *mem = &cpu_self()->mem;

    mem->free = RB_ROOT;
    INIT_LIST_HEAD(&mem->regions);
    mem->capacity = 0;
    mem->available = 0;
    return region_add(mem, (uintptr_t)base, size, 0);
}

void *mem_alloc(size_t size)
{
    struct mem_state *mem = &cpu_self()->mem;
    struct block *block;
    size_t need;
    size_t have;

    if (!size || size > SIZE_MAX - sizeof(struct block) - ALIGNMENT)
        return 0;
    need = ALIGN_UP(size + sizeof(struct block));
    if (need < MIN_BLOCK)
        need = MIN_BLOCK;

    block = tree_best(mem, need);
    if (!block) {
        if (region_grow(mem, need) < 0)
            return 0;
        block = tree_best(mem, need);
        if (!block)
            return 0;
    }

    tree_remove(mem, block);
    have = block_size(block);
    if (have - need >= MIN_BLOCK) {
        struct block *rest = (struct block *)((unsigned char *)block + need);
        struct block *next;

        rest->size = have - need;
        rest->prev_size = need;
        next = next_block(rest);
        next->prev_size = block_size(rest);
        block->size = need | USED;
        tree_add(mem, rest);
        mem->available -= need;
    } else {
        block->size = have | USED;
        next_block(block)->prev_size = have;
        mem->available -= have;
    }
    return block + 1;
}

void mem_free(void *ptr)
{
    struct mem_state *mem = &cpu_self()->mem;
    struct mem_region *region;
    struct block *block;
    struct block *next;
    size_t released;

    if (!ptr || ((uintptr_t)ptr & (ALIGNMENT - 1)))
        return;
    region = find_region(mem, ptr);
    if (!region)
        return;

    block = ((struct block *)ptr) - 1;
    if (!block_used(block) || block_size(block) < MIN_BLOCK)
        return;
    released = block_size(block);
    block->size &= ~USED;

    if (block->prev_size) {
        struct block *prev = (struct block *)
            ((unsigned char *)block - block->prev_size);

        if (!block_used(prev)) {
            tree_remove(mem, prev);
            prev->size = block_size(prev) + block_size(block);
            block = prev;
        }
    }

    next = next_block(block);
    if (!block_used(next)) {
        tree_remove(mem, next);
        block->size = block_size(block) + block_size(next);
        next = next_block(block);
    }
    next->prev_size = block_size(block);
    mem->available += released;

    if (!release_region(mem, region, block))
        tree_add(mem, block);
}

size_t mem_size(const void *ptr)
{
    struct mem_state *mem = &cpu_self()->mem;
    struct mem_region *region;
    const struct block *block;

    if (!ptr || ((uintptr_t)ptr & (ALIGNMENT - 1)))
        return 0;
    region = find_region(mem, ptr);
    if (!region)
        return 0;
    block = ((const struct block *)ptr) - 1;
    if (!block_used(block) || block_size(block) < sizeof(*block))
        return 0;
    return block_size(block) - sizeof(*block);
}

void *mem_resize(void *ptr, size_t size)
{
    size_t old_size;
    void *new_ptr;

    if (!ptr)
        return mem_alloc(size);
    if (!size) {
        mem_free(ptr);
        return 0;
    }
    old_size = mem_size(ptr);
    if (!old_size)
        return 0;
    if (size <= old_size)
        return ptr;

    new_ptr = mem_alloc(size);
    if (!new_ptr)
        return 0;
    memcpy(new_ptr, ptr, old_size);
    mem_free(ptr);
    return new_ptr;
}

size_t mem_capacity(void)
{
    return cpu_self()->mem.capacity;
}

size_t mem_available(void)
{
    return cpu_self()->mem.available;
}
