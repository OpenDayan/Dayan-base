#include "runtime_internal.h"

#include <errno.h>
#include <stdlib.h>

int npu_is_power_of_two(size_t value)
{
    return value && !(value & (value - 1U));
}

void npu_arena_init(struct npu_arena *arena, uint64_t base, uint64_t size)
{
    arena->base = base;
    arena->size = size;
    arena->allocations = 0;
}

void npu_arena_destroy(struct npu_arena *arena)
{
    struct npu_allocation *allocation = arena->allocations;

    while (allocation) {
        struct npu_allocation *next = allocation->next;

        free(allocation);
        allocation = next;
    }
    arena->allocations = 0;
}

static int arena_contains(const struct npu_arena *arena,
                          uint64_t address, uint64_t size)
{
    return size && size <= arena->size && address >= arena->base &&
           address - arena->base <= arena->size - size;
}

static int align_up(uint64_t value, uint64_t alignment, uint64_t *result)
{
    uint64_t mask = alignment - 1U;

    if (value > UINT64_MAX - mask)
        return -EOVERFLOW;
    *result = (value + mask) & ~mask;
    return 0;
}

static int insert_allocation(struct npu_arena *arena, uint64_t address,
                             uint64_t size, uint32_t kind, uint32_t owner,
                             struct npu_allocation **position)
{
    struct npu_allocation *allocation = malloc(sizeof(*allocation));

    if (!allocation)
        return -ENOMEM;
    allocation->address = address;
    allocation->size = size;
    allocation->kind = kind;
    allocation->owner = owner;
    allocation->next = *position;
    *position = allocation;
    (void)arena;
    return 0;
}

int npu_arena_alloc(struct npu_arena *arena, uint64_t size,
                    uint64_t alignment, uint32_t kind, uint32_t owner,
                    uint64_t *address_out)
{
    struct npu_allocation **position = &arena->allocations;
    uint64_t candidate;
    uint64_t limit;
    int status;

    if (!arena || !address_out || !size || !alignment ||
        (alignment & (alignment - 1U)) ||
        arena->size > UINT64_MAX - arena->base)
        return -EINVAL;
    limit = arena->base + arena->size;
    status = align_up(arena->base, alignment, &candidate);
    if (status < 0)
        return status;
    while (*position) {
        struct npu_allocation *allocation = *position;

        if (candidate <= allocation->address &&
            size <= allocation->address - candidate)
            break;
        if (allocation->address > UINT64_MAX - allocation->size)
            return -EOVERFLOW;
        status = align_up(allocation->address + allocation->size,
                          alignment, &candidate);
        if (status < 0)
            return status;
        position = &allocation->next;
    }
    if (candidate > limit || size > limit - candidate)
        return -ENOMEM;
    status = insert_allocation(arena, candidate, size, kind, owner, position);
    if (status < 0)
        return status;
    *address_out = candidate;
    return 0;
}

int npu_arena_reserve(struct npu_arena *arena, uint64_t address,
                      uint64_t size, uint32_t kind, uint32_t owner)
{
    struct npu_allocation **position;

    if (!arena || !arena_contains(arena, address, size))
        return -EINVAL;
    position = &arena->allocations;
    while (*position && (*position)->address < address) {
        if ((*position)->size > address - (*position)->address)
            return -EBUSY;
        position = &(*position)->next;
    }
    if (*position && size > (*position)->address - address)
        return -EBUSY;
    return insert_allocation(arena, address, size, kind, owner, position);
}

int npu_arena_free(struct npu_arena *arena, uint64_t address,
                   uint32_t kind, uint32_t owner)
{
    struct npu_allocation **position;

    if (!arena)
        return -EINVAL;
    position = &arena->allocations;
    while (*position && (*position)->address != address)
        position = &(*position)->next;
    if (!*position)
        return -ENOENT;
    if ((kind && (*position)->kind != kind) ||
        (owner != UINT32_MAX && (*position)->owner != owner))
        return -EPERM;
    {
        struct npu_allocation *allocation = *position;

        *position = allocation->next;
        free(allocation);
    }
    return 0;
}
