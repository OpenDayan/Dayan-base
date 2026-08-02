#include "hal_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define NPU_DEFAULT_RAM_BASE      UINT64_C(0x80000000)
#define NPU_DEFAULT_RAM_SIZE      UINT64_C(0x04000000)
#define NPU_DEFAULT_COHERENT_BASE UINT64_C(0x00400000)
#define NPU_DEFAULT_COHERENT_SIZE UINT64_C(0x00400000)

static int power_of_two(uint64_t value)
{
    return value && !(value & (value - 1U));
}

static int region_valid(const npu_hal_t *hal,
                        const npu_hal_memory_region_t *region)
{
    if (!region->size || region->base > UINT64_MAX - region->size ||
        (region->memory_class != NPU_HAL_MEMORY_NORMAL &&
         region->memory_class != NPU_HAL_MEMORY_COHERENT))
        return 0;
    switch (region->scope) {
    case NPU_HAL_MEMORY_DEVICE:
        return region->cluster == NPU_HAL_ANY_CLUSTER &&
               region->core == NPU_HAL_ANY_CORE;
    case NPU_HAL_MEMORY_CLUSTER:
        return region->cluster < hal->cluster_count &&
               region->core == NPU_HAL_ANY_CORE;
    case NPU_HAL_MEMORY_CORE:
        return region->cluster < hal->cluster_count &&
               region->core < hal->cores_per_cluster;
    default:
        return 0;
    }
}

static int shared_address_space(const npu_hal_memory_region_t *left,
                                const npu_hal_memory_region_t *right)
{
    if (left->scope == NPU_HAL_MEMORY_DEVICE ||
        right->scope == NPU_HAL_MEMORY_DEVICE)
        return 1;
    return left->cluster == right->cluster;
}

int npu_hal_memory_manager_init(npu_hal_t *hal,
                                const npu_hal_options_t *options)
{
    static const npu_hal_memory_region_t defaults[] = {
        { NPU_DEFAULT_RAM_BASE, NPU_DEFAULT_RAM_SIZE,
          NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_DEVICE,
          NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
        { NPU_DEFAULT_COHERENT_BASE, NPU_DEFAULT_COHERENT_SIZE,
          NPU_HAL_MEMORY_COHERENT, NPU_HAL_MEMORY_DEVICE,
          NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE },
    };
    const npu_hal_memory_region_t *regions = options->memory_regions;
    uint32_t count = options->memory_region_count;
    uint32_t index;

    if (!count) {
        regions = defaults;
        count = sizeof(defaults) / sizeof(defaults[0]);
    }
    hal->arenas = calloc(count, sizeof(*hal->arenas));
    if (!hal->arenas)
        return -ENOMEM;
    hal->arena_count = count;
    for (index = 0; index < count; ++index) {
        uint32_t previous;

        if (!region_valid(hal, &regions[index]))
            goto invalid;
        for (previous = 0; previous < index; ++previous) {
            const npu_hal_memory_region_t *other =
                &hal->arenas[previous].region;

            if (shared_address_space(&regions[index], other) &&
                regions[index].base < other->base + other->size &&
                other->base < regions[index].base + regions[index].size)
                goto invalid;
        }
        hal->arenas[index].region = regions[index];
    }
    return 0;

invalid:
    free(hal->arenas);
    hal->arenas = 0;
    hal->arena_count = 0;
    return -EINVAL;
}

void npu_hal_memory_manager_destroy(npu_hal_t *hal)
{
    uint32_t index;

    if (!hal)
        return;
    for (index = 0; index < hal->arena_count; ++index) {
        struct npu_hal_allocation *allocation =
            hal->arenas[index].allocations;

        while (allocation) {
            struct npu_hal_allocation *next = allocation->next;

            free(allocation);
            allocation = next;
        }
    }
    free(hal->arenas);
    hal->arenas = 0;
    hal->arena_count = 0;
}

static int request_location_valid(const npu_hal_t *hal,
                                  const npu_hal_memory_request_t *request)
{
    switch (request->scope) {
    case NPU_HAL_MEMORY_DEVICE:
        return request->location.cluster == NPU_HAL_ANY_CLUSTER &&
               request->location.core == NPU_HAL_ANY_CORE;
    case NPU_HAL_MEMORY_CLUSTER:
        return request->location.cluster < hal->cluster_count &&
               request->location.core == NPU_HAL_ANY_CORE;
    case NPU_HAL_MEMORY_CORE:
        return request->location.cluster < hal->cluster_count &&
               request->location.core < hal->cores_per_cluster;
    default:
        return 0;
    }
}

static int request_valid(const npu_hal_t *hal,
                         const npu_hal_memory_request_t *request)
{
    return request && request->version == NPU_HAL_MEMORY_REQUEST_VERSION &&
           (request->memory_class == NPU_HAL_MEMORY_NORMAL ||
            request->memory_class == NPU_HAL_MEMORY_COHERENT) &&
           request->usage >= NPU_HAL_MEMORY_BUFFER &&
           request->usage <= NPU_HAL_MEMORY_SHARED && !request->flags &&
           request->size && power_of_two(request->alignment) &&
           request_location_valid(hal, request);
}

static int arena_matches(const struct npu_hal_arena *arena,
                         const npu_hal_memory_request_t *request)
{
    return arena->region.memory_class == request->memory_class &&
           arena->region.scope == request->scope &&
           arena->region.cluster == request->location.cluster &&
           arena->region.core == request->location.core;
}

static int align_address(uint64_t address, uint64_t alignment,
                         uint64_t *aligned)
{
    uint64_t mask = alignment - 1U;

    if (address > UINT64_MAX - mask)
        return -EOVERFLOW;
    *aligned = (address + mask) & ~mask;
    return 0;
}

static int insert_allocation(struct npu_hal_arena *arena,
                             npu_hal_t *hal,
                             const npu_hal_memory_request_t *request,
                             uint64_t address,
                             npu_hal_allocation_t **allocation_out)
{
    struct npu_hal_allocation **link = &arena->allocations;
    struct npu_hal_allocation *allocation;

    while (*link && (*link)->address < address)
        link = &(*link)->next;
    allocation = calloc(1, sizeof(*allocation));
    if (!allocation)
        return -ENOMEM;
    allocation->hal = hal;
    allocation->address = address;
    allocation->size = request->size;
    allocation->memory_class = request->memory_class;
    allocation->scope = request->scope;
    allocation->usage = request->usage;
    allocation->owner = request->owner;
    allocation->location = request->location;
    allocation->arena = arena;
    allocation->next = *link;
    *link = allocation;
    *allocation_out = allocation;
    return 0;
}

/* Find a gap without relying on pointer-to-link container arithmetic. */
static int allocate_from_arena(struct npu_hal_arena *arena,
                               npu_hal_t *hal,
                               const npu_hal_memory_request_t *request,
                               npu_hal_allocation_t **allocation_out)
{
    struct npu_hal_allocation *current = arena->allocations;
    struct npu_hal_allocation *previous = 0;
    uint64_t cursor = arena->region.base;
    uint64_t limit = arena->region.base + arena->region.size;

    while (1) {
        uint64_t aligned;
        int status = align_address(cursor, request->alignment, &aligned);

        if (status < 0 || aligned > limit || request->size > limit - aligned)
            return -ENOMEM;
        if (!current || aligned + request->size <= current->address) {
            struct npu_hal_allocation *allocation =
                calloc(1, sizeof(*allocation));

            if (!allocation)
                return -ENOMEM;
            allocation->hal = hal;
            allocation->address = aligned;
            allocation->size = request->size;
            allocation->memory_class = request->memory_class;
            allocation->scope = request->scope;
            allocation->usage = request->usage;
            allocation->owner = request->owner;
            allocation->location = request->location;
            allocation->arena = arena;
            allocation->next = current;
            if (previous)
                previous->next = allocation;
            else
                arena->allocations = allocation;
            *allocation_out = allocation;
            return 0;
        }
        if (current->address > UINT64_MAX - current->size)
            return -EOVERFLOW;
        cursor = current->address + current->size;
        previous = current;
        current = current->next;
    }
}

int npu_hal_memory_alloc(npu_hal_t *hal,
                         const npu_hal_memory_request_t *request,
                         npu_hal_allocation_t **allocation_out)
{
    uint32_t index;
    int status = -ENOMEM;

    if (!hal || !allocation_out || !request_valid(hal, request) ||
        request->fixed_address)
        return -EINVAL;
    *allocation_out = 0;
    pthread_mutex_lock(&hal->memory_mutex);
    for (index = 0; index < hal->arena_count; ++index) {
        if (!arena_matches(&hal->arenas[index], request))
            continue;
        status = allocate_from_arena(&hal->arenas[index], hal, request,
                                     allocation_out);
        if (status == 0 || status != -ENOMEM)
            break;
    }
    pthread_mutex_unlock(&hal->memory_mutex);
    return status;
}

int npu_hal_memory_reserve(npu_hal_t *hal,
                           const npu_hal_memory_request_t *request,
                           npu_hal_allocation_t **allocation_out)
{
    uint32_t index;
    int status = -ENOMEM;

    if (!hal || !allocation_out || !request_valid(hal, request) ||
        request->fixed_address > UINT64_MAX - request->size)
        return -EINVAL;
    *allocation_out = 0;
    pthread_mutex_lock(&hal->memory_mutex);
    for (index = 0; index < hal->arena_count; ++index) {
        struct npu_hal_arena *arena = &hal->arenas[index];
        uint64_t limit;

        if (!arena_matches(arena, request))
            continue;
        limit = arena->region.base + arena->region.size;
        if (request->fixed_address < arena->region.base ||
            request->fixed_address > limit || request->size >
                limit - request->fixed_address ||
            (request->fixed_address & (request->alignment - 1U))) {
            status = -ERANGE;
            continue;
        }
        {
            struct npu_hal_allocation *current = arena->allocations;
            struct npu_hal_allocation *previous = 0;

            while (current && current->address < request->fixed_address) {
                previous = current;
                current = current->next;
            }
            if ((previous && previous->address + previous->size >
                             request->fixed_address) ||
                (current && request->fixed_address + request->size >
                            current->address)) {
                status = -EEXIST;
                continue;
            }
            status = insert_allocation(arena, hal, request,
                                       request->fixed_address,
                                       allocation_out);
        }
        break;
    }
    pthread_mutex_unlock(&hal->memory_mutex);
    return status;
}

int npu_hal_memory_free(npu_hal_t *hal, npu_hal_allocation_t *allocation)
{
    struct npu_hal_allocation **link;

    if (!hal || !allocation || allocation->hal != hal || !allocation->arena)
        return -EINVAL;
    pthread_mutex_lock(&hal->memory_mutex);
    link = &allocation->arena->allocations;
    while (*link && *link != allocation)
        link = &(*link)->next;
    if (!*link) {
        pthread_mutex_unlock(&hal->memory_mutex);
        return -ENOENT;
    }
    *link = allocation->next;
    pthread_mutex_unlock(&hal->memory_mutex);
    free(allocation);
    return 0;
}

int npu_hal_memory_info(npu_hal_t *hal,
                        const npu_hal_allocation_t *allocation,
                        npu_hal_allocation_info_t *info)
{
    if (!hal || !allocation || allocation->hal != hal || !info)
        return -EINVAL;
    info->address = allocation->address;
    info->size = allocation->size;
    info->memory_class = allocation->memory_class;
    info->scope = allocation->scope;
    info->usage = allocation->usage;
    info->owner = allocation->owner;
    info->location = allocation->location;
    return 0;
}

int npu_hal_memory_visible(npu_hal_t *hal,
                           const npu_hal_allocation_t *allocation,
                           npu_hal_core_t viewer)
{
    if (!hal || !allocation || allocation->hal != hal ||
        !npu_hal_core_valid(hal, viewer))
        return 0;
    if (allocation->scope == NPU_HAL_MEMORY_DEVICE)
        return 1;
    if (allocation->location.cluster != viewer.cluster)
        return 0;
    return allocation->scope == NPU_HAL_MEMORY_CLUSTER ||
           allocation->location.core == viewer.core;
}

int npu_hal_address_valid(npu_hal_t *hal, npu_hal_core_t viewer,
                          uint64_t address, uint64_t size)
{
    uint32_t index;

    if (!hal || !npu_hal_core_valid(hal, viewer) ||
        address > UINT64_MAX - size)
        return 0;
    for (index = 0; index < hal->arena_count; ++index) {
        const npu_hal_memory_region_t *region = &hal->arenas[index].region;
        int visible = region->scope == NPU_HAL_MEMORY_DEVICE ||
            (region->cluster == viewer.cluster &&
             (region->scope == NPU_HAL_MEMORY_CLUSTER ||
              region->core == viewer.core));

        if (visible && address >= region->base &&
            size <= region->size && address - region->base <=
                region->size - size)
            return 1;
    }
    return 0;
}

int npu_hal_memory_address(npu_hal_t *hal,
                           const npu_hal_allocation_t *allocation,
                           npu_hal_core_t viewer, uint64_t *address_out)
{
    if (!hal || !allocation || !address_out ||
        !npu_hal_memory_visible(hal, allocation, viewer))
        return -EACCES;
    *address_out = allocation->address;
    return 0;
}

int npu_hal_memory_write(npu_hal_t *hal,
                         const npu_hal_allocation_t *allocation,
                         npu_hal_core_t viewer, uint64_t offset,
                         const void *source, size_t size)
{
    uint64_t address;
    int status;

    if (!allocation || offset > allocation->size ||
        size > allocation->size - offset)
        return -ERANGE;
    status = npu_hal_memory_address(hal, allocation, viewer, &address);
    if (status < 0 || address > UINT64_MAX - offset)
        return status < 0 ? status : -EOVERFLOW;
    return npu_hal_copy_to_address(hal, viewer.cluster, address + offset,
                                   source, size);
}

int npu_hal_memory_read(npu_hal_t *hal,
                        const npu_hal_allocation_t *allocation,
                        npu_hal_core_t viewer, void *destination,
                        uint64_t offset, size_t size)
{
    uint64_t address;
    int status;

    if (!allocation || offset > allocation->size ||
        size > allocation->size - offset)
        return -ERANGE;
    status = npu_hal_memory_address(hal, allocation, viewer, &address);
    if (status < 0 || address > UINT64_MAX - offset)
        return status < 0 ? status : -EOVERFLOW;
    return npu_hal_copy_from_address(hal, viewer.cluster, destination,
                                     address + offset, size);
}
