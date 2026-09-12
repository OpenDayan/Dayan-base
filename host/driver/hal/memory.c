#include "hal_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define NPU_DEFAULT_RAM_BASE      UINT64_C(0x80000000)
#define NPU_DEFAULT_RAM_SIZE      UINT64_C(0x100000000) // 4 GiB default
#define NPU_DEFAULT_COHERENT_BASE UINT64_C(0x00400000)
#define NPU_DEFAULT_COHERENT_SIZE UINT64_C(0x00400000)
#define NPU_DEFAULT_URAM_BASE     UINT64_C(0x00800000)
#define NPU_DEFAULT_URAM_SIZE     UINT64_C(0x00180000)
#define NPU_DEFAULT_URAM_STRIDE   UINT64_C(0x00200000)
#define NPU_URAM_GRANULE          UINT64_C(32768)

/*
 * The Normal region is split into two disjoint physical pools: everything the
 * firmware itself owns at the bottom, Buffer backing above it.
 *
 * This split is what lets the MMU map the firmware window with a linear offset
 * while Buffers get non-identity virtual addresses, without any physical page
 * being reachable through two different virtual addresses.  A shared pool
 * would reintroduce identity aliases for Buffer pages, and an out-of-bounds
 * Buffer access would then land on a legally mapped address instead of
 * faulting.
 *
 * The firmware pool goes at the bottom so that in the default configuration
 * its base coincides with the region base, which is also the firmware's link
 * address (device/config.h CONFIG_RAM_BASE): the linear offset degenerates to
 * an identity and the firmware occupies the same physical addresses whether
 * the MMU is on or off.  Putting it at the top instead would move the firmware
 * whenever --normal-memory-size-mib changed, which is a needless dependency -
 * the physical address of a fixed image should not follow the size of memory
 * it never touches.  The offset itself is still honoured, because a caller
 * supplying its own memory_regions can base the Normal region anywhere.
 *
 * The boundary is aligned to 2 MiB so the firmware pool can be mapped with
 * 2 MiB leaves.  Because the pool now starts at the region base, that base has
 * to be 2 MiB-aligned for a split to happen at all.
 *
 * The firmware pool is capped rather than proportional.  --normal-memory-size-mib
 * can make the Normal region gigabytes wide, and none of that helps the
 * firmware: the pool only has to hold the BL1 image, its heap and stacks, the
 * page tables and the guard pages, and the firmware cannot address past its own
 * linker window (device/config.h CONFIG_RAM_SIZE) whatever we map.  Splitting
 * proportionally would strand half of a 4 GiB region where Buffers can never
 * reach it.  A region too small for the cap is halved instead.
 *
 * None of this is wanted when the MMU is off.  With FixMap identity-mapping
 * everything there are no virtual aliases to keep apart, and the split would
 * only cost every non-Buffer usage - heap, kernels, the profile buffer - the
 * whole Buffer pool it can no longer allocate from, for no benefit.  So it is
 * conditional on npu_hal_options_t::split_normal_memory, which dayan_server
 * sets from --mmu.
 */
#define NPU_FIRMWARE_POOL_ALIGN   UINT64_C(0x200000)
#define NPU_FIRMWARE_POOL_MAX     UINT64_C(0x04000000)

static int power_of_two(uint64_t value)
{
    return value && !(value & (value - 1U));
}

/*
 * Firmware-owned memory comes from below the split, Buffer backing from above.
 * Every Normal usage other than BUFFER is memory the firmware reaches through
 * the linear-offset window, so it has to sit in the firmware pool.  The two
 * usages that are never Normal - KERNEL_URAM (URAM) and ATOMIC_CACHE_BACKING
 * (Coherent) - are unaffected: their regions are never split.
 */
static int usage_is_firmware_pool(uint32_t usage)
{
    return usage != NPU_HAL_MEMORY_BUFFER;
}

/*
 * Size of the firmware pool, which starts at the region base; the Buffer pool
 * is the remainder above it.  Zero means the region is not split - either the
 * MMU is off and nothing needs it, or the region is too small or misaligned, in
 * which case the MMU falls back to mapping all of it as the firmware window.
 */
static uint64_t normal_split_offset(const npu_hal_t *hal,
                                    const npu_hal_memory_region_t *region)
{
    uint64_t reserve;

    if (!hal->split_normal ||
        region->memory_class != NPU_HAL_MEMORY_NORMAL ||
        region->size < 2U * NPU_FIRMWARE_POOL_ALIGN ||
        (region->base & (NPU_FIRMWARE_POOL_ALIGN - 1U)))
        return 0;
    /* Take the cap when the region can spare it, half the region otherwise, so
     * a small region still leaves the firmware room and a large one still
     * leaves nearly all of itself to Buffers. */
    reserve = region->size / 2U;
    if (reserve > NPU_FIRMWARE_POOL_MAX)
        reserve = NPU_FIRMWARE_POOL_MAX;
    /* Round the size down to a 2 MiB multiple.  The base is already aligned, so
     * this is what makes the pool an exact number of 2 MiB leaves; rounding
     * down rather than up keeps the cap from being exceeded. */
    reserve &= ~(NPU_FIRMWARE_POOL_ALIGN - 1U);
    if (!reserve || reserve >= region->size)
        return 0;
    return reserve;
}

static int region_valid(const npu_hal_t *hal,
                        const npu_hal_memory_region_t *region)
{
    if (!region->size || region->base > UINT64_MAX - region->size ||
        (region->memory_class != NPU_HAL_MEMORY_NORMAL &&
         region->memory_class != NPU_HAL_MEMORY_COHERENT &&
         region->memory_class != NPU_HAL_MEMORY_URAM))
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
    npu_hal_memory_region_t *defaults = 0;
    const npu_hal_memory_region_t *regions = options->memory_regions;
    uint64_t normal_size = options->normal_memory_size_mib ?
        (uint64_t)options->normal_memory_size_mib * UINT64_C(1048576) :
        NPU_DEFAULT_RAM_SIZE;
    uint32_t count = options->memory_region_count;
    uint32_t index;

    if (!count) {
        uint32_t cluster;
        uint32_t core;

        count = 2U + hal->cluster_count * hal->cores_per_cluster;
        defaults = calloc(count, sizeof(*defaults));
        if (!defaults)
            return -ENOMEM;
        defaults[0] = (npu_hal_memory_region_t){
            NPU_DEFAULT_RAM_BASE, normal_size,
            NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_DEVICE,
            NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE
        };
        defaults[1] = (npu_hal_memory_region_t){
            NPU_DEFAULT_COHERENT_BASE, NPU_DEFAULT_COHERENT_SIZE,
            NPU_HAL_MEMORY_COHERENT, NPU_HAL_MEMORY_DEVICE,
            NPU_HAL_ANY_CLUSTER, NPU_HAL_ANY_CORE
        };
        index = 2;
        for (cluster = 0; cluster < hal->cluster_count; ++cluster) {
            for (core = 0; core < hal->cores_per_cluster; ++core) {
                defaults[index++] = (npu_hal_memory_region_t){
                    NPU_DEFAULT_URAM_BASE + core * NPU_DEFAULT_URAM_STRIDE,
                    NPU_DEFAULT_URAM_SIZE, NPU_HAL_MEMORY_URAM,
                    NPU_HAL_MEMORY_CORE, cluster, core
                };
            }
        }
        regions = defaults;
    }
    hal->arenas = calloc(count, sizeof(*hal->arenas));
    if (!hal->arenas) {
        free(defaults);
        return -ENOMEM;
    }
    hal->arena_count = count;
    hal->split_normal = !!options->split_normal_memory;
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
    free(defaults);
    return 0;

invalid:
    free(defaults);
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
            request->memory_class == NPU_HAL_MEMORY_COHERENT ||
           request->memory_class == NPU_HAL_MEMORY_URAM) &&
           request->usage >= NPU_HAL_MEMORY_BUFFER &&
           request->usage <= NPU_HAL_MEMORY_ATOMIC_CACHE_BACKING &&
           !request->flags &&
           request->size && power_of_two(request->alignment) &&
           request_location_valid(hal, request) &&
           (request->memory_class == NPU_HAL_MEMORY_URAM ?
                (request->scope == NPU_HAL_MEMORY_CORE &&
                 request->usage == NPU_HAL_MEMORY_KERNEL_URAM &&
                 request->alignment >= NPU_URAM_GRANULE &&
                 !(request->size & (NPU_URAM_GRANULE - 1U))) :
                request->usage != NPU_HAL_MEMORY_KERNEL_URAM);
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
    uint64_t split = normal_split_offset(hal, &arena->region);
    uint64_t floor = arena->region.base;
    uint64_t limit = arena->region.base + arena->region.size;
    uint64_t cursor;

    /*
     * Confine the search to this usage's pool.  Both pools keep using the one
     * allocation list, ordered by address, so the walk below is unchanged -
     * only the window it may return from narrows.
     */
    if (split) {
        if (usage_is_firmware_pool(request->usage))
            limit = arena->region.base + split;
        else
            floor = arena->region.base + split;
    }
    cursor = floor;

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
        /*
         * The list spans both pools, so stepping over an allocation in the
         * other one would otherwise pull the cursor back below this pool's
         * base and hand out an address from the wrong side of the split.
         */
        if (cursor < floor)
            cursor = floor;
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

int npu_hal_firmware_pool(npu_hal_t *hal, uint64_t *base_out,
                          uint64_t *size_out)
{
    uint32_t index;
    int status = -ENODEV;

    if (!hal || !base_out || !size_out)
        return -EINVAL;
    /* Nothing was split, so there is no pool to report - distinct from a region
     * that could not be split, which -ENOENT below reports along with the whole
     * region the caller may map instead. */
    if (!hal->split_normal)
        return -ENODEV;
    pthread_mutex_lock(&hal->memory_mutex);
    for (index = 0; index < hal->arena_count; ++index) {
        const npu_hal_memory_region_t *region = &hal->arenas[index].region;
        uint64_t split;

        if (region->memory_class != NPU_HAL_MEMORY_NORMAL ||
            region->scope != NPU_HAL_MEMORY_DEVICE)
            continue;
        split = normal_split_offset(hal, region);
        if (!split) {
            /* Unsplit region: the caller maps all of it. */
            *base_out = region->base;
            *size_out = region->size;
            status = -ENOENT;
            break;
        }
        *base_out = region->base;
        *size_out = split;
        status = 0;
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
        uint64_t split;
        uint64_t floor;
        uint64_t limit;

        if (!arena_matches(arena, request))
            continue;
        floor = arena->region.base;
        limit = arena->region.base + arena->region.size;
        /*
         * A fixed address has to land in its usage's pool, exactly as an
         * ordinary allocation would.  Bounds-checking against the whole region
         * instead would let a caller reserve firmware memory inside the Buffer
         * pool - or a range straddling the boundary - and the disjointness the
         * split guarantees would hold for allocate() but not for reserve().
         */
        split = normal_split_offset(hal, &arena->region);
        if (split) {
            if (usage_is_firmware_pool(request->usage))
                limit = arena->region.base + split;
            else
                floor = arena->region.base + split;
        }
        if (request->fixed_address < floor ||
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
