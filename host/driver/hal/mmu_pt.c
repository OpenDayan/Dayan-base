/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

/*
 * ClusterMMU Sv39 page tables and guard pages.
 *
 * This is the mechanism half of the MMU: it knows the hardware's page-table
 * format and bring-up rules and nothing about Contexts, Buffers or firmware.
 * Which virtual addresses exist, where the firmware window sits and when a
 * TLB flush is safe are the Service's decisions (server/mmu.c); this file
 * only installs what it is told to.
 *
 * Every mapping is a 2 MiB level-1 leaf, so only root and level-1 tables
 * exist.  DAYAN_MMU_PAGE_SHIFT (12) is still the PPN unit and the size of a
 * table.
 *
 * A Cluster has one page-table walker base and one ASID, so there is one
 * table per Cluster, shared by everything mapped into it.
 *
 * A fault does not abort the access.  The MMU substitutes the whole master
 * address with a per-class redirect base and the access completes there.
 * There are no fault-status or fault-address registers, so the only way to
 * observe a fault is to point the redirect bases at pages filled with a known
 * pattern and inspect them afterwards.  That is what the guard pages are.
 *
 * Nothing here flushes the TLB.  Writing TLB_FLUSH while any AXI port still
 * has traffic in flight hangs the Cluster permanently, and only the caller
 * can know the Cluster is idle; see npu_hal_mmu_flush().
 *
 * The page tables live in device memory, which the Host cannot dereference.
 * Every table therefore keeps a Host-side shadow copy; writes go to the
 * shadow and are pushed to the device with npu_hal_copy_to_address().
 *
 * Only public HAL calls are used, so this file can be linked on its own
 * against a fake HAL in tests (hal/tests/mmu_pt_test.c).
 */

#include "hal.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "platform_regs.h"

#define NPU_MMU_PTE_V           UINT64_C(0x001)
#define NPU_MMU_PTE_R           UINT64_C(0x002)
#define NPU_MMU_PTE_W           UINT64_C(0x004)
#define NPU_MMU_PTE_X           UINT64_C(0x008)
#define NPU_MMU_PTE_U           UINT64_C(0x010)
#define NPU_MMU_PTE_G           UINT64_C(0x020)
#define NPU_MMU_PTE_A           UINT64_C(0x040)
#define NPU_MMU_PTE_D           UINT64_C(0x080)

#define NPU_MMU_PTE_PPN_SHIFT   10U
#define NPU_MMU_PTE_PER_TABLE   512U
#define NPU_MMU_VPN_BITS        9U
#define NPU_MMU_LEVELS          3U
#define NPU_MMU_TABLE_BYTES     (NPU_MMU_PTE_PER_TABLE * sizeof(uint64_t))

/* One level-1 Sv39 entry spans 2 MiB; a level-1 leaf maps all of it. */
#define NPU_MMU_LEAF_LEVEL      1U

/* Guard-page fill: recognisable, and encodes its own offset so a corrupted
 * word tells us which VA[11:0] the faulting access carried. */
#define NPU_MMU_POISON_BASE     UINT64_C(0xdead000000000000)

/*
 * One Sv39 table.  @entries is a Host shadow of the device-resident table at
 * @device_address.  @children mirrors the interior links so a walk never has
 * to read them back.
 */
struct npu_mmu_table {
    uint64_t device_address;
    npu_hal_allocation_t *allocation;
    uint64_t *entries;
    struct npu_mmu_table **children;
};

struct npu_mmu_cluster {
    struct npu_mmu_table *root;
    npu_hal_allocation_t *guard_allocation[DAYAN_MMU_REDIR_CLASS_COUNT];
    uint64_t guard_address[DAYAN_MMU_REDIR_CLASS_COUNT];
};

struct npu_hal_mmu {
    npu_hal_t *hal;
    pthread_mutex_t lock;
    uint32_t cluster_count;
    uint32_t active;
    /*
     * Where tables and guard pages come from, handed out top-down.  The
     * caller picks the range; the cursor moves only once a reservation
     * succeeds.
     */
    uint64_t pool_base;
    uint64_t pool_size;
    uint64_t pool_cursor;
    struct npu_mmu_cluster *clusters;
};

static uint64_t npu_mmu_poison_word(uint64_t offset)
{
    return NPU_MMU_POISON_BASE | offset;
}

static uint32_t npu_mmu_vpn(uint64_t va, unsigned level)
{
    return (uint32_t)((va >> (DAYAN_MMU_PAGE_SHIFT +
                              level * NPU_MMU_VPN_BITS)) &
                      (NPU_MMU_PTE_PER_TABLE - 1U));
}

/*
 * A and D are preset rather than left to the hardware.  The ClusterMMU does
 * not update them, so a leaf with A clear would fault forever.
 */
static uint64_t npu_mmu_pte_make(uint64_t pa, uint32_t prot, int leaf)
{
    uint64_t pte = NPU_MMU_PTE_V |
                   ((pa >> DAYAN_MMU_PAGE_SHIFT) << NPU_MMU_PTE_PPN_SHIFT);

    if (!leaf)
        return pte;
    pte |= NPU_MMU_PTE_A | NPU_MMU_PTE_D | NPU_MMU_PTE_U | NPU_MMU_PTE_G;
    if (prot & NPU_HAL_MMU_READ)
        pte |= NPU_MMU_PTE_R;
    if (prot & NPU_HAL_MMU_WRITE)
        pte |= NPU_MMU_PTE_W;
    if (prot & NPU_HAL_MMU_EXEC)
        pte |= NPU_MMU_PTE_X;
    return pte;
}

static uint64_t npu_mmu_pte_pa(uint64_t pte)
{
    return (pte >> NPU_MMU_PTE_PPN_SHIFT) << DAYAN_MMU_PAGE_SHIFT;
}

/* ---------------------------------------------------------------- */
/* Page tables                                                       */
/* ---------------------------------------------------------------- */

/*
 * Push one shadow entry to the device copy of the table.  Tables are written
 * a word at a time rather than wholesale: a full-table push would rewrite
 * entries the walker may be reading concurrently.
 */
static int npu_mmu_push_entry(npu_hal_mmu_t *mmu, uint32_t cluster,
                              const struct npu_mmu_table *table,
                              uint32_t index)
{
    return npu_hal_copy_to_address(mmu->hal, cluster,
                                   table->device_address +
                                   index * sizeof(uint64_t),
                                   &table->entries[index],
                                   sizeof(table->entries[index]));
}

/*
 * One page of Device memory for page tables and guard pages, taken from the
 * top of the caller's pool and working downwards.  The caller's reason for
 * going top-down (in this tree: BL1 is reserved at the pool base) is not this
 * file's business, but it does rely on it: an ordinary first-fit allocation
 * would hand out the pool base.
 */
static int npu_mmu_reserve_page(npu_hal_mmu_t *mmu, uint64_t size,
                                npu_hal_allocation_t **allocation_out)
{
    npu_hal_memory_request_t request;
    uint64_t address;
    int status;

    if (mmu->pool_cursor - mmu->pool_base < size)
        return -ENOMEM;
    address = mmu->pool_cursor - size;

    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_NORMAL;
    request.usage = NPU_HAL_MEMORY_SHARED;
    request.owner = UINT32_MAX;
    request.scope = NPU_HAL_MEMORY_DEVICE;
    request.location.cluster = NPU_HAL_ANY_CLUSTER;
    request.location.core = NPU_HAL_ANY_CORE;
    request.size = size;
    request.alignment = DAYAN_MMU_PAGE_SIZE;
    request.fixed_address = address;
    /* The cursor moves only once the range is actually ours.  Consuming it
     * first would retire a page on every failed attempt, so a caller that
     * retries would walk the cursor down into whatever sits below. */
    status = npu_hal_memory_reserve(mmu->hal, &request, allocation_out);
    if (status < 0)
        return status;
    mmu->pool_cursor = address;
    return 0;
}

static int npu_mmu_table_alloc(npu_hal_mmu_t *mmu, uint32_t cluster,
                               struct npu_mmu_table **table_out)
{
    struct npu_mmu_table *table;
    npu_hal_allocation_t *allocation = 0;
    npu_hal_allocation_info_t info;
    int status;

    *table_out = 0;
    table = calloc(1, sizeof(*table));
    if (!table)
        return -ENOMEM;
    table->entries = calloc(NPU_MMU_PTE_PER_TABLE, sizeof(*table->entries));
    if (!table->entries) {
        free(table);
        return -ENOMEM;
    }
    table->children = calloc(NPU_MMU_PTE_PER_TABLE, sizeof(*table->children));
    if (!table->children) {
        free(table->entries);
        free(table);
        return -ENOMEM;
    }
    /* A page table must be page-aligned for its PPN to be exact. */
    status = npu_mmu_reserve_page(mmu, NPU_MMU_TABLE_BYTES, &allocation);
    if (status < 0)
        goto fail;
    status = npu_hal_memory_info(mmu->hal, allocation, &info);
    if (status < 0)
        goto fail_free;
    table->allocation = allocation;
    table->device_address = info.address;
    /* The walker must never see stale bytes as valid entries. */
    status = npu_hal_copy_to_address(mmu->hal, cluster,
                                     table->device_address, table->entries,
                                     NPU_MMU_TABLE_BYTES);
    if (status < 0)
        goto fail_free;
    *table_out = table;
    return 0;

fail_free:
    (void)npu_hal_memory_free(mmu->hal, allocation);
fail:
    free(table->children);
    free(table->entries);
    free(table);
    return status;
}

static void npu_mmu_table_free(npu_hal_mmu_t *mmu, struct npu_mmu_table *table)
{
    uint32_t index;

    if (!table)
        return;
    for (index = 0; index < NPU_MMU_PTE_PER_TABLE; ++index)
        npu_mmu_table_free(mmu, table->children[index]);
    if (table->allocation)
        (void)npu_hal_memory_free(mmu->hal, table->allocation);
    free(table->children);
    free(table->entries);
    free(table);
}

/*
 * Walk to the 2 MiB leaf slot for @va, optionally creating the level-1 table.
 * Returns the owning table and the index rather than a pointer, because every
 * entry write has to be mirrored to device memory and so needs to know which
 * table it belongs to.
 */
static int npu_mmu_walk(npu_hal_mmu_t *mmu, uint32_t cluster, uint64_t va,
                        int create, struct npu_mmu_table **table_out,
                        uint32_t *index_out)
{
    struct npu_mmu_table *table = mmu->clusters[cluster].root;
    unsigned level;

    for (level = NPU_MMU_LEVELS - 1U; level > NPU_MMU_LEAF_LEVEL; --level) {
        uint32_t index = npu_mmu_vpn(va, level);
        struct npu_mmu_table *child = table->children[index];

        if (!child) {
            int status;

            /*
             * A valid entry with no child is a leaf.  Descending would mean
             * replacing it with an interior entry and unmapping everything it
             * covers, so refuse instead.  Nothing installs leaves above level
             * 1, which is why this is a guard and not a live path.
             */
            if (table->entries[index] & NPU_MMU_PTE_V)
                return -EEXIST;
            if (!create)
                return -ENOENT;
            status = npu_mmu_table_alloc(mmu, cluster, &child);
            if (status < 0)
                return status;
            table->children[index] = child;
            /*
             * The child's contents are already in device memory (zeroed by
             * npu_mmu_table_alloc) before this entry makes it reachable, so
             * the walker can never follow this pointer into uninitialised
             * memory.
             */
            table->entries[index] = npu_mmu_pte_make(child->device_address,
                                                     0, 0);
            status = npu_mmu_push_entry(mmu, cluster, table, index);
            if (status < 0) {
                /* Retract the link: an entry the device never received must
                 * not be left claimed in the shadow. */
                table->entries[index] = 0;
                table->children[index] = 0;
                npu_mmu_table_free(mmu, child);
                return status;
            }
        }
        table = table->children[index];
    }
    *table_out = table;
    *index_out = npu_mmu_vpn(va, NPU_MMU_LEAF_LEVEL);
    return 0;
}

/* ---------------------------------------------------------------- */
/* Guard pages                                                       */
/* ---------------------------------------------------------------- */

/*
 * One page per fault class, each filled with a self-describing pattern and
 * pointed to by that class's redirect base.  ERROR_SLV_OFF_MODE is set so the
 * redirect keeps VA[11:0], which means a corrupted word's offset within the
 * page recovers the low 12 bits of the faulting address.
 *
 * This is sampling, not reporting: it cannot give the VA's upper bits and
 * cannot count occurrences.  The hardware has no fault-status register, so
 * this is the ceiling on what is observable.
 */
static int npu_mmu_guard_fill(npu_hal_mmu_t *mmu, uint32_t cluster,
                              uint64_t address)
{
    uint64_t pattern[DAYAN_MMU_PAGE_SIZE / sizeof(uint64_t)];
    size_t word;

    for (word = 0; word < sizeof(pattern) / sizeof(pattern[0]); ++word)
        pattern[word] = npu_mmu_poison_word(word * sizeof(uint64_t));
    return npu_hal_copy_to_address(mmu->hal, cluster, address, pattern,
                                   sizeof(pattern));
}

static int npu_mmu_guard_init(npu_hal_mmu_t *mmu, uint32_t cluster)
{
    struct npu_mmu_cluster *state = &mmu->clusters[cluster];
    unsigned class_index;
    uint32_t cfg_err = 0;
    int status;

    for (class_index = 0; class_index < DAYAN_MMU_REDIR_CLASS_COUNT;
         ++class_index) {
        npu_hal_allocation_t *allocation = 0;
        npu_hal_allocation_info_t info;

        status = npu_mmu_reserve_page(mmu, DAYAN_MMU_PAGE_SIZE, &allocation);
        if (status < 0)
            return status;
        status = npu_hal_memory_info(mmu->hal, allocation, &info);
        if (status < 0) {
            (void)npu_hal_memory_free(mmu->hal, allocation);
            return status;
        }
        state->guard_allocation[class_index] = allocation;
        state->guard_address[class_index] = info.address;
        status = npu_mmu_guard_fill(mmu, cluster, info.address);
        if (status < 0)
            return status;
        /*
         * A redirect base is a physical address that the hardware substitutes
         * for the whole master address; it needs no mapping of its own.  It
         * must be 4 KiB aligned, which the reservation guarantees and
         * npu_hal_mmu_write64() rechecks.
         */
        status = npu_hal_mmu_write64(mmu->hal, cluster,
                                     DAYAN_MMU_REDIR_LO(class_index),
                                     info.address);
        if (status < 0)
            return status;
    }

    status = npu_hal_mmu_write_reg(mmu->hal, cluster,
                                   DAYAN_MMU_ERROR_SLV_OFF_MODE,
                                   DAYAN_MMU_ERROR_SLV_OFF_VA);
    if (status < 0)
        return status;
    /* The error slave reports its own misconfiguration here; a set bit means
     * the redirect programming above was not accepted. */
    status = npu_hal_mmu_read_reg(mmu->hal, cluster,
                                  DAYAN_MMU_ERROR_SLV_CFG_ERR, &cfg_err);
    if (status < 0)
        return status;
    if (cfg_err & DAYAN_MMU_ERROR_SLV_CFG_ERR_BIT) {
        (void)npu_hal_mmu_write_reg(mmu->hal, cluster,
                                    DAYAN_MMU_ERROR_SLV_CFG_CLR,
                                    DAYAN_MMU_ERROR_SLV_CFG_ERR_BIT);
        return -EIO;
    }
    return 0;
}

int npu_hal_mmu_check_faults(npu_hal_mmu_t *mmu, uint32_t cluster,
                             uint32_t *fault_class, uint32_t *fault_offset)
{
    struct npu_mmu_cluster *state;
    uint64_t observed[DAYAN_MMU_PAGE_SIZE / sizeof(uint64_t)];
    unsigned class_index;

    if (!mmu || cluster >= mmu->cluster_count)
        return -EINVAL;
    state = &mmu->clusters[cluster];
    for (class_index = 0; class_index < DAYAN_MMU_REDIR_CLASS_COUNT;
         ++class_index) {
        size_t word;
        int status;

        if (!state->guard_allocation[class_index])
            continue;
        status = npu_hal_copy_from_address(mmu->hal, cluster, observed,
                                           state->guard_address[class_index],
                                           sizeof(observed));
        if (status < 0)
            return status;
        for (word = 0; word < sizeof(observed) / sizeof(observed[0]); ++word) {
            if (observed[word] == npu_mmu_poison_word(word * sizeof(uint64_t)))
                continue;
            if (fault_class)
                *fault_class = class_index;
            if (fault_offset)
                *fault_offset = (uint32_t)(word * sizeof(uint64_t));
            /* Restore so the next check starts from a known state. */
            (void)npu_mmu_guard_fill(mmu, cluster,
                                     state->guard_address[class_index]);
            return 1;
        }
    }
    return 0;
}

/* ---------------------------------------------------------------- */
/* Lifetime and activation                                           */
/* ---------------------------------------------------------------- */

/*
 * The root table and PTW_BASE first, then the guard pages, all before the
 * caller installs a single mapping and long before npu_hal_mmu_activate()
 * narrows FixMap.  Until then every address still goes through the reset-time
 * identity FixMap, so nothing done here is visible to the Device.
 */
static int npu_mmu_cluster_init(npu_hal_mmu_t *mmu, uint32_t cluster)
{
    struct npu_mmu_cluster *state = &mmu->clusters[cluster];
    int status;

    status = npu_mmu_table_alloc(mmu, cluster, &state->root);
    if (status < 0)
        return status;
    status = npu_hal_mmu_write64(mmu->hal, cluster, DAYAN_MMU_PTW_BASE_LO,
                                 state->root->device_address);
    if (status < 0)
        return status;
    /* One shared ASID: the caller separates users by VA range, not ASID. */
    status = npu_hal_mmu_write_reg(mmu->hal, cluster, DAYAN_MMU_GLOBAL_ASID, 0);
    if (status < 0)
        return status;
    return npu_mmu_guard_init(mmu, cluster);
}

int npu_hal_mmu_create(npu_hal_t *hal, uint32_t cluster_count,
                       uint64_t pool_base, uint64_t pool_size,
                       npu_hal_mmu_t **mmu_out)
{
    npu_hal_mmu_t *mmu;
    uint32_t cluster;
    int status;

    if (!mmu_out)
        return -EINVAL;
    *mmu_out = 0;
    if (!hal || !cluster_count || !pool_size ||
        pool_base > UINT64_MAX - pool_size)
        return -EINVAL;

    for (cluster = 0; cluster < cluster_count; ++cluster) {
        int present = 0;

        status = npu_hal_mmu_present(hal, cluster, &present);
        if (status < 0)
            return status;
        if (!present)
            return -ENODEV;
    }

    mmu = calloc(1, sizeof(*mmu));
    if (!mmu)
        return -ENOMEM;
    mmu->clusters = calloc(cluster_count, sizeof(*mmu->clusters));
    if (!mmu->clusters) {
        free(mmu);
        return -ENOMEM;
    }
    pthread_mutex_init(&mmu->lock, 0);
    mmu->hal = hal;
    mmu->cluster_count = cluster_count;
    mmu->pool_base = pool_base;
    mmu->pool_size = pool_size;
    mmu->pool_cursor = pool_base + pool_size;

    for (cluster = 0; cluster < cluster_count; ++cluster) {
        status = npu_mmu_cluster_init(mmu, cluster);
        if (status < 0) {
            npu_hal_mmu_destroy(mmu);
            return status;
        }
    }
    *mmu_out = mmu;
    return 0;
}

/*
 * Narrow FixMap to the low 2 GiB; higher addresses then fall through to the
 * page table.  Every Cluster's table has to be complete before this - the
 * first access above 2 GiB after it walks whatever is there.
 */
int npu_hal_mmu_activate(npu_hal_mmu_t *mmu)
{
    uint32_t cluster;
    int status;

    if (!mmu)
        return -EINVAL;
    for (cluster = 0; cluster < mmu->cluster_count; ++cluster) {
        status = npu_hal_mmu_write64(mmu->hal, cluster,
                                     DAYAN_MMU_FIX_MAP_BASE_LO, 0);
        if (status < 0)
            return status;
        status = npu_hal_mmu_write64(mmu->hal, cluster,
                                     DAYAN_MMU_FIX_MAP_SIZE_LO,
                                     DAYAN_MMU_LOW_2G_PAGES);
        if (status < 0)
            return status;
        status = npu_hal_mmu_write_reg(mmu->hal, cluster,
                                       DAYAN_MMU_N_FIX_MAP_EN, 0);
        if (status < 0)
            return status;
        /* Set per Cluster, so a failure part-way still gets the ones already
         * narrowed restored by deactivate. */
        mmu->active = 1;
    }
    return 0;
}

/*
 * Back to the reset-time full-range identity FixMap.  FixMap is re-enabled
 * before it is widened, so the page tables stay valid during the transition.
 * Every Cluster is attempted even if one fails; the first error is returned.
 */
int npu_hal_mmu_deactivate(npu_hal_mmu_t *mmu)
{
    uint32_t cluster;
    int result = 0;

    if (!mmu)
        return -EINVAL;
    for (cluster = 0; cluster < mmu->cluster_count; ++cluster) {
        int status = npu_hal_mmu_write_reg(mmu->hal, cluster,
                                           DAYAN_MMU_N_FIX_MAP_EN, 0);

        if (status == 0)
            status = npu_hal_mmu_write64(mmu->hal, cluster,
                                         DAYAN_MMU_FIX_MAP_BASE_LO, 0);
        if (status == 0)
            status = npu_hal_mmu_write64(mmu->hal, cluster,
                                         DAYAN_MMU_FIX_MAP_SIZE_LO, 0);
        if (status < 0 && result == 0)
            result = status;
    }
    mmu->active = 0;
    return result;
}

/*
 * Returns translation to identity first if the caller has not, so nothing is
 * translating while the tables are dismantled.
 */
void npu_hal_mmu_destroy(npu_hal_mmu_t *mmu)
{
    uint32_t cluster;

    if (!mmu)
        return;
    if (mmu->active)
        (void)npu_hal_mmu_deactivate(mmu);
    for (cluster = 0; cluster < mmu->cluster_count; ++cluster) {
        struct npu_mmu_cluster *state = &mmu->clusters[cluster];
        unsigned class_index;

        for (class_index = 0; class_index < DAYAN_MMU_REDIR_CLASS_COUNT;
             ++class_index)
            if (state->guard_allocation[class_index])
                (void)npu_hal_memory_free(mmu->hal,
                                          state->guard_allocation[class_index]);
        npu_mmu_table_free(mmu, state->root);
    }
    pthread_mutex_destroy(&mmu->lock);
    free(mmu->clusters);
    free(mmu);
}

/* ---------------------------------------------------------------- */
/* Mapping                                                           */
/* ---------------------------------------------------------------- */

static int npu_mmu_range_valid(const npu_hal_mmu_t *mmu, uint32_t cluster,
                               uint64_t va, uint64_t size)
{
    if (!mmu || cluster >= mmu->cluster_count)
        return -EINVAL;
    if ((va & NPU_HAL_MMU_LEAF_MASK) || (size & NPU_HAL_MMU_LEAF_MASK))
        return -EINVAL;
    if (va > UINT64_MAX - size)
        return -EOVERFLOW;
    return 0;
}

int npu_hal_mmu_map(npu_hal_mmu_t *mmu, uint32_t cluster, uint64_t va,
                    uint64_t pa, uint64_t size, uint32_t prot)
{
    uint64_t offset;
    int status;

    status = npu_mmu_range_valid(mmu, cluster, va, size);
    if (status < 0)
        return status;
    /* The walker raises ILLEGAL_PTE for a 2 MiB leaf whose PPN is not 2 MiB
     * aligned, so a misaligned pa is refused here rather than faulting. */
    if ((pa & NPU_HAL_MMU_LEAF_MASK) || !size)
        return -EINVAL;
    if (pa > UINT64_MAX - size)
        return -EOVERFLOW;
    if ((pa + size - 1U) >> DAYAN_MMU_PA_BITS)
        return -EINVAL;
    /*
     * Below 2 GiB the MMU bypasses the page table entirely, and Sv39 cannot
     * express anything at or above its 39-bit limit.  A PTE in either place
     * would never be consulted - a caller asking for one has the wrong
     * address and would otherwise get a success it cannot use.
     */
    if (va < NPU_HAL_MMU_BYPASS_LIMIT || va + size > NPU_HAL_MMU_VA_LIMIT)
        return -EINVAL;

    pthread_mutex_lock(&mmu->lock);
    for (offset = 0; offset < size; offset += NPU_HAL_MMU_LEAF_SIZE) {
        struct npu_mmu_table *table = 0;
        uint32_t index = 0;

        status = npu_mmu_walk(mmu, cluster, va + offset, 1, &table, &index);
        if (status < 0)
            break;
        if (table->entries[index] & NPU_MMU_PTE_V) {
            status = -EEXIST;
            break;
        }
        table->entries[index] = npu_mmu_pte_make(pa + offset, prot, 1);
        status = npu_mmu_push_entry(mmu, cluster, table, index);
        if (status < 0) {
            table->entries[index] = 0;
            break;
        }
    }
    if (status < 0) {
        /*
         * Take back the leaves this call did install.  The caller frees the
         * backing memory on failure, so a leaf left behind here would point
         * at memory that is about to belong to someone else.  Nothing can
         * have translated these addresses yet, so no flush is needed.
         */
        uint64_t undo;

        for (undo = 0; undo < offset; undo += NPU_HAL_MMU_LEAF_SIZE) {
            struct npu_mmu_table *table = 0;
            uint32_t index = 0;

            if (npu_mmu_walk(mmu, cluster, va + undo, 0, &table, &index) < 0)
                continue;
            table->entries[index] = 0;
            (void)npu_mmu_push_entry(mmu, cluster, table, index);
        }
    }
    pthread_mutex_unlock(&mmu->lock);

    /*
     * No flush on map.  A newly valid entry cannot be stale in the TLB,
     * because nothing has translated that address before.
     */
    return status;
}

int npu_hal_mmu_unmap(npu_hal_mmu_t *mmu, uint32_t cluster, uint64_t va,
                      uint64_t size)
{
    uint64_t offset;
    int status;

    status = npu_mmu_range_valid(mmu, cluster, va, size);
    if (status < 0)
        return status;

    pthread_mutex_lock(&mmu->lock);
    for (offset = 0; offset < size; offset += NPU_HAL_MMU_LEAF_SIZE) {
        struct npu_mmu_table *table = 0;
        uint32_t index = 0;
        int walked = npu_mmu_walk(mmu, cluster, va + offset, 0, &table,
                                  &index);

        /* An absent interior table means nothing was mapped here. */
        if (walked == -ENOENT)
            continue;
        if (walked < 0) {
            status = walked;
            break;
        }
        if (!(table->entries[index] & NPU_MMU_PTE_V))
            continue;
        table->entries[index] = 0;
        status = npu_mmu_push_entry(mmu, cluster, table, index);
        if (status < 0)
            break;
    }
    pthread_mutex_unlock(&mmu->lock);
    return status;
}

int npu_hal_mmu_protect(npu_hal_mmu_t *mmu, uint32_t cluster, uint64_t va,
                        uint64_t size, uint32_t prot)
{
    uint64_t offset;
    int status;

    status = npu_mmu_range_valid(mmu, cluster, va, size);
    if (status < 0)
        return status;

    pthread_mutex_lock(&mmu->lock);
    for (offset = 0; offset < size; offset += NPU_HAL_MMU_LEAF_SIZE) {
        struct npu_mmu_table *table = 0;
        uint32_t index = 0;
        uint64_t previous;

        status = npu_mmu_walk(mmu, cluster, va + offset, 0, &table, &index);
        if (status < 0)
            break;
        if (!(table->entries[index] & NPU_MMU_PTE_V)) {
            status = -ENOENT;
            break;
        }
        previous = table->entries[index];
        table->entries[index] =
            npu_mmu_pte_make(npu_mmu_pte_pa(previous), prot, 1);
        status = npu_mmu_push_entry(mmu, cluster, table, index);
        if (status < 0) {
            table->entries[index] = previous;
            break;
        }
    }
    pthread_mutex_unlock(&mmu->lock);
    return status;
}
