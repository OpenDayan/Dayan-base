/*
 * ClusterMMU page-table mechanism tests (hal/mmu_pt.c).
 *
 * These run against a fake HAL (mmu_fake_hal.c) rather than a device: the
 * point is PTE encoding, 2 MiB leaf walks, rollback, guard pages and the
 * register sequence that brings translation up and down.  Which addresses get
 * mapped is Service policy and is covered by server/tests/mmu_test.c.
 */

#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>

#include "mmu_fake_hal.h"

/* Arbitrary 2 MiB-aligned VAs above the bypass limit. */
#define VA_A UINT64_C(0x200000000)
#define VA_B UINT64_C(0x300000000)

static npu_hal_mmu_t *create(uint32_t clusters)
{
    npu_hal_mmu_t *mmu = 0;

    fake_hal_reset();
    assert(npu_hal_mmu_create(FAKE_HAL, clusters, FW_POOL_BASE, FW_POOL_SIZE,
                              &mmu) == 0);
    assert(mmu);
    return mmu;
}

/*
 * create programs everything except FixMap: until activate, the reset-time
 * identity map is still what the Device sees, so a half-built table is never
 * walked.
 */
static void test_create_and_activation(void)
{
    npu_hal_mmu_t *mmu = create(1);
    unsigned index;
    unsigned class_index;
    uint64_t root = fake_ptw_base(0);

    assert(root >= FW_POOL_BASE && root < FW_POOL_BASE + FW_POOL_SIZE);
    /* The root is the first thing reserved, off the very top of the pool:
     * the caller relies on the bottom staying free (BL1 lives there). */
    assert(root == FW_POOL_BASE + FW_POOL_SIZE - 0x1000);
    /* One shared ASID: users are separated by VA range, not ASID. */
    assert(mmu_registers[DAYAN_MMU_GLOBAL_ASID >> 2] == 0);
    /* Every redirect points at an owned, 4 KiB-aligned page in the pool, and
     * offset mode is on so a guard hit records VA[11:0]. */
    for (class_index = 0; class_index < DAYAN_MMU_REDIR_CLASS_COUNT;
         ++class_index) {
        uint64_t guard = fake_reg64(DAYAN_MMU_REDIR_LO(class_index));

        assert(guard >= FW_POOL_BASE && guard < root);
        assert(!(guard & 0xfffU));
    }
    assert(mmu_registers[DAYAN_MMU_ERROR_SLV_OFF_MODE >> 2] ==
           DAYAN_MMU_ERROR_SLV_OFF_VA);
    /* Root plus one guard page per class, all inside the pool. */
    assert(fake_alloc_live == 1 + DAYAN_MMU_REDIR_CLASS_COUNT);
    for (index = 0; index < fake_reserved_count; ++index) {
        if (!fake_reserved[index].size)
            continue;
        assert(fake_reserved[index].address >= FW_POOL_BASE);
        assert(fake_reserved[index].address + fake_reserved[index].size <=
               FW_POOL_BASE + FW_POOL_SIZE);
    }
    assert(fake_bl1_reserve() == 0);
    /* FixMap untouched so far. */
    assert(fix_map_size_writes == 0);

    /* Activation narrows FixMap exactly once, after PTW_BASE went live. */
    assert(npu_hal_mmu_activate(mmu) == 0);
    assert(fix_map_size_writes == 1);
    assert(fake_reg64(DAYAN_MMU_FIX_MAP_BASE_LO) == 0);
    assert(fake_reg64(DAYAN_MMU_FIX_MAP_SIZE_LO) == DAYAN_MMU_LOW_2G_PAGES);
    assert(mmu_registers[DAYAN_MMU_N_FIX_MAP_EN >> 2] == 0);

    /* Deactivation restores the reset-time full-range identity FixMap. */
    assert(npu_hal_mmu_deactivate(mmu) == 0);
    assert(mmu_registers[DAYAN_MMU_N_FIX_MAP_EN >> 2] == 0);
    assert(fake_reg64(DAYAN_MMU_FIX_MAP_BASE_LO) == 0);
    assert(fake_reg64(DAYAN_MMU_FIX_MAP_SIZE_LO) == 0);

    /* Destroy gives back every page it took - and only those: the fake BL1
     * reservation above was never handed to the MMU. */
    npu_hal_mmu_destroy(mmu);
    assert(fake_alloc_live == 1);
}

/* Destroying an active MMU must drop back to identity before freeing the
 * tables, or the Device would keep walking memory about to be reused. */
static void test_destroy_while_active(void)
{
    npu_hal_mmu_t *mmu = create(1);

    assert(npu_hal_mmu_map(mmu, 0, VA_A, DEVICE_MEMORY_BASE, LEAF,
                           NPU_HAL_MMU_READ) == 0);
    assert(npu_hal_mmu_activate(mmu) == 0);
    assert(fake_reg64(DAYAN_MMU_FIX_MAP_SIZE_LO) == DAYAN_MMU_LOW_2G_PAGES);
    npu_hal_mmu_destroy(mmu);
    assert(fake_reg64(DAYAN_MMU_FIX_MAP_SIZE_LO) == 0);
    /* Root, the level-1 table under VA_A, and the guard pages. */
    assert(fake_alloc_live == 0);
}

static void test_create_refusals(void)
{
    npu_hal_mmu_t *mmu = 0;

    fake_hal_reset();
    /* Asked for, but not there: nothing is allocated or programmed. */
    fake_mmu_absent_cluster = 1;
    assert(npu_hal_mmu_create(FAKE_HAL, 2, FW_POOL_BASE, FW_POOL_SIZE,
                              &mmu) == -ENODEV);
    assert(!mmu);
    assert(fake_alloc_live == 0);
    assert(!ptw_base_written);

    fake_hal_reset();
    assert(npu_hal_mmu_create(0, 1, FW_POOL_BASE, FW_POOL_SIZE, &mmu) ==
           -EINVAL);
    assert(npu_hal_mmu_create(FAKE_HAL, 0, FW_POOL_BASE, FW_POOL_SIZE,
                              &mmu) == -EINVAL);
    assert(npu_hal_mmu_create(FAKE_HAL, 1, FW_POOL_BASE, 0, &mmu) == -EINVAL);
    /* A pool too small for the root and guard pages fails cleanly. */
    assert(npu_hal_mmu_create(FAKE_HAL, 1, FW_POOL_BASE, 0x3000, &mmu) ==
           -ENOMEM);
    assert(!mmu);
    assert(fake_alloc_live == 0);
}

static void test_pte_encoding_and_walk(void)
{
    npu_hal_mmu_t *mmu = create(1);
    uint64_t root = fake_ptw_base(0);
    uint64_t pte;

    /* A W|X mapping, to check every permission bit is placed correctly. */
    assert(npu_hal_mmu_map(mmu, 0, VA_A, DEVICE_MEMORY_BASE + LEAF, 2 * LEAF,
                           NPU_HAL_MMU_READ | NPU_HAL_MMU_WRITE |
                           NPU_HAL_MMU_EXEC) == 0);
    {
        int level = -1;

        /* A 2 MiB leaf at level 1; a level-0 leaf would pass every other
         * assertion below. */
        pte = walk_device_level(root, VA_A, &level);
        assert(level == 1);
        /* Any address inside the leaf resolves through it. */
        assert(walk_device_level(root, VA_A + LEAF - 0x1000, &level) == pte);
        assert(level == 1);
    }
    assert(pte & PTE_V);
    assert(pte & PTE_R);
    assert(pte & PTE_W);
    assert(pte & PTE_X);
    /* A and D are preset because the hardware never updates them; a leaf
     * with A clear would fault forever. */
    assert(pte & PTE_A);
    assert(pte & PTE_D);
    assert(pte & PTE_U);
    assert(pte & PTE_G);
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + LEAF);
    /* The second leaf of the range resolves independently. */
    pte = walk_device(root, VA_A + LEAF);
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + 2 * LEAF);

    /* Read-only must not carry W, or a redirect would never trigger. */
    assert(npu_hal_mmu_map(mmu, 0, VA_B, DEVICE_MEMORY_BASE + 4 * LEAF, LEAF,
                           NPU_HAL_MMU_READ) == 0);
    pte = walk_device(root, VA_B);
    assert((pte & PTE_R) && !(pte & PTE_W) && !(pte & PTE_X));

    /* Double mapping the same VA is a bookkeeping bug, not something to
     * silently overwrite. */
    assert(npu_hal_mmu_map(mmu, 0, VA_B, DEVICE_MEMORY_BASE + 5 * LEAF, LEAF,
                           NPU_HAL_MMU_READ) == -EEXIST);
    /*
     * A range that fails part-way must leave nothing behind.  The caller
     * frees the backing memory on failure, so a leaf installed for the first
     * part would otherwise point at memory about to be reused.
     */
    assert(npu_hal_mmu_map(mmu, 0, VA_B - 2 * LEAF,
                           DEVICE_MEMORY_BASE + 6 * LEAF, 3 * LEAF,
                           NPU_HAL_MMU_READ) == -EEXIST);
    assert(walk_device(root, VA_B - 2 * LEAF) == 0);
    assert(walk_device(root, VA_B - LEAF) == 0);
    /* The colliding leaf itself is untouched. */
    pte = walk_device(root, VA_B);
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + 4 * LEAF);

    /* Misalignment and zero length are rejected rather than rounded.  4 KiB
     * alignment is not enough: a 2 MiB leaf with an unaligned PPN is an
     * ILLEGAL_PTE to the walker. */
    assert(npu_hal_mmu_map(mmu, 0, VA_A + 0x10000000 + 0x1000,
                           DEVICE_MEMORY_BASE, LEAF,
                           NPU_HAL_MMU_READ) == -EINVAL);
    assert(npu_hal_mmu_map(mmu, 0, UINT64_C(0x400000000),
                           DEVICE_MEMORY_BASE + 0x1000, LEAF,
                           NPU_HAL_MMU_READ) == -EINVAL);
    assert(npu_hal_mmu_map(mmu, 0, UINT64_C(0x400000000), DEVICE_MEMORY_BASE,
                           0x1000, NPU_HAL_MMU_READ) == -EINVAL);
    assert(npu_hal_mmu_map(mmu, 0, UINT64_C(0x400000000), DEVICE_MEMORY_BASE,
                           0, NPU_HAL_MMU_READ) == -EINVAL);
    /* A PA beyond 56 bits cannot be encoded. */
    assert(npu_hal_mmu_map(mmu, 0, UINT64_C(0x400000000),
                           UINT64_C(0x0100000000000000), LEAF,
                           NPU_HAL_MMU_READ) == -EINVAL);
    /* Below 2 GiB the hardware bypasses the table, and Sv39 ends at 2^38:
     * a leaf in either place would never be consulted. */
    assert(npu_hal_mmu_map(mmu, 0, UINT64_C(0x40000000), DEVICE_MEMORY_BASE,
                           LEAF, NPU_HAL_MMU_READ) == -EINVAL);
    assert(npu_hal_mmu_map(mmu, 0, NPU_HAL_MMU_VA_LIMIT - LEAF,
                           DEVICE_MEMORY_BASE, 2 * LEAF,
                           NPU_HAL_MMU_READ) == -EINVAL);
    assert(npu_hal_mmu_map(mmu, 0, NPU_HAL_MMU_VA_LIMIT - LEAF,
                           DEVICE_MEMORY_BASE, LEAF, NPU_HAL_MMU_READ) == 0);
    /* No such Cluster. */
    assert(npu_hal_mmu_map(mmu, 1, UINT64_C(0x400000000), DEVICE_MEMORY_BASE,
                           LEAF, NPU_HAL_MMU_READ) == -EINVAL);

    /* protect() narrows in place and keeps the PA. */
    assert(npu_hal_mmu_protect(mmu, 0, VA_A, LEAF, NPU_HAL_MMU_READ) == 0);
    pte = walk_device(root, VA_A);
    assert((pte & PTE_R) && !(pte & PTE_W) && !(pte & PTE_X));
    assert(pte_pa(pte) == DEVICE_MEMORY_BASE + LEAF);
    /* Nothing mapped there means nothing to protect. */
    assert(npu_hal_mmu_protect(mmu, 0, UINT64_C(0x500000000), LEAF,
                               NPU_HAL_MMU_READ) == -ENOENT);
    assert(npu_hal_mmu_protect(mmu, 0, VA_A, 0x1000, NPU_HAL_MMU_READ) ==
           -EINVAL);

    /* unmap clears the leaf in device memory... */
    assert(npu_hal_mmu_unmap(mmu, 0, VA_A, 2 * LEAF) == 0);
    assert(walk_device(root, VA_A) == 0);
    assert(walk_device(root, VA_A + LEAF) == 0);
    /* ...and never flushes: that is the caller's, once the device is idle. */
    assert(flush_count == 0);
    /* Unmapping an absent range is not an error; reclaim retries are
     * allowed to be sloppy. */
    assert(npu_hal_mmu_unmap(mmu, 0, UINT64_C(0x600000000), LEAF) == 0);
    assert(npu_hal_mmu_unmap(mmu, 0, VA_A, 0x1000) == -EINVAL);

    npu_hal_mmu_destroy(mmu);
    assert(fake_alloc_live == 0);
}

/* Each Cluster has its own table: a leaf installed on one is invisible to
 * the other's walker. */
static void test_clusters_are_separate(void)
{
    npu_hal_mmu_t *mmu = create(2);
    uint64_t root0 = fake_ptw_base(0);
    uint64_t root1 = fake_ptw_base(1);

    assert(root0 && root1 && root0 != root1);
    assert(npu_hal_mmu_map(mmu, 1, VA_A, DEVICE_MEMORY_BASE, LEAF,
                           NPU_HAL_MMU_READ) == 0);
    assert(walk_device(root1, VA_A) != 0);
    assert(walk_device(root0, VA_A) == 0);
    assert(npu_hal_mmu_unmap(mmu, 0, VA_A, LEAF) == 0);
    assert(walk_device(root1, VA_A) != 0);
    npu_hal_mmu_destroy(mmu);
    assert(fake_alloc_live == 0);
}

static void test_guard_pages(void)
{
    npu_hal_mmu_t *mmu = create(1);
    uint32_t fault_class = UINT32_MAX;
    uint32_t fault_offset = UINT32_MAX;
    uint64_t guard;
    uint64_t corruption = 0;

    /* Untouched guard pages mean no fault was observed. */
    assert(npu_hal_mmu_check_faults(mmu, 0, &fault_class, &fault_offset) == 0);

    /* Simulate a redirected write landing in the write-permission guard.
     * With ERROR_SLV_OFF_MODE set the landing offset is the faulting
     * access's VA[11:0], so the offset is recoverable. */
    guard = fake_reg64(DAYAN_MMU_REDIR_WRITE_PERM_LO);
    assert(guard >= DEVICE_MEMORY_BASE);
    assert(npu_hal_copy_to_address(FAKE_HAL, 0, guard + 0x40, &corruption,
                                   sizeof(corruption)) == 0);
    assert(npu_hal_mmu_check_faults(mmu, 0, &fault_class, &fault_offset) == 1);
    /* Class 2 is write-permission, per the redirect base ordering. */
    assert(fault_class == 2);
    assert(fault_offset == 0x40);
    /* The page is restored, so the next check starts clean and a second
     * fault is distinguishable from the first. */
    assert(npu_hal_mmu_check_faults(mmu, 0, &fault_class, &fault_offset) == 0);
    assert(npu_hal_mmu_check_faults(mmu, 1, 0, 0) == -EINVAL);

    npu_hal_mmu_destroy(mmu);
}

int main(void)
{
    test_create_and_activation();
    test_destroy_while_active();
    test_create_refusals();
    test_pte_encoding_and_walk();
    test_clusters_are_separate();
    test_guard_pages();
    puts("mmu_pt_test: create/activate ordering, 2 MiB leaves, rollback, "
         "per-Cluster tables and guard pages OK");
    return 0;
}
