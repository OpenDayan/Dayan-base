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
 * Fake HAL for the ClusterMMU tests.
 *
 * hal/mmu_pt.c uses only public HAL calls, so it can be linked against these
 * stand-ins instead of the real HAL and a device.  Both mmu_pt_test (the HAL
 * page-table mechanism) and server/tests/mmu_test (the Service policy on top
 * of it) link this file, so they see the same device memory, register image
 * and allocator.
 */

#ifndef NPU_HAL_TESTS_MMU_FAKE_HAL_H
#define NPU_HAL_TESTS_MMU_FAKE_HAL_H

#include <stddef.h>
#include <stdint.h>

#include "hal.h"
#include "platform_regs.h"

/* Sv39 bits, duplicated so a change in mmu_pt.c has to be deliberate. */
#define PTE_V  UINT64_C(0x001)
#define PTE_R  UINT64_C(0x002)
#define PTE_W  UINT64_C(0x004)
#define PTE_X  UINT64_C(0x008)
#define PTE_U  UINT64_C(0x010)
#define PTE_G  UINT64_C(0x020)
#define PTE_A  UINT64_C(0x040)
#define PTE_D  UINT64_C(0x080)

#define DEVICE_MEMORY_BASE UINT64_C(0x80000000)
#define DEVICE_MEMORY_SIZE UINT64_C(0x00800000)

/*
 * The firmware pool: 4 MiB, which is two 2 MiB leaves - enough for "more than
 * one leaf" to mean something - with the Buffer pool at a different physical
 * address so the disjointness tests can tell them apart.
 *
 * FW_POOL_OFFSET is the point of this arrangement.  In the default server
 * configuration the pool sits at the Normal region base, which is also the
 * firmware's link address, so the linear offset degenerates to an identity and
 * a virtual address used as a physical one still names the right memory.  That
 * makes every address-domain confusion invisible.  Here the pool is deliberately
 * placed 2 MiB above the region base, so pa != va throughout and any code that
 * hands a firmware VA to the HAL reads or writes the wrong page.
 *
 * The HAL page-table tests use the same range as the pool their tables and
 * guard pages come from.
 */
#define FW_POOL_OFFSET     UINT64_C(0x00200000)
#define FW_POOL_SIZE       UINT64_C(0x00400000)
#define FW_POOL_BASE       (DEVICE_MEMORY_BASE + FW_POOL_OFFSET)
#define BUFFER_POOL_BASE   (FW_POOL_BASE + FW_POOL_SIZE)
#define BUFFER_POOL_LIMIT  (DEVICE_MEMORY_BASE + DEVICE_MEMORY_SIZE)

/* One mapping leaf: 2 MiB. */
#define LEAF               NPU_HAL_MMU_LEAF_SIZE

/* The fake ignores the handle, but npu_hal_mmu_create() rejects NULL. */
extern char fake_hal_token;
#define FAKE_HAL ((npu_hal_t *)(void *)&fake_hal_token)

/* Occupancy list, so reserve and alloc reject overlaps like the real HAL. */
#define FAKE_RESERVED_MAX 64
struct fake_reserved_range {
    uint64_t address;
    uint64_t size;
};
extern struct fake_reserved_range fake_reserved[FAKE_RESERVED_MAX];
extern unsigned fake_reserved_count;
extern uint32_t fake_alloc_live;

/* Recorded register image and counters, so ordering can be checked. */
extern uint32_t mmu_registers[DAYAN_MMU_APB_SIZE / 4];
extern uint32_t fix_map_size_writes;
extern uint32_t ptw_base_written;
extern uint32_t flush_count;
extern uint32_t flush_fail;
/* Makes npu_hal_mmu_present() report no MMU on this Cluster. */
extern uint32_t fake_mmu_absent_cluster;

/* Clear every piece of fake state. */
void fake_hal_reset(void);

int fake_reserve(uint64_t address, uint64_t size,
                 npu_hal_allocation_t **allocation_out);
/* First-fit inside [base, limit), mirroring the real HAL's pool walk. */
int fake_alloc_in(uint64_t base, uint64_t limit, uint64_t size,
                  uint64_t alignment, npu_hal_allocation_t **allocation_out);
/* Stands in for the BL1 segment reservation at the firmware pool base. */
int fake_bl1_reserve(void);

/* The table root PTW_BASE was last programmed with on @cluster (< 4).  The
 * register image itself is shared, so it only shows the last Cluster. */
#define FAKE_CLUSTER_MAX 4
uint64_t fake_ptw_base(uint32_t cluster);
/* A 64-bit LO/HI register pair from the image. */
uint64_t fake_reg64(uint32_t offset_lo);

/*
 * Walk the device-resident tables the way the hardware would, so a test
 * validates what the walker will actually see rather than the shadow.  Stops
 * on a leaf; @level_out reports where (1 for a 2 MiB leaf, -1 when absent).
 */
uint64_t walk_device_level(uint64_t root, uint64_t va, int *level_out);
uint64_t walk_device(uint64_t root, uint64_t va);
uint64_t pte_pa(uint64_t pte);

#endif
