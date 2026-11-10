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
 * Fake HAL for the ClusterMMU tests; see mmu_fake_hal.h.
 */

#define _POSIX_C_SOURCE 200809L

#include "mmu_fake_hal.h"

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

struct npu_hal_allocation {
    uint64_t address;
    uint64_t size;
    int freed;
};

char fake_hal_token;

/* Fake device memory, so page tables can be read back and inspected. */
static unsigned char device_memory[DEVICE_MEMORY_SIZE];

struct fake_reserved_range fake_reserved[FAKE_RESERVED_MAX];
unsigned fake_reserved_count;
uint32_t fake_alloc_live;

uint32_t mmu_registers[DAYAN_MMU_APB_SIZE / 4];
uint32_t fix_map_size_writes;
uint32_t ptw_base_written;
uint32_t flush_count;
uint32_t flush_fail;
uint32_t fake_mmu_absent_cluster = UINT32_MAX;
static uint64_t ptw_base_of[FAKE_CLUSTER_MAX];

void fake_hal_reset(void)
{
    memset(device_memory, 0, sizeof(device_memory));
    memset(mmu_registers, 0, sizeof(mmu_registers));
    memset(fake_reserved, 0, sizeof(fake_reserved));
    fake_reserved_count = 0;
    fake_alloc_live = 0;
    fix_map_size_writes = 0;
    ptw_base_written = 0;
    flush_count = 0;
    flush_fail = 0;
    fake_mmu_absent_cluster = UINT32_MAX;
    memset(ptw_base_of, 0, sizeof(ptw_base_of));
}

static int in_device_memory(uint64_t address, size_t size)
{
    return address >= DEVICE_MEMORY_BASE &&
           address - DEVICE_MEMORY_BASE <= DEVICE_MEMORY_SIZE - size;
}

static int fake_range_free(uint64_t address, uint64_t size)
{
    unsigned index;

    for (index = 0; index < fake_reserved_count; ++index) {
        uint64_t base = fake_reserved[index].address;

        if (!fake_reserved[index].size)
            continue;
        if (address < base + fake_reserved[index].size &&
            base < address + size)
            return 0;
    }
    return 1;
}

int fake_reserve(uint64_t address, uint64_t size,
                 npu_hal_allocation_t **allocation_out)
{
    npu_hal_allocation_t *allocation;
    unsigned index;

    if (address < DEVICE_MEMORY_BASE ||
        size > DEVICE_MEMORY_BASE + DEVICE_MEMORY_SIZE - address)
        return -ERANGE;
    if (!fake_range_free(address, size))
        return -EEXIST;
    if (fake_reserved_count == FAKE_RESERVED_MAX)
        return -ENOMEM;
    allocation = calloc(1, sizeof(*allocation));
    if (!allocation)
        return -ENOMEM;
    allocation->address = address;
    allocation->size = size;
    index = fake_reserved_count++;
    fake_reserved[index].address = address;
    fake_reserved[index].size = size;
    ++fake_alloc_live;
    *allocation_out = allocation;
    return 0;
}

int fake_alloc_in(uint64_t base, uint64_t limit, uint64_t size,
                  uint64_t alignment, npu_hal_allocation_t **allocation_out)
{
    uint64_t mask = alignment ? alignment - 1U : 0;
    uint64_t address;

    for (address = (base + mask) & ~mask; address + size <= limit;
         address += alignment ? alignment : size) {
        if (fake_range_free(address, size))
            return fake_reserve(address, size, allocation_out);
    }
    return -ENOMEM;
}

int fake_bl1_reserve(void)
{
    npu_hal_allocation_t *allocation = 0;
    int status = fake_reserve(FW_POOL_BASE, UINT64_C(0x20000), &allocation);

    if (status == 0)
        free(allocation);
    return status;
}

uint64_t fake_reg64(uint32_t offset_lo)
{
    return ((uint64_t)mmu_registers[(offset_lo + 4U) >> 2] << 32) |
           mmu_registers[offset_lo >> 2];
}

uint64_t fake_ptw_base(uint32_t cluster)
{
    assert(cluster < FAKE_CLUSTER_MAX);
    return ptw_base_of[cluster];
}

/* ---------------------------------------------------------------- */
/* HAL stand-ins                                                     */
/* ---------------------------------------------------------------- */

int npu_hal_memory_alloc(npu_hal_t *hal,
                         const npu_hal_memory_request_t *request,
                         npu_hal_allocation_t **allocation_out)
{
    (void)hal;
    *allocation_out = 0;
    if (request->version != NPU_HAL_MEMORY_REQUEST_VERSION ||
        !request->size || !request->alignment || request->fixed_address)
        return -EINVAL;
    /* Not a Buffer, so the firmware pool - the same routing the real HAL
     * applies through usage_is_firmware_pool(). */
    return fake_alloc_in(FW_POOL_BASE, FW_POOL_BASE + FW_POOL_SIZE,
                         request->size, request->alignment, allocation_out);
}

int npu_hal_firmware_pool(npu_hal_t *hal, uint64_t *base_out,
                          uint64_t *size_out)
{
    (void)hal;
    *base_out = FW_POOL_BASE;
    *size_out = FW_POOL_SIZE;
    return 0;
}

void npu_hal_memory_request_init(npu_hal_memory_request_t *request)
{
    memset(request, 0, sizeof(*request));
    request->version = NPU_HAL_MEMORY_REQUEST_VERSION;
    request->owner = UINT32_MAX;
    request->location.cluster = NPU_HAL_ANY_CLUSTER;
    request->location.core = NPU_HAL_ANY_CORE;
    request->alignment = 1;
}

int npu_hal_memory_reserve(npu_hal_t *hal,
                           const npu_hal_memory_request_t *request,
                           npu_hal_allocation_t **allocation_out)
{
    (void)hal;
    *allocation_out = 0;
    if (request->version != NPU_HAL_MEMORY_REQUEST_VERSION || !request->size ||
        !request->alignment ||
        (request->fixed_address & (request->alignment - 1U)))
        return -EINVAL;
    return fake_reserve(request->fixed_address, request->size, allocation_out);
}

int npu_hal_memory_info(npu_hal_t *hal, const npu_hal_allocation_t *allocation,
                        npu_hal_allocation_info_t *info)
{
    (void)hal;
    memset(info, 0, sizeof(*info));
    info->address = allocation->address;
    info->size = allocation->size;
    /* Every allocation in these tests is Normal memory, which is the only
     * class the MMU translates - Coherent and URAM are below the bypass
     * limit.  The class has to be reported, or address translation would
     * take the not-translated path and hand back a physical address. */
    info->memory_class = NPU_HAL_MEMORY_NORMAL;
    info->scope = NPU_HAL_MEMORY_DEVICE;
    return 0;
}

/* The HAL reports physical addresses; that is the whole reason
 * npu_service_mmu_device_address() exists as a separate call. */
int npu_hal_memory_address(npu_hal_t *hal,
                           const npu_hal_allocation_t *allocation,
                           npu_hal_core_t viewer, uint64_t *address_out)
{
    (void)hal;
    (void)viewer;
    if (!allocation)
        return -EACCES;
    *address_out = allocation->address;
    return 0;
}

int npu_hal_memory_free(npu_hal_t *hal, npu_hal_allocation_t *allocation)
{
    unsigned index;

    (void)hal;
    /* A double free here would mean something was released twice - the
     * reclaim list or a table teardown - which is what these tests guard. */
    assert(allocation && !allocation->freed);
    for (index = 0; index < fake_reserved_count; ++index) {
        if (fake_reserved[index].size &&
            fake_reserved[index].address == allocation->address) {
            fake_reserved[index].size = 0;
            break;
        }
    }
    --fake_alloc_live;
    free(allocation);
    return 0;
}

int npu_hal_copy_to_address(npu_hal_t *hal, uint32_t cluster, uint64_t address,
                            const void *source, size_t size)
{
    (void)hal;
    (void)cluster;
    assert(in_device_memory(address, size));
    memcpy(device_memory + (address - DEVICE_MEMORY_BASE), source, size);
    return 0;
}

int npu_hal_copy_from_address(npu_hal_t *hal, uint32_t cluster,
                              void *destination, uint64_t address, size_t size)
{
    (void)hal;
    (void)cluster;
    assert(in_device_memory(address, size));
    memcpy(destination, device_memory + (address - DEVICE_MEMORY_BASE), size);
    return 0;
}

int npu_hal_mmu_write_reg(npu_hal_t *hal, uint32_t cluster, uint32_t offset,
                          uint32_t value)
{
    (void)hal;
    (void)cluster;
    assert(offset < DAYAN_MMU_APB_SIZE && !(offset & 3U));
    if (offset == DAYAN_MMU_FIX_MAP_SIZE_LO && value) {
        /*
         * FixMap must not be narrowed before the walker base is programmed;
         * otherwise the first high address walks an empty table.
         */
        assert(ptw_base_written);
        ++fix_map_size_writes;
    }
    mmu_registers[offset >> 2] = value;
    return 0;
}

int npu_hal_mmu_read_reg(npu_hal_t *hal, uint32_t cluster, uint32_t offset,
                         uint32_t *value)
{
    (void)hal;
    (void)cluster;
    assert(offset < DAYAN_MMU_APB_SIZE && !(offset & 3U));
    *value = mmu_registers[offset >> 2];
    return 0;
}

int npu_hal_mmu_write64(npu_hal_t *hal, uint32_t cluster, uint32_t offset_lo,
                        uint64_t value)
{
    if (value >> DAYAN_MMU_PA_BITS)
        return -EINVAL;
    if (offset_lo == DAYAN_MMU_PTW_BASE_LO) {
        assert(cluster < FAKE_CLUSTER_MAX);
        ptw_base_of[cluster] = value;
        ptw_base_written = 1;
    }
    (void)npu_hal_mmu_write_reg(hal, cluster, offset_lo,
                                (uint32_t)(value & UINT32_C(0xffffffff)));
    return npu_hal_mmu_write_reg(hal, cluster, offset_lo + 4U,
                                 (uint32_t)((value >> 32) &
                                            DAYAN_MMU_PA_HI_MASK));
}

int npu_hal_mmu_present(npu_hal_t *hal, uint32_t cluster, int *present)
{
    (void)hal;
    *present = cluster != fake_mmu_absent_cluster;
    return 0;
}

int npu_hal_mmu_flush(npu_hal_t *hal, uint32_t cluster, int asid_selective,
                      uint8_t asid, uint32_t timeout_ms)
{
    (void)hal;
    (void)cluster;
    (void)asid_selective;
    (void)asid;
    (void)timeout_ms;
    ++flush_count;
    return flush_fail ? -ETIMEDOUT : 0;
}

/* ---------------------------------------------------------------- */
/* Hardware-eye view of the tables                                   */
/* ---------------------------------------------------------------- */

static uint64_t read_pte(uint64_t table_address, uint32_t index)
{
    uint64_t pte;

    assert(in_device_memory(table_address + index * 8U, sizeof(pte)));
    memcpy(&pte, device_memory + (table_address - DEVICE_MEMORY_BASE) +
                 index * 8U, sizeof(pte));
    return pte;
}

uint64_t pte_pa(uint64_t pte)
{
    return (pte >> 10) << DAYAN_MMU_PAGE_SHIFT;
}

static int pte_is_leaf(uint64_t pte)
{
    return (pte & (PTE_R | PTE_W | PTE_X)) != 0;
}

/*
 * The walker treats an entry with any of R/W/X set as a leaf and never reads
 * a table under it, which is how a 2 MiB mapping is found.
 */
uint64_t walk_device_level(uint64_t root, uint64_t va, int *level_out)
{
    uint64_t table = root;
    int level;

    if (level_out)
        *level_out = -1;
    for (level = 2; level > 0; --level) {
        uint32_t index = (uint32_t)((va >> (DAYAN_MMU_PAGE_SHIFT +
                                            level * 9)) & 0x1ff);
        uint64_t pte = read_pte(table, index);

        if (!(pte & PTE_V))
            return 0;
        if (pte_is_leaf(pte)) {
            if (level_out)
                *level_out = level;
            return pte;
        }
        table = pte_pa(pte);
    }
    if (level_out)
        *level_out = 0;
    return read_pte(table, (uint32_t)((va >> DAYAN_MMU_PAGE_SHIFT) & 0x1ff));
}

uint64_t walk_device(uint64_t root, uint64_t va)
{
    return walk_device_level(root, va, 0);
}
