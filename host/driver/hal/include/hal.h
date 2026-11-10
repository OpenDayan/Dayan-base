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

#ifndef NPU_HOST_HAL_H
#define NPU_HOST_HAL_H

/* Public HAL control and memory-management API. */

#include <stddef.h>
#include <stdint.h>

#include "npu_hal_config.h"
#include "dataplane.h"
#include "npu_pcu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_HAL_ABI_VERSION          6U
#define NPU_HAL_MEMORY_REQUEST_VERSION 1U
#define NPU_HAL_TIMEOUT_INFINITE     UINT32_MAX

typedef struct npu_hal npu_hal_t;
typedef struct npu_hal_allocation npu_hal_allocation_t;

typedef enum npu_hal_memory_usage {
    NPU_HAL_MEMORY_BUFFER = 1,
    NPU_HAL_MEMORY_FIRMWARE,
    NPU_HAL_MEMORY_KERNEL,
    NPU_HAL_MEMORY_HEAP,
    NPU_HAL_MEMORY_LAUNCH,
    NPU_HAL_MEMORY_SHARED,
    NPU_HAL_MEMORY_KERNEL_URAM,
    NPU_HAL_MEMORY_PROFILE,
    NPU_HAL_MEMORY_ATOMIC_CACHE_BACKING,
} npu_hal_memory_usage_t;

typedef struct npu_hal_info {
    uint32_t abi_version;
    uint32_t backend;
    uint32_t data_provider;
    uint32_t flags;
    uint32_t cluster_count;
    uint32_t cores_per_cluster;
    uint32_t data_handle_count;
    uint32_t memory_region_count;
    uint64_t max_transfer_size;
    uint64_t transfer_alignment;
} npu_hal_info_t;

typedef struct npu_hal_memory_request {
    uint32_t version;
    uint32_t memory_class;
    uint32_t scope;
    uint32_t usage;
    uint32_t owner;
    uint32_t flags;
    npu_hal_core_t location;
    uint64_t size;
    uint64_t alignment;
    uint64_t fixed_address;
} npu_hal_memory_request_t;

typedef struct npu_hal_allocation_info {
    uint64_t address;
    uint64_t size;
    uint32_t memory_class;
    uint32_t scope;
    uint32_t usage;
    uint32_t owner;
    npu_hal_core_t location;
} npu_hal_allocation_info_t;

void npu_hal_memory_request_init(npu_hal_memory_request_t *request);
int npu_hal_open(npu_hal_t **hal_out, const npu_hal_options_t *options);
void npu_hal_close(npu_hal_t *hal);
/* Returns the first fatal backend I/O error, or zero while usable. */
int npu_hal_device_status(const npu_hal_t *hal);
int npu_hal_get_info(npu_hal_t *hal, npu_hal_info_t *info);
int npu_hal_get_memory_region(npu_hal_t *hal, uint32_t index,
                              npu_hal_memory_region_t *region);
int npu_hal_core_valid(npu_hal_t *hal, npu_hal_core_t core);
/*
 * Total size of the regions of @memory_class visible to @viewer.  This is the
 * installed memory, not what any one allocation can reach: with the Normal
 * region split (see npu_hal_firmware_pool()) no single request can use all of
 * it, so this is a topology figure and not an allocation budget.
 */
int npu_hal_memory_capacity(npu_hal_t *hal,
                            npu_hal_memory_class_t memory_class,
                            npu_hal_core_t viewer, uint64_t *capacity_out);
/*
 * The device-visible Normal region is split into a firmware pool (low, at the
 * region base, a 2 MiB multiple) and a Buffer pool above it, when
 * npu_hal_options_t::split_normal_memory asked for it.  Reports the firmware
 * pool so the MMU can map it as one linear-offset window.  Returns -ENODEV when
 * the split was not requested, and -ENOENT when it was but the region cannot be
 * split, in which case the whole region is reported: mapping all of it is
 * possible but leaves Buffer pages inside the window, so a caller that needs
 * the pools disjoint has to treat -ENOENT as fatal.
 */
int npu_hal_firmware_pool(npu_hal_t *hal, uint64_t *base_out,
                          uint64_t *size_out);

/* Raw address access is reserved for BL1/RPC implementation inside Service. */
int npu_hal_copy_to_address(npu_hal_t *hal, uint32_t cluster,
                            uint64_t address, const void *source, size_t size);
int npu_hal_copy_from_address(npu_hal_t *hal, uint32_t cluster,
                              void *destination, uint64_t address,
                              size_t size);
/*
 * Copies through a Host bounce buffer in 64 KiB chunks.  A backend with a
 * copy check validates both complete ranges first, so a software range error
 * changes nothing; otherwise chunks before the failing one stay written.
 * Overlapping ranges are not supported.
 */
int npu_hal_copy_address(npu_hal_t *hal, uint32_t cluster,
                         uint64_t destination, uint64_t source, size_t size);

int npu_hal_register_read(npu_hal_t *hal, uint32_t cluster, uint64_t offset,
                          void *value, size_t size);
int npu_hal_register_write(npu_hal_t *hal, uint32_t cluster, uint64_t offset,
                           const void *value, size_t size);
/* Caller serializes the complete snapshot against other PCU users. */
int npu_hal_pcu_snapshot(npu_hal_t *hal, uint32_t cluster,
                          npu_pcu_batch_t *batch);
int npu_hal_irq_status(npu_hal_t *hal, uint32_t cluster, uint32_t *status);
/* Returns -ETIMEDOUT with cluster/status reset when no IRQ arrives. */
int npu_hal_irq_wait(npu_hal_t *hal, uint32_t timeout_ms,
                     uint32_t *cluster, uint32_t *status);
int npu_hal_irq_ack(npu_hal_t *hal, uint32_t cluster, uint32_t mask);
int npu_hal_irq_mailbox(npu_hal_t *hal, npu_hal_core_t core,
                        uint64_t *command);
int npu_hal_command_submit(npu_hal_t *hal, npu_hal_core_t core,
                           uint64_t command);
int npu_hal_wait_command_consumed(npu_hal_t *hal, npu_hal_core_t core,
                                  uint32_t timeout_ms);
int npu_hal_boot_core(npu_hal_t *hal, npu_hal_core_t core, uint64_t entry,
                      uint32_t timeout_ms);

/*
 * Hardware reset of one Cluster.  Returns -ENOTSUP while the backend has no
 * reset sequence, which is every backend today: the Service still restarts
 * Harts by parking them in the BootROM.  Success has to mean every Hart and
 * AXI master has stopped, so memory the Cluster could reach may be reused.
 */
int npu_hal_cluster_reset(npu_hal_t *hal, uint32_t cluster);

/* Clear only exclusively allocated Shared Resource entries before Proc launch.
 * The selected physical entries must have no outstanding users or transactions. */
int npu_hal_shared_resource_init(npu_hal_t *hal, uint32_t cluster, uint32_t mask);

/*
 * ClusterMMU APB4 register access.  Offsets are the DAYAN_MMU_* values from
 * platform_regs.h, relative to the MMU register page rather than to the
 * window npu_hal_register_* addresses.
 *
 * Writes to a base bound to a translation window are rejected with -EINVAL
 * when not 4 KiB aligned, because the hardware would silently ignore them
 * and report no error.  npu_hal_mmu_write64() likewise rejects values that
 * do not fit 56 bits instead of truncating them.
 */
int npu_hal_mmu_present(npu_hal_t *hal, uint32_t cluster, int *present);
int npu_hal_mmu_read_reg(npu_hal_t *hal, uint32_t cluster, uint32_t offset,
                         uint32_t *value);
int npu_hal_mmu_write_reg(npu_hal_t *hal, uint32_t cluster, uint32_t offset,
                          uint32_t value);
int npu_hal_mmu_read64(npu_hal_t *hal, uint32_t cluster, uint32_t offset_lo,
                       uint64_t *value);
int npu_hal_mmu_write64(npu_hal_t *hal, uint32_t cluster, uint32_t offset_lo,
                        uint64_t value);

/*
 * Invalidate the Cluster TLB.  Returns -EBUSY if the hardware rejected the
 * request because an earlier flush was still running (nothing was
 * invalidated; retry), and -ETIMEDOUT if FLUSH_BUSY never cleared.
 *
 * The caller must have established that the Cluster has no AXI traffic in
 * flight before calling this.  There is no quiesce register to enforce it:
 * writing TLB_FLUSH with transactions outstanding leaves the drain waiting
 * forever and FLUSH_BUSY permanently set, which no error path here can undo.
 * In this tree that means no active launches, which is what the Service's
 * deferred-reclaim path waits for.
 */
int npu_hal_mmu_flush(npu_hal_t *hal, uint32_t cluster, int asid_selective,
                      uint8_t asid, uint32_t timeout_ms);

/*
 * ClusterMMU Sv39 page tables and guard pages (hal/mmu_pt.c).
 *
 * Mechanism only: one table per Cluster, 2 MiB level-1 leaves, and one guard
 * page per fault class behind the redirect bases.  Which addresses are mapped,
 * and when a TLB flush is safe, are the caller's decisions - none of these
 * calls flushes.  Unmapping or narrowing permissions therefore leaves stale
 * TLB entries until the caller runs npu_hal_mmu_flush() on an idle Cluster.
 *
 * va, pa and size are 2 MiB multiples; va must lie in
 * [NPU_HAL_MMU_BYPASS_LIMIT, NPU_HAL_MMU_VA_LIMIT), because the hardware
 * bypasses the table below 2 GiB and Sv39 ends at 2^38.
 */
typedef struct npu_hal_mmu npu_hal_mmu_t;

#define NPU_HAL_MMU_READ          (1U << 0)
#define NPU_HAL_MMU_WRITE         (1U << 1)
#define NPU_HAL_MMU_EXEC          (1U << 2)

#define NPU_HAL_MMU_LEAF_SIZE     UINT64_C(0x200000)
#define NPU_HAL_MMU_LEAF_MASK     UINT64_C(0x1fffff)
#define NPU_HAL_MMU_BYPASS_LIMIT  UINT64_C(0x80000000)
#define NPU_HAL_MMU_VA_LIMIT      UINT64_C(0x0000004000000000)

/*
 * Build an empty table and the guard pages for every Cluster and program
 * PTW_BASE, GLOBAL_ASID and the redirect bases.  Translation is not switched
 * on: FixMap still identity-maps everything until npu_hal_mmu_activate().
 *
 * Tables and guard pages are reserved from [pool_base, pool_base + pool_size),
 * top-down; the range must be Device-reachable Normal memory.  Returns
 * -ENODEV if any Cluster has no MMU.
 */
int npu_hal_mmu_create(npu_hal_t *hal, uint32_t cluster_count,
                       uint64_t pool_base, uint64_t pool_size,
                       npu_hal_mmu_t **mmu_out);
/* Deactivates first if still active, then frees every table and guard page. */
void npu_hal_mmu_destroy(npu_hal_mmu_t *mmu);
/* Narrow FixMap to the low 2 GiB on every Cluster, so addresses at and above
 * NPU_HAL_MMU_BYPASS_LIMIT go through the tables.  Map before activating. */
int npu_hal_mmu_activate(npu_hal_mmu_t *mmu);
/* Restore the reset-time full-range identity FixMap on every Cluster. */
int npu_hal_mmu_deactivate(npu_hal_mmu_t *mmu);
/* Install leaves.  -EEXIST if any leaf is already valid; on any failure the
 * leaves this call installed are removed again. */
int npu_hal_mmu_map(npu_hal_mmu_t *mmu, uint32_t cluster, uint64_t va,
                    uint64_t pa, uint64_t size, uint32_t prot);
/* Clear leaves; absent ones are skipped. */
int npu_hal_mmu_unmap(npu_hal_mmu_t *mmu, uint32_t cluster, uint64_t va,
                      uint64_t size);
/* Rewrite permissions in place; -ENOENT if any leaf is absent. */
int npu_hal_mmu_protect(npu_hal_mmu_t *mmu, uint32_t cluster, uint64_t va,
                        uint64_t size, uint32_t prot);
/*
 * Returns 1 and fills the outputs when a guard page shows a fault, else 0,
 * and refills the page.  Sampling only: the offset is the faulting VA[11:0];
 * the upper bits and the count are not observable on this hardware.
 */
int npu_hal_mmu_check_faults(npu_hal_mmu_t *mmu, uint32_t cluster,
                             uint32_t *fault_class, uint32_t *fault_offset);

/* Physical address allocation and ownership live entirely in HAL. */
int npu_hal_memory_alloc(npu_hal_t *hal,
                         const npu_hal_memory_request_t *request,
                         npu_hal_allocation_t **allocation_out);
int npu_hal_memory_reserve(npu_hal_t *hal,
                           const npu_hal_memory_request_t *request,
                           npu_hal_allocation_t **allocation_out);
int npu_hal_memory_free(npu_hal_t *hal, npu_hal_allocation_t *allocation);
int npu_hal_memory_info(npu_hal_t *hal,
                        const npu_hal_allocation_t *allocation,
                        npu_hal_allocation_info_t *info);
int npu_hal_memory_visible(npu_hal_t *hal,
                           const npu_hal_allocation_t *allocation,
                           npu_hal_core_t viewer);
int npu_hal_address_valid(npu_hal_t *hal, npu_hal_core_t viewer,
                          uint64_t address, uint64_t size);
int npu_hal_memory_address(npu_hal_t *hal,
                           const npu_hal_allocation_t *allocation,
                           npu_hal_core_t viewer, uint64_t *address_out);
int npu_hal_memory_write(npu_hal_t *hal,
                         const npu_hal_allocation_t *allocation,
                         npu_hal_core_t viewer, uint64_t offset,
                         const void *source, size_t size);
int npu_hal_memory_read(npu_hal_t *hal,
                        const npu_hal_allocation_t *allocation,
                        npu_hal_core_t viewer, void *destination,
                        uint64_t offset, size_t size);

/* Control-side data plane. Runtime only consumes dataplane.h. */
int npu_hal_data_plane_export(npu_hal_t *hal,
                              npu_hal_data_export_t *exported,
                              int handles[NPU_HAL_DATA_MAX_HANDLES]);
int npu_hal_transfer_prepare(npu_hal_t *hal,
                             const npu_hal_allocation_t *allocation,
                             npu_hal_core_t viewer, uint64_t offset,
                             size_t size,
                             npu_hal_data_direction_t direction,
                             npu_hal_data_token_t *token);
int npu_hal_transfer_finish(npu_hal_t *hal,
                            const npu_hal_data_token_t *token,
                            int transfer_status);

#ifdef __cplusplus
}
#endif

#endif
