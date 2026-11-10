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

#ifndef NPU_HOST_HAL_PUBLIC_CONFIG_H
#define NPU_HOST_HAL_PUBLIC_CONFIG_H

/* Public HAL topology and backend configuration. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_HAL_OPTIONS_VERSION         7U
#define NPU_HAL_MAX_CLUSTERS            32U
#define NPU_HAL_MAX_CORES_PER_CLUSTER   32U

#define NPU_HAL_BACKEND_XILINX_DMA      1U
#define NPU_HAL_BACKEND_SOC_MMAP        2U
#define NPU_HAL_BACKEND_INPROCESS       3U
/* Source compatibility for existing simulator/HAL callers. */
#define NPU_HAL_BACKEND_XDMA            NPU_HAL_BACKEND_XILINX_DMA
#define NPU_HAL_ANY_CLUSTER             UINT32_MAX
#define NPU_HAL_ANY_CORE                UINT32_MAX

typedef struct npu_hal_core {
    uint32_t cluster;
    uint32_t core;
} npu_hal_core_t;

typedef enum npu_hal_memory_class {
    NPU_HAL_MEMORY_NORMAL = 1,
    NPU_HAL_MEMORY_COHERENT = 2,
    NPU_HAL_MEMORY_URAM = 3,
} npu_hal_memory_class_t;

typedef enum npu_hal_memory_scope {
    NPU_HAL_MEMORY_DEVICE = 1,
    NPU_HAL_MEMORY_CLUSTER = 2,
    NPU_HAL_MEMORY_CORE = 3,
} npu_hal_memory_scope_t;

/* A backend may expose multiple regions with different visibility scopes. */
typedef struct npu_hal_memory_region {
    uint64_t base;
    uint64_t size;
    uint32_t memory_class;
    uint32_t scope;
    uint32_t cluster;
    uint32_t core;
} npu_hal_memory_region_t;

typedef struct npu_hal_backend_options {
    uint32_t version;
    uint32_t backend;
} npu_hal_backend_options_t;

/* Configuration is consumed only by dayan_server, never by applications. */
typedef struct npu_hal_options {
    uint32_t version;
    uint32_t cluster_count;
    uint32_t cores_per_cluster;
    uint32_t normal_memory_size_mib;
    /* Per-core URAM bytes: 0x180000 (default) or 0xc0000.  Used by default
     * regions and SoC access bounds; explicit memory_regions define pools. */
    uint32_t uram_size;
    /* Mark explicit overrides, including values equal to the defaults.
     * Non-default cores_per_cluster/uram_size are also treated as overrides. */
    uint32_t manual_cores_per_cluster;
    uint32_t manual_uram_size;
    /*
     * Split the Normal region into a firmware pool and a Buffer pool.  Only the
     * MMU needs the split - it is what keeps a Buffer's physical pages out of
     * the linear-offset firmware window - and it costs the firmware pool's
     * worth of address space to every other usage, so it stays off unless the
     * caller enables the MMU.  npu_hal_firmware_pool() reports the boundary and
     * returns -ENODEV when the split is off.
     */
    uint32_t split_normal_memory;
    const npu_hal_backend_options_t *backend_options;
    const npu_hal_memory_region_t *memory_regions;
    uint32_t memory_region_count;
} npu_hal_options_t;

void npu_hal_options_init(npu_hal_options_t *options);

#ifdef __cplusplus
}
#endif

#endif
