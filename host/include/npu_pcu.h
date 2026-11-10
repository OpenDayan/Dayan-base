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

#ifndef NPU_HOST_PCU_H
#define NPU_HOST_PCU_H

#include <stdint.h>

/* npu-cluster PerformanceCounterUnit / RVTPerfEvent event order. */
#define NPU_PCU_MAX_CORES             4U
#define NPU_PCU_COUNTER_COUNT         112U
#define NPU_PCU_URAM_BANK_COUNT       32U
#define NPU_PCU_VEC_GROUP_COUNT       2U
#define NPU_PCU_VEC_GROUP_STRIDE      21U
#define NPU_PCU_MAT_EXEC              0U
#define NPU_PCU_LSU_READ              1U
#define NPU_PCU_LSU_WRITE             2U
#define NPU_PCU_VEC_ISSUE(group)      (3U + (group) * NPU_PCU_VEC_GROUP_STRIDE)
#define NPU_PCU_URAM_READ(bank)       (45U + (bank))
#define NPU_PCU_URAM_WRITE(bank)      (77U + (bank))
#define NPU_PCU_SCALAR_COMMIT         109U
#define NPU_PCU_AXI_SLAVE_READ        110U
#define NPU_PCU_AXI_SLAVE_WRITE       111U
#define NPU_PCU_ROUND_END             (1U << 0)

typedef struct npu_pcu_core_sample {
    uint64_t cycle;
    uint64_t counters[NPU_PCU_COUNTER_COUNT];
} npu_pcu_core_sample_t;

/* One cluster snapshot per read. Core array indices are physical hart IDs. */
typedef struct npu_pcu_batch {
    uint32_t cluster;
    uint32_t core_count;
    uint32_t flags;
    uint32_t reserved;
    npu_pcu_core_sample_t cores[NPU_PCU_MAX_CORES];
} npu_pcu_batch_t;

#ifdef __cplusplus
static_assert(sizeof(npu_pcu_batch_t) == 3632, "NPU PCU batch ABI");
#else
_Static_assert(sizeof(npu_pcu_batch_t) == 3632, "NPU PCU batch ABI");
#endif

#endif
