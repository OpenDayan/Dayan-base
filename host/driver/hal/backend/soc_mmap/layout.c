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

#include "layout.h"

#include <errno.h>
#include <string.h>

#include "npu_hal_config.h"
#include "platform_regs.h"

static const char *const uram_names[DAYAN_CLUSTER_MAX_HARTS] = {
    "uram0", "uram1", "uram2", "uram3",
};

/*
 * Access widths are the ones the Host driver uses and the rvtng model
 * accepts; the RTL has not confirmed them (see README.md):
 *   - Device and IRQ mailboxes, atomic-cache control: 64-bit.
 *   - IRQ control (STATUS, ENABLE, SET, CLR, ENABLE_SET, ENABLE_CLR), MSIP,
 *     resource init: 32-bit.
 *   - ClusterMMU: 32-bit only; a 56-bit value is a LO/HI pair the HAL writes
 *     LO first (hal/mmu.c).
 *   - PCU: 64-bit single beats.  Values are stable because a write to
 *     PCU_CONTROL latches every counter into its hold register first
 *     (hal/pcu.c); a split read could still tear, so it is refused.
 */
int soc_layout_build(uint32_t cores, uint32_t uram_size,
                     struct soc_layout_entry *entries,
                     uint32_t capacity)
{
    const struct soc_layout_entry fixed[] = {
        { "device-mailbox", SOC_LAYOUT_REGISTER, 8,
          DAYAN_DEVICE_MAILBOX_BASE, (uint64_t)cores * 8U, 0,
          NPU_HAL_ANY_CORE },
        { "irq-control", SOC_LAYOUT_REGISTER, 4,
          DAYAN_IRQ_STATUS, DAYAN_IRQ_CONTROL_SIZE, 0, NPU_HAL_ANY_CORE },
        { "irq-mailbox", SOC_LAYOUT_REGISTER, 8,
          DAYAN_IRQ_MAILBOX_BASE, (uint64_t)cores * 8U, 0,
          NPU_HAL_ANY_CORE },
        { "atomic-cache-control", SOC_LAYOUT_REGISTER, 8,
          DAYAN_ATOMIC_CACHE_CONTROL_BASE, DAYAN_ATOMIC_CACHE_CONTROL_SIZE, 0,
          NPU_HAL_ANY_CORE },
        { "mmu", SOC_LAYOUT_REGISTER, 4,
          DAYAN_MMU_APB_BASE, DAYAN_MMU_APB_SIZE, 0, NPU_HAL_ANY_CORE },
        { "resource-init", SOC_LAYOUT_REGISTER, 4,
          DAYAN_RESOURCE_GLOBAL_INIT, 4, 0, NPU_HAL_ANY_CORE },
        { "msip", SOC_LAYOUT_REGISTER, 4,
          DAYAN_CLINT_MSIP_BASE, (uint64_t)cores * 4U, 0, NPU_HAL_ANY_CORE },
        { "pcu", SOC_LAYOUT_REGISTER, 8,
          DAYAN_PCU_BASE, (uint64_t)cores * DAYAN_PCU_CORE_STRIDE, 0,
          NPU_HAL_ANY_CORE },
        { "coherent", SOC_LAYOUT_MEMORY, 0,
          DAYAN_COHERENT_WINDOW_BASE, DAYAN_ATOMIC_CACHE_WINDOW_SIZE,
          NPU_HAL_MEMORY_COHERENT, NPU_HAL_ANY_CORE },
    };
    uint32_t count = sizeof(fixed) / sizeof(fixed[0]);
    uint32_t index;

    if (!entries || !cores || cores > DAYAN_CLUSTER_MAX_HARTS ||
        (uram_size != DAYAN_URAM_SIZE_768K &&
         uram_size != DAYAN_URAM_SIZE_1536K))
        return -EINVAL;
    if (capacity < count + cores)
        return -E2BIG;
    memcpy(entries, fixed, sizeof(fixed));
    for (index = 0; index < cores; ++index) {
        entries[count++] = (struct soc_layout_entry){
            uram_names[index], SOC_LAYOUT_MEMORY, 0, DAYAN_URAM(index),
            uram_size, NPU_HAL_MEMORY_URAM, index
        };
    }
    /* The table is fixed, but a wrong edit to platform_regs.h should fail
     * here rather than alias two modules. */
    for (index = 0; index < count; ++index) {
        const struct soc_layout_entry *e = &entries[index];

        if (!e->size || e->offset > DAYAN_CLUSTER_APERTURE_SIZE ||
            e->size > DAYAN_CLUSTER_APERTURE_SIZE - e->offset ||
            (e->kind == SOC_LAYOUT_REGISTER &&
             ((e->offset | e->size) & (e->access_width - 1U))) ||
            (index && e->offset < entries[index - 1].offset +
                                  entries[index - 1].size))
            return -EPROTO;
    }
    return (int)count;
}
