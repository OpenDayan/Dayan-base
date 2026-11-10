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

#ifndef NPU_HOST_HAL_XILINX_DMA_H
#define NPU_HOST_HAL_XILINX_DMA_H
#include "npu_hal_config.h"
#ifdef __cplusplus
extern "C" {
#endif
#define NPU_HAL_XILINX_DMA_OPTIONS_VERSION 2U
#define NPU_HAL_XILINX_DMA_MMIO_LIMIT UINT64_C(0x80000000)
typedef enum {
    NPU_HAL_XILINX_DMA_DRIVER_XDMA = 1,
    NPU_HAL_XILINX_DMA_DRIVER_QDMA = 2,
} npu_hal_xilinx_dma_driver_t;
typedef enum {
    NPU_HAL_XILINX_DMA_PROFILE_SIMULATOR = 1,
    NPU_HAL_XILINX_DMA_PROFILE_FPGA = 2,
} npu_hal_xilinx_dma_profile_t;
typedef enum {
    NPU_HAL_XILINX_DMA_COMPLETION_IRQ = 1,
    NPU_HAL_XILINX_DMA_COMPLETION_POLLING = 2,
    /* Optional-event behavior for the legacy command-line interface only. */
    NPU_HAL_XILINX_DMA_COMPLETION_LEGACY = 3,
} npu_hal_xilinx_dma_completion_t;
typedef struct {
    npu_hal_backend_options_t base;
    uint32_t driver;
    uint32_t profile;
    uint32_t completion;
    const char *device_directory;
    const char *device_name;
    uint32_t device_index;
    uint32_t cluster_device_stride;
    uint32_t channel;
    uint32_t queue;
    uint32_t event;
    uint32_t user_irq;
    uint64_t mmio_address_offset;
} npu_hal_xilinx_dma_options_t;
int npu_hal_xilinx_dma_options_init(npu_hal_xilinx_dma_options_t *options,
                                    npu_hal_xilinx_dma_profile_t profile);
#ifdef __cplusplus
}
#endif
#endif
