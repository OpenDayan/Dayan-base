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

#include "npu_hal_xilinx_dma.h"
#include <errno.h>
#include <string.h>
int npu_hal_xilinx_dma_options_init(npu_hal_xilinx_dma_options_t *options,
                                    npu_hal_xilinx_dma_profile_t profile)
{
    if (!options || (profile != NPU_HAL_XILINX_DMA_PROFILE_SIMULATOR &&
                    profile != NPU_HAL_XILINX_DMA_PROFILE_FPGA))
        return -EINVAL;
    memset(options, 0, sizeof(*options));
    options->base.backend = NPU_HAL_BACKEND_XILINX_DMA;
    options->base.version = NPU_HAL_XILINX_DMA_OPTIONS_VERSION;
    options->driver = NPU_HAL_XILINX_DMA_DRIVER_XDMA;
    options->profile = profile;
    options->completion = NPU_HAL_XILINX_DMA_COMPLETION_IRQ;
    options->cluster_device_stride = 1;
    options->device_directory = profile == NPU_HAL_XILINX_DMA_PROFILE_FPGA ?
                                "/dev" : "/tmp/virtual_npu";
    options->mmio_address_offset = profile == NPU_HAL_XILINX_DMA_PROFILE_FPGA ?
                                  UINT64_C(0x4000000000) : 0;
    options->control = NPU_HAL_XILINX_DMA_CONTROL_AUTO;
    options->control_bar = NPU_HAL_XILINX_DMA_QDMA_CONTROL_BAR;
    options->control_offset = NPU_HAL_XILINX_DMA_QDMA_CONTROL_OFFSET;
    options->wide_bar = NPU_HAL_XILINX_DMA_QDMA_WIDE_BAR;
    options->wide_offset = NPU_HAL_XILINX_DMA_QDMA_WIDE_OFFSET;
    return 0;
}
