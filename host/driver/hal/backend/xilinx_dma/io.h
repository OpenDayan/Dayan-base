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

#ifndef NPU_HOST_HAL_XILINX_DMA_IO_H
#define NPU_HOST_HAL_XILINX_DMA_IO_H

#include <stddef.h>
#include <stdint.h>

int npu_hal_xilinx_dma_pread(int fd, uint64_t offset, void *data, size_t size);
int npu_hal_xilinx_dma_pwrite(int fd, uint64_t offset, const void *data, size_t size);

#endif
