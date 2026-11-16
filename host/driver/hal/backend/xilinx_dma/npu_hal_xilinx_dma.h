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
#define NPU_HAL_XILINX_DMA_OPTIONS_VERSION 4U
#define NPU_HAL_XILINX_DMA_MMIO_LIMIT UINT64_C(0x80000000)
/*
 * QDMA card (rvt-qdma-driver-interface.md): BAR1 is the QDMA AXI-Lite master,
 * translated to AXI 0x10000000; the Cluster aperture starts at AXI 0x11000000,
 * so IP offset A is BAR1 offset 0x01000000 + A.  The path only guarantees
 * aligned 32-bit accesses.
 */
#define NPU_HAL_XILINX_DMA_QDMA_CONTROL_BAR    1U
#define NPU_HAL_XILINX_DMA_QDMA_CONTROL_OFFSET UINT64_C(0x01000000)
/*
 * BAR4 (16 MiB) reaches the same aperture through the QDMA AXI Bridge master
 * (AXI 0x10_0000_0000, BAR offset A = IP offset A) and carries single 64-bit
 * MMIO accesses.  Registers that need one 64-bit beat (mailboxes, Atomic
 * Cache control, PCU) go through it; everything else stays on BAR1.
 */
#define NPU_HAL_XILINX_DMA_QDMA_WIDE_BAR       4U
#define NPU_HAL_XILINX_DMA_QDMA_WIDE_OFFSET    UINT64_C(0)
#define NPU_HAL_XILINX_DMA_NO_BAR              UINT32_MAX
#define NPU_HAL_XILINX_DMA_PCI_DEVICES         "/sys/bus/pci/devices"
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
/* How addresses below NPU_HAL_XILINX_DMA_MMIO_LIMIT (the Cluster aperture)
 * are reached. */
typedef enum {
    /* BAR for QDMA with the fpga profile, DMA otherwise. */
    NPU_HAL_XILINX_DMA_CONTROL_AUTO = 0,
    /* Through the H2C/C2H nodes at address + mmio_address_offset. */
    NPU_HAL_XILINX_DMA_CONTROL_DMA = 1,
    /* 32-bit MMIO on a mapped PCI BAR at control_offset + address. */
    NPU_HAL_XILINX_DMA_CONTROL_BAR = 2,
} npu_hal_xilinx_dma_control_t;
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
    /* Control path, used for one Cluster only. */
    uint32_t control;
    uint32_t control_bar;
    uint64_t control_offset;
    /* PCI function owning the BAR ("0000:01:00.0" or "01:00.0").  NULL
     * derives it from the QDMA device name, which the qdma-pf driver formats
     * as qdma%05x of (bus << 12 | device << 4 | function), domain 0. */
    const char *pci_device;
    /* File to map instead of <pci_devices>/<pci_device>/resource<bar>. */
    const char *control_resource;
    /* BAR for single 64-bit accesses, NPU_HAL_XILINX_DMA_NO_BAR for none.
     * A missing BAR falls back to two 32-bit accesses on the control BAR. */
    uint32_t wide_bar;
    uint64_t wide_offset;
    const char *wide_resource;
} npu_hal_xilinx_dma_options_t;
int npu_hal_xilinx_dma_options_init(npu_hal_xilinx_dma_options_t *options,
                                    npu_hal_xilinx_dma_profile_t profile);
#ifdef __cplusplus
}
#endif
#endif
