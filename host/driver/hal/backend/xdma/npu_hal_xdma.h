#ifndef NPU_HOST_HAL_BACKEND_XDMA_H
#define NPU_HOST_HAL_BACKEND_XDMA_H

#include "npu_hal_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_HAL_XDMA_OPTIONS_VERSION    1U
#define NPU_HAL_XDMA_MMIO_LIMIT         UINT64_C(0x80000000)

#define NPU_HAL_XDMA_SIMULATOR_DEVICE_DIRECTORY "/tmp/virtual_npu"
#define NPU_HAL_XDMA_FPGA_DEVICE_DIRECTORY      "/dev"
#define NPU_HAL_XDMA_SIMULATOR_MMIO_OFFSET      UINT64_C(0)
#define NPU_HAL_XDMA_FPGA_MMIO_OFFSET           UINT64_C(0x4000000000)

typedef enum npu_hal_xdma_profile {
    NPU_HAL_XDMA_PROFILE_SIMULATOR = 1,
    NPU_HAL_XDMA_PROFILE_FPGA = 2,
} npu_hal_xdma_profile_t;

typedef struct npu_hal_xdma_options {
    npu_hal_backend_options_t base;
    uint32_t profile;
    uint32_t device_index;
    uint32_t cluster_device_stride;
    uint32_t reserved;
    const char *device_directory;
    uint64_t mmio_address_offset;
} npu_hal_xdma_options_t;

int npu_hal_xdma_options_init(npu_hal_xdma_options_t *options,
                              npu_hal_xdma_profile_t profile);

#ifdef __cplusplus
}
#endif

#endif
