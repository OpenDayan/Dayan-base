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
    return 0;
}
