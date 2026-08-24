#include "npu_hal_xdma.h"

#include <errno.h>
#include <string.h>

int npu_hal_xdma_options_init(npu_hal_xdma_options_t *options,
                              npu_hal_xdma_profile_t profile)
{
    if (!options)
        return -EINVAL;
    memset(options, 0, sizeof(*options));
    options->base.version = NPU_HAL_XDMA_OPTIONS_VERSION;
    options->base.backend = NPU_HAL_BACKEND_XDMA;
    options->profile = (uint32_t)profile;
    options->cluster_device_stride = 1;
    switch (profile) {
    case NPU_HAL_XDMA_PROFILE_SIMULATOR:
        options->device_directory =
            NPU_HAL_XDMA_SIMULATOR_DEVICE_DIRECTORY;
        options->mmio_address_offset =
            NPU_HAL_XDMA_SIMULATOR_MMIO_OFFSET;
        return 0;
    case NPU_HAL_XDMA_PROFILE_FPGA:
        options->device_directory = NPU_HAL_XDMA_FPGA_DEVICE_DIRECTORY;
        options->mmio_address_offset = NPU_HAL_XDMA_FPGA_MMIO_OFFSET;
        return 0;
    default:
        memset(options, 0, sizeof(*options));
        return -EINVAL;
    }
}
