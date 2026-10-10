#ifndef NPU_HOST_HAL_DATAPLANE_INTERNAL_H
#define NPU_HOST_HAL_DATAPLANE_INTERNAL_H

#include "dataplane.h"

struct npu_hal_data_plane_ops {
    uint32_t provider;
    int (*import)(const npu_hal_data_export_t *exported, int *handles,
                  size_t handle_count, void **state_out);
    void (*close)(void *state);
    int (*transfer)(void *state, const npu_hal_data_token_t *token,
                    void *host_buffer, size_t size);
};

extern const struct npu_hal_data_plane_ops npu_hal_xilinx_dma_data_plane_ops;
extern const struct npu_hal_data_plane_ops npu_hal_soc_mmap_data_plane_ops;

#endif
