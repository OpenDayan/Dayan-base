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
extern const struct npu_hal_data_plane_ops npu_hal_inprocess_data_plane_ops;

#endif
