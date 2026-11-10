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

#define _POSIX_C_SOURCE 200809L

#include "dataplane_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <unistd.h>

#include "io.h"

struct xilinx_dma_data_endpoint {
    int h2d_fd;
    int d2h_fd;
};

struct xilinx_dma_data_plane {
    uint32_t endpoint_count;
    struct xilinx_dma_data_endpoint endpoints[];
};

static int xilinx_dma_data_import(const npu_hal_data_export_t *exported,
                            int *handles, size_t handle_count,
                            void **state_out)
{
    struct xilinx_dma_data_plane *data_plane;
    uint64_t endpoint_count = exported->opaque[0];
    size_t client_size;
    size_t index;

    if (!state_out || !endpoint_count ||
        endpoint_count > NPU_HAL_DATA_MAX_HANDLES / 2U ||
        handle_count != endpoint_count * 2U || exported->opaque[1] ||
        exported->opaque[2] || exported->opaque[3] ||
        endpoint_count > (SIZE_MAX - sizeof(*data_plane)) /
                             sizeof(data_plane->endpoints[0]))
        return -EPROTO;
    client_size = sizeof(*data_plane) +
                  (size_t)endpoint_count * sizeof(data_plane->endpoints[0]);
    data_plane = calloc(1, client_size);
    if (!data_plane)
        return -ENOMEM;
    data_plane->endpoint_count = (uint32_t)endpoint_count;
    for (index = 0; index < endpoint_count; ++index) {
        if (handles[index * 2U] < 0 || handles[index * 2U + 1U] < 0) {
            while (index) {
                --index;
                close(data_plane->endpoints[index].d2h_fd);
                close(data_plane->endpoints[index].h2d_fd);
            }
            free(data_plane);
            return -EBADF;
        }
        data_plane->endpoints[index].h2d_fd = handles[index * 2U];
        data_plane->endpoints[index].d2h_fd = handles[index * 2U + 1U];
        handles[index * 2U] = -1;
        handles[index * 2U + 1U] = -1;
    }
    *state_out = data_plane;
    return 0;
}

static void xilinx_dma_data_close(void *opaque)
{
    struct xilinx_dma_data_plane *data_plane = opaque;
    uint32_t index;

    if (!data_plane)
        return;
    for (index = 0; index < data_plane->endpoint_count; ++index) {
        close(data_plane->endpoints[index].d2h_fd);
        close(data_plane->endpoints[index].h2d_fd);
    }
    free(data_plane);
}

static int xilinx_dma_data_transfer(void *opaque,
                              const npu_hal_data_token_t *token,
                              void *host_buffer, size_t size)
{
    struct xilinx_dma_data_plane *data_plane = opaque;
    uint64_t offset = token->opaque[0];
    uint64_t endpoint = token->opaque[1];

    if (endpoint >= data_plane->endpoint_count || token->opaque[2] ||
        token->opaque[3])
        return -EACCES;
    switch (token->direction) {
    case NPU_HAL_DATA_HOST_TO_DEVICE:
        return npu_hal_xilinx_dma_pwrite(
            data_plane->endpoints[endpoint].h2d_fd, offset,
            host_buffer, size);
    case NPU_HAL_DATA_DEVICE_TO_HOST:
        return npu_hal_xilinx_dma_pread(
            data_plane->endpoints[endpoint].d2h_fd, offset,
            host_buffer, size);
    default:
        return -EINVAL;
    }
}

const struct npu_hal_data_plane_ops npu_hal_xilinx_dma_data_plane_ops = {
    .provider = NPU_HAL_DATA_PROVIDER_XILINX_DMA,
    .import = xilinx_dma_data_import,
    .close = xilinx_dma_data_close,
    .transfer = xilinx_dma_data_transfer,
};
