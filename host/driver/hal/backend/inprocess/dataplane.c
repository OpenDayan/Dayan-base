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

#include "dataplane_internal.h"
#include "io.h"

#include <errno.h>

static int import_data(const npu_hal_data_export_t *exported, int *handles,
                       size_t handle_count, void **state_out)
{
    struct npu_inprocess_device *device;
    int status;

    (void)handles;
    if (handle_count || exported->opaque[2] || exported->opaque[3])
        return -EPROTO;
    status = npu_inprocess_acquire(exported->opaque[0], exported->opaque[1],
                                  &device);
    if (status == 0)
        *state_out = device;
    return status;
}

static void close_data(void *state)
{
    npu_inprocess_release(state);
}

static int transfer(void *state, const npu_hal_data_token_t *token,
                     void *host_buffer, size_t size)
{
    if (token->opaque[1] >= npu_inprocess_clusters(state) ||
        token->opaque[2] != npu_inprocess_id(state) || token->opaque[3])
        return -EACCES;
    switch (token->direction) {
    case NPU_HAL_DATA_HOST_TO_DEVICE:
        return npu_inprocess_write(state, (uint32_t)token->opaque[1],
                                   token->opaque[0], host_buffer, size);
    case NPU_HAL_DATA_DEVICE_TO_HOST:
        return npu_inprocess_read(state, (uint32_t)token->opaque[1],
                                  token->opaque[0], host_buffer, size);
    default:
        return -EINVAL;
    }
}

const struct npu_hal_data_plane_ops npu_hal_inprocess_data_plane_ops = {
    .provider = NPU_HAL_DATA_PROVIDER_INPROCESS,
    .import = import_data,
    .close = close_data,
    .transfer = transfer,
};
