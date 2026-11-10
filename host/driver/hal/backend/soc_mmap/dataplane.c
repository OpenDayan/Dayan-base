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
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "io.h"

struct soc_data_plane {
    int fd;
    uint32_t coherence;
    pthread_mutex_t mutex;
    /* The shared DDR as the Server exported it; fixed at import. */
    uint64_t ddr_host_base;
    uint64_t ddr_size;
    /* Mapped on the first valid transfer. */
    struct npu_hal_soc_mmap_mapping ddr;
};

/* The same constraints the Server applied to its own DDR options. */
static int ddr_range_valid(uint64_t host_base, uint64_t size)
{
    long page = sysconf(_SC_PAGESIZE);

    return page > 0 && host_base && size &&
           !((host_base | size) & ((uint64_t)page - 1U)) &&
           host_base <= UINT64_MAX - size &&
           host_base + size <= (uint64_t)INT64_MAX && size <= SIZE_MAX;
}

static int soc_data_import(const npu_hal_data_export_t *exported,
                           int *handles, size_t handle_count,
                           void **state_out)
{
    struct soc_data_plane *data_plane;
    int status;

    if (!state_out || handle_count != 1U ||
        exported->opaque[0] != NPU_HAL_SOC_MMAP_TOKEN_LAYOUT ||
        exported->opaque[1] > UINT32_MAX ||
        !ddr_range_valid(exported->opaque[2], exported->opaque[3]))
        return -EPROTO;
    /* Refuse a coherence model this Runtime cannot honour. */
    if (npu_hal_soc_mmap_ddr_sync_for_cpu((uint32_t)exported->opaque[1],
                                          0, 0) < 0)
        return -ENOTSUP;
    if (handles[0] < 0)
        return -EBADF;
    data_plane = calloc(1, sizeof(*data_plane));
    if (!data_plane)
        return -ENOMEM;
    status = pthread_mutex_init(&data_plane->mutex, 0);
    if (status) {
        free(data_plane);
        return -status;
    }
    data_plane->fd = handles[0];
    data_plane->coherence = (uint32_t)exported->opaque[1];
    data_plane->ddr_host_base = exported->opaque[2];
    data_plane->ddr_size = exported->opaque[3];
    handles[0] = -1;
    *state_out = data_plane;
    return 0;
}

static void soc_data_close(void *opaque)
{
    struct soc_data_plane *data_plane = opaque;

    if (!data_plane)
        return;
    npu_hal_soc_mmap_unmap(&data_plane->ddr);
    pthread_mutex_destroy(&data_plane->mutex);
    close(data_plane->fd);
    free(data_plane);
}

/* Called with the mutex held, only after the token was validated. */
static int ddr_mapping(struct soc_data_plane *data_plane)
{
    if (data_plane->ddr.base)
        return 0;
    return npu_hal_soc_mmap_map(data_plane->fd, data_plane->ddr_host_base,
                                data_plane->ddr_size, &data_plane->ddr);
}

/*
 * The same protocol as the Server's own DDR copies (soc_mmap.c soc_copy()):
 * fence, copy with aligned accesses, hand ownership over, fence.  What makes
 * the data visible is the coherence model the Server declared, not the
 * fences.
 */
static int soc_data_transfer(void *opaque, const npu_hal_data_token_t *token,
                             void *host_buffer, size_t size)
{
    struct soc_data_plane *data_plane = opaque;
    uint64_t offset = token->opaque[2];
    volatile unsigned char *target;
    int status;

    /*
     * Every token, the first included, must name the exported DDR and stay
     * inside it; nothing is mapped or touched otherwise.  This keeps a stale
     * or corrupted token from reaching another physical range through this
     * provider.  It does not confine the process: the descriptor it holds can
     * be mmap()ed at any offset the device allows, token or not.
     */
    if (token->opaque[0] != data_plane->ddr_host_base ||
        token->opaque[1] != data_plane->ddr_size || token->opaque[3] ||
        offset > data_plane->ddr_size || !size ||
        size > data_plane->ddr_size - offset)
        return -EACCES;
    if (token->direction != NPU_HAL_DATA_HOST_TO_DEVICE &&
        token->direction != NPU_HAL_DATA_DEVICE_TO_HOST)
        return -EINVAL;
    status = pthread_mutex_lock(&data_plane->mutex);
    if (status)
        return -status;
    status = ddr_mapping(data_plane);
    if (status == 0) {
        target = data_plane->ddr.window + offset;
        npu_hal_soc_mmap_io_fence();
        if (token->direction == NPU_HAL_DATA_HOST_TO_DEVICE) {
            npu_hal_soc_mmap_mem_write(target, host_buffer, size);
            status = npu_hal_soc_mmap_ddr_sync_for_device(
                data_plane->coherence, target, size);
        } else {
            status = npu_hal_soc_mmap_ddr_sync_for_cpu(
                data_plane->coherence, target, size);
            if (status == 0)
                npu_hal_soc_mmap_mem_read(host_buffer, target, size);
        }
        npu_hal_soc_mmap_io_fence();
    }
    pthread_mutex_unlock(&data_plane->mutex);
    return status;
}

const struct npu_hal_data_plane_ops npu_hal_soc_mmap_data_plane_ops = {
    .provider = NPU_HAL_DATA_PROVIDER_SOC_MMAP,
    .import = soc_data_import,
    .close = soc_data_close,
    .transfer = soc_data_transfer,
};
