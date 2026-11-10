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

#include "io.h"

#include <errno.h>
#include <limits.h>
#include <sys/types.h>
#include <unistd.h>

#define NPU_HAL_XILINX_DMA_MAX_IO_SIZE \
    ((size_t)64U * (size_t)1024U * (size_t)1024U)

int npu_hal_xilinx_dma_pread(int fd, uint64_t offset, void *data, size_t size)
{
    unsigned char *cursor = data;
    size_t done = 0;

    if (fd < 0 || (!data && size) || offset > (uint64_t)INT64_MAX ||
        size > (uint64_t)INT64_MAX - offset)
        return -EINVAL;
    while (done < size) {
        size_t chunk = size - done;
        ssize_t result;

        if (chunk > NPU_HAL_XILINX_DMA_MAX_IO_SIZE)
            chunk = NPU_HAL_XILINX_DMA_MAX_IO_SIZE;
        result = pread(fd, cursor + done, chunk,
                       (off_t)(offset + done));

        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0)
            return result < 0 ? -errno : -EIO;
        done += (size_t)result;
    }
    return 0;
}

int npu_hal_xilinx_dma_pwrite(int fd, uint64_t offset, const void *data, size_t size)
{
    const unsigned char *cursor = data;
    size_t done = 0;

    if (fd < 0 || (!data && size) || offset > (uint64_t)INT64_MAX ||
        size > (uint64_t)INT64_MAX - offset)
        return -EINVAL;
    while (done < size) {
        size_t chunk = size - done;
        ssize_t result;

        if (chunk > NPU_HAL_XILINX_DMA_MAX_IO_SIZE)
            chunk = NPU_HAL_XILINX_DMA_MAX_IO_SIZE;
        result = pwrite(fd, cursor + done, chunk,
                        (off_t)(offset + done));

        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0)
            return result < 0 ? -errno : -EIO;
        done += (size_t)result;
    }
    return 0;
}
