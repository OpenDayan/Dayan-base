#ifndef NPU_HOST_HAL_XDMA_BACKEND_IO_H
#define NPU_HOST_HAL_XDMA_BACKEND_IO_H

#include <stddef.h>
#include <stdint.h>

int npu_hal_xdma_pread(int fd, uint64_t offset, void *data, size_t size);
int npu_hal_xdma_pwrite(int fd, uint64_t offset, const void *data, size_t size);

#endif
