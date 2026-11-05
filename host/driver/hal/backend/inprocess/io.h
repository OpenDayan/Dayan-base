#ifndef NPU_HAL_INPROCESS_IO_H
#define NPU_HAL_INPROCESS_IO_H

#include "npu_hal_inprocess.h"

struct npu_inprocess_device;
int npu_inprocess_acquire(uint64_t id, uint64_t process,
                         struct npu_inprocess_device **device);
void npu_inprocess_release(struct npu_inprocess_device *device);
uint64_t npu_inprocess_id(const struct npu_inprocess_device *device);
uint32_t npu_inprocess_clusters(const struct npu_inprocess_device *device);
int npu_inprocess_read(struct npu_inprocess_device *device, uint32_t cluster,
                      uint64_t address, void *destination, size_t size);
int npu_inprocess_write(struct npu_inprocess_device *device, uint32_t cluster,
                       uint64_t address, const void *source, size_t size);

#endif
