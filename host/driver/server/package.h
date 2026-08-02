#ifndef NPU_DRIVER_SERVER_PACKAGE_H
#define NPU_DRIVER_SERVER_PACKAGE_H

#include <stddef.h>
#include <stdint.h>

#include "npu_package.h"

struct npu_package_view {
    const unsigned char *image;
    size_t size;
    const npu_package_header_t *header;
    const npu_package_kernel_t *kernels;
    const npu_package_argument_t *arguments;
    const npu_package_sync_t *sync_resources;
    const unsigned char *object;
};

int npu_package_view_init(struct npu_package_view *view,
                          const void *image, size_t size);
const char *npu_package_string(const struct npu_package_view *view,
                               uint32_t offset);
const npu_package_kernel_t *npu_package_find_kernel(
    const struct npu_package_view *view, const char *name);
int npu_package_object_is_immutable(const struct npu_package_view *view);
uint64_t npu_package_hash(const void *image, size_t size);

#endif
