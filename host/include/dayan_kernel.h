#ifndef NPU_HOST_COMPILED_KERNEL_H
#define NPU_HOST_COMPILED_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#include "npu_package.h"
#include "npu_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_KERNEL_LAUNCH_ATTRS_VERSION 2U
#define NPU_KERNEL_LAUNCH_ASYNC         (1U << 0)

typedef struct npu_kernel_descriptor npu_kernel_descriptor_t;

typedef struct npu_kernel_launch_attrs {
    uint32_t version;
    uint32_t flags;
    npu_context_t *context;
    npu_stream_t *stream;
    NPULaunchConfig config;
    uint32_t timeout_ms;
    uint32_t reserved;
    uint64_t cookie;
    npu_event_t **event_out;
} npu_kernel_launch_attrs_t;

void npu_kernel_launch_attrs_init(npu_kernel_launch_attrs_t *attrs,
                                  npu_context_t *context,
                                  npu_stream_t *stream);

/* Entry points used by dycc-generated Host wrappers. */
int npu_kernel_descriptor_create(
    const void *object, size_t object_size, const char *entry_symbol,
    const npu_package_argument_t *arguments, size_t argument_count,
    npu_kernel_descriptor_t **descriptor_out);
void npu_kernel_descriptor_destroy(npu_kernel_descriptor_t *descriptor);
int npu_kernel_descriptor_launch(npu_kernel_descriptor_t *descriptor,
                                 const npu_kernel_arg_t *args, size_t argc,
                                 const npu_kernel_launch_attrs_t *attrs);

static inline uint64_t npu_kernel_pack_f32(float value)
{
    uint32_t bits;

    __builtin_memcpy(&bits, &value, sizeof(bits));
    return (uint64_t)bits;
}

static inline uint64_t npu_kernel_pack_f64(double value)
{
    uint64_t bits;

    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

#ifdef __cplusplus
}
#endif

#endif
