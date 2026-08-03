#ifndef NPU_HOST_COMPILED_KERNEL_H
#define NPU_HOST_COMPILED_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#include "npu_package.h"
#include "npu_runtime.h"

#ifndef DAYAN_KERNEL_RESOURCE_DESC_VERSION
#define DAYAN_KERNEL_RESOURCE_DESC_VERSION 1U
#endif
#ifndef DAYAN_HART_MAX_PROCS
#define DAYAN_HART_MAX_PROCS 16U
#endif
#ifndef DAYAN_KERNEL_MAX_CHILD_PROCS
#define DAYAN_KERNEL_MAX_CHILD_PROCS (DAYAN_HART_MAX_PROCS - 1U)
#endif
#ifndef DAYAN_URAM_ALLOCATION_GRANULE
#define DAYAN_URAM_ALLOCATION_GRANULE 1024U
#endif

#ifndef DAYAN_KERNEL_RESOURCE_DESC_DEFINED
#define DAYAN_KERNEL_RESOURCE_DESC_DEFINED
typedef struct dayan_kernel_resource_desc {
    uint32_t version;
    uint32_t size;
    uint32_t max_child_procs;
    uint32_t flags;
    uint64_t atomic_cache_size;
    uint64_t uram_size;
} dayan_kernel_resource_desc_t;
#endif

/*
 * Each __kernel__ must have exactly one descriptor with this macro.  dycc
 * keeps the exact <kernel>__desc symbol in the Device object and folds the
 * same constant values into the Host package metadata.
 */
#if defined(__DAYAN_DEVICE__)
#ifdef __cplusplus
#define DAYAN_KERNEL_DESC_LINKAGE extern "C"
#else
#define DAYAN_KERNEL_DESC_LINKAGE extern
#endif
#define DAYAN_KERNEL_DESC(kernel, child_procs, atomic_bytes, uram_bytes)       \
    DAYAN_KERNEL_DESC_LINKAGE __device__                                      \
    __attribute__((visibility("default"), section(".dayan.kernel_desc")))     \
    const dayan_kernel_resource_desc_t kernel##__desc = {                     \
        DAYAN_KERNEL_RESOURCE_DESC_VERSION,                                   \
        sizeof(dayan_kernel_resource_desc_t),                                 \
        (child_procs), 0U, (atomic_bytes), (uram_bytes)                        \
    }
#else
#define DAYAN_KERNEL_DESC(kernel, child_procs, atomic_bytes, uram_bytes)
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_KERNEL_LAUNCH_ATTRS_VERSION 3U
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
    char task_name[NPU_PROC_NAME_SIZE];
    npu_event_t **event_out;
} npu_kernel_launch_attrs_t;

void npu_kernel_launch_attrs_init(npu_kernel_launch_attrs_t *attrs,
                                  npu_context_t *context,
                                  npu_stream_t *stream,
                                  const char *task_name);

/* Entry points used by dycc-generated Host wrappers. */
int npu_kernel_descriptor_create(
    const void *object, size_t object_size, const char *entry_symbol,
    const npu_package_argument_t *arguments, size_t argument_count,
    const dayan_kernel_resource_desc_t *resources,
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
static_assert(sizeof(dayan_kernel_resource_desc_t) == 32,
              "Dayan kernel resource descriptor ABI");
static_assert(sizeof(npu_kernel_launch_attrs_t) == 80,
              "Dayan kernel launch attributes ABI");
static_assert(offsetof(npu_kernel_launch_attrs_t, task_name) == 56,
              "Dayan kernel launch task name offset");
static_assert(offsetof(npu_kernel_launch_attrs_t, event_out) == 72,
              "Dayan kernel launch event offset");
}
#else
_Static_assert(sizeof(dayan_kernel_resource_desc_t) == 32,
               "Dayan kernel resource descriptor ABI");
_Static_assert(sizeof(npu_kernel_launch_attrs_t) == 80,
               "Dayan kernel launch attributes ABI");
_Static_assert(offsetof(npu_kernel_launch_attrs_t, task_name) == 56,
               "Dayan kernel launch task name offset");
_Static_assert(offsetof(npu_kernel_launch_attrs_t, event_out) == 72,
               "Dayan kernel launch event offset");
#endif

#endif
