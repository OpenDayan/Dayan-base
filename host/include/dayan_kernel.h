#ifndef DAYAN_HOST_KERNEL_H
#define DAYAN_HOST_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DAYAN_KERNEL_LAUNCH_ATTRS_VERSION 1U
#define DAYAN_KERNEL_LAUNCH_ASYNC         (1U << 0)
#define DAYAN_KERNEL_MAX_ARGS             8U

typedef struct npu_runtime npu_runtime_t;
typedef struct npu_launch npu_launch_t;
typedef struct dayan_kernel_descriptor dayan_kernel_descriptor_t;

/*
 * launch_count is the Dayan launch scale.  Invocation i is submitted to
 * (first_hart + i) modulo the runtime hart count.  A synchronous call waits
 * for and releases every invocation.  An asynchronous call returns ownership
 * of every npu_launch_t through async_launches.
 */
typedef struct dayan_kernel_launch_attrs {
    uint32_t version;
    uint32_t flags;
    npu_runtime_t *runtime;
    uint32_t first_hart;
    uint32_t launch_count;
    uint32_t timeout_ms;
    uint32_t reserved;
    uint64_t stack_size;
    uint64_t shared_size;
    uint64_t printf_buffer_size;
    uint64_t cookie;
    npu_launch_t **async_launches;
    size_t async_launch_capacity;
    size_t *async_launch_count;
} dayan_kernel_launch_attrs_t;

void dayan_kernel_launch_attrs_init(dayan_kernel_launch_attrs_t *attrs,
                                    npu_runtime_t *runtime);

/* Register the public-symbol ELF used to resolve Device object relocations. */
int dayan_runtime_set_api_elf(npu_runtime_t *runtime,
                              const void *elf_image, size_t elf_size);
int dayan_runtime_set_api_elf_file(npu_runtime_t *runtime, const char *path);

/*
 * These entry points are used by dycc-generated wrappers.  The descriptor
 * references (rather than copies) object, so that storage must outlive the
 * descriptor.  dycc satisfies this with its embedded const array.
 */
int dayan_kernel_descriptor_create(const void *object, size_t object_size,
                                   const char *entry_symbol,
                                   dayan_kernel_descriptor_t **descriptor_out);
void dayan_kernel_descriptor_destroy(dayan_kernel_descriptor_t *descriptor);
int dayan_kernel_launch(dayan_kernel_descriptor_t *descriptor,
                        const uint64_t *args, size_t argc,
                        const dayan_kernel_launch_attrs_t *attrs);

static inline uint64_t dayan_kernel_pack_f32(float value)
{
    uint32_t bits;

    __builtin_memcpy(&bits, &value, sizeof(bits));
    return (uint64_t)bits;
}

static inline uint64_t dayan_kernel_pack_f64(double value)
{
    uint64_t bits;

    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

#ifdef __cplusplus
}
#endif

#endif
