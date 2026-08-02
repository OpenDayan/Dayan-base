#ifndef NPU_HOST_RUNTIME_H
#define NPU_HOST_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "npu_driver.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_RUNTIME_OPTIONS_VERSION 1U
#define NPU_LAUNCH_DESC_VERSION     1U
#define NPU_TIMEOUT_INFINITE        UINT32_MAX
#define NPU_LAUNCH_MAX_ARGS         8U

#define NPU_OUTPUT_CLEAR            (1U << 0)

typedef enum npu_launch_state {
    NPU_LAUNCH_SUBMITTED = 0,
    NPU_LAUNCH_RUNNING = 1,
    NPU_LAUNCH_EXITED = 2,
    NPU_LAUNCH_FAILED = 3,
} npu_launch_state_t;

typedef struct npu_runtime npu_runtime_t;
typedef struct npu_launch npu_launch_t;
struct trap_record;

typedef void (*npu_output_fn)(void *context, uint32_t hart,
                              uint32_t proc, const void *data,
                              size_t size, uint32_t flags);
typedef void (*npu_trap_fn)(void *context,
                            const struct trap_record *record);

typedef struct npu_runtime_options {
    uint32_t version;
    uint32_t hart_count;
    uint64_t ram_base;
    uint64_t ram_size;
    uint64_t coherent_base;
    uint64_t coherent_size;
    uint32_t rpc_poll_interval_us;
    uint32_t default_timeout_ms;
    npu_driver_options_t driver;
    npu_output_fn output;
    npu_trap_fn trap;
    void *callback_context;
} npu_runtime_options_t;

typedef struct npu_launch_desc {
    uint32_t version;
    uint32_t hart;
    uint32_t flags;
    uint32_t argc;
    uint64_t entry;
    uint64_t stack_size;
    uint64_t shared_base;
    uint64_t shared_size;
    uint64_t printf_buffer_size;
    uint64_t args[NPU_LAUNCH_MAX_ARGS];
    uint64_t cookie;
} npu_launch_desc_t;

typedef struct npu_launch_info {
    uint32_t state;
    uint32_t hart;
    uint32_t proc_id;
    uint32_t reserved;
    int64_t status;
    uint64_t shared_base;
    uint64_t shared_size;
    uint64_t cookie;
} npu_launch_info_t;

void npu_runtime_options_init(npu_runtime_options_t *options);
void npu_launch_desc_init(npu_launch_desc_t *desc);

int npu_runtime_open(npu_runtime_t **runtime_out,
                     const npu_runtime_options_t *options);
int npu_runtime_close(npu_runtime_t *runtime);

int npu_runtime_load_bl1(npu_runtime_t *runtime,
                         const void *elf_image, size_t elf_size,
                         uint32_t hart_mask);
int npu_runtime_load_bl1_file(npu_runtime_t *runtime,
                              const char *path, uint32_t hart_mask);

int npu_runtime_alloc(npu_runtime_t *runtime, size_t size,
                      size_t alignment, uint64_t *address_out);
int npu_runtime_free(npu_runtime_t *runtime, uint64_t address);
int npu_runtime_coherent_alloc(npu_runtime_t *runtime, size_t size,
                               size_t alignment, uint64_t *address_out);
int npu_runtime_coherent_free(npu_runtime_t *runtime, uint64_t address);

int npu_runtime_copy_to_device(npu_runtime_t *runtime, uint64_t destination,
                               const void *source, size_t size);
int npu_runtime_copy_from_device(npu_runtime_t *runtime, void *destination,
                                 uint64_t source, size_t size);
int npu_runtime_copy_device(npu_runtime_t *runtime, uint64_t destination,
                            uint64_t source, size_t size);

int npu_runtime_rpc_call(npu_runtime_t *runtime, uint32_t hart,
                         uint16_t opcode, uint64_t argument,
                         uint64_t *response_out, uint32_t timeout_ms);

int npu_runtime_launch(npu_runtime_t *runtime,
                       const npu_launch_desc_t *desc,
                       npu_launch_t **launch_out);
int npu_launch_query(npu_launch_t *launch, npu_launch_info_t *info);
int npu_launch_wait(npu_launch_t *launch, uint32_t timeout_ms,
                    npu_launch_info_t *info);
int npu_launch_release(npu_launch_t *launch);

#ifdef __cplusplus
}
#endif

#endif
