#ifndef NPU_HOST_SERVICE_DEVICE_H
#define NPU_HOST_SERVICE_DEVICE_H

#include <stddef.h>
#include <stdint.h>

#include "hal/hal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_SERVICE_OPTIONS_VERSION 3U
#define NPU_SERVICE_LAUNCH_DESC_VERSION 3U
#define NPU_SERVICE_TIMEOUT_INFINITE UINT32_MAX
#define NPU_SERVICE_LAUNCH_MAX_ARGS 8U
#define NPU_SERVICE_OUTPUT_CLEAR (1U << 0)

typedef struct npu_service_device npu_service_device_t;
typedef struct npu_service_launch npu_service_launch_t;
struct trap_record;

typedef enum npu_service_launch_state {
    NPU_SERVICE_LAUNCH_SUBMITTED = 0,
    NPU_SERVICE_LAUNCH_RUNNING = 1,
    NPU_SERVICE_LAUNCH_EXITED = 2,
    NPU_SERVICE_LAUNCH_FAILED = 3,
} npu_service_launch_state_t;

typedef void (*npu_service_output_fn)(void *context, uint32_t cluster,
                                      uint32_t core, uint32_t proc,
                                      const void *data, size_t size,
                                      uint32_t flags);
typedef void (*npu_service_trap_fn)(void *context, uint32_t cluster,
                                    uint32_t core,
                                    const struct trap_record *record);

typedef struct npu_service_options {
    uint32_t version;
    uint32_t rpc_poll_interval_us;
    uint32_t default_timeout_ms;
    uint32_t reserved;
    npu_service_output_fn output;
    npu_service_trap_fn trap;
    void *callback_context;
} npu_service_options_t;

typedef struct npu_service_launch_desc {
    uint32_t version;
    uint32_t flags;
    npu_hal_core_t target;
    uint32_t argc;
    uint32_t task_id[3];
    uint32_t task_dim[3];
    uint32_t task_mode;
    uint64_t entry;
    uint64_t stack_size;
    npu_hal_allocation_t *shared_allocation;
    uint64_t shared_offset;
    uint64_t shared_size;
    npu_hal_allocation_t *uram_allocation;
    uint64_t uram_offset;
    uint64_t uram_size;
    uint32_t max_child_procs;
    uint32_t resource_reserved;
    uint64_t printf_buffer_size;
    uint64_t args[NPU_SERVICE_LAUNCH_MAX_ARGS];
    uint64_t cookie;
} npu_service_launch_desc_t;

typedef struct npu_service_launch_info {
    uint32_t state;
    npu_hal_core_t target;
    uint32_t proc_id;
    int64_t status;
    uint32_t task_id[3];
    uint32_t task_mode;
    uint64_t cookie;
} npu_service_launch_info_t;

void npu_service_options_init(npu_service_options_t *options);
void npu_service_launch_desc_init(npu_service_launch_desc_t *desc);
int npu_service_open(npu_service_device_t **device_out,
                     npu_hal_t *hal,
                     const npu_service_options_t *options);
int npu_service_close(npu_service_device_t *device);

/* BL1 is a per-Cluster instance; core_mask addresses cores within Cluster. */
int npu_service_load_bl1(npu_service_device_t *device, uint32_t cluster,
                         const void *elf_image, size_t elf_size,
                         uint32_t core_mask);
int npu_service_load_bl1_file(npu_service_device_t *device,
                              uint32_t cluster, const char *path,
                              uint32_t core_mask);

int npu_service_set_api_elf(npu_service_device_t *device,
                            const void *elf_image, size_t elf_size);
int npu_service_set_api_elf_file(npu_service_device_t *device,
                                 const char *path);
int npu_service_kernel_resolve(npu_service_device_t *device,
                               npu_hal_core_t target,
                               const void *object, size_t object_size,
                               const char *entry_symbol,
                               uint64_t *entry_out);
int npu_service_kernel_forget(npu_service_device_t *device,
                              const void *object, size_t object_size);

int npu_service_launch(npu_service_device_t *device,
                       const npu_service_launch_desc_t *desc,
                       npu_service_launch_t **launch_out);
int npu_service_launch_query(npu_service_launch_t *launch,
                             npu_service_launch_info_t *info);
int npu_service_launch_wait(npu_service_launch_t *launch,
                            uint32_t timeout_ms,
                            npu_service_launch_info_t *info);
int npu_service_launch_release(npu_service_launch_t *launch);

const npu_service_options_t *npu_service_get_options(
    const npu_service_device_t *device);

#ifdef __cplusplus
}
#endif

#endif
