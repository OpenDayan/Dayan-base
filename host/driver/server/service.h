#ifndef NPU_HOST_SERVICE_DEVICE_H
#define NPU_HOST_SERVICE_DEVICE_H

#include <stddef.h>
#include <stdint.h>

#include "hal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_SERVICE_OPTIONS_VERSION 7U
#define NPU_SERVICE_LAUNCH_DESC_VERSION 4U
#define NPU_SERVICE_TIMEOUT_INFINITE UINT32_MAX
#define NPU_SERVICE_LAUNCH_MAX_ARGS 8U
#define NPU_SERVICE_PROC_NAME_SIZE 16U
#define NPU_SERVICE_OUTPUT_CLEAR (1U << 0)
#define NPU_SERVICE_PROC_MAX_RECORDS 19U
#define NPU_SERVICE_PROFILE_DEFAULT_BUFFER_SIZE (16U * 1024U * 1024U)

#define NPU_SERVICE_PROC_SYSTEM            (1U << 0)
#define NPU_SERVICE_PROC_HOST_ROOT         (1U << 1)
#define NPU_SERVICE_PROC_CHILD             (1U << 2)
#define NPU_SERVICE_PROC_ZOMBIE            (1U << 3)
#define NPU_SERVICE_PROC_KILL_WAIT_PRODUCE (1U << 4)

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
typedef void (*npu_service_proc_exit_fn)(void *context, uint32_t cluster,
                                         uint32_t core, uint32_t proc,
                                         int64_t result, uint64_t cookie);

typedef int (*npu_service_profile_fn)(
    void *context, uint32_t cluster, uint32_t core,
    const void *data, size_t size);

typedef struct npu_service_options {
    uint32_t version;
    uint32_t rpc_poll_interval_us;
    uint32_t default_timeout_ms;
    uint32_t reserved;
    npu_service_output_fn output;
    npu_service_trap_fn trap;
    npu_service_proc_exit_fn proc_exit;
    npu_service_profile_fn profile;
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
    const char *name;
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
    char name[NPU_SERVICE_PROC_NAME_SIZE];
} npu_service_launch_info_t;

typedef struct npu_service_proc_info {
    uint32_t proc_id;
    uint32_t parent_proc_id;
    uint32_t state;
    uint32_t flags;
    uint32_t live_children;
    uint32_t slot_index;
    uint32_t task_mode;
    uint32_t host_proc_id;
    uint32_t task_id[3];
    uint32_t task_dim[3];
    int64_t result;
    uint64_t pc;
    uint64_t cookie;
    char name[NPU_SERVICE_PROC_NAME_SIZE];
} npu_service_proc_info_t;

typedef struct npu_service_proc_snapshot {
    npu_hal_core_t target;
    uint32_t count;
    uint32_t global_free_slots;
    uint32_t reserved_slots;
    uint32_t reserved;
    npu_service_proc_info_t records[NPU_SERVICE_PROC_MAX_RECORDS];
} npu_service_proc_snapshot_t;

typedef struct npu_service_kill_result {
    npu_hal_core_t target;
    int64_t status;
    uint32_t killed_roots;
    uint32_t killed_children;
    uint32_t released_slots;
    uint32_t waiting_produce;
} npu_service_kill_result_t;

void npu_service_options_init(npu_service_options_t *options);
void npu_service_launch_desc_init(npu_service_launch_desc_t *desc);
int npu_service_open(npu_service_device_t **device_out,
                     npu_hal_t *hal,
                     const npu_service_options_t *options);
int npu_service_close(npu_service_device_t *device);
/* Cancels in-flight RPC waits and rejects future Device operations. */
void npu_service_abort(npu_service_device_t *device, int status);
int npu_service_status(npu_service_device_t *device);

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

/* Queues PROC_CREATE on the target Core; query/wait observes submission. */
int npu_service_launch(npu_service_device_t *device,
                       const npu_service_launch_desc_t *desc,
                       npu_service_launch_t **launch_out);
int npu_service_launch_query(npu_service_launch_t *launch,
                             npu_service_launch_info_t *info);
int npu_service_launch_wait(npu_service_launch_t *launch,
                            uint32_t timeout_ms,
                            npu_service_launch_info_t *info);
int npu_service_launch_release(npu_service_launch_t *launch);
/* Drops the Host handle; its Descriptor stays reserved until HAL close. */
int npu_service_launch_abandon(npu_service_launch_t *launch);

/* Administrative calls explicitly address one Cluster/Core (Hart). */
int npu_service_rpc_ps(npu_service_device_t *device, npu_hal_core_t target,
                       npu_service_proc_snapshot_t *snapshot);
int npu_service_rpc_kill_all(npu_service_device_t *device,
                             npu_hal_core_t target,
                             npu_service_kill_result_t *result);
int npu_service_rpc_kill_all_timeout(npu_service_device_t *device,
                                     npu_hal_core_t target,
                                     uint32_t timeout_ms,
                                     npu_service_kill_result_t *result);

/* One global allocation is partitioned into one FIFO per booted Hart. */
int npu_service_profile_start(npu_service_device_t *device,
                              uint64_t buffer_size);
int npu_service_profile_stop(npu_service_device_t *device);

const npu_service_options_t *npu_service_get_options(
    const npu_service_device_t *device);

#ifdef __cplusplus
}
#endif

#endif
