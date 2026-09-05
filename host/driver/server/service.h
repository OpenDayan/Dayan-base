#ifndef NPU_HOST_SERVICE_DEVICE_H
#define NPU_HOST_SERVICE_DEVICE_H

#include <stddef.h>
#include <stdint.h>

#include "hal.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_SERVICE_OPTIONS_VERSION 8U
#define NPU_SERVICE_LAUNCH_DESC_VERSION 5U
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
struct tensor_interrupt_record;

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
typedef void (*npu_service_tensor_interrupt_fn)(
    void *context, uint32_t cluster, uint32_t core,
    const struct tensor_interrupt_record *record);

typedef int (*npu_service_profile_fn)(
    void *context, uint32_t cluster, uint32_t core,
    const void *data, size_t size);

typedef struct npu_service_options {
    uint32_t version;
    uint32_t rpc_poll_interval_us;
    uint32_t default_timeout_ms;
    /*
     * Nonzero routes Device addresses through the ClusterMMU: the Service
     * builds Sv39 page tables, identity-maps the firmware and control
     * apertures, and hands Buffers virtual addresses.  Zero leaves the MMU in
     * its reset state, whose FixMap is a full-range identity map, so every
     * address behaves exactly as it did before this option existed.
     *
     * Occupies what was a reserved word, so the struct size does not change.
     */
    uint32_t enable_mmu;
    npu_service_output_fn output;
    npu_service_trap_fn trap;
    npu_service_tensor_interrupt_fn tensor_interrupt;
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
    uint32_t max_child_threads;
    uint32_t resource_context;
    uint64_t tensor_context;
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

/* Queues PROC_CREATE; completion is published by the Device's PROC_EXIT. */
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
int npu_service_rpc_kill_family(npu_service_device_t *device,
                                npu_hal_core_t target,
                                uint32_t root_proc_id,
                                int64_t failure_status,
                                npu_service_kill_result_t *result);
int npu_service_rpc_kill_family_timeout(npu_service_device_t *device,
                                        npu_hal_core_t target,
                                        uint32_t root_proc_id,
                                        int64_t failure_status,
                                        uint32_t timeout_ms,
                                        npu_service_kill_result_t *result);

/* One global allocation is partitioned into one FIFO per booted Hart. */
int npu_service_profile_start(npu_service_device_t *device,
                              uint64_t buffer_size);
int npu_service_profile_stop(npu_service_device_t *device);

const npu_service_options_t *npu_service_get_options(
    const npu_service_device_t *device);

/*
 * Device address space (ClusterMMU).
 *
 * All of these are no-ops returning success when the MMU is disabled, so
 * callers need not branch on it.  The exception is reserve_va(), which
 * returns -ENOTSUP because there is no virtual address to hand out and the
 * caller must keep using the physical one.
 */
#define NPU_SERVICE_MMU_READ        (1U << 0)
#define NPU_SERVICE_MMU_WRITE       (1U << 1)
#define NPU_SERVICE_MMU_EXEC        (1U << 2)

/* Buffer mapping granularity.  The firmware window uses 2 MiB leaves, but
 * that is internal to the page-table manager; everything mapped through this
 * API is 4 KiB. */
#define NPU_SERVICE_MMU_PAGE_SIZE   UINT64_C(0x1000)
#define NPU_SERVICE_MMU_PAGE_MASK   UINT64_C(0xfff)

/*
 * Physical address behind a firmware virtual address.  The firmware's
 * addresses are fixed by its linker script; the memory behind them is wherever
 * the Host placed the firmware pool, so any Host access by address has to be
 * converted here first.
 *
 * Returns -ENOTSUP with *pa_out = va when the MMU is off (FixMap makes the two
 * identical), and -ERANGE when the range does not fit inside the window.
 */
int npu_service_mmu_fw_pa(npu_service_device_t *device, uint64_t va,
                          uint64_t size, uint64_t *pa_out);
/*
 * The address a Device Core must use to reach @allocation - the inverse of
 * npu_service_mmu_fw_pa().  Use this, not npu_hal_memory_address(), wherever
 * an address is about to be handed to the Device; the HAL reports physical
 * addresses because XDMA needs them.
 *
 * Returns -EFAULT for a Normal allocation outside the firmware window: that is
 * a Buffer, whose virtual address comes from npu_service_mmu_reserve_va() and
 * cannot be derived arithmetically.
 */
int npu_service_mmu_device_address(npu_service_device_t *device,
                                   const npu_hal_allocation_t *allocation,
                                   npu_hal_core_t viewer,
                                   uint64_t *address_out);
int npu_service_mmu_reserve_va(npu_service_device_t *device, uint64_t size,
                               uint64_t *base_out);
void npu_service_mmu_release_va(npu_service_device_t *device, uint64_t base,
                                uint64_t size);
int npu_service_mmu_map(npu_service_device_t *device, uint32_t cluster,
                        uint64_t va, uint64_t pa, uint64_t size,
                        uint32_t prot);
/*
 * Clears PTEs only; it does not flush.  The backing memory must be released
 * through npu_service_mmu_defer_reclaim(), which is where the flush happens,
 * because a flush issued while the Device has AXI traffic in flight hangs the
 * Cluster and there is no quiesce register to prevent it.
 */
int npu_service_mmu_unmap(npu_service_device_t *device, uint32_t cluster,
                          uint64_t va, uint64_t size);
/* Narrowing permissions requires a flush; the caller owns it. */
int npu_service_mmu_protect(npu_service_device_t *device, uint32_t cluster,
                            uint64_t va, uint64_t size, uint32_t prot);
/* Park an allocation whose PTEs are cleared; it is freed once a flush has
 * happened.  Frees immediately when the MMU is disabled. */
int npu_service_mmu_defer_reclaim(npu_service_device_t *device,
                                  uint32_t cluster,
                                  npu_hal_allocation_t *allocation);
/* Returns 1 and fills the outputs when a guard page shows a fault, else 0.
 * Sampling only: it cannot report the faulting VA's upper bits or a count,
 * because the hardware has no fault-status register. */
int npu_service_mmu_check_faults(npu_service_device_t *device,
                                 uint32_t cluster, uint32_t *fault_class,
                                 uint32_t *fault_offset);

#ifdef __cplusplus
}
#endif

#endif
