#ifndef NPU_HOST_SERVICE_INTERNAL_H
#define NPU_HOST_SERVICE_INTERNAL_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "rpc.h"
#include "service.h"

/* Shared admission capacity for queued Host-to-Device RPC calls. */
#define NPU_SERVICE_RPC_QUEUE_CAPACITY 1024U

struct npu_api_symbol {
    char *name;
    uint64_t value;
};

struct npu_kernel_export {
    char *name;
    uint64_t value;
};

struct npu_kernel_object {
    const void *identity;
    struct npu_kernel_object *next;
};

struct npu_kernel_module {
    size_t object_size;
    uint32_t cluster;
    uint64_t base;
    uint64_t size;
    npu_hal_allocation_t *allocation;
    struct npu_kernel_export *exports;
    size_t export_count;
    struct npu_kernel_object *objects;
    struct npu_kernel_module *next;
};

struct npu_pending_call {
    uint16_t expected_opcode;
    /* pending tracks the mailbox response; active protects result pickup. */
    uint32_t pending;
    uint32_t active;
    uint64_t expected_argument;
    int status;
};

struct npu_rpc_queue_entry {
    struct npu_rpc_queue_entry *next;
    size_t core_index;
    uint32_t queued;
};

struct npu_service_launch_agent {
    npu_service_device_t *device;
    pthread_t thread;
    struct npu_service_launch *head;
    struct npu_service_launch *tail;
    uint32_t started;
};

struct npu_service_cluster {
    uint32_t firmware_loaded;
    uint32_t booted_cores;
    uint64_t firmware_entry;
    npu_hal_allocation_t *atomic_cache_backing;
    uint64_t atomic_cache_backing_address;
};

/*
 * Nothing below 2 GiB is mapped.  The MMU bypasses that whole range before it
 * ever consults a page table, so the control window, the BL0 ROM and URAM need
 * no PTEs and must not be given any: a mapping here would be dead weight that
 * disagrees with the hardware it is supposed to describe.  Only the firmware
 * window and Buffers are translated.
 *
 * This makes the firmware and QEMU version-locked.  A firmware that maps
 * nothing below 2 GiB against a QEMU that still translates the whole space
 * sends every low access into an empty table, redirected to a guard page.
 */
#define NPU_SERVICE_MMU_BYPASS_LIMIT UINT64_C(0x80000000)

/*
 * The firmware window.  Its virtual base is fixed by device/linker.ld (which
 * places every section at CONFIG_RAM_BASE), so the firmware needs no rebuild
 * and its internal pointers stay valid.  Its physical base is chosen by the
 * Host, and the two differ by a constant:
 *
 *     pa = va - NPU_SERVICE_MMU_FW_VA_BASE + fw_pool_base
 *
 * Decoupling the physical side is what removes the ordering deadlock that an
 * identity mapping would create: the window can be mapped in full at init
 * time, before BL1 is even loaded, because where BL1's segments land inside it
 * no longer matters.
 */
#define NPU_SERVICE_MMU_FW_VA_BASE  UINT64_C(0x80000000)

/* 2 MiB Sv39 leaf, used only for the firmware window: permissions there are
 * uniform and the pool base is 2 MiB-aligned.  Buffers stay on 4 KiB leaves so
 * each keeps its own permissions and its own out-of-bounds granularity. */
#define NPU_SERVICE_MMU_LARGE_SIZE  UINT64_C(0x00200000)

/* Unmapped gap between the firmware window and the Buffer VA window, so an
 * overrun off the end of the firmware window faults instead of sliding into
 * Buffer space. */
#define NPU_SERVICE_MMU_VA_SENTINEL NPU_SERVICE_MMU_LARGE_SIZE

/*
 * One Sv39 table.  @entries is a Host shadow of the device-resident table at
 * @device_address: the Host cannot dereference device memory, so entries are
 * edited here and pushed across.  @children mirrors the interior links so a
 * walk never has to read them back.
 */
struct npu_mmu_table {
    uint64_t device_address;
    npu_hal_allocation_t *allocation;
    uint64_t *entries;
    struct npu_mmu_table **children;
};

/*
 * An allocation whose PTEs are cleared but which must not be freed until a
 * TLB flush has happened.  The flush can only be issued when the device is
 * idle, so release is two-phase; see npu_service_mmu_defer_reclaim().
 */
struct npu_mmu_reclaim {
    npu_hal_allocation_t *allocation;
    uint32_t cluster;
    struct npu_mmu_reclaim *next;
};

/* Per-Cluster guard pages, one per redirect class. */
struct npu_mmu_cluster {
    npu_hal_allocation_t *guard_allocation[6];
    uint64_t guard_address[6];
};

struct npu_mmu {
    pthread_mutex_t lock;
    uint32_t enabled;
    uint32_t cluster_count;
    struct npu_mmu_table **root;
    struct npu_mmu_cluster *clusters;
    struct npu_mmu_reclaim *reclaim;
    uint64_t va_next;
    uint64_t va_limit;
    /*
     * The firmware pool: physical base and size of the window mapped at
     * NPU_SERVICE_MMU_FW_VA_BASE.  fw_pool_base is the whole linear-offset
     * relation, so every Host-to-Device address conversion goes through it.
     */
    uint64_t fw_pool_base;
    uint64_t fw_pool_size;
    /*
     * Downward cursor for page tables and guard pages, from the top of the
     * firmware pool.  The pool's base is the Host's choice, but the firmware's
     * offset inside it is not - BL1's segments start at fw_pool_base - so MMU
     * pages are taken from the far end.
     */
    uint64_t pool_cursor;
};

struct npu_service_heap_allocation {
    npu_hal_allocation_t *allocation;
    npu_hal_core_t owner;
    uint64_t address;
    uint64_t size;
    struct npu_service_heap_allocation *next;
};

struct npu_service_firmware_allocation {
    npu_hal_allocation_t *allocation;
    uint32_t scope;
    uint32_t cluster;
    uint64_t address;
    uint64_t size;
    struct npu_service_firmware_allocation *next;
};

struct npu_service_profile_core {
    npu_hal_core_t target;
    uint64_t address;
    uint64_t offset;
    uint32_t capacity;
    uint32_t configured;
};

struct npu_service_device {
    npu_hal_t *hal;
    npu_hal_info_t hal_info;
    npu_service_options_t options;
    pthread_mutex_t mutex;
    pthread_mutex_t kernel_mutex;
    pthread_cond_t condition;
    pthread_t worker;
    struct npu_pending_call *calls;
    struct npu_service_launch_agent *launch_agents;
    struct npu_service_launch *launches;
    size_t launch_agent_count;
    struct npu_rpc_queue_entry *rpc_queue_head;
    struct npu_rpc_queue_entry *rpc_queue_tail;
    size_t rpc_queue_count;
    size_t rpc_queue_capacity;
    size_t rpc_queue_waiters;
    uint32_t rpc_debug;
    struct npu_service_cluster *clusters;
    struct npu_service_heap_allocation *heap_allocations;
    struct npu_service_firmware_allocation *firmware_allocations;
    npu_hal_allocation_t *profile_allocation;
    struct npu_service_profile_core *profile_cores;
    size_t profile_core_count;
    uint64_t profile_generation;
    uint64_t profile_next_generation;
    uint32_t profile_active;
    uint32_t profile_transition;
    uint32_t profile_readers;
    uint32_t worker_started;
    uint32_t worker_stop;
    int abort_status;
    uint32_t active_launches;
    /*
     * Set while the idle-only MMU work (guard-page check, TLB flush) runs
     * outside the lock.  Launch submission waits on it: a launch queued
     * between the idle check and the flush would be running when TLB_FLUSH
     * lands, which hangs the Cluster.
     */
    uint32_t reclaim_in_flight;
    /*
     * Sticky.  A launch that reached the Device and was then abandoned has
     * unknown completion, so the Device can never again be shown idle and
     * no further flush is safe; reclaim waits for teardown.
     */
    uint32_t launch_abandoned;
    struct npu_api_symbol *api_symbols;
    size_t api_symbol_count;
    struct npu_kernel_module *kernel_modules;
    struct npu_mmu mmu;
};

struct npu_service_launch {
    npu_service_device_t *device;
    npu_hal_core_t target;
    npu_hal_allocation_t *request_allocation;
    struct npu_service_launch *queue_next;
    struct npu_service_launch *next;
    uint64_t request_address;
    uint64_t cookie;
    int64_t result;
    uint32_t task_id[3];
    uint32_t task_mode;
    uint32_t proc_id;
    uint32_t submit_started;
    uint32_t submit_done;
    uint32_t accepted;
    uint32_t exit_received;
    uint32_t completed;
    uint32_t descriptor_may_be_live;
    uint32_t abandoned;
    int submit_status;
    char name[NPU_SERVICE_PROC_NAME_SIZE];
};

uint64_t npu_now_ms(void);
void npu_delay_us(uint32_t delay_us);
size_t npu_service_core_index(const npu_service_device_t *device,
                              npu_hal_core_t core);
int npu_service_alloc_visible(npu_service_device_t *device,
                              npu_hal_core_t viewer,
                              npu_hal_memory_class_t memory_class,
                              npu_hal_memory_usage_t usage,
                              uint32_t owner, uint64_t size,
                              uint64_t alignment,
                              npu_hal_allocation_t **allocation_out);
int npu_service_alloc_rpc_descriptor(
    npu_service_device_t *device, npu_hal_core_t viewer,
    size_t descriptor_size, npu_hal_allocation_t **allocation_out);

/*
 * ClusterMMU lifecycle.  The mapping API itself is public in service.h,
 * because the Server layer needs it to give Buffers virtual addresses.
 */
int npu_service_mmu_init(npu_service_device_t *device);
void npu_service_mmu_fini(npu_service_device_t *device);
/*
 * Flush once and free everything parked by defer_reclaim().  Only call this
 * with no launch in flight - npu_hal_mmu_flush() hangs the Cluster otherwise.
 * @force skips the flush, for teardown after the MMU is switched off.
 */
int npu_service_mmu_run_reclaim(npu_service_device_t *device, int force);

int npu_service_load_bl1_elf(npu_service_device_t *device, uint32_t cluster,
                             const void *image, size_t size,
                             uint32_t core_mask);
void npu_service_dump_device(npu_service_device_t *device,
                             npu_hal_core_t core, const char *kind,
                             const void *record, size_t record_size);
int npu_service_rpc_call(npu_service_device_t *device, npu_hal_core_t core,
                         uint16_t opcode, uint64_t argument,
                         uint32_t timeout_ms,
                         int *argument_may_be_live_out);
int npu_service_launch_complete(npu_service_device_t *device,
                                npu_hal_core_t core, uint32_t proc_id,
                                int64_t result, uint64_t cookie);
void npu_service_launch_publish(npu_service_device_t *device,
                                npu_hal_core_t core, uint32_t proc_id,
                                int64_t result, uint64_t cookie);
void npu_rpc_debug_begin(npu_service_device_t *device, const char *direction,
                         npu_hal_core_t core, uint16_t opcode,
                         uint64_t argument);
void npu_rpc_debug_end(npu_service_device_t *device, const char *direction,
                       npu_hal_core_t core, uint16_t opcode,
                       uint64_t argument, int status);
void npu_service_kernel_state_destroy(npu_service_device_t *device);
void *npu_rpc_worker(void *arg);
int npu_rpc_dispatch(npu_service_device_t *device, npu_hal_core_t core,
                     uint64_t command,
                     struct rpc_proc_exit_request *proc_exit_out);
int npu_service_profile_copy(npu_service_device_t *device,
                             npu_hal_core_t target, uint64_t generation,
                             uint64_t address, uint32_t capacity,
                             uint32_t count, int *continue_out);
void npu_service_profile_discard(npu_service_device_t *device);

#endif
