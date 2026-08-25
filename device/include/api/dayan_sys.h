#ifndef DAYAN_SYS_H
#define DAYAN_SYS_H

#include <stddef.h>
#include <stdint.h>

#include "dayan_attrs.h"
#include "dayan_list.h"

/* Host-owned coherent allocation bound to gp; treat this pointer as read-only. */
register void *shared __asm__("gp");

#define DAYAN_HART_MAX_PROCS                16U
#define DAYAN_KERNEL_MAX_CHILD_THREADS      (DAYAN_HART_MAX_PROCS - 1U)
#define DAYAN_THREAD_NAME_SIZE              16U
#define DAYAN_URAM_ALLOCATION_GRANULE       (32U * 1024U)
#define DAYAN_TCSR_LOGICAL_GROUPS            8U
#define DAYAN_KERNEL_RESOURCE_LOCKED_TASK   (1U << 0)
#define DAYAN_KERNEL_RESOURCE_FLAGS_ALL DAYAN_KERNEL_RESOURCE_LOCKED_TASK

#ifndef DAYAN_KERNEL_RESOURCE_DESC_DEFINED
#define DAYAN_KERNEL_RESOURCE_DESC_DEFINED
/* Constant Device symbol named <kernel>__desc; consumed by dycc/Runtime. */
typedef struct dayan_kernel_resource_desc {
    uint16_t max_child_threads;
    uint16_t max_tasks_per_core;
    uint32_t flags;
    uint8_t lsu_tcsr_count;
    uint8_t mat_tcsr_count;
    uint8_t vec_tcsr_count;
    uint8_t macro_tcsr_count;
    uint32_t reserved;
    uint64_t atomic_cache_size;
    uint64_t uram_size;
} dayan_kernel_resource_desc_t;
#endif

typedef void (*dayan_thread_entry_t)(void *argument);

typedef struct dayan_thread_resources {
    uint8_t lsu_tcsr_count;
    uint8_t mat_tcsr_count;
    uint8_t vec_tcsr_count;
    uint8_t macro_tcsr_count;
    uint32_t reserved;
} dayan_thread_resources_t;

/*
 * Zero-initialized storage in the allocation rooted at shared.  arrivals is
 * a monotonic ticket counter; fields are private to dayan_cluster_barrier().
 */
typedef struct dayan_cluster_barrier {
    uint64_t arrivals;
    uint64_t reserved;
} dayan_cluster_barrier_t;

enum dayan_resource_state {
    DAYAN_RESOURCE_FREE = 0,
    DAYAN_RESOURCE_BUSY = 1,
    DAYAN_RESOURCE_READY = 2,
};

enum dayan_task_axis {
    DAYAN_TASK_X = 0,
    DAYAN_TASK_Y = 1,
    DAYAN_TASK_Z = 2,
};

/* Per-hart resource. Treat every field as private. */
typedef struct dayan_resource {
    uint32_t state;
    uint32_t producer_count;
    uint32_t consumer_count;
    volatile uint32_t lock;
    struct list_head wait_lock;
    struct list_head wait_ready;
    struct list_head wait_free;
} dayan_resource_t;

/*
 * Cross-hart resource. Place it within the allocation rooted at shared.
 * Fields are private. counts packs producer_remaining in bits 63:32 and
 * consumer_remaining in bits 31:0.
 */
typedef struct dayan_shared_resource {
    uint64_t counts;
    uint64_t reserved;
} dayan_shared_resource_t;

#ifdef __cplusplus
extern "C" {
#endif

DAYAN_API void dayan_yield(void);
DAYAN_API void dayan_thread_resources_init(dayan_thread_resources_t *resources);
/* Name must be nonempty, NUL-terminated, and fit in DAYAN_THREAD_NAME_SIZE. */
/* Creates a child Thread on the same Hart and returns its positive ID. */
DAYAN_API int dayan_create_thread(const char *name, dayan_thread_entry_t entry,
                                  void *argument,
                                  const dayan_thread_resources_t *resources);
/* Only direct child Threads can be joined. result may be null. */
DAYAN_API int dayan_thread_join(uint32_t thread_id, int64_t *result);
DAYAN_API int dayan_thread_join_all(void);
DAYAN_API DAYAN_NORETURN void dayan_thread_exit(int64_t result);
/* The returned range belongs to the current resident kernel instance. */
DAYAN_API void *dayan_uram_base(void);
DAYAN_API size_t dayan_uram_size(void);
/* Current Cluster ABI has exactly four participating Cores. */
DAYAN_API int dayan_cluster_barrier(uint32_t participants,
                                    dayan_cluster_barrier_t *barrier);
/* Logical launch coordinates. Physical Cluster/Core placement stays hidden. */
DAYAN_API uint32_t dayan_task_id(uint32_t axis);
DAYAN_API uint32_t dayan_task_dim(uint32_t axis);
DAYAN_API uint32_t dayan_task_mode(void);
/* Called from a Proc or child Thread; waits for earlier work on that unit. */
DAYAN_API void dayan_wait_lsu(void);
DAYAN_API void dayan_wait_mat(void);
DAYAN_API void dayan_wait_vec(void);
DAYAN_API void dayan_wait_macro(void);
/* DMA is implemented by LSU and retains the original API spelling. */
DAYAN_API void dayan_wait_dma(void);
/* Operate on every cache line intersecting [address, address + size). */
/* These functions do not wait for an execution unit or synchronize harts. */
DAYAN_API void dayan_cache_invalidate(const void *address, size_t size);
DAYAN_API void dayan_cache_flush(const void *address, size_t size);

/* Per-hart resource functions use no atomic operations. */
/* Functions return 0 on success and a negative value on failure. */
DAYAN_API int dayan_resource_init(dayan_resource_t *resource);
/* consumer_count must be nonzero; producer_count may be zero. */
DAYAN_API int dayan_resource_alloc(dayan_resource_t *resource,
                                   uint32_t producer_count,
                                   uint32_t consumer_count);
DAYAN_API int dayan_resource_produce(dayan_resource_t *resource);
DAYAN_API int dayan_resource_consume(dayan_resource_t *resource);
/* Nonblocking: apply the resource transition when prior unit work completes. */
DAYAN_API int dayan_resource_wait_lsu_produce(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_lsu_consume(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_mat_produce(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_mat_consume(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_vec_produce(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_vec_consume(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_macro_produce(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_macro_consume(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_ready(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_free(dayan_resource_t *resource);

/* Shared resources must come from shared and be initialized before publication. */
DAYAN_API int dayan_shared_resource_init(dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_alloc(dayan_shared_resource_t *resource,
                                          uint32_t producer_count,
                                          uint32_t consumer_count);
DAYAN_API int dayan_shared_resource_produce(dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_consume(dayan_shared_resource_t *resource);
/* The shared object must remain live until the deferred transition runs. */
DAYAN_API int dayan_shared_resource_wait_lsu_produce(
    dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_lsu_consume(
    dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_mat_produce(
    dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_mat_consume(
    dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_vec_produce(
    dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_vec_consume(
    dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_macro_produce(
    dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_macro_consume(
    dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_ready(
    dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_free(
    dayan_shared_resource_t *resource);

#ifdef __cplusplus
}
#endif

#endif
