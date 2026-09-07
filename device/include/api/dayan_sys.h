#ifndef DAYAN_SYS_H
#define DAYAN_SYS_H

#include <stddef.h>
#include <stdint.h>

#include "dayan_attrs.h"
/* Host-owned coherent allocation bound to gp; treat this pointer as read-only. */
register void *shared __asm__("gp");

#define DAYAN_HART_MAX_PROCS                16U
#define DAYAN_KERNEL_MAX_CHILD_THREADS      (DAYAN_HART_MAX_PROCS - 1U)
#define DAYAN_THREAD_NAME_SIZE              16U
#define DAYAN_URAM_ALLOCATION_GRANULE       (32U * 1024U)
#define DAYAN_TCSR_LOGICAL_GROUPS            8U
#define DAYAN_RESOURCE_LOGICAL_COUNT         32U
#ifndef DAYAN_RESOURCE_COUNT_PACK
#define DAYAN_RESOURCE_COUNT_PACK(local, shared) \
    ((uint32_t)(local) | ((uint32_t)(shared) << 8))
#define DAYAN_RESOURCE_LOCAL_COUNT(value)  ((uint8_t)(value))
#define DAYAN_RESOURCE_SHARED_COUNT(value) ((uint8_t)((value) >> 8))
#endif
#ifndef DAYAN_KERNEL_RESOURCE_LOCKED_TASK
#define DAYAN_KERNEL_RESOURCE_LOCKED_TASK   (1U << 0)
#endif
#ifndef DAYAN_KERNEL_RESOURCE_FLAGS_ALL
#define DAYAN_KERNEL_RESOURCE_FLAGS_ALL DAYAN_KERNEL_RESOURCE_LOCKED_TASK
#endif

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
    uint32_t resource_counts;
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
    uint8_t local_resource_count;
    uint8_t shared_resource_count;
    uint16_t reserved;
} dayan_thread_resources_t;

/*
 * Zero-initialized storage in the allocation rooted at shared.  arrivals is
 * a monotonic ticket counter; fields are private to dayan_cluster_barrier().
 */
typedef struct dayan_cluster_barrier {
    uint64_t arrivals;
    uint64_t reserved;
} dayan_cluster_barrier_t;

typedef uint8_t dayan_resource_t;
typedef uint64_t dayan_resource_mask_t;

#define DAYAN_LOCAL_RESOURCE(index)  ((dayan_resource_t)(index))
#define DAYAN_SHARED_RESOURCE(index) ((dayan_resource_t)(32U + (index)))
#define DAYAN_RESOURCE_MASK(resource) \
    ((dayan_resource_mask_t)1ULL << (dayan_resource_t)(resource))

enum dayan_resource_role {
    DAYAN_RESOURCE_ROLE_NORMAL = 0,
    DAYAN_RESOURCE_ROLE_PRODUCER = 1,
    DAYAN_RESOURCE_ROLE_CONSUMER = 2,
};

enum dayan_task_axis {
    DAYAN_TASK_X = 0,
    DAYAN_TASK_Y = 1,
    DAYAN_TASK_Z = 2,
};

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
/* Waits in four-Core mode; participants may be any nonzero count up to four.
 * The barrier polls briefly, then yields at the scheduler's lowest priority. */
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

/*
 * Logical resources 0..31 are Local and 32..63 are Shared.  Wait returns a
 * dayan_resource_role for election waits, or a negative error.  A normal
 * wait has disjoint masks.  An election wait overlaps them in one bit.
 * Every READY input must match its explicit lifecycle phase.  Bit r in phase is
 * the expected phase of logical Resource r; bits outside wait_ready are ignored.
 * For an election, this phase selects both the FREE producer and READY consumer
 * generation.  There is no implicit snapshot of the Resource's current phase.
 * The final Consume through Resource Produce returns it to FREE and flips phase.
 * Runtime initializes allocated resources to FREE, count zero, phase zero
 * before root Procs run; child Threads inherit without reinitializing them.
 * Track each physical buffer's actual uses, not the global loop or buffer index.
 * A pending participant must never be overtaken by two complete lifecycles.
 */
DAYAN_API int dayan_resource_wait(dayan_resource_mask_t wait_free,
                                  dayan_resource_mask_t wait_ready,
                                  dayan_resource_mask_t phase, int32_t qos);
DAYAN_API int dayan_resource_wait_free(dayan_resource_t resource);
/* phase must be 0 or 1. */
DAYAN_API int dayan_resource_wait_ready(dayan_resource_t resource, uint32_t phase);

/*
 * Publish output after earlier work on the selected unit completes.  At the
 * same completion point, every Resource in consume_ready loses one consumer.
 * The final consumption returns it to FREE and starts its next lifecycle.
 * WAIT_READY does not claim a consumer credit; software must match the count
 * and keep remote-copy sources alive until the copy has completed.
 */
DAYAN_API int dayan_resource_produce_lsu(
    dayan_resource_t output, uint32_t consumer_count,
    dayan_resource_mask_t consume_ready);
DAYAN_API int dayan_resource_produce_mat(
    dayan_resource_t output, uint32_t consumer_count,
    dayan_resource_mask_t consume_ready);
DAYAN_API int dayan_resource_produce_vec(
    dayan_resource_t output, uint32_t consumer_count,
    dayan_resource_mask_t consume_ready);
DAYAN_API int dayan_resource_produce_macro(
    dayan_resource_t output, uint32_t consumer_count,
    dayan_resource_mask_t consume_ready);

/*
 * Consume terminal inputs after earlier work on the selected unit completes,
 * without publishing another Resource.  This closes a finite dependency DAG
 * and returns its final READY Resources to FREE.
 */
DAYAN_API int dayan_resource_consume_lsu(
    dayan_resource_mask_t consume_ready);
DAYAN_API int dayan_resource_consume_mat(
    dayan_resource_mask_t consume_ready);
DAYAN_API int dayan_resource_consume_vec(
    dayan_resource_mask_t consume_ready);
DAYAN_API int dayan_resource_consume_macro(
    dayan_resource_mask_t consume_ready);

#ifdef __cplusplus
}
#endif

#endif
