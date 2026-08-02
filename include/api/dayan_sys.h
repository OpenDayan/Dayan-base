#ifndef DAYAN_SYS_H
#define DAYAN_SYS_H

#include <stdint.h>

#include "dayan_attrs.h"

#define DAYAN_COHERENT \
    __attribute__((section(".coherent"), aligned(64)))

enum dayan_resource_state {
    DAYAN_RESOURCE_FREE = 0,
    DAYAN_RESOURCE_BUSY = 1,
    DAYAN_RESOURCE_READY = 2,
};

/* Per-hart resource. Treat every field as private. */
typedef struct dayan_resource {
    uint32_t state;
    uint32_t producer_count;
    uint32_t consumer_count;
    volatile uint32_t lock;
} dayan_resource_t;

/* Cross-hart resource. The object itself must reside in coherent memory. */
typedef struct dayan_shared_resource {
    uint32_t state;
    uint32_t producer_count;
    uint32_t consumer_count;
    uint32_t lock;
} dayan_shared_resource_t;

#ifdef __cplusplus
extern "C" {
#endif

DAYAN_API void dayan_yield(void);
/* Called from a proc; each function waits for earlier work on that unit. */
DAYAN_API void dayan_wait_lsu(void);
DAYAN_API void dayan_wait_mat(void);
DAYAN_API void dayan_wait_vec(void);
DAYAN_API void dayan_wait_macro(void);
/* DMA is implemented by LSU and retains the original API spelling. */
DAYAN_API void dayan_wait_dma(void);

/* Per-hart resource functions use no atomic operations. */
/* Functions return 0 on success and a negative value on failure. */
DAYAN_API int dayan_resource_init(dayan_resource_t *resource);
/* consumer_count must be nonzero; producer_count may be zero. */
DAYAN_API int dayan_resource_alloc(dayan_resource_t *resource,
                                   uint32_t producer_count,
                                   uint32_t consumer_count);
DAYAN_API int dayan_resource_produce(dayan_resource_t *resource);
DAYAN_API int dayan_resource_consume(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_ready(dayan_resource_t *resource);
DAYAN_API int dayan_resource_wait_free(dayan_resource_t *resource);

/* Shared resources must be coherent and initialized before publication. */
DAYAN_API int dayan_shared_resource_init(dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_alloc(dayan_shared_resource_t *resource,
                                          uint32_t producer_count,
                                          uint32_t consumer_count);
DAYAN_API int dayan_shared_resource_produce(dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_consume(dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_ready(
    dayan_shared_resource_t *resource);
DAYAN_API int dayan_shared_resource_wait_free(
    dayan_shared_resource_t *resource);

#ifdef __cplusplus
}
#endif

#endif
