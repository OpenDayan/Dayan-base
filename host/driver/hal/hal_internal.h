#ifndef NPU_HOST_HAL_INTERNAL_H
#define NPU_HOST_HAL_INTERNAL_H

#include <pthread.h>

#include "hal.h"

struct npu_hal_ops {
    void (*close)(void *state);
    int (*get_info)(void *state, npu_hal_info_t *info);
    int (*copy_to_address)(void *state, uint32_t cluster, uint64_t address,
                           const void *source, size_t size);
    int (*copy_from_address)(void *state, uint32_t cluster, void *destination,
                             uint64_t address, size_t size);
    int (*register_read)(void *state, uint32_t cluster, uint64_t offset,
                         void *value, size_t size);
    int (*register_write)(void *state, uint32_t cluster, uint64_t offset,
                          const void *value, size_t size);
    int (*irq_wait)(void *state, uint32_t timeout_ms,
                    uint32_t *cluster, uint32_t *status);
    int (*export_data_plane)(void *state, npu_hal_data_export_t *exported,
                             int handles[NPU_HAL_DATA_MAX_HANDLES]);
    int (*prepare_transfer)(void *state, uint32_t cluster, uint64_t address,
                            size_t size,
                            npu_hal_data_direction_t direction,
                            npu_hal_data_token_t *token);
    int (*finish_transfer)(void *state,
                           const npu_hal_data_token_t *token,
                           int transfer_status);
};

struct npu_hal_arena;

struct npu_hal_allocation {
    struct npu_hal *hal;
    uint64_t address;
    uint64_t size;
    uint32_t memory_class;
    uint32_t scope;
    uint32_t usage;
    uint32_t owner;
    npu_hal_core_t location;
    struct npu_hal_arena *arena;
    struct npu_hal_allocation *next;
};

struct npu_hal_arena {
    npu_hal_memory_region_t region;
    struct npu_hal_allocation *allocations;
};

struct npu_hal {
    const struct npu_hal_ops *ops;
    void *state;
    uint32_t backend;
    uint32_t cluster_count;
    uint32_t cores_per_cluster;
    pthread_mutex_t irq_mutex;
    pthread_mutex_t memory_mutex;
    struct npu_hal_arena *arenas;
    uint32_t arena_count;
};

int npu_hal_fuse_open(const npu_hal_options_t *options,
                      const struct npu_hal_ops **ops_out, void **state_out);
int npu_hal_memory_manager_init(npu_hal_t *hal,
                                const npu_hal_options_t *options);
void npu_hal_memory_manager_destroy(npu_hal_t *hal);

#endif
