#ifndef NPU_HOST_HAL_INTERNAL_H
#define NPU_HOST_HAL_INTERNAL_H

#include <pthread.h>

#include "hal.h"

struct npu_hal_backend_ops {
    uint32_t backend;
    /* May resolve topology from ROM before the HAL creates its pools. */
    int (*open)(npu_hal_options_t *options, void **state_out);
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
    /*
     * Optional: validates a whole Device-to-Device copy (both ranges, every
     * target type) before npu_hal_copy_address() moves its first chunk.
     * Without it each chunk is checked on its own, as before, and a range
     * error after the first chunk leaves the earlier chunks written.
     */
    int (*copy_address_check)(void *state, uint32_t cluster,
                              uint64_t destination, uint64_t source,
                              size_t size);
    /* Optional; absent means the backend cannot reset a Cluster. */
    int (*reset_cluster)(void *state, uint32_t cluster);
};

typedef int (*npu_hal_build_id_reader_t)(void *state, uint32_t cluster,
                                         char *identifier, size_t size);
int npu_hal_detect_config(npu_hal_options_t *options,
                          npu_hal_build_id_reader_t read_id, void *state);

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
    const struct npu_hal_backend_ops *ops;
    void *state;
    int device_status;
    uint32_t backend;
    uint32_t cluster_count;
    uint32_t cores_per_cluster;
    /* npu_hal_options_t::split_normal_memory; see normal_split_offset(). */
    uint32_t split_normal;
    pthread_mutex_t irq_mutex;
    pthread_mutex_t memory_mutex;
    struct npu_hal_arena *arenas;
    uint32_t arena_count;
};

extern const struct npu_hal_backend_ops npu_hal_xilinx_dma_backend_ops;
extern const struct npu_hal_backend_ops npu_hal_soc_mmap_backend_ops;
extern const struct npu_hal_backend_ops npu_hal_inprocess_backend_ops;
int npu_hal_memory_manager_init(npu_hal_t *hal,
                                const npu_hal_options_t *options);
void npu_hal_memory_manager_destroy(npu_hal_t *hal);

#endif
