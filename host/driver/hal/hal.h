#ifndef NPU_HOST_HAL_H
#define NPU_HOST_HAL_H

#include <stddef.h>
#include <stdint.h>

#include "dataplane.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_HAL_ABI_VERSION          5U
#define NPU_HAL_OPTIONS_VERSION      3U
#define NPU_HAL_MEMORY_REQUEST_VERSION 1U
#define NPU_HAL_MAX_CLUSTERS         32U
#define NPU_HAL_MAX_CORES_PER_CLUSTER 32U
#define NPU_HAL_TIMEOUT_INFINITE     UINT32_MAX

#define NPU_HAL_BACKEND_FUSE_XDMA    1U
#define NPU_HAL_ANY_CLUSTER          UINT32_MAX
#define NPU_HAL_ANY_CORE             UINT32_MAX

typedef struct npu_hal npu_hal_t;
typedef struct npu_hal_allocation npu_hal_allocation_t;

typedef struct npu_hal_core {
    uint32_t cluster;
    uint32_t core;
} npu_hal_core_t;

typedef enum npu_hal_memory_class {
    NPU_HAL_MEMORY_NORMAL = 1,
    NPU_HAL_MEMORY_COHERENT = 2,
    NPU_HAL_MEMORY_URAM = 3,
} npu_hal_memory_class_t;

typedef enum npu_hal_memory_scope {
    NPU_HAL_MEMORY_DEVICE = 1,
    NPU_HAL_MEMORY_CLUSTER = 2,
    NPU_HAL_MEMORY_CORE = 3,
} npu_hal_memory_scope_t;

typedef enum npu_hal_memory_usage {
    NPU_HAL_MEMORY_BUFFER = 1,
    NPU_HAL_MEMORY_FIRMWARE,
    NPU_HAL_MEMORY_KERNEL,
    NPU_HAL_MEMORY_HEAP,
    NPU_HAL_MEMORY_LAUNCH,
    NPU_HAL_MEMORY_SHARED,
    NPU_HAL_MEMORY_KERNEL_URAM,
    NPU_HAL_MEMORY_PROFILE,
    NPU_HAL_MEMORY_ATOMIC_CACHE_BACKING,
} npu_hal_memory_usage_t;

/* A backend may expose multiple regions with different visibility scopes. */
typedef struct npu_hal_memory_region {
    uint64_t base;
    uint64_t size;
    uint32_t memory_class;
    uint32_t scope;
    uint32_t cluster;
    uint32_t core;
} npu_hal_memory_region_t;

/* Configuration is consumed only by dayan_server, never by applications. */
typedef struct npu_hal_options {
    uint32_t version;
    uint32_t backend;
    uint32_t device_index;
    uint32_t cluster_count;
    uint32_t cores_per_cluster;
    uint32_t cluster_device_stride;
    const char *device_prefix;
    uint64_t memory_address_bias;
    const npu_hal_memory_region_t *memory_regions;
    uint32_t memory_region_count;
    uint32_t normal_memory_size_mib;
} npu_hal_options_t;

typedef struct npu_hal_info {
    uint32_t abi_version;
    uint32_t backend;
    uint32_t data_provider;
    uint32_t flags;
    uint32_t cluster_count;
    uint32_t cores_per_cluster;
    uint32_t data_handle_count;
    uint32_t memory_region_count;
    uint64_t max_transfer_size;
    uint64_t transfer_alignment;
} npu_hal_info_t;

typedef struct npu_hal_memory_request {
    uint32_t version;
    uint32_t memory_class;
    uint32_t scope;
    uint32_t usage;
    uint32_t owner;
    uint32_t flags;
    npu_hal_core_t location;
    uint64_t size;
    uint64_t alignment;
    uint64_t fixed_address;
} npu_hal_memory_request_t;

typedef struct npu_hal_allocation_info {
    uint64_t address;
    uint64_t size;
    uint32_t memory_class;
    uint32_t scope;
    uint32_t usage;
    uint32_t owner;
    npu_hal_core_t location;
} npu_hal_allocation_info_t;

void npu_hal_options_init(npu_hal_options_t *options);
void npu_hal_memory_request_init(npu_hal_memory_request_t *request);
int npu_hal_open(npu_hal_t **hal_out, const npu_hal_options_t *options);
void npu_hal_close(npu_hal_t *hal);
/* Returns the first fatal backend I/O error, or zero while usable. */
int npu_hal_device_status(const npu_hal_t *hal);
int npu_hal_get_info(npu_hal_t *hal, npu_hal_info_t *info);
int npu_hal_get_memory_region(npu_hal_t *hal, uint32_t index,
                              npu_hal_memory_region_t *region);
int npu_hal_core_valid(npu_hal_t *hal, npu_hal_core_t core);
int npu_hal_memory_capacity(npu_hal_t *hal,
                            npu_hal_memory_class_t memory_class,
                            npu_hal_core_t viewer, uint64_t *capacity_out);

/* Raw address access is reserved for BL1/RPC implementation inside Service. */
int npu_hal_copy_to_address(npu_hal_t *hal, uint32_t cluster,
                            uint64_t address, const void *source, size_t size);
int npu_hal_copy_from_address(npu_hal_t *hal, uint32_t cluster,
                              void *destination, uint64_t address,
                              size_t size);
int npu_hal_copy_address(npu_hal_t *hal, uint32_t cluster,
                         uint64_t destination, uint64_t source, size_t size);

int npu_hal_register_read(npu_hal_t *hal, uint32_t cluster, uint64_t offset,
                          void *value, size_t size);
int npu_hal_register_write(npu_hal_t *hal, uint32_t cluster, uint64_t offset,
                           const void *value, size_t size);
int npu_hal_irq_status(npu_hal_t *hal, uint32_t cluster, uint32_t *status);
/* Returns -ETIMEDOUT with cluster/status reset when no IRQ arrives. */
int npu_hal_irq_wait(npu_hal_t *hal, uint32_t timeout_ms,
                     uint32_t *cluster, uint32_t *status);
int npu_hal_irq_ack(npu_hal_t *hal, uint32_t cluster, uint32_t mask);
int npu_hal_irq_mailbox(npu_hal_t *hal, npu_hal_core_t core,
                        uint64_t *command);
int npu_hal_command_submit(npu_hal_t *hal, npu_hal_core_t core,
                           uint64_t command);
int npu_hal_wait_command_consumed(npu_hal_t *hal, npu_hal_core_t core,
                                  uint32_t timeout_ms);
int npu_hal_boot_core(npu_hal_t *hal, npu_hal_core_t core, uint64_t entry,
                      uint32_t timeout_ms);

/* Physical address allocation and ownership live entirely in HAL. */
int npu_hal_memory_alloc(npu_hal_t *hal,
                         const npu_hal_memory_request_t *request,
                         npu_hal_allocation_t **allocation_out);
int npu_hal_memory_reserve(npu_hal_t *hal,
                           const npu_hal_memory_request_t *request,
                           npu_hal_allocation_t **allocation_out);
int npu_hal_memory_free(npu_hal_t *hal, npu_hal_allocation_t *allocation);
int npu_hal_memory_info(npu_hal_t *hal,
                        const npu_hal_allocation_t *allocation,
                        npu_hal_allocation_info_t *info);
int npu_hal_memory_visible(npu_hal_t *hal,
                           const npu_hal_allocation_t *allocation,
                           npu_hal_core_t viewer);
int npu_hal_address_valid(npu_hal_t *hal, npu_hal_core_t viewer,
                          uint64_t address, uint64_t size);
int npu_hal_memory_address(npu_hal_t *hal,
                           const npu_hal_allocation_t *allocation,
                           npu_hal_core_t viewer, uint64_t *address_out);
int npu_hal_memory_write(npu_hal_t *hal,
                         const npu_hal_allocation_t *allocation,
                         npu_hal_core_t viewer, uint64_t offset,
                         const void *source, size_t size);
int npu_hal_memory_read(npu_hal_t *hal,
                        const npu_hal_allocation_t *allocation,
                        npu_hal_core_t viewer, void *destination,
                        uint64_t offset, size_t size);

int npu_hal_export_data_plane(npu_hal_t *hal,
                              npu_hal_data_export_t *exported,
                              int handles[NPU_HAL_DATA_MAX_HANDLES]);
int npu_hal_prepare_transfer(npu_hal_t *hal,
                             const npu_hal_allocation_t *allocation,
                             npu_hal_core_t viewer, uint64_t offset,
                             size_t size,
                             npu_hal_data_direction_t direction,
                             npu_hal_data_token_t *token);
int npu_hal_finish_transfer(npu_hal_t *hal,
                            const npu_hal_data_token_t *token,
                            int transfer_status);

#ifdef __cplusplus
}
#endif

#endif
