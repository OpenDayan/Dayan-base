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

struct npu_api_symbol {
    char *name;
    uint64_t value;
};

struct npu_kernel_export {
    char *name;
    uint64_t value;
};

struct npu_kernel_module {
    const void *object_identity;
    size_t object_size;
    uint32_t cluster;
    uint64_t base;
    uint64_t size;
    npu_hal_allocation_t *allocation;
    struct npu_kernel_export *exports;
    size_t export_count;
    struct npu_kernel_module *next;
};

struct npu_pending_call {
    uint16_t expected_opcode;
    uint16_t reserved;
    uint32_t pending;
    uint64_t expected_argument;
    uint64_t response;
    int status;
};

struct npu_service_cluster {
    uint32_t firmware_loaded;
    uint32_t booted_cores;
    uint64_t firmware_entry;
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

struct npu_service_device {
    npu_hal_t *hal;
    npu_hal_info_t hal_info;
    npu_service_options_t options;
    pthread_mutex_t mutex;
    pthread_mutex_t kernel_mutex;
    pthread_cond_t condition;
    pthread_t worker;
    struct npu_pending_call *calls;
    struct npu_service_cluster *clusters;
    struct npu_service_heap_allocation *heap_allocations;
    struct npu_service_firmware_allocation *firmware_allocations;
    uint32_t worker_started;
    uint32_t worker_stop;
    uint32_t active_launches;
    struct npu_api_symbol *api_symbols;
    size_t api_symbol_count;
    struct npu_kernel_module *kernel_modules;
};

struct npu_service_launch {
    npu_service_device_t *device;
    npu_hal_core_t target;
    npu_hal_allocation_t *request_allocation;
    uint64_t request_address;
    uint32_t released;
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

int npu_service_load_bl1_elf(npu_service_device_t *device, uint32_t cluster,
                             const void *image, size_t size,
                             uint32_t core_mask);
int npu_service_rpc_call(npu_service_device_t *device, npu_hal_core_t core,
                         uint16_t opcode, uint64_t argument,
                         uint64_t *response_out, uint32_t timeout_ms);
void npu_service_kernel_state_destroy(npu_service_device_t *device);
void *npu_rpc_worker(void *arg);
int npu_rpc_dispatch(npu_service_device_t *device, npu_hal_core_t core,
                     uint64_t command);

#endif
