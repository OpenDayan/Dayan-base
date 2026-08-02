#ifndef NPU_HOST_RUNTIME_INTERNAL_H
#define NPU_HOST_RUNTIME_INTERNAL_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "npu_runtime.h"
#include "rpc.h"

enum npu_allocation_kind {
    NPU_ALLOC_USER = 1,
    NPU_ALLOC_FIRMWARE,
    NPU_ALLOC_HEAP,
    NPU_ALLOC_LAUNCH,
    NPU_ALLOC_COHERENT,
    NPU_ALLOC_KERNEL,
};

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
    uint64_t base;
    uint64_t size;
    struct npu_kernel_export *exports;
    size_t export_count;
    struct npu_kernel_module *next;
};

struct npu_allocation {
    uint64_t address;
    uint64_t size;
    uint32_t kind;
    uint32_t owner;
    struct npu_allocation *next;
};

struct npu_arena {
    uint64_t base;
    uint64_t size;
    struct npu_allocation *allocations;
};

struct npu_pending_call {
    uint16_t expected_opcode;
    uint16_t reserved;
    uint32_t pending;
    uint64_t expected_argument;
    uint64_t response;
    int status;
};

struct npu_runtime {
    npu_device_t device;
    npu_runtime_options_t options;
    pthread_mutex_t mutex;
    pthread_mutex_t kernel_mutex;
    pthread_cond_t condition;
    pthread_t worker;
    struct npu_arena normal;
    struct npu_arena coherent;
    struct npu_pending_call calls[NPU_DRIVER_MAX_HARTS];
    uint32_t worker_started;
    uint32_t worker_stop;
    uint32_t firmware_loaded;
    uint32_t booted_harts;
    uint32_t active_launches;
    uint64_t firmware_entry;
    struct npu_api_symbol *api_symbols;
    size_t api_symbol_count;
    struct npu_kernel_module *kernel_modules;
};

struct npu_launch {
    npu_runtime_t *runtime;
    uint64_t request_address;
    uint64_t owned_shared;
    uint64_t shared_base;
    uint64_t shared_size;
    uint32_t released;
};

int npu_is_power_of_two(size_t value);
uint64_t npu_now_ms(void);
void npu_delay_us(uint32_t delay_us);

void npu_arena_init(struct npu_arena *arena, uint64_t base, uint64_t size);
void npu_arena_destroy(struct npu_arena *arena);
int npu_arena_alloc(struct npu_arena *arena, uint64_t size,
                    uint64_t alignment, uint32_t kind, uint32_t owner,
                    uint64_t *address_out);
int npu_arena_reserve(struct npu_arena *arena, uint64_t address,
                      uint64_t size, uint32_t kind, uint32_t owner);
int npu_arena_free(struct npu_arena *arena, uint64_t address,
                   uint32_t kind, uint32_t owner);

int npu_runtime_load_bl1_elf(npu_runtime_t *runtime,
                             const void *image, size_t size,
                             uint32_t hart_mask);
int npu_kernel_resolve(npu_runtime_t *runtime,
                       const void *object, size_t object_size,
                       const char *entry_symbol, uint64_t *entry_out);
void npu_kernel_state_destroy(npu_runtime_t *runtime);
void *npu_rpc_worker(void *arg);
int npu_rpc_dispatch(npu_runtime_t *runtime, uint32_t hart,
                     uint64_t command);

#endif
