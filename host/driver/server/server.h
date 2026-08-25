#ifndef NPU_DRIVER_SERVER_H
#define NPU_DRIVER_SERVER_H

#include <stdint.h>

#include "service.h"

#define DAYAN_SERVER_OPTIONS_VERSION 11U

typedef struct dayan_server dayan_server_t;

typedef struct dayan_server_options {
    uint32_t version;
    uint32_t enabled_cluster_mask;
    uint32_t enabled_core_mask;
    uint32_t max_procs_per_hart;
    uint32_t lsu_tcsr_count_per_hart;
    uint32_t mat_tcsr_count_per_hart;
    uint32_t vec_tcsr_count_per_hart;
    uint32_t macro_tcsr_count_per_hart;
    const char *socket_path;
    const char *firmware_path;
    const char *api_elf_path;
    uint64_t resident_limit_per_core;
    npu_hal_options_t hal;
    npu_service_options_t service;
} dayan_server_options_t;

void dayan_server_options_init(dayan_server_options_t *options);
int dayan_server_create(dayan_server_t **server_out,
                        const dayan_server_options_t *options);
int dayan_server_run(dayan_server_t *server,
                     const volatile int *stop_requested);
void dayan_server_request_stop(dayan_server_t *server);
void dayan_server_destroy(dayan_server_t *server);

#endif
