/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#ifndef NPU_DRIVER_PROTOCOL_H
#define NPU_DRIVER_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

#include "dataplane.h"
#include "npu_pcu.h"
#include "npu_runtime.h"

#define NPU_IPC_MAGIC             UINT32_C(0x4e594144) /* DAYN, little endian */
#define NPU_IPC_VERSION           11U
#define NPU_IPC_HEADER_SIZE       40U
#define NPU_IPC_MAX_PAYLOAD       (64U * 1024U * 1024U)
#define NPU_IPC_RESPONSE          (1U << 0)

#define NPU_IPC_ROLE_CONTROL      1U
#define NPU_IPC_ROLE_DATA         2U
#define NPU_IPC_CAP_DIRECT_TRANSFER (1U << 0)
#define NPU_IPC_CAP_DEVICE_OUTPUT   (1U << 1)
#define NPU_IPC_CAP_PROFILE         (1U << 2)

#define NPU_IPC_OUTPUT_CLEAR        (1U << 0)
#define NPU_IPC_OUTPUT_CONTINUED    (1U << 1)
#define NPU_IPC_OUTPUT_CHUNK_SIZE   4096U
#define NPU_IPC_PROFILE_CHUNK_BYTES (64U * 1024U)
#define NPU_IPC_PROFILE_END          (1U << 0)
#define NPU_IPC_PROC_NAME_SIZE       16U

typedef enum npu_ipc_opcode {
    NPU_IPC_HELLO = 1,
    NPU_IPC_TOPOLOGY = 2,
    NPU_IPC_URAM_CAPACITY = 3,
    NPU_IPC_CONTEXT_CREATE = 0x0100,
    NPU_IPC_CONTEXT_DESTROY,
    NPU_IPC_BUFFER_CREATE = 0x0200,
    NPU_IPC_BUFFER_DESTROY,
    NPU_IPC_STREAM_CREATE = 0x0300,
    NPU_IPC_STREAM_DESTROY,
    NPU_IPC_STREAM_SYNCHRONIZE,
    NPU_IPC_STREAM_WAIT_EVENT,
    NPU_IPC_TRANSFER_SUBMIT = 0x0400,
    NPU_IPC_TRANSFER_ACQUIRE,
    NPU_IPC_TRANSFER_COMPLETE,
    NPU_IPC_MODULE_LOAD = 0x0500,
    NPU_IPC_MODULE_UNLOAD,
    NPU_IPC_KERNEL_GET,
    NPU_IPC_KERNEL_RELEASE,
    NPU_IPC_KERNEL_INFO,
    NPU_IPC_KERNEL_LAUNCH,
    NPU_IPC_EVENT_RECORD = 0x0600,
    NPU_IPC_EVENT_QUERY,
    NPU_IPC_EVENT_WAIT,
    NPU_IPC_EVENT_DESTROY,
    NPU_IPC_OUTPUT_READ = 0x0700,
    NPU_IPC_PROFILE_START = 0x0800,
    NPU_IPC_PROFILE_STOP,
    NPU_IPC_PROFILE_READ,
    NPU_IPC_PROFILE_DESTROY,
    NPU_IPC_PROFILE_PCU_READ,
} npu_ipc_opcode_t;

typedef struct npu_ipc_header {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint16_t opcode;
    uint16_t flags;
    uint32_t payload_size;
    uint64_t request_id;
    uint64_t object_id;
    int32_t status;
    uint32_t reserved;
} npu_ipc_header_t;

typedef struct npu_ipc_hello_request {
    uint32_t role;
    uint32_t runtime_version;
    uint64_t session_id;
} npu_ipc_hello_request_t;

typedef struct npu_ipc_hello_response {
    uint64_t session_id;
    uint32_t service_version;
    uint32_t capabilities;
    npu_hal_data_export_t data_plane;
} npu_ipc_hello_response_t;

typedef struct npu_ipc_context_create {
    uint64_t memory_quota;
    uint32_t max_pending_commands;
    uint32_t flags;
} npu_ipc_context_create_t;

typedef struct npu_ipc_buffer_create {
    uint64_t context_id;
    uint64_t size;
    uint64_t alignment;
    uint32_t flags;
    uint32_t reserved;
} npu_ipc_buffer_create_t;

typedef struct npu_ipc_stream_create {
    uint64_t context_id;
    int32_t priority;
    uint32_t flags;
} npu_ipc_stream_create_t;

typedef struct npu_ipc_stream_wait {
    uint64_t stream_id;
    uint64_t event_id;
    uint32_t timeout_ms;
    uint32_t reserved;
} npu_ipc_stream_wait_t;

typedef enum npu_ipc_transfer_direction {
    NPU_IPC_TRANSFER_H2D = 1,
    NPU_IPC_TRANSFER_D2H = 2,
    NPU_IPC_TRANSFER_D2D = 3,
} npu_ipc_transfer_direction_t;

typedef struct npu_ipc_transfer_submit {
    uint64_t context_id;
    uint64_t stream_id;
    uint64_t destination_buffer;
    uint64_t destination_offset;
    uint64_t source_buffer;
    uint64_t source_offset;
    uint64_t size;
    uint32_t direction;
    uint32_t reserved;
} npu_ipc_transfer_submit_t;

typedef struct npu_ipc_transfer_grant {
    uint64_t event_id;
    uint64_t transfer_id;
    npu_hal_data_token_t token;
} npu_ipc_transfer_grant_t;

typedef struct npu_ipc_transfer_complete {
    uint64_t event_id;
    uint64_t transfer_id;
    int32_t transfer_status;
    uint32_t reserved;
    npu_hal_data_token_t token;
} npu_ipc_transfer_complete_t;

typedef struct npu_ipc_module_load {
    uint64_t context_id;
    uint64_t package_size;
} npu_ipc_module_load_t;

typedef struct npu_ipc_kernel_get {
    uint64_t module_id;
    uint32_t name_size;
    uint32_t reserved;
} npu_ipc_kernel_get_t;

typedef struct npu_ipc_launch_argument {
    uint32_t kind;
    uint32_t access;
    uint32_t size;
    uint32_t reserved;
    uint64_t value_or_buffer;
    uint64_t offset;
} npu_ipc_launch_argument_t;

typedef struct npu_ipc_kernel_launch {
    uint64_t context_id;
    uint64_t stream_id;
    uint64_t kernel_id;
    uint64_t cookie;
    uint32_t flags;
    uint32_t task_x;
    uint32_t task_y;
    uint32_t task_z;
    uint32_t task_mode;
    uint32_t argument_count;
    uint32_t reserved;
    char task_name[NPU_IPC_PROC_NAME_SIZE];
    npu_ipc_launch_argument_t arguments[NPU_KERNEL_MAX_ARGS];
} npu_ipc_kernel_launch_t;

typedef struct npu_ipc_kernel_info {
    uint32_t supported_task_modes;
    uint32_t max_tasks_per_core;
    uint32_t max_child_threads;
    uint32_t argument_count;
    uint32_t sync_resource_count;
    uint32_t flags;
    uint8_t lsu_tcsr_count;
    uint8_t mat_tcsr_count;
    uint8_t vec_tcsr_count;
    uint8_t macro_tcsr_count;
    uint8_t local_resource_count;
    uint8_t shared_resource_count;
    uint16_t reserved;
    uint64_t stack_size;
    uint64_t atomic_cache_size;
    uint64_t resident_memory;
    uint64_t uram_size;
} npu_ipc_kernel_info_t;

typedef struct npu_ipc_event_info {
    uint32_t state;
    uint32_t command_type;
    int64_t status;
    uint64_t sequence;
    uint64_t cookie;
} npu_ipc_event_info_t;

typedef struct npu_ipc_event_wait {
    uint64_t event_id;
    uint32_t timeout_ms;
    uint32_t reserved;
} npu_ipc_event_wait_t;

typedef struct npu_ipc_output {
    uint32_t cluster;
    uint32_t core;
    uint32_t proc;
    uint32_t flags;
    uint32_t data_size;
    uint32_t reserved;
} npu_ipc_output_t;

typedef struct npu_ipc_profile_start {
    uint64_t buffer_size;
    uint32_t flags;
    uint32_t reserved;
} npu_ipc_profile_start_t;

typedef struct npu_ipc_profile_read {
    uint32_t max_bytes;
    uint32_t timeout_ms;
} npu_ipc_profile_read_t;

typedef struct npu_ipc_profile_batch {
    uint32_t cluster;
    uint32_t hart;
    uint32_t byte_count;
    uint32_t flags;
} npu_ipc_profile_batch_t;

#ifdef __cplusplus
static_assert(sizeof(npu_ipc_header_t) == NPU_IPC_HEADER_SIZE,
              "Dayan socket header ABI");
static_assert(sizeof(npu_ipc_hello_response_t) == 64,
              "Dayan HELLO response ABI");
static_assert(offsetof(npu_ipc_hello_response_t, data_plane) == 16,
              "Dayan data-plane export offset");
static_assert(sizeof(npu_ipc_output_t) == 24,
              "Dayan Device output header ABI");
static_assert(sizeof(npu_ipc_kernel_launch_t) == 1104,
              "Dayan kernel launch IPC ABI");
static_assert(offsetof(npu_ipc_kernel_launch_t, task_name) == 60,
              "Dayan kernel launch name IPC offset");
static_assert(offsetof(npu_ipc_kernel_launch_t, arguments) == 80,
              "Dayan kernel launch arguments IPC offset");
static_assert(sizeof(npu_ipc_kernel_info_t) == 64,
              "Dayan kernel info IPC ABI");
static_assert(sizeof(npu_ipc_profile_start_t) == 16,
              "Dayan profile start IPC ABI");
static_assert(sizeof(npu_ipc_profile_batch_t) == 16,
              "Dayan profile batch IPC ABI");
#else
_Static_assert(sizeof(npu_ipc_header_t) == NPU_IPC_HEADER_SIZE,
               "Dayan socket header ABI");
_Static_assert(sizeof(npu_ipc_hello_response_t) == 64,
               "Dayan HELLO response ABI");
_Static_assert(offsetof(npu_ipc_hello_response_t, data_plane) == 16,
               "Dayan data-plane export offset");
_Static_assert(sizeof(npu_ipc_output_t) == 24,
               "Dayan Device output header ABI");
_Static_assert(sizeof(npu_ipc_kernel_launch_t) == 1104,
               "Dayan kernel launch IPC ABI");
_Static_assert(offsetof(npu_ipc_kernel_launch_t, task_name) == 60,
               "Dayan kernel launch name IPC offset");
_Static_assert(offsetof(npu_ipc_kernel_launch_t, arguments) == 80,
               "Dayan kernel launch arguments IPC offset");
_Static_assert(sizeof(npu_ipc_kernel_info_t) == 64,
               "Dayan kernel info IPC ABI");
_Static_assert(sizeof(npu_ipc_profile_start_t) == 16,
               "Dayan profile start IPC ABI");
_Static_assert(sizeof(npu_ipc_profile_batch_t) == 16,
               "Dayan profile batch IPC ABI");
#endif

void npu_ipc_header_init(npu_ipc_header_t *header, uint16_t opcode,
                         uint64_t request_id, uint64_t object_id,
                         uint32_t payload_size);
int npu_ipc_connect(const char *path, uint32_t timeout_ms);
int npu_ipc_listen(const char *path, unsigned backlog);
int npu_ipc_send(int fd, const npu_ipc_header_t *header,
                 const void *payload, const int *handles,
                 size_t handle_count);
int npu_ipc_receive(int fd, npu_ipc_header_t *header, void **payload_out,
                    int handles[NPU_HAL_DATA_MAX_HANDLES],
                    size_t *handle_count_out);

#endif
