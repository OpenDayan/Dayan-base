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

#ifndef DAYAN_RUNTIME_INTERNAL_H
#define DAYAN_RUNTIME_INTERNAL_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#include "dataplane.h"
#include "npu_runtime.h"
#include "protocol.h"

struct runtime_transfer_job;

struct npu_runtime {
    int control_fd;
    int data_fd;
    npu_hal_data_plane_t *data_plane;
    uint64_t session_id;
    uint64_t next_request_id;
    uint32_t default_timeout_ms;
    uint32_t closing;
    npu_runtime_output_fn output;
    void *output_context;
    pthread_mutex_t control_mutex;
    pthread_mutex_t data_mutex;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    pthread_t data_worker;
    uint32_t data_worker_started;
    uint32_t data_worker_stop;
    struct runtime_transfer_job *job_head;
    struct runtime_transfer_job *job_tail;
    struct npu_context *contexts;
    struct npu_profile *profiles;
};

struct npu_context {
    npu_runtime_t *runtime;
    uint64_t id;
    uint32_t closing;
    uint32_t transfer_jobs;
    struct npu_buffer *buffers;
    struct npu_stream *streams;
    struct npu_module *modules;
    struct npu_event *events;
    struct npu_context *next;
};

struct npu_buffer {
    struct npu_context *context;
    uint64_t id;
    uint64_t size;
    uint64_t base_offset;
    /*
     * Nonzero on a view, which owns no Server-side buffer and so must not
     * send BUFFER_DESTROY.  Deliberately a flag and not a pointer to the root:
     * a view outlives its root whenever the caller destroys them in that
     * order, and @id already carries the root's Server id, so nothing here
     * ever has to dereference it.
     */
    uint32_t is_view;
    struct npu_buffer *next;
};

struct npu_stream {
    struct npu_context *context;
    uint64_t id;
    struct npu_stream *next;
};

struct npu_module {
    struct npu_context *context;
    uint64_t id;
    struct npu_kernel *kernels;
    struct npu_module *next;
};

struct npu_kernel {
    struct npu_module *module;
    uint64_t id;
    struct npu_kernel *next;
};

struct npu_event {
    struct npu_context *context;
    uint64_t id;
    npu_event_info_t cached;
    struct npu_event *next;
};

struct runtime_transfer_job {
    struct npu_context *context;
    /*
     * The event's id, not the struct npu_event.  The worker runs after the
     * async submit call has returned, by which point the caller owns the event
     * and may already have destroyed it - holding the pointer and reading
     * event->id in the worker is a use-after-free.  The id is all the worker
     * needs, and the Server rejects a stale one.
     */
    uint64_t event_id;
    void *host_buffer;
    size_t size;
    struct runtime_transfer_job *next;
};

struct npu_profile {
    npu_runtime_t *runtime;
    uint64_t id;
    uint32_t stopped;
    uint32_t stopping;
    uint32_t destroying;
    struct npu_profile *next;
};

int runtime_control_call(npu_runtime_t *runtime, uint16_t opcode,
                         uint64_t object_id, const void *request,
                         size_t request_size, void *response,
                         size_t expected_response_size,
                         uint64_t *response_object_id);
int runtime_data_call(npu_runtime_t *runtime, uint16_t opcode,
                      uint64_t object_id, const void *request,
                      size_t request_size, void *response,
                      size_t expected_response_size);
int runtime_control_call_variable(npu_runtime_t *runtime, uint16_t opcode,
                                  uint64_t object_id, const void *request,
                                  size_t request_size, void **response_out,
                                  size_t *response_size_out);
int runtime_event_adopt(struct npu_context *context, uint64_t id,
                        struct npu_event *event,
                        struct npu_event **event_out);
void runtime_event_remove(struct npu_event *event);
int runtime_transfer_enqueue(struct runtime_transfer_job *job);
void *runtime_data_worker(void *argument);
int runtime_context_valid(const struct npu_context *context);
int runtime_stream_valid(const struct npu_stream *stream);
void runtime_kernel_descriptors_forget_context(struct npu_context *context);
int runtime_profile_close_all(npu_runtime_t *runtime);

#endif
