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

#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>
#endif

#include <stddef.h>
#include <stdint.h>

struct alignas(64) TaskGridRecord {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t dim_x;
    uint32_t dim_y;
    uint32_t dim_z;
    uint32_t mode;
    uint32_t reserved;
    uint64_t marker;
    uint64_t padding[3];
};

static_assert(sizeof(TaskGridRecord) == 64, "task grid record ABI");

__kernel__ void task_grid_probe(TaskGridRecord *records)
{
    if (!records)
        dayan_thread_exit(-22);

    const uint32_t x = dayan_task_id(DAYAN_TASK_X);
    const uint32_t y = dayan_task_id(DAYAN_TASK_Y);
    const uint32_t z = dayan_task_id(DAYAN_TASK_Z);
    const uint32_t dim_x = dayan_task_dim(DAYAN_TASK_X);
    const uint32_t dim_y = dayan_task_dim(DAYAN_TASK_Y);
    const uint32_t dim_z = dayan_task_dim(DAYAN_TASK_Z);
    const uint32_t mode = dayan_task_mode();

    if (x >= 4U || y != 0U || z != 0U || dim_x != 4U || dim_y != 1U ||
        dim_z != 1U || mode != 1U)
        dayan_thread_exit(-22);

    TaskGridRecord *record = records + x;
    record->x = x;
    record->y = y;
    record->z = z;
    record->dim_x = dim_x;
    record->dim_y = dim_y;
    record->dim_z = dim_z;
    record->mode = mode;
    record->reserved = 0;
    record->marker = 0x7461736b00000000ULL | x;
    dayan_cache_flush(record, sizeof(*record));
}

DAYAN_KERNEL_DESC(task_grid_probe, 0, 0, 0, 0, 0, 0, 0);

#if !defined(__DAYAN_DEVICE__)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

extern "C" __host__ char *getenv(const char *name);

constexpr uint32_t kTaskCount = 4;
constexpr uint32_t kTimeoutMs = 300000;

__host__ static void preserve_error(int candidate, int *status)
{
    if (*status == 0 && candidate < 0)
        *status = candidate;
}

__host__ static int wait_and_destroy(npu_event_t *event)
{
    npu_event_info_t info;
    int status;
    int destroy_status;

    if (!event)
        return -1;
    status = npu_event_wait(event, kTimeoutMs, &info);
    destroy_status = npu_event_destroy(event);
    if (status < 0)
        return status;
    if (info.status < std::numeric_limits<int>::min() ||
        info.status > std::numeric_limits<int>::max())
        return -1;
    if (info.status < 0)
        return (int)info.status;
    return destroy_status;
}

__host__ static int upload(npu_stream_t *stream, npu_buffer_t *buffer,
                           const void *source, size_t size)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_h2d_async(stream, buffer, 0, source, size, &event);

    return status < 0 ? status : wait_and_destroy(event);
}

__host__ static int download(npu_stream_t *stream, void *destination,
                             npu_buffer_t *buffer, size_t size)
{
    npu_event_t *event = nullptr;
    int status = npu_memcpy_d2h_async(stream, destination, buffer, 0, size,
                                      &event);

    return status < 0 ? status : wait_and_destroy(event);
}

__host__ int main(int argc, char **argv)
{
    TaskGridRecord records[kTaskCount] = {};
    npu_runtime_options_t options;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    npu_buffer_t *buffer = nullptr;
    const char *socket_path;
    int status = 0;

    if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
        std::printf("usage: %s\n", argv[0]);
        return 0;
    }
    if (argc != 1)
        return 2;

    npu_runtime_options_init(&options);
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        options.server_socket = socket_path;
    status = npu_runtime_open(&runtime, &options);
    if (status == 0)
        status = npu_context_create(runtime, nullptr, &context);
    if (status == 0)
        status = npu_stream_create(context, nullptr, &stream);
    if (status == 0) {
        npu_buffer_desc_t desc;

        npu_buffer_desc_init(&desc, sizeof(records));
        desc.alignment = 64;
        status = npu_buffer_create(context, &desc, &buffer);
    }
    if (status == 0)
        status = upload(stream, buffer, records, sizeof(records));
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;

        npu_kernel_launch_attrs_init(&attrs, context, stream, "task-grid");
        attrs.config.task_dim = NPUTaskDim3{kTaskCount, 1, 1};
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms = kTimeoutMs;
        status = task_grid_probe(buffer, &attrs);
    }
    if (status == 0)
        status = download(stream, records, buffer, sizeof(records));

    for (uint32_t index = 0; status == 0 && index < kTaskCount; ++index) {
        const TaskGridRecord &record = records[index];
        const uint64_t expected_marker = 0x7461736b00000000ULL | index;

        std::printf("[host] slot=%u id=(%u,%u,%u) dim=(%u,%u,%u) "
                    "mode=%u marker=0x%016llx\n",
                    index, record.x, record.y, record.z, record.dim_x,
                    record.dim_y, record.dim_z, record.mode,
                    (unsigned long long)record.marker);
        if (record.x != index || record.y != 0U || record.z != 0U ||
            record.dim_x != kTaskCount || record.dim_y != 1U ||
            record.dim_z != 1U || record.mode != 1U ||
            record.marker != expected_marker)
            status = -1;
    }

    if (status == 0)
        std::printf("[host] PASS: task grid identities are distinct\n");
    else
        std::printf("[host] FAIL: task grid probe status=%d\n", status);

    if (buffer)
        preserve_error(npu_buffer_destroy(buffer), &status);
    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status < 0 ? 1 : 0;
}

#endif
