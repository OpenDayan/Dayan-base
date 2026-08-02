#include "runtime_internal.h"

#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "trap.h"

#define NPU_MAX_PRINT_BYTES (16U * 1024U * 1024U)

static int device_range_valid(npu_runtime_t *runtime,
                              uint64_t address, uint64_t size)
{
    return size <= runtime->options.ram_size &&
           address >= runtime->options.ram_base &&
           address - runtime->options.ram_base <=
               runtime->options.ram_size - size;
}

static int read_packet(npu_runtime_t *runtime, uint64_t address,
                       struct rpc_packet *packet)
{
    if ((address & 7U) ||
        !device_range_valid(runtime, address, sizeof(*packet)))
        return -EPROTO;
    return npu_device_read(&runtime->device, address,
                           packet, sizeof(*packet));
}

static int finish_packet(npu_runtime_t *runtime, uint64_t address,
                         int64_t status)
{
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return npu_device_write(&runtime->device,
                            address + offsetof(struct rpc_packet, status),
                            &status, sizeof(status));
}

static int packet_layout_valid(npu_runtime_t *runtime,
                               const struct rpc_packet *packet,
                               uint64_t request_size,
                               uint64_t response_size)
{
    if (packet->request_size != request_size ||
        packet->response_size != response_size)
        return 0;
    if (request_size && !device_range_valid(runtime, packet->request_addr,
                                            request_size))
        return 0;
    if (response_size && !device_range_valid(runtime, packet->response_addr,
                                             response_size))
        return 0;
    return 1;
}

static int service_print(npu_runtime_t *runtime, uint32_t hart,
                         uint64_t argument, uint32_t flags)
{
    struct rpc_print_request request;
    struct rpc_packet packet;
    unsigned char *data = 0;
    int status;

    status = read_packet(runtime, argument, &packet);
    if (status < 0)
        return status;
    if (!packet_layout_valid(runtime, &packet, sizeof(request), 0)) {
        status = -EPROTO;
        goto finish;
    }
    status = npu_device_read(&runtime->device, packet.request_addr,
                             &request, sizeof(request));
    if (status < 0)
        goto finish;
    if (request.hart_id != hart ||
        request.data_size > NPU_MAX_PRINT_BYTES ||
        (request.data_size &&
         !device_range_valid(runtime, request.data_addr,
                             request.data_size))) {
        status = -EPROTO;
        goto finish;
    }
    if (request.data_size) {
        data = malloc((size_t)request.data_size);
        if (!data) {
            status = -ENOMEM;
            goto finish;
        }
        status = npu_device_read(&runtime->device, request.data_addr,
                                 data, (size_t)request.data_size);
        if (status < 0)
            goto finish;
    }
    runtime->options.output(runtime->options.callback_context,
                            request.hart_id, request.proc_id,
                            data, (size_t)request.data_size, flags);
    status = 0;
finish:
    free(data);
    if (finish_packet(runtime, argument, status) < 0 && status == 0)
        status = -EIO;
    return status;
}

static int service_heap_acquire(npu_runtime_t *runtime, uint32_t hart,
                                uint64_t argument)
{
    struct rpc_heap_response response = { 0, 0 };
    struct rpc_heap_request request;
    struct rpc_packet packet;
    uint64_t address = 0;
    uint64_t size;
    int status;

    status = read_packet(runtime, argument, &packet);
    if (status < 0)
        return status;
    if (!packet_layout_valid(runtime, &packet, sizeof(request),
                             sizeof(response))) {
        status = -EPROTO;
        goto finish;
    }
    status = npu_device_read(&runtime->device, packet.request_addr,
                             &request, sizeof(request));
    if (status < 0)
        goto finish;
    if (!request.size || request.size > runtime->options.ram_size) {
        status = -EINVAL;
        goto finish;
    }
    size = (request.size + 63U) & ~UINT64_C(63);
    if (size < request.size) {
        status = -EOVERFLOW;
        goto finish;
    }
    pthread_mutex_lock(&runtime->mutex);
    status = npu_arena_alloc(&runtime->normal, size, 64,
                             NPU_ALLOC_HEAP, hart, &address);
    pthread_mutex_unlock(&runtime->mutex);
    if (status < 0)
        goto finish;
    response.addr = address;
    response.size = size;
    status = npu_device_write(&runtime->device, packet.response_addr,
                              &response, sizeof(response));
    if (status < 0) {
        pthread_mutex_lock(&runtime->mutex);
        (void)npu_arena_free(&runtime->normal, address,
                             NPU_ALLOC_HEAP, hart);
        pthread_mutex_unlock(&runtime->mutex);
    }
finish:
    if (finish_packet(runtime, argument, status) < 0 && status == 0)
        status = -EIO;
    return status;
}

static int service_heap_release(npu_runtime_t *runtime, uint32_t hart,
                                uint64_t argument)
{
    struct rpc_heap_release request;
    struct rpc_packet packet;
    int status;

    status = read_packet(runtime, argument, &packet);
    if (status < 0)
        return status;
    if (!packet_layout_valid(runtime, &packet, sizeof(request), 0)) {
        status = -EPROTO;
        goto finish;
    }
    status = npu_device_read(&runtime->device, packet.request_addr,
                             &request, sizeof(request));
    if (status < 0)
        goto finish;
    pthread_mutex_lock(&runtime->mutex);
    {
        struct npu_allocation *allocation = runtime->normal.allocations;

        while (allocation && allocation->address != request.addr)
            allocation = allocation->next;
        if (!allocation || allocation->size != request.size ||
            allocation->kind != NPU_ALLOC_HEAP || allocation->owner != hart)
            status = -EPERM;
        else
            status = npu_arena_free(&runtime->normal, request.addr,
                                    NPU_ALLOC_HEAP, hart);
    }
    pthread_mutex_unlock(&runtime->mutex);
finish:
    if (finish_packet(runtime, argument, status) < 0 && status == 0)
        status = -EIO;
    return status;
}

static int service_trap(npu_runtime_t *runtime, uint32_t hart,
                        uint64_t argument)
{
    struct trap_record record;
    int status;

    if (!device_range_valid(runtime, argument, sizeof(record)))
        return -EPROTO;
    status = npu_device_read(&runtime->device, argument,
                             &record, sizeof(record));
    if (status < 0)
        return status;
    if (record.version != TRAP_RECORD_VERSION ||
        record.size != sizeof(record) || record.hart_id != hart)
        return -EPROTO;
    runtime->options.trap(runtime->options.callback_context, &record);
    return 0;
}

int npu_rpc_dispatch(npu_runtime_t *runtime, uint32_t hart,
                     uint64_t command)
{
    uint16_t opcode = rpc_command_opcode(command);
    uint64_t argument = rpc_command_arg(command);

    switch (opcode) {
    case RPC_WRITE:
        return service_print(runtime, hart, argument, 0);
    case RPC_CLEAR:
        return service_print(runtime, hart, argument, NPU_OUTPUT_CLEAR);
    case RPC_HEAP_ACQUIRE:
        return service_heap_acquire(runtime, hart, argument);
    case RPC_HEAP_RELEASE:
        return service_heap_release(runtime, hart, argument);
    case RPC_TRAP_REPORT:
        return service_trap(runtime, hart, argument);
    default:
        return -ENOSYS;
    }
}

static int pending_response(npu_runtime_t *runtime, uint32_t hart,
                            uint64_t command)
{
    struct npu_pending_call *call = &runtime->calls[hart];

    return call->pending &&
           rpc_command_opcode(command) == call->expected_opcode &&
           rpc_command_arg(command) == call->expected_argument;
}

void *npu_rpc_worker(void *arg)
{
    npu_runtime_t *runtime = arg;

    for (;;) {
        uint32_t pending = 0;
        uint32_t hart;

        pthread_mutex_lock(&runtime->mutex);
        if (runtime->worker_stop) {
            pthread_mutex_unlock(&runtime->mutex);
            break;
        }
        pthread_mutex_unlock(&runtime->mutex);
        if (npu_device_read_irq_status(&runtime->device, &pending) < 0) {
            npu_delay_us(runtime->options.rpc_poll_interval_us);
            continue;
        }
        for (hart = 0; hart < runtime->options.hart_count; ++hart) {
            uint32_t mask = 1U << hart;
            uint64_t command;
            int is_response;

            if (!(pending & mask))
                continue;
            if (npu_device_read_irq_mailbox(&runtime->device, hart,
                                            &command) < 0)
                continue;
            pthread_mutex_lock(&runtime->mutex);
            is_response = pending_response(runtime, hart, command);
            pthread_mutex_unlock(&runtime->mutex);
            if (!is_response)
                (void)npu_rpc_dispatch(runtime, hart, command);
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            if (npu_device_clear_irq(&runtime->device, mask) < 0)
                continue;
            if (is_response) {
                pthread_mutex_lock(&runtime->mutex);
                if (pending_response(runtime, hart, command)) {
                    runtime->calls[hart].response = command;
                    runtime->calls[hart].status = 0;
                    runtime->calls[hart].pending = 0;
                    pthread_cond_broadcast(&runtime->condition);
                }
                pthread_mutex_unlock(&runtime->mutex);
            }
        }
        if (!pending)
            npu_delay_us(runtime->options.rpc_poll_interval_us);
    }
    return 0;
}
