#include "service_internal.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "trap.h"
#include "tensor_interrupt.h"

#define NPU_MAX_PRINT_BYTES (1024U * 1024U)
#define NPU_RPC_IRQ_WAIT_TIMEOUT_MS 100U

static int range_valid(npu_service_device_t *device, npu_hal_core_t core,
                       uint64_t address, uint64_t size)
{
    return npu_hal_address_valid(device->hal, core, address, size);
}

static int copy_from_device(npu_service_device_t *device,
                            npu_hal_core_t core, void *destination,
                            uint64_t address, size_t size)
{
    if (!range_valid(device, core, address, size))
        return -EPROTO;
    return npu_hal_copy_from_address(device->hal, core.cluster, destination,
                                     address, size);
}

static int copy_to_device(npu_service_device_t *device, npu_hal_core_t core,
                          uint64_t address, const void *source, size_t size)
{
    if (!range_valid(device, core, address, size))
        return -EPROTO;
    return npu_hal_copy_to_address(device->hal, core.cluster, address,
                                   source, size);
}

static int read_packet(npu_service_device_t *device, npu_hal_core_t core,
                       uint64_t address, struct rpc_packet *packet)
{
    if (address & (RPC_DESCRIPTOR_GRANULE - 1U))
        return -EPROTO;
    return copy_from_device(device, core, packet, address, sizeof(*packet));
}

static int finish_packet(npu_service_device_t *device, npu_hal_core_t core,
                         uint64_t address, int64_t status)
{
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    if (address > UINT64_MAX - offsetof(struct rpc_packet, status))
        return -EOVERFLOW;
    return copy_to_device(device, core,
                          address + offsetof(struct rpc_packet, status),
                          &status, sizeof(status));
}

static int packet_layout_valid(npu_service_device_t *device,
                               npu_hal_core_t core,
                               const struct rpc_packet *packet,
                               uint64_t request_size,
                               uint64_t response_size)
{
    return packet->request_size == request_size &&
           packet->response_size == response_size &&
           (!request_size ||
            !(packet->request_addr & (RPC_DESCRIPTOR_GRANULE - 1U))) &&
           (!response_size ||
            !(packet->response_addr & (RPC_DESCRIPTOR_GRANULE - 1U))) &&
           (!request_size || range_valid(device, core, packet->request_addr,
                                         request_size)) &&
           (!response_size || range_valid(device, core,
                                          packet->response_addr,
                                          response_size));
}

static int service_print(npu_service_device_t *device, npu_hal_core_t core,
                         uint64_t argument, uint32_t flags)
{
    struct rpc_print_request request;
    struct rpc_packet packet;
    unsigned char *data = 0;
    int status;

    status = read_packet(device, core, argument, &packet);
    if (status < 0)
        return status;
    if (!packet_layout_valid(device, core, &packet, sizeof(request), 0)) {
        status = -EPROTO;
        goto finish;
    }
    status = copy_from_device(device, core, &request, packet.request_addr,
                              sizeof(request));
    if (status < 0)
        goto finish;
    if (request.hart_id != core.core ||
        request.data_size > NPU_MAX_PRINT_BYTES ||
        (request.data_size &&
         !range_valid(device, core, request.data_addr, request.data_size))) {
        status = -EPROTO;
        goto finish;
    }
    if (request.data_size) {
        data = malloc((size_t)request.data_size);
        if (!data) {
            status = -ENOMEM;
            goto finish;
        }
        status = copy_from_device(device, core, data, request.data_addr,
                                  (size_t)request.data_size);
        if (status < 0)
            goto finish;
    }
    device->options.output(device->options.callback_context, core.cluster,
                           core.core, request.proc_id, data,
                           (size_t)request.data_size, flags);
    status = 0;

finish:
    free(data);
    if (finish_packet(device, core, argument, status) < 0 && status == 0)
        status = -EIO;
    return status;
}

static int service_heap_acquire(npu_service_device_t *device,
                                npu_hal_core_t core, uint64_t argument)
{
    struct rpc_heap_response response = { 0, 0 };
    struct npu_service_heap_allocation *tracking = 0;
    npu_hal_allocation_t *allocation = 0;
    struct rpc_heap_request request;
    struct rpc_packet packet;
    uint64_t size;
    int status;

    status = read_packet(device, core, argument, &packet);
    if (status < 0)
        return status;
    if (!packet_layout_valid(device, core, &packet, sizeof(request),
                             sizeof(response))) {
        status = -EPROTO;
        goto finish;
    }
    status = copy_from_device(device, core, &request, packet.request_addr,
                              sizeof(request));
    if (status < 0)
        goto finish;
    if (!request.size || request.size > UINT64_MAX - 63U) {
        status = -EINVAL;
        goto finish;
    }
    size = (request.size + 63U) & ~UINT64_C(63);
    tracking = calloc(1, sizeof(*tracking));
    if (!tracking) {
        status = -ENOMEM;
        goto finish;
    }
    status = npu_service_alloc_visible(
        device, core, NPU_HAL_MEMORY_NORMAL, NPU_HAL_MEMORY_HEAP,
        (uint32_t)npu_service_core_index(device, core), size, 64,
        &allocation);
    if (status < 0)
        goto finish;
    status = npu_hal_memory_address(device->hal, allocation, core,
                                    &response.addr);
    if (status < 0)
        goto finish;
    response.size = size;
    tracking->allocation = allocation;
    tracking->owner = core;
    tracking->address = response.addr;
    tracking->size = size;
    pthread_mutex_lock(&device->mutex);
    tracking->next = device->heap_allocations;
    device->heap_allocations = tracking;
    pthread_mutex_unlock(&device->mutex);
    tracking = 0;
    allocation = 0;
    status = copy_to_device(device, core, packet.response_addr, &response,
                            sizeof(response));
    if (status < 0) {
        struct npu_service_heap_allocation **link;

        pthread_mutex_lock(&device->mutex);
        link = &device->heap_allocations;
        while (*link && ((*link)->address != response.addr ||
                         (*link)->owner.cluster != core.cluster ||
                         (*link)->owner.core != core.core))
            link = &(*link)->next;
        if (*link) {
            tracking = *link;
            *link = tracking->next;
        }
        pthread_mutex_unlock(&device->mutex);
        if (tracking) {
            (void)npu_hal_memory_free(device->hal, tracking->allocation);
            free(tracking);
            tracking = 0;
        }
    }

finish:
    if (allocation)
        (void)npu_hal_memory_free(device->hal, allocation);
    free(tracking);
    if (finish_packet(device, core, argument, status) < 0 && status == 0)
        status = -EIO;
    return status;
}

static int service_heap_release(npu_service_device_t *device,
                                npu_hal_core_t core, uint64_t argument)
{
    struct npu_service_heap_allocation *tracking = 0;
    struct rpc_heap_release request;
    struct rpc_packet packet;
    int status;

    status = read_packet(device, core, argument, &packet);
    if (status < 0)
        return status;
    if (!packet_layout_valid(device, core, &packet, sizeof(request), 0)) {
        status = -EPROTO;
        goto finish;
    }
    status = copy_from_device(device, core, &request, packet.request_addr,
                              sizeof(request));
    if (status < 0)
        goto finish;
    pthread_mutex_lock(&device->mutex);
    {
        struct npu_service_heap_allocation **link =
            &device->heap_allocations;

        while (*link && ((*link)->address != request.addr ||
                         (*link)->size != request.size ||
                         (*link)->owner.cluster != core.cluster ||
                         (*link)->owner.core != core.core))
            link = &(*link)->next;
        if (*link) {
            tracking = *link;
            *link = tracking->next;
        }
    }
    pthread_mutex_unlock(&device->mutex);
    if (!tracking) {
        status = -EPERM;
        goto finish;
    }
    status = npu_hal_memory_free(device->hal, tracking->allocation);
    free(tracking);

finish:
    if (finish_packet(device, core, argument, status) < 0 && status == 0)
        status = -EIO;
    return status;
}

static int service_trap(npu_service_device_t *device, npu_hal_core_t core,
                        uint64_t argument)
{
    struct trap_record record;
    int status;

    status = copy_from_device(device, core, &record, argument,
                              sizeof(record));
    if (status < 0)
        return status;
    if (record.hart_id != core.core)
        return -EPROTO;
    device->options.trap(device->options.callback_context, core.cluster,
                         core.core, &record);
    npu_service_dump_device(device, core, "trap", &record, sizeof(record));
    return 0;
}

static int service_tensor_interrupt(npu_service_device_t *device,
                                    npu_hal_core_t core, uint64_t argument)
{
    struct tensor_interrupt_record record;
    int status;

    status = copy_from_device(device, core, &record, argument,
                              sizeof(record));
    if (status < 0)
        return status;
    if (record.hart_id != core.core ||
        record.recent_count > TENSOR_INTERRUPT_RECENT_COUNT)
        return -EPROTO;
    device->options.tensor_interrupt(device->options.callback_context,
                                     core.cluster, core.core, &record);
    npu_service_dump_device(device, core, "tensor-interrupt", &record,
                            sizeof(record));
    return 0;
}

static int service_proc_exit(npu_service_device_t *device,
                             npu_hal_core_t core, uint64_t argument)
{
    struct rpc_proc_exit_request request;
    struct rpc_packet packet;
    int status;

    status = read_packet(device, core, argument, &packet);
    if (status < 0)
        return status;
    if (!packet_layout_valid(device, core, &packet, sizeof(request), 0)) {
        status = -EPROTO;
        goto finish;
    }
    status = copy_from_device(device, core, &request, packet.request_addr,
                              sizeof(request));
    if (status < 0)
        goto finish;
    if (request.hart_id != core.core ||
        !request.proc_id || request.parent_proc_id ||
        request.flags != RPC_PROC_EXIT_ROOT) {
        status = -EPROTO;
        goto finish;
    }
    device->options.proc_exit(device->options.callback_context,
                              core.cluster, core.core, request.proc_id,
                              request.result, request.cookie);
    status = 0;

finish:
    if (finish_packet(device, core, argument, status) < 0 && status == 0)
        status = -EIO;
    return status;
}

static int service_profile_flush(npu_service_device_t *device,
                                 npu_hal_core_t core, uint64_t argument)
{
    struct rpc_profile_flush_response response;
    struct rpc_profile_flush_request request;
    struct rpc_packet packet;
    int continue_profile = 0;
    int status;

    status = read_packet(device, core, argument, &packet);
    if (status < 0)
        return status;
    if (!packet_layout_valid(device, core, &packet, sizeof(request),
                             sizeof(response))) {
        status = -EPROTO;
        goto finish;
    }
    status = copy_from_device(device, core, &request, packet.request_addr,
                              sizeof(request));
    if (status < 0)
        goto finish;
    if (request.hart_id != core.core ||
        request.flags || !request.count ||
        request.count > request.capacity || !request.generation) {
        status = -EPROTO;
        goto finish;
    }

    memset(&response, 0, sizeof(response));
    response.generation = request.generation;
    response.status = npu_service_profile_copy(
        device, core, request.generation, request.buffer_addr,
        request.capacity, request.count, &continue_profile);
    response.continue_profile = response.status == 0 && continue_profile;
    status = copy_to_device(device, core, packet.response_addr, &response,
                            sizeof(response));
    if (status == 0)
        status = finish_packet(device, core, argument, 0);
    return status;

finish:
    if (finish_packet(device, core, argument, status) < 0 && status == 0)
        status = -EIO;
    return status;
}

int npu_rpc_dispatch(npu_service_device_t *device, npu_hal_core_t core,
                     uint64_t command)
{
    uint16_t opcode = rpc_command_opcode(command);
    uint64_t argument = rpc_command_arg(command);

    switch (opcode) {
    case RPC_WRITE:
        return service_print(device, core, argument, 0);
    case RPC_CLEAR:
        return service_print(device, core, argument,
                             NPU_SERVICE_OUTPUT_CLEAR);
    case RPC_HEAP_ACQUIRE:
        return service_heap_acquire(device, core, argument);
    case RPC_HEAP_RELEASE:
        return service_heap_release(device, core, argument);
    case RPC_TRAP_REPORT:
        return service_trap(device, core, argument);
    case RPC_TENSOR_INTERRUPT_REPORT:
        return service_tensor_interrupt(device, core, argument);
    case RPC_PROC_EXIT:
        return service_proc_exit(device, core, argument);
    case RPC_PROFILE_FLUSH:
        return service_profile_flush(device, core, argument);
    default:
        return -ENOSYS;
    }
}

static int pending_response(npu_service_device_t *device,
                            npu_hal_core_t core, uint64_t command)
{
    struct npu_pending_call *call =
        &device->calls[npu_service_core_index(device, core)];

    return call->pending &&
           rpc_command_opcode(command) == call->expected_opcode &&
           rpc_command_arg(command) == call->expected_argument;
}

void *npu_rpc_worker(void *arg)
{
    npu_service_device_t *device = arg;
    int device_failed = 0;

    while (!device_failed) {
        uint32_t cluster;
        uint32_t pending;
        uint32_t core_index;
        int wait_status;

        pthread_mutex_lock(&device->mutex);
        if (device->worker_stop) {
            pthread_mutex_unlock(&device->mutex);
            break;
        }
        pthread_mutex_unlock(&device->mutex);
        wait_status = npu_hal_irq_wait(device->hal,
                                       NPU_RPC_IRQ_WAIT_TIMEOUT_MS,
                                       &cluster, &pending);
        if (wait_status == -ETIMEDOUT)
            continue;
        if (wait_status < 0) {
            int fatal_status = npu_hal_device_status(device->hal);

            if (fatal_status < 0) {
                npu_service_abort(device, fatal_status);
                break;
            }
            npu_delay_us(device->options.rpc_poll_interval_us);
            continue;
        }
        for (core_index = 0;
             core_index < device->hal_info.cores_per_cluster;
             ++core_index) {
            npu_hal_core_t core = { cluster, core_index };
            uint32_t mask = UINT32_C(1) << core_index;
            uint64_t command;
            int is_response;

            if (!(pending & mask))
                continue;
            if (npu_hal_irq_mailbox(device->hal, core, &command) < 0) {
                int fatal_status = npu_hal_device_status(device->hal);

                if (fatal_status < 0) {
                    npu_service_abort(device, fatal_status);
                    device_failed = 1;
                    break;
                }
                continue;
            }
            pthread_mutex_lock(&device->mutex);
            is_response = pending_response(device, core, command);
            pthread_mutex_unlock(&device->mutex);
            if (!is_response)
                (void)npu_rpc_dispatch(device, core, command);
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            if (npu_hal_irq_ack(device->hal, cluster, mask) < 0) {
                int fatal_status = npu_hal_device_status(device->hal);

                if (fatal_status < 0) {
                    npu_service_abort(device, fatal_status);
                    device_failed = 1;
                    break;
                }
                continue;
            }
            if (is_response) {
                struct npu_pending_call *call =
                    &device->calls[npu_service_core_index(device, core)];

                pthread_mutex_lock(&device->mutex);
                if (pending_response(device, core, command)) {
                    call->status = 0;
                    call->pending = 0;
                    pthread_cond_broadcast(&device->condition);
                }
                pthread_mutex_unlock(&device->mutex);
            }
        }
    }
    return 0;
}
