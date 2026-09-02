#include "server_internal.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SERVER_DEFAULT_STACK_SIZE   UINT64_C(0x10000)
#define SERVER_DEFAULT_PRINTF_SIZE  UINT64_C(0x1000)
#define SERVER_SHARED_ALIGNMENT     UINT64_C(64)
#define SERVER_SHARED_RESOURCE_SIZE UINT64_C(16)
#define SERVER_URAM_ALIGNMENT       DAYAN_CONTEXT_URAM_GRANULE
#define SERVER_TRAP_KILL_TIMEOUT_MS 30000U

static int calculate_shared_size(const struct server_command *command,
                                 uint64_t *size_out);

static int event_terminal(const struct server_event *event)
{
    return event->state == NPU_EVENT_COMPLETE ||
           event->state == NPU_EVENT_FAILED ||
           event->state == NPU_EVENT_CANCELLED;
}

static void remove_event_locked(struct server_context *context,
                                struct server_event *event)
{
    struct server_event **link = &context->events;

    while (*link && *link != event)
        link = &(*link)->next;
    if (*link)
        *link = event->next;
}

int server_enqueue_locked(dayan_server_t *server,
                          struct server_context *context,
                          struct server_stream *stream,
                          struct server_command *command,
                          uint32_t visible_event,
                          struct server_event **event_out)
{
    struct server_event *event;

    if (!server || !context || !stream || !command || !event_out ||
        context->closing || stream->context != context)
        return -EINVAL;
    if (context->max_pending_commands &&
        context->pending_commands >= context->max_pending_commands)
        return -EAGAIN;
    event = calloc(1, sizeof(*event));
    if (!event)
        return -ENOMEM;
    if (server->next_id == UINT64_MAX ||
        server->next_sequence == UINT64_MAX) {
        free(event);
        return -EOVERFLOW;
    }
    event->id = ++server->next_id;
    event->sequence = ++server->next_sequence;
    event->state = NPU_EVENT_QUEUED;
    event->command_type = command->kind;
    event->status = -EINPROGRESS;
    event->visible = visible_event;
    event->command = command;
    event->context = context;
    event->next = context->events;
    context->events = event;
    command->context = context;
    command->stream = stream;
    command->event = event;
    command->next = 0;
    if (stream->tail)
        stream->tail->next = command;
    else
        stream->head = command;
    stream->tail = command;
    ++context->pending_commands;
    *event_out = event;
    pthread_cond_broadcast(&server->condition);
    return 0;
}

static void free_launch_groups(dayan_server_t *server,
                               struct server_launch_group *group)
{
    while (group) {
        struct server_launch_group *next = group->next;

        if (group->shared_allocation)
            (void)npu_hal_memory_free(server->hal,
                                      group->shared_allocation);
        free(group);
        group = next;
    }
}

void server_complete_command_locked(dayan_server_t *server,
                                    struct server_command *command,
                                    int64_t status)
{
    struct server_context *context = command->context;
    struct server_stream *stream = command->stream;
    struct server_event *event = command->event;
    size_t index;

    if (stream->head != command || command->dispatching)
        return;
    stream->head = command->next;
    if (!stream->head)
        stream->tail = 0;
    if (context->pending_commands)
        --context->pending_commands;
    event->status = status;
    event->command = 0;
    if (status == -ECANCELED)
        event->state = NPU_EVENT_CANCELLED;
    else if (status < 0)
        event->state = NPU_EVENT_FAILED;
    else
        event->state = NPU_EVENT_COMPLETE;

    switch (command->kind) {
    case SERVER_COMMAND_WAIT:
        if (command->data.wait.dependency->references)
            --command->data.wait.dependency->references;
        break;
    case SERVER_COMMAND_H2D:
    case SERVER_COMMAND_D2H:
    case SERVER_COMMAND_D2D:
        if (command->data.transfer.destination &&
            command->data.transfer.destination->references)
            --command->data.transfer.destination->references;
        if (command->data.transfer.source &&
            command->data.transfer.source->references)
            --command->data.transfer.source->references;
        break;
    case SERVER_COMMAND_LAUNCH:
        for (index = 0; index < command->data.launch.argument_count; ++index) {
            if (command->data.launch.buffers[index] &&
                command->data.launch.buffers[index]->references)
                --command->data.launch.buffers[index]->references;
        }
        if (command->data.launch.kernel->references)
            --command->data.launch.kernel->references;
        if (command->data.launch.kernel->module->active_commands)
            --command->data.launch.kernel->module->active_commands;
        free_launch_groups(server, command->data.launch.groups);
        break;
    default:
        break;
    }
    free(command);
    if (!event->visible && !event->references) {
        remove_event_locked(context, event);
        free(event);
    }
    pthread_cond_broadcast(&server->condition);
}

void server_cancel_context_locked(dayan_server_t *server,
                                  struct server_context *context)
{
    struct server_stream *stream;

    if (!context || context->closing)
        return;
    context->closing = 1;
    for (stream = context->streams; stream; stream = stream->next) {
        struct server_command *command;

        for (command = stream->head; command; command = command->next) {
            command->cancelled = 1;
        }
    }
    pthread_cond_broadcast(&server->condition);
}

static uint64_t launch_resident_size(const npu_package_kernel_t *metadata)
{
    return metadata->resident_memory;
}

static uint32_t launch_reserved_procs(const npu_package_kernel_t *metadata)
{
    return metadata->max_child_threads + 1U;
}

static int launch_locks_task(const npu_package_kernel_t *metadata)
{
    return !!(metadata->flags & NPU_PACKAGE_KERNEL_LOCKED_TASK);
}

static int tcsr_range_available(uint64_t used, uint32_t capacity,
                                uint32_t count)
{
    uint32_t base;

    if (!count)
        return 1;
    if (count > DAYAN_CONTEXT_TCSR_MAX_COUNT || count > capacity)
        return 0;
    for (base = 0; base <= capacity - count; ++base) {
        uint64_t mask = ((UINT64_C(1) << count) - 1U) << base;

        if (!(used & mask))
            return 1;
    }
    return 0;
}

static int tcsr_range_reserve(uint64_t *used, uint32_t capacity,
                              uint32_t count, struct dayan_tcsr_range *range)
{
    uint32_t base;

    range->base = 0;
    range->count = 0;
    if (!count)
        return 0;
    if (!tcsr_range_available(*used, capacity, count))
        return -ENOSPC;
    for (base = 0; base <= capacity - count; ++base) {
        uint64_t mask = ((UINT64_C(1) << count) - 1U) << base;

        if (*used & mask)
            continue;
        *used |= mask;
        range->base = (uint8_t)base;
        range->count = (uint8_t)count;
        return 0;
    }
    return -ENOSPC;
}

static void tcsr_range_release(uint64_t *used,
                               struct dayan_tcsr_range range)
{
    uint64_t mask;

    if (!range.count)
        return;
    mask = ((UINT64_C(1) << range.count) - 1U) << range.base;
    *used &= ~mask;
}

static int slot_tcsr_accepts(const dayan_server_t *server,
                             const struct server_core_slot *slot,
                             const npu_package_kernel_t *metadata)
{
    return tcsr_range_available(slot->lsu_tcsr_used,
                                server->options.lsu_tcsr_count_per_hart,
                                metadata->lsu_tcsr_count) &&
           tcsr_range_available(slot->mat_tcsr_used,
                                server->options.mat_tcsr_count_per_hart,
                                metadata->mat_tcsr_count) &&
           tcsr_range_available(slot->vec_tcsr_used,
                                server->options.vec_tcsr_count_per_hart,
                                metadata->vec_tcsr_count) &&
           tcsr_range_available(slot->macro_tcsr_used,
                                server->options.macro_tcsr_count_per_hart,
                                metadata->macro_tcsr_count);
}

static int reserve_tcsr_ranges(dayan_server_t *server,
                               struct server_core_slot *slot,
                               const npu_package_kernel_t *metadata,
                               struct server_proc_resources *resources)
{
    int status;

    memset(resources, 0, sizeof(*resources));
    status = tcsr_range_reserve(&slot->lsu_tcsr_used,
                                server->options.lsu_tcsr_count_per_hart,
                                metadata->lsu_tcsr_count, &resources->lsu);
    if (status == 0)
        status = tcsr_range_reserve(&slot->mat_tcsr_used,
                                    server->options.mat_tcsr_count_per_hart,
                                    metadata->mat_tcsr_count,
                                    &resources->mat);
    if (status == 0)
        status = tcsr_range_reserve(&slot->vec_tcsr_used,
                                    server->options.vec_tcsr_count_per_hart,
                                    metadata->vec_tcsr_count,
                                    &resources->vec);
    if (status == 0)
        status = tcsr_range_reserve(&slot->macro_tcsr_used,
                                    server->options.macro_tcsr_count_per_hart,
                                    metadata->macro_tcsr_count,
                                    &resources->macro);
    if (status < 0) {
        tcsr_range_release(&slot->lsu_tcsr_used, resources->lsu);
        tcsr_range_release(&slot->mat_tcsr_used, resources->mat);
        tcsr_range_release(&slot->vec_tcsr_used, resources->vec);
        tcsr_range_release(&slot->macro_tcsr_used, resources->macro);
        memset(resources, 0, sizeof(*resources));
    }
    return status;
}

static void release_tcsr_ranges(struct server_core_slot *slot,
                                const struct server_proc_resources *resources)
{
    if (!resources)
        return;
    tcsr_range_release(&slot->lsu_tcsr_used, resources->lsu);
    tcsr_range_release(&slot->mat_tcsr_used, resources->mat);
    tcsr_range_release(&slot->vec_tcsr_used, resources->vec);
    tcsr_range_release(&slot->macro_tcsr_used, resources->macro);
}

static int slot_accepts(const dayan_server_t *server,
                        const struct server_command *command,
                        size_t slot_index)
{
    const struct server_core_slot *slot = &server->slots[slot_index];
    const struct server_kernel *kernel = command->data.launch.kernel;
    const npu_package_kernel_t *metadata = kernel->metadata;
    uint64_t resident = launch_resident_size(metadata);
    uint64_t atomic_cache = 0;
    uint32_t procs = launch_reserved_procs(metadata);
    size_t index;

    if (calculate_shared_size(command, &atomic_cache) < 0)
        return 0;
    if (!slot->enabled || resident == UINT64_MAX || slot->locked_task ||
        (launch_locks_task(metadata) && slot->active_tasks) ||
        kernel->active_per_core[slot_index] >=
            metadata->max_tasks_per_core)
        return 0;
    if (server->options.resident_limit_per_core &&
        (slot->resident_memory > server->options.resident_limit_per_core ||
         resident > server->options.resident_limit_per_core -
                        slot->resident_memory))
        return 0;
    if (slot->reserved_procs > server->options.max_procs_per_hart ||
        procs > server->options.max_procs_per_hart - slot->reserved_procs)
        return 0;
    if (slot->atomic_cache_memory > slot->atomic_cache_capacity ||
        atomic_cache > slot->atomic_cache_capacity -
                           slot->atomic_cache_memory)
        return 0;
    if (slot->uram_memory > slot->uram_capacity ||
        metadata->uram_size > slot->uram_capacity - slot->uram_memory)
        return 0;
    if (!slot_tcsr_accepts(server, slot, metadata))
        return 0;
    for (index = 0; index < command->data.launch.argument_count; ++index) {
        const struct server_buffer *buffer =
            command->data.launch.buffers[index];

        if (buffer && !npu_hal_memory_visible(
                          server->hal,
                          buffer->allocation, slot->target))
            return 0;
    }
    return 1;
}

static int choose_single_slot_locked(dayan_server_t *server,
                                     struct server_command *command,
                                     size_t *slot_out)
{
    size_t offset;

    for (offset = 0; offset < server->slot_count; ++offset) {
        size_t slot = (server->next_slot + offset) % server->slot_count;

        if (!slot_accepts(server, command, slot))
            continue;
        server->next_slot = (slot + 1U) % server->slot_count;
        *slot_out = slot;
        return 1;
    }
    return 0;
}

static size_t slot_index(const dayan_server_t *server, uint32_t cluster,
                         uint32_t core)
{
    return (size_t)cluster * server->hal_info.cores_per_cluster + core;
}

static int choose_cluster_locked(dayan_server_t *server,
                                 struct server_command *command,
                                 size_t selected[4])
{
    uint32_t offset;
    uint32_t first_cluster = server->slot_count ?
        (uint32_t)(server->next_slot /
                   server->hal_info.cores_per_cluster) : 0;

    if (server->hal_info.cores_per_cluster != 4U)
        return 0;
    for (offset = 0; offset < server->hal_info.cluster_count; ++offset) {
        uint32_t cluster = (first_cluster + offset) %
                           server->hal_info.cluster_count;
        uint32_t core;
        int accepted = 1;

        for (core = 0; core < 4U; ++core) {
            selected[core] = slot_index(server, cluster, core);
            if (!slot_accepts(server, command, selected[core])) {
                accepted = 0;
                break;
            }
        }
        if (accepted) {
            server->next_slot = slot_index(
                server, (cluster + 1U) % server->hal_info.cluster_count, 0);
            return 1;
        }
    }
    return 0;
}

static int reserve_slots_locked(dayan_server_t *server,
                                struct server_command *command,
                                const size_t *slots, size_t count,
                                struct server_proc_resources *resources)
{
    uint64_t resident = launch_resident_size(
        command->data.launch.kernel->metadata);
    uint64_t atomic_cache = 0;
    const npu_package_kernel_t *metadata =
        command->data.launch.kernel->metadata;
    size_t index;

    if (calculate_shared_size(command, &atomic_cache) < 0)
        return -EOVERFLOW;
    for (index = 0; index < count; ++index) {
        int status = reserve_tcsr_ranges(
            server, &server->slots[slots[index]], metadata,
            &resources[index]);

        if (status < 0) {
            while (index) {
                --index;
                release_tcsr_ranges(&server->slots[slots[index]],
                                    &resources[index]);
            }
            return status;
        }
    }
    for (index = 0; index < count; ++index) {
        if (launch_locks_task(metadata))
            server->slots[slots[index]].locked_task = 1;
        ++server->slots[slots[index]].active_tasks;
        server->slots[slots[index]].resident_memory += resident;
        server->slots[slots[index]].reserved_procs +=
            launch_reserved_procs(metadata);
        server->slots[slots[index]].atomic_cache_memory += atomic_cache;
        server->slots[slots[index]].uram_memory += metadata->uram_size;
        ++command->data.launch.kernel->active_per_core[slots[index]];
    }
    return 0;
}

static void release_slot_locked(dayan_server_t *server,
                                struct server_command *command,
                                size_t slot_index_value,
                                const struct server_proc_resources *resources)
{
    uint64_t resident = launch_resident_size(
        command->data.launch.kernel->metadata);
    uint64_t atomic_cache = 0;
    const npu_package_kernel_t *metadata =
        command->data.launch.kernel->metadata;
    struct server_core_slot *slot = &server->slots[slot_index_value];

    release_tcsr_ranges(slot, resources);
    (void)calculate_shared_size(command, &atomic_cache);
    if (slot->active_tasks)
        --slot->active_tasks;
    if (!slot->active_tasks)
        slot->locked_task = 0;
    if (slot->resident_memory >= resident)
        slot->resident_memory -= resident;
    else
        slot->resident_memory = 0;
    if (slot->reserved_procs >= launch_reserved_procs(metadata))
        slot->reserved_procs -= launch_reserved_procs(metadata);
    else
        slot->reserved_procs = 0;
    if (slot->atomic_cache_memory >= atomic_cache)
        slot->atomic_cache_memory -= atomic_cache;
    else
        slot->atomic_cache_memory = 0;
    if (slot->uram_memory >= metadata->uram_size)
        slot->uram_memory -= metadata->uram_size;
    else
        slot->uram_memory = 0;
    if (command->data.launch.kernel->active_per_core[slot_index_value])
        --command->data.launch.kernel->active_per_core[slot_index_value];
}

static int calculate_shared_size(const struct server_command *command,
                                 uint64_t *size_out)
{
    const struct server_kernel *kernel = command->data.launch.kernel;
    const npu_package_kernel_t *metadata = kernel->metadata;
    const struct npu_package_view *view = &kernel->module->package->view;
    uint64_t resource_count = 0;
    uint64_t resource_size;
    uint64_t size = metadata->atomic_cache_size;
    uint32_t index;

    for (index = 0; index < metadata->sync_count; ++index) {
        const npu_package_sync_t *sync =
            &view->sync_resources[metadata->first_sync + index];

        if (sync->count > UINT64_MAX - resource_count)
            return -EOVERFLOW;
        resource_count += sync->count;
    }
    if (resource_count > UINT64_MAX / SERVER_SHARED_RESOURCE_SIZE)
        return -EOVERFLOW;
    resource_size = resource_count * SERVER_SHARED_RESOURCE_SIZE;
    if (size < resource_size)
        size = resource_size;
    if (size > UINT64_MAX - (SERVER_SHARED_ALIGNMENT - 1U))
        return -EOVERFLOW;
    *size_out = (size + SERVER_SHARED_ALIGNMENT - 1U) &
                ~(SERVER_SHARED_ALIGNMENT - 1U);
    return 0;
}

static int empty_slot_accepts(const dayan_server_t *server,
                              const struct server_command *command,
                              size_t slot_index, uint64_t resident,
                              uint64_t atomic_cache)
{
    const struct server_core_slot *slot = &server->slots[slot_index];
    const npu_package_kernel_t *metadata =
        command->data.launch.kernel->metadata;
    size_t index;

    if (!slot->enabled ||
        (server->options.resident_limit_per_core &&
         resident > server->options.resident_limit_per_core) ||
        launch_reserved_procs(metadata) >
            server->options.max_procs_per_hart ||
        atomic_cache > slot->atomic_cache_capacity ||
        metadata->uram_size > slot->uram_capacity)
        return 0;
    if (!slot_tcsr_accepts(server, slot, metadata))
        return 0;
    for (index = 0; index < command->data.launch.argument_count; ++index) {
        const struct server_buffer *buffer =
            command->data.launch.buffers[index];

        if (buffer && !npu_hal_memory_visible(server->hal,
                                              buffer->allocation,
                                              slot->target))
            return 0;
    }
    return 1;
}

int server_launch_resources_valid(const dayan_server_t *server,
                                  const struct server_command *command)
{
    uint64_t resident;
    uint64_t atomic_cache;
    uint32_t mode;
    size_t index;
    int status;

    if (!server || !command || command->kind != SERVER_COMMAND_LAUNCH)
        return -EINVAL;
    resident = launch_resident_size(command->data.launch.kernel->metadata);
    if (resident == UINT64_MAX)
        return -EOVERFLOW;
    status = calculate_shared_size(command, &atomic_cache);
    if (status < 0)
        return status;
    mode = (uint32_t)command->data.launch.config.task_mode;
    if (mode == 1U) {
        for (index = 0; index < server->slot_count; ++index) {
            if (empty_slot_accepts(server, command, index, resident,
                                   atomic_cache))
                return 0;
        }
    } else if (mode == 4U && server->hal_info.cores_per_cluster == 4U) {
        uint32_t cluster;

        for (cluster = 0; cluster < server->hal_info.cluster_count;
             ++cluster) {
            uint32_t core;

            for (core = 0; core < 4U; ++core) {
                index = slot_index(server, cluster, core);
                if (!empty_slot_accepts(server, command, index, resident,
                                        atomic_cache))
                    break;
            }
            if (core == 4U)
                return 0;
        }
    } else {
        return -EINVAL;
    }
    return -ENOSPC;
}

static int allocation_request_for_group(
    dayan_server_t *server, struct server_command *command,
    const size_t *slots, size_t count, uint64_t size,
    npu_hal_allocation_t **allocation_out)
{
    npu_hal_memory_request_t request;
    npu_hal_memory_scope_t scopes[3];
    size_t scope_count;
    size_t index;
    int status = -ENOMEM;

    if (!size) {
        *allocation_out = 0;
        return 0;
    }
    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_COHERENT;
    request.usage = NPU_HAL_MEMORY_SHARED;
    request.owner = command->context->owner;
    request.size = size;
    request.alignment = SERVER_SHARED_ALIGNMENT;
    if (count == 4) {
        scopes[0] = NPU_HAL_MEMORY_CLUSTER;
        scopes[1] = NPU_HAL_MEMORY_DEVICE;
        scope_count = 2;
    } else {
        scopes[0] = NPU_HAL_MEMORY_CORE;
        scopes[1] = NPU_HAL_MEMORY_CLUSTER;
        scopes[2] = NPU_HAL_MEMORY_DEVICE;
        scope_count = 3;
    }
    for (index = 0; index < scope_count; ++index) {
        size_t viewer_index;

        request.scope = scopes[index];
        request.location.cluster = scopes[index] == NPU_HAL_MEMORY_DEVICE ?
            NPU_HAL_ANY_CLUSTER : server->slots[slots[0]].target.cluster;
        request.location.core = scopes[index] == NPU_HAL_MEMORY_CORE ?
            server->slots[slots[0]].target.core : NPU_HAL_ANY_CORE;
        status = npu_hal_memory_alloc(server->hal,
                                      &request, allocation_out);
        if (status < 0) {
            if (status != -ENOMEM)
                return status;
            continue;
        }
        for (viewer_index = 0; viewer_index < count; ++viewer_index) {
            if (!npu_hal_memory_visible(
                    server->hal, *allocation_out,
                    server->slots[slots[viewer_index]].target))
                break;
        }
        if (viewer_index == count)
            return 0;
        (void)npu_hal_memory_free(server->hal,
                                  *allocation_out);
        *allocation_out = 0;
    }
    return status;
}

static int initialize_shared(dayan_server_t *server,
                             struct server_command *command,
                             const size_t *slots, size_t count,
                             npu_hal_allocation_t **allocation_out,
                             uint64_t *size_out)
{
    const struct server_kernel *kernel = command->data.launch.kernel;
    const npu_package_kernel_t *metadata = kernel->metadata;
    const struct npu_package_view *view = &kernel->module->package->view;
    unsigned char *image;
    uint64_t shared_size;
    size_t resource_offset = 0;
    uint32_t index;
    int status;

    *allocation_out = 0;
    *size_out = 0;
    status = calculate_shared_size(command, &shared_size);
    if (status < 0 || !shared_size)
        return status;
    if (shared_size > SIZE_MAX)
        return -EOVERFLOW;
    status = allocation_request_for_group(server, command, slots, count,
                                          shared_size, allocation_out);
    if (status < 0)
        return status;
    image = calloc(1, (size_t)shared_size);
    if (!image) {
        status = -ENOMEM;
        goto fail;
    }
    for (index = 0; index < metadata->sync_count; ++index) {
        const npu_package_sync_t *sync =
            &view->sync_resources[metadata->first_sync + index];
        uint32_t instance;

        for (instance = 0; instance < sync->count; ++instance) {
            uint64_t counts =
                ((uint64_t)sync->producer_count << 32) |
                sync->consumer_count;

            memcpy(image + resource_offset, &counts, sizeof(counts));
            resource_offset += SERVER_SHARED_RESOURCE_SIZE;
        }
    }
    status = npu_hal_memory_write(
        server->hal, *allocation_out,
        server->slots[slots[0]].target, 0, image, (size_t)shared_size);
    free(image);
    if (status < 0)
        goto fail;
    *size_out = shared_size;
    return 0;

fail:
    if (*allocation_out) {
        (void)npu_hal_memory_free(server->hal,
                                  *allocation_out);
        *allocation_out = 0;
    }
    return status;
}

static int allocate_uram(dayan_server_t *server,
                         struct server_command *command,
                         npu_hal_core_t target, uint64_t size,
                         npu_hal_allocation_t **allocation_out)
{
    npu_hal_memory_request_t request;

    *allocation_out = 0;
    if (!size)
        return 0;
    if (size & (SERVER_URAM_ALIGNMENT - 1U))
        return -EINVAL;
    npu_hal_memory_request_init(&request);
    request.memory_class = NPU_HAL_MEMORY_URAM;
    request.scope = NPU_HAL_MEMORY_CORE;
    request.usage = NPU_HAL_MEMORY_KERNEL_URAM;
    request.owner = command->context->owner;
    request.location = target;
    request.size = size;
    request.alignment = SERVER_URAM_ALIGNMENT;
    return npu_hal_memory_alloc(server->hal, &request, allocation_out);
}

static int build_tensor_context(dayan_server_t *server,
                                npu_hal_core_t target,
                                npu_hal_allocation_t *uram_allocation,
                                uint64_t uram_size,
                                struct server_proc_resources *resources)
{
    struct dayan_context_fields fields;
    uint64_t address = 0;
    uint64_t offset = 0;
    int status;

    memset(&fields, 0, sizeof(fields));
    if (uram_size) {
        status = npu_hal_memory_address(server->hal, uram_allocation, target,
                                        &address);
        if (status < 0)
            return status;
        offset = address & (DAYAN_CONTEXT_URAM_WINDOW - 1U);
        if ((address & (DAYAN_CONTEXT_URAM_GRANULE - 1U)) ||
            (uram_size & (DAYAN_CONTEXT_URAM_GRANULE - 1U)) ||
            offset > DAYAN_CONTEXT_URAM_WINDOW ||
            uram_size > DAYAN_CONTEXT_URAM_WINDOW - offset)
            return -EINVAL;
        fields.uram_lo = (uint8_t)(offset / DAYAN_CONTEXT_URAM_GRANULE);
        fields.uram_hi = (uint8_t)((offset + uram_size) /
                                   DAYAN_CONTEXT_URAM_GRANULE);
    }
    fields.lsu = resources->lsu;
    fields.mat = resources->mat;
    fields.vec = resources->vec;
    fields.macro = resources->macro;
    resources->tensor_context = dayan_context_pack(&fields);
    return resources->tensor_context == UINT64_MAX ? -EINVAL : 0;
}

static void unit_task_ids(const struct server_command *command, uint64_t unit,
                          uint32_t lane, uint32_t task_id[3])
{
    const NPULaunchConfig *config = &command->data.launch.config;

    if ((uint32_t)config->task_mode == 4U) {
        task_id[0] = lane;
        task_id[1] = (uint32_t)(unit % config->task_dim.y);
        task_id[2] = (uint32_t)(unit / config->task_dim.y);
    } else {
        task_id[0] = (uint32_t)(unit % config->task_dim.x);
        unit /= config->task_dim.x;
        task_id[1] = (uint32_t)(unit % config->task_dim.y);
        task_id[2] = (uint32_t)(unit / config->task_dim.y);
    }
}

static int fill_launch_arguments(dayan_server_t *server,
                                 struct server_command *command,
                                 npu_hal_core_t target,
                                 npu_service_launch_desc_t *desc)
{
    size_t index;

    for (index = 0; index < command->data.launch.argument_count; ++index) {
        const npu_ipc_launch_argument_t *argument =
            &command->data.launch.arguments[index];

        if (argument->kind == NPU_KERNEL_ARG_VALUE) {
            desc->args[index] = argument->value_or_buffer;
        } else {
            struct server_buffer *buffer =
                command->data.launch.buffers[index];
            uint64_t address;
            /*
             * Visibility is still decided against the allocation, so this
             * call runs either way; only the address handed to the Device
             * changes.  A nonzero va means the ClusterMMU is on and the
             * Device must be given the virtual address instead.
             */
            int status = npu_hal_memory_address(
                server->hal, buffer->allocation, target,
                &address);

            if (status < 0)
                return status;
            if (buffer->va)
                address = buffer->va;
            if (address > UINT64_MAX - argument->offset)
                return -EOVERFLOW;
            desc->args[index] = address + argument->offset;
        }
    }
    return 0;
}

struct dispatch_result {
    struct server_launch_group *group;
    struct server_invocation *invocations;
    size_t launched;
    int status;
};

static struct dispatch_result dispatch_group(
    dayan_server_t *server, struct server_command *command,
    const size_t *slots, const struct server_proc_resources *resources,
    size_t count, uint64_t unit)
{
    const struct npu_package_kernel *metadata =
        command->data.launch.kernel->metadata;
    const struct npu_package_view *view =
        &command->data.launch.kernel->module->package->view;
    const char *entry_symbol = npu_package_string(view,
                                                   metadata->entry_offset);
    struct dispatch_result result = { 0, 0, 0, 0 };
    struct server_launch_group *group;
    uint64_t shared_size = 0;
    size_t lane;

    group = calloc(1, sizeof(*group));
    if (!group) {
        result.status = -ENOMEM;
        return result;
    }
    result.group = group;
    result.status = initialize_shared(server, command, slots, count,
                                      &group->shared_allocation,
                                      &shared_size);
    if (result.status < 0)
        return result;
    for (lane = 0; lane < count; ++lane) {
        struct server_invocation *invocation;
        npu_service_launch_desc_t desc;
        npu_hal_core_t target = server->slots[slots[lane]].target;
        npu_hal_allocation_t *uram_allocation = 0;
        uint64_t entry;
        int status;

        invocation = calloc(1, sizeof(*invocation));
        if (!invocation) {
            result.status = -ENOMEM;
            break;
        }
        status = allocate_uram(server, command, target, metadata->uram_size,
                               &uram_allocation);
        if (status < 0) {
            free(invocation);
            result.status = status;
            break;
        }
        invocation->resources = resources[lane];
        status = build_tensor_context(server, target, uram_allocation,
                                      metadata->uram_size,
                                      &invocation->resources);
        if (status < 0) {
            if (uram_allocation)
                (void)npu_hal_memory_free(server->hal, uram_allocation);
            free(invocation);
            result.status = status;
            break;
        }
        status = npu_service_kernel_resolve(
            server->device, target, view->object,
            (size_t)view->header->object_size, entry_symbol, &entry);
        if (status < 0) {
            if (uram_allocation)
                (void)npu_hal_memory_free(server->hal, uram_allocation);
            free(invocation);
            result.status = status;
            break;
        }
        npu_service_launch_desc_init(&desc);
        desc.target = target;
        desc.entry = entry;
        desc.argc = command->data.launch.argument_count;
        desc.stack_size = metadata->stack_size ? metadata->stack_size :
                                                 SERVER_DEFAULT_STACK_SIZE;
        desc.printf_buffer_size = metadata->printf_buffer_size ?
            metadata->printf_buffer_size : SERVER_DEFAULT_PRINTF_SIZE;
        desc.shared_allocation = group->shared_allocation;
        desc.shared_size = shared_size;
        desc.uram_allocation = uram_allocation;
        desc.uram_size = metadata->uram_size;
        desc.max_child_threads = metadata->max_child_threads;
        desc.tensor_context = invocation->resources.tensor_context;
        desc.name = command->data.launch.task_name;
        desc.task_dim[0] = command->data.launch.config.task_dim.x;
        desc.task_dim[1] = command->data.launch.config.task_dim.y;
        desc.task_dim[2] = command->data.launch.config.task_dim.z;
        desc.task_mode = (uint32_t)command->data.launch.config.task_mode;
        unit_task_ids(command, unit, (uint32_t)lane, desc.task_id);
        desc.cookie = command->data.launch.cookie + unit * count + lane;
        status = fill_launch_arguments(server, command, target, &desc);
        if (status == 0)
            status = npu_service_launch(server->device, &desc,
                                        &invocation->launch);
        if (status < 0) {
            fprintf(stderr,
                    "dayan_server: launch failed: task='%s' hart=%u "
                    "task_id=[%u,%u,%u] unit=%llu status=%d\n",
                    command->data.launch.task_name, target.core,
                    desc.task_id[0], desc.task_id[1], desc.task_id[2],
                    (unsigned long long)unit, status);
            if (uram_allocation)
                (void)npu_hal_memory_free(server->hal, uram_allocation);
            free(invocation);
            result.status = status;
            break;
        }
        invocation->slot_index = slots[lane];
        invocation->uram_allocation = uram_allocation;
        invocation->resident = launch_resident_size(metadata);
        invocation->cookie = desc.cookie;
        memcpy(invocation->task_id, desc.task_id,
               sizeof(invocation->task_id));
        invocation->group = group;
        invocation->next = result.invocations;
        result.invocations = invocation;
        ++result.launched;
        ++group->active;
    }
    return result;
}

static int find_transfer_viewer(dayan_server_t *server,
                                struct server_command *command,
                                npu_hal_core_t *viewer_out)
{
    size_t index;

    for (index = 0; index < server->slot_count; ++index) {
        npu_hal_core_t viewer = server->slots[index].target;

        if (!server->slots[index].enabled)
            continue;
        if (command->data.transfer.destination &&
            !npu_hal_memory_visible(server->hal,
                                    command->data.transfer.destination->allocation,
                                    viewer))
            continue;
        if (command->data.transfer.source &&
            !npu_hal_memory_visible(server->hal,
                                    command->data.transfer.source->allocation,
                                    viewer))
            continue;
        *viewer_out = viewer;
        return 0;
    }
    return -EXDEV;
}

static struct server_command *find_simple_action_locked(dayan_server_t *server)
{
    struct server_session *session;

    for (session = server->sessions; session; session = session->next) {
        struct server_context *context;

        for (context = session->contexts; context; context = context->next) {
            struct server_stream *stream;

            for (stream = context->streams; stream; stream = stream->next) {
                struct server_command *command = stream->head;

                if (!command || command->dispatching)
                    continue;
                if (command->cancelled) {
                    if (command->kind == SERVER_COMMAND_LAUNCH &&
                        command->data.launch.invocations)
                        continue;
                    if ((command->kind == SERVER_COMMAND_H2D ||
                         command->kind == SERVER_COMMAND_D2H) &&
                        command->data.transfer.granted &&
                        !command->data.transfer.completed)
                        continue;
                    return command;
                }
                switch (command->kind) {
                case SERVER_COMMAND_EVENT:
                    return command;
                case SERVER_COMMAND_WAIT:
                    if (event_terminal(command->data.wait.dependency))
                        return command;
                    break;
                case SERVER_COMMAND_H2D:
                case SERVER_COMMAND_D2H:
                    if (!command->data.transfer.token_ready ||
                        command->data.transfer.completed)
                        return command;
                    break;
                case SERVER_COMMAND_D2D:
                    if (!command->running)
                        return command;
                    break;
                case SERVER_COMMAND_LAUNCH:
                    break;
                }
            }
        }
    }
    return 0;
}

static int handle_simple_action(dayan_server_t *server,
                                struct server_command *command)
{
    int status = 0;

    if (command->cancelled) {
        server_complete_command_locked(server, command, -ECANCELED);
        return 1;
    }
    switch (command->kind) {
    case SERVER_COMMAND_EVENT:
    case SERVER_COMMAND_WAIT:
        server_complete_command_locked(server, command, 0);
        return 1;
    case SERVER_COMMAND_H2D:
    case SERVER_COMMAND_D2H:
        if (command->data.transfer.completed) {
            server_complete_command_locked(
                server, command, command->data.transfer.transfer_status);
            return 1;
        }
        if (!command->data.transfer.token_ready) {
            struct server_buffer *buffer =
                command->kind == SERVER_COMMAND_H2D ?
                    command->data.transfer.destination :
                    command->data.transfer.source;
            uint64_t offset = command->kind == SERVER_COMMAND_H2D ?
                command->data.transfer.destination_offset :
                command->data.transfer.source_offset;
            npu_hal_data_direction_t direction =
                command->kind == SERVER_COMMAND_H2D ?
                    NPU_HAL_DATA_HOST_TO_DEVICE :
                    NPU_HAL_DATA_DEVICE_TO_HOST;

            status = find_transfer_viewer(server, command,
                                           &command->data.transfer.viewer);
            if (status == 0)
                status = npu_hal_transfer_prepare(
                    server->hal, buffer->allocation,
                    command->data.transfer.viewer, offset,
                    (size_t)command->data.transfer.size, direction,
                    &command->data.transfer.token);
            if (status < 0) {
                server_complete_command_locked(server, command, status);
            } else {
                command->data.transfer.token_ready = 1;
                command->running = 1;
                command->event->state = NPU_EVENT_RUNNING;
                pthread_cond_broadcast(&server->condition);
            }
            return 1;
        }
        return 0;
    case SERVER_COMMAND_D2D: {
        uint64_t destination;
        uint64_t source;

        status = find_transfer_viewer(server, command,
                                       &command->data.transfer.viewer);
        if (status == 0)
            status = npu_hal_memory_address(
                server->hal,
                command->data.transfer.destination->allocation,
                command->data.transfer.viewer, &destination);
        if (status == 0)
            status = npu_hal_memory_address(
                server->hal,
                command->data.transfer.source->allocation,
                command->data.transfer.viewer, &source);
        if (status < 0) {
            server_complete_command_locked(server, command, status);
            return 1;
        }
        destination += command->data.transfer.destination_offset;
        source += command->data.transfer.source_offset;
        command->running = 1;
        command->dispatching = 1;
        command->event->state = NPU_EVENT_RUNNING;
        pthread_mutex_unlock(&server->mutex);
        status = npu_hal_copy_address(
            server->hal,
            command->data.transfer.viewer.cluster, destination, source,
            (size_t)command->data.transfer.size);
        pthread_mutex_lock(&server->mutex);
        command->dispatching = 0;
        server_complete_command_locked(server, command,
                                       command->cancelled ? -ECANCELED :
                                                            status);
        return 1;
    }
    case SERVER_COMMAND_LAUNCH:
        return 0;
    }
    return 0;
}

static struct server_command *find_launch_to_dispatch_locked(
    dayan_server_t *server, size_t selected[4],
    struct server_proc_resources resources[4], size_t *count_out,
    uint64_t *unit_out)
{
    struct server_session *session;

    for (session = server->sessions; session; session = session->next) {
        struct server_context *context;

        for (context = session->contexts; context; context = context->next) {
            struct server_stream *stream;

            for (stream = context->streams; stream; stream = stream->next) {
                struct server_command *command = stream->head;
                uint32_t mode;

                if (!command || command->kind != SERVER_COMMAND_LAUNCH ||
                    command->dispatching || command->cancelled ||
                    command->data.launch.aggregate_status < 0 ||
                    command->data.launch.next_unit >=
                        command->data.launch.total_units)
                    continue;
                mode = (uint32_t)command->data.launch.config.task_mode;
                if (mode == 1U) {
                    if (!choose_single_slot_locked(server, command,
                                                   &selected[0]))
                        continue;
                    *count_out = 1;
                } else {
                    if (!choose_cluster_locked(server, command, selected))
                        continue;
                    *count_out = 4;
                }
                *unit_out = command->data.launch.next_unit++;
                if (reserve_slots_locked(server, command, selected,
                                         *count_out, resources) < 0)
                    continue;
                command->dispatching = 1;
                command->running = 1;
                command->event->state = NPU_EVENT_RUNNING;
                return command;
            }
        }
    }
    return 0;
}

static void merge_dispatch_result_locked(
    dayan_server_t *server, struct server_command *command,
    const size_t *selected, const struct server_proc_resources *resources,
    size_t count, struct dispatch_result result)
{
    struct server_invocation *invocation;
    size_t launched_by_slot[4] = { 0, 0, 0, 0 };
    size_t index;

    for (invocation = result.invocations; invocation;
         invocation = invocation->next) {
        for (index = 0; index < count; ++index) {
            if (invocation->slot_index == selected[index])
                launched_by_slot[index] = 1;
        }
    }
    for (index = 0; index < count; ++index) {
        if (!launched_by_slot[index])
            release_slot_locked(server, command, selected[index],
                                &resources[index]);
    }
    if (result.launched) {
        struct server_invocation *tail = result.invocations;

        while (tail->next)
            tail = tail->next;
        tail->next = command->data.launch.invocations;
        command->data.launch.invocations = result.invocations;
        result.group->next = command->data.launch.groups;
        command->data.launch.groups = result.group;
        server_route_pending_outputs_locked(server);
    } else {
        if (result.group && result.group->shared_allocation)
            (void)npu_hal_memory_free(server->hal,
                                      result.group->shared_allocation);
        free(result.group);
    }
    if (result.status < 0 && !command->data.launch.aggregate_status) {
        command->data.launch.aggregate_status = result.status;
        command->data.launch.next_unit = command->data.launch.total_units;
    }
    command->dispatching = 0;
    if (!command->data.launch.invocations &&
        command->data.launch.next_unit == command->data.launch.total_units)
        server_complete_command_locked(
            server, command,
            command->cancelled ? -ECANCELED :
                command->data.launch.aggregate_status);
    pthread_cond_broadcast(&server->condition);
}

static struct server_command *find_invocation_to_poll_locked(
    dayan_server_t *server, struct server_invocation **invocation_out)
{
    struct server_session *session;

    for (session = server->sessions; session; session = session->next) {
        struct server_context *context;

        for (context = session->contexts; context; context = context->next) {
            struct server_stream *stream;

            for (stream = context->streams; stream; stream = stream->next) {
                struct server_command *command = stream->head;
                struct server_invocation *invocation;

                if (!command || command->kind != SERVER_COMMAND_LAUNCH ||
                    command->dispatching)
                    continue;
                for (invocation = command->data.launch.invocations;
                     invocation; invocation = invocation->next) {
                    if (!invocation->polling) {
                        invocation->polling = 1;
                        command->dispatching = 1;
                        *invocation_out = invocation;
                        return command;
                    }
                }
            }
        }
    }
    return 0;
}

static void remove_group_locked(dayan_server_t *server,
                                struct server_command *command,
                                struct server_launch_group *group)
{
    struct server_launch_group **link = &command->data.launch.groups;

    while (*link && *link != group)
        link = &(*link)->next;
    if (*link)
        *link = group->next;
    if (group->shared_allocation)
        (void)npu_hal_memory_free(server->hal,
                                  group->shared_allocation);
    free(group);
}

static int poll_invocation(dayan_server_t *server,
                           struct server_command *command,
                           struct server_invocation *invocation)
{
    npu_service_launch_info_t info;
    npu_service_kill_result_t kill_result;
    npu_hal_core_t target = server->slots[invocation->slot_index].target;
    uint32_t root_proc_id = invocation->proc_id;
    int64_t trap_status = invocation->trap_status;
    int kill_attempted = trap_status < 0 && !invocation->trap_handled &&
                         root_proc_id;
    int terminal = 0;
    int64_t completion = 0;
    int status;

    memset(&info, 0, sizeof(info));
    memset(&kill_result, 0, sizeof(kill_result));
    pthread_mutex_unlock(&server->mutex);
    if (kill_attempted) {
        status = npu_service_rpc_kill_family_timeout(
            server->device, target, root_proc_id, trap_status,
            SERVER_TRAP_KILL_TIMEOUT_MS,
            &kill_result);
        if (status < 0 && status != -ESRCH) {
            int abandon_status;

            fprintf(stderr,
                    "dayan_server: failed to terminate trapped kernel "
                    "cluster=%u core=%u root_proc=%u status=%d; "
                    "failing Host launch\n",
                    target.cluster, target.core, root_proc_id,
                    status);
            terminal = 1;
            completion = trap_status;
            abandon_status = npu_service_launch_abandon(invocation->launch);
            if (abandon_status < 0)
                completion = abandon_status;
            status = abandon_status;
        }
    }
    if (!terminal)
        status = npu_service_launch_query(invocation->launch, &info);
    if (status == 0 &&
        (info.state == NPU_SERVICE_LAUNCH_EXITED ||
         info.state == NPU_SERVICE_LAUNCH_FAILED)) {
        terminal = 1;
        completion = info.status;
        status = npu_service_launch_release(invocation->launch);
        if (status < 0)
            completion = status;
    } else if (status < 0 && npu_service_status(server->device) < 0) {
        int release_status;

        terminal = 1;
        completion = status;
        release_status = npu_service_launch_abandon(invocation->launch);
        if (release_status < 0)
            completion = release_status;
    }
    pthread_mutex_lock(&server->mutex);
    command->dispatching = 0;
    invocation->polling = 0;
    if (kill_attempted)
        invocation->trap_handled = 1;
    if (status == 0 && info.proc_id) {
        invocation->proc_id = info.proc_id;
        server_route_pending_outputs_locked(server);
    }
    if (!terminal) {
        if (status < 0 && !command->data.launch.aggregate_status) {
            command->data.launch.aggregate_status = status;
            command->data.launch.next_unit =
                command->data.launch.total_units;
        }
        return 0;
    }
    {
        struct server_invocation **link =
            &command->data.launch.invocations;
        struct server_launch_group *group = invocation->group;

        while (*link && *link != invocation)
            link = &(*link)->next;
        if (*link)
            *link = invocation->next;
        release_slot_locked(server, command, invocation->slot_index,
                            &invocation->resources);
        if (invocation->uram_allocation)
            (void)npu_hal_memory_free(server->hal,
                                      invocation->uram_allocation);
        if (group->active)
            --group->active;
        free(invocation);
        if (completion < 0 && !command->data.launch.aggregate_status) {
            command->data.launch.aggregate_status = completion;
            command->data.launch.next_unit =
                command->data.launch.total_units;
        }
        if (!group->active)
            remove_group_locked(server, command, group);
    }
    if (!command->data.launch.invocations &&
        (command->cancelled ||
         command->data.launch.next_unit == command->data.launch.total_units))
        server_complete_command_locked(
            server, command,
            command->cancelled ? -ECANCELED :
                command->data.launch.aggregate_status);
    return 1;
}

static int server_has_pending_locked(const dayan_server_t *server)
{
    const struct server_session *session;

    for (session = server->sessions; session; session = session->next) {
        const struct server_context *context;

        for (context = session->contexts; context; context = context->next) {
            if (context->pending_commands)
                return 1;
        }
    }
    return 0;
}

void *dayan_scheduler_main(void *argument)
{
    dayan_server_t *server = argument;

    pthread_mutex_lock(&server->mutex);
    for (;;) {
        struct server_command *command;
        struct server_invocation *invocation;
        size_t selected[4];
        struct server_proc_resources resources[4];
        size_t count;
        uint64_t unit;
        int progressed = 0;

        while ((command = find_simple_action_locked(server)) != 0) {
            if (!handle_simple_action(server, command))
                break;
            progressed = 1;
            if (server->stop && !server_has_pending_locked(server))
                break;
        }
        if (server->stop && !server_has_pending_locked(server))
            break;
        command = server->stop ? 0 : find_launch_to_dispatch_locked(
            server, selected, resources, &count, &unit);
        if (command) {
            struct dispatch_result result;

            pthread_mutex_unlock(&server->mutex);
            result = dispatch_group(server, command, selected, resources,
                                    count, unit);
            pthread_mutex_lock(&server->mutex);
            merge_dispatch_result_locked(server, command, selected, resources,
                                         count, result);
            progressed = 1;
        }
        invocation = 0;
        command = find_invocation_to_poll_locked(server, &invocation);
        if (command)
            progressed |= poll_invocation(server, command, invocation);
        server_collect_contexts_locked(server);
        if (!progressed) {
            struct timespec deadline;

            clock_gettime(CLOCK_REALTIME, &deadline);
            deadline.tv_nsec += 1000000L;
            if (deadline.tv_nsec >= 1000000000L) {
                ++deadline.tv_sec;
                deadline.tv_nsec -= 1000000000L;
            }
            (void)pthread_cond_timedwait(&server->condition, &server->mutex,
                                         &deadline);
        }
    }
    pthread_mutex_unlock(&server->mutex);
    return 0;
}
