#include "runtime_internal.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int npu_module_load(npu_context_t *context, const void *package_image,
                    size_t package_size, npu_module_t **module_out)
{
    npu_ipc_module_load_t *request;
    struct npu_module *module;
    npu_runtime_t *runtime;
    size_t payload_size;
    uint64_t id;
    int status;

    if (!runtime_context_valid(context) || !package_image || !package_size ||
        !module_out || package_size > NPU_IPC_MAX_PAYLOAD - sizeof(*request))
        return -EINVAL;
    *module_out = 0;
    payload_size = sizeof(*request) + package_size;
    request = malloc(payload_size);
    module = calloc(1, sizeof(*module));
    if (!request || !module) {
        free(request);
        free(module);
        return -ENOMEM;
    }
    request->context_id = context->id;
    request->package_size = package_size;
    memcpy(request + 1, package_image, package_size);
    runtime = context->runtime;
    status = runtime_control_call(runtime, NPU_IPC_MODULE_LOAD, 0, request,
                                  payload_size, 0, 0, &id);
    free(request);
    if (status < 0) {
        free(module);
        return status;
    }
    module->context = context;
    module->id = id;
    pthread_mutex_lock(&runtime->mutex);
    if (context->closing) {
        pthread_mutex_unlock(&runtime->mutex);
        (void)runtime_control_call(runtime, NPU_IPC_MODULE_UNLOAD, id,
                                   0, 0, 0, 0, 0);
        free(module);
        return -ECANCELED;
    }
    module->next = context->modules;
    context->modules = module;
    pthread_mutex_unlock(&runtime->mutex);
    *module_out = module;
    return 0;
}

int npu_module_load_file(npu_context_t *context, const char *path,
                         npu_module_t **module_out)
{
    unsigned char *image;
    long length;
    FILE *file;
    int status;

    if (!path || !module_out)
        return -EINVAL;
    file = fopen(path, "rb");
    if (!file)
        return -errno;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) <= 0 || (uint64_t)length > SIZE_MAX ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -EIO;
    }
    image = malloc((size_t)length);
    if (!image) {
        fclose(file);
        return -ENOMEM;
    }
    if (fread(image, 1, (size_t)length, file) != (size_t)length ||
        fclose(file) != 0) {
        free(image);
        return -EIO;
    }
    status = npu_module_load(context, image, (size_t)length, module_out);
    free(image);
    return status;
}

int npu_module_unload(npu_module_t *module)
{
    struct npu_module **link;
    npu_runtime_t *runtime;
    int status;

    if (!module || !runtime_context_valid(module->context) || module->kernels)
        return -EINVAL;
    runtime = module->context->runtime;
    status = runtime_control_call(runtime, NPU_IPC_MODULE_UNLOAD, module->id,
                                  0, 0, 0, 0, 0);
    if (status < 0)
        return status;
    pthread_mutex_lock(&runtime->mutex);
    link = &module->context->modules;
    while (*link && *link != module)
        link = &(*link)->next;
    if (*link)
        *link = module->next;
    pthread_mutex_unlock(&runtime->mutex);
    free(module);
    return 0;
}

int npu_module_get_kernel(npu_module_t *module, const char *name,
                          npu_kernel_t **kernel_out)
{
    npu_ipc_kernel_get_t *request;
    struct npu_kernel *kernel;
    npu_runtime_t *runtime;
    size_t name_size;
    size_t payload_size;
    uint64_t id;
    int status;

    if (!module || !runtime_context_valid(module->context) || !name ||
        !*name || !kernel_out)
        return -EINVAL;
    *kernel_out = 0;
    name_size = strlen(name) + 1U;
    if (name_size > UINT32_MAX ||
        name_size > NPU_IPC_MAX_PAYLOAD - sizeof(*request))
        return -EOVERFLOW;
    payload_size = sizeof(*request) + name_size;
    request = calloc(1, payload_size);
    kernel = calloc(1, sizeof(*kernel));
    if (!request || !kernel) {
        free(request);
        free(kernel);
        return -ENOMEM;
    }
    request->module_id = module->id;
    request->name_size = (uint32_t)name_size;
    memcpy(request + 1, name, name_size);
    runtime = module->context->runtime;
    status = runtime_control_call(runtime, NPU_IPC_KERNEL_GET, 0, request,
                                  payload_size, 0, 0, &id);
    free(request);
    if (status < 0) {
        free(kernel);
        return status;
    }
    kernel->module = module;
    kernel->id = id;
    pthread_mutex_lock(&runtime->mutex);
    if (module->context->closing) {
        pthread_mutex_unlock(&runtime->mutex);
        (void)runtime_control_call(runtime, NPU_IPC_KERNEL_RELEASE, id,
                                   0, 0, 0, 0, 0);
        free(kernel);
        return -ECANCELED;
    }
    kernel->next = module->kernels;
    module->kernels = kernel;
    pthread_mutex_unlock(&runtime->mutex);
    *kernel_out = kernel;
    return 0;
}

int npu_kernel_release(npu_kernel_t *kernel)
{
    struct npu_kernel **link;
    npu_runtime_t *runtime;
    int status;

    if (!kernel || !kernel->module ||
        !runtime_context_valid(kernel->module->context))
        return -EINVAL;
    runtime = kernel->module->context->runtime;
    status = runtime_control_call(runtime, NPU_IPC_KERNEL_RELEASE, kernel->id,
                                  0, 0, 0, 0, 0);
    if (status < 0)
        return status;
    pthread_mutex_lock(&runtime->mutex);
    link = &kernel->module->kernels;
    while (*link && *link != kernel)
        link = &(*link)->next;
    if (*link)
        *link = kernel->next;
    pthread_mutex_unlock(&runtime->mutex);
    free(kernel);
    return 0;
}

int npu_kernel_get_info(npu_kernel_t *kernel, npu_kernel_info_t *info)
{
    npu_ipc_kernel_info_t response;
    int status;

    if (!kernel || !info || !kernel->module ||
        !runtime_context_valid(kernel->module->context))
        return -EINVAL;
    status = runtime_control_call(kernel->module->context->runtime,
                                  NPU_IPC_KERNEL_INFO, kernel->id,
                                  0, 0, &response, sizeof(response), 0);
    if (status < 0)
        return status;
    info->supported_task_modes = response.supported_task_modes;
    info->max_tasks_per_core = response.max_tasks_per_core;
    info->max_child_procs = response.max_child_procs;
    info->arg_count = response.argument_count;
    info->sync_resource_count = response.sync_resource_count;
    info->reserved = 0;
    info->stack_size = response.stack_size;
    info->atomic_cache_size = response.atomic_cache_size;
    info->resident_memory = response.resident_memory;
    info->uram_size = response.uram_size;
    return 0;
}

static int launch_config_valid(const NPULaunchConfig *config)
{
    uint32_t mode;

    if (!config || !config->task_dim.x || !config->task_dim.y ||
        !config->task_dim.z)
        return 0;
    mode = (uint32_t)config->task_mode;
    return (mode == 1U || mode == 4U) &&
           (mode != 4U || config->task_dim.x == 4U);
}

int npu_kernel_launch(npu_stream_t *stream, npu_kernel_t *kernel,
                      const npu_launch_desc_t *desc,
                      npu_event_t **event_out)
{
    npu_launch_desc_t defaults;
    npu_ipc_kernel_launch_t request;
    struct npu_event *event;
    struct npu_context *context;
    uint64_t id;
    size_t index;
    int status;

    if (!runtime_stream_valid(stream) || !kernel || !kernel->module ||
        kernel->module->context != stream->context || !event_out)
        return -EINVAL;
    *event_out = 0;
    if (!desc) {
        npu_launch_desc_init(&defaults);
        desc = &defaults;
    }
    if (desc->version != NPU_LAUNCH_DESC_VERSION || desc->flags ||
        desc->reserved || desc->arg_count > NPU_KERNEL_MAX_ARGS ||
        !launch_config_valid(&desc->config))
        return -EINVAL;
    event = calloc(1, sizeof(*event));
    if (!event)
        return -ENOMEM;
    context = stream->context;
    memset(&request, 0, sizeof(request));
    request.context_id = context->id;
    request.stream_id = stream->id;
    request.kernel_id = kernel->id;
    request.cookie = desc->cookie;
    request.task_x = desc->config.task_dim.x;
    request.task_y = desc->config.task_dim.y;
    request.task_z = desc->config.task_dim.z;
    request.task_mode = (uint32_t)desc->config.task_mode;
    request.argument_count = desc->arg_count;
    for (index = 0; index < desc->arg_count; ++index) {
        const npu_kernel_arg_t *source = &desc->args[index];
        npu_ipc_launch_argument_t *destination = &request.arguments[index];

        if (source->reserved ||
            (source->kind != NPU_KERNEL_ARG_VALUE &&
             source->kind != NPU_KERNEL_ARG_BUFFER)) {
            status = -EINVAL;
            goto fail;
        }
        destination->kind = source->kind;
        destination->access = source->access;
        destination->size = source->size;
        if (source->kind == NPU_KERNEL_ARG_VALUE) {
            if (source->access ||
                (source->size != 1U && source->size != 2U &&
                 source->size != 4U && source->size != 8U)) {
                status = -EINVAL;
                goto fail;
            }
            destination->value_or_buffer = source->data.value;
        } else {
            npu_buffer_t *buffer = source->data.buffer.buffer;

            if (!buffer || buffer->context != context ||
                source->size != sizeof(uint64_t) || !source->access ||
                (source->access & ~(NPU_KERNEL_ARG_READ |
                                    NPU_KERNEL_ARG_WRITE)) ||
                source->data.buffer.offset >= buffer->size) {
                status = -EINVAL;
                goto fail;
            }
            destination->value_or_buffer = buffer->id;
            destination->offset = source->data.buffer.offset;
        }
    }
    status = runtime_control_call(context->runtime, NPU_IPC_KERNEL_LAUNCH, 0,
                                  &request, sizeof(request), 0, 0, &id);
    if (status < 0)
        goto fail;
    status = runtime_event_adopt(context, id, event, event_out);
    if (status < 0)
        free(event);
    return status;

fail:
    free(event);
    return status;
}
