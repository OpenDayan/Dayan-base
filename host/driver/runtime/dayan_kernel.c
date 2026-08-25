#include "runtime_internal.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dayan_kernel.h"

struct npu_kernel_binding {
    struct npu_context *context;
    npu_module_t *module;
    npu_kernel_t *kernel;
    struct npu_kernel_binding *next;
};

struct npu_kernel_descriptor {
    pthread_mutex_t mutex;
    void *package_image;
    size_t package_size;
    char *entry_symbol;
    struct npu_kernel_binding *bindings;
    struct npu_kernel_descriptor *global_next;
};

static pthread_mutex_t descriptor_list_mutex = PTHREAD_MUTEX_INITIALIZER;
static struct npu_kernel_descriptor *descriptor_list;

void npu_kernel_launch_attrs_init(npu_kernel_launch_attrs_t *attrs,
                                  npu_context_t *context,
                                  npu_stream_t *stream,
                                  const char *task_name)
{
    size_t name_length;

    if (!attrs)
        return;
    memset(attrs, 0, sizeof(*attrs));
    attrs->version = NPU_KERNEL_LAUNCH_ATTRS_VERSION;
    attrs->context = context;
    attrs->stream = stream;
    npu_launch_config_init(&attrs->config);
    if (context && context->runtime)
        attrs->timeout_ms = context->runtime->default_timeout_ms;
    if (!task_name || !task_name[0])
        return;
    name_length = strnlen(task_name, sizeof(attrs->task_name));
    if (name_length == sizeof(attrs->task_name))
        return;
    memcpy(attrs->task_name, task_name, name_length + 1U);
}

int npu_kernel_descriptor_create(
    const void *object, size_t object_size, const char *entry_symbol,
    const npu_package_argument_t *arguments, size_t argument_count,
    const dayan_kernel_resource_desc_t *resources,
    npu_kernel_descriptor_t **descriptor_out)
{
    npu_package_kernel_desc_t kernel;
    struct npu_kernel_descriptor *descriptor;
    int status;

    if (!object || !object_size || !entry_symbol || !*entry_symbol ||
        argument_count > NPU_PACKAGE_MAX_ARGS ||
        (argument_count && !arguments) || !resources ||
        resources->version != DAYAN_KERNEL_RESOURCE_DESC_VERSION ||
        resources->size != sizeof(*resources) ||
        resources->max_tasks_per_core > 4U ||
        resources->max_child_threads > DAYAN_KERNEL_MAX_CHILD_THREADS ||
        (resources->flags & ~DAYAN_KERNEL_RESOURCE_FLAGS_ALL) ||
        resources->reserved ||
        resources->lsu_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->mat_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->vec_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        resources->macro_tcsr_count > DAYAN_TCSR_LOGICAL_GROUPS ||
        ((resources->flags & DAYAN_KERNEL_RESOURCE_LOCKED_TASK) &&
         resources->max_tasks_per_core > 1U) ||
        (resources->uram_size & (DAYAN_URAM_ALLOCATION_GRANULE - 1U)) ||
        !descriptor_out)
        return -EINVAL;
    *descriptor_out = 0;
    descriptor = calloc(1, sizeof(*descriptor));
    if (!descriptor)
        return -ENOMEM;
    descriptor->entry_symbol = strdup(entry_symbol);
    if (!descriptor->entry_symbol) {
        free(descriptor);
        return -ENOMEM;
    }
    status = pthread_mutex_init(&descriptor->mutex, 0);
    if (status) {
        free(descriptor->entry_symbol);
        free(descriptor);
        return -status;
    }
    memset(&kernel, 0, sizeof(kernel));
    kernel.name = entry_symbol;
    kernel.entry_symbol = entry_symbol;
    kernel.supported_task_modes =
        resources->flags & DAYAN_KERNEL_RESOURCE_LOCKED_TASK ?
            NPU_PACKAGE_TASK_MODE_SINGLE_CORE : NPU_PACKAGE_TASK_MODE_ALL;
    kernel.max_tasks_per_core = resources->max_tasks_per_core ?
        resources->max_tasks_per_core : 1U;
    kernel.max_child_threads = resources->max_child_threads;
    kernel.flags = resources->flags & DAYAN_KERNEL_RESOURCE_LOCKED_TASK ?
        NPU_PACKAGE_KERNEL_LOCKED_TASK : 0U;
    kernel.lsu_tcsr_count = resources->lsu_tcsr_count;
    kernel.mat_tcsr_count = resources->mat_tcsr_count;
    kernel.vec_tcsr_count = resources->vec_tcsr_count;
    kernel.macro_tcsr_count = resources->macro_tcsr_count;
    kernel.atomic_cache_size = resources->atomic_cache_size;
    kernel.uram_size = resources->uram_size;
    kernel.arguments = arguments;
    kernel.argument_count = argument_count;
    status = npu_package_build(object, object_size, &kernel, 1, 0,
                               &descriptor->package_image,
                               &descriptor->package_size);
    if (status < 0) {
        pthread_mutex_destroy(&descriptor->mutex);
        free(descriptor->entry_symbol);
        free(descriptor);
        return status;
    }
    pthread_mutex_lock(&descriptor_list_mutex);
    descriptor->global_next = descriptor_list;
    descriptor_list = descriptor;
    pthread_mutex_unlock(&descriptor_list_mutex);
    *descriptor_out = descriptor;
    return 0;
}

static struct npu_kernel_binding *binding_get(
    struct npu_kernel_descriptor *descriptor, struct npu_context *context,
    int *status_out)
{
    struct npu_kernel_binding *binding;
    int status;

    for (binding = descriptor->bindings; binding; binding = binding->next) {
        if (binding->context == context) {
            *status_out = 0;
            return binding;
        }
    }
    binding = calloc(1, sizeof(*binding));
    if (!binding) {
        *status_out = -ENOMEM;
        return 0;
    }
    status = npu_module_load(context, descriptor->package_image,
                             descriptor->package_size, &binding->module);
    if (status == 0)
        status = npu_module_get_kernel(binding->module,
                                       descriptor->entry_symbol,
                                       &binding->kernel);
    if (status < 0) {
        if (binding->module)
            (void)npu_module_unload(binding->module);
        free(binding);
        *status_out = status;
        return 0;
    }
    binding->context = context;
    binding->next = descriptor->bindings;
    descriptor->bindings = binding;
    *status_out = 0;
    return binding;
}

int npu_kernel_descriptor_launch(npu_kernel_descriptor_t *descriptor,
                        const npu_kernel_arg_t *args, size_t argc,
                        const npu_kernel_launch_attrs_t *attrs)
{
    struct npu_kernel_binding *binding;
    npu_launch_desc_t launch;
    npu_event_info_t info;
    npu_event_t *event = 0;
    uint32_t timeout;
    int status;

    if (!descriptor || !attrs ||
        attrs->version != NPU_KERNEL_LAUNCH_ATTRS_VERSION ||
        (attrs->flags & ~NPU_KERNEL_LAUNCH_ASYNC) || attrs->reserved ||
        !runtime_context_valid(attrs->context) ||
        !runtime_stream_valid(attrs->stream) ||
        attrs->stream->context != attrs->context ||
        argc > NPU_KERNEL_MAX_ARGS || (argc && !args) ||
        !attrs->task_name[0] ||
        strnlen(attrs->task_name, sizeof(attrs->task_name)) ==
            sizeof(attrs->task_name) ||
        ((attrs->flags & NPU_KERNEL_LAUNCH_ASYNC) && !attrs->event_out))
        return -EINVAL;
    pthread_mutex_lock(&descriptor->mutex);
    binding = binding_get(descriptor, attrs->context, &status);
    pthread_mutex_unlock(&descriptor->mutex);
    if (!binding) {
        fprintf(stderr, "DBG-dl: binding_get failed status=%d\n", status);
        return status;
    }
    npu_launch_desc_init(&launch, attrs->task_name);
    launch.config = attrs->config;
    launch.cookie = attrs->cookie;
    launch.arg_count = (uint32_t)argc;
    if (argc)
        memcpy(launch.args, args, argc * sizeof(*args));
    status = npu_kernel_launch(attrs->stream, binding->kernel, &launch,
                               &event);
    if (status < 0) {
        fprintf(stderr, "DBG-dl: npu_kernel_launch status=%d\n", status);
        return status;
    }
    if (attrs->flags & NPU_KERNEL_LAUNCH_ASYNC) {
        *attrs->event_out = event;
        return 0;
    }
    timeout = attrs->timeout_ms ? attrs->timeout_ms :
              attrs->context->runtime->default_timeout_ms;
    status = npu_event_wait(event, timeout, &info);
    if (status < 0) {
        fprintf(stderr, "DBG-dl: npu_event_wait status=%d\n", status);
        return status;
    }
    status = npu_event_destroy(event);
    if (status < 0)
        return status;
    if (info.status < INT_MIN || info.status > INT_MAX)
        return -EIO;
    fprintf(stderr, "DBG-dl: device returned status=%ld event_status=%d\n", (long)info.status, (int)status);
    return (int)info.status;
}

void runtime_kernel_descriptors_forget_context(struct npu_context *context)
{
    struct npu_kernel_descriptor *descriptor;

    if (!context)
        return;
    pthread_mutex_lock(&descriptor_list_mutex);
    for (descriptor = descriptor_list; descriptor;
         descriptor = descriptor->global_next) {
        struct npu_kernel_binding **link;

        pthread_mutex_lock(&descriptor->mutex);
        link = &descriptor->bindings;
        while (*link) {
            struct npu_kernel_binding *binding = *link;

            if (binding->context == context) {
                *link = binding->next;
                free(binding);
            } else {
                link = &binding->next;
            }
        }
        pthread_mutex_unlock(&descriptor->mutex);
    }
    pthread_mutex_unlock(&descriptor_list_mutex);
}

void npu_kernel_descriptor_destroy(npu_kernel_descriptor_t *descriptor)
{
    struct npu_kernel_descriptor **link;
    struct npu_kernel_binding *binding;

    if (!descriptor)
        return;
    pthread_mutex_lock(&descriptor_list_mutex);
    link = &descriptor_list;
    while (*link && *link != descriptor)
        link = &(*link)->global_next;
    if (*link)
        *link = descriptor->global_next;
    pthread_mutex_unlock(&descriptor_list_mutex);
    pthread_mutex_lock(&descriptor->mutex);
    binding = descriptor->bindings;
    descriptor->bindings = 0;
    pthread_mutex_unlock(&descriptor->mutex);
    while (binding) {
        struct npu_kernel_binding *next = binding->next;

        if (runtime_context_valid(binding->context)) {
            (void)npu_kernel_release(binding->kernel);
            (void)npu_module_unload(binding->module);
        }
        free(binding);
        binding = next;
    }
    npu_package_free(descriptor->package_image);
    pthread_mutex_destroy(&descriptor->mutex);
    free(descriptor->entry_symbol);
    free(descriptor);
}
