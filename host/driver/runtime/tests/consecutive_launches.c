#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "npu_package.h"
#include "npu_runtime.h"

#define DEFAULT_LAUNCH_COUNT 16U
#define MAX_LAUNCH_COUNT 64U
#define TEST_TIMEOUT_MS 300000U

static int read_file(const char *path, void **data_out, size_t *size_out)
{
    FILE *file;
    void *data;
    long length;

    file = fopen(path, "rb");
    if (!file)
        return -errno;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (length = ftell(file)) <= 0 || (uint64_t)length > SIZE_MAX ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -EIO;
    }
    data = malloc((size_t)length);
    if (!data) {
        fclose(file);
        return -ENOMEM;
    }
    if (fread(data, 1, (size_t)length, file) != (size_t)length ||
        fclose(file) != 0) {
        free(data);
        return -EIO;
    }
    *data_out = data;
    *size_out = (size_t)length;
    return 0;
}

int main(int argc, char **argv)
{
    const uint64_t base = UINT64_C(0x4567890000006000);
    npu_package_argument_t argument = { 0 };
    npu_package_kernel_desc_t kernel_desc = { 0 };
    npu_runtime_options_t runtime_options;
    const char *socket_path;
    npu_runtime_t *runtime = 0;
    npu_context_t *context = 0;
    npu_module_t *module = 0;
    npu_kernel_t *kernel = 0;
    npu_stream_t **streams = 0;
    npu_buffer_t **buffers = 0;
    npu_event_t **executions = 0;
    npu_event_t **downloads = 0;
    uint64_t *outputs = 0;
    void *object = 0;
    void *package = 0;
    size_t object_size = 0;
    size_t package_size = 0;
    unsigned launch_count = DEFAULT_LAUNCH_COUNT;
    unsigned index;
    int status;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s KERNEL.o [LAUNCH_COUNT]\n", argv[0]);
        return 2;
    }
    if (argc == 3)
        launch_count = (unsigned)strtoul(argv[2], 0, 0);
    if (!launch_count || launch_count > MAX_LAUNCH_COUNT)
        return 2;
    status = read_file(argv[1], &object, &object_size);
    if (status < 0)
        goto out;
    argument.kind = NPU_PACKAGE_ARG_BUFFER;
    argument.size = sizeof(uint64_t);
    argument.alignment = sizeof(uint64_t);
    argument.access = NPU_PACKAGE_ARG_WRITE;
    kernel_desc.name = "sequence_kernel";
    kernel_desc.entry_symbol = "sequence_kernel";
    kernel_desc.supported_task_modes = NPU_PACKAGE_TASK_MODE_SINGLE_CORE;
    kernel_desc.max_tasks_per_core = 4;
    {
        npu_package_argument_t arguments[2] = { argument, { 0 } };

        arguments[1].kind = NPU_PACKAGE_ARG_VALUE;
        arguments[1].size = sizeof(uint64_t);
        kernel_desc.arguments = arguments;
        kernel_desc.argument_count = 2;
        status = npu_package_build(object, object_size, &kernel_desc, 1, 0,
                                   &package, &package_size);
    }
    if (status < 0)
        goto out;
    streams = calloc(launch_count, sizeof(*streams));
    buffers = calloc(launch_count, sizeof(*buffers));
    executions = calloc(launch_count, sizeof(*executions));
    downloads = calloc(launch_count, sizeof(*downloads));
    outputs = calloc(launch_count, sizeof(*outputs));
    if (!streams || !buffers || !executions || !downloads || !outputs) {
        status = -ENOMEM;
        goto out;
    }
    npu_runtime_options_init(&runtime_options);
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        runtime_options.server_socket = socket_path;
    status = npu_runtime_open(&runtime, &runtime_options);
    if (status < 0)
        goto out;
    status = npu_context_create(runtime, 0, &context);
    if (status < 0)
        goto out;
    status = npu_module_load(context, package, package_size, &module);
    if (status < 0)
        goto out;
    status = npu_module_get_kernel(module, "sequence_kernel", &kernel);
    if (status < 0)
        goto out;
    for (index = 0; index < launch_count; ++index) {
        npu_buffer_desc_t buffer_desc;

        status = npu_stream_create(context, 0, &streams[index]);
        if (status < 0)
            goto out;
        npu_buffer_desc_init(&buffer_desc, sizeof(outputs[index]));
        status = npu_buffer_create(context, &buffer_desc, &buffers[index]);
        if (status < 0)
            goto out;
    }
    for (index = 0; index < launch_count; ++index) {
        npu_launch_desc_t launch;
        uint64_t value = base + index;

        npu_launch_desc_init(&launch, "sequence-kernel");
        launch.arg_count = 2;
        npu_kernel_arg_buffer(&launch.args[0], buffers[index], 0,
                              NPU_KERNEL_ARG_WRITE);
        npu_kernel_arg_value(&launch.args[1], value, sizeof(value));
        status = npu_kernel_launch(streams[index], kernel, &launch,
                                   &executions[index]);
        if (status < 0)
            goto out;
    }
    for (index = 0; index < launch_count; ++index) {
        status = npu_memcpy_d2h_async(streams[index], &outputs[index],
                                      buffers[index], 0,
                                      sizeof(outputs[index]),
                                      &downloads[index]);
        if (status < 0)
            goto out;
    }
    for (index = 0; index < launch_count; ++index) {
        npu_event_info_t info;

        status = npu_event_wait(executions[index], TEST_TIMEOUT_MS, &info);
        if (status < 0 || info.status < 0)
            goto out;
        status = npu_event_destroy(executions[index]);
        if (status < 0)
            goto out;
        executions[index] = 0;
        status = npu_event_wait(downloads[index], TEST_TIMEOUT_MS, &info);
        if (status < 0 || info.status < 0)
            goto out;
        status = npu_event_destroy(downloads[index]);
        if (status < 0)
            goto out;
        downloads[index] = 0;
        if (outputs[index] != base + index) {
            status = -EIO;
            goto out;
        }
    }
    printf("consecutive launches passed: %u\n", launch_count);

out:
    if (status < 0)
        fprintf(stderr, "consecutive launches failed: %d\n", status);
    if (executions) {
        for (index = 0; index < launch_count; ++index) {
            if (executions[index]) {
                npu_event_info_t info;

                if (npu_event_wait(executions[index], TEST_TIMEOUT_MS,
                                   &info) == 0)
                    (void)npu_event_destroy(executions[index]);
            }
            if (downloads && downloads[index]) {
                npu_event_info_t info;

                if (npu_event_wait(downloads[index], TEST_TIMEOUT_MS,
                                   &info) == 0)
                    (void)npu_event_destroy(downloads[index]);
            }
        }
    }
    if (kernel)
        (void)npu_kernel_release(kernel);
    if (module)
        (void)npu_module_unload(module);
    if (streams) {
        for (index = 0; index < launch_count; ++index) {
            if (streams[index])
                (void)npu_stream_destroy(streams[index]);
            if (buffers && buffers[index])
                (void)npu_buffer_destroy(buffers[index]);
        }
    }
    if (context)
        (void)npu_context_destroy(context);
    if (runtime)
        (void)npu_runtime_close(runtime);
    npu_package_free(package);
    free(object);
    free(outputs);
    free(downloads);
    free(executions);
    free(buffers);
    free(streams);
    return status < 0 ? 1 : 0;
}
