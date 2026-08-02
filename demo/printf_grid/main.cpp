#include <dayan_runtime.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Host wrapper emitted by dycc from kernel.cpp. */
int printf_grid_kernel(uint64_t arg0, uint64_t arg1, uint64_t arg2,
                       uint64_t arg3, uint64_t arg4, uint64_t arg5,
                       uint64_t arg6,
                       const npu_kernel_launch_attrs_t *launch_attrs);

static void usage(const char *program)
{
    printf("usage: %s [ARG0 ARG1 ARG2 ARG3 ARG4 ARG5 ARG6]\n", program);
    printf("DAYAN_SERVER_SOCKET selects the dayan_server socket.\n");
}

static int parse_u64(const char *text, uint64_t *value)
{
    char *end = nullptr;
    unsigned long long parsed;

    errno = 0;
    parsed = strtoull(text, &end, 0);
    if (errno || end == text || *end != '\0')
        return -1;
    *value = (uint64_t)parsed;
    return 0;
}

static void preserve_error(int candidate, int *status)
{
    if (*status == 0 && candidate < 0)
        *status = candidate;
}

int main(int argc, char **argv)
{
    uint64_t args[7] = { 11, 22, 33, 44, 55, 66, 77 };
    npu_runtime_options_t runtime_options;
    npu_kernel_launch_attrs_t launch_attrs;
    npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;
    npu_stream_t *stream = nullptr;
    const char *socket_path;
    int status = 0;
    int index;

    if (argc == 2 && strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }
    if (argc != 1 && argc != 8) {
        usage(argv[0]);
        return 2;
    }
    for (index = 1; index < argc; ++index) {
        if (parse_u64(argv[index], &args[index - 1]) < 0) {
            fprintf(stderr, "invalid uint64 argument: %s\n", argv[index]);
            return 2;
        }
    }

    npu_runtime_options_init(&runtime_options);
    socket_path = getenv("DAYAN_SERVER_SOCKET");
    if (socket_path && *socket_path)
        runtime_options.server_socket = socket_path;

    status = npu_runtime_open(&runtime, &runtime_options);
    if (status == 0)
        status = npu_context_create(runtime, nullptr, &context);
    if (status == 0)
        status = npu_stream_create(context, nullptr, &stream);
    if (status == 0) {
        npu_kernel_launch_attrs_init(&launch_attrs, context, stream);
        launch_attrs.config.task_dim = NPUTaskDim3{2, 2, 1};
        launch_attrs.config.task_mode = NPUTaskMode::kSingleCore;
        printf("[host] launch grid=(2,2,1), args=[%llu,%llu,%llu,%llu,%llu,%llu,%llu]\n",
               (unsigned long long)args[0], (unsigned long long)args[1],
               (unsigned long long)args[2], (unsigned long long)args[3],
               (unsigned long long)args[4], (unsigned long long)args[5],
               (unsigned long long)args[6]);
        status = printf_grid_kernel(args[0], args[1], args[2], args[3],
                                    args[4], args[5], args[6], &launch_attrs);
    }
    if (status < 0)
        fprintf(stderr, "printf_grid_demo failed: %d\n", status);

    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status < 0 ? 1 : 0;
}
