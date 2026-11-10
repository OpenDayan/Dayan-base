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

#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern "C" __host__ char *getenv(const char *name);

__kernel__ void printf_grid_kernel(uint64_t arg0, uint64_t arg1,
                                   uint64_t arg2, uint64_t arg3,
                                   uint64_t arg4, uint64_t arg5,
                                   uint64_t arg6)
{
    const uint32_t x = dayan_task_id(DAYAN_TASK_X);
    const uint32_t y = dayan_task_id(DAYAN_TASK_Y);
    const uint32_t z = dayan_task_id(DAYAN_TASK_Z);

    // for(int i = 0 ; i < 1000 ; i ++) {
        printf("[device] grid_id=(%u,%u,%u) grid_dim=(%u,%u,%u) mode=%u "
            "args=[%llu,%llu,%llu,%llu,%llu,%llu,%llu]\n",
            (unsigned)x, (unsigned)y, (unsigned)z,
            (unsigned)dayan_task_dim(DAYAN_TASK_X),
            (unsigned)dayan_task_dim(DAYAN_TASK_Y),
            (unsigned)dayan_task_dim(DAYAN_TASK_Z),
            (unsigned)dayan_task_mode(),
            (unsigned long long)arg0, (unsigned long long)arg1,
            (unsigned long long)arg2, (unsigned long long)arg3,
            (unsigned long long)arg4, (unsigned long long)arg5,
            (unsigned long long)arg6);
        fflush(stdout);
    // }
}

DAYAN_KERNEL_DESC(printf_grid_kernel, 0, 0, 0, 0, 0, 0, 0);

__host__ static __attribute__((unused)) void usage(const char *program)
{
    printf("usage: %s [ARG0 ARG1 ARG2 ARG3 ARG4 ARG5 ARG6]\n", program);
    printf("DAYAN_SERVER_SOCKET selects the dayan_server socket.\n");
}

__host__ static __attribute__((unused)) int parse_u64(const char *text,
                                                       uint64_t *value)
{
    uint64_t parsed = 0;
    unsigned base = 10;

    if (!text || !*text || !value)
        return -1;
    if (text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        base = 16;
        text += 2;
        if (!*text)
            return -1;
    }
    while (*text) {
        unsigned digit;

        if (*text >= '0' && *text <= '9')
            digit = (unsigned)(*text - '0');
        else if (*text >= 'a' && *text <= 'f')
            digit = (unsigned)(*text - 'a') + 10U;
        else if (*text >= 'A' && *text <= 'F')
            digit = (unsigned)(*text - 'A') + 10U;
        else
            return -1;
        if (digit >= base || parsed > (UINT64_MAX - digit) / base)
            return -1;
        parsed = parsed * base + digit;
        ++text;
    }
    *value = parsed;
    return 0;
}

__host__ static __attribute__((unused)) void preserve_error(int candidate,
                                                             int *status)
{
    if (*status == 0 && candidate < 0)
        *status = candidate;
}

__host__ int main(int argc, char **argv)
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
            printf("invalid uint64 argument: %s\n", argv[index]);
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
        npu_kernel_launch_attrs_init(&launch_attrs, context, stream,
                                     "print-demo");
        launch_attrs.config.task_dim = NPUTaskDim3{4, 4, 4};
        launch_attrs.config.task_mode = NPUTaskMode::kSingleCore;
        printf("[host] launch grid=(4, 4, 4), "
               "args=[%llu,%llu,%llu,%llu,%llu,%llu,%llu]\n",
               (unsigned long long)args[0], (unsigned long long)args[1],
               (unsigned long long)args[2], (unsigned long long)args[3],
               (unsigned long long)args[4], (unsigned long long)args[5],
               (unsigned long long)args[6]);
        status = printf_grid_kernel(args[0], args[1], args[2], args[3],
                                    args[4], args[5], args[6], &launch_attrs);
    }
    if (status < 0)
        printf("print_demo failed: %d\n", status);

    if (stream)
        preserve_error(npu_stream_destroy(stream), &status);
    if (context)
        preserve_error(npu_context_destroy(context), &status);
    if (runtime)
        preserve_error(npu_runtime_close(runtime), &status);
    return status < 0 ? 1 : 0;
}
