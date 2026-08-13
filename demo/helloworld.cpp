#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>

__kernel__ void helloworld(uint64_t arg0, uint64_t arg1)
{
    const uint32_t x = dayan_task_id(DAYAN_TASK_X); const uint32_t y = dayan_task_id(DAYAN_TASK_Y);
    const uint32_t z = dayan_task_id(DAYAN_TASK_Z);

    printf("[device] grid_id=(%u,%u,%u) mode=%u "
           "args=[%llu,%llu]\n", (unsigned)x, (unsigned)y, (unsigned)z, (unsigned)dayan_task_mode(),
           (unsigned long long)arg0, (unsigned long long)arg1 );
    fflush(stdout);
}

DAYAN_KERNEL_DESC(helloworld, 0, 0, 0);

__host__ int main() {
    npu_runtime_options_t runtime_options; npu_runtime_t *runtime = nullptr;
    npu_context_t *context = nullptr;      npu_stream_t *stream = nullptr;
    npu_kernel_launch_attrs_t launch_attrs;

    npu_runtime_options_init(&runtime_options);
    npu_runtime_open(&runtime, &runtime_options);
    npu_context_create(runtime, nullptr, &context);
    npu_stream_create(context, nullptr, &stream);
    npu_kernel_launch_attrs_init(&launch_attrs, context, stream, 
        "hello-world");
    helloworld(1, 2, &launch_attrs);
}

