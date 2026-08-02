#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>
#include <stdio.h>
#include <stdint.h>
#else
#include <stdint.h>
#endif

__host__ uint64_t demo_host_bias = 1;
__device__ uint64_t demo_device_calls = 0;

uint64_t demo_clamp_count(uint64_t count)
{
    return count > 1024 ? 1024 : count;
}

__host__ uint64_t demo_prepare_count(uint64_t count)
{
    return demo_clamp_count(count) + demo_host_bias;
}

__device__ uint64_t demo_next_call(void)
{
    return ++demo_device_calls;
}

__kernel__ void demo_kernel(uint64_t count)
{
    count = demo_clamp_count(count);
    printf("demo kernel: count=%lu call=%lu\n", (unsigned long)count,
           (unsigned long)demo_next_call());
    fflush(stdout);
    dayan_yield();
}
