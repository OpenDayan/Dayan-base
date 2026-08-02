#include <dayan_kernel.h>
#include <stdint.h>

__device__ uint64_t executable_counter = 0;

__kernel__ void executable_kernel(uint64_t *value)
{
    *value += ++executable_counter;
}

DAYAN_KERNEL_DESC(executable_kernel, 0, 0, 0);
