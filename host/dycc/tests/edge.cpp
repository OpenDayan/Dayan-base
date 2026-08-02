#include <dayan_kernel.h>
#include <stdint.h>

__device__ int device_a = 1, device_b = 2;
__host__ int host_a = 3, host_b = 4;

struct holder {
    int value;
};

__device__ holder device_holders[2] = {{1}, {2}}, device_other{3};

__kernel__ void declared_kernel(int value = -1);

__kernel__ void declared_kernel(int value)
{
    device_a += value;
}

extern "C" __kernel__ void c_kernel(uintptr_t address)
{
    *(int *)address = device_b;
}
