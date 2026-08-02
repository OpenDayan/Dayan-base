#include <dayan_kernel.h>
#include <stdint.h>

__host__ int host_bias = 3;
__device__ uint64_t device_counter = 0;

namespace sample {
__kernel__ void vector_add(uint64_t *output, uint64_t value, float scale);
}

uint64_t add_bias(uint64_t value)
{
    return value + 1;
}

__host__ int host_helper(int value)
{
    return value + host_bias;
}

__host__ int host_launch(uint64_t *output, uint64_t value, float scale,
                         const dayan_kernel_launch_attrs_t *attrs)
{
    return sample::vector_add(output, value, scale, attrs);
}

__device__ uint64_t device_helper(uint64_t value)
{
    return value + device_counter;
}

namespace sample {

__kernel__ void vector_add(uint64_t *output, uint64_t value, float scale)
{
    *output = device_helper(value) + (uint64_t)scale;
}

} // namespace sample

__host__ int main() {
    return 0l;
}
