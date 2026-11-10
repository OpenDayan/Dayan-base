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
#include <stdint.h>

__host__ int host_bias = 3;
__device__ uint64_t device_counter = 0;

namespace sample {
__kernel__ void vector_add(uint64_t *output, uint64_t value, float scale);
__kernel__ void locked_kernel(uint64_t *output);
}

uint64_t add_bias(uint64_t value)
{
    return value + 1;
}

__host__ int host_helper(int value)
{
    return value + host_bias;
}

__host__ int host_launch(npu_buffer_t *output, uint64_t value, float scale,
                         const npu_kernel_launch_attrs_t *attrs)
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

DAYAN_KERNEL_DESC(vector_add, 3, 4096, 32768, 1, 2, 3, 4);

__kernel__ void locked_kernel(uint64_t *output)
{
    *output = 1;
}

DAYAN_KERNEL_LOCKED_TASK_DESC(locked_kernel, 0, 0, 0, 0, 0, 0, 0);

} // namespace sample

__host__ int main() {
    host_launch(nullptr, 42, 1.0f, nullptr);
}
