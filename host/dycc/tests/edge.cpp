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

DAYAN_KERNEL_DESC(declared_kernel, 0, 0, 0, 0, 0, 0, 0);

extern "C" __kernel__ void c_kernel(uintptr_t address)
{
    *(int *)address = device_b;
}

DAYAN_KERNEL_DESC(c_kernel, 0, 0, 0, 0, 0, 0, 0);
