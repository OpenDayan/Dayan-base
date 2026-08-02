#include <stdint.h>

#include <dayan_sys.h>

void grid_kernel(uint64_t *output, uint64_t base)
{
    uint32_t x = dayan_task_id(DAYAN_TASK_X);
    uint32_t y = dayan_task_id(DAYAN_TASK_Y);
    uint32_t z = dayan_task_id(DAYAN_TASK_Z);
    uint32_t dim_x = dayan_task_dim(DAYAN_TASK_X);
    uint32_t dim_y = dayan_task_dim(DAYAN_TASK_Y);
    uint64_t linear = x + (uint64_t)dim_x * (y + (uint64_t)dim_y * z);

    output[linear] = base + (uint64_t)dayan_task_mode() * 100U + linear;
    __asm__ volatile("fence rw, rw" ::: "memory");
}
