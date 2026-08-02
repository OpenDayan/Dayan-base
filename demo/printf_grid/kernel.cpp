#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#include <dayan_sys.h>
#include <stdio.h>
#endif

#include <stdint.h>

/*
 * dycc keeps this definition on Device and emits a Host wrapper with the same
 * name.  The seven scalar arguments occupy seven RV64 argument slots.
 */
__kernel__ void printf_grid_kernel(uint64_t arg0, uint64_t arg1,
                                   uint64_t arg2, uint64_t arg3,
                                   uint64_t arg4, uint64_t arg5,
                                   uint64_t arg6)
{
    const uint32_t x = dayan_task_id(DAYAN_TASK_X);
    const uint32_t y = dayan_task_id(DAYAN_TASK_Y);
    const uint32_t z = dayan_task_id(DAYAN_TASK_Z);

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
}
