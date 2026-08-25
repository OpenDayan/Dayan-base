#include <dayan_kernel.h>

__kernel__ int invalid_return(int value)
{
    return value;
}

DAYAN_KERNEL_DESC(invalid_return, 0, 0, 0, 0, 0, 0, 0);
