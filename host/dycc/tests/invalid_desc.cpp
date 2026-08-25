#include <dayan_kernel.h>

__kernel__ void invalid_desc_kernel()
{
}

DAYAN_KERNEL_DESC(invalid_desc_kernel, 1, 64, 1000, 0, 0, 0, 0);
