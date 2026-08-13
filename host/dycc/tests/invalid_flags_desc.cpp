#include <dayan_kernel.h>

__kernel__ void invalid_flags_kernel()
{
}

DAYAN_KERNEL_RESOURCE_DESC(invalid_flags_kernel, 1, 0, 2, 0, 0);
