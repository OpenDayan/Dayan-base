#include <dayan_kernel.h>

__kernel__ void too_many_children_kernel()
{
}

DAYAN_KERNEL_DESC(too_many_children_kernel, 16, 0, 32768, 0, 0, 0, 0);
