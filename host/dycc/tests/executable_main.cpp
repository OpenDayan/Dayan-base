#include <dayan_kernel.h>

int main(void)
{
    dayan_kernel_launch_attrs_t attrs;

    dayan_kernel_launch_attrs_init(&attrs, nullptr);
    return attrs.version == DAYAN_KERNEL_LAUNCH_ATTRS_VERSION ? 0 : 1;
}
