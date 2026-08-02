#include <dayan_kernel.h>

int main(void)
{
    npu_kernel_launch_attrs_t attrs;

    npu_kernel_launch_attrs_init(&attrs, nullptr, nullptr);
    return attrs.version == NPU_KERNEL_LAUNCH_ATTRS_VERSION ? 0 : 1;
}
