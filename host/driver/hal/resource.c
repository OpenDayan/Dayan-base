#include "hal_internal.h"

#include <errno.h>

#include "platform_regs.h"

int npu_hal_shared_resource_init(npu_hal_t *hal, uint32_t cluster, uint32_t mask)
{
    if (!hal || cluster >= hal->cluster_count)
        return -EINVAL;
    if (!mask)
        return 0;
    return npu_hal_copy_to_address(hal, cluster, DAYAN_RESOURCE_GLOBAL_INIT,
                                   &mask, sizeof(mask));
}
