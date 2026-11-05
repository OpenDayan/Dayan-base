#include "npu_hal_config.h"
#include "platform_regs.h"

#include <string.h>

void npu_hal_options_init(npu_hal_options_t *options)
{
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->version = NPU_HAL_OPTIONS_VERSION;
    options->cluster_count = 1;
    options->cores_per_cluster = 4;
    options->uram_size = DAYAN_URAM_SIZE;
}
