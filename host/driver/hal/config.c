/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

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
