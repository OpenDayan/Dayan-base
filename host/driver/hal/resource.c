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
