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

#ifndef DAYAN_DEMO_URAM_CONFIG_H
#define DAYAN_DEMO_URAM_CONFIG_H

#include <npu_runtime.h>
#include <cerrno>
#include <cstdio>

// Running allocations do not change this per-Core capacity. The Server
// schedules each launch when its declared resource budget becomes available.
static inline int demo_uram_capacity(npu_runtime_t *runtime, uint64_t required,
                                     uint64_t *capacity)
{
    const int status = npu_runtime_get_uram_capacity(runtime, capacity);
    if (status < 0) {
        std::fprintf(stderr, "URAM capacity query failed: %d\n", status);
        return status;
    }
    std::printf("[host] URAM capacity per Core: %llu KiB\n",
                (unsigned long long)(*capacity / 1024));
    if (*capacity < required) {
        std::fprintf(stderr, "operator requires at least %llu KiB per Core\n",
                     (unsigned long long)(required / 1024));
        return -ENOSPC;
    }
    return 0;
}

#endif
