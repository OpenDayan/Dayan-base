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

#ifndef NPU_HAL_INPROCESS_H
#define NPU_HAL_INPROCESS_H

#include <stddef.h>
#include "npu_hal_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_HAL_INPROCESS_OPTIONS_VERSION 1U

/* Callbacks run on HAL/Runtime worker threads. The owner must keep its model
 * advancing independently and return 0 or a negative errno for the full range.
 * No callback or context pointer is sent over the Runtime transport.
 * Server and Runtime must statically link the same registry in one process. */
typedef struct npu_hal_inprocess_device_desc {
    uint32_t cluster_count;
    void *context;
    int (*read)(void *context, uint32_t cluster, uint64_t address,
                void *destination, size_t size);
    int (*write)(void *context, uint32_t cluster, uint64_t address,
                 const void *source, size_t size);
} npu_hal_inprocess_device_desc_t;

typedef struct npu_hal_inprocess_options {
    npu_hal_backend_options_t base;
    uint64_t device_id;
} npu_hal_inprocess_options_t;

void npu_hal_inprocess_options_init(npu_hal_inprocess_options_t *options,
                                    uint64_t device_id);
int npu_hal_inprocess_device_register(
    const npu_hal_inprocess_device_desc_t *desc, uint64_t *device_id);
/* Returns -EBUSY while a HAL or provider retains the device. Callback context
 * may be destroyed only after successful unregister. IDs are never reused. */
int npu_hal_inprocess_device_unregister(uint64_t device_id);

#ifdef __cplusplus
}
#endif

#endif
