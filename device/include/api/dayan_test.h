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

#ifndef DAYAN_TEST_H
#define DAYAN_TEST_H

#include "dayan_sys.h"

#define DAYAN_TEST_DATA_BASE 0x84000000ULL
#define DAYAN_TEST_DATA_SIZE 0x40000000ULL
#define DAYAN_TEST_MAILBOX_BASE 0x83ffffc0ULL
#define DAYAN_TEST_MAILBOX_SIZE 64U
#define DAYAN_TEST_LOG_CAPACITY 4096U
#define DAYAN_TEST_LOG_HEADER_SIZE 64U
#define DAYAN_TEST_LOG_BASE \
    (DAYAN_TEST_MAILBOX_BASE - DAYAN_TEST_LOG_HEADER_SIZE - DAYAN_TEST_LOG_CAPACITY)
#define DAYAN_TEST_LOG_DATA_ADDR (DAYAN_TEST_LOG_BASE + DAYAN_TEST_LOG_HEADER_SIZE)
#define DAYAN_TEST_LOG_MAGIC 0x44594c4f47303031ULL
#define DAYAN_TEST_ALIVE_ADDR (DAYAN_TEST_MAILBOX_BASE + 0U)
#define DAYAN_TEST_RESULT_ADDR (DAYAN_TEST_MAILBOX_BASE + 8U)
#define DAYAN_TEST_DONE_ADDR (DAYAN_TEST_MAILBOX_BASE + 16U)
#define DAYAN_TEST_LAUNCH_FLAG_ADDR (DAYAN_TEST_MAILBOX_BASE + 32U)
#define DAYAN_TEST_LAUNCH_ENTRY_ADDR (DAYAN_TEST_MAILBOX_BASE + 40U)
#define DAYAN_TEST_ALIVE_VALUE 0xcafe0001ULL
#define DAYAN_TEST_DONE_VALUE 0xdead0001ULL
#define DAYAN_TEST_MAX_ARGS 32U

/* One local root Proc, run on Hart 0 in link order. Child Threads are allowed. */
typedef struct dayan_test_desc {
    void (*entry)(void);
    dayan_kernel_resource_desc_t resources;
    uintptr_t args[DAYAN_TEST_MAX_ARGS];
    uint32_t argc;
    char name[DAYAN_THREAD_NAME_SIZE];
} dayan_test_desc_t;

/* Pass a constant aggregate initializer; no C++ constructors are supported. */
#define DAYAN_TEST(symbol, ...) \
    static const dayan_test_desc_t symbol \
    __attribute__((used, section(".npu_test"), aligned(8))) = __VA_ARGS__

#endif
