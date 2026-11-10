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

#include "../../tensor_suite/tensor_test.h"

#include <stdio.h>

static constexpr unsigned kElements = 64;
static constexpr u32 kGuard = 0xa55a3cc3U;

static void tensor_run(void)
{
    auto left = reinterpret_cast<volatile u32 *>(DDR_BASE);
    auto right = reinterpret_cast<volatile u32 *>(DDR_BASE + 0x1000);
    auto output = reinterpret_cast<volatile u32 *>(OUT_DDR);
    auto guarded_input = reinterpret_cast<volatile u32 *>(DDR_BASE + 0x2000);
    auto guarded_output = reinterpret_cast<volatile u32 *>(OUT_DDR + 0x2000);
    auto diagnostic = reinterpret_cast<volatile u32 *>(OUT_DDR + 0x1000);

    for (unsigned lane = 0; lane < kElements; ++lane) {
        left[lane] = static_cast<u32>(lane * 7U - 200U);
        right[lane] = static_cast<u32>(lane * 3U - 70U);
    }
    fill_words(OUT_DDR, 128, 0);
    for (unsigned index = 0; index < 96; ++index)
        guarded_input[index] = kGuard;
    for (unsigned index = 0; index < 4; ++index)
        diagnostic[index] = 0;
    fence_io();

    dayan_cache_flush(reinterpret_cast<const void *>(DDR_BASE), 512U);
    dayan_cache_flush(reinterpret_cast<const void *>(DDR_BASE + 0x1000), 512U);
    dayan_cache_flush(reinterpret_cast<const void *>(DDR_BASE + 0x2000), 384U);

    configure_lsu_64b();
    write_tcsr<U_LSU, 0>(3);
    write_tcsr<U_LSU, 13>(3);
    emit32_a0_a1<dma_enc(0, 0, 0, 1, 0)>(DDR_BASE, URAM_BASE);
    wait_unit(U_LSU);
    emit32_a0_a1<dma_enc(0, 0, 0, 1, 0)>(DDR_BASE + 0x1000,
                                         URAM_BASE + 0x1000);
    wait_unit(U_LSU);
    write_tcsr<U_LSU, 0>(5);
    write_tcsr<U_LSU, 13>(5);
    emit32_a0_a1<dma_enc(0, 0, 0, 1, 0)>(DDR_BASE + 0x2000,
                                         URAM_BASE + 0x2000);
    wait_unit(U_LSU);

    configure_vec_single_thread();
    wait_vec_queue_capacity(4);
    emit32_a0<vec_read_enc(0, 0)>(URAM_BASE);
    emit32_a0<vec_read_enc(0)>(URAM_BASE + 0x1000);
    emit32<0x8400820bU>();
    emit32_a0<vec_write_enc(0, 0, 0)>(URAM_BASE + 0x3000);
    start_vec_block(0, 0, 4, 4, 4);

    write_tcsr<U_LSU, 0>(3);
    write_tcsr<U_LSU, 13>(3);
    emit32_a0_a1<dma_enc(0, 0, 0, 0, 1)>(URAM_BASE + 0x3000, OUT_DDR);
    wait_unit(U_LSU);
    dayan_cache_invalidate(reinterpret_cast<const void *>(OUT_DDR), 512U);

    write_tcsr<U_LSU, 0>(5);
    write_tcsr<U_LSU, 13>(5);
    dayan_cache_flush(reinterpret_cast<const void *>(OUT_DDR + 0x2000), 384U);
    emit32_a0_a1<dma_enc(0, 0, 0, 0, 1)>(URAM_BASE + 0x2000,
                                         OUT_DDR + 0x2000);
    wait_unit(U_LSU);
    dayan_cache_invalidate(reinterpret_cast<const void *>(OUT_DDR + 0x2000),
                           384U);

    for (unsigned index = 0; index < 96; ++index) {
        const bool guard = index < 16 || index >= 80;
        const unsigned lane = index - 16;
        const u32 actual = guard ? guarded_output[index] : output[lane];
        const u32 expected = guard
            ? kGuard
            : left[lane] + right[lane];
        if (actual != expected && diagnostic[0] == 0) {
            diagnostic[0] = guard ? 2U : 1U;
            diagnostic[1] = index;
            diagnostic[2] = actual;
            diagnostic[3] = expected;
        }
    }
    fence_io();
    dayan_cache_flush(reinterpret_cast<const void *>(OUT_DDR), 512U);
    dayan_cache_flush(reinterpret_cast<const void *>(OUT_DDR + 0x1000), 16U);

    if (diagnostic[0] != 0) {
        printf("[vec-vv-add-i32] kind=%u index=%u actual=%08x expected=%08x\n",
               diagnostic[0], diagnostic[1], diagnostic[2], diagnostic[3]);
        dayan_thread_exit(-22);
    }
}

DAYAN_TEST(vec_vv_int32_add_single_test, {
    .entry = tensor_run,
    .resources = {
        .max_child_threads = 0,
        .max_tasks_per_core = 1,
        .flags = 0,
        .lsu_tcsr_count = DAYAN_TCSR_LOGICAL_GROUPS,
        .mat_tcsr_count = DAYAN_TCSR_LOGICAL_GROUPS,
        .vec_tcsr_count = DAYAN_TCSR_LOGICAL_GROUPS,
        .macro_tcsr_count = DAYAN_TCSR_LOGICAL_GROUPS,
        .resource_counts = 0,
        .atomic_cache_size = 64,
        .uram_size = 0x180000,
    },
    .args = {0},
    .argc = 0,
    .name = "vec-vv-add-i32",
});
