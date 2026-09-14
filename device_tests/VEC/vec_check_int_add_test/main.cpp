#include "../../tensor_suite/tensor_test.h"

#include <stdio.h>

static constexpr unsigned VEC_LANES = 64;

static constexpr u32 input_left[VEC_LANES] = {
    0x00000000U, 0x00000001U, 0xffffffffU, 0x80000000U, 0x7fffffffU, 0x00800000U, 0x007fffffU, 0x00ffffffU,
    0xff000001U, 0xaaaaaaaaU, 0x55555555U, 0xdeadbeefU, 0x0000001fU, 0x00000020U, 0x12345678U, 0x80000001U,
    0x00000000U, 0x00000001U, 0xffffffffU, 0x80000000U, 0x7fffffffU, 0x00800000U, 0x007fffffU, 0x00ffffffU,
    0xff000001U, 0xaaaaaaaaU, 0x55555555U, 0xdeadbeefU, 0x0000001fU, 0x00000020U, 0x12345678U, 0x80000001U,
    0x00000000U, 0x00000001U, 0xffffffffU, 0x80000000U, 0x7fffffffU, 0x00800000U, 0x007fffffU, 0x00ffffffU,
    0xff000001U, 0xaaaaaaaaU, 0x55555555U, 0xdeadbeefU, 0x0000001fU, 0x00000020U, 0x12345678U, 0x80000001U,
    0x00000000U, 0x00000001U, 0xffffffffU, 0x80000000U, 0x7fffffffU, 0x00800000U, 0x007fffffU, 0x00ffffffU,
    0xff000001U, 0xaaaaaaaaU, 0x55555555U, 0xdeadbeefU, 0x0000001fU, 0x00000020U, 0x12345678U, 0x80000001U,
};

static constexpr u32 input_right[VEC_LANES] = {
    0x80000000U, 0x55555555U, 0x00000001U, 0xff000001U, 0x80000001U, 0x007fffffU, 0x00000020U, 0x7fffffffU,
    0xdeadbeefU, 0xffffffffU, 0xaaaaaaaaU, 0x00000000U, 0x00ffffffU, 0x12345678U, 0x00800000U, 0x0000001fU,
    0x7fffffffU, 0xdeadbeefU, 0xffffffffU, 0xaaaaaaaaU, 0x00000000U, 0x00ffffffU, 0x12345678U, 0x00800000U,
    0x0000001fU, 0x80000000U, 0x55555555U, 0x00000001U, 0xff000001U, 0x80000001U, 0x007fffffU, 0x00000020U,
    0x00800000U, 0x0000001fU, 0x80000000U, 0x55555555U, 0x00000001U, 0xff000001U, 0x80000001U, 0x007fffffU,
    0x00000020U, 0x7fffffffU, 0xdeadbeefU, 0xffffffffU, 0xaaaaaaaaU, 0x00000000U, 0x00ffffffU, 0x12345678U,
    0x007fffffU, 0x00000020U, 0x7fffffffU, 0xdeadbeefU, 0xffffffffU, 0xaaaaaaaaU, 0x00000000U, 0x00ffffffU,
    0x12345678U, 0x00800000U, 0x0000001fU, 0x80000000U, 0x55555555U, 0x00000001U, 0xff000001U, 0x80000001U,
};

static constexpr u32 expected_output[VEC_LANES] = {
    0x80000000U, 0x55555556U, 0x00000000U, 0x7f000001U, 0x00000000U, 0x00ffffffU, 0x0080001fU, 0x80fffffeU,
    0xddadbef0U, 0xaaaaaaa9U, 0xffffffffU, 0xdeadbeefU, 0x0100001eU, 0x12345698U, 0x12b45678U, 0x80000020U,
    0x7fffffffU, 0xdeadbef0U, 0xfffffffeU, 0x2aaaaaaaU, 0x7fffffffU, 0x017fffffU, 0x12b45677U, 0x017fffffU,
    0xff000020U, 0x2aaaaaaaU, 0xaaaaaaaaU, 0xdeadbef0U, 0xff000020U, 0x80000021U, 0x12b45677U, 0x80000021U,
    0x00800000U, 0x00000020U, 0x7fffffffU, 0xd5555555U, 0x80000000U, 0xff800001U, 0x80800000U, 0x017ffffeU,
    0xff000021U, 0x2aaaaaa9U, 0x34031444U, 0xdeadbeeeU, 0xaaaaaac9U, 0x00000020U, 0x13345677U, 0x92345679U,
    0x007fffffU, 0x00000021U, 0x7ffffffeU, 0x5eadbeefU, 0x7ffffffeU, 0xab2aaaaaU, 0x007fffffU, 0x01fffffeU,
    0x11345679U, 0xab2aaaaaU, 0x55555574U, 0x5eadbeefU, 0x55555574U, 0x00000021U, 0x11345679U, 0x00000002U,
};

static void tensor_run(void)
{
    auto left = reinterpret_cast<volatile u32 *>(DDR_BASE);
    auto right = reinterpret_cast<volatile u32 *>(DDR_BASE + 0x1000);
    auto output = reinterpret_cast<volatile u32 *>(OUT_DDR);
    auto guarded_input = reinterpret_cast<volatile u32 *>(DDR_BASE + 0x2000);
    auto guarded_output = reinterpret_cast<volatile u32 *>(OUT_DDR + 0x2000);
    auto diagnostic = reinterpret_cast<volatile u32 *>(OUT_DDR + 0x1000);
    for (unsigned lane = 0; lane < 64; ++lane) {
        left[lane] = input_left[lane];
        right[lane] = input_right[lane];
    }
    fill_words(OUT_DDR, 128, 0);
    for (unsigned index = 0; index < 96; ++index)
        guarded_input[index] = 0xa55a3cc3U;
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
    emit32_a0_a1<dma_enc(0, 0, 0, 1, 0)>(DDR_BASE + 0x1000, URAM_BASE + 0x1000);
    wait_unit(U_LSU);
    write_tcsr<U_LSU, 0>(5);
    write_tcsr<U_LSU, 13>(5);
    emit32_a0_a1<dma_enc(0, 0, 0, 1, 0)>(DDR_BASE + 0x2000, URAM_BASE + 0x3000 - 64);
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
    emit32_a0_a1<dma_enc(0, 0, 0, 0, 1)>(URAM_BASE + 0x3000 - 64, OUT_DDR + 0x2000);
    wait_unit(U_LSU);
    dayan_cache_invalidate(reinterpret_cast<const void *>(OUT_DDR + 0x2000), 384U);
    for (unsigned index = 0; index < 96; ++index) {
        const bool guard = index < 16 || index >= 80;
        const u32 actual = guard ? guarded_output[index] : output[index - 16];
        const u32 wanted = guard ? 0xa55a3cc3U : expected_output[index - 16];
        if (actual != wanted && diagnostic[0] == 0) {
            diagnostic[0] = guard ? 2 : 1;
            diagnostic[1] = index;
            diagnostic[2] = actual;
            diagnostic[3] = wanted;
        }
    }
    fence_io();
    dayan_cache_flush(reinterpret_cast<const void *>(OUT_DDR), 512U);
    dayan_cache_flush(reinterpret_cast<const void *>(OUT_DDR + 0x1000), 16U);
    if (diagnostic[0] != 0) {
        printf("[vec] kind=%u index=%u actual=%08x expected=%08x\n",
               diagnostic[0], diagnostic[1], diagnostic[2], diagnostic[3]);
        dayan_thread_exit(-22);
    }
}

DAYAN_TEST(tensor_test, {
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
    .name = "v004",
});
