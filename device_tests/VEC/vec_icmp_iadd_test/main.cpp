/* 从 dayan-samples/Instructions/VEC/dependency/vec_icmp_iadd.cpp 迁移（裸机版，
 * 发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：单 VEC 块内 int32 比较 -> 谓词 -> mux2 -> int32 add，9 条指令一次
 * CONTROL 启动，块内无等待。bank0: a=V1 f=V0；bank1: b=V18 t=V17 add=V16。
 * 期望（按位比较）：want=((a[i]<b[i])?t[i]:f[i])+add[i]，逐 lane。
 */

#include "../../instr_enc/tx_test.h"

#include <stdint.h>
#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(int32_t);
static constexpr u64 SPACING = 5U * KBYTES;

static void tensor_run(void)
{
    auto a = reinterpret_cast<volatile int32_t *>(txtst::DDR_BASE);
    auto b = reinterpret_cast<volatile int32_t *>(txtst::DDR_BASE + 1U * SPACING);
    auto f = reinterpret_cast<volatile int32_t *>(txtst::DDR_BASE + 2U * SPACING);
    auto t = reinterpret_cast<volatile int32_t *>(txtst::DDR_BASE + 3U * SPACING);
    auto add = reinterpret_cast<volatile int32_t *>(txtst::DDR_BASE + 4U * SPACING);

    for (unsigned i = 0; i < LANES; ++i) {
        a[i] = (int32_t)((i * 5U) % 17U) - 8;
        b[i] = (int32_t)((i * 7U) % 19U) - 9;
        f[i] = -100000 - (int32_t)i;
        t[i] = 100000 + (int32_t)(3 * i);
        add[i] = (int32_t)((i * 3U) % 11U) - 5;
    }
    for (u64 region = 0; region < 5U; ++region)
        txtst::flush_ddr(txtst::DDR_BASE + region * SPACING, KBYTES);

    txtst::configure_lsu_linear(KBYTES);
    for (u64 region = 0; region < 5U; ++region) {
        txtst::dma_dram_to_uram(txtst::DDR_BASE + region * SPACING,
                                txtst::URAM_BASE + region * SPACING);
    }
    txtst::wait_lsu();

    txtst::configure_vec_zero();

    /* 9 条：5 load + int32 cmp(LT) + mux2 + add + store。 */
    txtst::wait_vec_queue_capacity(9);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(txtst::URAM_BASE + 0U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(txtst::URAM_BASE + 1U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(txtst::URAM_BASE + 2U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(txtst::URAM_BASE + 3U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(txtst::URAM_BASE + 4U * SPACING);
    txenc::emit32<txenc::vec_cmp_enc(txenc::VEC_CMP_INT32, 2, 0b100, 1, 18)>();
    txenc::emit32<txenc::vec_mux2_enc(2, 0, 17, 0)>();
    txenc::emit32<txenc::vec_vv_int_enc(txenc::VEC_INT_ADD, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(txtst::URAM_BASE + 5U * SPACING);
    txtst::start_vec_block(9);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(txtst::URAM_BASE + 5U * SPACING, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    auto out = reinterpret_cast<volatile int32_t *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const int32_t want = ((a[i] < b[i]) ? t[i] : f[i]) + add[i];
        if (out[i] != want) {
            printf("[vdep17] mismatch i=%u actual=%d expected=%d\n", i, out[i], want);
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep17] PASS: %u lanes, int32 cmp->mux2->add in one block\n", LANES);
    txtst::flush_ddr(txtst::OUT_DDR, KBYTES);
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
    .name = "vdep17",
});
