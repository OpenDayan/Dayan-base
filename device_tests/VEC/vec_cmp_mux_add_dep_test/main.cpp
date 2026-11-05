/* 从 dayan-samples/Instructions/VEC/dependency/vec_cmp_mux_add_dep.cpp 迁移
 * （裸机版，发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：单 VEC 块内 fp32 比较 -> 谓词 -> mux2 -> fp32 add，9 条指令一次
 * CONTROL 启动，块内无等待。bank0: a=V1 f=V0；bank1: b=V18 t=V17 addend=V16。
 * 期望（按位比较）：want=((a[i]>b[i])?t[i]:f[i])+addend[i]，逐 lane。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(float);
static constexpr u64 SPACING = 5U * KBYTES;

static u32 bits_of(float value)
{
    u32 bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void tensor_run(void)
{
    auto a = reinterpret_cast<volatile float *>(txtst::DDR_BASE);
    auto b = reinterpret_cast<volatile float *>(txtst::DDR_BASE + 1U * SPACING);
    auto f = reinterpret_cast<volatile float *>(txtst::DDR_BASE + 2U * SPACING);
    auto t = reinterpret_cast<volatile float *>(txtst::DDR_BASE + 3U * SPACING);
    auto addend = reinterpret_cast<volatile float *>(txtst::DDR_BASE + 4U * SPACING);

    for (unsigned i = 0; i < LANES; ++i) {
        a[i] = (float)(int)((i * 5U) % 17U) - 8.0f;
        b[i] = (float)(int)((i * 7U) % 19U) - 9.0f;
        f[i] = (float)(int)(-1000 - (int)(3U * i));
        t[i] = (float)(int)(2000 + (int)(5U * i));
        addend[i] = (float)(int)((i * 3U) % 11U) - 5.0f;
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

    /* 9 条：5 load + cmp + mux2 + add + store。
     * 加载序 a(V0) b(V1) f(V0) t(V1) addend(V1) -> a=V1 f=V0；b=V18 t=V17 addend=V16。 */
    txtst::wait_vec_queue_capacity(9);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(txtst::URAM_BASE + 0U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(txtst::URAM_BASE + 1U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(txtst::URAM_BASE + 2U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(txtst::URAM_BASE + 3U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(txtst::URAM_BASE + 4U * SPACING);
    txenc::emit32<txenc::vec_cmp_enc(txenc::VEC_CMP_FP32, 2, 0b001, 1, 18)>();
    txenc::emit32<txenc::vec_mux2_enc(2, 0, 17, 0)>();
    txenc::emit32<txenc::vec_vv_fp32_enc(txenc::VEC_FP32_ADD, txenc::RM_RNE, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(txtst::URAM_BASE + 5U * SPACING);
    txtst::start_vec_block(9);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(txtst::URAM_BASE + 5U * SPACING, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const float selected = (a[i] > b[i]) ? t[i] : f[i];
        const u32 want_bits = bits_of(selected + addend[i]);
        if (out[i] != want_bits) {
            printf("[vdep05] mismatch i=%u actual=%08x expected=%08x\n",
                   i, out[i], want_bits);
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep05] PASS: %u lanes, fp32 cmp->mux2->add in one block\n", LANES);
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
    .name = "vdep05",
});
