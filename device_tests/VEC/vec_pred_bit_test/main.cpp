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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_pred_bit.cpp 迁移（裸机版，
 * 发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：单 VEC 块内双谓词组合链——
 *   fp32 GT cmp(S2) + fp32 LT cmp(S3) -> s.bitwise OR(S4=S2|S3) -> mux2 -> fp32 add
 * 共 13 条指令一次 CONTROL 启动，块内无等待。
 * bank0: a=V2 c=V1 f=V0；bank1: b=V19 d=V18 t=V17 bias=V16（age 编码进选择子）。
 * 原样例因编译器 builtin 的 s.bitwise LUT 编码 bug 用手编 .4byte 0xfa03b90b 绕过；
 * 本库编码正确，直接 txenc::vec_s_bitwise_enc(0xE, 2, 3, 4)（已验证位级等价）。
 *
 * 期望（按位比较）：use_t=(a>b)||(c<d); want=(use_t?t:f)+bias，逐 lane。
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
    auto c = reinterpret_cast<volatile float *>(txtst::DDR_BASE + 2U * SPACING);
    auto d = reinterpret_cast<volatile float *>(txtst::DDR_BASE + 3U * SPACING);
    auto f = reinterpret_cast<volatile float *>(txtst::DDR_BASE + 4U * SPACING);
    auto t = reinterpret_cast<volatile float *>(txtst::DDR_BASE + 5U * SPACING);
    auto bias = reinterpret_cast<volatile float *>(txtst::DDR_BASE + 6U * SPACING);

    for (unsigned i = 0; i < LANES; ++i) {
        b[i] = (float)i - 31.0f;
        a[i] = b[i] + ((i & 2U) ? 0.5f : -0.5f);
        d[i] = 17.0f - (float)i;
        c[i] = d[i] + ((i & 1U) ? -0.25f : 0.25f);
        f[i] = (float)(-1000 - (int)(3U * i));
        t[i] = (float)(2000 + (int)(5U * i));
        bias[i] = (float)(int)((i * 3U) % 11U) - 5.0f;
    }
    for (u64 region = 0; region < 7U; ++region)
        txtst::flush_ddr(txtst::DDR_BASE + region * SPACING, KBYTES);

    txtst::configure_lsu_linear(KBYTES);
    for (u64 region = 0; region < 7U; ++region) {
        txtst::dma_dram_to_uram(txtst::DDR_BASE + region * SPACING,
                                txtst::URAM_BASE + region * SPACING);
    }
    txtst::wait_lsu();

    txtst::configure_vec_zero();

    /* 13 条：7 load + 2 cmp + s.bitwise + mux2 + add + store。 */
    txtst::wait_vec_queue_capacity(13);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(txtst::URAM_BASE + 0U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(txtst::URAM_BASE + 1U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(txtst::URAM_BASE + 2U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(txtst::URAM_BASE + 3U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(txtst::URAM_BASE + 4U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(txtst::URAM_BASE + 5U * SPACING);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(txtst::URAM_BASE + 6U * SPACING);
    /* cmp<lhs,rhs,sd,cond>：a(V0 age2=2)>b(V1 age3=19) -> S2；c(V0 age1=1)<d(V1 age2=18) -> S3 */
    txenc::emit32<txenc::vec_cmp_enc(txenc::VEC_CMP_FP32, 2, 0b001, 2, 19)>();
    txenc::emit32<txenc::vec_cmp_enc(txenc::VEC_CMP_FP32, 3, 0b100, 1, 18)>();
    /* S4 = S2 | S3（LUT=0b1110），样例手编字 0xfa03b90b 的正确来源 */
    txenc::emit32<txenc::vec_s_bitwise_enc(0xE, 2, 3, 4)>();
    /* mux2: 谓词 S4 选 t(V17)/f(V0) -> V0；add: +bias(V16) -> V0 */
    txenc::emit32<txenc::vec_mux2_enc(4, 0, 17, 0)>();
    txenc::emit32<txenc::vec_vv_fp32_enc(txenc::VEC_FP32_ADD, txenc::RM_RNE, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(txtst::URAM_BASE + 7U * SPACING);
    txtst::start_vec_block(13);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(txtst::URAM_BASE + 7U * SPACING, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const bool use_t = (a[i] > b[i]) || (c[i] < d[i]);
        const float want = (use_t ? t[i] : f[i]) + bias[i];
        const u32 want_bits = bits_of(want);
        if (out[i] != want_bits) {
            printf("[vdep19] mismatch i=%u actual=%08x expected=%08x\n",
                   i, out[i], want_bits);
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep19] PASS: %u lanes, dual-cmp|bitwise-OR->mux2->add in one block\n", LANES);
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
    .name = "vdep19",
});
