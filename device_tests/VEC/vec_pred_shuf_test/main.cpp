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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_pred_shuf.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：单个 VEC 指令块内的谓词 -> 选择 -> 重排 -> 乘四操作依赖链——
 *   tx.vec.vv.fp32.cmp(Gt) -> tx.vec.vv.mux2 -> tx.vec.shuffle.xor -> tx.vec.vv.fp32.mul
 * 装载顺序决定年龄：bank0 依次装 a/f/t（a=V2 f=V1 t=V0），bank1 依次装
 * b/scale（b=V17 scale=V16）。cmp 求 a>b 写 64bit 谓词入 s2；mux2 按谓词
 * 逐 lane 取 t（bit=1 侧，vrs2=V0）/f（bit=0 侧，vrs1=V1）；shuffle.xor(1)
 * 交换相邻 lane（lane i 取 muxed[i^1]）；fp32 mul 乘 scale[i]。10 条指令
 * 一次 CONTROL 启动，中间无等待。
 *
 * 期望：src=(i^1)&63; selected=(a[src]>b[src])?t[src]:f[src];
 *       want=selected*scale[i]（逐 lane 按位对拍）。
 * 输入公式与原样例一致。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(float);          /* 256 */
static constexpr u64 SPACING = 5U * KBYTES;                   /* 原样例各 buffer 间距 */
static constexpr u64 A_DDR = txtst::DDR_BASE;
static constexpr u64 B_DDR = txtst::DDR_BASE + 1U * SPACING;
static constexpr u64 F_DDR = txtst::DDR_BASE + 2U * SPACING;
static constexpr u64 T_DDR = txtst::DDR_BASE + 3U * SPACING;
static constexpr u64 SCALE_DDR = txtst::DDR_BASE + 4U * SPACING;
static constexpr u64 A_URAM = txtst::URAM_BASE;
static constexpr u64 B_URAM = txtst::URAM_BASE + 1U * SPACING;
static constexpr u64 F_URAM = txtst::URAM_BASE + 2U * SPACING;
static constexpr u64 T_URAM = txtst::URAM_BASE + 3U * SPACING;
static constexpr u64 SCALE_URAM = txtst::URAM_BASE + 4U * SPACING;
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 5U * SPACING;

static constexpr int kCmpGt = 0b001;                          /* 原样例 cond 编码 */
static constexpr u32 kShuffleXorPair = 1U;                    /* 原样例 shuffle ctrl */

static inline u32 float_bits(float v)
{
    union { float f; u32 b; } u;
    u.f = v;
    return u.b;
}
static inline bool same_bits(float lhs, float rhs)
{
    return float_bits(lhs) == float_bits(rhs);
}

static void tensor_run(void)
{
    auto a = reinterpret_cast<volatile float *>(A_DDR);
    auto b = reinterpret_cast<volatile float *>(B_DDR);
    auto f = reinterpret_cast<volatile float *>(F_DDR);
    auto t = reinterpret_cast<volatile float *>(T_DDR);
    auto scale = reinterpret_cast<volatile float *>(SCALE_DDR);

    for (unsigned i = 0; i < LANES; ++i) {
        b[i] = (float)(int)(i % 16U) - 8.0f;
        a[i] = b[i] + ((i & 1U) ? 0.5f : -0.5f);
        f[i] = -1000.0f - (float)i;
        t[i] = 2000.0f + 2.0f * (float)i;
        scale[i] = 0.5f + (float)(i % 7U) * 0.25f;
    }
    txtst::flush_ddr(A_DDR, KBYTES);
    txtst::flush_ddr(B_DDR, KBYTES);
    txtst::flush_ddr(F_DDR, KBYTES);
    txtst::flush_ddr(T_DDR, KBYTES);
    txtst::flush_ddr(SCALE_DDR, KBYTES);

    /* 搬入 URAM（与原样例相同的整组线性配置 + 逐块 d2u）。 */
    txtst::configure_lsu_linear(KBYTES);
    txtst::dma_dram_to_uram(A_DDR, A_URAM);
    txtst::dma_dram_to_uram(B_DDR, B_URAM);
    txtst::dma_dram_to_uram(F_DDR, F_URAM);
    txtst::dma_dram_to_uram(T_DDR, T_URAM);
    txtst::dma_dram_to_uram(SCALE_DDR, SCALE_URAM);
    txtst::wait_lsu();

    /* VEC 全零描述符（含 vec_sync、v0 迭代器、ChannelStride 复位值）。 */
    txtst::configure_vec_zero();

    /* 10 条指令排一个块：5 loads + cmp + mux2 + shuffle.xor + mul + store。
     * bank0: a=V2(age2) f=V1(age1) t=V0(age0)；bank1: b=V17(age1) scale=V16(age0)。 */
    txtst::wait_vec_queue_capacity(10);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(A_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(B_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(F_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(T_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(SCALE_URAM);
    txenc::emit32<txenc::vec_cmp_enc(txenc::VEC_CMP_FP32, 2, kCmpGt, 2, 17)>();
    txenc::emit32<txenc::vec_mux2_enc(2, 1, 0, 0)>();
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_XOR, 0, 0)>(
        static_cast<u64>(kShuffleXorPair));
    txenc::emit32<txenc::vec_vv_fp32_enc(txenc::VEC_FP32_MUL, txenc::RM_RNE, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(10);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    /* 设备侧逐 lane 对拍（原样例的 host oracle 同款公式，按位比较）。 */
    auto out = reinterpret_cast<volatile float *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const u32 src = (i ^ 1U) & 63U;
        const float selected = (a[src] > b[src]) ? t[src] : f[src];
        const float want = selected * scale[i];
        if (!same_bits(out[i], want)) {
            printf("[vdep20] mismatch i=%u actual=%08x expected=%08x\n",
                   i, float_bits(out[i]), float_bits(want));
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep20] PASS: %u lanes, cmp->mux2->shuffle.xor->mul in one block\n", LANES);
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
    .name = "vdep20",
});
