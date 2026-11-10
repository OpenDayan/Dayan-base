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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_cvt_bf16_compress_dep.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图（原样例 VEC-DEP-002）：单个 VEC 指令块内的 cvt -> shuffle 依赖——
 *   tx.vec.cvt.bf16.fp32（窄化：FP32 lane 2k/2k+1 两个 BF16 打包进偶数目的
 *   lane 2k） -> tx.vec.shuffle.compress(ctrl=0x00, p=1)
 * compress 收集偶数 lane 的打包字成 32 字连续块并在 V0 两半复制。
 * 4 条指令（1 load + 1 cvt + 1 compress + 1 store）一次 CONTROL 启动，
 * 中间无等待；全部走 bank0 最新值（选择子 0）。
 *
 * 期望：out[i] == expected[i & 31]，其中
 *   expected[k] = bf16(in[2k]) | (bf16(in[2k+1]) << 16)  (k < 32, RNE)
 * 输入公式与原样例一致：in[i] = (float)((int)(i%23)-11)*0.375f
 *                        + (float)(i%7)*0.03125f（正负 RNE 均覆盖）。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(u32);              /* 256 */
static constexpr u64 IN_DDR = txtst::DDR_BASE;
static constexpr u64 IN_URAM = txtst::URAM_BASE;
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 4U * KBYTES; /* 原样例布局 */
static constexpr u32 kCtrlCompressP1 = 0x00;                    /* p=1 Compress */

/* 原样例 host helper 逐行照抄（memcpy 换 union，语义不变）。 */
static inline unsigned short f32_to_bf16(float value)
{
    union {
        float f;
        u32 b;
    } cvt;
    cvt.f = value;
    u32 bits = cvt.b;
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return (unsigned short)(bits >> 16);
}

static void tensor_run(void)
{
    auto in = reinterpret_cast<volatile float *>(IN_DDR);

    /* 相邻 lane 互异，正负 RNE 均有覆盖（原样例 host fill 公式）。 */
    for (unsigned i = 0; i < LANES; ++i) {
        const int centered = (int)(i % 23U) - 11;
        in[i] = (float)centered * 0.375f + (float)(i % 7U) * 0.03125f;
    }
    txtst::flush_ddr(IN_DDR, KBYTES);

    /* 搬入 URAM（与原样例相同的整组线性配置 + d2u）。 */
    txtst::configure_lsu_linear(KBYTES);
    txtst::dma_dram_to_uram(IN_DDR, IN_URAM);
    txtst::wait_lsu();

    /* VEC 全零描述符（含 vec_sync、v0 迭代器、ChannelStride 复位值）。 */
    txtst::configure_vec_zero();

    /* 4 条指令排一个块：u2v -> FP32->BF16 窄化 -> p=1 Compress -> v2u，
     * 全部 bank0 最新值（选择子 0）。 */
    txtst::wait_vec_queue_capacity(4);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(IN_URAM);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_BF16_FP32, 0, 0, 0)>();
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>(
        static_cast<u64>(kCtrlCompressP1));
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(4);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    /* 设备侧逐 lane 对拍（原样例 host oracle 同款公式：两半复制取 i&31）。 */
    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    u32 expected[32];
    for (unsigned k = 0; k < 32U; ++k) {
        expected[k] = (u32)f32_to_bf16(in[2U * k]) |
                      ((u32)f32_to_bf16(in[2U * k + 1U]) << 16);
    }
    for (unsigned i = 0; i < LANES; ++i) {
        const u32 want = expected[i & 31U];
        if (out[i] != want) {
            printf("[vdep06] mismatch i=%u actual=%08x expected=%08x\n",
                   i, out[i], want);
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep06] PASS: %u lanes, cvt fp32->bf16 -> compress in one block\n",
           LANES);
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
    .name = "vdep06",
});
