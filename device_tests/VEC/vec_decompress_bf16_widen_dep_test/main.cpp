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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_decompress_bf16_widen_dep.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图（原样例 VEC-DEP-003）：单个 VEC 指令块内的 shuffle -> cvt 依赖——
 *   tx.vec.shuffle.compress(ctrl=0x20, p=1 DeCompress) -> tx.vec.cvt.fp32.bf16
 * 输入 lane 0..31 是 32 个有效打包 BF16 字；DeCompress 把每个字展开到相邻
 * 两 lane，随后的宽化转换消费这些 lane 产生两个 FP32。输入 lane 32..63 是
 * 投毒数据（0xdead0000|i），不得影响结果。4 条指令一次 CONTROL 启动，
 * 中间无等待；全部走 bank0 最新值（选择子 0）。
 *
 * 期望：out[i] == bf16_to_f32(in[i/2] >> (16*(i&1)))，逐 lane 按位对拍。
 * 逻辑值公式与原样例一致：logical[i] = (float)((int)(i%19)-9)*0.4375f
 *                          + (float)(i%5)*0.0625f，再按 host 同款公式打包。
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
static constexpr u32 kCtrlDecompressP1 = 0x20;                  /* p=1 DeCompress */

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

static inline float bf16_to_f32(unsigned short value)
{
    union {
        float f;
        u32 b;
    } cvt;
    cvt.b = (u32)value << 16;
    return cvt.f;
}

static inline u32 float_bits(float value)
{
    union {
        float f;
        u32 b;
    } cvt;
    cvt.f = value;
    return cvt.b;
}

static void tensor_run(void)
{
    auto in = reinterpret_cast<volatile u32 *>(IN_DDR);

    /* 有效半段：logical 值按 host 同款公式打包；无效半段投毒。 */
    float logical[LANES];
    for (unsigned i = 0; i < LANES; ++i) {
        const int centered = (int)(i % 19U) - 9;
        logical[i] = (float)centered * 0.4375f + (float)(i % 5U) * 0.0625f;
    }
    for (unsigned k = 0; k < 32U; ++k) {
        const unsigned short lo = f32_to_bf16(logical[2U * k]);
        const unsigned short hi = f32_to_bf16(logical[2U * k + 1U]);
        in[k] = (u32)lo | ((u32)hi << 16);
    }
    for (unsigned k = 32U; k < LANES; ++k)
        in[k] = 0xdead0000U | k;
    txtst::flush_ddr(IN_DDR, KBYTES);

    /* 搬入 URAM（与原样例相同的整组线性配置 + d2u，含投毒半段）。 */
    txtst::configure_lsu_linear(KBYTES);
    txtst::dma_dram_to_uram(IN_DDR, IN_URAM);
    txtst::wait_lsu();

    /* VEC 全零描述符（含 vec_sync、v0 迭代器、ChannelStride 复位值）。 */
    txtst::configure_vec_zero();

    /* 4 条指令排一个块：u2v -> p=1 DeCompress -> BF16->FP32 宽化 -> v2u，
     * 全部 bank0 最新值（选择子 0）。 */
    txtst::wait_vec_queue_capacity(4);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(IN_URAM);
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>(
        static_cast<u64>(kCtrlDecompressP1));
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_FP32_BF16, 0, 0, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(4);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    /* 设备侧逐 lane 对拍（原样例 host oracle 同款公式，只认有效半段）。 */
    auto out = reinterpret_cast<volatile float *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const unsigned short packed =
            (unsigned short)(in[i / 2U] >> (16U * (i & 1U)));
        const float expected = bf16_to_f32(packed);
        if (float_bits(out[i]) != float_bits(expected)) {
            printf("[vdep10] mismatch i=%u actual=%08x expected=%08x\n",
                   i, float_bits(out[i]), float_bits(expected));
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep10] PASS: %u lanes, decompress -> bf16 widen in one block\n",
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
    .name = "vdep10",
});
