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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_shuffle_xor_dyn_add_dep.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：单个 VEC 指令块内的动态 shuffle -> ALU 跨操作 RAW 依赖——
 *   tx.vec.lsu.g2s -> tx.vec.shuffle.xor.dyn(s1) -> tx.vec.vv.fp32.add
 * 重排参数 kShuffleXorPair=3 先经 g2s 装入线程内标量寄存器 s1，再发无实参的
 * 动态 XOR-shuffle（lane i 取 a[i^3]），结果留在 bank0 V0；随后装 B 入
 * bank1（age0 = 选择子 16），fp32 add 消费 shuffle 结果。6 条指令一次
 * CONTROL 启动，中间无等待，验证标量通路与向量通路的块内依赖。
 *
 * 期望：out[i] == a[i ^ 3] + b[i]（逐 lane 按位对拍）。
 * 输入公式与原样例一致：a[i]=(float)(1000+13i)、b[i]=(float)(-300+7i)。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(float);          /* 256 */
static constexpr u64 SPACING = 5U * KBYTES;                   /* 原样例 a/b/out 间距 */
static constexpr u64 A_DDR = txtst::DDR_BASE;
static constexpr u64 B_DDR = txtst::DDR_BASE + 1U * SPACING;
static constexpr u64 A_URAM = txtst::URAM_BASE;
static constexpr u64 B_URAM = txtst::URAM_BASE + 1U * SPACING;
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 2U * SPACING;

static constexpr u32 kShuffleXorPair = 3U;                    /* 原样例 dyn ctrl 常量 */

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

    for (unsigned i = 0; i < LANES; ++i) {
        a[i] = (float)(1000 + (int)(13U * i));
        b[i] = (float)(-300 + (int)(7U * i));
    }
    txtst::flush_ddr(A_DDR, KBYTES);
    txtst::flush_ddr(B_DDR, KBYTES);

    /* 搬入 URAM（与原样例相同的整组线性配置 + 逐块 d2u）。 */
    txtst::configure_lsu_linear(KBYTES);
    txtst::dma_dram_to_uram(A_DDR, A_URAM);
    txtst::dma_dram_to_uram(B_DDR, B_URAM);
    txtst::wait_lsu();

    /* VEC 全零描述符（含 vec_sync、v0 迭代器、ChannelStride 复位值）。 */
    txtst::configure_vec_zero();

    /* 6 条指令排一个块：load A + g2s(重排参数入 s1) + shuffle.xor.dyn +
     * load B + fp32 add + store。
     * bank0: A 与 shuffle 结果 = V0；bank1: B = V16(age0)。 */
    txtst::wait_vec_queue_capacity(6);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(A_URAM);
    txenc::emit32_a0<txenc::vec_g2s_enc(1)>(static_cast<u64>(kShuffleXorPair));
    txenc::emit32<txenc::vec_shuffle_dyn_enc(txenc::VEC_SH_XOR_DYN, 1, 0, 0)>();
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(B_URAM);
    txenc::emit32<txenc::vec_vv_fp32_enc(txenc::VEC_FP32_ADD, txenc::RM_RNE, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(6);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    /* 设备侧逐 lane 对拍（原样例的 host oracle 同款公式，按位比较）。 */
    auto out = reinterpret_cast<volatile float *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const float expected = a[i ^ kShuffleXorPair] + b[i];
        if (!same_bits(out[i], expected)) {
            printf("[vdep26] mismatch i=%u actual=%08x expected=%08x\n",
                   i, float_bits(out[i]), float_bits(expected));
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep26] PASS: %u lanes, shuffle.xor.dyn(s1) -> fp32 add in one block\n", LANES);
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
    .name = "vdep26",
});
