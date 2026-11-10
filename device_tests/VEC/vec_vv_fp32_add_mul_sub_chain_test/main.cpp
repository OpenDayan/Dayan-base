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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_vv_fp32_add_mul_sub_chain.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：单个 VEC 指令块内的三级跨操作 RAW 依赖——
 *   tx.vec.vv.fp32.add -> tx.vec.vv.fp32.mul -> tx.vec.vv.fp32.sub
 * A 装入 bank0（最新值 = 选择子 0）；B/C/D 装入 bank1，age 分别为 2/1/0
 * （选择子 18/17/16）。四条 load 与三条运算、一条 store 共 8 条指令排同一块，
 * 一次 CONTROL 启动，中间无等待，验证块内生产者-消费者链路。
 *
 * 期望：out[i] == ((a[i] + b[i]) * c[i]) - d[i]，
 * 逐 lane 按位对拍（same_bits，裸机无 libm，打印 %08x 位值）。
 * 输入公式与原样例一致：a=(i%11-5)*0.5、b=(i%7-3)*0.25、
 * c=(1+(i%5))*0.5、d=(i%13-6)*0.25。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(float);           /* 256 */
static constexpr u64 SPACING = 5U * KBYTES;                    /* 原样例的 guard 间隔 */
static constexpr u64 A_DDR = txtst::DDR_BASE;
static constexpr u64 B_DDR = txtst::DDR_BASE + 1U * SPACING;
static constexpr u64 C_DDR = txtst::DDR_BASE + 2U * SPACING;
static constexpr u64 D_DDR = txtst::DDR_BASE + 3U * SPACING;
static constexpr u64 A_URAM = txtst::URAM_BASE;
static constexpr u64 B_URAM = txtst::URAM_BASE + 1U * SPACING;
static constexpr u64 C_URAM = txtst::URAM_BASE + 2U * SPACING;
static constexpr u64 D_URAM = txtst::URAM_BASE + 3U * SPACING;
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 4U * SPACING;

/* float 一律按位比较（与原样例的逐位 memcmp 语义一致）。 */
static u32 fbits(float f)
{
    union {
        float f;
        u32 u;
    } v;
    v.f = f;
    return v.u;
}

static int same_bits(float x, float y)
{
    return fbits(x) == fbits(y);
}

static void tensor_run(void)
{
    auto a = reinterpret_cast<volatile float *>(A_DDR);
    auto b = reinterpret_cast<volatile float *>(B_DDR);
    auto c = reinterpret_cast<volatile float *>(C_DDR);
    auto d = reinterpret_cast<volatile float *>(D_DDR);

    for (unsigned i = 0; i < LANES; ++i) {
        a[i] = (float)((int)(i % 11U) - 5) * 0.5f;
        b[i] = (float)((int)(i % 7U) - 3) * 0.25f;
        c[i] = (float)(1U + (i % 5U)) * 0.5f;
        d[i] = (float)((int)(i % 13U) - 6) * 0.25f;
    }
    txtst::flush_ddr(A_DDR, KBYTES);
    txtst::flush_ddr(B_DDR, KBYTES);
    txtst::flush_ddr(C_DDR, KBYTES);
    txtst::flush_ddr(D_DDR, KBYTES);

    /* 搬入 URAM（与原样例相同的整组线性配置 + 逐块 d2u）。 */
    txtst::configure_lsu_linear(KBYTES);
    txtst::dma_dram_to_uram(A_DDR, A_URAM);
    txtst::dma_dram_to_uram(B_DDR, B_URAM);
    txtst::dma_dram_to_uram(C_DDR, C_URAM);
    txtst::dma_dram_to_uram(D_DDR, D_URAM);
    txtst::wait_lsu();

    /* VEC 全零描述符（含 vec_sync、v0 迭代器、ChannelStride 复位值）。 */
    txtst::configure_vec_zero();

    /* 8 条指令排一个块：4 loads + add + mul + sub + store。
     * bank0: a=V0；bank1: b=V18(age2) c=V17(age1) d=V16(age0)。 */
    txtst::wait_vec_queue_capacity(8);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(A_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(B_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(C_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(D_URAM);
    txenc::emit32<txenc::vec_vv_fp32_enc(txenc::VEC_FP32_ADD, txenc::RM_RNE, 0, 18, 0)>();
    txenc::emit32<txenc::vec_vv_fp32_enc(txenc::VEC_FP32_MUL, txenc::RM_RNE, 0, 17, 0)>();
    txenc::emit32<txenc::vec_vv_fp32_enc(txenc::VEC_FP32_SUB, txenc::RM_RNE, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(8);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    /* 设备侧逐 lane 对拍（原样例的 host oracle 同款公式）。 */
    auto out = reinterpret_cast<volatile float *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const float expected = ((a[i] + b[i]) * c[i]) - d[i];
        if (!same_bits(out[i], expected)) {
            printf("[vdep27] mismatch i=%u actual=%08x expected=%08x\n",
                   i, fbits(out[i]), fbits(expected));
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep27] PASS: %u lanes, fp32 add->mul->sub chain in one block\n", LANES);
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
    .name = "vdep27",
});
