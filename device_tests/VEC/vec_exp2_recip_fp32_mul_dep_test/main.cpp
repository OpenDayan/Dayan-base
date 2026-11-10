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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_exp2_recip_fp32_mul_dep.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：单个 VEC 指令块内的两级 SFU 链 + FP32 乘 RAW 依赖——
 *   tx.vec.v.fp32.sfu.exp2 -> tx.vec.v.fp32.sfu.recip -> tx.vec.vv.fp32.mul
 * 6 条指令（load X、exp2 SFU、recip SFU、load scale、mul、store）一次
 * CONTROL 启动，中间无等待：exp2 结果写 bank0，recip 读 bank0 最新值再写回
 * bank0，mul 读 bank0 最新值（选择子 0）与 bank1 的 scale（选择子 16，age 0），
 * 验证 SFU 之间以及 SFU -> 乘法的生产者-消费者链路。
 *
 * 期望：out[i] == (1 / exp2(x[i])) * scale[i] = scale[i] / 2^x[i]，逐 lane 位
 * 对拍。输入公式与原样例一致：x 取整数指数 {-3..4} 循环、scale 取 8 个小尾数
 * 值循环。裸机无 libm，oracle 中 ldexp(1.0f, x) 换成 2^x 精确常量表（位级
 * 等价；除以 2 的幂仅改指数，精确）。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(float);   /* 256 */
static constexpr u64 X_DDR = txtst::DDR_BASE;
static constexpr u64 SCALE_DDR = txtst::DDR_BASE + 5U * KBYTES;
static constexpr u64 X_URAM = txtst::URAM_BASE;
static constexpr u64 SCALE_URAM = txtst::URAM_BASE + 5U * KBYTES;
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 10U * KBYTES;

/* 原样例 host 输入表（指数特意取整数 -3..4，2^x 全部精确可表）。 */
static const float X_VALUES[8] = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
static const float SCALE_VALUES[8] = {1.5f, 2.25f, 3.5f, 0.75f,
                                      5.0f, 1.25f, 0.625f, 2.75f};

/* 裸机无 libm：exp2(x) x∈{-3..4} 用 2^x 精确常量表，= 样例的 ldexp(1.0f, x)。 */
static const float POW2[8] = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f};

static u32 float_bits(float f)
{
    union { float f; u32 u; } v;
    v.f = f;
    return v.u;
}

static int same_bits(float a, float b)
{
    return float_bits(a) == float_bits(b);
}

static void tensor_run(void)
{
    auto x = reinterpret_cast<volatile float *>(X_DDR);
    auto scale = reinterpret_cast<volatile float *>(SCALE_DDR);

    for (unsigned i = 0; i < LANES; ++i) {
        x[i] = X_VALUES[i % 8U];
        scale[i] = SCALE_VALUES[i % 8U];
    }
    txtst::flush_ddr(X_DDR, KBYTES);
    txtst::flush_ddr(SCALE_DDR, KBYTES);

    /* 搬入 URAM（与原样例相同的整组线性配置 + 逐块 d2u，偏移沿用样例布局）。 */
    txtst::configure_lsu_linear(KBYTES);
    txtst::dma_dram_to_uram(X_DDR, X_URAM);
    txtst::dma_dram_to_uram(SCALE_DDR, SCALE_URAM);
    txtst::wait_lsu();

    /* VEC 全零描述符（样例只写 vec_sync/DIM/ImplDefHint/V0 迭代器，全零覆盖）。 */
    txtst::configure_vec_zero();

    /* 6 条指令排一个块：load X -> exp2 SFU -> recip SFU -> load scale -> mul
     * -> store。bank0: X/exp2/recip 结果 = V0；bank1: scale = V16(age 0)。 */
    txtst::wait_vec_queue_capacity(6);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(X_URAM);
    txenc::emit32<txenc::vec_sfu_enc(txenc::VEC_SFU_EXP2, false, 0, 0)>();
    txenc::emit32<txenc::vec_sfu_enc(txenc::VEC_SFU_RECIP, false, 0, 0)>();
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(SCALE_URAM);
    txenc::emit32<txenc::vec_vv_fp32_enc(txenc::VEC_FP32_MUL, 0, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(6);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    /* 设备侧逐 lane 位对拍（原样例 host oracle：scale / ldexp(1.0f, x)，
     * ldexp 项换成 2^x 常量表，位级等价——除以 2 的幂仅改指数，精确）。 */
    auto out = reinterpret_cast<volatile float *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const float xi = x[i];
        const float si = scale[i];
        const int exponent = (int)xi;
        const float expected = si / POW2[exponent + 3];
        if (!same_bits(out[i], expected)) {
            printf("[vdep13] mismatch i=%u actual=%08x expected=%08x\n",
                   i, float_bits(out[i]), float_bits(expected));
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep13] PASS: %u lanes, exp2 -> recip SFU -> fp32 mul in one block\n",
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
    .name = "vdep13",
});
