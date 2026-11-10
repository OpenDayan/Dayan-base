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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_recip_exp2_fp32_add_dep.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：单个 VEC 指令块内的两级 SFU 链 + FP32 加 RAW 依赖——
 *   tx.vec.v.fp32.sfu.recip -> tx.vec.v.fp32.sfu.exp2 -> tx.vec.vv.fp32.add
 * 6 条指令（load X、recip SFU、exp2 SFU、load addend、add、store）一次
 * CONTROL 启动，中间无等待：recip 结果写 bank0，exp2 读 bank0 最新值再写回
 * bank0，add 读 bank0 最新值（选择子 0）与 bank1 的 addend（选择子 16，age 0）。
 *
 * 期望：out[i] == exp2(1.0f / x[i]) + addend[i]，逐 lane 位对拍。输入公式与
 * 原样例一致：x 取 2 的幂 {0.5..4} 循环、addend 取 {0.25,-0.5,1.5,-2.25}
 * 循环。裸机无 libm（禁 exp2f）：1/x 精确为 {2,1,0.5,0.25}，exp2(1/x) 用查表
 * ——4.0/2.0 精确，exp2(0.5)=√2 与 exp2(0.25)=2^0.25 取正确舍入的 fp32 常量
 * （0x3fb504f3 / 0x3f9837f0，与样例 host exp2f 位级一致）；其后的 add 是普通
 * IEEE 运算，C 表达式直接移植。
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

/* 原样例 host 输入表（x 全部为 2 的整数幂，倒数精确可表）。 */
static const float X_VALUES[4] = {0.5f, 1.0f, 2.0f, 4.0f};
static const float SCALE_VALUES[4] = {0.25f, -0.5f, 1.5f, -2.25f};

/* 裸机无 libm：样例输入下 1/x 精确为 {2.0, 1.0, 0.5f, 0.25f}，exp2(1/x)
 * 查表：4.0/2.0 精确；√2、2^0.25 为无理数，取正确舍入的 fp32 常量
 * （1.41421353816986083984375f = 0x3fb504f3、1.1892070770263671875f =
 * 0x3f9837f0，与样例 host std::exp2 的位级结果一致）。 */
static float exp2_of_pow2(float q)
{
    if (q == 2.0f)
        return 4.0f;                            /* exp2(2)   = 0x40800000 */
    if (q == 1.0f)
        return 2.0f;                            /* exp2(1)   = 0x40000000 */
    if (q == 0.5f)
        return 1.41421353816986083984375f;      /* exp2(0.5) = 0x3fb504f3 */
    return 1.1892070770263671875f;              /* exp2(0.25)= 0x3f9837f0 */
}

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
        x[i] = X_VALUES[i % 4U];
        scale[i] = SCALE_VALUES[i % 4U];
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

    /* 6 条指令排一个块：load X -> recip SFU -> exp2 SFU -> load addend -> add
     * -> store。bank0: X/recip/exp2 结果 = V0；bank1: addend = V16(age 0)。 */
    txtst::wait_vec_queue_capacity(6);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(X_URAM);
    txenc::emit32<txenc::vec_sfu_enc(txenc::VEC_SFU_RECIP, false, 0, 0)>();
    txenc::emit32<txenc::vec_sfu_enc(txenc::VEC_SFU_EXP2, false, 0, 0)>();
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(SCALE_URAM);
    txenc::emit32<txenc::vec_vv_fp32_enc(txenc::VEC_FP32_ADD, 0, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(6);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    /* 设备侧逐 lane 位对拍（原样例 host oracle：exp2(1.0f / x) + scale，
     * exp2 换查表常量、除法照抄——x 为 2 的幂，1/x 精确；add 为 IEEE 运算）。 */
    auto out = reinterpret_cast<volatile float *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const float q = 1.0f / x[i];
        const float expected = exp2_of_pow2(q) + scale[i];
        if (!same_bits(out[i], expected)) {
            printf("[vdep21] mismatch i=%u actual=%08x expected=%08x\n",
                   i, float_bits(out[i]), float_bits(expected));
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep21] PASS: %u lanes, recip -> exp2 SFU -> fp32 add in one block\n",
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
    .name = "vdep21",
});
