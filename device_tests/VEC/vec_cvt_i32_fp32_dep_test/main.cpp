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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_cvt_i32_fp32_dep.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图（VEC-DEP-022）：单个 VEC 指令块内的跨类型 RAW 依赖——
 *   FP32 -> INT32 -> INT32 add -> FP32：
 *   u2v(src fp32, bank0) -> u2v(addend int32, bank1) ->
 *   tx.vec.cvt（fp32->s32，cvt_op=1）-> tx.vec.vv.int32.add（vrs2 选择子 16
 *   = bank1 age0）-> tx.vec.cvt（int32->fp32，cvt_op=0）-> v2u。
 * 样例注：Convert 模板名目的在前（dst_first），intrinsic 名与 txenc 枚举名
 * 一一对应（cvt_s32_fp32 -> VEC_CVT_S32_FP32=1，cvt_fp32_int32 ->
 * VEC_CVT_FP32_INT32=0）。
 *
 * 6 条指令一次 CONTROL 启动（原样例 0x0606060606 -> start_vec_block(6)）。
 * 输入公式与原样例一致：src[i]=(float)((int)(i%31)-15)、
 * add[i]=(int32_t)((int)(i%9)-4)；
 * 期望：out[i] 按位 == (float)((int)(i%31)-15 + (int)(i%9)-4)（纯整数和的
 * 精确 fp32 表示，设备侧按位对拍）。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(u32);            /* 256 = 原样例 kBytes */
static constexpr u64 SPACING = 5U * KBYTES;                   /* 原样例 URAM 布局间隔 */
static constexpr u64 SRC_DDR = txtst::DDR_BASE;
static constexpr u64 ADD_DDR = txtst::DDR_BASE + 1U * SPACING;
static constexpr u64 SRC_URAM = txtst::URAM_BASE;             /* 原样例 src_u */
static constexpr u64 ADD_URAM = txtst::URAM_BASE + 5U * KBYTES;   /* 原样例 add_u */
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 10U * KBYTES;  /* 原样例 out_u */

static u32 float_bits(float value)
{
    union {
        float f;
        u32 u;
    } cvt;
    cvt.f = value;
    return cvt.u;
}

static void tensor_run(void)
{
    auto src = reinterpret_cast<volatile float *>(SRC_DDR);
    auto add = reinterpret_cast<volatile int *>(ADD_DDR);

    /* 输入公式照抄原样例 host fill。 */
    for (unsigned i = 0; i < LANES; ++i) {
        src[i] = (float)((int)(i % 31U) - 15);
        add[i] = (int)(i % 9U) - 4;
    }
    txtst::flush_ddr(SRC_DDR, KBYTES);
    txtst::flush_ddr(ADD_DDR, KBYTES);

    /* 搬入 URAM（与原样例相同的整组线性配置 + 逐块 d2u）。 */
    txtst::configure_lsu_linear(KBYTES);
    txtst::dma_dram_to_uram(SRC_DDR, SRC_URAM);
    txtst::dma_dram_to_uram(ADD_DDR, ADD_URAM);
    txtst::wait_lsu();

    /* VEC 全零描述符（含 vec_sync、v0 迭代器、ChannelStride 复位值）。 */
    txtst::configure_vec_zero();

    /* 6 条指令排一个块：2 loads + fp32->s32 cvt + int32 add + s32->fp32 cvt
     * + store。bank0: src=V0；bank1: addend=V16（age0）。 */
    txtst::wait_vec_queue_capacity(6);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(SRC_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(ADD_URAM);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_S32_FP32, 0, 0, 0)>();
    txenc::emit32<txenc::vec_vv_int_enc(txenc::VEC_INT_ADD, 0, 16, 0)>();
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_FP32_INT32, 0, 0, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(6);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    /* 设备侧逐 lane 按位对拍（原样例 host oracle 同款公式；memcmp 语义）。 */
    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    unsigned mismatches = 0;
    for (unsigned i = 0; i < LANES; ++i) {
        const u32 expected = float_bits((float)((int)(i % 31U) - 15 + (int)(i % 9U) - 4));
        if (out[i] != expected) {
            if (mismatches < 12U)
                printf("[vdep09] mismatch i=%u actual=%08x expected=%08x\n",
                       i, out[i], expected);
            ++mismatches;
        }
    }
    if (mismatches != 0U) {
        printf("[vdep09] FAIL: elements=%u mismatches=%u\n", LANES, mismatches);
        dayan_thread_exit(-22);
    }
    printf("[vdep09] PASS: %u elements, fp32->s32 -> int32.add -> fp32 in one block\n",
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
    .name = "vdep09",
});
