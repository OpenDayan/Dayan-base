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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_fp16_pack_roundtrip.cpp 迁移
 * （裸机版，发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图（VEC-DEP-005）：单 VEC 块内 FP32 -> FP16 窄化 -> Compress -> DeCompress
 * -> FP16 宽化全链路 roundtrip，6 条指令一次 CONTROL 启动，块内无等待。
 * fp16 转换 helper 逐行照抄样例 host（RNE 含舍入进位/次正规/饱和分支）。
 * 期望（按位比较）：want=fp16_to_f32(f32_to_fp16(input[i]))，逐 lane。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(float);
static constexpr u64 IN_URAM = txtst::URAM_BASE;
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 4U * KBYTES;
static constexpr unsigned short kCtrlCompressP1 = 0x00;
static constexpr unsigned short kCtrlDecompressP1 = 0x20;

static u32 bits_of(float value)
{
    u32 bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float bits_to_f32(u32 bits)
{
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    return value;
}

static unsigned short f32_to_fp16(float value)
{
    const u32 bits = bits_of(value);
    const u32 sign = (bits >> 16) & 0x8000U;
    const u32 exponent = (bits >> 23) & 255U;
    const u32 mantissa = bits & 0x7fffffU;
    if (exponent == 255U)
        return (unsigned short)(sign | 0x7c00U | (mantissa ? 0x200U : 0));

    int half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31)
        return (unsigned short)(sign | 0x7c00U);
    if (half_exponent <= 0) {
        if (half_exponent < -10)
            return (unsigned short)sign;
        const u32 significand = mantissa | 0x800000U;
        const int shift = 14 - half_exponent;
        u32 half = significand >> shift;
        const u32 remainder = significand & ((1U << shift) - 1U);
        if (remainder > (1U << (shift - 1)) ||
            (remainder == (1U << (shift - 1)) && (half & 1U)))
            ++half;
        return (unsigned short)(sign | half);
    }

    u32 half_mantissa = mantissa >> 13;
    const u32 remainder = mantissa & 0x1fffU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U)))
        ++half_mantissa;
    if (half_mantissa == 0x400U) {
        half_mantissa = 0;
        if (++half_exponent >= 31)
            return (unsigned short)(sign | 0x7c00U);
    }
    return (unsigned short)(sign | ((u32)half_exponent << 10) | half_mantissa);
}

static float fp16_to_f32(unsigned short value)
{
    const u32 sign = (u32)(value & 0x8000U) << 16;
    const u32 exponent = (value >> 10) & 31U;
    u32 mantissa = value & 1023U;
    u32 bits;

    if (!exponent) {
        if (!mantissa) {
            bits = sign;
        } else {
            int exponent_adjustment = -14;
            while (!(mantissa & 1024U)) {
                mantissa <<= 1;
                --exponent_adjustment;
            }
            mantissa &= 1023U;
            bits = sign | (u32)(exponent_adjustment + 127) << 23 | (mantissa << 13);
        }
    } else if (exponent == 31U) {
        bits = sign | 0x7f800000U | (mantissa << 13);
    } else {
        bits = sign | (exponent + 112U) << 23 | (mantissa << 13);
    }
    return bits_to_f32(bits);
}

static void tensor_run(void)
{
    auto input = reinterpret_cast<volatile float *>(txtst::DDR_BASE);

    for (unsigned i = 0; i < LANES; ++i) {
        const int centered = (int)(i % 29U) - 14;
        input[i] = (float)centered * 0.28125f + (float)(i % 11U) * 0.013671875f;
    }
    txtst::flush_ddr(txtst::DDR_BASE, KBYTES);

    txtst::configure_lsu_linear(KBYTES);
    txtst::dma_dram_to_uram(txtst::DDR_BASE, IN_URAM);
    txtst::wait_lsu();

    txtst::configure_vec_zero();

    /* 6 条：load -> 窄化 -> Compress(0x00) -> DeCompress(0x20) -> 宽化 -> store。 */
    txtst::wait_vec_queue_capacity(6);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(IN_URAM);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_FP16_FP32, txenc::RM_RNE, 0, 0)>();
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>((u64)kCtrlCompressP1);
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>((u64)kCtrlDecompressP1);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_FP32_FP16, txenc::RM_RNE, 0, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(6);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const u32 want_bits = bits_of(fp16_to_f32(f32_to_fp16(input[i])));
        if (out[i] != want_bits) {
            printf("[vdep14] mismatch i=%u actual=%08x expected=%08x\n",
                   i, out[i], want_bits);
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep14] PASS: %u lanes, fp16 pack/compress/decompress/widen roundtrip\n", LANES);
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
    .name = "vdep14",
});
