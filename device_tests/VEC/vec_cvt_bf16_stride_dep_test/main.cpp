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

/* 从 dayan-samples/Instructions/VEC/dependency/vec_cvt_bf16_stride_dep.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图（VEC-DEP-027）：与 002 相同的 u2v -> BF16 narrow -> Compress -> v2u
 * 依赖链，但 V0 描述符非零：两个 DIM2 坐标的地址 =
 *   基址实参 + V0_Base(256) + d2 * V0_Dim2Stride(1024) + 4*lane（row=0,1）。
 * 每个 256B 活动行后跟 768B guard 区；选择子 0 由输入/输出共用（不同基址
 * 实参）；无 swizzle 改动。
 *
 * 输入区 2304B（= 256 前缀 + 2 行 × 1024 行距）全填 123.5f，两个活动行
 * （word 起点 (256+row*1024)/4）填样例公式；输出区 2304B 全填 kGuardWord
 * 并同样 d2u 种入 URAM，使错误描述符的越界写可见。
 *
 * 4 条指令一个块：u2v -> tx.vec.cvt（fp32->bf16，RNE；cvt_op=5，
 * 样例 intrinsic cvt_bf16_fp32，dst_first 命名）-> shuffle.compress（静态，
 * ctrl=0x00）-> v2u；原样例 CONTROL 0x0404040404 -> start_vec_block(4)。
 *
 * 对拍照原样例 host oracle：逐 word 检查全部 2304B（576 word）——活动行
 * word = (word-start)&31 选 pair，期望 = f32_to_bf16(in[start+2p]) |
 * f32_to_bf16(in[start+2p+1])<<16；其余 word 必须保持 guard 值。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned kElements = 64;
static constexpr unsigned kRows = 2;
static constexpr unsigned kPrefixBytes = 256;
static constexpr unsigned kRowStrideBytes = 1024;
static constexpr unsigned kStorageBytes = kPrefixBytes + kRows * kRowStrideBytes; /* 2304 */
static constexpr unsigned kStorageWords = kStorageBytes / sizeof(u32);            /* 576 */
static constexpr u32 kGuardWord = 0x5a93c76d;
static constexpr u64 kCtrlCompressP1 = 0x00;

static constexpr u64 IN_DDR = txtst::DDR_BASE;
static constexpr u64 IN_URAM = txtst::URAM_BASE;                       /* 原样例 in_uram */
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 4U * kStorageBytes; /* 原样例 out_uram */

/* f32 -> bf16（RNE）逐位照抄原样例 host helper（memcpy 换 union）。 */
static unsigned short f32_to_bf16(float value)
{
    union {
        float f;
        u32 u;
    } cvt;
    cvt.f = value;
    u32 bits = cvt.u;
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return (unsigned short)(bits >> 16);
}

static void tensor_run(void)
{
    auto in = reinterpret_cast<volatile float *>(IN_DDR);
    auto out_seed = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);

    /* 输入公式照抄原样例 host fill：全 123.5f + 两个活动行；输出区种 guard。 */
    for (unsigned word = 0; word < kStorageWords; ++word) {
        in[word] = 123.5f; /* 故意不是活动行取值。 */
        out_seed[word] = kGuardWord;
    }
    for (unsigned row = 0; row < kRows; ++row) {
        const unsigned start = (kPrefixBytes + row * kRowStrideBytes) / 4U;
        for (unsigned lane = 0; lane < kElements; ++lane) {
            const int centered = (int)(lane % 23U) - 11;
            in[start + lane] = (float)centered * 0.375f +
                (float)(lane % 7U) * 0.03125f + (float)row * 13.125f;
        }
    }
    txtst::flush_ddr(IN_DDR, kStorageBytes);
    txtst::flush_ddr(txtst::OUT_DDR, kStorageBytes);

    /* 两个区都 d2u（输出区用来种 guard）；整组线性配置按 kStorageBytes。 */
    txtst::configure_lsu_linear(kStorageBytes);
    txtst::dma_dram_to_uram(IN_DDR, IN_URAM);
    txtst::dma_dram_to_uram(txtst::OUT_DDR, OUT_URAM);
    txtst::wait_lsu();

    /* VEC 全零描述符后覆盖三项：Dim2=kRows-1、v0Base=kPrefixBytes、
     * v0Dim2Stride=kRowStrideBytes（v0 迭代器：Base=37，Dim0-5 Stride=38..43）。 */
    txtst::configure_vec_zero();
    txenc::write_tcsr<txenc::UNIT_VEC, 4>(kRows - 1U);
    txenc::write_tcsr<txenc::UNIT_VEC, 37>(kPrefixBytes);
    txenc::write_tcsr<txenc::UNIT_VEC, 40>(kRowStrideBytes);

    /* 4 条指令排一个块：u2v -> fp32->bf16 RNE -> p=1 Compress(ctrl=0x00) -> v2u。
     * 两行（DIM2 迭代）各自独立读写：地址 = 基址实参 + 256 + d2*1024。 */
    txtst::wait_vec_queue_capacity(4);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(IN_URAM);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_BF16_FP32, 0, 0, 0)>();
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>(kCtrlCompressP1);
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(4);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, kStorageBytes);

    /* 设备侧逐 word 对拍（原样例 host oracle 同款：活动行两个 packed 半区
     * + 全部未触碰 guard word）。 */
    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    unsigned active_words = 0, guard_words = 0, mismatches = 0;
    for (unsigned word = 0; word < kStorageWords; ++word) {
        u32 want = kGuardWord;
        bool active = false;
        for (unsigned row = 0; row < kRows; ++row) {
            const unsigned start = (kPrefixBytes + row * kRowStrideBytes) / 4U;
            if (word >= start && word < start + kElements) {
                const unsigned pair = (word - start) & 31U;
                want = (u32)f32_to_bf16(in[start + 2U * pair]) |
                    ((u32)f32_to_bf16(in[start + 2U * pair + 1U]) << 16);
                active = true;
            }
        }
        if (active)
            ++active_words;
        else
            ++guard_words;
        if (out[word] != want) {
            if (mismatches < 8U)
                printf("[vdep08] mismatch word=%u actual=%08x expected=%08x\n",
                       word, out[word], want);
            ++mismatches;
        }
    }
    if (mismatches != 0U) {
        printf("[vdep08] FAIL: rows=%u elements=%u guard_words=%u mismatches=%u\n",
               kRows, active_words, guard_words, mismatches);
        dayan_thread_exit(-22);
    }
    printf("[vdep08] rows=%u elements=%u guard_words=%u mismatches=%u\n",
           kRows, active_words, guard_words, mismatches);
    printf("[vdep08] PASS: descriptor-stride cvt -> bf16 compress dependency\n");
    txtst::flush_ddr(txtst::OUT_DDR, kStorageBytes);
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
    .name = "vdep08",
});
