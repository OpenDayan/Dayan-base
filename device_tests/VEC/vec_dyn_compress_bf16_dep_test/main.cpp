/* 从 dayan-samples/Instructions/VEC/dependency/vec_dyn_compress_bf16_dep.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图（VEC-DEP-025）：static/dynamic Compress -> DeCompress -> BF16 widen。
 * 原样例 kernel 带运行时参数（compress_ctrl, decompress_ctrl），合法对
 * (0x00,0x20) 与 (0x04,0x24)，host 分两次 launch 各跑一对；每对一个
 * 16 条指令的 VEC 块（原样例 CONTROL 0x101010101010），五个输出段：
 *   0=static packed、1=dynamic packed、2=static expanded、3=dynamic
 *   expanded、4=widened dynamic。v2u 与 g2s 不推进向量队列 age；static
 *   DeCompress 之后 dynamic DeCompress 读 age1（packed）而非 age0（已
 *   expanded），两路馈入都显式对拍。
 *
 * 裸机版差异：一个 tensor_run 顺序跑两个 ctrl 对——配置一次，每对排
 * 16 条指令 + start_vec_block(16) + wait_vec（样例源码里 if/else 二选一，
 * 这里编译期两个都写）。第二对的 5 段输出放 out_uram + 5*kBytes 起，
 * DDR 输出共 10 段 = 2560B 连续放 OUT_DDR 起，compare 全覆盖。
 *
 * 指令序列严格照样例（含两条 g2s 到 s1/s2、compress_dyn 读 s、static
 * compress/decompress 混排）。样例 intrinsic 名目的在前（dst_first）：
 * cvt_bf16_fp32（fp32->bf16 narrow）= VEC_CVT_BF16_FP32=5，
 * cvt_fp32_bf16（bf16->fp32 widen）= VEC_CVT_FP32_BF16=3。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned kElements = 64;
static constexpr u64 kBytes = kElements * sizeof(u32);          /* 256 = 原样例 kBytes */
static constexpr unsigned kOutputWords = 5U * kElements;        /* 320 / 对 */
static constexpr u64 kCaseBytes = kOutputWords * sizeof(u32);   /* 1280 / 对 */
static constexpr u64 kTotalOutBytes = 2U * kCaseBytes;          /* 2560 两对合计 */

static constexpr u64 IN_DDR = txtst::DDR_BASE;
static constexpr u64 IN_URAM = txtst::URAM_BASE;                /* 原样例 in_uram */
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 4U * kBytes; /* 原样例 out_uram */

/* ---- f32/bf16 位运算 helper 逐行照抄原样例 host（memcpy 换 union） ---- */
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

static float bf16_to_f32(unsigned short value)
{
    union {
        float f;
        u32 u;
    } cvt;
    cvt.u = (u32)value << 16;
    return cvt.f;
}

static u32 float_bits(float value)
{
    union {
        float f;
        u32 u;
    } cvt;
    cvt.f = value;
    return cvt.u;
}

/* 一个 ctrl 对的 16 条指令（严格照样例序列；ctrl 立即数经 a0 传入）。 */
static void run_ctrl_pair(u64 out_base, u64 compress_ctrl, u64 decompress_ctrl)
{
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(IN_URAM);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_BF16_FP32, 0, 0, 0)>();
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>(compress_ctrl);
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(out_base);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(IN_URAM);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_BF16_FP32, 0, 0, 0)>();
    txenc::emit32_a0<txenc::vec_g2s_enc(1)>(compress_ctrl);
    txenc::emit32<txenc::vec_shuffle_dyn_enc(txenc::VEC_SH_COMPRESS_DYN, 1, 0, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(out_base + 1U * kBytes);
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>(decompress_ctrl);
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(out_base + 2U * kBytes);
    txenc::emit32_a0<txenc::vec_g2s_enc(2)>(decompress_ctrl);
    txenc::emit32<txenc::vec_shuffle_dyn_enc(txenc::VEC_SH_COMPRESS_DYN, 2, 1, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(out_base + 3U * kBytes);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_FP32_BF16, 0, 0, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(out_base + 4U * kBytes);
}

static void tensor_run(void)
{
    auto in = reinterpret_cast<volatile float *>(IN_DDR);

    /* 输入公式照抄原样例 host fill（有限、互异，绕开 NaN/Inf 的 RNE）。 */
    for (unsigned i = 0; i < kElements; ++i) {
        const int centered = (int)(i % 29U) - 14;
        in[i] = (float)centered * 0.28125f + (float)(i % 11U) * 0.013671875f;
    }
    txtst::flush_ddr(IN_DDR, kBytes);

    txtst::configure_lsu_linear(kBytes);
    txtst::dma_dram_to_uram(IN_DDR, IN_URAM);
    txtst::wait_lsu();

    /* VEC 全零描述符（一条 64-lane fp32 向量，bank0，无 swizzle），两个
     * ctrl 对共用，仅配置一次。 */
    txtst::configure_vec_zero();

    /* 第一对 (0x00, 0x20)：输出段 0..4*kBytes。 */
    txtst::wait_vec_queue_capacity(16);
    run_ctrl_pair(OUT_URAM, 0x00U, 0x20U);
    txtst::start_vec_block(16);
    txtst::wait_vec();

    /* 第二对 (0x04, 0x24)：输出段 5..9*kBytes。 */
    txtst::wait_vec_queue_capacity(16);
    run_ctrl_pair(OUT_URAM + 5U * kBytes, 0x04U, 0x24U);
    txtst::start_vec_block(16);
    txtst::wait_vec();

    /* 两对的 10 段共 2560B 一次 u2d 搬回（原样例仅改 LSU_DIM0 后整段搬）。 */
    txtst::configure_lsu_linear(kTotalOutBytes);
    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, kTotalOutBytes);

    /* 设备侧逐段对拍（原样例 host oracle 同款：5 段输出/对 + 静态/动态
     * 一致性 parity）。packed/expanded 为栈上临时表。 */
    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    u32 packed[kElements];
    u32 expanded[kElements];
    unsigned mismatches = 0;
    for (unsigned cfg = 0; cfg < 2U; ++cfg) {
        unsigned case_mismatches = 0;
        for (unsigned i = 0; i < kElements; ++i) {
            const unsigned pair = cfg == 0U ? (i & 31U) : (i & 1U) * 16U;
            packed[i] = (u32)f32_to_bf16(in[2U * pair]) |
                ((u32)f32_to_bf16(in[2U * pair + 1U]) << 16);
        }
        for (unsigned i = 0; i < kElements; ++i)
            expanded[i] = packed[cfg == 0U ? i / 2U : i / 32U];
        for (unsigned stage = 0; stage < 5U; ++stage) {
            for (unsigned i = 0; i < kElements; ++i) {
                const unsigned src = cfg == 0U ? i :
                    (i < 32U ? 0U : 32U) + (i & 1U);
                const u32 want = stage < 2U ? packed[i] :
                    stage < 4U ? expanded[i] :
                    float_bits(bf16_to_f32(f32_to_bf16(in[src])));
                const u32 actual = out[cfg * 5U * kElements + stage * kElements + i];
                /* 独立 host oracle + 静态/动态 parity 双重检查。 */
                const bool parity = stage == 1U ?
                    actual == out[cfg * 5U * kElements + i] :
                    stage == 3U ?
                    actual == out[cfg * 5U * kElements + 2U * kElements + i] : true;
                if (actual != want || !parity) {
                    if (case_mismatches < 8U)
                        printf("[vdep11] mismatch cfg=%u stage=%u lane=%u "
                               "actual=%08x expected=%08x\n",
                               cfg, stage, i, actual, want);
                    ++case_mismatches;
                }
            }
        }
        mismatches += case_mismatches;
        printf("[vdep11] compress=0x%02x decompress=0x%02x elements=%u mismatches=%u\n",
               cfg == 0U ? 0x00U : 0x04U, cfg == 0U ? 0x20U : 0x24U,
               kOutputWords, case_mismatches);
    }
    if (mismatches != 0U) {
        printf("[vdep11] FAIL: cases=2 elements=%u mismatches=%u\n",
               2U * kOutputWords, mismatches);
        dayan_thread_exit(-22);
    }
    printf("[vdep11] cases=2 elements=%u mismatches=%u\n", 2U * kOutputWords, mismatches);
    printf("[vdep11] PASS: dynamic compress -> decompress -> bf16 dependency (two ctrl pairs)\n");
    txtst::flush_ddr(txtst::OUT_DDR, kTotalOutBytes);
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
    .name = "vdep11",
});
