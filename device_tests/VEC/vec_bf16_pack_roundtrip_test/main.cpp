/* 从 dayan-samples/Instructions/VEC/dependency/vec_bf16_pack_roundtrip.cpp 迁移
 * （裸机版，发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图（VEC-DEP-004）：单 VEC 块内 FP32 -> BF16 窄化 -> Compress -> DeCompress
 * -> BF16 宽化全链路 roundtrip，6 条指令一次 CONTROL 启动，块内无等待。
 * 窄化把相邻两 lane 打包进偶数 lane；Compress 收集成连续 32 字块；DeCompress
 * 还原成对 lane；宽化消费 lane 对。输入为有限且互异的值，覆盖 RNE 无 NaN/Inf。
 * 期望（按位比较）：want=bf16_to_f32(f32_to_bf16(input[i]))，逐 lane。
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

static unsigned short f32_to_bf16(float value)
{
    u32 bits = bits_of(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return (unsigned short)(bits >> 16);
}

static float bf16_to_f32(unsigned short value)
{
    return bits_to_f32((u32)value << 16);
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
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_BF16_FP32, txenc::RM_RNE, 0, 0)>();
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>((u64)kCtrlCompressP1);
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>((u64)kCtrlDecompressP1);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_FP32_BF16, txenc::RM_RNE, 0, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(6);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const u32 want_bits = bits_of(bf16_to_f32(f32_to_bf16(input[i])));
        if (out[i] != want_bits) {
            printf("[vdep03] mismatch i=%u actual=%08x expected=%08x\n",
                   i, out[i], want_bits);
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep03] PASS: %u lanes, bf16 pack/compress/decompress/widen roundtrip\n", LANES);
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
    .name = "vdep03",
});
