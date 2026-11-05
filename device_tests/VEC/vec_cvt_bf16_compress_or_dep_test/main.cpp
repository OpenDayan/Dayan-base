/* 从 dayan-samples/Instructions/VEC/dependency/vec_cvt_bf16_compress_or_dep.cpp
 * 迁移（裸机版，发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图（VEC-DEP-007）：单 VEC 块内 FP32 -> BF16 窄化 -> Compress -> int32 OR，
 * 6 条指令一次 CONTROL 启动，块内无等待。mask 在块首装入 bank1，窄化/Compress
 * 结果留在 V0，构成 Compress -> bitwise 直接依赖边。
 * 期望（按位比较）：packed[k]=bf16(in[2k])|bf16(in[2k+1])<<16（k<32）；
 * want(i)=packed[i&31] | mask[i]，逐 lane（两半复制的打包块都检查）。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(u32);
static constexpr u64 IN_URAM = txtst::URAM_BASE;
static constexpr u64 MASK_URAM = txtst::URAM_BASE + 5U * KBYTES;
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 10U * KBYTES;
static constexpr unsigned short kCtrlCompressP1 = 0x00;
static constexpr u32 kBitwiseBor = 0xE;

static u32 bits_of(float value)
{
    u32 bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static unsigned short f32_to_bf16(float value)
{
    u32 bits = bits_of(value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return (unsigned short)(bits >> 16);
}

static void tensor_run(void)
{
    auto input = reinterpret_cast<volatile float *>(txtst::DDR_BASE);
    auto mask = reinterpret_cast<volatile u32 *>(txtst::DDR_BASE + 5U * KBYTES);

    for (unsigned i = 0; i < LANES; ++i) {
        const int centered = (int)(i % 23U) - 11;
        input[i] = (float)centered * 0.375f + (float)(i % 7U) * 0.03125f;
        mask[i] = 0x00010001U * (i % 5U);
    }
    txtst::flush_ddr(txtst::DDR_BASE, KBYTES);
    txtst::flush_ddr(txtst::DDR_BASE + 5U * KBYTES, KBYTES);

    txtst::configure_lsu_linear(KBYTES);
    txtst::dma_dram_to_uram(txtst::DDR_BASE, IN_URAM);
    txtst::dma_dram_to_uram(txtst::DDR_BASE + 5U * KBYTES, MASK_URAM);
    txtst::wait_lsu();

    txtst::configure_vec_zero();

    /* 6 条：load in(V0) -> load mask(V1) -> 窄化 -> Compress(0x00)
     * -> OR LUT=0xE（V0 打包结果 | V16 mask）-> store。 */
    txtst::wait_vec_queue_capacity(6);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(IN_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(MASK_URAM);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_BF16_FP32, txenc::RM_RNE, 0, 0)>();
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>((u64)kCtrlCompressP1);
    txenc::emit32<txenc::vec_vv_bitwise_enc(kBitwiseBor, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(6);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    u32 packed[32];
    for (unsigned k = 0; k < 32U; ++k)
        packed[k] = (u32)f32_to_bf16(input[2U * k]) |
                    ((u32)f32_to_bf16(input[2U * k + 1U]) << 16);
    for (unsigned i = 0; i < LANES; ++i) {
        const u32 want = packed[i & 31U] | mask[i];
        if (out[i] != want) {
            printf("[vdep07] mismatch i=%u actual=%08x expected=%08x\n", i, out[i], want);
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep07] PASS: %u lanes, cvt->compress->bitwise-OR chain\n", LANES);
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
    .name = "vdep07",
});
