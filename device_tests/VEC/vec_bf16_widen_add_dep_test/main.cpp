/* 从 dayan-samples/Instructions/VEC/dependency/vec_bf16_widen_add_dep.cpp 迁移
 * （裸机版，发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图（VEC-DEP-006）：单 VEC 块内 DeCompress -> BF16 宽化 -> FP32 add，
 * 6 条指令一次 CONTROL 启动，块内无等待。输入前 32 字为有效打包 BF16 字，
 * lane 32..63 是投毒数据（0xdead0000|k），不得影响结果（DeCompress 只读前 32 字）。
 * 期望（按位比较）：half=packed[i/2]>>(16*(i&1)); want=bf16_to_f32(half)+addend[i]。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(u32);
static constexpr u64 ADDEND_URAM = txtst::URAM_BASE;
static constexpr u64 PACKED_URAM = txtst::URAM_BASE + 5U * KBYTES;
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 10U * KBYTES;
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
    auto logical = reinterpret_cast<volatile float *>(txtst::DDR_BASE);
    auto addend = reinterpret_cast<volatile float *>(txtst::DDR_BASE + 5U * KBYTES);
    auto packed = reinterpret_cast<volatile u32 *>(txtst::DDR_BASE + 10U * KBYTES);

    for (unsigned i = 0; i < LANES; ++i) {
        const int centered = (int)(i % 19U) - 9;
        logical[i] = (float)centered * 0.4375f + (float)(i % 5U) * 0.0625f;
        addend[i] = (float)((int)(i % 13U) - 6) * 0.1875f + (float)(i % 3U) * 0.03125f;
    }
    for (unsigned k = 0; k < 32U; ++k) {
        const unsigned short lo = f32_to_bf16(logical[2U * k]);
        const unsigned short hi = f32_to_bf16(logical[2U * k + 1U]);
        packed[k] = (u32)lo | ((u32)hi << 16);
    }
    for (unsigned k = 32; k < LANES; ++k)
        packed[k] = 0xdead0000U | k;
    txtst::flush_ddr(txtst::DDR_BASE + 5U * KBYTES, KBYTES);
    txtst::flush_ddr(txtst::DDR_BASE + 10U * KBYTES, KBYTES);

    txtst::configure_lsu_linear(KBYTES);
    txtst::dma_dram_to_uram(txtst::DDR_BASE + 5U * KBYTES, ADDEND_URAM);
    txtst::dma_dram_to_uram(txtst::DDR_BASE + 10U * KBYTES, PACKED_URAM);
    txtst::wait_lsu();

    txtst::configure_vec_zero();

    /* 6 条：addend 入 bank1、packed 入 bank0 -> DeCompress(0x20) -> 宽化 -> add -> store。cvt 命名 dst_src：cvt.fp32.bf16(op3)=bf16->fp32 宽化。 */
    txtst::wait_vec_queue_capacity(6);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(ADDEND_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(PACKED_URAM);
    txenc::emit32_a0<txenc::vec_shuffle_static_enc(txenc::VEC_SH_COMPRESS, 0, 0)>((u64)kCtrlDecompressP1);
    txenc::emit32<txenc::vec_cvt_enc(txenc::VEC_CVT_FP32_BF16, txenc::RM_RNE, 0, 0)>();
    txenc::emit32<txenc::vec_vv_fp32_enc(txenc::VEC_FP32_ADD, txenc::RM_RNE, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(6);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const unsigned short packed_half =
            (unsigned short)(packed[i / 2U] >> (16U * (i & 1U)));
        const float base = bf16_to_f32(packed_half);
        const u32 want_bits = bits_of(base + addend[i]);
        if (out[i] != want_bits) {
            printf("[vdep04] mismatch i=%u actual=%08x expected=%08x\n",
                   i, out[i], want_bits);
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep04] PASS: %u lanes, decompress->bf16 widen->add (poison tail ignored)\n", LANES);
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
    .name = "vdep04",
});
