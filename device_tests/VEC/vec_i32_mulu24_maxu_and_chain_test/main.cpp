/* 从 dayan-samples/Instructions/VEC/dependency/vec_i32_mulu24_maxu_and_chain.cpp 迁移
 * （dayan-base 裸机版，指令发射走 instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：单个 VEC 指令块内的三级跨操作 RAW 依赖——
 *   tx.vec.vv.int32.mulu24 -> tx.vec.vv.int32.maxu -> tx.vec.vv.int32.bitwise(AND)
 * A 装入 bank0（最新值 = 选择子 0）；B/C/D 装入 bank1，age 分别为 2/1/0
 * （选择子 18/17/16）。四条 load 与三条运算、一条 store 共 8 条指令排同一块，
 * 一次 CONTROL 启动，中间无等待，验证块内生产者-消费者链路。
 *
 * 期望：out[i] == (maxu(mulu24(a[i], b[i]), c[i])) & d[i]，逐 lane 对拍；
 * mulu24 = 双操作数低 24bit 无符号乘截断 32bit（原样例 host oracle 同款整数式）。
 * 输入公式与原样例一致：a=0x81000000+0x0012345*i、b=0x72000000+0x0005432*i、
 * c 奇偶 lane 两式、d 按 i%4 取 0x00ff00ff/0xff00ff00/0x0f0f0f0f/0xf0f0f0f0。
 */

#include "../../instr_enc/tx_test.h"

#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned LANES = 64;
static constexpr u64 KBYTES = LANES * sizeof(u32);             /* 256 */
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

static void tensor_run(void)
{
    auto a = reinterpret_cast<volatile u32 *>(A_DDR);
    auto b = reinterpret_cast<volatile u32 *>(B_DDR);
    auto c = reinterpret_cast<volatile u32 *>(C_DDR);
    auto d = reinterpret_cast<volatile u32 *>(D_DDR);

    for (unsigned i = 0; i < LANES; ++i) {
        a[i] = 0x81000000U + 0x0012345U * i;
        b[i] = 0x72000000U + 0x0005432U * i;
        c[i] = (i & 1U) ? (0xF0000000U - 0x00110011U * i) : (0x00010000U + 0x00030103U * i);
        d[i] = (i % 4U == 0U) ? 0x00ff00ffU : ((i % 4U == 1U) ? 0xff00ff00U : ((i % 4U == 2U) ? 0x0f0f0f0fU : 0xf0f0f0f0U));
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

    /* 8 条指令排一个块：4 loads + mulu24 + MAXU + AND + store。
     * bank0: a=V0；bank1: b=V18(age2) c=V17(age1) d=V16(age0)。 */
    txtst::wait_vec_queue_capacity(8);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 0)>(A_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(B_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(C_URAM);
    txenc::emit32_a0<txenc::vec_read_enc(txenc::VEC_RD_U2V, 0, 1)>(D_URAM);
    txenc::emit32<txenc::vec_vv_int_enc(txenc::VEC_INT_MULU24, 0, 18, 0)>();
    txenc::emit32<txenc::vec_vv_int_enc(txenc::VEC_INT_MAXU, 0, 17, 0)>();
    txenc::emit32<txenc::vec_vv_bitwise_enc(0x8, 0, 16, 0)>();
    txenc::emit32_a0<txenc::vec_write_enc(txenc::VEC_WR_V2U, 0, 0)>(OUT_URAM);
    txtst::start_vec_block(8);
    txtst::wait_vec();

    txtst::dma_uram_to_dram(OUT_URAM, txtst::OUT_DDR);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, KBYTES);

    /* 设备侧逐 lane 对拍（原样例的 host oracle 同款整数公式：
     * 24bit 截断无符号乘 -> 无符号 max -> AND）。 */
    auto out = reinterpret_cast<volatile u32 *>(txtst::OUT_DDR);
    for (unsigned i = 0; i < LANES; ++i) {
        const u32 ua = a[i];
        const u32 ub = b[i];
        const u32 uc = c[i];
        const u32 um = d[i];
        const u32 t0 = (ua & 0x00ffffffU) * (ub & 0x00ffffffU);
        const u32 t1 = t0 > uc ? t0 : uc;
        const u32 expected = t1 & um;
        if (out[i] != expected) {
            printf("[vdep16] mismatch i=%u actual=%08x expected=%08x\n",
                   i, out[i], expected);
            dayan_thread_exit(-22);
        }
    }
    printf("[vdep16] PASS: %u lanes, i32 mulu24->maxu->and chain in one block\n", LANES);
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
    .name = "vdep16",
});
