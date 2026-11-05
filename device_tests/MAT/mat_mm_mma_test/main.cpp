/* 从 dayan-samples/Instructions/MAT/mat_mm_mma.cpp 迁移（裸机版，发射走
 * instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：MAT 矩阵乘核心——tx.mat.mm（初始化，累加=0）与 tx.mat.mma（累加=1）。
 * 单 MAT 组：C = A·B，A 行主序（步长 K）、B 列主序（步长 K）、C FP32 行主序
 * （步长 N*4）。S8/S8（DataType=0x8208）使乘积和在 FP32 中精确，得到位精确
 * 参照。M=3/N=8/K=32（A/B 行步长 32B、C 行步长 32B，均满足 32B 端口对齐；
 * M 非瓦片倍数保留 M 尾步进覆盖）。
 * 每操作数独立 4K URAM slab；三次发射：c_mm=A1·B1（mm），c_mma 先 mm 种子
 * 再 mma 累加 A2·B2。每次发射后 bar 等 MAT 排空。
 * 期望（按位比较）：ref_mm[m][n]=Σk a1[m][k]*b1[n][k]；
 * ref_mma = ref_mm + Σk a2*b2，mm 与 mma 两个输出交由 ref/dut 对拍。
 */

#include "../../instr_enc/tx_test.h"

#include <stdint.h>
#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned kM = 3;
static constexpr unsigned kN = 8;
static constexpr unsigned kK = 32;
static constexpr unsigned kABytes = kM * kK;
static constexpr unsigned kBBytes = kN * kK;
static constexpr unsigned kCBytes = kM * kN * 4U;

static constexpr u64 SLAB = 4096;
static constexpr u64 A1_URAM = txtst::URAM_BASE;
static constexpr u64 B1_URAM = txtst::URAM_BASE + 1U * SLAB;
static constexpr u64 A2_URAM = txtst::URAM_BASE + 2U * SLAB;
static constexpr u64 B2_URAM = txtst::URAM_BASE + 3U * SLAB;
static constexpr u64 C_MM_URAM = txtst::URAM_BASE + 4U * SLAB;
static constexpr u64 C_MMA_URAM = txtst::URAM_BASE + 5U * SLAB;
static constexpr u64 A1_DDR = txtst::DDR_BASE;
static constexpr u64 B1_DDR = txtst::DDR_BASE + 1U * SLAB;
static constexpr u64 A2_DDR = txtst::DDR_BASE + 2U * SLAB;
static constexpr u64 B2_DDR = txtst::DDR_BASE + 3U * SLAB;
static constexpr u64 C_MM_DDR = txtst::OUT_DDR;
static constexpr u64 C_MMA_DDR = txtst::OUT_DDR + 1U * SLAB;

/* MAT group0 TCSR 偏移（SDK MAT_TCSR 枚举同值）。 */
static constexpr int MAT_K_ = 0;
static constexpr int MAT_M_ = 3;
static constexpr int MAT_N_ = 7;
static constexpr int MAT_ASTRIDE_ = 8;
static constexpr int MAT_PERSISTENTMODE_ = 9;
static constexpr int MAT_BSTRIDE_ = 12;
static constexpr int MAT_CSTRIDE_ = 13;
static constexpr int MAT_ASWIZZLE_ = 15;
static constexpr int MAT_BSWIZZLE_ = 16;
static constexpr int MAT_CSWIZZLE_ = 17;
static constexpr int MAT_DATATYPE_ = 18;

static void tensor_run(void)
{
    auto a1 = reinterpret_cast<volatile int8_t *>(A1_DDR);
    auto b1 = reinterpret_cast<volatile int8_t *>(B1_DDR);
    auto a2 = reinterpret_cast<volatile int8_t *>(A2_DDR);
    auto b2 = reinterpret_cast<volatile int8_t *>(B2_DDR);

    for (unsigned i = 0; i < kABytes; ++i) {
        a1[i] = (int8_t)(int)((i * 5U + 1U) % 13U) - 6;
        a2[i] = (int8_t)(int)((i * 7U + 3U) % 9U) - 4;
    }
    for (unsigned i = 0; i < kBBytes; ++i) {
        b1[i] = (int8_t)(int)((i * 11U + 2U) % 17U) - 8;
        b2[i] = (int8_t)(int)((i * 13U + 5U) % 11U) - 5;
    }
    for (u64 region = 0; region < 4U; ++region)
        txtst::flush_ddr(txtst::DDR_BASE + region * SLAB, SLAB);

    /* 搬入 URAM：逐 64B 块（操作数字节数非 64 倍数，尾部整块写 slab padding）。 */
    txtst::configure_lsu_chunked();
    txtst::dma_d2u_bytes(kABytes, A1_URAM, A1_DDR);
    txtst::dma_d2u_bytes(kBBytes, B1_URAM, B1_DDR);
    txtst::dma_d2u_bytes(kABytes, A2_URAM, A2_DDR);
    txtst::dma_d2u_bytes(kBBytes, B2_URAM, B2_DDR);
    txtst::wait_lsu();

    /* MAT group0：A 行主序（步长 K）、B 列主序（步长 K）、C 行主序（步长 N*4），
     * 线性布局无 swizzle；保留 PERSISTENTMODE=0 兼容写；S8/S8=0x8208。
     * MM 与 conv 的维度寄存器均按“实际值-1”编码。 */
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_K_>(kK - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_M_>(kM - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_N_>(kN - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_ASTRIDE_>(kK);
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_BSTRIDE_>(kK);
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_CSTRIDE_>(kN * 4U);
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_ASWIZZLE_>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_BSWIZZLE_>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_CSWIZZLE_>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_PERSISTENTMODE_>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, MAT_DATATYPE_>(0x8208);

    /* mm: c_mm = A1·B1（初始化）。rs1=a0(A) rs2=a1(B) rd=a2(C)。 */
    txenc::emit32_a0_a1_a2<txenc::mat_enc(txenc::MAT_MM, 0)>(A1_URAM, B1_URAM, C_MM_URAM);
    txtst::wait_mat();

    /* mma: c_mma 先 mm 种子 A1·B1，再 mma 累加 A2·B2。 */
    txenc::emit32_a0_a1_a2<txenc::mat_enc(txenc::MAT_MM, 0)>(A1_URAM, B1_URAM, C_MMA_URAM);
    txtst::wait_mat();
    txenc::emit32_a0_a1_a2<txenc::mat_enc(txenc::MAT_MMA, 0)>(A2_URAM, B2_URAM, C_MMA_URAM);
    txtst::wait_mat();

    txtst::dma_u2d_bytes(kCBytes, C_MM_DDR, C_MM_URAM);
    txtst::dma_u2d_bytes(kCBytes, C_MMA_DDR, C_MMA_URAM);
    txtst::wait_lsu();
    txtst::invalidate_ddr(C_MM_DDR, kCBytes);
    txtst::invalidate_ddr(C_MMA_DDR, kCBytes);
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
    .name = "mdep01",
});
