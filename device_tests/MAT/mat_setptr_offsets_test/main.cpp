/* 从 dayan-samples/Instructions/MAT/mat_setptr_offsets.cpp 迁移（裸机版，发射走
 * instr_enc 编码库 + tx_test 骨架）。
 *
 * 意图：tx.mat.setptr 解码/调度探针——在已知良好的 S8 MM/MMA harness 前发射
 * setptr（rs1/rs2 = A/B 侧额外矩阵指针，rd 字段固定 00000 不写寄存器），
 *随后 MM 结果位精确，作为 setptr 已被正确解码/调度的哨兵。
 * S8 不消费 scale 数据，不声明 MX scale 指针数据语义覆盖。
 * 保留 MAT_PERSISTENTMODE=0 兼容写。
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

    txtst::configure_lsu_chunked();
    txtst::dma_d2u_bytes(kABytes, A1_URAM, A1_DDR);
    txtst::dma_d2u_bytes(kBBytes, B1_URAM, B1_DDR);
    txtst::dma_d2u_bytes(kABytes, A2_URAM, A2_DDR);
    txtst::dma_d2u_bytes(kBBytes, B2_URAM, B2_DDR);
    txtst::wait_lsu();

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

    /* setptr：rd 字段固定 00000（a2 为哑元），rs1/rs2 = A/B 侧 scale 指针。 */
    txenc::emit32_a0_a1_a2<txenc::mat_setptr_enc(0)>(A1_URAM + 64U, B1_URAM + 128U, C_MM_URAM + 32U);

    txenc::emit32_a0_a1_a2<txenc::mat_enc(txenc::MAT_MM, 0)>(A1_URAM, B1_URAM, C_MM_URAM);
    txtst::wait_mat();
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
    .name = "mdep19",
});
