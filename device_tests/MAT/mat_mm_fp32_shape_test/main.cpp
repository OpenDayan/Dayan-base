/* 从 dayan-samples/Instructions/MAT/single/mat_mm_fp32_shape.cpp.cpp 迁移（裸机版，发射走
 * instr_enc 编码库 + tx_test 骨架；共享 mat_mm_single_impl/host 的逻辑内联展开）。
 *
 * 意图：FP32 存储 / TF32 pair（dtype=0x8410）单 MM：M=4/N=12/K=17，K 尾部 + 有效 A/B 均为 1.0，stride 96/96/64，输出含保护区。
 * 单条 tx.mat.mm（初始化，无累加）；A 行主序、B 行=输出列、C FP32 行主序，
 * 输入 padding 保持清零，输出预填 0x5a，完整物理输出交由 ref/dut 对拍。
 * 维度 M/N/K 写"实际-1"。
 */

#include "../../instr_enc/tx_test.h"

#include <stdint.h>
#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr unsigned kM = 4;
static constexpr unsigned kN = 12;
static constexpr unsigned kK = 17;
static constexpr unsigned kAStride = 96;
static constexpr unsigned kBStride = 96;
static constexpr unsigned kCStride = 64;
static constexpr unsigned kElemBytes = 4;
static constexpr unsigned kABytes = kM * kAStride;
static constexpr unsigned kBBytes = kN * kBStride;
static constexpr unsigned kCBytes = kM * kCStride;
static constexpr unsigned kDType = 0x8410;

static constexpr u64 SLAB = 8192;
static constexpr u64 A_URAM = txtst::URAM_BASE;
static constexpr u64 B_URAM = txtst::URAM_BASE + 1U * SLAB;
static constexpr u64 C_URAM = txtst::URAM_BASE + 2U * SLAB;
static constexpr u64 A_DDR = txtst::DDR_BASE;
static constexpr u64 B_DDR = txtst::DDR_BASE + 0x1000U;
static constexpr u64 C_DDR = txtst::OUT_DDR;

static_assert(kAStride >= kK * kElemBytes && kBStride >= kK * kElemBytes,
              "Input rows must fit their strides");
static_assert(kCStride > kN * sizeof(float) && kCStride % 32 == 0,
              "Every output row must contain aligned guard bytes");
static_assert(((kABytes + 63U) / 64U) * 64U <= 4096 &&
              ((kBBytes + 63U) / 64U) * 64U <= SLAB &&
              ((kCBytes + 63U) / 64U) * 64U <= SLAB, "DMA buffers must fit allocations");
static_assert(kCBytes == 256, "Update meta.json compare sizes when changing shape");

/* 输入原始位模式，小端展开到 kElemBytes 字节。 */
static void store_raw(u64 base, unsigned offset_elems)
{
    const u32 raw = 0x3f800000U;
    auto dst = reinterpret_cast<volatile uint8_t *>(base + offset_elems * kElemBytes);
    for (unsigned b = 0; b < kElemBytes; ++b)
        dst[b] = (uint8_t)(raw >> (8U * b));
}

static void tensor_run(void)
{
    /* 输入区清零（含 stride padding，与样例 zero-init 一致）；C 预播 0x5a。 */
    for (unsigned i = 0; i < kABytes; ++i)
        reinterpret_cast<volatile uint8_t *>(A_DDR)[i] = 0;
    for (unsigned i = 0; i < kBBytes; ++i)
        reinterpret_cast<volatile uint8_t *>(B_DDR)[i] = 0;
    for (unsigned i = 0; i < kCBytes; ++i)
        reinterpret_cast<volatile uint8_t *>(C_DDR)[i] = 0x5a;
    for (unsigned row = 0; row < kM; ++row)
        for (unsigned col = 0; col < kK; ++col)
            store_raw(A_DDR, row * kAStride / kElemBytes + col);
    for (unsigned row = 0; row < kN; ++row)
        for (unsigned col = 0; col < kK; ++col)
            store_raw(B_DDR, row * kBStride / kElemBytes + col);
    txtst::flush_ddr(A_DDR, kABytes);
    txtst::flush_ddr(B_DDR, kBBytes);
    txtst::flush_ddr(C_DDR, kCBytes);

    txtst::configure_lsu_chunked();
    txtst::dma_d2u_bytes(kABytes, A_URAM, A_DDR);
    txtst::dma_d2u_bytes(kBBytes, B_URAM, B_DDR);
    txtst::dma_d2u_bytes(kCBytes, C_URAM, C_DDR);
    txtst::wait_lsu();

    /* 配置矩阵乘所需字段；非 MX scale 步长和卷积窗口字段显式清零。 */
    txenc::write_tcsr<txenc::UNIT_MAT, 0>(kK - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 3>(kM - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 7>(kN - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 8>(kAStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 9>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 12>(kBStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 13>(kCStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 15>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 16>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 17>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 18>(kDType);
    txenc::write_tcsr<txenc::UNIT_MAT, 19>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 23>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 24>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 25>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 26>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 27>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 28>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 29>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 30>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 31>(0);

    txenc::emit32_a0_a1_a2<txenc::mat_enc(txenc::MAT_MM, 0)>(A_URAM, B_URAM, C_URAM);
    txtst::wait_mat();

    txtst::dma_u2d_bytes(kCBytes, C_DDR, C_URAM);
    txtst::wait_lsu();
    txtst::invalidate_ddr(C_DDR, kCBytes);
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
    .name = "mdep22",
});
