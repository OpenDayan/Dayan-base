/* 从 dayan-samples/Instructions/MAT/single/mat_conv2d_pointwise.cpp.cpp 迁移（裸机版，发射走
 * instr_enc 编码库 + tx_test 骨架；共享 mat_conv_single_impl/host 逻辑内联展开）。
 *
 * 意图：1×1 逐点卷积：C=16/Co=16/Hi=5/Wi=7/Hk=Wk=1/pad=0/dil=1/stride=1（Ho=5/Wo=7），WoStride=Co*4+32=96。
 * 单条 tx.mat.conv2d；维度写"实际-1"，StrideDilation=dil×输入步长。输入/核
 * 有效输入和核均为 1；物理 padding 预填 0xa5，输出预填 0x5a。
 * 每个输出槽保留 32B 保护区，完整输出交由 ref/dut 对拍。
 */

#include "../../instr_enc/tx_test.h"

#include <stdint.h>
#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr int kC = 16;
static constexpr int kCo = 16;
static constexpr int kHi = 5;
static constexpr int kWi = 7;
static constexpr int kHk = 1;
static constexpr int kWk = 1;
static constexpr int kPadL = 0, kPadU = 0;
static constexpr int kDilW = 1, kDilH = 1;
static constexpr int kStrideW = 1, kStrideH = 1;
static constexpr int kWo = (kWi + 2 * kPadL - kDilW * (kWk - 1) - 1) / kStrideW + 1;
static constexpr int kHo = (kHi + 2 * kPadU - kDilH * (kHk - 1) - 1) / kStrideH + 1;

static constexpr unsigned kWiStride = 32;
static constexpr unsigned kHiStride = kWi * kWiStride;
static constexpr unsigned kInputBytes = kHi * kHiStride;
static constexpr unsigned kCoStride = 32;
static constexpr unsigned kWkStride = kCo * kCoStride;
static constexpr unsigned kHkStride = kWk * kWkStride;
static constexpr unsigned kKernelBytes = kHk * kHkStride;
static constexpr unsigned kWoStride = kCo * 4 + 32;
static constexpr unsigned kHoStride = kWo * kWoStride;
static constexpr unsigned kOutputBytes = kHo * kHoStride;

static constexpr u64 SLAB = 8192;
static constexpr u64 IN_URAM = txtst::URAM_BASE;
static constexpr u64 KER_URAM = txtst::URAM_BASE + 1U * SLAB;
static constexpr u64 OUT_URAM = txtst::URAM_BASE + 2U * SLAB;
static constexpr u64 IN_DDR = txtst::DDR_BASE;
static constexpr u64 KER_DDR = txtst::DDR_BASE + 0x2000U;
static constexpr u64 OUT_DDR = txtst::OUT_DDR;

static_assert(kWoStride > kCo * sizeof(float) && kWoStride % 32 == 0,
              "Every output slot must contain aligned guard bytes");
static_assert(((kInputBytes + 63U) / 64U) * 64U <= SLAB &&
              ((kKernelBytes + 63U) / 64U) * 64U <= SLAB &&
              ((kOutputBytes + 63U) / 64U) * 64U <= SLAB, "DMA buffers must fit slabs");
static_assert(kOutputBytes == 3360, "Update meta.json compare sizes when changing shape");

static void tensor_run(void)
{
    /* 紧凑逻辑张量 + 0xa5 哨兵散布；输出 0x5a 预播。 */
    int8_t logical_in[kHi * kWi * kC];
    int8_t logical_ker[kHk * kWk * kCo * kC];
    for (unsigned i = 0; i < kHi * kWi * kC; ++i)
        logical_in[i] = 1;
    for (unsigned i = 0; i < kHk * kWk * kCo * kC; ++i)
        logical_ker[i] = 1;
    for (unsigned i = 0; i < kInputBytes; ++i)
        reinterpret_cast<volatile uint8_t *>(IN_DDR)[i] = 0xa5;
    for (unsigned i = 0; i < kKernelBytes; ++i)
        reinterpret_cast<volatile uint8_t *>(KER_DDR)[i] = 0xa5;
    for (unsigned i = 0; i < kOutputBytes; ++i)
        reinterpret_cast<volatile uint8_t *>(OUT_DDR)[i] = 0x5a;
    for (int hi = 0; hi < kHi; ++hi)
        for (int wi = 0; wi < kWi; ++wi)
            for (int c = 0; c < kC; ++c)
                reinterpret_cast<volatile int8_t *>(
                    IN_DDR)[hi * (int)kHiStride + wi * (int)kWiStride + c] =
                    logical_in[(hi * kWi + wi) * kC + c];
    for (int hk = 0; hk < kHk; ++hk)
        for (int wk = 0; wk < kWk; ++wk)
            for (int co = 0; co < kCo; ++co)
                for (int c = 0; c < kC; ++c)
                    reinterpret_cast<volatile int8_t *>(
                        KER_DDR)[hk * (int)kHkStride + wk * (int)kWkStride +
                                 co * (int)kCoStride + c] =
                        logical_ker[((hk * kWk + wk) * kCo + co) * kC + c];
    txtst::flush_ddr(IN_DDR, kInputBytes);
    txtst::flush_ddr(KER_DDR, kKernelBytes);
    txtst::flush_ddr(OUT_DDR, kOutputBytes);

    txtst::configure_lsu_chunked();
    txtst::dma_d2u_bytes(kInputBytes, IN_URAM, IN_DDR);
    txtst::dma_d2u_bytes(kKernelBytes, KER_URAM, KER_DDR);
    txtst::dma_d2u_bytes(kOutputBytes, OUT_URAM, OUT_DDR);
    txtst::wait_lsu();

    /* 完整 32 项 TCSR：维度"实际-1"，StrideDilation=dil×输入步长。 */
    txenc::write_tcsr<txenc::UNIT_MAT, 0>(kC - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 1>(kWi - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 2>(kHi - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 3>(kWo - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 4>(kHo - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 5>(kWk - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 6>(kHk - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 7>(kCo - 1);
    txenc::write_tcsr<txenc::UNIT_MAT, 8>(kWiStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 9>(kHiStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 10>(kWkStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 11>(kHkStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 12>(kCoStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 13>(kWoStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 14>(kHoStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 15>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 16>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 17>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 18>(0x8208);
    txenc::write_tcsr<txenc::UNIT_MAT, 19>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 20>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 21>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 22>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 23>(0);
    txenc::write_tcsr<txenc::UNIT_MAT, 24>(kPadL);
    txenc::write_tcsr<txenc::UNIT_MAT, 25>(kPadU);
    txenc::write_tcsr<txenc::UNIT_MAT, 26>(kDilW);
    txenc::write_tcsr<txenc::UNIT_MAT, 27>(kDilH);
    txenc::write_tcsr<txenc::UNIT_MAT, 28>(kStrideW);
    txenc::write_tcsr<txenc::UNIT_MAT, 29>(kStrideH);
    txenc::write_tcsr<txenc::UNIT_MAT, 30>(kDilW * (int)kWiStride);
    txenc::write_tcsr<txenc::UNIT_MAT, 31>(kDilH * (int)kHiStride);

    txenc::emit32_a0_a1_a2<txenc::mat_enc(txenc::MAT_CONV2D, 0)>(IN_URAM, KER_URAM, OUT_URAM);
    txtst::wait_mat();

    txtst::dma_u2d_bytes(kOutputBytes, OUT_DDR, OUT_URAM);
    txtst::wait_lsu();
    txtst::invalidate_ddr(OUT_DDR, kOutputBytes);
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
    .name = "mdep27",
});
