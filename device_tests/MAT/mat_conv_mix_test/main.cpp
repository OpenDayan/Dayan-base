/* S8/S8 conv2d/conv2da：C=8/Co=8、Hi=3/Wi=4、Hk=2/Wk=2。
 * pad(L,U)=(1,1)、dilation(W,H)=(2,2)、stride(W,H)=(1,1)，Ho=3/Wo=4。
 * 维度写实际值减 1，全部 32 项 TCSR 显式初始化；swizzle/非 MX scale 步长为零。
 * 非常量正负 S8 数据，输出 FP32 卷积与两倍卷积及槽内 padding，交由 ref/dut 对拍。
 */

#include "../../instr_enc/tx_test.h"

#include <stdint.h>
#include <stdio.h>

using txtst::u32;
using txtst::u64;

static constexpr int kC = 8;
static constexpr int kCo = 8;
static constexpr int kHi = 3;
static constexpr int kWi = 4;
static constexpr int kHk = 2;
static constexpr int kWk = 2;
static constexpr int kPadL = 1, kPadU = 1;
static constexpr int kDilW = 2, kDilH = 2;
static constexpr int kStrideW = 1, kStrideH = 1;
static constexpr int kWo = (kWi + 2 * kPadL - kDilW * (kWk - 1) - 1) / kStrideW + 1;
static constexpr int kHo = (kHi + 2 * kPadU - kDilH * (kHk - 1) - 1) / kStrideH + 1;

static constexpr unsigned kWiStride = 32;
static constexpr unsigned kHiStride = kWi * kWiStride;
static constexpr unsigned kInputByte = kHi * kHiStride;
static constexpr unsigned kCoStride = 32;
static constexpr unsigned kWkStride = kCo * kCoStride;
static constexpr unsigned kHkStride = kWk * kWkStride;
/* 核移动的输入字节增量=dilation×输入步长；核存储不插入膨胀空洞。 */
static constexpr unsigned StrideDilationW = kDilW * kWiStride;
static constexpr unsigned StrideDilationH = kDilH * kHiStride;
static constexpr unsigned kKernelByte = kHk * kHkStride;
static constexpr unsigned kWoStride = ((kCo * sizeof(float) + 31U) / 32U) * 32U;
static constexpr unsigned kHoStride = kWo * kWoStride;
static constexpr unsigned kOutputByte = kHo * kHoStride;
static constexpr unsigned kInV = kHi * kWi * kC;
static constexpr unsigned kKerV = kHk * kWk * kCo * kC;

static constexpr u64 SLAB = 4096;
static constexpr u64 IN_URAM = txtst::URAM_BASE;
static constexpr u64 KER_URAM = txtst::URAM_BASE + 1U * SLAB;
static constexpr u64 OUT1_URAM = txtst::URAM_BASE + 2U * SLAB;
static constexpr u64 OUT2_URAM = txtst::URAM_BASE + 3U * SLAB;
static constexpr u64 IN_DDR = txtst::DDR_BASE;
static constexpr u64 KER_DDR = txtst::DDR_BASE + 1U * SLAB;
static constexpr u64 OUT1_DDR = txtst::OUT_DDR;
static constexpr u64 OUT2_DDR = txtst::OUT_DDR + 1U * SLAB;


static_assert(kHo > 0 && kWo > 0, "Convolution output must be nonempty");
static_assert(kC <= kWiStride && kC <= kCoStride, "Channels must fit their slots");
static_assert(kWoStride >= kCo * sizeof(float) && kWoStride % 32 == 0,
              "Output slots must fit channels and be 32B aligned");
static_assert(((kInputByte + 63U) / 64U) * 64U <= SLAB &&
              ((kKernelByte + 63U) / 64U) * 64U <= SLAB &&
              ((kOutputByte + 63U) / 64U) * 64U <= SLAB, "DMA buffers must fit slabs");
static_assert(kOutputByte == 384, "Update meta.json compare sizes when changing shape");

static void tensor_run(void)
{
    /* 逻辑张量 in[hi][wi][c]、ker[hk][wk][co][c]（紧凑），再散布到对齐槽。 */
    int8_t logical_in[kInV];
    int8_t logical_ker[kKerV];
    for (unsigned index = 0; index < kInV; ++index)
        logical_in[index] = static_cast<int8_t>(static_cast<int>((index * 5U + 1U) % 13U) - 6);
    for (unsigned index = 0; index < kKerV; ++index)
        logical_ker[index] = static_cast<int8_t>(static_cast<int>((index * 3U + 1U) % 7U) - 4);

    auto in_fp = reinterpret_cast<volatile int8_t *>(IN_DDR);
    auto ker_fp = reinterpret_cast<volatile int8_t *>(KER_DDR);
    for (unsigned i = 0; i < kInputByte; ++i)
        in_fp[i] = 0;
    for (unsigned i = 0; i < kKernelByte; ++i)
        ker_fp[i] = 0;
    for (int hi = 0; hi < kHi; ++hi)
        for (int wi = 0; wi < kWi; ++wi)
            for (int c = 0; c < kC; ++c)
                in_fp[hi * (int)kHiStride + wi * (int)kWiStride + c] =
                    logical_in[(hi * kWi + wi) * kC + c];
    for (int hk = 0; hk < kHk; ++hk)
        for (int wk = 0; wk < kWk; ++wk)
            for (int co = 0; co < kCo; ++co)
                for (int c = 0; c < kC; ++c)
                    ker_fp[hk * (int)kHkStride + wk * (int)kWkStride +
                           co * (int)kCoStride + c] =
                        logical_ker[((hk * kWk + wk) * kCo + co) * kC + c];
    txtst::flush_ddr(IN_DDR, kInputByte);
    txtst::flush_ddr(KER_DDR, kKernelByte);

    for (unsigned byte = 0; byte < kOutputByte; ++byte) {
        reinterpret_cast<volatile uint8_t *>(OUT1_DDR)[byte] = 0x5a;
        reinterpret_cast<volatile uint8_t *>(OUT2_DDR)[byte] = 0xa5;
    }
    txtst::flush_ddr(OUT1_DDR, kOutputByte);
    txtst::flush_ddr(OUT2_DDR, kOutputByte);

    txtst::configure_lsu_chunked();
    txtst::dma_d2u_bytes(kInputByte, IN_URAM, IN_DDR);
    txtst::dma_d2u_bytes(kKernelByte, KER_URAM, KER_DDR);
    txtst::dma_d2u_bytes(kOutputByte, OUT1_URAM, OUT1_DDR);
    txtst::dma_d2u_bytes(kOutputByte, OUT2_URAM, OUT2_DDR);
    txtst::wait_lsu();

    /* 0..7 维度减 1；8..14 字节步长；15..23 swizzle/dtype/scale；24..31 卷积窗口。 */
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
    txenc::write_tcsr<txenc::UNIT_MAT, 30>(StrideDilationW);
    txenc::write_tcsr<txenc::UNIT_MAT, 31>(StrideDilationH);

    /* 同一 MAT 队列：conv2d 完成后，out2 的 conv2d 种子与 conv2da 累加背靠背
     * 入队（中间无 bar），最后一次 bar 等整批。 */
    txenc::emit32_a0_a1_a2<txenc::mat_enc(txenc::MAT_CONV2D, 0)>(IN_URAM, KER_URAM, OUT1_URAM);
    txtst::wait_mat();
    txenc::emit32_a0_a1_a2<txenc::mat_enc(txenc::MAT_CONV2D, 0)>(IN_URAM, KER_URAM, OUT2_URAM);
    txenc::emit32_a0_a1_a2<txenc::mat_enc(txenc::MAT_CONV2DA, 0)>(IN_URAM, KER_URAM, OUT2_URAM);
    txtst::wait_mat();

    txtst::dma_u2d_bytes(kOutputByte, OUT1_DDR, OUT1_URAM);
    txtst::dma_u2d_bytes(kOutputByte, OUT2_DDR, OUT2_URAM);
    txtst::wait_lsu();
    txtst::invalidate_ddr(OUT1_DDR, kOutputByte);
    txtst::invalidate_ddr(OUT2_DDR, kOutputByte);
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
    .name = "mdep17",
});
