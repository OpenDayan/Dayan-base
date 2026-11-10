/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include "../../../instr_enc/tx_test.h"
#include "../../../instr_enc/tx_macro_enc.h"

namespace matcheck {
using txtst::u64;
using txtst::u32;

static void fill(unsigned begin, unsigned bytes, unsigned value)
{
    auto memory = reinterpret_cast<volatile unsigned char *>(txtst::DDR_BASE);
    for (unsigned index = begin; index < begin + bytes; ++index)
        memory[index] = value;
}

static int sample(unsigned pattern, unsigned outer, unsigned inner, unsigned side)
{
    if (pattern == 0) return 0;
    if (pattern == 1) return 1;
    if (pattern == 2) return int((outer * 3 + inner * 2 + side) % 5) - 2;
    if (pattern == 3) return inner == (outer + side) % 7 ? 1 : 0;
    if (pattern == 4) return side ? 1 : (inner % 2 ? -1 : 1);
    if (pattern == 6) return side ? 1 : 2;
    return int((outer + side) % 3) - 1;
}

static void scalar(unsigned offset, unsigned dtype, int value)
{
    u64 address = txtst::DDR_BASE + offset;
    if (dtype == 0x8208) {
        *reinterpret_cast<volatile signed char *>(address) = value;
    } else if (dtype == 4) {
        unsigned magnitude = value < 0 ? -value : value;
        *reinterpret_cast<volatile unsigned char *>(address) =
            (value < 0 ? 0x80 : 0) | (magnitude == 2 ? 0x40 : magnitude == 1 ? 0x38 : 0);
    } else if (dtype == 2) {
        union { float number; u32 bits; } converted;
        converted.number = float(value);
        *reinterpret_cast<volatile unsigned short *>(address) = converted.bits >> 16;
    } else {
        *reinterpret_cast<volatile float *>(address) = float(value);
    }
}

[[maybe_unused]] static void operand(unsigned base, unsigned rows, unsigned count, unsigned stride,
                    unsigned dtype, unsigned width, unsigned pattern, unsigned side)
{
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned channel = 0; channel < count; ++channel)
            scalar(base + row * stride + channel * width, dtype,
                   sample(pattern, row, channel, side));
}

static void output(unsigned base, unsigned height, unsigned width, unsigned channels,
                   unsigned row_stride, unsigned pixel_stride, unsigned seed)
{
    for (unsigned row = 0; row < height; ++row)
        for (unsigned column = 0; column < width; ++column)
            for (unsigned channel = 0; channel < channels; ++channel)
                scalar(base + row * row_stride + column * pixel_stride + channel * 4,
                       0, int((row * 3 + column * 2 + channel + seed) % 7) - 3);
}

template <int Group, int Offset = 0>
static void configure(const unsigned *values)
{
    txenc::write_tcsr<txenc::UNIT_MAT, Group * 32 + Offset>(values[Offset]);
    if constexpr (Offset < 31) configure<Group, Offset + 1>(values);
}

template <int Group, int Offset = 0>
static void capture(const unsigned *used, unsigned slot)
{
    if (used[Offset])
        reinterpret_cast<volatile u64 *>(txtst::OUT_DDR + 0x100000)[slot * 32 + Offset] =
            txenc::read_tcsr<txenc::UNIT_MAT, Group * 32 + Offset>();
    if constexpr (Offset < 31) capture<Group, Offset + 1>(used, slot);
}

template <int Offset = 0>
static void configure_macro(const unsigned *values)
{
    txenc::write_tcsr<txenc::UNIT_MACRO, Offset>(values[Offset]);
    if constexpr (Offset < 16) configure_macro<Offset + 1>(values);
}

template <int Mode, int Group, int Registers = 0>
static void issue(u64 first, u64 second, u64 destination)
{
    if constexpr (Registers == 1) {
        asm volatile("mv t3, %0\nmv t4, %1\nmv t6, %2\n.insn 4, %3"
                     : : "r"(first), "r"(second), "r"(destination),
                     "i"(txenc::mat_enc(Mode, Group, 28, 29, Mode == 4 ? 0 : 31))
                     : "t3", "t4", "t6", "memory");
    } else if constexpr (Mode == 4) {
        txenc::emit32_a0_a1_a2<txenc::mat_setptr_enc(Group)>(first, second, destination);
    } else {
        txenc::emit32_a0_a1_a2<txenc::mat_enc(Mode, Group)>(first, second, destination);
    }
}
}

static void tensor_run()
{
    using namespace matcheck;
    txenc::sb_bar<txenc::UNIT_GLOBAL>();
    const u64 saved_context = txenc::read_tcsr<txenc::UNIT_GLOBAL, 16>();
    txenc::write_tcsr<txenc::UNIT_GLOBAL, 16>(0x2001020081800UL);
    reinterpret_cast<volatile u64 *>(txtst::OUT_DDR + 0x110000)[0] =
        txenc::read_tcsr<txenc::UNIT_GLOBAL, 16>();
    fill(0, 960, 0xd3);
    fill(32, 128, 90);
    fill(224, 512, 60);
    fill(800, 128, 109);
    operand(32, 2, 8, 64, 33288, 1, 2, 0);
    operand(224, 8, 8, 64, 33288, 1, 2, 1);
    output(800, 1, 2, 8, 0, 64, 1);
    txtst::flush_ddr(txtst::DDR_BASE + 0, 960);
    txtst::configure_lsu_chunked();
    txtst::dma_d2u_bytes(960, txtst::URAM_BASE + 0, txtst::DDR_BASE + 0);
    txtst::wait_lsu();
    const unsigned config_0[32] = {7, 0, 0, 1, 0, 0, 0, 7, 64, 0, 0, 0, 64, 64, 0, 0, 0, 0, 33288, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    configure<0>(config_0);
    txenc::emit32_a0_a1_a2<0x20b5460b>(txtst::URAM_BASE + 32, txtst::URAM_BASE + 224, txtst::URAM_BASE + -8388608);
    txtst::wait_mat();
    txtst::dma_u2d_bytes(960, txtst::OUT_DDR, txtst::URAM_BASE + 0);
    txtst::wait_lsu();
    txtst::invalidate_ddr(txtst::OUT_DDR, 960);
    txtst::flush_ddr(txtst::OUT_DDR + 0x110000, 8);
    txenc::sb_bar<txenc::UNIT_GLOBAL>();
    txenc::write_tcsr<txenc::UNIT_GLOBAL, 16>(saved_context);
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
        .uram_size = 0xc0000,
    },
    .args = {0},
    .argc = 0,
    .name = "mc469",
});
