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

#ifndef TENSOR_SUITE_TEST_H
#define TENSOR_SUITE_TEST_H

#include <dayan_test.h>

using u32 = unsigned int;
using u64 = unsigned long;

[[maybe_unused]] static constexpr u64 OUT_DDR = 0x92000000UL;
[[maybe_unused]] static constexpr u64 URAM_BASE = 0x00800000UL;
[[maybe_unused]] static constexpr u64 DDR_BASE = 0x90000000UL;

[[maybe_unused]] static constexpr int U_GLOBAL = 0;
[[maybe_unused]] static constexpr int U_LSU = 1;
[[maybe_unused]] static constexpr int U_MAT = 2;
[[maybe_unused]] static constexpr int U_VEC = 3;
[[maybe_unused]] static constexpr int U_MACRO = 4;

[[maybe_unused]] static inline void fence_io() {
    asm volatile("fence iorw, iorw" ::: "memory");
}

template <int Unit, int Address>
[[maybe_unused]] static inline void write_tcsr(u64 value) {
    asm volatile(
        "mv a0, %0\n"
        ".insn 4, %1\n"
        :
        : "r"(value),
          "i"((((Unit & 0xf) << 23) | ((Address & 0x3ff) << 13) |
                (10 << 7) | 0x0b))
        : "a0", "memory");
}

template <int Unit, int Address>
[[maybe_unused]] static inline u64 read_tcsr() {
    u64 value;
    asm volatile(
        ".insn 4, %1\n"
        "mv %0, a0\n"
        : "=r"(value)
        : "i"((((Unit & 0xf) << 23) | ((Address & 0x3ff) << 13) |
                (1 << 12) | (10 << 7) | 0x0b))
        : "a0", "memory");
    return value;
}

template <u32 Encoding>
[[maybe_unused]] static inline void emit32() {
    asm volatile(".4byte %0" : : "i"(Encoding) : "memory");
}

template <u32 Encoding>
[[maybe_unused]] static inline void emit32_a0(u64 value0) {
    asm volatile("mv a0, %0\n.insn 4, %1" : : "r"(value0), "i"(Encoding) : "a0", "memory");
}

template <u32 Encoding>
[[maybe_unused]] static inline void emit32_a0_a1(u64 value0, u64 value1) {
    asm volatile("mv a0, %0\nmv a1, %1\n.insn 4, %2" : : "r"(value0), "r"(value1), "i"(Encoding) : "a0", "a1", "memory");
}

template <u32 Encoding>
[[maybe_unused]] static inline void emit32_a0_a1_a2(u64 value0, u64 value1, u64 value2) {
    asm volatile("mv a0, %0\nmv a1, %1\nmv a2, %2\n.insn 4, %3" : : "r"(value0), "r"(value1), "r"(value2), "i"(Encoding) : "a0", "a1", "a2", "memory");
}

template <u64 Encoding>
[[maybe_unused]] static inline void emit64_a0_a1_a2(u64 value0, u64 value1, u64 value2) {
    asm volatile("mv a0, %0\nmv a1, %1\nmv a2, %2\n.insn 8, %3" : : "r"(value0), "r"(value1), "r"(value2), "i"(Encoding) : "a0", "a1", "a2", "memory");
}

template <u64 Encoding>
[[maybe_unused]] static inline void emit64_a0_a2(u64 value0, u64 value2) {
    asm volatile("mv a0, %0\nmv a2, %1\n.insn 8, %2" : : "r"(value0), "r"(value2), "i"(Encoding) : "a0", "a2", "memory");
}

constexpr u32 dma_enc(int group, int gather, int scatter, int src, int dst) {
    return (0x04u << 25) | ((gather & 1) << 23) | ((group & 7) << 20) |
           (10u << 15) | ((scatter & 1) << 14) | ((src & 1) << 13) |
           ((dst & 1) << 12) | (11u << 7) | 0x0b;
}

constexpr u32 mat_enc(int mode, int group = 0) {
    return (1u << 29) | ((group & 7) << 25) | (11u << 20) |
           (10u << 15) | ((mode & 7) << 12) | (12u << 7) | 0x0b;
}

constexpr u64 macro_enc(int extension, int high16, int group = 0) {
    return (static_cast<u64>(high16 & 0xffff) << 32) |
           (static_cast<u64>(extension & 0x7f) << 25) |
           (11UL << 20) | (10UL << 15) | ((group & 7) << 12) |
           (12UL << 7) | 0x3f;
}

constexpr u32 vec_read_enc(int subfunc, int vb = 0, int sel = 0) {
    return (0x10u << 26) | ((vb & 1) << 25) | ((sel & 3) << 15) |
           ((subfunc & 7) << 12) | (10u << 7) | 0x0b;
}

constexpr u32 vec_write_enc(int subfunc, int sel = 0, int vrs1 = 0) {
    return (0x11u << 26) | ((sel & 3) << 20) | ((vrs1 & 0x1f) << 15) |
           ((subfunc & 7) << 12) | (10u << 7) | 0x0b;
}

[[maybe_unused]] static inline void configure_lsu_64b(int group = 0) {
    const int base = group * 32;
    if (base == 0) {
        write_tcsr<U_LSU, 0>(0); write_tcsr<U_LSU, 1>(0); write_tcsr<U_LSU, 2>(0); write_tcsr<U_LSU, 3>(0);
        write_tcsr<U_LSU, 4>(64); write_tcsr<U_LSU, 5>(64); write_tcsr<U_LSU, 6>(64); write_tcsr<U_LSU, 7>(64);
        write_tcsr<U_LSU, 8>(64); write_tcsr<U_LSU, 9>(64); write_tcsr<U_LSU, 10>(64); write_tcsr<U_LSU, 11>(64);
        write_tcsr<U_LSU, 12>(6); write_tcsr<U_LSU, 13>(0); write_tcsr<U_LSU, 14>(63); write_tcsr<U_LSU, 15>(0);
        write_tcsr<U_LSU, 16>(0); write_tcsr<U_LSU, 17>(0); write_tcsr<U_LSU, 18>(63); write_tcsr<U_LSU, 19>(0); write_tcsr<U_LSU, 20>(64);
    }
}

[[maybe_unused]] static inline void configure_macro_1d(int count) {
    write_tcsr<U_MACRO, 0>(0); write_tcsr<U_MACRO, 1>(0); write_tcsr<U_MACRO, 2>(0); write_tcsr<U_MACRO, 3>(0);
    write_tcsr<U_MACRO, 4>(0); write_tcsr<U_MACRO, 5>(0); write_tcsr<U_MACRO, 6>(0);
    write_tcsr<U_MACRO, 7>(0); write_tcsr<U_MACRO, 8>(0); write_tcsr<U_MACRO, 9>(0);
    write_tcsr<U_MACRO, 10>(0); write_tcsr<U_MACRO, 11>(0); write_tcsr<U_MACRO, 12>(0);
    write_tcsr<U_MACRO, 13>(0); write_tcsr<U_MACRO, 14>(0); write_tcsr<U_MACRO, 15>(0);
    write_tcsr<U_MACRO, 16>(0); write_tcsr<U_MACRO, 17>(0); write_tcsr<U_MACRO, 18>(0);
    write_tcsr<U_MACRO, 19>(0); write_tcsr<U_MACRO, 20>(0); write_tcsr<U_MACRO, 21>(0);
    write_tcsr<U_MACRO, 22>(0); write_tcsr<U_MACRO, 23>(0); write_tcsr<U_MACRO, 24>(0);
    write_tcsr<U_MACRO, 25>(0); write_tcsr<U_MACRO, 26>(0); write_tcsr<U_MACRO, 27>(0);

    write_tcsr<U_MACRO, 0>(count - 1);
    write_tcsr<U_MACRO, 1>(0); write_tcsr<U_MACRO, 2>(0); write_tcsr<U_MACRO, 3>(0);
    write_tcsr<U_MACRO, 4>(count * 4); write_tcsr<U_MACRO, 5>(0); write_tcsr<U_MACRO, 6>(0);
    write_tcsr<U_MACRO, 7>(count * 4); write_tcsr<U_MACRO, 8>(0); write_tcsr<U_MACRO, 9>(0);
    write_tcsr<U_MACRO, 10>(count * 4); write_tcsr<U_MACRO, 11>(0); write_tcsr<U_MACRO, 12>(0);
    write_tcsr<U_MACRO, 13>(0); write_tcsr<U_MACRO, 14>(0); write_tcsr<U_MACRO, 15>(0);
}

[[maybe_unused]] static inline void configure_vec_single_thread() {
    write_tcsr<U_VEC, 2>(0); write_tcsr<U_VEC, 3>(0);
    write_tcsr<U_VEC, 4>(0); write_tcsr<U_VEC, 5>(0); write_tcsr<U_VEC, 6>(0); write_tcsr<U_VEC, 7>(0);
    write_tcsr<U_VEC, 8>(0);
    write_tcsr<U_VEC, 9>(0); write_tcsr<U_VEC, 10>(0); write_tcsr<U_VEC, 11>(0);
    write_tcsr<U_VEC, 12>(0); write_tcsr<U_VEC, 13>(0); write_tcsr<U_VEC, 14>(0);
    write_tcsr<U_VEC, 15>(0); write_tcsr<U_VEC, 16>(0); write_tcsr<U_VEC, 17>(0);
    write_tcsr<U_VEC, 18>(0); write_tcsr<U_VEC, 19>(0); write_tcsr<U_VEC, 20>(0);
    write_tcsr<U_VEC, 21>(0); write_tcsr<U_VEC, 22>(0); write_tcsr<U_VEC, 23>(0);
    write_tcsr<U_VEC, 24>(0); write_tcsr<U_VEC, 25>(0); write_tcsr<U_VEC, 26>(0);
    write_tcsr<U_VEC, 27>(0); write_tcsr<U_VEC, 28>(0); write_tcsr<U_VEC, 29>(0);
    write_tcsr<U_VEC, 30>(0); write_tcsr<U_VEC, 31>(0); write_tcsr<U_VEC, 32>(0);
    write_tcsr<U_VEC, 33>(0); write_tcsr<U_VEC, 34>(0); write_tcsr<U_VEC, 35>(0);
    write_tcsr<U_VEC, 36>(0);
    write_tcsr<U_VEC, 37>(0); write_tcsr<U_VEC, 38>(0); write_tcsr<U_VEC, 39>(0);
    write_tcsr<U_VEC, 40>(0); write_tcsr<U_VEC, 41>(0); write_tcsr<U_VEC, 42>(0);
    write_tcsr<U_VEC, 43>(0); write_tcsr<U_VEC, 44>(0); write_tcsr<U_VEC, 45>(0);
    write_tcsr<U_VEC, 46>(0); write_tcsr<U_VEC, 47>(0); write_tcsr<U_VEC, 48>(0);
    write_tcsr<U_VEC, 49>(0); write_tcsr<U_VEC, 50>(0); write_tcsr<U_VEC, 51>(0);
    write_tcsr<U_VEC, 52>(0); write_tcsr<U_VEC, 53>(0); write_tcsr<U_VEC, 54>(0);
    write_tcsr<U_VEC, 55>(0); write_tcsr<U_VEC, 56>(0); write_tcsr<U_VEC, 57>(0);
    write_tcsr<U_VEC, 58>(0); write_tcsr<U_VEC, 59>(0); write_tcsr<U_VEC, 60>(0);
    write_tcsr<U_VEC, 61>(0); write_tcsr<U_VEC, 62>(0); write_tcsr<U_VEC, 63>(0);
    write_tcsr<U_VEC, 64>(0);
    write_tcsr<U_VEC, 65>(32); write_tcsr<U_VEC, 66>(32);
    write_tcsr<U_VEC, 67>(32); write_tcsr<U_VEC, 68>(32);
}

[[maybe_unused]] static inline void wait_vec_queue_capacity(int count) {
    while (read_tcsr<U_VEC, 0>() < static_cast<u64>(count)) {
    }
}

[[maybe_unused]] static inline void start_vec_block(int prologue0_end,
                                                    int prologue1_end,
                                                    int loop_end,
                                                    int epilogue1_end,
                                                    int epilogue0_end) {
    const u64 control = (static_cast<u64>(prologue0_end) << 0) |
                        (static_cast<u64>(prologue1_end) << 8) |
                        (static_cast<u64>(loop_end) << 16) |
                        (static_cast<u64>(epilogue1_end) << 24) |
                        (static_cast<u64>(epilogue0_end) << 32);
    write_tcsr<U_VEC, 0>(control);
    write_tcsr<U_VEC, 1>(0);
    (void)read_tcsr<U_GLOBAL, U_VEC>();
}

[[maybe_unused]] static inline void start_vec_block(int count) {
    start_vec_block(0, 0, count, count, count);
}

[[maybe_unused]] static inline void wait_unit(int unit) {
    switch (unit) {
        case U_LSU: (void)read_tcsr<U_GLOBAL, U_LSU>(); break;
        case U_MAT: (void)read_tcsr<U_GLOBAL, U_MAT>(); break;
        case U_VEC: (void)read_tcsr<U_GLOBAL, U_VEC>(); break;
        case U_MACRO: (void)read_tcsr<U_GLOBAL, U_MACRO>(); break;
    }
}

[[maybe_unused]] static inline void fill_words(u64 address, int count, u32 seed) {
    auto ptr = reinterpret_cast<volatile u32 *>(address);
    for (int index = 0; index < count; ++index) ptr[index] = seed + static_cast<u32>(index);
    fence_io();
    if (address >= DAYAN_TEST_DATA_BASE)
        dayan_cache_flush(reinterpret_cast<const void *>(address), count * sizeof(u32));
}

[[maybe_unused]] static inline void copy_words(u64 source, u64 destination, int count) {
    if (source >= DAYAN_TEST_DATA_BASE)
        dayan_cache_invalidate(reinterpret_cast<const void *>(source), count * sizeof(u32));
    auto src = reinterpret_cast<volatile u32 *>(source);
    auto dst = reinterpret_cast<volatile u32 *>(destination);
    for (int index = 0; index < count; ++index) dst[index] = src[index];
    fence_io();
}

#endif
