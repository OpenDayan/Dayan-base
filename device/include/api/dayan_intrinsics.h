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

#ifndef DAYAN_INTRINSICS_H
#define DAYAN_INTRINSICS_H

#ifndef __cplusplus
#error "dayan_intrinsics.h requires C++"
#endif

#include <stdint.h>
#include <stdio.h>

/*
 * Instruction emitters come from Clang's C routine header <rvt_intrinsics.h>. Device compiles
 * must use the Dayan clang so <rvt_intrinsics.h> and __builtin_riscv_tx_* are
 * available. Host compilations skip that header; they only need the TCSR
 * identifiers and datatype constants below.
 */
#if defined(__riscv)
#define __tx_vec_vv_fp32_max __dayan_builtin_tx_vec_vv_fp32_max
#define __tx_vec_s_bitwise __dayan_builtin_tx_vec_s_bitwise
#define __tx_vec_cvt_fp32_int32 __dayan_builtin_tx_vec_cvt_fp32_int32
#define __tx_vec_cvt_s32_fp32 __dayan_builtin_tx_vec_cvt_s32_fp32
#define __tx_vec_cvt_fp32_fp16 __dayan_builtin_tx_vec_cvt_fp32_fp16
#define __tx_vec_cvt_fp32_bf16 __dayan_builtin_tx_vec_cvt_fp32_bf16
#define __tx_vec_cvt_fp16_fp32 __dayan_builtin_tx_vec_cvt_fp16_fp32
#define __tx_vec_cvt_bf16_fp32 __dayan_builtin_tx_vec_cvt_bf16_fp32
#define __tx_vec_cvt_mxfp4_bf16 __dayan_builtin_tx_vec_cvt_mxfp4_bf16
#include <rvt_intrinsics.h>
#undef __tx_vec_vv_fp32_max
#undef __tx_vec_s_bitwise
#undef __tx_vec_cvt_fp32_int32
#undef __tx_vec_cvt_s32_fp32
#undef __tx_vec_cvt_fp32_fp16
#undef __tx_vec_cvt_fp32_bf16
#undef __tx_vec_cvt_fp16_fp32
#undef __tx_vec_cvt_bf16_fp32
#undef __tx_vec_cvt_mxfp4_bf16
#endif

/*
 * Raw Dayan LSU, MAT, VEC, and MACRO instruction builders. Group and TCSR
 * identifiers are template arguments because they are immediate fields.
 */
 namespace BaseNPUInstBuilder {

constexpr long OPCODE_CUSTOM_32B = 0x0b;
constexpr long OPCODE_CUSTOM_64B = 0x3f;
constexpr long TCSR_LSU = 0x01;
constexpr long TCSR_MAT = 0x02;
constexpr long TCSR_VEC = 0x03;
constexpr long TCSR_MACRO = 0x04;
constexpr long MAT_FUNCT = 0x01;
constexpr long LSU_FUNCT7 = 0x04;

constexpr long TX_EXT_ELEMENTWISE = 0x00;
constexpr long TX_EXT_SFU = 0x02;
constexpr long TX_EXT_REDUCE = 0x03;

/* MAT legacy single-format encodings (EXT=0). */
constexpr long DT_FP32 = 0x00;
constexpr long DT_BF16 = 0x02;
constexpr long DT_FP16 = 0x03;
constexpr long DT_E4M3 = 0x04;
constexpr long DT_E5M2 = 0x05;
constexpr long DT_MXFP8_E4M3 = 0x06;
constexpr long DT_MXFP8_E5M2 = 0x07;
constexpr long DT_FP4_E2M1 = 0x08;
constexpr long DT_S32 = 0x10;
constexpr long DT_U32 = 0x11;
constexpr long DT_S16 = 0x12;
constexpr long DT_U16 = 0x13;
constexpr long DT_S8 = 0x14;
constexpr long DT_U8 = 0x15;
constexpr long DT_S4 = 0x18;
constexpr long DT_U4 = 0x19;

/* MAT A/B operand BaseFormat encodings (EXT=1, Version=0). */
constexpr long DT_MATIN_E0M7 = 0x00;
constexpr long DT_MATIN_E1M6 = 0x01;
constexpr long DT_MATIN_E2M5 = 0x02;
constexpr long DT_MATIN_E3M4 = 0x03;
constexpr long DT_MATIN_E4M3 = 0x04;
constexpr long DT_MATIN_E5M2 = 0x05;
constexpr long DT_MATIN_E6M1 = 0x06;
constexpr long DT_MATIN_S8 = 0x08;
constexpr long DT_MATIN_U8 = 0x09;
constexpr long DT_MATIN_TF16 = 0x0a;
constexpr long DT_MATIN_TF32 = 0x10;
constexpr long DT_MATIN_FP16 = 0x11;
constexpr long DT_MATIN_BF16 = 0x12;
constexpr long DT_MATIN_FP4_E2M1 = 0x18;
constexpr long DT_MATIN_S4 = 0x19;
constexpr long DT_MATIN_U4 = 0x1a;
constexpr long DT_MATIN_MX_BIT = 0x20;

constexpr long TX_ELEM_ADD = 0x00;
constexpr long TX_ELEM_SUB = 0x01;
constexpr long TX_ELEM_MUL = 0x02;
constexpr long TX_ELEM_MAX = 0x04;
constexpr long TX_ELEM_MIN = 0x05;

constexpr long TX_SFU_LOG2 = 0;
constexpr long TX_SFU_EXP2 = 1;
constexpr long TX_SFU_SIN = 2;
constexpr long TX_SFU_COS = 3;
constexpr long TX_SFU_RSQRT = 4;
constexpr long TX_SFU_RECIP = 5;

namespace detail {

#define DAYAN_DETAIL_MAP_REGISTER(operand, alias)                         \
    ".set " alias ",0\n"                                                \
    ".ifc " operand ",zero\n.set " alias ",0\n.endif\n"              \
    ".ifc " operand ",ra\n.set " alias ",1\n.endif\n"                \
    ".ifc " operand ",sp\n.set " alias ",2\n.endif\n"                \
    ".ifc " operand ",gp\n.set " alias ",3\n.endif\n"                \
    ".ifc " operand ",tp\n.set " alias ",4\n.endif\n"                \
    ".ifc " operand ",t0\n.set " alias ",5\n.endif\n"                \
    ".ifc " operand ",t1\n.set " alias ",6\n.endif\n"                \
    ".ifc " operand ",t2\n.set " alias ",7\n.endif\n"                \
    ".ifc " operand ",s0\n.set " alias ",8\n.endif\n"                \
    ".ifc " operand ",s1\n.set " alias ",9\n.endif\n"                \
    ".ifc " operand ",a0\n.set " alias ",10\n.endif\n"               \
    ".ifc " operand ",a1\n.set " alias ",11\n.endif\n"               \
    ".ifc " operand ",a2\n.set " alias ",12\n.endif\n"               \
    ".ifc " operand ",a3\n.set " alias ",13\n.endif\n"               \
    ".ifc " operand ",a4\n.set " alias ",14\n.endif\n"               \
    ".ifc " operand ",a5\n.set " alias ",15\n.endif\n"               \
    ".ifc " operand ",a6\n.set " alias ",16\n.endif\n"               \
    ".ifc " operand ",a7\n.set " alias ",17\n.endif\n"               \
    ".ifc " operand ",s2\n.set " alias ",18\n.endif\n"               \
    ".ifc " operand ",s3\n.set " alias ",19\n.endif\n"               \
    ".ifc " operand ",s4\n.set " alias ",20\n.endif\n"               \
    ".ifc " operand ",s5\n.set " alias ",21\n.endif\n"               \
    ".ifc " operand ",s6\n.set " alias ",22\n.endif\n"               \
    ".ifc " operand ",s7\n.set " alias ",23\n.endif\n"               \
    ".ifc " operand ",s8\n.set " alias ",24\n.endif\n"               \
    ".ifc " operand ",s9\n.set " alias ",25\n.endif\n"               \
    ".ifc " operand ",s10\n.set " alias ",26\n.endif\n"              \
    ".ifc " operand ",s11\n.set " alias ",27\n.endif\n"              \
    ".ifc " operand ",t3\n.set " alias ",28\n.endif\n"               \
    ".ifc " operand ",t4\n.set " alias ",29\n.endif\n"               \
    ".ifc " operand ",t5\n.set " alias ",30\n.endif\n"               \
    ".ifc " operand ",t6\n.set " alias ",31\n.endif\n"

template <long encoding>
__attribute__((always_inline)) static inline void emit_tcsr_write(long value)
{
    asm volatile(
        DAYAN_DETAIL_MAP_REGISTER("%0", "_DAYAN_R0")
        ".align 2\n"
        ".insn 4, %1 | (_DAYAN_R0 << 7)\n"
        :
        : "r"(value), "i"(encoding));
}

template <long encoding>
__attribute__((always_inline)) static inline void emit_tcsr_read(long &value)
{
    asm volatile(
        DAYAN_DETAIL_MAP_REGISTER("%0", "_DAYAN_R0")
        ".align 2\n"
        ".insn 4, %1 | (_DAYAN_R0 << 7)\n"
        : "=r"(value)
        : "i"(encoding));
}

template <long encoding>
__attribute__((always_inline)) static inline void emit_lsu(long destination,
                                                            long source)
{
    asm volatile(
        DAYAN_DETAIL_MAP_REGISTER("%0", "_DAYAN_R0")
        DAYAN_DETAIL_MAP_REGISTER("%1", "_DAYAN_R1")
        ".align 2\n"
        ".insn 4, %2 | (_DAYAN_R0 << 7) | (_DAYAN_R1 << 15)\n"
        :
        : "r"(destination), "r"(source), "i"(encoding));
}

template <long encoding>
__attribute__((always_inline)) static inline void emit_mat(long destination,
                                                            long left,
                                                            long right)
{
    asm volatile(
        DAYAN_DETAIL_MAP_REGISTER("%0", "_DAYAN_R0")
        DAYAN_DETAIL_MAP_REGISTER("%1", "_DAYAN_R1")
        DAYAN_DETAIL_MAP_REGISTER("%2", "_DAYAN_R2")
        ".align 2\n"
        ".insn 4, %3 | (_DAYAN_R0 << 7) | (_DAYAN_R1 << 15) | "
        "(_DAYAN_R2 << 20)\n"
        :
        : "r"(destination), "r"(left), "r"(right), "i"(encoding));
}

template <long encoding>
__attribute__((always_inline)) static inline void emit_void_32()
{
    asm volatile(
        ".align 2\n"
        ".insn 4, %0\n"
        :
        : "i"(encoding));
}

template <long encoding>
__attribute__((always_inline)) static inline void emit_macro_2(long destination,
                                                                long source)
{
    asm volatile(
        DAYAN_DETAIL_MAP_REGISTER("%0", "_DAYAN_R0")
        DAYAN_DETAIL_MAP_REGISTER("%1", "_DAYAN_R1")
        ".align 3\n"
        ".insn 8, %2 | (_DAYAN_R1 << 7) | (_DAYAN_R0 << 15)\n"
        :
        : "r"(source), "r"(destination), "i"(encoding));
}

template <long encoding>
__attribute__((always_inline)) static inline void emit_macro_3(long destination,
                                                                long left,
                                                                long right)
{
    asm volatile(
        DAYAN_DETAIL_MAP_REGISTER("%0", "_DAYAN_R0")
        DAYAN_DETAIL_MAP_REGISTER("%1", "_DAYAN_R1")
        DAYAN_DETAIL_MAP_REGISTER("%2", "_DAYAN_R2")
        ".align 3\n"
        ".insn 8, %3 | (_DAYAN_R2 << 7) | (_DAYAN_R0 << 15) | "
        "(_DAYAN_R1 << 20)\n"
        :
        : "r"(left), "r"(right), "r"(destination), "i"(encoding));
}

#undef DAYAN_DETAIL_MAP_REGISTER

template <long Vb, long Vrs1, long Rm, long Operation>
__attribute__((always_inline)) static inline void emit_vec_cvt()
{
    static_assert(Vb >= 0 && Vb < 2);
    static_assert(Vrs1 >= 0 && Vrs1 < 32);
    static_assert(Rm >= 0 && Rm < 8);
    static_assert(Operation >= 0 && Operation < 8);
    constexpr long encoding =
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B |
        (Vrs1 << 7) |
        (Rm << 12) |
        (Operation << 15) |
        (1L << 18) |
        (2L << 20) |
        (Vb << 25) |
        (0x20L << 26);
    emit_void_32<encoding>();
}

template <long Vb, long Vrs1, long Vrs2, long Rm, long Operation>
__attribute__((always_inline)) static inline void emit_vec_vv_fp32()
{
    static_assert(Vb >= 0 && Vb < 2);
    static_assert(Vrs1 >= 0 && Vrs1 < 32);
    static_assert(Vrs2 >= 0 && Vrs2 < 32);
    static_assert(Rm >= 0 && Rm < 8);
    static_assert(Operation >= 0 && Operation < 32);
    constexpr long encoding =
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B |
        (Rm << 7) |
        (Operation << 10) |
        (Vrs1 << 15) |
        (Vrs2 << 20) |
        (Vb << 25) |
        (0x22L << 26);
    emit_void_32<encoding>();
}

} // namespace detail
} // namespace BaseNPUInstBuilder

#if defined(__riscv)
template <long Sd, long S1, long S2, long Lut>
__attribute__((always_inline)) static inline void __tx_vec_s_bitwise()
{
    static_assert(Sd >= 0 && Sd < 8);
    static_assert(S1 >= 0 && S1 < 8);
    static_assert(S2 >= 0 && S2 < 8);
    static_assert(Lut >= 0 && Lut < 16);
    // The LUT spans func8[1:0] and func2.
    constexpr long encoding = BaseNPUInstBuilder::OPCODE_CUSTOM_32B |
        (S1 << 7) | ((Lut & 3) << 10) | (S2 << 12) | ((4 | (Lut >> 2)) << 15) | (Sd << 23) | (0x3eL << 26);
    BaseNPUInstBuilder::detail::emit_void_32<encoding>();
}

template <long Vb, long Vrs1, long Op>
__attribute__((always_inline)) static inline void __tx_vec_v_fp32_eru()
{
    static_assert(Vb >= 0 && Vb < 2);
    static_assert(Vrs1 >= 0 && Vrs1 < 32);
    static_assert(Op >= 0 && Op < 2);
    constexpr long encoding = BaseNPUInstBuilder::OPCODE_CUSTOM_32B |
        (Vrs1 << 7) | (Op << 12) | (1L << 17) | (2L << 20) | (Vb << 25) | (0x20L << 26);
    BaseNPUInstBuilder::detail::emit_void_32<encoding>();
}

template <long Vb, long Vrs1, long Vrs2, long Rm>
__attribute__((always_inline)) static inline void __tx_vec_vv_fp32_max()
{
    BaseNPUInstBuilder::detail::emit_vec_vv_fp32<Vb, Vrs1, Vrs2, Rm, 4>();
}

template <long Vb, long Vrs1, long Rm>
__attribute__((always_inline)) static inline void __tx_vec_cvt_fp32_int32()
{
    BaseNPUInstBuilder::detail::emit_vec_cvt<Vb, Vrs1, Rm, 0>();
}

template <long Vb, long Vrs1, long Rm>
__attribute__((always_inline)) static inline void __tx_vec_cvt_s32_fp32()
{
    BaseNPUInstBuilder::detail::emit_vec_cvt<Vb, Vrs1, Rm, 1>();
}

template <long Vb, long Vrs1, long Rm>
__attribute__((always_inline)) static inline void __tx_vec_cvt_fp32_fp16()
{
    BaseNPUInstBuilder::detail::emit_vec_cvt<Vb, Vrs1, Rm, 2>();
}

template <long Vb, long Vrs1, long Rm>
__attribute__((always_inline)) static inline void __tx_vec_cvt_fp32_bf16()
{
    BaseNPUInstBuilder::detail::emit_vec_cvt<Vb, Vrs1, Rm, 3>();
}

template <long Vb, long Vrs1, long Rm>
__attribute__((always_inline)) static inline void __tx_vec_cvt_fp16_fp32()
{
    BaseNPUInstBuilder::detail::emit_vec_cvt<Vb, Vrs1, Rm, 4>();
}

template <long Vb, long Vrs1, long Rm>
__attribute__((always_inline)) static inline void __tx_vec_cvt_bf16_fp32()
{
    BaseNPUInstBuilder::detail::emit_vec_cvt<Vb, Vrs1, Rm, 5>();
}

template <long Vb, long Vrs1, long Rm>
__attribute__((always_inline)) static inline void __tx_vec_cvt_mxfp4_bf16()
{
    BaseNPUInstBuilder::detail::emit_vec_cvt<Vb, Vrs1, Rm, 7>();
}
#endif

enum LSU_TCSR {
    LSU_BLOCK = 0,
    LSU_DIM0 = 1,
    LSU_DIM1 = 2,
    LSU_DIM2 = 3,
    LSU_INBLOCKSTRIDE = 4,
    LSU_INDIM0STRIDE = 5,
    LSU_INDIM1STRIDE = 6,
    LSU_INDIM2STRIDE = 7,
    LSU_OUTBLOCKSTRIDE = 8,
    LSU_OUTDIM0STRIDE = 9,
    LSU_OUTDIM1STRIDE = 10,
    LSU_OUTDIM2STRIDE = 11,
    LSU_BLOCKTRANSPOSE = 12,
    LSU_BLOCKPADDING = 13,
    LSU_REQUESTREADSIZE = 14,
    LSU_REQUESTPADDING = LSU_REQUESTREADSIZE,
    LSU_PADDINGVALUE = 15,
    LSU_URAMWRITESWIZZLEMBS = 16,
    LSU_URAMREADSWIZZLEMBS = 17,
    LSU_REQUESTWRITESIZE = 18,
    LSU_TABLESIZE = 19,
    LSU_TABLESTRIDE = 20,
};

#define LSU_GROUPSZ 32

enum VEC_TCSR {
    VEC_CONTROL = 0,
    VEC_SYNC = 1,
    VEC_DIM0 = 2,  /* Innermost fixed-control-flow loop; count minus one. */
    VEC_DIM1 = 3,  /* Outer fixed-control-flow loop; count minus one. */
    VEC_DIM2 = 4,
    VEC_DIM3 = 5,
    VEC_DIM4 = 6,
    VEC_DIM5 = 7,
    VEC_IMPL_DEF_HINT = 8,
    VEC_S0_BASE = 9,
    VEC_S0_DIM0_STRIDE = 10,
    VEC_S0_DIM1_STRIDE = 11,
    VEC_S0_DIM2_STRIDE = 12,
    VEC_S0_DIM3_STRIDE = 13,
    VEC_S0_DIM4_STRIDE = 14,
    VEC_S0_DIM5_STRIDE = 15,
    VEC_S1_BASE = 16,
    VEC_S1_DIM0_STRIDE = 17,
    VEC_S1_DIM1_STRIDE = 18,
    VEC_S1_DIM2_STRIDE = 19,
    VEC_S1_DIM3_STRIDE = 20,
    VEC_S1_DIM4_STRIDE = 21,
    VEC_S1_DIM5_STRIDE = 22,
    VEC_S2_BASE = 23,
    VEC_S2_DIM0_STRIDE = 24,
    VEC_S2_DIM1_STRIDE = 25,
    VEC_S2_DIM2_STRIDE = 26,
    VEC_S2_DIM3_STRIDE = 27,
    VEC_S2_DIM4_STRIDE = 28,
    VEC_S2_DIM5_STRIDE = 29,
    VEC_S3_BASE = 30,
    VEC_S3_DIM0_STRIDE = 31,
    VEC_S3_DIM1_STRIDE = 32,
    VEC_S3_DIM2_STRIDE = 33,
    VEC_S3_DIM3_STRIDE = 34,
    VEC_S3_DIM4_STRIDE = 35,
    VEC_S3_DIM5_STRIDE = 36,
    VEC_V0_BASE = 37,
    VEC_V0_DIM0_STRIDE = 38,
    VEC_V0_DIM1_STRIDE = 39,
    VEC_V0_DIM2_STRIDE = 40,
    VEC_V0_DIM3_STRIDE = 41,
    VEC_V0_DIM4_STRIDE = 42,
    VEC_V0_DIM5_STRIDE = 43,
    VEC_V1_BASE = 44,
    VEC_V1_DIM0_STRIDE = 45,
    VEC_V1_DIM1_STRIDE = 46,
    VEC_V1_DIM2_STRIDE = 47,
    VEC_V1_DIM3_STRIDE = 48,
    VEC_V1_DIM4_STRIDE = 49,
    VEC_V1_DIM5_STRIDE = 50,
    VEC_V2_BASE = 51,
    VEC_V2_DIM0_STRIDE = 52,
    VEC_V2_DIM1_STRIDE = 53,
    VEC_V2_DIM2_STRIDE = 54,
    VEC_V2_DIM3_STRIDE = 55,
    VEC_V2_DIM4_STRIDE = 56,
    VEC_V2_DIM5_STRIDE = 57,
    VEC_V3_BASE = 58,
    VEC_V3_DIM0_STRIDE = 59,
    VEC_V3_DIM1_STRIDE = 60,
    VEC_V3_DIM2_STRIDE = 61,
    VEC_V3_DIM3_STRIDE = 62,
    VEC_V3_DIM4_STRIDE = 63,
    VEC_V3_DIM5_STRIDE = 64,
    VEC_V0_URAM_CHANNEL_STRIDE = 65,
    VEC_V1_URAM_CHANNEL_STRIDE = 66,
    VEC_V2_URAM_CHANNEL_STRIDE = 67,
    VEC_V3_URAM_CHANNEL_STRIDE = 68,
};

#if defined(__riscv)
template <long GROUPBASE, bool TRANSPOSE = false,
          long TRANSPOSE_LOG2ELEMSIZE = 1>
static inline void set_lsu_matrix(long &checksum, long dim1, long dim0,
                                  long input_stride, long output_stride)
{
    if (TRANSPOSE) {
        const long sub_block_size = 64 / (1L << TRANSPOSE_LOG2ELEMSIZE);

        if (dim0 % 64 != 0)
            printf("Warning: transpose dim0(%ld) is not 64-byte aligned.\n",
                   dim0);
        if (dim1 % sub_block_size != 0)
            printf("Warning: transpose dim1(%ld) is not aligned to %ld "
                   "elements.\n", dim1, sub_block_size);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCK>(
            sub_block_size - 1);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM0>(
            dim0 / 64 - 1);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM1>(
            dim1 / sub_block_size - 1);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(
            input_stride);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM1STRIDE>(
            input_stride * sub_block_size);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(
            output_stride);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(
            output_stride * sub_block_size);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(64);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(
            TRANSPOSE_LOG2ELEMSIZE);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKPADDING>(
            sub_block_size - 1);
    } else {
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCK>(0);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM0>(
            dim0 / 64 - 1);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM1>(dim1 - 1);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM0STRIDE>(64);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_INDIM1STRIDE>(
            input_stride);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(64);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(
            output_stride);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
        __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    }
    __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(63);
    __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(63);
    __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<GROUPBASE * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
    checksum = 0;
}
#endif

enum MAT_TCSR {
    MAT_K = 0,
    MAT_M = 3,
    MAT_N = 7,
    MAT_ASTRIDE = 8,
    MAT_PERSISTENTMODE = 9,
    MAT_BSTRIDE = 12,
    MAT_CSTRIDE = 13,
    MAT_ASWIZZLEMBS = 15,
    MAT_BSWIZZLEMBS = 16,
    MAT_CSWIZZLEMBS = 17,
    MAT_DATATYPE = 18,
    MAT_ASCALESTRIDE = 19,
    MAT_BSCALESTRIDE = 23,
    MAT_PADL = 24,
    MAT_PADU = 25,
    MAT_DILATIONW = 26,
    MAT_DILATIONH = 27,
    MAT_STRIDEW = 28,
    MAT_STRIDEH = 29,
    MAT_STRIDEDILATIONW = 30,
    MAT_STRIDEDILATIONH = 31,
};

enum MACRO_TCSR {
    MACRO_DIM0 = 0,
    MACRO_DIM1 = 1,
    MACRO_DIM2 = 2,
    MACRO_DIM3 = 3,
    MACRO_SRC0DIM1STRIDE = 4,
    MACRO_SRC0DIM2STRIDE = 5,
    MACRO_SRC0DIM3STRIDE = 6,
    MACRO_SRC1DIM1STRIDE = 7,
    MACRO_SRC1DIM2STRIDE = 8,
    MACRO_SRC1DIM3STRIDE = 9,
    MACRO_DSTDIM1STRIDE = 10,
    MACRO_DSTDIM2STRIDE = 11,
    MACRO_DSTDIM3STRIDE = 12,
    MACRO_SRC0SWIZZLEMBS = 13,
    MACRO_SRC1SWIZZLEMBS = 14,
    MACRO_DSTSWIZZLEMBS = 15,
};

#define MAT_GROUPSZ 32
#define MACRO_GROUPSZ 32
#define MAT_DATATYPE_PAIR(a, b) (0x8000L | (((b) & 0x3fL) << 6) | ((a) & 0x3fL))
#define MAT_BATCH_SWIZZLE(matrix, batch, spatial)                         \
    (((uint64_t)(uint8_t)(matrix) << 16) |                                \
     ((uint64_t)(uint8_t)(batch) << 8) |                                  \
     (uint64_t)(uint8_t)(spatial))

#endif
