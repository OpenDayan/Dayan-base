#ifndef DAYAN_INTRINSICS_H
#define DAYAN_INTRINSICS_H

#ifndef __cplusplus
#error "dayan_intrinsics.h requires C++"
#endif

#include <stdint.h>
#include <stdio.h>

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

constexpr long DT_FP32 = 0x00;
constexpr long DT_BF16 = 0x02;
constexpr long DT_FP16 = 0x03;
constexpr long DT_E4M3 = 0x04;
constexpr long DT_E5M2 = 0x05;
constexpr long DT_MXFP8_E4M3 = 0x06;
constexpr long DT_MXFP8_E5M2 = 0x07;
constexpr long DT_S32 = 0x20;
constexpr long DT_U32 = 0x21;
constexpr long DT_S16 = 0x22;
constexpr long DT_U16 = 0x23;
constexpr long DT_S8 = 0x24;
constexpr long DT_U8 = 0x25;

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

} // namespace detail
} // namespace BaseNPUInstBuilder

template <long tcsr_id>
__attribute__((always_inline)) static inline void __tx_mat_tcsr_write(long value)
{
    static_assert(tcsr_id >= 0 && tcsr_id < 1024,
                  "MAT TCSR identifier is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TCSR_MAT << 23) | (tcsr_id << 13) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_write<encoding>(value);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void __tx_mat_tcsr_read(long &value)
{
    static_assert(tcsr_id >= 0 && tcsr_id < 1024,
                  "MAT TCSR identifier is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TCSR_MAT << 23) | (tcsr_id << 13) |
        (1L << 12) | BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_read<encoding>(value);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void __tx_lsu_tcsr_write(long value)
{
    static_assert(tcsr_id >= 0 && tcsr_id < 1024,
                  "LSU TCSR identifier is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TCSR_LSU << 23) | (tcsr_id << 13) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_write<encoding>(value);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void __tx_lsu_tcsr_read(long &value)
{
    static_assert(tcsr_id >= 0 && tcsr_id < 1024,
                  "LSU TCSR identifier is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TCSR_LSU << 23) | (tcsr_id << 13) |
        (1L << 12) | BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_read<encoding>(value);
}

template <long group, long direction>
__attribute__((always_inline)) static inline void __tx_lsu_dma(long destination,
                                                               long source)
{
    static_assert(group >= 0 && group < 8, "LSU group is out of range");
    static_assert(direction >= 0 && direction < 4,
                  "LSU direction is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::LSU_FUNCT7 << 25) | (direction << 12) |
        (group << 20) | BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_lsu<encoding>(destination, source);
}

template <long group>
__attribute__((always_inline)) static inline void __tx_lsu_dma_u2u(long destination,
                                                                   long source)
{
    __tx_lsu_dma<group, 0>(destination, source);
}

template <long group>
__attribute__((always_inline)) static inline void __tx_lsu_dma_u2d(long destination,
                                                                   long source)
{
    __tx_lsu_dma<group, 1>(destination, source);
}

template <long group>
__attribute__((always_inline)) static inline void __tx_lsu_dma_d2u(long destination,
                                                                   long source)
{
    __tx_lsu_dma<group, 2>(destination, source);
}

template <long group>
__attribute__((always_inline)) static inline void __tx_lsu_dma_d2d(long destination,
                                                                   long source)
{
    __tx_lsu_dma<group, 3>(destination, source);
}

template <long group, long operation>
__attribute__((always_inline)) static inline void __tx_mat(long destination,
                                                           long left,
                                                           long right)
{
    static_assert(group >= 0 && group < 8, "MAT group is out of range");
    static_assert(operation >= 0 && operation < 5,
                  "MAT operation is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::MAT_FUNCT << 29) | (operation << 12) |
        (group << 25) | BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_mat<encoding>(destination, left, right);
}

template <long group>
__attribute__((always_inline)) static inline void __tx_mat_mm(long destination,
                                                              long left,
                                                              long right)
{
    __tx_mat<group, 0>(destination, left, right);
}

template <long group>
__attribute__((always_inline)) static inline void __tx_mat_mma(long destination,
                                                               long left,
                                                               long right)
{
    __tx_mat<group, 1>(destination, left, right);
}

template <long group>
__attribute__((always_inline)) static inline void __tx_mat_conv2d(long destination,
                                                                  long left,
                                                                  long right)
{
    __tx_mat<group, 2>(destination, left, right);
}

template <long group>
__attribute__((always_inline)) static inline void __tx_mat_conv2da(long destination,
                                                                   long left,
                                                                   long right)
{
    __tx_mat<group, 3>(destination, left, right);
}

template <long group>
__attribute__((always_inline)) static inline void __tx_mat_setptr(long destination,
                                                                  long left,
                                                                  long right)
{
    __tx_mat<group, 4>(destination, left, right);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void __tx_vec_tcsr_write(long value)
{
    static_assert(tcsr_id >= 0 && tcsr_id < 1024,
                  "VEC TCSR identifier is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TCSR_VEC << 23) | (tcsr_id << 13) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_write<encoding>(value);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void __tx_vec_tcsr_read(long &value)
{
    static_assert(tcsr_id >= 0 && tcsr_id < 1024,
                  "VEC TCSR identifier is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TCSR_VEC << 23) | (tcsr_id << 13) |
        (1L << 12) | BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_read<encoding>(value);
}

template <long tcsr_id>
__attribute__((always_inline)) static inline void __tx_macro_tcsr_write(long value)
{
    static_assert(tcsr_id >= 0 && tcsr_id < 1024,
                  "MACRO TCSR identifier is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TCSR_MACRO << 23) | (tcsr_id << 13) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_write<encoding>(value);
}

__attribute__((always_inline)) static inline void __tx_vec_sb_bar(long &value)
{
    constexpr long encoding =
        (0x003L << 13) | (1L << 12) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_read<encoding>(value);
}

template <long vb, long sel>
__attribute__((always_inline)) static inline void __tx_vec_lsu_u2v(long address)
{
    static_assert(vb >= 0 && vb < 2, "VEC bank is out of range");
    static_assert(sel >= 0 && sel < 4, "VEC iterator is out of range");
    constexpr long encoding =
        (0b010000L << 26) | (vb << 25) | (sel << 15) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_write<encoding>(address);
}

template <long vb, long sel>
__attribute__((always_inline)) static inline void __tx_vec_lsu_u2v_broadcast32(long address)
{
    static_assert(vb >= 0 && vb < 2, "VEC bank is out of range");
    static_assert(sel >= 0 && sel < 4, "VEC iterator is out of range");
    constexpr long encoding =
        (0b010000L << 26) | (vb << 25) | (2L << 12) | (sel << 15) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_write<encoding>(address);
}

template <long sel, long source>
__attribute__((always_inline)) static inline void __tx_vec_lsu_v2u(long address)
{
    static_assert(sel >= 0 && sel < 4, "VEC iterator is out of range");
    static_assert(source >= 0 && source < 32,
                  "VEC source register is out of range");
    constexpr long encoding =
        (0b010001L << 26) | (sel << 20) | (source << 15) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_write<encoding>(address);
}

template <long sel, long source, long scalar>
__attribute__((always_inline)) static inline void __tx_vec_lsu_v2u_mask(long address)
{
    static_assert(sel >= 0 && sel < 4, "VEC iterator is out of range");
    static_assert(source >= 0 && source < 32,
                  "VEC source register is out of range");
    static_assert(scalar >= 0 && scalar < 8,
                  "VEC scalar register is out of range");
    constexpr long encoding =
        (0b010001L << 26) | (1L << 22) | (sel << 20) |
        (source << 15) | (scalar << 12) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_write<encoding>(address);
}

template <long vb, long source>
__attribute__((always_inline)) static inline void __tx_vec_shuffle_xor(long mask)
{
    static_assert(vb >= 0 && vb < 2, "VEC bank is out of range");
    static_assert(source >= 0 && source < 32,
                  "VEC source register is out of range");
    constexpr long encoding =
        (0b011111L << 26) | (vb << 25) | (source << 15) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_write<encoding>(mask);
}

template <long vb, long source>
__attribute__((always_inline)) static inline void __tx_vec_shuffle_compress(long stride)
{
    static_assert(vb >= 0 && vb < 2, "VEC bank is out of range");
    static_assert(source >= 0 && source < 32,
                  "VEC source register is out of range");
    constexpr long encoding =
        (0b011111L << 26) | (vb << 25) | (6L << 12) |
        (source << 15) | BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_tcsr_write<encoding>(stride);
}

template <long operation, long vb, long right, long left, long rounding>
__attribute__((always_inline)) static inline void __tx_vec_vv_fp32()
{
    static_assert(operation >= 0 && operation < 3,
                  "VEC FP32 operation is out of range");
    static_assert(vb >= 0 && vb < 2, "VEC bank is out of range");
    static_assert(right >= 0 && right < 32 && left >= 0 && left < 32,
                  "VEC source register is out of range");
    static_assert(rounding >= 0 && rounding < 8,
                  "VEC rounding mode is out of range");
    constexpr long encoding =
        (0b100010L << 26) | (vb << 25) | (right << 20) |
        (left << 15) | (operation << 10) | (rounding << 7) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_void_32<encoding>();
}

template <long vb, long right, long left, long rounding>
__attribute__((always_inline)) static inline void __tx_vec_vv_fp32_add()
{
    __tx_vec_vv_fp32<0, vb, right, left, rounding>();
}

template <long vb, long right, long left, long rounding>
__attribute__((always_inline)) static inline void __tx_vec_vv_fp32_sub()
{
    __tx_vec_vv_fp32<1, vb, right, left, rounding>();
}

template <long vb, long right, long left, long rounding>
__attribute__((always_inline)) static inline void __tx_vec_vv_fp32_mul()
{
    __tx_vec_vv_fp32<2, vb, right, left, rounding>();
}

template <long vb, long right, long left, long scalar>
__attribute__((always_inline)) static inline void __tx_vec_vv_mux2()
{
    static_assert(vb >= 0 && vb < 2, "VEC bank is out of range");
    static_assert(right >= 0 && right < 32 && left >= 0 && left < 32,
                  "VEC source register is out of range");
    static_assert(scalar >= 0 && scalar < 8,
                  "VEC scalar register is out of range");
    constexpr long encoding =
        (0b100011L << 26) | (vb << 25) | (right << 20) |
        (left << 15) | (scalar << 12) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_void_32<encoding>();
}

template <long is_float, long right, long left, long scalar, long predicate>
__attribute__((always_inline)) static inline void __tx_vec_vv_compare()
{
    static_assert(is_float == 0 || is_float == 1,
                  "VEC comparison type is out of range");
    static_assert(right >= 0 && right < 32 && left >= 0 && left < 32,
                  "VEC source register is out of range");
    static_assert(scalar >= 0 && scalar < 8,
                  "VEC scalar register is out of range");
    static_assert(predicate >= 0 && predicate < 8,
                  "VEC comparison predicate is out of range");
    constexpr long encoding =
        (0b100011L << 26) | (right << 20) | (left << 15) |
        (scalar << 12) | ((is_float ? 2L : 1L) << 10) |
        (predicate << 7) | BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_void_32<encoding>();
}

template <long right, long left, long scalar, long predicate>
__attribute__((always_inline)) static inline void __tx_vec_vv_int32_cmp()
{
    __tx_vec_vv_compare<0, right, left, scalar, predicate>();
}

template <long right, long left, long scalar, long predicate>
__attribute__((always_inline)) static inline void __tx_vec_vv_fp32_cmp()
{
    __tx_vec_vv_compare<1, right, left, scalar, predicate>();
}

template <long vb, long operation, long source>
__attribute__((always_inline)) static inline void __tx_vec_v_fp32_sfu()
{
    static_assert(vb >= 0 && vb < 2, "VEC bank is out of range");
    static_assert(operation >= 0 && operation < 64,
                  "VEC SFU operation is out of range");
    static_assert(source >= 0 && source < 32,
                  "VEC source register is out of range");
    constexpr long encoding =
        (1L << 31) | (vb << 25) | (1L << 21) | (operation << 12) |
        (source << 7) | BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_void_32<encoding>();
}

template <long conversion, long vb, long rounding, long source>
__attribute__((always_inline)) static inline void __tx_vec_convert()
{
    static_assert(conversion >= 0 && conversion < 8,
                  "VEC conversion is out of range");
    static_assert(vb >= 0 && vb < 2, "VEC bank is out of range");
    static_assert(rounding >= 0 && rounding < 8,
                  "VEC rounding mode is out of range");
    static_assert(source >= 0 && source < 32,
                  "VEC source register is out of range");
    constexpr long encoding =
        (1L << 31) | (vb << 25) | (1L << 21) | (1L << 18) |
        (conversion << 15) | (rounding << 12) | (source << 7) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_void_32<encoding>();
}

template <long vb, long rounding, long source>
__attribute__((always_inline)) static inline void __tx_vec_cvt_fp32_int32()
{
    __tx_vec_convert<0, vb, rounding, source>();
}

template <long vb, long rounding, long source>
__attribute__((always_inline)) static inline void __tx_vec_cvt_fp32_fp16()
{
    __tx_vec_convert<2, vb, rounding, source>();
}

template <long vb, long rounding, long source>
__attribute__((always_inline)) static inline void __tx_vec_cvt_fp32_bf16()
{
    __tx_vec_convert<3, vb, rounding, source>();
}

template <long vb, long rounding, long source>
__attribute__((always_inline)) static inline void __tx_vec_cvt_fp16_fp32()
{
    __tx_vec_convert<4, vb, rounding, source>();
}

template <long vb, long rounding, long source>
__attribute__((always_inline)) static inline void __tx_vec_cvt_bf16_fp32()
{
    __tx_vec_convert<5, vb, rounding, source>();
}

template <long destination, long right, long left>
__attribute__((always_inline)) static inline void __tx_vec_s_int32_add()
{
    static_assert(destination >= 0 && destination < 8 &&
                      right >= 0 && right < 8 && left >= 0 && left < 8,
                  "VEC scalar register is out of range");
    constexpr long encoding =
        (0b111110L << 26) | (destination << 23) | (1L << 15) |
        (right << 12) | (left << 7) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_void_32<encoding>();
}

template <long destination, long high_lut, long right, long low_lut, long left>
__attribute__((always_inline)) static inline void __tx_vec_s_bitwise()
{
    static_assert(destination >= 0 && destination < 8 &&
                      right >= 0 && right < 8 && left >= 0 && left < 8,
                  "VEC scalar register is out of range");
    static_assert(high_lut >= 0 && high_lut < 4 &&
                      low_lut >= 0 && low_lut < 4,
                  "VEC bitwise lookup is out of range");
    constexpr long encoding =
        (0b111110L << 26) | (destination << 23) | (1L << 17) |
        (high_lut << 15) | (right << 12) | (low_lut << 10) |
        (left << 7) | BaseNPUInstBuilder::OPCODE_CUSTOM_32B;
    BaseNPUInstBuilder::detail::emit_void_32<encoding>();
}

template <long group, long broadcast, long operation>
__attribute__((always_inline)) static inline void __tx_macro_elemwise_fp32(
    void *destination, void *left, void *right)
{
    static_assert(group >= 0 && group < 8, "MACRO group is out of range");
    static_assert(broadcast >= 0 && broadcast < 4,
                  "MACRO broadcast is out of range");
    static_assert(operation == BaseNPUInstBuilder::TX_ELEM_ADD ||
                      operation == BaseNPUInstBuilder::TX_ELEM_SUB ||
                      operation == BaseNPUInstBuilder::TX_ELEM_MUL ||
                      operation == BaseNPUInstBuilder::TX_ELEM_MAX ||
                      operation == BaseNPUInstBuilder::TX_ELEM_MIN,
                  "MACRO elementwise operation is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TX_EXT_ELEMENTWISE << 25) |
        (BaseNPUInstBuilder::DT_FP32 << 32) | (operation << 40) |
        (group << 12) | (broadcast << 46) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_64B;
    BaseNPUInstBuilder::detail::emit_macro_3<encoding>(
        (long)destination, (long)left, (long)right);
}

#define DAYAN_DEFINE_MACRO_ELEMENTWISE(name, operation)                         \
    template <long group, long broadcast>                                      \
    __attribute__((always_inline)) static inline void name(                     \
        void *destination, void *left, void *right)                             \
    {                                                                           \
        __tx_macro_elemwise_fp32<group, broadcast, operation>(                  \
            destination, left, right);                                          \
    }

DAYAN_DEFINE_MACRO_ELEMENTWISE(__tx_macro_elemwise_add_fp32,
                               BaseNPUInstBuilder::TX_ELEM_ADD)
DAYAN_DEFINE_MACRO_ELEMENTWISE(__tx_macro_elemwise_sub_fp32,
                               BaseNPUInstBuilder::TX_ELEM_SUB)
DAYAN_DEFINE_MACRO_ELEMENTWISE(__tx_macro_elemwise_mul_fp32,
                               BaseNPUInstBuilder::TX_ELEM_MUL)
DAYAN_DEFINE_MACRO_ELEMENTWISE(__tx_macro_elemwise_max_fp32,
                               BaseNPUInstBuilder::TX_ELEM_MAX)
DAYAN_DEFINE_MACRO_ELEMENTWISE(__tx_macro_elemwise_min_fp32,
                               BaseNPUInstBuilder::TX_ELEM_MIN)

#undef DAYAN_DEFINE_MACRO_ELEMENTWISE

template <long group, long index>
__attribute__((always_inline)) static inline void __tx_macro_reduce_max_fp32(
    void *destination, void *source, void *indices)
{
    static_assert(group >= 0 && group < 8, "MACRO group is out of range");
    static_assert(index >= 0 && index < 2, "MACRO index mode is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TX_EXT_REDUCE << 25) |
        (BaseNPUInstBuilder::DT_FP32 << 32) |
        (BaseNPUInstBuilder::TX_ELEM_MAX << 40) | (group << 12) |
        (index << 46) | BaseNPUInstBuilder::OPCODE_CUSTOM_64B;
    BaseNPUInstBuilder::detail::emit_macro_3<encoding>(
        (long)destination, (long)source, (long)indices);
}

template <long group>
__attribute__((always_inline)) static inline void __tx_macro_reduce_add_fp32(
    void *destination, void *source)
{
    static_assert(group >= 0 && group < 8, "MACRO group is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TX_EXT_REDUCE << 25) |
        (BaseNPUInstBuilder::DT_FP32 << 32) |
        (BaseNPUInstBuilder::TX_ELEM_ADD << 40) | (group << 12) |
        BaseNPUInstBuilder::OPCODE_CUSTOM_64B;
    BaseNPUInstBuilder::detail::emit_macro_2<encoding>(
        (long)destination, (long)source);
}

template <long group, long operation>
__attribute__((always_inline)) static inline void __tx_macro_sfu_fp32(
    void *destination, void *source)
{
    static_assert(group >= 0 && group < 8, "MACRO group is out of range");
    static_assert(operation >= BaseNPUInstBuilder::TX_SFU_LOG2 &&
                      operation <= BaseNPUInstBuilder::TX_SFU_RECIP,
                  "MACRO SFU operation is out of range");
    constexpr long encoding =
        (BaseNPUInstBuilder::TX_EXT_SFU << 25) | (operation << 40) |
        (group << 12) | BaseNPUInstBuilder::OPCODE_CUSTOM_64B;
    BaseNPUInstBuilder::detail::emit_macro_2<encoding>(
        (long)destination, (long)source);
}

#define DAYAN_DEFINE_MACRO_SFU(name, operation)                                 \
    template <long group>                                                       \
    __attribute__((always_inline)) static inline void name(                     \
        void *destination, void *source)                                        \
    {                                                                           \
        __tx_macro_sfu_fp32<group, operation>(destination, source);             \
    }

DAYAN_DEFINE_MACRO_SFU(__tx_macro_sfu_log2_fp32,
                       BaseNPUInstBuilder::TX_SFU_LOG2)
DAYAN_DEFINE_MACRO_SFU(__tx_macro_sfu_exp2_fp32,
                       BaseNPUInstBuilder::TX_SFU_EXP2)
DAYAN_DEFINE_MACRO_SFU(__tx_macro_sfu_sin_fp32,
                       BaseNPUInstBuilder::TX_SFU_SIN)
DAYAN_DEFINE_MACRO_SFU(__tx_macro_sfu_cos_fp32,
                       BaseNPUInstBuilder::TX_SFU_COS)
DAYAN_DEFINE_MACRO_SFU(__tx_macro_sfu_rsqrt_fp32,
                       BaseNPUInstBuilder::TX_SFU_RSQRT)
DAYAN_DEFINE_MACRO_SFU(__tx_macro_sfu_recip_fp32,
                       BaseNPUInstBuilder::TX_SFU_RECIP)

#undef DAYAN_DEFINE_MACRO_SFU

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
#define MAT_BATCH_SWIZZLE(matrix, batch, spatial)                         \
    (((uint64_t)(uint8_t)(matrix) << 16) |                                \
     ((uint64_t)(uint8_t)(batch) << 8) |                                  \
     (uint64_t)(uint8_t)(spatial))

#endif
