#ifndef DAYAN_INTRINSICS_H
#define DAYAN_INTRINSICS_H

#ifndef __cplusplus
#error "dayan_intrinsics.h requires C++"
#endif

#include <stdint.h>
#include <stdio.h>

/*
 * Raw Dayan LSU and MAT instruction builders.  Group and TCSR identifiers
 * are template arguments because they are immediate fields in the encoding.
 */
namespace BaseNPUInstBuilder {

constexpr long OPCODE_CUSTOM_32B = 0x0b;
constexpr long TCSR_LSU = 0x01;
constexpr long TCSR_MAT = 0x02;
constexpr long MAT_FUNCT = 0x01;
constexpr long LSU_FUNCT7 = 0x04;

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

#define MAT_GROUPSZ 32
#define MAT_BATCH_SWIZZLE(matrix, batch, spatial)                         \
    (((uint64_t)(uint8_t)(matrix) << 16) |                                \
     ((uint64_t)(uint8_t)(batch) << 8) |                                  \
     (uint64_t)(uint8_t)(spatial))

#endif
