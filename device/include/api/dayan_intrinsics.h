#ifndef DAYAN_INTRINSICS_H
#define DAYAN_INTRINSICS_H

#ifndef __cplusplus
#error "dayan_intrinsics.h requires C++"
#endif

#include <stdint.h>
#include <stdio.h>

/*
 * Instruction emitters come from Clang's RVT builtin header. Device compiles
 * must use the Dayan clang so <rvt_intrinsics.h> and __builtin_riscv_tx_* are
 * available. Host compilations skip that header; they only need the TCSR
 * identifiers and datatype constants below.
 */
#if defined(__riscv)
#include <rvt_intrinsics.h>
#endif

namespace BaseNPUInstBuilder {

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

} // namespace BaseNPUInstBuilder

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
#define MAT_BATCH_SWIZZLE(matrix, batch, spatial)                         \
    (((uint64_t)(uint8_t)(matrix) << 16) |                                \
     ((uint64_t)(uint8_t)(batch) << 8) |                                  \
     (uint64_t)(uint8_t)(spatial))

#endif
