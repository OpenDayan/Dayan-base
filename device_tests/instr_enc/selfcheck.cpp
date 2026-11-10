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

/* 指令编码库编译期自检：全部 static_assert，无运行时产物。
 * 锚点来源：Dayan clang builtin 实测（见 README.md）、vec_check_int_add_test
 * 在用裸编码、tensor_test.h 全局编码器公式交叉验证、文档位图手工展开。
 * 编译命令见 README.md；同时验证与 tensor_suite/tensor_test.h 同翻译单元共存。
 */

#include "instr_enc/instr_enc.h"
#include "tensor_suite/tensor_test.h"

namespace tx = txenc;

/* ---- 锚点 1：Dayan clang builtin 实测输出（逐位核对） ---- */
static_assert(tx::macro_convert_enc(tx::DT_FP32, tx::DT_BF16) == 0x000002000205853full);
static_assert(tx::macro_convert_enc(tx::DT_FP32, tx::DT_FP16) == 0x000003000205853full);
static_assert(tx::macro_convert_enc(tx::DT_FP32, tx::DT_FP8_E4M3) == 0x000004000205853full);
static_assert(tx::macro_convert_enc(tx::DT_S32, tx::DT_FP32) == 0x000000200205853full);
static_assert(tx::macro_convert_enc(tx::DT_FP32, tx::DT_S32) == 0x000020000205853full);
static_assert(tx::macro_elemwise_enc(tx::MACRO_ELEM_ADD, tx::MACRO_BD_VV) == 0x0000000000c5853full);
static_assert(tx::macro_elemwise_enc(tx::MACRO_ELEM_MUL, tx::MACRO_BD_VV) == 0x0000020000c5853full);
static_assert(tx::macro_elemwise_enc(tx::MACRO_ELEM_MIN, tx::MACRO_BD_VV) == 0x0000050000c5853full);
static_assert(tx::macro_elemwise_enc(tx::MACRO_ELEM_ADD, tx::MACRO_BD_VV, tx::DT_S32) == 0x0000002000c5853full);
static_assert(tx::macro_sfu_enc(tx::MACRO_SFU_EXP2) == 0x000001000405853full);
static_assert(tx::macro_sfu_enc(tx::MACRO_SFU_SIN) == 0x000002000405853full);
static_assert(tx::macro_sfu_enc(tx::MACRO_SFU_LOG2) == 0x000000000405853full);
static_assert(tx::macro_reduce_enc(tx::MACRO_REDUCE_MAX, false) == 0x0000040006c5853full);
static_assert(tx::macro_reduce_enc(tx::MACRO_REDUCE_MAX, true) == 0x0000440006c5853full);

/* ---- 锚点 2：vec_check_int_add_test 在用的裸编码 ---- */
static_assert(tx::vec_vv_int_enc(tx::VEC_INT_ADD, 1) == 0x8400820bu);

/* ---- 交叉验证：与 tensor_test.h 全局编码器公式一致 ---- */
static_assert(tx::lsu_dma_enc(0, false, false, tx::MEM_DRAM, tx::MEM_URAM) == dma_enc(0, 0, 0, 1, 0));
static_assert(tx::lsu_dma_enc(2, true, true, tx::MEM_URAM, tx::MEM_DRAM) == dma_enc(2, 1, 1, 0, 1));
static_assert(tx::mat_enc(tx::MAT_MM) == ::mat_enc(0));
static_assert(tx::mat_enc(tx::MAT_CONV2DA, 3) == ::mat_enc(3, 3));
static_assert(tx::mat_setptr_enc() == ((1u << 29) | (11u << 20) | (10u << 15) | (4u << 12) | 0x0b));
static_assert(tx::vec_read_enc(tx::VEC_RD_U2V, 0, 0) == ::vec_read_enc(0, 0, 0));
static_assert(tx::vec_read_enc(tx::VEC_RD_SM2V_B32, 1, 1) == ::vec_read_enc(3, 1, 1));
static_assert(tx::vec_write_enc(tx::VEC_WR_V2SM) == ::vec_write_enc(1, 0, 0));
static_assert(tx::tcsr_enc(tx::UNIT_LSU, 13, false) == (((1 & 0xf) << 23) | ((13 & 0x3ff) << 13) | (10 << 7) | 0x0b));
static_assert(tx::tcsr_enc(tx::UNIT_GLOBAL, tx::UNIT_VEC, true) == ((3 << 13) | (1 << 12) | (10 << 7) | 0x0b));

/* ---- 文档位图手工展开抽查（每族至少一条） ---- */
static_assert(tx::vec_s2v_broadcast32_enc(3, 1, 1) ==
              ((0x10u << 26) | (1u << 25) | (1u << 15) | (4u << 12) | (3u << 7) | 0x0b));
static_assert(tx::vec_u2s_enc(false, 5, 1) == 0x4280e50bu);
static_assert(tx::vec_g2s_enc(4) == ((0x10u << 26) | (4u << 23) | (0x20u << 12) | (10u << 7) | 0x0b));
static_assert(tx::vec_masked_read_enc(tx::VEC_RD_U2V_MASK, 0, 1) == 0x4010150bu);
static_assert(tx::vec_s2v_mask_broadcast32_enc(2) == ((0x10u << 26) | (0x0au << 20) | (2u << 7) | 0x0b));
static_assert(tx::vec_masked_write_enc(true, 2, 1, 3) ==
              ((0x11u << 26) | (2u << 22) | (2u << 20) | (1u << 15) | (3u << 12) | (10u << 7) | 0x0b));
static_assert(tx::vec_sparse_enc(tx::VEC_SM2V_SPARSE, 1, 0, 2, 1) == 0x4a10250bu);
static_assert(tx::vec_dyn_enc(tx::VEC_DYN_P0, 3, 1, 0, 2, 1) ==
              ((0x18u << 26) | (1u << 20) | (2u << 12) | (3u << 10) | (1u << 7) | 0x0b));
static_assert(tx::vec_shuffle_static_enc(tx::VEC_SH_COMPRESS, 2) ==
              ((0x1fu << 26) | (2u << 15) | (6u << 12) | (10u << 7) | 0x0b));
static_assert(tx::vec_shuffle_dyn_enc(tx::VEC_SH_ROR_DYN, 4, 2, 1) ==
              ((0x1fu << 26) | (1u << 25) | (2u << 15) | (3u << 12) | (4u << 7) | 0x0b));
static_assert(tx::vec_vv_bitwise_enc(0xe, 1, 2, 1) ==
              ((0x21u << 26) | (1u << 25) | (2u << 20) | (1u << 15) | (1u << 11) | (0xeu << 7) | 0x0b));
static_assert(tx::vec_vv_fp32_enc(tx::VEC_FP32_MUL, tx::RM_RTZ) == ((0x22u << 26) | (2u << 10) | (1u << 7) | 0x0b));
static_assert(tx::vec_mux2_enc(3) == ((0x23u << 26) | (3u << 12) | 0x0b));
static_assert(tx::vec_cmp_enc(tx::VEC_CMP_FP32, 1, 0b101, 1, 2) ==
              ((0x23u << 26) | (2u << 20) | (1u << 15) | (1u << 12) | (2u << 10) | (5u << 7) | 0x0b));
static_assert(tx::vec_shift_enc(tx::VEC_SHIFT_SRA, 7, 3) ==
              ((0x20u << 26) | (1u << 20) | (2u << 17) | (7u << 12) | (3u << 7) | 0x0b));
static_assert(tx::vec_sfu_enc(tx::VEC_SFU_RSQRT, true, 4) ==
              ((0x20u << 26) | (2u << 20) | (1u << 17) | (4u << 12) | (4u << 7) | 0x0b));
static_assert(tx::vec_cvt_enc(tx::VEC_CVT_FP32_BF16) == 0x8025800bu);
static_assert(tx::vec_s_enc(tx::VEC_S_ADD, 0, 1, 2, 3) == 0xf980a08bu);
static_assert(tx::vec_s_enc(tx::VEC_S_GETV, 2) == ((0x3eu << 26) | (3u << 15) | (2u << 10) | 0x0b));
static_assert(tx::vec_s_bitwise_enc(0xa, 1, 2, 3) ==
              ((0x3eu << 26) | (3u << 23) | (1u << 17) | (2u << 15) | (2u << 12) | (2u << 10) | (1u << 7) | 0x0b));
static_assert(tx::vec_branch_enc(tx::VEC_BR_BEQ, -2, 1, 2) == 0xfc7f208bu);
static_assert(tx::macro_elemwise_enc(tx::MACRO_ELEM_MIN, tx::MACRO_BD_SV, tx::DT_BF16, 2) ==
              ((1ull << 46) | (5ull << 40) | (0x02ull << 32) | (12ull << 20) |
               (11ull << 15) | (2ull << 12) | (10ull << 7) | 0x3f));
static_assert(tx::macro_shuffle_benes_enc(tx::MACRO_BENES_CMD_BROADCAST) ==
              ((2ull << 36) | (5ull << 25) | (12ull << 20) | (11ull << 15) | (10ull << 7) | 0x3f));
static_assert(tx::macro_shuffle_compress_enc() ==
              ((1ull << 32) | (5ull << 25) | (12ull << 20) | (11ull << 15) | (10ull << 7) | 0x3f));

/* ---- dtype 表 ---- */
static_assert(tx::dtype_bits(tx::DT_FP32) == 32 && tx::dtype_bits(tx::DT_BF16) == 16 &&
              tx::dtype_bits(tx::DT_S8) == 8 && tx::dtype_bits(tx::DT_MXFP4_E2M1) == 4);
static_assert(tx::DT_S32 == 0x20 && tx::DT_FP16 == 0x03);

/* ---- 发射模板实例化（asm 可编译性） ---- */
volatile unsigned long sink2;
void emit_all(unsigned long v) {
    tx::emit32<tx::vec_vv_int_enc(tx::VEC_INT_ADD, 1)>();
    tx::emit32_a0<tx::vec_read_enc(tx::VEC_RD_U2V)>(v);
    tx::emit32_a0_a1<tx::lsu_dma_enc()>(v, v);
    tx::emit32_a0_a1_a2<tx::mat_enc(tx::MAT_MM)>(v, v, v);
    tx::emit64_a0_a1<tx::macro_convert_enc(tx::DT_FP32, tx::DT_BF16)>(v, v);
    tx::emit64_a0_a1_a2<tx::macro_elemwise_enc(tx::MACRO_ELEM_ADD, tx::MACRO_BD_VV)>(v, v, v);
    tx::write_tcsr<tx::UNIT_LSU, 0>(v);
    sink2 = tx::read_tcsr<tx::UNIT_LSU, 0>();
    tx::sb_produce<tx::UNIT_VEC>(v);
    tx::sb_bar<tx::UNIT_MACRO>();
    sink2 = tx::sb_get_any();
}
