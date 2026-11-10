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

#ifndef TXENC_MAT_ENC_H
#define TXENC_MAT_ENC_H

/* 指令编码库 · Part 4/5：MAT 矩阵指令
 * 来源：instruction-encoding/mat-instruction-encoding.md（spec §4.1.2～§4.1.4）。
 * 共 5 条：mm/mma/conv2d/conv2da/setptr。funct3[31:29]=001，bit[28]=0，
 * group[27:25] 为配置组（不在 MACRO 的 [14:12]）。rd/rs1/rs2 为 GPR 编号，
 * 实际 URAM 地址取自 GPR 的值；dtype、M/N/K、卷积尺寸等在 TCSR 中。
 */

#include "tx_base_enc.h"

namespace txenc {

enum : int {
    MAT_MM      = 0,   /* C = A·B           rs1=A rs2=B rd=C */
    MAT_MMA     = 1,   /* C += A·B          同上 */
    MAT_CONV2D  = 2,   /* O = I⊛K           rs1=I rs2=K rd=O */
    MAT_CONV2DA = 3,   /* O += I⊛K          同上 */
    MAT_SETPTR  = 4,   /* 设置额外参数指针   rs1=A 侧额外矩阵（如 MX scale）rs2=B 侧 */
};

/* mm/mma/conv2d/conv2da 通用编码。默认 rs1=a0、rs2=a1、rd=a2（配 emit32_a0_a1_a2）。 */
constexpr uint32_t mat_enc(int mode, int group = 0,
                           int rs1 = GPR_A0, int rs2 = GPR_A1, int rd = GPR_A2)
{
    return (1u << 29) |
           (static_cast<uint32_t>(group & 7) << 25) |
           (static_cast<uint32_t>(rs2 & 0x1f) << 20) |
           (static_cast<uint32_t>(rs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(mode & 7) << 12) |
           (static_cast<uint32_t>(rd & 0x1f) << 7) | OPCODE_TX32;
}

/* setptr：rd 字段规范固定 00000（不是 C 指针），rs1/rs2 为两侧额外矩阵指针。
 * 发射用 emit32_a0_a1_a2 时 a0 为哑元（rd=x0 不写）。
 */
constexpr uint32_t mat_setptr_enc(int group = 0, int rs1 = GPR_A0, int rs2 = GPR_A1)
{
    return mat_enc(MAT_SETPTR, group, rs1, rs2, 0);
}

} /* namespace txenc */

#endif /* TXENC_MAT_ENC_H */
