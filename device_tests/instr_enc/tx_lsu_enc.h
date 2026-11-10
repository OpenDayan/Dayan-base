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

#ifndef TXENC_LSU_ENC_H
#define TXENC_LSU_ENC_H

/* 指令编码库 · Part 2/5：LSU DMA
 * 来源：instruction-encoding/lsu-instruction-encoding.md（spec §3.2）。
 * 仅一条指令族 tx.lsu.dma.[u|d]2[u|d][.g|.s|.sg]，funct7[31:25]=0000100，
 * 4 方向 × 4 地址模式 = 16 变体。形状/步长/转置等由 group 选中的 TCSR 组提供。
 */

#include "tx_base_enc.h"

namespace txenc {

/* 存储方向（src[13]/dst[12]：0=URAM，1=DRAM） */
enum : int {
    MEM_URAM = 0,
    MEM_DRAM = 1,
};

/* tx.lsu.dma.<dir>[.g][.s]
 *   group    [22:20] 访存配置寄存器组 0~7
 *   gather   [23]    源地址表使能（表在 URAM，rs1 为表基址，32B 对齐、64bit 表项）
 *   scatter  [14]    目的地址表使能（rd 为表基址）
 *   src_mem / dst_mem 选 u2u/u2d/d2u/d2d
 *   rs1 [19:15] 源基地址（或源地址表基址）GPR，默认 a0
 *   rd  [11:7]  目的基地址（或目的地址表基址）GPR，默认 a1
 * bit[24] 保留为 0；整条指令在最后一次写入完成后才提交（bar 等待整批）。
 */
constexpr uint32_t lsu_dma_enc(int group = 0, bool gather = false, bool scatter = false,
                               int src_mem = MEM_DRAM, int dst_mem = MEM_URAM,
                               int rs1 = GPR_A0, int rd = GPR_A1)
{
    return (0x04u << 25) |
           (static_cast<uint32_t>(gather ? 1 : 0) << 23) |
           (static_cast<uint32_t>(group & 7) << 20) |
           (static_cast<uint32_t>(rs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(scatter ? 1 : 0) << 14) |
           (static_cast<uint32_t>(src_mem & 1) << 13) |
           (static_cast<uint32_t>(dst_mem & 1) << 12) |
           (static_cast<uint32_t>(rd & 0x1f) << 7) | OPCODE_TX32;
}

} /* namespace txenc */

#endif /* TXENC_LSU_ENC_H */
