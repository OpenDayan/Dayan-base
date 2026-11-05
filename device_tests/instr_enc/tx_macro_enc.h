#ifndef TXENC_MACRO_ENC_H
#define TXENC_MACRO_ENC_H

/* 指令编码库 · Part 5/5：MACRO 64 位复合指令
 * 来源：instruction-encoding/macro-instruction-encoding.md（spec §6.2）。
 * opcode[6:0]=0111111，[63:48] 填 0（低 48 位有效）。Extension[31:25] 分族：
 * 0=elemwise 1=convert 2=sfu 3=reduce 5=shuffle。group[14:12] 为 MACRO 配置组。
 * Dim/stride/swizzle 等 shape 参数在 TCSR 中（Dim 存实际尺寸减 1）。
 *
 * GPR 约定（与 Dayan clang builtin 实测一致，配 emit64_* 使用）：
 *   三操作数族（elemwise/reduce/shuffle）：rd=a0(10)，rs1=a1(11)，rs2=a2(12)；
 *   两操作数族（convert/sfu）：rd=a0(10)，rs1=a1(11)，rs2 固定 0。
 */

#include "tx_base_enc.h"

namespace txenc {

enum : int {
    MACRO_EXT_ELEMWISE = 0,
    MACRO_EXT_CONVERT  = 1,
    MACRO_EXT_SFU      = 2,
    MACRO_EXT_REDUCE   = 3,
    MACRO_EXT_SHUFFLE  = 5,
};

/* 64 位布局公共低 32 位：rs2[24:20] rs1[19:15] group[14:12] rd[11:7] opcode。 */
constexpr uint64_t macro_low_enc(int extension, int group, int rd, int rs1, int rs2)
{
    return (static_cast<uint64_t>(extension & 0x7f) << 25) |
           (static_cast<uint64_t>(rs2 & 0x1f) << 20) |
           (static_cast<uint64_t>(rs1 & 0x1f) << 15) |
           (static_cast<uint64_t>(group & 7) << 12) |
           (static_cast<uint64_t>(rd & 0x1f) << 7) | OPCODE_TX64;
}

/* ---- elemwise（§6.2.1）：DataType[39:32] Operation[45:40] Bd[47:46] ---- */
enum : int {
    MACRO_ELEM_ADD = 0, MACRO_ELEM_SUB = 1, MACRO_ELEM_MUL = 2,
    MACRO_ELEM_MAX = 4, MACRO_ELEM_MIN = 5,
};
enum : int { MACRO_BD_VV = 0, MACRO_BD_SV = 1, MACRO_BD_IV = 2 };
/* Bd=sv：rs1 仍为源 0 tensor，仅复用 (d3,d2,d1) 的 Dim0=0；
 * Bd=iv：rs1 改为按 dtype 解释的立即标量位型，不访问源 0 URAM。 */
constexpr uint64_t macro_elemwise_enc(int operation, int bd, uint8_t dtype = DT_FP32,
                                     int group = 0, int rd = GPR_A0,
                                     int rs1 = GPR_A1, int rs2 = GPR_A2)
{
    return (static_cast<uint64_t>(bd & 3) << 46) |
           (static_cast<uint64_t>(operation & 0x3f) << 40) |
           (static_cast<uint64_t>(dtype) << 32) |
           macro_low_enc(MACRO_EXT_ELEMWISE, group, rd, rs1, rs2);
}

/* ---- convert（§6.2.2）：SrcType[39:32] DstType[47:40]，rs2 固定 00000 ---- */
constexpr uint64_t macro_convert_enc(uint8_t src_type, uint8_t dst_type, int group = 0,
                                     int rd = GPR_A0, int rs1 = GPR_A1)
{
    return (static_cast<uint64_t>(dst_type) << 40) |
           (static_cast<uint64_t>(src_type) << 32) |
           macro_low_enc(MACRO_EXT_CONVERT, group, rd, rs1, 0);
}

/* ---- sfu（§6.2.3）：Operation 实测在 [47:40]（Dayan clang builtin 逐位核对：
 * log2/exp2/sin = 0/1/2 → 高字 0x0000/0x0100/0x0200；文档 [39:32] 有误），
 * rs2 固定 00000。 */
enum : int {
    MACRO_SFU_LOG2 = 0, MACRO_SFU_EXP2 = 1, MACRO_SFU_SIN = 2, MACRO_SFU_COS = 3,
    MACRO_SFU_RSQRT = 4, MACRO_SFU_RECIP = 5,
};
constexpr uint64_t macro_sfu_enc(int operation, int group = 0,
                                 int rd = GPR_A0, int rs1 = GPR_A1)
{
    return (static_cast<uint64_t>(operation & 0xff) << 40) |
           macro_low_enc(MACRO_EXT_SFU, group, rd, rs1, 0);
}

/* ---- reduce（§6.2.4）：DataType[39:32] Operation[45:40] Id[46]，[47] 保留 0 ----
 * Id=1（仅 max/min）：rs2 为 uint32 索引 tensor 基地址，额外写回 Dim0 下标。
 */
enum : int {
    MACRO_REDUCE_ADD = 0, MACRO_REDUCE_MAX = 4, MACRO_REDUCE_MIN = 5,
};
constexpr uint64_t macro_reduce_enc(int operation, bool index_write,
                                    uint8_t dtype = DT_FP32, int group = 0,
                                    int rd = GPR_A0, int rs1 = GPR_A1,
                                    int rs2 = GPR_A2)
{
    return (static_cast<uint64_t>(index_write ? 1 : 0) << 46) |
           (static_cast<uint64_t>(operation & 0x3f) << 40) |
           (static_cast<uint64_t>(dtype) << 32) |
           macro_low_enc(MACRO_EXT_REDUCE, group, rd, rs1, rs2);
}

/* ---- shuffle.benes（§6.2.5）：ShuffleType[35:32]=0000 Bd[37:36]，[47:38] 保留 0 ----
 * rs1=源数据流基址，rs2=32B 命令流（Pattern）基址。命令块低 30B 有效：
 * Mask(8B，每 2bit 控制一个输出字节) + PaddingValue(4B) + Route(18B，9 级×16 个
 * 2×2 开关，须编码完整 32B 双射) + Reserve(2B=0)。
 */
enum : int {
    MACRO_BENES_VV = 0, MACRO_BENES_SRC_BROADCAST = 1,
    MACRO_BENES_CMD_BROADCAST = 2, MACRO_BENES_BOTH_BROADCAST = 3,
};
constexpr uint64_t macro_shuffle_benes_enc(int bd, int group = 0, int rd = GPR_A0,
                                           int rs1 = GPR_A1, int rs2 = GPR_A2)
{
    return (static_cast<uint64_t>(bd & 3) << 36) |
           macro_low_enc(MACRO_EXT_SHUFFLE, group, rd, rs1, rs2);
}

/* ---- shuffle.compress（§6.2.6）：ShuffleType[35:32]=0001，无 Bd，[47:36] 保留 0 ----
 * rs1=32B 源块基址，rs2=有效 bitmask 基址。
 */
constexpr uint64_t macro_shuffle_compress_enc(int group = 0, int rd = GPR_A0,
                                              int rs1 = GPR_A1, int rs2 = GPR_A2)
{
    return (1ull << 32) |
           macro_low_enc(MACRO_EXT_SHUFFLE, group, rd, rs1, rs2);
}

} /* namespace txenc */

#endif /* TXENC_MACRO_ENC_H */
