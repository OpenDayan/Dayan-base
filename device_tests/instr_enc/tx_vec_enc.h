#ifndef TXENC_VEC_ENC_H
#define TXENC_VEC_ENC_H

/* 指令编码库 · Part 3/5：VEC 全指令族
 * 来源：instruction-encoding/vec-instruction-encoding.md（spec §5.7，89 条）。
 * 家族判别码在 [31:26]；vrs1/vrs2 为 5 位向量源选择子（最高位选循环队列，
 * 低 4 位为队列内偏移，0 表示最近一次写入该队列的结果）；vb 选择写回队列；
 * s1/s2/sd 为 3 位线程内标量寄存器号；sel 为 2 位硬件迭代器组号。
 * 分支指令仅模式二可用。
 */

#include "tx_base_enc.h"

namespace txenc {

/* ---- 家族判别码 [31:26] ---- */
enum : uint32_t {
    VEC_P_READ     = 0x10u << 26,  /* 010000 顺序/掩码读、标量读入 */
    VEC_P_WRITE    = 0x11u << 26,  /* 010001 顺序/掩码写 */
    VEC_P_SHUFFLE  = 0x1fu << 26,  /* 011111 shuffle */
    VEC_P_V        = 0x20u << 26,  /* 100000 v 单输入（位移/SFU/cvt） */
    VEC_P_VV_INT   = 0x21u << 26,  /* 100001 vv 双输入整数 */
    VEC_P_VV_FP32  = 0x22u << 26,  /* 100010 vv 双输入 FP32 */
    VEC_P_MUX_CMP  = 0x23u << 26,  /* 100011 mux2 / cmp */
    VEC_P_S        = 0x3eu << 26,  /* 111110 s 标量运算 */
    VEC_P_BRANCH   = 0x3fu << 26,  /* 111111 s 分支（模式二） */
};

/* ---- §1 顺序读（无掩码，向量目标）----
 * subfunc：0=u2v 1=sm2v 2=u2v.broadcast32 3=sm2v.broadcast32 5=g2v.broadcast32。
 */
enum : int {
    VEC_RD_U2V = 0, VEC_RD_SM2V = 1, VEC_RD_U2V_B32 = 2,
    VEC_RD_SM2V_B32 = 3, VEC_RD_G2V_B32 = 5,
};
constexpr uint32_t vec_read_enc(int subfunc, int sel = 0, int vb = 0, int rd = GPR_A0)
{
    return VEC_P_READ | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(sel & 3) << 15) |
           (static_cast<uint32_t>(subfunc & 7) << 12) |
           (static_cast<uint32_t>(rd & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- §2 s2v.broadcast32（标量寄存器广播，[14:12] 固定 100） ---- */
constexpr uint32_t vec_s2v_broadcast32_enc(int s1, int sel = 0, int vb = 0)
{
    return VEC_P_READ | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(sel & 3) << 15) | (4u << 12) |
           (static_cast<uint32_t>(s1 & 7) << 7) | OPCODE_TX32;
}

/* ---- §3 u2s / sm2s（读入标量寄存器，subfunc 110/111） ---- */
constexpr uint32_t vec_u2s_enc(bool from_sm, int sd, int sel = 0, int rd = GPR_A0)
{
    return VEC_P_READ | (static_cast<uint32_t>(sd & 7) << 23) |
           (static_cast<uint32_t>(sel & 3) << 15) |
           (static_cast<uint32_t>(from_sm ? 7 : 6) << 12) |
           (static_cast<uint32_t>(rd & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- §4 g2s（通用寄存器 → 标量寄存器，func[22:12]=00000100000） ---- */
constexpr uint32_t vec_g2s_enc(int sd, int rd = GPR_A0)
{
    return VEC_P_READ | (static_cast<uint32_t>(sd & 7) << 23) |
           (0x20u << 12) | (static_cast<uint32_t>(rd & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- §5 掩码读（向量目标，subfunc5 在 [24:20]）----
 * 1=u2v.mask 3=sm2v.mask 5=u2v.mask.b32 7=sm2v.mask.b32 9=g2v.mask.b32；
 * vrs1 为掩码 bit=0 时的回退向量，s2 提供 64bit 谓词位图。
 */
enum : int {
    VEC_RD_U2V_MASK = 0x01, VEC_RD_SM2V_MASK = 0x03, VEC_RD_U2V_MASK_B32 = 0x05,
    VEC_RD_SM2V_MASK_B32 = 0x07, VEC_RD_G2V_MASK_B32 = 0x09,
};
constexpr uint32_t vec_masked_read_enc(int subfunc5, int vrs1 = 0, int s2 = 0,
                                       int rd = GPR_A0, int vb = 0)
{
    return VEC_P_READ | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(subfunc5 & 0x1f) << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(s2 & 7) << 12) |
           (static_cast<uint32_t>(rd & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- §6 s2v.mask.broadcast32（subfunc5 固定 01010） ---- */
constexpr uint32_t vec_s2v_mask_broadcast32_enc(int s1, int s2 = 0, int vrs1 = 0, int vb = 0)
{
    return VEC_P_READ | (static_cast<uint32_t>(vb & 1) << 25) | (0x0au << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(s2 & 7) << 12) |
           (static_cast<uint32_t>(s1 & 7) << 7) | OPCODE_TX32;
}

/* ---- §7 顺序写（无掩码，[25:22]=0000）----
 * subfunc：0=v2u 1=v2sm 2=s2u 3=s2sm。
 */
enum : int { VEC_WR_V2U = 0, VEC_WR_V2SM = 1, VEC_WR_S2U = 2, VEC_WR_S2SM = 3 };
constexpr uint32_t vec_write_enc(int subfunc, int sel = 0, int vrs1 = 0, int rd = GPR_A0)
{
    return VEC_P_WRITE | (static_cast<uint32_t>(sel & 3) << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(subfunc & 7) << 12) |
           (static_cast<uint32_t>(rd & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- §8 掩码写（subfunc4 在 [25:22]：0001=v2u.mask 0010=v2sm.mask） ---- */
constexpr uint32_t vec_masked_write_enc(bool to_sm, int sel = 0, int vrs1 = 0,
                                        int s2 = 0, int rd = GPR_A0)
{
    return VEC_P_WRITE | (static_cast<uint32_t>(to_sm ? 2 : 1) << 22) |
           (static_cast<uint32_t>(sel & 3) << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(s2 & 7) << 12) |
           (static_cast<uint32_t>(rd & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- §9 sparse 离散掩码读写 sm（func6 即家族判别码）----
 * vrs2 为逐 lane 字节偏移索引向量；写/原子类 vb 固定 0。
 */
enum : uint32_t {
    VEC_SM2V_SPARSE = 0x12u << 26, VEC_V2SM_SPARSE = 0x13u << 26,
    VEC_V2SM_SPARSE_ADD = 0x14u << 26, VEC_V2SM_SPARSE_AND = 0x15u << 26,
    VEC_V2SM_SPARSE_OR = 0x16u << 26, VEC_V2SM_SPARSE_XOR = 0x17u << 26,
};
constexpr uint32_t vec_sparse_enc(uint32_t func6_prefix, int vrs2, int vrs1, int s2,
                                  int vb = 0, int rd = GPR_A0)
{
    return func6_prefix | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(vrs2 & 0x1f) << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(s2 & 7) << 12) |
           (static_cast<uint32_t>(rd & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- §10 dyn 动态基址离散读写（func6 + op2 共同判别）----
 * s1 提供线程内动态基址；写/原子类 vb 固定 0。
 */
enum : uint32_t {
    VEC_DYN_P0 = 0x18u << 26,  /* sm2v(00) / v2sm(01) / add(10) / and(11) */
    VEC_DYN_P1 = 0x19u << 26,  /* or(00) / xor(01) */
};
constexpr uint32_t vec_dyn_enc(uint32_t func6_prefix, int op2, int vrs2, int vrs1,
                               int s2, int s1, int vb = 0)
{
    return func6_prefix | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(vrs2 & 0x1f) << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(s2 & 7) << 12) |
           (static_cast<uint32_t>(op2 & 3) << 10) |
           (static_cast<uint32_t>(s1 & 7) << 7) | OPCODE_TX32;
}

/* ---- §11/§12 shuffle 静态（rd 提供参数）/动态（s1 提供参数）变体 ----
 * subfunc：静态 0=xor 2=ror 4=broadcast 6=compress（LSB=0）；
 *          动态 1/3/5/7 为对应 .dyn（LSB=1）。
 */
enum : int {
    VEC_SH_XOR = 0, VEC_SH_XOR_DYN = 1, VEC_SH_ROR = 2, VEC_SH_ROR_DYN = 3,
    VEC_SH_BROADCAST = 4, VEC_SH_BROADCAST_DYN = 5,
    VEC_SH_COMPRESS = 6, VEC_SH_COMPRESS_DYN = 7,
};
constexpr uint32_t vec_shuffle_static_enc(int subfunc, int vrs1 = 0, int vb = 0,
                                          int rd = GPR_A0)
{
    return VEC_P_SHUFFLE | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(subfunc & 7) << 12) |
           (static_cast<uint32_t>(rd & 0x1f) << 7) | OPCODE_TX32;
}
constexpr uint32_t vec_shuffle_dyn_enc(int subfunc, int s1, int vrs1 = 0, int vb = 0)
{
    return VEC_P_SHUFFLE | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(subfunc & 7) << 12) |
           (static_cast<uint32_t>(s1 & 7) << 7) | OPCODE_TX32;
}

/* ---- §13 vv 双输入整数（func8 [14:7]） ---- */
enum : int {
    VEC_INT_ROL = 0, VEC_INT_SLL = 1, VEC_INT_SRA = 2, VEC_INT_SRL = 3,
    VEC_INT_ADD = 4, VEC_INT_SUB = 5, VEC_INT_MULU24 = 6, VEC_INT_MUL24 = 7,
    VEC_INT_MAXU = 8, VEC_INT_MINU = 9, VEC_INT_MAX = 10, VEC_INT_MIN = 11,
};
constexpr uint32_t vec_vv_int_enc(int func8, int vrs1 = 0, int vrs2 = 0, int vb = 0)
{
    return VEC_P_VV_INT | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(vrs2 & 0x1f) << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(func8 & 0xff) << 7) | OPCODE_TX32;
}

/* ---- §14 vv int32.bitwise（func4=0001，LUT[10:7] 为二输入布尔函数表） ---- */
constexpr uint32_t vec_vv_bitwise_enc(int lut, int vrs1 = 0, int vrs2 = 0, int vb = 0)
{
    return VEC_P_VV_INT | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(vrs2 & 0x1f) << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) | (1u << 11) |
           (static_cast<uint32_t>(lut & 0xf) << 7) | OPCODE_TX32;
}

/* ---- §15 vv 双输入 FP32（func5 [14:10] + rm [9:7]） ---- */
enum : int {
    VEC_FP32_ADD = 0, VEC_FP32_SUB = 1, VEC_FP32_MUL = 2,
    VEC_FP32_MAX = 4, VEC_FP32_MIN = 5,
};
constexpr uint32_t vec_vv_fp32_enc(int func5, int rm = RM_RNE,
                                   int vrs1 = 0, int vrs2 = 0, int vb = 0)
{
    return VEC_P_VV_FP32 | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(vrs2 & 0x1f) << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(func5 & 0x1f) << 10) |
           (static_cast<uint32_t>(rm & 7) << 7) | OPCODE_TX32;
}

/* ---- §16 vv mux2（逐 lane 三元选择，func5=00000，s2 提供 64bit 谓词） ---- */
constexpr uint32_t vec_mux2_enc(int s2, int vrs1 = 0, int vrs2 = 0, int vb = 0)
{
    return VEC_P_MUX_CMP | (static_cast<uint32_t>(vb & 1) << 25) |
           (static_cast<uint32_t>(vrs2 & 0x1f) << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(s2 & 7) << 12) | OPCODE_TX32;
}

/* ---- §17 vv int32.cmp / fp32.cmp（结果为 64bit 谓词写 sd，bit25 固定 0） ----
 * cond 的 bit2/bit1/bit0 分别控制 <、=、> 为真（[9:7]）。
 */
enum : int { VEC_CMP_INT32 = 1, VEC_CMP_FP32 = 2 };
constexpr uint32_t vec_cmp_enc(int klass, int sd, int cond,
                               int vrs1 = 0, int vrs2 = 0)
{
    return VEC_P_MUX_CMP |
           (static_cast<uint32_t>(vrs2 & 0x1f) << 20) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 15) |
           (static_cast<uint32_t>(sd & 7) << 12) |
           (static_cast<uint32_t>(klass & 3) << 10) |
           (static_cast<uint32_t>(cond & 7) << 7) | OPCODE_TX32;
}

/* ---- §18 v 单输入位移（unit=00001，shift_op [19:17]） ---- */
enum : int { VEC_SHIFT_SLL = 1, VEC_SHIFT_SRA = 2, VEC_SHIFT_SRL = 3 };
constexpr uint32_t vec_shift_enc(int shift_op, int sft, int vrs1 = 0, int vb = 0)
{
    return VEC_P_V | (static_cast<uint32_t>(vb & 1) << 25) | (1u << 20) |
           (static_cast<uint32_t>(shift_op & 7) << 17) |
           (static_cast<uint32_t>(sft & 0x1f) << 12) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- §19 v float32 SFU（unit=00010，class=00；precision[17]=1 走 ERU） ---- */
enum : int {
    VEC_SFU_LOG2 = 0, VEC_SFU_EXP2 = 1, VEC_SFU_SIN = 2, VEC_SFU_COS = 3,
    VEC_SFU_RSQRT = 4, VEC_SFU_RECIP = 5,
    VEC_ERU_EXP2 = 0, VEC_ERU_RECIP = 1,
};
constexpr uint32_t vec_sfu_enc(int sfu_op, bool fast_eru = false,
                               int vrs1 = 0, int vb = 0)
{
    return VEC_P_V | (static_cast<uint32_t>(vb & 1) << 25) | (2u << 20) |
           (static_cast<uint32_t>(fast_eru ? 1 : 0) << 17) |
           (static_cast<uint32_t>(sfu_op & 0x1f) << 12) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- §20 cvt 类型转换（unit=00010，class=01；类型组合编码在 cvt_op [17:15]） ---- */
enum : int {
    VEC_CVT_FP32_INT32 = 0, VEC_CVT_S32_FP32 = 1, VEC_CVT_FP32_FP16 = 2,
    VEC_CVT_FP32_BF16 = 3, VEC_CVT_FP16_FP32 = 4, VEC_CVT_BF16_FP32 = 5,
    VEC_CVT_MXFP8_BF16 = 6, VEC_CVT_MXFP4_BF16 = 7,  /* 7 预留不可执行 */
};
constexpr uint32_t vec_cvt_enc(int cvt_op, int rm = RM_RNE, int vrs1 = 0, int vb = 0)
{
    return VEC_P_V | (static_cast<uint32_t>(vb & 1) << 25) | (2u << 20) | (1u << 18) |
           (static_cast<uint32_t>(cvt_op & 7) << 15) |
           (static_cast<uint32_t>(rm & 7) << 12) |
           (static_cast<uint32_t>(vrs1 & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- §21 s 标量算术与迭代器读取（func8 [22:15] + func2 [11:10]） ----
 * 枚举值为 func8<<2（func2 由调用方按变体显式传入）：
 *   rol/sll/sra/srl → func8=0，func2=00/01/10/11；
 *   add/sub/mul     → func8=1，func2=00/01/10；
 *   getdim          → func8=1，func2=11，维度选择在 s1[1:0]（0..3 → Dim2..Dim5）；
 *   gets/getv       → func8=2/3，func2 复用为 2 位迭代器组号 sel（s0-s3 / v0-v3）。
 */
enum : int {
    VEC_S_ROL = 0, VEC_S_SLL = 1, VEC_S_SRA = 2, VEC_S_SRL = 3,        /* func8=0 */
    VEC_S_ADD = 4, VEC_S_SUB = 5, VEC_S_MUL = 6, VEC_S_GETDIM = 7,     /* func8=1 */
    VEC_S_GETS = 8,                                                    /* func8=2 */
    VEC_S_GETV = 12,                                                   /* func8=3 */
};
constexpr uint32_t vec_s_enc(int s_op, int func2, int s1 = 0, int s2 = 0, int sd = 0)
{
    const int func8 = s_op >> 2;
    return VEC_P_S | (static_cast<uint32_t>(sd & 7) << 23) |
           (static_cast<uint32_t>(func8 & 0xff) << 15) |
           (static_cast<uint32_t>(s2 & 7) << 12) |
           (static_cast<uint32_t>(func2 & 3) << 10) |
           (static_cast<uint32_t>(s1 & 7) << 7) | OPCODE_TX32;
}

/* ---- §22 s bitwise（64bit 谓词逐位布尔，LUT 分裂在 [16:15]/[11:10]） ---- */
constexpr uint32_t vec_s_bitwise_enc(int lut, int s1 = 0, int s2 = 0, int sd = 0)
{
    return VEC_P_S | (static_cast<uint32_t>(sd & 7) << 23) | (1u << 17) |
           (static_cast<uint32_t>((lut >> 2) & 3) << 15) |
           (static_cast<uint32_t>(s2 & 7) << 12) |
           (static_cast<uint32_t>(lut & 3) << 10) |
           (static_cast<uint32_t>(s1 & 7) << 7) | OPCODE_TX32;
}

/* ---- §23 s 分支（仅模式二；target 为 8bit 有符号线程内相对偏移） ---- */
enum : int { VEC_BR_BEQ = 0, VEC_BR_BNE = 1, VEC_BR_BLT = 2, VEC_BR_BGE = 3 };
constexpr uint32_t vec_branch_enc(int func2, int target, int s1 = 0, int s2 = 0)
{
    return VEC_P_BRANCH |
           (static_cast<uint32_t>(target & 0xff) << 15) |
           (static_cast<uint32_t>(s2 & 7) << 12) |
           (static_cast<uint32_t>(func2 & 3) << 10) |
           (static_cast<uint32_t>(s1 & 7) << 7) | OPCODE_TX32;
}

} /* namespace txenc */

#endif /* TXENC_VEC_ENC_H */
