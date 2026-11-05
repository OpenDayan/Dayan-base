#ifndef TXENC_BASE_ENC_H
#define TXENC_BASE_ENC_H

/* 指令编码库 · Part 1/5：共享基础
 * 覆盖：32/64 位指令格式常量、tx.base.tcsr.<unit>.read/write、Produce/Bar、
 * 共用数据类型编码表、GPR 约定与指令发射（emit）模板。
 * 来源：instruction-encoding/tcsr-instruction-encoding.md、data-type-encoding.md。
 * 仅编码，无运行时状态；与 tensor_suite/tensor_test.h 可共存（本库在 txenc 名字空间内）。
 */

#include <stdint.h>

namespace txenc {

/* ---- 指令格式（spec §1.3） ---- */
constexpr uint32_t OPCODE_TX32 = 0x0b;   /* 0001011：32 位张量指令 */
constexpr uint64_t OPCODE_TX64 = 0x3f;   /* 0111111：64 位 MACRO 复合指令，[63:48] 填 0 */

/* ---- 常用 GPR 编号（发射封装的默认操作数寄存器） ---- */
constexpr int GPR_A0 = 10;
constexpr int GPR_A1 = 11;
constexpr int GPR_A2 = 12;

/* ---- 单元编号（unit id [26:23]） ---- */
enum : int {
    UNIT_GLOBAL = 0,
    UNIT_LSU    = 1,
    UNIT_MAT    = 2,
    UNIT_VEC    = 3,
    UNIT_MACRO  = 4,
};

/* ---- tx.base.tcsr.<unit>.[read|write]（§2.2.1） ----
 * funct5[31:27]=0，unit[26:23]，addr[22:13]，re[12]，gpr[11:7]。
 * read：re=1，gpr 为 dest；write：re=0，gpr 为 src。
 * LSU/MAT/MACRO 的 group 为逻辑组号，经 Global Context(0x010) 映射到物理组。
 */
constexpr uint32_t tcsr_enc(int unit, int address, bool read, int gpr = GPR_A0)
{
    return (static_cast<uint32_t>(unit & 0xf) << 23) |
           (static_cast<uint32_t>(address & 0x3ff) << 13) |
           (static_cast<uint32_t>(read ? 1 : 0) << 12) |
           (static_cast<uint32_t>(gpr & 0x1f) << 7) | OPCODE_TX32;
}

/* ---- Produce/Bar（Global 单元特殊 TCSR，§2.1.1） ---- */
/* produce：向单元 X 发布同步令牌（write，addr=X）。 */
constexpr uint32_t sb_produce_enc(int unit) { return tcsr_enc(UNIT_GLOBAL, unit, false); }
/* bar：等待单元 X 已发射操作全部完成（read，addr=X，返回恒 0）。 */
constexpr uint32_t sb_bar_enc(int unit) { return tcsr_enc(UNIT_GLOBAL, unit, true); }
/* 全局队列消费：读 = 弹出队首令牌（addr=0）。 */
constexpr uint32_t sb_get_any_enc() { return tcsr_enc(UNIT_GLOBAL, 0, true); }

/* ---- 共用数据类型编码（8 位） ----
 * 浮点族与 data-type-encoding.md 一致（FP32/BF16/FP16/E4M3 已由 Dayan clang
 * builtin 的 convert/elemwise 输出逐位核对）。
 * 整数族两套约定冲突：文档表为 S32=0x10…，而编译器 builtin 实发 S32=0x20
 * （elemwise s32 的 DataType 字节与 convert s32→f32 的 SrcType 均为 0x20，
 * 即 sim/npu_kernels npu_instruction.hpp 一族）。本库从编译器：整数族取
 * 0x20 起；S16/S8 逐项未独立验证（按同族类推），S4/U4 不定义。
 */
enum : uint8_t {
    DT_FP32     = 0x00,
    DT_BF16     = 0x02,
    DT_FP16     = 0x03,
    DT_FP8_E4M3 = 0x04,
    DT_FP8_E5M2 = 0x05,
    DT_MXFP8_E4M3 = 0x06,
    DT_MXFP8_E5M2 = 0x07,
    DT_FP4_E2M1 = 0x08,
    DT_MXFP4_E2M1 = 0x09,
    DT_S32 = 0x20,
    DT_U32 = 0x21,
    DT_S16 = 0x22,
    DT_U16 = 0x23,
    DT_S8  = 0x24,
    DT_U8  = 0x25,
};

/* 元素位宽（bit）；未知编码返回 0。 */
constexpr unsigned dtype_bits(uint8_t dtype)
{
    switch (dtype) {
        case DT_FP32: case DT_S32: case DT_U32: return 32;
        case DT_BF16: case DT_FP16: case DT_S16: case DT_U16: return 16;
        case DT_FP8_E4M3: case DT_FP8_E5M2:
        case DT_MXFP8_E4M3: case DT_MXFP8_E5M2:
        case DT_S8: case DT_U8: return 8;
        case DT_FP4_E2M1: case DT_MXFP4_E2M1: return 4;
        default: return 0;
    }
}

/* ---- 浮点舍入模式 rm[2:0]（VEC 浮点运算与 cvt 通用） ---- */
enum : int {
    RM_RNE = 0,   /* 就近舍入，平局取偶 */
    RM_RTZ = 1,   /* 向 0 */
    RM_RDN = 2,   /* 向 -inf */
    RM_RUP = 3,   /* 向 +inf */
    RM_RMM = 4,   /* 就近舍入，平局取绝对值大者 */
};

/* ---- 指令发射模板（编码须为编译期常量） ---- */

template <uint32_t Encoding>
static inline void emit32()
{
    asm volatile(".4byte %0" : : "i"(Encoding) : "memory");
}

template <uint32_t Encoding>
static inline void emit32_a0(uint64_t value0)
{
    asm volatile("mv a0, %0\n.insn 4, %1" : : "r"(value0), "i"(Encoding) : "a0", "memory");
}

template <uint32_t Encoding>
static inline void emit32_a0_a1(uint64_t value0, uint64_t value1)
{
    asm volatile("mv a0, %0\nmv a1, %1\n.insn 4, %2"
                 : : "r"(value0), "r"(value1), "i"(Encoding) : "a0", "a1", "memory");
}

template <uint32_t Encoding>
static inline void emit32_a0_a1_a2(uint64_t value0, uint64_t value1, uint64_t value2)
{
    asm volatile("mv a0, %0\nmv a1, %1\nmv a2, %2\n.insn 4, %3"
                 : : "r"(value0), "r"(value1), "r"(value2), "i"(Encoding)
                 : "a0", "a1", "a2", "memory");
}

template <uint64_t Encoding>
static inline void emit64_a0_a1(uint64_t value0, uint64_t value1)
{
    asm volatile("mv a0, %0\nmv a1, %1\n.insn 8, %2"
                 : : "r"(value0), "r"(value1), "i"(Encoding) : "a0", "a1", "memory");
}

template <uint64_t Encoding>
static inline void emit64_a0_a1_a2(uint64_t value0, uint64_t value1, uint64_t value2)
{
    asm volatile("mv a0, %0\nmv a1, %1\nmv a2, %2\n.insn 8, %3"
                 : : "r"(value0), "r"(value1), "r"(value2), "i"(Encoding)
                 : "a0", "a1", "a2", "memory");
}

/* ---- TCSR 读写发射（等价 tensor_test.h 的 write_tcsr/read_tcsr） ---- */

template <int Unit, int Address>
static inline void write_tcsr(uint64_t value)
{
    asm volatile("mv a0, %0\n.insn 4, %1"
                 : : "r"(value), "i"(tcsr_enc(Unit, Address, false))
                 : "a0", "memory");
}

template <int Unit, int Address>
static inline uint64_t read_tcsr()
{
    uint64_t value;
    asm volatile(".insn 4, %1\nmv %0, a0"
                 : "=r"(value)
                 : "i"(tcsr_enc(Unit, Address, true))
                 : "a0", "memory");
    return value;
}

/* produce/bar/全局消费的发射封装。 */
template <int Unit> static inline void sb_produce(uint64_t token)
{
    asm volatile("mv a0, %0\n.insn 4, %1"
                 : : "r"(token), "i"(sb_produce_enc(Unit)) : "a0", "memory");
}
template <int Unit> static inline void sb_bar()
{
    asm volatile(".insn 4, %0" : : "i"(sb_bar_enc(Unit)) : "a0", "memory");
}
static inline uint64_t sb_get_any()
{
    uint64_t token;
    asm volatile(".insn 4, %1\nmv %0, a0"
                 : "=r"(token) : "i"(sb_get_any_enc()) : "a0", "memory");
    return token;
}

} /* namespace txenc */

#endif /* TXENC_BASE_ENC_H */
