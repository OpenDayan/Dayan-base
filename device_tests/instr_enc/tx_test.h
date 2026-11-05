#ifndef TXENC_TEST_H
#define TXENC_TEST_H

/* 指令编码库 · 测试骨架（公共库）
 * 基于 txenc 编码器（tx_base/tx_lsu/tx_vec_enc.h）提供裸机张量测试的常用
 * 配置、同步与搬运封装，替代 tensor_suite/tensor_test.h 的角色：
 *   - TCSR 读写 / Produce-Bar 一律走 txenc::write_tcsr / sb_bar；
 *   - 指令发射一律走 txenc::*_enc + emit 模板；
 *   - 测试注册 / 线程退出 / cache 维护沿用 <dayan_test.h> 协议层。
 * 测试通过相对路径引入：#include "../../instr_enc/tx_test.h"
 * 迁移意图参考 /home/juyi/instruction_test_all/dayan-samples 的样例。
 */

#include "tx_base_enc.h"
#include "tx_lsu_enc.h"
#include "tx_mat_enc.h"
#include "tx_vec_enc.h"

#include <dayan_test.h>

namespace txtst {

using u32 = unsigned int;
using u64 = unsigned long;

/* ---- 固定地址布局（与 tensor_suite 约定一致） ---- */
[[maybe_unused]] static constexpr u64 OUT_DDR = 0x92000000UL;
[[maybe_unused]] static constexpr u64 URAM_BASE = 0x00800000UL;
[[maybe_unused]] static constexpr u64 DDR_BASE = 0x90000000UL;

[[maybe_unused]] static inline void fence_io() {
    asm volatile("fence iorw, iorw" ::: "memory");
}

/* ---- DDR cache 维护（URAM 不经过 CPU cache，无需维护） ---- */
[[maybe_unused]] static inline void flush_ddr(u64 address, u64 bytes) {
    fence_io();
    if (address >= DAYAN_TEST_DATA_BASE)
        dayan_cache_flush(reinterpret_cast<const void *>(address), bytes);
}
[[maybe_unused]] static inline void invalidate_ddr(u64 address, u64 bytes) {
    fence_io();
    if (address >= DAYAN_TEST_DATA_BASE)
        dayan_cache_invalidate(reinterpret_cast<const void *>(address), bytes);
}

/* ---- LSU：一维连续搬数整组配置（dayan-samples harness 同款 21 项）----
 * 组号 Group 为模板参数（TCSR 地址须为编译期常量）；bytes 字节连续块：
 * Block=0、Dim0=bytes/64-1、In/Out Dim0 步长 64、Dim1 步长 bytes、
 * 不转置（BlockTranspose=6，数据地址可任意字节对齐）、整 64B 读写事务、
 * 无填充 / 重排 / 地址表。bytes 须为 64 的倍数。
 * 发射 DMA 前配置；快照语义保证在途指令不受后续改写影响。
 */
template <int Group = 0>
[[maybe_unused]] static inline void configure_lsu_linear(u64 bytes) {
    constexpr int base = Group * 32;
    txenc::write_tcsr<txenc::UNIT_LSU, base + 0>(0);              /* Block */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 1>(bytes / 64U - 1U); /* Dim0 */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 2>(0);              /* Dim1 */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 3>(0);              /* Dim2 */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 4>(0);              /* InBlockStride */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 5>(64);             /* InDim0Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 6>(bytes);          /* InDim1Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 7>(0);              /* InDim2Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 8>(0);              /* OutBlockStride */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 9>(64);             /* OutDim0Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 10>(bytes);         /* OutDim1Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 11>(0);             /* OutDim2Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 12>(6);             /* BlockTranspose=6 不转置 */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 13>(0);             /* BlockPadding */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 14>(63);            /* RequestReadSize */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 15>(0);             /* PaddingValue */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 16>(0);             /* UramWriteSwizzleMBS */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 17>(0);             /* UramReadSwizzleMBS */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 18>(63);            /* RequestWriteSize */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 19>(0);             /* TableSize */
    txenc::write_tcsr<txenc::UNIT_LSU, base + 20>(64);            /* TableStride */
}

/* ---- 常用 DMA（组 0；其他组直接用 txenc::lsu_dma_enc 发射）----
 * 实参顺序统一 (src, dst)：rs1=a0 源基地址，rd=a1 目的基地址。
 */
[[maybe_unused]] static inline void dma_dram_to_uram(u64 dram_src, u64 uram_dst) {
    txenc::emit32_a0_a1<txenc::lsu_dma_enc(0, false, false, txenc::MEM_DRAM, txenc::MEM_URAM)>(
        dram_src, uram_dst);
}
[[maybe_unused]] static inline void dma_uram_to_dram(u64 uram_src, u64 dram_dst) {
    txenc::emit32_a0_a1<txenc::lsu_dma_enc(0, false, false, txenc::MEM_URAM, txenc::MEM_DRAM)>(
        uram_src, dram_dst);
}

/* ---- LSU：任意字节数的逐 64B 块拷贝（dayan-samples MAT harness 同款）----
 * 适用字节数非 64 倍数的操作数（如 S8 矩阵）。先 configure_lsu_chunked()
 * 配置一次（Dim1=0 单行、64B 端口步长、不转置、整 64B 写事务）；每块只
 * 重写 Dim0(=0) 与 RequestReadSize（尾块 = bytes-1），RequestWriteSize 恒 63。
 * 尾块会整 64B 写入目的侧，源区后需留 padding（4K slab 布局天然满足）。
 */
[[maybe_unused]] static inline void configure_lsu_chunked() {
    txenc::write_tcsr<txenc::UNIT_LSU, 0>(0);   /* Block */
    txenc::write_tcsr<txenc::UNIT_LSU, 2>(0);   /* Dim1 */
    txenc::write_tcsr<txenc::UNIT_LSU, 3>(0);   /* Dim2 */
    txenc::write_tcsr<txenc::UNIT_LSU, 4>(0);   /* InBlockStride */
    txenc::write_tcsr<txenc::UNIT_LSU, 5>(64);  /* InDim0Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, 6>(64);  /* InDim1Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, 7>(0);   /* InDim2Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, 8>(0);   /* OutBlockStride */
    txenc::write_tcsr<txenc::UNIT_LSU, 9>(64);  /* OutDim0Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, 10>(64); /* OutDim1Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, 11>(0);  /* OutDim2Stride */
    txenc::write_tcsr<txenc::UNIT_LSU, 12>(6);  /* BlockTranspose */
    txenc::write_tcsr<txenc::UNIT_LSU, 13>(0);  /* BlockPadding */
    txenc::write_tcsr<txenc::UNIT_LSU, 15>(0);  /* PaddingValue */
    txenc::write_tcsr<txenc::UNIT_LSU, 16>(0);  /* UramWriteSwizzleMBS */
    txenc::write_tcsr<txenc::UNIT_LSU, 17>(0);  /* UramReadSwizzleMBS */
    txenc::write_tcsr<txenc::UNIT_LSU, 18>(63); /* RequestWriteSize */
    txenc::write_tcsr<txenc::UNIT_LSU, 19>(0);  /* TableSize */
    txenc::write_tcsr<txenc::UNIT_LSU, 20>(64); /* TableStride */
}

[[maybe_unused]] static inline void dma_d2u_bytes(u64 bytes, u64 uram, u64 dram) {
    while (bytes >= 64U) {
        txenc::write_tcsr<txenc::UNIT_LSU, 1>(0);
        txenc::write_tcsr<txenc::UNIT_LSU, 14>(63);
        txenc::emit32_a0_a1<txenc::lsu_dma_enc(0, false, false, txenc::MEM_DRAM, txenc::MEM_URAM)>(
            dram, uram);
        uram += 64U;
        dram += 64U;
        bytes -= 64U;
    }
    if (bytes) {
        txenc::write_tcsr<txenc::UNIT_LSU, 1>(0);
        txenc::write_tcsr<txenc::UNIT_LSU, 14>(bytes - 1U);
        txenc::emit32_a0_a1<txenc::lsu_dma_enc(0, false, false, txenc::MEM_DRAM, txenc::MEM_URAM)>(
            dram, uram);
    }
}

[[maybe_unused]] static inline void dma_u2d_bytes(u64 bytes, u64 dram, u64 uram) {
    while (bytes >= 64U) {
        txenc::write_tcsr<txenc::UNIT_LSU, 1>(0);
        txenc::write_tcsr<txenc::UNIT_LSU, 14>(63);
        txenc::emit32_a0_a1<txenc::lsu_dma_enc(0, false, false, txenc::MEM_URAM, txenc::MEM_DRAM)>(
            uram, dram);
        uram += 64U;
        dram += 64U;
        bytes -= 64U;
    }
    if (bytes) {
        txenc::write_tcsr<txenc::UNIT_LSU, 1>(0);
        txenc::write_tcsr<txenc::UNIT_LSU, 14>(bytes - 1U);
        txenc::emit32_a0_a1<txenc::lsu_dma_enc(0, false, false, txenc::MEM_URAM, txenc::MEM_DRAM)>(
            uram, dram);
    }
}

/* ---- 单元排空等待（= dayan-samples 的 dayan_wait_lsu / dayan_wait_vec，
 * Global Produce/Bar 单元侧读语义） ---- */
[[maybe_unused]] static inline void wait_lsu() { txenc::sb_bar<txenc::UNIT_LSU>(); }
[[maybe_unused]] static inline void wait_vec() { txenc::sb_bar<txenc::UNIT_VEC>(); }
[[maybe_unused]] static inline void wait_mat() { txenc::sb_bar<txenc::UNIT_MAT>(); }
[[maybe_unused]] static inline void wait_macro() { txenc::sb_bar<txenc::UNIT_MACRO>(); }

/* ---- VEC：全零描述符（vec_sync + Dim0-5 + ImplDefHint + 全部迭代器清零，
 * UramChannelStride 写复位值 32）。spec 只保证 ChannelStride 复位值，
 * 其余 TCSR 复位值未定义，s1-s3 / v1-v3 用前必须显式配置——本函数全量清零。
 * 注意偏移 0（Control）不写：写 0 即启动。 ---- */
[[maybe_unused]] static inline void configure_vec_zero() {
    txenc::write_tcsr<txenc::UNIT_VEC, 1>(0);   /* vec_sync */
    txenc::write_tcsr<txenc::UNIT_VEC, 2>(0);   /* Dim0 */
    txenc::write_tcsr<txenc::UNIT_VEC, 3>(0);   /* Dim1 */
    txenc::write_tcsr<txenc::UNIT_VEC, 4>(0);   /* Dim2 */
    txenc::write_tcsr<txenc::UNIT_VEC, 5>(0);   /* Dim3 */
    txenc::write_tcsr<txenc::UNIT_VEC, 6>(0);   /* Dim4 */
    txenc::write_tcsr<txenc::UNIT_VEC, 7>(0);   /* Dim5 */
    txenc::write_tcsr<txenc::UNIT_VEC, 8>(0);   /* ImplDefHint */
    txenc::write_tcsr<txenc::UNIT_VEC, 9>(0);   /* s0Base */
    txenc::write_tcsr<txenc::UNIT_VEC, 10>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 11>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 12>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 13>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 14>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 15>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 16>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 17>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 18>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 19>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 20>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 21>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 22>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 23>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 24>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 25>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 26>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 27>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 28>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 29>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 30>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 31>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 32>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 33>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 34>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 35>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 36>(0);  /* s0-s3 Base + 六维步长 */
    txenc::write_tcsr<txenc::UNIT_VEC, 37>(0);  /* v0Base */
    txenc::write_tcsr<txenc::UNIT_VEC, 38>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 39>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 40>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 41>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 42>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 43>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 44>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 45>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 46>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 47>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 48>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 49>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 50>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 51>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 52>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 53>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 54>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 55>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 56>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 57>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 58>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 59>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 60>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 61>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 62>(0);  txenc::write_tcsr<txenc::UNIT_VEC, 63>(0);
    txenc::write_tcsr<txenc::UNIT_VEC, 64>(0);  /* v0-v3 Base + 六维步长 */
    txenc::write_tcsr<txenc::UNIT_VEC, 65>(32); /* v0-v3 UramChannelStride 复位值 */
    txenc::write_tcsr<txenc::UNIT_VEC, 66>(32);
    txenc::write_tcsr<txenc::UNIT_VEC, 67>(32);
    txenc::write_tcsr<txenc::UNIT_VEC, 68>(32);
}

/* Control（偏移 0）读 = inst queue 空闲条数；发射 n 条前轮询到 >= n。
 * 与 spec 5.4 软件流程一致；连续两次读结果可不同属正常。 */
[[maybe_unused]] static inline void wait_vec_queue_capacity(int count) {
    while (txenc::read_tcsr<txenc::UNIT_VEC, 0>() < static_cast<u64>(count)) {
    }
}

/* ---- 写 CONTROL 启动 Vec 指令块（41 位载荷的五段下标 + 模式位 0）----
 * 段结束下标 = “从该下标开始进入下一段”。等值五参为直线码；
 * 单参版本沿用本仓库已验证约定 (0,0,N,N,N)：空 prologue、loop 覆盖全块、
 * Dim1=0 时等价于样例的 0xNNNNNNNNNN 全等值形式。
 */
[[maybe_unused]] static inline void start_vec_block(int prologue0_end, int prologue1_end,
                                                    int loop_end, int epilogue1_end,
                                                    int epilogue0_end) {
    const u64 control = (static_cast<u64>(prologue0_end & 0xff) << 0) |
                        (static_cast<u64>(prologue1_end & 0xff) << 8) |
                        (static_cast<u64>(loop_end & 0xff) << 16) |
                        (static_cast<u64>(epilogue1_end & 0xff) << 24) |
                        (static_cast<u64>(epilogue0_end & 0xff) << 32);
    txenc::write_tcsr<txenc::UNIT_VEC, 0>(control);
}
[[maybe_unused]] static inline void start_vec_block(int count) {
    start_vec_block(0, 0, count, count, count);
}

} /* namespace txtst */

#endif /* TXENC_TEST_H */
