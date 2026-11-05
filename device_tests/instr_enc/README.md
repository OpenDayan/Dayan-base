# 指令编码库（instr_enc）

张量扩展指令的纯 `constexpr` 编码函数 + 内联发射模板，供 `device_tests` 下的
裸机测试手工构造指令流。无 `meta.json`，不是测试目录，构建系统不会当作测试
目标；测试通过相对路径引入：

```cpp
#include "../../instr_enc/instr_enc.h"

txenc::emit32_a0_a1<txenc::lsu_dma_enc()>(dram_addr, uram_addr);  // d2u
txenc::emit64_a0_a1<txenc::macro_convert_enc(txenc::DT_FP32, txenc::DT_BF16)>(
    uram_dst, uram_src);
txenc::emit32<txenc::vec_vv_int_enc(txenc::VEC_INT_ADD, /*vrs1=*/1)>();
```

全部实体位于 `namespace txenc`，与 `tensor_suite/tensor_test.h` 的同名全局
编码器（`dma_enc`/`mat_enc`/`vec_read_enc`…）可共存；但不要在对两者都可见的
作用域里 `using namespace txenc;` 后再无限定调用发射模板。

## 五个部分

| 部分 | 头文件 | 覆盖 |
| --- | --- | --- |
| 1 共享基础 | `tx_base_enc.h` | 32/64 位格式常量、`tx.base.tcsr.<unit>.read/write`、Produce/Bar、共用数据类型编码表、`emit32*`/`emit64*` 发射模板、`write_tcsr`/`read_tcsr` |
| 2 LSU | `tx_lsu_enc.h` | `tx.lsu.dma` 4 方向 × 4 地址模式共 16 变体 |
| 3 VEC | `tx_vec_enc.h` | VEC LSU/shuffle/vv/v/cvt/s/分支全族（spec §5.7，89 条） |
| 4 MAT | `tx_mat_enc.h` | `mm`/`mma`/`conv2d`/`conv2da`/`setptr` 共 5 条 |
| 5 MACRO | `tx_macro_enc.h` | 64 位复合指令：elemwise/convert/sfu/reduce/shuffle.benes/shuffle.compress |

`instr_enc.h` 为伞形头。各单元 TCSR 寄存器逐项地址（LSU_BLOCK=0 等）不在本库
范围内，继续使用 `device/include/api/dayan_intrinsics.h` 的枚举。

另附 `tx_test.h`（2026-10-20 增）：基于本编码库的**公共测试骨架**（`txtst::`
名字空间）——DDR/URAM 固定地址、cache 维护、LSU 一维线性整组配置
`configure_lsu_linear<Group>(bytes)`、VEC 全零描述符 `configure_vec_zero()`、
队列容量轮询 `wait_vec_queue_capacity(n)`、块启动 `start_vec_block(...)`、
单元排空等待 `wait_lsu/wait_vec/...`（Produce/Bar）、`dma_dram_to_uram` 等。
测试注册 / 线程退出沿用 `<dayan_test.h>` 协议层。迁移 dayan-samples 用例时
`#include "../../instr_enc/tx_test.h"` 即可，无需 tensor_suite。

## 来源与锚点

编码提取自 `/home/juyi/CLUSTER/npu-cluster/instruction-encoding/` 各文档
（spec-rvt-cn-2026.08.31 的编码向提取）。`selfcheck.cpp` 为编译期自检
（全部 `static_assert`，无运行时产物），用设备同款 flags 编译验证：

```sh
clang --target=riscv64-unknown-elf -march=rv64imaf_zicsr_zifencei_zacas -mabi=lp64 \
  -Xclang -target-feature -Xclang +txlsu -Xclang -target-feature -Xclang +txmat \
  -Xclang -target-feature -Xclang +txvec -Xclang -target-feature -Xclang +txbase \
  -Xclang -target-feature -Xclang +txmacro -Wno-ignored-attributes \
  -mcmodel=medany -O2 -std=c++17 -fno-exceptions -fno-rtti -ffreestanding \
  -I<repo>/device_tests -I<repo>/device/include/api -c selfcheck.cpp -o /tmp/sc.o
```

以下常量已与 Dayan clang builtin（`/opt/llvm` 的 `__builtin_riscv_tx_macro_*`）
实测输出逐位核对，可作为回归锚点：

| 指令 | 编码 |
| --- | --- |
| `macro.convert.fp32→bf16`（rd=a0, rs1=a1） | `0x000002000205853f` |
| `macro.convert.fp32→fp16` | `0x000003000205853f` |
| `macro.convert.fp32→e4m3` | `0x000004000205853f` |
| `macro.convert.s32→f32`（SrcType=0x20） | `0x000000200205853f` |
| `macro.elemwise.vv.add.fp32`（rd=a0, rs1=a1, rs2=a2） | `0x0000000000c5853f` |
| `macro.elemwise.vv.add` dtype=S32 | `0x0000002000c5853f` |
| `macro.sfu.log2 / exp2 / sin`（op 在 [47:40]） | `0x…000405853f / 0x000001000405853f / 0x000002000405853f` |
| `macro.reduce.max / max_index`（Id[46]） | `0x0000040006c5853f / 0x0000440006c5853f` |
| `vec.vv.int32.add`（vrs1=1, vrs2=0, vb=0） | `0x8400820b`（vec_check_int_add_test 在用） |

## 文档与编译器实测的两处差异（以编译器为准）

1. **MACRO SFU 的 Operation 字段在 [47:40]**，文档 §6.2.3 写 [39:32]。
   实测 log2/exp2/sin = 0/1/2 分别落在高字 `0x0000/0x0100/0x0200`。
2. **整数 dtype 编码为 0x20 族**，文档表写 0x10 族。实测 elemwise s32 的
   DataType 字节与 convert s32→f32 的 SrcType 均为 `0x20`（即
   npu_instruction.hpp 一族）。浮点族两表一致（FP32=0x00、BF16=0x02、
   FP16=0x03、E4M3=0x04 已实测确认）。S16/S8 逐项未独立验证（按族类推），
   S4/U4 未定义。

另：reduce 的 Operation[45:40] + Id[46]、elemwise 的 Operation[45:40] +
Bd[47:46]、convert 的 SrcType[39:32]/DstType[47:40] 均与文档一致（已实测）。
