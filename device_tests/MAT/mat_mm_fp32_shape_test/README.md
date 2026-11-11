# mat_mm_fp32_shape_test

从 `dayan-samples/Instructions/MAT/single/mat_mm_fp32_shape.cpp.cpp` 迁移的裸机版本（共享
mat_*_single_impl/host 逻辑内联展开），发射走 `instr_enc`（txenc）+ `tx_test.h`。

## 覆盖意图

FP32 存储 / TF32 pair（dtype=0x8410）单 MM：M=4/N=12/K=17，K 尾部 + 有效 A/B 均为 1.0，stride 96/96/64，输出保护区检查。

## 当前模型限制

保留 TF32 pair 正向语义，不降级为 S8/BF16，也不把 tensor interrupt 视作成功。
当前 `dayan-emu/emu/mat.cpp` 的 operand decoder 未实现 TF32，运行会触发
张量中断并保持输出哨兵。本项属于模型能力阻塞，不能计入已通过的正向回归。
实现 TF32 支持后，应保持本测试并要求数值及保护区全部通过。
输入均为 TF32 精确可表示的小值，乘积和也可在 FP32 中精确表示；仅由 meta 定义 ref/dut 按位比较。

输出每行含保护区，预填 0x5a 并纳入 ref/dut 逐字节对拍；完整物理输出 256B 纳入 meta.json 对拍。

## 差异

设备侧保留填数和 padding 哨兵播种，不做尾部自判断；完整输出由 ref/dut 对拍；URAM 8K slab
（a/in@0、b/ker@8K、c/out@16K）；MAT 指令由 DAYAN_TEST 主线程直接发射。

## 简化数据

有效 A/B 均为 FP32 容器中的 1.0（0x3f800000），有效输出应为 17.0；仍需模型支持 TF32。
原有输入 padding、输出哨兵、形状和指令时序不变，不缩小 meta.json 对拍范围。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_mm_fp32_shape_test
```
