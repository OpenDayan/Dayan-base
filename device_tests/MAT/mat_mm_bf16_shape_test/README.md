# mat_mm_bf16_shape_test

从 `dayan-samples/Instructions/MAT/single/mat_mm_bf16_shape.cpp.cpp` 迁移的裸机版本（共享
mat_*_single_impl/host 逻辑内联展开），发射走 `instr_enc`（txenc）+ `tx_test.h`。

## 覆盖意图

BF16 单 MM（dtype=0x02 legacy）：M=3/N=16/K=16，stride 32/32/96，有效 A/B 均为 BF16 1.0。

输出每行含保护区，预填 0x5a 并纳入 ref/dut 逐字节对拍；完整物理输出 288B 纳入 meta.json 对拍。

## 差异

设备侧保留填数和 padding 哨兵播种，不做尾部自判断；完整输出由 ref/dut 对拍；URAM 8K slab
（a/in@0、b/ker@8K、c/out@16K）；MAT 指令由 DAYAN_TEST 主线程直接发射。

## 简化数据

有效 A/B 均为 BF16 1.0（0x3f80），有效输出应为 16.0。
原有输入 padding、输出哨兵、形状和指令时序不变，不缩小 meta.json 对拍范围。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_mm_bf16_shape_test
```
