# mat_mm_s8_tail_stride_test

从 `dayan-samples/Instructions/MAT/single/mat_mm_s8_tail_stride.cpp.cpp` 迁移的裸机版本（共享
mat_*_single_impl/host 逻辑内联展开），发射走 `instr_enc`（txenc）+ `tx_test.h`。

## 覆盖意图

S8 K=33 尾部 + 物理 stride 64/64/128 单 MM：M=3/N=24，输出保护区检查。

输出每行含保护区，预填 0x5a 并纳入 ref/dut 逐字节对拍；完整物理输出 384B 纳入 meta.json 对拍。

S8 原始字节表与 golden 数值表由编译期断言核对；有效输出按 FP32 位模式精确比较。

## 差异

设备侧保留填数和 padding 哨兵播种，不做尾部自判断；完整输出由 ref/dut 对拍；URAM 8K slab
（a/in@0、b/ker@8K、c/out@16K）；MAT 指令由 DAYAN_TEST 主线程直接发射。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_mm_s8_tail_stride_test
```
