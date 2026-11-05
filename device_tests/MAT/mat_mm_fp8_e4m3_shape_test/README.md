# mat_mm_fp8_e4m3_shape_test

从 `dayan-samples/Instructions/MAT/single/mat_mm_fp8_e4m3_shape.cpp.cpp` 迁移的裸机版本（共享
mat_*_single_impl/host 逻辑内联展开），发射走 `instr_enc`（txenc）+ `tx_test.h`。

## 覆盖意图

FP8 E4M3 单 MM（dtype=0x04）：M=3/N=16/K=32，stride 32/32/96，含最大有限值 ±448（E4M3 无 inf 语义）。

输出每行含保护区，预填 0x5a 并纳入 ref/dut 逐字节对拍；完整物理输出 288B 纳入 meta.json 对拍。

## 差异

设备侧保留填数和 padding 哨兵播种，不做尾部自判断；完整输出由 ref/dut 对拍；URAM 8K slab
（a/in@0、b/ker@8K、c/out@16K）；MAT 指令由 DAYAN_TEST 主线程直接发射。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_mm_fp8_e4m3_shape_test
```
