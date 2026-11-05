# mat_conv2d_pointwise_test

从 `dayan-samples/Instructions/MAT/single/mat_conv2d_pointwise.cpp.cpp` 迁移的裸机版本（共享
mat_*_single_impl/host 逻辑内联展开），发射走 `instr_enc`（txenc）+ `tx_test.h`。

## 覆盖意图

1×1 逐点卷积：C=16/Co=16/Hi=5/Wi=7/Hk=Wk=1/pad=0/dil=1/stride=1（Ho=5/Wo=7），WoStride=Co*4+32=96。

每个输出空间位置增加 32B 的 0x5a 保护区，通过 ref/dut 对拍观察是否保持不变；完整物理输出 3360B，纳入 exact 对拍。

## 差异

设备侧保留填数和 padding 哨兵播种，不做尾部自判断；完整输出由 ref/dut 对拍；URAM 8K slab
（a/in@0、b/ker@8K、c/out@16K）；MAT 指令由 DAYAN_TEST 主线程直接发射。

## 简化数据

有效输入和核均为 S8 1，有效输出应为 16.0。
原有输入 padding、输出哨兵、形状和指令时序不变，不缩小 meta.json 对拍范围。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_conv2d_pointwise_test
```
