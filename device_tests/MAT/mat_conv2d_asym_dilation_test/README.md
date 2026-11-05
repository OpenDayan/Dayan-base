# mat_conv2d_asym_dilation_test

从 `dayan-samples/Instructions/MAT/single/mat_conv2d_asym_dilation.cpp.cpp` 迁移的裸机版本（共享
mat_*_single_impl/host 逻辑内联展开），发射走 `instr_enc`（txenc）+ `tx_test.h`。

## 覆盖意图

不对称 padding×dilation×stride：C=8/Co=24/Hi=5/Wi=6/Hk=3/Wk=2/pad=(1,2)/dil=(2,1)/stride=(2,1)。

每个输出空间位置增加 32B 的 0x5a 保护区，通过 ref/dut 对拍观察是否保持不变；完整物理输出 2688B，纳入 exact 对拍。

## 差异

设备侧保留填数和 padding 哨兵播种，不做尾部自判断；完整输出由 ref/dut 对拍；URAM 8K slab
（a/in@0、b/ker@8K、c/out@16K）；MAT 指令由 DAYAN_TEST 主线程直接发射。

## 简化数据

有效输入和核均为 S8 1，每个输出为有效窗口位置数乘 8；越界输入仍按零处理。
原有输入 padding、输出哨兵、形状和指令时序不变，不缩小 meta.json 对拍范围。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_conv2d_asym_dilation_test
```
