# mat_mm_asym_stride_test

从 `dayan-samples/Instructions/MAT/mat_mm_asym_stride.cpp.cpp` 迁移的裸机版本，发射走
`instr_enc`（txenc mat_enc）+ `instr_enc/tx_test.h`（txtst）骨架。

## 覆盖意图

A/B/C 步长不对称（64/128/64）：A 行 64B 槽、B 行 128B 槽、C 行 64B 槽。

A 数据同时依赖行号和 K 索引，多行用例不会因 A 各行相同而掩盖寻址错误。

## 与原样例差异

设备侧保留填数，不做尾部自判断，完整输出由 ref/dut 位精确对拍；MAT 指令由 DAYAN_TEST 主线程
直接发射（资源块已声明 mat_tcsr_count，无需样例的 worker 线程）；逐 64B 分块
DMA 语义与样例一致。C slab 保留哨兵播种，存在的 padding 纳入完整物理输出对拍。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_mm_asym_stride_test
cd sim && python3 tools/dayan_emu_runner.py \
  ../dayan-base/device_tests/MAT/mat_mm_asym_stride_test
```
