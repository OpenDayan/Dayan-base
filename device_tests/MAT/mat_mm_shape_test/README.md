# mat_mm_shape_test

从 `dayan-samples/Instructions/MAT/mat_mm_shape.cpp.cpp` 迁移的裸机版本，发射走
`instr_enc`（txenc mat_enc）+ `instr_enc/tx_test.h`（txtst）骨架。

## 覆盖意图

交替形状 M=2/N=16/K=32，CSTRIDE=128（64 有效 + 64 padding），输出按物理步长布局，交由 ref/dut 对拍。

## 与原样例差异

设备侧保留填数，不做尾部自判断，完整输出由 ref/dut 位精确对拍；MAT 指令由 DAYAN_TEST 主线程
直接发射（资源块已声明 mat_tcsr_count，无需样例的 worker 线程）；逐 64B 分块
DMA 语义与样例一致。C slab 保留哨兵播种，存在的 padding 纳入完整物理输出对拍。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_mm_shape_test
cd sim && python3 tools/dayan_emu_runner.py \
  ../dayan-base/device_tests/MAT/mat_mm_shape_test
```
