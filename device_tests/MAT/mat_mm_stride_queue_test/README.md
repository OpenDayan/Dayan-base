# mat_mm_stride_queue_test

从 `dayan-samples/Instructions/MAT/mat_mm_stride_queue.cpp.cpp` 迁移的裸机版本，发射走
`instr_enc`（txenc mat_enc）+ `instr_enc/tx_test.h`（txtst）骨架。

## 覆盖意图

MM 使用 A/B/C=128/128/128B；发射 OUT2 的 MM 后不等待，改写 A/B 步长为 64/64B，再发依赖同一 OUT2 的 MMA。MM 必须保留旧配置，MMA 使用新配置，最后统一等待；同时检查数值与输出 padding。

A 数据同时依赖行号和 K 索引，多行用例不会因 A 各行相同而掩盖寻址错误。

## 与原样例差异

设备侧保留填数，不做尾部自判断，完整输出由 ref/dut 位精确对拍；MAT 指令由 DAYAN_TEST 主线程
直接发射（资源块已声明 mat_tcsr_count，无需样例的 worker 线程）；逐 64B 分块
DMA 语义与样例一致。C slab 保留哨兵播种，存在的 padding 纳入完整物理输出对拍。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_mm_stride_queue_test
cd sim && python3 tools/qemu_rvtng_cmodel_runner.py \
  ../dayan-base/device_tests/MAT/mat_mm_stride_queue_test
```
