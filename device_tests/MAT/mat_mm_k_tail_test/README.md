# mat_mm_k_tail_test

从 `dayan-samples/Instructions/MAT/mat_mm_k_tail.cpp.cpp` 迁移的裸机版本，发射走
`instr_enc`（txenc mat_enc）+ `instr_enc/tx_test.h`（txtst）骨架。

## 覆盖意图

K=33 尾块：A/B 行 64B 槽（33 有效 + 31 哨兵 padding），验证 K 尾乘加与 padding 不参与。

## 与原样例差异

设备侧保留填数，不做尾部自判断，完整输出由 ref/dut 位精确对拍；MAT 指令由 DAYAN_TEST 主线程
直接发射（资源块已声明 mat_tcsr_count，无需样例的 worker 线程）；逐 64B 分块
DMA 语义与样例一致。C slab 保留哨兵播种，存在的 padding 纳入完整物理输出对拍。

## 简化数据

有效 A1/A2/B1/B2 均为 1，MM/MMA 的有效输出应为 33/66。
原有输入 padding、输出哨兵、形状和指令时序不变，不缩小 meta.json 对拍范围。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_mm_k_tail_test
cd sim && python3 tools/dayan_emu_runner.py \
  ../dayan-base/device_tests/MAT/mat_mm_k_tail_test
```
