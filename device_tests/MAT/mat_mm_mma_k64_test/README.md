# mat_mm_mma_k64_test

从 `dayan-samples/Instructions/MAT/mat_mm_mma_k64.cpp.cpp` 迁移的裸机版本，发射走
`instr_enc`（txenc mat_enc）+ `instr_enc/tx_test.h`（txtst）骨架。

## 覆盖意图

K=64 满行（无 K 尾块）基线：mm 初始化 + mma 累加两条路径。

## 与原样例差异

设备侧保留填数，不做尾部自判断，完整输出由 ref/dut 位精确对拍；MAT 指令由 DAYAN_TEST 主线程
直接发射（资源块已声明 mat_tcsr_count，无需样例的 worker 线程）；逐 64B 分块
DMA 语义与样例一致。

## 构建运行

```sh
make -C dayan-base/device_tests MAT/mat_mm_mma_k64_test
cd sim && python3 tools/qemu_rvtng_dayan_emu_runner.py \
  ../dayan-base/device_tests/MAT/mat_mm_mma_k64_test
```
