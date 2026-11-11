# VEC 谓词 -> mux2 -> shuffle.xor -> mul 单块依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_pred_shuf.cpp` 迁移的裸机版本，
指令发射走 `instr_enc`（txenc）+ `instr_enc/tx_test.h`（txtst）骨架。

## 覆盖意图

- 单 VEC 块内：fp32 cmp（谓词写 s2）→ mux2（消费 s2）→ shuffle.xor（lane 两两交换）
  → fp32 mul，10 条指令一次 CONTROL 启动，块内无等待；
- 期望（按位比较）：`src=(i^1)&63; want=((a[src]>b[src])?t[src]:f[src])*scale[i]`。

## 与原样例差异

host 填数/oracle 移植到设备侧；CONTROL 约定 (0,0,N,N,N)；发指令前轮询队列容量。

## 构建运行

```sh
make -C dayan-base/device_tests VEC/vec_pred_shuf_test
cd dayan-base/sim && python3 tools/qemu_rvtng_dayan_emu_runner.py \
  ../dayan-base/device_tests/VEC/vec_pred_shuf_test
```
