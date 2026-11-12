# vec_fp16_pack_roundtrip_test

从 `dayan-samples/Instructions/VEC/dependency/vec_fp16_pack_roundtrip.cpp` 迁移的裸机版本，指令发射走
`instr_enc`（txenc）+`instr_enc/tx_test.h`（txtst）骨架，不使用 tensor_suite。

## 覆盖意图

单块内 FP32->FP16 窄化->Compress->DeCompress->宽化 roundtrip，6 条；fp16 RNE helper 逐行照抄样例。

host 填数 / oracle 原样移植到设备侧（float 按位比较）；CONTROL 约定 (0,0,N,N,N)；
发指令前轮询队列容量。详见 main.cpp 头注释。

## 构建运行

```sh
make -C dayan-base/device_tests VEC/vec_fp16_pack_roundtrip_test
cd dayan-base/sim && python3 tools/dayan_emu_runner.py ../dayan-base/device_tests/VEC/vec_fp16_pack_roundtrip_test
```
