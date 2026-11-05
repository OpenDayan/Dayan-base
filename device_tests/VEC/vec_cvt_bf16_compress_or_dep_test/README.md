# vec_cvt_bf16_compress_or_dep_test

从 `dayan-samples/Instructions/VEC/dependency/vec_cvt_bf16_compress_or_dep.cpp` 迁移的裸机版本，指令发射走
`instr_enc`（txenc）+`instr_enc/tx_test.h`（txtst）骨架，不使用 tensor_suite。

## 覆盖意图

单块内 窄化->Compress->int32 OR，6 条；两半复制的打包块都检查。

host 填数 / oracle 原样移植到设备侧（float 按位比较）；CONTROL 约定 (0,0,N,N,N)；
发指令前轮询队列容量。详见 main.cpp 头注释。

## 构建运行

```sh
make -C dayan-base/device_tests VEC/vec_cvt_bf16_compress_or_dep_test
cd dayan-base/sim && python3 tools/qemu_rvtng_cmodel_runner.py ../dayan-base/device_tests/VEC/vec_cvt_bf16_compress_or_dep_test
```
