# vec_pred_bit_test

从 `dayan-samples/Instructions/VEC/dependency/vec_pred_bit.cpp` 迁移的裸机版本，指令发射走
`instr_enc`（txenc）+`instr_enc/tx_test.h`（txtst）骨架，不使用 tensor_suite。

## 覆盖意图

单块内双谓词组合：fp32 GT cmp(S2)+LT cmp(S3) -> s.bitwise OR(S4) -> mux2 -> add，13 条。样例因编译器 builtin LUT bug 用手编 .4byte，本库直接 vec_s_bitwise_enc(0xE,2,3,4)（位级等价）。

host 填数 / oracle 原样移植到设备侧（float 按位比较）；CONTROL 约定 (0,0,N,N,N)；
发指令前轮询队列容量。详见 main.cpp 头注释。

## 构建运行

```sh
make -C dayan-base/device_tests VEC/vec_pred_bit_test
cd dayan-base/sim && python3 tools/dayan_emu_runner.py ../dayan-base/device_tests/VEC/vec_pred_bit_test
```
