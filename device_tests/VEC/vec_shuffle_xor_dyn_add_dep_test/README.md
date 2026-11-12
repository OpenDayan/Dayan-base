# VEC 动态 shuffle.xor -> fp32 add 单块依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_shuffle_xor_dyn_add_dep.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`），不使用
`tensor_suite/tensor_test.h`。

## 覆盖意图（与原样例一致）

- 单个 VEC 指令块内的动态 shuffle -> ALU 跨操作 RAW 依赖：
  `tx.vec.lsu.g2s(s1=3)` -> `tx.vec.shuffle.xor.dyn(s1)` -> `tx.vec.vv.fp32.add`；
- 重排参数（xor pair = 3）先经 g2s 装入线程内标量寄存器 s1，再发无实参的
  动态 XOR-shuffle（lane i 取 a[i^3]），验证标量通路到向量通路的块内依赖；
- 6 条指令（load A + g2s + dyn shuffle + load B + fp32 add + store）一次
  `CONTROL` 启动，块内无等待；
- 双循环队列年龄选择：bank0 装入 A 与 shuffle 结果（选择子 0），bank1 装入
  B（选择子 16，age 0）；
- 期望：`out[i] == a[i ^ 3] + b[i]`，输入公式与原样例一致。

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` DMA |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x060606060606L)` | `txtst::start_vec_block(6)`（`(0,0,6,6,6)` 约定，直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(6)`（spec 5.4 流程） |
| host 逐 lane 对拍 | 设备侧逐 lane 按位对拍 + meta.json compare(exact) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## 构建 / 运行

```sh
make -C dayan-base/device_tests VEC/vec_shuffle_xor_dyn_add_dep_test
cd dayan-base/sim && python3 tools/dayan_emu_runner.py \
  ../dayan-base/device_tests/VEC/vec_shuffle_xor_dyn_add_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
