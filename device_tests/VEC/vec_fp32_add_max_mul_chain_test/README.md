# VEC fp32 add -> max -> mul 单块依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_fp32_add_max_mul_chain.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`），不使用
`tensor_suite/tensor_test.h`。

## 覆盖意图（与原样例一致）

- 单个 VEC 指令块内的三级跨操作 RAW 依赖：
  `tx.vec.vv.fp32.add -> tx.vec.vv.fp32.max -> tx.vec.vv.fp32.mul`；
- 8 条指令（4 load + 3 运算 + 1 store）一次 `CONTROL` 启动，块内无等待；
- 双循环队列年龄选择：bank0 装入 a（选择子 0），bank1 装入 b/c/d
  （选择子 18/17/16，age 2/1/0）；
- 期望：`out[i] == ((a[i] + b[i]) > c[i] ? (a[i] + b[i]) : c[i]) * d[i]`，
  输入公式与原样例一致（a/b/c/d 均为 2 的幂次小数，运算精确无舍入歧义）。

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` DMA |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x0808080808)` | `txtst::start_vec_block(8)`（`(0,0,8,8,8)` 约定，直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(8)`（spec 5.4 流程） |
| host 逐 lane `out[i] != expected` 对拍 | 设备侧逐 lane 按位对拍（`same_bits`，union 转 u32）+ meta.json compare(exact) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## 构建 / 运行

```sh
cd <包含 dayan-base 与 sim 的检出目录>
flock /tmp/dayan-build.lock make -C release/dayan-base/device_tests VEC/vec_fp32_add_max_mul_chain_test
cd release/sim && python3 tools/qemu_rvtng_dayan_emu_runner.py \
  ../dayan-base/device_tests/VEC/vec_fp32_add_max_mul_chain_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
