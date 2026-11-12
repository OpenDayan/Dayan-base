# VEC exp2 -> recip SFU -> fp32 mul 单块依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_exp2_recip_fp32_mul_dep.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`）。

## 覆盖意图（与原样例一致）

- 单个 VEC 指令块内的两级 SFU 链 + FP32 乘 RAW 依赖：
  `tx.vec.v.fp32.sfu.exp2 -> tx.vec.v.fp32.sfu.recip -> tx.vec.vv.fp32.mul`；
- 6 条指令（load X、exp2 SFU、recip SFU、load scale、mul、store）一次
  `CONTROL` 启动，块内无等待；
- 双循环队列年龄选择：bank0 装 X / exp2 / recip 结果（选择子 0，逐级最新），
  bank1 装 scale（选择子 16，age 0）；
- 期望：`out[i] == scale[i] / exp2(x[i])`，输入公式与原样例一致
  （x 取整数指数 {-3..4} 循环、scale 取 8 个小尾数值循环）。

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` DMA |
| host `scale / std::ldexp(1.0f, x)` 对拍 | 设备侧位对拍；ldexp 项换 2^x 精确常量表（位级等价，裸机无 libm） |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x060606060606)` | `txtst::start_vec_block(6)`（`(0,0,6,6,6)` 约定，直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(6)`（spec 5.4 流程） |
| host 逐 lane 对拍（`!=`） | 设备侧逐 lane `same_bits` 位对拍 + meta.json compare(exact) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## oracle 移植说明（裸机 freestanding，禁 exp2f/ldexp）

- x 全部为整数指数 -3..4，`exp2(x) = 2^x` 用 8 项精确常量表
  `{0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f}`（下标 = 指数 + 3），
  与样例 `ldexp(1.0f, exponent)` 位级相同；
- 期望式 `scale / 2^x` 照抄样例除法（除以 2 的幂仅改指数，`fdiv.s` 精确）。

## 构建 / 运行

```sh
make -C dayan-base/device_tests VEC/vec_exp2_recip_fp32_mul_dep_test
cd dayan-base/sim && python3 tools/dayan_emu_runner.py \
  ../dayan-base/device_tests/VEC/vec_exp2_recip_fp32_mul_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
