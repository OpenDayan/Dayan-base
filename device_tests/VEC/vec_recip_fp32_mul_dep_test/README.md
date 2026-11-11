# VEC recip SFU -> fp32 mul 单块依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_recip_fp32_mul_dep.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`）。

## 覆盖意图（与原样例一致）

- 单个 VEC 指令块内的 SFU -> FP32 乘 RAW 依赖：
  `tx.vec.v.fp32.sfu.recip -> tx.vec.vv.fp32.mul`；
- 5 条指令（load X、recip SFU、load scale、mul、store）一次 `CONTROL` 启动，
  块内无等待；
- 双循环队列年龄选择：bank0 装 X / SFU 结果（选择子 0），bank1 装 scale
  （选择子 16，age 0）；
- 期望：`out[i] == (1.0f / x[i]) * scale[i]`，输入公式与原样例一致
  （x 取 2 的幂 {0.25..8} 循环、scale 取 6 个小尾数值循环）。

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` DMA |
| host `(1.0f / x) * scale` 对拍 | 设备侧 `same_bits` 位对拍，公式原样照抄 |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x0505050505)` | `txtst::start_vec_block(5)`（`(0,0,5,5,5)` 约定，直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(5)`（spec 5.4 流程） |
| host 逐 lane 对拍（`!=`） | 设备侧逐 lane `same_bits` 位对拍 + meta.json compare(exact) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## oracle 移植说明（裸机 freestanding）

- x 全部为 2 的整数幂（0.25..8），`1.0f / x` 由硬件 `fdiv.s`（正确舍入）
  精确得出 2 的幂，无需查表，直接照抄样例除法表达式；
- 其后的 mul 是普通 IEEE 运算，C 表达式直接移植
  （`expected = (1.0f / x[i]) * scale[i]`，均为精确值）。

## 构建 / 运行

```sh
make -C dayan-base/device_tests VEC/vec_recip_fp32_mul_dep_test
cd dayan-base/sim && python3 tools/qemu_rvtng_dayan_emu_runner.py \
  ../dayan-base/device_tests/VEC/vec_recip_fp32_mul_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
