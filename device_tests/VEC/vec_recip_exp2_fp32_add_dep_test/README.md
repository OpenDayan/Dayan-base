# VEC recip -> exp2 SFU -> fp32 add 单块依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_recip_exp2_fp32_add_dep.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`）。

## 覆盖意图（与原样例一致）

- 单个 VEC 指令块内的两级 SFU 链 + FP32 加 RAW 依赖：
  `tx.vec.v.fp32.sfu.recip -> tx.vec.v.fp32.sfu.exp2 -> tx.vec.vv.fp32.add`；
- 6 条指令（load X、recip SFU、exp2 SFU、load addend、add、store）一次
  `CONTROL` 启动，块内无等待；
- 双循环队列年龄选择：bank0 装 X / recip / exp2 结果（选择子 0，逐级最新），
  bank1 装 addend（选择子 16，age 0）；
- 期望：`out[i] == exp2(1.0f / x[i]) + addend[i]`，输入公式与原样例一致
  （x 取 2 的幂 {0.5..4} 循环、addend 取 {0.25,-0.5,1.5,-2.25} 循环）。

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` DMA |
| host `std::exp2(1.0f / x) + addend` 对拍 | 设备侧位对拍；exp2 换查表常量（见下），除法 / 加法照抄 |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x060606060606)` | `txtst::start_vec_block(6)`（`(0,0,6,6,6)` 约定，直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(6)`（spec 5.4 流程） |
| host 逐 lane 对拍（`!=`） | 设备侧逐 lane `same_bits` 位对拍 + meta.json compare(exact) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## oracle 移植说明（裸机 freestanding，禁 exp2f）

- x 全部为 2 的整数幂，`1.0f / x` 由硬件 `fdiv.s` 精确得出
  `{2.0, 1.0, 0.5f, 0.25f}`；
- `exp2(1/x)` 查表：`exp2(2)=4.0f`、`exp2(1)=2.0f` 精确；`exp2(0.5)=√2`、
  `exp2(0.25)=2^0.25` 为无理数，取正确舍入的 fp32 常量
  `1.41421353816986083984375f`（0x3fb504f3）、
  `1.1892070770263671875f`（0x3f9837f0），与样例 host `std::exp2` 的位级
  结果一致；
- 其后的 add 是普通 IEEE 运算，C 表达式直接移植
  （`expected = exp2_of_pow2(1.0f / x[i]) + addend[i]`）。

## 构建 / 运行

```sh
make -C dayan-base/device_tests VEC/vec_recip_exp2_fp32_add_dep_test
cd dayan-base/sim && python3 tools/qemu_rvtng_cmodel_runner.py \
  ../dayan-base/device_tests/VEC/vec_recip_exp2_fp32_add_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
