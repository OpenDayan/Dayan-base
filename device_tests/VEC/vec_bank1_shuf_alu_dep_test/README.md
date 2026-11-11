# VEC bank1 shuffle.ror -> shuffle.xor -> fp32 sub 单块依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_bank1_shuf_alu_dep.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`），不使用
`tensor_suite/tensor_test.h`。

## 覆盖意图（与原样例一致）

- 与 ror/xor/sub 同形样例（同输入、同 ctrl、同 oracle）仅 bank 挪位的
  对照测试：shuffle -> shuffle -> ALU 三级 RAW 依赖全部发生在 bank1；
- A 装入 bank1（`u2v vb=1`），两次 shuffle 均以 vrs1=16（bank1 age0）读、
  结果留 bank1；B 装入 bank0（选择子 0）；fp32 sub 以 vrs1=16/vrs2=0/vb=1
  收敛到 bank1；最后 `v2u` 以 vrs1=16 从 bank1 age0 写出（注意 v2u 模板
  第一参数是 vrs1 选择子）；
- 6 条指令一次 `CONTROL` 启动，块内无等待；
- 静态 shuffle 的 ctrl 参数走 GPR 实参（`emit32_a0`，a0=ctrl），
  `kShuffleRor = 3`、`kShuffleXorPair = 1` 与原样例一致；
- 期望：`out[i] == a[((i ^ 1) + 3) & 63] - b[i]`（按位比较），输入公式
  `a[i]=(float)(1000+13i)`、`b[i]=(float)(-300+7i)` 与原样例一致。

## 指令序列

```
u2v  vb=1 sel=0        <- A_URAM          (bank1 装入 A)
shuf ror vrs1=16 vb=1  ctrl=3            (lane i <- (i+3)&63，留 bank1)
shuf xor vrs1=16 vb=1  ctrl=1            (lane i <- 上一步的 i^1，仍留 bank1)
u2v  vb=0 sel=0        <- B_URAM          (bank0 装入 B)
fp32 sub vrs1=16 vrs2=0 vb=1 rm=0        (bank1 shuffle 结果 - B，回 bank1)
v2u  vrs1=16 sel=0     -> OUT_URAM        (bank1 age0 写出)
```

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` DMA |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x060606060606L)` | `txtst::start_vec_block(6)`（`(0,0,6,6,6)` 约定，直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(6)`（spec 5.4 流程） |
| host 逐 lane same_bits 对拍（%f 打印） | 设备侧逐 lane 按位对拍（union 位比较，%08x 打印）+ meta.json compare(exact) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## 构建 / 运行

```sh
flock /tmp/dayan-build.lock make -C release/dayan-base/device_tests VEC/vec_bank1_shuf_alu_dep_test
cd release/sim && python3 tools/qemu_rvtng_dayan_emu_runner.py \
  ../dayan-base/device_tests/VEC/vec_bank1_shuf_alu_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
