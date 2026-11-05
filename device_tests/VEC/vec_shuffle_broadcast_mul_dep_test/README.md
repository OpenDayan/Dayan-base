# VEC shuffle.broadcast -> fp32 mul 单块依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_shuffle_broadcast_mul_dep.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`），不使用
`tensor_suite/tensor_test.h`。

## 覆盖意图（与原样例一致）

- 单个 VEC 指令块内的 shuffle -> ALU 跨操作 RAW 依赖：
  `tx.vec.shuffle.broadcast(ctrl=13) -> tx.vec.vv.fp32.mul`，广播结果留在
  bank0 V0 被同块后续 mul 消费；
- 5 条指令（load A + shuffle.broadcast + load B + mul + store）一次
  `CONTROL` 启动，块内无等待；
- 双循环队列年龄选择：bank0 = A 与广播结果（选择子 0），bank1 = B
  （选择子 16，age 0）；
- 静态 shuffle 的 ctrl 参数走 GPR 实参（`emit32_a0`，a0=ctrl），
  `kShuffleBroadcastLane = 13` 与原样例一致；
- 期望：`out[i] == a[13] * b[i]`（按位比较），输入公式
  `a[i]=(float)(1000+13i)`、`b[i]=(float)(-300+7i)` 与原样例一致。

## 指令序列

```
u2v  vb=0 sel=0        <- A_URAM          (bank0 装入 A)
shuf broadcast vrs1=0 vb=0 ctrl=13        (每 lane <- a[13]，留 bank0)
u2v  vb=1 sel=0        <- B_URAM          (bank1 装入 B)
fp32 mul vrs1=0 vrs2=16 vb=0 rm=0         (广播结果 * B)
v2u  vrs1=0 sel=0      -> OUT_URAM        (bank0 age0 写出)
```

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` DMA |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x0505050505L)` | `txtst::start_vec_block(5)`（`(0,0,5,5,5)` 约定，直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(5)`（spec 5.4 流程） |
| host 逐 lane same_bits 对拍（%f 打印） | 设备侧逐 lane 按位对拍（union 位比较，%08x 打印）+ meta.json compare(exact) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## 构建 / 运行

```sh
flock /tmp/dayan-build.lock make -C release/dayan-base/device_tests VEC/vec_shuffle_broadcast_mul_dep_test
cd release/sim && python3 tools/qemu_rvtng_cmodel_runner.py \
  ../dayan-base/device_tests/VEC/vec_shuffle_broadcast_mul_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
