# VEC int32 add -> xor -> rol 单块依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_i32_add_xor_rol_chain.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走同目录新增的公共骨架 `instr_enc/tx_test.h`（`txtst::`），不使用
`tensor_suite/tensor_test.h`。

## 覆盖意图（与原样例一致）

- 单个 VEC 指令块内的三级跨操作 RAW 依赖：
  `tx.vec.vv.int32.add -> tx.vec.vv.int32.bitwise(XOR) -> tx.vec.vv.int32.rol`；
- 8 条指令（4 load + 3 运算 + 1 store）一次 `CONTROL` 启动，块内无等待；
- 双循环队列年龄选择：bank0 装入 a（选择子 0），bank1 装入 b/c/d
  （选择子 18/17/16，age 2/1/0）；
- 期望：`out[i] == rotl32((u32(a[i]) + u32(b[i])) ^ u32(c[i]), d[i])`，
  输入公式与原样例一致（d 旋转量取 1/7/13/29，非 0 无 UB）。

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` DMA |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x0808080808)` | `txtst::start_vec_block(8)`（`(0,0,8,8,8)` 约定，直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(8)`（spec 5.4 流程） |
| host 逐 lane 对拍 | 设备侧逐 lane 对拍 + meta.json compare(exact) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## 构建 / 运行

```sh
make -C dayan-base/device_tests VEC/vec_i32_add_xor_rol_dep_test
cd dayan-base/sim && python3 tools/qemu_rvtng_cmodel_runner.py \
  ../dayan-base/device_tests/VEC/vec_i32_add_xor_rol_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
