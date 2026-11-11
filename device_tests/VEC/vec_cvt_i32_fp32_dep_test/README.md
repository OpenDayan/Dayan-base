# VEC fp32 -> s32 -> int32.add -> fp32 单块跨类型依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_cvt_i32_fp32_dep.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`），不使用
`tensor_suite/tensor_test.h`。

## 覆盖意图（与原样例一致，VEC-DEP-022）

- 单个 VEC 指令块内的跨操作数类型 RAW 依赖：
  `u2v(fp32) -> tx.vec.cvt(fp32->s32) -> tx.vec.vv.int32.add -> tx.vec.cvt(s32->fp32) -> v2u`；
- 6 条指令（2 load + 2 cvt + 1 add + 1 store）一次 `CONTROL` 启动，
  块内无等待（原样例 `0x0606060606`）；
- 双循环队列年龄选择：bank0 装 src（选择子 0），bank1 装 addend（选择子 16）；
- Convert 模板名目的在前（dst_first），与 txenc 枚举一一对应：
  `cvt_s32_fp32 -> VEC_CVT_S32_FP32=1`、`cvt_fp32_int32 -> VEC_CVT_FP32_INT32=0`；
- 期望：`out[i]` 按位 == `(float)((int)(i%31)-15 + (int)(i%9)-4)`，
  输入公式与原样例一致（纯整数的精确 fp32 表示，无舍入歧义）。

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` DMA |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x0606060606)` | `txtst::start_vec_block(6)`（`(0,0,6,6,6)` 约定，直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(6)`（spec 5.4 流程） |
| host 逐元素 memcmp 对拍 | 设备侧逐元素按位对拍 + meta.json compare(exact) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## 构建 / 运行

```sh
make -C release/dayan-base/device_tests VEC/vec_cvt_i32_fp32_dep_test
cd release/sim && python3 tools/qemu_rvtng_dayan_emu_runner.py \
  ../dayan-base/device_tests/VEC/vec_cvt_i32_fp32_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
