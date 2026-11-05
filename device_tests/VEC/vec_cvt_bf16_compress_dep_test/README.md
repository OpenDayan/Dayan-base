# VEC cvt fp32->bf16 -> compress 单块依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_cvt_bf16_compress_dep.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`），不使用
`tensor_suite/tensor_test.h`。

## 覆盖意图（与原样例一致）

- 单个 VEC 指令块内的 cvt -> shuffle RAW 依赖：
  `tx.vec.cvt.bf16.fp32`（窄化，FP32 lane 2k/2k+1 两个 BF16 打包进偶 lane 2k）
  -> `tx.vec.shuffle.compress(ctrl=0x00, p=1)`；
- 4 条指令（load + cvt + compress + store）一次 `CONTROL` 启动，块内无等待；
- 全部读写 bank0 最新值（选择子 0）；
- 期望：`out[i] == expected[i & 31]`，
  `expected[k] = bf16(in[2k]) | (bf16(in[2k+1]) << 16)`（RNE），
  输入 `in[i] = (float)((int)(i%23)-11)*0.375f + (float)(i%7)*0.03125f`。

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` DMA |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x0404040404)` | `txtst::start_vec_block(4)`（`(0,0,4,4,4)` 约定，直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(4)`（spec 5.4 流程） |
| host 逐 lane 对拍（f32_to_bf16 helper） | 设备侧逐 lane 对拍（helper 逐行照抄，memcpy 换 union）+ meta.json compare(exact) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## 构建 / 运行

```sh
make -C dayan-base/device_tests VEC/vec_cvt_bf16_compress_dep_test
cd dayan-base/sim && python3 tools/qemu_rvtng_cmodel_runner.py \
  ../dayan-base/device_tests/VEC/vec_cvt_bf16_compress_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
