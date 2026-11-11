# VEC 非零 V0 描述符（Dim2 行距）bf16 narrow -> compress 依赖测试

从 `dayan-samples/Instructions/VEC/dependency/vec_cvt_bf16_stride_dep.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`），不使用
`tensor_suite/tensor_test.h`。

## 覆盖意图（与原样例一致，VEC-DEP-027）

- 与 002 相同的 `u2v -> tx.vec.cvt(fp32->bf16, RNE) -> shuffle.compress(0x00)
  -> v2u` 依赖链，但 V0 描述符非零：
  `Dim2 = kRows-1 = 1`、`v0Base = 256`、`v0Dim2Stride = 1024`；
- 地址 = rd 基址实参（URAM in/out 基址）+ `256 + d2*1024 + 4*lane`，
  两行（row=0,1）各自独立读写，每行 256B 后跟 768B guard；
- 输入/输出区各 2304B（= 256 前缀 + 2 行 × 1024）：输入全填 123.5f，
  两个活动行（word 起点 `(256+row*1024)/4`）填样例公式；输出区全填
  `kGuardWord=0x5a93c76d` 并同样 `d2u` 种入 URAM，使越界/错位写可见；
- 4 条指令一个块（原样例 `0x0404040404` -> `start_vec_block(4)`）；
- Convert 模板名目的在前（dst_first）：样例 `cvt_bf16_fp32`（fp32->bf16
  narrow）= txenc `VEC_CVT_BF16_FP32=5`；
- 期望（host oracle 同款）：活动行 word 的期望 =
  `f32_to_bf16(in[start+2p]) | f32_to_bf16(in[start+2p+1])<<16`，
  `p = (word-start)&31`；其余 448 个 word 必须保持 guard 值。

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| host 填 input/output buffer + h2d（output 用来种 guard） | 设备侧直写 DDR 两个 2304B 区 + cache flush + 双 `d2u` |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x0404040404)` | `txtst::start_vec_block(4)`（直线码等价） |
| 不轮询队列容量 | `txtst::wait_vec_queue_capacity(4)`（spec 5.4 流程） |
| host 逐 word 对拍 2304B（活动 + guard） | 设备侧逐 word 对拍 + meta.json compare(exact, 2304B) |
| `dayan_wait_lsu/vec` | `txtst::wait_lsu/wait_vec`（Produce/Bar 单元侧读） |

## 构建 / 运行

```sh
make -C release/dayan-base/device_tests VEC/vec_cvt_bf16_stride_dep_test
cd release/sim && python3 tools/qemu_rvtng_dayan_emu_runner.py \
  ../dayan-base/device_tests/VEC/vec_cvt_bf16_stride_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
