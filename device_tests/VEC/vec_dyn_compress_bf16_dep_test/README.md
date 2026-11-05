# VEC 静态/动态 Compress -> DeCompress -> BF16 widen 依赖测试（双 ctrl 对）

从 `dayan-samples/Instructions/VEC/dependency/vec_dyn_compress_bf16_dep.cpp`
迁移到 `dayan-base/device_tests` 的裸机版本，指令发射全部走
`device_tests/instr_enc` 编码库（`txenc::*_enc` + emit 模板），配置 / 同步 /
搬运走公共骨架 `instr_enc/tx_test.h`（`txtst::`），不使用
`tensor_suite/tensor_test.h`。

## 覆盖意图（与原样例一致，VEC-DEP-025）

- 每对一个 16 条指令的 VEC 块（原样例 `0x101010101010` ->
  `start_vec_block(16)`）：u2v -> bf16 narrow -> static compress ->
  v2u；再 u2v -> narrow -> g2s(s1) -> compress_dyn -> v2u；static
  decompress -> v2u；g2s(s2) -> compress_dyn(age1) -> v2u；widen -> v2u；
- 五个输出段/对：0=static packed、1=dynamic packed、2=static expanded、
  3=dynamic expanded、4=widened dynamic；v2u 与 g2s 不推进向量队列
  age（dynamic DeCompress 读 age1=packed 而非 age0=expanded）；
- ctrl 对 (0x00,0x20)：compress 保留全部 BF16 值；(0x04,0x24)：选
  packed pair0/16 并扩展成 32-lane 组（非恒等）；
- 期望：样例 host oracle 同款（`f32_to_bf16`/`bf16_to_f32`/`float_bits`
  位运算 helper 逐行照抄），外加静态/动态 parity（段1==段0、段3==段2）。

## 与原样例的差异

| 原样例（host runtime） | 本裸机版 |
| --- | --- |
| kernel 带运行时参数，host 分两次 launch（每对重配描述符、复用同一 1280B 输出 buffer） | **一个 tensor_run 顺序跑两个 ctrl 对**：LSU/VEC 配置一次，每对 16 条指令 + `start_vec_block(16)` + `wait_vec`（样例 if/else 二选一，这里编译期两对都写） |
| 每次launch后 u2d 1280B 回 host | 第二对输出放 `out_uram + 5*kBytes` 起，两对 10 段 = 2560B 连续放 OUT_DDR，一次 u2d + compare(exact, 2560B) |
| host 填 buffer + h2d memcpy | 设备侧直写 DDR + cache flush + `d2u` |
| `__tx_vec_tcsr_write<VEC_CONTROL>(0x101010101010)` | `txtst::start_vec_block(16)`（直线码等价） |
| 不轮询队列容量 | 每块前 `txtst::wait_vec_queue_capacity(16)`（spec 5.4 流程） |
| host 逐段对拍 | 设备侧逐段对拍（5 段/对 × 2 对 + parity） |

指令序列严格照样例（含两条 `g2s` 到 s1/s2、compress_dyn 读 s、
static compress/decompress 混排；立即数经 a0 传入）。样例 Convert 模板名
目的在前（dst_first）：`cvt_bf16_fp32`（fp32->bf16 narrow）=
`VEC_CVT_BF16_FP32=5`，`cvt_fp32_bf16`（bf16->fp32 widen）=
`VEC_CVT_FP32_BF16=3`。

## 构建 / 运行

```sh
make -C release/dayan-base/device_tests VEC/vec_dyn_compress_bf16_dep_test
cd release/sim && python3 tools/qemu_rvtng_cmodel_runner.py \
  ../dayan-base/device_tests/VEC/vec_dyn_compress_bf16_dep_test
```

QEMU 侧通过后生成 `sim/sim/ref/<name>/` 参考 artifact，供 RTL 对拍。
