# MAT 设备测试

共 28 个独立用例：11 个卷积、14 个矩阵乘、3 个 setptr 探针。
使用 `instr_enc` 发射指令；设备侧只负责填数、执行和搬回输出，不做尾部自判断；
`meta.json` 定义完整物理输出的 ref/dut 对拍范围。维度统一写实际值减 1。

## 正向回归与模型限制

当前 QEMU+Dayan-emu 的正向回归集合为除 `mat_mm_fp32_shape_test` 外的 27 项。
`mat_mm_fp32_shape_test` 保留 FP32 容器 / TF32 pair（0x8410）的正向语义，
但当前 Dayan-emu 未实现 TF32 operand 转换，运行会触发 tensor interrupt。
这属于模型能力阻塞，不是成功的异常测试；不要吞掉错误、修改通过标准或改成其他 dtype。
模型支持补齐后，应将它加入正向通过集合。

`setptr` 三项仅验证规范 rd=x0 编码及后续 S8 MM/MMA 哨兵，
不验证 MX scale 指针指向的数据是否被正确消费。
所有用例当前使用 MAT group 0，不声明多组隔离、swizzle 或 MX scale 覆盖。

## 校验约定

- 所有用例均移除设备侧 golden、数值/padding 比较及相应退出逻辑；输出统一交由 ref/dut 对拍。conv2da 的预期仍为种子卷积加同一卷积。
- 上次回归失败的 7 项使用有效值全 1 的简单数据，便于观察结果；其他用例维持原输入。形状、dtype、padding、dilation、stride 和指令时序均不变。
- 存在输出槽/行 padding 的用例先播种哨兵，再通过完整 ref/dut 对拍观察是否保持不变；single-conv 和 single-MM 都保留实际保护字节，保护字节继续纳入对拍。
- 两个 single-S8 MM 的原始字节表与 golden 表有编译期一致性检查。
- 除简化的 `mat_mm_m2_k_tail_test` 外，多行 stride-MM 保留行间差异以区分错误复用第 0 行。
- `mat_mm_stride_queue_test` 在发射 MM 后、不等待完成时改 A/B stride，再发 MMA，覆盖旧/新配置与 RAW 依赖。
- shape/stride 改动必须同时更新 meta.json；源码中的大小断言记录对应约定，不能代替对 meta.json 的检查。
- FP32/TF32 用例使用 TF32 精确可表示的小值，避免把输入量化误差混入寻址测试。

## 简化用例

- `mat_mm_k1_queue_test`：有效 A1/A2/B1/B2 均为 1，MM/MMA 的有效输出应为 1/2。
- `mat_mm_k_tail_test`：有效 A1/A2/B1/B2 均为 1，MM/MMA 的有效输出应为 33/66。
- `mat_mm_m2_k_tail_test`：有效 A1/A2/B1/B2 均为 1，MM/MMA 的有效输出应为 33/66；本项不再依赖行间数据差异覆盖寻址。
- `mat_mm_bf16_shape_test`：有效 A/B 均为 BF16 1.0（0x3f80），有效输出应为 16.0。
- `mat_mm_fp32_shape_test`：有效 A/B 均为 FP32 容器中的 1.0（0x3f800000），有效输出应为 17.0；仍需模型支持 TF32。
- `mat_conv2d_pointwise_test`：有效输入和核均为 S8 1，有效输出应为 16.0。
- `mat_conv2d_asym_dilation_test`：有效输入和核均为 S8 1，每个输出为有效窗口位置数乘 8；越界输入仍按零处理。

## 构建和运行

从 `release/` 目录执行单项：

```sh
make -C dayan-base/device_tests MAT/mat_conv2d_test
cd sim
python3 tools/qemu_rvtng_dayan_emu_runner.py ../dayan-base/device_tests/MAT/mat_conv2d_test
```

从 `release/` 构建全部并运行当前可支持的正向集合，任一失败立即退出：

```sh
set -e
for manifest in dayan-base/device_tests/MAT/*/meta.json; do
    test_dir=${manifest%/meta.json}
    name=${test_dir##*/}
    make -C dayan-base/device_tests "MAT/$name"
    if [ "$name" = mat_mm_fp32_shape_test ]; then
        echo "BLOCKED: $name requires TF32 support in dayan_emu"
        continue
    fi
    (cd sim && python3 tools/qemu_rvtng_dayan_emu_runner.py "../$test_dir")
done
```

参考模型执行通过只说明执行流程未报错，不代表数值正确或 RTL 已通过；数值结论以 ref/dut 对拍为准。
简化数据减少了部分数值/寻址覆盖，但保留非零输入 padding 和输出保护区，不缩小对拍范围。
