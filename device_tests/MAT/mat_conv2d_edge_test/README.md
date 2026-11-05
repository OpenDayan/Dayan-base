# mat_conv2d_edge_test

## 参数与覆盖

S8/S8（DataType=0x8208），MAT group 0；维度寄存器写实际值减 1。

| 参数 | 值 |
| --- | --- |
| C / Co | 8 / 8 |
| Hi / Wi | 5 / 6 |
| Hk / Wk | 2 / 3 |
| PadL / PadU | 2 / 1 |
| DilationW / DilationH | 1 / 2 |
| StrideW / StrideH | 2 / 1 |
| Ho / Wo | 5 / 4 |

输出尺寸按上下各 PadU、左右各 PadL 计算；输入采样坐标为
`hi=ho*StrideH-PadU+hk*DilationH`、`wi=wo*StrideW-PadL+wk*DilationW`，越界按零处理。
输入/核使用随逻辑索引变化的正负数据：`(index*5+1)%13-6` / `(index*3+1)%7-4`。
不依赖常量数据绕过寻址问题。

## 布局与校验

- 输入 `[hi][wi][c]`、核 `[hk][wk][co][c]` 的 C 向量各占 32B 槽。
- 输出 `[ho][wo][co]` 为 FP32，每个空间位置占 32B，其中 32B 有效。
- 每份输出共 640B，meta.json 对两份完整物理输出 exact 对拍。
- OUT1/OUT2 预填 0x5a/0xa5，再搬入 URAM；OUT1=卷积、OUT2=两倍卷积，有效数据及存在的槽内 padding 统一交由 ref/dut 对拍。
- 每条 MAT 指令后等待 MAT 完成。跨 LSU/MAT 数据依赖均有单元等待。
- 全部 32 项 TCSR 显式初始化；swizzle 关闭，非 MX scale 步长为零，不声明对应功能覆盖。
- 编译期检查输出大小、对齐及包含 DMA 尾块的 slab 容量；修改 shape 时须同步 meta.json。

## 构建运行

从 `release/` 目录执行：

```sh
make -C dayan-base/device_tests MAT/mat_conv2d_edge_test
cd sim
python3 tools/qemu_rvtng_cmodel_runner.py ../dayan-base/device_tests/MAT/mat_conv2d_edge_test
```

参考模型通过不代表 RTL 已通过；非零 padding、尾通道和非常量寻址错误应保留暴露，不能通过弱化数据或对拍范围绕过。
