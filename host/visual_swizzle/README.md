# VisualSwizzle

该目录包含自 VisualSwizzle 项目引入的单文件 NPU URAM swizzle
可视化与搜索工具。模型与 RVTNG `SwizzleAddr` 一致：使用字节地址，只改变
`addr[14:5]`，保留 32B 字内偏移和 32 KiB 区域以上的地址位。

## 启动

工具只依赖 Python 3：

```sh
python3 host/visual_swizzle/visual_swizzle.py
```

默认打开 `http://127.0.0.1:8765`；端口占用时会依次尝试后续端口。无图形环境下可用：

```sh
python3 host/visual_swizzle/visual_swizzle.py \
  --host 0.0.0.0 --port 8765 --no-browser
python3 host/visual_swizzle/visual_swizzle.py --self-test
```

页面中填写元素位宽、行数、每行字节数、stride、基地址和 Bank 数，然后点击
“搜索最优 M/B/S”。搜索按以下顺序比较候选：

1. 降低所有访问拍的最坏 Bank 峰值。
2. 提高任意水平连续四块的 Bank 覆盖率。
3. 减少额外 Bank 请求和发生冲突的拍数。
4. 在性能相同时，选择更小的重排窗口和更少的 XOR 位。

页面参数按 `M/B/S` 显示；代码中使用：

```cpp
constexpr uint64_t swizzle = MAT_BATCH_SWIZZLE(M, B, S);
```

同一块 URAM 数据的生产者和消费者必须使用完全相同的有效映射。例如 DRAM 到
URAM 的 Q load 应在 `LSU_URAMWRITESWIZZLEMBS` 配置该值，MAT 读取 Q 时应在
`MAT_ASWIZZLEMBS` 配置同一个值。只改一端会读到错误数据。

## FlashAttention 选择

当前 `D=128` FlashAttention 使用 32 个 Bank、每拍 8 行 x 32B 的访问模型：

| 数据 | 类型 | row bytes/stride | 搜索结果 | 最坏峰值 |
| --- | --- | ---: | --- | ---: |
| Q/KT | E4M3 | 128B | `M5/B0/S1` identity | 1x |
| Q/KT | FP16 | 256B | `M7/B1/S3` | 1x |
| score/P | FP32 | 512B | `M7/B2/S3` | 1x |

score/P 的搜索结果暂未用于算子。当前 FlexVec TCSR 没有 URAM swizzle 配置，故
MAT C、FlexVec score/P 访问和 LSU P store 全部使用 identity。待 FlexVec ISA
补齐 swizzle 后，三端必须一起切换，不能只启用 MAT C。

## 地址原语

```text
local = (addr >> 5) & 0x3ff
mask = ((((1 << B) - 1) & 0xff) << ((M - 5) & 0xf)) & 0x3ff
mapped = (addr & ~0x7fe0) | ((local ^ ((local >> S) & mask)) << 5)
```

`B=0` 时 mask 为 0，是恒等映射。硬件只映射每个 32 KiB 区域内部的地址，因此
还需确认选定参数的重排窗口不会破坏 buffer 边界和别名关系。
