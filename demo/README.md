# Dayan kernel demos

该目录放置最小、可独立构建的 Dayan kernel 示例。目前包含 Device `printf` 和
单 Core BF16 矩阵乘两个例子。

## 目录

```text
demo/
├── Makefile
├── device_include/cstdint       # freestanding Device 指令头兼容层
├── matmul_bf16_1c.cpp           # BF16 OS kernel、Host launcher 与 CPU 校验
├── run_matmul_cmodel.sh
├── run_cmodel.sh
└── print_demo.cpp   # __kernel__ Device 实现与 Runtime Host launcher
```

`print_demo.cpp` 由 dycc 编译。dycc 在 Device 侧保留 kernel，在 Host 侧生成同名
wrapper 和 main；Host 代码只使用公开的 `npu_` Runtime API，不指定物理 Cluster 或
Core。

## 构建

```sh
make -C demo
```

产物为 `build/demo/print_demo` 和 `build/demo/matmul_bf16_1c`。矩阵指令定义来自相邻的
`npu-kernels` 仓库，默认路径是 `../npu-kernels`，也可显式覆盖：

```sh
make -C demo NPU_KERNELS_DIR=/path/to/npu-kernels
```

两个 Host 程序默认连接
`/tmp/dayan-server.sock`，也可以通过环境变量覆盖：

```sh
DAYAN_SERVER_SOCKET=/tmp/my-dayan.sock \
  build/demo/print_demo 1 2 3 4 5 6 7
```

参数省略时使用 `11 22 33 44 55 66 77`。示例采用
`NPUTaskDim3{2, 2, 1}` 和 `NPUTaskMode::kSingleCore`，因此 Device 应输出四行；不同
logical task 的输出顺序由 Service 调度决定，不应依赖其顺序。Device `printf` 由
Service 转发给发起算子的 Runtime Session，并由 demo 进程输出；`dayan_server`
本身不再打印算子文本。

## 单 Core BF16 matmul

`matmul_bf16_1c` 计算以下矩阵乘，输出为 FP32：

```text
A[M][K] (BF16) × BT[N][K] (BF16) -> C[M][N] (FP32)
```

这里 Host 直接按转置布局提供 B。Device 算法沿用 output-stationary 分块：
`tileM=384`、`tileN=256`、`tileK=256`。一个 root proc 负责 B 的 DMA producer；三个
子 proc 各自负责一个固定的 M slice。A 使用三个 slice buffer，B 使用两个 full-tile
buffer 真双缓冲，C 使用三个跨 K 循环常驻的 slice buffer。B 的代际与 producer /
consumer 关系由 `dayan_resource_init/alloc/produce/wait/consume` 管理；LSU 和 MAT
完成等待分别使用 `dayan_wait_lsu()` 与 `dayan_wait_mat()`。

默认 Host case 为 `M=384, N=64, K=256`，它会生成确定性 BF16 输入、执行 Device
kernel、下载结果并逐元素和 CPU 参考结果比较：

```sh
build/demo/matmul_bf16_1c
build/demo/matmul_bf16_1c 768 256 512
```

当前保留原分块算法的对齐约束：`N` 必须是 16 的整数倍，`K` 必须是 256 的整数倍。
编译期可把切片模式改成单 slice；切换已有产物时用 `-B` 强制重新构建：

```sh
make -B -C demo MATMUL_SLICES=1
```

## 使用 cmodel 一键运行

默认使用 `~/rvtng-cmodel` 和 `~/npu-qemu/build/dl-riscv64`：

```sh
make -C demo run
```

也可以直接向脚本传入 7 个参数：

```sh
bash demo/run_cmodel.sh 10 20 30 40 50 60 70
```

可通过 `RVTNG_CMODEL_DIR`、`RVTNG_QEMU`、`NPU_DEVICE_PREFIX`、
`NPU_CMODEL_HARTS` 和 `DAYAN_SERVER_SOCKET` 覆盖模拟器与运行路径。脚本保留
`build/demo/cmodel.log`、`build/demo/dayan_server.log` 和
`build/demo/print_demo.log` 便于排查。

matmul 使用独立的一键脚本，默认只启动一个 Hart：

```sh
make -C demo run-matmul
bash demo/run_matmul_cmodel.sh 384 64 256
```

对应日志为 `build/demo/matmul_cmodel.log`、
`build/demo/matmul_dayan_server.log` 和 `build/demo/matmul_bf16_1c.log`。
