# Dayan kernel demos

该目录放置最小、可独立构建的 Dayan kernel 示例。首个示例是
`printf_grid`：每个 logical task 在 Device 上打印自己的三维 grid ID、完整 grid
维度、调度模式，以及 Host 传入的 7 个 `uint64_t` 参数。

## 目录

```text
demo/
├── Makefile
├── run_cmodel.sh
└── printf_grid/
    ├── kernel.cpp   # __kernel__ Device 实现
    └── main.cpp     # Runtime Host launcher
```

`kernel.cpp` 由 dycc 编译。dycc 在 Device 侧保留 kernel，在 Host 侧生成同名 wrapper；
`main.cpp` 只使用公开的 `npu_` Runtime API，不指定物理 Cluster 或 Core。

## 构建

```sh
make -C demo
```

产物为 `build/demo/printf_grid_demo`。它默认连接
`/tmp/dayan-server.sock`，也可以通过环境变量覆盖：

```sh
DAYAN_SERVER_SOCKET=/tmp/my-dayan.sock \
  build/demo/printf_grid_demo 1 2 3 4 5 6 7
```

参数省略时使用 `11 22 33 44 55 66 77`。示例采用
`NPUTaskDim3{2, 2, 1}` 和 `NPUTaskMode::kSingleCore`，因此 Device 应输出四行；不同
logical task 的输出顺序由 Service 调度决定，不应依赖其顺序。Device `printf` 由
Service 输出到 `dayan_server` 的标准输出，而不是 Runtime 应用的标准输出。

## 使用 cmodel 一键运行

默认使用 `~/rvtng-cmodel` 和 `~/npu-qemu/build/dl-riscv64`：

```sh
make -C demo run
```

也可以直接向脚本传入 7 个参数：

```sh
sh demo/run_cmodel.sh 10 20 30 40 50 60 70
```

可通过 `RVTNG_CMODEL_DIR`、`RVTNG_QEMU`、`NPU_DEVICE_PREFIX`、
`NPU_CMODEL_HARTS` 和 `DAYAN_SERVER_SOCKET` 覆盖模拟器与运行路径。脚本保留
`build/demo/cmodel.log` 和 `build/demo/dayan_server.log` 便于排查。
