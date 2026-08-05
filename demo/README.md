# Dayan kernel demos

该目录放置最小、可独立构建的 Dayan kernel 示例。目前包含 Device `printf`、
单 Core BF16 矩阵乘和四 Core 协作 BF16 矩阵乘。

## 目录

```text
demo/
├── Makefile
├── device_include/cstdint       # freestanding Device 指令头兼容层
├── lsu_bandwidth.cpp            # 单核 D2U / 远端 U2D 带宽微基准
├── shared_notify_latency.cpp    # core 0 -> core 1 Shared Resource 通知延迟
├── matmul_bf16_1c.cpp           # BF16 OS kernel、Host launcher 与 CPU 校验
├── matmul_bf16_4c.cpp           # 2x2 Cluster OS：DRAM Load、远端 Copy、双 MatStore
├── run_matmul_cmodel.sh
├── run_matmul_4c_cmodel.sh
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

产物为 `build/demo/print_demo`、`build/demo/lsu_bandwidth`、
`build/demo/shared_notify_latency`、
`build/demo/matmul_bf16_1c`、`build/demo/matmul_bf16_4c` 和
`build/demo/matmul_bf16_4c_d2a_model`。矩阵指令定义来自相邻的
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

## LSU 单核带宽微基准

`lsu_bandwidth` 隔离测试一个 Core 发射的连续 D2U，以及从本 Core URAM 到相邻 Core
URAM 地址窗的 U2D。四个 Core 仅用于建立 Cluster URAM 映射；测量区间内只有 lane 0
发射 DMA。每个尺寸都严格采用“发射一条 DMA、立即 `dayan_wait_lsu()`”的顺序，并用
Device `mcycle` 记录 4 KiB、16 KiB、64 KiB、256 KiB 和 1 MiB 的端到端周期。测试结束
后另行回读 64 KiB 做数据校验，回读不计入带宽：

```sh
build/demo/lsu_bandwidth       # 依次测试 D2U 和远端 U2D
build/demo/lsu_bandwidth d2u
build/demo/lsu_bandwidth u2d
```

## 跨 Core 通知延迟微基准

`shared_notify_latency` 隔离测量 Atomic Cache 上
`dayan_shared_resource_t` 的 core 0 -> core 1 通知路径。core 1 先用另一项
Shared Resource 发布 armed 握手，保证自己已经进入接收流程；core 0 随后在
`dayan_shared_resource_produce()` 前记录 `mcycle`，core 1 在
`dayan_shared_resource_wait_ready()` 返回后立即记录 `mcycle`。此外 core 1 会立刻
回 ACK，得到一个不依赖跨 Core 计数器对齐的 round-trip 对照。默认预热 16 次并采集
64 个样本，可指定不超过 512 的样本数：

```sh
build/demo/shared_notify_latency
build/demo/shared_notify_latency 512
```

主指标包含 producer API、Atomic Cache 状态可见性、接收 Core 的 scheduler 轮询和
线程恢复；不包含 kernel 启动、初始化 barrier 和结果回传。为 TestCluster 的
`nd` 波形窗口定位时，可传入第二个参数让 core 0 在初始化后打印 marker 并延迟指定
周期；该延时不进入样本时间戳：

```sh
build/demo/shared_notify_latency 1 200000
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

默认 Host case 为 `M=768, N=512, K=1536`，与四 Core 默认 case 中每个 Core
承担的矩阵范围和 tile 数量完全一致，便于直接比较单核流水与跨核协作开销。它会生成
确定性 BF16 输入、执行 Device kernel、回传无 profile 扰动的 Device `mcycle`
吞吐，并逐元素和 CPU 参考结果比较：

```sh
build/demo/matmul_bf16_1c
build/demo/matmul_bf16_1c 384 256 512
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

四 Core 版本固定启动 4 个 Hart，并以 Cluster mode 提交 `task_dim={4,1,1}`：

```sh
make -C demo run-matmul-4c
bash demo/run_matmul_4c_cmodel.sh 1024 1536 1536
```

默认 case 为 `M=1024, N=1536, K=1536`：M 正好覆盖一个完整的 2-Core tile，
N 正好覆盖两个完整的 2-Core tile，K 包含 6 个分片，可以充分展开 slice 流水并
避免 M/N 边缘碎片。每个 Core 使用 `tileM=512`、`tileN=384`、
`tileK=256`；三个 M slice 分别为 170、171、171 行，并严格按线性顺序执行。
A 的每个 slice 使用单缓冲，B 保持两个 full-tile buffer 真双缓冲，三个 FP32 C
slice 在 K 循环内常驻，合计占用 1.375MiB URAM。M 维在两行 Core 之间连续等分，
每行再按 512 行推进；N 维按 384 列 tile 交错分给两列 Core。

每对 Core 只从 DRAM 读取一份 A 或 B tile：core 00/11 负责 A，core 01/10 负责
B。每个 Hart 创建一个 DRAM Loader、一个远端 Copy worker 和两个 MatStore
worker。DRAM Loader 发射本代首条 D2U 后通过本地 `dayan_resource_t` 允许 Copy
进入；源数据真正完成后再通过 Atomic Cache 中的 `dayan_shared_resource_t`
发布给远端。Copy 使用 D2U 从源 Core URAM 拉取已完成的数据，MAT 只消费本地
Resource。两个 MatStore 静态交替处理线性 job：前一条 MAT 成功入队后才用本地
Resource 向下一线程交棒，因此最多两条 MAT outstanding，但发射顺序不会反转。
整个数据面不使用自写 generation 原子轮询或动态抢任务。
对应日志为
`build/demo/matmul_4c_cmodel.log`、`build/demo/matmul_4c_dayan_server.log` 和
`build/demo/matmul_bf16_4c.log`。

## 四 Core D2A 性能模型

`matmul_bf16_4c_d2a_model` 是由同一份 4C 源码条件编译出的假想性能版本。未来 D2A
被建模为一次源端 D2U：该 DMA 完成时，数据被视为已经同时写入源端和远端 URAM。
远端 Copy worker 仍负责把 Atomic Cache 中的 shared resource 转换为本地 resource，
从而保留真实算子的线程调度、顺序发布和 buffer 生命周期；但它不发远端 D2U，也不执行
`dayan_wait_lsu()`。通知器直接依赖源端 shared resource，不再等待本核心 Loader 的
`load_issued`；后者是旧 Copy-DMA 的本地发射顺序约束，会在无 DMA 模型里造成跨核等待
环。A 仍按三个 170/171/171-row slice 逐片发布，MAT、C 写回和所有本地数据依赖保持不变。

当前硬件没有 D2A，远端 URAM 中不会获得正确数据，因此该目标只报告 Device `mcycle`
性能并明确跳过数值校验。默认尺寸同 4C 实算版本，为 `1024x1536x1536`：

```sh
build/demo/matmul_bf16_4c_d2a_model
build/demo/matmul_bf16_4c_d2a_model 768 512 256
```

## 四 Core 直接广播性能模型

`matmul_bf16_broadcast_4c` 建模更完整的未来广播指令：源端 Loader 的一次 D2U
同时填充本地和远端 URAM，完成后直接发布本地 Resource 和 Atomic Cache Shared
Resource。tile 固定为 `512x384x256`，A 沿 M 维等分为四个 `128x256` slice。
每个 Core 只创建一个 DRAM Loader 和两个有序 MatStore worker，不再创建 Copy worker；
源端 MAT 消费本地 Resource，远端 MAT 直接等待并消费 Shared Resource。B 的远端
Shared Resource 由四个 M slice 分别消费。模型保留真实的 Load/MAT/C 写回时序与同步
成本，但由于当前指令不会真实写入远端 URAM，仍跳过数值校验。

```sh
build/demo/matmul_bf16_broadcast_4c
```
