# Dayan kernel demos

该目录放置最小、可独立构建的 Dayan kernel 示例。目前包含 Device `printf`、
FP32 Add、FP16/BF16 FlashAttention、Gated DeltaNet、BF16/FP8/FP4 矩阵乘，
以及吞吐、延迟和调度探针。

## 目录

```text
demo/
├── Makefile
├── README.md
├── check_device_isa.sh          # 检查 Device 目标文件不含 LR/SC
├── uram_config.h               # 运行时 URAM 容量查询
├── print_demo.cpp              # Device printf 与完整 Host launcher
├── add_fp32_macro.cpp          # 最小 FP32 Add：LSU + MACRO
├── add_fp32_vec.cpp            # 最小 FP32 Add：LSU + VEC
├── flash_attention.cpp         # FP16/BF16、1c/4c、Split-KV 及 TGMT 变体
├── gdn.cpp                     # GDN Host 输入、启动与 CPU 参考校验
├── gdn_chunk.h                 # GDN TGMT / SW + Vec TGMT / SW 共用实现
├── matmul_bf16_1c.cpp           # 独立分片 BF16，支持 1c/4c Task group
├── matmul_bf16_4c.cpp           # 四核协作 BF16
├── matmul_fp8_4c.cpp            # 四核协作 E4M3
├── matmul_fp4_1c.cpp            # 单核打包 E2M1，按 URAM 容量选择 tile
├── matmul_fp4_4c.cpp            # 四核协作打包 E2M1
├── lsu_bandwidth.cpp           # LSU 带宽
├── embedding_dma_bandwidth.cpp # 单核、双线程、双指针缓冲随机 gather/scatter
├── vec_peak.cpp                # VEC 稳态吞吐
├── launch_latency.cpp          # 连续算子启动延迟
├── shared_notify_latency.cpp   # 跨核 Shared Resource 通知
├── scheduler_yield_bench.cpp   # 跨 Hart 调度通知
├── context_thread_probe.cpp    # Context/Thread 探针
├── task_grid_probe.cpp         # Task 网格探针
└── tensor_interrupt_probe.cpp  # 张量中断探针
```

`print_demo.cpp` 由 dycc 编译。dycc 在 Device 侧保留 kernel，在 Host 侧生成同名
wrapper 和 main；Host 代码只使用公开的 `npu_` Runtime API，不指定物理 Cluster 或
Core。

## 构建

```sh
make -C demo
```

产物位于 `build/demo/`。Makefile 显式列出普通示例、六个 FlashAttention 变体和
三个 GDN 变体；查看完整目标及执行 Host 启动参数检查：

```sh
make -C demo list
make -C demo check
```

`check` 先构建所有目标并检查 Device ISA，再执行各目标的 `--help`，无需启动 Server。
它不进行设备数值验证；算子运行时的 CPU 参考校验仍需连接 Server。
单独构建可指定 `../build/demo/<目标名>`，也可用 `BUILD=/绝对路径` 隔离输出。
每个目标的 Device 中间文件保存在其 `<目标名>-temps/` 目录中。
历史软流水 FlashAttention、软流水 Matmul 和广播性能模型在当前目录没有源码，
因此不再提供对应构建入口。

Device 自定义指令由 Clang 的
`<rvt_intrinsics.h>`（`__builtin_riscv_tx_*`）提供；`dayan_intrinsics.h` 包含该头文件，
并保留 LSU/MAT TCSR 编号和 `set_lsu_matrix` 等 Dayan 辅助接口。dycc 会启用 TX
扩展并把 Clang resource dir 加入 Device 编译，构建 demo 不需要 npu-kernels。

## FlexVec 固定控制流

`DIM0` 是最内层循环，`DIM1` 是外层循环；两者写入的值均为执行次数减一。
TCSR 地址仍是 `VEC_DIM0 = 2`、`VEC_DIM1 = 3`，迭代器参数按 Dim0、Dim1、
Dim2、Dim3 的顺序传递。Dim2..5 用于硬件任务划分。
五段 Control 边界编码仍为 `(prologue0, prologue1, loop, epilogue1, epilogue0)`。
每次 Dim0 循环结束，epilogue1 在 `(d0, d1) = (0, 当前外层索引)` 执行；
Dim1 循环结束后，epilogue0 在 `(0, 0)` 执行。

## 选择设备后端

demo 通过 `DAYAN_SERVER_SOCKET` 连接已经启动的 Server，不接收 DMA driver/profile 参数。
Server 的统一后端使用 `--backend xilinx-dma`，`--dma-driver xdma|qdma` 选择驱动，
`--dma-profile simulator|fpga` 选择默认目录与 MMIO 偏移。

使用 URAM 的算子在创建 Buffer、启动 kernel 前调用
`npu_runtime_get_uram_capacity()`。返回值是启用核心中最小的每核总容量，单位为字节，
不随正在运行的任务占用而变化。Server 默认配置为 1.5 MiB；768 KiB 核心启动时指定
`--uram-size 0xc0000`，demo 自动选择小容量入口及对应的资源描述符。
查询失败或容量不足时直接报错，不猜测容量。

| 算子 | 1.5 MiB 配置 | 768 KiB 配置 |
| --- | --- | --- |
| BF16 Matmul，1c / 4c | 512×384×256 | 192×384×256 |
| FP8 Matmul，4c | 512×384×512 | 192×384×512 |
| FP4 Matmul，4c | 512×512×640 | 192×512×640 |
| FlashAttention，所有入口 | Q/KV tile 128×128 | Q/KV tile 32×128 |
| LSU bandwidth | 最大 DMA 1 MiB | 最大 DMA 768 KiB |
| FP32 Add、VEC peak、Context probe | 32 KiB | 相同配置 |

小容量 Matmul 保留 N/K、slice 数和行跨度，缩小 A/C Buffer；BF16/FP8 两种
缓冲方向实际占用 672 / 768 KiB，FP4 占用 664 / 764 KiB，均声明 768 KiB。
按实际 Buffer 基地址重算 VisualSwizzle，所有 A/B/C 每拍 Bank 峰值仍为 1x，
重排地址不跨 Buffer 边界。容量选择不改变输入矩阵的尺寸和数值参考计算。

例如，在已启动双向 MM queue 0 的 QDMA FPGA 上：

```sh
build/host/driver/dayan_server \
  --firmware build/device/npu-os.elf --api-elf build/device/npu-os-api.elf \
  --backend xilinx-dma --dma-driver qdma --dma-profile fpga \
  --dma-device-name qdma01000 --dma-queue 0 --dma-user-irq 0 \
  --dma-completion irq --cluster-count 1 --core-mask 0xf
# 在另一个终端中执行：
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock build/demo/add_fp32_vec
```

IRQ 模式要求 QDMA 驱动提供扩展的 `user_irq0` 节点；没有该扩展时使用
`--dma-completion polling`（每轮请求休眠 1 微秒）。现有 `run_server.sh` 仍只检查
XDMA 节点，QDMA 使用上述直接命令。demo 静态链接 Runtime，Runtime 更新后需要重新链接。
设备端构建需要支持 Dayan 自定义指令的 LLVM 工具链，配置方式见
[`host/dycc/README.md`](../host/dycc/README.md)。

## 性能采集

普通算子只执行计算并校验结果，耗时和吞吐分析统一使用 `dyperf`。连接已启动的 Server
后，可直接由 `dyperf` 启动算子并采集时间线：

```sh
make -C host/dyperf
build/host/dyperf/dyperf -o matmul-trace.json -- \
  build/demo/matmul_bf16_1c
```

专门的延迟、带宽基准确实需要 Device 时间戳时，调用 `<dayan_sys.h>` 中的
`uint64_t dayan_get_time(void)`。它通过 `rdcycle` 返回当前 Hart 的周期计数，不做纳秒
换算，也不等待执行单元完成；跨核相减要求平台的计数器同源且对齐。
demo 不自行定义 `read_cycle` 或直接读取周期 CSR。

## 随机 Embedding DMA 带宽

`embedding_dma_bandwidth.cpp` 测试单核从大表随机读取、或向大表随机写入数据。
默认表大小为 **20,000,000,000 B**，依次遍历 **64B、128B、512B、2KB、4KB**，
每种粒度执行四种模式，共 **20 项测试**。所有粒度复用同一张大表和辅助 Buffer。

默认每遍工作集（仅数据，不含指针）为 **1,392,640 B = 1.328125 MB**，沿用原始
128 B 场景的 `64 * 200 * (1 - 15%) = 10,880` 项规模，通过改变 Entry 数保持一致：

| 访问粒度 | Entry 数 / 遍 | 数据字节数 / 遍 | 指针字节数 / 遍 |
| --- | ---: | ---: | ---: |
| 64B | 21,760 | 1,392,640 | 174,080 |
| 128B | 10,880 | 1,392,640 | 87,040 |
| 512B | 2,720 | 1,392,640 | 21,760 |
| 2KB | 680 | 1,392,640 | 5,440 |
| 4KB | 340 | 1,392,640 | 2,720 |

自定义工作集时，`Entry 数 = 工作集字节数 / 访问粒度`，四舍五入到整数项，
实际数据量与目标相差至多半个 Entry。工作集至少为所选最大粒度，每种粒度最多 1,048,576 项。
输入视为已去重，算子自身不做去重。

本次仿真采用 **DRAMSim3**，配置为
`/workspace/npu-cluster-sim/configs/lpddr5-6400-2ch-32g.ini`，即 2 通道、32GB 版本。

随机指针由 Host 按粒度 G 分别生成：以 `SEED` 初始化 `std::mt19937_64`，再通过
`std::uniform_int_distribution<uint64_t>` 在 `[0, floor(TABLE_BYTES / G) - 1]` 上抽取行号。
整张表中所有完整的 G 字节块均参与抽样，尾部不足 G 字节的区域不使用。
用 `unordered_set` 拒绝重复行号并重新抽取，直到得到当前粒度所需的不同地址，形成均匀无放回样本。
表基址按 4KB 对齐，地址为 `table_device_base + row_id * G`，保证按访问粒度对齐。
按抽中顺序保存，不按地址排序、不分组、
不采用固定地址步长，保留随机访问顺序。指针数组本身连续存储并由 DMA 分批读取，
随机性体现在它指向的 Embedding 行。

默认 `SEED=1`，相同标准库实现下可复现同一序列；修改 `SEED` 可获取另一组样本。
每种粒度有各自的样本，同一粒度的四个模式、预热及重复遍数复用同一组指针，便于比较不同 DMA 路径。

默认依次测试四种路径，也可以通过启动参数单独选择：

| 模式 | 数据路径 | 数据 DRAM 读 / 项 | 数据 DRAM 写 / 项 | 指针 DRAM 读 / 项 |
| --- | --- | ---: | ---: | ---: |
| `gather-uram` | 随机 DRAM → URAM，完成后丢弃 | G B | 0 B | 8 B |
| `scatter-uram` | URAM → 随机 DRAM | 0 B | G B | 8 B |
| `gather-dram` | 随机 DRAM → 连续 DRAM | G B | G B | 8 B |
| `scatter-dram` | 连续 DRAM → 随机 DRAM | G B | G B | 8 B |

两个线程各自拥有指针缓冲，独立执行：

```text
线程 0：DMA 获取前半段指针 → 等待 → 派发所选数据 DMA → 等待 → 下一批
线程 1：DMA 获取后半段指针 → 等待 → 派发所选数据 DMA → 等待 → 下一批
```

两线程不互相交接、不使用线程间 barrier 或 Named Resource。每个线程独占 384 KiB URAM
和两组逻辑 LSU TCSR，总共预留 768 KiB。它们使用同一核心的 LSU，不代表两个物理 DMA 引擎。
每批通过一条地址表 DMA 搬运多个 G 字节数据项，等待完成后才能复用指针缓冲。

- URAM 模式：每线程项数为 `floor(384 * 1024 / (G + 8))`，指针区向上对齐 64 B，
  后接数据区。D2U 将数据读入该缓冲，下一批覆盖；U2D 重复使用该缓冲的数据。
- D2D 模式：每线程全部 384 KiB 用作指针缓冲，容纳 49,152 项，不分配数据 URAM 缓冲。

| 粒度 | URAM 项数 / 线程 / 批 | 指针区 B / 线程 | 数据区 B / 线程 |
| --- | ---: | ---: | ---: |
| 64B | 5,461 | 43,712 | 349,504 |
| 128B | 2,891 | 23,168 | 370,048 |
| 512B | 756 | 6,080 | 387,072 |
| 2KB | 191 | 1,536 | 391,168 |
| 4KB | 95 | 768 | 389,120 |

默认工作集下，各粒度每遍 URAM 模式均提交 4 批数据 DMA，D2D 模式均提交 2 批。
每一遍仍重新 DMA 获取指针，指针开销没有移到计时外。
指针尾块使用短 DMA，DRAM 端不越过有效长度，不要求调用方填充指针数组。

Device 入口为：

```cpp
embedding_dma_bandwidth(const uint64_t *pointers, uint64_t count,
                        uint8_t *packed, uint64_t mode,
                        uint64_t iterations, uint64_t entry_bytes,
                        EmbeddingDmaResult *result);
```

- `pointers`：设备内存中的 64-bit 绝对 D-side 地址数组，数组至少 8 B 对齐。
  每项指向可访问的 `entry_bytes` 字节数据，所有被间接引用的 Buffer 必须保留在同一个 Context 内。
- `entry_bytes`：64、128、512、2048 或 4096，控制 DMA 单项长度、地址表步长和 URAM 分块。
- `mode=0`：D2D Gather；`mode=1`：D2D Scatter；`mode=2`：D2U Gather；`mode=3`：U2D Scatter。
- `packed`：D2D 模式使用至少 `count * entry_bytes` B、64 B 对齐的连续设备区域。
  `mode=0` 将 `pointers[i]` gather 到 `packed + i*entry_bytes`；`mode=1` 反向 scatter。
  `mode=2` 忽略该参数，Host wrapper 传入有效的结果 Buffer 句柄即可。
  `mode=3` 在启动两个线程前，用 DMA 从 packed 的各半段起点预装一批数据到对应 URAM，
  完成后才开始计时；每线程后续各批重复使用这批源数据，尾批只使用有效项。
  Host 生成相同周期的 FP32 位模式作参考，不在设备端生成测试数据。
  除被忽略的参数外，packed、指针数组、结果区和随机表互不重叠；写入目标也不得重叠。
- `iterations`：同一指针序列的重复遍数，必须大于零；`count=0` 在 Device 入口是空操作。
  默认重复 10 遍，用于降低短测量波动；它不是每遍重新生成随机地址。
- `result`：总周期、访问项数、访问粒度以及各线程完成项数。两个线程的时间戳在同一 Hart，
  周期取最早开始至最晚完成，包含指针 DMA、数据 DMA、等待和调度间隙。

一键构建并运行独立仿真环境：

```sh
./run_test.sh embedding_dma_bandwidth
./run_test.sh embedding_dma_bandwidth --granularity 4KB
```

该入口自动创建独立设备目录、socket 和模拟器工作目录，使用 DRAMSim3 的 2ch、32GB 配置，
运行结束后清理进程和挂载。报告和日志保存于打印的 `build/test-runs/` 子目录。

也可单独构建并连接已有 Server：

```sh
make -C demo ../build/demo/embedding_dma_bandwidth
build/demo/embedding_dma_bandwidth
# 默认：all 20000000000 10880 10 1000000000 1 --granularity all
# 位置参数：[MODE [TABLE_BYTES [COUNT128 [ITERATIONS [CLOCK_HZ [SEED]]]]]]
# 只跑某个粒度的四种模式：
build/demo/embedding_dma_bandwidth --granularity 4KB
# 各粒度使用 2 MB 目标工作集，仍执行全部四种模式：
build/demo/embedding_dma_bandwidth --working-set 2097152
# 只跑各粒度的 D2U Gather，目标工作集为 10000 * 128 B：
build/demo/embedding_dma_bandwidth gather-uram 20000000000 10000 10
# 复现原 128 B 的模式选择及 Entry 数参数：
build/demo/embedding_dma_bandwidth both 20000000000 10880 10 --granularity 128
# 指定 Markdown 报告位置（默认写当前目录的 embedding_dma_bandwidth.md）：
build/demo/embedding_dma_bandwidth --report demo/embedding_dma_bandwidth.md
```

`--granularity` 支持 `all`、字节数 `64/128/512/2048/4096` 或名称 `64B/128B/512B/2KB/4KB`。
位置参数 `COUNT128` 保持原来的 128 B 参考项数含义，目标工作集为 `COUNT128 * 128`；
指定 `--working-set BYTES` 时改用显式字节数。`read/write/both` 仍选择原 D2D 模式，
默认对所有粒度生效。`--verify-uram` 保留独立、不计时的 D2U 回读验证。

Host 通过 `npu_buffer_get_device_address()` 获取表的设备可见基址，生成绝对地址数组，
用 H2D 上传指针数组及被访问行的初始数据，无需创建或传输 20 GB 的 Host 镜像。
设备端只有带宽测试 kernel；没有数据准备 kernel 或标量核读写 URAM 的初始化循环。
开启 MMU 时地址查询返回 Buffer 的虚拟地址，Host 不假设物理地址或固定表基址。
使用新增地址查询接口需重启更新后的 `dayan_server`；旧 Server 会返回明确的升级提示。
每个方向先预热一遍，再执行计时启动；准备、Host 传输、预热和结果校验均在计时外。
写测试使用与初值不同的数据；D2D Gather 和两个 Scatter 模式读回全部数据做 FP32 位模式
比较，并直接按 Host offset 抽查 17 个表项，检查地址转换。Scatter 的验证 Gather 不计时。
默认 D2U 只核对完成状态和项数，**丢弃数据且不回写 DRAM**；若需数值验证，显式添加
`--verify-uram`，另起一次不计时的 D2U/U2D 回读启动，逐项比较全部数据。
测试使用独立分配的表，写模式会覆盖其中被选中的行。

成功运行后，终端和 Markdown 文件输出工作集配置表及两张带宽表，默认带宽表各有 20 行：

- 工作集表：访问粒度、Entry 数、每遍实际数据字节数、指针字节数和 URAM 批容量。
- 数据表：访问粒度、模式、总周期、`Payload B/cycle`、`Payload MB/s`、
  `Total Read MB/s`、`Total Write MB/s`、`Total Utilization` 和 `Data Utilization`。
  Total 包含数据及指针流量；Payload 和 Data Utilization 只包含数据。
  有效载荷为 `count * entry_bytes * iterations`；D2D 的 DRAM 数据流量为读、写各一份。
- 指针表：访问粒度、模式、指针总字节数 `count * 8 * iterations`、指针 B/cycle、指针 MB/s、
  指针带宽利用率，作为 Total 指标中指针部分的独立明细。

采用 **1 MB = 1024 KB = 1024 * 1024 B**。双通道 LPDDR5-6400 的基准保持为
**51,200,000,000 B/s（原 51.2 GB/s）**，换算为 **48,828.125 MB/s**：

```text
Total Read = 数据读取 MB/s + 指针读取 MB/s
Total Write = 数据写入 MB/s（指针无写入流量）
Total Utilization = (Total Read + Total Write) / 48828.125 * 100%
Data Utilization = (数据读取 MB/s + 数据写入 MB/s) / 48828.125 * 100%
Pointer Utilization = 指针读取 MB/s / 48828.125 * 100%
```

所有指标共享同一测量周期，包含指针 DMA、数据 DMA、等待和调度。
准备、预热、U2D 源预装、结果写回和校验不计时。D2D 的 Data Utilization 包含数据的读、写两份流量。

默认按 **1 GHz** 折算：`MB/s = (B/cycle) * CLOCK_HZ / 1048576`。
可通过 `CLOCK_HZ` 修改频率，设为 0 则省略 MB/s 和利用率。
默认报告文件为当前目录的 `embedding_dma_bandwidth.md`，使用 `--report FILE` 可改路径；
每次成功运行会覆盖指定报告。报告同时记录 DRAMSim3 配置和指针生成方法。

默认 20 GB 表还需指针、packed Buffer 和运行时内存；例如 cmodel 可配置 24 GiB：

```sh
NPU_CMODEL_HARTS=1 NPU_CMODEL_MEMORY_MIB=24576 ./run_cmodel_server.sh
# 另一终端：
build/demo/embedding_dma_bandwidth
```

如果报 `allocation failed: embedding table ... status=-12`，这是设备 DRAM 分配失败。
`URAM capacity per Core: 768 KiB` 是片上容量，两者分别管理。
XDMA Server 不指定 `--normal-memory-size-mib` 时，Normal 内存池默认只有 **4 GiB**。
在后端确实提供 24 GiB 地址空间时，可在原 Server 启动参数后增加
`--normal-memory-size-mib 24576`，例如：

```sh
./run_server.sh --normal-memory-size-mib 24576
```

该参数只改变分配器的内存范围，不能扩展真实硬件或 RTL 后端的 DRAM。
后端容量不足时，显式使用较小的 `TABLE_BYTES` 做功能验证；算子不会自动缩小表后
继续把结果标成 20 GB 场景。失败信息会注明步骤、Buffer 名称和申请字节数。

## 连续算子启动性能

`launch_latency` 默认连续同步启动 1000 次，每次四核各执行一个固定 450000 轮的标量循环。
循环计数保留在寄存器中，编译器不能将循环折叠。支持单核模式，不使用张量计算单元或子线程。
Runtime、Context、Stream 和结果 buffer 均复用；前 5 次启动作为预热，首次冷启动耗时单独报告。
每次 kernel 只额外记录循环前后的 cycle、完成轮数和启动编号，并写回一个独占 cache line。
Host 在全部启动结束后一次性读回和校验结果，启动循环中没有结果回读或逐次打印。

```sh
make -C demo ../build/demo/launch_latency
# 已有 Server 使用原来的 IRQ 配置，设备实际频率为 10 MHz：
build/demo/launch_latency 1000 4 10000000
# 单核启动：
build/demo/launch_latency 1000 1 10000000
# 可选采集，按设备实际频率换算 trace：
build/host/dyperf/dyperf --clock-hz 10000000 -o launch-trace.json -- \
  build/demo/launch_latency 1000 4 10000000
```

参数顺序为 `[LAUNCHES [CORES [CLOCK_HZ]]]`，默认 `1000 4 0`。
`CLOCK_HZ=0` 只输出原始 device cycles，不假设设备运行在 1 GHz。启动次数上限为 100000。
报告包含 min、mean、p50、p95、p99、max：

- `host_call`：一次同步 wrapper 调用的耗时，包括提交、设备执行、完成通知和 event 销毁，不能当作纯启动延迟。
- `host_between_calls`：上一调用返回到下一调用开始之间的 Host 时间。
- `device_loop`：单个 Hart 执行 450000 轮循环的 cycle 数，不包含结果写回。
- `device_gap_per_lane`：同一 lane 上一次循环结束到下一次循环开始的 cycle 数，包含结果写回、退出、Host 往返、下一次启动及其他核未完成造成的等待，不等于纯 IRQ 延迟。

测试要求只启用一个 Cluster。四核模式按同一物理 Hart 的相邻样本计算间隔，不进行跨核时间戳相减。
单核任务可能迁移到其他 Hart；仅当 Server core mask 只有一位时输出 device gap，否则只输出 Host 和循环统计。
性能采集可在启动 Server 时设置 `DAYAN_RPC_DEBUG=0`，避免额外 RPC 日志；demo 不改变 Server 的 IRQ/polling 模式。
Dayan-emu 用于验证启动次数和循环结果，性能结论应使用实机或 RTL 数据。

## 最小 FP32 Add

两个示例都计算固定的 `A[64] + B[64] -> C[64]`，输入输出均为 FP32。
Device 部分只有一个 kernel，不定义或调用自定义 Device 子函数，也不创建
子 Thread。LSU 配置、计算单元配置以及 `load -> calc -> store` 全部在
kernel 体内直接展开：

- `add_fp32_macro.cpp` 使用 `__tx_macro_elemwise_add_fp32`。
- `add_fp32_vec.cpp` 使用 4 条 Vec 队列指令：读 A、读 B、FP32 Add、写回 URAM。

可以分别构建：

```sh
make -C demo ../build/demo/add_fp32_macro
make -C demo ../build/demo/add_fp32_vec
```

两个 Host 程序默认连接
`/tmp/dayan-server.sock`，也可以通过环境变量覆盖：

```sh
DAYAN_SERVER_SOCKET=/tmp/my-dayan.sock \
  build/demo/print_demo 1 2 3 4 5 6 7
```

## FP16/BF16 FlashAttention 优化流水线

`flash_attention.cpp` 将 llama.cpp 的
`ggml/src/ggml-dayan/ggml-dayan-flash-attention.h` 移植为单文件 demo，
包含配置、Device kernel、Host launcher 和 CPU 参考校验，不依赖 llama.cpp 头文件。
同一模块支持单核和四核，默认使用单核 FP16；`--cores 1 --bf16` 选择单核 BF16。
单核入口实例化 `run<Format, 1, false>`，使用 30 个 Local Resource、0 个 Shared
Resource，不执行跨核 barrier 或远端 K/V 复制。单核指每个 Task 使用一颗 Core，
不绑定物理 Core 0；`--tasks` 可配置多个独立单核 Task。Split-KV 仅支持四核。

1.5 MiB 配置下，每个 Core 使用两个 128-row Q slice，共享双缓冲的 128-column K/V tile。
root 执行 QK，六个子 Thread 分别负责 Q/K/V 加载、softmax、PV 和输出更新，
通过 30 个 Local Resource 管理缓冲区和状态版本。score 原地转为 FP32 P，
PV 使用 `TF32 × FP16/BF16 -> FP32`，省去单独的 P16 缓冲区和格式转换。
alpha、M/L、PV 存在 URAM 中供不同阶段或下一轮使用；O 在最后一块更新时归一化。
softmax 和 output 各用一组单消费者 State Resource；复用 M/L/alpha 存储前，
两组都必须 FREE。softmax 自己消费末 tile 的 State，每个 Resource 都逐代消费，
不再与 output 共用消费计数或跳过尾代。Split-KV 的 output State 保留到合并结束。
该配置每核申请 1536 KiB URAM，布局实际占用 1489 KiB。768 KiB 配置将 Q slice
缩到 32 行，布局实际占用 565 KiB；KV 仍为 128 列，softmax 的 VEC 指令序列不变。

四核 prefill 用四个 Shared Resource 选出 K/V loader，其余 Core 从 loader 的
URAM 复制；Shared Resource 的消费完成后才允许复用源缓冲区。
四核 decode 按 KV 维度切分工作，把同一 KV head 对应的 Q heads 打包为 MAT 行，
最后在 lane 0 根据各核 M/L 合并输出。跨核 barrier 保护归并期间的远端数据。

Q/K/V 接收 FP16 或 BF16 的 `[batch, head, row, D]`，支持 64B 对齐的行、head、
batch stride；V 转置在 LSU 内完成。输出为连续的 FP32 `[batch, row, q_head, D]`。
Q 不需要预先缩放，配置中的 `scale_bits` 是普通 attention scale，demo 使用
`1/sqrt(D)`，Device 再乘 `log2(e)`。支持 `D=64/128/192/256`、GQA、
因果掩码和 Q/KV 尾块。因果条件为 `key <= causal_offset + query_row`。

使用 VisualSwizzle 对实际基址、stride 和 32-bank 配置重新检查后，保留源实现的映射：

| MAT 数据 | 字节 stride | M/B/S | 线性 / 采用映射的 Bank 峰值 |
| --- | --- | --- | --- |
| QK A/B，D=64/192 | 128 / 384 | identity | 1 / 1 |
| QK A/B，D=128 | 256 | 5/1/5 | 2 / 1 |
| QK A/B，D=256 | 512 | 5/2/5 | 4 / 1 |
| PV B，即 Vt | 256 | 5/1/5 | 2 / 1 |
| QK C / PV A，即 score/P | 512 | identity | 4 / 4 |
| PV C，D=64/128/192/256 | 256/512/768/1024 | identity | 2/4/2/8，保持不变 |

工具按四块 Bank 覆盖率进一步排序时，256B/512B stride 的候选最优分别为
`7/1/3`、`7/2/3`；这里保留源实现的更小重排窗口，输入的每拍峰值同样为 1。
score/P 和 PV 的消费者包括没有 swizzle iterator 的 FlexVec，因此保持 identity。
这些是地址模型结果；吞吐和调度块周期需要实机或 RTL 测量。

```sh
make -C demo ../build/demo/flash_attention
build/demo/flash_attention --help

# 使用已启动的 Dayan Server；默认 FP16、单核、B=1、Hq=4、Hkv=2、Q=256、KV=512、D=128
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock build/demo/flash_attention

# 单核 BF16
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock build/demo/flash_attention --cores 1 --bf16

# 四核 prefill，因果掩码、Q/KV 尾块、带 padding 的 stride，以及两次重复 launch
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock build/demo/flash_attention \
  --bf16 --cores 4 --q-len 529 --kv-len 641 --causal --padded --repeat 2

# 四核 decode；Q=1 且 KV>=512 时自动使用 Split-KV
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock build/demo/flash_attention \
  --cores 4 --q-len 1 --kv-len 769 --q-heads 16 --kv-heads 2
```

`--split-kv` 可显式选择 decode 路径，要求四核、Q=1、至少四个可见 key。
`--causal-offset` 指定因果偏移；只给 `--causal` 时默认为 `KV-Q`。
`--tasks` 在单核模式分配独立 Task，四核模式分配独立 Cluster 工作组。
Host 使用确定性的可精确表示输入，按输入数据计算双精度完整 attention 参考值，
逐元素检查输出、每个 Task 的返回状态及输出尾部 guard。
报告最大绝对/相对误差、RMSE 和 mismatch 数；允许误差为
`3e-2 + 3e-2 * abs(reference)`，使用放宽阈值做 Resource 稳定性复测，日志同时打印
`atol`/`rtol`。NaN/Inf、超过阈值的误差、Task 失败及 guard 损坏仍报错。
`--repeat` 每轮重置输出和状态再验证。

### FlashAttention 构建变体

六个目标均来自 `flash_attention.cpp`，通过独立的编译宏和中间文件目录选择实现：

| 可执行文件 | 行为 |
| --- | --- |
| `flash_attention` | 1c/4c 完整流水线，保留 Split-KV decode |
| `flash_attention_tgmt_barrier` | 独立 Q group 分工，各核加载自己的 K/V，核间仅首尾 barrier |
| `flash_attention_tgmt_shared_4c` | 四核共享 K/V，合并 K/V loader，以 barrier 协调 peer copy |
| `flash_attention_tgmt_resource_4c` | 四核共享 K/V，以 Shared Resource 协调独立 K/V loader |
| `flash_attention_tgmt_resource_timed_4c` | Resource 版本，增加每核启动窗口与计算窗口计时 |
| `flash_attention_tgmt_resource_opt_4c` | 带计时的 Resource 版本，缓存 peer LSU 配置并合并输出 drain |

后三个 Resource 目标仅保留四核 prefill，关闭 Split-KV。独立 K、V loader
分别协调两个 ping-pong slot，共 4 个 Shared Resource、30 个 Local Resource、
6 个子线程。producer 在 LSU 完成后发布共享 token，peer 在 URAM 复制完成后
消费 token；下一次写入同时受本地 MAT 读者和远端复制完成约束。
K/V 循环内无 barrier，仅保留首尾初始化和汇合 barrier。

共享 K/V 的 barrier 版本将独立 K/V loader 合并为一个线程。每轮选一个有有效
Q 行的 Core 作为 DRAM owner，其他 Core 从 owner 的 URAM 复制 K 和转置后的 V。
两次 barrier 分别确认 owner 加载和 peer 复制完成，保留双缓冲及下一 tile 预取。

```sh
make -C demo ../build/demo/flash_attention_tgmt_resource_4c
# DAYAN_SERVER_SOCKET 指向启用四核的 Server。
build/demo/flash_attention_tgmt_resource_4c --bf16 --cores 4 --tasks 1 \
  --q-len 1024 --kv-len 512 --head-dim 128 --q-heads 1 --kv-heads 1
```

两个计时目标在首个 QK MAT 提交前记录时间，结束点在输出写回、线程汇合和末尾
barrier 完成之后。Host 分别对四核的 startup/post_startup/total 取最大值；
各列最大值可能来自不同 Core，不能将最大 startup 与最大 post_startup 相加。
post_startup 包含剩余加载、计算和排空，完整 launch 周期仍用于端到端比较。
优化版本通过 `FLASH_ATTENTION_OPT=1` 缓存 peer LSU 配置、将 shared_keys 提到
Q group 外层，并将两个 O slice 的 LSU drain 合并到 Q group 边界。
Resource 生命周期及数据共享方式保持一致。

## LSU 单核带宽微基准

`lsu_bandwidth` 隔离测试一个 Core 发射的连续 D2U，以及从本 Core URAM 到相邻 Core
URAM 地址窗的 U2D。四个 Core 仅用于建立 Cluster URAM 映射；测量区间内只有 lane 0
发射 DMA。每个尺寸都严格采用“发射一条 DMA、立即 `dayan_wait_lsu()`”的顺序，并用
`dayan_get_time()` 记录 4 KiB、16 KiB、64 KiB、256 KiB 和 1 MiB 的端到端周期；
768 KiB 核心的最后一档改为 768 KiB，输入 Buffer 同步缩小。测试结束
后另行回读 64 KiB 做数据校验，回读不计入带宽：

```sh
build/demo/lsu_bandwidth       # 依次测试 D2U 和远端 U2D
build/demo/lsu_bandwidth d2u
build/demo/lsu_bandwidth u2d
```

## VEC 吞吐微基准

`vec_peak` 测量 FlexVec 的稳态吞吐，并与设计峰值 128 FLOP/cycle（每周期 2 条 64-lane
ALU 指令，1 lane-op 记 1 FLOP）对比。每个配置是一个只使用寄存器的 VFork：prologue 用
30 条 broadcast 填满两个寄存器队列，循环体是 `BODY` 条 ALU 指令，由硬件循环
`(DIM1 + 1) × (DIM0 + 1)` 在 `TASKS`（`DIM2 + 1`）个 task 中重放，epilogue 每个 task 写回一个 256B
槽位。循环体不访问 URAM（`ld-*` 模式除外），因此结果只反映发射、寄存器堆和 ALU。

每个配置运行两次，仅外层循环次数不同（`DIM1 + 1` = 1 和 2；`flat` 为 VFork 个数），并用
`dayan_get_time()` 记录“发射指令块 → VFork → `dayan_wait_vec()`”的周期。Host 报告两次的
差值：

```text
ALU/cycle  = 长短两次相差的 ALU 指令数 / (long - short)
FLOP/cycle = ALU/cycle × 64        peak = FLOP/cycle / 128
```

指令块发射、VFork、prologue、task 的爬升与排空以及单元等待在两次运行中各出现一次，相减后
抵消，剩下的是稳态吞吐。测量开始前先执行一个不计时的小 VFork 吸收冷启动开销。操作数
取值保证每个 task 的最终结果是已知常数（`x+0`、`x*1`、`recip(1)` 恒为 1.0，`0+0`、
`0*0` 恒为 0.0）；kernel 在每次运行前用 poison 覆盖结果槽位，结束后检查每个 task 的
lane 0 和 lane 63。被静默丢弃的 VFork 会报告为 `BAD`，而不是一个快得不合理的周期数。

```sh
make -C demo ../build/demo/vec_peak
build/demo/vec_peak                     # 默认 summary sweep
build/demo/vec_peak threads mix         # 指定 sweep，一次 launch 内顺序执行
build/demo/vec_peak am-1r:16:15         # 单个配置：MODE[:TASKS[:DISTANCE[:BODY]]]
build/demo/vec_peak --cores 4 mix       # 四核：每个 Core 在自己的 VEC 上各跑一遍
build/demo/vec_peak --work 1024 all     # Dayan-emu 功能验证时缩小每次运行的规模
```

`--help` 列出全部 sweep 和模式。sweep 包括 `summary`（默认，各结论的代表点）、
`threads`（task 数 1..64）、`depend`（依赖距离与延迟）、`ops`（add/mul/sfu/load）、
`body`（循环体长度）、`flat`（不使用硬件循环，标量核逐块发射）、`mix`（add 与 mul
交错）、`reads`（add+mul 的操作数模式）和 `all`。`DISTANCE` 是以 push 计的依赖距离：
1 为串行链，15 是 16 项队列允许的最大独立度；取值限 1、2、4、8、15，因为寄存器 age
是指令的立即数字段，kernel 为每个取值各实例化一份发射代码。`--work N` 是长短两次运行
相差的循环体指令总数，默认 16384。

`--cores 4` 以 `NPUTaskMode::kFourCore`、`task_dim={4,1,1}` 启动，要求 Server 启用同一
Cluster 的 4 个 Core。task X 在 Core X 上执行整张配置表并写入 `results[X]`，Host 逐核
输出结果，并按“各核 ALU 指令之和 / 最慢一核的周期”给出 Cluster 吞吐，对比 4 × 128。
各核的 `sweep cycles` 窗口用于确认四份 sweep 确实同时运行；跨核比较要求计数器同源。

Device 部分只有一个 kernel，不创建子 Thread，使用 1 个 LSU 和 1 个 VEC TCSR 组、
32 KiB URAM（16 KiB 的 1.0f load 源和 16 KiB 结果槽位）。descriptor 为
`DAYAN_KERNEL_TASK_DESC(vec_peak, 1, ...)`：同一个 kernel 还要以四核模式启动，因此不能
声明为 locked Task，测量时 Server 上不应同时运行其它算子。Vec 指令队列在 Proc 退出后
仍然存在，kernel 退出前比较队列空闲项数，有指令未被 VFork 时以 `-5` 结束。

周期数只在 RTL 或实机上有意义；Dayan-emu 没有时序模型，长短两次的差值可能为负，此时该行
输出 `timing unusable`，PASS 只表示数据校验通过。使用 TestCluster 时按
[`Readme_cluster.md`](../Readme_cluster.md) 的方式运行，例如：

```sh
python3 ./scripts/run_testcluster_operator.py \
  --repo . --sim-dir "$NPU_TC_ROOT" --sim-bin "$NPU_TC_BIN" \
  --dram-config "$NPU_TC_DRAM" --startup-timeout 300 \
  --server-timeout-ms 36000000 --core-mask 0x1 --expect '[host] PASS:' \
  -- ./build/demo/vec_peak
```

四核 sweep 使用 `--core-mask 0xf` 并在 `vec_peak` 后追加 `--cores 4`。单个配置适合配合
`dyperf` 观察 Proc 状态时间线，trace 中只有该配置的一短一长两次运行：

```sh
build/host/dyperf/dyperf -o vec-peak-trace.json -- \
  build/demo/vec_peak am-1r:16:15
```

## 用 dyperf 测量跨 Hart 调度通知

`scheduler_yield_bench` 使用四核 Task：Hart0 在 Shared Resource 0 上 wait-free，
返回后立即 produce；Hart1 wait-ready 后 consume，允许下一轮发送。Hart2/3 在独立的
Shared Resource 1 上等待结束，测量期间不忙等。默认预热 64 次，随后执行 128 次通知。
测试内部不读取周期、不计算时间，只回传各 Hart 的完成次数，结果按缓存行隔离。

连接已启动的四核 Server 后运行：

```sh
make -C demo ../build/demo/scheduler_yield_bench
build/host/dyperf/dyperf --buffer-size 256K -o perf.json -- \
  build/demo/scheduler_yield_bench 128
python3 scripts/analyze_scheduler_notify.py perf.json --iterations 128
```

分析脚本从 `dyperf` 的原始 counter 配对每轮通知，丢弃预热并检查完整的等待次数。
主指标是 Hart0 的 wait-free 被调度为 RUNNING，到 Hart1 的 wait-ready 被调度为
RUNNING 的间隔；另报 Hart1 等待区间和 Hart0 相邻通知的往返间隔。该指标包含发送侧
恢复、Produce API 和 profile 开销，不等于 Produce 指令到接收函数返回的精确延迟。
RUNNING 在上下文恢复前记录。跨 Hart 相减要求各 Hart 的 cycle counter 同源且对齐；
`dyperf` 时间轴上的 `1 cycle = 1 ns` 是显示约定，统计结果以周期为单位。

## 跨 Core 通知延迟微基准

`shared_notify_latency` 隔离测量硬件 Shared Resource 的 core 0 -> core 1 通知路径。
core 1 先通过逻辑 Shared Resource 0 发布 armed 握手，保证自己已经进入接收流程；
core 0 随后在 `dayan_resource_produce_lsu()` 前调用 `dayan_get_time()`，core 1 在
`dayan_resource_wait()` 返回后立即调用同一接口记录周期。此外 core 1 会立刻
回 ACK，得到一个不依赖跨 Core 计数器对齐的 round-trip 对照。默认预热 16 次并采集
64 个样本，可指定不超过 512 的样本数：

```sh
build/demo/shared_notify_latency
build/demo/shared_notify_latency 512
```

core 2/3 完成初始化 barrier 后直接退出，Host 等四个 Proc 全部退出后才完成 launch。
core 0 在最后一次 ACK 消费后等待 ACK 变为 FREE，确认 Shared Resource 更新已经提交；
这个等待会将 ACK 预留为 BUSY，留给 Runtime 在下次分配时重新初始化。
该收尾不计入样本时间戳，也不依赖结束 barrier。

主指标包含 producer API、Cluster 级 Shared Resource 仲裁和接收 Thread 恢复；不包含
kernel 启动、初始化 barrier 和结果回传。为 TestCluster 的
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

Host 按转置布局提供 BT，三个 stride 均以字节为单位。核内流水沿用
`matmul_bf16_4c` 的 tile（1.5 MiB 配置为 `512×384×256`）、4 个 slice、两个 Loader Thread、
根 Calc Proc 和一个 Post Thread，按输出 tile → K tile → slice 的顺序推进。
`M<N` 时 A 双缓冲，BT 分为四份 `96×256`；否则 BT 双缓冲，A 分为四份
`128×256`。每个 slice 保留独立 C，直到全部 K 累加完成，再由 Post 写回。

每个 Task 自己从 DRAM 加载 A/BT，只用 10 个 Local Resource 管理核内依赖，
不做跨核复制、Shared Resource 选举或 Cluster barrier，也不申请 atomic/shared
存储。1.5 MiB 配置的两种布局分别使用 1,472 KiB、1,408 KiB URAM，统一申请 1,536 KiB，
每 Core 同时驻留一个 Task。

默认 `M=1024, N=1536, K=1536, TASKS=1`，启动单 Core Task。可选第四个参数
指定独立 Task 数，始终使用 `kSingleCore` 模式：输出 tile 按 M/N 行优先展开，
Task `t` 处理 `t, t+TASKS, ...`，各自完成整个 K 循环。Runtime 将 Task 调度到
可用 Core；Task 数多于输出 tile 时，多余 Task 直接返回。核间不共享 URAM 输入
切片，也不做部分和归约。程序生成确定性 BF16 输入并逐元素比较 CPU 参考结果：

```sh
make -C demo ../build/demo/matmul_bf16_1c
build/demo/matmul_bf16_1c
build/demo/matmul_bf16_1c 128 5088 1272
# 连接启用了多个 Core 的 Server，启动四个独立分片
build/demo/matmul_bf16_1c 1024 1536 1536 4
```

`N` 必须是 16 的整数倍；K 尾块通过实际 `MAT_K` 和窄 DMA 请求处理，不要求是 256
的整数倍。M/N 尾块只计算有效行列，Resource 消费计数按有效 slice 数设置。

VisualSwizzle 按实际类型、尺寸、URAM 基地址和 32 个 Bank 搜索：A/BT 的 512B
行跨度与 C 的 1536B 行跨度使用 `M=7,B=2,S=3`，每拍 Bank 峰值由 4x 降至 1x，
连续四块覆盖从 8/32 提升到 32/32；C 的 384B 行跨度在线性布局下已为 1x、32/32，
使用 identity `M=5,B=0,S=1`。每个 buffer 的 LSU/MAT 映射一致，重排不跨分配边界。
这些是布局模型的冲突指标，不是 RTL 性能测量。

## 四 Core BF16 matmul

`matmul_bf16_4c` 默认计算 `M=1024, N=1536, K=1536`，1.5 MiB 配置下每核使用
`512×384×256` tile，分成 4 个 slice。`M<N` 时 A 双缓冲供 4 个 B slice 共用，
B 每片为 `96×256 BF16`；反之 B 双缓冲供 4 个 A slice 共用，A 每片为
`128×256 BF16`。切片输入和 C 都是每 slice 单缓冲。C 每份为 `512×96` 或
`128×384` 个 FP32 元素，占用 192 KiB，4 份共 768 KiB。

A 共享路径实际使用 1,472 KiB URAM，B 共享路径实际使用 1,408 KiB。两种布局
统一声明 1,536 KiB URAM、10 个 Local Resource，四核共享 12 个 Shared Resource。
共享输入按连续 K 代次选择双缓冲，每个输出 slice 的所有 K 步使用固定的 C 缓冲；
Post 写回并释放后，该缓冲才能用于下一输出 tile。

Loader、MAT 和 Post 恢复统一的 `MAT_BATCH_SWIZZLE(5,3,5)`，跨核复制仍按原始
物理字节搬运。

每核运行两个 Loader、根 Calc Proc 和一个 Post Thread。Local Resource 管理各阶段
缓冲，Calc 与 Post 分别维护逐资源的期望 Phase，使用 `dayan_resource_wait()`
显式指定 Phase 等待输入。MAT/LSU 工作与资源 Consume 提交后，循环立即进入下一轮；
仅在线程末尾执行一次单元完成等待。

Local 的期望 Phase 从运行时初始化后的 0 开始：切片输入每次使用后推进，供多个有效切片共用的
输入在最后一个有效切片提交后推进，跳过的空切片不改变 Phase。Post 每次提交 C 的
存储和 Consume 后推进该 C 资源的期望 Phase。资源 Phase 不会在跨 C tile 时重置，
硬件 Phase 仅在最后一份消费计数通过 Produce 提交后翻转。

本地 D2U 完成即可发布本地输入 READY，Local 消费计数只包含本地 MAT 使用次数。
Shared READY 只有一个远端复制消费者：对端 LSU 完成复制后，在同一次 Produce 中
发布目标 Local READY 并消费 Shared。Loader 通过一次 Wait 同时预留 Local FREE
和参与指定 Phase 的 Shared 选举，确保本地 MAT 与上一代远端复制均完成后才能复用
源缓冲。发出 Produce 后即处理下一份输入，不再单独等待复制确认。

Shared 的期望 Phase 从 0 开始，按每个共享槽的实际使用推进。尾块只有一侧 Core 有输出时，
无输出的一侧仍等待并消费 Shared，但不预留 Local 或发出 DMA；两侧都无输出时才共同跳过。
这样从配对加载进入单侧尾块、再回到配对加载时，仍受同一套 Shared 生命周期保护。
Post 提交 U2D 和异步 Consume 后继续处理下一个 slice；Calc 在下一输出 tile 的
第一个 K 步等待对应 C slice 释放。

```sh
make -C demo ../build/demo/matmul_bf16_4c
# 连接已启动的四核 Server；非默认 socket 可设置 DAYAN_SERVER_SOCKET。
build/demo/matmul_bf16_4c
```

可用 `./run_dayan_emu_server.sh` 启动四核 dayan_emu Server，再在另一终端运行上述命令。
吞吐收益需以 RTL 测量为准。

## 使用 Dayan-emu 运行

默认使用同级的 `dayan-emu` 和其配套的 QEMU。先在一个终端运行仓库根目录的
`run_dayan_emu_server.sh`，再从另一个终端启动 demo。
可通过 `DAYAN_EMU_DIR`、`QEMU_DL_BIN`、`NPU_DEVICE_PREFIX`、
`DAYAN_EMU_HARTS` 和 `DAYAN_SERVER_SOCKET` 配置运行路径及连接。
脚本会构建固件与 Host 工具，模拟器日志默认位于 `build/dayan_emu/dayan_emu.log`。
它使用固定名称的 Dayan-emu IPC 资源，仅支持一个实例；已有模拟器运行时请复用其
Server，不要另起脚本。

单 Core matmul 可用以下两个终端运行：

```sh
# 终端一
DAYAN_EMU_HARTS=1 ./run_dayan_emu_server.sh
# 终端二
build/demo/matmul_bf16_1c
```

模拟器日志为 `build/dayan_emu/dayan_emu.log`。测试多个独立 Task 时，可将
`DAYAN_EMU_HARTS` 设为 4，并在 Host 的 `[M N K]` 后追加 Task 数。

四 Core 版本固定启动 4 个 Hart，并以 Cluster mode 提交 `task_dim={4,1,1}`：

```sh
# 终端一
DAYAN_EMU_HARTS=4 ./run_dayan_emu_server.sh
# 终端二
build/demo/matmul_bf16_4c
```

M 维在两行 Core 之间连续等分，各行 Core 按 512 行推进；N 维按 384 列 tile
交错分给两列 Core。默认 case 每核负责 `512×768` 个输出，M 方向包含一个
512 行 tile，N 方向包含两个 384 列 tile，K 包含 6 个分片。K 尾块使用实际
`MAT_K` 和窄 DMA 请求处理，不要求 K 是 256 的整数倍。
每对 Loader 通过 Shared Resource 选举 DRAM 加载者，对端从其 URAM 复制同一份输入。
最后一个 K 分片完成后发布对应 C slice，Post 写回并异步消费，随后释放该缓冲。
dayan_emu 日志默认位于 `build/dayan_emu/dayan_emu.log`。

## 四 Core FP8 / FP4 matmul

`matmul_fp8_4c` 和 `matmul_fp4_4c` 沿用 `matmul_bf16_4c` 的 2×2 Core
划分、两个 Loader、根 Calc Proc、独立 Post、输入共享与 Resource Phase 流程。
FP8 每核保留 4 个 slice，FP4 每核使用 2 个 slice，总 tile 为 `512×512×640`。
输入仍为 `A[M][K]` 和预转置的 `BT[N][K]`，输出为 FP32。

| 算子 | 输入格式 | 每核 M×N×K tile | 默认 M×N×K |
| --- | --- | --- | --- |
| `matmul_fp8_4c` | E4M3，每元素 1 字节 | 512×384×512 | 1024×1536×3072 |
| `matmul_fp4_4c` | E2M1，每字节 2 个元素 | 512×512×640 | 1024×2048×6144 |

以下描述 1.5 MiB 配置。FP8 相对 BF16 的 K tile 扩大到 2 倍，M/N tile 和 slice 尺寸保持不变；
A/B 的 URAM 行跨度为 512 字节，A 共享路径使用 1,472 KiB，B 共享路径使用
1,408 KiB。FP4 的 M/N tile 保持 512，K tile 为 640，A/B 行跨度为 320 字节。
共享 A 时每个 slice 为 `512×256×640`，共享 B 时为 `256×512×640`，
即专有维度均为 256。每个 FP32 C 缓冲占 512 KiB，2 份共 1,024 KiB；
共享输入双缓冲占 320 KiB，2 份专有输入各占 80 KiB，共 160 KiB。
两条路径均使用 1,504 KiB，剩余 32 KiB；申请 6 个 Local、8 个 Shared Resource。
两种格式均直接执行 MAT MM/MMA，不使用分块 scale。

FP4 收尾时，各核在 join worker 后等待全部 Local Resource 变为 FREE；四核汇合后，
由 core 0 等待全部 Shared Resource 变为 FREE，再通过一次 barrier 共同退出。
这些最终等待会将资源预留为 BUSY，直到 Runtime 重新初始化；它们用于确认消费已提交，
避免仅等待执行单元完成就释放 Resource 窗口和远端仍可能读取的 URAM。

FP4 Host 的 C 使用 `N*4+64` 字节 DRAM 行跨度，默认 N=2048 时为 8256 字节。
在当前模拟器按物理地址 `[13:12]` 选择四个 DRAM 通道的配置下，这使每轮写回
分散到四个通道。A 和 BT 的底层 Buffer 均按 16 KiB 对齐，BT 在其 Buffer 内
偏移 4 KiB 创建 View，保证起始通道相对 A 错开一个；上传和 kernel 参数均使用
该 View。A/BT 的 DRAM 行跨度仍为 `(K+1)/2` 字节。C 下载包含行尾 padding，
Host 按跨度校验有效输出，并检查预填 NaN 的 padding 未被覆盖。这些是 DRAM
布局调整，URAM 行跨度、Swizzle 和 Tile 尺寸不变。

FP4 的 A/B Loader 在输入 Buffer 可复用、跨核复制依赖满足后即可预取下一输出
tile，与当前 C 写回重叠。Post 在每个有效 C slice 的 U2D 后提交 Consume，
对应 Buffer 在写回完成后释放，供下一轮 MAT 使用。

FP4 沿 K 维打包：偶数索引在低半字节，奇数索引在高半字节，每行占
`(K+1)/2` 字节。支持奇数 K，最后一个高半字节不参与计算；Host 校验时故意在
该半字节放入非零值。FP8/FP4 的 K 尾块分别搬运 `real_k` 和 `(real_k+1)/2`
字节，MAT 使用实际 K。N 仍须为 16 的整数倍，配置中的 stride 均以字节计。

针对 FP8 的 A、BT、C 实际布局，用 `host/visual_swizzle/visual_swizzle.py` 在
32 Bank 下搜索：A/BT 的 512B stride 使用 `M=7,B=2,S=3`，每拍 Bank 峰值
由线性的 4x 降为 1x；C 的 384B stride 使用 identity `M=5,B=0,S=1`，峰值
为 1x；C 的 1536B stride 使用 `M=7,B=2,S=3`，峰值由 4x 降为 1x。
三种映射的连续四块覆盖均为 32/32。

FP4 的 320B 输入行在线性布局下每拍 Bank 峰值已经为 1x，使用 identity
`M=5,B=0,S=1`。VisualSwizzle 的无约束最优值 `(9,1,1)` 会跨行重排，
不适用于当前尾行只复制 `real_rows×320B` 物理字节的跨核路径，因此不采用。
C 的 1024B 行跨度使用 `M=7,B=3,S=3`，2048B 行跨度使用 `M=7,B=3,S=4`，
两者每拍 Bank 峰值均由 8x 降为 1x，连续四块覆盖均为 32/32。
旧 K=512 的输入 swizzle `(7,1,3)` 和四切片 C 的 `(7,2,3)` 保留在源码注释中。
LSU 与 MAT 使用一致映射，跨核复制仍直接搬运物理字节。
这里的冲突指标来自布局模型，不是 RTL 性能测量。

```sh
make -C demo ../build/demo/matmul_fp8_4c ../build/demo/matmul_fp4_4c
# 终端一：启动四核 Dayan-emu
DAYAN_EMU_HARTS=4 ./run_dayan_emu_server.sh
# 终端二：运行默认尺寸并逐元素校验 CPU 参考结果
build/demo/matmul_fp8_4c
build/demo/matmul_fp4_4c
```

也可使用 `[M N K]` 参数指定尺寸；Host 根据对应格式生成确定性输入，解码后
计算 FP32 参考结果，报告最大绝对/相对误差及 PASS/FAIL。

BF16（1c/4c）和 FP4 的配置通过 Host DMA 上传到 DRAM。Device CPU 的数据缓存
与 DMA 不保持一致，buffer 地址复用后可能仍缓存上一个算子的尺寸和步长，因此每个
Hart 在 kernel 入口、第一次解引用 `config` 之前执行
`dayan_cache_invalidate(config, sizeof(*config))`。FP8 直接传递尺寸和步长标量，
不会替其他算子刷新配置缓存；重复运行 FP8 不能修复旧 FP4 入口读到的缓存配置。

切换回归应连接同一个四核 Server，期间不重启设备。在板子或 TestCluster 上执行：

```sh
make -C demo ../build/demo/matmul_bf16_1c ../build/demo/matmul_bf16_4c \
  ../build/demo/matmul_fp8_4c ../build/demo/matmul_fp4_4c
for op in matmul_bf16_4c matmul_fp4_4c matmul_fp4_4c matmul_fp8_4c \
          matmul_fp4_4c matmul_bf16_1c matmul_fp4_4c; do
  build/demo/"$op" 16 1024 640 || break
done
```

每轮应通过数值校验，FP4 的行尾 padding 也应保持完整。Dayan-emu 可验证数值和
执行顺序，但其 QEMU CPU 不模拟这类数据缓存残留，不能单独验证此问题。

## FP4 单核心矩阵乘

`matmul_fp4_1c.cpp` 基于 `matmul_fp4_4c.cpp`，使用一个 `kSingleCore` Task
（grid 为 `{1,1,1}`）计算完整的 `A[M][K] × BT[N][K] -> C[M][N]`。
保留两个 Loader、根 Calc Proc 和一个 Post Thread，仅使用 6 个 Local Resource，
不申请 Shared Resource 或 Atomic Cache，也不进行跨核复制或 barrier。

Host 通过 `demo_uram_capacity()` 调用 `npu_runtime_get_uram_capacity()` 获取
每核 URAM 容量，选择对应的 kernel descriptor 和 tile；小于 768 KiB 时拒绝启动。
这里查询的是每核容量，不是当前空闲字节数。Device 同时通过 `dayan_uram_size()`
检查本次分配的空间，使用调度到的物理 Core 的 URAM 窗口偏移。

| URAM 容量 | M×N×K tile | A 双缓冲时的 slice | B 双缓冲时的 slice | 实际 URAM：A/B 双缓冲 |
| --- | --- | --- | --- | --- |
| 1.5 MiB | 512×512×640 | 512×256×640 | 256×512×640 | 1504 / 1504 KiB |
| 768 KiB | 192×512×640 | 192×256×640 | 96×512×640 | 664 / 764 KiB |

当 `M < N` 时双缓冲 A、沿 N 分两片；否则双缓冲 B、沿 M 分两片。
每片保留 FP32 C，首个 K 块用 MM 初始化，后续用 MMA 累加，最后由 Post 写回。
输入沿 K 打包 E2M1，低半字节在先，支持奇数 K 和 M/N/K 尾块，N 须为 16 的倍数。
Host 保留原四核版本的输入生成、CPU 参考校验、DRAM 通道偏移和 C 行尾 padding 检查。

单核不需要维持跨核物理字节复制的布局，因此 A/BT 使用 VisualSwizzle 搜索得到的
`M9/B1/S1`。两种容量、两条分支中所有输入缓冲均按 1 KiB 对齐且大小为其整数倍，
映射不会越过缓冲边界；每拍 Bank 峰值保持 1x，连续四块最小覆盖由 identity 的
22/32 提高到 28/32。C 的 1024B/2048B 行跨度分别使用 `M7/B3/S3`、`M7/B3/S4`，
峰值由 8x 降为 1x、覆盖 32/32。LSU 与 MAT 使用相同映射；这些是布局模型指标。

```sh
make -C demo ../build/demo/matmul_fp4_1c
# 终端一：启动单核 cmodel
NPU_CMODEL_HARTS=1 ./run_cmodel_server.sh
# 终端二：默认尺寸为 1024×2048×6144，也可指定 M N K
build/demo/matmul_fp4_1c
build/demo/matmul_fp4_1c 513 528 1281
```

## Gated DeltaNet: TGMT, SW + Vec TGMT, and SW

`gdn.cpp` and `gdn_chunk.h` reproduce the chunked scalar-gate path in
`ggml/src/ggml-dayan/ggml-dayan-gated-delta-net.h` in the llama.cpp repository.
All three executables share the same arithmetic, FP32 tensors, TF32 MAT inputs,
64-token chunks, 128-element heads, and 768 KiB URAM layout.

```bash
make -C demo -j3 ../build/demo/gdn_tgmt ../build/demo/gdn_sw_vec_tgmt ../build/demo/gdn_sw
./build/demo/gdn_tgmt [tokens heads sequences]
./build/demo/gdn_sw_vec_tgmt [tokens heads sequences]
./build/demo/gdn_sw [tokens heads sequences]
```

Defaults are `1024 1 1`. All launch one kernel Task on one Core. `gdn_tgmt` uses
loader and precompute child Threads plus the root state-update loop, with
four Local Resources and internal VEC task parallelism. `gdn_sw_vec_tgmt`
uses only the root Proc, one VEC TCSR context, and no Named Resources, but
DIM2/3 still fork multiple VEC hardware tasks. This is the VEC policy of the old
`gdn_sw` measured on 2026-10-28; its measurements and binary snapshots
remain in the experiment record under **SW + Vec TGMT**.

The new `gdn_sw` has at most one outstanding VEC hardware task: DIM2..5
are zero, rows execute sequentially using DIM1, and DIM0 handles the
triangular recurrence or the two halves of a state row. Each VEC block
completes before the next is submitted. This does not insert per-row CPU
submissions or waits. Both SW variants prefetch the next chunk during
current state projections and prepare the next chunk while the final
state MAT update can still be in flight. All retain block-local MAT/VEC/LSU
overlap and the same arithmetic.

The SW prologue and next-chunk input loads are expanded directly in
`run_chunked`, including the paired Q/K and gate/beta DMA submissions.
The prologue waits for LSU before the first Prepare; the loop waits for
prefetched inputs and output stores before Prepare(next). These are explicit
LSU dependencies, independent of MAT waits. TGMT retains its loader helper.
Historical timings describe their saved binaries, before this source expansion.

Both SW variants are explicitly software-pipelined baselines: their single
Proc can submit asynchronous MAT/LSU work, prefetch into the second input
buffer, and overlap work within a chunk. The TGMT variant keeps the backend's
coarse loader / prepare / root split; the heavy prepare stage remains one
sequential issuer. Holding the internal VEC parallelism switch fixed therefore
does not guarantee a large speedup from this stage split. See the pipeline
review in `../dayan-samples/data_zb_lattn.md` for the source and trace evidence.

This small benchmark supports contiguous FP32 Q/K/V, one scalar gate and
beta per token/head, and one final state snapshot. It requires at least
16 tokens; KDA, shared Q/K heads/sequences, and multiple snapshots are not
exposed. Independent head/sequence chains are distributed across Tasks;
each Task processes all tokens of its assigned chains in order.
Tail input slots are zero-filled, and all paths materialize all 64 MAT
columns before reuse; Host stores only valid tokens. Full chunks retain
the backend dimensions. The input seed is fixed at 20261028; Q/K rows are
L2-normalized.

Each run checks every output and final-state element against an independent
FP32 recurrence. It uses the backend's TF32 tolerances: output atol=2e-5,
state atol=2e-4, rtol=3e-3, and separate output/state relative L2 <= 3e-3.
Set `GDN_OUTPUT=/path/to/result.bin` to save `[output | final state]` as
raw FP32 for direct comparison between the three executables.
Use `DAYAN_SERVER_SOCKET` to select an isolated simulator instance.
Experiment records are in `../dayan-samples/data_zb_lattn.md` relative to the
firmware repository; original traces and logs are under
`build/testcluster-runs/gdn-*`.

### Independent-work four-core matmul and GDN

`matmul_bf16_1c` and all three GDN variants support one or four cores. Each core owns
complete independent work items, loads its own inputs, and writes disjoint
outputs. Four-core groups synchronize only at the initial and final
`dayan_cluster_barrier(4, ...)`; no Shared Resource, peer copy or broadcast is
used. TGMT's producer/consumer Resources remain local to each core.
Idle lanes still participate in both barriers.

```sh
# M N K, number of groups, cores per group; same tiles and full K reduction.
build/demo/matmul_bf16_1c 1024 1536 1024 1 4

# Tokens, heads, sequences, cores, number of groups (last defaults to 1).
# Four independent chains keep all four cores busy; tokens are not sharded.
build/demo/gdn_tgmt        128 4 1 4
build/demo/gdn_sw_vec_tgmt 128 4 1 4
build/demo/gdn_sw          128 4 1 4
```

`matmul_bf16_1c` retains its historical executable name but also accepts
four-core launch groups. `matmul_bf16_4c` retains the existing cooperative
implementation and is a different baseline. `matmul_bf16_1c` defaults to
M=1024, N=1536, K=1024.

For GDN, strict SW means at most one VEC hardware task **per core** at a time;
four cores may therefore execute four such tasks concurrently. The hybrid
variant permits VEC TGMT within each core. Only TGMT creates child Threads.
Input-load expansions and explicit LSU waits remain in place.

Measurements and active device-source counts are appended to
`../dayan-samples/data_zb_matmul.md` and `../dayan-samples/data_zb_lattn.md`; earlier results
remain tied to their saved source/binary snapshots. The four-core timing
metric is the maximum root-Proc duration over the four cores, including
entry/exit barriers. It is not the sum of core durations.
