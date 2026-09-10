# Dayan kernel demos

该目录放置最小、可独立构建的 Dayan kernel 示例。目前包含 Device `printf`、
FP32 Add、单 Core BF16 矩阵乘和四 Core 协作 BF16/FP8/FP4 矩阵乘。

## 目录

```text
demo/
├── Makefile
├── add_fp32_macro.cpp          # 最小 FP32 Add：LSU + MACRO
├── add_fp32_vec.cpp            # 最小 FP32 Add：LSU + VEC
├── flash_attention_1c.cpp      # 128-row FP16 FlashAttention / FP8 softmax pipeline
├── lsu_bandwidth.cpp            # 单核 D2U / 远端 U2D 带宽微基准
├── shared_notify_latency.cpp    # core 0 -> core 1 Shared Resource 通知延迟
├── matmul_bf16_1c.cpp           # BF16 OS kernel、Host launcher 与 CPU 校验
├── matmul_bf16_4c.cpp           # 2x2 Cluster：4 个 slice、共享输入双缓冲、独立 Post
├── matmul_fp8_4c.cpp            # 相同四核流水线，E4M3 输入，K tile=512
├── matmul_fp4_4c.cpp            # 四核流水线，打包 E2M1，512×512×512 tile、4 个 slice
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

产物为 `build/demo/print_demo`、`build/demo/add_fp32_macro`、
`build/demo/add_fp32_vec`、`build/demo/lsu_bandwidth`、
`build/demo/shared_notify_latency`、
`build/demo/flash_attention_1c`、
`build/demo/matmul_bf16_1c`、`build/demo/matmul_bf16_4c`、
`build/demo/matmul_fp8_4c`、`build/demo/matmul_fp4_4c` 和
`build/demo/matmul_bf16_broadcast_4c`。Device 自定义指令由 Clang 的
`<rvt_intrinsics.h>`（`__builtin_riscv_tx_*`）提供；`dayan_intrinsics.h` 包含该头文件，
并保留 LSU/MAT TCSR 编号和 `set_lsu_matrix` 等 Dayan 辅助接口。dycc 会启用 TX
扩展并把 Clang resource dir 加入 Device 编译，构建 demo 不需要 npu-kernels。

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

## 单 Core FP8/FP16 FlashAttention tile

`flash_attention_1c` 固定处理 128 行、`D=128` 和 4 个 256-column KV tile。同一份 Device
object 同时提供 `flash_attention_fp16_1c` 和 `flash_attention_fp8_1c` 两个 kernel，
共享 tile、Resource 和 online-softmax 实现，不再通过编译宏二选一：

```text
S       = Q[128][128] FP16 x KT[4][256][128] FP16
P       = exp2(S - M_new)
L_new   = alpha * L_old + rowsum(P)
O_acc   = alpha * O_acc + FP16(P) x VT[4][128][256] FP16
O       = O_acc / L
```

Device 创建 LSU、MAT、VEC 三个 DAPI worker，并用 Resource 串联数据生命周期。
KT/VT 双缓冲，score/P 三缓冲。MAT worker 使用两个常驻配置组，按
`QK(0), QK(1), PV(0), QK(2), PV(1), ...` 交错执行，QK 和 PV 都各用一条覆盖
128 row 的大 MAT 指令。Vec worker 在 softmax 流中原地计算 `alpha * O_acc`，随后
在同一个 VFork 尾部用 VEC 将 P 从 FP32 转成连续 FP16；PV 直接以 `mma`
累加到 URAM 中的 FP32 O_acc。
LSU 的通用配置只写一次，循环中没有重复刷新整组 TCSR。

唯一的 Vec worker 每个 KV tile 发射一次 `DIM2=15` 的 VFork，生成 16 个
FlexVec task。硬件将 task 分配到两个 Vec Group，每组最多同时运行 8 个 thread，
因此一次覆盖 `16 x 8 = 128` 行。每个 task 执行与 RVTNG
`FlashAttentionVecFlowSpec` 的 online-softmax 流，并追加 O_acc 缩放及逐行
FP32-to-FP16 convert/compress，共 421 条 FlexVec 指令；最终 normalization 使用
另一个 50 条指令的 VFork。`KT` 和 `VT`
分别按 key row 与 output-channel row 转置存储。Q 必须预先折叠 attention scale
和 `log2(e)`，因为向量阶段直接计算 `exp2(score - newM)`。M/L 使用 Vec 的物理
布局，每行标量复制 8 份。

FP16 URAM 实际使用 936 KiB，descriptor 申请 960 KiB：Q 32 KiB、KT/VT 双缓冲
256 KiB、score/P 三缓冲 576 KiB、O_acc 64 KiB、M/L 8 KiB，低于单 Core
1.5 MiB 容量。

Q/KT 的 swizzle 由 `host/visual_swizzle` 穷举选择，并在 LSU 写入和 MAT A/B
读取两端配置相同的 M/B/S。FP8 的 128B stride 在线性布局下已经无冲突，采用显式
identity `M=5,B=0,S=1`；FP16 Q/KT 的 256B stride 采用 `M=7,B=1,S=3`，VT 的
512B stride 采用 `M=7,B=2,S=3`，模型中的每拍 Bank 峰值降为 1x。score/P
目前保持 identity，因为 FlexVec TCSR
尚未提供 swizzle 字段；MAT C、FlexVec 和 LSU 回写必须保持一致。

`flash_attention_fp8_1c` kernel 暂时保留原来的 QK + online-softmax 路径，只校验
P/M/L，不执行低精度 P 转换、PV 和 O normalization。Host 使用确定性输入并逐元素
校验 FP16 O 或 FP8 P/M/L；可执行文件默认运行 FP16，`--fp8` 选择 FP8：

```sh
make -C demo ../build/demo/flash_attention_1c
build/demo/flash_attention_1c
build/demo/flash_attention_1c --fp8
```

## LSU 单核带宽微基准

`lsu_bandwidth` 隔离测试一个 Core 发射的连续 D2U，以及从本 Core URAM 到相邻 Core
URAM 地址窗的 U2D。四个 Core 仅用于建立 Cluster URAM 映射；测量区间内只有 lane 0
发射 DMA。每个尺寸都严格采用“发射一条 DMA、立即 `dayan_wait_lsu()`”的顺序，并用
`dayan_get_time()` 记录 4 KiB、16 KiB、64 KiB、256 KiB 和 1 MiB 的端到端周期。测试结束
后另行回读 64 KiB 做数据校验，回读不计入带宽：

```sh
build/demo/lsu_bandwidth       # 依次测试 D2U 和远端 U2D
build/demo/lsu_bandwidth d2u
build/demo/lsu_bandwidth u2d
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

这里 Host 直接按转置布局提供 B。Device 严格沿用原版 output-stationary 的一个 producer
和三个 consumer Thread：当 `M < N` 时使用 `256x384x256` tile，producer 双缓冲读取
A，三个 consumer 各处理 128 列并读取自己的 BT；否则使用原版的 `384x256x256`
tile，producer 双缓冲读取 BT，三个 consumer 各处理 128 行并读取自己的 A。共享 tile
的代际与 producer / consumer 关系由两个硬件 Local Resource 管理。producer 通过
`dayan_resource_wait_free()` 预留双缓冲，并用 `dayan_resource_produce_lsu()` 发布；
MAT consumer 在完成后用 `dayan_resource_consume_mat()` 释放输入。三切片模式的 URAM
占用为 832 KiB。

默认 Host case 为 `M=768, N=512, K=1536`。程序生成确定性 BF16 输入、执行 Device
kernel，并逐元素和 CPU 参考结果比较：

```sh
build/demo/matmul_bf16_1c
build/demo/matmul_bf16_1c 128 5088 1272
```

`N` 必须是 16 的整数倍；K 尾块通过实际 `MAT_K` 和窄 DMA 请求处理，不要求是 256
的整数倍。编译期可把切片模式改成单 slice；切换已有产物时用 `-B` 强制重新构建：

```sh
make -B -C demo MATMUL_SLICES=1
```

## 四 Core BF16 matmul

`matmul_bf16_4c` 默认计算 `M=1024, N=1536, K=1536`，每核固定使用
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

可用 `./run_cmodel_server.sh` 启动四核 cmodel Server，再在另一终端运行上述命令。
吞吐收益需以 RTL 测量为准。

## 使用 cmodel 一键运行

默认使用 `~/rvtng-cmodel` 和其配套的 `deps/npu-qemu/build/dl-riscv64`：

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
bash demo/run_matmul_cmodel.sh 64 384 248
```

对应日志为 `build/demo/matmul_cmodel.log`、
`build/demo/matmul_dayan_server.log` 和 `build/demo/matmul_bf16_1c.log`。

四 Core 版本固定启动 4 个 Hart，并以 Cluster mode 提交 `task_dim={4,1,1}`：

```sh
# 终端一
NPU_CMODEL_HARTS=4 ./run_cmodel_server.sh
# 终端二
build/demo/matmul_bf16_4c
```

M 维在两行 Core 之间连续等分，各行 Core 按 512 行推进；N 维按 384 列 tile
交错分给两列 Core。默认 case 每核负责 `512×768` 个输出，M 方向包含一个
512 行 tile，N 方向包含两个 384 列 tile，K 包含 6 个分片。K 尾块使用实际
`MAT_K` 和窄 DMA 请求处理，不要求 K 是 256 的整数倍。
每对 Loader 通过 Shared Resource 选举 DRAM 加载者，对端从其 URAM 复制同一份输入。
最后一个 K 分片完成后发布对应 C slice，Post 写回并异步消费，随后释放该缓冲。
cmodel 日志默认位于 `build/cmodel/cmodel.log`。

## 四 Core FP8 / FP4 matmul

`matmul_fp8_4c` 和 `matmul_fp4_4c` 沿用 `matmul_bf16_4c` 的 2×2 Core
划分、两个 Loader、根 Calc Proc、独立 Post、输入共享与 Resource Phase 流程。
FP8 和 FP4 每核均保留 4 个 slice，FP4 的每核 tile 为 `512×512×512`。
输入仍为 `A[M][K]` 和预转置的 `BT[N][K]`，输出为 FP32。

| 算子 | 输入格式 | 每核 M×N×K tile | 默认 M×N×K |
| --- | --- | --- | --- |
| `matmul_fp8_4c` | E4M3，每元素 1 字节 | 512×384×512 | 1024×1536×3072 |
| `matmul_fp4_4c` | E2M1，每字节 2 个元素 | 512×512×512 | 1024×1536×6144 |

FP8 相对 BF16 的 K tile 扩大到 2 倍，M/N tile 和 slice 尺寸保持不变；
A/B 的 URAM 行跨度为 512 字节，A 共享路径使用 1,472 KiB，B 共享路径使用
1,408 KiB。FP4 的 M/N/K tile 尺寸均为 512，A/B 行跨度为 256 字节。
共享 A 时每个 slice 为 `512×128×512`，共享 B 时为 `128×512×512`，
即专有维度均为 128。每个 FP32 C 缓冲占 256 KiB，4 份共 1,024 KiB；
共享输入双缓冲占 256 KiB，4 份专有输入各占 32 KiB，共 128 KiB。
两条路径均使用 1,408 KiB，低于每核 1,536 KiB 的 URAM 容量。
两种格式均直接执行 MAT MM/MMA，不使用分块 scale。

FP4 沿 K 维打包：偶数索引在低半字节，奇数索引在高半字节，每行占
`(K+1)/2` 字节。支持奇数 K，最后一个高半字节不参与计算；Host 校验时故意在
该半字节放入非零值。FP8/FP4 的 K 尾块分别搬运 `real_k` 和 `(real_k+1)/2`
字节，MAT 使用实际 K。N 仍须为 16 的整数倍，配置中的 stride 均以字节计。

针对 FP8 的 A、BT、C 实际布局，用 `host/visual_swizzle/visual_swizzle.py` 在
32 Bank 下搜索：A/BT 的 512B stride 使用 `M=7,B=2,S=3`，每拍 Bank 峰值
由线性的 4x 降为 1x；C 的 384B stride 使用 identity `M=5,B=0,S=1`，峰值
为 1x；C 的 1536B stride 使用 `M=7,B=2,S=3`，峰值由 4x 降为 1x。
三种映射的连续四块覆盖均为 32/32。FP4 的 A/BT 使用 `M=7,B=1,S=3`，
C 的 512B 行跨度使用 `M=7,B=2,S=3`，2048B 行跨度使用 `M=7,B=3,S=4`。
LSU 与 MAT 使用一致映射，跨核复制仍直接搬运物理字节。
这里的冲突指标来自布局模型，不是 RTL 性能测量。

```sh
make -C demo ../build/demo/matmul_fp8_4c ../build/demo/matmul_fp4_4c
# 终端一：启动四核 cmodel
NPU_CMODEL_HARTS=4 ./run_cmodel_server.sh
# 终端二：运行默认尺寸并逐元素校验 CPU 参考结果
build/demo/matmul_fp8_4c
build/demo/matmul_fp4_4c
```

也可使用 `[M N K]` 参数指定尺寸；Host 根据对应格式生成确定性输入，解码后
计算 FP32 参考结果，报告最大绝对/相对误差及 PASS/FAIL。

## 四 Core 直接广播性能模型

`matmul_bf16_broadcast_4c` 建模更完整的未来广播指令：源端 Loader 的一次 D2U
同时填充本地和远端 URAM，完成后直接发布 Local/Shared Named Resource。tile 固定为
`512x384x256`，A 沿 M 维等分为四个 `128x256` slice。
每个 Core 只创建一个 DRAM Loader 和两个有序 MatStore worker，不再创建 Copy worker；
源端 MAT 消费 Local Resource，远端 MAT 直接等待并消费 Shared Resource。B 的远端
Shared Resource 由四个 M slice 分别消费。模型保留真实的 Load/MAT/C 写回时序与同步
成本，但由于当前指令不会真实写入远端 URAM，仍跳过数值校验。

```sh
build/demo/matmul_bf16_broadcast_4c
```
