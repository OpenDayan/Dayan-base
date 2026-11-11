# Dayan Host 用户态驱动

`host/driver/` 是与当前 BL0/BL1 配套的 Linux 用户态驱动，分为 HAL、Server 和
Runtime 三层。应用只使用 Runtime 的逻辑对象接口；物理拓扑、设备地址、固件邮箱和
中断均由 Server 与 HAL 管理。

## 分层

```text
用户程序 / dycc 包装层
        |
        | libDayanRuntime：Context / Buffer / Stream / Kernel / Event
        | 控制连接                         传输授权连接：短消息、令牌
        v                                             |
dayan_server：会话、对象校验、网格调度                 |  不传输用户数据
        |                                             |
        | Service：BL1、RPC、ET_REL 装载              |
        v                                             v
HAL 配置与控制面                               用户进程中的 HAL 数据提供器
拓扑 / 寄存器 / 中断 / 内存                    导入设备句柄，解释传输令牌
        |                                             |
        +---------- Xilinx DMA (XDMA / QDMA) -----------+
                         直接 pread/pwrite
                  /dev 或 /tmp/virtual_npu
```

- `hal/include/`：公开的通用配置、控制面和数据面 ABI。`npu_hal_config.h` 只描述设备
  拓扑、内存和抽象后端配置。
- `hal/`：通用 HAL 生命周期、内存管理和数据提供器分派；`hal/backend/xilinx_dma/` 包含
  统一的 Xilinx DMA 配置、设备端点、低地址翻译、中断和直传实现；`driver` 区分 XDMA/QDMA。
  两者共享同一套状态、Ops 和 Runtime 数据面。每个后端提供一张完整的能力
  表，`npu_hal_open()` 通过注册表选择能力表，不使用后端选择 `switch`。
- `server/`：`service.[ch]` 负责 BL1 装载、Host 与 BL1 之间的 RPC、Device 堆和动态
  算子重定位；`server.[ch]` 与 `scheduler.c` 提供 Unix 域 Socket 服务、会话隔离、
  Runtime 对象和逻辑任务调度。
- `runtime/`：实现 `host/include/npu_runtime.h`。它只看到逻辑 Context、Buffer、
  Stream、Module、Kernel、Event 和设备级性能采样会话，不向启动接口暴露
  Cluster/Core 或 Device 地址；
  对外动态符号均为 `npu_` 前缀。

`dayan_server` 先打开具体 HAL 后端，再把不透明的 `npu_hal_t` 交给 Service。Service
配置不包含 DMA 设备路径等后端私有字段。`libDayanRuntime` 只链接通用数据面分派器和已
支持的数据提供器，不链接 HAL 控制面或 Service。以后加入基于 `ioctl` 或 `mmap` 的
实现时，不需要改变 Runtime 公共头文件或会话协议。

所有 C 接口成功返回 0，失败返回负 Linux errno。

## 多 Cluster / 多 Core

HAL 使用 `npu_hal_core_t { cluster, core }` 指定寄存器、RPC、内存可见性和 kernel
执行目标。Server 启动时读取 HAL topology，并为每个启用的物理 Core 建立内部 slot。
当前 ABI 固定每个 Cluster 有 4 个 Core。

`--cores-per-cluster N`（默认 4）告诉 HAL 硬件实际有几个 Core，例如 1c RTL 用 `--cores-per-cluster 1`。`--core-mask` 会被裁剪为低 N 位（`mask & ((1 << N) - 1)`）；只选了不存在的 Core 时 Server 启动失败。少于 4 个 Core 时只接受单核（`kSingleCore`）launch。

`npu_runtime_get_topology(runtime, &topology)` 返回 HAL 的 `cluster_count`、`cores_per_cluster` 和 Server 实际启用的 `enabled_cluster_mask`、`enabled_core_mask`。Core mask 对每个启用的 Cluster 都适用。应用可据此选择协作模式和逻辑网格大小；查询不暴露寄存器或分配物理 Core。旧 Server 不支持此查询时返回负 errno。

公开 Runtime 使用下列逻辑配置：

```cpp
struct NPUTaskDim3 { uint32_t x, y, z; };

enum class NPUTaskMode : uint32_t {
    kSingleCore = 1,
    kFourCore = 4,
};

struct NPULaunchConfig {
    NPUTaskDim3 task_dim;
    NPUTaskMode task_mode;
};
```

- `kSingleCore`：一个 logical task 是一个调度单元。Server 将 `x*y*z` 个 task 展平，
  在满足 Buffer 可见性、并发数和 resident memory 限制的 Core 间调度。
- `kFourCore`：一个 Cluster 是一个调度单元，必须满足 `task_dim.x == 4`。每个
  `(y,z)` group 原子保留同一 Cluster 的 Core 0–3，`x=0..3` 分别交给四个 Core；
  logical task 总数仍为 `4*y*z`。

Runtime、Server 和 BL1 都会校验 mode 只能为 1 或 4，并在 mode 4 时拒绝其它 x
维度。Device kernel 可用 `dayan_task_id(DAYAN_TASK_X/Y/Z)`、
`dayan_task_dim(...)` 和 `dayan_task_mode()` 读取逻辑坐标；这些 API 不返回物理位置。
每次 `npu_kernel_launch` 还必须在 `npu_launch_desc_init(&desc, task_name)` 中提供 Task
名称；dycc wrapper 则通过
`npu_kernel_launch_attrs_init(&attrs, context, stream, task_name)` 提供。名称非空且最多
15 bytes，Runtime 会把它复制进每个 logical task 对应的 root Proc。

### HAL 拓扑与内存

`npu_hal_options_t` 只配置设备拓扑、内存区域和抽象的 `backend_options` 指针。
当前注册的是 `NPU_HAL_BACKEND_XILINX_DMA`，配置结构为
`npu_hal_xilinx_dma_options_t`，头文件位于 `hal/backend/xilinx_dma/npu_hal_xilinx_dma.h`。
统一初始化函数 `npu_hal_xilinx_dma_options_init()` 保留 `simulator/fpga` profile；
`driver` 独立选择 XDMA 或 QDMA。旧 `npu_hal_xdma_options_t` 配置接口已移除。
`backend_options` 为空时采用统一后端的默认配置：XDMA、simulator、IRQ completion。
未指定 `device_name` 的 XDMA 模式将 Cluster `c` 映射到：

```text
xdma[device_index + c * cluster_device_stride]
```

两个内置运行环境是：

- `simulator`：设备目录 `/tmp/virtual_npu`，MMIO 偏移 `0x0`；
- `fpga`：设备目录 `/dev`，MMIO 偏移 `0x4000000000`，与当前 AXVU13P DUT 地址图一致。

这些默认值可以通过 `--dma-device-dir` 和 `--dma-mmio-offset` 分别覆盖，显式值优先于
profile 默认值，不受参数顺序影响。两种驱动的 H2C/C2H 地址翻译相同：设备地址低于
`0x80000000` 时加 `mmio_address_offset`，从
`0x80000000` 开始的 DRAM 地址保持不变。后端把翻译后的地址写入传输令牌，Runtime
本身不解释偏移或 Cluster。XDMA 的 `_user` 设备是独立的相对寄存器窗口，不重复
应用该物理地址偏移。

| 用途 | XDMA 示例 | QDMA 示例 |
| --- | --- | --- |
| H2C | `xdma0_h2c_0`，写方式打开 | `qdma01000-MM-0`，写方式打开 |
| C2H | `xdma0_c2h_0`，读方式打开 | 同一个 `qdma01000-MM-0`，另以读方式打开 |
| 控制窗口 | 可选 `xdma0_user` | 通过 MM 数据节点访问 |
| 事件节点 | `xdma0_events_0` | `qdma01000-user_irq0` |

QDMA 的 user IRQ 节点来自本地驱动扩展，原生驱动不保证提供。MM queue 与 user IRQ
编号独立；XDMA channel 与 event 编号也独立。QDMA 不需要 XDMA 名称的符号链接。
当前 QDMA 或显式 `device_name` 配置只支持单 Cluster；多 Cluster 只支持上述 XDMA
索引/步长映射，不会将同一 QDMA function 的多个 queue 自动视为不同 Cluster。

内存 region 同时描述 memory class 与 visibility scope：

- `NPU_HAL_MEMORY_DEVICE`：所有 Cluster/Core 可见；
- `NPU_HAL_MEMORY_CLUSTER`：只对一个 Cluster 可见；
- `NPU_HAL_MEMORY_CORE`：只对一个 Core 可见；
- class 为普通 DRAM、coherent window 或每 Core 私有 URAM；usage 区分 Buffer、
  Firmware、Kernel、Device heap、Launch descriptor、Shared/atomic-cache allocation
  和 Kernel URAM。

Server 的 Buffer 分配、调度目标选择、kernel image、BL1 固定段和反向 heap RPC 都使用
HAL allocation handle。应用侧只有 `npu_buffer_t *`，不会取得物理地址。默认 HAL
包含 device-scope DRAM/coherent region，以及每个 Core 一个 URAM region。
HAL 打开后端后、创建内存池前，读取各 Cluster BootROM `0x10100` 处的
NUL 结尾版本字符串（最多 256 字节），格式为 `<BuildConfigKey>-<gitStr>`。
启动日志逐个打印 Cluster 编号和原始配置串；配置名未识别时也会打印，空串或
读取失败分别显示 `<empty>` / `<unavailable>`。
支持当前 npu-cluster 的配置名：`Single` 为单核，`Qual` 为四核；`Tiny` 为每核
768 KiB URAM（`0xc0000`），其余为 1.5 MiB（`0x180000`）。XDMA、QDMA 和
SoC mmap 均使用该结果生成拓扑和默认 URAM 内存池，SoC 的读写范围同步更新。

`--cores-per-cluster` 和 `--uram-size` 可手动覆盖各自的探查结果；与硬件不一致时
输出包含 Cluster 编号、版本字符串、手动值和探查值的 `Warning`，保留手动值。
未识别、无 NUL、读取失败或未提供版本字符串时，输出 `Warning` 并保留原配置
（默认四核、每核 1.5 MiB）。当前 HAL 要求各 Cluster 使用统一拓扑；自动探查发现
核数或容量不同会拒绝初始化；部分 Cluster 无有效标识时整体回退到原配置。

程序化配置使用 `npu_hal_options_t.cores_per_cluster` / `uram_size`，非默认值视为
手动配置。显式指定默认值时还需将 `manual_cores_per_cluster` / `manual_uram_size`
置为 1，避免自动探查覆盖。显式 `memory_regions` 仍决定可分配的内存池，需要与
最终拓扑一致。SoC 未提供 regions 时，HAL 根据探查结果和 SoC DDR 参数生成。
URAM 从 `0x00800000` 起、Core 间窗口 stride 为 2 MiB，两种容量使用相同地址布局。
真实多 Cluster 平台可通过
`dayan_server_options_t.hal.memory_regions` 提供准确的物理布局。对同一 Cluster
可见的 region 不得重叠；不同 Cluster 的局部地址空间可以使用相同数值地址。

每个 kernel 的 Package 元信息声明 `max_child_threads`、`atomic_cache_size`、
`uram_size`、四类 TCSR 数量和 kernel flags。Server 选 Core 时累加检查 proc slot、
Atomic Cache、URAM 及 LSU/MAT/VEC/MACRO TCSR 位图：每个 invocation 按
`1 + max_child_threads` 预留 proc slot，同一 Hart 最多 16 个（root 计入，init/RPC
proc 不计入）。URAM allocation 固定为 Core scope，以 32 KiB 为分配和释放粒度；
Server 把 URAM 与 TCSR 物理范围压缩进 64-bit Context。物理 Cluster/Core 仍不会
出现在 Runtime 启动接口中。

Host Runtime 不单独配置容量；Server 按 HAL region 容量校验和调度，单个实例需求超过
容量时 launch 返回 `-ENOSPC`。Device 的 `dayan_uram_size()` 返回当前实例获分配的
大小，因此同一份受 Host 管理的 BL1 可用于两种容量。独立运行的裸机测试需用
`make -C device URAM_SIZE=0xc0000` 构建 768 KiB 版本；该选项限制裸机测试描述符
允许申请的容量，不改变 Host 下发的分配范围。

`NPU_PACKAGE_KERNEL_LOCKED_TASK` 只允许 single-Core kernel。带该标记的一个 logical
Task 只能派发到空 Hart；从 root Proc 建立到整个 family 终态期间，该 Hart 不接受任何
其它 logical Task，包括同一 grid 的下一 Task，但该 Task 自己的 child Thread 可在其真实
预留范围内运行。root 退出前会隐式 join-all，所以 invocation 终态就是安全解锁点；
Device launch 失败则沿预留回滚路径立即解锁。该标记不虚增 Proc 计数，
`max_child_threads=15` 仍表示真实预留 root 加 15 个 child Thread。

## 用户进程直接搬运数据

Runtime 为一个会话建立两条 Unix `SOCK_STREAM` 连接。控制连接负责对象、队列和算子
请求；传输授权连接只传递设备文件描述符、传输令牌和完成状态。两条连接都不传输 H2D
或 D2H 的用户数据。

Xilinx DMA 数据提供器本身不使用 Socket。Server 打开 HAL 时打开 `/dev` 或
`/tmp/virtual_npu` 下所选驱动的设备节点，并在建立会话时通过 `SCM_RIGHTS` 把文件描述符交给用户
进程。`SCM_RIGHTS` 只传递一个已打开文件的引用，不复制设备数据。Runtime 随后直接
对这些文件描述符执行 `pread/pwrite`。

H2D/D2H 流程如下：

1. Runtime 通过控制连接把传输排入 Stream，只发送逻辑 Buffer、偏移和长度。
2. 命令到达 Stream 队首后，Runtime 的传输线程申请令牌。Server 校验内存归属、方向、
   范围和可见 Cluster，再由 HAL 生成仅用于该次传输的令牌。
3. Runtime 把 Host 缓冲区和令牌交给 Xilinx DMA 数据提供器。提供器根据方向选择 H2C 或
   C2H 文件，并直接执行 `pread/pwrite`。数据不经过 Socket 或 Server 缓冲区。
4. Runtime 原样交还令牌和传输结果。Server 核对后完成 Event，并允许相关设备内存
   被释放或复用。

已授权传输被取消时，Server 会等待完成通知或传输授权连接断开，不会提前复用对应物理
地址。D2D 不涉及 Host 缓冲区，由 Server 直接调用 HAL 控制面搬运。

这条路径避免了“用户进程到 Server、Server 再到设备”的中转拷贝，但不表示内核和
XDMA/QDMA 驱动内部一定是零拷贝。令牌用于可信 Runtime 与 Server 之间的范围校验和生命周期
管理；设备文件权限仍是防止恶意进程越权访问的系统安全边界。其它 HAL 后端也应采用
相同模式：Server 签发令牌，用户进程中的数据提供器直接使用设备资源完成传输。

## BL1 生命周期与 RPC

Service 为每个 Cluster 分别加载 BL1 `ET_EXEC` 的 `PT_LOAD` 段，向选定 Core 的 BL0
mailbox 写入口、置 MSIP，并等待 BL1 打开对应 Host IRQ bit。后台 RPC worker 调用
`npu_hal_irq_wait()`，由 HAL 在超时前返回发生中断的 Cluster 和 IRQ status。
`--dma-completion irq` 要求成功打开 XDMA 的 `_events_<N>` 或 QDMA 的 `-user_irq<N>`；
事件节点缺失或读写出错时直接报错，不自动退回 polling。`--dma-completion polling`
不打开事件节点，直接读取 NPU IRQ status，每轮请求休眠 1 微秒；实际周期还包含
PCIe 访问和 Linux 调度开销。IRQ 模式等待前以及被唤醒后也会读取 status，以确认待处理事件。
只有完全使用旧命令行入口时才保留 optional-event 兼容行为；Service 不区分等待方式。

`run_server.sh` 默认设置 `DAYAN_RPC_DEBUG=1`，打印双向 RPC 的 begin/end、Cluster/Hart、
opcode、argument 和 status；使用 `DAYAN_RPC_DEBUG=0 ./run_server.sh` 可关闭。直接启动
`dayan_server` 时默认关闭，可显式设置 `DAYAN_RPC_DEBUG=1` 开启。

- Device→Host：输出、显式输出 flush、Device heap acquire/release、trap report、root proc
  exit 和 profile FIFO flush；退出通知携带返回值与 cookie，子 Thread 不发送 Host 生命周期消息。子 Thread 的
  独立 Device printf buffer 使用所属 Host root ID 输出；Cluster、Core 与 root proc ID
  由 Server 归属到发起任务的 Runtime Session。Runtime
  在 Event/Stream 同步时读取并调用应用侧 output callback，Server 不直接打印算子文本。
- Server 默认只打印 trap 和 tensor interrupt 的寄存器报告，不在故障处理路径读取设备
  内存。需要完整 dump 时显式设置 `DAYAN_TRAP_DUMP_DIR`；dump 只读取故障 Core 可见的
  Device、Cluster 和本 Core region，避免访问其它 Core 的私有 URAM。
- Host→Device：`RPC_PROC_CREATE`、`RPC_DCACHE_FLUSH`、`RPC_FENCEI`、`RPC_PROC_PS`、
  `RPC_PROC_KILL_ALL` 与 `RPC_PROFILE_CONFIG`；每个 Core 同时至多一笔，不同 Core 可并行。Service 管理接口
  `npu_service_rpc_ps/kill_all/kill_family` 显式接收 Cluster/Core，Runtime launch 接口不暴露物理拓扑。

### 张量中断源编号

`dayan_server` 默认报告中的 `source=N(name)` 使用以下硬件中断源编号表，
依据 `interrupts-20261103.csv` 更新。实现位于
`server/device.c` 的 `tensor_interrupt_source_name()`；未列出的编号打印 `unknown`。
`source` 取最近一条中断记录的 bits 47:32，`info` 为 bits 31:0 的原始硬件信息；
该 CSV 仅定义编号和名称，没有定义 `info` 的各位含义。

| source_id | source_name |
| --- | --- |
| 0 | `neomat_uram_a_read_range` |
| 1 | `neomat_uram_a_read_align` |
| 2 | `neomat_uram_b_read_range` |
| 3 | `neomat_uram_b_read_align` |
| 4 | `neomat_uram_a_scale_read_range` |
| 5 | `neomat_uram_a_scale_read_align` |
| 6 | `neomat_uram_b_scale_read_range` |
| 7 | `neomat_uram_b_scale_read_align` |
| 8 | `neomat_uram_c_read_range` |
| 9 | `neomat_uram_c_read_align` |
| 10 | `neomat_uram_c_write_range` |
| 11 | `neomat_uram_c_write_align` |
| 12 | `neomat_tcsr_access_error` |
| 13 | `neomat_decode_error` |
| 14 | `neomat_datatype_error` |
| 15 | `neodma_tcsr_access_error` |
| 16 | `neodma_uram_table_range` |
| 17 | `neodma_uram_table_align` |
| 18 | `neodma_uram_read_range` |
| 19 | `neodma_uram_read_align` |
| 20 | `neodma_uram_write_range` |
| 21 | `neodma_uram_write_align` |
| 22 | `neodma_decode_error` |
| 23 | `flexvec_decode_error` |
| 24 | `flexvec_uram_group_0_range` |
| 25 | `flexvec_uram_group_0_align` |
| 26 | `neovec_uram_lda_range` |
| 27 | `neovec_uram_lda_align` |
| 28 | `neovec_uram_ldb_range` |
| 29 | `neovec_uram_ldb_align` |
| 30 | `neovec_uram_store_range` |
| 31 | `neovec_uram_store_align` |
| 32 | `neovec_tcsr_access_error` |
| 33 | `neovec_decode_error` |
| 34 | `instruction_decode_error` |
| 35 | `tensor_inst_watchdog_timeout` |
| 36 | `resource_scheduler_error` |

### Proc 创建 ABI

create-proc ABI 使用 64-byte 对齐的 `rpc_proc_create_request`：前 11 个 RV64 word
依次为 entry、cookie、shared offset/size、启动配置、Tensor Context、Resource Context、
三个 task ID/dimension word 和两个 name word。启动配置依次打包
`stack_size_m1:u16/printf_size:u16/argc:u8/task_mode:u8/max_child_threads:u8/reserved:u8`；
`stack_size_m1 + 1` 支持完整 64KiB 栈。shared offset 相对 `0x00400000`，
Resource Context 高 32 位清零；URAM 范围由 Tensor Context 携带。
word 11..15 放 args 0..4，后续每条缓存行放 8 个参数：
0–5/6–13/14–21/22–29/30–32 个参数分别占 2/3/4/5/6 条缓存行。
Host 只分配和写入这些缓存行，Device 对同一范围执行 invalidate/flush。
Device 消费 entry 后，用 word 0 返回 `status:i32/proc_id:u16/hart_id:u8/state:u8`，
Host 读取前 88 bytes 校验返回值、cookie、task 和名称。此布局要求 Host 与固件同步更新。
Runtime launch 必须提供
非空 Task 名称；名称最多 15 bytes，最后一 byte 留给 NUL。BL1 创建 root 时一次性专属
预留全部子 slot，并复制启动数据。创建事务全程阻塞，Host 读取创建回复后即可回收描述符；
最终结果由 `RPC_PROC_EXIT` 单独上报。`dayan_create_thread(name, entry, argument, resources)` 在同一 Core
创建有独立名称的子 Thread，返回正的 ID 并继承 logical task、`shared/gp` 和 URAM；
TCSR 资源由参数显式声明并形成独立 Context。
`dayan_thread_join`/`dayan_thread_join_all` 回收直属子 Thread；父
执行对象返回前隐式等待全部子 Thread，因而 root 的终态覆盖整棵子树。每个执行对象退出
统一调用输出 flush，空缓冲不会产生 RPC，也不再额外发送 CLEAR。只有 root 主动向 Host
上报返回值；Host 在响应 `RPC_PROC_EXIT` 前记录终态，Device
收到响应后由 send Thread 归还整组静态 slot。普通完成路径不调用 `RPC_PROC_PS`。

`RPC_PROC_PS` 返回目标 Hart 的系统/算子 proc 快照，每条记录包含相同的 16-byte
名称。系统 Proc 分别命名为 `rpc-receive`、`rpc-send` 和 `profile-flush`。
`RPC_PROC_KILL_ALL` 不碰 init 和
三个系统 proc；普通等待者先从对应队列摘除，非 Produce slot 立即释放，异步 Produce
等待者标为 `KILL_WAIT_PRODUCE`，结果 token 到达后再释放。被 kill 的 root 以
`-ECANCELED` 完成，子 Thread 不单独通知 Host。定向 family kill 复用同一 RPC
描述符，按 root proc ID 选择 family，并用调用方提供的负错误码完成 root。
Service 接受 create response 后已经快速进入终态仍是合法情况。

Device 用 `dayan_uram_base()` / `dayan_uram_size()` 读取当前实例的 Core-local URAM。
四核模式可把零初始化的 `dayan_cluster_barrier_t` 放在 shared allocation 中，并调用
`dayan_cluster_barrier(4, barrier)`；Server 在启动四个 lane 前初始化本次 launch 的
shared allocation。该 barrier 使用 `AMOADD.D` ticket，可跨多轮复用；其它
参与者数量或非 `kFourCore` launch 会被拒绝。

共享区大小取整、设备端 allocation 对齐，以及 Host 侧初始化 image 的 `aligned_alloc`
均使用 `SERVER_SHARED_ALIGNMENT`（当前为 64 字节）；image 分配后显式清零。

动态 kernel loader 接收 RISC-V `ET_REL`，只从 `npu-os-api.elf` 的白名单符号表解析
未定义符号。每个 Cluster 在 BL1 启动前预留 16 MiB kernel arena，按 ELF 对齐要求
单调分配所有加载节段，包含 `.bss`。已写入地址在卸载或失败后仍不复用，arena 用尽返回
`-ENOMEM`，Service 关闭时统一释放；新模块上传不发送 D-cache RPC。
上传后并行向全部已启动 Hart 发送 `RPC_FENCEI`，全部回复后才发布入口。该操作复用
`0x0101/0x8101`，`RPC_ICACHE_FLUSH` 为兼容别名，argument 是不解引用的模块地址 cookie。
模块复用无需上传或 fence，`RPC_PROC_CREATE` 不再重复执行 `fence.i`。loader 互斥锁覆盖
分配、上传与发布。直接通过 Service 启动自行上传的入口时，调用者需先完成各目标 Hart 的 fence。
Profile START、STOP 及失败回滚复用跨核并发 helper；同一 Hart 的 RPC 继续排队串行，
所有已提交配置结束后才汇总结果或释放共享 buffer。

Device 发起的标量 RPC 将 packet、request 和 response 分别放入独占的 64-byte 对齐
slot，Host 只写 response/status，Device 确认后 invalidate 并复制回调用者。Host-owned
标量控制 descriptor 统一按 64-byte 对齐地址和 64-byte allocation 粒度分配，opcode
唯一确定结构体布局和长度。Device 在解析前 invalidate，请求字段在生命周期内只读；
结果只在 RPC 响应或 Proc family 终态写入并 clean，Host 确认可见后再释放 allocation。
提交前失败可立即释放；一旦尝试发布命令却未收到匹配响应，allocation 不重新分配，只在
HAL 关闭时统一回收。

## proc profile trace

profile 是设备级、单会话能力。`npu_profile_start()` 默认通过 HAL 分配 16 MiB 普通
Device RAM；Service 根据已启动 Hart 数量把它均分为独立 FIFO，避免多个 Hart 写同一
write index。profile record 是紧密排列的变长字节流，容量、写入计数和 flush 计数都以
byte 为单位。容量可通过 `npu_profile_options_t.buffer_size` 配置。

普通状态记录固定 8 B：低 48 bit 是 Device counter，中间 8 bit 是 Hart 内 slot，最高
8 bit 是状态。状态 `0..7` 与 Device 的 `THREAD_*` 一致；`0/1` 分别是 RUNNING/READY，
`2..7` 是各种等待、退出与 kill-wait-produce 状态。tag `0x80 | root_slot` 表示 24-B
`SLOT_DEFINE`：相同的 8-B header 后紧跟 16-B NUL 结尾 Proc 名称。`root_slot` 为所属
Proc family 的根 slot，系统线程为 0；仅使用定义 tag 的空闲位，记录长度不变。用户 Proc 使用 slot
`1..16`，三个系统 Proc 固定使用 `17..19`，slot 0 留给 `dyperf` 合成 scheduler 轨道。

START 会为该 Hart 所有现存 Proc 依次写 `SLOT_DEFINE + state`。之后状态切换只增加 8 B；
只有一个 slot 分配给新的 Thread 时，才先增加 24-B `SLOT_DEFINE`，再增加 8-B 初始
state。Device FIFO 内记录无 padding；若尾部不足以原子容纳最坏的 32 B 组合，则提前
flush，未使用尾部不计入 `count`。Host Runtime batch 可以在任意 byte 边界切分，解析器
必须根据首个 8-B header 的 state/tag 得到长度，并保留跨 batch 残片。FIFO flush 时专用
profile 系统 Proc clean cache 并发送 `RPC_PROFILE_FLUSH`；Host response 的
`continue_profile` 决定该 Hart 是否继续记录。STOP 会关闭记录、等待正在进行的 flush，
并把未满尾批次交给 Host。

Server 可同时服务多个 Runtime 会话；每个会话使用独立的控制连接和传输授权连接，每条
连接由独立线程处理。性能采样读取在等待条件变量时会释放 Server 互斥锁，因此
`dyperf` 长轮询不会阻塞其它会话提交算子或搬运数据。设备级性能采样资源只允许一个
Runtime 占用，第二个采样程序会得到 `-EBUSY`。同一个 Runtime 句柄的控制 API 仍按单条
连接串行化；NPU Stream 命令提交到 Server 后，设备调度不受该连接串行化限制。Runtime
API 为：

```c
npu_profile_options_t options;
npu_profile_t *profile;
unsigned char trace[NPU_PROFILE_MAX_BATCH_BYTES];
npu_profile_batch_info_t batch;

npu_profile_options_init(&options);
npu_profile_start(runtime, &options, &profile);
npu_profile_read(profile, trace, sizeof(trace),
                 1000, &batch);
npu_profile_stop(profile);       /* flush 每个 Hart 的尾批次 */
/* 继续 read，直到 batch.flags 含 NPU_PROFILE_BATCH_END */
npu_profile_destroy(profile);
```

`batch.cluster/hart` 只描述观测数据来源，`batch.byte_count` 是本次返回的原始字节数；
这些字段不参与 Runtime 的 kernel 调度接口。
`host/dyperf` 封装上述流程并生成 Chrome Trace Viewer JSON；每个 Hart 是独立 pid，
Hart 按 Cluster/Hart 编号升序排列。每次 slot 定义分配全局唯一的 32-bit tid；同一 Hart
内先放 scheduler 和系统线程，再按创建时间排列 Proc family，族内父 Thread 在前，
子 Thread 按名称排序，同名时按创建顺序排列。`scheduler`
轨道由相邻 Proc 的 `RUNNING` 边界合成。`thread_name` 只使用 `SLOT_DEFINE` 中的名称，
各目标状态被展开为连续的状态区间。Device counter 按 1 ns 解释，
Trace `ts` 写为带纳秒精度的小数微秒。
推荐使用
`dyperf [OPTIONS] -- COMMAND [ARG...]`：工具先启动 profile，再以另一个 Runtime Session
运行算子，并自动向子进程传递 `DAYAN_SERVER_SOCKET`。

## 构建与运行

从仓库根目录构建全部内容，或只构建 Host 驱动：

```sh
make
make -C host/driver
```

`host/driver` 是 Host C 代码，默认由系统 `cc` 编译（可通过 `CC` 覆盖）。Device 固件与
demo 需要支持 Dayan 自定义指令的 LLVM 工具链，配置方式见
[`host/dycc/README.md`](../dycc/README.md)。Runtime 静态库更新后，静态链接它的
demo 和 llama 需要重新链接才能包含更新后的数据面实现。

主要产物位于 `build/host/driver/`：

- `dayan_server`；
- `dyperf`（由 `make -C host/dyperf` 生成到 `build/host/dyperf/`）；
- `libDayanRuntime.a`、`libDayanRuntime.so` 和兼容名 `libnpu_runtime.a`；
- `libDayanHal.a`、`libDayanService.a`。

开发树中启动 daemon 的示例：

```sh
build/host/driver/dayan_server \
  --firmware build/device/npu-os.elf \
  --api-elf build/device/npu-os-api.elf \
  --backend xilinx-dma --dma-driver xdma --dma-profile simulator \
  --dma-completion polling \
  --cluster-count 1 --core-mask 0xf
```

上例显式选择 polling；模拟器提供事件节点时可改为 `--dma-completion irq`。
XDMA 模式未显式指定设备名时，多 Cluster 可追加 `--cluster-count N --cluster-stride S`；也可用
`--cluster-mask` 关闭部分 Cluster。应用默认连接 `/tmp/dayan-server.sock`，可在
`npu_runtime_options_t.server_socket` 中覆盖。

QDMA FPGA 示例（先加载驱动并启动双向 MM queue 0）：

```sh
build/host/driver/dayan_server \
  --firmware build/device/npu-os.elf \
  --api-elf build/device/npu-os-api.elf \
  --backend xilinx-dma --dma-driver qdma --dma-profile fpga \
  --dma-device-name qdma01000 --dma-queue 0 --dma-user-irq 0 \
  --dma-completion irq --cluster-count 1 --core-mask 0xf
```

若驱动未提供 user IRQ 扩展节点，应显式使用 `--dma-completion polling`。
`--xdma-profile/--xdma-device-dir/--xdma-mmio-offset` 仅作为旧 XDMA 命令行别名保留；
旧 `--backend qdma` 和 `--qdma-*` 不再是有效参数。完整选项见
[`hal/backend/xilinx_dma/README.md`](hal/backend/xilinx_dma/README.md)。

典型 Runtime 顺序为：

1. `npu_runtime_open()`；
2. `npu_context_create()`，随后创建 Buffer 与 Stream；
3. 用异步 memcpy 上传数据并等待 Event；
4. `npu_module_load()` / `npu_module_get_kernel()`；
5. 填写 `npu_launch_desc_t.config` 和参数，调用 `npu_kernel_launch()`；
6. 等待/销毁 Event，再按 Kernel、Module、Stream、Buffer、Context、Runtime 的顺序
   释放。销毁 Context 会取消并排空其所有队列与直传 token。

## 验证

不依赖模拟器的 Host 测试覆盖双 Cluster 内存可见性、完整 64-bit RPC mailbox、包格式、
跨 Context 的不可变 Module 共享与私有 Module 隔离、Socket 对象、令牌直传和
取消回收：

```sh
make -C host/driver check
```

完整链路默认使用 `~/dayan-emu`、`~/QEMU/build/dl-riscv64`、4 个 Core：

```sh
make -C host/driver dayan_emu-smoke
```

模拟器或硬件已经独立运行并暴露 XDMA 兼容设备文件时，从仓库根目录启动 Server：

```sh
NPU_DEVICE_PREFIX=/tmp/virtual_npu ./run_server.sh
```

该脚本不启动具体后端，因此可连接 dayan_emu、`npu-cluster-sim` 的 RTL 模拟器或 XDMA 硬件。
它仍检查 `xdma<index>_h2c_0/c2h_0` 并构造旧 XDMA 参数；QDMA 或自定义 channel/name
应使用上面的 `dayan_server` 直接命令，不要将新参数直接传给该脚本。
脚本和 Server 会把启动进度直接写到标准错误并立即刷新，包括等待 XDMA endpoint、打开
HAL、识别 topology、加载 API ELF、为各 Cluster 配置 Atomic Cache backing、启动各
Core、等待 BL1 ready，以及 Unix socket 就绪状态。若启动被阻塞，终端中的最后一条
`[startup]` 或 `[firmware Cn]` 日志就是当前等待阶段；看到 `[ready]` 后 Runtime 才能连接。
Server 还会记录 Runtime control/data 连接的建立与断开，以及停止过程。
收到 SIGINT/SIGTERM 后，Server 会停止接收请求，并行向各已启用 Hart 发送短超时的
`RPC_PROC_KILL_ALL`，随后取消仍在等待的 Host RPC 并回收 scheduler 资源。HAL 会保存
首个不可恢复的设备 I/O 错误；DMA endpoint 返回 `EIO`、`ENODEV`、`EBADF` 等
错误时，Server 无需额外信号便会自动退出，并跳过已经不可能完成的 Device 侧清理。

需要由脚本同时启动 dayan_emu 与 `dayan_server` 时运行：

```sh
./run_dayan_emu_server.sh
```

默认 QEMU DRAM 和 HAL Normal memory 均为 4096 MiB。可用 `DAYAN_EMU_MEMORY_MIB` 同时覆盖两者；BL1 的 `RAM_SIZE` 仍只描述固件链接区域。还可用 `DAYAN_EMU_DIR`、`QEMU_DL_BIN`、`NPU_DEVICE_PREFIX` 和 `DAYAN_EMU_HARTS=1..4` 覆盖其余配置。4-core smoke 同时验证 BL0/BL1 启动、API ELF 动态
重定位、令牌化 H2D/D2H 直传、`kSingleCore` 网格、`kFourCore` Cluster 网格，以及
`kFourCore` 对 `task_dim.x == 4` 的约束和性能采样满缓冲区、尾批次的端到端传输。
