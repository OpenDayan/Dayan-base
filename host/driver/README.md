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
        +------------------- XDMA ---------------------+
                         直接 pread/pwrite
                  /dev 或 /tmp/virtual_npu
```

- `hal/include/`：公开的通用配置、控制面和数据面 ABI。`npu_hal_config.h` 只描述设备
  拓扑、内存和抽象后端配置。
- `hal/`：通用 HAL 生命周期、内存管理和数据提供器分派；`hal/backend/xdma/` 包含
  XDMA 私有配置、设备端点、低地址翻译、中断和直传实现。每个后端提供一张完整的能力
  表，`npu_hal_open()` 通过注册表选择能力表，不使用后端选择 `switch`。
- `server/`：`service.[ch]` 负责 BL1 装载、Host 与 BL1 之间的 RPC、Device 堆和动态
  算子重定位；`server.[ch]` 与 `scheduler.c` 提供 Unix 域 Socket 服务、会话隔离、
  Runtime 对象和逻辑任务调度。
- `runtime/`：实现 `host/include/npu_runtime.h`。它只看到逻辑 Context、Buffer、
  Stream、Module、Kernel、Event 和设备级性能采样会话，不向启动接口暴露
  Cluster/Core 或 Device 地址；
  对外动态符号均为 `npu_` 前缀。

`dayan_server` 先打开具体 HAL 后端，再把不透明的 `npu_hal_t` 交给 Service。Service
配置不包含 XDMA 路径等后端私有字段。`libDayanRuntime` 只链接通用数据面分派器和已
支持的数据提供器，不链接 HAL 控制面或 Service。以后加入基于 `ioctl` 或 `mmap` 的
实现时，不需要改变 Runtime 公共头文件或会话协议。

所有 C 接口成功返回 0，失败返回负 Linux errno。

## 多 Cluster / 多 Core

HAL 使用 `npu_hal_core_t { cluster, core }` 指定寄存器、RPC、内存可见性和 kernel
执行目标。Server 启动时读取 HAL topology，并为每个启用的物理 Core 建立内部 slot。
当前 ABI 固定每个 Cluster 有 4 个 Core。

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

`npu_hal_options_t` 只配置设备拓扑、内存区域和抽象的 `backend_options` 指针。XDMA
后端的私有配置包含运行环境、设备目录、设备编号、Cluster 步长和 MMIO 地址偏移。
`backend_options` 为空时，HAL 注册表选择默认后端；当前默认项是 XDMA 模拟器配置。
XDMA 后端将 Cluster `c` 映射到：

```text
xdma[device_index + c * cluster_device_stride]
```

两个内置运行环境是：

- `simulator`：设备目录 `/tmp/virtual_npu`，MMIO 偏移 `0x0`；
- `fpga`：设备目录 `/dev`，MMIO 偏移 `0x4000000000`，与当前 AXVU13P DUT 地址图一致。

这些默认值可以通过 `--xdma-device-dir` 和 `--xdma-mmio-offset` 分别覆盖。XDMA
H2C/C2H 地址翻译为：设备地址低于 `0x80000000` 时加 `mmio_address_offset`，从
`0x80000000` 开始的 DRAM 地址保持不变。后端把翻译后的地址写入传输令牌，Runtime
本身不知道偏移或 Cluster。`_user` 设备是独立的相对寄存器窗口，不重复
应用该物理地址偏移。

内存 region 同时描述 memory class 与 visibility scope：

- `NPU_HAL_MEMORY_DEVICE`：所有 Cluster/Core 可见；
- `NPU_HAL_MEMORY_CLUSTER`：只对一个 Cluster 可见；
- `NPU_HAL_MEMORY_CORE`：只对一个 Core 可见；
- class 为普通 DRAM、coherent window 或每 Core 私有 URAM；usage 区分 Buffer、
  Firmware、Kernel、Device heap、Launch descriptor、Shared/atomic-cache allocation
  和 Kernel URAM。

Server 的 Buffer 分配、调度目标选择、kernel image、BL1 固定段和反向 heap RPC 都使用
HAL allocation handle。应用侧只有 `npu_buffer_t *`，不会取得物理地址。默认 HAL
包含 device-scope DRAM/coherent region，以及每个 Core 一个 1.5 MiB URAM region；
URAM 从 `0x00800000` 起、Core 间窗口 stride 为 2 MiB。真实多 Cluster 平台可通过
`dayan_server_options_t.hal.memory_regions` 提供准确的物理布局。对同一 Cluster
可见的 region 不得重叠；不同 Cluster 的局部地址空间可以使用相同数值地址。

每个 kernel 的 Package 元信息声明 `max_child_threads`、`atomic_cache_size`、
`uram_size`、四类 TCSR 数量和 kernel flags。Server 选 Core 时累加检查 proc slot、
Atomic Cache、URAM 及 LSU/MAT/VEC/MACRO TCSR 位图：每个 invocation 按
`1 + max_child_threads` 预留 proc slot，同一 Hart 最多 16 个（root 计入，init/RPC
proc 不计入）。URAM allocation 固定为 Core scope，以 32 KiB 为分配和释放粒度；
Server 把 URAM 与 TCSR 物理范围压缩进 64-bit Context。物理 Cluster/Core 仍不会
出现在 Runtime 启动接口中。

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

XDMA 数据提供器本身不使用 Socket。Server 在建立会话时打开 `/dev` 或
`/tmp/virtual_npu` 下的 XDMA 设备文件，并通过 `SCM_RIGHTS` 把文件描述符交给用户
进程。`SCM_RIGHTS` 只传递一个已打开文件的引用，不复制设备数据。Runtime 随后直接
对这些文件描述符执行 `pread/pwrite`。

H2D/D2H 流程如下：

1. Runtime 通过控制连接把传输排入 Stream，只发送逻辑 Buffer、偏移和长度。
2. 命令到达 Stream 队首后，Runtime 的传输线程申请令牌。Server 校验内存归属、方向、
   范围和可见 Cluster，再由 HAL 生成仅用于该次传输的令牌。
3. Runtime 把 Host 缓冲区和令牌交给 XDMA 数据提供器。提供器根据方向选择 H2C 或
   C2H 文件，并直接执行 `pread/pwrite`。数据不经过 Socket 或 Server 缓冲区。
4. Runtime 原样交还令牌和传输结果。Server 核对后完成 Event，并允许相关设备内存
   被释放或复用。

已授权传输被取消时，Server 会等待完成通知或传输授权连接断开，不会提前复用对应物理
地址。D2D 不涉及 Host 缓冲区，由 Server 直接调用 HAL 控制面搬运。

这条路径避免了“用户进程到 Server、Server 再到设备”的中转拷贝，但不表示内核和
XDMA 驱动内部一定是零拷贝。令牌用于可信 Runtime 与 Server 之间的范围校验和生命周期
管理；设备文件权限仍是防止恶意进程越权访问的系统安全边界。其它 HAL 后端也应采用
相同模式：Server 签发令牌，用户进程中的数据提供器直接使用设备资源完成传输。

## BL1 生命周期与 RPC

Service 为每个 Cluster 分别加载 BL1 `ET_EXEC` 的 `PT_LOAD` 段，向选定 Core 的 BL0
mailbox 写入口、置 MSIP，并等待 BL1 打开对应 Host IRQ bit。后台 RPC worker 调用
`npu_hal_irq_wait()`，由 HAL 在超时前返回发生中断的 Cluster 和 IRQ status。XDMA
后端优先等待 `_events_0`；事件接口缺失或不支持阻塞等待时，兼容轮询也只存在于 HAL
内部。

- Device→Host：输出、清屏、Device heap acquire/release、trap report、root proc
  exit 和 profile FIFO flush；退出通知携带返回值与 cookie，子 Thread 不发送 Host 生命周期消息。子 Thread 的
  独立 Device printf buffer 使用所属 Host root ID 输出；Cluster、Core 与 root proc ID
  由 Server 归属到发起任务的 Runtime Session。Runtime
  在 Event/Stream 同步时读取并调用应用侧 output callback，Server 不直接打印算子文本。
- Server 默认只打印 trap 和 tensor interrupt 的寄存器报告，不在故障处理路径读取设备
  内存。需要完整 dump 时显式设置 `DAYAN_TRAP_DUMP_DIR`；dump 只读取故障 Core 可见的
  Device、Cluster 和本 Core region，避免访问其它 Core 的私有 URAM。
- Host→Device：`RPC_PROC_CREATE`、`RPC_DCACHE_FLUSH`、`RPC_ICACHE_FLUSH`、`RPC_PROC_PS`、
  `RPC_PROC_KILL_ALL` 与 `RPC_PROFILE_CONFIG`；每个 Core 同时至多一笔，不同 Core 可并行。Service 管理接口
  `npu_service_rpc_ps/kill_all/kill_family` 显式接收 Cluster/Core，Runtime launch 接口不暴露物理拓扑。

create-proc ABI 使用 224-byte `rpc_proc_create_request`，包含 entry、栈、shared、
printf buffer、0–8 个 RV64 参数、cookie、`task_id[3]`、`task_dim[3]`、`task_mode`，
最大子 Thread 数、URAM base/size、64-bit Tensor Context，以及固定 16-byte Proc 名称。Runtime launch 必须提供
非空 Task 名称；名称最多 15 bytes，最后一 byte 留给 NUL。BL1 创建 root 时一次性专属
预留全部子 slot；`dayan_create_thread(name, entry, argument, resources)` 在同一 Core
创建有独立名称的子 Thread，返回正的 ID 并继承 logical task、`shared/gp` 和 URAM；
TCSR 资源由参数显式声明并形成独立 Context。
`dayan_thread_join`/`dayan_thread_join_all` 回收直属子 Thread；父
执行对象返回前隐式等待全部子 Thread，因而 root 的终态覆盖整棵子树。每个执行对象退出会
flush/CLEAR 输出；只有 root 主动向 Host 上报返回值。最后由 scheduler 栈归还整组静态
slot，并把描述符置为 `EXITED` 或 `FAILED`。

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

动态 kernel loader 接收 RISC-V `ET_REL`，只从 `npu-os-api.elf` 的白名单符号表解析
未定义符号。覆盖模块前，会对该 Cluster 所有已启动 Core 分别执行带地址和长度的
`RPC_DCACHE_FLUSH`；全部完成后写入新的 text/rodata，再执行 `RPC_ICACHE_FLUSH` 并发布入口。

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
`2..7` 是各种等待、退出与 kill-wait-produce 状态。tag `0x80` 表示 24-B
`SLOT_DEFINE`：相同的 8-B header 后紧跟 16-B NUL 结尾 Proc 名称。用户 Proc 使用 slot
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
Hart 按 Cluster/Hart 编号升序排列。每次 slot 定义是独立 generation，Host 使用
`(cluster, hart, slot, generation)` 区分轨道；同一 Hart 内按 Proc 名称排序。`scheduler`
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
  --xdma-profile simulator \
  --cluster-count 1 --core-mask 0xf
```

多 Cluster 可追加 `--cluster-count N --cluster-stride S`；也可用
`--cluster-mask` 关闭部分 Cluster。应用默认连接 `/tmp/dayan-server.sock`，可在
`npu_runtime_options_t.server_socket` 中覆盖。

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

完整链路默认使用 `~/rvtng-cmodel`、`~/npu-qemu/build/dl-riscv64`、4 个 Core：

```sh
make -C host/driver cmodel-smoke
```

模拟器或硬件已经独立运行并暴露 XDMA 兼容设备文件时，从仓库根目录启动 Server：

```sh
NPU_DEVICE_PREFIX=/tmp/virtual_npu ./run_server.sh
```

该脚本不启动具体后端，因此可连接 cmodel、`npu-cluster-sim` 的 RTL 模拟器或硬件。
脚本和 Server 会把启动进度直接写到标准错误并立即刷新，包括等待 XDMA endpoint、打开
HAL、识别 topology、加载 API ELF、为各 Cluster 配置 Atomic Cache backing、启动各
Core、等待 BL1 ready，以及 Unix socket 就绪状态。若启动被阻塞，终端中的最后一条
`[startup]` 或 `[firmware Cn]` 日志就是当前等待阶段；看到 `[ready]` 后 Runtime 才能连接。
Server 还会记录 Runtime control/data 连接的建立与断开，以及停止过程。
收到 SIGINT/SIGTERM 后，Server 会停止接收请求，并行向各已启用 Hart 发送短超时的
`RPC_PROC_KILL_ALL`，随后取消仍在等待的 Host RPC 并回收 scheduler 资源。HAL 会保存
首个不可恢复的设备 I/O 错误；XDMA endpoint 返回 `EIO`、`ENODEV`、`EBADF` 等
错误时，Server 无需额外信号便会自动退出，并跳过已经不可能完成的 Device 侧清理。

需要由脚本同时启动 cmodel 与 `dayan_server` 时运行：

```sh
./run_cmodel_server.sh
```

默认 QEMU DRAM 和 HAL Normal memory 均为 4096 MiB。可用 `NPU_CMODEL_MEMORY_MIB` 同时覆盖两者；BL1 的 `RAM_SIZE` 仍只描述固件链接区域。还可用 `RVTNG_CMODEL_DIR`、`RVTNG_QEMU`、`NPU_DEVICE_PREFIX` 和 `NPU_CMODEL_HARTS=1..4` 覆盖其余配置。4-core smoke 同时验证 BL0/BL1 启动、API ELF 动态
重定位、令牌化 H2D/D2H 直传、`kSingleCore` 网格、`kFourCore` Cluster 网格，以及
`kFourCore` 对 `task_dim.x == 4` 的约束和性能采样满缓冲区、尾批次的端到端传输。
