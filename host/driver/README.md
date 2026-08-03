# Dayan Host 用户态驱动

`host/driver/` 是与当前 BL0/BL1 配套的 Linux 用户态驱动，分为 `hal`、`server`、
`runtime` 三层。应用只使用 Runtime 的逻辑对象接口；Cluster/Core、物理地址、固件
mailbox 和 IRQ 都留在 Server 与 HAL 内部。

## 分层

```text
Application / dycc wrapper
        |
        | libDayanRuntime: Context / Buffer / Stream / Kernel / Event
        | control socket                         data socket: token
        v                                             |
dayan_server: session、对象校验、grid scheduler       |
        |                                             |
        | Service: BL1、RPC、ET_REL loader            |
        v                                             v
HAL control plane                              HAL data-plane provider ABI
register / IRQ / memory allocator              opaque export / token / handles
        |                                             |
        +------------- FD_OFFSET / cmodel FUSE -------+
                         (future: ioctl provider)
```

- `hal/`：对 Service 暴露设备无关的 topology、register、IRQ、boot、allocation 和
  transfer 接口；FUSE/XDMA 只是当前 control backend。独立的 `dataplane.h` 使用
  provider ID、opaque export/token 和通用 handle 数组，不含 Cluster/Core 语义。
- `server/`：`service.[ch]` 以下负责 BL1 装载、Host↔BL1 RPC、Device heap 和动态
  kernel 重定位；`server.[ch]` 与 `scheduler.c` 提供 Unix socket daemon、会话隔离、
  Runtime 对象和 logical task 调度。
- `runtime/`：实现 `host/include/npu_runtime.h`。它只看到逻辑 Context、Buffer、
  Stream、Module、Kernel、Event 和设备级 profile 会话，不向 launch 接口暴露
  Cluster/Core 或 Device 地址；
  对外动态符号均为 `npu_` 前缀。

`dayan_server` 先打开具体 HAL backend，再把 opaque `npu_hal_t` 注入 Service；Service
配置中没有 FUSE 路径等 backend 字段。`libDayanRuntime` 只链接通用 data-plane
dispatcher 与已支持的 provider，不链接 HAL 控制面或 Service。加入 ioctl 实现时无需
改变 Runtime 公共头文件或 socket 对象协议。

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

### HAL topology 与内存

`npu_hal_options_t` 配置 `cluster_count`、`cores_per_cluster` 和
`cluster_device_stride`。FUSE/XDMA backend 将 Cluster `c` 映射到：

```text
xdma[device_index + c * cluster_device_stride]
```

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

每个 kernel 的 Package 元信息声明 `max_child_procs`、`atomic_cache_size` 和
`uram_size`。Server 选 Core 时累加检查三类资源：每个 invocation 按
`1 + max_child_procs` 预留 proc slot，同一 Hart 最多 16 个（root 计入，init/RPC
proc 不计入）；atomic cache 与 URAM 分别受 HAL 对该 Core 可见的容量限制。URAM
allocation 固定为 Core scope，以 1 KiB 为分配和释放粒度；物理 Cluster/Core 仍不会
出现在 Runtime 启动接口中。

## socket 与零中转 memcpy

Runtime 为一个 session 建立 control 和 data 两条 Unix `SOCK_STREAM` 连接。control
连接负责对象、队列和算子请求；data 连接只负责 memcpy token 的授权与完成。

H2D/D2H 流程如下：

1. Runtime 通过 control socket 把传输排入 Stream，只发送 Buffer handle、offset 和
   size，不发送 payload。
2. 该命令到达 Stream 队首后，Runtime 的 data worker 请求 token。Server 校验
   allocation、方向、范围和可见 Cluster，再由 HAL 生成 opaque token。
3. Runtime 使用 HELLO 阶段经 `SCM_RIGHTS` 获得的 data-plane handle，把 Host buffer
   直接交给匹配的 provider；当前 `FD_OFFSET` provider 执行 `pread/pwrite`，数据不
   经过 socket 或 Server buffer。
4. Runtime 原样回传 token、transfer ID 和状态。Server 校验后才完成 Event 并允许
   allocation 被回收。

已授权传输被取消时，Server 会等待 `TRANSFER_COMPLETE` 或 data socket 断开，不会
提前复用对应物理地址。D2D 没有 Host payload，由 Server 直接调用控制侧 HAL 搬运。

## BL1 生命周期与 RPC

Service 为每个 Cluster 分别加载 BL1 `ET_EXEC` 的 `PT_LOAD` 段，向选定 Core 的 BL0
mailbox 写入口、置 MSIP，并等待 BL1 打开对应 Host IRQ bit。后台 RPC worker 调用
`npu_hal_irq_wait()`，由 HAL 在超时前返回发生中断的 Cluster 和 IRQ status。FUSE/XDMA
后端优先等待 `_events_0`；事件接口缺失或不支持阻塞等待时，兼容轮询也只存在于 HAL
内部。

- Device→Host：输出、清屏、Device heap acquire/release、trap report、root proc
  exit 和 profile FIFO flush；退出通知携带返回值与 cookie，子 proc 不发送 Host 生命周期消息。子 proc 的
  独立 Device printf buffer 使用所属 Host root ID 输出；Cluster、Core 与 root proc ID
  由 Server 归属到发起任务的 Runtime Session。Runtime
  在 Event/Stream 同步时读取并调用应用侧 output callback，Server 不直接打印算子文本。
- Host→Device：`RPC_PROC_CREATE`、`RPC_ICACHE_FLUSH`、`RPC_PROC_PS`、
  `RPC_PROC_KILL_ALL` 与 `RPC_PROFILE_CONFIG`；每个 Core 同时至多一笔，不同 Core 可并行。Service 管理接口
  `npu_service_rpc_ps/kill_all` 显式接收 Cluster/Core，Runtime launch 接口不暴露物理拓扑。

create-proc ABI v4 使用 224-byte `rpc_proc_create_request`，包含 entry、栈、shared、
printf buffer、0–8 个 RV64 参数、cookie、`task_id[3]`、`task_dim[3]`、`task_mode`，
最大子 proc 数、URAM base/size，以及固定 16-byte Proc 名称。Runtime launch 必须提供
非空 Task 名称；名称最多 15 bytes，最后一 byte 留给 NUL。BL1 创建 root 时一次性专属
预留全部子 slot；`dayan_create_proc(name, entry, argument)` 在同一 Core 创建有独立名称的
子 proc，返回正的 ID 并继承 logical task、`shared/gp` 和 URAM。
`dayan_join`/`dayan_join_all` 回收直属子 proc；父
proc 返回前隐式等待全部子 proc，因而 root 的终态覆盖整棵子树。每个 proc 退出会
flush/CLEAR 输出；只有 root 主动向 Host 上报返回值。最后由 scheduler 栈归还整组静态
slot，并把描述符置为 `EXITED` 或 `FAILED`。

`RPC_PROC_PS` v3 返回目标 Hart 的系统/算子 proc 快照，每条记录包含相同的 16-byte
名称。系统 Proc 分别命名为 `rpc-receive`、`rpc-send` 和 `profile-flush`。
`RPC_PROC_KILL_ALL` 不碰 init 和
三个系统 proc；普通等待者先从对应队列摘除，非 Produce slot 立即释放，异步 Produce
等待者标为 `KILL_WAIT_PRODUCE`，结果 token 到达后再释放。被 kill 的 root 以
`-ECANCELED` 完成，子 proc 不单独通知 Host。
Service 接受 create response 后已经快速进入终态仍是合法情况。

Device 用 `dayan_uram_base()` / `dayan_uram_size()` 读取当前实例的 Core-local URAM。
四核模式可把零初始化的 `dayan_cluster_barrier_t` 放在 shared allocation 中，并调用
`dayan_cluster_barrier(4, barrier)`；该 barrier 可复用，使用 release/acquire 语义，
其它参与者数量或非 `kFourCore` launch 会被拒绝。

动态 kernel loader 接收 RISC-V `ET_REL`，只从 `npu-os-api.elf` 的白名单符号表解析
未定义符号。代码写入某个 Cluster 后，会对该 Cluster 所有已启动 Core 分别执行
`RPC_ICACHE_FLUSH`，再发布入口。

## proc profile trace

profile 是设备级、单会话能力。`npu_profile_start()` 默认通过 HAL 分配 16 MiB 普通
Device RAM；Service 根据已启动 Hart 数量把它均分为独立 FIFO，避免多个 Hart 写同一
write index。profile v3 的每条原始事件为 24 bytes：8-byte cycle/proc/state 头加
16-byte Proc 名称。容量可通过
`npu_profile_options_t.buffer_size` 配置。

每个事件头的低 32 bit 是 `cycle`，中间 16 bit 是 proc ID，最高 16 bit 是状态变化后的
目标状态，数值与 Device 的 `THREAD_*` 状态一致：
0/1 分别为 RUNNING/READY，2--7 为各种等待、退出与 kill-wait-produce 状态。START 会先
写入该 Hart 所有现存 proc 的初始状态（receive、send、profile 以及最多 16 个算子
proc），之后每次 proc 状态实际变化只写一个事件。profile 运行期间新建的 Proc 会先写
带名称的 `WAIT_START`，因此 Host 能立即建立新的 tid/name 映射。每 Hart FIFO 至少要
容纳完整的 19 项初始快照，事件数不要求成对。FIFO 写满时，专用 profile 系统 proc 清理 cache 并
发送 `RPC_PROFILE_FLUSH`；Host response 的 `continue_profile` 决定该 Hart 是否继续记录。
STOP 会关闭记录、等待正在进行的 flush，并把未满尾批次交给 Host。

Server 可同时服务多个 Runtime Session；每个 Session 使用独立的 control/data socket，
每条连接由独立线程处理。profile read 的条件变量等待会释放 Server mutex，因此
`dyperf` 长轮询不会阻塞其它 Session 的算子提交和数据搬运。设备级 profile 资源本身
只允许一个 Runtime 占用，第二个 profiler 会得到 `-EBUSY`。同一个 Runtime 句柄的
control API 仍按单 socket 串行化；NPU Stream 命令提交到 Server 后，设备调度不受该
socket 串行化限制。Runtime API 为：

```c
npu_profile_options_t options;
npu_profile_t *profile;
npu_profile_event_t events[NPU_PROFILE_MAX_BATCH_EVENTS];
npu_profile_batch_info_t batch;

npu_profile_options_init(&options);
npu_profile_start(runtime, &options, &profile);
npu_profile_read(profile, events, NPU_PROFILE_MAX_BATCH_EVENTS,
                 1000, &batch);
npu_profile_stop(profile);       /* flush 每个 Hart 的尾批次 */
/* 继续 read，直到 batch.flags 含 NPU_PROFILE_BATCH_END */
npu_profile_destroy(profile);
```

`batch.cluster/hart` 只描述观测数据来源，不参与 Runtime 的 kernel 调度接口。
`host/dyperf` 封装上述流程并生成 Chrome Trace Viewer JSON；每个 Hart 是独立 pid，
Hart 按 Cluster/Hart 编号升序排列。每个 proc ID 是独立 tid，同一 Hart 内按 Proc 名称
排序；tid 0 的 `scheduler` 轨道由相邻 Proc 的 `RUNNING` 边界合成。`thread_name` 使用
事件携带的 Proc 名称，各目标状态被展开为连续的状态区间。Device cycle 按 1 ns 解释，
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
跨 Context 的 immutable Module 共享与 private Module 隔离、socket 对象、token 直传和
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
首个不可恢复的设备 I/O 错误；XDMA/FUSE endpoint 返回 `EIO`、`ENODEV`、`EBADF` 等
错误时，Server 无需额外信号便会自动退出，并跳过已经不可能完成的 Device 侧清理。

需要由脚本同时启动 cmodel 与 `dayan_server` 时运行：

```sh
./run_cmodel_server.sh
```

可用 `RVTNG_CMODEL_DIR`、`RVTNG_QEMU`、`NPU_DEVICE_PREFIX` 和
`NPU_CMODEL_HARTS=1..4` 覆盖。4-core smoke 同时验证 BL0/BL1 启动、API ELF 动态
重定位、socket/token H2D/D2H、`kSingleCore` grid、`kFourCore` Cluster grid，以及
`kFourCore` 对 `task_dim.x == 4` 的约束，以及 profile 满缓冲/尾批次的端到端传输。
