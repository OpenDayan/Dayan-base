# NPU cooperative OS skeleton

这是一个面向 RISC-V NPU Device 的最小裸机系统骨架。当前版本已经建立启动、
每核资源隔离、协作式调度、mailbox RPC、同步异常处理、张量外部中断、基础 libc
和堆管理。系统不使用时钟抢占：mailbox 的 MSIP 由专用 proc 协作式轮询；张量错误
通过机器外部中断进入 trap，同步异常仍通过 `mtvec` 处理。

源码按运行位置组织：`device/` 包含 BL0/BL1、Dayan OS、Device API 和 Device
测试；`host/` 包含 dycc、Runtime、Service 与 HAL；`demo/` 保存同时使用 Host 和
Device 代码的算子示例。顶层 Makefile 只负责调度这些子项目。

## 启动流程

1. `device/bl0/bootrom.S` 从当前 hart 的 mailbox 读取入口地址，并把 hart id 留在 `a0`。
2. `device/bl1/bl1.S` 为该 hart 选择独立栈，令 `tp/current` 指向 CPU-local 中的 init
   thread，清零该区，将 `gp/shared` 置空，配置独立 trap 栈与 `mtvec`，仅启用张量
   机器外部中断并初始化浮点状态。
3. `device/boot/boot.c:init()` 初始化平台、调度器、该 hart 的链接期初始堆，以及 RPC
   收发 proc 和 profile proc。
4. BL1 调用 `device/sche/sche.c:scheduler()`，此后不再返回。

默认构建假设四个 hart 运行四份相互独立的 OS 状态：

| 区域 | 默认配置 | 用途 |
| --- | --- | --- |
| RAM | `0x80000000`, 64 MiB | 镜像、CPU-local、每核栈和每核初始堆 |
| CPU-local | 4 KiB / hart | 每核 scheduler、allocator、RPC、trap 状态与 init thread |
| init 栈 | 64 KiB / hart | BL1、初始化和 scheduler 的每核栈 |
| RPC 栈 | 4 KiB × 2 / hart | RPC receive proc 和 send proc 的私有栈 |
| profile 栈 | 4 KiB / hart | proc profile 满缓冲上报 proc 的私有栈 |
| trap 栈 | 4 KiB / hart | 同步异常入口使用的独立栈 |
| 初始堆 | 1 MiB / hart | 红黑树分配器的第一个 region |
| Host coherent window | `0x00400000`, 4 MiB | Host 按 proc 动态切分，基址由 `gp/shared` 访问 |

固件 RAM 地址集中在 `device/Makefile` 中，接入真实硬件时应覆盖 `RAM_BASE` 和
`RAM_SIZE`。coherent memory 不属于固件链接布局：算子提交用量后由 Host 动态分配，
创建 proc 时把分配基址交给内核。`COHERENT_BASE=0x00400000` 和 4 MiB 的
`COHERENT_SIZE` 只描述硬件窗口并用于运行时地址校验，不会传给链接器或产生本地预留。

`tp` 通过全局寄存器变量 `current` 始终解释为布局固定的 `struct thread *`，而不是
`struct cpu *`。公共 thread
依次保存 32-bit ID、32-bit state、当前 CPU 指针、`sp`、恢复 `pc` 和 32-bit type。
init thread 的 ID 与 type 都是 0；其它类型把 `struct thread` 作为结构体首成员，并
使用非零 type。每个独立 OS 实例的 scheduler 从 1 开始统一分配后续 ID。上下文切换
同步切换 `tp`，thread 的 CPU 指针使内存分配器等每核服务能找到正确的私有状态；
proc 一旦绑定到某个 hart，当前版本不允许迁移。
由于 `tp` 已承担 `current` ABI，固件和算子都不能使用 ELF TLS；构建会拒绝
TP-relative/TLS relocation，链接脚本也会拒绝 `.tdata` 和 `.tbss`。

thread state 的公共值从 `THREAD_RUNNING == 0`、`THREAD_READY == 1` 开始，其余值
表示不同等待原因。`thread_init()` 构造期间短暂使用 `THREAD_WAIT_START`；proc 初始化
完成后直接进入 `THREAD_READY`，运行期使用 `THREAD_WAIT_POLL`、
`THREAD_WAIT_EVENT`、`THREAD_WAIT_PRODUCE`、`THREAD_KILL_WAIT_PRODUCE` 和
`THREAD_WAIT_EXIT`。CPU 私有结构聚合 scheduler、allocator、RPC 和 trap 状态；hart ID
由平台直接读取 `mhartid`，不在 CPU-local 中重复保存。

普通 `.data`/`.bss` 会被链接脚本拒绝，以免不一致缓存下多个 hart 意外共享
可变状态。每核状态应放入 `struct cpu`。固件侧同样拒绝 `.coherent` section；跨核
对象只能放入 Host 给当前 proc 分配的 coherent allocation，并从 `shared` 基址寻址。

## 构建

需要本机 `clang`、`ld.lld`、LLVM binutils 和 `qemu-system-riscv64`：

```sh
make
make check
```

`make check` 会先执行独立脚本中的 ELF/ABI 静态检查，再把 `device/tests/` 中的 list、
RBTree、allocator、math、RPC ABI 和 printf 测试编译为 RV64 bare-metal ELF，并在
QEMU `virt` 上逐一运行。QEMU 不在 `PATH` 时可显式指定：

```sh
make check QEMU_SYSTEM_RISCV64=/path/to/qemu-system-riscv64
```

这些测试覆盖标准 RV64IMAF 上可独立执行的 Device 基础组件；Dayan 自定义执行单元和
Host↔Device 完整链路仍由 `host/driver/run_cmodel_smoke.sh` 在 RVTNG cmodel 中验证。

默认目标为 `rv64imaf_zicsr_zifencei` / `lp64`，使用标准 RV64 `lp64` 调用约定，
同时允许编译器在函数内部使用 F 指令。`zicsr` 和 `zifencei` 是裸机启动所需的系统
扩展。固件输出位于 `build/device/`：

`gp` 专用于全局寄存器变量 `shared`，表示当前 proc 的 Host coherent allocation
基址；它不再承担 RISC-V small-data global pointer 的传统职责。BL1 为 scheduler/init
设为 0，新 proc 从启动帧装入 Host 给定值，之后随上下文保存和恢复。链接脚本不定义
`__global_pointer$`，固件和运行时加载的算子 `.o` 均须使用
`-mno-relax -msmall-data-limit=0`，并且不得包含 GP-relative relocation；普通数据以及
对系统/lib 符号的引用仍由加载器处理 PC-relative relocation。对 `shared` 的显式访问
则由编译器直接生成 `gp` 寄存器读操作。
固件链接不启用 section GC，确保未被固件自身调用的公共 API 仍保留在静态符号表中，
可供运行时算子重定位使用。

- `npu-os.elf`：供加载和调试的 ELF；
- `npu-os-api.elf`：与固件地址一致、但符号表只保留公开 API 的算子链接视图；
- `npu-os.bin`：只包含可加载镜像的平坦二进制；
- `npu-os.objdump`、`npu-os.map`：反汇编和链接布局。

### 安装布局

GNU 风格安装默认使用 `/usr/local`，也可以选择独立 SDK 前缀或先安装到打包目录：

```sh
make install
make install PREFIX=/opt/dayan
make install PREFIX=/usr DESTDIR="$PWD/package-root"
```

默认安装内容如下；所有路径都会随 `PREFIX` 改变：

| 内容 | 默认路径 |
| --- | --- |
| `dycc` driver | `/usr/local/bin/dycc` |
| `dyperf` profiler | `/usr/local/bin/dyperf` |
| Host 公共头文件 | `/usr/local/include/{dayan_*,npu_*}.h` |
| Host runtime 静态/动态库 | `/usr/local/lib/libDayanRuntime.{a,so}` |
| `pkg-config` 元数据 | `/usr/local/lib/pkgconfig/dayan-runtime.pc` |
| Device 交叉编译头文件及 dycc 内部资源 | `/usr/local/lib/dayan/` |
| BL1、公开 API ELF 和固件二进制 | `/usr/local/lib/dayan/firmware/` |

Device 的 `stdio.h`、`stdlib.h` 等 freestanding 头文件不会放进系统
`/usr/local/include`，避免覆盖 Host libc；它们由安装后的 `dycc` 从私有 SDK 目录自动
加入。`DESTDIR` 只用于发行包暂存。安装后的 driver 会相对自身定位 SDK，也可用
`DAYAN_ROOT` 或 `--dayan-root` 显式覆盖。

Host 应用可用 `pkg-config --cflags --libs dayan-runtime` 获取编译和动态链接参数；静态
链接时使用 `pkg-config --static --cflags --libs dayan-runtime`。

### dycc 算子编译器

`host/dycc` 提供类似 `nvcc` 的 Dayan C++ compiler driver。源码使用 `__host__`、
`__device__` 和 `__kernel__` 标注；未标注的全局函数进入 Host 与 Device 两侧，所有
全局变量则必须显式标注所属侧。算子源文件必须显式包含 `<dayan_kernel.h>`，以引用
`npu_kernel_launch_attrs_t` 等公共类型。构建与调用示例：

```sh
make dycc
host/dycc/dycc -c demo.cpp -o build/demo.o
```

带 `-c` 时输出可供后续链接的 Host `.o`；不带 `-c` 时，dycc 会像 gcc 一样继续调用
Host clang++ 并自动链接 `libDayanRuntime.a`：

```sh
host/dycc/dycc operator.cpp main.cpp -o build/app
```

编译阶段产生的 Host `.o` 内部已经嵌入 RISC-V Device `.o`。自动生成的同名 kernel
wrapper 追加 `npu_kernel_launch_attrs_t *` 参数，并以线程安全的 static descriptor
缓存完成延迟加载。`dayan_server` 启动时注册 `npu-os-api.elf`；Host Runtime 只通过
socket 提交 package 与 logical task，不接触固件符号表或物理拓扑。Host driver 产出
`libDayanRuntime.a` 与 `libDayanRuntime.so`。
完整选项、切分规则和同步/异步调用示例见 [`host/dycc/README.md`](host/dycc/README.md)。
- `host/driver/hal/`：多 Cluster/Core 设备访问与物理内存管理；
- `host/driver/server/`：BL1/RPC Service、socket daemon 与 logical grid scheduler；
- `host/driver/runtime/`：不暴露物理拓扑的应用对象接口；
- `build/host/driver/`：`dayan_server`、Dayan Runtime/HAL/Service 库。

### dyperf proc profile

`host/dyperf` 通过 Runtime 开启 Device proc 状态 trace，并输出 Chrome Trace Viewer
JSON。默认申请一个 16 MiB 的全局 profile allocation，Service 按已启动 Hart 切成互不
覆盖的 FIFO；每个 24-byte 原始事件包含 cycle、proc ID、状态和固定 16-byte Proc 名称，
但不附加物理拓扑。采集时 Cluster/Hart 只在
Runtime 的批次头中返回，用来为每个 Hart 建立独立时间线：

```sh
make dyperf
build/host/dyperf/dyperf --duration-ms 1000 -o trace.json
```

每个 Hart 映射为 Trace Viewer 的独立 pid，每个 Device proc ID 映射为 tid，并用 Proc
名称设置 `thread_name`；时间线直接展示 RUNNING、READY 和各种等待/退出状态区间。
profile 启动时先记录所有现存 proc 的初始状态，随后每次状态变化写入目标状态；采样中
新建的 Proc 也会从带名称的 `WAIT_START` 开始记录。一个 cycle 按 1 ns 处理。
详细选项见 [`host/dyperf/README.md`](host/dyperf/README.md)。

### Kernel demos

[`demo/`](demo/README.md) 提供可独立构建和运行的基础 kernel。当前单文件
`print_demo.cpp` 示例在 Device 上打印每个 logical task 的三维 grid ID、grid 维度、
task mode 和 Host 传入的 7 个参数：

```sh
make demo-check
make -C demo run
```

后端已经单独启动时，可用通用脚本启动 Runtime server。它只等待 XDMA 兼容设备文件，
不关心设备来自 cmodel、`npu-cluster-sim` 的 RTL 模拟器还是硬件：

```sh
NPU_DEVICE_PREFIX=/tmp/virtual_npu ./run_server.sh
```

需要由一个脚本同时启动 cmodel 与 Runtime server 时运行：

```sh
./run_cmodel_server.sh
```

脚本默认给 QEMU DRAM 和 HAL Normal memory 配置 4096 MiB，可用 `NPU_CMODEL_MEMORY_MIB` 同时覆盖两者。BL1 的 `RAM_SIZE` 仍只描述固件链接区域。

脚本就绪后，在另一个终端运行：

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock build/demo/print_demo
```

Device `printf` 经对应 Runtime Session 返回并由应用进程输出，不会写到
`dayan_server` 的标准输出。

配置示例：

```sh
make HARTS=4 STACK_SIZE=0x20000 HEAP_SIZE=0x200000 \
     RPC_STACK_SIZE=0x1000 TRAP_STACK_SIZE=0x1000 \
     RAM_BASE=0x80000000 RAM_SIZE=0x08000000 CONTROL_BASE=0
```

## 子系统约定

`device/include/container_of.h`、`device/include/list.h` 和
`device/include/rbtree.h` 提供一组独立实现的
Linux 风格 intrusive 泛型。当前兼容常用的 `container_of/list_entry`、双向循环
`list_head`，以及 `rb_root/rb_node/rb_link_node/rb_insert_color/rb_erase/rb_entry`
接口；不引入 Linux 内核源码、RCU、锁、内存屏障、poison 或调试配置。类型相关操作
由宏在编译期展开，基础指针操作使用 `static inline`，RBTree 平衡代码位于
`device/lib/rbtree.c`。

`device/sche/` 的调度队列和 `device/mem/` 的 region 集合都使用 `list_head`。
`device/mem/` 的空闲块
使用按块大小排序的 intrusive RBTree 做 best-fit 分配，并通过边界信息合并
相邻空闲块。初始 region 永不归还；堆不足时通过 `RPC_HEAP_ACQUIRE` 请求 Host，
Host region 完全空闲后通过 `RPC_HEAP_RELEASE` 立即归还。Host 返回的 region 必须不小于
请求值并能通过地址、对齐和重叠检查；无效 region 会立即通过 release RPC 回滚，且不会
进入 allocator 的 RBTree。动态 region 接收后先 invalidate 本核缓存；归还前先 clean，
Host 确认回收后再 invalidate，避免非一致性缓存中的旧行污染后续所有者。

Cache 接口保留为弱符号，默认实现采用当前平台的 64-byte cache line 和
`cbo.flush`/`cbo.inval` 指令编码。RPC 在发布请求前 clean Device 缓存，在 Host
完成事务后 invalidate 描述符和响应区。

### RPC 与异常

RPC 命令固定为一个 64-bit word：高 16 位是 opcode，低 48 位是参数。默认
`CONTROL_BASE=0` 时，每个 hart 使用下列寄存器；配置非零 `CONTROL_BASE` 时，表中
地址均作为 offset 加到该基址上。

| 方向 | 命令 mailbox | doorbell / 完成确认 |
| --- | --- | --- |
| Host → Device | `0x1000 + hart * 8` | Host 置位 `msip[hart]`（`0x20000 + hart * 4`）；Device 处理后写 0 |
| Device → Host | `0x2020 + hart * 8` | Device 向 `IRQ_STATUS_SET` (`0x2008`) 写 `1 << hart`；Host 通过控制器清除状态位 |

Device 发送前后都轮询 `IRQ_STATUS` (`0x2000`) 中自己的 bit，只有 Host 清除前一笔
事务后才复用 mailbox。初始化还会清除本核遗留状态，并通过 `IRQ_ENABLE_SET`
(`0x2010`) 使能对应的 Host IRQ 输出。上述 doorbell 只用于握手，不会触发 NPU 上的
抢占：BL1 用 `csrw mie, 0x800` 仅启用张量机器外部中断，RPC MSIP 保持轮询。

每个 hart 有三个内核 proc。receive proc 以 `proc_wait_poll()` 等待本核 `msip`，读取
Host 命令，调用弱的 `rpc_receive(opcode, arg, &response)` 分派接口，并在需要回复时
把响应加入发送队列，最后清除 `msip`。send proc 独占本核发送队列；队列为空时使用
`proc_wait_event()`，有事务时写 Device → Host mailbox，等待 Host 清除状态位，再执行
事务 cleanup 并唤醒调用者。receive handler 返回 1 表示发送 `response`，返回 0 表示
无响应，负值表示请求已消费但失败。handler 在清除当前 `msip` 前运行；若它发起阻塞式
反向 RPC，Host 必须在等待 response 的同时继续服务 Device→Host mailbox。仓库内置的
create-proc handler 在初始堆不足时可能通过 `RPC_HEAP_ACQUIRE` 嵌套扩堆，配套
`host/driver/server` Service 支持这一流程，并用 `(cluster, core)` 标识请求来源。

`rpc_call()` 是 Device 主动请求的统一入口。其低 48 位参数指向固定 40-byte 的
`struct rpc_packet`，字段依次是 request 地址/长度、response 地址/长度和 Host 回填的
有符号 status。当前 opcode 1–8 分别为字符串输出、清空输出缓冲、申请堆 region、
归还堆 region、异常报告、张量中断报告、root proc 退出通知和 profile 满缓冲上报。
异常与张量中断报告不使用 packet：前者的低 48 位参数直接指向一份
稳定的 312-byte `struct trap_record`，后者直接指向 176-byte
`struct tensor_interrupt_record`。输出和清空操作的 packet request 指向固定
24-byte 的 `rpc_print_request`，其中包含 hart ID、proc ID、数据地址和长度，Host 因而
可以分别管理每个 proc 的输出。堆申请及释放分别使用固定 8-byte 请求和 16-byte
响应/释放描述符，Device 侧由 `rpc_heap_acquire/release()` 统一封装。

Host 创建算子使用 opcode `0x0100`。其参数指向 Host 放在普通 Device DRAM 中的固定
224-byte `rpc_proc_create_request`，包含 entry、栈/输出缓冲容量、shared allocation、
0–8 个 ABI 参数、`task_id[3]`、`task_dim[3]`、`task_mode`、最大子 Thread 数、URAM
base/size、64-bit Tensor Context、固定 16-byte Proc 名称以及完成状态。BL1 校验
task mode 只能为 1 或 4，并在 mode 4 时要求 `task_dim.x == 4`；随后分配托管 proc 并
回复 opcode `0x8100`。root 创建时会物理摘下 `1 + max_child_threads` 个静态 slot；
子孙创建、退出与 join 不再经过 Host。每个 proc 退出都 flush/CLEAR 自己的输出；只有
Host root 通过 32-byte `RPC_PROC_EXIT` 请求主动上报 64-bit 返回值与 cookie。随后
scheduler 在自己的栈上调用 cleanup，把描述符从 `RUNNING` 更新为 `EXITED` 或
`FAILED`。因此 Host 必须保留描述符直到终态，不能只把 create response 当作执行完成。

Host 管理目标 Hart 使用 `RPC_PROC_PS` (`0x0102`) 和 `RPC_PROC_KILL_ALL` (`0x0103`)，
响应分别为 `0x8102`、`0x8103`。PS 固定返回三个系统 proc（RPC receive/send 与
profile）和至多 16 个算子 proc
的名称、状态、父 ID、直属子计数、logical task、返回值与 cookie。kill-all 保留 init/RPC/profile
proc，摘除其它 proc 的 ready/poll/event/system-wait 节点；非 Produce slot 立即释放，
等待 Produce 的 slot 进入 `THREAD_KILL_WAIT_PRODUCE`，收到硬件 token 后才释放。

Host 覆盖动态模块前，先通过 `RPC_DCACHE_FLUSH` (`0x0105`) 让所有已启动 Hart
失效目标模块的 D-cache 范围；全部返回 `0x8105` 后才写入新的 text/rodata。写入完成后
再使用 `RPC_ICACHE_FLUSH` (`0x0101`) 发布代码，并等待 `0x8101` 响应。`fence.i` 只作用于
当前 hart，因此两阶段发布都逐个处理所有已启动 hart；创建 proc 不再重复刷新 cache。

双向标量 RPC descriptor 都隔离 cache line。Device 发起的 packet、request、response
各占一个独立的 64-byte 对齐 slot；Host 只写 response/status，Device 在收到确认后
invalidate 并复制回调用者。Host-owned 控制描述符（proc create、PS、kill、profile
config 和 D-cache flush）
从 64-byte 对齐地址开始，allocation 大小也向上取整到 64-byte 粒度，避免与其它对象共享
cache line；opcode 唯一确定描述符布局和长度。Device 在解析前 invalidate，
请求字段在生命周期内保持只读，只在 RPC 响应或 proc family 终态写入结果和 state，并在
通知 Host 前 clean。Host 读取到响应或终态后才能释放对应 allocation；提交前失败可立即
释放，一旦尝试发布命令却未收到匹配响应，allocation 不再进入分配池，只在 HAL 关闭时
统一回收。

同步异常从 `mtvec` 进入独立的每核 trap 栈，保存 x0–x31（其中 x2 为异常前的 sp）、
`mstatus`、`mepc`、`mcause`、`mtval` 和 `fcsr`。C handler 再附加版本、记录大小、
hart ID 与当前 thread 的 ID/type/state，异步排入 RPC 发送队列，然后把故障 proc 置为
退出状态并回到 scheduler。init/scheduler 上下文异常、嵌套异常，或缺少有效
`current` 时无法安全恢复，系统会停机；异常报告无法入队时也会停机，避免静默丢失
故障信息。

`device/sche/` 是有独立 proc 栈的非抢占式调度器。`switch_context()` 在当前栈保存
`ra`、`s0-s11`、`gp` 和 `fcsr`，把保存帧的 `sp` 及恢复入口 `pc` 写入当前 thread，
再加载目标 thread 的 `sp/pc` 并同步更新 `tp/current`。恢复旧 proc 时从切换帧恢复
`gp/shared`；首次执行新 proc 时由 bootstrap 从启动帧加载它。软浮点 `lp64` ABI 没有需要跨 C 函数调用
保留的 F 寄存器；有存活浮点值时，编译器会在调用 `proc_yield()` 等函数前自行溢出，
而每个 thread 的浮点控制状态由 `fcsr` 保存。

`proc_init(proc, entry, stack, stack_size, shared_base, printf_buffer,
printf_capacity, argc, ...)` 使用调用者提供的栈区间、Host 分配的 shared 基址，以及
proc 私有输出缓冲区构造任务，
栈顶按 16 字节向下对齐，支持 0 到 8 个寄存器参数。可变参数是底层 ABI 机器字，调用者
必须逐个按 `uintptr_t` 传入；后续类型安全封装可负责指针转换、浮点位模式打包以及自动
加入 proc 指针。实际实现位于接收 `va_list` 的 `proc_vinit()`；`proc_init()` 仅负责
建立和销毁参数列表，因此其它可变参数封装可以直接调用 `proc_vinit()`，不需要中间
数组。bootstrap 开头按 `a7` 到 `a0` 连续排列八条定长加载指令，新 proc 的
初始 `pc` 为 `proc_bootstrap + (8 - argc) * 4`，因此只恢复实际提供的参数寄存器，再
加载任务入口和返回地址。未使用的高位参数寄存器不作初始化。该栈与输出缓冲区在 proc
到达 `THREAD_WAIT_EXIT` 前必须一直有效；传入空指针和 0 可为不使用 stdio 的内核 proc
禁用输出。

例如，底层接口创建一个具有两个机器字参数的 proc：

```c
proc_init(&proc, (proc_fn)task_main, stack, stack_size, shared_base,
          printf_buffer, printf_capacity, 2,
          (uintptr_t)&proc, (uintptr_t)arg);
```

`proc_init()`/`proc_vinit()`/`proc_init_args()` 成功时直接把 proc 加入当前 CPU 的
ready queue；最后一个接口供 RPC 这类已有机器字数组的调用者使用。可选 cleanup 在
proc 停止使用自己的栈后由 scheduler 调用，因而可以安全回收托管栈。运行中的
proc 可调用 `proc_yield()` 主动让出。`proc_wait_poll(poll, arg)` 进入轮询等待队列，
scheduler 每轮调用其 predicate，返回非零后移回 ready queue；该队列非空时不会执行
`wfi`。`proc_wait_event(event)` 则立即把当前 proc 的 intrusive `node` 挂入指定事件
链表，scheduler 不遍历它；事件所有者调用 `proc_event_wake_one/all()` 或
`proc_wake()` 将等待者移回 ready queue。传入空指针会使用 scheduler 的默认外部事件
链表。`entry()` 正常返回等价于调用 `proc_exit()`，不会被重新排队。普通 proc 仍由
所有者管理生命周期；Host RPC 创建的 root proc 按资源描述符预留 `1 + max_children`
个 proc slot。每个 Hart 固定有 16 个算子 proc slot，root 也计入，init、两个 RPC
proc 和 profile proc 不计入，因此单个 kernel 最多声明 15 个子 Thread。
`dayan_create_thread(name, entry, argument, resources)` 创建的子 Thread 保持在同一 hart，
拥有不超过 15 bytes 的名称，继承 logical task、`gp/shared` 与 URAM，并返回正的 Thread
ID。传入空资源表示不占用 TCSR；显式资源由固件映射为该 Thread 从逻辑组 0 开始的
独立 Context。`dayan_thread_join(id, &result)` 精确等待直属子 Thread，
`dayan_thread_join_all()` 等待全部直属子 Thread；每个执行对象退出前都会隐式
join-all，因此 root 进入终态时整棵子树已经完成。`dayan_thread_exit(result)` 可显式
设置 64-bit 返回值。
托管 cleanup 在退出后把 slot 归还静态池；slot 的 64 KiB 栈和 4 KiB 输出缓冲在
固件链接时已经按 Hart 预留，不走运行时堆分配。当前没有抢占或跨 hart 迁移。

## 算子 API 与同步

`device/include/api/` 是源码树中外部算子唯一受支持的头文件根目录。它包含当前的小型 libc 标准头，
以及只声明 Dayan 系统接口的 `dayan_sys.h`；该头文件不会包含或暴露 `proc`、`thread`、
`cpu` 和 scheduler 等内部类型。算子应使用 `-Idevice/include/api` 编译，不能加入内部
`device/include/`。旧的 `device/lib/include/` 仅保留为转发层，公开声明以
`device/include/api/` 为准。

所有公开 libc 函数和 DAPI 分别标注到 `.text.api.libc`、`.text.api.dapi`。完整的
`npu-os.elf` 仍保留全部系统符号，供固件加载和调试；构建同时根据
`device/dayan-api.symbols` 生成 `npu-os-api.elf`，其中调试信息及非白名单符号已经移除。
运行时算子重定位器应只读取这个 API ELF 的符号表，而不能用完整固件 ELF 解析算子
未定义符号。`make check` 会检查白名单、两个 API section 和 API ELF 的全局符号集合
完全一致，并分别用 C11 与 C++17 在只有公开 include 路径的环境下编译算子样例。

第一批 proc 包装为 `dayan_yield()`，以及 `dayan_wait_lsu/mat/vec/macro()` 四个同步
入口。它们把当前 proc 的 `tp` 作为 token，分别通过 BASE 单元地址 1、2、3、4 发出
Produce，然后进入默认 event 等待；`dayan_wait_dma()` 是 LSU 入口的兼容别名。
scheduler 每轮通过 BASE 全局地址 0 的 pop 读取所有已完成 token，并由
`proc_scan_produce()` 调用内部 `proc_wake()`。event 链表本身仍不遍历；只要
其中存在等待者，scheduler 就持续扫描 Produce，而不会进入 `wfi`。每次等待占用一个
Produce 项，软件仍须遵守硬件的全局 32 项和单执行单元 2 项的无阻塞保证，不能无界
发射同步项。

`dayan_uram_base()` 和 `dayan_uram_size()` 返回 Service 为当前 kernel 实例在本 Core
分配的私有 URAM；默认每 Core 容量为 1.5 MiB，地址窗口从 `0x00800000` 起并以 2 MiB
为 Core stride。声明、分配和释放大小以 32 KiB 为粒度。
`dayan_create_thread(name, entry, arg, resources)` 使用 kernel 描述符预留的 slot 在同
Core 创建一个有名称的单参数子 Thread；子 Thread 继承 URAM，TCSR 必须显式申请，达到
声明上限或 TCSR 池耗尽时返回失败。

### 推荐的算子分片模型

普通激活和 Element-wise 算子优先使用无子 Thread 的顺序 Proc：Host 将输入/
输出分成多个逻辑分片，每个 Task 坐标启动一个 Proc，Proc 在自己的循环中执行
`Load -> 计算 -> Store`。用
`DAYAN_KERNEL_TASK_DESC(kernel, 4, 0, ...)` 可声明每 Hart 最多共驻 4 个这类
Proc。它们在执行单元等待点主动让出，Scheduler 使其 Load、计算和 Store 在同
Hart 上交错执行，从而提高 LSU 与计算单元的利用率。

资源描述符按单 Proc 声明 URAM 和 TCSR 最坏需求；Server 仅在 Proc slot、URAM
与四类 TCSR 都足够时允许共驻。每个 Proc 都拥有独立 Context，算子仍然使用
逻辑配置组 0。对分片独立的算子，不应为 Load/计算/Store 各创建一个 Thread。

多 Thread 保留给需要核间数据广播、远程 URAM 复制或显式多阶段生产者/消费者
流水的特殊算子，典型例子是需要在四个 Core 间广播 A/B tile 的矩阵乘。这类
算子再使用 `kFourCore`、`shared` resource 和独立的 loader/copy/MAT/store Thread。

`dayan_cluster_barrier(4, barrier)` 只用于四 Core 调度模式。barrier 是放在当前
`shared` allocation 内的零初始化 8-byte 对象，四个 lane 到达后以 release/acquire
语义推进 generation，因此同一对象可以重复使用。

资源分为单核和跨核两种类型，两者提供相同的
`FREE -> BUSY -> READY -> FREE` 状态机。`alloc()` 设置生产者和消费者计数；生产者为
0 时直接进入 READY，消费者必须非零。最后一次 `produce()` 进入 READY，最后一次
`consume()` 回到 FREE。所有返回 `int` 的资源接口均以 0 表示成功、负值表示参数或
状态错误。

`dayan_resource_t` 只允许当前 hart 的 proc 共享，可放在该 hart 的栈或普通私有堆中。
它内嵌 lock/ready/free 三条核私有事件链表；`wait_ready/free()` 直接挂入对应链表，
状态转换时由资源操作唤醒，而不需要 scheduler 扫描。其状态、计数、上锁和解锁全部
使用普通 load/store，不包含 `__atomic`、AMO 或 fence；非抢占调度保证资源操作不会
与本核另一个 proc 同时执行。初始化后，存在等待者的资源对象不得复制、移动或再次
初始化。

`dayan_shared_resource_t` 供多个 hart 共享。producer/consumer 剩余计数打包在同一个
64-bit 状态字中：`alloc()` 使用 `amocas.d` 从 FREE 原子安装新一代计数，`produce()`
和 `consume()` 分别使用一条 `amoadd.d` 递减计数，全程没有自旋锁。它不能保存只对某个
hart 私有的 proc 链表节点，因此
`wait_ready/free()` 使用 `proc_wait_poll()` 检查一致性状态。`dayan_sys.h` 以全局寄存器
变量的形式把 `shared` 直接绑定到 `gp`；它没有内存符号，也不是固件定义的普通全局
变量，算子代码不得改写它。算子应在 Host
分配区内定义自己的布局，
例如把第一个对象放在基址处：

```c
dayan_shared_resource_t *resource = (dayan_shared_resource_t *)shared;
```

一个 hart 应在发布对象前调用一次 `dayan_shared_resource_init()`。该函数要求 `gp`
位于 `0x00400000` 开始的 4 MiB 硬件窗口内，并按 RPC 传入的当前 allocation 精确长度
检查对象边界和类型对齐。算子仍须让自己的布局与 `atomic_cache_size` 声明一致。不能用
普通 `malloc()` 为跨核资源分配内存，也不能在算子 `.o` 中静态定义 `.coherent`
section。

`device/lib/` 提供 `mem*`、常用字符串函数、`malloc/calloc/realloc/free`，以及整数、字符串
和指针格式的 `printf/snprintf`。数学部分先提供不依赖软双精度 runtime 的单精度
`fabsf/copysignf/fminf/fmaxf/sqrtf/rsqrtf/expf/logf`。浮点格式化暂未实现，
`printf/puts/putchar` 共享当前 proc 在创建时配置的有限缓冲区；缓冲恰好填满时自动通过
RPC 发送，未填满的内容跨调用保留，直到 `fflush(stdout)` 或 `fflush(NULL)`。RPC
失败时普通 flush 会保留缓冲内容以便重试。proc 退出时会做一次 best-effort flush，
随后发送 CLEAR 清理该 kernel root 的 Host 输出流；若要求确认尾部输出已经送达，仍应
在返回前显式调用并检查 `fflush`。

## Host 用户态驱动

`host/driver/` 参考旧 `npu-runtime` 的 XDMA 文件访问方式，并按 `hal`、`server`、
`runtime` 分层。HAL 显式使用 `(cluster, core)` 访问 register/mailbox、管理普通与
coherent 物理 region；Server 为每个 Cluster 加载 BL1、服务反向 RPC，并把 Runtime
提交的 logical task 调度到物理 Core。公开 Runtime 只提供 Context、Buffer、Stream、
Module、Kernel 和 Event，不展示 Cluster/Core 或物理地址，动态库只导出 `npu_`
前缀的应用 ABI。Server 打开具体 HAL backend 后把 opaque HAL handle 注入 Service；
Runtime 只依赖无拓扑语义的 data-plane provider ABI，因此后续可以增加 ioctl backend。

任务配置使用 `NPUTaskDim3 {x,y,z}` 与 `NPUTaskMode`。mode 1 以单 Core 为调度粒度，
将 `x*y*z` 个 task 分发到可用 Core；mode 4 以包含四个 Core 的 Cluster 为调度粒度，
要求 `task_dim.x == 4`，每个 `(y,z)` group 的四个 x lane 同时占用同一 Cluster。
Device 算子通过 `dayan_task_id/dim/mode()` 获取逻辑坐标，不感知物理放置。

每个 `__kernel__` 旁必须定义
`DAYAN_KERNEL_DESC(kernel, max_child_threads, atomic_bytes, uram_bytes,
lsu_tcsr, mat_tcsr, vec_tcsr, macro_tcsr)`。dycc 在 Device object 中保留
`<kernel>__desc`，并把相同资源值写入 Package。Server 对同一 Core 上所有 resident
Task 的 proc slot、Atomic Cache、URAM 和四类 TCSR 声明求和后再调度。每个 invocation
占用 `1 + max_child_threads` 个 slot，同一 Hart 的总和不得超过 16。

Runtime 与 Server 使用 Unix socket 控制协议。H2D/D2H payload 不进入 socket：Runtime
先在 data socket 申请受范围和方向约束的 token，再用 HELLO 时经 `SCM_RIGHTS` 导入的
HAL data-plane FD 直接搬运，最后回报同一 token；Server 在完成回报前不会回收对应
allocation。详细架构、生命周期、API 和 cmodel 命令见
[`host/driver/README.md`](host/driver/README.md)。
