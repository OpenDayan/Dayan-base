# NPU cooperative OS skeleton

这是一个面向 RISC-V NPU Device 的最小裸机系统骨架。当前版本已经建立启动、
每核资源隔离、协作式调度、mailbox RPC、同步异常处理、张量外部中断、基础 libc
和堆管理。系统不使用时钟抢占：上下文切换时通过 `mip.MSIP` 选择 RPC 接收线程；张量错误
通过机器外部中断进入 trap，同步异常仍通过 `mtvec` 处理。

源码按运行位置组织：`device/` 包含 BL0/BL1、Dayan OS、Device API 和 Device
测试；`host/` 包含 dycc、Runtime、Service 与 HAL；`demo/` 保存同时使用 Host 和
Device 代码的算子示例。顶层 Makefile 只负责调度这些子项目。

## 启动流程

1. `device/bl0/bootrom.S` 从当前 hart 的 mailbox 读取入口地址，并把 hart id 留在 `a0`。
2. `device/bl1/bl1.S` 为该 hart 选择独立启动栈，清零 CPU-local，将 `tp/current` 和
   `gp/shared` 置空，配置独立 trap 栈与 `mtvec`，仅启用张量
   机器外部中断并初始化浮点状态。
3. `device/boot/boot.c:init()` 初始化平台、调度器、该 hart 的链接期初始堆，以及 RPC
   收发 proc 和 profile proc。
4. BL1 直接进入 `device/sche/context.S:switch_context()`，首次分派不保存启动上下文。

默认构建假设四个 hart 运行四份相互独立的 OS 状态：

| 区域 | 默认配置 | 用途 |
| --- | --- | --- |
| RAM | `0x80000000`, 64 MiB | 镜像、CPU-local、每核栈和每核初始堆 |
| CPU-local | 8 KiB / hart | `struct cpu`：线程 slot、调度元数据、profile、allocator、RPC 和异常记录 |
| 启动栈 | 64 KiB / hart | BL1 和初始化，首次分派后不再使用 |
| RPC 栈 | 4 KiB × 2 / hart | RPC receive proc 和 send proc 的私有栈 |
| profile 栈 | 4 KiB / hart | proc profile 满缓冲上报 proc 的私有栈 |
| trap 栈 | 4 KiB / hart | 同步异常入口使用的独立栈 |
| 初始堆 | 1 MiB / hart | 红黑树分配器的第一个 region |
| Host coherent window | `0x00400000`, 4 MiB | Host 按 proc 动态切分，基址由 `gp/shared` 访问 |

固件 RAM 地址集中在 `device/Makefile` 中，接入真实硬件时应覆盖 `RAM_BASE` 和
`RAM_SIZE`。coherent memory 不属于固件链接布局：算子提交用量后由 Host 动态分配，
创建 proc 时把分配基址交给内核。`COHERENT_BASE=0x00400000` 和 4 MiB 的
`COHERENT_SIZE` 只描述硬件窗口并用于运行时地址校验，不会传给链接器或产生本地预留。

启动完成后，`tp/current` 指向布局固定的 `struct thread`，保存 type、state、16-bit ID、
父 slot、事件链表节点、`sp/pc`、Context1/Context2 和退出信息。ID 由 slot 和 generation
组成；系统没有独立的 init/scheduler thread。上下文切换
同步切换 `tp`；内存分配器等每核服务根据 `mhartid` 定位 CPU-local 私有状态。proc
一旦绑定到某个 hart，当前版本不允许迁移。
由于 `tp` 已承担 `current` ABI，固件和算子都不能使用 ELF TLS；构建会拒绝
TP-relative/TLS relocation，链接脚本也会拒绝 `.tdata` 和 `.tbss`。

thread state 的公共值从 `THREAD_RUNNING == 0`、`THREAD_READY == 1` 开始，其余值
表示不同等待原因。`thread_start()` 构造后使用 `THREAD_WAIT_START`；root 提交无依赖 Wait 后
进入 `THREAD_READY`，创建回复完成后才可能被选中执行。运行期使用 `THREAD_POLLING`、
`THREAD_WAIT_EVENT`、`THREAD_WAIT_START`、`THREAD_WAIT_EXIT`、`THREAD_WAIT_LSU`、
`THREAD_WAIT_MAT`、`THREAD_WAIT_VEC`、`THREAD_WAIT_MACRO`、`THREAD_YIELD`、
`THREAD_WAIT_JOIN`、`THREAD_WAIT_RESOURCE` 和 `THREAD_KILL_WAIT_PRODUCE`；旧的
`THREAD_WAIT_PRODUCE` 仍保留为兼容状态。CPU 私有结构同时持有这些线程的 slot 和运行期元数据；hart ID
由平台直接读取 `mhartid`，不在 CPU-local 中重复保存。

固件不允许定义独立的可变全局或函数内 static 变量，包括普通 `.data`/`.bss` 和
自定义 section。所有每核持久状态都由 `struct cpu` 持有，线程私有元数据放在其中的
slot 数组，按线程索引访问；不能把多个线程的独立状态合并成一个每核标量。
`struct cpu` 本身也不是 `.bss` 全局变量，而是位于链接器预留的 `.cpu` 空间，
通过 `cpu_self()` 定位。每核预留容量为 2 的幂且覆盖完整 cache line，BL1 负责清零。

栈、printf buffer 和初始 heap 是链接脚本按 Hart 隔离的原始存储预留，不定义 C 全局
数组；其地址由 CPU Local 中的线程上下文、buffer 指针和 allocator 管理。
链接器同时检查这些区域的步长，避免只对齐数组首地址、却在 Hart 边界共享 cache line。
只读常量可以保留在 `.rodata`。固件侧同样拒绝 `.coherent` section；跨核可变对象
只能放入 Host 给当前 proc 分配的 coherent allocation，并从 `shared` 基址寻址。

## 构建

需要本机 `clang`、`ld.lld`、LLVM binutils 和 `qemu-system-riscv64`：

```sh
make
make check
```

`make check` 会先执行 ELF/ABI 静态检查，并验证链接器拒绝可变全局、自定义可写 section、
固件 coherent 对象和未按 cache line 隔离的存储步长，再把 `device/tests/` 中的 list、
RBTree、allocator、math、RPC ABI、printf 和 profile 测试编译为 RV64 bare-metal ELF，并在
QEMU `virt` 上逐一运行。QEMU 不在 `PATH` 时可显式指定：

```sh
make check QEMU_SYSTEM_RISCV64=/path/to/qemu-system-riscv64
```

这些测试覆盖标准 RV64IMAF 上可独立执行的 Device 基础组件；Dayan 自定义执行单元和
Host↔Device 完整链路仍由 `host/driver/run_dayan_emu_smoke.sh` 在 dayan_emu 中验证。

默认目标为 `rv64imaf_zicsr_zifencei` / `lp64`，使用标准 RV64 `lp64` 调用约定，
同时允许编译器在函数内部使用 F 指令。`zicsr` 和 `zifencei` 是裸机启动所需的系统
扩展。固件输出位于 `build/device/`：

`gp` 专用于全局寄存器变量 `shared`，表示当前 proc 的 Host coherent allocation
基址；它不再承担 RISC-V small-data global pointer 的传统职责。BL1 在启动时将它
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

模拟器或 XDMA 硬件已经单独启动时，可用现有脚本启动 Runtime server。它只等待 XDMA 兼容设备文件，
不关心设备来自 dayan_emu、`npu-cluster-sim` 的 RTL 模拟器还是硬件：

```sh
NPU_DEVICE_PREFIX=/tmp/virtual_npu ./run_server.sh
```

HAL 已统一为 Xilinx DMA backend，通过 `driver` 选择 XDMA/QDMA，通过独立的
`profile` 选择 simulator/fpga。上面的脚本仍构造旧 XDMA 参数和节点检查，QDMA 应直接
启动 Server（先加载驱动并启动双向 MM queue）：

```sh
build/host/driver/dayan_server \
  --firmware build/device/npu-os.elf --api-elf build/device/npu-os-api.elf \
  --backend xilinx-dma --dma-driver qdma --dma-profile fpga \
  --dma-device-name qdma01000 --dma-queue 0 --dma-user-irq 0 \
  --dma-completion irq --cluster-count 1 --core-mask 0xf
```

QDMA IRQ 模式需要本地驱动扩展的 `/dev/qdma01000-user_irq0`；若没有此节点，显式
使用 `--dma-completion polling`，HAL 将读取 NPU IRQ status，并在每轮请求休眠 1 微秒。
无需创建 XDMA 名称的符号链接。完整配置与限制见
[`Xilinx DMA 说明`](host/driver/hal/backend/xilinx_dma/README.md)。

需要由一个脚本同时启动 dayan_emu 与 Runtime server 时运行：

```sh
./run_dayan_emu_server.sh
```

脚本默认使用同级的 `dayan-emu` 及其 `deps/QEMU/build/dl-riscv64`，保持 MMU 和
Shared Resource clear 协议匹配；可用 `DAYAN_EMU_DIR` 与 `QEMU_DL_BIN` 覆盖。
脚本默认给 QEMU DRAM 和 HAL Normal memory 配置 4096 MiB，可用 `DAYAN_EMU_MEMORY_MIB` 同时覆盖两者。BL1 的 `RAM_SIZE` 仍只描述固件链接区域。

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

`device/sche/` 的事件等待链表和 `device/mem/` 的 region 集合都使用 `list_head`；就绪与
Named Resource 调度由硬件队列完成，不维护软件 ready/resource list。
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

每个 hart 有三个系统 Thread。`switch_context()` 仅根据本核 `mip.MSIP` 直接选择
receive Thread；它调用 `rpc_receive(opcode, arg, &response)`，同步发送回复并等待 Host
确认，最后清除 `msip`。整个接收事务不会挂起，也不向硬件队列提交接收线程 token。
send Thread 在发送队列为空时使用 event wait，有事务时阻塞写 Device → Host mailbox，
忙等 Host 清除状态位，再执行事务 cleanup 并唤醒调用者。send Thread 检查发送队列和
待上报中断后直接登记等待，不屏蔽中断；Tensor 中断表示不可恢复异常，不保证异常后的执行结果。
receive handler 返回 1 表示发送 `response`，返回 0 表示
无响应，负值表示请求已消费但失败。handler 在清除当前 `msip` 前运行；若它发起阻塞式
反向 RPC，Host 必须在等待 response 的同时继续服务 Device→Host mailbox。
create-proc 只使用静态 slot 和栈，不扩堆、不等待其它线程。配套 Service 用
`(cluster, core)` 标识请求来源，并串行化同核的创建事务。

`rpc_call()` 是 Device 主动请求的统一入口。其低 48 位参数指向固定 40-byte 的
`struct rpc_packet`，字段依次是 request 地址/长度、response 地址/长度和 Host 回填的
有符号 status。当前 opcode 1–8 分别为字符串输出、刷新 Host 输出流、申请堆 region、
归还堆 region、异常报告、张量中断报告、root proc 退出通知和 profile 满缓冲上报。
异常与张量中断报告不使用 packet：前者的低 48 位参数直接指向一份
稳定的 328-byte `struct trap_record`，Host 默认 trap 回调会原子地打印线程身份、
`mstatus/mepc/mcause/mtval/fcsr` 和 x0–x31；后者直接指向 472-byte
`struct tensor_interrupt_record`。硬件中断源编号与 Server 日志名称见
[张量中断源编号](host/driver/README.md#张量中断源编号)。tensor 报告由 send Thread 使用双缓冲直接同步发送，
不创建通用 RPC transaction。输出和清空操作的 packet request 指向固定
24-byte 的 `rpc_print_request`，其中包含 hart ID、proc ID、数据地址和长度，Host 因而
可以分别管理每个 proc 的输出。堆申请及释放分别使用固定 8-byte 请求和 16-byte
响应/释放描述符，Device 侧由 `rpc_heap_acquire/release()` 统一封装。

Host 创建算子使用 opcode `0x0100`。其参数指向 Host 放在普通 Device DRAM 中的固定
416-byte `rpc_proc_create_request`，包含 entry、栈/输出缓冲容量、shared allocation、
0–32 个 ABI 参数、`task_id[3]`、`task_dim[3]`、`task_mode`、最大子 Thread 数、URAM
base/size、64-bit Tensor Context、固定 16-byte Proc 名称以及创建状态。Host 校验
启动参数，BL1 只检查本固件的栈/输出容量及空闲 slot。Device invalidate 描述符，
把启动数据复制到静态 slot 和启动帧，完整 flush 创建结果并提交 root token。
接收线程同步回复 `0x8100`、清除 `msip` 后才切换，因此新 kernel 不会抢在回复前执行。
Host 读取回复后即可释放创建描述符；运行结果只通过 `RPC_PROC_EXIT` 上报。root 创建时会摘下
`1 + max_child_threads` 个静态 slot；
子孙创建、退出与 join 不再经过 Host。每个 proc 退出都通过统一 flush 路径提交剩余输出；
缓冲为空时直接返回，不发送 `WRITE` 或 `CLEAR`。只有 Host root 通过 32-byte
`RPC_PROC_EXIT` 请求主动上报 64-bit 返回值与 cookie。Host 在成功响应前记录终态；
Device 收到响应后立即由 sender 归还 Thread slot。普通 launch
完成只依赖这次通知，不再用 `RPC_PROC_PS` 轮询确认。

Host 管理目标 Hart 使用 `RPC_PROC_PS` (`0x0102`) 和 `RPC_PROC_KILL_ALL` (`0x0103`)，
响应分别为 `0x8102`、`0x8103`。PS 固定返回三个系统 proc（RPC receive/send 与
profile）和至多 16 个算子 proc 的名称、状态、父 ID、直属子计数、logical task、返回值与
cookie。kill-all 保留 RPC/profile proc，摘除 event/system-wait 节点；尚未回到软件的
硬件 token 进入 `THREAD_KILL_WAIT_PRODUCE`，token 返回后再释放对应 slot。这里也包括
`dayan_yield()` 提交的无依赖 Wait token，不只包括执行单元 Produce。
kill 回复中的 `released_slots` 只统计当时已经释放的 slot；尚有 token 的 family
在接收事务结束、dispatcher 取回全部旧 token 后释放。

Host 在启动 BL1 前为每个 Cluster 预留 16 MiB kernel arena，存放模块所有加载节段，
包括 `.bss` 零初始化。模块按 ELF 对齐要求单调分配，卸载及上传失败都不复用已写入的范围；
arena 在 Service 关闭时统一释放，用尽返回 `-ENOMEM`。因此新模块上传无需 D-cache RPC。
上传完成后向该 Cluster 所有已启动 Hart 并行发送 `RPC_FENCEI` (`0x0101`)，全部回复
`0x8101` 后才发布 entry；loader 互斥锁覆盖分配、上传及发布。`argument` 是不解引用的
模块地址 cookie。旧名称 `RPC_ICACHE_FLUSH` 保留为同一 opcode 的别名。
驻留模块复用无需上传或 fence；`RPC_PROC_CREATE` 不再重复执行 `fence.i`。
Profile START、STOP 和失败回滚也通过同一个并发 helper 跨核配置；同一 Hart 仍按 RPC
队列串行执行，所有已提交配置结束后才汇总结果或释放共享 buffer。

双向标量 RPC descriptor 都隔离 cache line。Device 发起的 packet、request、response
各占一个独立的 64-byte 对齐 slot；Host 只写 response/status，Device 在收到确认后
invalidate 并复制回调用者。Host-owned 控制描述符（proc create、PS、kill、profile
config 和 D-cache flush）
从 64-byte 对齐地址开始，allocation 大小也向上取整到 64-byte 粒度，避免与其它对象共享
cache line；opcode 唯一确定描述符布局和长度。Device 在解析前 invalidate，
请求字段在事务内保持只读，只在 RPC 响应前写入结果和 state，并完整 flush。
Device 回复后不再访问描述符；Host 读取响应后即可释放对应 allocation。提交前失败可立即
释放，一旦尝试发布命令却未收到匹配响应，allocation 不再进入分配池，只在 HAL 关闭时
统一回收。

同步异常从 `mtvec` 进入独立的每核 trap 栈，保存 x0–x31（其中 x2 为异常前的 sp）、
`mstatus`、`mepc`、`mcause`、`mtval` 和 `fcsr`。C handler 再附加版本、记录大小、
hart ID 与当前 thread 的 ID/type/state，异步排入 RPC 发送队列，然后把故障 proc 置为
退出状态并切换上下文。启动阶段异常、嵌套异常，或缺少有效
`current` 时无法安全恢复，系统会停机；异常报告无法入队时也会停机，避免静默丢失
故障信息。

`device/sche/` 管理 Thread 生命周期和非抢占式上下文切换，不设独立 scheduler Thread。
`switch_context()` 先用临时寄存器
检查 `mip.MSIP` 或弹出 BASE Produce/Bar[0]，选定目标且确认不同于当前 `tp` 后，才在当前
栈保存 `ra`、`s0-s11`、`gp` 和 `fcsr`。旧寄存器 store 与目标 Thread 的 `sp/pc`、
Context1/Context2 load 交错执行；恢复旧 Thread 时从切换帧恢复 `gp/shared`，首次执行时
由 bootstrap 从 96-byte 启动帧加载入口、`a0-a7`、返回地址和 `gp`。软浮点 `lp64` ABI
没有需要跨 C 调用保存的 F 寄存器；有存活浮点值时，编译器会在调用 `thread_yield()` 等
函数前自行溢出，Thread 的浮点控制状态由 `fcsr` 保存。

内部 `thread_start()` 初始化 Thread，`thread_schedule()` 提交无依赖硬件 Wait。
运行中的 Thread 通过 `thread_yield()` 提交同类 Wait 主动让出。
`thread_wait_event()` 把短 slot node 挂入 join/RPC/profile 事件链表，条件满足时由事件
所有者摘下并提交无依赖硬件 Wait。join 挂在目标子 Thread 的退出链表上，只在目标退出时
唤醒，不轮询条件。scheduler 本身不维护或扫描 ready、
poll、resource 队列。入口正常返回等价于 `thread_exit()`。Host RPC 创建的 root Proc 按
资源描述符预留 `1 + max_children`
个 proc slot。每个 Hart 固定有 16 个算子 proc slot，root 也计入，两个 RPC
proc 和 profile proc 不计入，因此单个 kernel 最多声明 15 个子 Thread。
`dayan_create_thread(name, entry, argument, resources)` 创建的子 Thread 保持在同一 hart，
拥有不超过 15 bytes 的名称，继承 logical task、`gp/shared` 与 URAM，并返回正的 Thread
ID。传入空资源表示不占用 TCSR 或 Named Resource；显式 TCSR 由固件映射为该 Thread 从
逻辑组 0 开始的独立 Context1，Local/Shared count 限制 Context2 可见前缀。
`dayan_thread_join(id, &result)` 精确等待直属子 Thread，
`dayan_thread_join_all()` 等待全部直属子 Thread；每个执行对象退出前都会隐式
join-all，因此 root 进入终态时整棵子树已经完成。`dayan_thread_exit(result)` 可显式
设置 64-bit 返回值。
托管 cleanup 在退出后把 slot 归还静态池；slot 的 64 KiB 栈和 4 KiB 输出缓冲在
固件链接时已经按 Hart 预留，不走运行时堆分配。当前没有抢占或跨 hart 迁移。

固件 C 代码直接调用 RVT 的 `__builtin_riscv_tx_base_tcsr_read/write`，地址必须是编译期
常量；它们也是 `rvt_intrinsics.h` 中 C++ template 的底层接口。无依赖唤醒和 yield
利用 `WAIT` 提交后自动清零的语义，不再重复写入零掩码。

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

`dayan_yield()` 直接登记一个无资源依赖的硬件 Wait 项，token 就是当前 Thread 的
`tp`。`dayan_wait_lsu/mat/vec/macro()` 仍通过对应执行单元的 Produce/Bar 队列等待前序
工作；`dayan_wait_dma()` 是 LSU 入口的兼容别名。全局 Produce/Bar[0] 中只传递 Thread
token，不再编码 Resource 指针或软件事件类型。

选线程和上下文切换位于同一个汇编函数中。热路径先读 `mip.MSIP`，有 Host RPC 时选择
`rpc-receive` Thread；否则弹出 Produce/Bar[0]。token 为 0 时继续等待，等于当前 `tp`
时直接返回。只有确定目标 Thread 不同后，才保存 `s0--s11`、`gp` 和 `fcsr`，并交错读取
新 Thread 的栈、PC、Context1 和 Context2。软件 ready/resource 链表与扫描循环已经删除。

`dayan_uram_base()` 和 `dayan_uram_size()` 返回 Service 为当前 kernel 实例在本 Core
分配的私有 URAM；每 Core 容量默认 1.5 MiB（`0x180000`），768 KiB 核心需在
`dayan_server` 启动时指定 `--uram-size 0xc0000`。两种容量的地址窗口均从
`0x00800000` 起并以 2 MiB 为 Core stride。声明、分配和释放大小以 32 KiB 为粒度。
Host 管理的 BL1 从 Context 获取分配范围；独立裸机测试则用
`make -C device URAM_SIZE=0xc0000` 选择 768 KiB，默认仍为 1.5 MiB。
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
算子再使用 `kFourCore`、Shared Named Resource 和独立的 loader/copy/MAT/store Thread。

`dayan_cluster_barrier(participants, barrier)` 只用于四 Core 调度模式，`participants` 可为 1--4。barrier 是放在当前
`shared` allocation 内的零初始化 16-byte 对象。每个 lane 通过 `AMOADD.D`
领取单调 ticket，先轮询 64 次；未到达本轮边界时以最低优先级的 polling Wait 让出，再继续等待，因此不需要 AMOSWAP、LR/SC 或清零
计数，同一对象也可以跨多轮复用。Server 在启动任何 lane 前，会把本次 launch 的
整个 shared allocation 初始化为零；不会清空可能仍被其它 kernel 使用的全局
Atomic Cache backing。

Named Resource 状态由硬件表维护，每个 Core 有 32 项 Local Resource，同一 Cluster
共享 32 项 Shared Resource。`dayan_resource_t` 只是 8-bit 逻辑编号：0--31 表示
Local，32--63 表示 Shared。Host 为每个常驻 kernel 分配物理窗口，Device 在切换 Thread
时恢复 Context2；算子只使用 `DAYAN_LOCAL_RESOURCE(i)`、
`DAYAN_SHARED_RESOURCE(i)` 和 `DAYAN_RESOURCE_MASK(r)`，不保存资源对象或状态字段。

两类资源使用相同的 `FREE -> BUSY -> READY -> FREE` 状态机。`dayan_resource_wait()`
把 `WAIT_FREE` 输出和 `WAIT_READY` 输入连同当前 `tp` 一次登记到硬件；发射时所有
`WAIT_FREE` 项会原子预留为 BUSY。两个掩码 one-hot 重叠时执行生产者选举，返回
`DAYAN_RESOURCE_ROLE_PRODUCER` 或 `DAYAN_RESOURCE_ROLE_CONSUMER`。单资源便捷接口
`dayan_resource_wait_free/ready()` 也遵循相同语义，因此 `wait_free()` 成功后调用方必须
最终发布该资源。

`dayan_resource_wait(free, ready, phase, qos)` 强制要求每个 READY 输入匹配
`phase` 位图中对应逻辑资源的期望值。硬件只在最后一个 Consume 经 Resource Produce
提交、使资源回到 FREE 时翻转 Phase。消费者提交异步工作和 Consume 后即可等待下一代，
无需逐轮等待 MAT/LSU，Thread 退出前仍须排空最后的工作及资源提交。
每次启动前，Host 通过 Cluster `0x6000` clear 接口初始化本组 Proc 的 Shared 资源；
Device 在根 Proc 首次调度前用 `LOCAL_RESOURCE_INIT` 初始化分配的 Local 资源。
初始化按物理掩码只清选中项，状态为 FREE、计数 0、Phase 0；子 Thread 不重复初始化。
单资源 `dayan_resource_wait_ready(resource, phase)` 的 phase 必须为 0 或 1。
选举同样使用软件指定的期望 Phase，不自动快照资源当前代，也不提供关闭 Phase 比较的模式。

`dayan_resource_produce_<unit>(output, consumers, inputs)` 在对应执行单元的前序工作完成
后，原子消费 `inputs` 并把已预留的 `output` 发布为 READY。有限依赖图的最后一批输入
用 `dayan_resource_consume_<unit>(inputs)` 终止消费，使资源回到 FREE。Resource 操作
本身不生成调度 token，跨 Hart 的 Shared Resource 提交由硬件在 Cluster 范围内原子化。
kernel 和子 Thread 分别在 descriptor 与 `dayan_thread_resources_t` 中声明所需的 Local/
Shared 数量；`shared` 指向的 Host coherent allocation 只存放屏障和算子数据，不再承载
Resource 控制对象。

`device/lib/` 提供 `mem*`、常用字符串函数、`malloc/calloc/realloc/free`，以及整数、字符串
和指针格式的 `printf/snprintf`。数学部分先提供不依赖软双精度 runtime 的单精度
`fabsf/copysignf/fminf/fmaxf/sqrtf/rsqrtf/expf/logf`。浮点格式化暂未实现，
`printf/puts/putchar` 共享当前 proc 在创建时配置的有限缓冲区；缓冲恰好填满时自动通过
RPC 发送，未填满的内容跨调用保留，直到 `fflush(stdout)` 或 `fflush(NULL)`。统一 flush
路径在 `output_size == 0` 时直接成功，RPC 失败时保留缓冲内容以便重试。Host 收到输出后
立即交给输出回调，不保存按 Proc 拼接的流状态。proc 退出时只做一次 best-effort flush，
不额外发送 CLEAR；若要求确认尾部输出已经送达，仍应在返回前显式调用并检查 `fflush`。

## Host 用户态驱动

`host/driver/` 按 `hal`、`server`、`runtime` 分层，统一 Xilinx DMA backend 支持
XDMA/QDMA 驱动的设备文件访问。HAL 显式使用 `(cluster, core)` 访问 register/mailbox、管理普通与
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
allocation。详细架构、生命周期、API 和 dayan_emu 命令见
[`host/driver/README.md`](host/driver/README.md)。
