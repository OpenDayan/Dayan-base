# NPU cooperative OS skeleton

这是一个面向 RISC-V NPU Device 的最小裸机系统骨架。当前版本已经建立启动、
每核资源隔离、协作式调度、mailbox RPC、同步异常处理、基础 libc 和堆管理。
系统不使用抢占式调度：包括 CLINT 在内的功能中断始终屏蔽，mailbox 状态由专用
proc 协作式轮询；同步异常仍通过 `mtvec` 进入 trap。

## 启动流程

1. `bl0/bootrom.S` 从当前 hart 的 mailbox 读取入口地址，并把 hart id 留在 `a0`。
2. `bl1/bl1.S` 为该 hart 选择独立栈，令 `tp/current` 指向 CPU-local 中的 init
   thread，清零该区，将 `gp/shared` 置空，配置独立 trap 栈与 `mtvec`，屏蔽功能
   中断并初始化浮点状态。
3. `boot/boot.c:init()` 初始化平台、调度器、该 hart 的链接期初始堆，以及 RPC
   收发 proc。
4. BL1 调用 `sche/sche.c:scheduler()`，此后不再返回。

默认构建假设四个 hart 运行四份相互独立的 OS 状态：

| 区域 | 默认配置 | 用途 |
| --- | --- | --- |
| RAM | `0x80000000`, 64 MiB | 镜像、CPU-local、每核栈和每核初始堆 |
| CPU-local | 4 KiB / hart | 每核 scheduler、allocator、RPC、trap 状态与 init thread |
| init 栈 | 64 KiB / hart | BL1、初始化和 scheduler 的每核栈 |
| RPC 栈 | 4 KiB × 2 / hart | RPC receive proc 和 send proc 的私有栈 |
| trap 栈 | 4 KiB / hart | 同步异常入口使用的独立栈 |
| 初始堆 | 1 MiB / hart | 红黑树分配器的第一个 region |
| Host coherent window | `0x00400000`, 4 MiB | Host 按 proc 动态切分，基址由 `gp/shared` 访问 |

固件 RAM 地址集中在顶层 `Makefile` 中，接入真实硬件时应覆盖 `RAM_BASE` 和
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
`THREAD_WAIT_EVENT` 和 `THREAD_WAIT_EXIT`。CPU 私有结构聚合 scheduler、allocator、
RPC 和 trap 状态；hart ID 由平台直接读取 `mhartid`，不在 CPU-local 中重复保存。

普通 `.data`/`.bss` 会被链接脚本拒绝，以免不一致缓存下多个 hart 意外共享
可变状态。每核状态应放入 `struct cpu`。固件侧同样拒绝 `.coherent` section；跨核
对象只能放入 Host 给当前 proc 分配的 coherent allocation，并从 `shared` 基址寻址。

## 构建

需要本机 `clang`、`ld.lld` 和 LLVM binutils：

```sh
make
make check
```

默认目标为 `rv64imaf_zicsr_zifencei` / `lp64`，与 `npu-kernels` 的软浮点调用
约定兼容，同时允许编译器在函数内部使用 F 指令。`zicsr` 和 `zifencei` 是裸机
启动所需的系统扩展。输出位于 `build/`：

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

配置示例：

```sh
make HARTS=4 STACK_SIZE=0x20000 HEAP_SIZE=0x200000 \
     RPC_STACK_SIZE=0x1000 TRAP_STACK_SIZE=0x1000 \
     RAM_BASE=0x80000000 RAM_SIZE=0x08000000 CONTROL_BASE=0
```

## 子系统约定

`include/container_of.h`、`include/list.h` 和 `include/rbtree.h` 提供一组独立实现的
Linux 风格 intrusive 泛型。当前兼容常用的 `container_of/list_entry`、双向循环
`list_head`，以及 `rb_root/rb_node/rb_link_node/rb_insert_color/rb_erase/rb_entry`
接口；不引入 Linux 内核源码、RCU、锁、内存屏障、poison 或调试配置。类型相关操作
由宏在编译期展开，基础指针操作使用 `static inline`，RBTree 平衡代码位于
`lib/rbtree.c`。

`sche/` 的调度队列和 `mem/` 的 region 集合都使用 `list_head`。`mem/` 的空闲块
使用按块大小排序的 intrusive RBTree 做 best-fit 分配，并通过边界信息合并
相邻空闲块。初始 region 永不归还；堆不足时通过 `RPC_HEAP_ACQUIRE` 请求 Host，
Host region 完全空闲后通过 `RPC_HEAP_RELEASE` 立即归还。

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
抢占：BL1 将 `mie` 清零并清除 `mstatus.MIE`。

每个 hart 有两个内核 proc。receive proc 以 `proc_wait_poll()` 等待本核 `msip`，读取
Host 命令，调用弱的 `rpc_receive(opcode, arg, &response)` 分派接口，并在需要回复时
把响应加入发送队列，最后清除 `msip`。send proc 独占本核发送队列；队列为空时使用
`proc_wait_event()`，有事务时写 Device → Host mailbox，等待 Host 清除状态位，再执行
事务 cleanup 并唤醒调用者。receive handler 返回 1 表示发送 `response`，返回 0 表示
无响应，负值表示请求已消费但失败。handler 在清除当前 `msip` 前运行，因此不应发起
需要 Host 嵌套处理的阻塞式反向 RPC。

`rpc_call()` 是 Device 主动请求的统一入口。其低 48 位参数指向固定 40-byte 的
`struct rpc_packet`，字段依次是 request 地址/长度、response 地址/长度和 Host 回填的
有符号 status。当前 opcode 1–5 分别为字符串输出、清空输出缓冲、申请堆 region、
归还堆 region 和异常报告。前四类使用 packet；异常报告的低 48 位参数直接指向一份
稳定的 320-byte `struct trap_record`。

同步异常从 `mtvec` 进入独立的每核 trap 栈，保存 x0–x31（其中 x2 为异常前的 sp）、
`mstatus`、`mepc`、`mcause`、`mtval` 和 `fcsr`。C handler 再附加版本、记录大小、
hart ID 与当前 thread 的 ID/type/state，异步排入 RPC 发送队列，然后把故障 proc 置为
退出状态并回到 scheduler。init/scheduler 上下文异常、嵌套异常，或缺少有效
`current` 时无法安全恢复，系统会停机；异常报告无法入队时也会停机，避免静默丢失
故障信息。

`sche/` 是有独立 proc 栈的非抢占式调度器。`switch_context()` 在当前栈保存
`ra`、`s0-s11`、`gp` 和 `fcsr`，把保存帧的 `sp` 及恢复入口 `pc` 写入当前 thread，
再加载目标 thread 的 `sp/pc` 并同步更新 `tp/current`。恢复旧 proc 时从切换帧恢复
`gp/shared`；首次执行新 proc 时由 bootstrap 从启动帧加载它。软浮点 `lp64` ABI 没有需要跨 C 函数调用
保留的 F 寄存器；有存活浮点值时，编译器会在调用 `proc_yield()` 等函数前自行溢出，
而每个 thread 的浮点控制状态由 `fcsr` 保存。

`proc_init(proc, entry, stack, stack_size, shared_base, argc, ...)` 使用调用者提供的
栈区间和 Host 分配的 shared 基址构造启动帧，
栈顶按 16 字节向下对齐，支持 0 到 8 个寄存器参数。可变参数是底层 ABI 机器字，调用者
必须逐个按 `uintptr_t` 传入；后续类型安全封装可负责指针转换、浮点位模式打包以及自动
加入 proc 指针。实际实现位于接收 `va_list` 的 `proc_vinit()`；`proc_init()` 仅负责
建立和销毁参数列表，因此其它可变参数封装可以直接调用 `proc_vinit()`，不需要中间
数组。bootstrap 开头按 `a7` 到 `a0` 连续排列八条定长加载指令，新 proc 的
初始 `pc` 为 `proc_bootstrap + (8 - argc) * 4`，因此只恢复实际提供的参数寄存器，再
加载任务入口和返回地址。未使用的高位参数寄存器不作初始化。该栈在 proc 到达
`THREAD_WAIT_EXIT` 前必须一直有效。

例如，底层接口创建一个具有两个机器字参数的 proc：

```c
proc_init(&proc, (proc_fn)task_main, stack, stack_size, shared_base, 2,
          (uintptr_t)&proc, (uintptr_t)arg);
```

`proc_init()`/`proc_vinit()` 成功时直接把 proc 加入当前 CPU 的 ready queue。运行中的
proc 可调用 `proc_yield()` 主动让出。`proc_wait_poll(poll, arg)` 进入轮询等待队列，
scheduler 每轮调用其 predicate，返回非零后移回 ready queue；该队列非空时不会执行
`wfi`。`proc_wait_event(event)` 则立即把当前 proc 的 intrusive `node` 挂入指定事件
链表，scheduler 不遍历它；事件所有者调用 `proc_event_wake_one/all()` 或
`proc_wake()` 将等待者移回 ready queue。传入空指针会使用 scheduler 的默认外部事件
链表。`entry()` 正常返回等价于调用 `proc_exit()`，不会被重新排队。当前没有抢占、
跨 hart 迁移或退出后自动回收 proc/栈。

## 算子 API 与同步

`include/api/` 是外部算子唯一受支持的头文件根目录。它包含当前的小型 libc 标准头，
以及只声明 Dayan 系统接口的 `dayan_sys.h`；该头文件不会包含或暴露 `proc`、`thread`、
`cpu` 和 scheduler 等内部类型。算子应使用 `-Iinclude/api` 编译，不能加入内部
`include/`。旧的 `lib/include/` 仅保留为转发层，公开声明以 `include/api/` 为准。

所有公开 libc 函数和 DAPI 分别标注到 `.text.api.libc`、`.text.api.dapi`。完整的
`npu-os.elf` 仍保留全部系统符号，供固件加载和调试；构建同时根据
`dayan-api.symbols` 生成 `npu-os-api.elf`，其中调试信息及非白名单符号已经移除。
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

`dayan_shared_resource_t` 供多个 hart 共享，所有字段通过 `__atomic` 访问，临界区使用
RV64A 原子锁。它不能保存只对某个 hart 私有的 proc 链表节点，因此
`wait_ready/free()` 使用 `proc_wait_poll()` 检查一致性状态。`dayan_sys.h` 以全局寄存器
变量的形式把 `shared` 直接绑定到 `gp`；它没有内存符号，也不是固件定义的普通全局
变量，算子代码不得改写它。算子应在 Host
分配区内定义自己的布局，
例如把第一个对象放在基址处：

```c
dayan_shared_resource_t *resource = (dayan_shared_resource_t *)shared;
```

一个 hart 应在发布对象前调用一次 `dayan_shared_resource_init()`。该函数要求 `gp`
位于 `0x00400000` 开始的 4 MiB 硬件窗口内，并检查资源地址不低于 `shared`、没有越过
硬件窗口且满足类型对齐。内核仍不持有 Host 为单个 proc 分配的实际长度，因此 allocation
内部的精确上界和算子布局由 Host/加载器负责。不能用普通 `malloc()` 为跨核资源分配
内存，也不能在算子 `.o` 中静态定义 `.coherent` section。

`lib/` 提供 `mem*`、常用字符串函数、`malloc/calloc/realloc/free`，以及整数、字符串
和指针格式的 `printf/snprintf`。数学部分先提供不依赖软双精度 runtime 的单精度
`fabsf/copysignf/fminf/fmaxf/sqrtf/rsqrtf/expf/logf`。浮点格式化暂未实现，
`printf` 的输出统一走 RPC。
