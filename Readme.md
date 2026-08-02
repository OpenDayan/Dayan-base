# NPU cooperative OS skeleton

这是一个面向 RISC-V NPU Device 的最小裸机系统骨架。当前版本只建立启动、
每核资源隔离、协作式调度、RPC 边界、基础 libc 和堆管理，不包含具体的 Host
传输协议、设备驱动或抢占式上下文切换。

## 启动流程

1. `bl0/bootrom.S` 从当前 hart 的 mailbox 读取入口地址，并把 hart id 留在 `a0`。
2. `bl1/bl1.S` 为该 hart 选择独立栈，令 `tp` 指向 CPU-local 中的 init thread，
   清零该区并初始化浮点状态。
3. `boot/boot.c:init()` 初始化平台、调度器，以及该 hart 的链接期初始堆。
4. BL1 调用 `sche/sche.c:scheduler()`，此后不再返回。

默认构建假设四个 hart 运行四份相互独立的 OS 状态：

| 区域 | 默认配置 | 用途 |
| --- | --- | --- |
| RAM | `0x80000000`, 64 MiB | 镜像、CPU-local、每核栈和每核初始堆 |
| CPU-local | 4 KiB / hart | 每核 scheduler、allocator 与 init thread |
| init 栈 | 64 KiB / hart | BL1、初始化和 scheduler 的每核栈 |
| 初始堆 | 1 MiB / hart | 红黑树分配器的第一个 region |
| 一致性区 | `0x90000000`, 4 MiB | 显式跨核通信 |

地址均集中在顶层 `Makefile` 中，接入真实硬件时应覆盖 `RAM_BASE`、
`RAM_SIZE` 和 `COHERENT_BASE`。一致性区默认按 4 MiB 预留。

`tp` 始终解释为布局固定的 `struct thread *`，而不是 `struct cpu *`。公共 thread
依次保存 32-bit ID、32-bit state、当前 CPU 指针、`sp`、恢复 `pc` 和 32-bit type。
init thread 的 ID 与 type 都是 0；其它类型把 `struct thread` 作为结构体首成员，并
使用非零 type。每个独立 OS 实例的 scheduler 从 1 开始统一分配后续 ID。上下文切换
同步切换 `tp`，thread 的 CPU 指针使内存分配器等每核服务能找到正确的私有状态；
proc 一旦绑定到某个 hart，当前版本不允许迁移。

thread state 的公共值从 `THREAD_RUNNING == 0`、`THREAD_READY == 1` 开始，其余值
表示不同等待原因。`thread_init()` 构造期间短暂使用 `THREAD_WAIT_START`；proc 初始化
完成后直接进入 `THREAD_READY`，运行期使用 `THREAD_WAIT_INTERNAL_EVENT`、
`THREAD_WAIT_EXTERNAL_EVENT` 和 `THREAD_WAIT_EXIT`。CPU 私有结构仅聚合 scheduler 与
allocator 状态；hart ID 由平台直接读取 `mhartid`，不在 CPU-local 中重复保存。

普通 `.data`/`.bss` 会被链接脚本拒绝，以免不一致缓存下多个 hart 意外共享
可变状态。每核状态应放入 `struct cpu`，跨核对象应显式使用 `OS_COHERENT`；
一致性区是 `NOLOAD` 区域，初始化和跨核协议由平台代码负责。

## 构建

需要本机 `clang`、`ld.lld` 和 LLVM binutils：

```sh
make
make check
```

默认目标为 `rv64imaf_zicsr_zifencei` / `lp64`，与 `npu-kernels` 的软浮点调用
约定兼容，同时允许编译器在函数内部使用 F 指令。`zicsr` 和 `zifencei` 是裸机
启动所需的系统扩展。输出位于 `build/`：

固件 ABI 不使用 `gp`：BL1 不初始化它，链接脚本不定义 `__global_pointer$`，构建
同时使用 `-mno-relax -msmall-data-limit=0`。运行时加载的算子 `.o` 也应
使用 `-mno-relax -msmall-data-limit=0`，并且不得包含 GP-relative relocation；其
内部数据以及对系统/lib 符号的引用统一由加载器处理 PC-relative relocation。
固件链接不启用 section GC，确保未被固件自身调用的公共 API 仍保留在静态符号表中，
可供运行时算子重定位使用。

- `npu-os.elf`：供加载和调试的 ELF；
- `npu-os-api.elf`：与固件地址一致、但符号表只保留公开 API 的算子链接视图；
- `npu-os.bin`：只包含可加载镜像的平坦二进制；
- `npu-os.objdump`、`npu-os.map`：反汇编和链接布局。

配置示例：

```sh
make HARTS=4 STACK_SIZE=0x20000 HEAP_SIZE=0x200000 \
     RAM_BASE=0x80000000 COHERENT_BASE=0x90000000
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
Host region 完全空闲后通过 `RPC_HEAP_RELEASE` 立即归还。RPC 尚未接通时，扩堆会
正常失败并使 `malloc()` 返回空指针。

`rpc_call()` 是统一的 Device 侧 RPC 入口。字符串输出、清空 Host 输出缓冲、申请
和归还堆分别占用四个最小服务号。`platform_rpc()` 当前是返回 `-ENOSYS` 的弱符号，
平台只需覆盖它即可接入 mailbox/doorbell。Cache 接口同样保留为弱符号，默认实现
采用当前平台的 64-byte cache line 和 `cbo.flush`/`cbo.inval` 指令编码。

`sche/` 是有独立 proc 栈的非抢占式调度器。`switch_context()` 在当前栈保存
`ra`、`s0-s11` 和 `fcsr`，把保存帧的 `sp` 及恢复入口 `pc` 写入当前 thread，再加载
目标 thread 的 `sp/pc` 并同步更新 `tp`。软浮点 `lp64` ABI 没有需要跨 C 函数调用
保留的 F 寄存器；有存活浮点值时，编译器会在调用 `proc_yield()` 等函数前自行溢出，
而每个 thread 的浮点控制状态由 `fcsr` 保存。

`proc_init(proc, entry, stack, stack_size, argc, ...)` 使用调用者提供的区间构造启动帧，
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
proc_init(&proc, (proc_fn)task_main, stack, stack_size, 2,
          (uintptr_t)&proc, (uintptr_t)arg);
```

`proc_init()`/`proc_vinit()` 成功时直接把 proc 加入当前 CPU 的 ready queue。运行中的
proc 可调用 `proc_yield()` 主动让出。`proc_wait_internal(ready, arg)` 进入内部等待队列，
scheduler 每轮调用其 predicate，返回非零后移回 ready queue；内部等待非空时不会执行
`wfi`。`proc_wait_external()` 进入不被遍历的外部等待队列，外部事件 callback 通过
`proc_wake()` 将其移回 ready queue。`entry()` 正常返回等价于调用 `proc_exit()`，不会
被重新排队。当前没有抢占、跨 hart 迁移或退出后自动回收 proc/栈。

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
Produce，然后进入 external-event 等待；`dayan_wait_dma()` 是 LSU 入口的兼容别名。
scheduler 每轮通过 BASE 全局地址 0 的 pop 读取所有已完成 token，并由
`proc_scan_produce()` 调用内部 `proc_wake()`。external-wait 链表本身仍不遍历；只要
其中存在等待者，scheduler 就持续扫描 Produce，而不会进入 `wfi`。每次等待占用一个
Produce 项，软件仍须遵守硬件的全局 32 项和单执行单元 2 项的无阻塞保证，不能无界
发射同步项。

资源分为单核和跨核两种类型，两者提供相同的
`FREE -> BUSY -> READY -> FREE` 状态机。`alloc()` 设置生产者和消费者计数；生产者为
0 时直接进入 READY，消费者必须非零。最后一次 `produce()` 进入 READY，最后一次
`consume()` 回到 FREE。`wait_ready/free()` 使用 internal-event predicate 协作式切出，
状态满足后再返回。所有返回 `int` 的资源接口均以 0 表示成功、负值表示参数或状态错误。

`dayan_resource_t` 只允许当前 hart 的 proc 共享，可放在该 hart 的栈或普通私有堆中。
其状态、计数、上锁和解锁全部使用普通 load/store，不包含 `__atomic`、AMO 或 fence；
非抢占调度保证资源操作不会与本核另一个 proc 同时执行。

`dayan_shared_resource_t` 供多个 hart 共享，所有字段通过 `__atomic` 访问，临界区使用
RV64A 原子锁。对象必须分配在 4 MiB 一致性窗口中，例如：

```c
DAYAN_COHERENT dayan_shared_resource_t shared_resource;
```

一个 hart 应在发布对象前调用一次 `dayan_shared_resource_init()`；该函数会拒绝一致性
窗口之外的地址。运行时加载器必须把算子 `.o` 的 `.coherent` section 分配到该窗口，
不能用普通 `malloc()` 为跨核资源分配内存。

`lib/` 提供 `mem*`、常用字符串函数、`malloc/calloc/realloc/free`，以及整数、字符串
和指针格式的 `printf/snprintf`。数学部分先提供不依赖软双精度 runtime 的单精度
`fabsf/copysignf/fminf/fmaxf/sqrtf/rsqrtf/expf/logf`。浮点格式化暂未实现，
`printf` 的输出统一走 RPC。
