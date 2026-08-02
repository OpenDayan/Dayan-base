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
表示不同等待原因。当前 proc 使用 `THREAD_WAIT_EVENT`、`THREAD_WAIT_START` 和
`THREAD_WAIT_EXIT`。CPU 私有结构仅聚合 scheduler 与 allocator 状态；hart ID
由平台直接读取 `mhartid`，不在 CPU-local 中重复保存。

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

`sche/` 的 ready queue 和 `mem/` 的 region 集合都使用 `list_head`。`mem/` 的空闲块
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

`proc_init(proc, run, arg, stack, stack_size)` 使用调用者提供的区间构造启动帧，栈顶按
16 字节向下对齐。新 proc 的初始 `pc` 直接指向 `proc_bootstrap`；bootstrap 从栈中加载
任务入口、返回地址和 `a0-a7`，其中 `a0` 是 proc、`a1` 是 arg，其余参数为 0。该栈
在 proc 到达 `THREAD_WAIT_EXIT` 前必须一直有效。
`proc_start()` 只负责加入当前 CPU 的 ready queue。运行中的 proc 可调用
`proc_yield()` 主动让出，或调用 `proc_wait()` 睡眠并等待 `proc_wake()`；`run()` 正常
返回等价于调用 `proc_exit()`，不会被重新排队。当前没有抢占、跨 hart 迁移或退出后
自动回收 proc/栈。

`lib/` 提供 `mem*`、常用字符串函数、`malloc/calloc/realloc/free`，以及整数、字符串
和指针格式的 `printf/snprintf`。数学部分先提供不依赖软双精度 runtime 的单精度
`fabsf/copysignf/fminf/fmaxf/sqrtf/rsqrtf/expf/logf`。浮点格式化暂未实现，
`printf` 的输出统一走 RPC。
