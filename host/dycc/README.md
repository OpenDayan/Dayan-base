# dycc

`dycc` 是 Dayan C++ 算子的 Clang compiler driver。它接收同时包含 Host 与 Device
声明的源文件；使用 `-c` 时生成普通 Host relocatable object，不使用 `-c` 时继续调用
Host clang++ 生成可执行程序。Device 的 RISC-V `ET_REL` 会作为本地只读数组嵌入每个
Host `.o`。

## 构建与调用

```sh
make -C host/dycc
host/dycc/dycc -O2 -c demo.cpp -o build/demo.o
```

完整 SDK 可以从仓库根目录一键安装；安装后直接使用 `dycc`：

```sh
make install PREFIX=/opt/dayan
/opt/dayan/bin/dycc -O2 -c demo.cpp -o demo.o
```

## 编译与链接模式

行为与 gcc/clang driver 的核心模式一致：

```sh
# 只编译；省略 -o 时生成 operator.o
dycc -c operator.cpp

# 编译并链接；省略 -o 时生成 a.out
dycc operator.cpp main.cpp -o app

# 可追加预先编译的 Host 对象、静态库和常见链接参数
dycc operator.cpp main.o support.a -L/path/to/lib -lfoo -o app
```

链接模式会自动加入 `libDayanRuntime.a` 和 `-pthread`。源码树中从
`build/host/driver` 查找 Runtime，安装后从 SDK 相邻的 `lib`/`lib64` 查找；特殊部署
可用 `DYCC_RUNTIME_LIBRARY=/absolute/path/libDayanRuntime.a` 覆盖。`-Xlinker <arg>`、
`-Wl,...`、`-L` 和 `-l` 等参数只传给最终 Host 链接，`-Xhost` 参数传给 Host 编译与
链接阶段。

默认 Device ABI 固定为 `riscv64-unknown-elf`、
`rv64imaf_zicsr_zifencei`、`lp64`、`medany`、`-mno-relax` 和
`-msmall-data-limit=0`，并启用 `txlsu`/`txmat`/`txvec`/`txbase`/`txmacro`，
以便使用 Clang `<rvt_intrinsics.h>` 中的 `__builtin_riscv_tx_*`。Device 编译会查询
`clang++ -print-resource-dir` 以找到该头文件。源码树中的 Device 公开头文件目录为 `device/include/api`，
安装后的 SDK 目录为 `include/api`；Host 公开头文件目录为 `host/include`。dycc 会把
对应目录作为 system include path 加入。可用 `DYCC_DEVICE_CXX`、
`DYCC_HOST_CXX` 或 `--device-cxx`、`--host-cxx` 选择 Clang；`-Xdevice` 和 `-Xhost`
传递单侧参数，其余编译参数传给两侧。

安装后的 driver 会优先从自身旁边的 `lib/dayan`、`lib64/dayan` 或 `share/dayan`
定位私有 SDK。源码树 launcher 则回退到编译时的仓库根目录；两种模式都可通过
`DAYAN_ROOT` 或 `--dayan-root` 覆盖。

调试切分结果时可使用：

```sh
host/dycc/dycc --save-temps=build/dycc-demo \
  --emit-host build/demo.host.cpp \
  --emit-device build/demo.device.cpp \
  -c demo.cpp -o build/demo.o
```

## 源码规则

- 每个算子源文件必须显式 `#include <dayan_kernel.h>`。该头文件声明 Host wrapper 使用
  的 `npu_kernel_launch_attrs_t`、descriptor 和 launch API；dycc 不再为输入源码
  隐式强制包含它。
- `__kernel__`：只在 Device 保留定义；Host 原位置生成同名 wrapper。kernel 必须返回
  `void`、具有外部链接、不是模板或可变参数函数，并且最多接收 8 个机器字参数。
- 每个 `__kernel__` 必须有且只有一个同名资源描述符：
  `DAYAN_KERNEL_DESC(kernel, child_threads, atomic_bytes, uram_bytes,
  lsu_tcsr, mat_tcsr, vec_tcsr, macro_tcsr);`。
  允许同一 Hart 共驻多个 logical Task 时使用
  `DAYAN_KERNEL_TASK_DESC(kernel, tasks_per_core, child_threads, atomic_bytes,
  uram_bytes, lsu_tcsr, mat_tcsr, vec_tcsr, macro_tcsr);`。会修改且依赖未保存/
  恢复的非 TCSR Hart 状态时，必须改用同参数形式的
  `DAYAN_KERNEL_LOCKED_TASK_DESC`；它只支持 `kSingleCore`，并从 root 建立到整个
  family 结束独占该 Hart，自己的 child Thread 仍可运行。
  `child_threads` 是该 kernel 实例最多同时存活的子 Thread 数，上限为 15。root Proc
  也占一个 slot；Runtime 保证同一 Hart 的 `1 + child_threads` 总和不超过 16。
  `atomic_bytes` 是由
  `shared/gp` 寻址的 coherent allocation 大小；`uram_bytes` 是当前 Core 的私有
  URAM 大小（必须是 32 KiB 的整数倍），以及 LSU、MAT、VEC、MACRO 四类 TCSR
  组数。对应项填 0 表示不申请该资源。
- `__device__`：函数或全局变量只进入 Device。
- `__host__`：函数或全局变量只进入 Host。
- 未标注的命名空间/文件作用域函数同时进入 Host 与 Device。
- 所有命名空间/文件作用域变量必须显式标注 `__host__` 或 `__device__`，否则 dycc
  直接报源码位置诊断。一个逗号声明组中的变量必须属于同一侧。

kernel 参数支持整数、枚举、指针、引用、`float` 和 `double`。Host wrapper 把 Device
指针或引用参数改写为 `npu_buffer_t *`，再根据 const 属性生成 Buffer read/write
metadata；它不会接收或传递 Host raw pointer。其它参数按 RV64 `lp64` ABI 打包为至多
8 个机器字，浮点参数传递位模式。聚合体参数当前会被拒绝，调用者应改用 Device
Buffer。

Host wrapper 的追加参数为：

```cpp
const npu_kernel_launch_attrs_t *launch_attrs
```

wrapper 返回 `int` 状态。它包含每个 kernel 独立的 static descriptor、原子快路径和
初始化自旋锁。首次调用使用嵌入的 `__kernel_obj__`、对象尺寸和准确的 C++ mangled
entry 名以及资源声明创建描述符；后续调用直接复用。dycc 同时在 Device object 中保留
外部 C 符号 `<kernel>__desc`，并把相同常量写入 Host Package。描述符版本为 3、固定
40 bytes，包含 resource flags 和四类 TCSR 数量，当前最多声明 15 个子 Thread。
  `locked_task` 是独立的调度约束，不改变 `child_threads` 的真实资源计数；不要用虚报
  15 个 child 来替代它。
实际 Device module 按 runtime 实例缓存，因此
同一 Host `.o` 可用于多个 runtime。

对激活、Element-wise 等分片独立算子，推荐使用
`DAYAN_KERNEL_TASK_DESC(kernel, 4, 0, atomic_bytes, uram_bytes, ...)`：每个
Task 坐标是一个无子 Thread 的 Proc，在内部循环中顺序完成自己分片的
Load、计算和 Store。`tasks_per_core=4` 允许 Server 在资源充足时向一个
Hart 放入最多四个 Proc；Proc 在 `dayan_wait_*` 时交替执行，不同执行单元可以
自然重叠。这类算子不应为普通三阶段分别创建 Thread。

只有需要核间广播、远程 URAM 复制或明确的紧耦合生产者/消费者流水时，
才声明子 Thread 并使用 `dayan_create_thread`。典型场景是 `kFourCore` 矩阵乘在
Core 间广播输入 tile；每个 Thread 仍从自己 Context 的逻辑 TCSR 组 0 开始编程。

源文件可根据 `__DAYAN_DEVICE__` / `__DAYAN_HOST__` 条件包含单侧头文件。比如
`dayan_sys.h` 只应在 Device 分支包含。

## Runtime 对接

BL1 与 API ELF 由独立的 `dayan_server` 加载。应用连接 Server，创建 Context、Stream
和 Buffer，再把这些逻辑对象交给 wrapper；应用不选择物理 Cluster/Core：

```cpp
#include <dayan_runtime.h>

int demo_kernel(npu_buffer_t *output, uint64_t count,
                const npu_kernel_launch_attrs_t *launch_attrs);

npu_runtime_t *runtime;
npu_context_t *context;
npu_stream_t *stream;
npu_buffer_t *output;

npu_runtime_open(&runtime, nullptr);
npu_context_create(runtime, nullptr, &context);
npu_stream_create(context, nullptr, &stream);

npu_buffer_desc_t buffer_desc;
npu_buffer_desc_init(&buffer_desc, 4096);
npu_buffer_create(context, &buffer_desc, &output);

npu_kernel_launch_attrs_t attrs;
npu_kernel_launch_attrs_init(&attrs, context, stream, "demo-task");
attrs.config.task_dim = {8, 1, 1};
attrs.config.task_mode = NPUTaskMode::kSingleCore;

int status = demo_kernel(output, 8, &attrs);
```

第四个参数是本次 launch 的 Task/根 Proc 名称，必须非空且最多 15 bytes。它不要求与
kernel 符号名相同，因此同一个 kernel 的不同用途可以在 `dyperf` 中显示不同名称。

同步 wrapper 等待并销毁执行 Event，返回 kernel status。异步模式通过一个 Event 表示
整个 logical grid：

```cpp
npu_event_t *event = nullptr;
attrs.flags = NPU_KERNEL_LAUNCH_ASYNC;
attrs.event_out = &event;
status = demo_kernel(output, 8, &attrs);
/* npu_event_wait(event, ...); npu_event_destroy(event); */
```

Cluster 调度使用 `NPUTaskMode::kFourCore`，并要求 `attrs.config.task_dim.x == 4`；此时
同一 `(y,z)` group 的 x=0..3 会映射到一个 Cluster 的四个 Core。Device 侧可通过
`dayan_task_id()`、`dayan_task_dim()` 和 `dayan_task_mode()` 获取逻辑坐标。

`host/driver` 同时产出 `libDayanRuntime.a`、`libDayanRuntime.so` 和兼容名称
`libnpu_runtime.a`。Server 的加载器只用启动时注册的 API ELF 解析 Device 未定义符号，
支持 dycc 默认代码模型产生的 RISC-V 绝对、PC-relative、CALL/JAL/BRANCH 与
ADD/SUB/SET relocation；TLS、GP-relative、动态链接和构造器 section 会被拒绝。
