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
`build/host/runtime` 查找 runtime，安装后从 SDK 相邻的 `lib`/`lib64` 查找；特殊部署
可用 `DYCC_RUNTIME_LIBRARY=/absolute/path/libDayanRuntime.a` 覆盖。`-Xlinker <arg>`、
`-Wl,...`、`-L` 和 `-l` 等参数只传给最终 Host 链接，`-Xhost` 参数传给 Host 编译与
链接阶段。

默认 Device ABI 固定为 `riscv64-unknown-elf`、
`rv64imaf_zicsr_zifencei`、`lp64`、`medany`、`-mno-relax` 和
`-msmall-data-limit=0`。Device 公开头文件目录 `include/api` 与 Host 公开头文件目录
`host/include` 会分别作为 system include path 加入。可用 `DYCC_DEVICE_CXX`、
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
  的 `dayan_kernel_launch_attrs_t`、descriptor 和 launch API；dycc 不再为输入源码
  隐式强制包含它。
- `__kernel__`：只在 Device 保留定义；Host 原位置生成同名 wrapper。kernel 必须返回
  `void`、具有外部链接、不是模板或可变参数函数，并且最多接收 8 个机器字参数。
- `__device__`：函数或全局变量只进入 Device。
- `__host__`：函数或全局变量只进入 Host。
- 未标注的命名空间/文件作用域函数同时进入 Host 与 Device。
- 所有命名空间/文件作用域变量必须显式标注 `__host__` 或 `__device__`，否则 dycc
  直接报源码位置诊断。一个逗号声明组中的变量必须属于同一侧。

kernel 参数支持整数、枚举、指针、引用、`float` 和 `double`。wrapper 将它们按 RV64
`lp64` ABI 打包为至多 8 个 64-bit 机器字；浮点参数传递位模式。聚合体参数当前会被
拒绝，调用者应改传指针。

Host wrapper 的追加参数为：

```cpp
const dayan_kernel_launch_attrs_t *launch_attrs
```

wrapper 返回 `int` 状态。它包含每个 kernel 独立的 static descriptor、原子快路径和
初始化自旋锁。首次调用使用嵌入的 `__kernel_obj__`、对象尺寸和准确的 C++ mangled
entry 名创建描述符；后续调用直接复用。实际 Device module 按 runtime 实例缓存，因此
同一 Host `.o` 可用于多个 runtime。

源文件可根据 `__DAYAN_DEVICE__` / `__DAYAN_HOST__` 条件包含单侧头文件。比如
`dayan_sys.h` 只应在 Device 分支包含。

## Runtime 对接

应用先加载固件，并将只含公开白名单符号的 API ELF 注册到同一 runtime：

```cpp
#include <dayan_runtime.h>

int demo_kernel(uint64_t count,
                const dayan_kernel_launch_attrs_t *launch_attrs);

dayan_runtime_set_api_elf_file(runtime, "build/npu-os-api.elf");

dayan_kernel_launch_attrs_t attrs;
dayan_kernel_launch_attrs_init(&attrs, runtime);
attrs.first_hart = 0;
attrs.launch_count = 1;

int status = demo_kernel(count, &attrs);
```

`DAYAN_KERNEL_LAUNCH_ASYNC` 会把 `npu_launch_t*` 写入调用者提供的
`async_launches` 数组；同步模式会等待并释放 launch。`launch_count` 是启动规模，
从 `first_hart` 起连续分配 hart。

```cpp
npu_launch_t *handles[4];
size_t submitted = 0;
attrs.flags = DAYAN_KERNEL_LAUNCH_ASYNC;
attrs.launch_count = 4;
attrs.async_launches = handles;
attrs.async_launch_capacity = 4;
attrs.async_launch_count = &submitted;
status = demo_kernel(count, &attrs);
/* wait/query/release handles[0..submitted) with the low-level runtime API */
```

`host/runtime` 同时产出 `libDayanRuntime.a`、`libDayanRuntime.so` 和兼容名称
`libnpu_runtime.a`。加载器只用注册的 API ELF 解析 Device 未定义符号，支持 dycc 默认
代码模型产生的 RISC-V 绝对、PC-relative、CALL/JAL/BRANCH 与 ADD/SUB/SET relocation；
TLS、GP-relative、动态链接和构造器 section 会被拒绝。
