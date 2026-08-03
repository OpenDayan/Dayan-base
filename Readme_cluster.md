# 在 TestCluster 中运行 Dayan Runtime 与 demo 算子

本文说明如何使用 `TestCluster` RTL 模拟器启动 Dayan BL0/BL1、
`dayan_server` 和 Host Runtime，并运行 `demo/` 中的 Device `printf`、单核 BF16
矩阵乘及四核协作 BF16 矩阵乘。

推荐优先使用生命周期 runner。它会管理 TestCluster 的 PTY、FUSE mount、Server、
Unix socket、日志和退出清理。需要观察交互式 CLI、单步推进或波形时，再使用后文的
手工流程。

## 1. 运行链路

```text
Host demo / dycc wrapper
        |
        | libDayanRuntime：Context、Buffer、Stream、Kernel、Event
        | Unix control/data socket
        v
dayan_server
        |
        | Service：加载 BL1、配置 Atomic Cache、处理 RPC、调度 logical task
        | HAL：寄存器、IRQ、物理内存和 memcpy token
        v
TestCluster FUSE/XDMA endpoints
        |
        | BL0 mailbox -> BL1 -> scheduler/root proc -> Device kernel
        v
四个 NPU Core、私有 URAM、Coherent/Atomic Cache、外部 DRAM
```

Host 应用只看到 Runtime 的逻辑对象和 `NPUTaskDim3/NPUTaskMode`。物理 Cluster、Core、
mailbox、IRQ、Atomic Cache backing 地址和 URAM allocation 都由 Server/HAL 管理。

Runtime 与 Server 使用两条 Unix `SOCK_STREAM` 连接：control socket 传递对象和命令，
data socket 申请并完成 memcpy token。实际 H2D/D2H payload 通过 HAL 导出的 XDMA/FUSE
handle 直接搬运，不经过 Server 的中间缓冲区。

一个 `dayan_server` 可以同时接受多个 Runtime Session；每个 Session 都有自己的
control/data socket，已接受的连接由独立线程处理。不同 Session 可以分别运行 `dyperf`
和算子。设备 profile 是全局观测能力，同一时刻只能有一个 profiler owner，但它会记录
所有 Session 创建的 Device proc，并不独占算子调度。

## 2. 路径与前置条件

以下命令均从 firmware 仓库根目录执行：

```sh
cd /home/wangzhe/npu-firmware
```

当前环境中有两种常见的 TestCluster 布局。

独立发布目录：

```sh
export NPU_TC_ROOT=/home/wangzhe/npu-cluster-sim
export NPU_TC_BIN="$NPU_TC_ROOT/TestCluster"
export NPU_TC_DRAM="$NPU_TC_ROOT/configs/LPDDR5_8Gb_x16_6400_4ch.ini"
```

从 `npu-cluster` 源码仓库构建的默认目录：

```sh
export NPU_TC_ROOT=/home/wangzhe/npu-cluster/sim
export NPU_TC_BIN="$NPU_TC_ROOT/build/TestCluster"
export NPU_TC_DRAM="$NPU_TC_ROOT/configs/LPDDR5_8Gb_x16_6400_4ch.ini"
```

本地构建布局中，`NPU_TC_ROOT` 必须是 `.../npu-cluster/sim`，不能设成
`.../sim/build`。TestCluster binary 在 `build/` 下，但 DRAM 配置和运行时资源相对于
`sim/` 目录查找。

还需要：

- 可执行的 TestCluster 和匹配其 RTL 配置的 DRAM 配置文件；
- Linux FUSE 支持、可访问的 `/dev/fuse` 和 `fusermount3`；
- Device 构建所需的 LLVM/RISC-V 工具链；
- `/home/wangzhe/npu-kernels`，或构建时通过 `NPU_KERNELS_DIR` 指定其它位置；
- 没有其它进程占用本次使用的 FUSE mount point 和 Dayan socket。

不要执行 `TestCluster --help`。该 binary 会把参数交给 Verilator，仍然启动模拟器。
需要 CLI 帮助时，应在 TestCluster 交互界面输入 `h`。

## 3. 构建 firmware、Runtime/Server 和 demo

先构建 BL1、公开 API ELF、Host driver、dycc 和 demo：

```sh
make -C . -j72 all
make -C ./demo -j72 all
```

机器核心数较少时应降低 `-j72`。关键产物为：

```text
build/device/npu-os.elf
build/device/npu-os-api.elf
build/host/driver/dayan_server
build/host/driver/libDayanRuntime.a
build/host/dyperf/dyperf
build/demo/print_demo
build/demo/matmul_bf16_1c
build/demo/matmul_bf16_4c
```

可以先做不依赖 RTL 的 Host 测试和 demo 静态检查：

```sh
make -C ./host/driver -j72 check
make -C ./host/dyperf -j72 check
make -C ./demo -j72 check
```

demo 构建末尾的 `device-isa-check: PASS` 很重要。当前硬件不支持 LR/SC，检查脚本会
拒绝 Device object 中的 `lr.w/lr.d/sc.w/sc.d`。如果怀疑使用了旧 object，可强制重建：

```sh
make -B -C ./demo -j72 all
```

## 4. 推荐：使用生命周期 runner

runner 位于 `./scripts/run_testcluster_operator.py`。

先查看完整选项：

```sh
python3 ./scripts/run_testcluster_operator.py --help
```

runner 会依次完成：

1. 检查 TestCluster、DRAM 配置、firmware、Server 和算子 binary；
2. 创建 PTY 并启动 TestCluster，等待 CLI 与 FUSE/XDMA endpoint；
3. 启动 `dayan_server`，等待第一笔 FUSE 请求进入队列；
4. 自动向 TestCluster 发送 `n 1`，确认该单周期命令返回；
5. 发送 `c` 进入连续仿真，并等待 Dayan Unix socket；
6. 设置 `DAYAN_SERVER_SOCKET` 后执行 Host demo；
7. 检查退出码和 `--expect` 指定的输出；
8. 停止本次 Server/TestCluster，精确清理 mount 和 socket，保存日志。

TestCluster 启动和四个 Core 的 BL1 加载可能超过 30 秒，因此下面统一使用
`--startup-timeout 180`。这只是 runner 的启动编排超时，不会修改算子的 Event timeout。

### 4.1 Device printf

`print_demo` 使用 `NPUTaskMode::kSingleCore` 和 `task_dim={2,2,1}`。只启动 Core 0
即可完成四个 logical task：

```sh
python3 ./scripts/run_testcluster_operator.py \
  --repo . \
  --sim-dir "$NPU_TC_ROOT" \
  --sim-bin "$NPU_TC_BIN" \
  --dram-config "$NPU_TC_DRAM" \
  --startup-timeout 180 \
  --core-mask 0x1 \
  --expect '[device] grid_id=' \
  -- ./build/demo/print_demo 1 2 3 4 5 6 7
```

成功时应看到一行 Host launch 信息和四行 Device 输出。四个 `grid_id` 的输出顺序由
调度决定，不应据此判断对错。

### 4.2 单核 BF16 matmul

```sh
python3 ./scripts/run_testcluster_operator.py \
  --repo . \
  --sim-dir "$NPU_TC_ROOT" \
  --sim-bin "$NPU_TC_BIN" \
  --dram-config "$NPU_TC_DRAM" \
  --startup-timeout 180 \
  --core-mask 0x1 \
  --expect '[host] PASS:' \
  -- ./build/demo/matmul_bf16_1c
```

默认 shape 为 `M=384, N=64, K=256`。也可在 `--` 后传入显式的 `M N K`，例如：

```sh
python3 ./scripts/run_testcluster_operator.py \
  --repo . \
  --sim-dir "$NPU_TC_ROOT" \
  --sim-bin "$NPU_TC_BIN" \
  --dram-config "$NPU_TC_DRAM" \
  --startup-timeout 180 \
  --core-mask 0x1 \
  --expect '[host] PASS:' \
  -- ./build/demo/matmul_bf16_1c 384 256 256
```

### 4.3 四核协作 BF16 matmul

四核 kernel 使用 `NPUTaskMode::kFourCore` 和 `task_dim={4,1,1}`，必须启动同一
Cluster 的 Core 0--3。忘记 `--core-mask 0xf` 时，Runtime 通常会因找不到完整的
四核调度单元而返回 `-ENODEV`（`-19`）。

建议先运行一个无边缘 tile 的 RTL smoke case：

```sh
python3 ./scripts/run_testcluster_operator.py \
  --repo . \
  --sim-dir "$NPU_TC_ROOT" \
  --sim-bin "$NPU_TC_BIN" \
  --dram-config "$NPU_TC_DRAM" \
  --startup-timeout 180 \
  --core-mask 0xf \
  --expect '[host] PASS:' \
  -- ./build/demo/matmul_bf16_4c 1024 512 256
```

当前 4c 默认 case 是 `2048x2048x2048`，每 Core tile 为 `512x256x256`，三片 M
slice 为 `160/192/160`，Cluster output tile 为 `1024x512`。完整默认 case 可省略参数：

```sh
python3 ./scripts/run_testcluster_operator.py \
  --repo . \
  --sim-dir "$NPU_TC_ROOT" \
  --sim-bin "$NPU_TC_BIN" \
  --dram-config "$NPU_TC_DRAM" \
  --startup-timeout 180 \
  --core-mask 0xf \
  --expect '[host] PASS:' \
  -- ./build/demo/matmul_bf16_4c
```

`2048^3` 包含约 8.59G 次 BF16 MAC。Verilator RTL 仿真会比 cmodel 慢得多，适合作为
长时间回归，不适合作为首次 bring-up。当前 demo 内部 launch/memcpy timeout 为 30
分钟；如果所用 TestCluster 无法在这个时间内完成，需先提高 demo 的 timeout 并重新
构建。日常调试优先使用前面的单 Cluster tile case，完整数值验证可先运行：

```sh
make -C ./demo run-matmul-4c
```

### 4.4 同时运行 dyperf 与算子

`dyperf` 的 command 模式会先开启 profile，再启动使用独立 Runtime Session 的算子。
runner 设置的 `DAYAN_SERVER_SOCKET` 会被 `dyperf` 读取并继续传给子进程：

```sh
python3 ./scripts/run_testcluster_operator.py \
  --repo . \
  --sim-dir "$NPU_TC_ROOT" \
  --sim-bin "$NPU_TC_BIN" \
  --dram-config "$NPU_TC_DRAM" \
  --startup-timeout 180 \
  --core-mask 0xf \
  --expect '[host] PASS:' \
  -- ./build/host/dyperf/dyperf \
     --output ./build/matmul-4c-trace.json \
     -- ./build/demo/matmul_bf16_4c 1024 512 256
```

未显式指定 `--duration-ms` 时，采样持续到算子退出，`dyperf` 返回算子的退出码。显式
指定时，该值限制采样时长，但工具仍会等待算子退出。一个 Server 同时只允许一个
profile owner；重复启动第二个 `dyperf` 会得到 `-EBUSY`，不会影响已运行的算子。
不启动子命令而持续观察 Server 时使用 `--follow`，按 Ctrl-C 结束。采样开始后新创建的
root/child Proc 也会产生带 16-byte 名称的 `WAIT_START` 事件；Trace Viewer 以 proc ID
作为 tid、Proc 名称作为 `thread_name`。时间轴按 `1 cycle = 1 ns` 显示。

### 4.5 runner 日志

每次运行会创建独立目录：

```text
build/testcluster-runs/<timestamp>-<pid>/
├── dayan_server.log
├── operator.log
└── testcluster-tail.raw
```

- `operator.log`：Host demo、转发后的 Device printf 和数值校验结果；
- `dayan_server.log`：firmware/Service/RPC 错误；
- `testcluster-tail.raw`：TestCluster 全屏 PTY 输出的最后一段，包含 ANSI 控制字符。

查看最近一次运行：

```sh
latest_run=$(find ./build/testcluster-runs \
  -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | \
  sort -n | tail -1 | cut -d' ' -f2-)
sed -n '1,240p' "$latest_run/operator.log"
sed -n '1,240p' "$latest_run/dayan_server.log"
strings "$latest_run/testcluster-tail.raw" | tail -200
```

## 5. 手工启动：三个终端的完整流程

手工流程适合需要执行 `n/nd`、观察当前周期、检查 FUSE 请求或进入波形调试的情况。
以下示例使用一个 Cluster、四个 Core，并为本次运行选择独立 socket。

### 5.1 公共变量

三个终端中使用一致的路径：

```sh
export NPU_TC_ROOT=/home/wangzhe/npu-cluster-sim
export NPU_TC_BIN="$NPU_TC_ROOT/TestCluster"
export NPU_TC_DRAM="$NPU_TC_ROOT/configs/LPDDR5_8Gb_x16_6400_4ch.ini"
export NPU_DEVICE_PREFIX=/tmp/virtual_npu
export DAYAN_SERVER_SOCKET=/tmp/dayan-testcluster.sock
```

若使用源码仓库中的默认 build，只需替换三项 TestCluster 变量：

```sh
export NPU_TC_ROOT=/home/wangzhe/npu-cluster/sim
export NPU_TC_BIN="$NPU_TC_ROOT/build/TestCluster"
export NPU_TC_DRAM="$NPU_TC_ROOT/configs/LPDDR5_8Gb_x16_6400_4ch.ini"
```

启动前确认 mount point 没有被其它活跃进程使用：

```sh
mountpoint "$NPU_DEVICE_PREFIX" || true
test ! -S "$DAYAN_SERVER_SOCKET"
```

如果该 mount 属于另一个活跃 TestCluster，不能卸载或复用，应为本次运行选择其它目录。

### 5.2 终端 A：启动 TestCluster

```sh
cd "$NPU_TC_ROOT"
SIM_DRAM_CONFIG="$NPU_TC_DRAM" \
SIM_MOUNT_POINT="$NPU_DEVICE_PREFIX" \
"$NPU_TC_BIN"
```

等待界面出现 `Starting main CLI loop...` 和“等待命令”，并在另一个终端确认：

```sh
ls -l "$NPU_DEVICE_PREFIX"/xdma0_h2c_0 \
      "$NPU_DEVICE_PREFIX"/xdma0_c2h_0 \
      "$NPU_DEVICE_PREFIX"/xdma0_events_0
```

当前 TestCluster 不提供 `xdma0_user`。Dayan FUSE HAL 会通过 H2C/C2H 的物理控制窗口
访问寄存器，因此缺少 `_user` 文件不是错误。

### 5.3 终端 B：启动 dayan_server

四核通用配置：

```sh
server_log=./build/testcluster-dayan_server.log
./build/host/driver/dayan_server \
  --socket "$DAYAN_SERVER_SOCKET" \
  --firmware ./build/device/npu-os.elf \
  --api-elf ./build/device/npu-os-api.elf \
  --device-prefix "$NPU_DEVICE_PREFIX" \
  --cluster-count 1 \
  --core-mask 0xf \
  >"$server_log" 2>&1 &
dayan_server_pid=$!
```

只调试单核算子时可把 `--core-mask` 改成 `0x1`，firmware 启动会更快。若希望先跑单核
再跑四核，则一开始就使用 `0xf`，不需要重启 Server。

Server 在创建 Unix socket 前会打开 HAL、预留 BL1 段、分配并配置 Atomic Cache
backing、加载 firmware、启动指定 Core 并等待 BL1 ready。Server 会在每个可能阻塞的
阶段之前写日志并立即刷新，因此可以直接观察进度：

```sh
tail -f "$server_log"
```

典型阶段依次包括 `[startup] opening HAL`、
`[firmware C0] configuring Atomic Cache backing`、
`[firmware C0] waiting for BL1 ready` 和最终的 `[ready]`。socket 在 `[ready]` 之前尚未
创建是正常现象；若长时间没有进展，最后一条日志就是当前等待点。

可以观察 Server 是否已经把第一笔 FUSE 请求排入内核：

```sh
ps -L -p "$dayan_server_pid" -o pid,tid,stat,wchan:32,comm
```

常见等待点是 `request_wait_answer`。此时返回终端 A 推进仿真。

### 5.4 终端 A：先 prime，再连续运行

在 TestCluster CLI 中输入：

```text
n 1
```

确认这一个周期执行结束并重新出现“等待命令”。随后输入：

```text
c
```

`c` 会持续推进 RTL，直到按 `Ctrl-C`。不要在 Server 的第一笔 FUSE 请求尚未排队前
抢先发送 `c`；实践中这容易造成启动时序不稳定或 Server 初始化超时 `-110`。

CLI 的其它常用命令：

- `n <cycles>`：执行固定周期数；
- `nd <cycles>`：执行固定周期数并用于波形调试；
- `c`：Host 驱动场景所需的连续运行；
- `h`：显示 CLI 帮助；
- `quit`：退出模拟器。

周期后缀只使用 CLI 支持的 `k/K`。不要使用 `n 1g`。普通正确性测试不要开启完整波形，
长时间 FST 会占用大量磁盘并显著拖慢仿真。

### 5.5 终端 B：等待 Server socket

TestCluster 进入 `c` 后等待 socket：

```sh
while [ ! -S "$DAYAN_SERVER_SOCKET" ]; do
  if ! kill -0 "$dayan_server_pid" 2>/dev/null; then
    sed -n '1,240p' "$server_log"
    break
  fi
  sleep 0.2
done
ls -l "$DAYAN_SERVER_SOCKET"
```

### 5.6 终端 C：运行 Runtime demo

Device printf：

```sh
DAYAN_SERVER_SOCKET="$DAYAN_SERVER_SOCKET" \
  ./build/demo/print_demo 1 2 3 4 5 6 7
```

单核 matmul：

```sh
DAYAN_SERVER_SOCKET="$DAYAN_SERVER_SOCKET" \
  ./build/demo/matmul_bf16_1c
```

四核 smoke matmul：

```sh
DAYAN_SERVER_SOCKET="$DAYAN_SERVER_SOCKET" \
  ./build/demo/matmul_bf16_4c 1024 512 256
```

四核完整默认 case：

```sh
DAYAN_SERVER_SOCKET="$DAYAN_SERVER_SOCKET" \
  ./build/demo/matmul_bf16_4c
```

在已有 Server 上同时采样四核 smoke case：

```sh
DAYAN_SERVER_SOCKET="$DAYAN_SERVER_SOCKET" \
  ./build/host/dyperf/dyperf \
    --output ./build/matmul-4c-trace.json \
    -- ./build/demo/matmul_bf16_4c 1024 512 256
```

同一个 Server 可以并行接受 `dyperf` 和多个 Host Runtime；算子能否同时执行还受 Core、
proc slot、URAM、coherent memory 和 resident memory 配额约束。不要同时启动多个 runner
或其它生命周期脚本去管理同一个 TestCluster mount point；那是模拟器实例管理冲突，
不是 Server 的多 Session 限制。

## 6. TestCluster 启动期间发生了什么

### 6.1 BL1 与 RPC

`dayan_server` 解析 `npu-os.elf` 的 `PT_LOAD` 段，将其预留并写入模拟 DRAM。随后为每个
启用 Core 向 BL0 mailbox 写入 BL1 entry、置位 MSIP，并等待 BL1 通过 Host IRQ enable
状态宣告 ready。BL1 启动 scheduler、RPC receive/send proc 和 profile proc。

Host kernel 第一次提交时，Server 使用 `npu-os-api.elf` 解析 Device object 的公开符号，
完成 RISC-V `ET_REL` 重定位，将代码写入目标 Cluster，并对已启动 Core 执行 I-cache
flush。之后通过 `RPC_PROC_CREATE` 创建 root proc。

### 6.2 Atomic Cache backing

Coherent window 为 `0x00400000--0x007fffff`。它不是普通 DRAM，必须先给 Atomic Cache
配置合法的 backing memory。Server 在 firmware 固定段预留完成后，为每个 Cluster：

1. 从 `0x80000000` 起的普通 DRAM 分配 4 MiB；
2. 要求 backing 地址按 4 MiB 对齐并避免覆盖 firmware/kernel/buffer；
3. 向物理控制寄存器 `0x4008` 写入 64-bit backing base；
4. 轮询 `0x4000`，等待 busy 清零且 done 置位；
5. 在 Server 生命周期内持有该 allocation。

因此旧 RTL 中不合法的复位默认 backing base 不会被算子使用，Host 配置值才是实际值。
如需诊断，可在 Server 完成启动后读取控制页：

```sh
dd if="$NPU_DEVICE_PREFIX/xdma0_c2h_0" \
  bs=1 skip=16384 count=16 status=none | od -An -tx8 -v
```

第一项是状态，正常完成通常为 `2`；第二项是动态 backing 地址，例如：

```text
0000000000000002 0000000080c00000
```

地址不要求固定为 `0x80c00000`，但必须位于合法 DRAM region 且 4 MiB 对齐。

### 6.3 四核 URAM 与数据广播

每个 Core 的 URAM window 从 `0x00800000` 起，Core 间 stride 为 2 MiB：

```text
Core 0: 0x00800000
Core 1: 0x00a00000
Core 2: 0x00c00000
Core 3: 0x00e00000
```

`matmul_bf16_4c` 在四个 Core 间共享 coherent resource 状态，并通过远端 URAM window
广播 A/B tile。Cluster barrier 和 shared resource 使用硬件 AMO，不使用不受支持的
LR/SC。

## 7. 如何判断一次运行成功

一次有效测试应同时满足：

1. TestCluster 在 Host demo 结束前没有退出或触发 RTL assertion；
2. `dayan_server` 成功创建 socket，并在 demo 结束前保持运行；
3. Host demo 退出码为 0；
4. `print_demo` 输出预期数量的 grid 行，或 matmul 输出 `[host] PASS:`；
5. matmul 的 `max_abs/max_rel` 和逐元素校验满足程序阈值；
6. 使用 `dyperf` 时，trace JSON 能被解析且包含目标 Hart/proc 的状态事件。

当前 demo 的典型成功输出：

```text
[host] launch matmul_bf16_4c: A[1024,256] x BT[512,256] -> C[1024,512], ...
[host] max_abs=0 max_rel=0 at C[0,0]
[host] PASS: all 524288 outputs match the BF16 reference
```

TestCluster 的 `Narrow read/write burst may cause issues` 或
`burst crosses 4K boundary` 是模拟器诊断信息，不能单独判定算子失败。只有同时出现
trap/assertion、Server/Host 非零退出、timeout 或数值 mismatch 时，才应把测试判为失败。

## 8. 常见问题

### 8.1 runner 报 missing TestCluster

检查 `--sim-bin`。源码仓库默认 binary 是：

```text
/home/wangzhe/npu-cluster/sim/build/TestCluster
```

### 8.2 runner 报 missing DRAM configuration

通常是把 `--sim-dir` 错设成了 `.../sim/build`。应使用：

```text
--sim-dir /home/wangzhe/npu-cluster/sim
--sim-bin /home/wangzhe/npu-cluster/sim/build/TestCluster
```

### 8.3 FUSE endpoint 没有出现

确认：

- TestCluster 已进入主 CLI；
- `SIM_DRAM_CONFIG` 是存在的绝对路径；
- `SIM_MOUNT_POINT` 可创建且未被其它实例占用；
- TestCluster 进程仍然存活。

### 8.4 Transport endpoint is not connected

这是异常退出留下的 stale FUSE mount。先确认没有活跃 TestCluster 使用该目录，再清理
精确的 mount point：

```sh
fusermount3 -u "$NPU_DEVICE_PREFIX" ||
  fusermount3 -uz "$NPU_DEVICE_PREFIX"
```

runner 会自动处理它自己目标目录上的 stale mount，但不会卸载其它路径。

### 8.5 dayan_server socket 一直不出现

依次检查：

1. TestCluster 是否已经执行 `c`；
2. 是否遵循“Server 请求排队 -> `n 1` -> `c`”的顺序；
3. Server 是否仍存活；
4. `dayan_server.log` 是否出现 firmware、Atomic Cache 或 BL1 ready 错误；
5. runner 的 `--startup-timeout` 是否过短。

根据 Server log 的最后一条阶段日志定位等待点。例如停在 `[startup] opening HAL` 或
firmware 装载阶段时，可通过 `wchan` 进一步判断是否阻塞在正常的 FUSE
`request_wait_answer` 中。

### 8.6 四核算子立即返回 -19

`-19` 是 `-ENODEV`。最常见原因是 Server/runner 只启用了 Core 0。四核 launch 必须使用：

```text
--core-mask 0xf
```

并确保 TestCluster 本身是四核配置。

### 8.7 返回 -110

`-110` 是 `-ETIMEDOUT`。启动阶段通常与没有连续推进 RTL、prime 顺序错误或
`--startup-timeout` 太短有关；算子阶段则检查 kernel launch timeout、Device trap、
异步 LSU/MAT produce 和各 Core 是否全部在线。

### 8.8 卡在第一个 Atomic/Cluster barrier

先读取 `0x4000/0x4008` 控制页，确认 status done 且 backing 地址合法、4 MiB 对齐。
若 backing 正常，再检查 Device object 中是否混入 LR/SC，以及四个 root proc 是否都已
创建。当前正确路径使用 AMOSWAP，初始化 barrier 的 generation 应从 0 推进到 1。

### 8.9 需要更多 Device 日志

4c matmul 可在构建时开启 trace：

```sh
make -B -C ./demo -j72 MATMUL_4C_TRACE=1 all
```

trace 会显著增加 Device printf/RPC 流量并拖慢 RTL，只用于定位问题。完成后恢复：

```sh
make -B -C ./demo -j72 MATMUL_4C_TRACE=0 all
```

### 8.10 dyperf 与算子并发问题

`profile start failed: Device or resource busy` 表示已有 profiler 占用了全局 profile，
不是 Server 只支持一个客户端。先确认是否已有 `dyperf`：

```sh
pgrep -af 'dyperf|dayan_server'
ss -xnp | grep "$DAYAN_SERVER_SOCKET"
```

不同进程各自调用 `npu_runtime_open()` 会建立独立 Session，可以并行使用同一 Server。
同一个 `npu_runtime_t` 内的 control API 则由一把 mutex 按请求串行化；不要在一个线程做
无限 profile read、同时期待另一个线程通过同一 Runtime 句柄提交 kernel。独立运行
`dyperf`，或使用它的 `-- COMMAND` 模式即可把长轮询放到单独 Session。

## 9. 正确停止与清理

使用 runner 时无需额外操作。它会只停止本次创建的进程并清理本次 socket/mount。

手工运行结束后：

1. 在 TestCluster 的 `c` 运行界面按 `Ctrl-C`；
2. CLI 返回后输入 `quit`，等待 TestCluster/FUSE thread 完整退出；
3. 在终端 B 停止本次 `dayan_server`；
4. 仅在 mount 仍存在时卸载本次 mount point；
5. 删除本次使用的 socket。

```sh
kill "$dayan_server_pid" 2>/dev/null || true
wait "$dayan_server_pid" 2>/dev/null || true
if mountpoint -q "$NPU_DEVICE_PREFIX"; then
  fusermount3 -u "$NPU_DEVICE_PREFIX" ||
    fusermount3 -uz "$NPU_DEVICE_PREFIX"
fi
rm -f "$DAYAN_SERVER_SOCKET"
```

连续命令被 `Ctrl-C` 中止后，TestCluster UI 可能把该条 `c` 命令显示为 `Failed`。如果
Host 算子已经报告 PASS 且清理发生在其后，这只是交互命令的停止状态，不是算子失败。

## 10. 推荐的日常验证顺序

1. `make ... check`，确认 Host、firmware ABI 和 Device ISA 静态检查通过；
2. 用 cmodel 运行目标算子的完整数值 case；
3. 用 TestCluster runner 跑 `print_demo`，确认 BL0/BL1、RPC 和 Device printf；
4. 用 TestCluster 跑单核或一个对齐的四核 Cluster tile；
5. 只有需要发布回归或硬件时序证据时，才运行 `2048^3` 长 case 或开启波形；
6. 以 Host 退出码和数值参考校验为最终判据，并保留本次 runner 日志目录。
