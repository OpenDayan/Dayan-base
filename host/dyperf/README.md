# dyperf

`dyperf` 通过 Dayan Runtime 开启 Device proc profile，持续读取各 Hart 的状态变化事件，
并输出 Chrome Trace Viewer 可读取的 JSON。工具不直接访问 HAL，也不暴露算子调度所用
的物理 Cluster/Core 选择；只有 profile 批次携带 `(cluster, hart)`，用于构造时间线。

```sh
make -C host/dyperf
build/host/dyperf/dyperf --duration-ms 1000 -o trace.json
```

持续观察当前 Server，直到按下 Ctrl-C：

```sh
build/host/dyperf/dyperf --follow -o trace.json
```

最常用的方式是让 `dyperf` 直接启动待观测算子：

```sh
build/host/dyperf/dyperf -o trace.json -- \
  build/demo/matmul_bf16_4c 1024 512 256
```

`dyperf` 会先开启设备级 profile，再 fork/exec 算子。子进程通过自己的 Dayan Runtime
Session 连接同一个 Server；`dyperf` 会把自身选择的 socket 写入子进程的
`DAYAN_SERVER_SOCKET`。未显式指定 `--duration-ms` 时，采样持续到子进程退出；工具最终
返回算子的退出码。显式指定时，该值是最长采样时间，采样结束后仍会等待算子退出。

常用参数：

- `--socket PATH`：Dayan Server socket；未指定时优先读取
  `DAYAN_SERVER_SOCKET`，再使用 `/tmp/dayan-server.sock`。
- `--duration-ms MS`：采集时长；设为 `0` 时持续采集到 Ctrl-C。
- `--follow`：持续采集到 Ctrl-C；command 模式下则持续到子进程退出。不能与
  `--duration-ms` 同时使用。
- `--buffer-size SIZE`：全局 Device profile 缓冲区，默认 16 MiB，支持 `K/M/G` 后缀。
- `--output FILE`：输出文件，默认 `dayan-trace.json`。

## PCU 轮询

```sh
# 每轮扫描结束后等待 0.1 秒；也可使用 1 或 1s。
build/host/dyperf/dyperf --pcu-interval 0.1 -o trace.json -- \
  build/demo/matmul_bf16_4c 1024 512 256

# 扫完一轮立即开始下一轮。
build/host/dyperf/dyperf --pcu-interval 0 --follow -o trace.json
```

`--pcu` 等价于默认间隔 0.1 秒的 PCU 采集。PCU 默认关闭，需要同时更新并重启
`dayan_server`。`--clock-hz` 指定设备周期频率，默认 `1e9`；该频率同时用于 proc 和
PCU 时间轴以及 LSU 带宽换算。`--lsu-beat-bytes` 默认 64，对应当前 RTL 的 512-bit
LSU AXI 数据口。这里统计成功传输的总线字节，不扣除末尾 beat 中未使用的字节。

轮询通过 `npu_profile_pcu_read()` 走 Runtime control socket。Server 每次扫描一个
启用的 Cluster，返回四个物理 Core 的快照；`NPU_PCU_ROUND_END` 表示本轮结束。
只有当前 profile owner 可以触发快照。扫描时释放 Server 全局锁，允许其他 Session
继续提交工作。每轮之间继续读取 proc 事件；连续模式下这次 proc read 不阻塞。
启动 COMMAND 前采集基线，停止 profile 前补采最后一轮，因此短于轮询间隔的命令
也能产生区段指标。实际采样间距包括扫描耗时，不能由 Host 的等待秒数推算设备周期。

Host 按 `~/npu-cluster` 的 `PerformanceCounter.scala` 和 `RVTPMU.scala` 读取
每核事件 0..111，不读取空洞或 uncore 事件。写入快照控制寄存器的 bit 0，等待写响应
完成，再逐个发起 8 字节读取；不能对该 PCU 发起多 beat 的突发扫描。

### 时间戳预留约定

当前 RTL 尚未提供周期时间戳；Host 先在 `device/include/platform_regs.h` 预留
`DAYAN_PCU_CYCLE(hart)`，后续按最终 RTL 映射调整此宏：

| 寄存器 | 物理地址 |
| --- | --- |
| Core `h` 的事件 `e` | `0x30000 + h * 0x2000 + e * 8` |
| Core `h` 的周期快照（预留槽位 1022） | `0x30000 + h * 0x2000 + 0x1ff0` |
| 四核全局快照控制 | `0x37ff8` |

周期快照必须是与该核计数器同步锁存的 64-bit `rdcycle` 时间戳，与 proc profile
同源、同起点。独立起点的 free-running 计数器不能直接对齐 proc 时间线。整个 Cluster
的时间戳都是零时返回 `-ENODATA`，`dyperf` 报错退出；单个从未启动的核为零时不输出
该核指标。时间戳相等且计数器未变时忽略重复快照，时间戳倒退时报错，不计算错误的
大区间。累积事件计数器按无符号 64-bit 差值处理回绕。

### 区段指标

令 `C = cycle_end - cycle_start`，`delta(e)` 为两次快照的事件计数差值。

| 指标 | 计算口径 |
| --- | --- |
| URAM 读/写平均占用率 | 每个 bank 的 `100 * delta(bank) / C`，对 32 个 bank 求均值 |
| URAM 读/写总体方差 | 上述 32 个百分比对均值的平方偏差之和除以 32，单位 `%²` |
| URAM 读/写标准差 | 总体方差的平方根，单位为百分点 |
| MAT 利用率 | `100 * delta(matExecEvent) / C` |
| LSU 读/写带宽 | `delta(beat) * beat_bytes * clock_hz / C / 1e9`，十进制 GB/s |
| VEC group 0/1 IPC | 各组 `delta(Issue) / C`，为发射 IPC |
| VEC 总 IPC | 两组发射 IPC 之和 |
| Scalar IPC | `delta(scalarCommit) / C`，为退休 IPC |

这些指标在对应 Hart 下输出为 Chrome Trace `C` 计数器轨道，值从区段起点生效。
`PCU intervals` 轨道同时输出 `X` 区间，含起止周期及全部指标；`PCU snapshot` 瞬时
事件保留周期和按事件编号排列的 112 个原始计数值。采集结束处的 `C=0` 是轨道终止点，
不代表又采集了一段空闲时间。跨 Cluster 的快照先后采集，各自使用设备时间戳定位。

## 并发模型

`dayan_server` 可以同时接受多个 Runtime Session，每个 Session 有独立的 control/data
socket，Server 为每条已接受连接创建处理线程。因此 `dyperf` 的阻塞式 profile read
只占用自己的 control socket，等待期间也会释放 Server 全局状态锁，不会阻止算子的
Runtime 连接、提交 memcpy 或 launch。

设备 profile 本身仍是全局独占资源，所以同一时刻只能有一个 `dyperf` 或一个
`npu_profile_start()` owner；第二个 profiler 会得到 `-EBUSY`。这不限制被观测的算子
数量。单个 Runtime 句柄内部的 control 调用仍按 socket 串行化；需要独立的长轮询时，
应像 `dyperf` 一样使用单独 Runtime Session。

每个 Hart 映射为独立 `pid`，并通过 `process_sort_index` 固定按 Cluster、Hart 编号升序
排列；同一 Hart 内的轨道通过 `thread_sort_index` 按 Proc 名称排序，同名 Proc 再按 tid
排序。Device 输入是紧密排列的 profile v4 字节流：普通状态记录为 8 B，`SLOT_DEFINE`
为 24 B。工具读取每个 8-B header 的 state/tag 决定记录长度，并保留跨 Runtime batch
的未完整记录。每次 slot 重新定义都会产生新的 generation，因此同一 slot 上先后运行的
Thread 不会合并为一个 `tid`。`scheduler` 轨道由工具根据当前 Proc 离开
`RUNNING` 到下一个 Proc 进入 `RUNNING` 之间的区间合成该轨道，不增加 Device profile
记录。Device 在 START 时为所有现存 Proc 先报告 `SLOT_DEFINE` 和初始状态，之后状态
切换只报告 8-B 目标状态；采样期间 slot 分配给新 Thread 时再报告一次 24-B 定义，名称
不会在每次状态切换中重复。工具把这些记录展开成
`RUNNING`、`READY`、`POLLING`、`WAIT_EVENT`、`WAIT_START`、`WAIT_EXIT`、
`WAIT_PRODUCE`、`WAIT_LSU`、`WAIT_MAT`、`WAIT_VEC`、`WAIT_MACRO`、`YIELD`、
`WAIT_JOIN`、`WAIT_RESOURCE` 和 `KILL_WAIT_PRODUCE` 状态区间。JSON 中的 `B/E` 只用于表达区间，不是
Device 原始状态记录类型。工具会按 Hart 独立扩展 48-bit counter 回绕，并按
默认 `1 cycle = 1 ns` 转换时间，可通过 `--clock-hz` 修改。Chrome Trace 的 `ts` 使用
带小数的微秒值保存精度，
同时设置 `displayTimeUnit: "ns"`。
