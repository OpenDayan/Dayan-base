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

## 并发模型

`dayan_server` 可以同时接受多个 Runtime Session，每个 Session 有独立的 control/data
socket，Server 为每条已接受连接创建处理线程。因此 `dyperf` 的阻塞式 profile read
只占用自己的 control socket，等待期间也会释放 Server 全局状态锁，不会阻止算子的
Runtime 连接、提交 memcpy 或 launch。

设备 profile 本身仍是全局独占资源，所以同一时刻只能有一个 `dyperf` 或一个
`npu_profile_start()` owner；第二个 profiler 会得到 `-EBUSY`。这不限制被观测的算子
数量。单个 Runtime 句柄内部的 control 调用仍按 socket 串行化；需要独立的长轮询时，
应像 `dyperf` 一样使用单独 Runtime Session。

每个 Hart 映射为独立 `pid`，每个 Device proc ID 映射为独立 `tid`，每条
`thread_name` metadata 使用该 Proc 的固定 16-byte 名称。Device 在 START 时先报告所有
现存 proc 的初始状态，之后报告每次状态切换后的目标状态；采样期间新建的 Proc 会先
产生带名称的 `WAIT_START`，因此新 tid 会被持续发现并立即命名。工具把这些事件展开成
`RUNNING`、`READY`、`WAIT_POLL`、`WAIT_EVENT`、`WAIT_START`、`WAIT_EXIT`、
`WAIT_PRODUCE` 和 `KILL_WAIT_PRODUCE` 状态区间。JSON 中的 `B/E` 只用于表达区间，不是
Device 原始事件类型。工具会按 Hart 独立扩展 32-bit cycle 回绕，并按
`1 cycle = 1 ns` 转换时间。Chrome Trace 的 `ts` 使用带三位小数的微秒值保存纳秒精度，
同时设置 `displayTimeUnit: "ns"`。
