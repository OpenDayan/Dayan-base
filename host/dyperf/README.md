# dyperf

`dyperf` 通过 Dayan Runtime 开启 Device proc profile，持续读取各 Hart 的状态变化事件，
并输出 Chrome Trace Viewer 可读取的 JSON。工具不直接访问 HAL，也不暴露算子调度所用
的物理 Cluster/Core 选择；只有 profile 批次携带 `(cluster, hart)`，用于构造时间线。

```sh
make -C host/dyperf
build/host/dyperf/dyperf --duration-ms 1000 -o trace.json
```

常用参数：

- `--socket PATH`：Dayan Server socket。
- `--duration-ms MS`：采集时长；设为 `0` 时持续采集到 Ctrl-C。
- `--buffer-size SIZE`：全局 Device profile 缓冲区，默认 16 MiB，支持 `K/M/G` 后缀。
- `--output FILE`：输出文件，默认 `dayan-trace.json`。

每个 Hart 映射为独立 `pid`，每个 Device proc ID 映射为独立 `tid`。Device 在 START 时
先报告所有现存 proc 的初始状态，之后报告每次状态切换后的目标状态；工具把这些事件
展开成 `RUNNING`、`READY`、`WAIT_POLL`、`WAIT_EVENT`、`WAIT_START`、`WAIT_EXIT`、
`WAIT_PRODUCE` 和 `KILL_WAIT_PRODUCE` 状态区间。JSON 中的 `B/E` 只用于表达区间，不是
Device 原始事件类型。工具会按 Hart 独立扩展 32-bit cycle 回绕；因为当前 ABI 没有约定
固定时钟频率，JSON 的 `ts` 保留设备 cycle 单位，不换算为纳秒或微秒。
