# NPU host runtime

这里提供与本仓库 BL0/BL1 协议配套的 Linux 用户态驱动库。它直接打开 XDMA
字符设备；在 `rvtng-cmodel` 下则打开 QEMU 暴露的同名 FUSE 文件，因此上层接口不需要
区分模拟器和 FPGA。

## 能力与分层

- `npu_driver.[ch]`：`pread/pwrite` 设备访问、IRQ mailbox、MSIP doorbell 和 BL0
  启动确认；
- `npu_elf.c`：校验并加载 RV64 little-endian `ET_EXEC` BL1 的所有 `PT_LOAD`，清零
  `p_memsz - p_filesz`，再启动选定 hart；
- `npu_rpc.c`：后台服务 BL1 的输出、清空、扩堆、归还堆和 trap RPC，同时匹配
  Host→BL1 调用的响应；
- `npu_memory.c`：管理普通 DRAM 与 coherent window，阻止固件、Host buffer、动态
  Device heap 和 launch 描述符重叠；
- `npu_runtime.[ch]`：内存搬运、raw RPC、创建 proc、等待完成和资源回收 API。

所有接口返回 0 或负的 Linux errno。每个 hart 同时只允许一笔 Host→BL1 RPC；不同
hart 可并行。后台线程必须贯穿 BL1 生命周期，因为 BL1 在创建大栈或运行算子期间也
可能反向申请堆、输出或报告异常。

## 构建与链接

顶层 `make` 会生成 `build/host/runtime/libnpu_runtime.a`。也可以单独构建：

```sh
make -C host/runtime
cc app.c -Ihost/include build/host/runtime/libnpu_runtime.a -pthread
```

从仓库根目录执行 `make install PREFIX=/opt/dayan` 后，公共头文件位于
`/opt/dayan/include`，`libDayanRuntime.a`、`libDayanRuntime.so` 和兼容库
`libnpu_runtime.a` 位于 `/opt/dayan/lib`，固件及公开 API ELF 位于
`/opt/dayan/lib/dayan/firmware`。

最小使用流程：

```c
npu_runtime_options_t options;
npu_runtime_t *runtime;
npu_launch_desc_t desc;
npu_launch_t *launch;
npu_launch_info_t result;
uint64_t code, data;

npu_runtime_options_init(&options);
npu_runtime_open(&runtime, &options);
npu_runtime_load_bl1_file(runtime, "build/npu-os.elf", 0xf);

npu_runtime_alloc(runtime, code_size, 64, &code);
npu_runtime_copy_to_device(runtime, code, code_image, code_size);
npu_runtime_alloc(runtime, data_size, 64, &data);
npu_runtime_copy_to_device(runtime, data, host_data, data_size);

npu_launch_desc_init(&desc);
desc.hart = 0;
desc.entry = code;
desc.argc = 1;
desc.args[0] = data;
npu_runtime_launch(runtime, &desc, &launch);
npu_launch_wait(launch, NPU_TIMEOUT_INFINITE, &result);
npu_runtime_copy_from_device(runtime, host_data, data, data_size);
npu_launch_release(launch);
```

当前库接收已经链接好的算子代码和明确的 Device entry；它不会把任意 `ET_REL` 算子
对象重定位。算子可预先针对 `build/npu-os-api.elf` 链接，也可由上层加载器完成重定位，
然后用 `npu_runtime_alloc` 和 `npu_runtime_copy_to_device` 上传。每次创建最多传 8 个
RV64 ABI 机器字参数。

`shared_size != 0 && shared_base == 0` 时，runtime 自动从 coherent window 分配并在
`npu_launch_release` 时回收；给出非零 `shared_base` 时，其生命周期由调用者管理。
launch 描述符在 Device 报告 `EXITED/FAILED` 前不会复用。若提交 RPC 超时，
`npu_runtime_launch` 仍会返回一个可查询的 handle，因为 doorbell 写出后不能判定 BL1
是否已经消费请求。

## 启动与 RPC 协议

加载器保留 ELF 的完整内存段（包括 BL1 的每核栈和初始堆），复制内容后向
`0x1000 + hart * 8` 写 entry，再置位 `0x20000 + hart * 4`。BL0 清 MSIP 表示已经消费
entry；runtime 还会等待 BL1 在 IRQ enable 中使能该 hart，才把固件标为 ready。

Host 创建 proc 时把固定 152-byte `rpc_proc_create_request` 放进普通 DRAM，发送
`RPC_PROC_CREATE (0x0100)`。BL1 校验 entry、栈、shared window 和参数，分配托管的
`proc`/栈/输出缓冲，返回 `RPC_PROC_CREATE_RESPONSE (0x8100)`。算子退出后，scheduler
在自己的栈上清理这些对象，并最后把同一描述符置为 `RPC_PROC_EXITED`；异常则置为
`RPC_PROC_FAILED`，同时通过 trap RPC 上报寄存器快照。

## cmodel 验证

环境默认使用 `~/rvtng-cmodel`、`~/npu-qemu/build/dl-riscv64` 和
`/tmp/virtual_npu`，均可通过 `RVTNG_CMODEL_DIR`、`RVTNG_QEMU`、
`NPU_DEVICE_PREFIX` 覆盖：

```sh
make check
./host/runtime/run_cmodel_smoke.sh
```

默认启动 1 个 hart；可用 `NPU_CMODEL_HARTS=4` 覆盖并验证四核 BL0/BL1 启动。

smoke test 会覆盖 BL1 ELF 加载、BL0 启动、普通搬运、create-proc、2 MiB 栈触发的
反向扩堆 RPC、proc 输出 RPC，以及非法指令 trap/失败完成。脚本只在没有现存 cmodel
进程时启动，并在退出时卸载自己的 FUSE mount、终止模拟器和清理对应 IPC 对象。
