# soc_mmap 后端

NPU IP 集成进 SoC、Host CPU 运行 RISC-V Linux 时使用的 HAL 后端与 Runtime 数据面。
`dayan_server` 和 Runtime 都通过 `/dev/mem`（或偏移相同的受限设备节点）的
`mmap` 访问 NPU。IRQ 采用轮询。

状态：只做过离线验证（`build/host/driver/tests/soc_mmap_test` 用普通文件代替
`/dev/mem`），并通过 RISC-V 交叉编译。SoC QEMU 尚未加入 NPU 模型，未在 QEMU NPU
或芯片上运行过。

## 启动命令

```sh
dayan_server --backend soc-mmap \
    --soc-base 0x<NPU IP 的 Host PA> \
    --soc-ddr-base 0x<为 NPU 保留的共享 DDR 的 Host PA> \
    --soc-ddr-size 0x<该 DDR 大小> \
    --soc-ddr-coherence uncached|coherent \
    [--soc-ddr-npu-base 0x80000000] \
    [--soc-memory-device /dev/mem] [--soc-data-device PATH] \
    [--soc-platform-control external] \
    --socket /tmp/dayan-server.sock \
    --firmware build/device/npu-os.elf --api-elf build/device/npu-os-api.elf \
    [--enable-mmu 0|1] [--core-mask 0xf] [--timeout-ms N]
```

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `--soc-base` | NPU Cluster IP 地址窗口在 Host 物理地址空间的基址，需页对齐 | 必填 |
| `--soc-ddr-base` | 共享 DDR 的 Host PA，需页对齐，必须已从 Linux 保留（见下文） | 必填 |
| `--soc-ddr-size` | 共享 DDR 大小，需页对齐；同时是 HAL Normal 内存的容量 | 必填 |
| `--soc-ddr-npu-base` | NPU 主端口看到的该 DDR 地址，需 4 KiB 对齐 | `DAYAN_DRAM_BASE`（0x80000000） |
| `--soc-ddr-coherence` | Host 与 NPU 对共享 DDR 的一致性模型 | 必填，无默认 |
| `--soc-memory-device` | Server 映射用的设备 | `/dev/mem` |
| `--soc-data-device` | 传给 Runtime 的设备，DDR 偏移须与前者相同 | 同上 |
| `--soc-platform-control` | 只有 `external`：clock/reset/电源由 BSP 负责 | `external` |

在 soc-mmap 后端下：
- `--cluster-count` 只能为 1；
- `--normal-memory-size-mib` 不可用，Normal 容量由 `--soc-ddr-size` 决定；
- 已移除 `--soc-window`；
- Xilinx DMA 的 `--dma-*` 选项会被拒绝。

## 地址模型

四种地址各不相同，不能混用：

| 地址 | 含义 | 来源 |
| --- | --- | --- |
| Host PA | `/dev/mem` 的 mmap 偏移 | `--soc-base`、`--soc-ddr-base` |
| IP offset | 模块在 Cluster IP 地址窗口中的位置。NPU 在 `CONFIG_CONTROL_BASE = 0` 处看到该窗口，因此它也是该模块的 NPU 侧地址 | `device/include/platform_regs.h` |
| NPU 侧 DDR PA | NPU 主端口访问共享 DDR 使用的地址，可以与 Host PA 不同 | `--soc-ddr-npu-base` |
| ClusterMMU VA | 开启 MMU 后固件与算子使用的地址，经页表翻译为 NPU 侧 PA。HAL 只处理 PA | `server/mmu.c` |

翻译规则：

```text
IP 窗口：Host PA = soc-base + IP offset            （IP offset < DAYAN_CLUSTER_APERTURE_SIZE）
DDR：    Host PA = soc-ddr-base + (NPU PA - soc-ddr-npu-base)
```

Normal DDR 不属于 IP 窗口，而是独立的 SoC 共享内存。依据是：rvtng QEMU 模型把 DRAM 作为
独立共享内存挂在 NPU 地址 0x80000000，IP 内部模块全在 16 MiB 以下；FPGA 路径中低 2 GiB
加偏移、DDR 不加偏移。因此 DDR 单独配置，不使用 `--soc-base`。

`--enable-mmu 0` 时，固件按链接地址 `CONFIG_RAM_BASE`（`device/Makefile` 中的
`RAM_BASE`，默认 0x80000000）直接访问内存，`--soc-ddr-npu-base` 必须等于它，否则 BL1
段预留会在启动阶段失败。`--enable-mmu 1` 时固件窗口按线性偏移映射到 Normal 池，NPU 侧 PA
可以是其他值。

## 固定布局

布局只来源于 `platform_regs.h`。`layout.c` 把它转成访问规则，其他源文件不再写死模块地址；
`hal/memory.c` 的默认区域也改为引用这些常量。

| 模块 | IP offset | 大小（4 核） | 规则 |
| --- | --- | --- | --- |
| Device mailbox | `0x001000` | 4 × 8 | 寄存器，64 位 |
| IRQ 控制（STATUS/ENABLE/SET/CLR/ENABLE_SET/ENABLE_CLR） | `0x002000` | 0x18 | 寄存器，32 位 |
| IRQ mailbox | `0x002020` | 4 × 8 | 寄存器，64 位 |
| Atomic-cache 控制（STATUS/BACKING） | `0x004000` | 0x10 | 寄存器，64 位 |
| ClusterMMU | `0x005000` | 0x1000 | 寄存器，32 位 |
| Resource init | `0x006000` | 4 | 寄存器，32 位 |
| CLINT MSIP | `0x020000` | 4 × 4 | 寄存器，32 位 |
| PCU | `0x030000` | 4 × 0x2000 | 寄存器，64 位 |
| Coherent（atomic-cache）窗口 | `0x400000` | 4 MiB | 内存 |
| URAM，每核一份 | `0x800000 + core × 0x200000` | 0x180000 | 内存，仅该核可见 |

窗口内的其他偏移（BootROM、CLINT 其余部分、保留区）Host 不访问，访问会返回 `-EFAULT`。

HAL 分配区域由 `npu_hal_soc_mmap_memory_regions()` 按上表和 DDR 参数生成：
- Normal 覆盖整段共享 DDR；
- Coherent；
- 每核一个 URAM。

后端打开时逐个检查区域：
- 每个区域必须完整落在同类内存模块内；
- URAM 必须标记为对应核私有；
- 区域不能放在寄存器上。

不满足任一条件则 `npu_hal_open()` 返回 `-EINVAL`。

## 访问规则

一次 HAL 访问必须完整落在三类目标之一：
- 共享 DDR；
- 一个内存模块（Coherent 窗口或某个核的 URAM）；
- 一个寄存器。

不满足时：
- 越界、跨模块、落在空洞：返回 `-EFAULT`；
- 地址溢出：返回 `-EOVERFLOW`；
- 寄存器宽度或对齐不符，或把寄存器当作批量拷贝目标：返回 `-EINVAL`。

### 失败时不写入的保证覆盖哪些接口

软件范围错误（越界、空洞、溢出、类型或宽度不符）在第一次设备访问之前检查，失败时不写入任何
数据。保证以单次调用为单位，只覆盖以下接口：

| 接口 | 预检范围 |
| --- | --- |
| `npu_hal_copy_to_address()` / `copy_from_address()`，以及基于它们的 `npu_hal_memory_write/read()`、`npu_hal_register_*()` | 整个请求 |
| `npu_hal_copy_address()`（Device 到 Device，经 64 KiB 分块中转） | 源和目的的完整范围，由 `copy_address_check` 在搬运第一块之前检查。只允许内存目标（共享 DDR、Coherent、URAM），寄存器返回 `-EINVAL` |
| Runtime `npu_hal_data_plane_transfer()` | token 与导出的 DDR 范围，在 mmap 和访问之前检查 |

不覆盖：
- 由多次 HAL 调用组成的操作，作为整体不保证。例如 ELF 段装载和清零按 64 KiB 分次调用，
  56 位 MMU 寄存器分 LO、HI 两次写入。每一次调用仍然各自预检。
- Xilinx DMA 后端没有 `copy_address_check`，`npu_hal_copy_address()` 仍逐块检查。
  HAL 通用层只增加了整体地址溢出检查。
- `npu_hal_copy_address()` 不支持源和目的范围重叠。

预检只能避免软件范围错误造成的部分写入。真实芯片上的总线错误、超时或复位可能打断任意一次
访问，这类情况无法保证原子性。

### 宽度规则

- **寄存器**：访问长度必须等于该寄存器宽度，且自然对齐；不会降宽，也不会拆分。
  一次访问编译为一条 `sd/sw/ld/lw` 指令（已用交叉编译反汇编确认），前后各一条
  `fence iorw,iorw`。
- **56 位 MMU 寄存器**：由 `hal/mmu.c` 拆成两次 32 位写入，先 LO 后 HI。
- **PCU**：先写 `PCU_CONTROL` 锁存所有计数器，再逐个做单次 64 位读取（`hal/pcu.c`）。
  32 位拆分读取会撕裂，因此被拒绝。Host 必须是 64 位（`io.c` 编译期检查）。
- **内存**：自然对齐的 volatile 访问，最大 8 字节，只在不对齐的首尾收窄。

## 共享 DDR：保留与所有权

HAL 分配器不会替 Linux 保留物理内存。`--soc-ddr-base/size` 必须指向 BSP 已从 Linux 可用 RAM
中保留的区域，不能与 kernel、OpenSBI 或用户进程内存重叠。不要默认 NPU 侧的 0x80000000 等于
Host PA 0x80000000。当前 `nanhu-v5-dayan` 设备树把 `0x80000000` 起的内存全部交给 Linux
（2 GiB 配置），没有 `reserved-memory` 节点。使用前需由 BSP 增加保留区，例如：

```dts
reserved-memory {
    #address-cells = <2>;
    #size-cells = <2>;
    ranges;
    npu_ddr: npu@f0000000 {
        reg = <0x0 0xf0000000 0x0 0x10000000>;
        no-map;
    };
};
```

需要 `no-map`：否则内核线性映射仍以可缓存属性映射这段内存，会与用户态映射形成属性不同的
别名。具体地址和大小由 BSP 决定，上例只是格式示例。

## 缓存一致性

以下三者都不能保证 DDR 一致性：
- `/dev/mem` 的 `O_SYNC`：只决定内核选择什么页表属性，见下文；
- volatile 访问：只阻止编译器合并或省略访问；
- fence：只保证顺序，不清理或失效任何缓存。

### `/dev/mem` 实际选择的属性

`drivers/char/mem.c` 在 `O_SYNC`（包含 `O_DSYNC` 位），或物理地址高于 `high_memory` 时调用
`pgprot_noncached()`。在 RISC-V（`arch/riscv/include/asm/pgtable.h`）上，两个函数选择的
PBMT 类型不同：

| 函数 | PTE 位 | Svpbmt 类型 |
| --- | --- | --- |
| `pgprot_noncached()`，`/dev/mem` 使用 | `_PAGE_IO` | PBMT=10 IO：不可缓存、非幂等、强顺序 |
| `pgprot_writecombine()` / `pgprot_dmacoherent()` | `_PAGE_NOCACHE` | PBMT=01 NC：不可缓存、幂等、弱顺序的主存 |

因此即使启用了 Svpbmt，`/dev/mem` 加 `O_SYNC` 得到的也是 IO 类型，而不是 NC。IO 类型下，
共享 DDR 的每次访问都按 I/O 处理，性能较低，且非对齐访问可能失败；本后端的内存拷贝只使用
自然对齐访问。

这两个位只有在 CPU 实现 Svpbmt，且内核以 `CONFIG_RISCV_ISA_SVPBMT` 编译并在启动时检测到
该扩展时才会写入。否则 `ALT_SVPBMT` 得到 0，`pgprot_noncached()` 不改变 PTE，访问属性
完全由硬件 PMA 决定。

当前平台：
- 设备树声明 CPU 支持 `svpbmt`；
- 内核（`riscv-linux/.config`）未开启 `CONFIG_RISCV_ISA_SVPBMT`；
- 因此 DDR 的用户态映射完全按 PMA 处理，普通 DRAM 通常是可缓存的；
- IP 窗口中的寄存器、Coherent 和 URAM 同样由 PMA 决定，属性需由 SoC 集成确认。

### 一致性模式

`--soc-ddr-coherence` 没有默认值，由部署者声明平台实际满足的模型：

| 模式 | 前提 | 当前平台 |
| --- | --- | --- |
| `uncached` | Host 对该 DDR 的实际映射不可缓存，NPU 侧也不缓存 | 不满足：内核未启用 Svpbmt，映射按 PMA 可缓存 |
| `coherent` | 互连保证 NPU 访问与 Host 缓存一致 | 未知，需 SoC 集成确认 |

两种模式都不需要软件维护缓存。后端无法检测声明是否属实，声明错误会导致数据不一致。

启用 `CONFIG_RISCV_ISA_SVPBMT` 并为保留区加上 `no-map`，是满足 `uncached` 的可能途径，
但不等于平台已经验证。还需确认：
- CPU 实际实现并启用了 Svpbmt；
- 该 DDR 的 PMA 允许 PBMT 覆盖属性；
- 没有其他可缓存别名（包括 bootloader、OpenSBI 和其他主设备）；
- NPU 侧访问路径不经过缓存；
- 在目标硬件上实测确认。

内核支持 Zicbom（`CONFIG_RISCV_ISA_ZICBOM=y`），但用户态能否执行 `cbo.clean/inval`
取决于内核设置的 `senvcfg`，当前没有对外接口。

如果平台两种模式都不满足，需要增加内核缓存维护辅助层。接入点是
`npu_hal_soc_mmap_ddr_sync_for_device/for_cpu()`（`io.c`）：
- Server 的 DDR 拷贝（`soc_copy()`）和 Runtime 的传输（`dataplane.c`）都调用这两个函数，
  顺序一致：fence → 拷贝 → `for_device`，或 fence → `for_cpu` → 拷贝，最后 fence。
- 导出给 Runtime 的数据中带有一致性模式，Runtime 拒绝未知模式。

QEMU 或文件测试通过，不能证明真实 SoC 的缓存一致性。

## Runtime 传输

- 只允许传输到共享 DDR。目标是 Coherent 或 URAM 的 Buffer 会在 `prepare` 阶段返回
  `-ENOTSUP`：这两个模块在 IP 窗口内，Host 批量访问路径以及 atomic-cache 窗口相对 backing
  DDR 的语义尚未确认。
- Server 导出（协议 layout 3）带有 DDR 的 Host PA、大小和一致性模式。Runtime 在 import 时
  校验这些值（非零、页对齐、不溢出），并固定为唯一允许的范围。layout 2 由第一个 token
  决定范围，指向 IP 窗口的第一个 token 会被接受并写入 mailbox；layout 3 不再如此。
- 每个 token（包括第一个）的 Host PA 和大小都必须与导出值一致，偏移和长度必须落在其中，
  且不能溢出。不满足时返回 `-EACCES`，不执行 mmap，也不访问内存。DDR 只在第一个有效 token
  到达时映射。
- token 校验只能防止过期或损坏的 token 经本 provider 访问其他物理地址，不能约束进程本身：
  持有 `/dev/mem` 描述符的进程可以自行 mmap 内核允许的任意物理地址，与 token 无关。
  需要限制时，用 `--soc-data-device` 指定只暴露共享 DDR 的节点。

## Clock、reset、电源

`external` 模式下，`dayan_server` 启动前由 bootloader/BSP 完成上电、解除隔离、配置时钟和
释放复位，后端不做任何操作。`soc_platform_prepare()` 与 `npu_hal_cluster_reset()` 为占位：
后者返回 `-ENOTSUP`，Server 不调用。

## 待确认项

1. IP 窗口在 SoC 中的 Host PA（`--soc-base`），以及窗口内各模块 offset 是否与
   `platform_regs.h` 一致。SoC QEMU 尚未加入 NPU，表中数值来自 rvtng QEMU 模型和 FPGA
   寄存器表。
2. 各寄存器的实际访问宽度：
   - 64 位寄存器是否接受单次 64 位访问；
   - MMU LO/HI 的锁存语义（HI 写入时提交，还是各自立即生效）。
3. PCU 窗口大小及锁存协议（rvtng QEMU 尚未建模 PCU）。
4. NPU 主端口访问共享 DDR 的地址映射，即 `--soc-ddr-npu-base` 与 Host PA 的关系。
5. 共享 DDR 的一致性模型，以及 BSP 保留区（地址、大小、`no-map`）。
6. Coherent 窗口与 URAM 是否允许 Host 批量访问，访问宽度是否受限。
7. 多 Cluster 的基址关系（当前只支持 1 个 Cluster）。
8. Clock/reset 序列及停稳确认。
9. Host IRQ 接入方式（UIO 或平台驱动；内核未启用 `CONFIG_UIO`）。

## 验证范围

| 验证 | 证明了什么 | 不能证明什么 |
| --- | --- | --- |
| `soc_mmap_test`（普通文件代替 `/dev/mem`） | 地址翻译、布局与区域校验、寄存器宽度/对齐规则、失败不写入（含大于 64 KiB 的 Device 到 Device 拷贝）、IRQ 轮询、PCU 读取顺序、数据面导出范围绑定与 token 校验、reset 占位 | 总线宽度、访问顺序、缓存、寄存器语义 |
| RISC-V 交叉编译与反汇编 | 寄存器访问为单条指令，fence 为 `fence iorw,iorw` | 在真实 RISC-V 上运行 |
| 文件模拟启动 `dayan_server` | 参数解析、区域生成、HAL 打开 | 固件启动（需要 NPU） |
| 需要 QEMU NPU 模型 | 模块 offset、寄存器宽度、BL1 启动、RPC、H2D/D2H 与 NPU 读写一致 | 缓存一致性、时序、总线错误 |
| 需要真实硬件 | 缓存一致性、PMA/Svpbmt 属性、LO/HI 锁存、clock/reset、总线错误行为 | — |
