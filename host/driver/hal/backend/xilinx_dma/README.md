# Xilinx DMA backend

One HAL backend and one Runtime data plane support both XDMA and QDMA.
`--dma-driver` selects device opening rules; all subsequent operations share
the same implementation and state. No XDMA symlinks are required for QDMA.

## QDMA FPGA startup examples

Run these examples from the repository root after building the server and
firmware and starting QDMA MM queue 0. The examples use one Cluster and
one Core. Set the PCI function, device name, queue, and topology for your
card. Both examples set NPU Normal memory to 30720 MiB (30 GiB). Adjust
the value to the Normal memory available on your card.

### Control BAR and user IRQ

The card must expose the control BAR and raise NPU user interrupt 0. The
driver must provide a usable `/dev/qdma01000-user_irq0` node.

```sh
sudo ./build/host/driver/dayan_server \
  --socket /tmp/dayan-server.sock \
  --firmware ./build/device/npu-os.elf \
  --api-elf ./build/device/npu-os-api.elf \
  --backend xilinx-dma --dma-driver qdma --dma-profile fpga \
  --dma-device-name qdma01000 --dma-pci-device 0000:01:00.0 --dma-queue 0 \
  --dma-control bar --dma-completion irq --dma-user-irq 0 \
  --cluster-count 1 --cores-per-cluster 1 --core-mask 0x1 \
  --normal-memory-size-mib 30720
```

### No control BAR or user IRQ

Use polling and access the NPU register window through the MM queue.
The bitstream must map that window at the FPGA profile offset
`0x4000000000`.

```sh
sudo ./build/host/driver/dayan_server \
  --socket /tmp/dayan-server.sock \
  --firmware ./build/device/npu-os.elf \
  --api-elf ./build/device/npu-os-api.elf \
  --backend xilinx-dma --dma-driver qdma --dma-profile fpga \
  --dma-device-name qdma01000 --dma-pci-device 0000:01:00.0 --dma-queue 0 \
  --dma-control dma --dma-completion polling \
  --cluster-count 1 --cores-per-cluster 1 --core-mask 0x1 \
  --normal-memory-size-mib 30720
```

## XDMA FPGA options

```sh
--backend xilinx-dma --dma-driver xdma --dma-profile fpga \
--dma-device-name xdma0 --dma-channel 0 --dma-event 0 \
--dma-completion irq
```

QDMA opens `qdma01000-MM-0` separately for reading and writing, and the custom
driver's `qdma01000-user_irq0` for IRQ notification. This event node is a local
QDMA driver extension, not a guarantee of the upstream QDMA driver. XDMA opens
`xdma0_h2c_0`, `xdma0_c2h_0`, optional `xdma0_user`, and `xdma0_events_0`.
Queue/channel and IRQ/event indices are independent.

`--dma-completion irq` requires an event node. Missing/inaccessible nodes fail
startup, and event read/poll errors do not silently switch to polling.
`--dma-completion polling` does not open an event node; HAL reads NPU IRQ status
and requests a 1,000 ns (1 us) sleep between polling passes. Actual wall-clock
period includes PCIe reads, OS scheduling and sleep latency; it is not a 1 MHz
timing guarantee. Polling does not disable interrupt generation in hardware.
The event read ABI for both drivers is a 32-bit notification count.

## Control path

Addresses below `0x80000000` are the Cluster aperture (mailbox, MSIP, IRQ
controller, Atomic Cache, MMU, PCU, Coherent window, URAM). `--dma-control`
selects how they are reached:

| Mode | Access | Default for |
| --- | --- | --- |
| `dma` | H2C/C2H nodes at `address + mmio_address_offset` | XDMA, QDMA simulator, QDMA fpga without the control BAR |
| `bar` | 32-bit MMIO on a PCI BAR at `control_offset + address` | QDMA fpga when the control BAR exists |

Without `--dma-control`, QDMA fpga checks for the control BAR's sysfs
`resource` file (or `--dma-control-resource`): a PDI that exposes BAR1 uses it,
a PDI without it (only BAR0/BAR2) reaches the aperture through the MM queue at
`address + mmio_address_offset` (`0x40_0000_0000` by default). The server
logs which path was chosen.

On the QDMA card (rvt-qdma-driver-interface.md) the MM queue only reaches
DDR, and BAR1 is the QDMA AXI-Lite master translated to AXI `0x10000000`,
with the aperture at AXI `0x11000000`. IP offset A is therefore BAR1 offset
`0x01000000 + A`. The backend maps 16 MiB of
`/sys/bus/pci/devices/<BDF>/resource1` from that offset (root required). The
path only guarantees aligned 32-bit accesses: partial words are read, merged
and written back.

Registers that need one 64-bit beat (device and IRQ mailboxes, Atomic Cache
STATUS/BACKING_BASE, PCU) are reached through BAR4 instead: the QDMA AXI
Bridge master maps the same aperture at AXI `0x10_0000_0000`, BAR4 offset A
being IP offset A, and carries single 64-bit MMIO accesses. Every single,
naturally aligned 8-byte access goes there as one 64-bit load or store;
everything else stays on BAR1. When `resource4` does not exist (an older PDI)
the backend warns and falls back to two 32-bit accesses on BAR1, low word
first; `--dma-wide-bar none` selects that explicitly. Addresses at or above the 16 MiB aperture but below 2 GiB return
`-EFAULT`. Runtime transfers to Coherent or URAM Buffers return `-ENOTSUP`,
since the DMA queue cannot reach them; Normal memory is unchanged.

| Option | Default |
| --- | --- |
| `--dma-control dma\|bar` | QDMA fpga: `bar` if the control BAR exists, else `dma`; `dma` otherwise |
| `--dma-pci-device BDF` | from the device name: `qdma%05x` is `bus << 12 \| device << 4 \| function`, so `qdma01000` is `0000:01:00.0` |
| `--dma-control-bar N` | 1 |
| `--dma-control-offset N` | `0x1000000` |
| `--dma-control-resource PATH` | none; maps this file instead of the sysfs BAR (tests) |
| `--dma-wide-bar N\|none` | 4 |
| `--dma-wide-offset N` | `0` |
| `--dma-wide-resource PATH` | none; maps this file instead of the sysfs wide BAR (tests) |

Profiles remain independent of driver type:

| Profile | Default directory | Low-address offset |
| --- | --- | --- |
| simulator | /tmp/virtual_npu | 0 |
| fpga | /dev | 0x4000000000 |

`--dma-device-dir` and `--dma-mmio-offset` override profile defaults regardless
of argument order. The default profile is simulator, default driver is XDMA,
and the new interface defaults to IRQ completion. A simulator must implement
the selected driver's node names and event ABI.

Without an explicit device name, XDMA retains the `--device-index` and
`--cluster-stride` mapping for multiple clusters. An explicit device name or
QDMA currently requires a single cluster, avoiding implicit reuse of the same
PCI function for different clusters. Driver-specific CLI options cannot be mixed.

Only `xilinx-dma` is registered as a backend. All HAL callers use
`npu_hal_xilinx_dma.h` and `npu_hal_xilinx_dma_options_init()`. The duplicate
XDMA configuration interface and version-1 options adapter have been removed.
Simulator and FPGA defaults are maintained only in `config.c`. Old
`--xdma-profile`, `--xdma-device-dir`, `--xdma-mmio-offset`, and `--device-prefix`
arguments are accepted. An invocation using only the legacy interface retains
the old optional-event behavior. Explicit `--dma-completion` always overrides it.
The short-lived independent `--backend qdma` / `--qdma-*` interface is replaced
by the unified options above.

The Runtime wire provider remains numeric 1, now named
`NPU_HAL_DATA_PROVIDER_XILINX_DMA`; the old macro is a source-compatible alias.
The fd order and token/export layout are unchanged. Existing Runtime clients
remain compatible. The implementation and I/O helpers reside in `xilinx_dma/`.
