# Xilinx DMA backend

One HAL backend and one Runtime data plane support both XDMA and QDMA.
`--dma-driver` selects device opening rules; all subsequent operations share
the same implementation and state. No XDMA symlinks are required for QDMA.

QDMA FPGA options (append socket, firmware, topology and service options):

```sh
--backend xilinx-dma --dma-driver qdma --dma-profile fpga \
--dma-device-name qdma01000 --dma-queue 0 --dma-user-irq 0 \
--dma-completion irq
```

XDMA FPGA options:

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
