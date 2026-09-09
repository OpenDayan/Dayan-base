# Smoke test metadata

This test follows the [device test build](../README.md) and
[bare-metal memory layout](../../device/BAREMETAL.md). Its two local launch
descriptors run sequentially: `first`, then `second`.

## Files

- `main.c`: test logic and `DAYAN_TEST` descriptors.
- `meta.json`: simulator memory initialization and comparison regions,
  maintained alongside the source.
- `build/device_tests/smoke_test/smoke_test.bin`: generated binary, with the path
  relative to the repository root.

Build from the repository root with `make device-tests`, or build only this
test with `make -C device_tests smoke_test`. The simulator receives the source
directory as `TEST_DIR` and locates the binary by directory name. No program
path belongs in `meta.json`.

## Test data

The runtime reserves `[0x84000000, 0xc4000000)`, exactly `0x40000000` bytes
(1 GiB), for test input and output. It neither allocates nor initializes this
area. `meta.json` initializes only the three compared regions, 516 bytes in
total, to zero before the program starts.

| Address | Bytes | Contents after execution | Comparison |
| --- | --- | --- | --- |
| `0x84000000` | 256 | 64 little-endian `uint32_t` values, `input[i] = i * i` | Exact |
| `0x84001000` | 256 | 64 little-endian `uint32_t` values, `output[i] = i * i * 3 + 7` | Exact |
| `0x84002000` | 8 | Address returned by the first allocation | Omitted |
| `0xc3fffffc` | 4 | `0xfeedface`, bytes `ce fa ed fe` | Exact |

`first` writes its own input values; it does not consume external random input.
Its child computes the output. `second` checks the output, allocation progress,
fresh coherent shared memory and the reservation's final word. The saved
allocation address is internal bookkeeping, not a portable numerical result.
The test's coherent `shared` storage is separate from the reserved data area.

`exact` compares QEMU and RTL bytes. It does not independently evaluate the
formulas above. The device test also checks its results internally.

## Completion protocol

The launcher calls `platform_test_complete(result)` only after both descriptors
and their child threads have finished and their output has drained. The default
implementation in `device/platform/test.c` publishes the aggregate result and
DONE. `platform_test_init` clears stale state and publishes ALIVE at startup.
The metadata selects these 64-bit words:

| Address | Word | Expected value |
| --- | --- | --- |
| `0x83ffffc0` | ALIVE | `0xcafe0001` |
| `0x83ffffc8` | RESULT | `0` |
| `0x83ffffd0` | DONE | `0xdead0001` |

The linker excludes the mailbox cache line from the test heap, leaving the
1 GiB data reservation intact. The second descriptor flushes all three compared
regions before returning; the platform hooks fence and flush the status words.
Both QEMU and RTL check RESULT after observing DONE. A nonzero aggregate result
fails the run, even when the two backends produced identical output.

`meta.json` configures `timeout.alive_cycles: 100000` and
`timeout.done_cycles: 100000000` for RTL. These are independent clock-cycle
budgets for the two waits (5 ms and 5 s with the current 50 ns clock).
`timeout.qemu_seconds: 120` sets a separate wall-clock budget for each QEMU
startup, ALIVE and DONE stage. Increase the DONE budget for longer tests;
the QEMU runner also accepts `--timeout-s` as an override.

The boundary sentinel is written by `first`, before `second` checks the results;
it must not be used as a suite completion flag. Printed output goes to the fixed
4096-byte DRAM circular buffer described in `BAREMETAL.md`. QEMU and RTL program
runners export the retained text as `device_console.log` in their artifact
directories; no UART model or additional JSON configuration is required.
