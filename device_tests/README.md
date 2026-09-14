# Device tests

Create a directory ending in `_test`, following `device_tests/smoke_test/`.
Nested directories such as `device_tests/VEC/one_test/` are supported. Each test
is discovered recursively through its own `meta.json`; helper directories are not tests.
Put its C, C++, assembly and local header files directly in that directory.
Each directory produces one ELF containing all of its registered `DAYAN_TEST`
descriptors. The build automatically discovers `.c`, `.cpp`, `.cc` and `.S`
sources and links them with `build/device/libnpu-os.a`.

```text
device_tests/
  smoke_test/
    main.c
    meta.json
    README.md

build/device_tests/
  smoke_test/
    main.c.o
    main.c.d
    smoke_test.elf
    smoke_test.bin
    smoke_test.map
```

```sh
make device-tests                    # All *_test directories
make -C device_tests                 # Same build
make -C device_tests smoke_test      # One directory
make -C device_tests VEC/vec_check_int_add_test
make -C device_tests clean           # Remove build/device_tests only
```

`make -C device all` also builds every test. The device Makefile supplies the
same compiler, ABI, configuration and linker symbols to both the library and
the tests. `CLANG`, `CLANGXX`, `HARTS` and the other device build settings can be
passed on the make command line. Dependencies on included headers are tracked.

The ELF entry and the BIN load address are `0x80000000`. Default output uses
a fixed 4096-byte DRAM circular buffer, retaining the most recent bytes without
UART hardware. The sim program runners export it as `device_console.log` in
each backend's artifact directory. Every test must register at least one constant local
launch descriptor using `<dayan_test.h>`; there is no `main` entry or Host
launcher. See [the bare-metal API and memory layout](../device/BAREMETAL.md).

## Simulation metadata

Test input and output use the fixed 1 GiB range `[0x84000000, 0xc4000000)`:
`DAYAN_TEST_DATA_BASE = 0x84000000`, `DAYAN_TEST_DATA_SIZE = 0x40000000`.
The runtime neither allocates nor initializes this range. It is separate from
firmware RAM, heaps and per-test coherent `shared` storage.

Maintain `meta.json` in each test's source directory. It describes optional
`memory_init` regions and nonempty `compare` regions for simulation; each region
uses a byte address and byte size inside the reservation. Do not initialize or
dump the entire 1 GiB unless the test actually needs it. Program paths are
inferred from the test directory and are not metadata fields.

The build generates `build/device_tests/<relative-test-path>/<leaf-name>.bin`; it does not generate
or copy `meta.json`. See [smoke_test/README.md](smoke_test/README.md) for its exact
regions and the simulator completion protocol.

| Metadata field | Meaning |
| --- | --- |
| `schema_version` | Write `1` for the current format |
| `name` | Test label for logs and artifacts; the directory determines the binary name |
| `memory_init` | Optional array of regions to initialize before execution |
| `compare` | Nonempty array of regions to compare between QEMU and RTL |
| `mailbox` | Optional override of the sim driver's ALIVE/DONE protocol; requires matching platform code |
| `timeout` | Optional positive integers: `alive_cycles` (default 100000), `done_cycles` (100000000), `qemu_seconds` (120) |

Each region specifies `name`, `base` and `size` in bytes. Encode addresses as
hexadecimal strings without separators, such as `"0x84000000"`.
Initialization defaults to `pattern: "random"`; `seed: 0` or an omitted seed
uses the run seed shared by both backends. For zero initialization use
`pattern: "constant", value: 0`. Comparison defaults to `policy: "exact"`.

For device-test images, include the following mailbox mapping. The default
platform hooks publish ALIVE at startup and RESULT/DONE after the suite finishes;
RESULT must be zero for a passing run. The mailbox occupies a separate 64-byte
line immediately below the test data reservation and is excluded from the test heap.

```json
"mailbox": {
  "alive": { "addr": "0x83ffffc0", "value": "0xcafe0001" },
  "result": { "addr": "0x83ffffc8", "value": 0 },
  "done": { "addr": "0x83ffffd0", "value": "0xdead0001" }
}
```

See [the platform protocol](../device/BAREMETAL.md#simulation-completion-protocol)
for ordering, failure handling and the QEMU boot handshake.
