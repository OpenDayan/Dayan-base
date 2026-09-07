# Bare-metal operator tests

`make -C device all` builds `build/device/libnpu-os.a` from exactly the object
list used for `npu-os.elf`. Build only the archive with:

```sh
make -C device libnpu-os.a
```

The archive includes BL1, the scheduler, Device API, libc, platform code and both
RPC transports. Link it with `device/linker.ld`; `_start` is extracted from the
archive and placed at `0x80000000`. The linker retains `.npu_test` even with
`--gc-sections`. With no descriptors the image uses the normal Host RPC path.

## Register a test

Include `<dayan_test.h>` from `device/include/api`. For example:

```c
#include <dayan_test.h>
#include <stdio.h>

static void kernel(uintptr_t output)
{
    *(uint32_t *)output = 42;
    printf("result=%u\n", *(uint32_t *)output);
}

DAYAN_TEST(example, {
    .entry = (void (*)(void))kernel,
    .args = {DAYAN_TEST_DATA_BASE},
    .argc = 1,
    .name = "example",
});
```

Descriptors are constant 128-byte records. `resources` uses the existing
`dayan_kernel_resource_desc_t`: declare the root family's child Threads, URAM,
TCSR groups, named resources and coherent `shared` bytes there. Unspecified
resources are zero. Up to eight machine-word arguments are loaded into a0-a7.
Names must contain 1-15 bytes followed by NUL. C++17 can use ordinary aggregate
initializers; dynamic initialization and writable globals remain unsupported.

This version executes one root Proc at a time on Hart 0, in descriptor link
order. Each root receives task mode 1, task ID `{0,0,0}` and dimensions `{1,1,1}`.
`max_tasks_per_core` does not affect this sequential launcher. Child Threads use
the existing create/join APIs. Other configured Harts enter their local system
workers without launching tests. Four-Core launches and task-grid expansion are
not part of this local descriptor ABI.

The launcher waits for the whole family to exit and its output to drain before
starting the next record. Returning from a kernel means success;
`dayan_thread_exit(result)` supplies an explicit result. Invalid descriptors and
failed tests are reported and the remaining records still run. Traps and tensor
interrupt reports fail the current test. A test that never exits still requires
an external timeout. The suite prints its aggregate status, calls the weak
`platform_test_complete(result)` hook and leaves the workers waiting.

## Build and boot

Build the complete example with `make -C device_tests smoke_test`. The result is
`build/device_tests/smoke_test/smoke_test.elf` and `.bin`; it checks argument passing, child
Threads, allocation, reserved data and sequential completion.

Each `device_tests/*_test/` directory is discovered automatically. Its `.c`,
`.cpp`, `.cc` and `.S` files are compiled and linked together with `libnpu-os.a`;
all objects, dependency files, ELF, BIN and map go under
`build/device_tests/<directory>/`. Run `make device-tests` or
`make -C device_tests` to build every test directory. `make -C device all`
also builds these tests. See `device_tests/README.md` for the directory layout.

For an external object, use the same target ABI and configuration as the
archive. For a default build, run from the repository root:

```sh
clang --target=riscv64-unknown-elf -march=rv64imaf_zicsr_zifencei_zacas \
  -mabi=lp64 -mcmodel=medany -mno-relax -msmall-data-limit=0 \
  -ffreestanding -fno-builtin -fno-pic -fno-stack-protector \
  -Idevice/include/api -c example.c -o example.o
clang --target=riscv64-unknown-elf -march=rv64imaf_zicsr_zifencei_zacas \
  -mabi=lp64 -mcmodel=medany -mno-relax -nostdlib -static -fuse-ld=lld \
  -Wl,-T,device/linker.ld,--gc-sections \
  example.o build/device/libnpu-os.a -o example.elf
```

Supply descriptor objects directly. If they are themselves in an archive, use
`--whole-archive` around that test archive so unreferenced registrations are
extracted. For a nondefault firmware configuration, pass the same linker
`CONFIG_*` values as the device Makefile. C++ code must also disable exceptions
and RTTI, and must not require a C++ standard library. Tensor kernels need the
same compiler TX feature flags used by the existing Device build.

Load the ELF segments, cold-reset the Cluster, then start the participating
Harts in machine mode at `0x80000000`. BL1 reads `mhartid` itself; no initial a0,
Host process, RPC mailbox exchange or operating system is required. The board
must provide its reset identity mapping for DRAM. Global hardware resource state
comes from cold reset, while the firmware initializes each Hart's local state.

## Memory and platform

| Range | Owner |
| --- | --- |
| `0x80000000` through `__ram_used_end` | Image, CPU-local state, stacks, per-Hart heaps and test coherent backing |
| `__test_heap_start` through `__test_heap_end` | Local RPC bump allocator, 64-byte granularity |
| `[0x84000000, 0xc4000000)` | Fixed 1 GiB test input/output area, never allocated or initialized by the runtime |

The RPC heap uses the remaining configured firmware RAM below `0x84000000`.
Overflow or exhaustion returns `-ENOMEM`; release acknowledges without reusing
the backing range. Existing `malloc/free` still manage blocks inside each heap
region, and request additional regions through this local RPC allocator.
Allocation state does not reset between tests. The 1 GiB data reservation is
represented by linker symbols, so it does not add a 1 GiB ELF load segment.

The test image reserves 4 MiB below the bump heap for Atomic Cache backing and
programs its base before launch. Each test's declared coherent `shared` region
is zeroed before execution; it is separate from the fixed input/output area.
URAM and TCSR resources start at physical group zero and are reused after the
preceding test finishes.

The default weak `platform_print(data, size)` writes 32-bit AXI UARTLite TX FIFO
accesses at `0x40600004`, polling TX-full at `0x40600008`. A board object linked
before the archive can override `platform_print`, `platform_test_init` and
`platform_test_complete`; these hooks must not call RPC-backed `printf`.

## Validation

```sh
make -C device check
make -C device check-baremetal-cmodel
```

The first command checks archive membership, linker retention, C/C++ descriptor
ABI, allocator boundaries, RPC services and the existing firmware tests. The
second uses `~/rvtng-cmodel` and its QEMU build, with no Host server, and sets all
four PCs directly to `0x80000000`. It checks both successful execution and an
illegal-instruction fault followed by another test. Override `RVTNG_CMODEL_DIR`
or `RVTNG_QEMU` to select compatible simulator builds. Its temporary processes,
mount and IPC resources are removed after each case.

QEMU's `rvtng-v0` has no UART model. Only the cmodel test image links a
semihosting implementation of `platform_print`; the production archive and the
`baremetal-smoke` image retain AXI UARTLite output. Cmodel verifies functional
behavior, not UART electrical behavior or RTL timing.
