# Device tests

Create a directory ending in `_test`, following `device_tests/smoke_test/`.
Put its C, C++, assembly and local header files directly in that directory.
Each directory produces one ELF containing all of its registered `DAYAN_TEST`
descriptors. The build automatically discovers `.c`, `.cpp`, `.cc` and `.S`
sources and links them with `build/device/libnpu-os.a`.

```text
device_tests/
  smoke_test/
    main.c

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
make -C device_tests clean           # Remove build/device_tests only
```

`make -C device all` also builds every test. The device Makefile supplies the
same compiler, ABI, configuration and linker symbols to both the library and
the tests. `CLANG`, `CLANGXX`, `HARTS` and the other device build settings can be
passed on the make command line. Dependencies on included headers are tracked.

The ELF entry and the BIN load address are `0x80000000`. Default output uses
the Cluster AXI UARTLite. Every test must register at least one constant local
launch descriptor using `<dayan_test.h>`; there is no `main` entry or Host
launcher. See [the bare-metal API and memory layout](../device/BAREMETAL.md).
