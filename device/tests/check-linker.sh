#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
device_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
build=${TEST_BUILD_DIR:-"$device_root/../build/device/tests/linker"}
clang=${CLANG:-clang}
mkdir -p "$build"

flags=(--target=riscv64-unknown-elf -march=rv64imaf_zicsr_zifencei
       -mabi=lp64 -mcmodel=medany -ffreestanding -fno-pic)
link=("$clang" "${flags[@]}" -nostdlib -static -fuse-ld=lld
      -Wl,-T,"$device_root/linker.ld",--build-id=none
      "$build/global.o" -o "$build/global.elf")

# Read-only globals remain legal; mutable globals must fail even in old pools
# or a section whose name misleadingly looks like read-only data or code.
"$clang" "${flags[@]}" -c "$script_dir/linker_global.c" -o "$build/global.o"
"${link[@]}"
for section in .data .bss .cpu .proc_slots .profile_state .trap_records \
               .tensor_interrupt_records .proc_stack .proc_printf .coherent \
               .custom_state .rodata.mutable .text.mutable .npu_test; do
    "$clang" "${flags[@]}" -DTEST_QUALIFIER= "-DTEST_SECTION=\"$section\"" \
        -c "$script_dir/linker_global.c" -o "$build/global.o"
    if "${link[@]}" >"$build/rejected.log" 2>&1; then
        echo "linker-check: FAIL: accepted writable $section" >&2
        exit 1
    fi
    grep -Fq 'keep state in struct cpu' "$build/rejected.log"
done

# A read-only coherent declaration is forbidden too: Host owns this window.
"$clang" "${flags[@]}" '-DTEST_SECTION=".coherent"' \
    -c "$script_dir/linker_global.c" -o "$build/global.o"
if "${link[@]}" >"$build/rejected.log" 2>&1; then
    echo 'linker-check: FAIL: accepted firmware coherent storage' >&2
    exit 1
fi
grep -Fq 'coherent storage is allocated by Host' "$build/rejected.log"

"$clang" "${flags[@]}" '-DTEST_SECTION=".npu_test"' \
    -c "$script_dir/linker_global.c" -o "$build/global.o"
if "${link[@]}" >"$build/rejected.log" 2>&1; then
    echo 'linker-check: FAIL: accepted a truncated test descriptor' >&2
    exit 1
fi
grep -Fq 'test descriptor table must contain 128-byte records' "$build/rejected.log"

"$clang" "${flags[@]}" -c "$script_dir/linker_global.c" -o "$build/global.o"
if "${link[@]}" -Wl,--defsym=CONFIG_REQUIRE_TESTS=1 >"$build/rejected.log" 2>&1; then
    echo 'linker-check: FAIL: accepted a test image without descriptors' >&2
    exit 1
fi
grep -Fq 'device test must register at least one DAYAN_TEST descriptor' "$build/rejected.log"
for pool in STACK PROC_STACK PROC_PRINTF RPC_STACK PROFILE_STACK TRAP_STACK HEAP; do
    if "${link[@]}" "-Wl,--defsym=CONFIG_${pool}_SIZE=0x1010" \
            >"$build/rejected.log" 2>&1; then
        echo "linker-check: FAIL: accepted unaligned $pool stride" >&2
        exit 1
    fi
    grep -Fq 'cover whole cache lines' "$build/rejected.log"
done
if "${link[@]}" -Wl,--defsym=CONFIG_CPU_LOCAL_SIZE=0x1800 \
        >"$build/rejected.log" 2>&1; then
    echo 'linker-check: FAIL: accepted non-power-of-two CPU stride' >&2
    exit 1
fi
grep -Fq 'CPU_LOCAL_SIZE must be a power of two' "$build/rejected.log"
echo 'linker-check: PASS'
