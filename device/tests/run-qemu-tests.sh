#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
device_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
repo_root=$(CDPATH= cd -- "$device_root/.." && pwd)
build=${TEST_BUILD_DIR:-"$repo_root/build/device/tests/qemu"}
clang=${CLANG:-clang}
qemu=${QEMU_SYSTEM_RISCV64:-}
qemu_timeout=${QEMU_TIMEOUT:-20}

if [[ -z "$qemu" ]]; then
    if command -v qemu-system-riscv64 >/dev/null 2>&1; then
        qemu=$(command -v qemu-system-riscv64)
    elif [[ -x "$repo_root/../npu-qemu/build/qemu-system-riscv64" ]]; then
        qemu="$repo_root/../npu-qemu/build/qemu-system-riscv64"
    else
        echo "qemu-tests: error: qemu-system-riscv64 was not found" >&2
        echo "set QEMU_SYSTEM_RISCV64=/path/to/qemu-system-riscv64" >&2
        exit 2
    fi
fi
[[ -x "$qemu" ]] || {
    echo "qemu-tests: error: QEMU is not executable: $qemu" >&2
    exit 2
}
command -v "$clang" >/dev/null 2>&1 || {
    echo "qemu-tests: error: compiler was not found: $clang" >&2
    exit 2
}
command -v timeout >/dev/null 2>&1 || {
    echo "qemu-tests: error: GNU timeout is required" >&2
    exit 2
}

mkdir -p "$build"

target_flags=(
    --target=riscv64-unknown-elf
    -march=rv64imaf_zicsr_zifencei
    -mabi=lp64
    -mcmodel=medany
    -mno-relax
    -msmall-data-limit=0
)
cflags=(
    -O2 -g -std=c11
    -ffreestanding -fno-builtin -fno-common -fno-pic -fno-pie
    -fno-stack-protector -fno-unwind-tables -fno-asynchronous-unwind-tables
    -ffunction-sections -fdata-sections
    -Wall -Wextra -Werror
)
includes=(
    -I"$script_dir/qemu/include"
    -I"$device_root/include/api"
    -I"$device_root/include"
    -I"$device_root/lib/include"
)

compile_c()
{
    local output=$1
    local source=$2
    shift 2
    "$clang" "${target_flags[@]}" "${cflags[@]}" "$@" \
        "${includes[@]}" -c "$source" -o "$output"
}

compile_c "$build/runtime.o" "$script_dir/qemu/runtime.c"
"$clang" "${target_flags[@]}" -ffreestanding -c \
    "$script_dir/qemu/crt0.S" -o "$build/crt0.o"
compile_c "$build/string.o" "$device_root/lib/string.c"
compile_c "$build/rbtree.o" "$device_root/lib/rbtree.c"
compile_c "$build/math.o" "$device_root/lib/math.c"
compile_c "$build/mem.o" "$device_root/mem/mem.c" -I"$script_dir/include"
compile_c "$build/printf.o" "$device_root/lib/printf.c" \
    -I"$script_dir/printf_include"

compile_c "$build/list_test.o" "$script_dir/list_test.c"
compile_c "$build/rbtree_test.o" "$script_dir/rbtree_test.c"
compile_c "$build/mem_test.o" "$script_dir/mem_test.c" \
    -I"$script_dir/include"
compile_c "$build/math_test.o" "$script_dir/math_test.c"
compile_c "$build/rpc_test.o" "$script_dir/rpc_test.c"
compile_c "$build/printf_test.o" "$script_dir/printf_test.c" \
    -I"$script_dir/printf_include"

link_test()
{
    local name=$1
    shift
    "$clang" "${target_flags[@]}" -nostdlib -static -fuse-ld=lld \
        -Wl,-T,"$script_dir/qemu/linker.ld",--gc-sections,--build-id=none \
        -Wl,-Map,"$build/$name.map" \
        "$build/crt0.o" "$build/runtime.o" "$@" -o "$build/$name.elf"
}

link_test list_test "$build/list_test.o"
link_test rbtree_test "$build/rbtree_test.o" "$build/rbtree.o"
link_test mem_test "$build/mem_test.o" "$build/mem.o" \
    "$build/rbtree.o" "$build/string.o"
link_test math_test "$build/math_test.o" "$build/math.o"
link_test rpc_test "$build/rpc_test.o"
link_test printf_test "$build/printf_test.o" "$build/printf.o" \
    "$build/string.o"

tests=(list_test rbtree_test mem_test math_test rpc_test printf_test)
for test_name in "${tests[@]}"; do
    log="$build/$test_name.log"
    echo "qemu-tests: RUN $test_name"
    set +e
    timeout --foreground "${qemu_timeout}s" "$qemu" \
        -machine virt -cpu rv64 -smp 1 -m 128M \
        -bios none -kernel "$build/$test_name.elf" \
        -display none -serial stdio -monitor none -no-reboot \
        >"$log" 2>&1
    status=$?
    set -e
    if [[ $status -ne 0 ]] || ! grep -Fq 'dayan-qemu-test: PASS' "$log"; then
        cat "$log" >&2
        if [[ $status -eq 124 ]]; then
            echo "qemu-tests: FAIL $test_name (timeout)" >&2
        else
            echo "qemu-tests: FAIL $test_name (QEMU status $status)" >&2
        fi
        exit 1
    fi
    echo "qemu-tests: PASS $test_name"
done
