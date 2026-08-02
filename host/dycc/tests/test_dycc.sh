#!/bin/sh
set -eu

if [ -z "${DYCC_BIN:-}" ]; then
    echo "test_dycc.sh: DYCC_BIN is required" >&2
    exit 2
fi

test_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT HUP INT TERM

"$DYCC_BIN" --save-temps="$work_dir" \
    --emit-host "$work_dir/basic.host.cpp" \
    --emit-device "$work_dir/basic.device.cpp" \
    -c "$test_dir/basic.cpp" -o "$work_dir/basic.o"

rg -q 'int vector_add\(' "$work_dir/basic.host.cpp"
rg -q 'npu_kernel_descriptor_create' "$work_dir/basic.host.cpp"
rg -q 'npu_kernel_arg_buffer' "$work_dir/basic.host.cpp"
rg -q 'npu_buffer_t \*output' "$work_dir/basic.host.cpp"
rg -q '__atomic_test_and_set' "$work_dir/basic.host.cpp"
rg -q '_ZN6sample10vector_addEPmmf' "$work_dir/basic.host.cpp"
rg -Fq '3U, 0U, UINT64_C(4096), UINT64_C(2048)' \
    "$work_dir/basic.host.cpp"
rg -Fq 'return sample::vector_add(output, value, scale, attrs);' \
    "$work_dir/basic.host.cpp"
if rg -q 'device_helper\(' "$work_dir/basic.host.cpp"; then
    echo "device-only function leaked into Host source" >&2
    exit 1
fi
rg -q '__kernel__ void vector_add\(' "$work_dir/basic.device.cpp"
if rg -q 'host_helper\(' "$work_dir/basic.device.cpp"; then
    echo "host-only function leaked into Device source" >&2
    exit 1
fi

llvm-readelf --file-header "$work_dir/basic.o" | rg -q 'Machine:.*X86-64'
llvm-readelf --symbols "$work_dir/basic.o" | \
    rg -q 'LOCAL +HIDDEN.*__kernel_obj__$'
llvm-readelf --file-header "$work_dir/basic.cpp.dycc.device.o" | \
    rg -q 'Machine:.*RISC-V'
llvm-nm --defined-only "$work_dir/basic.cpp.dycc.device.o" | \
    rg -q '_ZN6sample10vector_addEPmmf'
llvm-nm --defined-only "$work_dir/basic.cpp.dycc.device.o" | \
    rg -q ' vector_add__desc$'

"$DYCC_BIN" --save-temps="$work_dir/edge" \
    --emit-host "$work_dir/edge.host.cpp" \
    --emit-device "$work_dir/edge.device.cpp" \
    -c "$test_dir/edge.cpp" -o "$work_dir/edge.o"
rg -q 'int declared_kernel\(int value, const' "$work_dir/edge.host.cpp"
rg -q 'int declared_kernel\(int value, const' "$work_dir/edge.host.cpp"
rg -q '"_Z15declared_kerneli"' "$work_dir/edge.host.cpp"
rg -q '"c_kernel"' "$work_dir/edge.host.cpp"
if rg -q 'device_a = 1' "$work_dir/edge.host.cpp"; then
    echo "device variable group leaked into Host source" >&2
    exit 1
fi
if rg -q 'host_a = 3' "$work_dir/edge.device.cpp"; then
    echo "host variable group leaked into Device source" >&2
    exit 1
fi
llvm-nm --defined-only "$work_dir/edge/edge.cpp.dycc.device.o" | \
    rg -q ' c_kernel$'

"$DYCC_BIN" "$test_dir/executable_kernel.cpp" \
    "$test_dir/executable_main.cpp" -o "$work_dir/dycc-program"
test -x "$work_dir/dycc-program"
"$work_dir/dycc-program"
llvm-readelf --file-header "$work_dir/dycc-program" | \
    rg -q 'Machine:.*X86-64'
llvm-nm --defined-only "$work_dir/dycc-program" | \
    rg -q 'npu_kernel_descriptor_create'

mkdir "$work_dir/default-output"
(
    cd "$work_dir/default-output"
    "$DYCC_BIN" "$test_dir/executable_kernel.cpp" \
        "$test_dir/executable_main.cpp"
    test -x a.out
    ./a.out
)

if "$DYCC_BIN" --split-only --emit-host "$work_dir/bad.host.cpp" \
    "$test_dir/unannotated_global.cpp" >"$work_dir/bad.log" 2>&1; then
    echo "unannotated global variable was accepted" >&2
    exit 1
fi
rg -q 'global variable requires an explicit __host__ or __device__' \
    "$work_dir/bad.log"

if "$DYCC_BIN" --split-only --emit-host "$work_dir/invalid.host.cpp" \
    "$test_dir/invalid_kernel.cpp" >"$work_dir/invalid.log" 2>&1; then
    echo "invalid kernel signature was accepted" >&2
    exit 1
fi
rg -q '__kernel__ functions must return void' "$work_dir/invalid.log"

if "$DYCC_BIN" --split-only --emit-host "$work_dir/missing.host.cpp" \
    "$test_dir/missing_desc.cpp" >"$work_dir/missing.log" 2>&1; then
    echo "kernel without resource descriptor was accepted" >&2
    exit 1
fi
rg -q '__kernel__ requires DAYAN_KERNEL_DESC' "$work_dir/missing.log"

if "$DYCC_BIN" --split-only --emit-host "$work_dir/resource.host.cpp" \
    "$test_dir/invalid_desc.cpp" >"$work_dir/resource.log" 2>&1; then
    echo "misaligned URAM descriptor was accepted" >&2
    exit 1
fi
rg -q '1 KiB-aligned URAM size' "$work_dir/resource.log"
