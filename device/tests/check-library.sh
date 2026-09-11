#!/usr/bin/env bash
set -euo pipefail

device_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build=${BUILD_DIR:-"$device_root/../build/device"}
ar=${LLVM_AR:-llvm-ar}
nm=${NM:-llvm-nm}
readelf=${READELF:-llvm-readelf}

for source in "$device_root"/{boot,lib,mem,platform,rpc,sche}/*.c \
              "$device_root"/{bl1,platform,sche}/*.S; do
    name=${source##*/}
    printf '%s.o\n' "${name%.*}"
done | LC_ALL=C sort > "$build/tests/library.expected"
"$ar" t "$build/libnpu-os.a" | LC_ALL=C sort > "$build/tests/library.actual"
diff -u "$build/tests/library.expected" "$build/tests/library.actual"

symbol()
{
    "$nm" --defined-only --format=posix "$1" | \
        awk -v name="$2" '$1 == name { print "0x" $3 }'
}

elf=${SMOKE_ELF:-"$device_root/../build/device_tests/smoke_test/smoke_test.elf"}
[[ -z $("$nm" --undefined-only "$elf") ]]
[[ $(symbol "$elf" _start) == 0x80000000 ]]
[[ $(( $(symbol "$elf" __npu_test_end) - $(symbol "$elf" __npu_test_start) )) == 640 ]]
[[ $(( $(symbol "$elf" __test_data_start) )) == $((0x84000000)) ]]
[[ $(( $(symbol "$elf" __test_data_end) )) == $((0xc4000000)) ]]
[[ $(( $(symbol "$elf" __test_mailbox_start) )) == $((0x83ffffc0)) ]]
[[ $(( $(symbol "$elf" __test_mailbox_end) )) == $((0x84000000)) ]]
[[ $(( $(symbol "$elf" __test_heap_start) )) -lt $((0x84000000)) ]]
[[ $(( $(symbol "$elf" __test_log_start) )) == $((0x83ffef80)) ]]
[[ $(( $(symbol "$elf" __test_log_end) )) == $((0x83ffffc0)) ]]
[[ $(( $(symbol "$elf" __test_heap_end) )) -le $((0x83ffef80)) ]]
[[ $(symbol "$build/npu-os.elf" __npu_test_start) == \
   $(symbol "$build/npu-os.elf" __npu_test_end) ]]
"$readelf" -SW "$build/tests/baremetal_api.o" | \
    grep -Eq '\.npu_test[[:space:]]+PROGBITS.*000140.* A '
if "$readelf" -SW "$build/tests/baremetal_api.o" | grep -Fq '.init_array'; then
    echo 'library-check: C++ descriptor requires a constructor' >&2
    exit 1
fi
echo 'library-check: PASS'
