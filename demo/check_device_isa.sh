#!/usr/bin/env bash
set -eu

objdump=${OBJDUMP:-llvm-objdump}

if [ "$#" -eq 0 ]; then
    echo "device-isa-check: error: no Device object supplied" >&2
    exit 2
fi

for object in "$@"; do
    [ -f "$object" ] || {
        echo "device-isa-check: error: missing Device object: $object" >&2
        exit 2
    }
    dump=$($objdump --disassemble --no-show-raw-insn "$object")
    if grep -Eq '(^|[[:space:]])(lr\.[wd]|sc\.[wd])([.[:space:]]|$)' \
            <<<"$dump"; then
        echo "device-isa-check: error: unsupported LR/SC in $object" >&2
        grep -E '(^|[[:space:]])(lr\.[wd]|sc\.[wd])([.[:space:]]|$)' \
            <<<"$dump" >&2
        exit 1
    fi
done

echo "device-isa-check: PASS"
