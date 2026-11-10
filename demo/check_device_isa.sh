#!/usr/bin/env bash
#***************************************************************************************
# Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
#
# Dayan-base is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

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
