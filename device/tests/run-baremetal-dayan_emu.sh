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

set -euo pipefail

device_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
repo_root=$(CDPATH= cd -- "$device_root/.." && pwd)
build=${BUILD_DIR:-"$repo_root/build/device"}
emu_dir=${DAYAN_EMU_DIR:-}
if [ -z "$emu_dir" ] && [ -d "$repo_root/../dayan-emu" ]; then
    emu_dir=$(CDPATH='' cd -- "$repo_root/../dayan-emu" && pwd)
fi
if [ -z "$emu_dir" ]; then
    echo "DAYAN_EMU_DIR must identify the simulator" >&2
    echo "set it explicitly, or place dayan-emu next to this repository" >&2
    exit 2
fi
qemu=${QEMU_DL_BIN:-"$emu_dir/deps/QEMU/build/dl-riscv64"}
emulator="$emu_dir/bin/dayan-emu"
mount_dir=/tmp/virtual_npu
emulator_pid=
resources=(/tmp/card0_dayan_inst_req /tmp/card0_dayan_inst_rsp /tmp/card0_dayan_msg
           /tmp/card0_dayan_mmu /dev/shm/dram_shm_0
           /dev/shm/uram_shm_0 /dev/shm/uram_shm_1
           /dev/shm/uram_shm_2 /dev/shm/uram_shm_3)

cleanup()
{
    if [[ -z "$emulator_pid" ]]; then
        return
    fi
    kill -TERM -- "-$emulator_pid" 2>/dev/null || true
    wait "$emulator_pid" 2>/dev/null || true
    emulator_pid=
    if findmnt -rn -M "$mount_dir" >/dev/null; then
        fusermount3 -u "$mount_dir"
    fi
    for path in "${resources[@]}"; do
        if [[ -e "$path" || -S "$path" ]]; then
            rm -- "$path"
        fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -x "$emulator" && -x "$qemu" ]]
if pgrep -x dayan-emu >/dev/null || pgrep -x dl-riscv64 >/dev/null ||
   pgrep -x TestCluster >/dev/null || findmnt -rn -M "$mount_dir" >/dev/null; then
    echo 'baremetal-dayan_emu: simulator resources are already in use' >&2
    exit 1
fi
for path in "${resources[@]}"; do
    if [[ -e "$path" || -S "$path" ]]; then
        echo "baremetal-dayan_emu: existing simulator resource: $path" >&2
        exit 1
    fi
done
mkdir -p "$mount_dir"

for test in baremetal_dayan_emu baremetal_failure; do
    elf="$build/tests/$test.elf"
    log="$build/tests/$test.log"
    [[ -f "$elf" ]]
    # The loader sets every PC directly, bypassing the Host/BL0 boot exchange.
    setsid "$emulator" -c 4 "$qemu" -machine dayan-v0 \
        -bios "$device_root/bl0/bootrom.elf" \
        -device "loader,file=$elf,cpu-num=0" \
        -device loader,addr=0x80000000,cpu-num=1 \
        -device loader,addr=0x80000000,cpu-num=2 \
        -device loader,addr=0x80000000,cpu-num=3 \
        -display none -serial none -monitor none -m 4096 \
        -semihosting-config enable=on,target=native >"$log" 2>&1 &
    emulator_pid=$!
    completed=0
    for ((attempt = 0; attempt < 300; ++attempt)); do
        if grep -Fq '[test] suite:' "$log"; then
            completed=1
            break
        fi
        if ! kill -0 "$emulator_pid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if [[ $completed != 1 ]]; then
        tail -n 60 "$log" >&2
        echo "baremetal-dayan_emu: FAIL $test (no completion)" >&2
        exit 1
    fi
    if [[ $test == baremetal_dayan_emu ]]; then
        grep -Fq '[test] first: PASS (0)' "$log"
        grep -Fq '[test] second: PASS (0)' "$log"
        grep -Fq '[test] suite: PASS (0)' "$log"
    else
        grep -Fq '[test] trap:' "$log"
        grep -Fq '[test] fault: FAIL (-5)' "$log"
        grep -Fq '[test] after-fault: PASS (0)' "$log"
        grep -Fq '[test] suite: FAIL (-5)' "$log"
    fi
    cleanup
    echo "baremetal-dayan_emu: PASS $test"
done
