#!/bin/sh
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

runtime_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH='' cd -- "$runtime_dir/../.." && pwd)
emu_dir=${DAYAN_EMU_DIR:-}
if [ -z "$emu_dir" ] && [ -d "$repo_dir/../dayan-emu" ]; then
    emu_dir=$(CDPATH='' cd -- "$repo_dir/../dayan-emu" && pwd)
fi
if [ -z "$emu_dir" ]; then
    echo "DAYAN_EMU_DIR must identify the simulator" >&2
    echo "set it explicitly, or place dayan-emu next to this repository" >&2
    exit 2
fi
qemu=${QEMU_DL_BIN:-"$repo_dir/deps/QEMU/build/dl-riscv64"}
mount_dir=${NPU_DEVICE_PREFIX:-/tmp/virtual_npu}
dayan_emu_harts=${DAYAN_EMU_HARTS:-4}
log_file=${DAYAN_EMU_LOG:-"$repo_dir/build/host/driver/dayan_emu.log"}
emulator_pid=

cleanup() {
    if [ -n "$emulator_pid" ]; then
        kill "$emulator_pid" 2>/dev/null || true
        wait "$emulator_pid" 2>/dev/null || true
    fi
    fusermount3 -uz "$mount_dir" 2>/dev/null || \
        fusermount -uz "$mount_dir" 2>/dev/null || true
    rm -f /tmp/card0_rvt_msg /tmp/card0_rvt_inst_req \
        /tmp/card0_rvt_inst_rsp /dev/shm/dram_shm_0 \
        /dev/shm/uram_shm_0 /dev/shm/uram_shm_1 \
        /dev/shm/uram_shm_2 /dev/shm/uram_shm_3 \
        /dev/shm/uram_shm_4 /dev/shm/uram_shm_5 \
        /dev/shm/uram_shm_6 /dev/shm/uram_shm_7
}
trap cleanup EXIT INT TERM

if pgrep -f "$emu_dir/bin/dayan-emu" >/dev/null 2>&1 || \
   pgrep -f "$qemu.*rvtng-v0" >/dev/null 2>&1; then
    echo "a dayan_emu instance is already running" >&2
    exit 1
fi
case "$dayan_emu_harts" in
    1|2|3|4) ;;
    *) echo "DAYAN_EMU_HARTS must be between 1 and 4" >&2; exit 1 ;;
esac
cleanup
mkdir -p "$repo_dir/build/host/driver" "$mount_dir"
make -C "$repo_dir" -j4 all
make -C "$runtime_dir" dayan_emu-smoke-bin

"$emu_dir/bin/dayan-emu" -c "$dayan_emu_harts" \
    "$qemu" -monitor null -bios "$repo_dir/device/bl0/bootrom.elf" \
    -machine rvtng-v0 -nographic -serial none -m 4096 >"$log_file" 2>&1 &
emulator_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 300 ]; do
    if [ -r "$mount_dir/xdma0_c2h_0" ] && \
       [ -w "$mount_dir/xdma0_h2c_0" ] && \
       [ -r "$mount_dir/xdma0_user" ]; then
        ready=1
        break
    fi
    if ! kill -0 "$emulator_pid" 2>/dev/null; then
        echo "dayan_emu exited during startup; see $log_file" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 0.1
done
if [ "$ready" -ne 1 ]; then
    echo "dayan_emu device did not appear; see $log_file" >&2
    exit 1
fi

NPU_DEVICE_PREFIX="$mount_dir" \
    "$repo_dir/build/host/driver/tests/dayan_emu_smoke" \
    "$repo_dir/build/device/npu-os.elf" \
    "$repo_dir/build/device/npu-os-api.elf" \
    "$repo_dir/build/host/driver/tests/dayan_emu_kernel.o" \
    "$dayan_emu_harts"
