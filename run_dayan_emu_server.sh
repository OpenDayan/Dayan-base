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

repo_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
emu_dir=${DAYAN_EMU_DIR:-}
if [ -z "$emu_dir" ] && [ -d "$repo_dir/../dayan-emu" ]; then
    emu_dir=$(CDPATH='' cd -- "$repo_dir/../dayan-emu" && pwd)
fi
qemu=${QEMU_DL_BIN:-"$emu_dir/deps/QEMU/build/dl-riscv64"}
mount_dir=${NPU_DEVICE_PREFIX:-/tmp/virtual_npu}
dayan_emu_harts=${DAYAN_EMU_HARTS:-4}
dayan_emu_memory_mib=${DAYAN_EMU_MEMORY_MIB:-4096}
socket_path=${DAYAN_SERVER_SOCKET:-/tmp/dayan-server.sock}
build_jobs=${DAYAN_BUILD_JOBS:-4}
build_dir=${DAYAN_EMU_BUILD_DIR:-"$repo_dir/build/dayan_emu"}
dayan_emu_log="$build_dir/dayan_emu.log"
emulator_pid=
server_pid=

cleanup_devices()
{
    fusermount3 -uz "$mount_dir" 2>/dev/null || \
        fusermount -uz "$mount_dir" 2>/dev/null || true
    rm -f /tmp/card0_rvt_msg /tmp/card0_rvt_inst_req \
        /tmp/card0_rvt_inst_rsp /dev/shm/dram_shm_0 \
        /dev/shm/uram_shm_0 /dev/shm/uram_shm_1 \
        /dev/shm/uram_shm_2 /dev/shm/uram_shm_3 \
        /dev/shm/uram_shm_4 /dev/shm/uram_shm_5 \
        /dev/shm/uram_shm_6 /dev/shm/uram_shm_7
}

cleanup()
{
    if [ -n "$server_pid" ]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [ -n "$emulator_pid" ]; then
        kill "$emulator_pid" 2>/dev/null || true
        wait "$emulator_pid" 2>/dev/null || true
    fi
    rm -f "$socket_path"
    cleanup_devices
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

case "$dayan_emu_harts" in
    1|2|3|4) ;;
    *) echo "DAYAN_EMU_HARTS must be between 1 and 4" >&2; exit 2 ;;
esac
case "$dayan_emu_memory_mib" in
    ''|*[!0-9]*|0) echo "DAYAN_EMU_MEMORY_MIB must be a positive integer" >&2; exit 2 ;;
esac
case "$build_jobs" in
    ''|*[!0-9]*|0) echo "DAYAN_BUILD_JOBS must be a positive integer" >&2; exit 2 ;;
esac
if [ -z "$emu_dir" ]; then
    echo "DAYAN_EMU_DIR must identify the simulator" >&2
    echo "set it explicitly, or place dayan-emu next to this repository" >&2
    exit 2
fi
if [ ! -x "$emu_dir/bin/rvt_emulator" ]; then
    echo "dayan_emu is not executable: $emu_dir/bin/rvt_emulator" >&2
    exit 2
fi
if [ ! -x "$qemu" ]; then
    echo "RVTNG QEMU is not executable: $qemu" >&2
    exit 2
fi
if pgrep -f "$emu_dir/bin/rvt_emulator" >/dev/null 2>&1 || \
   pgrep -f "$qemu.*rvtng-v0" >/dev/null 2>&1; then
    echo "a dayan_emu instance is already running" >&2
    exit 1
fi

mkdir -p "$build_dir" "$mount_dir"
make -C "$repo_dir" -j"$build_jobs" all
if [ ! -f "$repo_dir/device/bl0/bootrom.elf" ]; then
    if ! command -v riscv64-unknown-elf-gcc >/dev/null 2>&1; then
        echo "missing device/bl0/bootrom.elf and riscv64-unknown-elf-gcc" >&2
        exit 2
    fi
    make -C "$repo_dir/device/bl0" all
fi
cleanup_devices

"$emu_dir/bin/rvt_emulator" -c "$dayan_emu_harts" \
    "$qemu" -monitor null -bios "$repo_dir/device/bl0/bootrom.elf" \
    -machine rvtng-v0 -nographic -serial none -m "$dayan_emu_memory_mib" >"$dayan_emu_log" 2>&1 &
emulator_pid=$!

ready=0
for _ in $(seq 1 300); do
    if [ -r "$mount_dir/xdma0_c2h_0" ] && \
       [ -w "$mount_dir/xdma0_h2c_0" ] && \
       [ -r "$mount_dir/xdma0_user" ]; then
        ready=1
        break
    fi
    if ! kill -0 "$emulator_pid" 2>/dev/null; then
        echo "dayan_emu exited during startup; see $dayan_emu_log" >&2
        exit 1
    fi
    sleep 0.1
done
if [ "$ready" -ne 1 ]; then
    echo "dayan_emu device did not appear; see $dayan_emu_log" >&2
    exit 1
fi

DAYAN_SKIP_BUILD=1 \
NPU_DEVICE_PREFIX="$mount_dir" \
NPU_CORE_MASK="$(( (1 << dayan_emu_harts) - 1 ))" \
DAYAN_SERVER_SOCKET="$socket_path" \
    "$repo_dir/run_server.sh" \
        --normal-memory-size-mib "$dayan_emu_memory_mib" &
server_pid=$!

ready=0
for _ in $(seq 1 300); do
    if [ -S "$socket_path" ]; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid" || true
        echo "dayan_server exited during startup" >&2
        exit 1
    fi
    sleep 0.1
done
if [ "$ready" -ne 1 ]; then
    echo "dayan_server socket did not appear: $socket_path" >&2
    exit 1
fi

printf '%s\n' "dayan_emu server is ready: $socket_path"
printf '%s\n' "Device Normal memory: $dayan_emu_memory_mib MiB"
printf '%s\n' "Run clients with: DAYAN_SERVER_SOCKET=$socket_path <program>"
printf '%s\n' "Press Ctrl-C to stop; dayan_emu log: $dayan_emu_log"

while kill -0 "$server_pid" 2>/dev/null &&
      kill -0 "$emulator_pid" 2>/dev/null; do
    sleep 0.2
done
if ! kill -0 "$emulator_pid" 2>/dev/null; then
    echo "dayan_emu exited unexpectedly; see $dayan_emu_log" >&2
    exit 1
fi
set +e
wait "$server_pid"
status=$?
set -e
server_pid=
exit "$status"
