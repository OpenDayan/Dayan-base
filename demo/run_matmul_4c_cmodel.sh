#!/usr/bin/env bash
set -euo pipefail

demo_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH='' cd -- "$demo_dir/.." && pwd)
cmodel_dir=${RVTNG_CMODEL_DIR:-/home/wangzhe/rvtng-cmodel}
qemu=${RVTNG_QEMU:-/home/wangzhe/npu-qemu/build/dl-riscv64}
mount_dir=${NPU_DEVICE_PREFIX:-/tmp/virtual_npu}
socket_path=${DAYAN_SERVER_SOCKET:-/tmp/dayan-matmul-4c-demo-$$.sock}
build_dir="$repo_dir/build/demo"
cmodel_log="$build_dir/matmul_4c_cmodel.log"
server_log="$build_dir/matmul_4c_dayan_server.log"
demo_log="$build_dir/matmul_bf16_4c.log"
emulator_pid=
server_pid=

cleanup() {
    [ -z "$server_pid" ] || kill "$server_pid" 2>/dev/null || true
    [ -z "$emulator_pid" ] || kill "$emulator_pid" 2>/dev/null || true
    fusermount3 -u "$mount_dir" 2>/dev/null || \
        fusermount3 -uz "$mount_dir" 2>/dev/null || \
        fusermount -u "$mount_dir" 2>/dev/null || \
        fusermount -uz "$mount_dir" 2>/dev/null || true
    for pid in "$server_pid" "$emulator_pid"; do
        [ -n "$pid" ] || continue
        attempt=0
        while kill -0 "$pid" 2>/dev/null && [ "$attempt" -lt 20 ]; do
            attempt=$((attempt + 1))
            sleep 0.05
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
        wait "$pid" 2>/dev/null || true
    done
    rm -f "$socket_path" /tmp/card0_rvt_msg /tmp/card0_rvt_inst_req \
        /tmp/card0_rvt_inst_rsp /dev/shm/dram_shm_0 \
        /dev/shm/uram_shm_0 /dev/shm/uram_shm_1 \
        /dev/shm/uram_shm_2 /dev/shm/uram_shm_3 \
        /dev/shm/uram_shm_4 /dev/shm/uram_shm_5 \
        /dev/shm/uram_shm_6 /dev/shm/uram_shm_7
}
trap cleanup EXIT INT TERM

if [ "$#" -ne 0 ] && [ "$#" -ne 3 ]; then
    echo "usage: $0 [M N K]" >&2
    exit 2
fi
if [ ! -x "$cmodel_dir/bin/rvt_emulator" ]; then
    echo "missing cmodel: $cmodel_dir/bin/rvt_emulator" >&2
    exit 1
fi
if [ ! -x "$qemu" ]; then
    echo "missing QEMU: $qemu" >&2
    exit 1
fi
if pgrep -f "$cmodel_dir/bin/rvt_emulator" >/dev/null 2>&1 || \
   pgrep -f "$qemu.*rvtng-v0" >/dev/null 2>&1; then
    echo "an RVTNG cmodel instance is already running" >&2
    exit 1
fi

cleanup
mkdir -p "$build_dir" "$mount_dir"
make -C "$repo_dir" -j4 all
make -C "$demo_dir" all

"$cmodel_dir/bin/rvt_emulator" -c 4 \
    "$qemu" -monitor null -bios "$repo_dir/device/bl0/bootrom.elf" \
    -machine rvtng-v0 -nographic -serial none -m 4096 \
    >"$cmodel_log" 2>&1 &
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
        echo "cmodel exited during startup; see $cmodel_log" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 0.1
done
if [ "$ready" -ne 1 ]; then
    echo "cmodel device did not appear; see $cmodel_log" >&2
    exit 1
fi

"$repo_dir/build/host/driver/dayan_server" \
    --socket "$socket_path" \
    --firmware "$repo_dir/build/device/npu-os.elf" \
    --api-elf "$repo_dir/build/device/npu-os-api.elf" \
    --device-prefix "$mount_dir" \
    --core-mask 0xf \
    >"$server_log" 2>&1 &
server_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 600 ]; do
    if [ -S "$socket_path" ]; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "dayan_server exited during startup:" >&2
        sed -n '1,200p' "$server_log" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 0.1
done
if [ "$ready" -ne 1 ]; then
    echo "dayan_server socket did not appear:" >&2
    sed -n '1,200p' "$server_log" >&2
    exit 1
fi

demo_status=0
set +e
DAYAN_SERVER_SOCKET="$socket_path" \
    "$build_dir/matmul_bf16_4c" "$@" 2>&1 | tee "$demo_log"
demo_status=${PIPESTATUS[0]}
set -e
if ! grep -Fq '[host] PASS:' "$demo_log"; then
    echo "matmul validation did not report PASS" >&2
    demo_status=1
fi
if [ "$demo_status" -ne 0 ]; then
    echo "dayan_server log:" >&2
    sed -n '1,240p' "$server_log" >&2
fi
exit "$demo_status"
