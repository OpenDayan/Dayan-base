#!/usr/bin/env bash
set -euo pipefail

demo_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH='' cd -- "$demo_dir/.." && pwd)
cmodel_dir=${RVTNG_CMODEL_DIR:-/home/wangzhe/rvtng-cmodel}
qemu=${RVTNG_QEMU:-"$repo_dir/deps/npu-qemu/build/dl-riscv64"}
mount_dir=${NPU_DEVICE_PREFIX:-/tmp/virtual_npu}
cmodel_harts=${NPU_CMODEL_HARTS:-4}
socket_path=${DAYAN_SERVER_SOCKET:-/tmp/dayan-print-demo-$$.sock}
build_dir="$repo_dir/build/demo"
cmodel_log="$build_dir/cmodel.log"
server_log="$build_dir/dayan_server.log"
demo_log="$build_dir/print_demo.log"
context_log="$build_dir/context_thread_probe.log"
interrupt_log="$build_dir/tensor_interrupt_probe.log"
emulator_pid=
server_pid=

cleanup() {
    if [ -n "$server_pid" ]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [ -n "$emulator_pid" ]; then
        kill "$emulator_pid" 2>/dev/null || true
        wait "$emulator_pid" 2>/dev/null || true
    fi
    fusermount3 -uz "$mount_dir" 2>/dev/null || \
        fusermount -uz "$mount_dir" 2>/dev/null || true
    rm -f "$socket_path" /tmp/card0_rvt_msg /tmp/card0_rvt_inst_req \
        /tmp/card0_rvt_inst_rsp /dev/shm/dram_shm_0 \
        /dev/shm/uram_shm_0 /dev/shm/uram_shm_1 \
        /dev/shm/uram_shm_2 /dev/shm/uram_shm_3 \
        /dev/shm/uram_shm_4 /dev/shm/uram_shm_5 \
        /dev/shm/uram_shm_6 /dev/shm/uram_shm_7
}
trap cleanup EXIT INT TERM

case "$cmodel_harts" in
    1|2|3|4) ;;
    *) echo "NPU_CMODEL_HARTS must be between 1 and 4" >&2; exit 2 ;;
esac
if [ "$#" -ne 0 ] && [ "$#" -ne 7 ]; then
    echo "usage: $0 [ARG0 ARG1 ARG2 ARG3 ARG4 ARG5 ARG6]" >&2
    exit 2
fi
case "${DAYAN_DEMO_TEST:-all}" in
    all|print|context|interrupt) ;;
    *) echo "DAYAN_DEMO_TEST must be all, print, context, or interrupt" >&2; exit 2 ;;
esac
if pgrep -f "$cmodel_dir/bin/rvt_emulator" >/dev/null 2>&1 || \
   pgrep -f "$qemu.*rvtng-v0" >/dev/null 2>&1; then
    echo "an RVTNG cmodel instance is already running" >&2
    exit 1
fi

cleanup
mkdir -p "$build_dir" "$mount_dir"
make -C "$repo_dir" -j4 all
make -C "$demo_dir" all

"$cmodel_dir/bin/rvt_emulator" -c "$cmodel_harts" \
    "$qemu" -monitor null -bios "$repo_dir/device/bl0/bootrom.elf" \
    -machine rvtng-v0 -nographic -serial none -m 4096 >"$cmodel_log" 2>&1 &
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
    --core-mask "$(( (1 << cmodel_harts) - 1 ))" >"$server_log" 2>&1 &
server_pid=$!

ready=0
attempt=0
while [ "$attempt" -lt 300 ]; do
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
if [ "${DAYAN_DEMO_TEST:-all}" = all ] ||
   [ "${DAYAN_DEMO_TEST:-all}" = print ]; then
    set +e
    DAYAN_SERVER_SOCKET="$socket_path" \
        "$build_dir/print_demo" "$@" 2>&1 | tee "$demo_log"
    demo_status=${PIPESTATUS[0]}
    set -e
    for grid_id in '(0,0,0)' '(1,0,0)' '(0,1,0)' '(1,1,0)'; do
        if ! grep -Fq "[device] grid_id=$grid_id grid_dim=(4,4,4) mode=1 " \
                "$demo_log"; then
            echo "missing Device output for grid_id=$grid_id" >&2
            demo_status=1
        fi
    done
    if [ "$demo_status" -ne 0 ]; then
        echo "dayan_server log:" >&2
        sed -n '1,200p' "$server_log" >&2
    fi
fi

context_status=0
if [ "${DAYAN_DEMO_TEST:-all}" = all ] ||
   [ "${DAYAN_DEMO_TEST:-all}" = context ]; then
    set +e
    DAYAN_SERVER_SOCKET="$socket_path" \
        "$build_dir/context_thread_probe" 2>&1 | tee "$context_log"
    context_status=${PIPESTATUS[0]}
    set -e
    if ! grep -Fq \
            '[host] PASS: Thread Contexts isolate logical TCSR 0 and inherit URAM' \
            "$context_log"; then
        context_status=1
    fi
    if [ "$context_status" -ne 0 ]; then
        echo "dayan_server log:" >&2
        sed -n '1,240p' "$server_log" >&2
    fi
fi

interrupt_status=0
if [ "${DAYAN_DEMO_TEST:-all}" = all ] ||
   [ "${DAYAN_DEMO_TEST:-all}" = interrupt ]; then
    set +e
    DAYAN_SERVER_SOCKET="$socket_path" \
        "$build_dir/tensor_interrupt_probe" 2>&1 | tee "$interrupt_log"
    interrupt_status=${PIPESTATUS[0]}
    set -e
    if ! grep -Fq \
            '[host] PASS: tensor interrupt resumes the faulting Proc' \
            "$interrupt_log" ||
       ! grep -Fq 'source=12(neodma_tcsr_access_error)' "$server_log"; then
        interrupt_status=1
    fi
    if [ "$interrupt_status" -ne 0 ]; then
        echo "dayan_server log:" >&2
        sed -n '1,280p' "$server_log" >&2
    fi
fi
if [ "$demo_status" -ne 0 ]; then
    exit "$demo_status"
fi
if [ "$context_status" -ne 0 ]; then
    exit "$context_status"
fi
exit "$interrupt_status"
