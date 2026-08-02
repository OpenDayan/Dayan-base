#!/bin/sh
set -eu

runtime_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH='' cd -- "$runtime_dir/../.." && pwd)
cmodel_dir=${RVTNG_CMODEL_DIR:-/home/wangzhe/rvtng-cmodel}
qemu=${RVTNG_QEMU:-/home/wangzhe/npu-qemu/build/dl-riscv64}
mount_dir=${NPU_DEVICE_PREFIX:-/tmp/virtual_npu}
cmodel_harts=${NPU_CMODEL_HARTS:-4}
log_file=${NPU_CMODEL_LOG:-"$repo_dir/build/host/driver/cmodel.log"}
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

if pgrep -f "$cmodel_dir/bin/rvt_emulator" >/dev/null 2>&1 || \
   pgrep -f "$qemu.*rvtng-v0" >/dev/null 2>&1; then
    echo "an RVTNG cmodel instance is already running" >&2
    exit 1
fi
case "$cmodel_harts" in
    1|2|3|4) ;;
    *) echo "NPU_CMODEL_HARTS must be between 1 and 4" >&2; exit 1 ;;
esac
cleanup
mkdir -p "$repo_dir/build/host/driver" "$mount_dir"
make -C "$repo_dir" -j4 all
make -C "$runtime_dir" cmodel-smoke-bin

"$cmodel_dir/bin/rvt_emulator" -c "$cmodel_harts" \
    "$qemu" -monitor null -bios "$repo_dir/bl0/bootrom.elf" \
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
        echo "cmodel exited during startup; see $log_file" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    sleep 0.1
done
if [ "$ready" -ne 1 ]; then
    echo "cmodel device did not appear; see $log_file" >&2
    exit 1
fi

NPU_DEVICE_PREFIX="$mount_dir" \
    "$repo_dir/build/host/driver/tests/cmodel_smoke" \
    "$repo_dir/build/npu-os.elf" \
    "$repo_dir/build/npu-os-api.elf" \
    "$repo_dir/build/host/driver/tests/cmodel_kernel.o" \
    "$cmodel_harts"
