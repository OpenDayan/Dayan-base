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
server_bin=${DAYAN_SERVER_BIN:-"$repo_dir/build/host/driver/dayan_server"}
firmware=${DAYAN_FIRMWARE:-"$repo_dir/build/device/npu-os.elf"}
api_elf=${DAYAN_API_ELF:-"$repo_dir/build/device/npu-os-api.elf"}
xdma_profile=${NPU_XDMA_PROFILE:-simulator}
xdma_device_dir=${NPU_XDMA_DEVICE_DIR:-${NPU_DEVICE_PREFIX:-}}
xdma_mmio_offset=${NPU_XDMA_MMIO_OFFSET:-}
device_index=${NPU_DEVICE_INDEX:-0}
cluster_count=${NPU_CLUSTER_COUNT:-1}
cluster_stride=${NPU_CLUSTER_STRIDE:-1}
cluster_mask=${NPU_CLUSTER_MASK:-}
enable_mmu=${NPU_ENABLE_MMU:-0}
core_mask=${NPU_CORE_MASK:-0xf}
cores_per_cluster=${NPU_CORES_PER_CLUSTER:-}
socket_path=${DAYAN_SERVER_SOCKET:-/tmp/dayan-server.sock}
wait_seconds=${DAYAN_DEVICE_WAIT_SECONDS:-30}
build_jobs=${DAYAN_BUILD_JOBS:-4}
skip_build=${DAYAN_SKIP_BUILD:-0}
rpc_debug=${DAYAN_RPC_DEBUG:-1}

status()
{
    printf '[run_server] %s\n' "$*" >&2
}

usage()
{
    cat <<EOF
usage: $0 [dayan_server options]

Start dayan_server after an external backend exposes its XDMA-compatible
device files.  The backend may be cmodel, npu-cluster-sim, or hardware.

Environment:
  NPU_XDMA_PROFILE           simulator or fpga (default: simulator)
  NPU_XDMA_DEVICE_DIR        override the profile's XDMA device directory
  NPU_XDMA_MMIO_OFFSET       override the profile's low-address MMIO offset
  NPU_DEVICE_INDEX           first XDMA index (default: 0)
  NPU_CLUSTER_COUNT          number of Clusters (default: 1)
  NPU_CLUSTER_STRIDE         XDMA index stride (default: 1)
  NPU_CLUSTER_MASK           optional enabled Cluster mask
  NPU_CORES_PER_CLUSTER      optional manual Core count (default: auto from
                             BootROM, fallback 4); warns on hardware mismatch.
                             NPU_CORE_MASK is clipped to the Core count.
  NPU_CORE_MASK              enabled Core mask (default: 0xf)
  DAYAN_SERVER_SOCKET        Runtime socket (default: /tmp/dayan-server.sock)
  DAYAN_FIRMWARE             BL1 ELF path
  DAYAN_API_ELF              public Device API ELF path
  DAYAN_SERVER_BIN           dayan_server executable path
  DAYAN_DEVICE_WAIT_SECONDS  device wait timeout (default: 30)
  DAYAN_BUILD_JOBS           build parallelism (default: 4)
  DAYAN_SKIP_BUILD           set to 1 to use existing artifacts
  DAYAN_RPC_DEBUG            Host RPC debug output, 0 or 1 (default: 1)

Command-line dayan_server options override the corresponding environment
values; unrecognized options are forwarded unchanged.
EOF
}

if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
    usage
    exit 0
fi

forwarded_args=("$@")
extra_args=()
for ((arg_index = 0; arg_index < ${#forwarded_args[@]}; ++arg_index)); do
    option=${forwarded_args[arg_index]}
    case "$option" in
        --socket|--firmware|--api-elf|--xdma-profile|--xdma-device-dir|\
        --xdma-mmio-offset|--device-prefix|--device-index|--cluster-count|\
        --cluster-stride|--cluster-mask|--core-mask|--cores-per-cluster)
            if [ "$((arg_index + 1))" -ge "${#forwarded_args[@]}" ]; then
                echo "missing value after $option" >&2
                exit 2
            fi
            value=${forwarded_args[++arg_index]}
            case "$option" in
                --socket) socket_path=$value ;;
                --firmware) firmware=$value ;;
                --api-elf) api_elf=$value ;;
                --xdma-profile) xdma_profile=$value ;;
                --xdma-device-dir|--device-prefix) xdma_device_dir=$value ;;
                --xdma-mmio-offset) xdma_mmio_offset=$value ;;
                --device-index) device_index=$value ;;
                --enable-mmu) enable_mmu=$value ;;
                --cluster-count) cluster_count=$value ;;
                --cluster-stride) cluster_stride=$value ;;
                --cluster-mask) cluster_mask=$value ;;
                --core-mask) core_mask=$value ;;
                --cores-per-cluster) cores_per_cluster=$value ;;
            esac
            ;;
        *) extra_args+=("$option") ;;
    esac
done

case "$xdma_profile" in
    simulator)
        xdma_device_dir=${xdma_device_dir:-/tmp/virtual_npu}
        ;;
    fpga)
        xdma_device_dir=${xdma_device_dir:-/dev}
        ;;
    *)
        echo "NPU_XDMA_PROFILE must be simulator or fpga" >&2
        exit 2
        ;;
esac

positive_decimal()
{
    case "$1" in
        ''|*[!0-9]*|0) return 1 ;;
        *) return 0 ;;
    esac
}

nonnegative_decimal()
{
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

if ! positive_decimal "$cluster_count"; then
    echo "NPU_CLUSTER_COUNT must be between 1 and 32" >&2
    exit 2
fi
if ! positive_decimal "$cluster_stride"; then
    echo "NPU_CLUSTER_STRIDE must be a positive decimal integer" >&2
    exit 2
fi
if [ -n "$cores_per_cluster" ] && ! positive_decimal "$cores_per_cluster"; then
    echo "NPU_CORES_PER_CLUSTER must be a positive decimal integer" >&2
    exit 2
fi
if ! nonnegative_decimal "$device_index"; then
    echo "NPU_DEVICE_INDEX must be a non-negative decimal integer" >&2
    exit 2
fi
if ! nonnegative_decimal "$wait_seconds"; then
    echo "DAYAN_DEVICE_WAIT_SECONDS must be a non-negative integer" >&2
    exit 2
fi
if ! positive_decimal "$build_jobs"; then
    echo "DAYAN_BUILD_JOBS must be a positive integer" >&2
    exit 2
fi
case "$skip_build" in
    0|1) ;;
    *) echo "DAYAN_SKIP_BUILD must be 0 or 1" >&2; exit 2 ;;
esac
case "$rpc_debug" in
    0) rpc_debug_state=disabled ;;
    1) rpc_debug_state=enabled ;;
    *) echo "DAYAN_RPC_DEBUG must be 0 or 1" >&2; exit 2 ;;
esac

cluster_count=$((10#$cluster_count))
cluster_stride=$((10#$cluster_stride))
device_index=$((10#$device_index))
enable_mmu=$((10#$enable_mmu))
if [ -n "$cores_per_cluster" ]; then
    cores_per_cluster=$((10#$cores_per_cluster))
    if [ "$cores_per_cluster" -gt 4 ]; then
        echo "NPU_CORES_PER_CLUSTER must be between 1 and 4" >&2
        exit 2
    fi
fi
wait_seconds=$((10#$wait_seconds))
build_jobs=$((10#$build_jobs))
if [ "$cluster_count" -gt 32 ]; then
    echo "NPU_CLUSTER_COUNT must be between 1 and 32" >&2
    exit 2
fi

if [ "$skip_build" -eq 0 ]; then
    status "building firmware and host driver with $build_jobs jobs"
    make -C "$repo_dir" -j"$build_jobs" device host-driver
    status "build completed"
else
    status "using existing build artifacts (DAYAN_SKIP_BUILD=1)"
fi
if [ ! -x "$server_bin" ]; then
    echo "dayan_server is not executable: $server_bin" >&2
    exit 2
fi
if [ ! -f "$firmware" ]; then
    echo "Device firmware does not exist: $firmware" >&2
    exit 2
fi
if [ ! -f "$api_elf" ]; then
    echo "Device API ELF does not exist: $api_elf" >&2
    exit 2
fi

devices_ready()
{
    local cluster
    local index

    for ((cluster = 0; cluster < cluster_count; ++cluster)); do
        index=$((device_index + cluster * cluster_stride))
        if [ ! -w "$xdma_device_dir/xdma${index}_h2c_0" ] ||
           [ ! -r "$xdma_device_dir/xdma${index}_c2h_0" ]; then
            return 1
        fi
    done
    return 0
}

attempt=0
max_attempts=$((wait_seconds * 10))
status "waiting up to ${wait_seconds}s for ${cluster_count} XDMA device(s) under $xdma_device_dir"
while ! devices_ready; do
    if [ "$attempt" -ge "$max_attempts" ]; then
        echo "XDMA device files did not appear under $xdma_device_dir" >&2
        exit 1
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -gt 0 ] && [ "$((attempt % 50))" -eq 0 ]; then
        status "still waiting for XDMA endpoints ($((attempt / 10))s elapsed)"
    fi
    sleep 0.1
done
status "XDMA endpoints are ready"

server_args=(
    --socket "$socket_path"
    --firmware "$firmware"
    --api-elf "$api_elf"
    --xdma-profile "$xdma_profile"
    --xdma-device-dir "$xdma_device_dir"
    --device-index "$device_index"
    --enable-mmu "$enable_mmu"
    --cluster-count "$cluster_count"
    --cluster-stride "$cluster_stride"
    --core-mask "$core_mask"
)
if [ -n "$xdma_mmio_offset" ]; then
    server_args+=(--xdma-mmio-offset "$xdma_mmio_offset")
fi
if [ -n "$cluster_mask" ]; then
    server_args+=(--cluster-mask "$cluster_mask")
fi
if [ -n "$cores_per_cluster" ]; then
    server_args+=(--cores-per-cluster "$cores_per_cluster")
fi
server_args+=("${extra_args[@]}")

status "starting dayan_server: socket=$socket_path clusters=$cluster_count cores_per_cluster=${cores_per_cluster:-default} core_mask=$core_mask"
status "firmware=$firmware api_elf=$api_elf"
status "Host RPC debug output: $rpc_debug_state"
export DAYAN_RPC_DEBUG="$rpc_debug"
exec "$server_bin" "${server_args[@]}"
