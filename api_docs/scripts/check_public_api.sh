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

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
missing=0

check_identifier()
{
    local chapter=$1
    local identifier=$2

    if ! grep -Fq "\\code{$identifier}" "$repo_root/$chapter"; then
        printf 'error: public identifier %s is not documented in %s\n' \
            "$identifier" "$chapter" >&2
        missing=1
    fi
}

check_pattern()
{
    local chapter=$1
    local pattern=$2
    shift 2

    while IFS= read -r identifier; do
        case "$identifier" in
        *_H|*_DEFINED|*_LINKAGE|*_)
            continue
            ;;
        esac
        check_identifier "$chapter" "$identifier"
    done < <(
        rg --no-filename -o "$pattern" "$@" | sort -u
    )
}

cd "$repo_root"

check_pattern api_docs/chapters/01_hal.tex \
    '\b(?:NPU|DAYAN)_[A-Z][A-Z0-9_]+\b' \
    host/driver/hal/include/*.h
check_pattern api_docs/chapters/01_hal.tex \
    '\b(?:npu|dayan)_[a-z0-9_]+_t\b' \
    host/driver/hal/include/*.h

check_pattern api_docs/chapters/02_service.tex \
    '\b(?:NPU|DAYAN)_[A-Z][A-Z0-9_]+\b' \
    host/driver/server/service.h host/driver/server/server.h
check_pattern api_docs/chapters/02_service.tex \
    '\b(?:npu_service|dayan_server)_[a-z0-9_]+_t\b' \
    host/driver/server/service.h host/driver/server/server.h

check_pattern api_docs/chapters/03_runtime.tex \
    '\b(?:NPU|DAYAN)_[A-Z][A-Z0-9_]+\b' \
    host/include/npu_runtime.h host/include/npu_package.h \
    host/include/dayan_kernel.h host/dycc/include/dayan_compiler.h
check_pattern api_docs/chapters/03_runtime.tex \
    '\b(?:npu|dayan)_[a-z0-9_]+_t\b|\bNPUTaskDim3\b|\bNPULaunchConfig\b' \
    host/include/npu_runtime.h host/include/npu_package.h \
    host/include/dayan_kernel.h
for identifier in kSingleCore kFourCore __host__ __device__ __kernel__; do
    check_identifier api_docs/chapters/03_runtime.tex "$identifier"
done

check_pattern api_docs/chapters/04_rpc.tex \
    '\bRPC_[A-Z][A-Z0-9_]+\b|\brpc_[a-z0-9_]+\b' \
    device/include/rpc.h

check_pattern api_docs/chapters/05_dapi.tex \
    '\bDAYAN_[A-Z][A-Z0-9_]+\b|\bdayan_[a-z0-9_]+_t\b' \
    device/include/api/dayan_attrs.h device/include/api/dayan_sys.h
for identifier in \
    shared dayan_resource_role dayan_task_axis \
    FILE stdout EOF \
    INFINITY NAN HUGE_VALF isnan isinf isfinite signbit \
    size_t ptrdiff_t wchar_t max_align_t NULL offsetof \
    bool true false va_list va_start va_arg va_end va_copy \
    int8_t uint8_t int16_t uint16_t int32_t uint32_t int64_t uint64_t \
    intptr_t uintptr_t intmax_t uintmax_t \
    INT8_MIN INT8_MAX UINT8_MAX INT16_MIN INT16_MAX UINT16_MAX \
    INT32_MIN INT32_MAX UINT32_MAX INT64_MIN INT64_MAX UINT64_MAX \
    INTPTR_MAX UINTPTR_MAX SIZE_MAX \
    CHAR_BIT SCHAR_MIN SCHAR_MAX UCHAR_MAX SHRT_MIN SHRT_MAX USHRT_MAX \
    INT_MIN INT_MAX UINT_MAX LONG_MIN LONG_MAX ULONG_MAX; do
    check_identifier api_docs/chapters/05_dapi.tex "$identifier"
done

exit "$missing"
