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

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$script_dir/.." && pwd)

prefix=${PREFIX:-/usr/local}
destdir=${DESTDIR:-}
bindir=${BINDIR:-"$prefix/bin"}
includedir=${INCLUDEDIR:-"$prefix/include"}
libdir=${LIBDIR:-"$prefix/lib"}
datadir=${DATADIR:-"$prefix/share"}
sdkdir=${DAYAN_SDKDIR:-"$libdir/dayan"}
firmwaredir=${DAYAN_FIRMWAREDIR:-"$sdkdir/firmware"}
install_tool=${INSTALL:-install}

for destination in "$bindir" "$includedir" "$libdir" "$datadir" \
        "$sdkdir" "$firmwaredir"; do
    case "$destination" in
        /*) ;;
        *)
            echo "install-dayan: error: installation paths must be absolute" >&2
            exit 2
            ;;
    esac
    case "$destination/" in
        */../*)
            echo "install-dayan: error: installation paths must be normalized" >&2
            exit 2
            ;;
    esac
done

required=(
    "$root/build/host/dycc/dycc-bin"
    "$root/build/host/dyperf/dyperf"
    "$root/build/host/dyrun/dyrun"
    "$root/build/host/driver/libDayanRuntime.a"
    "$root/build/host/driver/libDayanRuntime.so"
    "$root/build/host/driver/libnpu_runtime.a"
    "$root/build/host/driver/libDayanHal.a"
    "$root/build/host/driver/libDayanService.a"
    "$root/build/host/driver/dayan_server"
    "$root/build/device/npu-os.elf"
    "$root/build/device/npu-os-api.elf"
    "$root/build/device/npu-os.bin"
    "$root/build/device/libnpu-os.a"
)
for file in "${required[@]}"; do
    [[ -f "$file" ]] || {
        echo "install-dayan: error: missing build artifact: $file" >&2
        exit 2
    }
done

host_include_target="$destdir$includedir"
library_target="$destdir$libdir"
sdk_target="$destdir$sdkdir"
firmware_target="$destdir$firmwaredir"
pkgconfig_target="$library_target/pkgconfig"
documentation_target="$destdir$datadir/doc/dayan"

"$install_tool" -d "$destdir$bindir" "$host_include_target" \
    "$library_target" "$sdk_target/include/api" \
    "$sdk_target/host/include" "$sdk_target/host/dycc/include" \
    "$sdk_target/host/driver/hal" "$sdk_target/host/driver/server" \
    "$firmware_target" "$pkgconfig_target" "$documentation_target"

"$install_tool" -m 0755 "$root/build/host/dycc/dycc-bin" \
    "$destdir$bindir/dycc"
"$install_tool" -m 0755 "$root/build/host/dyperf/dyperf" \
    "$destdir$bindir/dyperf"
"$install_tool" -m 0755 "$root/build/host/dyrun/dyrun" \
    "$destdir$bindir/dyrun"
"$install_tool" -m 0755 "$root/build/host/driver/dayan_server" \
    "$destdir$bindir/"
"$install_tool" -m 0644 "$root/build/host/driver/libDayanRuntime.a" \
    "$root/build/host/driver/libnpu_runtime.a" \
    "$root/build/host/driver/libDayanHal.a" \
    "$root/build/host/driver/libDayanService.a" "$library_target/"
"$install_tool" -m 0755 "$root/build/host/driver/libDayanRuntime.so" \
    "$library_target/"
"$install_tool" -m 0644 "$root/host/driver/dayan-runtime.pc" \
    "$pkgconfig_target/"
"$install_tool" -m 0644 "$root/host/include/"*.h "$host_include_target/"

"$install_tool" -m 0644 "$root/device/include/api/"*.h \
    "$sdk_target/include/api/"
"$install_tool" -m 0644 "$root/host/include/"*.h \
    "$sdk_target/host/include/"
"$install_tool" -m 0644 "$root/host/dycc/include/dayan_compiler.h" \
    "$sdk_target/host/dycc/include/"
"$install_tool" -m 0644 "$root/host/driver/hal/include/"*.h \
    "$sdk_target/host/driver/hal/"
"$install_tool" -m 0644 "$root/host/driver/server/server.h" \
    "$root/host/driver/server/service.h" \
    "$sdk_target/host/driver/server/"
"$install_tool" -m 0644 "$root/host/driver/protocol.h" \
    "$sdk_target/host/driver/"
"$install_tool" -m 0644 "$root/device/dayan-api.symbols" "$sdk_target/"

"$install_tool" -m 0644 "$root/build/device/npu-os.elf" \
    "$root/build/device/npu-os-api.elf" "$root/build/device/npu-os.bin" \
    "$root/build/device/libnpu-os.a" "$root/device/linker.ld" \
    "$firmware_target/"
"$install_tool" -m 0644 "$root/Readme.md" \
    "$documentation_target/README.md"
"$install_tool" -m 0644 "$root/host/dycc/README.md" \
    "$documentation_target/dycc.md"
"$install_tool" -m 0644 "$root/host/driver/README.md" \
    "$documentation_target/driver.md"
"$install_tool" -m 0644 "$root/host/dyperf/README.md" \
    "$documentation_target/dyperf.md"
"$install_tool" -m 0644 "$root/host/dyrun/README.md" \
    "$documentation_target/dyrun.md"
"$install_tool" -m 0644 "$root/device/BAREMETAL.md" \
    "$documentation_target/baremetal.md"

echo "install-dayan: dycc       -> $bindir/dycc"
echo "install-dayan: dyperf     -> $bindir/dyperf"
echo "install-dayan: dyrun      -> $bindir/dyrun"
echo "install-dayan: server     -> $bindir/dayan_server"
echo "install-dayan: Host SDK   -> $includedir, $libdir"
echo "install-dayan: Device SDK -> $sdkdir"
echo "install-dayan: firmware   -> $firmwaredir"
