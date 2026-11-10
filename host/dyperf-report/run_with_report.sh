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

# One command captures with the unchanged dyperf, then displays its report.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$HERE/../.." && pwd)"
if [[ $# -lt 1 ]]; then
  printf 'Usage: bash %s TRACE_JSON [DYPERF_OPTIONS...] -- COMMAND [ARGS...]\n' "$0" >&2
  exit 2
fi
TRACE="$1"
shift
# Keep a single explicit output path so the analyzed file matches the capture.
for ARG in "$@"; do
  [[ "$ARG" == -- ]] && break
  case "$ARG" in
    -o*|--output*) printf 'Use TRACE_JSON to specify the output path.\n' >&2; exit 2 ;;
  esac
done
DYPERF="${DYPERF:-$REPO/build/host/dyperf/dyperf}"
"$DYPERF" -o "$TRACE" "$@"
"${PYTHON:-python3}" "$HERE/dyperf_report.py" -- "$TRACE"
