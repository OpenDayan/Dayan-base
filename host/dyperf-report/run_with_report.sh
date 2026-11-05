#!/usr/bin/env bash
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
