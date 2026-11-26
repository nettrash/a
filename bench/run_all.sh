#!/usr/bin/env bash
# bench/run_all.sh [--build-only] [--args "ARGS"] [AREA ...]: runs the scoreboard of every area in
# turn (or of the areas named), one area at a time, and leaves the TSVs in bench/results/<area>.tsv
# ($BENCH_RESULTS). An area that fails is reported and the others still run; the exit status is
# non-zero when one failed. --build-only builds every area's probes without timing anything.
# --args passes ARGS (split at spaces, no globbing) to the area's run.sh, e.g. its groups:
#   bench/run_all.sh --args "wmul wdiv" poly
# Each area's own options (groups, filters, sampling) are in its run.sh and in README.md.
set -uo pipefail
BENCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ALL="nt numbers numerics lp float curves special rational fft poly numtypes fields lattice sparse"
ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --build-only) export BENCH_BUILD_ONLY=1; shift ;;
    --args) set -f; read -r -a ARGS <<< "$2"; set +f; shift 2 ;;
    *) break ;;
  esac
done
AREAS=${*:-$ALL}
RESULTS=${BENCH_RESULTS:-$BENCH_ROOT/results}
export BENCH_RESULTS=$RESULTS
mkdir -p "$RESULTS"
failed=()
for a in $AREAS; do
  if [ ! -x "$BENCH_ROOT/$a/run.sh" ]; then
    echo "run_all: no area $a (areas: $ALL)" >&2
    failed+=("$a")
    continue
  fi
  start=$(date +%s)
  echo "=== $a ($(date '+%Y-%m-%d %H:%M:%S'))" >&2
  if "$BENCH_ROOT/$a/run.sh" ${ARGS[@]+"${ARGS[@]}"}; then
    echo "=== $a done in $(( $(date +%s) - start )) s" >&2
  else
    echo "=== $a FAILED after $(( $(date +%s) - start )) s" >&2
    failed+=("$a")
  fi
done
if [ ${#failed[@]} -gt 0 ]; then
  echo "run_all: failed: ${failed[*]}" >&2
  exit 1
fi
