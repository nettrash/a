#!/usr/bin/env bash
# The lp scoreboard: a::math::numerics::LinearProgram::solve (the dense primal simplex for small
# programmes, the dual simplex on a sparse LU above LpOptions::dual_threshold) against HiGHS 1.12
# (dual simplex, and interior point with crossover) through SciPy's bindings, timing HiGHS's run
# alone. The probe is the numerics area's (bench/numerics/probe, its group lp).
#
#   bench/lp/run.sh [-o OUT.tsv]
#   NUMPROBE_REUSE_PYREF=1 ...   keep the Python references of the last run
#
# Without SciPy (python3 with numpy and scipy) the HiGHS rows, i.e. all of them, are left out.
# Ubuntu: python3-numpy python3-scipy (and the numerics area's libgsl-dev libboost-dev, which the
# probe links).
AREA=lp
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
[ "$(uname -s)" = Darwin ] && export NUMPROBE_PREFIX=${NUMPROBE_PREFIX:-$(brew_prefix)}
if ! have_header gsl/gsl_errno.h gsl || ! have_header boost/version.hpp; then
  empty_board "GSL or Boost not found (the numerics probe links them)"
  exit 0
fi
if ! python3 -c 'import numpy, scipy' 2>/dev/null; then
  empty_board "no SciPy (HiGHS is reached through scipy.optimize)"
  exit 0
fi
EXPORT=$WORK/export

cargo build --release --manifest-path "$BENCH_ROOT/numerics/probe/Cargo.toml"
PROBE=$CARGO_TARGET_DIR/release/numprobe
built

mkdir -p "$EXPORT"
NUMPROBE_EXPORT=$EXPORT "$PROBE" export
if [ -n "${NUMPROBE_REUSE_PYREF:-}" ] && [ -s "$EXPORT/py_ref.tsv" ]; then :; else
  slot python3 "$BENCH_ROOT/numerics/py/refs.py" "$EXPORT" lp > "$EXPORT/py_ref.tsv"
fi
export NUMPROBE_PYREF=$EXPORT/py_ref.tsv

header > "$OUT.tmp"
slot "$PROBE" lp >> "$OUT.tmp" || echo "bench/$AREA: the lp group failed" >&2
mv "$OUT.tmp" "$OUT"
echo "bench/$AREA: $(($(wc -l < "$OUT") - 1)) rows in $OUT" >&2
