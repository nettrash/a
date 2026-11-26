#!/usr/bin/env bash
# The numerics scoreboard (a::math::numerics: roots, systems, quadrature, cubature, Gauss rules,
# ODEs, optimisation, L-BFGS-B, interpolation, AAA) against GSL, Boost, the crates argmin,
# ode_solvers and roots, and SciPy. The lp group (LinearProgram against HiGHS) is the area lp:
# bench/lp/run.sh, with the same probe.
#
#   bench/numerics/run.sh [-o OUT.tsv] [GROUP ...]
#     GROUP: roots systems quad cubature gauss ode opt lbfgsb interp aaa (default all)
#   NUMPROBE_REUSE_PYREF=1 ...   keep the Python references of the last run (they do not depend
#                                on this crate); NUMPROBE_NO_PY=1: no SciPy rows
#   NUMPROBE_QUICK=1 ...         a functional check, not a timing (fewer samples)
#
# References: GSL and Boost (headers) found by pkg-config or under $NUMPROBE_PREFIX (default
# Homebrew's prefix on macOS, /usr on Linux): both required (the area is skipped without them);
# python3 with NumPy and SciPy for the SciPy rows (left out when SciPy is missing).
# Ubuntu: libgsl-dev libboost-dev python3-numpy python3-scipy.
AREA=numerics
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
GROUPS_=${*:-roots systems quad cubature gauss ode opt lbfgsb interp aaa}
[ "$(uname -s)" = Darwin ] && export NUMPROBE_PREFIX=${NUMPROBE_PREFIX:-$(brew_prefix)}
if ! have_header gsl/gsl_errno.h gsl || ! have_header boost/version.hpp; then
  empty_board "GSL or Boost not found"
  exit 0
fi
EXPORT=$WORK/export

cargo build --release --manifest-path "$BENCH_ROOT/numerics/probe/Cargo.toml"
PROBE=$CARGO_TARGET_DIR/release/numprobe
built

mkdir -p "$EXPORT"
NUMPROBE_EXPORT=$EXPORT "$PROBE" export
if [ -n "${NUMPROBE_NO_PY:-}" ]; then
  echo "bench/$AREA: NUMPROBE_NO_PY: the SciPy/HiGHS rows are left out" >&2
elif [ -n "${NUMPROBE_REUSE_PYREF:-}" ] && [ -s "$EXPORT/py_ref.tsv" ]; then
  export NUMPROBE_PYREF=$EXPORT/py_ref.tsv
elif python3 -c 'import numpy, scipy' 2>/dev/null; then
  # Every Python group but lp (bench/lp has it).
  slot python3 "$BENCH_ROOT/numerics/py/refs.py" "$EXPORT" lbfgsb ode fits cub gauss splines cheb bary aaa > "$EXPORT/py_ref.tsv"
  export NUMPROBE_PYREF=$EXPORT/py_ref.tsv
else
  echo "bench/$AREA: no SciPy, the SciPy/HiGHS rows are left out" >&2
fi

header > "$OUT.tmp"
for g in $GROUPS_; do
  echo "== $g" >&2
  slot "$PROBE" "$g" >> "$OUT.tmp" || echo "bench/$AREA: group $g failed" >&2
done
mv "$OUT.tmp" "$OUT"
echo "bench/$AREA: $(($(wc -l < "$OUT") - 1)) rows in $OUT" >&2
