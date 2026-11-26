#!/usr/bin/env bash
# The special-functions scoreboard (a::math::special) against statrs, special, puruspe, libm,
# lambert_w, the system libm (Apple's on macOS, glibc's on Linux), Boost.Math, GSL and SciPy.
#
#   bench/special/run.sh [-o OUT.tsv] [--quick] [GROUP ...]
#     GROUP: gamma incgamma incbeta erf complex bessel01 bessel_n bessel_v sph_bessel airy zeta
#            polylog lambert expint elliptic carlson orthopoly f32 scipy   (default: all)
#     --quick  5 samples a row instead of 21 (a smoke test; not for the record)
#   A run over some groups replaces their rows and keeps the others' of earlier runs in $WORK
#   (merge.py keeps the newest row of each case and reference); SB_THIS_RUN=1 keeps this run's only.
#   SB_NO_BOOST=1 / SB_NO_GSL=1 leave those references out; SB_FILTER=text keeps the Rust/C cases
#   whose key contains the text; PYTHON (default python3).
#
# Three timed passes, one slot each: ours alone with fat LTO (the "ours with fat LTO" notes), then
# every Rust/C reference against ours in the default release build, then SciPy against ours
# through the cdylib (ctypes). The C/C++ drivers are built at -O3 -mcpu=native (aarch64) or
# -march=native (x86_64); Boost and GSL are found by pkg-config or in Homebrew's prefix,
# /usr/local and /usr, and a missing one only loses its rows. SciPy rows need python3 with scipy.
# Ubuntu: libgsl-dev libboost-dev pkg-config python3-scipy.
AREA=special
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
RUNS=$WORK/run
mkdir -p "$RUNS"
PY=${PYTHON:-python3}
# slot SECONDS CMD...: CMD in a timing slot, under a timeout where `timeout` exists
tslot() {
  local t=$1
  shift
  if command -v timeout >/dev/null; then slot timeout "$t" "$@"; else slot "$@"; fi
}

SAMPLES=21
if [ "${1:-}" = --quick ]; then SAMPLES=5; shift; fi
ALL="gamma incgamma incbeta erf complex bessel01 bessel_n bessel_v sph_bessel airy zeta polylog lambert expint elliptic carlson orthopoly f32 scipy"
SEL=${*:-$ALL}
RUST=""
DO_SCIPY=0
for g in $SEL; do
  if [ "$g" = scipy ]; then DO_SCIPY=1; else RUST=${RUST:+$RUST,}$g; fi
done
if [ $DO_SCIPY = 1 ] && ! "$PY" -c "import scipy.special" 2>/dev/null; then
  echo "bench/$AREA: no scipy: SciPy rows left out" >&2
  DO_SCIPY=0
fi
TAG=$(date +%Y%m%d-%H%M%S)

if command -v pkg-config >/dev/null && pkg-config --exists gsl 2>/dev/null; then
  SB_INCLUDE=$(pkg-config --variable=includedir gsl)
  SB_LIBDIR=$(pkg-config --variable=libdir gsl)
  export SB_INCLUDE SB_LIBDIR
fi

(cd "$AREA_DIR" && cargo build --release && cargo build --profile lto)
built
BIN=$CARGO_TARGET_DIR/release/special-bench
LTO=$CARGO_TARGET_DIR/lto/special-bench
case $(uname -s) in Darwin) CDY=$CARGO_TARGET_DIR/release/libsbench.dylib ;; *) CDY=$CARGO_TARGET_DIR/release/libsbench.so ;; esac
"$BIN" --list > "$RUNS/cases.txt"

FILTER=()
[ -n "${SB_FILTER:-}" ] && FILTER=(--filter "$SB_FILTER")
if [ -n "$RUST" ]; then
  echo "== ours with fat LTO: $RUST" >&2
  tslot 1800 "$LTO" --group "$RUST" ${FILTER[@]+"${FILTER[@]}"} --samples "$SAMPLES" --ours-only > "$RUNS/lto_$TAG.txt" 2> "$RUNS/lto_$TAG.err" || echo "bench/$AREA: the LTO pass failed (see $RUNS/lto_$TAG.err)" >&2
  echo "== references: $RUST" >&2
  tslot 3600 "$BIN" --group "$RUST" ${FILTER[@]+"${FILTER[@]}"} --samples "$SAMPLES" --lto "$RUNS/lto_$TAG.txt" > "$RUNS/rows_$TAG.tsv" 2> "$RUNS/rows_$TAG.err" || echo "bench/$AREA: the references pass failed (see $RUNS/rows_$TAG.err)" >&2
fi
if [ $DO_SCIPY = 1 ]; then
  echo "== scipy" >&2
  tslot 3600 "$PY" "$AREA_DIR/scipy_bench.py" "$CDY" --samples "$SAMPLES" > "$RUNS/rows_${TAG}_scipy.tsv" 2> "$RUNS/rows_${TAG}_scipy.err" || echo "bench/$AREA: the SciPy pass failed (see $RUNS/rows_${TAG}_scipy.err)" >&2
fi

# Oldest first: merge.py keeps the last row it reads for each case and reference.
if [ -n "${SB_THIS_RUN:-}" ]; then
  FILES=$(ls -tr "$RUNS"/rows_"$TAG"*.tsv 2>/dev/null || true)
else
  FILES=$(ls -tr "$RUNS"/rows_*.tsv 2>/dev/null || true)
fi
# shellcheck disable=SC2086
"$PY" "$AREA_DIR/merge.py" "$RUNS/cases.txt" $FILES > "$OUT"
echo "bench/$AREA: $(($(wc -l < "$OUT") - 1)) rows in $OUT" >&2
