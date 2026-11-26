#!/usr/bin/env bash
# The poly scoreboard (a::math::poly: polynomials over word and multi-limb moduli, Z, Q, floats
# and GF(2); products, division, gcd, evaluation, roots, factoring) against FLINT 3, NTL, PARI/GP,
# LAPACK (Accelerate on macOS, liblapack on Linux) and rustfft/realfft.
#
#   bench/poly/run.sh [-o OUT.tsv] [GROUP ...]
#     GROUP: wmul wdiv mlimb zarith qarith farith gcd eval roots factor (default all)
#   PROFILE=lto ...   the second figure: fat LTO, one codegen unit
#   LIMIT (2400)      seconds after which a group is cut
#
# The C references are found by pkg-config or the standard include paths (Homebrew's prefix on
# macOS); a missing one is replaced by a stub and its rows are left out (POLYBENCH_NO_FLINT=1,
# POLYBENCH_NO_NTL=1, POLYBENCH_NO_PARI=1 force that).
# Ubuntu: libflint-dev libntl-dev libpari-dev liblapack-dev pkg-config.
AREA=poly
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
PROFILE=${PROFILE:-release}
# (Not GROUPS: bash keeps that name for the user's group ids and ignores an assignment.)
SEL=${*:-wmul wdiv mlimb zarith qarith farith gcd eval roots factor}
LIMIT=${LIMIT:-2400}
# Single-threaded on every side: PARI's MT engine is set to one thread in its shim, FLINT and NTL
# to one thread in theirs; Accelerate (vecLib) and OpenBLAS by their environment.
export VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cargo build --profile "$PROFILE" --manifest-path "$AREA_DIR/Cargo.toml"
BIN=$CARGO_TARGET_DIR/$PROFILE/polybench
built

run() {
  if command -v timeout > /dev/null; then slot timeout "$LIMIT" "$@"; else slot "$@"; fi
}

header > "$OUT.tmp"
for g in $SEL; do
  echo "== $g $(date '+%H:%M:%S')" >&2
  part=$WORK/$g.tsv
  rm -f "$part"
  if ! run "$BIN" --out "$part" "$g" 2> "$WORK/$g.log"; then
    echo "bench/$AREA: group $g failed or timed out (see $WORK/$g.log)" >&2
  fi
  if [ -f "$part" ]; then cat "$part" >> "$OUT.tmp"; fi
  grep -h MISMATCH "$WORK/$g.log" >&2 || true
done
mv "$OUT.tmp" "$OUT"
echo "bench/$AREA: $(($(wc -l < "$OUT") - 1)) rows in $OUT" >&2
