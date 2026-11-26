#!/usr/bin/env bash
# The lattice scoreboard (issue #23: LLL, deep insertions, BKZ, enumeration, PSLQ, normal forms,
# exact linear algebra, ModMatrix, BitMatrix) against FLINT, fplll, NTL, M4RI, PARI and mpmath.
#
#   bench/lattice/run.sh [-o OUT.tsv] [JOB ...]   JOB: group[:filter], e.g. lll:knapsack modp
#                                                 (default: every job of JOBS below)
#   LATBENCH_PROFILE=fat ...   the second figure (fat LTO, one codegen unit)
#   LATBENCH_TIMEOUT (5400)    seconds a job; LATBENCH_SKIP=pari,ntl skips references on purpose
#
# The references are found by pkg-config or in Homebrew's prefix, /usr/local and /usr
# (LATBENCH_PREFIXES adds more); a missing one only loses its rows. PSLQ's mpmath rows need
# python3 with mpmath (LATBENCH_PYTHON). Ubuntu: libflint-dev libfplll-dev libntl-dev libm4ri-dev
# libpari-dev libgmp-dev libmpfr-dev python3-mpmath pkg-config.
AREA=lattice
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
PROFILE=${LATBENCH_PROFILE:-release}
# Single-threaded references (FLINT and NTL default to one thread already; M4RI here has no
# OpenMP; these cover builds that do).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
TIMEOUT=${LATBENCH_TIMEOUT:-5400}

cargo build --profile "$PROFILE" --manifest-path "$AREA_DIR/Cargo.toml" >&2
BUILT=$CARGO_TARGET_DIR/$PROFILE/latbench
[ "$PROFILE" = dev ] && BUILT=$CARGO_TARGET_DIR/debug/latbench
built
# Run a copy, so that a rebuild during the run cannot touch the running binary.
BIN=$WORK/latbench
cp "$BUILT" "$BIN"
"$BIN" versions >&2

# group[:filter]: the filters split the long groups into jobs of minutes.
JOBS=${*:-lll:knapsack lll:NTRU lll:q-ary lll:uniform lll:Goldstein lll-exact deep isred \
bkz:knapsack bkz:q-ary bkz:NTRU bkz:Goldstein bkz:uniform enum pslq normal exact \
modp:* modp:ple modp:rank modp:det modp:solve modp:inverse modp:rref modp:null_space modp:from_matrix,+,-,scale,transpose \
gf2:n=256,n=1024 gf2:n=4096 gf2:n=8192 gf2:n=16384 gf2:add_row,swap_rows}
tmo=()
command -v timeout >/dev/null 2>&1 && tmo=(timeout "$TIMEOUT")

header > "$OUT.tmp"
set -f  # the jobs hold * (modp:*)
for job in $JOBS; do
  group=${job%%:*}
  filter=""
  case $job in *:*) filter=${job#*:} ;; esac
  echo "== $group ${filter:+($filter)}" >&2
  if [ -n "$filter" ]; then
    slot env LATBENCH_FILTER="$filter" ${tmo[@]+"${tmo[@]}"} "$BIN" "$group" >> "$OUT.tmp" || echo "bench/$AREA: $job failed" >&2
  else
    slot ${tmo[@]+"${tmo[@]}"} "$BIN" "$group" >> "$OUT.tmp" || echo "bench/$AREA: $job failed" >&2
  fi
done
set +f
mv "$OUT.tmp" "$OUT"
echo "bench/$AREA: $(($(wc -l < "$OUT") - 1)) rows in $OUT" >&2
