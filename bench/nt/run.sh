#!/usr/bin/env bash
# The nt scoreboard (issue #15: sieves, primes, factoring, modular and quadratic arithmetic,
# arithmetic functions, sequences of a::math::numbers) against primesieve, primecount, GMP, FLINT,
# PARI, GMP-ECM, msieve and the crates primal, machine-prime, machine-factor and malachite.
#
#   bench/nt/run.sh [-o OUT.tsv] [TASK ...]   every task by default (the list: `ntbench list`)
#   NT_LONG=1 ...         also the rows of minutes (pi to 2^64, siqs at 75/80 digits, ...)
#   NT_FILTER="10^12" ... only the rows whose "operation params" contain the text
#   NT_SAMPLES (15), NT_BUDGET (seconds a row, 15), NT_MIN_BATCH_MS (1): the sampling
#   NT_CPUTIME=1          the thread's CPU time instead of wall time (a machine loaded past its
#                         cores; not for msieve, a separate process)
#   PROFILE=fat ...       the crate and the probe with fat LTO and one codegen unit
#
# The C references are found by pkg-config or in Homebrew's prefix, /usr/local and /usr
# (build.rs; extra prefixes in NT_PREFIXES, colon-separated; NT_SKIP=gmp,pari,... leaves some out
# on purpose); a missing one is left out with a warning and its rows time ours only. msieve is optional, found on PATH.
# Ubuntu: libgmp-dev libflint-dev libpari-dev pari-gp libecm-dev libprimesieve-dev
# libprimecount-dev (msieve where packaged).
AREA=nt
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
PROFILE=${PROFILE:-release}
NT_SCRATCH=$WORK/scratch
export NT_SCRATCH
mkdir -p "$NT_SCRATCH" "$WORK/parts"
# FLINT's quadratic sieve may write temporary files: keep them in the working directory.
TMPDIR=$NT_SCRATCH
export TMPDIR

cargo build --profile "$PROFILE" --manifest-path "$AREA_DIR/Cargo.toml"
BIN=$CARGO_TARGET_DIR/$PROFILE/ntbench
[ "$PROFILE" = dev ] && BIN=$CARGO_TARGET_DIR/debug/ntbench
built

TASKS=${*:-$("$BIN" list | cut -f1)}
slot "$BIN" meta > "$WORK/meta.txt"
cat "$WORK/meta.txt" >&2
for t in $TASKS; do
  for long in "" 1; do
    [ -n "$long" ] && [ -z "${NT_LONG:-}" ] && continue
    suf=${long:+.long}
    echo "== $t$suf ($(date '+%H:%M:%S'))" >&2
    if slot env NT_LONG="${long:+only}" "$BIN" "$t" > "$WORK/parts/$t$suf.tsv.new" 2> "$WORK/parts/$t$suf.log"; then :; else
      echo "!! $t$suf failed: see $WORK/parts/$t$suf.log" >&2
    fi
    mv "$WORK/parts/$t$suf.tsv.new" "$WORK/parts/$t$suf.tsv"
    tail -n 1 "$WORK/parts/$t$suf.log" >&2
  done
done
# The scoreboard: every task's latest rows (and its long rows), in the order of `ntbench list`.
"$BIN" header > "$OUT.tmp"
for t in $("$BIN" list | cut -f1); do
  for f in "$WORK/parts/$t.tsv" "$WORK/parts/$t.long.tsv"; do
    if [ -f "$f" ]; then cat "$f" >> "$OUT.tmp"; fi
  done
done
mv "$OUT.tmp" "$OUT"
echo "bench/nt: $(($(wc -l < "$OUT") - 1)) rows in $OUT" >&2
