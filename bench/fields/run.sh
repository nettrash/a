#!/usr/bin/env bash
# The fields scoreboard (a::math::field: prime fields, extension fields, the BN254/BLS12-381
# towers, binary fields, GF(2^8)/GF(2^128)/GHASH) against FLINT, NTL, GMP, OpenSSL, mcl,
# arkworks, blst, the bls12_381 crate and RustCrypto's ghash and polyval.
#
#   bench/fields/run.sh [-o OUT.tsv] [GROUP ...]   GROUP: prime ext tower binary gf2 (default all),
#                                                 or group:case,case (e.g. gf2:gf2_8) for some cases
#   FIELDBENCH_SAMPLES (21); FIELDBENCH_SPLIT=1: one timed process per case instead of per group
#   FIELDBENCH_CHECK=1: only the checks of the outputs, untimed (a quick test)
#   FIELDBENCH_NO_FLINT=1, _NO_NTL, _NO_GMP, _NO_OSSL, _NO_MCL: leave a reference out
#   NO_MCL=1 / MCL_DIR=dir: as in bench/curves (mcl is built by bench/mcl.sh)
#
# The C references are found by build.rs with pkg-config or in Homebrew's prefix, /usr/local and
# /usr (FIELDBENCH_PREFIXES adds prefixes); a missing one only loses its rows. The primes of the
# prime-field rows are in rs/src/consts.rs and rs/src/arkf.rs, written by gen/gen.py from
# gen/primes.txt (gen/primes.gp, PARI/GP). Ubuntu: libflint-dev libntl-dev libgmp-dev libssl-dev
# pkg-config.
AREA=fields
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
: "${FIELDBENCH_SAMPLES:=21}"
if [ -n "${NO_MCL:-}" ]; then
  export FIELDBENCH_NO_MCL=1
elif [ -z "${MCL_DIR:-}" ]; then
  if MCL_DIR=$("$BENCH_ROOT/mcl.sh"); then export MCL_DIR; else
    echo "bench/$AREA: mcl could not be built, its rows are left out" >&2
    export FIELDBENCH_NO_MCL=1
  fi
fi

cargo build --release --manifest-path "$AREA_DIR/rs/Cargo.toml"
BIN=$CARGO_TARGET_DIR/release/fieldbench
built

tmo=()
command -v timeout >/dev/null 2>&1 && tmo=(timeout 3600)
timed() {
  if [ -n "${FIELDBENCH_CHECK:-}" ]; then
    ${tmo[@]+"${tmo[@]}"} "$BIN" --check "$@"
  else
    slot ${tmo[@]+"${tmo[@]}"} "$BIN" --samples "$FIELDBENCH_SAMPLES" "$@"
  fi
}

PRIME="p64 p128 p192 bn254 bn254r bls381r c25519 secp256k1 p256 bls381 p384 p512 p521 p576 p600 p768 p1024 p2048 p4096"
EXT="3/5 3/12 1000000007/2 1000000007/8 1000000007/24 1000000007/64 bn254/2 bn254/6 bn254/12 bn254/24 bn254/64 bls381/2 bls381/12 p600/3 p600/8"

LOG=$WORK/run.log
PART=$WORK/scoreboard.partial.tsv  # the rows so far, while it runs
: > "$LOG"
"$BIN" --header > "$PART" 2>> "$LOG"
# One timed process per group (a few minutes each); FIELDBENCH_SPLIT=1 takes one per case instead.
each() { # group, cases...
  local g=$1 c
  shift
  echo "== $g" >&2
  if [ -n "${FIELDBENCH_SPLIT:-}" ]; then
    for c in "$@"; do timed "$g:$c" >> "$PART" 2>> "$LOG" || echo "bench/$AREA: $g:$c failed (see $LOG)" >&2; done
  else
    timed "$g:$(echo "$@" | tr ' ' ',')" >> "$PART" 2>> "$LOG" || echo "bench/$AREA: $g failed (see $LOG)" >&2
  fi
}
for g in ${*:-prime ext tower binary gf2}; do
  case $g in
    prime) each prime $PRIME ;;
    ext) each ext $EXT ;;
    tower) each tower bn254 bls12_381 ;;
    binary) each binary 8 64 113 128 131 163 193 233 283 409 571 1024 2048 normal clmul ;;
    gf2) each gf2 gf2_8 gf2_128 ghash ;;
    *:*) echo "== $g" >&2; timed "$g" >> "$PART" 2>> "$LOG" || echo "bench/$AREA: $g failed (see $LOG)" >&2 ;;
    *) echo "bench/$AREA: unknown group $g" >&2; exit 2 ;;
  esac
done
mv "$PART" "$OUT"
echo "bench/$AREA: $(($(wc -l < "$OUT") - 1)) rows in $OUT; log in $LOG" >&2
if grep -q MISMATCH "$OUT" "$LOG"; then
  echo "bench/$AREA: MISMATCHES: see $LOG" >&2
fi
