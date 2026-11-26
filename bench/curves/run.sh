#!/usr/bin/env bash
# The curves scoreboard: pairings (BLS12-381, BN254), hashing to curves, binary curves (the 18
# SEC 2 curves) and SHA-2 against their references.
#
#   bench/curves/run.sh [-o OUT.tsv] [PART ...]   PART: pairing h2c binary sha (default all)
#   NO_MCL=1 ...      without mcl (its rows are left out); MCL_DIR=dir: an mcl build of your own
#
# References:
#   crates.io (in the probe crate only): arkworks 0.5, blst 0.3.17, bls12_381 0.8, substrate-bn 0.6,
#     sha2 0.9/0.10.9/0.11.1, ring 0.17.14, elliptic-curve 0.13.8, p256/p384/p521/k256 0.13,
#     curve25519-dalek 5.0.0.
#   mcl (herumi/mcl): cloned and built at a pinned commit by bench/mcl.sh into bench/ext/mcl
#     (needs git, make, a C++ compiler and, the first time, the network).
#   OpenSSL 3 and 4: loaded at run time (dlopen) from OPENSSL3_LIB / OPENSSL4_LIB if set, else
#     Homebrew's openssl@3 / openssl@4 (macOS) or libcrypto.so.3 / .so.4 (Linux, e.g. Ubuntu's
#     libssl3t64); a missing one is skipped.
# Nothing here uses a macOS-only framework.
AREA=curves
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
PARTS=("$@")
[ ${#PARTS[@]} -gt 0 ] || PARTS=(pairing h2c binary sha)

if [ -n "${NO_MCL:-}" ]; then
  export MCL_DIR=/nonexistent
elif [ -z "${MCL_DIR:-}" ]; then
  if MCL_DIR=$("$BENCH_ROOT/mcl.sh"); then export MCL_DIR; else
    echo "bench/$AREA: mcl could not be built, its rows are left out" >&2
    export MCL_DIR=/nonexistent
  fi
fi

cargo build --release --bins --manifest-path "$AREA_DIR/probe/Cargo.toml"
BIN=$CARGO_TARGET_DIR/release
built

header > "$OUT.tmp"
for part in "${PARTS[@]}"; do
  echo "== $part" >&2
  case $part in
    pairing)
      slot "$BIN/pairing" bls12_381 >> "$OUT.tmp" || echo "bench/$AREA: pairing bls12_381 failed" >&2
      slot "$BIN/pairing" bn254 >> "$OUT.tmp" || echo "bench/$AREA: pairing bn254 failed" >&2
      ;;
    h2c)
      slot "$BIN/h2c" >> "$OUT.tmp" || echo "bench/$AREA: h2c failed" >&2
      ;;
    binary)
      # One process per curve, to keep each timed command short.
      for c in sect113r1 sect113r2 sect131r1 sect131r2 sect163k1 sect163r1 sect163r2 sect193r1 \
               sect193r2 sect233k1 sect233r1 sect239k1 sect283k1 sect283r1 sect409k1 sect409r1 \
               sect571k1 sect571r1; do
        slot "$BIN/binary" "$c" >> "$OUT.tmp" || echo "bench/$AREA: binary $c failed" >&2
      done
      ;;
    sha)
      slot "$BIN/sha" >> "$OUT.tmp" || echo "bench/$AREA: sha failed" >&2
      ;;
    *)
      echo "bench/$AREA: unknown part $part (pairing h2c binary sha)" >&2
      exit 2
      ;;
  esac
done
mv "$OUT.tmp" "$OUT"
echo "bench/$AREA: $(($(wc -l < "$OUT") - 1)) rows in $OUT" >&2
