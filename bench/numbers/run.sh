#!/usr/bin/env bash
# The numbers scoreboard: the shared kernels of `Number` that the v0.2 speed pass added, against
# GMP. A white-box probe (wb/): the crate's source compiled into the binary so that its
# pub(crate) kernels are reachable. Rows: short products (nat::mulhigh_to, nat::sqrhigh_to)
# against GMP's mpn_mul_n/mpn_sqr and the crate's whole products, 12-512 limbs; the Jacobi symbol
# (Number::jacobi and the nat::euclid_steps kernel) against mpz_jacobi, 8-2048 limbs;
# quotients_into against quotients, 2-1024 limbs. The public operations of Number (add, mul,
# div, gcd, pow, roots, radix, ...) are measured by the pages' own tables (docs/math/*.md); the
# number-theory functions by bench/nt.
#
#   bench/numbers/run.sh [-o OUT.tsv] [GROUP ...]   GROUP: mul quadratic gcd (default all)
#   NUMBERS_SAMPLES (21)
# GMP is found by pkg-config or in Homebrew's prefix / the standard directories. Ubuntu: libgmp-dev.
AREA=numbers
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
if ! have_lib gmp gmp; then
  empty_board "GMP not found"
  exit 0
fi
cargo build --release --manifest-path "$AREA_DIR/wb/Cargo.toml"
BIN=$CARGO_TARGET_DIR/release/numbers-wb
built
# Same-work checks first (untimed): the error bound of the short products, the Jacobi symbols.
"$BIN" check | tee "$WORK/check.txt" >&2
if grep -E ' [1-9][0-9]* bad' "$WORK/check.txt" >&2; then
  echo "bench/numbers: same-work checks failed" >&2
  exit 1
fi
header > "$OUT.tmp"
for g in ${*:-mul quadratic gcd}; do
  echo "== $g" >&2
  slot env NUMBERS_FILTER="$g" "$BIN" time "${NUMBERS_SAMPLES:-21}" >> "$OUT.tmp" || echo "bench/$AREA: group $g failed" >&2
done
mv "$OUT.tmp" "$OUT"
echo "bench/numbers: $(($(wc -l < "$OUT") - 1)) rows in $OUT" >&2
