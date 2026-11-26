#!/usr/bin/env bash
# The rational scoreboard (a::math::rational, and Matrix<Rational>/Matrix<Number>) against GMP's
# mpq, FLINT's fmpq, PARI, malachite-q, num-rational and rug.
#
#   bench/rational/run.sh [-o OUT.tsv] [GROUP ...]   all groups by default (see `ratbench list`)
#   REF_PREFIX=dir[:dir]   where GMP / FLINT / PARI live, before Homebrew's prefix, /usr/local, /usr
#   REF_NO_FLINT=1, REF_NO_PARI=1   leave those references out
#   NO_RUG=1               build without rug (rug needs GMP's headers to build against)
#   RB_SAMPLES (21), RB_SAMPLES_LONG (15, for operations past 30 ms), RB_TARGET_MS (1.2)
#   SKIP_LONG=1            leave out seq_long (B_(10^6): about five minutes)
#   LTO=1                  the probe with fat LTO and one codegen unit (profile `lto`): the second
#                          figure for ours
#
# GMP is required (the area is skipped without it); FLINT and PARI are optional.
# Ubuntu: libgmp-dev libflint-dev libpari-dev pari-gp.
AREA=rational
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
if ! have_header gmp.h gmp; then
  empty_board "GMP not found"
  exit 0
fi
profile=release
[ -n "${LTO:-}" ] && profile=lto
parts=$WORK/parts-$profile
mkdir -p "$parts"

# GMP's headers for rug's build check (gmp-mpfr-sys with use-system-libs compiles `cc -lgmp`).
rp=${REF_PREFIX:-}
for p in ${rp//:/ } $(brew_prefix) /usr/local; do
  if [ -f "$p/include/gmp.h" ]; then
    export CPATH="$p/include${CPATH:+:$CPATH}" LIBRARY_PATH="$p/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    break
  fi
done

feat=()
[ -n "${NO_RUG:-}" ] && feat=(--no-default-features)
build() { cargo build --manifest-path "$AREA_DIR/Cargo.toml" --profile "$profile" ${feat[@]+"${feat[@]}"} "$@"; }
if ! build; then
  echo "bench/$AREA: build with rug failed; retrying without it" >&2
  feat=(--no-default-features)
  build
fi
bin=$CARGO_TARGET_DIR/$profile/ratbench
built

if [ $# -gt 0 ]; then
  groups=("$@")
else
  read -r -a groups <<< "$("$bin" list)"
fi
for g in "${groups[@]}"; do
  if [ "$g" = seq_long ] && [ -n "${SKIP_LONG:-}" ]; then
    continue
  fi
  echo "== $g" >&2
  # Each group is one timed process of a few minutes at most.
  slot "$bin" "$g" > "$parts/$g.tsv" 2> "$parts/$g.err" || echo "bench/$AREA: group $g failed (see $parts/$g.err)" >&2
  grep -h MISMATCH "$parts/$g.err" >&2 || true
done

# Assemble in the order of the group list (all groups' parts that exist).
{
  "$bin" header
  read -r -a all <<< "$("$bin" list)"
  for g in "${all[@]}"; do
    if [ -f "$parts/$g.tsv" ]; then cat "$parts/$g.tsv"; fi
  done
} > "$OUT"
echo "bench/$AREA: $(($(wc -l < "$OUT") - 1)) rows in $OUT" >&2
