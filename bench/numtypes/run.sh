#!/usr/bin/env bash
# The numtypes scoreboard (a::math::{quaternion, dual, uint}) against glam, nalgebra, num-dual,
# ruint, crypto-bigint, bnum, malachite-nz, GMP and OpenSSL's BN.
#
#   bench/numtypes/run.sh [-o OUT.tsv] [BINARY ...]   BINARY: uint mont quat dual (default all)
#   PROFILE=lto ...                ours and the Rust references with fat LTO, one codegen unit
#   NTBENCH_FILTER='uint div_rem'  only the rows whose "group operation" contains the text
#   UINT_NS='4 8' MONT_NS='4'      only these widths (limbs) of uint and mont
#
# Every timed command has a 30-minute timeout where `timeout` exists. GMP and OpenSSL are found by
# pkg-config or in the standard places (Homebrew, /usr/lib/<multiarch>); a library that is missing
# only drops its rows. Ubuntu: libgmp-dev libssl-dev.
AREA=numtypes
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
profile=${PROFILE:-release}
tmo=()
command -v timeout >/dev/null 2>&1 && tmo=(timeout 1800)

cargo build --profile "$profile" --bins --manifest-path "$AREA_DIR/Cargo.toml"
bindir=$CARGO_TARGET_DIR/$profile
built

rows=$WORK/rows-$profile.tsv
: > "$rows"
run() { # name [VAR=value ...]: one timed process
  local name=$1
  shift
  echo "== $name $*" >&2
  slot env "$@" ${tmo[@]+"${tmo[@]}"} "$bindir/$name" >> "$rows" || echo "bench/$AREA: $name $* failed" >&2
}
for b in ${*:-uint mont quat dual}; do
  case $b in
    uint) for n in ${UINT_NS:-1 2 4 6 8 16}; do run uint NTBENCH_N="$n"; done ;;
    mont) for n in ${MONT_NS:-1 2 4 6 8 16 32 64}; do run mont NTBENCH_N="$n"; done ;;
    *) run "$b" ;;
  esac
done

{
  header
  cat "$rows"
} > "$OUT"
echo "bench/$AREA: $(($(wc -l < "$OUT") - 1)) rows in $OUT" >&2
