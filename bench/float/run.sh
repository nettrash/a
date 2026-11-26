#!/usr/bin/env bash
# The float scoreboard (a::math::float: BigFloat and BigComplex) against MPFR, GNU MPC, rug,
# malachite-float, astro-float, dashu-float and Arb (FLINT 3).
#
#   bench/float/run.sh [-o OUT.tsv] [--quick] [--filter s] [--skip s] [--samples n] [GROUP ...]
#     --filter keeps the rows whose "group/operation/params" contain s, --skip leaves out those
#     that do; GROUP is a prefix of the names in RUNS below (default: all). Each group is one
#     timed command; its rows go to $WORK/raw/<name>.tsv (stderr, with any MISMATCH report, to
#     .err), and the scoreboard is then rebuilt from every raw file present, in the order of RUNS,
#     so that a subset can be re-run.
#   PROFILE=fat ...   ours (and the Rust references) with fat LTO and one codegen unit
#   Same-work checks alone, untimed: $CARGO_TARGET_DIR/release/floatbench <group> <sub> --check.
#
# The probe links GMP, MPFR and MPC statically from the system's archives (Homebrew on macOS;
# libgmp-dev, libmpfr-dev, libmpc-dev on Ubuntu; required: the area is skipped without them) and
# FLINT dynamically (libflint-dev; without it the Arb rows are left out, FLOATBENCH_NO_ARB=1 forces
# that); rug uses the same system libraries (gmp-mpfr-sys with use-system-libs).
AREA=float
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
if ! have_header mpfr.h mpfr || ! have_header mpc.h; then
  empty_board "MPFR or MPC not found"
  exit 0
fi
if [ -n "$(brew_prefix)" ]; then
  # for gmp-mpfr-sys's check of the system libraries (it compiles a C program with -lgmp)
  export CPATH=$(brew_prefix)/include${CPATH:+:$CPATH}
  export LIBRARY_PATH=$(brew_prefix)/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
fi
OPTS=()
SEL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) OPTS+=(--quick); shift ;;
    --filter) OPTS+=(--filter "$2"); shift 2 ;;
    --samples) OPTS+=(--samples "$2"); shift 2 ;;
    --skip) OPTS+=(--skip "$2"); shift 2 ;;
    *) SEL+=("$1"); shift ;;
  esac
done
PROFILE=${PROFILE:-release}
cargo build --profile "$PROFILE" --manifest-path "$AREA_DIR/probe/Cargo.toml"
B=$CARGO_TARGET_DIR/$PROFILE/floatbench
built
"$B" info >&2
RAW=$WORK/raw-$PROFILE
mkdir -p "$RAW"

# name|arguments of the probe (with an optional --filter to split long groups)
RUNS=(
  "arith-into|arith into"
  "arith-new|arith new"
  "arith-assign|arith assign"
  "arith-ops|arith ops"
  "sum-dot|arith sumdot"
  "roots|func roots"
  "conv|conv"
  "text|text"
  "explog-exp|func explog --filter explog/exp"
  "explog-ln|func explog --filter explog/ln"
  "explog-log|func explog --filter explog/log"
  "explog-pow|func explog --filter explog/pow"
  "hyper|func hyper"
  "trig-sin|func trig --filter trig/sin"
  "trig-cos|func trig --filter trig/cos"
  "trig-tan|func trig --filter trig/tan"
  "trig-rec|func trig --filter trig/sec"
  "trig-csc|func trig --filter trig/csc"
  "trig-cot|func trig --filter trig/cot"
  "trig-atan|func trig --filter trig/atan"
  "trig-asin|func trig --filter trig/asin"
  "trig-acos|func trig --filter trig/acos"
  "trig-huge|func trig-huge"
  "consts-first-pi|consts first --filter /pi_"
  "consts-first-e|consts first --filter /e_"
  "consts-first-ln2|consts first --filter /ln2_"
  "consts-first-ln10|consts first --filter /ln10_"
  "consts-first-gamma|consts first --filter /euler_gamma_"
  "consts-first-catalan|consts first --filter /catalan_"
  "consts-cached|consts cached"
  "complex-into|complex into"
  "complex-new|complex new"
  "complex-assign|complex assign"
  "complex-fns|complex fns"
  "matrix-product|matrix product"
  "matrix-solve|matrix solve"
  "matrix-complex|matrix complex"
)

for g in "${RUNS[@]}"; do
  name=${g%%|*}
  args=${g#*|}
  if [ ${#SEL[@]} -gt 0 ]; then
    hit=0
    for s in "${SEL[@]}"; do case $name in "$s"*) hit=1 ;; esac; done
    [ $hit = 1 ] || continue
  fi
  read -r -a argv <<< "$args"
  # a group's own --filter and the caller's are both applied (the probe ANDs them)
  echo "== $name: ${argv[*]} ${OPTS[*]:-}" >&2
  start=$(date +%s)
  slot "$B" "${argv[@]}" ${OPTS[@]+"${OPTS[@]}"} > "$RAW/$name.tsv" 2> "$RAW/$name.err" || echo "!! $name failed" >&2
  echo "   $(wc -l < "$RAW/$name.tsv") rows in $(( $(date +%s) - start )) s; $(grep -c MISMATCH "$RAW/$name.err" || true) mismatch lines" >&2
done

{
  header
  for g in "${RUNS[@]}"; do
    name=${g%%|*}
    if [ -f "$RAW/$name.tsv" ]; then cat "$RAW/$name.tsv"; fi
  done
} > "$OUT"
echo "bench/$AREA: $(( $(wc -l < "$OUT") - 1 )) rows in $OUT" >&2
