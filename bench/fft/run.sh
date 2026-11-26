#!/usr/bin/env bash
# The fft scoreboard: a::math::fft (complex, real, DCT/DST, n-dimensional transforms,
# convolutions, planning) against FFTW 3, rustfft/realfft/rustdct and, on macOS, Accelerate's vDSP.
#
#   bench/fft/run.sh [-o OUT.tsv] [JOB-FILTER ...]
#     JOB-FILTER: substrings of the job names below ("complex f64", "conv c32 conv", ...); none = all.
#   FFTBENCH_PATIENT      1 = also FFTW_PATIENT plans for powers of two 64..2^20 (default 1)
#   FFTBENCH_SAMPLES / FFTBENCH_MIN_MS / FFTBENCH_WARM_MS   timing (default 15 / 1.0 / 3.0)
#   RUNS                  how many full runs (default 1); with more, each row is the median of
#                         the runs' figures (merge.py), the runs kept as $WORK/run<K>.tsv
#   PROFILE               release (default) or fat (fat LTO, one codegen unit)
#   FFTW_LIB_DIR          where libfftw3 and libfftw3f are, when pkg-config does not know
# FFTW 3 (double and float) is required: the area is skipped without it. FFTW's wisdom is kept in
# $WORK/wisdom. Ubuntu: libfftw3-dev pkg-config; the vDSP rows exist on macOS only.
AREA=fft
. "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
PROFILE=${PROFILE:-release}
if [ -z "${FFTW_LIB_DIR:-}" ] && ! { have_lib fftw3 fftw3 && have_lib fftw3f fftw3f; }; then
  empty_board "FFTW 3 (libfftw3, libfftw3f) not found"
  exit 0
fi
export FFTBENCH_WISDOM_DIR=$WORK/wisdom
export FFTBENCH_PATIENT=${FFTBENCH_PATIENT:-1}
LOGS=$WORK/logs-$PROFILE
mkdir -p "$FFTBENCH_WISDOM_DIR" "$LOGS"

(cd "$AREA_DIR" && cargo build --profile "$PROFILE")
BIN=$CARGO_TARGET_DIR/$PROFILE/fftbench
built

JOBS=()
for t in f64 f32; do
  for p in pow2 smooth prime inverse oop batch; do JOBS+=("complex $t $p"); done
  for p in forward inverse batch; do JOBS+=("real $t $p"); done
  for p in dct dst; do JOBS+=("dct $t $p"); done
  JOBS+=("nd $t all")
done
for t in f32 f64 c32 c64; do
  for p in conv corr circ; do JOBS+=("conv $t $p"); done
done
JOBS+=("plan f64 all" "plan f32 all")

selected() {
  [ $# -eq 1 ] && return 0
  local job=$1
  shift
  for f in "$@"; do case $job in *"$f"*) return 0 ;; esac; done
  return 1
}

RUNS=${RUNS:-1}
start=$(date +%s)
fails=0
for ((run = 1; run <= RUNS; run++)); do
  TMP=$WORK/run$run.tsv.part
  "$BIN" header > "$TMP"
  for job in "${JOBS[@]}"; do
    selected "$job" "$@" || continue
    log=$LOGS/${job// /_}.run$run.log
    echo "== run $run/$RUNS: $job ($(date +%H:%M:%S))" >&2
    read -r -a argv <<< "$job"
    slot "$BIN" "${argv[@]}" >> "$TMP" 2> "$log" || { echo "FAILED: $job (see $log)" >&2; tail -3 "$log" >&2; }
    tail -1 "$log" >&2
    fails=$((fails + $(grep -c 'CHECK FAILED' "$log" || true)))
  done
  mv "$TMP" "$WORK/run$run.tsv"
done
if [ "$RUNS" -gt 1 ]; then
  runs=()
  for ((run = 1; run <= RUNS; run++)); do runs+=("$WORK/run$run.tsv"); done
  python3 -I "$AREA_DIR/merge.py" "${runs[@]}" > "$OUT"
else
  cp "$WORK/run1.tsv" "$OUT"
fi
echo "bench/$AREA: $(($(wc -l < "$OUT") - 1)) rows in $OUT, $fails same-work check failures, $(( $(date +%s) - start )) s" >&2
