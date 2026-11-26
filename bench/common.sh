# shellcheck shell=bash
# Sourced by every bench/<area>/run.sh after it has set AREA, before anything else:
#
#   AREA=nt
#   . "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
#
# It removes `-o FILE` from the script's arguments and sets:
#
#   BENCH_ROOT        bench/ (the directory of this file)
#   REPO_ROOT         the repository; the crate is $REPO_ROOT/lib
#   AREA_DIR          bench/<area>
#   OUT               the scoreboard to write: FILE of `-o FILE`, else $BENCH_RESULTS/<area>.tsv
#   WORK              the area's working directory, $BENCH_WORK/<area> (default bench/work/<area>):
#                     raw parts, logs, generated inputs; never committed
#   CARGO_TARGET_DIR  default bench/target: one build directory for every area
#   CARGO_BUILD_JOBS  default: every core
#
# and the helpers below (slot, header, empty_board, built, have_header, have_lib, brew_prefix).
set -euo pipefail

BENCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname "$BENCH_ROOT")
AREA_DIR=$BENCH_ROOT/$AREA
BENCH_RESULTS=${BENCH_RESULTS:-$BENCH_ROOT/results}
BENCH_WORK=${BENCH_WORK:-$BENCH_ROOT/work}
CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$BENCH_ROOT/target}
# Absolute, since the scripts change directory.
case $BENCH_RESULTS in /*) ;; *) BENCH_RESULTS=$PWD/$BENCH_RESULTS ;; esac
case $BENCH_WORK in /*) ;; *) BENCH_WORK=$PWD/$BENCH_WORK ;; esac
case $CARGO_TARGET_DIR in /*) ;; *) CARGO_TARGET_DIR=$PWD/$CARGO_TARGET_DIR ;; esac
export CARGO_TARGET_DIR
WORK=$BENCH_WORK/$AREA
if [ -z "${CARGO_BUILD_JOBS:-}" ]; then
  CARGO_BUILD_JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
fi
export CARGO_BUILD_JOBS
# The timing wrapper: bench/slot.sh (one timing process at a time on the machine) unless
# BENCH_SLOT names another one; BENCH_SLOT= (set and empty) runs the timed commands directly.
BENCH_SLOT=${BENCH_SLOT-$BENCH_ROOT/slot.sh}

OUT=
_bench_args=()
while [ $# -gt 0 ]; do
  case $1 in
    -o) OUT=$2; shift 2 ;;
    -o*) OUT=${1#-o}; shift ;;
    *) _bench_args+=("$1"); shift ;;
  esac
done
set -- ${_bench_args[@]+"${_bench_args[@]}"}
unset _bench_args
OUT=${OUT:-$BENCH_RESULTS/$AREA.tsv}
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
mkdir -p "$(dirname "$OUT")" "$WORK"

# slot CMD...: one timed command, through the timing wrapper.
slot() {
  if [ -n "$BENCH_SLOT" ]; then "$BENCH_SLOT" "$@"; else "$@"; fi
}

# The scoreboard's header line.
header() {
  printf 'area\tgroup\toperation\tparams\tours_ns\treference\tref_ns\tratio\tnotes\n'
}

# empty_board WHY: a scoreboard with the header only (the area cannot run here), and why.
empty_board() {
  echo "bench/$AREA: skipped: $*" >&2
  [ -n "${BENCH_BUILD_ONLY:-}" ] && return 0
  header > "$OUT"
  echo "bench/$AREA: skipped: $*" > "$WORK/skipped.txt"
}

# built: called by run.sh once its probes are built; with BENCH_BUILD_ONLY=1 it stops there
# (bench/run_all.sh --build-only builds every area without timing anything).
built() {
  if [ -n "${BENCH_BUILD_ONLY:-}" ]; then
    echo "bench/$AREA: built" >&2
    exit 0
  fi
}

# Homebrew's prefix on macOS (empty elsewhere), found once.
BREW_PREFIX=
if [ "$(uname -s)" = Darwin ]; then
  if command -v brew >/dev/null 2>&1; then BREW_PREFIX=$(brew --prefix); elif [ -d /opt/homebrew ]; then BREW_PREFIX=/opt/homebrew; else BREW_PREFIX=/usr/local; fi
fi
brew_prefix() { echo "$BREW_PREFIX"; }

# have_header REL [PKG]: true when the C header REL (e.g. gmp.h, suitesparse/cholmod.h) is in
# pkg-config PKG's include directories, Homebrew's prefix, /usr/local/include or /usr/include.
have_header() {
  local rel=$1 pkg=${2:-} d
  if [ -n "$pkg" ] && command -v pkg-config >/dev/null 2>&1 && pkg-config --exists "$pkg" 2>/dev/null; then
    for d in $(pkg-config --cflags-only-I "$pkg" | sed 's/-I//g') "$(pkg-config --variable=includedir "$pkg")"; do
      [ -f "$d/$rel" ] && return 0
    done
  fi
  for d in "$(brew_prefix)/include" /usr/local/include /usr/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu; do
    [ -f "$d/$rel" ] && return 0
  done
  return 1
}

# have_lib NAME [PKG]: true when libNAME.{so,dylib,a} is in pkg-config PKG's library directory or a
# standard one.
have_lib() {
  local name=$1 pkg=${2:-} d e
  local dirs=("$(brew_prefix)/lib" /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib)
  if [ -n "$pkg" ] && command -v pkg-config >/dev/null 2>&1 && pkg-config --exists "$pkg" 2>/dev/null; then
    dirs=("$(pkg-config --variable=libdir "$pkg")" "${dirs[@]}")
  fi
  for d in "${dirs[@]}"; do
    for e in so dylib a; do
      [ -e "$d/lib$name.$e" ] && return 0
    done
  done
  return 1
}
