#!/usr/bin/env bash
# bench/mcl.sh [DIR]: clones herumi/mcl (BSD-3-Clause) at the commit the scoreboards were measured
# with and builds its static library lib/libmcl.a with the default configuration (MCL_FP_BIT=384,
# MCL_FR_BIT=256: one build serves BN254 and BLS12-381), for the rows of the areas curves and
# fields. DIR defaults to $MCL_DIR, else bench/ext/mcl. Prints DIR. Nothing is installed; mcl is
# never vendored in this repository. Needs git, make, a C++ compiler and, once, the network.
set -euo pipefail
MCL_COMMIT=a1c4e29e4192e1a68baa10d28dc3fe1aa8b15a52  # herumi/mcl, 2026-11-18 ("Merge branch 'dev'")
MCL_URL=https://github.com/herumi/mcl.git
BENCH_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
D=${1:-${MCL_DIR:-$BENCH_ROOT/ext/mcl}}
JOBS=${CARGO_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
if [ -f "$D/lib/libmcl.a" ] && [ "$(git -C "$D" rev-parse HEAD 2>/dev/null)" = "$MCL_COMMIT" ]; then
  echo "$D"
  exit 0
fi
if [ "$(git -C "$D" rev-parse HEAD 2>/dev/null || true)" != "$MCL_COMMIT" ]; then
  rm -rf "$D"
  mkdir -p "$D"
  git -C "$D" init -q
  git -C "$D" remote add origin "$MCL_URL"
  git -C "$D" fetch -q --depth 1 origin "$MCL_COMMIT"
  git -C "$D" checkout -q FETCH_HEAD
fi
if [ "$(uname -s)" = Darwin ]; then
  make -C "$D" -j "$JOBS" lib/libmcl.a >&2
else
  make -C "$D" -j "$JOBS" lib/libmcl.a CXX="${CXX:-g++}" >&2
fi
echo "$D"
