#!/usr/bin/env bash
# bench/slot.sh CMD...: runs one timed command in one of $BENCH_SLOTS (default 1) timing slots of
# this machine, so that the benchmarks of several areas started at once never time at the same
# time. It waits for a free slot, runs CMD, frees the slot, and exits with CMD's status. Compiles
# need no slot. The slots are directories in $BENCH_SLOT_DIR (default $TMPDIR/nettrash-a-slots);
# one whose owner has died is taken over.
set -u
dir=${BENCH_SLOT_DIR:-${TMPDIR:-/tmp}/nettrash-a-slots}
n=${BENCH_SLOTS:-1}
mkdir -p "$dir"
got=
while [ -z "$got" ]; do
  i=1
  while [ "$i" -le "$n" ]; do
    if mkdir "$dir/s$i" 2>/dev/null; then
      got=$dir/s$i
      echo $$ > "$got/pid"
      break
    fi
    pid=$(cat "$dir/s$i/pid" 2>/dev/null || true)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      rm -rf "$dir/s$i"
      continue
    fi
    i=$((i + 1))
  done
  [ -n "$got" ] || sleep 2
done
trap 'rm -rf "$got"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$@"
