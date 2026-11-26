// Shared declarations of the reference shims of the nt scoreboard.
// A pool is a set of numbers passed from Rust as little-endian 64-bit limbs: number i is
// limbs[off[i] .. off[i + 1]] (an empty range is zero). Each library converts a pool once into
// its own representation (a handle), so that conversions stay out of the timed loops. Every op
// runs `k` calls on the items (i0 + j) mod n and returns a checksum that the Rust side compares
// with its own over the same items.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
  const uint64_t *limbs;
  const uint64_t *off;
  size_t n;
} raw_pool;

static inline size_t item(size_t i0, size_t j, size_t n) { return (i0 + j) % n; }
