// primesieve and primecount references, one thread each.
#include <primesieve.h>
#include <primecount.h>
#include <stdint.h>
#include "common.h"

void ps_setup(void) {
  primesieve_set_num_threads(1);
  primecount_set_num_threads(1);
}

// The number of primes in [a, b].
uint64_t ps_count(uint64_t a, uint64_t b) { return primesieve_count_primes(a, b); }

// The primes of [a, b] formed one by one by the iterator, summed.
uint64_t ps_iter_sum(uint64_t a, uint64_t b) {
  primesieve_iterator it;
  primesieve_init(&it);
  primesieve_jump_to(&it, a, b);
  uint64_t s = 0, p;
  while ((p = primesieve_next_prime(&it)) <= b) s += p;
  primesieve_free_iterator(&it);
  return s;
}

// The first k primes from a (a included), summed: a new iterator each call.
uint64_t ps_first(uint64_t a, uint64_t k) {
  primesieve_iterator it;
  primesieve_init(&it);
  primesieve_jump_to(&it, a, a + 1000 * k);
  uint64_t s = 0;
  for (uint64_t j = 0; j < k; j++) s += primesieve_next_prime(&it);
  primesieve_free_iterator(&it);
  return s;
}

// The 128-bit API (no cost below 2^63, its own fall-back), so that x up to 2^64 - 1 is counted.
uint64_t pc_pi(uint64_t x) {
  pc_int128_t a;
  a.lo = x;
  a.hi = 0;
  return primecount_pi_128(a).lo;
}
uint64_t pc_nth(uint64_t n) { return (uint64_t)primecount_nth_prime((int64_t)n); }

const char *ps_version_str(void) { return primesieve_version(); }
const char *pc_version_str(void) { return primecount_version(); }
