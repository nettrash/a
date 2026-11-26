// FLINT references, one thread (flint_set_num_threads(1)).
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_vec.h>
#include <flint/fmpz_factor.h>
#include <flint/fmpz_poly.h>
#include <flint/ulong_extras.h>
#include <flint/qsieve.h>
#include <flint/arith.h>
#include <flint/partitions.h>
#include <stdlib.h>
#include "common.h"

typedef struct {
  fmpz *z;
  size_t n;
} fpool;

void flint_setup(void) { flint_set_num_threads(1); }

void *flint_pool(const uint64_t *limbs, const uint64_t *off, size_t n) {
  fpool *p = malloc(sizeof(fpool));
  p->z = _fmpz_vec_init(n ? n : 1);
  p->n = n;
  for (size_t i = 0; i < n; i++) fmpz_set_ui_array(p->z + i, (const ulong *)limbs + off[i], off[i + 1] - off[i]);
  return p;
}

void flint_pool_free(void *h) {
  fpool *p = h;
  _fmpz_vec_clear(p->z, p->n ? p->n : 1);
  free(p);
}

static inline uint64_t lo(const fmpz_t z) { return fmpz_get_ui(z); }

// The smaller of f and n/f, the low limb: the same for either factor of a semiprime.
static uint64_t lo_minpair(const fmpz_t f, const fmpz_t n) {
  fmpz_t g;
  fmpz_init(g);
  fmpz_divexact(g, n, f);
  uint64_t r = fmpz_cmp(g, f) < 0 ? lo(g) : lo(f);
  fmpz_clear(g);
  return r;
}

// The checksum of a factorisation: the sum of the low limbs of its primes times their exponents.
static uint64_t fac_sum(const fmpz_factor_t f) {
  uint64_t s = 0;
  for (slong i = 0; i < f->num; i++) s += lo(f->p + i) * f->exp[i];
  return s;
}

enum {
  F_BPSW = 1,       // fmpz_is_probabprime_BPSW
  F_NEXT = 2,       // fmpz_nextprime(r, n, 0)
  F_FACTOR = 3,     // fmpz_factor
  F_QSIEVE = 4,     // qsieve_factor
  F_TRIAL = 5,      // fmpz_factor_trial_range(f, n, 0, arg): arg = number of primes
  F_RHO = 6,        // fmpz_factor_pollard_brent, tries again until it finds
  F_ECM = 7,        // fmpz_factor_ecm(f, curves = arg, B1 = b1, B2 = b2)
  F_PP1 = 8,        // fmpz_factor_pp1 with c = 2/7, 6/5, 6 mod n... (FLINT's c is a word: 3, 5, 7)
  F_PPOW = 9,       // fmpz_is_perfect_power
  F_CRT2 = 10,      // fmpz_CRT: pool a = residues, pool b = moduli, pairs (2i, 2i + 1)
  F_MCRT = 11,      // fmpz_multi_CRT: groups of arg
  F_KRON = 12,      // fmpz_kronecker(a, b)
  F_SQRTMOD = 13,   // fmpz_sqrtmod(r, a, p): pool a, pool b = p
  F_LUCCHAIN = 14,  // fmpz_lucas_chain_full(V, V1, P, Q, m, n): pools a = (P, Q, m, n) in fours
  F_LUCCHAIN1 = 15, // fmpz_lucas_chain(V, V1, P, m, n) for Q = 1: the same fours
  F_DIVISORS = 16,  // arith_divisors(n)
  F_FACPHI = 17,    // fmpz_factor then fmpz_factor_euler_phi (a number's totient)
  F_FACMU = 18,     // fmpz_factor then fmpz_factor_moebius_mu
  F_JACOBI = 19,    // fmpz_jacobi(a, p)
  F_ISPRIME = 20,   // fmpz_is_prime (proves)
};

uint64_t flint_op(int op, void *ha, void *hb, size_t i0, size_t k, uint64_t arg, uint64_t b1, uint64_t b2) {
  fpool *a = ha, *b = hb;
  size_t n = a->n;
  uint64_t s = 0;
  fmpz_t r, r1;
  fmpz_init(r);
  fmpz_init(r1);
  fmpz_factor_t fac;
  flint_rand_t st;
  flint_rand_init(st);
  switch (op) {
    case F_BPSW:
      for (size_t j = 0; j < k; j++) s += fmpz_is_probabprime_BPSW(a->z + item(i0, j, n)) != 0;
      break;
    case F_ISPRIME:
      for (size_t j = 0; j < k; j++) s += fmpz_is_prime(a->z + item(i0, j, n)) == 1;
      break;
    case F_NEXT:
      for (size_t j = 0; j < k; j++) {
        fmpz_nextprime(r, a->z + item(i0, j, n), 0);
        s += lo(r);
      }
      break;
    case F_FACTOR:
      for (size_t j = 0; j < k; j++) {
        fmpz_factor_init(fac);
        fmpz_factor(fac, a->z + item(i0, j, n));
        s += fac_sum(fac);
        fmpz_factor_clear(fac);
      }
      break;
    case F_QSIEVE:
      for (size_t j = 0; j < k; j++) {
        fmpz_factor_init(fac);
        qsieve_factor(fac, a->z + item(i0, j, n));
        s += fac_sum(fac);
        fmpz_factor_clear(fac);
      }
      break;
    case F_TRIAL:
      for (size_t j = 0; j < k; j++) {
        fmpz_factor_init(fac);
        fmpz_factor_trial_range(fac, a->z + item(i0, j, n), 0, arg);
        s += fac->num;
        fmpz_factor_clear(fac);
      }
      break;
    case F_RHO:
      for (size_t j = 0; j < k; j++) {
        fmpz *x = a->z + item(i0, j, n);
        // One try with a long sequence, as `rho(n, iterations)` runs one: again until found.
        while (!fmpz_factor_pollard_brent(r, st, x, 1, arg)) {
        }
        s += lo_minpair(r, x);
      }
      break;
    case F_ECM:
      for (size_t j = 0; j < k; j++) {
        fmpz *x = a->z + item(i0, j, n);
        s += fmpz_factor_ecm(r, arg, b1, b2, st, x) != 0;
      }
      break;
    case F_PP1:
      for (size_t j = 0; j < k; j++) {
        fmpz *x = a->z + item(i0, j, n);
        // Three starting values, as `pp1` tries three.
        ulong cs[3] = {3, 5, 7};
        for (int t = 0; t < 3; t++)
          if (fmpz_factor_pp1(r, x, b1, b2, cs[t])) {
            s += 1;
            break;
          }
      }
      break;
    case F_PPOW:
      for (size_t j = 0; j < k; j++) s += (uint64_t)fmpz_is_perfect_power(r, a->z + item(i0, j, n));
      break;
    case F_CRT2:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n / 2) * 2;
        fmpz_CRT(r, a->z + i, b->z + i, a->z + i + 1, b->z + i + 1, 0);
        s += lo(r);
      }
      break;
    case F_MCRT:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n / arg) * arg;
        if (fmpz_multi_CRT(r, b->z + i, a->z + i, arg, 0)) s += lo(r);
      }
      break;
    case F_KRON:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n);
        s += (uint64_t)(fmpz_kronecker(a->z + i, b->z + i) + 1);
      }
      break;
    case F_JACOBI:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n);
        s += (uint64_t)(fmpz_jacobi(a->z + i, b->z + i) + 1);
      }
      break;
    case F_SQRTMOD:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n);
        if (fmpz_sqrtmod(r, a->z + i, b->z + i)) {
          fmpz_sub(r1, b->z + i, r);
          s += fmpz_cmp(r1, r) < 0 ? lo(r1) : lo(r);
        }
      }
      break;
    case F_LUCCHAIN:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n / 4) * 4;
        fmpz_lucas_chain_full(r, r1, a->z + i, a->z + i + 1, a->z + i + 2, a->z + i + 3);
        s += lo(r);
      }
      break;
    case F_LUCCHAIN1:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n / 4) * 4;
        fmpz_lucas_chain(r, r1, a->z + i, a->z + i + 2, a->z + i + 3);
        s += lo(r);
      }
      break;
    case F_DIVISORS:
      for (size_t j = 0; j < k; j++) {
        fmpz_poly_t d;
        fmpz_poly_init(d);
        arith_divisors(d, a->z + item(i0, j, n));
        s += fmpz_poly_length(d);
        fmpz_poly_clear(d);
      }
      break;
    case F_FACPHI:
      for (size_t j = 0; j < k; j++) {
        fmpz_factor_init(fac);
        fmpz_factor(fac, a->z + item(i0, j, n));
        fmpz_factor_euler_phi(r, fac);
        s += lo(r);
        fmpz_factor_clear(fac);
      }
      break;
    case F_FACMU:
      for (size_t j = 0; j < k; j++) {
        fmpz_factor_init(fac);
        fmpz_factor(fac, a->z + item(i0, j, n));
        s += (uint64_t)(fmpz_factor_moebius_mu(fac) + 1);
        fmpz_factor_clear(fac);
      }
      break;
  }
  flint_rand_clear(st);
  fmpz_clear(r);
  fmpz_clear(r1);
  return s;
}

// Prepared factorisations: group g of pool `primes` is primes[goff[g] .. goff[g + 1]], with the
// exponents exps[] at the same places.
typedef struct {
  fmpz_factor_struct *f;
  size_t n;
} ffpool;

void *flint_facpool(void *hp, const uint32_t *exps, const uint64_t *goff, size_t ng) {
  fpool *p = hp;
  ffpool *q = malloc(sizeof(ffpool));
  q->f = malloc(sizeof(fmpz_factor_struct) * ng);
  q->n = ng;
  for (size_t g = 0; g < ng; g++) {
    fmpz_factor_init(q->f + g);
    for (uint64_t i = goff[g]; i < goff[g + 1]; i++) _fmpz_factor_append(q->f + g, p->z + i, exps[i]);
  }
  return q;
}

enum {
  FF_PHI = 1,     // fmpz_factor_euler_phi
  FF_MU = 2,      // fmpz_factor_moebius_mu
  FF_SIGMA = 3,   // fmpz_factor_divisor_sigma(arg)
};

uint64_t flint_facop(int op, void *hq, size_t i0, size_t k, uint64_t arg) {
  ffpool *q = hq;
  size_t n = q->n;
  uint64_t s = 0;
  fmpz_t r;
  fmpz_init(r);
  for (size_t j = 0; j < k; j++) {
    fmpz_factor_struct *f = q->f + item(i0, j, n);
    switch (op) {
      case FF_PHI:
        fmpz_factor_euler_phi(r, f);
        s += lo(r);
        break;
      case FF_MU:
        s += (uint64_t)(fmpz_factor_moebius_mu(f) + 1);
        break;
      case FF_SIGMA:
        fmpz_factor_divisor_sigma(r, arg, f);
        s += lo(r);
        break;
    }
  }
  fmpz_clear(r);
  return s;
}

enum {
  FN_ISPRIME = 1,     // n_is_prime
  FN_NEXT = 2,        // n_nextprime
  FN_FACTOR = 3,      // n_factor: sum of p * e
  FN_PRIMROOT = 4,    // n_primitive_root_prime
  FN_PHI = 5,         // n_euler_phi
  FN_MU = 6,          // n_moebius_mu
  FN_SQFREE = 7,      // n_is_squarefree
  FN_SQRTMOD = 8,     // n_sqrtmod(xs[2i], xs[2i + 1]): min(r, p - r)
  FN_DLOG = 9,        // n_discrete_log_bsgs(h = xs[3i], g = xs[3i + 1], p = xs[3i + 2])
  FN_PRIMEPI = 10,    // n_prime_pi after flint_cleanup (its first call)
  FN_NTH = 11,        // n_nth_prime after flint_cleanup
  FN_FIB = 12,        // fmpz_fib_ui
  FN_PRIMORIAL = 13,  // fmpz_primorial
  FN_PART = 14,       // partitions_fmpz_ui, caches kept
  FN_PART_CLEAN = 15, // partitions_fmpz_ui after flint_cleanup
  FN_SQRTMODN = 16,   // n_sqrtmodn(a = xs[2i], m = xs[2i + 1]) with n_factor of m before: count + sum
};

uint64_t flint_nop(int op, const uint64_t *xs, size_t n, size_t i0, size_t k) {
  uint64_t s = 0;
  n_factor_t nf;
  fmpz_t r;
  fmpz_init(r);
  switch (op) {
    case FN_ISPRIME:
      for (size_t j = 0; j < k; j++) s += n_is_prime(xs[item(i0, j, n)]);
      break;
    case FN_NEXT:
      for (size_t j = 0; j < k; j++) s += n_nextprime(xs[item(i0, j, n)], 0);
      break;
    case FN_FACTOR:
      for (size_t j = 0; j < k; j++) {
        n_factor_init(&nf);
        n_factor(&nf, xs[item(i0, j, n)], 0);
        for (int t = 0; t < nf.num; t++) s += nf.p[t] * (uint64_t)nf.exp[t];
      }
      break;
    case FN_PRIMROOT:
      for (size_t j = 0; j < k; j++) s += n_primitive_root_prime(xs[item(i0, j, n)]);
      break;
    case FN_PHI:
      for (size_t j = 0; j < k; j++) s += n_euler_phi(xs[item(i0, j, n)]);
      break;
    case FN_MU:
      for (size_t j = 0; j < k; j++) s += (uint64_t)(n_moebius_mu(xs[item(i0, j, n)]) + 1);
      break;
    case FN_SQFREE:
      for (size_t j = 0; j < k; j++) s += n_is_squarefree(xs[item(i0, j, n)]) != 0;
      break;
    case FN_SQRTMOD:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n / 2) * 2;
        ulong p = xs[i + 1], x = n_sqrtmod(xs[i], p);
        s += (p - x < x) ? p - x : x;
      }
      break;
    case FN_SQRTMODN:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n / 2) * 2;
        ulong *roots = NULL;
        n_factor_init(&nf);
        n_factor(&nf, xs[i + 1], 1);
        slong c = n_sqrtmodn(&roots, xs[i], &nf);
        s += (uint64_t)c;
        for (slong t = 0; t < c; t++) s += roots[t];
        flint_free(roots);
      }
      break;
    case FN_DLOG:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n / 3) * 3;
        s += n_discrete_log_bsgs(xs[i], xs[i + 1], xs[i + 2]);
      }
      break;
    case FN_PRIMEPI:
      for (size_t j = 0; j < k; j++) {
        flint_cleanup();
        s += n_prime_pi(xs[item(i0, j, n)]);
      }
      break;
    case FN_NTH:
      for (size_t j = 0; j < k; j++) {
        flint_cleanup();
        s += n_nth_prime(xs[item(i0, j, n)]);
      }
      break;
    case FN_FIB:
      for (size_t j = 0; j < k; j++) {
        fmpz_fib_ui(r, xs[item(i0, j, n)]);
        s += lo(r);
      }
      break;
    case FN_PRIMORIAL:
      for (size_t j = 0; j < k; j++) {
        fmpz_primorial(r, xs[item(i0, j, n)]);
        s += lo(r);
      }
      break;
    case FN_PART:
      for (size_t j = 0; j < k; j++) {
        partitions_fmpz_ui(r, xs[item(i0, j, n)]);
        s += lo(r);
      }
      break;
    case FN_PART_CLEAN:
      for (size_t j = 0; j < k; j++) {
        flint_cleanup();
        partitions_fmpz_ui(r, xs[item(i0, j, n)]);
        s += lo(r);
      }
      break;
  }
  fmpz_clear(r);
  return s;
}

const char *flint_version_str(void) { return FLINT_VERSION; }
