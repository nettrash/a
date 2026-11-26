// GMP references: primality, next/previous prime, Kronecker/Jacobi symbols, perfect powers,
// Fibonacci/Lucas numbers and primorials. Results go into numbers allocated before the loop.
#include <gmp.h>
#include <stdlib.h>
#include "common.h"

typedef struct {
  mpz_t *z;
  size_t n;
} gpool;

static void set_from(mpz_t z, const uint64_t *l, size_t len) {
  mpz_import(z, len, -1, sizeof(uint64_t), 0, 0, l);
}

void *gmp_pool(const uint64_t *limbs, const uint64_t *off, size_t n) {
  gpool *p = malloc(sizeof(gpool));
  p->z = malloc(sizeof(mpz_t) * (n ? n : 1));
  p->n = n;
  for (size_t i = 0; i < n; i++) {
    mpz_init(p->z[i]);
    set_from(p->z[i], limbs + off[i], off[i + 1] - off[i]);
  }
  return p;
}

void gmp_pool_free(void *h) {
  gpool *p = h;
  for (size_t i = 0; i < p->n; i++) mpz_clear(p->z[i]);
  free(p->z);
  free(p);
}

static inline uint64_t lo(const mpz_t z) { return mpz_size(z) ? mpz_getlimbn(z, 0) : 0; }

enum {
  G_PRIME = 1,   // mpz_probab_prime_p(n, 24) != 0
  G_NEXT = 2,    // mpz_nextprime
  G_PREV = 3,    // mpz_prevprime
  G_KRON = 4,    // mpz_kronecker(a, b), pools a and b
  G_JACOBI = 5,  // mpz_jacobi(a, b)
  G_PPOW = 6,    // mpz_perfect_power_p
};

uint64_t gmp_op(int op, void *ha, void *hb, size_t i0, size_t k) {
  gpool *a = ha, *b = hb;
  size_t n = a->n;
  uint64_t s = 0;
  mpz_t r;
  mpz_init2(r, 64 * 80);
  switch (op) {
    case G_PRIME:
      for (size_t j = 0; j < k; j++) s += mpz_probab_prime_p(a->z[item(i0, j, n)], 24) != 0;
      break;
    case G_NEXT:
      for (size_t j = 0; j < k; j++) {
        mpz_nextprime(r, a->z[item(i0, j, n)]);
        s += lo(r);
      }
      break;
    case G_PREV:
      for (size_t j = 0; j < k; j++) {
        if (mpz_prevprime(r, a->z[item(i0, j, n)])) s += lo(r);
      }
      break;
    case G_KRON:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n);
        s += (uint64_t)(mpz_kronecker(a->z[i], b->z[i]) + 1);
      }
      break;
    case G_JACOBI:
      for (size_t j = 0; j < k; j++) {
        size_t i = item(i0, j, n);
        s += (uint64_t)(mpz_jacobi(a->z[i], b->z[i]) + 1);
      }
      break;
    case G_PPOW:
      for (size_t j = 0; j < k; j++) s += mpz_perfect_power_p(a->z[item(i0, j, n)]) != 0;
      break;
  }
  mpz_clear(r);
  return s;
}

enum {
  G_FIB = 1,        // mpz_fib_ui(n)
  G_LUC = 2,        // mpz_lucnum_ui(n)
  G_FIB2L = 3,      // mpz_fib2_ui(n) and L_n = F_n + 2 F_(n-1): U_n, V_n for (1, -1)
  G_PRIMORIAL = 4,  // mpz_primorial_ui(n)
};

// The sequences take a list of indices; the result numbers are allocated once, at the size of
// the largest result, so that the loop does not reallocate (as the earlier measurements did).
uint64_t gmp_seq(int op, const uint64_t *ns, size_t n, size_t i0, size_t k) {
  uint64_t s = 0;
  static mpz_t f, f1;
  static int init = 0;
  if (!init) {
    mpz_init(f);
    mpz_init(f1);
    init = 1;
  }
  switch (op) {
    case G_FIB:
      for (size_t j = 0; j < k; j++) {
        mpz_fib_ui(f, ns[item(i0, j, n)]);
        s += lo(f);
      }
      break;
    case G_LUC:
      for (size_t j = 0; j < k; j++) {
        mpz_lucnum_ui(f, ns[item(i0, j, n)]);
        s += lo(f);
      }
      break;
    case G_FIB2L:
      for (size_t j = 0; j < k; j++) {
        uint64_t m = ns[item(i0, j, n)];
        mpz_fib2_ui(f, f1, m);
        s += lo(f);
        mpz_mul_2exp(f1, f1, 1);  // 2 F_(n-1)
        mpz_add(f1, f1, f);
        s += lo(f1);
      }
      break;
    case G_PRIMORIAL:
      for (size_t j = 0; j < k; j++) {
        mpz_primorial_ui(f, ns[item(i0, j, n)]);
        s += lo(f);
      }
      break;
  }
  return s;
}

const char *gmp_version_str(void) { return gmp_version; }
