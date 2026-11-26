// Reference kernels for the rational scoreboard: GMP's mpq_t and FLINT's fmpq, driven in
// batches from the Rust harness. Every batch cycles over a pool of inputs, writing into a
// result variable that persists across calls (the way these libraries are used), and folds
// anything scalar into a sink so the compiler keeps the work.
#include <gmp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

volatile uint64_t g_sink;
volatile double g_dsink;

static void imp(mpz_t z, const uint8_t *b, size_t len, int neg) {
  mpz_import(z, len, -1, 1, 0, 0, b);
  if (neg) mpz_neg(z, z);
}

static size_t put(char *buf, size_t cap, const char *s) {
  size_t l = strlen(s);
  if (l + 1 > cap) l = cap - 1;
  memcpy(buf, s, l);
  buf[l] = 0;
  return l;
}

// ---------------------------------------------------------------------------------------
// GMP
// ---------------------------------------------------------------------------------------

typedef struct {
  size_t n;
  mpq_t *q;
  mpq_t r;
  mpz_t z, z2;
  long *si;
  double *d;
  char **s;
  char *buf;
  size_t bufcap;
  size_t cur;
} gpool;

void *gq_new(size_t n) {
  gpool *p = calloc(1, sizeof *p);
  p->n = n;
  p->q = malloc(n * sizeof(mpq_t));
  for (size_t i = 0; i < n; i++) mpq_init(p->q[i]);
  mpq_init(p->r);
  mpz_init(p->z);
  mpz_init(p->z2);
  p->si = calloc(n, sizeof(long));
  p->d = calloc(n, sizeof(double));
  p->s = calloc(n, sizeof(char *));
  return p;
}

void gq_free(void *v) {
  gpool *p = v;
  for (size_t i = 0; i < p->n; i++) {
    mpq_clear(p->q[i]);
    free(p->s[i]);
  }
  free(p->q);
  mpq_clear(p->r);
  mpz_clear(p->z);
  mpz_clear(p->z2);
  free(p->si);
  free(p->d);
  free(p->s);
  free(p->buf);
  free(p);
}

void gq_set(void *v, size_t i, const uint8_t *nb, size_t nl, int neg, const uint8_t *db,
            size_t dl, int canon) {
  gpool *p = v;
  imp(mpq_numref(p->q[i]), nb, nl, neg);
  imp(mpq_denref(p->q[i]), db, dl, 0);
  if (canon) mpq_canonicalize(p->q[i]);
}

void gq_set_si(void *v, size_t i, long x) { ((gpool *)v)->si[i] = x; }
void gq_set_d(void *v, size_t i, double x) { ((gpool *)v)->d[i] = x; }
void gq_set_s(void *v, size_t i, const char *s) {
  gpool *p = v;
  free(p->s[i]);
  p->s[i] = strdup(s);
}
void gq_reserve_buf(void *v, size_t cap) {
  gpool *p = v;
  free(p->buf);
  p->buf = malloc(cap);
  p->bufcap = cap;
}

// which: 0 = item i, 1 = the result variable r, 2 = the integer z.
size_t gq_str(void *v, size_t i, int which, int base, char *buf, size_t cap) {
  gpool *p = v;
  char *s = which == 0 ? mpq_get_str(NULL, base, p->q[i])
          : which == 1 ? mpq_get_str(NULL, base, p->r)
                       : mpz_get_str(NULL, base, p->z);
  size_t l = put(buf, cap, s);
  void (*fr)(void *, size_t);
  mp_get_memory_functions(NULL, NULL, &fr);
  fr(s, strlen(s) + 1);
  return l;
}

#define CYCLE(body)                                                                       \
  for (size_t k = 0; k < count; k++) {                                                    \
    body;                                                                                 \
    if (++j == n) j = 0;                                                                  \
  }                                                                                       \
  break;

// Binary operations of A[j] and B[j]. `one` < SIZE_MAX runs once at index `one`.
int64_t gq_bin(int op, void *va, void *vb, size_t count, size_t one) {
  gpool *A = va, *B = vb;
  size_t n = A->n, j = A->cur;
  int64_t s = 0;
  if (one != SIZE_MAX) {
    j = one;
    count = 1;
  }
  switch (op) {
  case 0: CYCLE(mpq_add(A->r, A->q[j], B->q[j]))
  case 1: CYCLE(mpq_sub(A->r, A->q[j], B->q[j]))
  case 2: CYCLE(mpq_mul(A->r, A->q[j], B->q[j]))
  case 3: CYCLE(mpq_div(A->r, A->q[j], B->q[j]))
  case 4: CYCLE({ int c = mpq_cmp(A->q[j], B->q[j]); s += (c > 0) - (c < 0); })
  case 5: CYCLE(s += mpq_equal(A->q[j], B->q[j]))
  case 6: CYCLE({ mpq_set(A->r, A->q[j]); mpq_add(A->r, A->r, B->q[j]); })
  case 7: CYCLE({ mpq_set(A->r, A->q[j]); mpq_sub(A->r, A->r, B->q[j]); })
  case 8: CYCLE({ mpq_set(A->r, A->q[j]); mpq_mul(A->r, A->r, B->q[j]); })
  case 9: CYCLE({ mpq_set(A->r, A->q[j]); mpq_div(A->r, A->r, B->q[j]); })
  default: abort();
  }
  if (one == SIZE_MAX) A->cur = j;
  g_sink += (uint64_t)s;
  return s;
}

// Unary operations of A[j]; `e` is an exponent or a radix where the operation takes one.
int64_t gq_un(int op, void *va, size_t count, size_t one, long e) {
  gpool *A = va;
  size_t n = A->n, j = A->cur;
  int64_t s = 0;
  double d = 0;
  if (one != SIZE_MAX) {
    j = one;
    count = 1;
  }
  switch (op) {
  case 0: CYCLE(mpz_fdiv_q(A->z, mpq_numref(A->q[j]), mpq_denref(A->q[j])))
  case 1: CYCLE(mpz_cdiv_q(A->z, mpq_numref(A->q[j]), mpq_denref(A->q[j])))
  case 2: CYCLE(mpz_tdiv_q(A->z, mpq_numref(A->q[j]), mpq_denref(A->q[j])))
  case 3: CYCLE(mpq_abs(A->r, A->q[j]))
  case 4: CYCLE(mpq_neg(A->r, A->q[j]))
  case 5: CYCLE(mpq_inv(A->r, A->q[j]))
  case 6: CYCLE(mpq_set(A->r, A->q[j]))
  // Canonical form of a raw fraction, on a fresh copy.
  case 7: CYCLE({ mpq_set(A->r, A->q[j]); mpq_canonicalize(A->r); })
  case 8: CYCLE(d += mpq_get_d(A->q[j]))
  // A power of a rational: the powers of its parts (GMP has no mpq_pow).
  case 9: CYCLE({
    long ee = e < 0 ? -e : e;
    mpz_pow_ui(mpq_numref(A->r), mpq_numref(A->q[j]), (unsigned long)ee);
    mpz_pow_ui(mpq_denref(A->r), mpq_denref(A->q[j]), (unsigned long)ee);
    if (e < 0) mpq_inv(A->r, A->r);
  })
  case 10: CYCLE(s += mpq_sgn(A->q[j]))
  case 11: CYCLE(mpq_set_si(A->r, A->si[j], 1))
  case 12: CYCLE(mpq_set_d(A->r, A->d[j]))
  // A new mpq_t each time, as a new rational is.
  case 13: CYCLE({ mpq_t t; mpq_init(t); mpq_set_d(t, A->d[j]); s += mpq_sgn(t); mpq_clear(t); })
  case 14: CYCLE(s += (int64_t)strlen(mpq_get_str(A->buf, (int)e, A->q[j])))
  // Reading text: as written (14 no canon) and canonicalised.
  case 15: CYCLE(s += mpq_set_str(A->r, A->s[j], (int)e))
  case 16: CYCLE({ s += mpq_set_str(A->r, A->s[j], (int)e); mpq_canonicalize(A->r); })
  // The integer of an integral rational.
  case 17: CYCLE(mpz_set(A->z, mpq_numref(A->q[j])))
  default: abort();
  }
  if (one == SIZE_MAX) A->cur = j;
  g_sink += (uint64_t)s;
  g_dsink += d;
  if (one != SIZE_MAX && op == 8) {
    union { double d; int64_t i; } u = {d};
    return u.i;
  }
  return s;
}

// Sums and products of the pool's terms, each call the whole of it: mode 0 term by term
// (h += t), 1 in a binary-counter tree, 2 a product term by term, 3 a product in the tree.
static void gtree(gpool *A, int mul, mpq_t out) {
  size_t n = A->n;
  mpq_t st[64];
  int rk[64];
  int top = 0;
  for (size_t i = 0; i < n; i++) {
    mpq_init(st[top]);
    mpq_set(st[top], A->q[i]);
    rk[top++] = 0;
    while (top >= 2 && rk[top - 1] == rk[top - 2]) {
      if (mul) mpq_mul(st[top - 2], st[top - 2], st[top - 1]);
      else mpq_add(st[top - 2], st[top - 2], st[top - 1]);
      rk[top - 2]++;
      mpq_clear(st[--top]);
    }
  }
  mpq_set_ui(out, mul ? 1 : 0, 1);
  while (top > 0) {
    if (mul) mpq_mul(out, out, st[top - 1]);
    else mpq_add(out, out, st[top - 1]);
    mpq_clear(st[--top]);
  }
}

void gq_sum(void *va, int mode, size_t count) {
  gpool *A = va;
  for (size_t k = 0; k < count; k++) {
    if (mode == 0 || mode == 2) {
      mpq_set_ui(A->r, mode == 0 ? 0 : 1, 1);
      for (size_t i = 0; i < A->n; i++) {
        if (mode == 0) mpq_add(A->r, A->r, A->q[i]);
        else mpq_mul(A->r, A->r, A->q[i]);
      }
    } else {
      gtree(A, mode == 3, A->r);
    }
  }
  g_sink += mpq_sgn(A->r);
}

// ---------------------------------------------------------------------------------------
// FLINT
// ---------------------------------------------------------------------------------------
#ifdef HAS_FLINT
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_vec.h>
#include <flint/fmpq.h>
#include <flint/fmpq_vec.h>
#include <flint/fmpz_mat.h>
#include <flint/fmpq_mat.h>
#include <flint/bernoulli.h>
#include <flint/arith.h>

typedef struct {
  size_t n;
  fmpq *q;
  fmpq_t r;
  fmpz_t z, z2;
  slong *si;
  double *d;
  char **s;
  char *buf;
  // Continued fractions: the terms of each item.
  fmpz **terms;
  slong *nterms;
  size_t cur;
} fpool;

void *fq_new(size_t n) {
  fpool *p = calloc(1, sizeof *p);
  p->n = n;
  p->q = malloc(n * sizeof(fmpq));
  for (size_t i = 0; i < n; i++) fmpq_init(p->q + i);
  fmpq_init(p->r);
  fmpz_init(p->z);
  fmpz_init(p->z2);
  p->si = calloc(n, sizeof(slong));
  p->d = calloc(n, sizeof(double));
  p->s = calloc(n, sizeof(char *));
  p->terms = calloc(n, sizeof(fmpz *));
  p->nterms = calloc(n, sizeof(slong));
  return p;
}

void fq_free(void *v) {
  fpool *p = v;
  for (size_t i = 0; i < p->n; i++) {
    fmpq_clear(p->q + i);
    free(p->s[i]);
    if (p->terms[i]) _fmpz_vec_clear(p->terms[i], p->nterms[i]);
  }
  free(p->q);
  fmpq_clear(p->r);
  fmpz_clear(p->z);
  fmpz_clear(p->z2);
  free(p->si);
  free(p->d);
  free(p->s);
  free(p->terms);
  free(p->nterms);
  free(p->buf);
  free(p);
}

void fq_set(void *v, size_t i, const uint8_t *nb, size_t nl, int neg, const uint8_t *db,
            size_t dl, int canon) {
  fpool *p = v;
  mpz_t t;
  mpz_init(t);
  imp(t, nb, nl, neg);
  fmpz_set_mpz(fmpq_numref(p->q + i), t);
  imp(t, db, dl, 0);
  fmpz_set_mpz(fmpq_denref(p->q + i), t);
  mpz_clear(t);
  if (canon) fmpq_canonicalise(p->q + i);
}

void fq_set_si(void *v, size_t i, long x) { ((fpool *)v)->si[i] = x; }
void fq_set_d(void *v, size_t i, double x) { ((fpool *)v)->d[i] = x; }
void fq_set_s(void *v, size_t i, const char *s) {
  fpool *p = v;
  free(p->s[i]);
  p->s[i] = strdup(s);
}
void fq_reserve_buf(void *v, size_t cap) {
  fpool *p = v;
  free(p->buf);
  p->buf = malloc(cap);
}

// Stores the continued fraction terms of item i (from FLINT's own expansion).
void fq_prepare_cfrac(void *v, size_t i) {
  fpool *p = v;
  slong b = fmpq_cfrac_bound(p->q + i);
  fmpz *c = _fmpz_vec_init(b);
  fmpq_t rem;
  fmpq_init(rem);
  slong m = fmpq_get_cfrac(c, rem, p->q + i, b);
  fmpq_clear(rem);
  if (p->terms[i]) _fmpz_vec_clear(p->terms[i], p->nterms[i]);
  p->terms[i] = c;
  p->nterms[i] = b;
  p->si[i] = m;
}

size_t fq_str(void *v, size_t i, int which, int base, char *buf, size_t cap) {
  fpool *p = v;
  char *s = which == 0 ? fmpq_get_str(NULL, base, p->q + i)
          : which == 1 ? fmpq_get_str(NULL, base, p->r)
                       : fmpz_get_str(NULL, base, p->z);
  size_t l = put(buf, cap, s);
  flint_free(s);
  return l;
}

int64_t fq_bin(int op, void *va, void *vb, size_t count, size_t one) {
  fpool *A = va, *B = vb;
  size_t n = A->n, j = A->cur;
  int64_t s = 0;
  if (one != SIZE_MAX) {
    j = one;
    count = 1;
  }
  switch (op) {
  case 0: CYCLE(fmpq_add(A->r, A->q + j, B->q + j))
  case 1: CYCLE(fmpq_sub(A->r, A->q + j, B->q + j))
  case 2: CYCLE(fmpq_mul(A->r, A->q + j, B->q + j))
  case 3: CYCLE(fmpq_div(A->r, A->q + j, B->q + j))
  case 4: CYCLE({ int c = fmpq_cmp(A->q + j, B->q + j); s += (c > 0) - (c < 0); })
  case 5: CYCLE(s += fmpq_equal(A->q + j, B->q + j))
  case 6: CYCLE({ fmpq_set(A->r, A->q + j); fmpq_add(A->r, A->r, B->q + j); })
  case 7: CYCLE({ fmpq_set(A->r, A->q + j); fmpq_sub(A->r, A->r, B->q + j); })
  case 8: CYCLE({ fmpq_set(A->r, A->q + j); fmpq_mul(A->r, A->r, B->q + j); })
  case 9: CYCLE({ fmpq_set(A->r, A->q + j); fmpq_div(A->r, A->r, B->q + j); })
  // With an integer: B[j]'s numerator.
  case 10: CYCLE(fmpq_add_fmpz(A->r, A->q + j, fmpq_numref(B->q + j)))
  case 11: CYCLE(fmpq_mul_fmpz(A->r, A->q + j, fmpq_numref(B->q + j)))
  case 12: CYCLE(fmpq_simplest_between(A->r, A->q + j, B->q + j))
  // Rational reconstruction of A[j]'s numerator modulo B[j]'s numerator.
  case 13: CYCLE(s += fmpq_reconstruct_fmpz(A->r, fmpq_numref(A->q + j), fmpq_numref(B->q + j)))
  // ... with the bounds N = B[j]'s denominator... (see fq_recon2).
  default: abort();
  }
  if (one == SIZE_MAX) A->cur = j;
  g_sink += (uint64_t)s;
  return s;
}

// Reconstruction with bounds: a = A[j].num, m = B[j].num, N = C[j].num, D = C[j].den.
int64_t fq_recon2(void *va, void *vb, void *vc, size_t count, size_t one) {
  fpool *A = va, *B = vb, *C = vc;
  size_t n = A->n, j = A->cur;
  int64_t s = 0;
  if (one != SIZE_MAX) {
    j = one;
    count = 1;
  }
  for (size_t k = 0; k < count; k++) {
    s += fmpq_reconstruct_fmpz_2(A->r, fmpq_numref(A->q + j), fmpq_numref(B->q + j),
                                 fmpq_numref(C->q + j), fmpq_denref(C->q + j));
    if (++j == n) j = 0;
  }
  if (one == SIZE_MAX) A->cur = j;
  g_sink += (uint64_t)s;
  return s;
}

int64_t fq_un(int op, void *va, size_t count, size_t one, long e) {
  fpool *A = va;
  size_t n = A->n, j = A->cur;
  int64_t s = 0;
  double d = 0;
  if (one != SIZE_MAX) {
    j = one;
    count = 1;
  }
  switch (op) {
  case 0: CYCLE(fmpz_fdiv_q(A->z, fmpq_numref(A->q + j), fmpq_denref(A->q + j)))
  case 1: CYCLE(fmpz_cdiv_q(A->z, fmpq_numref(A->q + j), fmpq_denref(A->q + j)))
  case 2: CYCLE(fmpz_tdiv_q(A->z, fmpq_numref(A->q + j), fmpq_denref(A->q + j)))
  case 3: CYCLE(fmpq_abs(A->r, A->q + j))
  case 4: CYCLE(fmpq_neg(A->r, A->q + j))
  case 5: CYCLE(fmpq_inv(A->r, A->q + j))
  case 6: CYCLE(fmpq_set(A->r, A->q + j))
  case 7: CYCLE({ fmpq_set(A->r, A->q + j); fmpq_canonicalise(A->r); })
  case 8: CYCLE(d += fmpq_get_d(A->q + j))
  case 9: CYCLE(fmpq_pow_si(A->r, A->q + j, e))
  case 10: CYCLE(s += fmpq_sgn(A->q + j))
  case 11: CYCLE(fmpq_set_si(A->r, A->si[j], 1))
  // Round to nearest (FLINT's ties: away from zero? see the notes), into z.
  case 12: CYCLE(fmpz_ndiv_qr(A->z, A->z2, fmpq_numref(A->q + j), fmpq_denref(A->q + j)))
  case 14: CYCLE(s += (int64_t)strlen(fmpq_get_str(A->buf, (int)e, A->q + j)))
  case 16: CYCLE({ s += fmpq_set_str(A->r, A->s[j], (int)e); fmpq_canonicalise(A->r); })
  // With a primitive integer A->si[j].
  case 20: CYCLE(fmpq_add_si(A->r, A->q + j, A->si[j]))
  case 21: CYCLE(fmpq_sub_si(A->r, A->q + j, A->si[j]))
  case 22: CYCLE(fmpq_mul_si(A->r, A->q + j, A->si[j]))
  // In place on a fresh copy.
  case 23: CYCLE({ fmpq_set(A->r, A->q + j); fmpq_add_si(A->r, A->r, A->si[j]); })
  case 24: CYCLE({ int c = fmpq_cmp_si(A->q + j, A->si[j]); s += (c > 0) - (c < 0); })
  // Continued fraction with its bound and the vector's allocation, as a caller does it.
  case 30: CYCLE({
    slong b = fmpq_cfrac_bound(A->q + j);
    fmpz *c = _fmpz_vec_init(b);
    s += fmpq_get_cfrac(c, A->r, A->q + j, b);
    _fmpz_vec_clear(c, b);
  })
  case 31: CYCLE(fmpq_set_cfrac(A->r, A->terms[j], A->si[j]))
  case 32: {
    // fmpq_get_cfrac into a vector that exists, grown to the largest bound once.
    static fmpz *cbuf = NULL;
    static slong ccap = 0;
    CYCLE({
      slong b = fmpq_cfrac_bound(A->q + j);
      if (b > ccap) {
        cbuf = realloc(cbuf, b * sizeof(fmpz));
        for (slong i = ccap; i < b; i++) fmpz_init(cbuf + i);
        ccap = b;
      }
      s += fmpq_get_cfrac(cbuf, A->r, A->q + j, b);
    })
  }
  case 40: CYCLE(fmpz_set(A->z, fmpq_numref(A->q + j)))
  default: abort();
  }
  if (one == SIZE_MAX) A->cur = j;
  g_sink += (uint64_t)s;
  g_dsink += d;
  if (one != SIZE_MAX && op == 8) {
    union { double d; int64_t i; } u = {d};
    return u.i;
  }
  return s;
}

static void ftree(fpool *A, int mul, fmpq_t out) {
  size_t n = A->n;
  fmpq st[64];
  int rk[64];
  int top = 0;
  for (size_t i = 0; i < n; i++) {
    fmpq_init(st + top);
    fmpq_set(st + top, A->q + i);
    rk[top++] = 0;
    while (top >= 2 && rk[top - 1] == rk[top - 2]) {
      if (mul) fmpq_mul(st + top - 2, st + top - 2, st + top - 1);
      else fmpq_add(st + top - 2, st + top - 2, st + top - 1);
      rk[top - 2]++;
      fmpq_clear(st + --top);
    }
  }
  fmpq_set_si(out, mul ? 1 : 0, 1);
  while (top > 0) {
    if (mul) fmpq_mul(out, out, st + top - 1);
    else fmpq_add(out, out, st + top - 1);
    fmpq_clear(st + --top);
  }
}

void fq_sum(void *va, int mode, size_t count) {
  fpool *A = va;
  for (size_t k = 0; k < count; k++) {
    if (mode == 0 || mode == 2) {
      fmpq_set_si(A->r, mode == 0 ? 0 : 1, 1);
      for (size_t i = 0; i < A->n; i++) {
        if (mode == 0) fmpq_add(A->r, A->r, A->q + i);
        else fmpq_mul(A->r, A->r, A->q + i);
      }
    } else {
      ftree(A, mode == 3, A->r);
    }
  }
  g_sink += fmpq_sgn(A->r);
}

// Bernoulli and harmonic numbers, the indices n, n + step, ... (four in turn); the result
// in r. kind: 0 _bernoulli_fmpq_ui, 1 bernoulli_fmpq_ui (cached), 2 fmpq_harmonic_ui,
// 3 _bernoulli_fmpq_ui_zeta (the formula alone).
void fl_seq(void *va, int kind, unsigned long n0, unsigned long step, size_t count) {
  fpool *A = va;
  unsigned long n = n0 + step * (A->cur % 4);
  for (size_t k = 0; k < count; k++) {
    switch (kind) {
    case 0: _bernoulli_fmpq_ui(fmpq_numref(A->r), fmpq_denref(A->r), n); break;
    case 1: bernoulli_fmpq_ui(A->r, n); break;
    case 2: fmpq_harmonic_ui(A->r, n); break;
    case 3: _bernoulli_fmpq_ui_zeta(fmpq_numref(A->r), fmpq_denref(A->r), n); break;
    }
    n += step;
    if (n >= n0 + 4 * step) n = n0;
  }
  A->cur = (n - n0) / step;
  g_sink += fmpq_sgn(A->r);
}

// The vector B_0..B_n: kind 0 bernoulli_fmpq_vec_no_cache, 1 arith_bernoulli_number_vec.
// Item i of the pool (if it has n + 1 items) receives B_i, for the check.
void fl_bvec(void *va, int kind, unsigned long n, size_t count) {
  fpool *A = va;
  fmpq *v = _fmpq_vec_init(n + 1);
  for (size_t k = 0; k < count; k++) {
    if (kind == 0) bernoulli_fmpq_vec_no_cache(v, 0, n + 1);
    else arith_bernoulli_number_vec(v, n + 1);
  }
  for (size_t i = 0; i < A->n && i <= n; i++) fmpq_set(A->q + i, v + i);
  _fmpq_vec_clear(v, n + 1);
}

// Matrices: a pool of k pairs (A, B) of n x n and n x m rational matrices; ops det, solve,
// inv, mul; and integer matrices for solve / inverse over Q.
typedef struct {
  size_t k, n, m;
  fmpq_mat_struct *a, *b;
  fmpz_mat_struct *az, *bz;
  fmpq_mat_t x;
  fmpz_mat_t xz;
  fmpq_t d;
  fmpz_t den;
  size_t cur;
} fmatpool;

void *fm_new(size_t k, size_t n, size_t m) {
  fmatpool *p = calloc(1, sizeof *p);
  p->k = k;
  p->n = n;
  p->m = m;
  p->a = malloc(k * sizeof(fmpq_mat_struct));
  p->b = malloc(k * sizeof(fmpq_mat_struct));
  p->az = malloc(k * sizeof(fmpz_mat_struct));
  p->bz = malloc(k * sizeof(fmpz_mat_struct));
  for (size_t i = 0; i < k; i++) {
    fmpq_mat_init(p->a + i, n, n);
    fmpq_mat_init(p->b + i, n, m);
    fmpz_mat_init(p->az + i, n, n);
    fmpz_mat_init(p->bz + i, n, m);
  }
  fmpq_mat_init(p->x, n, m > n ? m : n);
  fmpz_mat_init(p->xz, n, n);
  fmpq_init(p->d);
  fmpz_init(p->den);
  return p;
}

void fm_free(void *v) {
  fmatpool *p = v;
  for (size_t i = 0; i < p->k; i++) {
    fmpq_mat_clear(p->a + i);
    fmpq_mat_clear(p->b + i);
    fmpz_mat_clear(p->az + i);
    fmpz_mat_clear(p->bz + i);
  }
  free(p->a);
  free(p->b);
  free(p->az);
  free(p->bz);
  fmpq_mat_clear(p->x);
  fmpz_mat_clear(p->xz);
  fmpq_clear(p->d);
  fmpz_clear(p->den);
  free(p);
}

// which 0: A, 1: B (rational entries); 2: Az, 3: Bz (the numerators).
void fm_set(void *v, int which, size_t t, size_t i, size_t j, const uint8_t *nb, size_t nl,
            int neg, const uint8_t *db, size_t dl) {
  fmatpool *p = v;
  mpz_t z;
  mpz_init(z);
  imp(z, nb, nl, neg);
  if (which == 0 || which == 1) {
    fmpq *e = fmpq_mat_entry(which == 0 ? p->a + t : p->b + t, i, j);
    fmpz_set_mpz(fmpq_numref(e), z);
    imp(z, db, dl, 0);
    fmpz_set_mpz(fmpq_denref(e), z);
    fmpq_canonicalise(e);
  } else {
    fmpz_set_mpz(fmpz_mat_entry(which == 2 ? p->az + t : p->bz + t, i, j), z);
  }
  mpz_clear(z);
}

// op 0 det, 1 solve, 2 inv, 3 mul (A * A), 4 solve of integers, 5 inverse of integers over Q.
int64_t fm_run(void *v, int op, size_t count, size_t one) {
  fmatpool *p = v;
  size_t j = p->cur;
  int64_t s = 0;
  if (one != SIZE_MAX) {
    j = one;
    count = 1;
  }
  for (size_t c = 0; c < count; c++) {
    switch (op) {
    case 0: fmpq_mat_det(p->d, p->a + j); s += fmpq_sgn(p->d); break;
    case 1: {
      fmpq_mat_t x;
      fmpq_mat_window_init(x, p->x, 0, 0, p->n, p->m);
      s += fmpq_mat_solve(x, p->a + j, p->b + j);
      fmpq_mat_window_clear(x);
      break;
    }
    case 2: {
      fmpq_mat_t x;
      fmpq_mat_window_init(x, p->x, 0, 0, p->n, p->n);
      s += fmpq_mat_inv(x, p->a + j);
      fmpq_mat_window_clear(x);
      break;
    }
    case 3: {
      fmpq_mat_t x;
      fmpq_mat_window_init(x, p->x, 0, 0, p->n, p->n);
      fmpq_mat_mul(x, p->a + j, p->a + j);
      fmpq_mat_window_clear(x);
      break;
    }
    case 4: {
      fmpq_mat_t x;
      fmpq_mat_window_init(x, p->x, 0, 0, p->n, p->m);
      s += fmpq_mat_solve_fmpz_mat(x, p->az + j, p->bz + j);
      fmpq_mat_window_clear(x);
      break;
    }
    case 5: {
      fmpq_mat_t x;
      fmpq_mat_window_init(x, p->x, 0, 0, p->n, p->n);
      s += fmpz_mat_inv(p->xz, p->den, p->az + j);
      fmpq_mat_set_fmpz_mat_div_fmpz(x, p->xz, p->den);
      fmpq_mat_window_clear(x);
      break;
    }
    // The determinant of the integer matrix.
    case 7: fmpz_mat_det(p->den, p->az + j); s += fmpz_sgn(p->den); break;
    // The reduced row echelon form (its rank returned): rank and null space come from it.
    case 6: {
      fmpq_mat_t x;
      fmpq_mat_window_init(x, p->x, 0, 0, p->n, p->n);
      s += fmpq_mat_rref(x, p->a + j);
      fmpq_mat_window_clear(x);
      break;
    }
    }
    if (++j == p->k) j = 0;
  }
  if (one == SIZE_MAX) p->cur = j;
  g_sink += (uint64_t)s;
  return s;
}

// The result's entry (i, j) (op 0: the determinant) as text, for the check.
size_t fm_str(void *v, int op, size_t i, size_t j, char *buf, size_t cap) {
  fmatpool *p = v;
  char *s = op == 0 ? fmpq_get_str(NULL, 16, p->d)
          : op == 7 ? fmpz_get_str(NULL, 16, p->den)
                    : fmpq_get_str(NULL, 16, fmpq_mat_entry(p->x, i, j));
  size_t l = put(buf, cap, s);
  flint_free(s);
  return l;
}
#endif
