// PARI/GP references for the rational scoreboard: continued fractions, best approximations,
// rational reconstruction, Bernoulli and harmonic numbers, matrices of rationals. Inputs are
// clones on PARI's heap; every call runs on the stack and the stack is reset after it.
#include <pari/pari.h>
#include <stdint.h>
#include <string.h>

volatile uint64_t p_sink;
static int inited;

/* PARI's version, (major << 16) + (minor << 8) + patch. */
long pa_version_code(void) { return PARI_VERSION_CODE; }

void pa_init(void) {
  if (inited) return;
  // No signal handlers, no GMP memory functions of PARI's.
  pari_init_opts(1UL << 30, 1UL << 20, INIT_DFTm | INIT_noINTGMPm | INIT_noIMTm);
  // The multithread engine, started by hand with one thread: single-threaded as the others.
  pari_mt_nbthreads = 1;
  pari_mt_init();
  inited = 1;
}

static GEN mkint(const uint8_t *b, size_t len, int neg) {
  while (len > 0 && b[len - 1] == 0) len--;
  if (len == 0) return gen_0;
  size_t L = (len + 7) / 8;
  GEN x = cgeti(L + 2);
  x[1] = evalsigne(neg ? -1 : 1) | evallgefint(L + 2);
  for (size_t l = 0; l < L; l++) {
    ulong w = 0;
    for (size_t t = 0; t < 8 && 8 * l + t < len; t++) w |= (ulong)b[8 * l + t] << (8 * t);
    *int_W(x, l) = w;
  }
  return x;
}

static GEN mkq(const uint8_t *nb, size_t nl, int neg, const uint8_t *db, size_t dl) {
  GEN n = mkint(nb, nl, neg), d = mkint(db, dl, 0);
  if (equali1(d)) return n;
  return gdiv(n, d);
}

typedef struct {
  size_t n;
  GEN *x;
  GEN last;
  size_t cur;
} ppool;

void *pp_new(size_t n) {
  ppool *p = pari_calloc(sizeof *p);
  p->n = n;
  p->x = pari_calloc(n * sizeof(GEN));
  return p;
}

void pp_free(void *v) {
  ppool *p = v;
  for (size_t i = 0; i < p->n; i++)
    if (p->x[i]) gunclone(p->x[i]);
  if (p->last) gunclone(p->last);
  pari_free(p->x);
  pari_free(p);
}

void pp_set(void *v, size_t i, const uint8_t *nb, size_t nl, int neg, const uint8_t *db,
            size_t dl) {
  ppool *p = v;
  pari_sp av = avma;
  if (p->x[i]) gunclone(p->x[i]);
  p->x[i] = gclone(mkq(nb, nl, neg, db, dl));
  set_avma(av);
}

static GEN pp_one(int op, ppool *A, ppool *B, size_t j) {
  switch (op) {
  case 0: return contfrac0(A->x[j], NULL, 0);
  case 1: return bestappr(A->x[j], B->x[j]);
  case 2: return bestappr(gmodulo(A->x[j], B->x[j]), NULL);
  }
  return NULL;
}

int64_t pp_run(int op, void *va, void *vb, size_t count, size_t one) {
  ppool *A = va, *B = vb;
  size_t n = A->n, j = A->cur;
  int64_t s = 0;
  if (one != SIZE_MAX) {
    j = one;
    count = 1;
  }
  pari_sp av = avma;
  for (size_t k = 0; k < count; k++) {
    GEN r = pp_one(op, A, B, j);
    s += lg(r);
    if (one != SIZE_MAX) {
      if (A->last) gunclone(A->last);
      A->last = gclone(r);
    }
    set_avma(av);
    if (++j == n) j = 0;
  }
  if (one == SIZE_MAX) A->cur = j;
  p_sink += (uint64_t)s;
  return s;
}

// kind 0: bernfrac(n), 1: harmonic(n); four indices n, n + step, ... in turn.
void pp_seq(void *va, int kind, unsigned long n0, unsigned long step, size_t count) {
  ppool *A = va;
  unsigned long n = n0 + step * (A->cur % 4);
  pari_sp av = avma;
  for (size_t k = 0; k < count; k++) {
    GEN r = kind == 0 ? bernfrac((long)n) : harmonic(n);
    p_sink += lg(r);
    if (k + 1 == count) {
      if (A->last) gunclone(A->last);
      A->last = gclone(r);
    }
    set_avma(av);
    n += step;
    if (n >= n0 + 4 * step) n = n0;
  }
  A->cur = (n - n0) / step;
}

// bernvec(m): B_0, B_2, ..., B_2m.
void pp_bvec(void *va, long m, size_t count) {
  ppool *A = va;
  pari_sp av = avma;
  for (size_t k = 0; k < count; k++) {
    GEN r = bernvec(m);
    p_sink += lg(r);
    if (k + 1 == count) {
      if (A->last) gunclone(A->last);
      A->last = gclone(r);
    }
    set_avma(av);
  }
}

// The last result (or an entry of a vector result, `idx` >= 0, 1-based) as decimal text.
size_t pp_str(void *v, long idx, char *buf, size_t cap) {
  ppool *p = v;
  GEN x = p->last;
  if (idx >= 0) x = gel(x, idx);
  char *s = GENtostr(x);
  size_t l = strlen(s);
  if (l + 1 > cap) l = cap - 1;
  memcpy(buf, s, l);
  buf[l] = 0;
  pari_free(s);
  return l;
}

// Matrices: k pairs (A, B), n x n and n x m.
typedef struct {
  size_t k, n, m;
  GEN *a, *b;
  GEN last;
  size_t cur;
} pmpool;

void *pm_new(size_t k, size_t n, size_t m) {
  pmpool *p = pari_calloc(sizeof *p);
  p->k = k;
  p->n = n;
  p->m = m;
  p->a = pari_calloc(k * sizeof(GEN));
  p->b = pari_calloc(k * sizeof(GEN));
  return p;
}

void pm_free(void *v) {
  pmpool *p = v;
  for (size_t i = 0; i < p->k; i++) {
    if (p->a[i]) gunclone(p->a[i]);
    if (p->b[i]) gunclone(p->b[i]);
  }
  if (p->last) gunclone(p->last);
  pari_free(p->a);
  pari_free(p->b);
  pari_free(p);
}

// Sets entry (i, j) of A (which 0) or B (which 1) of pair t. The matrices are built on the
// stack as the entries come and cloned once the last one is set (`done`).
static GEN pm_build_a, pm_build_b;
static pari_sp pm_av;

void pm_begin(void *v) {
  pmpool *p = v;
  pm_av = avma;
  pm_build_a = zeromatcopy(p->n, p->n);
  pm_build_b = zeromatcopy(p->n, p->m);
}

void pm_entry(int which, size_t i, size_t j, const uint8_t *nb, size_t nl, int neg,
              const uint8_t *db, size_t dl) {
  GEN M = which == 0 ? pm_build_a : pm_build_b;
  gcoeff(M, i + 1, j + 1) = mkq(nb, nl, neg, db, dl);
}

void pm_end(void *v, size_t t) {
  pmpool *p = v;
  if (p->a[t]) gunclone(p->a[t]);
  if (p->b[t]) gunclone(p->b[t]);
  p->a[t] = gclone(pm_build_a);
  p->b[t] = gclone(pm_build_b);
  set_avma(pm_av);
}

// op 0 det, 1 RgM_solve, 2 RgM_inv, 3 A * A.
int64_t pm_run(void *v, int op, size_t count, size_t one) {
  pmpool *p = v;
  size_t j = p->cur;
  int64_t s = 0;
  if (one != SIZE_MAX) {
    j = one;
    count = 1;
  }
  pari_sp av = avma;
  for (size_t c = 0; c < count; c++) {
    GEN r;
    switch (op) {
    case 0: r = det(p->a[j]); break;
    case 1: r = RgM_solve(p->a[j], p->b[j]); break;
    case 2: r = RgM_inv(p->a[j]); break;
    case 4: r = stoi(rank(p->a[j])); break;
    case 5: r = ker(p->a[j]); break;
    default: r = gmul(p->a[j], p->a[j]); break;
    }
    s += r ? lg(r) : 0;
    if (one != SIZE_MAX) {
      if (p->last) gunclone(p->last);
      p->last = gclone(r);
    }
    set_avma(av);
    if (++j == p->k) j = 0;
  }
  if (one == SIZE_MAX) p->cur = j;
  p_sink += (uint64_t)s;
  return s;
}

size_t pm_str(void *v, int op, size_t i, size_t j, char *buf, size_t cap) {
  pmpool *p = v;
  GEN x = op == 0 ? p->last : gcoeff(p->last, i + 1, j + 1);
  char *s = GENtostr(x);
  size_t l = strlen(s);
  if (l + 1 > cap) l = cap - 1;
  memcpy(buf, s, l);
  buf[l] = 0;
  pari_free(s);
  return l;
}
