// PARI/GP references (the library, one thread). Inputs are cloned onto the heap once; each call
// runs on PARI's stack, which is reset after it.
#include <pari/pari.h>
#include <stdlib.h>
#include "common.h"

typedef struct {
  GEN *z;
  size_t n;
} ppool;

void pari_setup(void) {
  pari_init_opts(1UL << 28, 1UL << 20, INIT_DFTm);
  paristack_setsize(1UL << 28, 1UL << 33);
  setdefault("nbthreads", "1", d_SILENT);
}

static GEN from_limbs(const uint64_t *l, size_t len) {
  while (len && !l[len - 1]) len--;
  if (!len) return gen_0;
  GEN z = cgetipos(len + 2);
  for (size_t i = 0; i < len; i++) *int_W(z, i) = l[i];
  return z;
}

void *pari_pool(const uint64_t *limbs, const uint64_t *off, size_t n) {
  ppool *p = malloc(sizeof(ppool));
  p->z = malloc(sizeof(GEN) * (n ? n : 1));
  p->n = n;
  pari_sp av = avma;
  for (size_t i = 0; i < n; i++) p->z[i] = gclone(from_limbs(limbs + off[i], off[i + 1] - off[i]));
  set_avma(av);
  return p;
}

static inline uint64_t lo(GEN x) { return signe(x) ? (uint64_t)*int_LSW(x) : 0; }

static uint64_t lo_min(GEN r, GEN p) {
  GEN t = subii(p, r);
  return cmpii(t, r) < 0 ? lo(t) : lo(r);
}

static uint64_t fac_sum(GEN F) {
  GEN P = gel(F, 1), E = gel(F, 2);
  uint64_t s = 0;
  for (long i = 1; i < lg(P); i++) s += lo(gel(P, i)) * (uint64_t)itos(gel(E, i));
  return s;
}

enum {
  P_ISPSEUDO = 1,   // ispseudoprime(n, 0)
  P_NEXT = 2,       // nextprime
  P_PREC = 3,       // precprime
  P_FACTOR = 4,     // Z_factor
  P_MPQS = 5,       // factorint(n, 14): MPQS alone
  P_RHO = 6,        // Z_pollardbrent(n, arg, seed), seeds 1, 2, ... until found
  P_ISPOWER = 7,    // ispower(n, NULL, &r): the largest k
  P_KRON = 8,       // kronecker(a, b)
  P_FPSQRT = 9,     // Fp_sqrt(a, p): min(r, p - r)
  P_CORN = 10,      // qfbcornacchia(d = a, p = b): x + y
  P_CHINESE = 11,   // chinese(Mod(r1, m1), Mod(r2, m2)): pairs (2i, 2i + 1) of a = residues, b = moduli
  P_ZVCHINESE = 12, // ZV_chinese of groups of arg
  P_ZNORDER = 13,   // znorder(Mod(a, m)): a, b = m
  P_PRIMROOT = 14,  // znprimroot(m)
  P_ZNLOG = 15,     // znlog(h = a, Mod(g, p)): b = (g, p) in pairs
  P_FPLOG = 16,     // Fp_log(h = a, g, ord, p): b = (g, ord, p) in threes
  P_PHI = 17,       // eulerphi(n), factoring it
  P_MU = 18,        // moebius(n)
  P_SQFREE = 19,    // Z_issquarefree(n)
  P_NUMBPART = 20,  // numbpart(n)
  P_PRIMEPI = 21,   // primepi(n)
  P_PRIME = 22,     // prime(n)
  P_ZNSQRT = 23,    // Zn_sqrt(a, n): b = n, factored by PARI
  P_SIGMA1 = 24,    // sumdiv(n), factoring it
  P_NUMDIV = 25,    // numdiv(n)
};

uint64_t pari_op(int op, void *ha, void *hb, size_t i0, size_t k, uint64_t arg) {
  ppool *a = ha, *b = hb;
  size_t n = a->n;
  uint64_t s = 0;
  pari_sp av = avma;
  for (size_t j = 0; j < k; j++) {
    size_t i = item(i0, j, n);
    GEN x = a->z[i];
    switch (op) {
      case P_ISPSEUDO: s += ispseudoprime(x, 0) != 0; break;
      case P_NEXT: s += lo(nextprime(x)); break;
      case P_PREC: s += lo(precprime(x)); break;
      case P_FACTOR: s += fac_sum(Z_factor(x)); break;
      case P_MPQS: s += fac_sum(factorint(x, 14)); break;
      case P_RHO: {
        GEN f = NULL;
        for (long seed = 1; !f; seed++) f = Z_pollardbrent(x, (long)arg, seed);
        // A vector of factors: the smallest of them, which for a semiprime is the smaller prime.
        GEN m = NULL;
        for (long t = 1; t < lg(f); t++)
          if (typ(gel(f, t)) == t_INT && (!m || cmpii(gel(f, t), m) < 0)) m = gel(f, t);
        s += m ? lo(m) : 0;
        break;
      }
      case P_ISPOWER: {
        GEN r;
        s += (uint64_t)ispower(x, NULL, &r);
        break;
      }
      case P_KRON: s += (uint64_t)(kronecker(x, b->z[i]) + 1); break;
      case P_FPSQRT: {
        GEN r = Fp_sqrt(x, b->z[i]);
        if (r) s += lo_min(r, b->z[i]);
        break;
      }
      case P_ZNSQRT: {
        GEN r = Zn_sqrt(x, b->z[i]);
        if (r) s += 1;
        break;
      }
      case P_CORN: {
        GEN r = qfbcornacchia(x, b->z[i]);
        if (lg(r) > 2) s += lo(gel(r, 1)) + lo(gel(r, 2));
        break;
      }
      case P_CHINESE: {
        size_t t = item(i0, j, n / 2) * 2;
        GEN r = chinese(mkintmod(a->z[t], b->z[t]), mkintmod(a->z[t + 1], b->z[t + 1]));
        s += lo(gel(r, 2));
        break;
      }
      case P_ZVCHINESE: {
        size_t g = arg, t = item(i0, j, n / g) * g;
        GEN A = cgetg(g + 1, t_VEC), P = cgetg(g + 1, t_VEC);
        for (size_t u = 0; u < g; u++) {
          gel(A, u + 1) = a->z[t + u];
          gel(P, u + 1) = b->z[t + u];
        }
        s += lo(ZV_chinese(A, P, NULL));
        break;
      }
      case P_ZNORDER: s += lo(znorder(mkintmod(x, b->z[i]), NULL)); break;
      case P_PRIMROOT: s += lo(gel(znprimroot(x), 2)); break;
      case P_ZNLOG: {
        size_t t = i * 2;
        GEN r = znlog(x, mkintmod(b->z[t], b->z[t + 1]), NULL);
        if (typ(r) == t_INT) s += lo(r);
        break;
      }
      case P_FPLOG: {
        size_t t = i * 3;
        s += lo(Fp_log(x, b->z[t], b->z[t + 1], b->z[t + 2]));
        break;
      }
      case P_PHI: s += lo(eulerphi(x)); break;
      case P_MU: s += (uint64_t)(moebius(x) + 1); break;
      case P_SQFREE: s += Z_issquarefree(x) != 0; break;
      case P_SIGMA1: s += lo(sumdiv(x)); break;
      case P_NUMDIV: s += lo(numdiv(x)); break;
      case P_NUMBPART: s += lo(numbpart(x)); break;
      case P_PRIMEPI: s += lo(primepi(x)); break;
      case P_PRIME: s += lo(prime(itos(x))); break;
    }
    set_avma(av);
  }
  return s;
}

// Prepared factorisations, as factorisation matrices: group g is primes[goff[g] .. goff[g + 1]]
// with the exponents at the same places.
void *pari_facpool(void *hp, const uint32_t *exps, const uint64_t *goff, size_t ng) {
  ppool *p = hp, *q = malloc(sizeof(ppool));
  q->z = malloc(sizeof(GEN) * ng);
  q->n = ng;
  pari_sp av = avma;
  for (size_t g = 0; g < ng; g++) {
    long m = (long)(goff[g + 1] - goff[g]);
    GEN P = cgetg(m + 1, t_COL), E = cgetg(m + 1, t_COL);
    for (long t = 0; t < m; t++) {
      gel(P, t + 1) = p->z[goff[g] + t];
      gel(E, t + 1) = utoipos(exps[goff[g] + t]);
    }
    // Sorted, as PARI's factorisations are.
    GEN F = mkmat2(P, E);
    F = sort_factor(F, (void *)cmpii, cmp_nodata);
    q->z[g] = gclone(F);
    set_avma(av);
  }
  return q;
}

enum {
  PF_PHI = 1,       // eulerphi(fa)
  PF_NUMDIV = 2,    // numdiv(fa)
  PF_SIGMA = 3,     // sumdivk(fa, arg)
  PF_MU = 4,        // moebius(fa)
  PF_DIVISORS = 5,  // divisors(fa): count + low limb of the largest
  PF_RADICAL = 6,   // factorback(primes)
  PF_ZNSQRT = 7,    // Zn_sqrt(a, fa): a from pool hb, item i
  PF_VALUE = 8,     // factorback(fa)
};

uint64_t pari_facop(int op, void *hq, void *hb, size_t i0, size_t k, uint64_t arg) {
  ppool *q = hq, *b = hb;
  size_t n = q->n;
  uint64_t s = 0;
  pari_sp av = avma;
  for (size_t j = 0; j < k; j++) {
    size_t i = item(i0, j, n);
    GEN F = q->z[i];
    switch (op) {
      case PF_PHI: s += lo(eulerphi(F)); break;
      case PF_NUMDIV: s += lo(numdiv(F)); break;
      case PF_SIGMA: s += lo(arg == 1 ? sumdiv(F) : sumdivk(F, (long)arg)); break;
      case PF_MU: s += (uint64_t)(moebius(F) + 1); break;
      case PF_DIVISORS: {
        GEN d = divisors(F);
        s += (uint64_t)(lg(d) - 1) + lo(gel(d, lg(d) - 1));
        break;
      }
      case PF_RADICAL: s += lo(factorback(gel(F, 1))); break;
      case PF_VALUE: s += lo(factorback(F)); break;
      case PF_ZNSQRT: {
        GEN r = Zn_sqrt(b->z[i], F);
        if (r) s += 1;
        break;
      }
    }
    set_avma(av);
  }
  return s;
}

const char *pari_version_str(void) { return PARIVERSION; }
