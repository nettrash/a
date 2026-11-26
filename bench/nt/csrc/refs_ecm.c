// GMP-ECM references (libecm): one curve of the elliptic curve method, Pollard's p - 1 and
// Williams's p + 1 per call, with explicit B1 and B2 (B2 = 0: stage 1 alone, as B2 < B2min = B1
// skips stage 2). The parameters are kept across calls (as ecm's batch product for B1, which
// GMP-ECM itself reuses when B1 is unchanged), the starting point set again each call.
#include <gmp.h>
#include <ecm.h>
#include <stdlib.h>
#include "common.h"

typedef struct {
  mpz_t *z;
  size_t n;
} gpool;  // the same layout as refs_gmp.c's pools

enum {
  E_ECM = 1,  // one curve: param = arg >> 32, sigma = arg & 0xffffffff
  E_PM1 = 2,  // x0 = 3
  E_PP1 = 3,  // x0 = 2/7, 6/5, 6 in turn (the three of `pp1`), each to the end unless found
};

uint64_t ecm_op(int op, void *ha, size_t i0, size_t k, uint64_t arg, double b1, double b2) {
  gpool *a = ha;
  size_t n = a->n;
  uint64_t s = 0;
  mpz_t f;
  mpz_init(f);
  ecm_params p;
  ecm_init(p);
  p->verbose = 0;
  if (b2 > 0) mpz_set_d(p->B2, b2);
  else mpz_set_ui(p->B2, 0);
  if (op == E_ECM) {
    p->method = ECM_ECM;
    p->param = (int)(arg >> 32);
  } else if (op == E_PM1) p->method = ECM_PM1;
  else p->method = ECM_PP1;
  for (size_t j = 0; j < k; j++) {
    mpz_t *x = &a->z[item(i0, j, n)];
    if (op == E_ECM) {
      mpz_set_ui(p->sigma, arg & 0xffffffffu);
      mpz_set_ui(p->x, 0);
      p->B1done = ECM_DEFAULT_B1_DONE;
      int r = ecm_factor(f, *x, b1, p);
      s += r > 0;
    } else if (op == E_PM1) {
      mpz_set_ui(p->x, 3);
      p->B1done = ECM_DEFAULT_B1_DONE;
      int r = ecm_factor(f, *x, b1, p);
      s += r > 0;
    } else {
      // 2/7, 6/5 and 6 modulo n.
      unsigned long num[3] = {2, 6, 6}, den[3] = {7, 5, 1};
      for (int t = 0; t < 3; t++) {
        mpz_set_ui(p->x, den[t]);
        mpz_invert(p->x, p->x, *x);
        mpz_mul_ui(p->x, p->x, num[t]);
        mpz_mod(p->x, p->x, *x);
        p->B1done = ECM_DEFAULT_B1_DONE;
        int r = ecm_factor(f, *x, b1, p);
        if (r > 0) {
          s += 1;
          break;
        }
      }
    }
  }
  ecm_clear(p);
  mpz_clear(f);
  return s;
}

const char *ecm_version_str(void) { return ECM_VERSION; }
