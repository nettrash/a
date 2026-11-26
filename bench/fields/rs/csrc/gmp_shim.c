/* GMP (mpz) reference of the fields scoreboard for prime fields: a product is mpz_mul and
 * mpz_tdiv_r by p, inverses mpz_invert, symbols mpz_legendre, powers mpz_powm. GMP has no modular
 * square root. See shim.h for the interface. */
#include <stdlib.h>
#include <gmp.h>
#include "shim.h"

static volatile unsigned long sink;

typedef struct {
    mpz_t p, e, acc, t, u;
    mpz_t *xs;
    size_t n, count;
} g_pf;

static void imp(mpz_t z, const uint64_t *l, size_t n) { mpz_import(z, n, -1, 8, 0, 0, l); }

static void exp_limbs(uint64_t *out, const mpz_t z, size_t n) {
    for (size_t j = 0; j < n; j++) out[j] = 0;
    size_t c = 0;
    mpz_export(out, &c, -1, 8, 0, 0, z);
}

void *gmp_pf_new(const uint64_t *p, size_t n, const uint64_t *xs, size_t count, const uint64_t *e,
                 size_t en) {
    g_pf *h = calloc(1, sizeof *h);
    mpz_inits(h->p, h->e, h->acc, h->t, h->u, NULL);
    imp(h->p, p, n); imp(h->e, e, en);
    h->n = n; h->count = count;
    h->xs = malloc(count * sizeof(mpz_t));
    for (size_t i = 0; i < count; i++) { mpz_init(h->xs[i]); imp(h->xs[i], xs + i * n, n); }
    mpz_set(h->acc, h->xs[0]);
    /* room for the products, so that the timed loops do not grow them */
    mpz_realloc2(h->t, 2 * 64 * n + 64);
    mpz_realloc2(h->acc, 2 * 64 * n + 64);
    return h;
}

void gmp_pf_free(void *v) {
    g_pf *h = v;
    for (size_t i = 0; i < h->count; i++) mpz_clear(h->xs[i]);
    free(h->xs);
    mpz_clears(h->p, h->e, h->acc, h->t, h->u, NULL);
    free(h);
}

static void addmod(mpz_t r, const mpz_t a, const mpz_t b, const mpz_t p) {
    mpz_add(r, a, b);
    if (mpz_cmp(r, p) >= 0) mpz_sub(r, r, p);
}

int gmp_pf_out(void *v, int op, size_t i, uint64_t *out) {
    g_pf *h = v;
    size_t j = (i + 1) & (h->count - 1);
    mpz_t r; mpz_init(r);
    int ret = 0;
    switch (op) {
    case OP_MUL: mpz_mul(r, h->xs[i], h->xs[j]); mpz_tdiv_r(r, r, h->p); break;
    case OP_SQR: mpz_mul(r, h->xs[i], h->xs[i]); mpz_tdiv_r(r, r, h->p); break;
    case OP_ADD: addmod(r, h->xs[i], h->xs[j], h->p); break;
    case OP_SUB: mpz_sub(r, h->xs[i], h->xs[j]); if (mpz_sgn(r) < 0) mpz_add(r, r, h->p); break;
    case OP_INV: mpz_invert(r, h->xs[i], h->p); break;
    case OP_LEG: ret = mpz_legendre(h->xs[i], h->p); break;
    case OP_POW: mpz_powm(r, h->xs[i], h->e, h->p); break;
    default: ret = UNSUPPORTED;
    }
    exp_limbs(out, r, h->n);
    mpz_clear(r);
    return ret;
}

void gmp_pf_run(void *v, int op, uint64_t iters) {
    g_pf *h = v;
    size_t m = h->count - 1;
    long s = 0;
    switch (op) {
    case OP_MUL:
        for (uint64_t i = 0; i < iters; i++) { mpz_mul(h->t, h->acc, h->xs[i & m]); mpz_tdiv_r(h->acc, h->t, h->p); }
        break;
    case OP_SQR:
        for (uint64_t i = 0; i < iters; i++) { mpz_mul(h->t, h->acc, h->acc); mpz_tdiv_r(h->acc, h->t, h->p); }
        break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) addmod(h->acc, h->acc, h->xs[i & m], h->p); break;
    case OP_SUB:
        for (uint64_t i = 0; i < iters; i++) {
            mpz_sub(h->acc, h->acc, h->xs[i & m]);
            if (mpz_sgn(h->acc) < 0) mpz_add(h->acc, h->acc, h->p);
        }
        break;
    case OP_INV:
        for (uint64_t i = 0; i < iters; i++) { mpz_invert(h->t, h->xs[i & m], h->p); addmod(h->acc, h->acc, h->t, h->p); }
        break;
    case OP_LEG: for (uint64_t i = 0; i < iters; i++) s += mpz_legendre(h->xs[i & m], h->p); break;
    case OP_POW:
        for (uint64_t i = 0; i < iters; i++) { mpz_powm(h->t, h->xs[i & m], h->e, h->p); addmod(h->acc, h->acc, h->t, h->p); }
        break;
    }
    if (mpz_sgn(h->acc) == 0) mpz_set(h->acc, h->xs[0]);
    sink = mpz_get_ui(h->acc) + (unsigned long)s;
}
