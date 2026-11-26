/* FLINT references of the fields scoreboard: fmpz_mod, nmod and mpn_mod for prime fields; fq,
 * fq_nmod and fq_zech for extension fields. Built by build.rs at -O3 with the native CPU flags
 * when FLINT is found. See shim.h for the interface. */
#include <stdlib.h>
#include <string.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_vec.h>
#include <flint/fmpz_poly.h>
#include <flint/fmpz_mod.h>
#include <flint/fmpz_mod_poly.h>
#include <flint/nmod.h>
#include <flint/nmod_poly.h>
#include <flint/ulong_extras.h>
#include <flint/fq.h>
#include <flint/fq_nmod.h>
#include <flint/fq_zech.h>
#include <flint/gr.h>
#if __FLINT_RELEASE >= 30100
#include <flint/mpn_mod.h>
#define HAVE_MPN_MOD 1
#endif
#include "shim.h"

static volatile ulong sink;

static void set_limbs(fmpz_t f, const uint64_t *l, size_t n) { fmpz_set_ui_array(f, (const ulong *)l, n); }
static void get_limbs(uint64_t *out, const fmpz_t f, size_t n) { fmpz_get_ui_array((ulong *)out, n, f); }

/* ---------------------------------------------------------------- fmpz_mod */

typedef struct {
    fmpz_mod_ctx_t ctx;
    fmpz_t p, e, acc, t;
    fmpz *xs, *sq;
    size_t n, count;
    long s;
} fm_pf;

void *flint_fmpz_mod_pf_new(const uint64_t *p, size_t n, const uint64_t *xs, size_t count,
                            const uint64_t *e, size_t en) {
    fm_pf *h = calloc(1, sizeof *h);
    fmpz_init(h->p); fmpz_init(h->e); fmpz_init(h->acc); fmpz_init(h->t);
    set_limbs(h->p, p, n);
    set_limbs(h->e, e, en);
    fmpz_mod_ctx_init(h->ctx, h->p);
    h->n = n; h->count = count;
    h->xs = _fmpz_vec_init(count); h->sq = _fmpz_vec_init(count);
    for (size_t i = 0; i < count; i++) {
        set_limbs(h->xs + i, xs + i * n, n);
        fmpz_mod_mul(h->sq + i, h->xs + i, h->xs + i, h->ctx);
    }
    fmpz_set(h->acc, h->xs + 0); /* a random start: a chain from 1 would square 1 */
    return h;
}

void flint_fmpz_mod_pf_free(void *v) {
    fm_pf *h = v;
    _fmpz_vec_clear(h->xs, h->count); _fmpz_vec_clear(h->sq, h->count);
    fmpz_mod_ctx_clear(h->ctx);
    fmpz_clear(h->p); fmpz_clear(h->e); fmpz_clear(h->acc); fmpz_clear(h->t);
    free(h);
}

int flint_fmpz_mod_pf_out(void *v, int op, size_t i, uint64_t *out) {
    fm_pf *h = v;
    size_t m = h->count - 1, j = (i + 1) & m;
    fmpz_t r; fmpz_init(r);
    int ret = 0;
    switch (op) {
    case OP_MUL: fmpz_mod_mul(r, h->xs + i, h->xs + j, h->ctx); break;
    case OP_SQR: fmpz_mod_mul(r, h->xs + i, h->xs + i, h->ctx); break;
    case OP_ADD: fmpz_mod_add(r, h->xs + i, h->xs + j, h->ctx); break;
    case OP_SUB: fmpz_mod_sub(r, h->xs + i, h->xs + j, h->ctx); break;
    case OP_INV: fmpz_mod_inv(r, h->xs + i, h->ctx); break;
    case OP_SQRT: ret = fmpz_sqrtmod(r, h->sq + i, h->p); break;
    case OP_LEG: ret = fmpz_jacobi(h->xs + i, h->p); break;
    case OP_POW: fmpz_mod_pow_fmpz(r, h->xs + i, h->e, h->ctx); break;
    default: ret = UNSUPPORTED;
    }
    get_limbs(out, r, h->n);
    fmpz_clear(r);
    return ret;
}

void flint_fmpz_mod_pf_run(void *v, int op, uint64_t iters) {
    fm_pf *h = v;
    size_t m = h->count - 1;
    fmpz *acc = h->acc, *t = h->t;
    long s = 0;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) fmpz_mod_mul(acc, acc, h->xs + (i & m), h->ctx); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) fmpz_mod_mul(acc, acc, acc, h->ctx); break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) fmpz_mod_add(acc, acc, h->xs + (i & m), h->ctx); break;
    case OP_SUB: for (uint64_t i = 0; i < iters; i++) fmpz_mod_sub(acc, acc, h->xs + (i & m), h->ctx); break;
    case OP_INV:
        for (uint64_t i = 0; i < iters; i++) { fmpz_mod_inv(t, h->xs + (i & m), h->ctx); fmpz_mod_add(acc, acc, t, h->ctx); }
        break;
    case OP_SQRT:
        for (uint64_t i = 0; i < iters; i++) { fmpz_sqrtmod(t, h->sq + (i & m), h->p); fmpz_mod_add(acc, acc, t, h->ctx); }
        break;
    case OP_LEG: for (uint64_t i = 0; i < iters; i++) s += fmpz_jacobi(h->xs + (i & m), h->p); break;
    case OP_POW:
        for (uint64_t i = 0; i < iters; i++) { fmpz_mod_pow_fmpz(t, h->xs + (i & m), h->e, h->ctx); fmpz_mod_add(acc, acc, t, h->ctx); }
        break;
    }
    if (fmpz_is_zero(acc)) fmpz_one(acc); /* a product chain must not collapse to 0 */
    sink = (ulong)s + fmpz_get_ui(acc);
}

/* ---------------------------------------------------------------- nmod (one limb) */

typedef struct {
    nmod_t mod;
    ulong *xs, *sq, e, acc;
    size_t count;
} nm_pf;

void *flint_nmod_pf_new(const uint64_t *p, size_t n, const uint64_t *xs, size_t count,
                        const uint64_t *e, size_t en) {
    if (n != 1) return NULL;
    nm_pf *h = calloc(1, sizeof *h);
    nmod_init(&h->mod, p[0]);
    h->count = count;
    h->xs = malloc(count * sizeof(ulong)); h->sq = malloc(count * sizeof(ulong));
    for (size_t i = 0; i < count; i++) { h->xs[i] = xs[i]; h->sq[i] = nmod_mul(xs[i], xs[i], h->mod); }
    h->e = en ? e[0] : 0;
    h->acc = h->xs[0];
    return h;
}

void flint_nmod_pf_free(void *v) { nm_pf *h = v; free(h->xs); free(h->sq); free(h); }

int flint_nmod_pf_out(void *v, int op, size_t i, uint64_t *out) {
    nm_pf *h = v;
    size_t j = (i + 1) & (h->count - 1);
    ulong r = 0, p = h->mod.n;
    int ret = 0;
    switch (op) {
    case OP_MUL: r = nmod_mul(h->xs[i], h->xs[j], h->mod); break;
    case OP_SQR: r = nmod_mul(h->xs[i], h->xs[i], h->mod); break;
    case OP_ADD: r = nmod_add(h->xs[i], h->xs[j], h->mod); break;
    case OP_SUB: r = nmod_sub(h->xs[i], h->xs[j], h->mod); break;
    case OP_INV: r = nmod_inv(h->xs[i], h->mod); break;
    case OP_SQRT: r = n_sqrtmod(h->sq[i], p); ret = 1; break;
    case OP_LEG: ret = n_jacobi_unsigned(h->xs[i], p); break;
    case OP_POW: r = nmod_pow_ui(h->xs[i], h->e, h->mod); break;
    default: ret = UNSUPPORTED;
    }
    out[0] = r;
    return ret;
}

void flint_nmod_pf_run(void *v, int op, uint64_t iters) {
    nm_pf *h = v;
    size_t m = h->count - 1;
    ulong acc = h->acc, p = h->mod.n;
    long s = 0;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) acc = nmod_mul(acc, h->xs[i & m], h->mod); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) acc = nmod_mul(acc, acc, h->mod); break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) acc = nmod_add(acc, h->xs[i & m], h->mod); break;
    case OP_SUB: for (uint64_t i = 0; i < iters; i++) acc = nmod_sub(acc, h->xs[i & m], h->mod); break;
    case OP_INV: for (uint64_t i = 0; i < iters; i++) acc = nmod_add(acc, nmod_inv(h->xs[i & m], h->mod), h->mod); break;
    case OP_SQRT: for (uint64_t i = 0; i < iters; i++) acc = nmod_add(acc, n_sqrtmod(h->sq[i & m], p), h->mod); break;
    case OP_LEG: for (uint64_t i = 0; i < iters; i++) s += n_jacobi_unsigned(h->xs[i & m], p); break;
    case OP_POW: for (uint64_t i = 0; i < iters; i++) acc = nmod_add(acc, nmod_pow_ui(h->xs[i & m], h->e, h->mod), h->mod); break;
    }
    h->acc = acc ? acc : 1;
    sink = acc + (ulong)s;
}

/* ---------------------------------------------------------------- mpn_mod (2 to 16 limbs) */

#ifdef HAVE_MPN_MOD
typedef struct {
    gr_ctx_t ctx;
    ulong *xs, *sq, *acc, *t;
    fmpz_t e;
    size_t n, count;
} mm_pf;

void *flint_mpn_mod_pf_new(const uint64_t *p, size_t n, const uint64_t *xs, size_t count,
                           const uint64_t *e, size_t en) {
    if (n < 2 || n > MPN_MOD_MAX_LIMBS) return NULL;
    mm_pf *h = calloc(1, sizeof *h);
    if (_gr_ctx_init_mpn_mod(h->ctx, (const ulong *)p, n) != GR_SUCCESS) { free(h); return NULL; }
    h->n = n; h->count = count;
    h->xs = malloc(count * n * sizeof(ulong)); h->sq = malloc(count * n * sizeof(ulong));
    h->acc = calloc(n, sizeof(ulong)); h->t = calloc(n, sizeof(ulong));
    memcpy(h->xs, xs, count * n * sizeof(ulong));
    for (size_t i = 0; i < count; i++) mpn_mod_mul(h->sq + i * n, h->xs + i * n, h->xs + i * n, h->ctx);
    fmpz_init(h->e); set_limbs(h->e, e, en);
    memcpy(h->acc, h->xs, n * sizeof(ulong));
    return h;
}

void flint_mpn_mod_pf_free(void *v) {
    mm_pf *h = v;
    free(h->xs); free(h->sq); free(h->acc); free(h->t); fmpz_clear(h->e);
    gr_ctx_clear(h->ctx);
    free(h);
}

int flint_mpn_mod_pf_out(void *v, int op, size_t i, uint64_t *out) {
    mm_pf *h = v;
    size_t n = h->n, j = (i + 1) & (h->count - 1);
    ulong *x = h->xs + i * n, *y = h->xs + j * n, *r = (ulong *)out;
    int ret = 0;
    switch (op) {
    case OP_MUL: mpn_mod_mul(r, x, y, h->ctx); break;
    case OP_SQR: mpn_mod_mul(r, x, x, h->ctx); break;
    case OP_ADD: mpn_mod_add(r, x, y, h->ctx); break;
    case OP_SUB: mpn_mod_sub(r, x, y, h->ctx); break;
    case OP_INV: mpn_mod_inv(r, x, h->ctx); break;
    case OP_SQRT: ret = gr_sqrt(r, h->sq + i * n, h->ctx) == GR_SUCCESS ? 1 : UNSUPPORTED; break; /* GR_UNABLE: not offered */
    case OP_POW: gr_pow_fmpz(r, x, h->e, h->ctx); break;
    default: memset(out, 0, n * 8); ret = UNSUPPORTED;
    }
    return ret;
}

void flint_mpn_mod_pf_run(void *v, int op, uint64_t iters) {
    mm_pf *h = v;
    size_t n = h->n, m = h->count - 1;
    ulong *acc = h->acc, *t = h->t;
    int st = 0;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) st |= mpn_mod_mul(acc, acc, h->xs + (i & m) * n, h->ctx); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) st |= mpn_mod_mul(acc, acc, acc, h->ctx); break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) st |= mpn_mod_add(acc, acc, h->xs + (i & m) * n, h->ctx); break;
    case OP_SUB: for (uint64_t i = 0; i < iters; i++) st |= mpn_mod_sub(acc, acc, h->xs + (i & m) * n, h->ctx); break;
    case OP_INV:
        for (uint64_t i = 0; i < iters; i++) { st |= mpn_mod_inv(t, h->xs + (i & m) * n, h->ctx); st |= mpn_mod_add(acc, acc, t, h->ctx); }
        break;
    case OP_SQRT:
        for (uint64_t i = 0; i < iters; i++) { st |= gr_sqrt(t, h->sq + (i & m) * n, h->ctx); st |= mpn_mod_add(acc, acc, t, h->ctx); }
        break;
    case OP_POW:
        for (uint64_t i = 0; i < iters; i++) { st |= gr_pow_fmpz(t, h->xs + (i & m) * n, h->e, h->ctx); st |= mpn_mod_add(acc, acc, t, h->ctx); }
        break;
    }
    if (mpn_mod_is_zero(acc, h->ctx) == T_TRUE) mpn_mod_one(acc, h->ctx);
    sink = acc[0] + (ulong)st;
}
#else
void *flint_mpn_mod_pf_new(const uint64_t *p, size_t n, const uint64_t *xs, size_t count,
                           const uint64_t *e, size_t en) { return NULL; }
void flint_mpn_mod_pf_free(void *v) {}
int flint_mpn_mod_pf_out(void *v, int op, size_t i, uint64_t *out) { return UNSUPPORTED; }
void flint_mpn_mod_pf_run(void *v, int op, uint64_t iters) {}
#endif

/* ---------------------------------------------------------------- fq (fmpz coefficients) */

typedef struct {
    fmpz_mod_ctx_t pctx;
    fq_ctx_t ctx;
    fmpz_t p, e, nacc, nt;
    fq_struct *xs, *sq;
    fq_t acc, t;
    size_t n, k, count;
} fq_ext;

static void fq_set_limbs(fq_t x, const uint64_t *c, size_t k, size_t n, fq_ctx_t ctx) {
    fmpz_t v; fmpz_init(v);
    fq_zero(x, ctx);
    for (size_t j = 0; j < k; j++) { set_limbs(v, c + j * n, n); fmpz_poly_set_coeff_fmpz(x, j, v); }
    fmpz_clear(v);
}

static void fq_get_limbs(uint64_t *out, const fq_t x, size_t k, size_t n) {
    fmpz_t v; fmpz_init(v);
    for (size_t j = 0; j < k; j++) { fmpz_poly_get_coeff_fmpz(v, x, j); get_limbs(out + j * n, v, n); }
    fmpz_clear(v);
}

void *flint_fq_ext_new(const uint64_t *p, size_t n, const uint64_t *g, size_t k, const uint64_t *xs,
                       size_t count, const uint64_t *e, size_t en) {
    fq_ext *h = calloc(1, sizeof *h);
    fmpz_init(h->p); fmpz_init(h->e); fmpz_init(h->nacc); fmpz_init(h->nt);
    set_limbs(h->p, p, n); set_limbs(h->e, e, en);
    fmpz_mod_ctx_init(h->pctx, h->p);
    fmpz_mod_poly_t gp; fmpz_mod_poly_init(gp, h->pctx);
    fmpz_t c; fmpz_init(c);
    for (size_t j = 0; j <= k; j++) { set_limbs(c, g + j * n, n); fmpz_mod_poly_set_coeff_fmpz(gp, j, c, h->pctx); }
    fq_ctx_init_modulus(h->ctx, gp, h->pctx, "x");
    fmpz_mod_poly_clear(gp, h->pctx); fmpz_clear(c);
    h->n = n; h->k = k; h->count = count;
    h->xs = malloc(count * sizeof(fq_struct)); h->sq = malloc(count * sizeof(fq_struct));
    for (size_t i = 0; i < count; i++) {
        fq_init(h->xs + i, h->ctx); fq_init(h->sq + i, h->ctx);
        fq_set_limbs(h->xs + i, xs + i * k * n, k, n, h->ctx);
        fq_sqr(h->sq + i, h->xs + i, h->ctx);
    }
    fq_init(h->acc, h->ctx); fq_init(h->t, h->ctx); fq_set(h->acc, h->xs, h->ctx);
    return h;
}

void flint_fq_ext_free(void *v) {
    fq_ext *h = v;
    for (size_t i = 0; i < h->count; i++) { fq_clear(h->xs + i, h->ctx); fq_clear(h->sq + i, h->ctx); }
    free(h->xs); free(h->sq);
    fq_clear(h->acc, h->ctx); fq_clear(h->t, h->ctx);
    fq_ctx_clear(h->ctx); fmpz_mod_ctx_clear(h->pctx);
    fmpz_clear(h->p); fmpz_clear(h->e); fmpz_clear(h->nacc); fmpz_clear(h->nt);
    free(h);
}

int flint_fq_ext_out(void *v, int op, size_t i, uint64_t *out) {
    fq_ext *h = v;
    size_t j = (i + 1) & (h->count - 1);
    fq_t r; fq_init(r, h->ctx);
    fmpz_t s; fmpz_init(s);
    int ret = 0, scalar = 0;
    switch (op) {
    case OP_MUL: fq_mul(r, h->xs + i, h->xs + j, h->ctx); break;
    case OP_ADD: fq_add(r, h->xs + i, h->xs + j, h->ctx); break;
    case OP_SQR: fq_sqr(r, h->xs + i, h->ctx); break;
    case OP_INV: fq_inv(r, h->xs + i, h->ctx); break;
    case OP_FROB: fq_frobenius(r, h->xs + i, 1, h->ctx); break;
    case OP_NORM: fq_norm(s, h->xs + i, h->ctx); scalar = 1; break;
    case OP_TRACE: fq_trace(s, h->xs + i, h->ctx); scalar = 1; break;
    case OP_SQRT: ret = fq_sqrt(r, h->sq + i, h->ctx); break;
    case OP_ISSQ: ret = fq_is_square(h->xs + i, h->ctx); break;
    case OP_POW: fq_pow(r, h->xs + i, h->e, h->ctx); break;
    default: ret = UNSUPPORTED;
    }
    if (scalar) get_limbs(out, s, h->n); else fq_get_limbs(out, r, h->k, h->n);
    fq_clear(r, h->ctx); fmpz_clear(s);
    return ret;
}

void flint_fq_ext_run(void *v, int op, uint64_t iters) {
    fq_ext *h = v;
    size_t m = h->count - 1;
    long s = 0;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) fq_mul(h->acc, h->acc, h->xs + (i & m), h->ctx); break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) fq_add(h->acc, h->acc, h->xs + (i & m), h->ctx); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) fq_sqr(h->acc, h->acc, h->ctx); break;
    case OP_INV:
        for (uint64_t i = 0; i < iters; i++) { fq_inv(h->t, h->xs + (i & m), h->ctx); fq_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    case OP_FROB:
        for (uint64_t i = 0; i < iters; i++) { fq_frobenius(h->t, h->xs + (i & m), 1, h->ctx); fq_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    case OP_NORM:
        for (uint64_t i = 0; i < iters; i++) { fq_norm(h->nt, h->xs + (i & m), h->ctx); fmpz_mod_add(h->nacc, h->nacc, h->nt, h->pctx); }
        break;
    case OP_TRACE:
        for (uint64_t i = 0; i < iters; i++) { fq_trace(h->nt, h->xs + (i & m), h->ctx); fmpz_mod_add(h->nacc, h->nacc, h->nt, h->pctx); }
        break;
    case OP_SQRT:
        for (uint64_t i = 0; i < iters; i++) { fq_sqrt(h->t, h->sq + (i & m), h->ctx); fq_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    case OP_ISSQ: for (uint64_t i = 0; i < iters; i++) s += fq_is_square(h->xs + (i & m), h->ctx); break;
    case OP_POW:
        for (uint64_t i = 0; i < iters; i++) { fq_pow(h->t, h->xs + (i & m), h->e, h->ctx); fq_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    }
    if (fq_is_zero(h->acc, h->ctx)) fq_one(h->acc, h->ctx);
    sink = (ulong)s + fmpz_get_ui(h->nacc);
}

/* ---------------------------------------------------------------- fq_nmod (word p) */

typedef struct {
    fq_nmod_ctx_t ctx;
    nmod_t mod;
    fmpz_t e, nt;
    ulong nacc;
    fq_nmod_struct *xs, *sq;
    fq_nmod_t acc, t;
    size_t k, count;
} fqn_ext;

static void fqn_set(fq_nmod_t x, const uint64_t *c, size_t k, fq_nmod_ctx_t ctx) {
    fq_nmod_zero(x, ctx);
    for (size_t j = 0; j < k; j++) nmod_poly_set_coeff_ui(x, j, c[j]);
}

static void fqn_get(uint64_t *out, const fq_nmod_t x, size_t k) {
    for (size_t j = 0; j < k; j++) out[j] = nmod_poly_get_coeff_ui(x, j);
}

static void fqn_ctx(fq_nmod_ctx_t ctx, ulong p, const uint64_t *g, size_t k) {
    nmod_poly_t gp; nmod_poly_init(gp, p);
    for (size_t j = 0; j <= k; j++) nmod_poly_set_coeff_ui(gp, j, g[j]);
    fq_nmod_ctx_init_modulus(ctx, gp, "x");
    nmod_poly_clear(gp);
}

void *flint_fq_nmod_ext_new(const uint64_t *p, size_t n, const uint64_t *g, size_t k, const uint64_t *xs,
                            size_t count, const uint64_t *e, size_t en) {
    if (n != 1) return NULL;
    fqn_ext *h = calloc(1, sizeof *h);
    fqn_ctx(h->ctx, p[0], g, k);
    nmod_init(&h->mod, p[0]);
    fmpz_init(h->e); fmpz_init(h->nt); set_limbs(h->e, e, en);
    h->k = k; h->count = count;
    h->xs = malloc(count * sizeof(fq_nmod_struct)); h->sq = malloc(count * sizeof(fq_nmod_struct));
    for (size_t i = 0; i < count; i++) {
        fq_nmod_init(h->xs + i, h->ctx); fq_nmod_init(h->sq + i, h->ctx);
        fqn_set(h->xs + i, xs + i * k, k, h->ctx);
        fq_nmod_sqr(h->sq + i, h->xs + i, h->ctx);
    }
    fq_nmod_init(h->acc, h->ctx); fq_nmod_init(h->t, h->ctx); fq_nmod_set(h->acc, h->xs, h->ctx);
    return h;
}

void flint_fq_nmod_ext_free(void *v) {
    fqn_ext *h = v;
    for (size_t i = 0; i < h->count; i++) { fq_nmod_clear(h->xs + i, h->ctx); fq_nmod_clear(h->sq + i, h->ctx); }
    free(h->xs); free(h->sq);
    fq_nmod_clear(h->acc, h->ctx); fq_nmod_clear(h->t, h->ctx);
    fq_nmod_ctx_clear(h->ctx); fmpz_clear(h->e); fmpz_clear(h->nt);
    free(h);
}

int flint_fq_nmod_ext_out(void *v, int op, size_t i, uint64_t *out) {
    fqn_ext *h = v;
    size_t j = (i + 1) & (h->count - 1);
    fq_nmod_t r; fq_nmod_init(r, h->ctx);
    fmpz_t s; fmpz_init(s);
    int ret = 0, scalar = 0;
    switch (op) {
    case OP_MUL: fq_nmod_mul(r, h->xs + i, h->xs + j, h->ctx); break;
    case OP_ADD: fq_nmod_add(r, h->xs + i, h->xs + j, h->ctx); break;
    case OP_SQR: fq_nmod_sqr(r, h->xs + i, h->ctx); break;
    case OP_INV: fq_nmod_inv(r, h->xs + i, h->ctx); break;
    case OP_FROB: fq_nmod_frobenius(r, h->xs + i, 1, h->ctx); break;
    case OP_NORM: fq_nmod_norm(s, h->xs + i, h->ctx); scalar = 1; break;
    case OP_TRACE: fq_nmod_trace(s, h->xs + i, h->ctx); scalar = 1; break;
    case OP_SQRT: ret = fq_nmod_sqrt(r, h->sq + i, h->ctx); break;
    case OP_ISSQ: ret = fq_nmod_is_square(h->xs + i, h->ctx); break;
    case OP_POW: fq_nmod_pow(r, h->xs + i, h->e, h->ctx); break;
    default: ret = UNSUPPORTED;
    }
    if (scalar) out[0] = fmpz_get_ui(s); else fqn_get(out, r, h->k);
    fq_nmod_clear(r, h->ctx); fmpz_clear(s);
    return ret;
}

void flint_fq_nmod_ext_run(void *v, int op, uint64_t iters) {
    fqn_ext *h = v;
    size_t m = h->count - 1;
    long s = 0;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) fq_nmod_mul(h->acc, h->acc, h->xs + (i & m), h->ctx); break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) fq_nmod_add(h->acc, h->acc, h->xs + (i & m), h->ctx); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) fq_nmod_sqr(h->acc, h->acc, h->ctx); break;
    case OP_INV:
        for (uint64_t i = 0; i < iters; i++) { fq_nmod_inv(h->t, h->xs + (i & m), h->ctx); fq_nmod_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    case OP_FROB:
        for (uint64_t i = 0; i < iters; i++) { fq_nmod_frobenius(h->t, h->xs + (i & m), 1, h->ctx); fq_nmod_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    case OP_NORM:
        for (uint64_t i = 0; i < iters; i++) { fq_nmod_norm(h->nt, h->xs + (i & m), h->ctx); h->nacc = nmod_add(h->nacc, fmpz_get_ui(h->nt), h->mod); }
        break;
    case OP_TRACE:
        for (uint64_t i = 0; i < iters; i++) { fq_nmod_trace(h->nt, h->xs + (i & m), h->ctx); h->nacc = nmod_add(h->nacc, fmpz_get_ui(h->nt), h->mod); }
        break;
    case OP_SQRT:
        for (uint64_t i = 0; i < iters; i++) { fq_nmod_sqrt(h->t, h->sq + (i & m), h->ctx); fq_nmod_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    case OP_ISSQ: for (uint64_t i = 0; i < iters; i++) s += fq_nmod_is_square(h->xs + (i & m), h->ctx); break;
    case OP_POW:
        for (uint64_t i = 0; i < iters; i++) { fq_nmod_pow(h->t, h->xs + (i & m), h->e, h->ctx); fq_nmod_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    }
    if (fq_nmod_is_zero(h->acc, h->ctx)) fq_nmod_one(h->acc, h->ctx);
    sink = (ulong)s + h->nacc;
}

/* ---------------------------------------------------------------- fq_zech (small fields, logs) */

typedef struct {
    fq_nmod_ctx_t nctx;
    fq_zech_ctx_t ctx;
    nmod_t mod;
    fmpz_t e, nt;
    ulong nacc;
    fq_zech_struct *xs, *sq;
    fq_zech_t acc, t;
    size_t k, count;
} fqz_ext;

void *flint_fq_zech_ext_new(const uint64_t *p, size_t n, const uint64_t *g, size_t k, const uint64_t *xs,
                            size_t count, const uint64_t *e, size_t en) {
    if (n != 1) return NULL;
    /* Zech logarithms need the modulus to be primitive and the field small. */
    double q = 1; for (size_t j = 0; j < k; j++) q *= (double)p[0];
    if (q > 1e7) return NULL;
    fqz_ext *h = calloc(1, sizeof *h);
    fqn_ctx(h->nctx, p[0], g, k);
    if (!fq_zech_ctx_init_fq_nmod_ctx_check(h->ctx, h->nctx)) { fq_nmod_ctx_clear(h->nctx); free(h); return NULL; }
    nmod_init(&h->mod, p[0]);
    fmpz_init(h->e); fmpz_init(h->nt); set_limbs(h->e, e, en);
    h->k = k; h->count = count;
    h->xs = malloc(count * sizeof(fq_zech_struct)); h->sq = malloc(count * sizeof(fq_zech_struct));
    fq_nmod_t a; fq_nmod_init(a, h->nctx);
    for (size_t i = 0; i < count; i++) {
        fq_zech_init(h->xs + i, h->ctx); fq_zech_init(h->sq + i, h->ctx);
        fqn_set(a, xs + i * k, k, h->nctx);
        fq_zech_set_fq_nmod(h->xs + i, a, h->ctx);
        fq_zech_sqr(h->sq + i, h->xs + i, h->ctx);
    }
    fq_nmod_clear(a, h->nctx);
    fq_zech_init(h->acc, h->ctx); fq_zech_init(h->t, h->ctx); fq_zech_set(h->acc, h->xs, h->ctx);
    return h;
}

void flint_fq_zech_ext_free(void *v) {
    fqz_ext *h = v;
    for (size_t i = 0; i < h->count; i++) { fq_zech_clear(h->xs + i, h->ctx); fq_zech_clear(h->sq + i, h->ctx); }
    free(h->xs); free(h->sq);
    fq_zech_clear(h->acc, h->ctx); fq_zech_clear(h->t, h->ctx);
    fq_zech_ctx_clear(h->ctx); fq_nmod_ctx_clear(h->nctx);
    fmpz_clear(h->e); fmpz_clear(h->nt);
    free(h);
}

int flint_fq_zech_ext_out(void *v, int op, size_t i, uint64_t *out) {
    fqz_ext *h = v;
    size_t j = (i + 1) & (h->count - 1);
    fq_zech_t r; fq_zech_init(r, h->ctx);
    fmpz_t s; fmpz_init(s);
    int ret = 0, scalar = 0;
    switch (op) {
    case OP_MUL: fq_zech_mul(r, h->xs + i, h->xs + j, h->ctx); break;
    case OP_ADD: fq_zech_add(r, h->xs + i, h->xs + j, h->ctx); break;
    case OP_SQR: fq_zech_sqr(r, h->xs + i, h->ctx); break;
    case OP_INV: fq_zech_inv(r, h->xs + i, h->ctx); break;
    case OP_FROB: fq_zech_frobenius(r, h->xs + i, 1, h->ctx); break;
    case OP_NORM: fq_zech_norm(s, h->xs + i, h->ctx); scalar = 1; break;
    case OP_TRACE: fq_zech_trace(s, h->xs + i, h->ctx); scalar = 1; break;
    case OP_SQRT: ret = fq_zech_sqrt(r, h->sq + i, h->ctx); break;
    case OP_ISSQ: ret = fq_zech_is_square(h->xs + i, h->ctx); break;
    case OP_POW: fq_zech_pow(r, h->xs + i, h->e, h->ctx); break;
    default: ret = UNSUPPORTED;
    }
    if (scalar) out[0] = fmpz_get_ui(s);
    else {
        fq_nmod_t a; fq_nmod_init(a, h->nctx);
        fq_zech_get_fq_nmod(a, r, h->ctx);
        fqn_get(out, a, h->k);
        fq_nmod_clear(a, h->nctx);
    }
    fq_zech_clear(r, h->ctx); fmpz_clear(s);
    return ret;
}

void flint_fq_zech_ext_run(void *v, int op, uint64_t iters) {
    fqz_ext *h = v;
    size_t m = h->count - 1;
    long s = 0;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) fq_zech_mul(h->acc, h->acc, h->xs + (i & m), h->ctx); break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) fq_zech_add(h->acc, h->acc, h->xs + (i & m), h->ctx); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) fq_zech_sqr(h->acc, h->acc, h->ctx); break;
    case OP_INV:
        for (uint64_t i = 0; i < iters; i++) { fq_zech_inv(h->t, h->xs + (i & m), h->ctx); fq_zech_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    case OP_FROB:
        for (uint64_t i = 0; i < iters; i++) { fq_zech_frobenius(h->t, h->xs + (i & m), 1, h->ctx); fq_zech_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    case OP_NORM:
        for (uint64_t i = 0; i < iters; i++) { fq_zech_norm(h->nt, h->xs + (i & m), h->ctx); h->nacc = nmod_add(h->nacc, fmpz_get_ui(h->nt), h->mod); }
        break;
    case OP_TRACE:
        for (uint64_t i = 0; i < iters; i++) { fq_zech_trace(h->nt, h->xs + (i & m), h->ctx); h->nacc = nmod_add(h->nacc, fmpz_get_ui(h->nt), h->mod); }
        break;
    case OP_SQRT:
        for (uint64_t i = 0; i < iters; i++) { fq_zech_sqrt(h->t, h->sq + (i & m), h->ctx); fq_zech_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    case OP_ISSQ: for (uint64_t i = 0; i < iters; i++) s += fq_zech_is_square(h->xs + (i & m), h->ctx); break;
    case OP_POW:
        for (uint64_t i = 0; i < iters; i++) { fq_zech_pow(h->t, h->xs + (i & m), h->e, h->ctx); fq_zech_add(h->acc, h->acc, h->t, h->ctx); }
        break;
    }
    if (fq_zech_is_zero(h->acc, h->ctx)) fq_zech_one(h->acc, h->ctx);
    sink = (ulong)s + h->nacc;
}
