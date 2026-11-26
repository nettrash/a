/* FLINT reference shim: nmod_poly, fmpz_mod_poly, fmpz_poly, fmpq_poly, Arb's roots. */
#include <flint/flint.h>
#include <flint/thread_support.h>
#include <flint/fmpz.h>
#include <flint/fmpz_vec.h>
#include <flint/fmpq_vec.h>
#include <flint/nmod.h>
#include <flint/fmpq.h>
#include <flint/nmod_poly.h>
#include <flint/nmod_poly_factor.h>
#include <flint/fmpz_mod.h>
#include <flint/fmpz_mod_poly.h>
#include <flint/fmpz_mod_poly_factor.h>
#include <flint/fmpz_poly.h>
#include <flint/fmpz_poly_factor.h>
#include <flint/fmpq_poly.h>
#include <flint/arb.h>
#include <flint/acb.h>
#include <flint/arb_fmpz_poly.h>
#include <flint/acb_poly.h>
#include "shim.h"

SHIM_EXPORTS(sf)
const char* sf_label(void* h);
const char* sf_label(void* h) { (void)h; return ""; }

enum {
    W_MUL = 1, W_SQR, W_DIVREM, W_INV_SERIES, W_MULMOD, W_POWMOD_X, W_POWMOD_X_BINEXP, W_POWMOD,
    W_COMPOSE, W_COMPOSE_MOD, W_POW, W_EVAL, W_DERIVATIVE, W_ADD, W_SCALE, W_GCD, W_XGCD,
    W_RESULTANT, W_DISCRIMINANT, W_EVAL_MANY, W_EVAL_MANY_FAST, W_FROM_ROOTS, W_INTERPOLATE,
    W_INTERP_NEWTON, W_INTERP_LAGRANGE, W_FACTOR, W_FACTOR_BERLEKAMP, W_IS_IRREDUCIBLE, W_ROOTS,
    W_SQUAREFREE, W_DDF, W_EDF,
    M_MUL, M_SQR, M_DIVREM, M_POWMOD_X, M_INV_SERIES, M_MULMOD, M_COMPOSE_MOD, M_GCD, M_XGCD,
    M_RESULTANT, M_DISCRIMINANT, M_EVAL_MANY, M_EVAL_MANY_FAST, M_FROM_ROOTS, M_FACTOR,
    M_IS_IRREDUCIBLE, M_ROOTS,
    Z_MUL, Z_SQR, Z_PDIVREM, Z_DIV_EXACT, Z_COMPOSE, Z_POW, Z_EVAL, Z_DERIVATIVE, Z_ADD, Z_CONTENT,
    Z_GCD, Z_LCM, Z_RESULTANT, Z_DISCRIMINANT, Z_EVAL_MANY, Z_FROM_ROOTS, Z_FACTOR, Z_SQUAREFREE,
    Z_COUNT_REAL, Z_COUNT_REAL_STURM, Z_ISOLATE, Z_REAL_ROOTS, Z_COMPLEX_ROOTS,
    Q_MUL, Q_DIVREM, Q_INV_SERIES, Q_GCD, Q_XGCD, Q_RESULTANT, Q_COMPOSE, Q_INTERPOLATE,
    F_ROOTS, C_ROOTS
};

static const struct { const char* name; int op; } OPS[] = {
    {"w_mul", W_MUL}, {"w_sqr", W_SQR}, {"w_divrem", W_DIVREM}, {"w_inv_series", W_INV_SERIES},
    {"w_mulmod", W_MULMOD}, {"w_powmod_x", W_POWMOD_X}, {"w_powmod_x_binexp", W_POWMOD_X_BINEXP},
    {"w_powmod", W_POWMOD}, {"w_compose", W_COMPOSE}, {"w_compose_mod", W_COMPOSE_MOD},
    {"w_pow", W_POW}, {"w_eval", W_EVAL}, {"w_derivative", W_DERIVATIVE}, {"w_add", W_ADD},
    {"w_scale", W_SCALE}, {"w_gcd", W_GCD}, {"w_xgcd", W_XGCD}, {"w_resultant", W_RESULTANT},
    {"w_discriminant", W_DISCRIMINANT}, {"w_eval_many", W_EVAL_MANY},
    {"w_eval_many_fast", W_EVAL_MANY_FAST}, {"w_from_roots", W_FROM_ROOTS},
    {"w_interpolate", W_INTERPOLATE}, {"w_interp_newton", W_INTERP_NEWTON},
    {"w_interp_lagrange", W_INTERP_LAGRANGE}, {"w_factor", W_FACTOR},
    {"w_factor_berlekamp", W_FACTOR_BERLEKAMP}, {"w_is_irreducible", W_IS_IRREDUCIBLE},
    {"w_roots", W_ROOTS}, {"w_squarefree", W_SQUAREFREE}, {"w_ddf", W_DDF}, {"w_edf", W_EDF},
    {"m_mul", M_MUL}, {"m_sqr", M_SQR}, {"m_divrem", M_DIVREM}, {"m_powmod_x", M_POWMOD_X},
    {"m_inv_series", M_INV_SERIES}, {"m_mulmod", M_MULMOD}, {"m_compose_mod", M_COMPOSE_MOD},
    {"m_gcd", M_GCD}, {"m_xgcd", M_XGCD}, {"m_resultant", M_RESULTANT},
    {"m_discriminant", M_DISCRIMINANT}, {"m_eval_many", M_EVAL_MANY},
    {"m_eval_many_fast", M_EVAL_MANY_FAST}, {"m_from_roots", M_FROM_ROOTS}, {"m_factor", M_FACTOR},
    {"m_is_irreducible", M_IS_IRREDUCIBLE}, {"m_roots", M_ROOTS},
    {"z_mul", Z_MUL}, {"z_sqr", Z_SQR}, {"z_pdivrem", Z_PDIVREM}, {"z_div_exact", Z_DIV_EXACT},
    {"z_compose", Z_COMPOSE}, {"z_pow", Z_POW}, {"z_eval", Z_EVAL}, {"z_derivative", Z_DERIVATIVE},
    {"z_add", Z_ADD}, {"z_content", Z_CONTENT}, {"z_gcd", Z_GCD}, {"z_lcm", Z_LCM},
    {"z_resultant", Z_RESULTANT}, {"z_discriminant", Z_DISCRIMINANT},
    {"z_eval_many", Z_EVAL_MANY}, {"z_from_roots", Z_FROM_ROOTS}, {"z_factor", Z_FACTOR},
    {"z_squarefree", Z_SQUAREFREE}, {"z_count_real", Z_COUNT_REAL},
    {"z_count_real_sturm", Z_COUNT_REAL_STURM}, {"z_isolate", Z_ISOLATE},
    {"z_real_roots", Z_REAL_ROOTS}, {"z_complex_roots", Z_COMPLEX_ROOTS},
    {"q_mul", Q_MUL}, {"q_divrem", Q_DIVREM}, {"q_inv_series", Q_INV_SERIES}, {"q_gcd", Q_GCD},
    {"q_xgcd", Q_XGCD}, {"q_resultant", Q_RESULTANT}, {"q_compose", Q_COMPOSE},
    {"q_interpolate", Q_INTERPOLATE}, {"f_roots", F_ROOTS}, {"c_roots", C_ROOTS},
    {NULL, 0}};

typedef struct {
    int op, fam; /* fam: 'w', 'm', 'z', 'q', 'f' */
    size_t V;
    slong k;      /* inv_series length, pow exponent, edf degree, precision */
    ulong n;      /* word modulus */
    nmod_t mod;
    fmpz_t N, e;  /* modulus (m), exponent */
    fmpz_mod_ctx_t ctx;
    int has_ctx;
    /* per-variant inputs: up to three polynomials, two vectors, one scalar */
    nmod_poly_struct* w[3];
    fmpz_mod_poly_struct* m[3];
    fmpz_poly_struct* z[3];
    fmpq_poly_struct* q[3];
    nn_ptr* wx[2];
    fmpz** zx[2];
    slong vlen;   /* vector length */
    ulong* us;    /* scalar per variant (eval point, scale) */
    fmpz* zs;     /* scalar per variant (eval point) */
    double** dp;  /* f64 coefficients (complex interleaved) */
    slong dlen;
    /* results */
    nmod_poly_t wr[3];
    fmpz_mod_poly_t mr[3];
    fmpz_poly_t zr[3];
    fmpq_poly_t qr[3];
    ulong ur;
    fmpz_t zsr;
    fmpq_t qsr;
    slong sr;
    nn_ptr wv;
    fmpz* zv;
    nmod_poly_factor_t wf;
    fmpz_mod_poly_factor_t mf;
    fmpz_poly_factor_t zf;
    fmpq* exact; fmpz* carr; slong* karr; slong nexact, nint;
    arb_ptr rr; acb_ptr cr; slong nroots;
    double* droots;
    acb_ptr* cpoly;
    slong* degs;
    int npoly;
} H;

static void set_fmpz(fmpz_t f, const uint64_t* q) {
    size_t nl = (size_t)q[1];
    if (nl == 0) fmpz_zero(f);
    else fmpz_set_ui_array(f, (const ulong*)(q + 2), (slong)nl);
    if (q[0]) fmpz_neg(f, f);
}
static void set_fmpz_arg(fmpz_t f, const arg_t* a) {
    if (a->kind == K_U64) { fmpz_set_ui(f, a->u); return; }
    if (a->len == 0) fmpz_zero(f);
    else fmpz_set_ui_array(f, (const ulong*)a->p, (slong)a->len);
    if (a->neg) fmpz_neg(f, f);
}
static fmpz* fmpz_vec_from(const arg_t* a) {
    fmpz* v = _fmpz_vec_init((slong)a->len);
    const uint64_t* q = a->p;
    for (size_t i = 0; i < a->len; i++) { set_fmpz(v + i, q); q += enc_int_len(q); }
    return v;
}
static uint64_t fp_fmpz(const fmpz_t f) { return fmpz_fdiv_ui(f, FP); }
static uint64_t fp_inv(uint64_t a) {
    uint64_t r = 1, b = a, e = FP - 2;
    while (e) { if (e & 1) r = fp_mul(r, b); b = fp_mul(b, b); e >>= 1; }
    return r;
}
static void put_wpoly(outbuf_t* o, const nmod_poly_t p) {
    put(o, (uint64_t)p->length); put(o, fp_wvec((const uint64_t*)p->coeffs, (size_t)p->length));
}
static uint64_t fp_fmpz_vec(const fmpz* c, slong len) {
    uint64_t h = 0;
    for (slong i = len; i-- > 0;) h = fp_add(fp_mul(h, FP_R), fp_fmpz(c + i));
    return h;
}
static void put_mpoly(outbuf_t* o, const fmpz_mod_poly_t p) {
    put(o, (uint64_t)p->length); put(o, fp_fmpz_vec(p->coeffs, p->length));
}
static void put_zpoly(outbuf_t* o, const fmpz_poly_t p) {
    put(o, (uint64_t)p->length); put(o, fp_fmpz_vec(p->coeffs, p->length));
}
static uint64_t fp_qpoly(const fmpq_poly_t p) {
    return fp_mul(fp_fmpz_vec(p->coeffs, p->length), fp_inv(fp_fmpz(p->den)));
}
static void put_qpoly(outbuf_t* o, const fmpq_poly_t p) {
    put(o, (uint64_t)p->length); put(o, fp_qpoly(p));
}

static void w_from(nmod_poly_t p, const arg_t* a, ulong n) {
    nmod_poly_init2(p, n, (slong)a->len + 1);
    for (size_t i = 0; i < a->len; i++) p->coeffs[i] = a->p[i];
    p->length = (slong)a->len;
    _nmod_poly_normalise(p);
}
static void z_from(fmpz_poly_t p, const arg_t* a) {
    fmpz_poly_init2(p, (slong)a->len);
    const uint64_t* q = a->p;
    for (size_t i = 0; i < a->len; i++) { set_fmpz(p->coeffs + i, q); q += enc_int_len(q); }
    _fmpz_poly_set_length(p, (slong)a->len);
    _fmpz_poly_normalise(p);
}
static void m_from(fmpz_mod_poly_t p, const arg_t* a, const fmpz_mod_ctx_t ctx) {
    fmpz_poly_t t;
    z_from(t, a);
    fmpz_mod_poly_init(p, ctx);
    fmpz_mod_poly_set_fmpz_poly(p, t, ctx);
    fmpz_poly_clear(t);
}
static void q_from(fmpq_poly_t p, const arg_t* a) {
    fmpq_poly_init(p);
    const uint64_t* q = a->p;
    fmpq_t c;
    fmpq_init(c);
    for (size_t i = 0; i < a->len; i++) {
        set_fmpz(fmpq_numref(c), q); q += enc_int_len(q);
        set_fmpz(fmpq_denref(c), q); q += enc_int_len(q);
        fmpq_canonicalise(c);
        fmpq_poly_set_coeff_fmpq(p, (slong)i, c);
    }
    fmpq_clear(c);
}

static int lookup(const char* op) {
    for (int i = 0; OPS[i].name; i++) if (!strcmp(OPS[i].name, op)) return OPS[i].op;
    return 0;
}

/* Inverse of the reversal of f modulo x^len(f), for the _preinv functions. */
static void w_finv(nmod_poly_t finv, const nmod_poly_t f) {
    nmod_poly_t r;
    nmod_poly_init_mod(r, f->mod);
    nmod_poly_init_mod(finv, f->mod);
    nmod_poly_reverse(r, f, f->length);
    nmod_poly_inv_series(finv, r, f->length);
    nmod_poly_clear(r);
}
static void m_finv(fmpz_mod_poly_t finv, const fmpz_mod_poly_t f, const fmpz_mod_ctx_t ctx) {
    fmpz_mod_poly_t r;
    fmpz_mod_poly_init(r, ctx);
    fmpz_mod_poly_init(finv, ctx);
    fmpz_mod_poly_reverse(r, f, f->length, ctx);
    fmpz_mod_poly_inv_series(finv, r, f->length, ctx);
    fmpz_mod_poly_clear(r, ctx);
}

void* sf_prep(const char* name, const uint64_t* buf, size_t nbuf) {
    flint_set_num_threads(1); /* one thread (FLINT's default, made explicit) */
    int op = lookup(name);
    if (!op) return NULL;
    input_t in;
    parse_input(buf, nbuf, &in);
    H* h = (H*)calloc(1, sizeof(H));
    h->op = op;
    h->fam = name[0];
    h->V = in.v;
    fmpz_init(h->N); fmpz_init(h->e); fmpz_init(h->zsr); fmpq_init(h->qsr);
    size_t si = 0; /* shared args in order: modulus (w, m), then op parameters */
    if (h->fam == 'w') { h->n = in.shared[si++].u; nmod_init(&h->mod, h->n); }
    if (h->fam == 'm') {
        set_fmpz_arg(h->N, &in.shared[si++]);
        fmpz_mod_ctx_init(h->ctx, h->N);
        h->has_ctx = 1;
    }
    /* op parameters */
    switch (op) {
    case W_INV_SERIES: case M_INV_SERIES: case Q_INV_SERIES: case W_POW: case Z_POW: case W_EDF:
    case Z_REAL_ROOTS: case Z_COMPLEX_ROOTS:
        h->k = (slong)in.shared[si++].u; break;
    case W_POWMOD_X: case W_POWMOD_X_BINEXP: case W_POWMOD:
        set_fmpz_arg(h->e, &in.shared[si++]); break;
    case M_POWMOD_X: fmpz_set(h->e, h->N); break;
    default: break;
    }
    size_t V = h->V, A = in.a;
    /* inputs by family: polynomials first, then vectors / scalars as the op wants */
    int npoly = 0, nvec = 0, nscal = 0;
    switch (op) {
    case W_EVAL_MANY: case W_EVAL_MANY_FAST: case M_EVAL_MANY: case M_EVAL_MANY_FAST: case Z_EVAL_MANY:
        npoly = 1; nvec = 1; break;
    case W_FROM_ROOTS: case M_FROM_ROOTS: case Z_FROM_ROOTS: nvec = 1; break;
    case W_INTERPOLATE: case W_INTERP_NEWTON: case W_INTERP_LAGRANGE: case Q_INTERPOLATE: nvec = 2; break;
    case W_EVAL: case W_SCALE: case Z_EVAL: npoly = 1; nscal = 1; break;
    case F_ROOTS: case C_ROOTS: break;
    default: npoly = (int)A; break;
    }
    h->npoly = npoly;
    for (int j = 0; j < npoly; j++) {
        switch (h->fam) {
        case 'w': h->w[j] = calloc(V, sizeof(nmod_poly_struct));
            for (size_t v = 0; v < V; v++) w_from(h->w[j] + v, VA(&in, v, j), h->n); break;
        case 'm': h->m[j] = calloc(V, sizeof(fmpz_mod_poly_struct));
            for (size_t v = 0; v < V; v++) m_from(h->m[j] + v, VA(&in, v, j), h->ctx); break;
        case 'z': h->z[j] = calloc(V, sizeof(fmpz_poly_struct));
            for (size_t v = 0; v < V; v++) z_from(h->z[j] + v, VA(&in, v, j)); break;
        case 'q': h->q[j] = calloc(V, sizeof(fmpq_poly_struct));
            for (size_t v = 0; v < V; v++) q_from(h->q[j] + v, VA(&in, v, j)); break;
        }
    }
    for (int j = 0; j < nvec; j++) {
        size_t col = (size_t)(npoly + j);
        h->vlen = (slong)VA(&in, 0, col)->len;
        if (h->fam == 'w') {
            h->wx[j] = calloc(V, sizeof(nn_ptr));
            for (size_t v = 0; v < V; v++) {
                h->wx[j][v] = flint_malloc(sizeof(ulong) * (size_t)(h->vlen + 1));
                memcpy(h->wx[j][v], VA(&in, v, col)->p, sizeof(ulong) * (size_t)h->vlen);
            }
        } else {
            h->zx[j] = calloc(V, sizeof(fmpz*));
            for (size_t v = 0; v < V; v++) h->zx[j][v] = fmpz_vec_from(VA(&in, v, col));
        }
    }
    if (nscal) {
        size_t col = (size_t)npoly;
        if (h->fam == 'w') {
            h->us = calloc(V, sizeof(ulong));
            for (size_t v = 0; v < V; v++) h->us[v] = VA(&in, v, col)->u;
        } else {
            h->zs = _fmpz_vec_init((slong)V);
            for (size_t v = 0; v < V; v++) set_fmpz_arg(h->zs + v, VA(&in, v, col));
        }
    }
    if (op == F_ROOTS || op == C_ROOTS) {
        /* kept complex interleaved: 2 * (degree + 1) doubles */
        h->dp = calloc(V, sizeof(double*));
        size_t len = VA(&in, 0, 0)->len;
        h->dlen = (slong)(op == F_ROOTS ? 2 * len : len);
        for (size_t v = 0; v < V; v++) {
            h->dp[v] = calloc((size_t)h->dlen, sizeof(double));
            const double* src = (const double*)VA(&in, v, 0)->p;
            if (op == F_ROOTS) for (size_t i = 0; i < len; i++) h->dp[v][2 * i] = src[i];
            else memcpy(h->dp[v], src, sizeof(double) * len);
        }
        h->op = op = F_ROOTS;
        h->droots = malloc(sizeof(double) * (size_t)h->dlen);
        h->cpoly = calloc(V, sizeof(acb_ptr));
        for (size_t v = 0; v < V; v++) {
            h->cpoly[v] = _acb_vec_init(h->dlen / 2);
            for (slong i = 0; i < h->dlen / 2; i++)
                acb_set_d_d(h->cpoly[v] + i, h->dp[v][2 * i], h->dp[v][2 * i + 1]);
        }
        h->cr = _acb_vec_init(h->dlen / 2);
    }
    /* results and precomputations */
    for (int j = 0; j < 3; j++) {
        if (h->fam == 'w') nmod_poly_init_mod(h->wr[j], h->mod);
        if (h->fam == 'm') fmpz_mod_poly_init(h->mr[j], h->ctx);
        if (h->fam == 'z') fmpz_poly_init(h->zr[j]);
        if (h->fam == 'q') fmpq_poly_init(h->qr[j]);
    }
    if (h->fam == 'w') nmod_poly_factor_init(h->wf);
    if (h->fam == 'm') fmpz_mod_poly_factor_init(h->mf, h->ctx);
    if (h->fam == 'z') fmpz_poly_factor_init(h->zf);
    if (h->vlen) {
        if (h->fam == 'w') h->wv = flint_malloc(sizeof(ulong) * (size_t)(h->vlen + 1));
        else h->zv = _fmpz_vec_init(h->vlen + 1);
    }
    /* precomputed inverses (outside the timing, as the page's harness did) */
    if (op == W_POWMOD_X || op == W_POWMOD_X_BINEXP || op == W_POWMOD || op == W_COMPOSE_MOD) {
        int fj = (op == W_POWMOD_X || op == W_POWMOD_X_BINEXP) ? 0 : (op == W_POWMOD ? 1 : 2);
        nmod_poly_struct* finv = calloc(V, sizeof(nmod_poly_struct));
        for (size_t v = 0; v < V; v++) w_finv(finv + v, h->w[fj] + v);
        /* the inverses go to the free slot w[2], or (compose_mod, whose f is w[2]) to wx[1] */
        if (op == W_COMPOSE_MOD) h->wx[1] = (nn_ptr*)finv; else h->w[2] = finv;
    }
    if (op == M_POWMOD_X || op == M_COMPOSE_MOD) {
        int fj = op == M_POWMOD_X ? 0 : 2;
        fmpz_mod_poly_struct* finv = calloc(V, sizeof(fmpz_mod_poly_struct));
        for (size_t v = 0; v < V; v++) m_finv(finv + v, h->m[fj] + v, h->ctx);
        if (op == M_COMPOSE_MOD) h->zx[1] = (fmpz**)finv; else h->m[2] = finv;
    }
    if (op == W_POWMOD_X_BINEXP) { nmod_poly_set_coeff_ui(h->wr[1], 1, 1); }
    if (op == W_DDF) h->degs = calloc((size_t)(h->w[0][0].length + 2), sizeof(slong));
    if (op == Z_ISOLATE) {
        slong d = h->z[0][0].length;
        h->exact = _fmpq_vec_init(d + 1);
        h->carr = _fmpz_vec_init(d + 1);
        h->karr = calloc((size_t)d + 1, sizeof(slong));
    }
    if (op == Z_REAL_ROOTS) h->rr = _arb_vec_init(h->z[0][0].length + 1);
    if (op == Z_COMPLEX_ROOTS) h->cr = _acb_vec_init(h->z[0][0].length + 1);
    free(in.var);
    return h;
}

static void run1(H* h, size_t v) {
    if (h->fam == 'w') h->wf->num = 0;
    if (h->fam == 'm') h->mf->num = 0;
    if (h->fam == 'z') h->zf->num = 0;
    nmod_poly_struct *wa = h->w[0] ? h->w[0] + v : NULL, *wb = h->w[1] ? h->w[1] + v : NULL,
                     *wc = h->w[2] ? h->w[2] + v : NULL;
    fmpz_mod_poly_struct *ma = h->m[0] ? h->m[0] + v : NULL, *mb = h->m[1] ? h->m[1] + v : NULL,
                         *mc = h->m[2] ? h->m[2] + v : NULL;
    fmpz_poly_struct *za = h->z[0] ? h->z[0] + v : NULL, *zb = h->z[1] ? h->z[1] + v : NULL;
    fmpq_poly_struct *qa = h->q[0] ? h->q[0] + v : NULL, *qb = h->q[1] ? h->q[1] + v : NULL;
    switch (h->op) {
    case W_MUL: nmod_poly_mul(h->wr[0], wa, wb); break;
    case W_SQR: nmod_poly_mul(h->wr[0], wa, wa); break;
    case W_DIVREM: nmod_poly_divrem(h->wr[0], h->wr[1], wa, wb); break;
    case W_INV_SERIES: nmod_poly_inv_series(h->wr[0], wa, h->k); break;
    case W_MULMOD: nmod_poly_mulmod(h->wr[0], wa, wb, wc); break;
    case W_POWMOD_X: nmod_poly_powmod_x_fmpz_preinv(h->wr[0], h->e, wa, wc); break;
    case W_POWMOD_X_BINEXP: nmod_poly_powmod_fmpz_binexp_preinv(h->wr[0], h->wr[1], h->e, wa, wc); break;
    case W_POWMOD: nmod_poly_powmod_fmpz_binexp_preinv(h->wr[0], wa, h->e, wb, wc); break;
    case W_COMPOSE: nmod_poly_compose(h->wr[0], wa, wb); break;
    case W_COMPOSE_MOD:
        nmod_poly_compose_mod_brent_kung_preinv(h->wr[0], wa, wb, wc, ((nmod_poly_struct*)h->wx[1]) + v);
        break;
    case W_POW: nmod_poly_pow(h->wr[0], wa, (ulong)h->k); break;
    case W_EVAL: h->ur = nmod_poly_evaluate_nmod(wa, h->us[v]); break;
    case W_DERIVATIVE: nmod_poly_derivative(h->wr[0], wa); break;
    case W_ADD: nmod_poly_add(h->wr[0], wa, wb); break;
    case W_SCALE: nmod_poly_scalar_mul_nmod(h->wr[0], wa, h->us[v]); break;
    case W_GCD: nmod_poly_gcd(h->wr[0], wa, wb); break;
    case W_XGCD: nmod_poly_xgcd(h->wr[0], h->wr[1], h->wr[2], wa, wb); break;
    case W_RESULTANT: h->ur = nmod_poly_resultant(wa, wb); break;
    case W_DISCRIMINANT: h->ur = nmod_poly_discriminant(wa); break;
    case W_EVAL_MANY: nmod_poly_evaluate_nmod_vec(h->wv, wa, h->wx[0][v], h->vlen); break;
    case W_EVAL_MANY_FAST: nmod_poly_evaluate_nmod_vec_fast(h->wv, wa, h->wx[0][v], h->vlen); break;
    case W_FROM_ROOTS: nmod_poly_product_roots_nmod_vec(h->wr[0], h->wx[0][v], h->vlen); break;
    case W_INTERPOLATE: nmod_poly_interpolate_nmod_vec_fast(h->wr[0], h->wx[0][v], h->wx[1][v], h->vlen); break;
    case W_INTERP_NEWTON: nmod_poly_interpolate_nmod_vec_newton(h->wr[0], h->wx[0][v], h->wx[1][v], h->vlen); break;
    case W_INTERP_LAGRANGE: nmod_poly_interpolate_nmod_vec_barycentric(h->wr[0], h->wx[0][v], h->wx[1][v], h->vlen); break;
    case W_FACTOR: h->ur = nmod_poly_factor(h->wf, wa); break;
    case W_FACTOR_BERLEKAMP: h->ur = nmod_poly_factor_with_berlekamp(h->wf, wa); break;
    case W_IS_IRREDUCIBLE: h->sr = nmod_poly_is_irreducible(wa); break;
    case W_ROOTS: nmod_poly_roots(h->wf, wa, 1); break;
    case W_SQUAREFREE: nmod_poly_factor_squarefree(h->wf, wa); break;
    case W_DDF: { slong* d = h->degs; nmod_poly_factor_distinct_deg(h->wf, wa, &d); break; }
    case W_EDF: h->wf->num = 0; nmod_poly_factor_equal_deg(h->wf, wa, h->k); break;
    case M_MUL: fmpz_mod_poly_mul(h->mr[0], ma, mb, h->ctx); break;
    case M_SQR: fmpz_mod_poly_sqr(h->mr[0], ma, h->ctx); break;
    case M_DIVREM: fmpz_mod_poly_divrem(h->mr[0], h->mr[1], ma, mb, h->ctx); break;
    case M_POWMOD_X: fmpz_mod_poly_powmod_x_fmpz_preinv(h->mr[0], h->e, ma, mc, h->ctx); break;
    case M_INV_SERIES: fmpz_mod_poly_inv_series(h->mr[0], ma, h->k, h->ctx); break;
    case M_MULMOD: fmpz_mod_poly_mulmod(h->mr[0], ma, mb, mc, h->ctx); break;
    case M_COMPOSE_MOD:
        fmpz_mod_poly_compose_mod_brent_kung_preinv(h->mr[0], ma, mb, mc, ((fmpz_mod_poly_struct*)h->zx[1]) + v, h->ctx);
        break;
    case M_GCD: fmpz_mod_poly_gcd(h->mr[0], ma, mb, h->ctx); break;
    case M_XGCD: fmpz_mod_poly_xgcd(h->mr[0], h->mr[1], h->mr[2], ma, mb, h->ctx); break;
    case M_RESULTANT: fmpz_mod_poly_resultant(h->zsr, ma, mb, h->ctx); break;
    case M_DISCRIMINANT: fmpz_mod_poly_discriminant(h->zsr, ma, h->ctx); break;
    case M_EVAL_MANY: fmpz_mod_poly_evaluate_fmpz_vec(h->zv, ma, h->zx[0][v], h->vlen, h->ctx); break;
    case M_EVAL_MANY_FAST: fmpz_mod_poly_evaluate_fmpz_vec_fast(h->zv, ma, h->zx[0][v], h->vlen, h->ctx); break;
    case M_FROM_ROOTS: fmpz_mod_poly_product_roots_fmpz_vec(h->mr[0], h->zx[0][v], h->vlen, h->ctx); break;
    case M_FACTOR: fmpz_mod_poly_factor(h->mf, ma, h->ctx); break;
    case M_IS_IRREDUCIBLE: h->sr = fmpz_mod_poly_is_irreducible(ma, h->ctx); break;
    case M_ROOTS: fmpz_mod_poly_roots(h->mf, ma, 1, h->ctx); break;
    case Z_MUL: fmpz_poly_mul(h->zr[0], za, zb); break;
    case Z_SQR: fmpz_poly_sqr(h->zr[0], za); break;
    case Z_PDIVREM: { ulong d; fmpz_poly_pseudo_divrem(h->zr[0], h->zr[1], &d, za, zb); h->sr = (slong)d; break; }
    case Z_DIV_EXACT: h->sr = fmpz_poly_divides(h->zr[0], za, zb); break;
    case Z_COMPOSE: fmpz_poly_compose(h->zr[0], za, zb); break;
    case Z_POW: fmpz_poly_pow(h->zr[0], za, (ulong)h->k); break;
    case Z_EVAL: fmpz_poly_evaluate_fmpz(h->zsr, za, h->zs + v); break;
    case Z_DERIVATIVE: fmpz_poly_derivative(h->zr[0], za); break;
    case Z_ADD: fmpz_poly_add(h->zr[0], za, zb); break;
    case Z_CONTENT: fmpz_poly_content(h->zsr, za); break;
    case Z_GCD: fmpz_poly_gcd(h->zr[0], za, zb); break;
    case Z_LCM: fmpz_poly_lcm(h->zr[0], za, zb); break;
    case Z_RESULTANT: fmpz_poly_resultant(h->zsr, za, zb); break;
    case Z_DISCRIMINANT: fmpz_poly_discriminant(h->zsr, za); break;
    case Z_EVAL_MANY: fmpz_poly_evaluate_fmpz_vec(h->zv, za, h->zx[0][v], h->vlen); break;
    case Z_FROM_ROOTS: fmpz_poly_product_roots_fmpz_vec(h->zr[0], h->zx[0][v], h->vlen); break;
    case Z_FACTOR: fmpz_poly_factor(h->zf, za); break;
    case Z_SQUAREFREE: fmpz_poly_factor_squarefree(h->zf, za); break;
    case Z_COUNT_REAL: h->sr = fmpz_poly_num_real_roots(za); break;
    case Z_COUNT_REAL_STURM: h->sr = fmpz_poly_num_real_roots_sturm(za); break;
    case Z_ISOLATE:
        fmpz_poly_isolate_real_roots(h->exact, &h->nexact, h->carr, h->karr, &h->nint, za);
        break;
    case Z_REAL_ROOTS: h->nroots = arb_fmpz_poly_real_roots(h->rr, za, 0, h->k); break;
    case Z_COMPLEX_ROOTS: arb_fmpz_poly_complex_roots(h->cr, za, 0, h->k); break;
    case Q_MUL: fmpq_poly_mul(h->qr[0], qa, qb); break;
    case Q_DIVREM: fmpq_poly_divrem(h->qr[0], h->qr[1], qa, qb); break;
    case Q_INV_SERIES: fmpq_poly_inv_series(h->qr[0], qa, h->k); break;
    case Q_GCD: fmpq_poly_gcd(h->qr[0], qa, qb); break;
    case Q_XGCD: fmpq_poly_xgcd(h->qr[0], h->qr[1], h->qr[2], qa, qb); break;
    case Q_RESULTANT: fmpq_poly_resultant(h->qsr, qa, qb); break;
    case Q_COMPOSE: fmpq_poly_compose(h->qr[0], qa, qb); break;
    case Q_INTERPOLATE: fmpq_poly_interpolate_fmpz_vec(h->qr[0], h->zx[0][v], h->zx[1][v], h->vlen); break;
    case F_ROOTS: {
        /* the public numerical root finder (Aberth/WDK, not certified) at 53 bits, its own
         * iteration count; _acb_poly_find_roots_double with maxiter 0 does not iterate */
        acb_poly_struct P = {h->cpoly[v], h->dlen / 2, h->dlen / 2};
        acb_poly_find_roots(h->cr, &P, NULL, 0, 53);
        break;
    }
    }
}

void sf_run(void* hv, uint64_t reps) {
    H* h = (H*)hv;
    size_t V = h->V;
    for (uint64_t r = 0; r < reps; r++) run1(h, (size_t)(r % V));
}

static int cmp_u64(const void* a, const void* b) {
    uint64_t x = *(const uint64_t*)a, y = *(const uint64_t*)b;
    return x < y ? -1 : x > y;
}
static int cmp_d2(const void* a, const void* b) {
    const double *x = a, *y = b;
    if (x[0] != y[0]) return x[0] < y[0] ? -1 : 1;
    return x[1] < y[1] ? -1 : x[1] > y[1];
}

static void put_double(outbuf_t* o, double d) { uint64_t u; memcpy(&u, &d, 8); put(o, u); }

static void out1(H* h, size_t v, outbuf_t* o) {
    switch (h->op) {
    case W_DIVREM: put_wpoly(o, h->wr[0]); put_wpoly(o, h->wr[1]); break;
    case W_XGCD: put_wpoly(o, h->wr[0]); put_wpoly(o, h->wr[1]); put_wpoly(o, h->wr[2]); break;
    case W_EVAL: case W_RESULTANT: case W_DISCRIMINANT: put(o, h->ur % FP); break;
    case W_EVAL_MANY: case W_EVAL_MANY_FAST:
        put(o, (uint64_t)h->vlen); put(o, fp_wvec((const uint64_t*)h->wv, (size_t)h->vlen)); break;
    case W_FACTOR: case W_FACTOR_BERLEKAMP: case W_SQUAREFREE: {
        uint64_t s = 0, mix = 0;
        for (slong i = 0; i < h->wf->num; i++) {
            s += (uint64_t)(h->wf->exp[i] * (h->wf->p[i].length - 1));
            mix = fp_add(mix, fp_mul((uint64_t)h->wf->exp[i], fp_mix(fp_wvec((const uint64_t*)h->wf->p[i].coeffs, (size_t)h->wf->p[i].length))));
        }
        uint64_t lc = h->op == W_SQUAREFREE ? h->w[0][v].coeffs[h->w[0][v].length - 1] : h->ur;
        put(o, (uint64_t)h->wf->num); put(o, s); put(o, mix); put(o, lc % FP);
        break;
    }
    case W_DDF: {
        /* the non-constant products with their degrees */
        uint64_t cnt = 0, mix = 0;
        for (slong i = 0; i < h->wf->num; i++) {
            if (h->wf->p[i].length <= 1) continue;
            cnt++;
            mix = fp_add(mix, fp_mix(fp_add(fp_wvec((const uint64_t*)h->wf->p[i].coeffs, (size_t)h->wf->p[i].length), (uint64_t)h->degs[i])));
        }
        put(o, cnt); put(o, mix);
        break;
    }
    case W_EDF: {
        uint64_t mix = 0;
        for (slong i = 0; i < h->wf->num; i++)
            mix = fp_add(mix, fp_mix(fp_wvec((const uint64_t*)h->wf->p[i].coeffs, (size_t)h->wf->p[i].length)));
        put(o, (uint64_t)h->wf->num); put(o, mix);
        break;
    }
    case W_IS_IRREDUCIBLE: case M_IS_IRREDUCIBLE: put(o, (uint64_t)h->sr); break;
    case W_ROOTS: {
        slong k = h->wf->num;
        uint64_t* rs = malloc(sizeof(uint64_t) * 2 * (size_t)(k + 1));
        for (slong i = 0; i < k; i++) {
            ulong c0 = h->wf->p[i].length ? h->wf->p[i].coeffs[0] : 0;
            rs[2 * i] = c0 ? h->n - c0 : 0;
            rs[2 * i + 1] = (uint64_t)h->wf->exp[i];
        }
        qsort(rs, (size_t)k, 16, cmp_u64);
        put(o, (uint64_t)k);
        for (slong i = 0; i < 2 * k; i++) put(o, rs[i]);
        free(rs);
        break;
    }
    case M_DIVREM: put_mpoly(o, h->mr[0]); put_mpoly(o, h->mr[1]); break;
    case M_XGCD: put_mpoly(o, h->mr[0]); put_mpoly(o, h->mr[1]); put_mpoly(o, h->mr[2]); break;
    case M_RESULTANT: case M_DISCRIMINANT: put(o, fp_fmpz(h->zsr)); break;
    case M_EVAL_MANY: case M_EVAL_MANY_FAST:
        put(o, (uint64_t)h->vlen); put(o, fp_fmpz_vec(h->zv, h->vlen)); break;
    case M_FACTOR: {
        uint64_t s = 0, mix = 0;
        for (slong i = 0; i < h->mf->num; i++) {
            s += (uint64_t)(h->mf->exp[i] * (h->mf->poly[i].length - 1));
            mix = fp_add(mix, fp_mul((uint64_t)h->mf->exp[i], fp_mix(fp_fmpz_vec(h->mf->poly[i].coeffs, h->mf->poly[i].length))));
        }
        fmpz_mod_poly_struct* a = h->m[0] + v;
        put(o, (uint64_t)h->mf->num); put(o, s); put(o, mix); put(o, fp_fmpz(a->coeffs + a->length - 1));
        break;
    }
    case M_ROOTS: {
        uint64_t mix = 0, s = 0;
        for (slong i = 0; i < h->mf->num; i++) {
            fmpz_t r; fmpz_init(r);
            fmpz_mod_neg(r, h->mf->poly[i].coeffs + 0, h->ctx);
            s += (uint64_t)h->mf->exp[i];
            mix = fp_add(mix, fp_mul((uint64_t)h->mf->exp[i], fp_mix(fp_fmpz(r))));
            fmpz_clear(r);
        }
        put(o, (uint64_t)h->mf->num); put(o, s); put(o, mix);
        break;
    }
    case Z_PDIVREM: {
        /* scale FLINT's quotient and remainder to the full power l^(m-n+1) */
        fmpz_poly_struct *a = h->z[0] + v, *b = h->z[1] + v;
        slong full = a->length - b->length + 1;
        if (full < 0) full = 0;
        fmpz_t l; fmpz_init(l);
        fmpz_pow_ui(l, b->coeffs + b->length - 1, (ulong)(full - h->sr));
        fmpz_poly_scalar_mul_fmpz(h->zr[2], h->zr[0], l); put_zpoly(o, h->zr[2]);
        fmpz_poly_scalar_mul_fmpz(h->zr[2], h->zr[1], l); put_zpoly(o, h->zr[2]);
        fmpz_clear(l);
        break;
    }
    case Z_DIV_EXACT:
        if (h->sr) { put(o, 1); put_zpoly(o, h->zr[0]); } else put(o, 0);
        break;
    case Z_EVAL: case Z_CONTENT: case Z_RESULTANT: case Z_DISCRIMINANT: put(o, fp_fmpz(h->zsr)); break;
    case Z_EVAL_MANY: put(o, (uint64_t)h->vlen); put(o, fp_fmpz_vec(h->zv, h->vlen)); break;
    case Z_FACTOR: case Z_SQUAREFREE: {
        uint64_t s = 0, mix = 0;
        for (slong i = 0; i < h->zf->num; i++) {
            s += (uint64_t)(h->zf->exp[i] * (h->zf->p[i].length - 1));
            mix = fp_add(mix, fp_mul((uint64_t)h->zf->exp[i], fp_mix(fp_fmpz_vec(h->zf->p[i].coeffs, h->zf->p[i].length))));
        }
        put(o, (uint64_t)h->zf->num); put(o, s); put(o, mix); put(o, fp_fmpz(&h->zf->c));
        break;
    }
    case Z_COUNT_REAL: case Z_COUNT_REAL_STURM: put(o, (uint64_t)h->sr); break;
    case Z_ISOLATE: put(o, (uint64_t)(h->nexact + h->nint)); break;
    case Z_REAL_ROOTS:
        put(o, (uint64_t)h->nroots);
        for (slong i = 0; i < h->nroots; i++) put_double(o, arf_get_d(arb_midref(h->rr + i), ARF_RND_NEAR));
        break;
    case Z_COMPLEX_ROOTS: {
        slong d = h->z[0][v].length - 1;
        double* z = malloc(sizeof(double) * 2 * (size_t)(d + 1));
        for (slong i = 0; i < d; i++) {
            z[2 * i] = arf_get_d(arb_midref(acb_realref(h->cr + i)), ARF_RND_NEAR);
            z[2 * i + 1] = arf_get_d(arb_midref(acb_imagref(h->cr + i)), ARF_RND_NEAR);
        }
        qsort(z, (size_t)d, 16, cmp_d2);
        put(o, (uint64_t)d);
        for (slong i = 0; i < 2 * d; i++) put_double(o, z[i]);
        free(z);
        break;
    }
    case Q_DIVREM: put_qpoly(o, h->qr[0]); put_qpoly(o, h->qr[1]); break;
    case Q_XGCD: put_qpoly(o, h->qr[0]); put_qpoly(o, h->qr[1]); put_qpoly(o, h->qr[2]); break;
    case Q_RESULTANT:
        put(o, fp_mul(fp_fmpz(fmpq_numref(h->qsr)), fp_inv(fp_fmpz(fmpq_denref(h->qsr))))); break;
    case F_ROOTS: {
        slong n = h->dlen / 2 - 1;
        double* z = malloc(sizeof(double) * 2 * (size_t)(n + 1));
        for (slong i = 0; i < n; i++) {
            z[2 * i] = arf_get_d(arb_midref(acb_realref(h->cr + i)), ARF_RND_NEAR);
            z[2 * i + 1] = arf_get_d(arb_midref(acb_imagref(h->cr + i)), ARF_RND_NEAR);
        }
        qsort(z, (size_t)n, 16, cmp_d2);
        put(o, (uint64_t)n);
        for (slong i = 0; i < 2 * n; i++) put_double(o, z[i]);
        free(z);
        break;
    }
    default:
        if (h->fam == 'w') put_wpoly(o, h->wr[0]);
        else if (h->fam == 'm') put_mpoly(o, h->mr[0]);
        else if (h->fam == 'z') put_zpoly(o, h->zr[0]);
        else if (h->fam == 'q') put_qpoly(o, h->qr[0]);
        break;
    }
}

size_t sf_check(void* hv, uint64_t* out, size_t cap) {
    H* h = (H*)hv;
    outbuf_t o = {out, cap, 0};
    for (size_t v = 0; v < h->V; v++) { run1(h, v); out1(h, v, &o); }
    return o.n;
}

void sf_free(void* hv) {
    H* h = (H*)hv;
    size_t V = h->V;
    for (int j = 0; j < 3; j++) {
        if (h->w[j]) { for (size_t v = 0; v < V; v++) nmod_poly_clear(h->w[j] + v); free(h->w[j]); }
        if (h->m[j]) { for (size_t v = 0; v < V; v++) fmpz_mod_poly_clear(h->m[j] + v, h->ctx); free(h->m[j]); }
        if (h->z[j]) { for (size_t v = 0; v < V; v++) fmpz_poly_clear(h->z[j] + v); free(h->z[j]); }
        if (h->q[j]) { for (size_t v = 0; v < V; v++) fmpq_poly_clear(h->q[j] + v); free(h->q[j]); }
        if (h->fam == 'w') nmod_poly_clear(h->wr[j]);
        if (h->fam == 'm') fmpz_mod_poly_clear(h->mr[j], h->ctx);
        if (h->fam == 'z') fmpz_poly_clear(h->zr[j]);
        if (h->fam == 'q') fmpq_poly_clear(h->qr[j]);
    }
    if (h->op == W_COMPOSE_MOD) {
        nmod_poly_struct* f = (nmod_poly_struct*)h->wx[1];
        for (size_t v = 0; v < V; v++) nmod_poly_clear(f + v);
        free(f); h->wx[1] = NULL;
    }
    if (h->op == M_COMPOSE_MOD) {
        fmpz_mod_poly_struct* f = (fmpz_mod_poly_struct*)h->zx[1];
        for (size_t v = 0; v < V; v++) fmpz_mod_poly_clear(f + v, h->ctx);
        free(f); h->zx[1] = NULL;
    }
    for (int j = 0; j < 2; j++) {
        if (h->wx[j]) { for (size_t v = 0; v < V; v++) flint_free(h->wx[j][v]); free(h->wx[j]); }
        if (h->zx[j]) { for (size_t v = 0; v < V; v++) _fmpz_vec_clear(h->zx[j][v], h->vlen); free(h->zx[j]); }
    }
    if (h->wv) flint_free(h->wv);
    if (h->zv) _fmpz_vec_clear(h->zv, h->vlen + 1);
    if (h->us) free(h->us);
    if (h->zs) _fmpz_vec_clear(h->zs, (slong)V);
    if (h->dp) {
        for (size_t v = 0; v < V; v++) { free(h->dp[v]); _acb_vec_clear(h->cpoly[v], h->dlen / 2); }
        free(h->dp); free(h->droots); free(h->cpoly); _acb_vec_clear(h->cr, h->dlen / 2);
    }
    if (h->fam == 'w') nmod_poly_factor_clear(h->wf);
    if (h->fam == 'm') fmpz_mod_poly_factor_clear(h->mf, h->ctx);
    if (h->fam == 'z') fmpz_poly_factor_clear(h->zf);
    if (h->degs) free(h->degs);
    if (h->exact) {
        /* sized by the first variant's length */
        free(h->karr);
    }
    if (h->has_ctx) fmpz_mod_ctx_clear(h->ctx);
    fmpz_clear(h->N); fmpz_clear(h->e); fmpz_clear(h->zsr); fmpq_clear(h->qsr);
    free(h);
}
