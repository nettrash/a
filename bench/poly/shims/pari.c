/* PARI reference shim: Flx / F2x (word moduli), FpX (any modulus), ZX and RgX over Z and Q,
 * polroots / polrootsreal / polsturm. One thread; every call runs between avma marks. */
#include <pari/pari.h>
#include "shim.h"

SHIM_EXPORTS(sp)
const char* sp_label(void* h);

enum {
    W_MUL = 1, W_SQR, W_DIVREM, W_INV_SERIES, W_MULMOD, W_POWMOD_X, W_POWMOD, W_COMPOSE_MOD, W_POW,
    W_EVAL, W_DERIVATIVE, W_ADD, W_SCALE, W_GCD, W_XGCD, W_RESULTANT, W_DISCRIMINANT, W_EVAL_MANY,
    W_FROM_ROOTS, W_INTERPOLATE, W_FACTOR, W_IS_IRREDUCIBLE, W_ROOTS, W_SQUAREFREE, W_DDF,
    Z_MUL, Z_SQR, Z_PDIVREM, Z_COMPOSE, Z_POW, Z_EVAL, Z_DERIVATIVE, Z_ADD, Z_CONTENT, Z_GCD,
    Z_RESULTANT, Z_DISCRIMINANT, Z_EVAL_MANY, Z_FROM_ROOTS, Z_FACTOR, Z_SQUAREFREE,
    Z_IS_IRREDUCIBLE, Z_COUNT_REAL, Z_REAL_ROOTS, Z_COMPLEX_ROOTS,
    Q_MUL, Q_DIVREM, Q_INV_SERIES, Q_GCD, Q_RESULTANT, Q_COMPOSE, Q_INTERPOLATE,
    F_ROOTS, C_ROOTS
};
static const struct { const char* n; int op; } T[] = {
    {"mul", W_MUL}, {"sqr", W_SQR}, {"divrem", W_DIVREM}, {"inv_series", W_INV_SERIES},
    {"mulmod", W_MULMOD}, {"powmod_x", W_POWMOD_X}, {"powmod", W_POWMOD},
    {"compose_mod", W_COMPOSE_MOD}, {"pow", W_POW}, {"eval", W_EVAL}, {"derivative", W_DERIVATIVE},
    {"add", W_ADD}, {"scale", W_SCALE}, {"gcd", W_GCD}, {"xgcd", W_XGCD},
    {"resultant", W_RESULTANT}, {"discriminant", W_DISCRIMINANT}, {"eval_many", W_EVAL_MANY},
    {"from_roots", W_FROM_ROOTS}, {"interpolate", W_INTERPOLATE}, {"factor", W_FACTOR},
    {"is_irreducible", W_IS_IRREDUCIBLE}, {"roots", W_ROOTS}, {"squarefree", W_SQUAREFREE},
    {"ddf", W_DDF}, {NULL, 0}};
static const struct { const char* n; int op; } ZQ[] = {
    {"z_mul", Z_MUL}, {"z_sqr", Z_SQR}, {"z_pdivrem", Z_PDIVREM}, {"z_compose", Z_COMPOSE},
    {"z_pow", Z_POW}, {"z_eval", Z_EVAL}, {"z_derivative", Z_DERIVATIVE}, {"z_add", Z_ADD},
    {"z_content", Z_CONTENT}, {"z_gcd", Z_GCD}, {"z_resultant", Z_RESULTANT},
    {"z_discriminant", Z_DISCRIMINANT}, {"z_eval_many", Z_EVAL_MANY},
    {"z_from_roots", Z_FROM_ROOTS}, {"z_factor", Z_FACTOR}, {"z_squarefree", Z_SQUAREFREE},
    {"z_is_irreducible", Z_IS_IRREDUCIBLE}, {"z_count_real", Z_COUNT_REAL},
    {"z_real_roots", Z_REAL_ROOTS}, {"z_complex_roots", Z_COMPLEX_ROOTS},
    {"q_mul", Q_MUL}, {"q_divrem", Q_DIVREM}, {"q_inv_series", Q_INV_SERIES}, {"q_gcd", Q_GCD},
    {"q_resultant", Q_RESULTANT}, {"q_compose", Q_COMPOSE}, {"q_interpolate", Q_INTERPOLATE},
    {"f_roots", F_ROOTS}, {"c_roots", C_ROOTS}, {NULL, 0}};

static int inited = 0;
static void init(void) {
    if (inited) return;
    pari_init_opts(1UL << 30, 0, INIT_JMPm | INIT_DFTm);
    paristack_setsize(1UL << 30, 1UL << 33);
    /* libpari starts its MT engine with one thread a core: ZX_resultant, ZX_disc, the modular
     * gcds and more then run in parallel. The scoreboard is single-threaded on both sides. */
    (void)setdefault("nbthreads", "1", d_SILENT);
    if (mt_nbthreads() != 1) abort();
    inited = 1;
}

typedef struct {
    int op, fam;     /* 'w' (Flx), '2' (F2x), 'm' (FpX), 'z', 'q', 'f', 'c' */
    size_t V;
    ulong p;         /* word modulus */
    GEN P;           /* t_INT modulus (clone) */
    GEN e;           /* exponent (clone) */
    long k;
    GEN* in[3];      /* per-variant clones */
    GEN res, res2, res3; /* clones of the last result (for the check only) */
    long sres;
    const char* label;
} H;

static GEN int_from(const uint64_t* q) {
    int neg = (int)q[0];
    size_t nl = (size_t)q[1];
    const uint64_t* l = q + 2;
    while (nl && l[nl - 1] == 0) nl--;
    if (!nl) return gen_0;
    GEN z = cgeti((long)nl + 2);
    z[1] = evalsigne(neg ? -1 : 1) | evallgefint((long)nl + 2);
    GEN w = int_LSW(z);
    for (size_t i = 0; i < nl; i++) { *w = (long)l[i]; w = int_nextW(w); }
    return z;
}
static GEN int_arg(const arg_t* a) {
    if (a->kind == K_U64) return utoi(a->u);
    uint64_t hdr[2] = {(uint64_t)a->neg, (uint64_t)a->len};
    /* rebuild the (neg, nl, limbs) layout the decoder wants */
    uint64_t* t = (uint64_t*)malloc(sizeof(uint64_t) * (2 + a->len));
    t[0] = hdr[0]; t[1] = hdr[1];
    memcpy(t + 2, a->p, sizeof(uint64_t) * a->len);
    GEN z = int_from(t);
    free(t);
    return z;
}
static GEN flx_from(const arg_t* a) {
    GEN z = cgetg((long)a->len + 2, t_VECSMALL);
    z[1] = evalvarn(0);
    for (size_t i = 0; i < a->len; i++) z[i + 2] = (long)a->p[i];
    return Flx_renormalize(z, (long)a->len + 2);
}
static GEN zx_from(const arg_t* a) {
    GEN z = cgetg((long)a->len + 2, t_POL);
    z[1] = evalvarn(0) | evalsigne(1);
    const uint64_t* q = a->p;
    for (size_t i = 0; i < a->len; i++) { gel(z, i + 2) = int_from(q); q += enc_int_len(q); }
    return normalizepol(z);
}
static GEN zx_from_w(const arg_t* a) {
    GEN z = cgetg((long)a->len + 2, t_POL);
    z[1] = evalvarn(0) | evalsigne(1);
    for (size_t i = 0; i < a->len; i++) gel(z, i + 2) = utoi(a->p[i]);
    return normalizepol(z);
}
static GEN qx_from(const arg_t* a) {
    GEN z = cgetg((long)a->len + 2, t_POL);
    z[1] = evalvarn(0) | evalsigne(1);
    const uint64_t* q = a->p;
    for (size_t i = 0; i < a->len; i++) {
        GEN n = int_from(q); q += enc_int_len(q);
        GEN d = int_from(q); q += enc_int_len(q);
        gel(z, i + 2) = gdiv(n, d);
    }
    return normalizepol(z);
}
static GEN zv_from(const arg_t* a) {   /* t_VEC of t_INT */
    GEN z = cgetg((long)a->len + 1, t_VEC);
    if (a->kind == K_WVEC) for (size_t i = 0; i < a->len; i++) gel(z, i + 1) = utoi(a->p[i]);
    else {
        const uint64_t* q = a->p;
        for (size_t i = 0; i < a->len; i++) { gel(z, i + 1) = int_from(q); q += enc_int_len(q); }
    }
    return z;
}
static GEN flv_from(const arg_t* a) {  /* t_VECSMALL */
    GEN z = cgetg((long)a->len + 1, t_VECSMALL);
    for (size_t i = 0; i < a->len; i++) z[i + 1] = (long)a->p[i];
    return z;
}
static GEN rx_from(const arg_t* a, int cplx) {   /* doubles; complex interleaved */
    size_t n = cplx ? a->len / 2 : a->len;
    const double* d = (const double*)a->p;
    GEN z = cgetg((long)n + 2, t_POL);
    z[1] = evalvarn(0) | evalsigne(1);
    for (size_t i = 0; i < n; i++)
        gel(z, i + 2) = cplx ? mkcomplex(dbltor(d[2 * i]), dbltor(d[2 * i + 1])) : dbltor(d[i]);
    return z;
}

static uint64_t fp_int(GEN x) { return umodiu(x, FP); }
static uint64_t fp_flx(GEN z) {
    uint64_t h = 0;
    for (long i = lg(z) - 1; i >= 2; i--) h = fp_step(h, (uint64_t)z[i]);
    return h;
}
static void put_flx(outbuf_t* o, GEN z) { put(o, (uint64_t)(lg(z) - 2)); put(o, fp_flx(z)); }
static uint64_t fp_inv(uint64_t a) {
    uint64_t r = 1, b = a, e = FP - 2;
    while (e) { if (e & 1) r = fp_mul(r, b); b = fp_mul(b, b); e >>= 1; }
    return r;
}
static uint64_t fp_q(GEN x) {
    if (typ(x) == t_INT) return fp_int(x);
    return fp_mul(fp_int(gel(x, 1)), fp_inv(fp_int(gel(x, 2))));
}
static uint64_t fp_pol(GEN z) {   /* t_POL of t_INT / t_FRAC */
    uint64_t h = 0;
    for (long i = lg(z) - 1; i >= 2; i--) h = fp_add(fp_mul(h, FP_R), fp_q(gel(z, i)));
    return h;
}
static void put_pol(outbuf_t* o, GEN z) { put(o, (uint64_t)(lg(z) - 2)); put(o, fp_pol(z)); }

void* sp_prep(const char* name, const uint64_t* buf, size_t nbuf) {
    init();
    int op = 0, fam = name[0];
    if (fam == 'w' || fam == 'm') {
        for (int i = 0; T[i].n; i++) if (!strcmp(T[i].n, name + 2)) op = T[i].op;
    } else {
        for (int i = 0; ZQ[i].n; i++) if (!strcmp(ZQ[i].n, name)) op = ZQ[i].op;
    }
    if (!op) return NULL;
    input_t in;
    parse_input(buf, nbuf, &in);
    pari_sp av = avma;
    H* h = (H*)calloc(1, sizeof(H));
    h->op = op; h->fam = fam; h->V = in.v;
    size_t si = 0;
    if (fam == 'w') {
        h->p = in.shared[si++].u;
        h->P = gclone(utoi(h->p));
        if (h->p == 2) h->fam = '2';
    }
    if (fam == 'm') h->P = gclone(int_arg(&in.shared[si++]));
    switch (op) {
    case W_INV_SERIES: case W_POW: case Q_INV_SERIES: case Z_POW:
    case Z_REAL_ROOTS: case Z_COMPLEX_ROOTS:
        h->k = (long)in.shared[si++].u; break;
    case W_POWMOD_X: case W_POWMOD:
        if (fam == 'm') h->e = gclone(h->P); else h->e = gclone(int_arg(&in.shared[si++]));
        break;
    default: break;
    }
    if (h->fam == '2') {
        switch (op) {
        case W_MUL: case W_SQR: case W_DIVREM: case W_GCD: case W_XGCD: case W_FACTOR:
        case W_IS_IRREDUCIBLE: case W_DDF: case W_ADD: case W_DERIVATIVE: case W_MULMOD:
        case W_POWMOD_X: case W_POWMOD: break;
        default: set_avma(av); free(in.var); free(h); return NULL;
        }
    }
    if ((h->fam == 'm' || h->fam == 'w' || h->fam == '2') && op == W_DISCRIMINANT && h->fam != 'm') {
        set_avma(av); free(in.var); free(h); return NULL;
    }
    for (int j = 0; j < 3 && j < (int)in.a; j++) h->in[j] = (GEN*)calloc(in.v, sizeof(GEN));
    for (size_t v = 0; v < in.v; v++) {
        for (size_t j = 0; j < in.a && j < 3; j++) {
            const arg_t* a = VA(&in, v, j);
            GEN g = NULL;
            pari_sp av2 = avma;
            switch (h->fam) {
            case 'w': case '2':
                if (a->kind == K_U64) g = utoi(a->u);
                else if ((op == W_EVAL_MANY && j == 1) || op == W_FROM_ROOTS || op == W_INTERPOLATE) g = flv_from(a);
                else { g = flx_from(a); if (h->fam == '2') g = Flx_to_F2x(g); }
                break;
            case 'm':
                if (a->kind == K_U64 || a->kind == K_INT) g = int_arg(a);
                else if ((op == W_EVAL_MANY && j == 1) || op == W_FROM_ROOTS || op == W_INTERPOLATE) g = zv_from(a);
                else g = a->kind == K_WVEC ? zx_from_w(a) : zx_from(a);
                break;
            case 'z':
                if (a->kind == K_U64 || a->kind == K_INT) g = int_arg(a);
                else if ((op == Z_EVAL_MANY && j == 1) || op == Z_FROM_ROOTS) g = zv_from(a);
                else g = zx_from(a);
                break;
            case 'q':
                if (op == Q_INTERPOLATE) g = zv_from(a);
                else g = qx_from(a);
                break;
            case 'f': g = rx_from(a, 0); break;
            case 'c': g = rx_from(a, 1); break;
            }
            h->in[j][v] = gclone(g);
            set_avma(av2);
        }
    }
    h->label = h->fam == '2' ? "F2x" : h->fam == 'w' ? "Flx" : h->fam == 'm' ? "FpX" : "";
    set_avma(av);
    free(in.var);
    return h;
}

/* One call; returns the result on the PARI stack (the caller resets avma). */
static GEN call(H* h, size_t v, GEN* r2, GEN* r3) {
    GEN a = h->in[0] ? h->in[0][v] : NULL, b = h->in[1] ? h->in[1][v] : NULL,
        c = h->in[2] ? h->in[2][v] : NULL;
    ulong p = h->p;
    GEN P = h->P;
    if (h->fam == '2') {
        switch (h->op) {
        case W_MUL: return F2x_mul(a, b);
        case W_SQR: return F2x_sqr(a);
        case W_DIVREM: return F2x_divrem(a, b, r2);
        case W_GCD: return F2x_gcd(a, b);
        case W_XGCD: return F2x_extgcd(a, b, r2, r3);
        case W_FACTOR: return F2x_factor(a);
        case W_IS_IRREDUCIBLE: h->sres = F2x_is_irred(a); return gen_0;
        case W_DDF: return F2x_ddf(a);
        case W_ADD: return F2x_add(a, b);
        case W_DERIVATIVE: return F2x_deriv(a);
        case W_MULMOD: return F2xq_mul(a, b, c);
        case W_POWMOD_X: return F2xq_pow(polx_F2x(0), h->e, a);
        case W_POWMOD: return F2xq_pow(a, h->e, b);
        }
        return gen_0;
    }
    if (h->fam == 'w') {
        switch (h->op) {
        case W_MUL: return Flx_mul(a, b, p);
        case W_SQR: return Flx_sqr(a, p);
        case W_DIVREM: return Flx_divrem(a, b, p, r2);
        case W_INV_SERIES: return Flxn_inv(a, h->k, p);
        case W_MULMOD: return Flxq_mul(a, b, c, p);
        case W_POWMOD_X: return Flxq_pow(polx_Flx(0), h->e, a, p);
        case W_POWMOD: return Flxq_pow(a, h->e, b, p);
        case W_COMPOSE_MOD: return Flx_Flxq_eval(a, b, c, p);
        case W_POW: return Flx_powu(a, (ulong)h->k, p);
        case W_EVAL: h->sres = (long)Flx_eval(a, itou(b), p); return gen_0;
        case W_DERIVATIVE: return Flx_deriv(a, p);
        case W_ADD: return Flx_add(a, b, p);
        case W_SCALE: return Flx_Fl_mul(a, itou(b), p);
        case W_GCD: return Flx_gcd(a, b, p);
        case W_XGCD: return Flx_extgcd(a, b, p, r2, r3);
        case W_RESULTANT: h->sres = (long)Flx_resultant(a, b, p); return gen_0;
        case W_EVAL_MANY: return Flx_Flv_multieval(a, b, p);
        case W_FROM_ROOTS: return Flv_roots_to_pol(a, p, 0);
        case W_INTERPOLATE: return Flv_polint(a, b, p, 0);
        case W_FACTOR: return Flx_factor(a, p);
        case W_IS_IRREDUCIBLE: h->sres = Flx_is_irred(a, p); return gen_0;
        case W_ROOTS: return Flx_roots(a, p);
        case W_SQUAREFREE: return Flx_factor_squarefree(a, p);
        case W_DDF: return Flx_ddf(a, p);
        }
        return gen_0;
    }
    if (h->fam == 'm') {
        switch (h->op) {
        case W_MUL: return FpX_mul(a, b, P);
        case W_SQR: return FpX_sqr(a, P);
        case W_DIVREM: return FpX_divrem(a, b, P, r2);
        case W_INV_SERIES: return FpXn_inv(a, h->k, P);
        case W_MULMOD: return FpXQ_mul(a, b, c, P);
        case W_POWMOD_X: return FpXQ_pow(pol_x(0), h->e, a, P);
        case W_POWMOD: return FpXQ_pow(a, h->e, b, P);
        case W_COMPOSE_MOD: return FpX_FpXQ_eval(a, b, c, P);
        case W_GCD: return FpX_gcd(a, b, P);
        case W_XGCD: return FpX_extgcd(a, b, P, r2, r3);
        case W_RESULTANT: return FpX_resultant(a, b, P);
        case W_DISCRIMINANT: return FpX_disc(a, P);
        case W_EVAL_MANY: return FpX_FpV_multieval(a, b, P);
        case W_FROM_ROOTS: return FpV_roots_to_pol(a, P, 0);
        case W_INTERPOLATE: return FpV_polint(a, b, P, 0);
        case W_FACTOR: return FpX_factor(a, P);
        case W_IS_IRREDUCIBLE: h->sres = FpX_is_irred(a, P); return gen_0;
        case W_ROOTS: return FpX_roots(a, P);
        }
        return gen_0;
    }
    switch (h->op) {
    case Z_MUL: return ZX_mul(a, b);
    case Z_SQR: return ZX_sqr(a);
    case Z_PDIVREM: return RgX_pseudodivrem(a, b, r2);
    case Z_COMPOSE: return poleval(a, b);
    case Z_POW: return gpowgs(a, h->k);
    case Z_EVAL: return ZX_Z_eval(a, b);
    case Z_DERIVATIVE: return ZX_deriv(a);
    case Z_ADD: return ZX_add(a, b);
    case Z_CONTENT: return content(a);
    case Z_GCD: return ZX_gcd(a, b);
    case Z_RESULTANT: return ZX_resultant(a, b);
    case Z_DISCRIMINANT: return ZX_disc(a);
    case Z_EVAL_MANY: {
        long n = lg(b) - 1;
        GEN y = cgetg(n + 1, t_VEC);
        for (long i = 1; i <= n; i++) gel(y, i) = ZX_Z_eval(a, gel(b, i));
        return y;
    }
    case Z_FROM_ROOTS: return roots_to_pol(a, 0);
    case Z_FACTOR: return ZX_factor(a);
    case Z_SQUAREFREE: { GEN ex; GEN f = ZX_squff(a, &ex); *r2 = ex; return f; }
    case Z_IS_IRREDUCIBLE: h->sres = ZX_is_irred(a); return gen_0;
    case Z_COUNT_REAL: h->sres = ZX_sturm(a); return gen_0;
    case Z_REAL_ROOTS: return realroots(a, NULL, h->k);
    case Z_COMPLEX_ROOTS: return roots(a, h->k);
    case Q_MUL: return RgX_mul(a, b);
    case Q_DIVREM: return RgX_divrem(a, b, r2);
    case Q_INV_SERIES: return RgXn_inv(a, h->k);
    case Q_GCD: return QX_gcd(a, b);
    case Q_RESULTANT: return QX_resultant(a, b);
    case Q_COMPOSE: return poleval(a, b);
    case Q_INTERPOLATE: return RgV_polint(a, b, 0);
    case F_ROOTS: case C_ROOTS: return roots(a, 64);
    }
    return gen_0;
}

void sp_run(void* hv, uint64_t reps) {
    H* h = (H*)hv;
    size_t V = h->V;
    for (uint64_t r = 0; r < reps; r++) {
        pari_sp av = avma;
        GEN r2 = NULL, r3 = NULL;
        (void)call(h, (size_t)(r % V), &r2, &r3);
        set_avma(av);
    }
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
static int cmp_d(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : x > y;
}
static void put_double(outbuf_t* o, double d) { uint64_t u; memcpy(&u, &d, 8); put(o, u); }

/* fingerprint of a factorisation matrix [factors, exponents] (Flx, FpX or ZX), constants skipped */
static void put_fac(outbuf_t* o, H* h, GEN F, GEN E, uint64_t lc) {
    long n = lg(F) - 1;
    uint64_t cnt = 0, sdeg = 0, mix = 0;
    for (long i = 1; i <= n; i++) {
        GEN f = gel(F, i);
        long e = E ? (typ(E) == t_VECSMALL ? E[i] : itos(gel(E, i))) : 1;
        uint64_t hf;
        long d;
        if (h->fam == 'w') { d = degpol(f); hf = fp_flx(f); }
        else if (h->fam == '2') { GEN g = F2x_to_Flx(f); d = degpol(g); hf = fp_flx(g); }
        else { d = degpol(f); hf = fp_pol(f); }
        if (d <= 0) continue;
        cnt++;
        sdeg += (uint64_t)(e * d);
        mix = fp_add(mix, fp_mul((uint64_t)e, fp_mix(hf)));
    }
    put(o, cnt); put(o, sdeg); put(o, mix); put(o, lc);
}

static uint64_t lc_of(H* h, size_t v) {
    GEN a = h->in[0][v];
    if (h->fam == 'w') return (uint64_t)a[lg(a) - 1] % FP;
    if (h->fam == '2') return 1;
    return fp_q(leading_coeff(a));
}

size_t sp_check(void* hv, uint64_t* out, size_t cap) {
    H* h = (H*)hv;
    outbuf_t o = {out, cap, 0};
    for (size_t v = 0; v < h->V; v++) {
        pari_sp av = avma;
        GEN r2 = NULL, r3 = NULL;
        GEN r = call(h, v, &r2, &r3);
        int w = h->fam == 'w', f2 = h->fam == '2', m = h->fam == 'm';
        if (f2 && h->op != W_FACTOR && h->op != W_DDF && h->op != W_IS_IRREDUCIBLE) {
            r = F2x_to_Flx(r);
            if (r2) r2 = F2x_to_Flx(r2);
            if (r3) r3 = F2x_to_Flx(r3);
            w = 1;
        }
        switch (h->op) {
        case W_DIVREM: /* = Z_* ids differ; the families below keep them apart */
            if (h->fam == 'w' || h->fam == '2' || m) {
                if (w) { put_flx(&o, r); put_flx(&o, r2); } else { put_pol(&o, r); put_pol(&o, r2); }
                goto next;
            }
            break;
        default: break;
        }
        if (h->fam == 'w' || h->fam == '2' || m) {
            switch (h->op) {
            case W_GCD:
                if (w) { r = Flx_normalize(r, f2 ? 2 : h->p); put_flx(&o, r); }
                else { r = FpX_normalize(r, h->P); put_pol(&o, r); }
                break;
            case W_XGCD: {
                if (w) {
                    ulong p = f2 ? 2 : h->p;
                    ulong li = Fl_inv((ulong)r[lg(r) - 1], p);
                    put_flx(&o, Flx_Fl_mul(r, li, p)); put_flx(&o, Flx_Fl_mul(r2, li, p)); put_flx(&o, Flx_Fl_mul(r3, li, p));
                } else {
                    GEN li = Fp_inv(leading_coeff(r), h->P);
                    put_pol(&o, FpX_Fp_mul(r, li, h->P)); put_pol(&o, FpX_Fp_mul(r2, li, h->P)); put_pol(&o, FpX_Fp_mul(r3, li, h->P));
                }
                break;
            }
            case W_EVAL: case W_RESULTANT:
                if (w) put(&o, (uint64_t)h->sres % FP); else put(&o, fp_int(r));
                break;
            case W_DISCRIMINANT: put(&o, fp_int(r)); break;
            case W_EVAL_MANY: {
                long n = lg(r) - 1;
                uint64_t hh = 0;
                for (long i = n; i >= 1; i--) hh = fp_add(fp_mul(hh, FP_R), w ? (uint64_t)r[i] % FP : fp_int(gel(r, i)));
                put(&o, (uint64_t)n); put(&o, hh);
                break;
            }
            case W_FACTOR: put_fac(&o, h, gel(r, 1), gel(r, 2), lc_of(h, v)); break;
            case W_IS_IRREDUCIBLE: put(&o, (uint64_t)h->sres); break;
            case W_ROOTS: {
                long n = lg(r) - 1;
                if (w) {
                    uint64_t* rs = malloc(sizeof(uint64_t) * 2 * (size_t)(n + 1));
                    for (long i = 1; i <= n; i++) { rs[2 * (i - 1)] = (uint64_t)r[i]; rs[2 * (i - 1) + 1] = 0; }
                    qsort(rs, (size_t)n, 16, cmp_u64);
                    put(&o, (uint64_t)n);
                    for (long i = 0; i < 2 * n; i++) put(&o, rs[i]);
                    free(rs);
                } else {
                    uint64_t mix = 0;
                    for (long i = 1; i <= n; i++) mix = fp_add(mix, fp_mix(fp_int(gel(r, i))));
                    put(&o, (uint64_t)n); put(&o, 0); put(&o, mix);
                }
                break;
            }
            case W_SQUAREFREE: {
                /* r[i] = the product of the factors of multiplicity i */
                long n = lg(r) - 1;
                uint64_t cnt = 0, sdeg = 0, mix = 0;
                for (long i = 1; i <= n; i++) {
                    GEN f = gel(r, i);
                    if (degpol(f) <= 0) continue;
                    f = Flx_normalize(f, h->p);
                    cnt++; sdeg += (uint64_t)(i * degpol(f));
                    mix = fp_add(mix, fp_mul((uint64_t)i, fp_mix(fp_flx(f))));
                }
                put(&o, cnt); put(&o, sdeg); put(&o, mix); put(&o, lc_of(h, v));
                break;
            }
            case W_DDF: {
                /* r = [products, degrees]: the product of the irreducible factors of each degree */
                GEN F = gel(r, 1), D = gel(r, 2);
                long n = lg(F) - 1;
                uint64_t cnt = 0, mix = 0;
                for (long i = 1; i <= n; i++) {
                    GEN f = gel(F, i);
                    if (f2) f = F2x_to_Flx(f);
                    if (degpol(f) <= 0) continue;
                    if (!f2) f = Flx_normalize(f, h->p); /* PARI's products are not monic */
                    cnt++;
                    mix = fp_add(mix, fp_mix(fp_add(fp_flx(f), (uint64_t)D[i])));
                }
                put(&o, cnt); put(&o, mix);
                break;
            }
            default:
                if (w) put_flx(&o, r); else put_pol(&o, r);
                break;
            }
            goto next;
        }
        switch (h->op) {
        case Z_PDIVREM: put_pol(&o, r); put_pol(&o, r2); break;
        case Q_DIVREM: put_pol(&o, r); put_pol(&o, r2); break;
        case Z_EVAL: case Z_CONTENT: case Z_RESULTANT: case Z_DISCRIMINANT: put(&o, fp_int(r)); break;
        case Q_RESULTANT: put(&o, fp_q(r)); break;
        case Z_GCD: if (signe(leading_coeff(r)) < 0) r = ZX_neg(r); put_pol(&o, r); break;
        case Q_GCD: r = RgX_Rg_div(r, leading_coeff(r)); put_pol(&o, r); break;
        case Z_EVAL_MANY: {
            long n = lg(r) - 1;
            uint64_t hh = 0;
            for (long i = n; i >= 1; i--) hh = fp_add(fp_mul(hh, FP_R), fp_int(gel(r, i)));
            put(&o, (uint64_t)n); put(&o, hh);
            break;
        }
        case Z_FACTOR: {
            GEN a = h->in[0][v];
            GEN c = content(a);
            if (signe(leading_coeff(a)) < 0) c = negi(c);
            put_fac(&o, h, gel(r, 1), gel(r, 2), fp_int(c));
            break;
        }
        case Z_SQUAREFREE: {
            GEN a = h->in[0][v];
            GEN c = content(a);
            if (signe(leading_coeff(a)) < 0) c = negi(c);
            put_fac(&o, h, r, r2, fp_int(c));
            break;
        }
        case Z_IS_IRREDUCIBLE: case Z_COUNT_REAL: put(&o, (uint64_t)h->sres); break;
        case Z_REAL_ROOTS: {
            long n = lg(r) - 1;
            double* d = malloc(sizeof(double) * (size_t)(n + 1));
            for (long i = 1; i <= n; i++) d[i - 1] = rtodbl(gel(r, i));
            qsort(d, (size_t)n, sizeof(double), cmp_d);
            put(&o, (uint64_t)n);
            for (long i = 0; i < n; i++) put_double(&o, d[i]);
            free(d);
            break;
        }
        case Z_COMPLEX_ROOTS: case F_ROOTS: case C_ROOTS: {
            long n = lg(r) - 1;
            double* d = malloc(sizeof(double) * 2 * (size_t)(n + 1));
            for (long i = 1; i <= n; i++) {
                GEN z = gel(r, i);
                d[2 * (i - 1)] = typ(z) == t_COMPLEX ? rtodbl(gel(z, 1)) : rtodbl(z);
                d[2 * (i - 1) + 1] = typ(z) == t_COMPLEX ? rtodbl(gel(z, 2)) : 0.0;
            }
            qsort(d, (size_t)n, 16, cmp_d2);
            put(&o, (uint64_t)n);
            for (long i = 0; i < 2 * n; i++) put_double(&o, d[i]);
            free(d);
            break;
        }
        default: put_pol(&o, r); break;
        }
    next:
        set_avma(av);
    }
    return o.n;
}

void sp_free(void* hv) {
    H* h = (H*)hv;
    for (int j = 0; j < 3; j++) if (h->in[j]) {
        for (size_t v = 0; v < h->V; v++) if (h->in[j][v]) gunclone(h->in[j][v]);
        free(h->in[j]);
    }
    if (h->P) gunclone(h->P);
    if (h->e) gunclone(h->e);
    free(h);
}
const char* sp_label(void* hv) { return ((H*)hv)->label; }
