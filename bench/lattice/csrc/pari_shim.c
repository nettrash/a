// PARI/GP reference shim (libpari): qflll (with and without FLATTER), mathnf, matsnf, det,
// matsolve, matinverse, products, lindep, qfminim and qfcvp. PARI's lattices are the columns of
// a matrix: the crate's rows are passed as columns (the transpose), so that both sides reduce
// the same vectors.
#include "common.h"
#include <pari/pari.h>

static int g_init = 0;

void pa_init(void) {
    if (g_init) return;
    pari_init(1UL << 28, 0);
    paristack_setsize(1UL << 28, 1UL << 34);
    g_init = 1;
}

// Every operation runs under pari_CATCH: an error of PARI (a precision too low, a singular
// matrix) returns "ERROR: <message>" instead of ending the process, as PARI does uncaught.
static char *pa_errtext(void) {
    char *m = pari_err2str(pari_err_last());
    lb_buf b = {0, 0, 0};
    lb_put(&b, "ERROR: ");
    lb_put(&b, m);
    pari_free(m);
    for (char *q = b.s; *q; q++)
        if (*q == '\n' || *q == '\t') *q = ' ';
    return b.s;
}
#define PA_BEGIN                                                                                \
    char *volatile out = NULL;                                                                  \
    pari_sp av = avma;                                                                          \
    pari_CATCH(CATCH_ALL) { out = pa_errtext(); }                                               \
    pari_TRY {
#define PA_END                                                                                  \
    }                                                                                           \
    pari_ENDCATCH;                                                                              \
    set_avma(av);                                                                               \
    return out;

static char *gen_text(GEN x) {
    char *t = GENtostr(x);
    char *s = strdup(t);
    pari_free(t);
    return s;
}

// The crate's rows of a matrix as text; columns of the PARI matrix when transpose (lattices),
// as they are otherwise (linear algebra).
void *pa_zmat_new(const char *text, long r, long c, int transpose) {
    pari_sp av = avma;
    long R = transpose ? c : r, C = transpose ? r : c;
    GEN M = cgetg(C + 1, t_MAT);
    for (long j = 1; j <= C; j++) gel(M, j) = cgetg(R + 1, t_COL);
    const char *p = text;
    char *tok = NULL;
    size_t cap = 0;
    for (long i = 0; i < r; i++)
        for (long j = 0; j < c; j++) {
            if (!lb_token(&p, &tok, &cap)) abort();
            int neg = tok[0] == '-';
            GEN v = strtoi(neg ? tok + 1 : tok);
            if (neg) v = negi(v);
            if (transpose) gcoeff(M, j + 1, i + 1) = v; else gcoeff(M, i + 1, j + 1) = v;
        }
    free(tok);
    GEN k = gclone(M);
    set_avma(av);
    return k;
}

// A rational matrix from the crate's text "p/q" entries, rows as rows.
void *pa_qmat_new(const char *text, long r, long c) {
    pari_sp av = avma;
    GEN M = cgetg(c + 1, t_MAT);
    for (long j = 1; j <= c; j++) gel(M, j) = cgetg(r + 1, t_COL);
    const char *p = text;
    char *tok = NULL;
    size_t cap = 0;
    for (long i = 0; i < r; i++)
        for (long j = 0; j < c; j++) {
            if (!lb_token(&p, &tok, &cap)) abort();
            gcoeff(M, i + 1, j + 1) = gp_read_str(tok);
        }
    free(tok);
    GEN k = gclone(M);
    set_avma(av);
    return k;
}

void pa_free(void *x) { gunclone((GEN)x); }

// The vector of reals m_i·2^e_i at prec_i bits: mantissas as text, exponents and precisions as
// arrays.
void *pa_reals_new(const char *mants, const long *exps, const long *precs, long n) {
    pari_sp av = avma;
    GEN v = cgetg(n + 1, t_VEC);
    const char *p = mants;
    char *tok = NULL;
    size_t cap = 0;
    for (long i = 0; i < n; i++) {
        if (!lb_token(&p, &tok, &cap)) abort();
        int neg = tok[0] == '-';
        GEN m = strtoi(neg ? tok + 1 : tok);
        if (neg) m = negi(m);
        GEN x = itor(m, nbits2prec(precs[i]));
        gel(v, i + 1) = shiftr(x, exps[i]);
    }
    free(tok);
    GEN k = gclone(v);
    set_avma(av);
    return k;
}

// The transpose of a PARI matrix of columns as the crate's rows, as text.
static char *cols_as_rows(GEN B) {
    lb_buf b = {0, 0, 0};
    lb_put(&b, "");
    long C = lg(B) - 1;
    long R = C ? nbrows(B) : 0;
    for (long j = 1; j <= C; j++)
        for (long i = 1; i <= R; i++) {
            char *t = GENtostr(gcoeff(B, i, j));
            lb_put(&b, t);
            lb_put(&b, " ");
            pari_free(t);
        }
    return b.s;
}

// qflll(x, flag) (T), then x·T, the reduced basis. flag 0 default (FLATTER), 1 no FLATTER.
char *pa_qflll(const void *x_, long flag, int want) {
    PA_BEGIN
    GEN x = (GEN)x_;
    GEN T = qflll0(x, flag);
    GEN B = ZM_mul(x, T);
    out = want ? cols_as_rows(B) : NULL;
    PA_END
}

char *pa_mathnf(const void *x_, long flag, int want) {
    PA_BEGIN
    GEN r = mathnf0((GEN)x_, flag);
    out = want ? gen_text(flag ? gel(r, 1) : r) : NULL;
    PA_END
}

char *pa_matsnf(const void *x_, long flag, int want) {
    PA_BEGIN
    GEN r = matsnf0((GEN)x_, flag);
    out = want ? gen_text(flag ? gel(r, 3) : r) : NULL;
    PA_END
}

char *pa_det(const void *x_, int want) {
    PA_BEGIN
    GEN d = det((GEN)x_);
    out = want ? gen_text(d) : NULL;
    PA_END
}

char *pa_solve(const void *x_, const void *b_, int want) {
    PA_BEGIN
    GEN s = gauss((GEN)x_, (GEN)b_);
    out = want ? gen_text(s) : NULL;
    PA_END
}

char *pa_inv(const void *x_, int want) {
    PA_BEGIN
    GEN s = RgM_inv((GEN)x_);
    out = want ? (s ? gen_text(s) : strdup("")) : NULL;
    PA_END
}

char *pa_mul(const void *x_, const void *y_, int want) {
    PA_BEGIN
    GEN s = ZM_mul((GEN)x_, (GEN)y_);
    out = want ? gen_text(s) : NULL;
    PA_END
}

char *pa_lindep(const void *v_, int want) {
    PA_BEGIN
    GEN s = lindep((GEN)v_);
    out = want ? gen_text(s) : NULL;
    PA_END
}

// The shortest vector of the lattice of the columns of x: the Gram matrix, then qfminim with
// no bound (the minimum), at most one pair stored. Returns the minimum.
char *pa_svp(const void *x_, long flag, int want) {
    PA_BEGIN
    GEN x = (GEN)x_;
    GEN G = ZM_transmultosym(x, x);
    GEN r = qfminim0(G, NULL, gen_1, flag, DEFAULTPREC);
    out = want ? gen_text(gel(r, 2)) : NULL;
    PA_END
}

// Every vector of norm at most `bound` (half of them): returns their number (pairs).
char *pa_short(const void *x_, const char *bound, int want) {
    PA_BEGIN
    GEN x = (GEN)x_;
    GEN G = ZM_transmultosym(x, x);
    GEN r = qfminim0(G, strtoi(bound), NULL, 0, DEFAULTPREC);
    out = want ? gen_text(gel(r, 1)) : NULL;
    PA_END
}

// The closest vector to y (a column of the crate's target, the lattice square and of full
// rank): the coordinates t = x⁻¹·y, the Gram matrix, then qfcvp. Returns the squared distance.
char *pa_cvp(const void *x_, const void *y_, int want) {
    PA_BEGIN
    GEN x = (GEN)x_, y = (GEN)y_;
    GEN t = gauss(x, y);
    GEN G = ZM_transmultosym(x, x);
    GEN r = qfcvp0(G, gel(t, 1), NULL, gen_1, 0);
    out = want ? gen_text(gel(r, 2)) : NULL;
    PA_END
}

const char *pa_version(void) { return PARIVERSION; }
