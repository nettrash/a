// FLINT reference shim: integer matrices given as text, word matrices modulo p as arrays.
// Each operation does the whole work on a copy of its input (as the crate does, which returns
// new matrices) and frees what it made; with `want` it also returns the result as text
// (malloc'd; free with lb_free) or a digest.
#include "common.h"
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>
#include <flint/fmpz_lll.h>
#include <flint/nmod_mat.h>
#include <stdio.h>

void lb_free(char *s) { free(s); }

static char *zmat_text(const fmpz_mat_t A) {
    lb_buf b = {0, 0, 0};
    lb_put(&b, "");
    for (slong i = 0; i < A->r; i++)
        for (slong j = 0; j < A->c; j++) {
            char *t = fmpz_get_str(NULL, 10, fmpz_mat_entry(A, i, j));
            lb_put(&b, t);
            lb_put(&b, " ");
            flint_free(t);
        }
    return b.s;
}

static char *qmat_text(const fmpq_mat_t A) {
    lb_buf b = {0, 0, 0};
    lb_put(&b, "");
    for (slong i = 0; i < A->r; i++)
        for (slong j = 0; j < A->c; j++) {
            char *t = fmpq_get_str(NULL, 10, fmpq_mat_entry(A, i, j));
            lb_put(&b, t);
            lb_put(&b, " ");
            flint_free(t);
        }
    return b.s;
}

static char *fmpz_text(const fmpz_t x) {
    char *t = fmpz_get_str(NULL, 10, x);
    char *s = strdup(t);
    flint_free(t);
    return s;
}

void *fl_zmat_new(const char *text, long r, long c) {
    fmpz_mat_struct *A = malloc(sizeof(fmpz_mat_struct));
    fmpz_mat_init(A, r, c);
    const char *p = text;
    char *tok = NULL;
    size_t cap = 0;
    for (slong i = 0; i < r; i++)
        for (slong j = 0; j < c; j++) {
            if (!lb_token(&p, &tok, &cap)) abort();
            if (fmpz_set_str(fmpz_mat_entry(A, i, j), tok, 10)) abort();
        }
    free(tok);
    return A;
}

void fl_zmat_free(void *A) {
    fmpz_mat_clear((fmpz_mat_struct *)A);
    free(A);
}

void *fl_qmat_new(const char *text, long r, long c) {
    fmpq_mat_struct *A = malloc(sizeof(fmpq_mat_struct));
    fmpq_mat_init(A, r, c);
    const char *p = text;
    char *tok = NULL;
    size_t cap = 0;
    for (slong i = 0; i < r; i++)
        for (slong j = 0; j < c; j++) {
            if (!lb_token(&p, &tok, &cap)) abort();
            if (fmpq_set_str(fmpq_mat_entry(A, i, j), tok, 10)) abort();
            fmpq_canonicalise(fmpq_mat_entry(A, i, j));
        }
    free(tok);
    return A;
}

void fl_qmat_free(void *A) {
    fmpq_mat_clear((fmpq_mat_struct *)A);
    free(A);
}

// ---- lattices ----

// fmpz_lll on a copy (Z_BASIS, APPROX Gram), with the transformation when with_u.
char *fl_lll(const void *A_, double delta, double eta, int with_u, int want) {
    const fmpz_mat_struct *A = A_;
    fmpz_mat_t B;
    fmpz_mat_init_set(B, A);
    fmpz_lll_t fl;
    fmpz_lll_context_init(fl, delta, eta, Z_BASIS, APPROX);
    if (with_u) {
        fmpz_mat_t U;
        fmpz_mat_init(U, A->r, A->r);
        fmpz_mat_one(U);
        fmpz_lll(B, U, fl);
        fmpz_mat_clear(U);
    } else {
        fmpz_lll(B, NULL, fl);
    }
    char *out = want ? zmat_text(B) : NULL;
    fmpz_mat_clear(B);
    return out;
}

int fl_lll_is_reduced(const void *A_, double delta, double eta, long prec) {
    fmpz_lll_t fl;
    fmpz_lll_context_init(fl, delta, eta, Z_BASIS, APPROX);
    return fmpz_lll_is_reduced((const fmpz_mat_struct *)A_, fl, prec);
}

// ---- normal forms ----

char *fl_hnf(const void *A_, int with_u, int want) {
    const fmpz_mat_struct *A = A_;
    fmpz_mat_t H;
    fmpz_mat_init(H, A->r, A->c);
    char *out = NULL;
    if (with_u) {
        fmpz_mat_t U;
        fmpz_mat_init(U, A->r, A->r);
        fmpz_mat_hnf_transform(H, U, A);
        if (want) {
            char *h = zmat_text(H), *u = zmat_text(U);
            lb_buf b = {0, 0, 0};
            lb_put(&b, h);
            lb_put(&b, "| ");
            lb_put(&b, u);
            free(h);
            free(u);
            out = b.s;
        }
        fmpz_mat_clear(U);
    } else {
        fmpz_mat_hnf(H, A);
        if (want) out = zmat_text(H);
    }
    fmpz_mat_clear(H);
    return out;
}

// The diagonal of the Smith form, as text; with_uv computes the transformations too.
char *fl_snf(const void *A_, int with_uv, int want) {
    const fmpz_mat_struct *A = A_;
    fmpz_mat_t S;
    fmpz_mat_init(S, A->r, A->c);
    if (with_uv) {
        fmpz_mat_t U, V;
        fmpz_mat_init(U, A->r, A->r);
        fmpz_mat_init(V, A->c, A->c);
        fmpz_mat_snf_transform(S, U, V, A);
        fmpz_mat_clear(U);
        fmpz_mat_clear(V);
    } else {
        fmpz_mat_snf(S, A);
    }
    char *out = NULL;
    if (want) {
        lb_buf b = {0, 0, 0};
        lb_put(&b, "");
        slong k = A->r < A->c ? A->r : A->c;
        for (slong i = 0; i < k; i++) {
            char *t = fmpz_get_str(NULL, 10, fmpz_mat_entry(S, i, i));
            lb_put(&b, t);
            lb_put(&b, " ");
            flint_free(t);
        }
        out = b.s;
    }
    fmpz_mat_clear(S);
    return out;
}

// ---- exact linear algebra ----

char *fl_det(const void *A_, int want) {
    fmpz_t d;
    fmpz_init(d);
    fmpz_mat_det(d, (const fmpz_mat_struct *)A_);
    char *out = want ? fmpz_text(d) : NULL;
    fmpz_clear(d);
    return out;
}

char *fl_det_bareiss(const void *A_, int want) {
    fmpz_t d;
    fmpz_init(d);
    fmpz_mat_det_bareiss(d, (const fmpz_mat_struct *)A_);
    char *out = want ? fmpz_text(d) : NULL;
    fmpz_clear(d);
    return out;
}

// fmpq_mat_solve_fmpz_mat: the solution over Q, "" if singular.
char *fl_solve(const void *A_, const void *B_, int want) {
    const fmpz_mat_struct *A = A_, *B = B_;
    fmpq_mat_t X;
    fmpq_mat_init(X, A->c, B->c);
    int ok = fmpq_mat_solve_fmpz_mat(X, A, B);
    char *out = want ? (ok ? qmat_text(X) : strdup("")) : NULL;
    fmpq_mat_clear(X);
    return out;
}

// fmpz_mat_solve_fflu (Bareiss), then the rational solution.
char *fl_solve_fflu(const void *A_, const void *B_, int want) {
    const fmpz_mat_struct *A = A_, *B = B_;
    fmpz_mat_t X;
    fmpz_t den;
    fmpz_mat_init(X, A->c, B->c);
    fmpz_init(den);
    int ok = fmpz_mat_solve_fflu(X, den, A, B);
    fmpq_mat_t Q;
    fmpq_mat_init(Q, A->c, B->c);
    if (ok) fmpq_mat_set_fmpz_mat_div_fmpz(Q, X, den);
    char *out = want ? (ok ? qmat_text(Q) : strdup("")) : NULL;
    fmpq_mat_clear(Q);
    fmpz_mat_clear(X);
    fmpz_clear(den);
    return out;
}

// fmpz_mat_inv, then the rational inverse (fmpq_mat_set_fmpz_mat_div_fmpz) when rational.
char *fl_inv(const void *A_, int rational, int want) {
    const fmpz_mat_struct *A = A_;
    fmpz_mat_t B;
    fmpz_t den;
    fmpz_mat_init(B, A->r, A->c);
    fmpz_init(den);
    int ok = fmpz_mat_inv(B, den, A);
    char *out = NULL;
    if (rational) {
        fmpq_mat_t X;
        fmpq_mat_init(X, A->r, A->c);
        if (ok) fmpq_mat_set_fmpz_mat_div_fmpz(X, B, den);
        if (want) out = ok ? qmat_text(X) : strdup("");
        fmpq_mat_clear(X);
    } else if (want) {
        out = ok ? fmpz_text(den) : strdup("");
    }
    fmpz_mat_clear(B);
    fmpz_clear(den);
    return out;
}

char *fl_mul(const void *A_, const void *B_, int want) {
    const fmpz_mat_struct *A = A_, *B = B_;
    fmpz_mat_t C;
    fmpz_mat_init(C, A->r, B->c);
    fmpz_mat_mul(C, A, B);
    char *out = want ? zmat_text(C) : NULL;
    fmpz_mat_clear(C);
    return out;
}

char *fl_qsolve(const void *A_, const void *B_, int want) {
    const fmpq_mat_struct *A = A_, *B = B_;
    fmpq_mat_t X;
    fmpq_mat_init(X, A->c, B->c);
    int ok = fmpq_mat_solve(X, A, B);
    char *out = want ? (ok ? qmat_text(X) : strdup("")) : NULL;
    fmpq_mat_clear(X);
    return out;
}

char *fl_qdet(const void *A_, int want) {
    fmpq_t d;
    fmpq_init(d);
    fmpq_mat_det(d, (const fmpq_mat_struct *)A_);
    char *out = NULL;
    if (want) {
        char *t = fmpq_get_str(NULL, 10, d);
        out = strdup(t);
        flint_free(t);
    }
    fmpq_clear(d);
    return out;
}

char *fl_qinv(const void *A_, int want) {
    const fmpq_mat_struct *A = A_;
    fmpq_mat_t X;
    fmpq_mat_init(X, A->r, A->c);
    int ok = fmpq_mat_inv(X, A);
    char *out = want ? (ok ? qmat_text(X) : strdup("")) : NULL;
    fmpq_mat_clear(X);
    return out;
}

// ---- matrices over GF(p) ----

static uint64_t nmat_digest(const nmod_mat_t A) {
    uint64_t h = LB_FNV_INIT;
    for (slong i = 0; i < A->r; i++)
        h = lb_fnv_words(h, (const uint64_t *)nmod_mat_entry_ptr(A, i, 0), (size_t)A->c);
    return h;
}

void *fl_nmat_new(const uint64_t *w, long r, long c, uint64_t p) {
    nmod_mat_struct *A = malloc(sizeof(nmod_mat_struct));
    nmod_mat_init(A, r, c, p);
    for (slong i = 0; i < r; i++)
        for (slong j = 0; j < c; j++) nmod_mat_entry(A, i, j) = w[i * c + j] % p;
    return A;
}

void fl_nmat_free(void *A) {
    nmod_mat_clear((nmod_mat_struct *)A);
    free(A);
}

// The operations: 0 mul(A, B), 1 lu (rank), 2 rank, 3 det, 4 solve(A, B), 5 inv, 6 rref, 7
// nullspace, 8 add, 9 sub, 10 transpose, 11 scalar mul by 12345, 12 from fmpz_mat (A_ is a
// fmpz_mat handle, B_ unused, p in the low word of `arg`).
uint64_t fl_nmat_op(int op, const void *A_, const void *B_, uint64_t arg, int want) {
    const nmod_mat_struct *A = A_, *B = B_;
    uint64_t h = 0;
    switch (op) {
    case 0: {
        nmod_mat_t C;
        nmod_mat_init(C, A->r, B->c, A->mod.n);
        nmod_mat_mul(C, A, B);
        if (want) h = nmat_digest(C);
        nmod_mat_clear(C);
        break;
    }
    case 1: {
        nmod_mat_t T;
        nmod_mat_init_set(T, A);
        slong *P = flint_malloc(sizeof(slong) * (A->r > 0 ? A->r : 1));
        h = (uint64_t)nmod_mat_lu(P, T, 0);
        flint_free(P);
        nmod_mat_clear(T);
        break;
    }
    case 2: h = (uint64_t)nmod_mat_rank(A); break;
    case 3: h = nmod_mat_det(A); break;
    case 4: {
        nmod_mat_t X;
        nmod_mat_init(X, A->c, B->c, A->mod.n);
        int ok = nmod_mat_solve(X, A, B);
        if (want) h = ok ? nmat_digest(X) : 0;
        nmod_mat_clear(X);
        break;
    }
    case 5: {
        nmod_mat_t X;
        nmod_mat_init(X, A->r, A->c, A->mod.n);
        int ok = nmod_mat_inv(X, A);
        if (want) h = ok ? nmat_digest(X) : 0;
        nmod_mat_clear(X);
        break;
    }
    case 6: {
        nmod_mat_t T;
        nmod_mat_init_set(T, A);
        slong r = nmod_mat_rref(T);
        if (want) h = nmat_digest(T) ^ (uint64_t)r;
        nmod_mat_clear(T);
        break;
    }
    case 7: {
        nmod_mat_t X;
        nmod_mat_init(X, A->c, A->c, A->mod.n);
        slong k = nmod_mat_nullspace(X, A);
        if (want) {
            // The digest of the first k columns, row-major, as the crate's n × k matrix.
            uint64_t g = LB_FNV_INIT;
            for (slong i = 0; i < A->c; i++)
                g = lb_fnv_words(g, (const uint64_t *)nmod_mat_entry_ptr(X, i, 0), (size_t)k);
            h = g;
        }
        nmod_mat_clear(X);
        break;
    }
    case 8:
    case 9: {
        nmod_mat_t C;
        nmod_mat_init(C, A->r, A->c, A->mod.n);
        if (op == 8) nmod_mat_add(C, A, B); else nmod_mat_sub(C, A, B);
        if (want) h = nmat_digest(C);
        nmod_mat_clear(C);
        break;
    }
    case 10: {
        nmod_mat_t C;
        nmod_mat_init(C, A->c, A->r, A->mod.n);
        nmod_mat_transpose(C, A);
        if (want) h = nmat_digest(C);
        nmod_mat_clear(C);
        break;
    }
    case 11: {
        nmod_mat_t C;
        nmod_mat_init(C, A->r, A->c, A->mod.n);
        nmod_mat_scalar_mul(C, A, 12345 % A->mod.n);
        if (want) h = nmat_digest(C);
        nmod_mat_clear(C);
        break;
    }
    case 12: {
        const fmpz_mat_struct *Z = A_;
        nmod_mat_t C;
        nmod_mat_init(C, Z->r, Z->c, arg);
        fmpz_mat_get_nmod_mat(C, Z);
        if (want) h = nmat_digest(C);
        nmod_mat_clear(C);
        break;
    }
    }
    return h;
}

const char *fl_version(void) { return FLINT_VERSION; }
