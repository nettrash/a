// NTL reference shim: LLL_FP (deep insertions), G_LLL_FP, the exact LLL, BKZ_FP, HNF, mat_zz_p
// and mat_GF2.
#include "common.h"
#include <NTL/HNF.h>
#include <NTL/LLL.h>
#include <NTL/mat_GF2.h>
#include <NTL/mat_ZZ.h>
#include <NTL/mat_lzz_p.h>
#include <NTL/version.h>
#include <sstream>
#include <string>

using namespace NTL;

static char *mat_text(const mat_ZZ &B) {
    std::ostringstream o;
    for (long i = 0; i < B.NumRows(); i++)
        for (long j = 0; j < B.NumCols(); j++) o << B[i][j] << ' ';
    return strdup(o.str().c_str());
}

static char *zz_text(const ZZ &x) {
    std::ostringstream o;
    o << x;
    return strdup(o.str().c_str());
}

// NTL reports errors by exceptions (LLL_FP's "numbers too big", a singular matrix): each
// operation catches them and returns "ERROR: <message>" instead of ending the process.
static char *err_text(const char *what) {
    std::string m = std::string("ERROR: ") + what;
    for (auto &ch : m)
        if (ch == '\n' || ch == '\t') ch = ' ';
    return strdup(m.c_str());
}
#define NT_TRY try {
#define NT_CATCH                                                                                \
    }                                                                                           \
    catch (const std::exception &e) { return err_text(e.what()); }                              \
    catch (...) { return err_text("NTL error"); }

extern "C" {

void *nt_new(const char *text, long r, long c) {
    mat_ZZ *A = new mat_ZZ;
    A->SetDims(r, c);
    const char *p = text;
    char *tok = NULL;
    size_t cap = 0;
    for (long i = 0; i < r; i++)
        for (long j = 0; j < c; j++) {
            if (!lb_token(&p, &tok, &cap)) abort();
            (*A)[i][j] = conv<ZZ>(tok);
        }
    free(tok);
    return A;
}

void nt_free(void *A) { delete (mat_ZZ *)A; }

// kind 0: LLL_FP(δ, deep), 1: G_LLL_FP(δ, deep), 2: the exact LLL with δ = 99/100, 3: LLL_XD,
// 4: G_LLL_XD (doubles with an extended exponent, for entries LLL_FP refuses).
char *nt_lll(const void *A_, int kind, double delta, long deep, int with_u, int want) {
    NT_TRY
    mat_ZZ B(*(const mat_ZZ *)A_);
    mat_ZZ U;
    if (kind == 0) {
        if (with_u) LLL_FP(B, U, delta, deep); else LLL_FP(B, delta, deep);
    } else if (kind == 1) {
        if (with_u) G_LLL_FP(B, U, delta, deep); else G_LLL_FP(B, delta, deep);
    } else if (kind == 3) {
        if (with_u) LLL_XD(B, U, delta, deep); else LLL_XD(B, delta, deep);
    } else if (kind == 4) {
        if (with_u) G_LLL_XD(B, U, delta, deep); else G_LLL_XD(B, delta, deep);
    } else {
        ZZ det2;
        if (with_u) LLL(det2, B, U, 99, 100); else LLL(det2, B, 99, 100);
    }
    return want ? mat_text(B) : NULL;
    NT_CATCH
}

// BKZ_FP (kind 0), G_BKZ_FP (kind 1) or BKZ_XD (kind 2), no pruning.
char *nt_bkz(const void *A_, int kind, long beta, double delta, int with_u, int want) {
    NT_TRY
    mat_ZZ B(*(const mat_ZZ *)A_);
    mat_ZZ U;
    if (kind == 0) {
        if (with_u) BKZ_FP(B, U, delta, beta); else BKZ_FP(B, delta, beta);
    } else if (kind == 1) {
        if (with_u) G_BKZ_FP(B, U, delta, beta); else G_BKZ_FP(B, delta, beta);
    } else {
        if (with_u) BKZ_XD(B, U, delta, beta); else BKZ_XD(B, delta, beta);
    }
    return want ? mat_text(B) : NULL;
    NT_CATCH
}

// HNF of a square nonsingular matrix: D = |det A|, then HNF(W, A, D). Returns the product of
// the diagonal of W (= D) for the check.
char *nt_hnf(const void *A_, int want) {
    NT_TRY
    const mat_ZZ &A = *(const mat_ZZ *)A_;
    ZZ d;
    determinant(d, A);
    abs(d, d);
    mat_ZZ W;
    HNF(W, A, d);
    if (!want) return NULL;
    ZZ prod(1);
    for (long i = 0; i < W.NumRows(); i++) prod *= W[i][i];
    abs(prod, prod);
    return zz_text(prod);
    NT_CATCH
}

char *nt_det(const void *A_, int want) {
    NT_TRY
    ZZ d;
    determinant(d, *(const mat_ZZ *)A_);
    return want ? zz_text(d) : NULL;
    NT_CATCH
}

long nt_sp_nbits(void) { return NTL_SP_NBITS; }
const char *nt_version(void) { return NTL_VERSION; }

// ---- mat_zz_p ----

struct ZP {
    zz_pContext ctx;
    mat_zz_p A, At;
};

void *nt_zp_new(const uint64_t *w, long r, long c, uint64_t p) {
    ZP *h = new ZP;
    h->ctx = zz_pContext((long)p);
    h->ctx.restore();
    h->A.SetDims(r, c);
    for (long i = 0; i < r; i++)
        for (long j = 0; j < c; j++) h->A[i][j] = conv<zz_p>((long)(w[i * c + j] % p));
    transpose(h->At, h->A);
    return h;
}

void nt_zp_free(void *h_) {
    ZP *h = (ZP *)h_;
    h->ctx.restore();
    delete h;
}

static uint64_t zp_digest(const mat_zz_p &X) {
    uint64_t h = LB_FNV_INIT;
    for (long i = 0; i < X.NumRows(); i++)
        for (long j = 0; j < X.NumCols(); j++) {
            uint64_t v = (uint64_t)rep(X[i][j]);
            h = lb_fnv_words(h, &v, 1);
        }
    return h;
}

// 0 mul(A, B), 2 rank (gauss on a copy), 3 det, 4 solve A·x = b (b the first column of B), 5
// inv, 7 kernel of the transpose (the right null space of A).
uint64_t nt_zp_op(int op, void *A_, void *B_, int want) {
    ZP *a = (ZP *)A_, *b = (ZP *)B_;
    a->ctx.restore();
    uint64_t h = 0;
    switch (op) {
    case 0: {
        mat_zz_p C;
        mul(C, a->A, b->A);
        if (want) h = zp_digest(C);
        break;
    }
    case 2: {
        mat_zz_p T(a->A);
        h = (uint64_t)gauss(T);
        break;
    }
    case 3: {
        zz_p d;
        determinant(d, a->A);
        h = (uint64_t)rep(d);
        break;
    }
    case 4: {
        zz_p d;
        vec_zz_p x, rhs;
        rhs.SetLength(b->A.NumRows());
        for (long i = 0; i < b->A.NumRows(); i++) rhs[i] = b->A[i][0];
        solve(d, a->A, x, rhs);
        if (want && !IsZero(d)) {
            uint64_t g = LB_FNV_INIT;
            for (long i = 0; i < x.length(); i++) {
                uint64_t v = (uint64_t)rep(x[i]);
                g = lb_fnv_words(g, &v, 1);
            }
            h = g;
        }
        break;
    }
    case 5: {
        zz_p d;
        mat_zz_p X;
        inv(d, X, a->A);
        if (want && !IsZero(d)) h = zp_digest(X);
        break;
    }
    case 7: {
        mat_zz_p X;
        kernel(X, a->At);
        h = (uint64_t)X.NumRows();
        break;
    }
    }
    return h;
}

// ---- mat_GF2 ----

void *nt_gf2_new(const uint64_t *w, long r, long c) {
    mat_GF2 *A = new mat_GF2;
    A->SetDims(r, c);
    long wpr = (c + 63) / 64;
    for (long i = 0; i < r; i++) {
        vec_GF2 &row = (*A)[i];
        // NTL keeps a row in words of NTL_BITS_PER_LONG bits, the element j in bit j % 64 of
        // word j / 64, as the crate does.
        for (long k = 0; k < wpr; k++) row.rep[k] = (_ntl_ulong)w[i * wpr + k];
    }
    return A;
}

void nt_gf2_free(void *A) { delete (mat_GF2 *)A; }

static uint64_t gf2_digest(const mat_GF2 &X) {
    uint64_t h = LB_FNV_INIT;
    long wpr = (X.NumCols() + 63) / 64;
    for (long i = 0; i < X.NumRows(); i++)
        h = lb_fnv_words(h, (const uint64_t *)X[i].rep.elts(), (size_t)wpr);
    return h;
}

// 0 mul, 2 rank (gauss), 3 det, 4 solve A·x = b (b the first column of B), 5 inv, 7 kernel of
// the transpose (its dimension), 10 transpose. B_ for 4 is a matrix whose first column is b;
// for 7 and 4 the caller passes the matrix the operation needs.
uint64_t nt_gf2_op(int op, const void *A_, const void *B_, int want) {
    const mat_GF2 &A = *(const mat_GF2 *)A_;
    uint64_t h = 0;
    switch (op) {
    case 0: {
        mat_GF2 C;
        mul(C, A, *(const mat_GF2 *)B_);
        if (want) h = gf2_digest(C);
        break;
    }
    case 2: {
        mat_GF2 T(A);
        h = (uint64_t)gauss(T);
        break;
    }
    case 3: {
        GF2 d;
        determinant(d, A);
        h = IsOne(d) ? 1 : 0;
        break;
    }
    case 4: {
        const mat_GF2 &Bm = *(const mat_GF2 *)B_;
        vec_GF2 x, rhs;
        rhs.SetLength(Bm.NumRows());
        for (long i = 0; i < Bm.NumRows(); i++) rhs[i] = Bm[i][0];
        GF2 d;
        solve(d, A, x, rhs);
        h = IsOne(d) ? 1 : 0;
        break;
    }
    case 5: {
        GF2 d;
        mat_GF2 X;
        inv(d, X, A);
        if (want && IsOne(d)) h = gf2_digest(X);
        break;
    }
    case 7: {
        mat_GF2 X;
        kernel(X, A);
        h = (uint64_t)X.NumRows();
        break;
    }
    case 10: {
        mat_GF2 X;
        transpose(X, A);
        if (want) h = gf2_digest(X);
        break;
    }
    }
    return h;
}

void *nt_gf2_transpose(const void *A_) {
    mat_GF2 *X = new mat_GF2;
    transpose(*X, *(const mat_GF2 *)A_);
    return X;
}

} // extern "C"
