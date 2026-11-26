// SuiteSparse references for the sparse scoreboard of nettrash-a.
//
// Thin wrappers with plain array arguments, so that the Rust harness can call each reference
// operation once per timed repetition. Matrices are compressed sparse COLUMN arrays with int32
// indices (int64 for SPQR's C interface). A function that makes a matrix has an `_h` form that
// returns a handle (read with ss_sp_*, released with ss_sp_free); the harness times the call and
// the release together, as a user pays both.
//
// Built by build.rs at -O3 -mcpu=native (or -march=native on x86_64).

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "SuiteSparseQR_C.h"
#include "amd.h"
#include "camd.h"
#include "cholmod.h"
#include "colamd.h"
#include "cs.h"
#include "klu.h"
#include "ldl.h"
#include "umfpack.h"

// ------------------------------------------------------------------------------------------------
// CHOLMOD commons: one int32, one int64 (for SPQR's C interface).

static cholmod_common C32;
static int C32_started = 0;
static cholmod_common C64;
static int C64_started = 0;

static cholmod_common *cc(void) {
    if (!C32_started) {
        cholmod_start(&C32);
        C32.print = 0;
        C32.nthreads_max = 1;
        C32_started = 1;
    }
    return &C32;
}

static cholmod_common *cc64(void) {
    if (!C64_started) {
        cholmod_l_start(&C64);
        C64.print = 0;
        C64.nthreads_max = 1;
        C64.SPQR_nthreads = 1;
        C64_started = 1;
    }
    return &C64;
}

static cholmod_sparse wrap(int nrow, int ncol, const int32_t *p, const int32_t *i, const double *x,
                           int stype) {
    cholmod_sparse A;
    memset(&A, 0, sizeof A);
    A.nrow = (size_t)nrow;
    A.ncol = (size_t)ncol;
    A.nzmax = (size_t)p[ncol];
    A.p = (void *)p;
    A.i = (void *)i;
    A.nz = NULL;
    A.x = (void *)x;
    A.z = NULL;
    A.stype = stype;
    A.itype = CHOLMOD_INT;
    A.xtype = x ? CHOLMOD_REAL : CHOLMOD_PATTERN;
    A.dtype = CHOLMOD_DOUBLE;
    A.sorted = 1;
    A.packed = 1;
    return A;
}

static cholmod_sparse wrap64(int64_t nrow, int64_t ncol, const int64_t *p, const int64_t *i,
                             const double *x) {
    cholmod_sparse A;
    memset(&A, 0, sizeof A);
    A.nrow = (size_t)nrow;
    A.ncol = (size_t)ncol;
    A.nzmax = (size_t)p[ncol];
    A.p = (void *)p;
    A.i = (void *)i;
    A.x = (void *)x;
    A.stype = 0;
    A.itype = CHOLMOD_LONG;
    A.xtype = CHOLMOD_REAL;
    A.dtype = CHOLMOD_DOUBLE;
    A.sorted = 1;
    A.packed = 1;
    return A;
}

static cholmod_dense wrapd(int nrow, int ncol, double *x) {
    cholmod_dense X;
    memset(&X, 0, sizeof X);
    X.nrow = (size_t)nrow;
    X.ncol = (size_t)ncol;
    X.nzmax = (size_t)nrow * (size_t)ncol;
    X.d = (size_t)nrow;
    X.x = x;
    X.xtype = CHOLMOD_REAL;
    X.dtype = CHOLMOD_DOUBLE;
    return X;
}

// ------------------------------------------------------------------------------------------------
// Sparse results: handles.

int ss_sp_nrow(void *h) { return (int)((cholmod_sparse *)h)->nrow; }
int ss_sp_ncol(void *h) { return (int)((cholmod_sparse *)h)->ncol; }
long ss_sp_nnz(void *h) { return (long)cholmod_nnz((cholmod_sparse *)h, cc()); }
// Copies a packed result out (pointers ncol+1, then indices and values).
void ss_sp_copy(void *h, int32_t *p, int32_t *i, double *x) {
    cholmod_sparse *A = (cholmod_sparse *)h;
    int32_t *Ap = (int32_t *)A->p;
    memcpy(p, Ap, sizeof(int32_t) * (A->ncol + 1));
    memcpy(i, A->i, sizeof(int32_t) * (size_t)Ap[A->ncol]);
    if (x && A->x) memcpy(x, A->x, sizeof(double) * (size_t)Ap[A->ncol]);
}
int ss_sp_sorted(void *h) { return ((cholmod_sparse *)h)->sorted; }
void ss_sp_free(void *h) {
    cholmod_sparse *A = (cholmod_sparse *)h;
    cholmod_free_sparse(&A, cc());
}

// ------------------------------------------------------------------------------------------------
// Products, sums, transposes, conversions.

// Y = A·X (transpose = 0) or Aᵀ·X (transpose = 1), X of k columns.
int ss_sdmult(int nrow, int ncol, const int32_t *p, const int32_t *i, const double *x, int transpose,
              int k, double *X, double *Y) {
    cholmod_sparse A = wrap(nrow, ncol, p, i, x, 0);
    cholmod_dense Xd = wrapd(transpose ? nrow : ncol, k, X);
    cholmod_dense Yd = wrapd(transpose ? ncol : nrow, k, Y);
    double one[2] = {1, 0}, zero[2] = {0, 0};
    return cholmod_sdmult(&A, transpose, one, zero, &Xd, &Yd, cc());
}

// C = A·B, sorted.
void *ss_ssmult_h(int am, int an, const int32_t *ap, const int32_t *ai, const double *ax, int bm,
                  int bn, const int32_t *bp, const int32_t *bi, const double *bx) {
    cholmod_sparse A = wrap(am, an, ap, ai, ax, 0);
    cholmod_sparse B = wrap(bm, bn, bp, bi, bx, 0);
    return cholmod_ssmult(&A, &B, 0, 1, 1, cc());
}

// C = A + B (sorted).
void *ss_add_h(int m, int n, const int32_t *ap, const int32_t *ai, const double *ax,
               const int32_t *bp, const int32_t *bi, const double *bx) {
    cholmod_sparse A = wrap(m, n, ap, ai, ax, 0);
    cholmod_sparse B = wrap(m, n, bp, bi, bx, 0);
    double one[2] = {1, 0};
    return cholmod_add(&A, &B, one, one, 1, 1, cc());
}

// C = Aᵀ (numerical).
void *ss_transpose_h(int m, int n, const int32_t *p, const int32_t *i, const double *x) {
    cholmod_sparse A = wrap(m, n, p, i, x, 0);
    return cholmod_transpose(&A, 1, cc());
}

// Triplets (duplicates summed) to compressed columns.
void *ss_triplet_h(int m, int n, long nnz, const int32_t *ti, const int32_t *tj, const double *tx) {
    cholmod_triplet T;
    memset(&T, 0, sizeof T);
    T.nrow = (size_t)m;
    T.ncol = (size_t)n;
    T.nzmax = (size_t)nnz;
    T.nnz = (size_t)nnz;
    T.i = (void *)ti;
    T.j = (void *)tj;
    T.x = (void *)tx;
    T.stype = 0;
    T.itype = CHOLMOD_INT;
    T.xtype = CHOLMOD_REAL;
    T.dtype = CHOLMOD_DOUBLE;
    return cholmod_triplet_to_sparse(&T, 0, cc());
}

// C = A(rset, cset); a size of -1 means all.
void *ss_submatrix_h(int m, int n, const int32_t *p, const int32_t *i, const double *x,
                     int32_t *rset, long rsize, int32_t *cset, long csize) {
    cholmod_sparse A = wrap(m, n, p, i, x, 0);
    return cholmod_submatrix(&A, rset, rsize, cset, csize, 1, 1, cc());
}

// Diagonals k1..k2 of A (k1 = 0, k2 = n: upper triangle; k1 = -m, k2 = 0: lower).
void *ss_band_h(int m, int n, const int32_t *p, const int32_t *i, const double *x, long k1,
                long k2) {
    cholmod_sparse A = wrap(m, n, p, i, x, 0);
    return cholmod_band(&A, k1, k2, 1, cc());
}

// norm 0: infinity, 1: one.
double ss_norm(int m, int n, const int32_t *p, const int32_t *i, const double *x, int norm) {
    cholmod_sparse A = wrap(m, n, p, i, x, 0);
    return cholmod_norm_sparse(&A, norm, cc());
}

// Writes the dense column-major matrix of A into out.
int ss_to_dense(int m, int n, const int32_t *p, const int32_t *i, const double *x, double *out) {
    cholmod_sparse A = wrap(m, n, p, i, x, 0);
    cholmod_dense *D = cholmod_sparse_to_dense(&A, cc());
    if (!D) return 0;
    memcpy(out, D->x, sizeof(double) * (size_t)m * (size_t)n);
    cholmod_free_dense(&D, cc());
    return 1;
}

void *ss_from_dense_h(int m, int n, double *x) {
    cholmod_dense X = wrapd(m, n, x);
    return cholmod_dense_to_sparse(&X, 1, cc());
}

// cholmod_symmetry, option 1: whether the matrix is symmetric (returns its kind).
int ss_symmetry(int n, const int32_t *p, const int32_t *i, const double *x) {
    cholmod_sparse A = wrap(n, n, p, i, x, 0);
    int32_t xmatched, pmatched, nzoffdiag, nzdiag;
    return cholmod_symmetry(&A, 1, &xmatched, &pmatched, &nzoffdiag, &nzdiag, cc());
}

// ------------------------------------------------------------------------------------------------
// CSparse (CXSparse, int32 'di' interface).

static cs_di cswrap(int m, int n, const int32_t *p, const int32_t *i, const double *x) {
    cs_di A;
    A.nzmax = p[n];
    A.m = m;
    A.n = n;
    A.p = (int32_t *)p;
    A.i = (int32_t *)i;
    A.x = (double *)x;
    A.nz = -1;
    return A;
}

long ss_cs_nnz(void *h) { cs_di *A = (cs_di *)h; return A->p[A->n]; }
void ss_cs_copy(void *h, int32_t *p, int32_t *i, double *x) {
    cs_di *A = (cs_di *)h;
    memcpy(p, A->p, sizeof(int32_t) * (size_t)(A->n + 1));
    memcpy(i, A->i, sizeof(int32_t) * (size_t)A->p[A->n]);
    if (x && A->x) memcpy(x, A->x, sizeof(double) * (size_t)A->p[A->n]);
}
void ss_cs_free(void *h) { cs_di_spfree((cs_di *)h); }

void *ss_cs_multiply_h(int am, int an, const int32_t *ap, const int32_t *ai, const double *ax,
                       int bm, int bn, const int32_t *bp, const int32_t *bi, const double *bx) {
    cs_di A = cswrap(am, an, ap, ai, ax), B = cswrap(bm, bn, bp, bi, bx);
    return cs_di_multiply(&A, &B);
}
void *ss_cs_add_h(int m, int n, const int32_t *ap, const int32_t *ai, const double *ax,
                  const int32_t *bp, const int32_t *bi, const double *bx) {
    cs_di A = cswrap(m, n, ap, ai, ax), B = cswrap(m, n, bp, bi, bx);
    return cs_di_add(&A, &B, 1.0, 1.0);
}
void *ss_cs_transpose_h(int m, int n, const int32_t *p, const int32_t *i, const double *x) {
    cs_di A = cswrap(m, n, p, i, x);
    return cs_di_transpose(&A, 1);
}
// Triplets to compressed columns, duplicates summed (cs_compress + cs_dupl); `sort` != 0 sorts
// the columns by a double transpose, as CSparse's documentation suggests.
void *ss_cs_compress_h(int m, int n, long nnz, const int32_t *ti, const int32_t *tj,
                       const double *tx, int sort) {
    cs_di T;
    T.nzmax = (int32_t)nnz;
    T.m = m;
    T.n = n;
    T.p = (int32_t *)tj;
    T.i = (int32_t *)ti;
    T.x = (double *)tx;
    T.nz = (int32_t)nnz;
    cs_di *C = cs_di_compress(&T);
    if (!C) return NULL;
    cs_di_dupl(C);
    if (sort) {
        cs_di *D = cs_di_transpose(C, 1);
        cs_di_spfree(C);
        C = cs_di_transpose(D, 1);
        cs_di_spfree(D);
    }
    return C;
}
// C = P·A·Qᵀ with pinv (inverse row permutation) and q (column permutation); unsorted columns.
void *ss_cs_permute_h(int m, int n, const int32_t *p, const int32_t *i, const double *x,
                      const int32_t *pinv, const int32_t *q) {
    cs_di A = cswrap(m, n, p, i, x);
    return cs_di_permute(&A, pinv, q, 1);
}
// Elimination tree of A (upper triangle used) and its postorder; returns the sum of parents
// (a checksum).
long ss_cs_etree(int n, const int32_t *p, const int32_t *i, int32_t *parent) {
    cs_di A = cswrap(n, n, p, i, NULL);
    int32_t *par = cs_di_etree(&A, 0);
    if (!par) return -1;
    memcpy(parent, par, sizeof(int32_t) * (size_t)n);
    cs_di_free(par);
    return 0;
}
long ss_cs_post(int n, const int32_t *parent, int32_t *post) {
    int32_t *po = cs_di_post(parent, n);
    if (!po) return -1;
    memcpy(post, po, sizeof(int32_t) * (size_t)n);
    cs_di_free(po);
    return 0;
}

// CSparse LU: order 0 natural, 1 amd(A+Aᵀ), 2 amd(AᵀA without dense rows), 3 amd(AᵀA).
typedef struct {
    cs_dis *S;
    cs_din *N;
    int n;
} cslu;
void *ss_cs_lu(int n, const int32_t *p, const int32_t *i, const double *x, int order, double tol) {
    cs_di A = cswrap(n, n, p, i, x);
    cslu *h = (cslu *)calloc(1, sizeof(cslu));
    h->n = n;
    h->S = cs_di_sqr(order, &A, 0);
    h->N = cs_di_lu(&A, h->S, tol);
    if (!h->N) {
        cs_di_sfree(h->S);
        free(h);
        return NULL;
    }
    return h;
}
long ss_cs_lu_nnz(void *hh) {
    cslu *h = (cslu *)hh;
    return h->N->L->p[h->n] + h->N->U->p[h->n];
}
void ss_cs_lu_solve(void *hh, const double *b, double *x, double *work) {
    cslu *h = (cslu *)hh;
    cs_di_ipvec(h->N->pinv, b, work, h->n);
    cs_di_lsolve(h->N->L, work);
    cs_di_usolve(h->N->U, work);
    cs_di_ipvec(h->S->q, work, x, h->n);
}
void ss_cs_lu_free(void *hh) {
    cslu *h = (cslu *)hh;
    cs_di_sfree(h->S);
    cs_di_nfree(h->N);
    free(h);
}

// CSparse Cholesky (up-looking, simplicial) with AMD of A+Aᵀ (order 1) or natural (0); reads the
// upper triangle of A.
void *ss_cs_chol(int n, const int32_t *p, const int32_t *i, const double *x, int order) {
    cs_di A = cswrap(n, n, p, i, x);
    cslu *h = (cslu *)calloc(1, sizeof(cslu));
    h->n = n;
    h->S = cs_di_schol(order, &A);
    h->N = cs_di_chol(&A, h->S);
    if (!h->N) {
        cs_di_sfree(h->S);
        free(h);
        return NULL;
    }
    return h;
}
long ss_cs_chol_nnz(void *hh) {
    cslu *h = (cslu *)hh;
    return h->N->L->p[h->n];
}
void ss_cs_chol_solve(void *hh, const double *b, double *x, double *work) {
    cslu *h = (cslu *)hh;
    cs_di_ipvec(h->S->pinv, b, work, h->n);
    cs_di_lsolve(h->N->L, work);
    cs_di_ltsolve(h->N->L, work);
    cs_di_pvec(h->S->pinv, work, x, h->n);
}
void ss_cs_chol_free(void *hh) { ss_cs_lu_free(hh); }

// CSparse QR (left-looking Householder) of an m × n matrix, m ≥ n, order 3 = amd(AᵀA).
typedef struct {
    cs_dis *S;
    cs_din *N;
    int m, n;
} csqr;
void *ss_cs_qr(int m, int n, const int32_t *p, const int32_t *i, const double *x, int order) {
    cs_di A = cswrap(m, n, p, i, x);
    csqr *h = (csqr *)calloc(1, sizeof(csqr));
    h->m = m;
    h->n = n;
    h->S = cs_di_sqr(order, &A, 1);
    h->N = cs_di_qr(&A, h->S);
    if (!h->N) {
        cs_di_sfree(h->S);
        free(h);
        return NULL;
    }
    return h;
}
long ss_cs_qr_nnz(void *hh) {
    csqr *h = (csqr *)hh;
    return h->N->L->p[h->n] + h->N->U->p[h->n];
}
// Least squares: x = R \ (Qᵀ b), as cs_qrsol does; work of S->m2 entries.
void ss_cs_qr_solve(void *hh, const double *b, double *x, double *work) {
    csqr *h = (csqr *)hh;
    int k;
    for (k = 0; k < h->S->m2; k++) work[k] = 0;
    cs_di_ipvec(h->S->pinv, b, work, h->m);
    for (k = 0; k < h->n; k++) cs_di_happly(h->N->L, k, h->N->B[k], work);
    cs_di_usolve(h->N->U, work);
    cs_di_ipvec(h->S->q, work, x, h->n);
}
int ss_cs_qr_m2(void *hh) { return ((csqr *)hh)->S->m2; }
void ss_cs_qr_free(void *hh) {
    csqr *h = (csqr *)hh;
    cs_di_sfree(h->S);
    cs_di_nfree(h->N);
    free(h);
}

// ------------------------------------------------------------------------------------------------
// Orderings.

// AMD with its defaults; returns nnz(L) as AMD counts it (Info[AMD_LNZ], diagonal excluded).
double ss_amd(int n, const int32_t *p, const int32_t *i, int32_t *perm) {
    double Control[AMD_CONTROL], Info[AMD_INFO];
    amd_defaults(Control);
    if (amd_order(n, p, i, perm, Control, Info) < 0) return -1;
    return Info[AMD_LNZ];
}

double ss_camd(int n, const int32_t *p, const int32_t *i, int32_t *perm) {
    double Control[CAMD_CONTROL], Info[CAMD_INFO];
    camd_defaults(Control);
    if (camd_order(n, p, i, perm, Control, Info, NULL) < 0) return -1;
    return Info[CAMD_LNZ];
}

// COLAMD of an m × n matrix (compressed columns); the copy into colamd's workspace is part of the
// call, as every user must make it.
int ss_colamd(int m, int n, const int32_t *p, const int32_t *i, int32_t *perm) {
    int32_t nnz = p[n];
    size_t alen = colamd_recommended(nnz, m, n);
    int32_t *A = (int32_t *)malloc(sizeof(int32_t) * alen);
    memcpy(A, i, sizeof(int32_t) * (size_t)nnz);
    memcpy(perm, p, sizeof(int32_t) * (size_t)(n + 1));
    int32_t stats[COLAMD_STATS];
    int ok = colamd(m, n, (int32_t)alen, A, perm, NULL, stats);
    free(A);
    return ok;
}

int ss_symamd(int n, const int32_t *p, const int32_t *i, int32_t *perm) {
    int32_t stats[COLAMD_STATS];
    return symamd(n, (int32_t *)i, (int32_t *)p, perm, NULL, stats, calloc, free);
}

// METIS's nested dissection, as CHOLMOD calls it (METIS_NodeND), of the symmetric A (lower part
// read); no postorder, as METIS gives it.
int ss_metis(int n, const int32_t *p, const int32_t *i, int32_t *perm, int postorder) {
    cholmod_sparse A = wrap(n, n, p, i, NULL, -1);
    return cholmod_metis(&A, NULL, 0, postorder, perm, cc());
}

// CHOLMOD's nested dissection (NESDIS: METIS bisections, CAMD on the leaves and separators).
long ss_nesdis(int n, const int32_t *p, const int32_t *i, int32_t *perm, int32_t *cparent,
               int32_t *cmember) {
    cholmod_sparse A = wrap(n, n, p, i, NULL, -1);
    return (long)cholmod_nested_dissection(&A, NULL, 0, perm, cparent, cmember, cc());
}

// The elimination tree of the symmetric A (upper triangle of the compressed columns read).
int ss_etree(int n, const int32_t *p, const int32_t *i, int32_t *parent) {
    cholmod_sparse A = wrap(n, n, p, i, NULL, 1);
    return cholmod_etree(&A, parent, cc());
}

long ss_postorder(int n, const int32_t *parent, int32_t *post) {
    return (long)cholmod_postorder((int32_t *)parent, n, NULL, post, cc());
}

// ------------------------------------------------------------------------------------------------
// CHOLMOD Cholesky.
//
// mode: 0 simplicial, 1 automatic (CHOLMOD's default), 2 supernodal.
// ll: 1 for L·Lᵀ, 0 for L·D·Lᵀ (simplicial only: CHOLMOD's supernodal factor is always L·Lᵀ).
// ordering: -1 CHOLMOD's default strategy (AMD, then METIS when AMD fills much), else one
// method: 0 natural, 1 the given perm, 2 AMD, 3 METIS, 4 NESDIS.
// A is the upper triangle in compressed columns (stype 1).

void *ss_chol(int n, const int32_t *p, const int32_t *i, const double *x, int mode, int ll,
              int ordering, const int32_t *perm) {
    cholmod_common *c = cc();
    c->supernodal = mode == 0 ? CHOLMOD_SIMPLICIAL : (mode == 1 ? CHOLMOD_AUTO : CHOLMOD_SUPERNODAL);
    c->final_ll = ll;
    if (ordering < 0) {
        c->nmethods = 0;
    } else {
        c->nmethods = 1;
        c->method[0].ordering = ordering;
    }
    c->postorder = 1;
    cholmod_sparse A = wrap(n, n, p, i, x, 1);
    cholmod_factor *L =
        ordering == CHOLMOD_GIVEN ? cholmod_analyze_p(&A, (int32_t *)perm, NULL, 0, c)
                                  : cholmod_analyze(&A, c);
    if (!L) return NULL;
    if (!cholmod_factorize(&A, L, c) || c->status != CHOLMOD_OK) {
        cholmod_free_factor(&L, c);
        return NULL;
    }
    return L;
}

double ss_chol_lnz(void *h) {
    cholmod_factor *L = (cholmod_factor *)h;
    // The nonzeros of L as the symbolic analysis counts them (diagonal included).
    double s = 0;
    int32_t *cc_ = (int32_t *)L->ColCount;
    for (size_t j = 0; j < L->n; j++) s += cc_[j];
    return s;
}
int ss_chol_is_super(void *h) { return ((cholmod_factor *)h)->is_super; }
long ss_chol_nsuper(void *h) { return (long)((cholmod_factor *)h)->nsuper; }

// X = A⁻¹·B, B of k columns.
int ss_chol_solve(void *h, int n, int k, double *b, double *x) {
    cholmod_factor *L = (cholmod_factor *)h;
    cholmod_dense B = wrapd(n, k, b);
    cholmod_dense *X = cholmod_solve(CHOLMOD_A, L, &B, cc());
    if (!X) return 0;
    memcpy(x, X->x, sizeof(double) * (size_t)n * (size_t)k);
    cholmod_free_dense(&X, cc());
    return 1;
}
// The same through cholmod_solve2, which reuses its workspace: what a user solving repeatedly
// would call.
static cholmod_dense *S2_X = NULL, *S2_Y = NULL, *S2_E = NULL;
int ss_chol_solve2(void *h, int n, int k, double *b, double *x) {
    cholmod_factor *L = (cholmod_factor *)h;
    cholmod_dense B = wrapd(n, k, b);
    if (!cholmod_solve2(CHOLMOD_A, L, &B, NULL, &S2_X, NULL, &S2_Y, &S2_E, cc())) return 0;
    memcpy(x, S2_X->x, sizeof(double) * (size_t)n * (size_t)k);
    return 1;
}
void ss_chol_free(void *h) {
    cholmod_factor *L = (cholmod_factor *)h;
    cholmod_free_factor(&L, cc());
}

// ------------------------------------------------------------------------------------------------
// LDL (Davis's concise up-looking L·D·Lᵀ), with the given permutation (or none).

typedef struct {
    int n;
    int32_t *Lp, *Li, *Parent, *Lnz, *Flag, *Pattern, *P, *Pinv;
    double *Lx, *D, *Y;
} ldlh;

void *ss_ldl(int n, const int32_t *p, const int32_t *i, const double *x, const int32_t *perm) {
    ldlh *h = (ldlh *)calloc(1, sizeof(ldlh));
    h->n = n;
    h->Lp = (int32_t *)malloc(sizeof(int32_t) * (size_t)(n + 1));
    h->Parent = (int32_t *)malloc(sizeof(int32_t) * (size_t)n);
    h->Lnz = (int32_t *)malloc(sizeof(int32_t) * (size_t)n);
    h->Flag = (int32_t *)malloc(sizeof(int32_t) * (size_t)n);
    h->Pattern = (int32_t *)malloc(sizeof(int32_t) * (size_t)n);
    h->D = (double *)malloc(sizeof(double) * (size_t)n);
    h->Y = (double *)malloc(sizeof(double) * (size_t)n);
    if (perm) {
        h->P = (int32_t *)malloc(sizeof(int32_t) * (size_t)n);
        h->Pinv = (int32_t *)malloc(sizeof(int32_t) * (size_t)n);
        memcpy(h->P, perm, sizeof(int32_t) * (size_t)n);
    }
    ldl_symbolic(n, (int32_t *)p, (int32_t *)i, h->Lp, h->Parent, h->Lnz, h->Flag, h->P, h->Pinv);
    int32_t lnz = h->Lp[n];
    h->Li = (int32_t *)malloc(sizeof(int32_t) * (size_t)(lnz > 0 ? lnz : 1));
    h->Lx = (double *)malloc(sizeof(double) * (size_t)(lnz > 0 ? lnz : 1));
    int d = ldl_numeric(n, (int32_t *)p, (int32_t *)i, (double *)x, h->Lp, h->Parent, h->Lnz, h->Li,
                        h->Lx, h->D, h->Y, h->Pattern, h->Flag, h->P, h->Pinv);
    if (d != n) {
        // A zero pivot: give the handle back anyway, flagged by a NULL D.
        free(h->D);
        h->D = NULL;
    }
    return h;
}
int ss_ldl_ok(void *hh) { return ((ldlh *)hh)->D != NULL; }
long ss_ldl_lnz(void *hh) { ldlh *h = (ldlh *)hh; return h->Lp[h->n]; }
void ss_ldl_solve(void *hh, const double *b, double *x) {
    ldlh *h = (ldlh *)hh;
    int n = h->n;
    if (h->P) {
        ldl_perm(n, h->Y, (double *)b, h->P);
        ldl_lsolve(n, h->Y, h->Lp, h->Li, h->Lx);
        ldl_dsolve(n, h->Y, h->D);
        ldl_ltsolve(n, h->Y, h->Lp, h->Li, h->Lx);
        ldl_permt(n, x, h->Y, h->P);
    } else {
        memcpy(x, b, sizeof(double) * (size_t)n);
        ldl_lsolve(n, x, h->Lp, h->Li, h->Lx);
        ldl_dsolve(n, x, h->D);
        ldl_ltsolve(n, x, h->Lp, h->Li, h->Lx);
    }
}
void ss_ldl_free(void *hh) {
    ldlh *h = (ldlh *)hh;
    free(h->Lp); free(h->Li); free(h->Lx); free(h->Parent); free(h->Lnz); free(h->Flag);
    free(h->Pattern); free(h->D); free(h->Y); free(h->P); free(h->Pinv);
    free(h);
}

// ------------------------------------------------------------------------------------------------
// UMFPACK.
//
// ordering: -1 the default (UMFPACK_ORDERING_AMD... as umfpack_di_defaults sets), else
// UMFPACK_ORDERING_* (0 CHOLMOD, 1 AMD, 2 given, 3 METIS, 4 best, 5 none, 6 user);
// strategy: -1 default (auto), 1 unsymmetric, 3 symmetric; pivot_tol < 0: default (0.1);
// scale: -1 default (sum), 0 none.

typedef struct {
    void *Symbolic, *Numeric;
    int n;
    const int32_t *p, *i;
    const double *x;
    double Control[UMFPACK_CONTROL];
    double Info[UMFPACK_INFO];
} umfh;

void *ss_umf(int n, const int32_t *p, const int32_t *i, const double *x, int ordering, int strategy,
             double pivot_tol, int scale, const int32_t *qinit) {
    umfh *h = (umfh *)calloc(1, sizeof(umfh));
    h->n = n;
    h->p = p;
    h->i = i;
    h->x = x;
    umfpack_di_defaults(h->Control);
    if (ordering >= 0) h->Control[UMFPACK_ORDERING] = ordering;
    if (strategy >= 0) h->Control[UMFPACK_STRATEGY] = strategy;
    if (pivot_tol >= 0) h->Control[UMFPACK_PIVOT_TOLERANCE] = pivot_tol;
    if (scale >= 0) h->Control[UMFPACK_SCALE] = scale;
    double Info[UMFPACK_INFO];
    int st;
    if (qinit)
        st = umfpack_di_qsymbolic(n, n, p, i, x, qinit, &h->Symbolic, h->Control, Info);
    else
        st = umfpack_di_symbolic(n, n, p, i, x, &h->Symbolic, h->Control, Info);
    if (st != UMFPACK_OK) {
        free(h);
        return NULL;
    }
    st = umfpack_di_numeric(p, i, x, h->Symbolic, &h->Numeric, h->Control, h->Info);
    if (st != UMFPACK_OK) {
        umfpack_di_free_symbolic(&h->Symbolic);
        if (h->Numeric) umfpack_di_free_numeric(&h->Numeric);
        free(h);
        return NULL;
    }
    return h;
}
long ss_umf_nnz(void *hh) {
    umfh *h = (umfh *)hh;
    int32_t lnz, unz, nr, nc, nzud;
    umfpack_di_get_lunz(&lnz, &unz, &nr, &nc, &nzud, h->Numeric);
    // L's unit diagonal is counted in lnz; U's diagonal in unz.
    return (long)lnz + (long)unz;
}
// sys 0: A·x = b, 1: Aᵀ·x = b.
int ss_umf_solve(void *hh, int sys, const double *b, double *x) {
    umfh *h = (umfh *)hh;
    double Info[UMFPACK_INFO];
    return umfpack_di_solve(sys == 0 ? UMFPACK_A : UMFPACK_At, h->p, h->i, h->x, x, b, h->Numeric,
                            h->Control, Info);
}
// The reciprocal condition estimate UMFPACK reports (min |U_ii| / max |U_ii|).
double ss_umf_rcond(void *hh) { return ((umfh *)hh)->Info[UMFPACK_RCOND]; }
void ss_umf_free(void *hh) {
    umfh *h = (umfh *)hh;
    umfpack_di_free_symbolic(&h->Symbolic);
    umfpack_di_free_numeric(&h->Numeric);
    free(h);
}

// ------------------------------------------------------------------------------------------------
// KLU. ordering 0 AMD (default), 1 COLAMD; btf 1 (default) or 0; tol: partial pivoting
// threshold (default 0.001); scale: -1 default (2: max), 0 none.

typedef struct {
    klu_common C;
    klu_symbolic *S;
    klu_numeric *N;
    int n;
    double *work;
} kluh;

void *ss_klu(int n, const int32_t *p, const int32_t *i, const double *x, int ordering, int btf,
             double tol, int scale) {
    kluh *h = (kluh *)calloc(1, sizeof(kluh));
    h->n = n;
    klu_defaults(&h->C);
    h->C.ordering = ordering;
    h->C.btf = btf;
    if (tol >= 0) h->C.tol = tol;
    if (scale >= 0) h->C.scale = scale;
    h->S = klu_analyze(n, (int32_t *)p, (int32_t *)i, &h->C);
    if (!h->S) {
        free(h);
        return NULL;
    }
    h->N = klu_factor((int32_t *)p, (int32_t *)i, (double *)x, h->S, &h->C);
    if (!h->N) {
        klu_free_symbolic(&h->S, &h->C);
        free(h);
        return NULL;
    }
    return h;
}
long ss_klu_nnz(void *hh) {
    kluh *h = (kluh *)hh;
    // lnz and unz include the diagonals; the off-diagonal blocks of the BTF form are nzoff.
    return (long)h->N->lnz + (long)h->N->unz + (long)h->N->nzoff;
}
// sys 0: A·x = b, 1: Aᵀ·x = b; in place on x (b copied first).
int ss_klu_solve(void *hh, int sys, const double *b, double *x) {
    kluh *h = (kluh *)hh;
    memcpy(x, b, sizeof(double) * (size_t)h->n);
    if (sys == 0) return klu_solve(h->S, h->N, h->n, 1, x, &h->C);
    return klu_tsolve(h->S, h->N, h->n, 1, x, &h->C);
}
double ss_klu_rcond(void *hh, const int32_t *p, const int32_t *i, const double *x) {
    kluh *h = (kluh *)hh;
    klu_condest((int32_t *)p, (double *)x, h->S, h->N, &h->C);
    return h->C.condest;
}
void ss_klu_free(void *hh) {
    kluh *h = (kluh *)hh;
    klu_free_numeric(&h->N, &h->C);
    klu_free_symbolic(&h->S, &h->C);
    free(h);
}

// ------------------------------------------------------------------------------------------------
// SPQR through its C interface (int64). ordering: SPQR_ORDERING_* (2 COLAMD, 7 default, 1 given
// is treated as fixed, 0 fixed, 6 AMD of AᵀA, 3 best...). tol: -2 default.

void *ss_spqr(int64_t m, int64_t n, const int64_t *p, const int64_t *i, const double *x,
              int ordering, double tol) {
    cholmod_sparse A = wrap64(m, n, p, i, x);
    return SuiteSparseQR_C_factorize(ordering, tol, &A, cc64());
}
// Least squares: x = E·(R \ (Qᵀ·b)).
int ss_spqr_solve(void *h, int64_t m, int64_t n, int k, double *b, double *x) {
    SuiteSparseQR_C_factorization *QR = (SuiteSparseQR_C_factorization *)h;
    cholmod_dense B;
    memset(&B, 0, sizeof B);
    B.nrow = (size_t)m;
    B.ncol = (size_t)k;
    B.nzmax = (size_t)m * (size_t)k;
    B.d = (size_t)m;
    B.x = b;
    B.xtype = CHOLMOD_REAL;
    B.dtype = CHOLMOD_DOUBLE;
    cholmod_dense *Y = SuiteSparseQR_C_qmult(SPQR_QTX, QR, &B, cc64());
    if (!Y) return 0;
    cholmod_dense *X = SuiteSparseQR_C_solve(SPQR_RETX_EQUALS_B, QR, Y, cc64());
    cholmod_l_free_dense(&Y, cc64());
    if (!X) return 0;
    memcpy(x, X->x, sizeof(double) * (size_t)n * (size_t)k);
    cholmod_l_free_dense(&X, cc64());
    return 1;
}
long ss_spqr_rank(void) { return (long)cc64()->SPQR_istat[4]; }
void ss_spqr_free(void *h) {
    SuiteSparseQR_C_factorization *QR = (SuiteSparseQR_C_factorization *)h;
    SuiteSparseQR_C_free(&QR, cc64());
}
// nnz(R) and nnz(H) of the last factorisation, as SPQR reports them.
double ss_spqr_nnzR(void) { return cc64()->SPQR_istat[0]; }
double ss_spqr_nnzH(void) { return cc64()->SPQR_istat[1]; }

// R of A in the column order given (SPQR_ORDERING_FIXED), without rank detection (SPQR_NO_TOL),
// n×n in compressed columns, for a comparison of R up to the signs of its rows. The arrays are
// malloc'd copies (free them with ss_free); returns the rank SPQR reports, -1 on failure.
int64_t ss_spqr_r(int64_t m, int64_t n, const int64_t *p, const int64_t *i, const double *x,
                  int64_t **rp, int64_t **ri, double **rx) {
    cholmod_sparse A = wrap64(m, n, p, i, x);
    cholmod_sparse *R = NULL;
    int64_t *E = NULL;
    int64_t rank = SuiteSparseQR_C(SPQR_ORDERING_FIXED, SPQR_NO_TOL, n, 0, &A, NULL, NULL, NULL,
                                   NULL, &R, &E, NULL, NULL, NULL, cc64());
    if (rank < 0 || !R) return -1;
    if (E) {
        // A column permutation after all (singletons): not comparable row by row.
        for (int64_t k = 0; k < n; k++)
            if (E[k] != k) { free(E); cholmod_l_free_sparse(&R, cc64()); return -2; }
        free(E);
    }
    int64_t nc = (int64_t)R->ncol, nz = ((int64_t *)R->p)[nc];
    *rp = malloc(sizeof(int64_t) * (size_t)(nc + 1));
    *ri = malloc(sizeof(int64_t) * (size_t)(nz > 0 ? nz : 1));
    *rx = malloc(sizeof(double) * (size_t)(nz > 0 ? nz : 1));
    memcpy(*rp, R->p, sizeof(int64_t) * (size_t)(nc + 1));
    memcpy(*ri, R->i, sizeof(int64_t) * (size_t)nz);
    memcpy(*rx, R->x, sizeof(double) * (size_t)nz);
    cholmod_l_free_sparse(&R, cc64());
    return rank;
}
void ss_free(void *p) { free(p); }
