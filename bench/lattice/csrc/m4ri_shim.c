// M4RI reference shim: products (mzd_mul, mzd_mul_m4rm), echelon forms (default, M4RI, PLUQ),
// inverse, solve, kernel, transpose, sum. Matrices come as packed words, the crate's layout
// (element j of a row in bit j % 64 of word j / 64), which is M4RI's.
#include "common.h"
#include <m4ri/m4ri.h>

static uint64_t mzd_digest(const mzd_t *X) {
    uint64_t h = LB_FNV_INIT;
    rci_t wpr = (X->ncols + 63) / 64;
    for (rci_t i = 0; i < X->nrows; i++)
        h = lb_fnv_words(h, (const uint64_t *)mzd_row((mzd_t *)X, i), (size_t)wpr);
    return h;
}

void *m4_new(const uint64_t *w, long r, long c) {
    mzd_t *A = mzd_init((rci_t)r, (rci_t)c);
    long wpr = (c + 63) / 64;
    for (long i = 0; i < r; i++) memcpy(mzd_row(A, (rci_t)i), w + i * wpr, (size_t)wpr * 8);
    return A;
}

void m4_free(void *A) { mzd_free((mzd_t *)A); }

// 0 mul (mzd_mul, cutoff arg; 0 its default), 1 mul_m4rm, 2 rank (echelonize, method arg: 0
// default, 1 M4RI, 2 PLUQ), 3 rref (full echelon form, same methods), 4 inverse (mzd_inv_m4ri),
// 5 solve A·X = B (mzd_solve_left, consistency checked), 6 kernel (mzd_kernel_left_pluq), 7
// transpose, 8 add.
uint64_t m4_op(int op, const void *A_, const void *B_, int arg, int want) {
    const mzd_t *A = A_, *B = B_;
    uint64_t h = 0;
    switch (op) {
    case 0: {
        mzd_t *C = mzd_mul(NULL, A, B, arg);
        if (want) h = mzd_digest(C);
        mzd_free(C);
        break;
    }
    case 1: {
        mzd_t *C = mzd_mul_m4rm(NULL, A, B, 0);
        if (want) h = mzd_digest(C);
        mzd_free(C);
        break;
    }
    case 2:
    case 3: {
        mzd_t *T = mzd_copy(NULL, A);
        int full = op == 3;
        rci_t r = arg == 1 ? mzd_echelonize_m4ri(T, full, 0)
                  : arg == 2 ? mzd_echelonize_pluq(T, full)
                             : mzd_echelonize(T, full);
        h = (uint64_t)r;
        if (want && full) h ^= mzd_digest(T);
        mzd_free(T);
        break;
    }
    case 4: {
        mzd_t *X = mzd_inv_m4ri(NULL, A, 0);
        if (want) h = mzd_digest(X);
        mzd_free(X);
        break;
    }
    case 5: {
        mzd_t *T = mzd_copy(NULL, A);
        mzd_t *X = mzd_copy(NULL, B);
        int bad = mzd_solve_left(T, X, 0, 1);
        if (want) h = bad ? 0 : mzd_digest(X);
        mzd_free(T);
        mzd_free(X);
        break;
    }
    case 6: {
        mzd_t *T = mzd_copy(NULL, A);
        mzd_t *X = mzd_kernel_left_pluq(T, 0);
        if (want) h = X ? (uint64_t)X->ncols : 0;
        if (X) mzd_free(X);
        mzd_free(T);
        break;
    }
    case 7: {
        mzd_t *X = mzd_transpose(NULL, A);
        if (want) h = mzd_digest(X);
        mzd_free(X);
        break;
    }
    case 8: {
        mzd_t *X = mzd_add(NULL, A, B);
        if (want) h = mzd_digest(X);
        mzd_free(X);
        break;
    }
    }
    return h;
}

// Row operations in place: 0 add row src to row dst, 1 swap rows a and b.
void m4_rowop(int op, void *A_, long a, long b) {
    mzd_t *A = A_;
    if (op == 0) mzd_row_add(A, (rci_t)a, (rci_t)b); else mzd_row_swap(A, (rci_t)a, (rci_t)b);
}
