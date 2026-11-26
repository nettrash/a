// Apple Accelerate (Sparse Solvers) references for the sparse scoreboard of nettrash-a.
// macOS only; built by build.rs at -O3 -mcpu=native when the target is Apple.
//
// Matrices are compressed sparse columns: long column starts, int row indices.

#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

static void quiet(const char *message) { (void)message; }

static SparseMatrix_Double mat(int m, int n, long *cs, int *ri, double *x, int kind_sym,
                               int transpose) {
    SparseMatrix_Double A;
    memset(&A, 0, sizeof A);
    A.structure.rowCount = m;
    A.structure.columnCount = n;
    A.structure.columnStarts = cs;
    A.structure.rowIndices = ri;
    A.structure.attributes.kind = kind_sym ? SparseSymmetric : SparseOrdinary;
    A.structure.attributes.triangle = SparseLowerTriangle;
    A.structure.attributes.transpose = transpose ? 1 : 0;
    A.structure.blockSize = 1;
    A.data = x;
    return A;
}

int ac_single_threaded(void) {
    // BLASSetThreading is available from macOS 15.
    if (__builtin_available(macOS 15.0, *)) {
        return BLASSetThreading(BLAS_THREADING_SINGLE_THREADED);
    }
    return -1;
}

// Y = A·X (transpose = 0) or Aᵀ·X (transpose = 1), X of k columns, column-major.
void ac_mul(int m, int n, long *cs, int *ri, double *x, int transpose, int k, double *X,
            double *Y) {
    SparseMatrix_Double A = mat(m, n, cs, ri, x, 0, transpose);
    int ym = transpose ? n : m, xm = transpose ? m : n;
    if (k == 1) {
        DenseVector_Double xv = {xm, X}, yv = {ym, Y};
        SparseMultiply(A, xv, yv);
    } else {
        DenseMatrix_Double Xd = {xm, k, xm, {0}, X}, Yd = {ym, k, ym, {0}, Y};
        SparseMultiply(A, Xd, Yd);
    }
}

void ac_mul_f32(int m, int n, long *cs, int *ri, float *x, int transpose, float *X, float *Y) {
    SparseMatrix_Float A;
    memset(&A, 0, sizeof A);
    A.structure.rowCount = m;
    A.structure.columnCount = n;
    A.structure.columnStarts = cs;
    A.structure.rowIndices = ri;
    A.structure.attributes.kind = SparseOrdinary;
    A.structure.attributes.transpose = transpose ? 1 : 0;
    A.structure.blockSize = 1;
    A.data = x;
    int ym = transpose ? n : m, xm = transpose ? m : n;
    DenseVector_Float xv = {xm, X}, yv = {ym, Y};
    SparseMultiply(A, xv, yv);
}

// Triplets (duplicates summed) to compressed columns; returns the number of stored elements and
// releases the matrix (the conversion is what is timed).
long ac_from_coo(int m, int n, long nnz, const int *ri, const int *ci, const double *x) {
    SparseAttributes_t att;
    memset(&att, 0, sizeof att);
    att.kind = SparseOrdinary;
    SparseMatrix_Double A = SparseConvertFromCoordinate(m, n, nnz, 1, att, ri, ci, x);
    long s = A.structure.columnStarts[n];
    SparseCleanup(A);
    return s;
}

// Factorisations. type: 0 Cholesky, 1 LDLT (pivoted), 2 LDLT unpivoted, 3 LDLT SBK, 4 LDLT TPP,
// 40 QR, 41 Cholesky of AᵀA, 50.. LU (see Solve.h); kind_sym: the matrix is symmetric and its
// lower triangle is given. order: 0 default, 1 user (perm), 2 AMD, 3 METIS, 4 COLAMD.
typedef struct {
    SparseOpaqueFactorization_Double F;
    int m, n;
} ach;

void *ac_factor(int type, int m, int n, long *cs, int *ri, double *x, int kind_sym, int order,
                int *perm) {
    SparseMatrix_Double A = mat(m, n, cs, ri, x, kind_sym, 0);
    SparseSymbolicFactorOptions so;
    memset(&so, 0, sizeof so);
    so.control = SparseDefaultControl;
    so.orderMethod = (SparseOrder_t)order;
    so.order = perm;
    so.ignoreRowsAndColumns = NULL;
    so.malloc = malloc;
    so.free = free;
    so.reportError = quiet;
    SparseNumericFactorOptions no;
    memset(&no, 0, sizeof no);
    no.control = SparseDefaultControl;
    no.scalingMethod = SparseScalingDefault;
    no.pivotTolerance = 0.01;
    no.zeroTolerance = 1e-4 * 2.220446049250313e-16;
    ach *h = (ach *)calloc(1, sizeof(ach));
    h->m = m;
    h->n = n;
    h->F = SparseFactor((SparseFactorization_t)type, A, so, no);
    if (h->F.status != SparseStatusOK) {
        SparseCleanup(h->F);
        free(h);
        return NULL;
    }
    return h;
}

// The default numeric options of SparseFactor (pivot tolerance 0.01 is Accelerate's documented
// default for LDLᵀ TPP; the plain two-argument call is used for the default-factor rows).
void *ac_factor_default(int type, int m, int n, long *cs, int *ri, double *x, int kind_sym) {
    SparseMatrix_Double A = mat(m, n, cs, ri, x, kind_sym, 0);
    ach *h = (ach *)calloc(1, sizeof(ach));
    h->m = m;
    h->n = n;
    h->F = SparseFactor((SparseFactorization_t)type, A);
    if (h->F.status != SparseStatusOK) {
        SparseCleanup(h->F);
        free(h);
        return NULL;
    }
    return h;
}

double ac_factor_size(void *hh) {
    ach *h = (ach *)hh;
    return (double)h->F.symbolicFactorization.factorSize_Double;
}

// x = A⁻¹·b (or the least-squares solution for QR), B of k columns: b is m × k, x is n × k.
void ac_solve(void *hh, int k, double *b, double *x) {
    ach *h = (ach *)hh;
    if (k == 1) {
        DenseVector_Double bv = {h->m, b}, xv = {h->n, x};
        SparseSolve(h->F, bv, xv);
    } else {
        DenseMatrix_Double B = {h->m, k, h->m, {0}, b}, X = {h->n, k, h->n, {0}, x};
        SparseSolve(h->F, B, X);
    }
}

void ac_free(void *hh) {
    ach *h = (ach *)hh;
    SparseCleanup(h->F);
    free(h);
}

// Iterative methods. method 0: CG, 1: GMRES (variant, nvec), precond 0 none, 2 diagonal.
// Returns the status (0 converged); x holds the initial guess on entry.
int ac_iterative(int method, int variant, int nvec, int n, long *cs, int *ri, double *x_, int sym,
                 double *b, double *x, double rtol, int maxit, int precond) {
    SparseMatrix_Double A = mat(n, n, cs, ri, x_, sym, 0);
    DenseMatrix_Double B = {n, 1, n, {0}, b}, X = {n, 1, n, {0}, x};
    SparseIterativeMethod M;
    if (method == 0) {
        SparseCGOptions o;
        memset(&o, 0, sizeof o);
        o.maxIterations = maxit;
        o.atol = 0.0;
        o.rtol = rtol;
        o.reportError = quiet;
        M = SparseConjugateGradient(o);
    } else {
        SparseGMRESOptions o;
        memset(&o, 0, sizeof o);
        o.variant = (SparseGMRESVariant_t)variant;
        o.nvec = nvec;
        o.maxIterations = maxit;
        o.atol = 0.0;
        o.rtol = rtol;
        o.reportError = quiet;
        M = SparseGMRES(o);
    }
    // SparsePreconditionerNone through the preconditioned overload is refused at once: without
    // a preconditioner the plain overload is called.
    if (precond == 0) return SparseSolve(M, A, B, X);
    return SparseSolve(M, A, B, X, (SparsePreconditioner_t)precond);
}

// The same method through a counting operator: the number of products with A it made (and the
// diagonal preconditioner when precond is 2).
static long ac_count = 0;
int ac_iterative_count(int method, int variant, int nvec, int n, long *cs, int *ri, double *x_,
                       int sym, double *b, double *x, double rtol, int maxit, int precond,
                       long *products) {
    SparseMatrix_Double A = mat(n, n, cs, ri, x_, sym, 0);
    DenseMatrix_Double B = {n, 1, n, {0}, b}, X = {n, 1, n, {0}, x};
    SparseIterativeMethod M;
    if (method == 0) {
        SparseCGOptions o;
        memset(&o, 0, sizeof o);
        o.maxIterations = maxit;
        o.rtol = rtol;
        o.reportError = quiet;
        M = SparseConjugateGradient(o);
    } else {
        SparseGMRESOptions o;
        memset(&o, 0, sizeof o);
        o.variant = (SparseGMRESVariant_t)variant;
        o.nvec = nvec;
        o.maxIterations = maxit;
        o.rtol = rtol;
        o.reportError = quiet;
        M = SparseGMRES(o);
    }
    ac_count = 0;
    void (^op)(bool, enum CBLAS_TRANSPOSE, DenseMatrix_Double, DenseMatrix_Double) =
        ^(bool accumulate, enum CBLAS_TRANSPOSE trans, DenseMatrix_Double X_,
          DenseMatrix_Double Y_) {
          ac_count += X_.columnCount;
          SparseMatrix_Double At = trans == CblasTrans ? SparseGetTranspose(A) : A;
          if (accumulate) {
              SparseMultiplyAdd(At, X_, Y_);
          } else {
              SparseMultiply(At, X_, Y_);
          }
        };
    int st;
    if (precond) {
        SparseOpaquePreconditioner_Double P =
            SparseCreatePreconditioner((SparsePreconditioner_t)precond, A);
        st = SparseSolve(M, op, B, X, P);
        SparseCleanup(P);
    } else {
        st = SparseSolve(M, op, B, X);
    }
    *products = ac_count;
    return st;
}
