/* LAPACK reference shim: the roots of a float polynomial as the eigenvalues of its companion
 * matrix, as numpy.roots computes them (dgeev / zgeev, no eigenvectors, the matrix built in each
 * call). macOS: Accelerate's LAPACK; Linux: the system's liblapack. */
#include "shim.h"
#include <complex.h>

SHIM_EXPORTS(sl)
const char* sl_label(void* h);

extern void dgeev_(const char* jobvl, const char* jobvr, const int* n, double* a, const int* lda,
                   double* wr, double* wi, double* vl, const int* ldvl, double* vr, const int* ldvr,
                   double* work, const int* lwork, int* info);
extern void zgeev_(const char* jobvl, const char* jobvr, const int* n, double* a, const int* lda,
                   double* w, double* vl, const int* ldvl, double* vr, const int* ldvr,
                   double* work, const int* lwork, double* rwork, int* info);

typedef struct {
    int cplx;
    size_t V;
    int n;              /* degree */
    double** p;         /* coefficients (complex interleaved for cplx) */
    double *a, *wr, *wi, *w, *work, *rwork;
    int lwork;
} H;

void* sl_prep(const char* op, const uint64_t* buf, size_t nbuf) {
    int cplx;
    if (!strcmp(op, "f_roots")) cplx = 0;
    else if (!strcmp(op, "c_roots")) cplx = 1;
    else return NULL;
    input_t in;
    parse_input(buf, nbuf, &in);
    H* h = (H*)calloc(1, sizeof(H));
    h->cplx = cplx;
    h->V = in.v;
    size_t len = VA(&in, 0, 0)->len;
    h->n = (int)((cplx ? len / 2 : len) - 1);
    h->p = calloc(h->V, sizeof(double*));
    for (size_t v = 0; v < h->V; v++) {
        h->p[v] = malloc(sizeof(double) * len);
        memcpy(h->p[v], VA(&in, v, 0)->p, sizeof(double) * len);
    }
    free(in.var);
    int n = h->n, lda = n, one = 1, info = 0, lw = -1;
    size_t nn = (size_t)n * (size_t)n * (cplx ? 2 : 1);
    h->a = malloc(sizeof(double) * nn + 16);
    double q[2] = {0, 0};
    if (!cplx) {
        h->wr = malloc(sizeof(double) * (size_t)n + 8);
        h->wi = malloc(sizeof(double) * (size_t)n + 8);
        dgeev_("N", "N", &n, h->a, &lda, h->wr, h->wi, NULL, &one, NULL, &one, q, &lw, &info);
        h->lwork = (int)q[0];
        h->work = malloc(sizeof(double) * (size_t)h->lwork + 8);
    } else {
        h->w = malloc(sizeof(double) * 2 * (size_t)n + 8);
        h->rwork = malloc(sizeof(double) * 2 * (size_t)n + 8);
        zgeev_("N", "N", &n, h->a, &lda, h->w, NULL, &one, NULL, &one, q, &lw, h->rwork, &info);
        h->lwork = (int)q[0];
        h->work = malloc(sizeof(double) * 2 * (size_t)h->lwork + 8);
    }
    return h;
}

static void run1(H* h, size_t v) {
    int n = h->n, lda = n, one = 1, info = 0, lw = h->lwork;
    const double* p = h->p[v];
    if (!h->cplx) {
        /* ones under the diagonal, -p_i/p_n in the last column (column-major) */
        memset(h->a, 0, sizeof(double) * (size_t)n * (size_t)n);
        for (int i = 1; i < n; i++) h->a[(size_t)(i - 1) * n + i] = 1.0;
        for (int i = 0; i < n; i++) h->a[(size_t)(n - 1) * n + i] = -p[i] / p[n];
        dgeev_("N", "N", &n, h->a, &lda, h->wr, h->wi, NULL, &one, NULL, &one, h->work, &lw, &info);
    } else {
        double complex* A = (double complex*)h->a;
        const double complex* P = (const double complex*)p;
        memset(h->a, 0, sizeof(double) * 2 * (size_t)n * (size_t)n);
        for (int i = 1; i < n; i++) A[(size_t)(i - 1) * n + i] = 1.0;
        for (int i = 0; i < n; i++) A[(size_t)(n - 1) * n + i] = -P[i] / P[n];
        zgeev_("N", "N", &n, h->a, &lda, h->w, NULL, &one, NULL, &one, h->work, &lw, h->rwork, &info);
    }
}

void sl_run(void* hv, uint64_t reps) {
    H* h = (H*)hv;
    for (uint64_t r = 0; r < reps; r++) run1(h, (size_t)(r % h->V));
}

static int cmp_d2(const void* a, const void* b) {
    const double *x = a, *y = b;
    if (x[0] != y[0]) return x[0] < y[0] ? -1 : 1;
    return x[1] < y[1] ? -1 : x[1] > y[1];
}

size_t sl_check(void* hv, uint64_t* out, size_t cap) {
    H* h = (H*)hv;
    outbuf_t o = {out, cap, 0};
    int n = h->n;
    double* z = malloc(sizeof(double) * 2 * (size_t)n + 16);
    for (size_t v = 0; v < h->V; v++) {
        run1(h, v);
        for (int i = 0; i < n; i++) {
            z[2 * i] = h->cplx ? h->w[2 * i] : h->wr[i];
            z[2 * i + 1] = h->cplx ? h->w[2 * i + 1] : h->wi[i];
        }
        qsort(z, (size_t)n, 16, cmp_d2);
        put(&o, (uint64_t)n);
        for (int i = 0; i < 2 * n; i++) { uint64_t u; memcpy(&u, &z[i], 8); put(&o, u); }
    }
    free(z);
    return o.n;
}

void sl_free(void* hv) {
    H* h = (H*)hv;
    for (size_t v = 0; v < h->V; v++) free(h->p[v]);
    free(h->p); free(h->a); free(h->wr); free(h->wi); free(h->w); free(h->work); free(h->rwork);
    free(h);
}
const char* sl_label(void* h) { (void)h; return ""; }
