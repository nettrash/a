// fplll reference shim: LLL (wrapper, proved, heuristic, fast), BKZ 1.0 and 2.0, SVP, CVP.
#include "common.h"
#include <fplll.h>
#include <string>
#include <vector>

using namespace fplll;
typedef ZZ_mat<mpz_t> Mat;

static char *mat_text(const Mat &B) {
    lb_buf b = {0, 0, 0};
    lb_put(&b, "");
    for (int i = 0; i < B.get_rows(); i++)
        for (int j = 0; j < B.get_cols(); j++) {
            char *t = mpz_get_str(NULL, 10, B[i][j].get_data());
            lb_put(&b, t);
            lb_put(&b, " ");
            free(t);
        }
    return b.s;
}

static std::vector<Strategy> *g_strategies = nullptr;

extern "C" {

void *fp_new(const char *text, long r, long c) {
    Mat *A = new Mat((int)r, (int)c);
    const char *p = text;
    char *tok = NULL;
    size_t cap = 0;
    for (long i = 0; i < r; i++)
        for (long j = 0; j < c; j++) {
            if (!lb_token(&p, &tok, &cap)) abort();
            if (mpz_set_str((*A)[i][j].get_data(), tok, 10)) abort();
        }
    free(tok);
    return A;
}

void fp_free(void *A) { delete (Mat *)A; }

// method: 0 wrapper, 1 proved, 2 heuristic, 3 fast.
char *fp_lll(const void *A_, double delta, double eta, int method, int with_u, int want) {
    Mat B(*(const Mat *)A_);
    int st;
    if (with_u) {
        Mat U;
        U.gen_identity(B.get_rows());
        st = lll_reduction(B, U, delta, eta, (LLLMethod)method, FT_DEFAULT, 0, LLL_DEFAULT);
    } else {
        st = lll_reduction(B, delta, eta, (LLLMethod)method, FT_DEFAULT, 0, LLL_DEFAULT);
    }
    if (st != RED_SUCCESS) return want ? strdup("FAIL") : NULL;
    return want ? mat_text(B) : NULL;
}

// variant 1: BKZ 1.0 (no strategies, no pruning), 2: BKZ 2.0 (default.json strategies, auto
// abort, the Gaussian heuristic bound on the radius). The basis is LLL-reduced first by the wrapper, as fplll's own tools do.
char *fp_bkz(const void *A_, int beta, double delta, int variant, int with_u, int want) {
    Mat B(*(const Mat *)A_);
    Mat U;
    if (with_u) U.gen_identity(B.get_rows());
    int st = with_u ? lll_reduction(B, U, delta, LLL_DEF_ETA) : lll_reduction(B, delta, LLL_DEF_ETA);
    if (st != RED_SUCCESS) return want ? strdup("FAIL") : NULL;
    std::vector<Strategy> empty;
    int flags = BKZ_DEFAULT;
    if (variant == 2) {
        if (!g_strategies)
            g_strategies = new std::vector<Strategy>(
                load_strategies_json(strategy_full_path(default_strategy())));
        flags |= BKZ_AUTO_ABORT | BKZ_GH_BND;
    }
    BKZParam param(beta, variant == 2 ? *g_strategies : empty, delta, flags);
    st = bkz_reduction(&B, with_u ? &U : nullptr, param, FT_DEFAULT, 0);
    if (st != RED_SUCCESS) return want ? strdup("FAIL") : NULL;
    return want ? mat_text(B) : NULL;
}

static void norm_of(const Mat &B, const std::vector<Z_NR<mpz_t>> &x, const Mat *t, mpz_t out) {
    mpz_set_ui(out, 0);
    mpz_t s;
    mpz_init(s);
    for (int j = 0; j < B.get_cols(); j++) {
        mpz_set_ui(s, 0);
        for (int i = 0; i < B.get_rows(); i++) mpz_addmul(s, x[i].get_data(), B[i][j].get_data());
        if (t) mpz_sub(s, s, (*t)[0][j].get_data());
        mpz_addmul(out, s, s);
    }
    mpz_clear(s);
}

// LLL (wrapper) then shortest_vector; method 2 proved, 0 fast. Returns |v|^2.
char *fp_svp(const void *A_, int method, int want) {
    Mat B(*(const Mat *)A_);
    if (lll_reduction(B) != RED_SUCCESS) return want ? strdup("FAIL") : NULL;
    // Drop the zero rows that a dependent basis leaves first.
    int z = 0;
    while (z < B.get_rows()) {
        bool zero = true;
        for (int j = 0; j < B.get_cols(); j++)
            if (B[z][j].sgn() != 0) zero = false;
        if (!zero) break;
        z++;
    }
    if (z > 0) {
        Mat C(B.get_rows() - z, B.get_cols());
        for (int i = z; i < B.get_rows(); i++)
            for (int j = 0; j < B.get_cols(); j++) C[i - z][j] = B[i][j];
        B = C;
    }
    std::vector<Z_NR<mpz_t>> sol;
    int st = shortest_vector(B, sol, (SVPMethod)method, SVP_DEFAULT);
    if (st != RED_SUCCESS) return want ? strdup("FAIL") : NULL;
    if (!want) return NULL;
    mpz_t n;
    mpz_init(n);
    norm_of(B, sol, nullptr, n);
    char *s = mpz_get_str(NULL, 10, n);
    mpz_clear(n);
    return s;
}

// LLL (wrapper) then closest_vector (CVPM_FAST unless method 2). Target is a 1 × cols matrix.
// Returns the squared distance.
char *fp_cvp(const void *A_, const void *T_, int method, int want) {
    Mat B(*(const Mat *)A_);
    const Mat &T = *(const Mat *)T_;
    if (lll_reduction(B) != RED_SUCCESS) return want ? strdup("FAIL") : NULL;
    std::vector<Z_NR<mpz_t>> target(T.get_cols());
    for (int j = 0; j < T.get_cols(); j++) target[j] = T[0][j];
    std::vector<Z_NR<mpz_t>> sol;
    int st = closest_vector(B, target, sol, method, CVP_DEFAULT);
    if (st != RED_SUCCESS) return want ? strdup("FAIL") : NULL;
    if (!want) return NULL;
    mpz_t n;
    mpz_init(n);
    norm_of(B, sol, &T, n);
    char *s = mpz_get_str(NULL, 10, n);
    mpz_clear(n);
    return s;
}

#define LB_STR2(x) #x
#define LB_STR(x) LB_STR2(x)
const char *fp_version(void) { return LB_STR(FPLLL_VERSION); }

} // extern "C"
