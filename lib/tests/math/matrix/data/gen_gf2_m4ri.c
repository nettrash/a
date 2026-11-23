/* The reference outputs of gf2_reference.txt from M4RI, for the tests of BitMatrix:
 *   cc -O2 gen_gf2_m4ri.c -I/opt/homebrew/include -L/opt/homebrew/lib -lm4ri -o gen_gf2_m4ri
 *   ./gen_gf2_m4ri > gf2_reference.txt
 * Lines (hashes as gen_gf2.h computes them, of the outputs laid out as BitMatrix has them):
 *   mul KA m l SA KB n SB  hash(A·B)                     mzd_mul_m4rm, checked by mzd_mul
 *   ech K m n S  rank hash(rref) hash(null space)        mzd_echelonize_m4ri(full), checked by
 *                                                        mzd_echelonize_pluq and the kernel
 *   inv K n S  hash(A⁻¹) | singular                      mzd_inv_m4ri, checked by A·A⁻¹ = I
 *   solve KA m n SA KB k SB  rank(A) rank([A|B]) hash(X) | none
 * where X is the solution of A·X = B whose rows of the columns without a pivot are zero, from the
 * reduced echelon form of [A | B]; KB "image" is B = A·X0 for X0 = gen("dense", n, k, SB). */
#include <m4ri/m4ri.h>
#include "gen_gf2.h"

static mzd_t *to_mzd(const bm *a) {
  mzd_t *m = mzd_init(a->rows, a->cols);
  for (int i = 0; i < a->rows; ++i) memcpy(mzd_row(m, i), a->w + (size_t)i * a->stride, (size_t)a->stride * 8);
  return m;
}
static bm from_mzd(const mzd_t *m) {
  bm a = bm_new(m->nrows, m->ncols);
  for (int i = 0; i < a.rows; ++i)
    for (int w = 0; w < a.stride; ++w)
      a.w[(size_t)i * a.stride + w] = mzd_row_const(m, i)[w] & (w == a.stride - 1 ? last_mask(a.cols) : ~0ULL);
  return a;
}
static int empty(const bm *a) { return a->rows == 0 || a->cols == 0; }
static void fail(const char *what) {
  fprintf(stderr, "check failed: %s\n", what);
  exit(1);
}

static uint64_t SEED = 0x1000;

static void mul(const char *ka, int m, int l, const char *kb, int n) {
  uint64_t sa = SEED++, sb = SEED++;
  bm a = gen(ka, m, l, sa), b = gen(kb, l, n, sb), c;
  if (empty(&a) || empty(&b)) {
    c = bm_naive_mul(&a, &b);
  } else {
    mzd_t *ma = to_mzd(&a), *mb = to_mzd(&b);
    mzd_t *mc = mzd_mul_m4rm(NULL, ma, mb, 0), *ms = mzd_mul(NULL, ma, mb, 0);
    if (!mzd_equal(mc, ms)) fail("mzd_mul_m4rm = mzd_mul");
    c = from_mzd(mc);
    mzd_free(ma);
    mzd_free(mb);
    mzd_free(mc);
    mzd_free(ms);
  }
  printf("mul %s %d %d %llu %s %d %llu %016llx\n", ka, m, l, (unsigned long long)sa, kb, n,
         (unsigned long long)sb, (unsigned long long)bm_hash(&c));
  free(a.w);
  free(b.w);
  free(c.w);
}

/* The reduced row echelon form of a and its rank, checked against the PLUQ route. */
static bm rref(const bm *a, int *rank) {
  if (empty(a)) {
    *rank = 0;
    bm r = bm_new(a->rows, a->cols);
    return r;
  }
  mzd_t *m = to_mzd(a), *p = to_mzd(a);
  *rank = mzd_echelonize_m4ri(m, 1, 0);
  if (mzd_echelonize_pluq(p, 1) != *rank || !mzd_equal(m, p)) fail("mzd_echelonize_m4ri = mzd_echelonize_pluq");
  bm r = from_mzd(m);
  mzd_free(m);
  mzd_free(p);
  return r;
}

static void ech(const char *k, int m, int n) {
  uint64_t s = SEED++;
  bm a = gen(k, m, n, s);
  int rank;
  bm r = rref(&a, &rank), x = null_from_rref(&r, rank);
  /* The null space: A·X = 0, and M4RI's own kernel has as many columns. */
  bm ax = bm_naive_mul(&a, &x);
  for (size_t i = 0; i < (size_t)ax.rows * ax.stride; ++i)
    if (ax.w[i]) fail("A·N = 0");
  if (!empty(&a)) {
    mzd_t *ma = to_mzd(&a);
    mzd_t *kern = mzd_kernel_left_pluq(ma, 0);
    int dim = kern ? kern->ncols : 0;
    if (dim != n - rank) fail("kernel dimension");
    if (kern) mzd_free(kern);
    mzd_free(ma);
  }
  printf("ech %s %d %d %llu %d %016llx %016llx\n", k, m, n, (unsigned long long)s, rank,
         (unsigned long long)bm_hash(&r), (unsigned long long)bm_hash(&x));
  free(a.w);
  free(r.w);
  free(x.w);
  free(ax.w);
}

static void inv(const char *k, int n, uint64_t s) {
  bm a = gen(k, n, n, s);
  int rank;
  bm r = rref(&a, &rank);
  if (rank < n) {
    printf("inv %s %d %llu singular\n", k, n, (unsigned long long)s);
  } else if (n == 0) {
    printf("inv %s %d %llu %016llx\n", k, n, (unsigned long long)s, (unsigned long long)bm_hash(&a));
  } else {
    mzd_t *ma = to_mzd(&a);
    mzd_t *mi = mzd_inv_m4ri(NULL, ma, 0);
    mzd_t *pr = mzd_mul(NULL, ma, mi, 0);
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        if (mzd_read_bit(pr, i, j) != (i == j)) fail("A·A⁻¹ = I");
    bm b = from_mzd(mi);
    printf("inv %s %d %llu %016llx\n", k, n, (unsigned long long)s, (unsigned long long)bm_hash(&b));
    free(b.w);
    mzd_free(ma);
    mzd_free(mi);
    mzd_free(pr);
  }
  free(a.w);
  free(r.w);
}

/* The first seed from `s` whose dense n × n matrix is invertible. */
static uint64_t invertible(int n, uint64_t s) {
  for (;; ++s) {
    bm a = gen("dense", n, n, s);
    int rank;
    bm r = rref(&a, &rank);
    free(a.w);
    free(r.w);
    if (rank == n) return s;
  }
}

static void solve(const char *ka, int m, int n, const char *kb, int k) {
  uint64_t sa = SEED++, sb = SEED++;
  bm a = gen(ka, m, n, sa), b;
  if (!strcmp(kb, "image")) {
    bm x0 = gen("dense", n, k, sb);
    b = bm_naive_mul(&a, &x0);
    free(x0.w);
  } else {
    b = gen(kb, m, k, sb);
  }
  /* [A | B] */
  bm ab = bm_new(m, n + k);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n + k; ++j)
      if (j < n ? bm_get(&a, i, j) : bm_get(&b, i, j - n)) bm_flip(&ab, i, j);
  int ra, rab;
  bm r = rref(&a, &ra), rr = rref(&ab, &rab);
  printf("solve %s %d %d %llu %s %d %llu %d %d ", ka, m, n, (unsigned long long)sa, kb, k,
         (unsigned long long)sb, ra, rab);
  if (rab > ra) {
    printf("none\n");
  } else {
    bm x = bm_new(n, k);
    for (int i = 0; i < ra; ++i) {
      int p = 0;
      while (!bm_get(&rr, i, p)) ++p;
      for (int j = 0; j < k; ++j)
        if (bm_get(&rr, i, n + j)) bm_flip(&x, p, j);
    }
    bm ax = bm_naive_mul(&a, &x);
    for (size_t i = 0; i < (size_t)ax.rows * ax.stride; ++i)
      if (ax.w[i] != b.w[i]) fail("A·X = B");
    printf("%016llx\n", (unsigned long long)bm_hash(&x));
    free(x.w);
    free(ax.w);
  }
  free(a.w);
  free(b.w);
  free(ab.w);
  free(r.w);
  free(rr.w);
}

int main(void) {
  static const char *KINDS[] = {"dense", "sparse", "deficient", "low"};
  for (int s = 0; s < NSIZES; ++s) {
    int n = SIZES[s];
    mul("dense", n, n, "dense", n);
    if (n <= 1000) mul("sparse", n, n, "deficient", n);
  }
  for (int t = 0; t < NTRIPLES; ++t) mul("dense", TRIPLES[t][0], TRIPLES[t][1], "dense", TRIPLES[t][2]);
  for (int s = 0; s < NSIZES; ++s) {
    int n = SIZES[s];
    for (int k = 0; k < 4; ++k) ech(KINDS[k], n, n);
    ech("dense", n, n + 65);
    ech("deficient", n, n + 65);
    ech("dense", n + 65, n);
    ech("deficient", n + 65, n);
    ech("permuted", n, n);
    ech("permuted", n + 65, n);
  }
  for (int s = 0; s < NSIZES; ++s) {
    int n = SIZES[s];
    inv("dense", n, invertible(n, 7000 + (uint64_t)n));
    inv("deficient", n, SEED++);
    inv("sparse", n, SEED++);
    inv("permuted", n, SEED++);
  }
  for (int s = 0; s < NSIZES; ++s) {
    int n = SIZES[s];
    solve("dense", n, n, "image", 3);
    solve("dense", n, n, "dense", 5);
    solve("deficient", n + 65, n, "image", 2);
    solve("deficient", n + 65, n, "dense", 1);
    solve("dense", n, n + 65, "dense", 4);
    solve("low", n, n, "image", 1);
  }
  return 0;
}
