/* The seeded matrices of the tests of BitMatrix (lib/tests/math/matrix/gf2.rs) and the cases of
 * gf2_reference.txt, shared by gen_gf2_m4ri.c and gen_gf2_ntl.cpp. The test builds the same
 * matrices from the same seeds; a change here needs the same change there. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int rows, cols, stride;
  uint64_t *w;
} bm;

/* SplitMix64 (Steele, Lea and Flood 2014). */
static uint64_t mix(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static uint64_t next(uint64_t *s) {
  *s += 0x9E3779B97F4A7C15ULL;
  return mix(*s);
}

static bm bm_new(int rows, int cols) {
  bm a;
  a.rows = rows;
  a.cols = cols;
  a.stride = (cols + 63) / 64;
  a.w = (uint64_t *)calloc((size_t)rows * a.stride + 1, 8);
  return a;
}
static int bm_get(const bm *a, int i, int j) { return (int)(a->w[(size_t)i * a->stride + j / 64] >> (j % 64) & 1); }
static void bm_flip(bm *a, int i, int j) { a->w[(size_t)i * a->stride + j / 64] ^= 1ULL << (j % 64); }
static uint64_t last_mask(int cols) { return cols % 64 ? (1ULL << (cols % 64)) - 1 : ~0ULL; }

/* The hash of the tests: the shape, then every word, each through SplitMix64's finaliser. */
static uint64_t bm_hash(const bm *a) {
  uint64_t h = mix(mix((uint64_t)a->rows) ^ (uint64_t)a->cols);
  for (size_t k = 0; k < (size_t)a->rows * a->stride; ++k) h = mix(h ^ a->w[k]);
  return h;
}

/* a·b, a row of b for each 1 of a row of a. */
static bm bm_naive_mul(const bm *a, const bm *b) {
  bm c = bm_new(a->rows, b->cols);
  for (int i = 0; i < a->rows; ++i)
    for (int j = 0; j < a->cols; ++j)
      if (bm_get(a, i, j))
        for (int w = 0; w < b->stride; ++w) c.w[(size_t)i * c.stride + w] ^= b->w[(size_t)j * b->stride + w];
  return c;
}

/* Kinds: "dense" (uniform), "sparse" (each bit 1 with probability 1/8), "deficient" (dense, then
 * row i = row i-1 + row i-3 for i % 4 == 3, then column j = column j-1 + column j-2 for
 * j % 3 == 2), "low" (X·Y for dense X of m × r and Y of r × n, r = min(m, n) / 3, seeds `seed`
 * and `seed ^ 0xA5A5A5A5A5A5A5A5`), "permuted" (sparse, then the bits below the diagonal cleared
 * and those on it set, then the rows shuffled by Fisher and Yates, row i swapped with row
 * next() % (i + 1) for i from m - 1 down to 1: invertible when square). */
static bm gen(const char *kind, int m, int n, uint64_t seed) {
  if (!strcmp(kind, "low")) {
    int r = (m < n ? m : n) / 3;
    bm x = gen("dense", m, r, seed), y = gen("dense", r, n, seed ^ 0xA5A5A5A5A5A5A5A5ULL);
    bm a = bm_naive_mul(&x, &y);
    free(x.w);
    free(y.w);
    return a;
  }
  bm a = bm_new(m, n);
  uint64_t s = seed;
  int sparse = !strcmp(kind, "sparse") || !strcmp(kind, "permuted");
  for (int i = 0; i < m; ++i)
    for (int w = 0; w < a.stride; ++w) {
      uint64_t v = next(&s);
      if (sparse) {
        v &= next(&s);
        v &= next(&s);
      }
      if (w == a.stride - 1) v &= last_mask(n);
      a.w[(size_t)i * a.stride + w] = v;
    }
  if (!strcmp(kind, "permuted")) {
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n && j <= i; ++j)
        if (bm_get(&a, i, j) != (j == i)) bm_flip(&a, i, j);
    for (int i = m - 1; i >= 1; --i) {
      int j = (int)(next(&s) % (uint64_t)(i + 1));
      for (int w = 0; w < a.stride; ++w) {
        uint64_t t = a.w[(size_t)i * a.stride + w];
        a.w[(size_t)i * a.stride + w] = a.w[(size_t)j * a.stride + w];
        a.w[(size_t)j * a.stride + w] = t;
      }
    }
  }
  if (!strcmp(kind, "deficient")) {
    for (int i = 3; i < m; ++i)
      if (i % 4 == 3)
        for (int w = 0; w < a.stride; ++w)
          a.w[(size_t)i * a.stride + w] = a.w[(size_t)(i - 1) * a.stride + w] ^ a.w[(size_t)(i - 3) * a.stride + w];
    for (int j = 2; j < n; ++j)
      if (j % 3 == 2)
        for (int i = 0; i < m; ++i)
          if (bm_get(&a, i, j) != (bm_get(&a, i, j - 1) ^ bm_get(&a, i, j - 2))) bm_flip(&a, i, j);
  }
  return a;
}

/* The canonical basis of the null space from a reduced row echelon form r of n columns: for each
 * column f without a pivot, the vector with a 1 in row f and r[i][f] in the row of the pivot of
 * row i. */
static bm null_from_rref(const bm *r, int rank) {
  int n = r->cols;
  int *piv = (int *)calloc((size_t)rank + 1, sizeof(int));
  int *isp = (int *)calloc((size_t)n + 1, sizeof(int));
  for (int i = 0; i < rank; ++i) {
    int j = 0;
    while (!bm_get(r, i, j)) ++j;
    piv[i] = j;
    isp[j] = 1;
  }
  bm x = bm_new(n, n - rank);
  int f = 0;
  for (int j = 0; j < n; ++j) {
    if (isp[j]) continue;
    bm_flip(&x, j, f);
    for (int i = 0; i < rank; ++i)
      if (bm_get(r, i, j)) bm_flip(&x, piv[i], f);
    ++f;
  }
  free(piv);
  free(isp);
  return x;
}

static const int SIZES[] = {0, 1, 63, 64, 65, 127, 128, 129, 1000, 2048};
#define NSIZES (int)(sizeof SIZES / sizeof SIZES[0])

/* The products beyond the squares: m, l, n. The last is the product of the ignored test. */
static const int TRIPLES[][3] = {
    {0, 5, 7},       {5, 0, 7},        {7, 5, 0},       {1, 1, 1},       {1, 2048, 1},
    {2048, 1, 2048}, {65, 129, 63},    {63, 65, 129},   {129, 1000, 127}, {1000, 64, 1000},
    {64, 1000, 65},  {2048, 2048, 65}, {65, 2048, 2048}, {2048, 65, 2048}, {2113, 2177, 2100},
    {4096, 4096, 4096},
};
#define NTRIPLES (int)(sizeof TRIPLES / sizeof TRIPLES[0])
