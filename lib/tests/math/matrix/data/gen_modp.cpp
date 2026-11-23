// Writes modp_flint.txt, the reference results of the tests of `ModMatrix` (tests/math/matrix/modp.rs):
// for each seeded matrix over GF(p), its rank, determinant, and hashes of its reduced row echelon
// form, inverse, null space basis and product by its transpose, from FLINT's nmod_mat; NTL's
// mat_zz_p (mat_ZZ_p beyond 60 bits) is run on every matrix too and must agree on the rank, the
// determinant and the inverse, and its kernel must span the same space as FLINT's null space,
// or the generator stops.
//
//   c++ -std=c++17 -O2 -I/opt/homebrew/include gen_modp.cpp -L/opt/homebrew/lib -lflint -lntl -lgmp -lmpfr -o gen_modp
//   ./gen_modp > modp_flint.txt
//
// The matrices are those of `random` in modp.rs: xorshift64 (<< 13, >> 7, << 17), each element
// `x = next() % p` kept when `next() % 8 < density`; below full rank, the product of an m × rank and a
// rank × n matrix drawn so. A line is `seed m n rank density p rank det rref inv null prod`, the
// hashes FNV-1a over the words (rows, cols, then the elements row after row), `-` where there is none.
#include <flint/nmod_mat.h>
#include <NTL/mat_ZZ_p.h>
#include <NTL/mat_lzz_p.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

typedef unsigned __int128 u128;
typedef std::vector<uint64_t> Vec;

struct Rng {
    uint64_t s;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

static Vec product(const Vec &a, const Vec &b, size_t m, size_t k, size_t n, uint64_t p) {
    Vec c(m * n);
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) {
            u128 s = 0;
            for (size_t t = 0; t < k; t++) s = (s + (u128)a[i * k + t] * b[t * n + j]) % p;
            c[i * n + j] = (uint64_t)s;
        }
    return c;
}

static Vec random(Rng &r, size_t m, size_t n, size_t rank, uint64_t density, uint64_t p) {
    auto draw = [&]() {
        uint64_t x = r.next() % p;
        return r.next() % 8 < density ? x : 0;
    };
    if (rank >= (m < n ? m : n)) {
        Vec a(m * n);
        for (auto &x : a) x = draw();
        return a;
    }
    Vec b(m * rank), c(rank * n);
    for (auto &x : b) x = draw();
    for (auto &x : c) x = draw();
    return product(b, c, m, rank, n, p);
}

struct Hash {
    uint64_t h = 0xcbf29ce484222325ull;
    void mix(uint64_t x) {
        h ^= x;
        h *= 0x100000001b3ull;
    }
};

static uint64_t hash(const nmod_mat_t a, slong rows, slong cols) {
    Hash h;
    h.mix(rows);
    h.mix(cols);
    for (slong i = 0; i < rows; i++)
        for (slong j = 0; j < cols; j++) h.mix(nmod_mat_entry(a, i, j));
    return h.h;
}

static void fail(const char *what, uint64_t seed) {
    fprintf(stderr, "NTL and FLINT disagree on the %s of seed %llu\n", what, (unsigned long long)seed);
    exit(1);
}

// NTL's rank, determinant, inverse and kernel against FLINT's, for the field type `F` and its matrices `M`.
template <class F, class M>
static void check(const Vec &a, size_t m, size_t n, uint64_t seed, slong rank, const nmod_mat_t A,
                  const nmod_mat_t null, slong nullity) {
    auto elem = [](uint64_t x) {
        NTL::ZZ z;
        NTL::conv(z, (unsigned long)x);
        F f;
        NTL::conv(f, z);
        return f;
    };
    M x;
    x.SetDims(m, n);
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) x[i][j] = elem(a[i * n + j]);
    M g = x;
    if (NTL::gauss(g) != rank) fail("rank", seed);
    if (m == n) {
        F d;
        M inv;
        NTL::inv(d, inv, x);
        if (d != elem(nmod_mat_det(A))) fail("determinant", seed);
        if (!NTL::IsZero(d)) {
            nmod_mat_t B;
            nmod_mat_init(B, m, m, A->mod.n);
            nmod_mat_inv(B, A);
            for (size_t i = 0; i < m; i++)
                for (size_t j = 0; j < m; j++)
                    if (inv[i][j] != elem(nmod_mat_entry(B, i, j))) fail("inverse", seed);
            nmod_mat_clear(B);
        }
    }
    // The kernel of `x ↦ x·Aᵀ` is the null space of `A`: as many vectors as FLINT's, and with FLINT's
    // vectors no more than as many dimensions.
    M t, k;
    NTL::transpose(t, x);
    NTL::kernel(k, t);
    if (k.NumRows() != nullity) fail("nullity", seed);
    M both;
    both.SetDims(2 * nullity, n);
    for (slong i = 0; i < nullity; i++)
        for (size_t j = 0; j < n; j++) {
            both[i][j] = k[i][j];
            both[nullity + i][j] = elem(nmod_mat_entry(null, j, i));
        }
    if (nullity > 0 && NTL::gauss(both) != nullity) fail("null space", seed);
}

int main() {
    const uint64_t primes[] = {2ull,
                               3ull,
                               5ull,
                               251ull,
                               65521ull,
                               2147483647ull,
                               4294967291ull,
                               1125899906842597ull,
                               2305843009213693951ull,
                               4611686018427387847ull,
                               9223372036854775783ull,
                               18446744073709551557ull,
                               1000003ull};
    struct Shape {
        size_t m, n, rank;
        bool large;
    };
    const Shape shapes[] = {{1, 1, 1, false},   {2, 2, 2, false},   {2, 2, 1, false},   {3, 3, 3, false},
                            {4, 4, 2, false},   {3, 6, 3, false},   {6, 3, 3, false},   {5, 5, 0, false},
                            {8, 8, 8, false},   {10, 10, 7, false}, {7, 13, 7, false},  {13, 7, 5, false},
                            {20, 20, 20, false}, {24, 24, 16, false}, {16, 40, 16, false}, {40, 16, 9, false},
                            {64, 64, 64, true}, {64, 64, 50, true}, {80, 120, 80, true}, {120, 80, 61, true},
                            {150, 150, 150, true}, {150, 150, 140, true}};
    uint64_t seed = 0x5EED0000ull;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        const Shape &sh = shapes[s];
        for (size_t j = 0; j < sizeof primes / sizeof primes[0]; j++) {
            if (sh.large && j % 4 != s % 4) continue;
            const uint64_t p = primes[j];
            for (uint64_t density : {8ull, 2ull}) {
                seed++;
                Rng r{seed};
                Vec a = random(r, sh.m, sh.n, sh.rank, density, p);
                const size_t m = sh.m, n = sh.n;
                nmod_mat_t A, R, X, T, P;
                nmod_mat_init(A, m, n, p);
                for (size_t i = 0; i < m; i++)
                    for (size_t c = 0; c < n; c++) nmod_mat_entry(A, i, c) = a[i * n + c];
                slong rank = nmod_mat_rank(A);
                printf("%llu %zu %zu %zu %llu %llu %ld", (unsigned long long)seed, m, n, sh.rank,
                       (unsigned long long)density, (unsigned long long)p, (long)rank);
                if (m == n)
                    printf(" %llu", (unsigned long long)nmod_mat_det(A));
                else
                    printf(" -");
                nmod_mat_init_set(R, A);
                if (nmod_mat_rref(R) != rank) fail("FLINT rref rank", seed);
                printf(" %llu", (unsigned long long)hash(R, m, n));
                if (m == n && rank == (slong)n) {
                    nmod_mat_init(X, n, n, p);
                    nmod_mat_inv(X, A);
                    printf(" %llu", (unsigned long long)hash(X, n, n));
                    nmod_mat_clear(X);
                } else
                    printf(" -");
                nmod_mat_init(X, n, n, p);
                slong nullity = nmod_mat_nullspace(X, A);
                if (nullity != (slong)n - rank) fail("FLINT nullity", seed);
                printf(" %llu", (unsigned long long)hash(X, n, nullity));
                nmod_mat_init(T, n, m, p);
                nmod_mat_transpose(T, A);
                nmod_mat_init(P, m, m, p);
                nmod_mat_mul(P, A, T);
                printf(" %llu\n", (unsigned long long)hash(P, m, m));
                if (p < NTL_SP_BOUND) {
                    NTL::zz_p::init(p);
                    check<NTL::zz_p, NTL::mat_zz_p>(a, m, n, seed, rank, A, X, nullity);
                } else {
                    NTL::ZZ q;
                    NTL::conv(q, (unsigned long)p);
                    NTL::ZZ_p::init(q);
                    check<NTL::ZZ_p, NTL::mat_ZZ_p>(a, m, n, seed, rank, A, X, nullity);
                }
                nmod_mat_clear(A);
                nmod_mat_clear(R);
                nmod_mat_clear(X);
                nmod_mat_clear(T);
                nmod_mat_clear(P);
            }
        }
    }
    return 0;
}
