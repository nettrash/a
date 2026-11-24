// Reference outputs for tests/math/field/binary.rs: products, squares, inverses, quotients, square
// roots, powers, traces, half-traces and reductions in GF(2^m) for the SEC 2 fields and others,
// computed by NTL 11.6 (GF2X, GF2E) and by OpenSSL's BN_GF2m_* (4.1), which must agree; and the
// irreducibility of random polynomials by NTL's IterIrredTest.
//
// Inputs come from splitmix64, as the test generates them, so only digests (FNV-1a over the
// little-endian bytes of the output words) are stored.
//
//   c++ -O2 -std=c++17 gen_binary.cpp -I/opt/homebrew/include -I/opt/homebrew/opt/openssl@4/include \
//       -L/opt/homebrew/lib -L/opt/homebrew/opt/openssl@4/lib -lntl -lgmp -lcrypto -o gen_binary
//   ./gen_binary > binary_ntl.txt
//
// NTL computes every output; OpenSSL every one it accepts (its inverses and quotients take at most
// five terms and degrees up to 661), and the two must agree. `openssl=` counts its outputs.
#include <NTL/GF2E.h>
#include <NTL/GF2XFactoring.h>
#include <openssl/bn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
using namespace NTL;
using std::vector;

static uint64_t next(uint64_t &s) {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

struct Fnv {
    uint64_t h = 0xcbf29ce484222325ULL;
    void word(uint64_t w) {
        for (int i = 0; i < 8; i++) { h ^= (w >> (8 * i)) & 0xff; h *= 0x100000001b3ULL; }
    }
    void words(const vector<uint64_t> &v) { for (auto w : v) word(w); }
};

static GF2X from_words(const vector<uint64_t> &w) {
    vector<unsigned char> b(8 * w.size());
    for (size_t i = 0; i < w.size(); i++) for (int j = 0; j < 8; j++) b[8 * i + j] = w[i] >> (8 * j);
    return GF2XFromBytes(b.data(), b.size());
}
static vector<uint64_t> to_words(const GF2X &x, size_t n) {
    vector<unsigned char> b(8 * n);
    BytesFromGF2X(b.data(), x, b.size());
    vector<uint64_t> w(n, 0);
    for (size_t i = 0; i < n; i++) for (int j = 0; j < 8; j++) w[i] |= uint64_t(b[8 * i + j]) << (8 * j);
    return w;
}
static BIGNUM *bn(const vector<uint64_t> &w) {
    vector<unsigned char> b(8 * w.size());
    for (size_t i = 0; i < w.size(); i++) for (int j = 0; j < 8; j++) b[8 * i + j] = w[i] >> (8 * j);
    return BN_lebin2bn(b.data(), b.size(), nullptr);
}
static vector<uint64_t> words_bn(const BIGNUM *x, size_t n) {
    vector<unsigned char> b(8 * n);
    if (BN_bn2lebinpad(x, b.data(), b.size()) < 0) { fprintf(stderr, "bn too big\n"); exit(1); }
    vector<uint64_t> w(n, 0);
    for (size_t i = 0; i < n; i++) for (int j = 0; j < 8; j++) w[i] |= uint64_t(b[8 * i + j]) << (8 * j);
    return w;
}
static void agree(const vector<uint64_t> &a, const vector<uint64_t> &b, const char *what, long m) {
    if (a != b) { fprintf(stderr, "NTL and OpenSSL disagree on %s for m = %ld\n", what, m); exit(1); }
}

static long total_checked = 0;

static void field(const vector<long> &exps) {
    long checked = 0;
    long m = exps[0];
    // OpenSSL's arrays end in the constant term (f = x has none); its inverses and quotients take
    // at most five terms and degrees up to OPENSSL_ECC_MAX_FIELD_BITS = 661, so a call it refuses
    // (returns 0 for) is left to NTL alone.
    bool ossl = exps.back() == 0;
    fprintf(stderr, "m = %ld\n", m);
    size_t n = (m + 63) / 64;
    GF2X f;
    for (long e : exps) SetCoeff(f, e);
    if (!IterIrredTest(f)) { fprintf(stderr, "reducible\n"); exit(1); }
    GF2E::init(f);
    vector<int> arr(exps.begin(), exps.end());
    arr.push_back(-1);
    BN_CTX *ctx = BN_CTX_new();
    uint64_t s = 0;
    for (long e : exps) s = s * 1000003 + uint64_t(e) + 1;
    int cases = m <= 600 ? 24 : 8;
    Fnv dmul, dsqr, dinv, ddiv, dsqrt, dpow, dtr, dhalf, dlong;
    auto elem = [&](void) {
        vector<uint64_t> w(n);
        for (auto &x : w) x = next(s);
        if (m % 64) w[n - 1] &= (uint64_t(1) << (m % 64)) - 1;
        return w;
    };
    for (int c = 0; c < cases; c++) {
        vector<uint64_t> aw = elem(), bw = elem();
        uint64_t e = next(s);
        vector<uint64_t> lw(3 * n + 1);
        for (auto &x : lw) x = next(s);
        GF2E a = conv<GF2E>(from_words(aw)), b = conv<GF2E>(from_words(bw));
        BIGNUM *A = bn(aw), *B = bn(bw), *R = BN_new(), *E = BN_new();
        BN_set_word(E, 0);
        BN_lshift(E, BN_value_one(), 0);
        { vector<uint64_t> ev{e}; BN_free(E); E = bn(ev); }
        // a·b
        vector<uint64_t> r1 = to_words(rep(a * b), n);
        if (ossl && BN_GF2m_mod_mul_arr(R, A, B, arr.data(), ctx)) { agree(r1, words_bn(R, n), "mul", m); checked++; } dmul.words(r1);
        // a²
        r1 = to_words(rep(sqr(a)), n);
        if (ossl && BN_GF2m_mod_sqr_arr(R, A, arr.data(), ctx)) { agree(r1, words_bn(R, n), "sqr", m); checked++; } dsqr.words(r1);
        // 1/a, 0 for 0
        if (IsZero(a)) r1.assign(n, 0); else {
            r1 = to_words(rep(inv(a)), n);
            if (ossl && BN_GF2m_mod_inv_arr(R, A, arr.data(), ctx)) { agree(r1, words_bn(R, n), "inv", m); checked++; }
        }
        dinv.words(r1);
        // a/b, 0 for b = 0
        if (IsZero(b)) r1.assign(n, 0); else {
            r1 = to_words(rep(a / b), n);
            if (ossl && BN_GF2m_mod_div_arr(R, A, B, arr.data(), ctx)) { agree(r1, words_bn(R, n), "div", m); checked++; }
        }
        ddiv.words(r1);
        // √a = a^(2^(m−1))
        r1 = to_words(rep(power(a, power2_ZZ(m - 1))), n);
        if (ossl && BN_GF2m_mod_sqrt_arr(R, A, arr.data(), ctx)) { agree(r1, words_bn(R, n), "sqrt", m); checked++; } dsqrt.words(r1);
        // a^e
        ZZ ez = conv<ZZ>((unsigned long)(e >> 32)) * power2_ZZ(32) + conv<ZZ>((unsigned long)(e & 0xffffffff));
        r1 = to_words(rep(power(a, ez)), n);
        if (ossl && BN_GF2m_mod_exp_arr(R, A, E, arr.data(), ctx)) { agree(r1, words_bn(R, n), "pow", m); checked++; } dpow.words(r1);
        // Tr(a)
        dtr.word(IsOne(trace(a)) ? 1 : 0);
        // the half-trace, for odd m
        if (m % 2 == 1) {
            GF2E h = a, t = a;
            for (long i = 0; i < (m - 1) / 2; i++) { t = sqr(sqr(t)); h += t; }
            r1 = to_words(rep(h), n);
            if (IsZero(trace(a))) {
                // OpenSSL solves z² + z = a by the half-trace for odd m.
                if (ossl && BN_GF2m_mod_solve_quad_arr(R, A, arr.data(), ctx)) { agree(r1, words_bn(R, n), "half-trace", m); checked++; }
            }
            dhalf.words(r1);
        }
        // a long polynomial reduced
        r1 = to_words(from_words(lw) % f, n);
        if (ossl && BN_GF2m_mod_arr(R, bn(lw), arr.data())) { agree(r1, words_bn(R, n), "mod", m); checked++; } dlong.words(r1);
        BN_free(A); BN_free(B); BN_free(R); BN_free(E);
    }
    for (size_t i = 0; i < exps.size(); i++) printf("%s%ld", i ? "," : "", exps[i]);
    printf(" %d mul=%016llx sqr=%016llx inv=%016llx div=%016llx sqrt=%016llx pow=%016llx tr=%016llx",
           cases, (unsigned long long)dmul.h, (unsigned long long)dsqr.h, (unsigned long long)dinv.h,
           (unsigned long long)ddiv.h, (unsigned long long)dsqrt.h, (unsigned long long)dpow.h,
           (unsigned long long)dtr.h);
    if (m % 2 == 1) printf(" half=%016llx", (unsigned long long)dhalf.h);
    printf(" long=%016llx openssl=%ld\n", (unsigned long long)dlong.h, checked);
    total_checked += checked;
    BN_CTX_free(ctx);
}

static vector<long> exps_of(const GF2X &f) {
    vector<long> e;
    for (long i = deg(f); i >= 0; i--) if (IsOne(coeff(f, i))) e.push_back(i);
    return e;
}

int main() {
    SetSeed(conv<ZZ>(20261123));
    printf("# gen_binary.cpp: NTL 11.6 GF2E and OpenSSL 4.1 BN_GF2m_*, which agree on every output.\n");
    printf("# <exponents> <cases> <op>=<FNV-1a digest of the outputs>; inputs from splitmix64.\n");
    vector<vector<long>> fields = {
        // SEC 2 (and X9.62's x^239 + x^36 + 1), AES, GCM.
        {113, 9, 0}, {131, 8, 3, 2, 0}, {163, 7, 6, 3, 0}, {193, 15, 0}, {233, 74, 0},
        {239, 36, 0}, {239, 158, 0}, {283, 12, 7, 5, 0}, {409, 87, 0}, {571, 10, 5, 2, 0},
        {8, 4, 3, 1, 0}, {128, 7, 2, 1, 0},
        // The reciprocals of some: terms close to x^m, so short chunks or Barrett's reduction.
        {163, 160, 157, 156, 0}, {233, 159, 0}, {239, 203, 0}, {409, 322, 0}, {571, 569, 566, 561, 0},
        {1, 0}, {1},
        // Moduli of one word that are dense or have terms close to x^m, for which the sparse
        // reduction would take chunks of a few bits: Barrett's reduction under a carry-less
        // instruction (the auditor's examples among them).
        {3, 2, 0}, {5, 4, 3, 2, 0}, {8, 7, 6, 5, 4, 2, 0}, {13, 12, 11, 8, 2, 1, 0},
        {13, 12, 7, 5, 0}, {31, 28, 0}, {33, 32, 20, 10, 0},
        {33, 32, 31, 29, 28, 27, 25, 23, 22, 21, 19, 18, 17, 15, 14, 11, 9, 4, 0},
        {47, 46, 38, 28, 0}, {50, 49, 11, 7, 0}, {63, 62, 0}, {63, 62, 48, 32, 0},
        {63, 61, 60, 59, 56, 55, 53, 51, 49, 48, 46, 43, 42, 40, 34, 32, 31, 30, 29, 28, 25, 22, 17,
         13, 12, 11, 10, 9, 8, 4, 3, 2, 0},
        {64, 63, 37, 33, 0}, {64, 62, 50, 43, 0},
        {64, 63, 62, 60, 59, 58, 56, 55, 53, 52, 47, 46, 43, 42, 41, 35, 34, 33, 32, 26, 24, 23, 22,
         21, 20, 18, 16, 13, 12, 11, 10, 9, 8, 2, 0},
    };
    for (long d : {2, 3, 4, 5, 7, 13, 31, 32, 33, 61, 63, 64, 65, 96, 127, 128, 129, 191, 192, 255,
                   256, 257, 383, 511, 512, 513, 600, 1024, 1279, 2000}) {
        GF2X f; BuildSparseIrred(f, d); fields.push_back(exps_of(f));
    }
    for (long d : {3, 8, 17, 63, 64, 65, 100, 127, 128, 160, 200, 257, 300, 500, 571, 777, 1000, 1500}) {
        GF2X g; BuildSparseIrred(g, d);
        GF2X f; BuildRandomIrred(f, g); fields.push_back(exps_of(f));
    }
    for (auto &e : fields) field(e);
    printf("# OpenSSL agreed with NTL on %ld outputs.\n", total_checked);
    // Irreducibility of random polynomials: trinomials, pentanomials (terms that repeat cancel)
    // and dense ones, of degrees 2 to 601, as tests/math/field/binary.rs draws them.
    uint64_t s = 0x1eeddeadULL;
    int count = 600;
    vector<int> bits;
    for (int c = 0; c < count; c++) {
        long d = 2 + next(s) % 600;
        long kind = next(s) % 3;
        GF2X f; SetCoeff(f, d); SetCoeff(f, 0);
        if (kind < 2) {
            for (int j = 0; j < (kind == 0 ? 1 : 3); j++) {
                long k = 1 + next(s) % (d - 1);
                SetCoeff(f, k, coeff(f, k) + 1);
            }
        } else {
            for (long w = 0; w < (d + 63) / 64; w++) {
                uint64_t x = next(s);
                for (int j = 0; j < 64; j++) {
                    long i = 64 * w + j;
                    if (i >= 1 && i < d && (x >> j & 1)) SetCoeff(f, i);
                }
            }
        }
        bits.push_back(IterIrredTest(f) ? 1 : 0);
    }
    printf("irreducible %d ", count);
    for (int i = 0; i < count; i += 4) {
        int v = 0;
        for (int j = 0; j < 4 && i + j < count; j++) v |= bits[i + j] << j;
        printf("%x", v);
    }
    printf("\n");
}
