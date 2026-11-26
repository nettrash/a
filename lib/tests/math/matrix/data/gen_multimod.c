/* Writes multimod_flint.txt, the determinants of the large seeded integer matrices of the tests of
 * the multimodular determinant (tests/math/matrix/multimodular.rs), by FLINT's fmpz_mat_det.
 *
 *   cc -O2 -I/opt/homebrew/include gen_multimod.c -L/opt/homebrew/lib -lflint -lgmp -lmpfr -o gen_multimod
 *   ./gen_multimod > multimod_flint.txt
 *
 * An element of `bits` bits is ceil(bits / 8) bytes, each the low byte of the next xorshift64 word
 * (<< 13, >> 7, << 17), little-endian, the top byte cut to the bits, then negated when the next word
 * is odd. A matrix is drawn row after row; kind `r` is that, kind `s` has its last row the sum of its
 * first two, kind `u` is L·U for L unit lower and U unit upper triangular with elements so drawn
 * (L's below its diagonal row after row, then U's above), whose determinant is 1, and kind `l` has
 * its first row of `bits` bits and the other rows of 8 bits, kind `m` the same with its last row
 * the sum of its first two. The orders and lengths are chosen
 * inside the bands of the determinant's methods (`DET_BANDS` in rational.rs: Bareiss's
 * elimination, the residues, Dixon's divisor) and at their edges; kind `l` reaches the remainder
 * tree of the residues at a small cost. A line is
 * `kind seed n bits`, then the determinant's residues modulo 2^61 - 1 and 2^64 - 59 (in [0, q)),
 * its sign (-1, 0, 1) and its length in bits. */
#include <gmp.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <stdint.h>
#include <stdio.h>

static uint64_t s;
static uint64_t next(void) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
}

static void entry(fmpz_t x, int bits) {
    int bytes = (bits + 7) / 8;
    static unsigned char v[1 << 16];
    for (int i = 0; i < bytes; i++) v[i] = (unsigned char)next();
    if (bits % 8) v[bytes - 1] &= (unsigned char)((1u << (bits % 8)) - 1);
    mpz_t z;
    mpz_init(z);
    mpz_import(z, bytes, -1, 1, 0, 0, v);
    if (next() & 1) mpz_neg(z, z);
    fmpz_set_mpz(x, z);
    mpz_clear(z);
}

static void line(char kind, uint64_t seed, int n, int bits) {
    s = seed;
    fmpz_mat_t a;
    fmpz_mat_init(a, n, n);
    if (kind == 'u') {
        fmpz_mat_t l, u;
        fmpz_mat_init(l, n, n);
        fmpz_mat_init(u, n, n);
        for (int i = 0; i < n; i++) {
            fmpz_one(fmpz_mat_entry(l, i, i));
            fmpz_one(fmpz_mat_entry(u, i, i));
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++) entry(fmpz_mat_entry(l, i, j), bits);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) entry(fmpz_mat_entry(u, i, j), bits);
        fmpz_mat_mul(a, l, u);
        fmpz_mat_clear(l);
        fmpz_mat_clear(u);
    } else {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) entry(fmpz_mat_entry(a, i, j), (kind == 'l' || kind == 'm') && i > 0 ? 8 : bits);
        if (kind == 's' || kind == 'm')
            for (int j = 0; j < n; j++)
                fmpz_add(fmpz_mat_entry(a, n - 1, j), fmpz_mat_entry(a, 0, j), fmpz_mat_entry(a, 1, j));
    }
    fmpz_t d, q, r;
    fmpz_init(d);
    fmpz_init(q);
    fmpz_init(r);
    fmpz_mat_det(d, a);
    printf("%c %llu %d %d", kind, (unsigned long long)seed, n, bits);
    fmpz_set_ui(q, 1);
    fmpz_mul_2exp(q, q, 61);
    fmpz_sub_ui(q, q, 1);
    fmpz_mod(r, d, q);
    printf(" %llu", (unsigned long long)fmpz_get_ui(r));
    fmpz_set_ui(q, UINT64_C(18446744073709551557));
    fmpz_mod(r, d, q);
    printf(" %llu", (unsigned long long)fmpz_get_ui(r));
    printf(" %d %llu\n", fmpz_sgn(d), (unsigned long long)fmpz_bits(d));
    fmpz_clear(d);
    fmpz_clear(q);
    fmpz_clear(r);
    fmpz_mat_clear(a);
}

int main(void) {
    /* Elimination: below the residues' band. */
    const int bareiss[][2] = {{9, 64}, {12, 1000}, {16, 100}, {19, 64}};
    /* Residues: inside the bands for r and s (and for u, whose elements are about twice as long). */
    const int residues[][2] = {{24, 30}, {30, 100}, {36, 250}, {48, 500}, {20, 1400}, {32, 2000}, {16, 4000}};
    /* Dixon's divisor: above the residues' band. */
    const int divisor[][2] = {{30, 8}, {40, 64}, {60, 30}, {60, 100}, {64, 8}, {80, 64}, {100, 8}};
    /* One long row: the remainder tree of the residues. */
    const int tree[][2] = {{32, 34000}, {36, 40000}};
    uint64_t seed = 0xD37000;
    for (unsigned i = 0; i < sizeof bareiss / sizeof bareiss[0]; i++)
        for (const char *k = "rsu"; *k; k++) line(*k, ++seed, bareiss[i][0], bareiss[i][1]);
    for (unsigned i = 0; i < sizeof residues / sizeof residues[0]; i++)
        for (const char *k = "rsu"; *k; k++) line(*k, ++seed, residues[i][0], residues[i][1]);
    for (unsigned i = 0; i < sizeof divisor / sizeof divisor[0]; i++)
        for (const char *k = "rsu"; *k; k++) line(*k, ++seed, divisor[i][0], divisor[i][1]);
    for (unsigned i = 0; i < sizeof tree / sizeof tree[0]; i++)
        for (const char *k = "ls"; *k; k++) line(*k == 's' ? 'm' : *k, ++seed, tree[i][0], tree[i][1]);
    return 0;
}
