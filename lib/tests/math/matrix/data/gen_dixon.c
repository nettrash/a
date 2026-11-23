/* Writes dixon_flint.txt, the solutions of the seeded systems of the tests of Dixon's lifting
 * (tests/math/matrix/dixon.rs), by FLINT's fmpq_mat_solve_fmpz_mat_dixon (systems of integers) and
 * fmpq_mat_solve_dixon (systems of rationals).
 *
 *   cc -O2 -I/opt/homebrew/include gen_dixon.c -L/opt/homebrew/lib -lflint -lgmp -lmpfr -o gen_dixon
 *   ./gen_dixon > dixon_flint.txt
 *
 * The elements are drawn as in gen_multimod.c: ceil(bits / 8) bytes, each the low byte of the next
 * xorshift64 word (<< 13, >> 7, << 17), little-endian, the top byte cut to the bits, negated when the
 * next word is odd; a rational is such a numerator over 1 + (next word mod 1000). `A` (n × n) is drawn
 * row after row, then `B` (n × k). Kind `z` is integers, `s` integers with the last row of `A` the sum
 * of its first two (singular), `q` rationals. A line is `kind seed n k bits`, then `none` for a
 * singular system, or the least common multiple of the denominators of `X` modulo 2^61 - 1, its
 * length in bits, and the digest of `X`: h = (h·1000003 + (y mod q) + 7·(d mod q)) mod q over the
 * elements y / d in lowest terms, column after column, q = 2^61 - 1, from h = 0. */
#include <gmp.h>
#include <flint/fmpq.h>
#include <flint/fmpq_mat.h>
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

static void rational(fmpq_t x, int bits) {
    fmpz_t n, d;
    fmpz_init(n);
    fmpz_init(d);
    entry(n, bits);
    fmpz_set_ui(d, 1 + next() % 1000);
    fmpq_set_fmpz_frac(x, n, d);
    fmpz_clear(n);
    fmpz_clear(d);
}

static void line(char kind, uint64_t seed, int n, int k, int bits) {
    s = seed;
    fmpq_mat_t x;
    fmpq_mat_init(x, n, k);
    int ok;
    if (kind == 'q') {
        fmpq_mat_t a, b;
        fmpq_mat_init(a, n, n);
        fmpq_mat_init(b, n, k);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) rational(fmpq_mat_entry(a, i, j), bits);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++) rational(fmpq_mat_entry(b, i, j), bits);
        ok = fmpq_mat_solve_dixon(x, a, b);
        fmpq_mat_clear(a);
        fmpq_mat_clear(b);
    } else {
        fmpz_mat_t a, b;
        fmpz_mat_init(a, n, n);
        fmpz_mat_init(b, n, k);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) entry(fmpz_mat_entry(a, i, j), bits);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++) entry(fmpz_mat_entry(b, i, j), bits);
        if (kind == 's')
            for (int j = 0; j < n; j++)
                fmpz_add(fmpz_mat_entry(a, n - 1, j), fmpz_mat_entry(a, 0, j), fmpz_mat_entry(a, 1, j));
        ok = fmpq_mat_solve_fmpz_mat_dixon(x, a, b);
        fmpz_mat_clear(a);
        fmpz_mat_clear(b);
    }
    printf("%c %llu %d %d %d", kind, (unsigned long long)seed, n, k, bits);
    if (!ok) {
        printf(" none\n");
        fmpq_mat_clear(x);
        return;
    }
    fmpz_t q, l, r, t;
    fmpz_init(q);
    fmpz_init(l);
    fmpz_init(r);
    fmpz_init(t);
    fmpz_set_ui(q, 1);
    fmpz_mul_2exp(q, q, 61);
    fmpz_sub_ui(q, q, 1);
    fmpz_one(l);
    fmpz_zero(r);
    for (int j = 0; j < k; j++)
        for (int i = 0; i < n; i++) {
            fmpq *e = fmpq_mat_entry(x, i, j);
            fmpz_lcm(l, l, fmpq_denref(e));
            fmpz_mul_ui(r, r, 1000003);
            fmpz_mod(t, fmpq_numref(e), q);
            fmpz_add(r, r, t);
            fmpz_mod(t, fmpq_denref(e), q);
            fmpz_addmul_ui(r, t, 7);
            fmpz_mod(r, r, q);
        }
    fmpz_mod(t, l, q);
    printf(" %llu %llu %llu\n", (unsigned long long)fmpz_get_ui(t), (unsigned long long)fmpz_bits(l),
           (unsigned long long)fmpz_get_ui(r));
    fmpz_clear(q);
    fmpz_clear(l);
    fmpz_clear(r);
    fmpz_clear(t);
    fmpq_mat_clear(x);
}

int main(void) {
    /* n, k, bits: from 24 rows, where Dixon's lifting takes the systems with at most n / 8 columns, to
     * 100; and below and past n / 8 columns, where the elimination does. */
    const int z[][3] = {{24, 1, 8},  {24, 3, 64},  {24, 4, 64},  {30, 2, 1000}, {32, 1, 100}, {40, 5, 8},
                        {50, 1, 64}, {50, 6, 200}, {64, 8, 32},  {80, 2, 64},   {100, 1, 8},  {100, 12, 64},
                        {100, 3, 100}, {16, 2, 200}};
    const int q[][3] = {{16, 1, 8}, {20, 2, 64}, {24, 3, 8}, {40, 1, 100}, {60, 4, 16}, {100, 2, 8}};
    uint64_t seed = 0xD1C0000;
    for (unsigned i = 0; i < sizeof z / sizeof z[0]; i++) {
        line('z', ++seed, z[i][0], z[i][1], z[i][2]);
        line('s', ++seed, z[i][0], z[i][1], z[i][2]);
    }
    for (unsigned i = 0; i < sizeof q / sizeof q[0]; i++) line('q', ++seed, q[i][0], q[i][1], q[i][2]);
    return 0;
}
