/* The Hermite and Smith normal forms by FLINT, for gen_normal.py.
 *
 * Reads the number of matrices, then each as "m n" and its m·n elements row after row, in decimal,
 * from stdin; writes for each the lines
 *   H <fmpz_mat_hnf(A)>
 *   L <the first n columns of fmpz_mat_hnf([A | I])>
 *   U <the last m columns of fmpz_mat_hnf([A | I])>
 *   S <the diagonal of fmpz_mat_snf(A)>
 * each matrix as its rows joined by ';', the elements of a row by ','.
 *
 * cc -O2 -I/opt/homebrew/include gen_normal.c -L/opt/homebrew/lib -lflint -lgmp -lmpfr
 */
#include <stdio.h>
#include <stdlib.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>

static void put(const fmpz_t x) {
    char *s = fmpz_get_str(NULL, 10, x);
    fputs(s, stdout);
    flint_free(s);
}

/* The columns c0..c1 of A, rows joined by ';', elements by ','. */
static void put_block(const fmpz_mat_t a, slong c0, slong c1) {
    for (slong i = 0; i < fmpz_mat_nrows(a); i++) {
        if (i) putchar(';');
        for (slong j = c0; j < c1; j++) {
            if (j > c0) putchar(',');
            put(fmpz_mat_entry(a, i, j));
        }
    }
}

int main(void) {
    long count;
    if (scanf("%ld", &count) != 1) return 1;
    for (long k = 0; k < count; k++) {
        long m, n;
        if (scanf("%ld %ld", &m, &n) != 2) return 1;
        fmpz_mat_t a, h, w, hw, s;
        fmpz_mat_init(a, m, n);
        for (slong i = 0; i < m; i++)
            for (slong j = 0; j < n; j++)
                if (!fmpz_fread(stdin, fmpz_mat_entry(a, i, j))) return 1;
        fmpz_mat_init(h, m, n);
        fmpz_mat_hnf(h, a);
        if (!fmpz_mat_is_in_hnf(h)) return 2;
        fputs("H ", stdout);
        put_block(h, 0, n);
        putchar('\n');
        /* [A | I] */
        fmpz_mat_init(w, m, n + m);
        fmpz_mat_init(hw, m, n + m);
        for (slong i = 0; i < m; i++) {
            for (slong j = 0; j < n; j++) fmpz_set(fmpz_mat_entry(w, i, j), fmpz_mat_entry(a, i, j));
            fmpz_one(fmpz_mat_entry(w, i, n + i));
        }
        fmpz_mat_hnf(hw, w);
        fputs("L ", stdout);
        put_block(hw, 0, n);
        fputs("\nU ", stdout);
        put_block(hw, n, n + m);
        putchar('\n');
        fmpz_mat_init(s, m, n);
        fmpz_mat_snf(s, a);
        if (!fmpz_mat_is_in_snf(s)) return 3;
        fputs("S ", stdout);
        for (slong i = 0; i < (m < n ? m : n); i++) {
            if (i) putchar(',');
            put(fmpz_mat_entry(s, i, i));
        }
        putchar('\n');
        fmpz_mat_clear(a);
        fmpz_mat_clear(h);
        fmpz_mat_clear(w);
        fmpz_mat_clear(hw);
        fmpz_mat_clear(s);
    }
    return 0;
}
