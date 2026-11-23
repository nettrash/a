/* Reads a basis in fplll's format ("[[1 2]\n[3 4]]") from stdin, reduces its rows by FLINT's
 * fmpz_lll with the delta and eta given as arguments, and writes the result in the same format.
 * A helper of gen_lll_reference.py. Build:
 *   cc -O2 gen_lll_flint.c -I/opt/homebrew/include -L/opt/homebrew/lib -lflint -lgmp -lmpfr -o gen_lll_flint
 */
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <flint/fmpz_lll.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    double delta = argc > 1 ? atof(argv[1]) : 0.99, eta = argc > 2 ? atof(argv[2]) : 0.51;
    size_t cap = 1 << 20, len = 0;
    char *text = malloc(cap);
    int c;
    while ((c = getchar()) != EOF) {
        if (len + 1 >= cap) text = realloc(text, cap *= 2);
        text[len++] = (char)c;
    }
    text[len] = 0;
    /* Count the rows (lines with a number) and the columns (numbers on the first of them). */
    long rows = 0, cols = 0, count = 0;
    char *copy = strdup(text);
    for (char *line = strtok(copy, "\n"); line; line = strtok(NULL, "\n")) {
        long k = 0;
        for (char *p = line; *p;) {
            if (*p == '-' || isdigit((unsigned char)*p)) {
                k++;
                while (*p == '-' || isdigit((unsigned char)*p)) p++;
            } else p++;
        }
        if (k > 0) {
            if (rows == 0) cols = k;
            rows++;
            count += k;
        }
    }
    free(copy);
    if (rows == 0 || count != rows * cols) { fprintf(stderr, "bad input\n"); return 1; }
    fmpz_mat_t B;
    fmpz_mat_init(B, rows, cols);
    long i = 0, j = 0;
    for (char *p = text; *p;) {
        if (*p == '-' || isdigit((unsigned char)*p)) {
            char *q = p;
            while (*q == '-' || isdigit((unsigned char)*q)) q++;
            char save = *q;
            *q = 0;
            fmpz_set_str(fmpz_mat_entry(B, i, j), p, 10);
            *q = save;
            p = q;
            if (++j == cols) { j = 0; i++; }
        } else p++;
    }
    fmpz_lll_t fl;
    fmpz_lll_context_init(fl, delta, eta, Z_BASIS, APPROX);
    fmpz_lll(B, NULL, fl);
    printf("[");
    for (i = 0; i < rows; i++) {
        printf("[");
        for (j = 0; j < cols; j++) {
            if (j) printf(" ");
            fmpz_print(fmpz_mat_entry(B, i, j));
        }
        printf(i + 1 < rows ? "]\n" : "]]\n");
    }
    fmpz_mat_clear(B);
    free(text);
    return 0;
}
