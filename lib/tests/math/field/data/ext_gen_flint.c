/* The second reference for ext_reference.txt: FLINT's fq (FLINT 3) on the same inputs as PARI/GP.
 *
 * Reads, from standard input, whitespace-separated decimal numbers:
 *   C p k g_0 ... g_k        a field GF(p)[x]/g, g monic of degree k
 *   S a_0..a_{k-1} b_0..b_{k-1} e i ord
 *                            a sample: a, b, the exponent e, the Frobenius index i, and whether
 *                            to compute the multiplicative order (0 or 1)
 *   E                        the end
 * and writes for each sample one line:
 *   mul <coeffs> inv <coeffs> pow <coeffs> frob <coeffs> norm <n> tr <t> sq <0|1>
 *   sqrt <coeffs|-> ord <n|->
 * the coefficients the constant first, separated by commas, in decimal.
 *
 * Build: cc -O2 ext_gen_flint.c -I/opt/homebrew/include -L/opt/homebrew/lib -lflint -lgmp -lmpfr
 */
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mod.h>
#include <flint/fmpz_mod_poly.h>
#include <flint/fq.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char buf[1 << 20];

static int next_token(void) { return scanf("%1048575s", buf) == 1; }

static void read_fmpz(fmpz_t x) {
    if (!next_token() || fmpz_set_str(x, buf, 10) != 0) {
        fprintf(stderr, "bad number\n");
        exit(1);
    }
}

static void print_elem(const char *name, const fq_t a, slong k, const fmpz_mod_ctx_t mctx,
                       const fq_ctx_t ctx) {
    fmpz_mod_poly_t p;
    fmpz_t c;
    fmpz_mod_poly_init(p, mctx);
    fmpz_init(c);
    fq_get_fmpz_mod_poly(p, a, ctx);
    printf("%s ", name);
    for (slong i = 0; i < k; i++) {
        fmpz_mod_poly_get_coeff_fmpz(c, p, i, mctx);
        if (i) printf(",");
        fmpz_print(c);
    }
    printf(" ");
    fmpz_clear(c);
    fmpz_mod_poly_clear(p, mctx);
}

static void read_elem(fq_t a, slong k, const fmpz_mod_ctx_t mctx, const fq_ctx_t ctx) {
    fmpz_mod_poly_t p;
    fmpz_t c;
    fmpz_mod_poly_init(p, mctx);
    fmpz_init(c);
    for (slong i = 0; i < k; i++) {
        read_fmpz(c);
        fmpz_mod_poly_set_coeff_fmpz(p, i, c, mctx);
    }
    fq_set_fmpz_mod_poly(a, p, ctx);
    fmpz_clear(c);
    fmpz_mod_poly_clear(p, mctx);
}

int main(void) {
    fmpz_t pr, tmp, e, n;
    fmpz_init(pr);
    fmpz_init(tmp);
    fmpz_init(e);
    fmpz_init(n);
    fmpz_mod_ctx_t mctx;
    fq_ctx_t ctx;
    int have = 0;
    slong k = 0;
    while (next_token()) {
        if (strcmp(buf, "E") == 0) break;
        if (strcmp(buf, "C") == 0) {
            if (have) {
                fq_ctx_clear(ctx);
                fmpz_mod_ctx_clear(mctx);
            }
            read_fmpz(pr);
            read_fmpz(tmp);
            k = fmpz_get_si(tmp);
            fmpz_mod_ctx_init(mctx, pr);
            fmpz_mod_poly_t g;
            fmpz_mod_poly_init(g, mctx);
            for (slong i = 0; i <= k; i++) {
                read_fmpz(tmp);
                fmpz_mod_poly_set_coeff_fmpz(g, i, tmp, mctx);
            }
            fq_ctx_init_modulus(ctx, g, mctx, "x");
            fmpz_mod_poly_clear(g, mctx);
            have = 1;
            continue;
        }
        if (strcmp(buf, "S") != 0) {
            fprintf(stderr, "bad line %s\n", buf);
            return 1;
        }
        fq_t a, b, r;
        fq_init(a, ctx);
        fq_init(b, ctx);
        fq_init(r, ctx);
        read_elem(a, k, mctx, ctx);
        read_elem(b, k, mctx, ctx);
        read_fmpz(e);
        read_fmpz(tmp);
        slong i = fmpz_get_si(tmp);
        read_fmpz(tmp);
        int want_ord = !fmpz_is_zero(tmp);
        fq_mul(r, a, b, ctx);
        print_elem("mul", r, k, mctx, ctx);
        if (fq_is_zero(a, ctx)) {
            printf("inv - ");
        } else {
            fq_inv(r, a, ctx);
            print_elem("inv", r, k, mctx, ctx);
        }
        fq_pow(r, a, e, ctx);
        print_elem("pow", r, k, mctx, ctx);
        fq_frobenius(r, a, i % k, ctx);
        print_elem("frob", r, k, mctx, ctx);
        fq_norm(n, a, ctx);
        printf("norm ");
        fmpz_print(n);
        fq_trace(n, a, ctx);
        printf(" tr ");
        fmpz_print(n);
        printf(" sq %d ", fq_is_square(a, ctx));
        if (fq_sqrt(r, a, ctx)) {
            print_elem("sqrt", r, k, mctx, ctx);
        } else {
            printf("sqrt - ");
        }
        if (want_ord && !fq_is_zero(a, ctx)) {
            fq_multiplicative_order(n, a, ctx);
            printf("ord ");
            fmpz_print(n);
        } else {
            printf("ord -");
        }
        printf("\n");
        fq_clear(a, ctx);
        fq_clear(b, ctx);
        fq_clear(r, ctx);
    }
    if (have) {
        fq_ctx_clear(ctx);
        fmpz_mod_ctx_clear(mctx);
    }
    fmpz_clear(pr);
    fmpz_clear(tmp);
    fmpz_clear(e);
    fmpz_clear(n);
    return 0;
}
