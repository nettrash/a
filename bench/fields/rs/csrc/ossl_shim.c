/* OpenSSL references of the fields scoreboard: BIGNUM prime fields (Montgomery products by
 * BN_mod_mul_montgomery, with its assembly on aarch64 and x86_64), binary fields (BN_GF2m_*_arr),
 * and GHASH through AES-128-GCM's additional data (EVP_EncryptUpdate with no output, which runs
 * GHASH alone). See shim.h for the interface. */
#include <stdlib.h>
#include <string.h>
#include <openssl/bn.h>
#include <openssl/evp.h>
#include <openssl/crypto.h>
#include "shim.h"

static volatile unsigned long sink;

const char *ossl_version(void) { return OpenSSL_version(OPENSSL_VERSION); }

static BIGNUM *bn_of(const uint64_t *l, size_t n) {
    return BN_lebin2bn((const unsigned char *)l, (int)(n * 8), NULL);
}

static void limbs_of(uint64_t *out, const BIGNUM *b, size_t n) {
    memset(out, 0, n * 8);
    BN_bn2lebinpad(b, (unsigned char *)out, (int)(n * 8));
}

/* ---------------------------------------------------------------- prime fields */

typedef struct {
    BIGNUM *p, *e, *acc, *t;
    BIGNUM **xm, **xc, **sqc; /* Montgomery forms, canonical values, canonical squares */
    BN_MONT_CTX *mont;
    BN_CTX *ctx;
    size_t n, count;
} o_pf;

void *ossl_pf_new(const uint64_t *p, size_t n, const uint64_t *xs, size_t count, const uint64_t *e,
                  size_t en) {
    o_pf *h = calloc(1, sizeof *h);
    h->ctx = BN_CTX_new();
    h->p = bn_of(p, n); h->e = bn_of(e, en);
    h->mont = BN_MONT_CTX_new();
    BN_MONT_CTX_set(h->mont, h->p, h->ctx);
    h->n = n; h->count = count;
    h->xm = malloc(count * sizeof(BIGNUM *)); h->xc = malloc(count * sizeof(BIGNUM *));
    h->sqc = malloc(count * sizeof(BIGNUM *));
    for (size_t i = 0; i < count; i++) {
        h->xc[i] = bn_of(xs + i * n, n);
        h->xm[i] = BN_new();
        BN_to_montgomery(h->xm[i], h->xc[i], h->mont, h->ctx);
        h->sqc[i] = BN_new();
        BN_mod_mul(h->sqc[i], h->xc[i], h->xc[i], h->p, h->ctx);
    }
    h->acc = BN_dup(h->xm[0]);
    h->t = BN_new();
    BN_set_bit(h->t, (int)(2 * 64 * n + 64)); /* room, so that the loops do not grow it */
    BN_zero(h->t);
    return h;
}

void ossl_pf_free(void *v) {
    o_pf *h = v;
    for (size_t i = 0; i < h->count; i++) { BN_free(h->xm[i]); BN_free(h->xc[i]); BN_free(h->sqc[i]); }
    free(h->xm); free(h->xc); free(h->sqc);
    BN_free(h->p); BN_free(h->e); BN_free(h->acc); BN_free(h->t);
    BN_MONT_CTX_free(h->mont); BN_CTX_free(h->ctx);
    free(h);
}

int ossl_pf_out(void *v, int op, size_t i, uint64_t *out) {
    o_pf *h = v;
    size_t j = (i + 1) & (h->count - 1);
    BIGNUM *r = BN_new();
    int ret = 0, mont = 0;
    switch (op) {
    case OP_MUL: BN_mod_mul_montgomery(r, h->xm[i], h->xm[j], h->mont, h->ctx); mont = 1; break;
    case OP_SQR: BN_mod_mul_montgomery(r, h->xm[i], h->xm[i], h->mont, h->ctx); mont = 1; break;
    case OP_ADD: BN_mod_add_quick(r, h->xm[i], h->xm[j], h->p); mont = 1; break;
    case OP_SUB: BN_mod_sub_quick(r, h->xm[i], h->xm[j], h->p); mont = 1; break;
    case OP_INV: BN_mod_inverse(r, h->xc[i], h->p, h->ctx); break;
    case OP_SQRT: ret = BN_mod_sqrt(r, h->sqc[i], h->p, h->ctx) != NULL; break;
    case OP_LEG: ret = BN_kronecker(h->xc[i], h->p, h->ctx); break;
    case OP_POW: BN_mod_exp_mont(r, h->xc[i], h->e, h->p, h->ctx, h->mont); break;
    default: ret = UNSUPPORTED;
    }
    if (mont) BN_from_montgomery(r, r, h->mont, h->ctx);
    limbs_of(out, r, h->n);
    BN_free(r);
    return ret;
}

void ossl_pf_run(void *v, int op, uint64_t iters) {
    o_pf *h = v;
    size_t m = h->count - 1;
    long s = 0;
    BN_CTX *c = h->ctx;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) BN_mod_mul_montgomery(h->acc, h->acc, h->xm[i & m], h->mont, c); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) BN_mod_mul_montgomery(h->acc, h->acc, h->acc, h->mont, c); break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) BN_mod_add_quick(h->acc, h->acc, h->xm[i & m], h->p); break;
    case OP_SUB: for (uint64_t i = 0; i < iters; i++) BN_mod_sub_quick(h->acc, h->acc, h->xm[i & m], h->p); break;
    case OP_INV:
        for (uint64_t i = 0; i < iters; i++) { BN_mod_inverse(h->t, h->xc[i & m], h->p, c); BN_mod_add_quick(h->acc, h->acc, h->t, h->p); }
        break;
    case OP_SQRT:
        for (uint64_t i = 0; i < iters; i++) { BN_mod_sqrt(h->t, h->sqc[i & m], h->p, c); BN_mod_add_quick(h->acc, h->acc, h->t, h->p); }
        break;
    case OP_LEG: for (uint64_t i = 0; i < iters; i++) s += BN_kronecker(h->xc[i & m], h->p, c); break;
    case OP_POW:
        for (uint64_t i = 0; i < iters; i++) { BN_mod_exp_mont(h->t, h->xc[i & m], h->e, h->p, c, h->mont); BN_mod_add_quick(h->acc, h->acc, h->t, h->p); }
        break;
    }
    if (BN_is_zero(h->acc)) BN_copy(h->acc, h->xm[0]);
    sink = BN_get_word(h->acc) + (unsigned long)s;
}

/* ---------------------------------------------------------------- binary fields */

#ifndef OPENSSL_NO_EC2M

typedef struct {
    int p[16];
    BIGNUM **xs, *acc, *t, *e;
    BN_CTX *ctx;
    size_t w, count;
} o_bin;

void *ossl_bin_new(const uint32_t *exps, size_t nexp, const uint64_t *xs, size_t count, uint64_t e) {
    if (nexp + 1 > 16) return NULL;
    o_bin *h = calloc(1, sizeof *h);
    for (size_t j = 0; j < nexp; j++) h->p[j] = (int)exps[j]; /* decreasing, ending with 0 */
    h->p[nexp] = -1;
    h->w = (exps[0] + 63) / 64; h->count = count;
    h->ctx = BN_CTX_new();
    h->xs = malloc(count * sizeof(BIGNUM *));
    for (size_t i = 0; i < count; i++) h->xs[i] = bn_of(xs + i * h->w, h->w);
    h->acc = BN_dup(h->xs[0]); h->t = BN_new();
    h->e = BN_new(); BN_set_word(h->e, e);
    return h;
}

void ossl_bin_free(void *v) {
    o_bin *h = v;
    for (size_t i = 0; i < h->count; i++) BN_free(h->xs[i]);
    free(h->xs); BN_free(h->acc); BN_free(h->t); BN_free(h->e); BN_CTX_free(h->ctx);
    free(h);
}

int ossl_bin_out(void *v, int op, size_t i, uint64_t *out) {
    o_bin *h = v;
    size_t j = (i + 1) & (h->count - 1);
    BIGNUM *r = BN_new();
    int ret = 0, ok = 1;
    switch (op) {
    case OP_MUL: ok = BN_GF2m_mod_mul_arr(r, h->xs[i], h->xs[j], h->p, h->ctx); break;
    case OP_SQR: ok = BN_GF2m_mod_sqr_arr(r, h->xs[i], h->p, h->ctx); break;
    case OP_INV: ok = BN_GF2m_mod_inv_arr(r, h->xs[i], h->p, h->ctx); break;
    case OP_DIV: ok = BN_GF2m_mod_div_arr(r, h->xs[i], h->xs[j], h->p, h->ctx); break;
    case OP_SQRT: ok = BN_GF2m_mod_sqrt_arr(r, h->xs[i], h->p, h->ctx); break;
    case OP_QROOT: ok = BN_GF2m_mod_solve_quad_arr(r, h->xs[i], h->p, h->ctx); ret = ok; ok = 1; break;
    case OP_POWU: ok = BN_GF2m_mod_exp_arr(r, h->xs[i], h->e, h->p, h->ctx); break;
    default: ret = UNSUPPORTED;
    }
    if (!ok) ret = UNSUPPORTED;
    limbs_of(out, r, h->w);
    BN_free(r);
    return ret;
}

void ossl_bin_run(void *v, int op, uint64_t iters) {
    o_bin *h = v;
    size_t m = h->count - 1;
    BN_CTX *c = h->ctx;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) BN_GF2m_mod_mul_arr(h->acc, h->acc, h->xs[i & m], h->p, c); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) BN_GF2m_mod_sqr_arr(h->acc, h->acc, h->p, c); break;
    case OP_INV:
        for (uint64_t i = 0; i < iters; i++) { BN_GF2m_mod_inv_arr(h->t, h->xs[i & m], h->p, c); BN_GF2m_add(h->acc, h->acc, h->t); }
        break;
    case OP_DIV: for (uint64_t i = 0; i < iters; i++) BN_GF2m_mod_div_arr(h->acc, h->acc, h->xs[i & m], h->p, c); break;
    case OP_SQRT:
        for (uint64_t i = 0; i < iters; i++) { BN_GF2m_mod_sqrt_arr(h->t, h->xs[i & m], h->p, c); BN_GF2m_add(h->acc, h->acc, h->t); }
        break;
    case OP_QROOT:
        for (uint64_t i = 0; i < iters; i++) { BN_GF2m_mod_solve_quad_arr(h->t, h->xs[i & m], h->p, c); BN_GF2m_add(h->acc, h->acc, h->t); }
        break;
    case OP_POWU:
        for (uint64_t i = 0; i < iters; i++) { BN_GF2m_mod_exp_arr(h->t, h->xs[i & m], h->e, h->p, c); BN_GF2m_add(h->acc, h->acc, h->t); }
        break;
    }
    if (BN_is_zero(h->acc)) BN_copy(h->acc, h->xs[0]);
    sink = BN_get_word(h->acc);
}

#else /* an OpenSSL built without binary fields */
void *ossl_bin_new(const uint32_t *exps, size_t nexp, const uint64_t *xs, size_t count, uint64_t e) { return NULL; }
void ossl_bin_free(void *v) {}
int ossl_bin_out(void *v, int op, size_t i, uint64_t *out) { return UNSUPPORTED; }
void ossl_bin_run(void *v, int op, uint64_t iters) {}
#endif

/* ---------------------------------------------------------------- GHASH through AES-GCM */

typedef struct { EVP_CIPHER_CTX *c; } o_gcm;

/* H = AES_K(0^128) and E(K, J0) for the 96-bit IV, by AES-128-ECB, for the harness's check. */
int ossl_gcm_keys(const uint8_t key[16], const uint8_t iv[12], uint8_t h[16], uint8_t ej0[16]) {
    EVP_CIPHER_CTX *c = EVP_CIPHER_CTX_new();
    uint8_t in[32] = {0}, out[48];
    int l1 = 0, l2 = 0;
    memcpy(in + 16, iv, 12); in[31] = 1;
    EVP_EncryptInit_ex(c, EVP_aes_128_ecb(), NULL, key, NULL);
    EVP_CIPHER_CTX_set_padding(c, 0);
    int ok = EVP_EncryptUpdate(c, out, &l1, in, 32) && EVP_EncryptFinal_ex(c, out + l1, &l2);
    memcpy(h, out, 16); memcpy(ej0, out + 16, 16);
    EVP_CIPHER_CTX_free(c);
    return ok;
}

/* The tag of AES-128-GCM over the additional data `aad` and an empty plaintext. */
int ossl_gcm_tag(const uint8_t key[16], const uint8_t iv[12], const uint8_t *aad, size_t len, uint8_t tag[16]) {
    EVP_CIPHER_CTX *c = EVP_CIPHER_CTX_new();
    int l = 0, ok = EVP_EncryptInit_ex(c, EVP_aes_128_gcm(), NULL, key, iv);
    ok = ok && EVP_EncryptUpdate(c, NULL, &l, aad, (int)len);
    ok = ok && EVP_EncryptFinal_ex(c, NULL, &l);
    ok = ok && EVP_CIPHER_CTX_ctrl(c, EVP_CTRL_GCM_GET_TAG, 16, tag);
    EVP_CIPHER_CTX_free(c);
    return ok;
}

void *ossl_gcm_new(const uint8_t key[16], const uint8_t iv[12]) {
    o_gcm *h = calloc(1, sizeof *h);
    h->c = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(h->c, EVP_aes_128_gcm(), NULL, key, iv);
    return h;
}

/* `reps` calls of GHASH over `len` bytes of additional data each, continuing one message. */
void ossl_gcm_aad(void *v, const uint8_t *aad, size_t len, uint64_t reps) {
    o_gcm *h = v;
    int l = 0;
    for (uint64_t i = 0; i < reps; i++) EVP_EncryptUpdate(h->c, NULL, &l, aad, (int)len);
    sink = (unsigned long)l;
}

void ossl_gcm_free(void *v) { o_gcm *h = v; EVP_CIPHER_CTX_free(h->c); free(h); }
