/* Reference values for tests/math/field/gf2_128.rs: GHASH(H, A, C) of AES-128-GCM by OpenSSL 4.1
 * (EVP_aes_128_gcm) on random keys, IVs, additional data and ciphertexts.
 *
 * GCM's tag is T = GHASH(H, A, C) + E(K, J0), with H = E(K, 0^128) and J0 = IV || 0^31 || 1 for a
 * 96-bit IV. Each case draws K, IV, A and C from splitmix64 (one draw a byte, as the test does),
 * takes the plaintext that encrypts to C (C plus the keystream, the encryption of zeros), lets
 * OpenSSL encrypt it, checks that the ciphertext is C, and prints H and T + E(K, J0) = GHASH.
 *
 *   cc -O2 gen_gf2_128.c -I/opt/homebrew/opt/openssl@4/include -L/opt/homebrew/opt/openssl@4/lib \
 *      -lcrypto -o gen_gf2_128 && ./gen_gf2_128 > gf2_128_openssl.txt
 */
#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t next(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void die(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }

static void gcm(const uint8_t *k, const uint8_t *iv, const uint8_t *a, int alen, const uint8_t *p,
                int plen, uint8_t *c, uint8_t *tag) {
    EVP_CIPHER_CTX *x = EVP_CIPHER_CTX_new();
    int n;
    if (!EVP_EncryptInit_ex(x, EVP_aes_128_gcm(), NULL, k, iv)) die("init");
    if (alen && !EVP_EncryptUpdate(x, NULL, &n, a, alen)) die("aad");
    if (plen && !EVP_EncryptUpdate(x, c, &n, p, plen)) die("update");
    if (!EVP_EncryptFinal_ex(x, c + plen, &n)) die("final");
    if (!EVP_CIPHER_CTX_ctrl(x, EVP_CTRL_GCM_GET_TAG, 16, tag)) die("tag");
    EVP_CIPHER_CTX_free(x);
}

static void ecb(const uint8_t *k, const uint8_t *in, uint8_t *out) {
    EVP_CIPHER_CTX *x = EVP_CIPHER_CTX_new();
    int n;
    EVP_EncryptInit_ex(x, EVP_aes_128_ecb(), NULL, k, NULL);
    EVP_CIPHER_CTX_set_padding(x, 0);
    if (!EVP_EncryptUpdate(x, out, &n, in, 16) || n != 16) die("ecb");
    EVP_CIPHER_CTX_free(x);
}

int main(void) {
    printf("# gen_gf2_128.c: OpenSSL 4.1 AES-128-GCM; per case <H> <GHASH(H, A, C)>, GCM blocks.\n");
    for (int i = 0; i < 64; i++) {
        uint64_t s = 0x67636d0000000000ULL + i;
        uint8_t k[16], iv[12], a[70], c[90], p[90], ks[90], z[90] = {0}, ct[106], tag[16], kt[16];
        for (int j = 0; j < 16; j++) k[j] = next(&s);
        for (int j = 0; j < 12; j++) iv[j] = next(&s);
        int alen = next(&s) % 70, clen = next(&s) % 90;
        for (int j = 0; j < alen; j++) a[j] = next(&s);
        for (int j = 0; j < clen; j++) c[j] = next(&s);
        gcm(k, iv, NULL, 0, z, clen, ks, kt);
        for (int j = 0; j < clen; j++) p[j] = c[j] ^ ks[j];
        gcm(k, iv, a, alen, p, clen, ct, tag);
        if (memcmp(ct, c, clen)) die("ciphertext");
        uint8_t zero[16] = {0}, h[16], j0[16], e[16];
        ecb(k, zero, h);
        memcpy(j0, iv, 12);
        j0[12] = j0[13] = j0[14] = 0;
        j0[15] = 1;
        ecb(k, j0, e);
        for (int j = 0; j < 16; j++) printf("%02x", h[j]);
        printf(" ");
        for (int j = 0; j < 16; j++) printf("%02x", tag[j] ^ e[j]);
        printf("\n");
    }
}
