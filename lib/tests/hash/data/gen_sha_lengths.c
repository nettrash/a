/* The digests by OpenSSL's SHA-224, SHA-256, SHA-384 and SHA-512 of the messages of every length
 * from 0 to 300 bytes, for the test `every_length_up_to_300_bytes_matches_openssl`.
 *
 * The message of length n is the first n bytes of m[i] = (i * 167 + 13) mod 256.
 *
 * Build and run (OpenSSL 3 or 4):
 *   cc -O2 gen_sha_lengths.c -I/opt/homebrew/opt/openssl@4/include \
 *      -L/opt/homebrew/opt/openssl@4/lib -lcrypto -o gen_sha_lengths
 *   ./gen_sha_lengths 8 > sha_lengths.txt    # the first 8 bytes of each digest, as the test has
 *   ./gen_sha_lengths 64 > sha_lengths_full.txt  # whole digests, for a probe
 *
 * Each line: the length, then the first `prefix` bytes of each of the four digests in hexadecimal.
 */
#include <openssl/evp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int prefix = argc > 1 ? atoi(argv[1]) : 8;
    unsigned char msg[300];
    for (int i = 0; i < 300; i++) msg[i] = (unsigned char)((i * 167 + 13) & 0xff);
    const EVP_MD *mds[4] = {EVP_sha224(), EVP_sha256(), EVP_sha384(), EVP_sha512()};
    printf("# The first %d bytes of SHA-224, SHA-256, SHA-384 and SHA-512 by %s\n", prefix,
           OPENSSL_VERSION_TEXT);
    printf("# of the message m[i] = (i * 167 + 13) mod 256 of each length; from gen_sha_lengths.c\n");
    for (int n = 0; n <= 300; n++) {
        printf("%d", n);
        for (int k = 0; k < 4; k++) {
            unsigned char md[64];
            unsigned int len = 0;
            if (!EVP_Digest(msg, (size_t)n, md, &len, mds[k], NULL)) return 1;
            printf(" ");
            for (unsigned int j = 0; j < len && (int)j < prefix; j++) printf("%02x", md[j]);
        }
        printf("\n");
    }
    return 0;
}
