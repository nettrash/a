// Shared helpers of the reference shims: text of integer matrices, FNV-1a digests of words.
#ifndef LATBENCH_COMMON_H
#define LATBENCH_COMMON_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// FNV-1a over 64-bit words, each word fed as 8 little-endian bytes.
static inline uint64_t lb_fnv_words(uint64_t h, const uint64_t *w, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint64_t x = w[i];
        for (int b = 0; b < 8; b++) {
            h ^= (x >> (8 * b)) & 0xff;
            h *= 0x100000001b3ULL;
        }
    }
    return h;
}
#define LB_FNV_INIT 0xcbf29ce484222325ULL

// A growable string.
typedef struct {
    char *s;
    size_t len, cap;
} lb_buf;

static inline void lb_put(lb_buf *b, const char *t) {
    size_t n = strlen(t);
    if (b->len + n + 2 > b->cap) {
        b->cap = (b->len + n + 2) * 2;
        b->s = (char *)realloc(b->s, b->cap);
    }
    memcpy(b->s + b->len, t, n);
    b->len += n;
    b->s[b->len] = 0;
}

// The next whitespace-separated token of *p, copied into tok (cap bytes), or 0 at the end.
static inline int lb_token(const char **p, char **tok, size_t *cap) {
    const char *s = *p;
    while (*s == ' ' || *s == '\n' || *s == '\t') s++;
    if (!*s) return 0;
    const char *e = s;
    while (*e && *e != ' ' && *e != '\n' && *e != '\t') e++;
    size_t n = (size_t)(e - s);
    if (n + 1 > *cap) {
        *cap = n + 1;
        *tok = (char *)realloc(*tok, *cap);
    }
    memcpy(*tok, s, n);
    (*tok)[n] = 0;
    *p = e;
    return 1;
}

#ifdef __cplusplus
}
#endif
#endif
