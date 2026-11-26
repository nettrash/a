/* Common input decoding and output fingerprints for the reference shims (FLINT, NTL, PARI).
 *
 * Every shim exports, for its prefix P (flint, ntl, pari, lapack):
 *   void*  P_prep(const char* op, const uint64_t* buf, size_t n);  NULL when unsupported
 *   void   P_run(void* h, uint64_t reps);                           reps calls, inputs cycled
 *   size_t P_check(void* h, uint64_t* out, size_t cap);             one call per variant, outputs
 *   void   P_free(void* h);
 *
 * The buffer: [S, S shared args, V, A, V*A variant args]. An argument is a kind word and data:
 *   1 U64   [1, x]
 *   2 WVEC  [2, len, x0 .. x(len-1)]                     words (residues, points)
 *   3 INT   [3, neg, nl, limb0 .. limb(nl-1)]            a signed integer, limbs little-endian
 *   4 IVEC  [4, len, then len times: neg, nl, limbs]     integers (coefficients, points)
 *   5 F64V  [5, len, bits0 ..]                           doubles
 *   6 QVEC  [6, len, then len times: numerator, denominator as (neg, nl, limbs)]
 *
 * The outputs are words: a polynomial is [len, h] with h = sum c_i R^i mod FP (FP = 2^61 - 1,
 * R = FP_R), its coefficients taken mod FP (the residue of a negative one is FP - |c| mod FP);
 * each operation's own layout is in the Rust harness (src/main.rs, fn compare). */
#ifndef POLYBENCH_SHIM_H
#define POLYBENCH_SHIM_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define FP ((uint64_t)0x1FFFFFFFFFFFFFFFULL)
#define FP_R ((uint64_t)0x123456789ABCDEFULL)

static inline uint64_t fp_red(unsigned __int128 x) {
    uint64_t lo = (uint64_t)(x & FP), hi = (uint64_t)(x >> 61);
    uint64_t s = lo + (hi & FP) + (uint64_t)(hi >> 61);
    while (s >= FP) s -= FP;
    return s;
}
static inline uint64_t fp_mul(uint64_t a, uint64_t b) { return fp_red((unsigned __int128)a * b); }
static inline uint64_t fp_add(uint64_t a, uint64_t b) { uint64_t s = a + b; return s >= FP ? s - FP : s; }
static inline uint64_t fp_neg(uint64_t a) { return a ? FP - a : 0; }
/* Horner step from the top coefficient down: h = h*R + c. */
static inline uint64_t fp_step(uint64_t h, uint64_t c) { return fp_add(fp_mul(h, FP_R), c % FP); }
/* A signed integer given by limbs, mod FP. */
static inline uint64_t fp_limbs(int neg, const uint64_t* l, size_t nl) {
    uint64_t h = 0;
    for (size_t i = nl; i-- > 0;) {
        /* h = h * 2^64 + l[i] (mod FP); 2^64 = 8 (mod 2^61 - 1) */
        h = fp_add(fp_mul(h, 8), l[i] % FP);
    }
    return neg ? fp_neg(h) : h;
}
/* fp of a word vector as a polynomial, constant first. */
static inline uint64_t fp_wvec(const uint64_t* c, size_t len) {
    uint64_t h = 0;
    for (size_t i = len; i-- > 0;) h = fp_step(h, c[i]);
    return h;
}

enum { K_U64 = 1, K_WVEC = 2, K_INT = 3, K_IVEC = 4, K_F64V = 5, K_QVEC = 6 };

typedef struct {
    int kind;
    uint64_t u;          /* K_U64 */
    size_t len;          /* vector length, or nl for K_INT */
    int neg;             /* K_INT */
    const uint64_t* p;   /* data start: words, limbs, or the first encoded integer */
} arg_t;

typedef struct {
    size_t ns;           /* shared args */
    arg_t shared[8];
    size_t v, a;         /* variants, args per variant */
    arg_t* var;          /* v * a */
} input_t;

/* The words an encoded integer (neg, nl, limbs) takes. */
static inline size_t enc_int_len(const uint64_t* p) { return 2 + (size_t)p[1]; }

static inline const uint64_t* parse_arg(const uint64_t* p, arg_t* a) {
    memset(a, 0, sizeof *a);
    a->kind = (int)p[0];
    switch (a->kind) {
    case K_U64: a->u = p[1]; return p + 2;
    case K_WVEC: case K_F64V: a->len = p[1]; a->p = p + 2; return p + 2 + a->len;
    case K_INT: a->neg = (int)p[1]; a->len = p[2]; a->p = p + 3; return p + 3 + a->len;
    case K_IVEC: {
        a->len = p[1]; a->p = p + 2;
        const uint64_t* q = p + 2;
        for (size_t i = 0; i < a->len; i++) q += enc_int_len(q);
        return q;
    }
    case K_QVEC: {
        a->len = p[1]; a->p = p + 2;
        const uint64_t* q = p + 2;
        for (size_t i = 0; i < a->len; i++) { q += enc_int_len(q); q += enc_int_len(q); }
        return q;
    }
    default: abort();
    }
}

static inline void parse_input(const uint64_t* buf, size_t n, input_t* in) {
    (void)n;
    const uint64_t* p = buf;
    in->ns = (size_t)*p++;
    for (size_t i = 0; i < in->ns; i++) p = parse_arg(p, &in->shared[i]);
    in->v = (size_t)*p++;
    in->a = (size_t)*p++;
    in->var = (arg_t*)calloc(in->v * in->a + 1, sizeof(arg_t));
    for (size_t i = 0; i < in->v * in->a; i++) p = parse_arg(p, &in->var[i]);
}
static inline arg_t* VA(input_t* in, size_t v, size_t i) { return &in->var[v * in->a + i]; }

/* Output collector. */
typedef struct { uint64_t* out; size_t cap, n; } outbuf_t;
static inline void put(outbuf_t* o, uint64_t x) { if (o->n < o->cap) o->out[o->n] = x; o->n++; }

/* Order-independent combination of factor fingerprints (with multiplicities). */
static inline uint64_t fp_mix(uint64_t h) { return fp_add(fp_mul(h, h), fp_mul(h, 0x2545F4914F6CDD1DULL % FP)); }

#define SHIM_EXPORTS(P) \
    void* P##_prep(const char* op, const uint64_t* buf, size_t n); \
    void P##_run(void* h, uint64_t reps); \
    size_t P##_check(void* h, uint64_t* out, size_t cap); \
    void P##_free(void* h);

#endif
