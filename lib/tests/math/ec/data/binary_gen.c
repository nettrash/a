/* Generates binary_openssl.txt: the 18 binary curves of SEC 2 as OpenSSL's EC2M has them, and
 * products, sums and SEC 1 encodings of points on each, for tests/math/ec/binary.rs.
 *
 *   cc -O2 -I/opt/homebrew/opt/openssl@4/include binary_gen.c \
 *      -L/opt/homebrew/opt/openssl@4/lib -lcrypto -o binary_gen
 *   ./binary_gen 1 > binary_openssl.txt      (the argument multiplies the random entries)
 *
 * Scalars and points come from a xorshift generator with a fixed seed for each curve. Each product
 * k·P is computed twice, by EC_POINT_mul and by a double-and-add of EC_POINT_dbl and EC_POINT_add,
 * and the program stops if the two differ: OpenSSL 4.1's EC_POINT_mul is wrong for some points
 * outside the prime-order subgroup on the Koblitz curves of cofactor 4, as for (#E - 1)·(1, 0) on
 * sect283k1, though none came up with these seeds (at 1 and at 12). Lines, all numbers and encodings in hexadecimal:
 *
 *   curve NAME                        the curve's parameters, from EC_GROUP_get_curve,
 *   modulus E1 E2 ...                 EC_GROUP_get0_generator, the order and the cofactor
 *   a A / b B / gx X / gy Y / order N / cofactor H
 *   mulg K R                          R = K·G
 *   mulp P K R                        R = K·P
 *   add P Q R                         R = P + Q
 *   enc U C H                         the point U, uncompressed, and the first bytes of its
 *                                     compressed and hybrid forms
 *   check U V                         V, a 0 or 1 for each, tells whether EC_POINT_oct2point
 *                                     accepts each of 15 changes of the encoding U, in the order of
 *                                     changes() below
 *   nox X V                           an x with no point, and whether 02 X and 03 X are accepted
 *   zero V                            whether 02 0…0, 03 0…0, 00, 00 00, the empty string, and
 *                                     the hybrid forms 06 and 07 of (0, √b) are accepted
 *
 * Points other than U are in the compressed form, "00" for the point at infinity.
 */
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/err.h>
#include <openssl/obj_mac.h>
#include <openssl/objects.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long state;
static unsigned long long next(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

static BN_CTX *ctx;
static const EC_GROUP *g;
static int len; /* bytes of an element */
static int m;

static void hex(const unsigned char *b, size_t n) {
  for (size_t i = 0; i < n; i++) printf("%02X", b[i]);
}

static void put_point(const EC_POINT *p, point_conversion_form_t form) {
  unsigned char buf[400];
  size_t n = EC_POINT_point2oct(g, p, form, buf, sizeof buf, ctx);
  if (n == 0) { fprintf(stderr, "point2oct\n"); exit(1); }
  hex(buf, n);
}

static void put_bn(const BIGNUM *x) {
  char *s = BN_bn2hex(x);
  printf("%s", s);
  OPENSSL_free(s);
}

/* A random number of `bits` bits. */
static BIGNUM *random_bn(int bits) {
  int n = (bits + 7) / 8;
  unsigned char buf[400];
  for (int i = 0; i < n; i++) buf[i] = (unsigned char)next();
  if (bits % 8) buf[0] &= (1 << (bits % 8)) - 1;
  return BN_bin2bn(buf, n, NULL);
}

/* k·P by double-and-add on the group law alone, checked against EC_POINT_mul. */
static EC_POINT *product(const EC_POINT *p, const BIGNUM *k) {
  EC_POINT *r = EC_POINT_new(g), *s = EC_POINT_new(g);
  EC_POINT_set_to_infinity(g, r);
  for (int i = BN_num_bits(k) - 1; i >= 0; i--) {
    EC_POINT_dbl(g, r, r, ctx);
    if (BN_is_bit_set(k, i)) EC_POINT_add(g, r, r, p, ctx);
  }
  EC_POINT_mul(g, s, NULL, p, k, ctx);
  if (EC_POINT_cmp(g, r, s, ctx) != 0) { fprintf(stderr, "EC_POINT_mul differs\n"); exit(1); }
  EC_POINT_free(s);
  return r;
}

/* A random point of the curve, of any order: a random x with a solution, decompressed. */
static EC_POINT *random_point(void) {
  EC_POINT *p = EC_POINT_new(g);
  unsigned char buf[200];
  for (;;) {
    buf[0] = 2 + (next() & 1);
    for (int i = 1; i <= len; i++) buf[i] = (unsigned char)next();
    if (m % 8) buf[1] &= (1 << (m % 8)) - 1;
    ERR_set_mark();
    int ok = EC_POINT_oct2point(g, p, buf, 1 + len, ctx);
    ERR_pop_to_mark();
    if (ok) return p;
  }
}

/* Prints 1 if OpenSSL accepts the encoding, 0 if it does not. */
static void verdict(const unsigned char *b, size_t n) {
  EC_POINT *p = EC_POINT_new(g);
  ERR_set_mark();
  int ok = EC_POINT_oct2point(g, p, b, n, ctx);
  ERR_pop_to_mark();
  printf("%d", ok ? 1 : 0);
  EC_POINT_free(p);
}

/* The verdicts on 15 changes of the uncompressed encoding u of nu bytes. */
static void changes(const unsigned char *u, size_t nu) {
  unsigned char buf[400];
  /* 0: the last bit of y flipped; 1, 2: bit m of x, of y, set. */
  memcpy(buf, u, nu); buf[nu - 1] ^= 1; verdict(buf, nu);
  memcpy(buf, u, nu); buf[1] |= 1 << (m % 8); verdict(buf, nu);
  memcpy(buf, u, nu); buf[1 + len] |= 1 << (m % 8); verdict(buf, nu);
  /* 3, 4: a byte short, a zero byte more. */
  verdict(u, nu - 1);
  memcpy(buf, u, nu); buf[nu] = 0; verdict(buf, nu + 1);
  /* 5 to 10: the first byte 1, 5, 8, FF, 2 and 3 on the whole encoding. */
  int tags[6] = {1, 5, 8, 0xFF, 2, 3};
  for (int s = 0; s < 6; s++) { memcpy(buf, u, nu); buf[0] = (unsigned char)tags[s]; verdict(buf, nu); }
  /* 11, 12: the hybrid form with each bit; 13, 14: the compressed form with each bit. */
  memcpy(buf, u, nu); buf[0] = 6; verdict(buf, nu);
  buf[0] = 7; verdict(buf, nu);
  buf[0] = 2; verdict(buf, 1 + len);
  buf[0] = 3; verdict(buf, 1 + len);
}

/* Prints a point in the compressed form, "00" at infinity. */
static void put_short(const EC_POINT *p) {
  if (EC_POINT_is_at_infinity(g, p)) printf("00"); else put_point(p, POINT_CONVERSION_COMPRESSED);
}

int main(int argc, char **argv) {
  int count = argc > 1 ? atoi(argv[1]) : 1;
  int nids[] = {NID_sect113r1, NID_sect113r2, NID_sect131r1, NID_sect131r2, NID_sect163k1,
                NID_sect163r1, NID_sect163r2, NID_sect193r1, NID_sect193r2, NID_sect233k1,
                NID_sect233r1, NID_sect239k1, NID_sect283k1, NID_sect283r1, NID_sect409k1,
                NID_sect409r1, NID_sect571k1, NID_sect571r1};
  ctx = BN_CTX_new();
  for (int c = 0; c < 18; c++) {
    state = 0x9E3779B97F4A7C15ULL + (unsigned long long)c * 0x1000193ULL;
    EC_GROUP *grp = EC_GROUP_new_by_curve_name(nids[c]);
    g = grp;
    BIGNUM *p = BN_new(), *a = BN_new(), *b = BN_new(), *x = BN_new(), *y = BN_new();
    EC_GROUP_get_curve(g, p, a, b, ctx);
    const EC_POINT *gen = EC_GROUP_get0_generator(g);
    EC_POINT_get_affine_coordinates(g, gen, x, y, ctx);
    const BIGNUM *n = EC_GROUP_get0_order(g);
    m = EC_GROUP_get_degree(g);
    len = (m + 7) / 8;
    int e[8], k = BN_GF2m_poly2arr(p, e, 8);
    printf("curve %s\nmodulus", OBJ_nid2sn(nids[c]));
    for (int j = 0; j < k; j++) printf(" %d", e[j]);
    printf("\na "); put_bn(a);
    printf("\nb "); put_bn(b);
    printf("\ngx "); put_bn(x);
    printf("\ngy "); put_bn(y);
    printf("\norder "); put_bn(n);
    printf("\ncofactor "); put_bn(EC_GROUP_get0_cofactor(g));
    printf("\n");
    int nb = BN_num_bits(n);
    for (int r = 0; r < count; r++) {
      /* k·G: below the order, short, and twice the order's length. */
      int sizes[3] = {nb, 32, 2 * nb + 5};
      for (int s = 0; s < 3; s++) {
        BIGNUM *kk = random_bn(sizes[s]);
        EC_POINT *q = product(gen, kk);
        printf("mulg "); put_bn(kk); printf(" "); put_short(q); printf("\n");
        BN_free(kk);
        EC_POINT_free(q);
      }
      /* k·P: P in the subgroup, then P a random point of the curve, with a scalar of the order's
         length and one of twice it. */
      for (int s = 0; s < 3; s++) {
        EC_POINT *pt;
        if (s == 0) {
          BIGNUM *j = random_bn(nb);
          pt = product(gen, j);
          BN_free(j);
        } else {
          pt = random_point();
        }
        BIGNUM *kk = random_bn(s == 2 ? 2 * nb + 5 : nb);
        EC_POINT *q = product(pt, kk);
        printf("mulp "); put_short(pt); printf(" "); put_bn(kk); printf(" "); put_short(q);
        printf("\n");
        BN_free(kk);
        EC_POINT_free(q);
        EC_POINT_free(pt);
      }
      /* P + Q, P + P and P + (-P). */
      {
        EC_POINT *p1 = random_point(), *p2 = random_point(), *q = EC_POINT_new(g), *mp = EC_POINT_new(g);
        EC_POINT_copy(mp, p1);
        EC_POINT_invert(g, mp, ctx);
        const EC_POINT *rhs[3] = {p2, p1, mp};
        for (int s = 0; s < 3; s++) {
          EC_POINT_add(g, q, p1, rhs[s], ctx);
          printf("add "); put_short(p1); printf(" "); put_short(rhs[s]); printf(" "); put_short(q);
          printf("\n");
        }
        EC_POINT_free(p1); EC_POINT_free(p2); EC_POINT_free(q); EC_POINT_free(mp);
      }
      /* The encodings of a random point, of the generator and, once, of the point (0, √b). */
      {
        EC_POINT *pts[3];
        int np = 2;
        pts[0] = random_point();
        pts[1] = EC_POINT_dup(gen, g);
        if (r == 0) {
          unsigned char t[200] = {2};
          pts[2] = EC_POINT_new(g);
          if (!EC_POINT_oct2point(g, pts[2], t, 1 + len, ctx)) { fprintf(stderr, "(0, sqrt b)\n"); exit(1); }
          np = 3;
        }
        for (int s = 0; s < np; s++) {
          unsigned char cb[400], hb[400];
          EC_POINT_point2oct(g, pts[s], POINT_CONVERSION_COMPRESSED, cb, sizeof cb, ctx);
          size_t nh = EC_POINT_point2oct(g, pts[s], POINT_CONVERSION_HYBRID, hb, sizeof hb, ctx);
          unsigned char ub[400];
          EC_POINT_point2oct(g, pts[s], POINT_CONVERSION_UNCOMPRESSED, ub, sizeof ub, ctx);
          /* The forms share x, and the hybrid one y, with the uncompressed one. */
          if (memcmp(cb + 1, ub + 1, len) || memcmp(hb + 1, ub + 1, nh - 1)) { fprintf(stderr, "forms\n"); exit(1); }
          printf("enc "); put_point(pts[s], POINT_CONVERSION_UNCOMPRESSED);
          printf(" %02X %02X\n", cb[0], hb[0]);
          EC_POINT_free(pts[s]);
        }
      }
      /* Encodings, valid and not, and OpenSSL's verdict on each. */
      {
        unsigned char buf[400], u[400];
        EC_POINT *pt = random_point();
        size_t nu = EC_POINT_point2oct(g, pt, POINT_CONVERSION_UNCOMPRESSED, u, sizeof u, ctx);
        printf("check ");
        hex(u, nu);
        printf(" ");
        changes(u, nu);
        printf("\n");
        /* A compressed x with no point: random x until OpenSSL refuses it. */
        for (;;) {
          buf[0] = 2;
          for (int i = 1; i <= len; i++) buf[i] = (unsigned char)next();
          if (m % 8) buf[1] &= (1 << (m % 8)) - 1;
          EC_POINT *t = EC_POINT_new(g);
          ERR_set_mark();
          int ok = EC_POINT_oct2point(g, t, buf, 1 + len, ctx);
          ERR_pop_to_mark();
          EC_POINT_free(t);
          if (!ok) break;
        }
        printf("nox ");
        hex(buf + 1, len);
        printf(" ");
        verdict(buf, 1 + len);
        buf[0] = 3;
        verdict(buf, 1 + len);
        printf("\n");
        if (r == 0) {
          memset(buf, 0, sizeof buf);
          printf("zero ");
          buf[0] = 2;
          verdict(buf, 1 + len);
          buf[0] = 3;
          verdict(buf, 1 + len);
          buf[0] = 0;
          verdict(buf, 1);
          verdict(buf, 2);
          verdict(buf, 0);
          EC_POINT *t = EC_POINT_new(g);
          buf[0] = 2;
          EC_POINT_oct2point(g, t, buf, 1 + len, ctx);
          size_t nt = EC_POINT_point2oct(g, t, POINT_CONVERSION_UNCOMPRESSED, buf, sizeof buf, ctx);
          buf[0] = 6;
          verdict(buf, nt);
          buf[0] = 7;
          verdict(buf, nt);
          EC_POINT_free(t);
          printf("\n");
        }
        EC_POINT_free(pt);
      }
    }
    BN_free(p); BN_free(a); BN_free(b); BN_free(x); BN_free(y);
    EC_GROUP_free(grp);
  }
  return 0;
}
