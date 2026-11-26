// NTL references of the fields scoreboard: ZZ_p for prime fields, ZZ_pE and zz_pE for extension
// fields, GF2E (over GF2X) for binary fields. Built by build.rs at -O3 with the native CPU flags
// when NTL is found. NTL keeps its moduli in global contexts, so each handle saves its own and
// restores it before it works. See shim.h for the interface.
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <NTL/lzz_p.h>
#include <NTL/lzz_pX.h>
#include <NTL/lzz_pE.h>
#include <NTL/GF2X.h>
#include <NTL/GF2E.h>
#include <NTL/GF2XFactoring.h>
#include <vector>
#include <cstring>
#include "shim.h"

using namespace NTL;

static volatile unsigned long sink;

static ZZ zz_of(const uint64_t *l, size_t n) {
    ZZ z;
    ZZFromBytes(z, reinterpret_cast<const unsigned char *>(l), (long)(n * 8));
    return z;
}

static void limbs_of(uint64_t *out, const ZZ &z, size_t n) {
    std::memset(out, 0, n * 8);
    BytesFromZZ(reinterpret_cast<unsigned char *>(out), z, (long)(n * 8));
}

extern "C" {

// ---------------------------------------------------------------- ZZ_p

struct NtlPf {
    ZZ_pContext ctx;
    ZZ p, e;
    std::vector<ZZ_p> xs;
    std::vector<ZZ> xz, sqz;
    ZZ_p acc, t;
    size_t n, count;
};

void *ntl_pf_new(const uint64_t *p, size_t n, const uint64_t *xs, size_t count, const uint64_t *e, size_t en) {
    NtlPf *h = new NtlPf;
    h->p = zz_of(p, n);
    h->e = zz_of(e, en);
    ZZ_p::init(h->p);
    h->ctx.save();
    h->n = n;
    h->count = count;
    for (size_t i = 0; i < count; i++) {
        ZZ z = zz_of(xs + i * n, n);
        h->xz.push_back(z);
        h->sqz.push_back(MulMod(z, z, h->p));
        h->xs.push_back(conv<ZZ_p>(z));
    }
    h->acc = h->xs[0];
    return h;
}

void ntl_pf_free(void *v) {
    NtlPf *h = static_cast<NtlPf *>(v);
    h->ctx.restore();
    delete h;
}

int ntl_pf_out(void *v, int op, size_t i, uint64_t *out) {
    NtlPf *h = static_cast<NtlPf *>(v);
    h->ctx.restore();
    size_t j = (i + 1) & (h->count - 1);
    ZZ_p r;
    ZZ rz;
    int ret = 0, isz = 0;
    switch (op) {
    case OP_MUL: mul(r, h->xs[i], h->xs[j]); break;
    case OP_SQR: sqr(r, h->xs[i]); break;
    case OP_ADD: add(r, h->xs[i], h->xs[j]); break;
    case OP_SUB: sub(r, h->xs[i], h->xs[j]); break;
    case OP_INV: inv(r, h->xs[i]); break;
    case OP_SQRT: SqrRootMod(rz, h->sqz[i], h->p); isz = 1; ret = 1; break;
    case OP_LEG: ret = (int)Jacobi(h->xz[i], h->p); break;
    case OP_POW: power(r, h->xs[i], h->e); break;
    default: ret = UNSUPPORTED;
    }
    limbs_of(out, isz ? rz : rep(r), h->n);
    return ret;
}

void ntl_pf_run(void *v, int op, uint64_t iters) {
    NtlPf *h = static_cast<NtlPf *>(v);
    h->ctx.restore();
    size_t m = h->count - 1;
    long s = 0;
    ZZ_p &acc = h->acc, &t = h->t;
    ZZ tz;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) mul(acc, acc, h->xs[i & m]); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) sqr(acc, acc); break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) add(acc, acc, h->xs[i & m]); break;
    case OP_SUB: for (uint64_t i = 0; i < iters; i++) sub(acc, acc, h->xs[i & m]); break;
    case OP_INV: for (uint64_t i = 0; i < iters; i++) { inv(t, h->xs[i & m]); add(acc, acc, t); } break;
    case OP_SQRT:
        for (uint64_t i = 0; i < iters; i++) { SqrRootMod(tz, h->sqz[i & m], h->p); add(acc, acc, conv<ZZ_p>(tz)); }
        break;
    case OP_LEG: for (uint64_t i = 0; i < iters; i++) s += Jacobi(h->xz[i & m], h->p); break;
    case OP_POW: for (uint64_t i = 0; i < iters; i++) { power(t, h->xs[i & m], h->e); add(acc, acc, t); } break;
    }
    if (IsZero(acc)) acc = h->xs[0];
    sink = (unsigned long)s + (unsigned long)conv<long>(rep(acc) % 1000);
}

// ---------------------------------------------------------------- ZZ_pE

struct NtlExt {
    ZZ_pContext pctx;
    ZZ_pEContext ectx;
    ZZ p, e;
    ZZ_pX g;
    ZZ_pXModulus G;
    std::vector<ZZ_pE> xs;
    ZZ_pE acc, t;
    ZZ_p nacc, nt;
    size_t n, k, count;
};

static ZZ_pE ext_of(const uint64_t *c, size_t k, size_t n) {
    ZZ_pX a;
    for (size_t j = 0; j < k; j++) SetCoeff(a, (long)j, conv<ZZ_p>(zz_of(c + j * n, n)));
    return conv<ZZ_pE>(a);
}

static void ext_limbs(uint64_t *out, const ZZ_pE &x, size_t k, size_t n) {
    for (size_t j = 0; j < k; j++) limbs_of(out + j * n, rep(coeff(rep(x), (long)j)), n);
}

void *ntl_ext_new(const uint64_t *p, size_t n, const uint64_t *g, size_t k, const uint64_t *xs, size_t count,
                  const uint64_t *e, size_t en) {
    NtlExt *h = new NtlExt;
    h->p = zz_of(p, n);
    h->e = zz_of(e, en);
    ZZ_p::init(h->p);
    h->pctx.save();
    for (size_t j = 0; j <= k; j++) SetCoeff(h->g, (long)j, conv<ZZ_p>(zz_of(g + j * n, n)));
    build(h->G, h->g);
    ZZ_pE::init(h->g);
    h->ectx.save();
    h->n = n;
    h->k = k;
    h->count = count;
    for (size_t i = 0; i < count; i++) h->xs.push_back(ext_of(xs + i * k * n, k, n));
    h->acc = h->xs[0];
    return h;
}

void ntl_ext_free(void *v) {
    NtlExt *h = static_cast<NtlExt *>(v);
    h->pctx.restore();
    h->ectx.restore();
    delete h;
}

int ntl_ext_out(void *v, int op, size_t i, uint64_t *out) {
    NtlExt *h = static_cast<NtlExt *>(v);
    h->pctx.restore();
    h->ectx.restore();
    size_t j = (i + 1) & (h->count - 1);
    ZZ_pE r;
    ZZ_p s;
    int ret = 0, scalar = 0;
    switch (op) {
    case OP_MUL: mul(r, h->xs[i], h->xs[j]); break;
    case OP_ADD: add(r, h->xs[i], h->xs[j]); break;
    case OP_SQR: sqr(r, h->xs[i]); break;
    case OP_INV: inv(r, h->xs[i]); break;
    case OP_FROB: power(r, h->xs[i], h->p); break;
    case OP_NORM: NormMod(s, rep(h->xs[i]), h->g); scalar = 1; break;
    case OP_TRACE: TraceMod(s, rep(h->xs[i]), h->G); scalar = 1; break;
    case OP_POW: power(r, h->xs[i], h->e); break;
    default: ret = UNSUPPORTED;
    }
    if (scalar) limbs_of(out, rep(s), h->n); else ext_limbs(out, r, h->k, h->n);
    return ret;
}

void ntl_ext_run(void *v, int op, uint64_t iters) {
    NtlExt *h = static_cast<NtlExt *>(v);
    h->pctx.restore();
    h->ectx.restore();
    size_t m = h->count - 1;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) mul(h->acc, h->acc, h->xs[i & m]); break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) add(h->acc, h->acc, h->xs[i & m]); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) sqr(h->acc, h->acc); break;
    case OP_INV: for (uint64_t i = 0; i < iters; i++) { inv(h->t, h->xs[i & m]); add(h->acc, h->acc, h->t); } break;
    case OP_FROB: for (uint64_t i = 0; i < iters; i++) { power(h->t, h->xs[i & m], h->p); add(h->acc, h->acc, h->t); } break;
    case OP_NORM: for (uint64_t i = 0; i < iters; i++) { NormMod(h->nt, rep(h->xs[i & m]), h->g); add(h->nacc, h->nacc, h->nt); } break;
    case OP_TRACE: for (uint64_t i = 0; i < iters; i++) { TraceMod(h->nt, rep(h->xs[i & m]), h->G); add(h->nacc, h->nacc, h->nt); } break;
    case OP_POW: for (uint64_t i = 0; i < iters; i++) { power(h->t, h->xs[i & m], h->e); add(h->acc, h->acc, h->t); } break;
    }
    if (IsZero(h->acc)) h->acc = h->xs[0];
    sink = (unsigned long)deg(rep(h->acc)) + (unsigned long)IsZero(h->nacc);
}

// ---------------------------------------------------------------- zz_pE (p below NTL_SP_BOUND)

struct NtlSpExt {
    zz_pContext pctx;
    zz_pEContext ectx;
    ZZ e;
    long p;
    zz_pX g;
    zz_pXModulus G;
    std::vector<zz_pE> xs;
    zz_pE acc, t;
    zz_p nacc, nt;
    size_t k, count;
};

void *ntl_sp_ext_new(const uint64_t *p, size_t n, const uint64_t *g, size_t k, const uint64_t *xs, size_t count,
                     const uint64_t *e, size_t en) {
    if (n != 1 || p[0] >= (uint64_t)NTL_SP_BOUND) return nullptr;
    NtlSpExt *h = new NtlSpExt;
    h->p = (long)p[0];
    h->e = zz_of(e, en);
    zz_p::init(h->p);
    h->pctx.save();
    for (size_t j = 0; j <= k; j++) SetCoeff(h->g, (long)j, conv<zz_p>((long)g[j]));
    build(h->G, h->g);
    zz_pE::init(h->g);
    h->ectx.save();
    h->k = k;
    h->count = count;
    for (size_t i = 0; i < count; i++) {
        zz_pX a;
        for (size_t j = 0; j < k; j++) SetCoeff(a, (long)j, conv<zz_p>((long)xs[i * k + j]));
        h->xs.push_back(conv<zz_pE>(a));
    }
    h->acc = h->xs[0];
    return h;
}

void ntl_sp_ext_free(void *v) {
    NtlSpExt *h = static_cast<NtlSpExt *>(v);
    h->pctx.restore();
    h->ectx.restore();
    delete h;
}

int ntl_sp_ext_out(void *v, int op, size_t i, uint64_t *out) {
    NtlSpExt *h = static_cast<NtlSpExt *>(v);
    h->pctx.restore();
    h->ectx.restore();
    size_t j = (i + 1) & (h->count - 1);
    zz_pE r;
    zz_p s;
    int ret = 0, scalar = 0;
    switch (op) {
    case OP_MUL: mul(r, h->xs[i], h->xs[j]); break;
    case OP_ADD: add(r, h->xs[i], h->xs[j]); break;
    case OP_SQR: sqr(r, h->xs[i]); break;
    case OP_INV: inv(r, h->xs[i]); break;
    case OP_FROB: power(r, h->xs[i], h->p); break;
    case OP_NORM: NormMod(s, rep(h->xs[i]), h->g); scalar = 1; break;
    case OP_TRACE: TraceMod(s, rep(h->xs[i]), h->G); scalar = 1; break;
    case OP_POW: power(r, h->xs[i], h->e); break;
    default: ret = UNSUPPORTED;
    }
    if (scalar) out[0] = (uint64_t)rep(s);
    else for (size_t c = 0; c < h->k; c++) out[c] = (uint64_t)rep(coeff(rep(r), (long)c));
    return ret;
}

void ntl_sp_ext_run(void *v, int op, uint64_t iters) {
    NtlSpExt *h = static_cast<NtlSpExt *>(v);
    h->pctx.restore();
    h->ectx.restore();
    size_t m = h->count - 1;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) mul(h->acc, h->acc, h->xs[i & m]); break;
    case OP_ADD: for (uint64_t i = 0; i < iters; i++) add(h->acc, h->acc, h->xs[i & m]); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) sqr(h->acc, h->acc); break;
    case OP_INV: for (uint64_t i = 0; i < iters; i++) { inv(h->t, h->xs[i & m]); add(h->acc, h->acc, h->t); } break;
    case OP_FROB: for (uint64_t i = 0; i < iters; i++) { power(h->t, h->xs[i & m], h->p); add(h->acc, h->acc, h->t); } break;
    case OP_NORM: for (uint64_t i = 0; i < iters; i++) { NormMod(h->nt, rep(h->xs[i & m]), h->g); add(h->nacc, h->nacc, h->nt); } break;
    case OP_TRACE: for (uint64_t i = 0; i < iters; i++) { TraceMod(h->nt, rep(h->xs[i & m]), h->G); add(h->nacc, h->nacc, h->nt); } break;
    case OP_POW: for (uint64_t i = 0; i < iters; i++) { power(h->t, h->xs[i & m], h->e); add(h->acc, h->acc, h->t); } break;
    }
    if (IsZero(h->acc)) h->acc = h->xs[0];
    sink = (unsigned long)deg(rep(h->acc)) + (unsigned long)rep(h->nacc);
}

// ---------------------------------------------------------------- GF2E

struct NtlBin {
    GF2EContext ctx;
    GF2X f;
    long m;
    ZZ half; // 2^(m-1), for square roots by a power
    std::vector<GF2E> xs;
    GF2E acc, t;
    long e;
    size_t w, count;
};

static GF2X gf2x_of(const uint64_t *l, size_t w) {
    GF2X x;
    GF2XFromBytes(x, reinterpret_cast<const unsigned char *>(l), (long)(w * 8));
    return x;
}

static void gf2x_limbs(uint64_t *out, const GF2X &x, size_t w) {
    std::memset(out, 0, w * 8);
    BytesFromGF2X(reinterpret_cast<unsigned char *>(out), x, (long)(w * 8));
}

void *ntl_bin_new(const uint32_t *exps, size_t nexp, const uint64_t *xs, size_t count, uint64_t e) {
    NtlBin *h = new NtlBin;
    for (size_t j = 0; j < nexp; j++) SetCoeff(h->f, (long)exps[j]);
    h->m = (long)exps[0];
    GF2E::init(h->f);
    h->ctx.save();
    h->half = power2_ZZ(h->m - 1);
    h->w = (exps[0] + 63) / 64;
    h->count = count;
    h->e = (long)e;
    for (size_t i = 0; i < count; i++) h->xs.push_back(conv<GF2E>(gf2x_of(xs + i * h->w, h->w)));
    h->acc = h->xs[0];
    return h;
}

void ntl_bin_free(void *v) {
    NtlBin *h = static_cast<NtlBin *>(v);
    h->ctx.restore();
    delete h;
}

int ntl_bin_out(void *v, int op, size_t i, uint64_t *out) {
    NtlBin *h = static_cast<NtlBin *>(v);
    h->ctx.restore();
    size_t j = (i + 1) & (h->count - 1);
    GF2E r;
    int ret = 0;
    switch (op) {
    case OP_MUL: mul(r, h->xs[i], h->xs[j]); break;
    case OP_SQR: sqr(r, h->xs[i]); break;
    case OP_INV: inv(r, h->xs[i]); break;
    case OP_DIV: div(r, h->xs[i], h->xs[j]); break;
    case OP_SQRT: power(r, h->xs[i], h->half); break;
    case OP_TRACE: ret = (int)rep(trace(h->xs[i])); break;
    case OP_POWU: power(r, h->xs[i], h->e); break;
    default: ret = UNSUPPORTED;
    }
    gf2x_limbs(out, rep(r), h->w);
    return ret;
}

void ntl_bin_run(void *v, int op, uint64_t iters) {
    NtlBin *h = static_cast<NtlBin *>(v);
    h->ctx.restore();
    size_t m = h->count - 1;
    long s = 0;
    switch (op) {
    case OP_MUL: for (uint64_t i = 0; i < iters; i++) mul(h->acc, h->acc, h->xs[i & m]); break;
    case OP_SQR: for (uint64_t i = 0; i < iters; i++) sqr(h->acc, h->acc); break;
    case OP_INV: for (uint64_t i = 0; i < iters; i++) { inv(h->t, h->xs[i & m]); add(h->acc, h->acc, h->t); } break;
    case OP_DIV: for (uint64_t i = 0; i < iters; i++) div(h->acc, h->acc, h->xs[i & m]); break;
    case OP_SQRT: for (uint64_t i = 0; i < iters; i++) { power(h->t, h->xs[i & m], h->half); add(h->acc, h->acc, h->t); } break;
    case OP_TRACE: for (uint64_t i = 0; i < iters; i++) s ^= (long)rep(trace(h->xs[i & m])); break;
    case OP_POWU: for (uint64_t i = 0; i < iters; i++) { power(h->t, h->xs[i & m], h->e); add(h->acc, h->acc, h->t); } break;
    }
    if (IsZero(h->acc)) h->acc = h->xs[0];
    sink = (unsigned long)deg(rep(h->acc)) + (unsigned long)s;
}

// Rabin's test of irreducibility over GF(2), as BinaryField::is_irreducible: NTL's IterIrredTest.
long ntl_gf2x_irred(const uint32_t *exps, size_t nexp, uint64_t reps) {
    GF2X f;
    for (size_t j = 0; j < nexp; j++) SetCoeff(f, (long)exps[j]);
    long r = 0;
    for (uint64_t i = 0; i < reps; i++) r += IterIrredTest(f);
    return r;
}

} // extern "C"
