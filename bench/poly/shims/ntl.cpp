// NTL reference shim: zz_pX (moduli below NTL_SP_BOUND), ZZ_pX (larger moduli), GF2X (modulus 2),
// ZZX (integers).
#include <NTL/lzz_pX.h>
#include <NTL/lzz_pXFactoring.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pXFactoring.h>
#include <NTL/GF2X.h>
#include <NTL/GF2XFactoring.h>
#include <NTL/ZZX.h>
#include <NTL/ZZXFactoring.h>
#include <NTL/BasicThreadPool.h>
#include <vector>
#include <string>
#include <algorithm>
#include <type_traits>
#include "shim.h"

using namespace NTL;

extern "C" {
SHIM_EXPORTS(sn)
const char* sn_label(void* h);
}

static void set_zz(ZZ& z, const uint64_t* q) {
    size_t nl = (size_t)q[1];
    if (nl == 0) { clear(z); return; }
    ZZFromBytes(z, (const unsigned char*)(q + 2), (long)(nl * 8));
    if (q[0]) NTL::negate(z, z);
}
static void set_zz_arg(ZZ& z, const arg_t* a) {
    if (a->kind == K_U64) { conv(z, (unsigned long)a->u); return; }
    if (a->len == 0) { clear(z); return; }
    ZZFromBytes(z, (const unsigned char*)a->p, (long)(a->len * 8));
    if (a->neg) NTL::negate(z, z);
}
static uint64_t fp_zz(const ZZ& z) {
    long r = rem(z, (long)FP);
    if (r < 0) r += (long)FP;
    return (uint64_t)r;
}
static inline uint64_t fp_c(const zz_p& c) { return (uint64_t)rep(c) % FP; }
static inline uint64_t fp_c(const ZZ_p& c) { return fp_zz(rep(c)); }
static inline uint64_t fp_c(const ZZ& c) { return fp_zz(c); }
static inline uint64_t fp_c(GF2 c) { return (uint64_t)rep(c); }

template <class P> static uint64_t fp_poly(const P& p) {
    uint64_t h = 0;
    for (long i = deg(p); i >= 0; i--) h = fp_add(fp_mul(h, FP_R), fp_c(coeff(p, i)));
    return h;
}
template <class P> static void put_poly(outbuf_t* o, const P& p) {
    put(o, (uint64_t)(deg(p) + 1)); put(o, fp_poly(p));
}

enum Op {
    W_MUL = 1, W_SQR, W_DIVREM, W_INV_SERIES, W_MULMOD, W_POWMOD_X, W_POWMOD, W_COMPOSE_MOD, W_POW,
    W_EVAL, W_DERIVATIVE, W_ADD, W_SCALE, W_GCD, W_XGCD, W_RESULTANT, W_EVAL_MANY, W_FROM_ROOTS,
    W_INTERPOLATE, W_FACTOR, W_FACTOR_BERLEKAMP, W_IS_IRREDUCIBLE, W_SQUAREFREE, W_DDF, W_EDF,
    Z_MUL, Z_SQR, Z_PDIVREM, Z_DIV_EXACT, Z_DERIVATIVE, Z_ADD, Z_CONTENT, Z_GCD, Z_RESULTANT,
    Z_DISCRIMINANT, Z_FACTOR, Z_SQUAREFREE
};

static int lookup(const char* name, char* fam) {
    static const struct { const char* n; int op; } T[] = {
        {"mul", W_MUL}, {"sqr", W_SQR}, {"divrem", W_DIVREM}, {"inv_series", W_INV_SERIES},
        {"mulmod", W_MULMOD}, {"powmod_x", W_POWMOD_X}, {"powmod", W_POWMOD},
        {"compose_mod", W_COMPOSE_MOD}, {"pow", W_POW}, {"eval", W_EVAL},
        {"derivative", W_DERIVATIVE}, {"add", W_ADD}, {"scale", W_SCALE}, {"gcd", W_GCD},
        {"xgcd", W_XGCD}, {"resultant", W_RESULTANT}, {"eval_many", W_EVAL_MANY},
        {"from_roots", W_FROM_ROOTS}, {"interpolate", W_INTERPOLATE}, {"factor", W_FACTOR},
        {"factor_berlekamp", W_FACTOR_BERLEKAMP}, {"is_irreducible", W_IS_IRREDUCIBLE},
        {"squarefree", W_SQUAREFREE}, {"ddf", W_DDF}, {"edf", W_EDF}, {nullptr, 0}};
    static const struct { const char* n; int op; } Z[] = {
        {"z_mul", Z_MUL}, {"z_sqr", Z_SQR}, {"z_pdivrem", Z_PDIVREM}, {"z_div_exact", Z_DIV_EXACT},
        {"z_derivative", Z_DERIVATIVE}, {"z_add", Z_ADD}, {"z_content", Z_CONTENT},
        {"z_gcd", Z_GCD}, {"z_resultant", Z_RESULTANT}, {"z_discriminant", Z_DISCRIMINANT},
        {"z_factor", Z_FACTOR}, {"z_squarefree", Z_SQUAREFREE}, {nullptr, 0}};
    *fam = name[0];
    if (name[0] == 'z') {
        for (int i = 0; Z[i].n; i++) if (!strcmp(Z[i].n, name)) return Z[i].op;
        return 0;
    }
    if ((name[0] != 'w' && name[0] != 'm') || name[1] != '_') return 0;
    for (int i = 0; T[i].n; i++) if (!strcmp(T[i].n, name + 2)) return T[i].op;
    return 0;
}

struct Base {
    int op; size_t V; std::string label;
    virtual ~Base() {}
    virtual void run1(size_t v) = 0;
    virtual void out1(size_t v, outbuf_t* o) = 0;
    virtual void enter() {}
};

// zz_pX / ZZ_pX share one template.
template <class P, class C, class VC, class Mod, class VP, class Pair, class Ctx> struct ModCase : Base {
    Ctx ctx;
    std::vector<P> a, b, c;
    std::vector<Mod> F;
    std::vector<VC> x, y;
    std::vector<C> s;
    ZZ e; long k = 0;
    P r0, r1, r2; C cr; VC vr;
    Pair fac; VP vfac;
    void enter() override { ctx.restore(); }
    void run1(size_t v) override {
        switch (op) {
        case W_MUL: mul(r0, a[v], b[v]); break;
        case W_SQR: sqr(r0, a[v]); break;
        case W_DIVREM: DivRem(r0, r1, a[v], b[v]); break;
        case W_INV_SERIES: InvTrunc(r0, a[v], k); break;
        case W_MULMOD: MulMod(r0, a[v], b[v], c[v]); break;
        case W_POWMOD_X: PowerXMod(r0, e, F[v]); break;
        case W_POWMOD: PowerMod(r0, a[v], e, F[v]); break;
        case W_COMPOSE_MOD: CompMod(r0, a[v], b[v], F[v]); break;
        case W_POW: power(r0, a[v], k); break;
        case W_EVAL: eval(cr, a[v], s[v]); break;
        case W_DERIVATIVE: diff(r0, a[v]); break;
        case W_ADD: add(r0, a[v], b[v]); break;
        case W_SCALE: mul(r0, a[v], s[v]); break;
        case W_GCD: GCD(r0, a[v], b[v]); break;
        case W_XGCD: XGCD(r0, r1, r2, a[v], b[v]); break;
        case W_RESULTANT: resultant(cr, a[v], b[v]); break;
        case W_EVAL_MANY: eval(vr, a[v], x[v]); break;
        case W_FROM_ROOTS: BuildFromRoots(r0, x[v]); break;
        case W_INTERPOLATE: interpolate(r0, x[v], y[v]); break;
        case W_FACTOR: CanZass(fac, a[v]); break;
        case W_FACTOR_BERLEKAMP: berlekamp(fac, a[v]); break;
        case W_IS_IRREDUCIBLE: k = DetIrredTest(a[v]); break;
        case W_SQUAREFREE: SquareFreeDecomp(fac, a[v]); break;
        case W_DDF: { Mod Fm(a[v]); PowerXMod(r0, ZZ_mod(), Fm); DDF(fac, a[v], r0); break; }
        case W_EDF: { Mod Fm(a[v]); PowerXMod(r0, ZZ_mod(), Fm); EDF(vfac, a[v], r0, k); break; }
        }
    }
    ZZ ZZ_mod();
    void out1(size_t v, outbuf_t* o) override {
        switch (op) {
        case W_DIVREM: put_poly(o, r0); put_poly(o, r1); break;
        case W_XGCD: put_poly(o, r0); put_poly(o, r1); put_poly(o, r2); break;
        case W_EVAL: case W_RESULTANT: put(o, fp_c(cr)); break;
        case W_EVAL_MANY: {
            uint64_t hh = 0;
            for (long i = vr.length() - 1; i >= 0; i--) hh = fp_add(fp_mul(hh, FP_R), fp_c(vr[i]));
            put(o, (uint64_t)vr.length()); put(o, hh); break;
        }
        case W_FACTOR: case W_FACTOR_BERLEKAMP: case W_SQUAREFREE: {
            uint64_t sdeg = 0, mix = 0;
            for (long i = 0; i < fac.length(); i++) {
                sdeg += (uint64_t)(fac[i].b * deg(fac[i].a));
                mix = fp_add(mix, fp_mul((uint64_t)fac[i].b, fp_mix(fp_poly(fac[i].a))));
            }
            put(o, (uint64_t)fac.length()); put(o, sdeg); put(o, mix); put(o, fp_c(LeadCoeff(a[v])));
            break;
        }
        case W_DDF: {
            uint64_t mix = 0;
            for (long i = 0; i < fac.length(); i++)
                mix = fp_add(mix, fp_mix(fp_add(fp_poly(fac[i].a), (uint64_t)fac[i].b)));
            put(o, (uint64_t)fac.length()); put(o, mix); break;
        }
        case W_EDF: {
            uint64_t mix = 0;
            for (long i = 0; i < vfac.length(); i++) mix = fp_add(mix, fp_mix(fp_poly(vfac[i])));
            put(o, (uint64_t)vfac.length()); put(o, mix); break;
        }
        case W_IS_IRREDUCIBLE: put(o, (uint64_t)k); break;
        default: put_poly(o, r0); break;
        }
    }
};
template <> ZZ ModCase<zz_pX, zz_p, vec_zz_p, zz_pXModulus, vec_zz_pX, vec_pair_zz_pX_long, zz_pContext>::ZZ_mod() { return ZZ(zz_p::modulus()); }
template <> ZZ ModCase<ZZ_pX, ZZ_p, vec_ZZ_p, ZZ_pXModulus, vec_ZZ_pX, vec_pair_ZZ_pX_long, ZZ_pContext>::ZZ_mod() { return ZZ_p::modulus(); }

typedef ModCase<zz_pX, zz_p, vec_zz_p, zz_pXModulus, vec_zz_pX, vec_pair_zz_pX_long, zz_pContext> SmallCase;
typedef ModCase<ZZ_pX, ZZ_p, vec_ZZ_p, ZZ_pXModulus, vec_ZZ_pX, vec_pair_ZZ_pX_long, ZZ_pContext> BigCase;

struct Gf2Case : Base {
    std::vector<GF2X> a, b, c;
    std::vector<GF2XModulus> F;
    ZZ e; long k = 0;
    GF2X r0, r1, r2;
    vec_pair_GF2X_long fac; vec_GF2X vfac;
    void run1(size_t v) override {
        switch (op) {
        case W_MUL: mul(r0, a[v], b[v]); break;
        case W_SQR: sqr(r0, a[v]); break;
        case W_DIVREM: DivRem(r0, r1, a[v], b[v]); break;
        case W_INV_SERIES: InvTrunc(r0, a[v], k); break;
        case W_MULMOD: MulMod(r0, a[v], b[v], c[v]); break;
        case W_POWMOD_X: PowerXMod(r0, e, F[v]); break;
        case W_POWMOD: PowerMod(r0, a[v], e, F[v]); break;
        case W_GCD: GCD(r0, a[v], b[v]); break;
        case W_XGCD: XGCD(r0, r1, r2, a[v], b[v]); break;
        case W_FACTOR: CanZass(fac, a[v]); break;
        case W_IS_IRREDUCIBLE: k = IterIrredTest(a[v]); break;
        case W_SQUAREFREE: SquareFreeDecomp(fac, a[v]); break;
        case W_DDF: DDF(fac, a[v]); break;
        case W_EDF: EDF(vfac, a[v], k); break;
        case W_ADD: add(r0, a[v], b[v]); break;
        case W_DERIVATIVE: diff(r0, a[v]); break;
        }
    }
    void out1(size_t v, outbuf_t* o) override {
        (void)v;
        switch (op) {
        case W_DIVREM: put_poly(o, r0); put_poly(o, r1); break;
        case W_XGCD: put_poly(o, r0); put_poly(o, r1); put_poly(o, r2); break;
        case W_FACTOR: case W_SQUAREFREE: {
            uint64_t sdeg = 0, mix = 0;
            for (long i = 0; i < fac.length(); i++) {
                sdeg += (uint64_t)(fac[i].b * deg(fac[i].a));
                mix = fp_add(mix, fp_mul((uint64_t)fac[i].b, fp_mix(fp_poly(fac[i].a))));
            }
            put(o, (uint64_t)fac.length()); put(o, sdeg); put(o, mix); put(o, 1); break;
        }
        case W_DDF: {
            uint64_t mix = 0;
            for (long i = 0; i < fac.length(); i++)
                mix = fp_add(mix, fp_mix(fp_add(fp_poly(fac[i].a), (uint64_t)fac[i].b)));
            put(o, (uint64_t)fac.length()); put(o, mix); break;
        }
        case W_EDF: {
            uint64_t mix = 0;
            for (long i = 0; i < vfac.length(); i++) mix = fp_add(mix, fp_mix(fp_poly(vfac[i])));
            put(o, (uint64_t)vfac.length()); put(o, mix); break;
        }
        case W_IS_IRREDUCIBLE: put(o, (uint64_t)k); break;
        default: put_poly(o, r0); break;
        }
    }
};

struct ZCase : Base {
    std::vector<ZZX> a, b;
    ZZX r0, r1; ZZ zr; long k = 0;
    vec_pair_ZZX_long fac; ZZ c;
    void run1(size_t v) override {
        switch (op) {
        case Z_MUL: mul(r0, a[v], b[v]); break;
        case Z_SQR: sqr(r0, a[v]); break;
        case Z_PDIVREM: PseudoDivRem(r0, r1, a[v], b[v]); break;
        case Z_DIV_EXACT: k = divide(r0, a[v], b[v]); break;
        case Z_DERIVATIVE: diff(r0, a[v]); break;
        case Z_ADD: add(r0, a[v], b[v]); break;
        case Z_CONTENT: content(zr, a[v]); break;
        case Z_GCD: GCD(r0, a[v], b[v]); break;
        case Z_RESULTANT: resultant(zr, a[v], b[v]); break;
        case Z_DISCRIMINANT: discriminant(zr, a[v]); break;
        case Z_FACTOR: factor(c, fac, a[v]); break;
        case Z_SQUAREFREE: {
            // NTL wants a primitive polynomial of positive leading coefficient
            content(c, a[v]);
            if (sign(LeadCoeff(a[v])) < 0) NTL::negate(c, c);
            divide(r1, a[v], c);
            SquareFreeDecomp(fac, r1);
            break;
        }
        }
    }
    void out1(size_t v, outbuf_t* o) override {
        (void)v;
        switch (op) {
        case Z_PDIVREM: put_poly(o, r0); put_poly(o, r1); break;
        case Z_DIV_EXACT: if (k) { put(o, 1); put_poly(o, r0); } else put(o, 0); break;
        case Z_CONTENT: put(o, fp_zz(abs(zr))); break; // NTL's content takes the sign of the leading coefficient
        case Z_RESULTANT: case Z_DISCRIMINANT: put(o, fp_zz(zr)); break;
        case Z_FACTOR: case Z_SQUAREFREE: {
            uint64_t sdeg = 0, mix = 0;
            for (long i = 0; i < fac.length(); i++) {
                sdeg += (uint64_t)(fac[i].b * deg(fac[i].a));
                mix = fp_add(mix, fp_mul((uint64_t)fac[i].b, fp_mix(fp_poly(fac[i].a))));
            }
            put(o, (uint64_t)fac.length()); put(o, sdeg); put(o, mix); put(o, fp_zz(c)); break;
        }
        default: put_poly(o, r0); break;
        }
    }
};

template <class P> static void poly_from(P& p, const arg_t* a);
template <> void poly_from<zz_pX>(zz_pX& p, const arg_t* a) {
    p.SetLength((long)a->len);
    for (size_t i = 0; i < a->len; i++) p[(long)i] = zz_p((long)(a->p[i] % (uint64_t)zz_p::modulus()));
    p.normalize();
}
template <> void poly_from<ZZ_pX>(ZZ_pX& p, const arg_t* a) {
    p.SetLength((long)a->len);
    if (a->kind == K_WVEC) {
        for (size_t i = 0; i < a->len; i++) { ZZ t; conv(t, (unsigned long)a->p[i]); conv(p[(long)i], t); }
    } else {
        const uint64_t* q = a->p;
        for (size_t i = 0; i < a->len; i++) { ZZ t; set_zz(t, q); q += enc_int_len(q); conv(p[(long)i], t); }
    }
    p.normalize();
}
static void gf2_from(GF2X& p, const arg_t* a) {
    clear(p);
    for (size_t i = 0; i < a->len; i++) if (a->p[i] & 1) SetCoeff(p, (long)i);
}
static void zzx_from(ZZX& p, const arg_t* a) {
    p.SetLength((long)a->len);
    const uint64_t* q = a->p;
    for (size_t i = 0; i < a->len; i++) { set_zz(p[(long)i], q); q += enc_int_len(q); }
    p.normalize();
}
template <class VC> static void vec_from(VC& x, const arg_t* a);
template <> void vec_from<vec_zz_p>(vec_zz_p& x, const arg_t* a) {
    x.SetLength((long)a->len);
    for (size_t i = 0; i < a->len; i++) x[(long)i] = zz_p((long)(a->p[i] % (uint64_t)zz_p::modulus()));
}
template <> void vec_from<vec_ZZ_p>(vec_ZZ_p& x, const arg_t* a) {
    x.SetLength((long)a->len);
    if (a->kind == K_WVEC) {
        for (size_t i = 0; i < a->len; i++) { ZZ t; conv(t, (unsigned long)a->p[i]); conv(x[(long)i], t); }
    } else {
        const uint64_t* q = a->p;
        for (size_t i = 0; i < a->len; i++) { ZZ t; set_zz(t, q); q += enc_int_len(q); conv(x[(long)i], t); }
    }
}
static void scal_from(zz_p& c, const arg_t* a) { c = zz_p((long)(a->u % (uint64_t)zz_p::modulus())); }
static void scal_from(ZZ_p& c, const arg_t* a) { ZZ t; set_zz_arg(t, a); conv(c, t); }

static int npoly_of(int op, size_t A) {
    switch (op) {
    case W_EVAL_MANY: return 1;
    case W_FROM_ROOTS: case W_INTERPOLATE: return 0;
    case W_EVAL: case W_SCALE: return 1;
    default: return (int)A;
    }
}

template <class Case, class P, class VC>
static Base* build_mod(int op, input_t& in, size_t si) {
    Case* h = new Case();
    h->ctx.save();
    h->op = op; h->V = in.v;
    switch (op) {
    case W_INV_SERIES: case W_POW: case W_EDF: h->k = (long)in.shared[si++].u; break;
    case W_POWMOD_X: case W_POWMOD: set_zz_arg(h->e, &in.shared[si++]); break;
    default: break;
    }
    int np = npoly_of(op, in.a);
    for (size_t v = 0; v < in.v; v++) {
        P p[3];
        for (int j = 0; j < np; j++) poly_from<P>(p[j], VA(&in, v, (size_t)j));
        if (np > 0) h->a.push_back(p[0]);
        if (np > 1) h->b.push_back(p[1]);
        if (np > 2) h->c.push_back(p[2]);
        if (op == W_EVAL || op == W_SCALE) { typename std::remove_reference<decltype(h->s[0])>::type c; scal_from(c, VA(&in, v, 1)); h->s.push_back(c); }
        if (op == W_EVAL_MANY) { VC x; vec_from<VC>(x, VA(&in, v, 1)); h->x.push_back(x); }
        if (op == W_FROM_ROOTS) { VC x; vec_from<VC>(x, VA(&in, v, 0)); h->x.push_back(x); }
        if (op == W_INTERPOLATE) { VC x, y; vec_from<VC>(x, VA(&in, v, 0)); vec_from<VC>(y, VA(&in, v, 1)); h->x.push_back(x); h->y.push_back(y); }
    }
    if (op == W_POWMOD_X) for (size_t v = 0; v < in.v; v++) h->F.emplace_back(h->a[v]);
    if (op == W_POWMOD) for (size_t v = 0; v < in.v; v++) h->F.emplace_back(h->b[v]);
    if (op == W_COMPOSE_MOD) for (size_t v = 0; v < in.v; v++) h->F.emplace_back(h->c[v]);
    // a ZZ_pX powmod_x for the m_ family takes the modulus as the exponent
    return h;
}

extern "C" void* sn_prep(const char* name, const uint64_t* buf, size_t nbuf) {
    SetNumThreads(1); // one thread (NTL's default, made explicit)
    char fam;
    int op = lookup(name, &fam);
    if (!op) return nullptr;
    input_t in;
    parse_input(buf, nbuf, &in);
    Base* h = nullptr;
    size_t si = 0;
    if (fam == 'z') {
        ZCase* z = new ZCase();
        z->op = op; z->V = in.v;
        for (size_t v = 0; v < in.v; v++) {
            ZZX p[2];
            for (size_t j = 0; j < in.a && j < 2; j++) zzx_from(p[j], VA(&in, v, j));
            z->a.push_back(p[0]);
            if (in.a > 1) z->b.push_back(p[1]);
        }
        z->label = "ZZX";
        h = z;
    } else {
        ZZ N;
        set_zz_arg(N, &in.shared[si++]);
        bool unsupported = false;
        if (N == 2) {
            switch (op) {
            case W_MUL: case W_SQR: case W_DIVREM: case W_INV_SERIES: case W_MULMOD: case W_POWMOD_X:
            case W_POWMOD: case W_GCD: case W_XGCD: case W_FACTOR: case W_IS_IRREDUCIBLE:
            case W_SQUAREFREE: case W_DDF: case W_EDF: case W_ADD: case W_DERIVATIVE: break;
            default: unsupported = true;
            }
            if (!unsupported) {
                Gf2Case* g = new Gf2Case();
                g->op = op; g->V = in.v;
                switch (op) {
                case W_INV_SERIES: case W_EDF: g->k = (long)in.shared[si++].u; break;
                case W_POWMOD_X: case W_POWMOD: set_zz_arg(g->e, &in.shared[si++]); break;
                default: break;
                }
                for (size_t v = 0; v < in.v; v++) {
                    GF2X p[3];
                    for (size_t j = 0; j < in.a && j < 3; j++) gf2_from(p[j], VA(&in, v, j));
                    g->a.push_back(p[0]);
                    if (in.a > 1) g->b.push_back(p[1]);
                    if (in.a > 2) g->c.push_back(p[2]);
                }
                if (op == W_POWMOD_X) for (size_t v = 0; v < in.v; v++) g->F.emplace_back(g->a[v]);
                if (op == W_POWMOD) for (size_t v = 0; v < in.v; v++) g->F.emplace_back(g->b[v]);
                g->label = "GF2X";
                h = g;
            }
        } else if (N < NTL_SP_BOUND) {
            if (op == W_FACTOR_BERLEKAMP || op == W_FACTOR || op == W_SQUAREFREE || op == W_DDF || op == W_EDF || op == W_IS_IRREDUCIBLE || true) {
                zz_p::init(conv<long>(N));
                h = build_mod<SmallCase, zz_pX, vec_zz_p>(op, in, si);
                h->label = "zz_pX";
            }
        } else {
            if (op == W_FACTOR_BERLEKAMP) unsupported = true; // the page's NTL berlekamp is zz_pX only
            if (!unsupported) {
                ZZ_p::init(N);
                if (fam == 'm' && op == W_POWMOD_X) {
                    // x^n mod f with e = n
                    BigCase* b = (BigCase*)build_mod<BigCase, ZZ_pX, vec_ZZ_p>(W_MULMOD + 1000, in, si);
                    b->op = W_POWMOD_X; b->e = N;
                    for (size_t v = 0; v < in.v; v++) b->F.emplace_back(b->a[v]);
                    h = b;
                } else {
                    h = build_mod<BigCase, ZZ_pX, vec_ZZ_p>(op, in, si);
                }
                h->label = "ZZ_pX";
            }
        }
    }
    free(in.var);
    return h;
}

extern "C" void sn_run(void* hv, uint64_t reps) {
    Base* h = (Base*)hv;
    h->enter();
    size_t V = h->V;
    for (uint64_t r = 0; r < reps; r++) h->run1((size_t)(r % V));
}
extern "C" size_t sn_check(void* hv, uint64_t* out, size_t cap) {
    Base* h = (Base*)hv;
    h->enter();
    outbuf_t o = {out, cap, 0};
    for (size_t v = 0; v < h->V; v++) { h->run1(v); h->out1(v, &o); }
    return o.n;
}
extern "C" void sn_free(void* hv) { delete (Base*)hv; }
extern "C" const char* sn_label(void* hv) { return ((Base*)hv)->label.c_str(); }
