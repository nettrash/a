"""The error bounds of the erf area of a::math::special (docs/math/special.md, "Error bounds"):
a first-order running error analysis of the paths that end in a short chain of roundings, each
operation as erf/mod.rs and erf/complex.rs do it, every rounding taken at its largest at once.

Run from this directory: `python3 erf_bounds.py` (numpy and mpmath; some minutes). It prints the
largest bound of each region, maximised over random arguments and a local search around the
largest: the bound is a smooth function of the argument (but at the seams of the pieces), so this
finds its maximum, unlike a search for the largest error, which depends on how the roundings
happen to fall. For the real functions a dense grid of the interval does; for the complex ones the
search can miss a little (runs of other budgets and seeds differ by 1 % in the boxes and by up to
7 % beyond them, where the bound has many local maxima), so the docs state the largest of several
runs with 3 % more, and a run of this script prints below them (by up to 10 % for erfc beyond
the box).

Each value carries a bound on its absolute rounding error: an operation's result v gets u|v|
(u = 2^-53) plus its inputs' errors times the partial derivatives, a fused multiply-add one
rounding, a rounded constant u|c|. What an algorithm leaves out (the aliasing of the trapezoidal
rule, 0.3 eps of |w| measured in exact arithmetic; the series truncations; exp_dd's 2^-60) and
the polynomials' own errors with their coefficients as doubles (measured here with mpmath) are
added. The values are float64 without fused operations: they serve only for the magnitudes. The
real bounds are in ulps of the result, its last rounding counted as half an ulp; the complex ones
in eps = 2^-52 of the scale of the docs (the larger of the two terms a formula subtracts, or |w|).
The Fresnel integrals from 1 (added after the review of 2026-11-22) count every rounding as half
an ulp of its result, the tighter form of the same bound: their errors come from a few roundings
of values near 1/sqrt(2) and 1/2, at the top of their binades, where u|v| would count twice as much.
"""
import os
import re

import mpmath as mp
import numpy as np


# ---- complex
U = 2.0 ** -53
EPS = 2.0 ** -52

# What the algorithms leave out, relative to |w| (or to the term), on top of the roundings:
TRAP_ALIAS = 0.35 * EPS  # the aliasing e^(-pi^2/h^2) of the shifted rule: 0.3 eps of |w| measured in exact arithmetic
TRAP_TAYLOR = 2.0 ** -60  # e^(-t0^2), cosh, sinh to their Taylor degree
ASYM_TRUNC = 2.0 ** -60  # the first omitted term at |z| = 10
EXP_DD = 2.0 ** -58  # exp_dd's double-double m, before its rounding to a double
SINCOS = 1.2 * U  # fdlibm's kernels with the exact reduction: below 0.6 ulp, relative


class V:
    """A value array with an absolute error bound array."""

    __slots__ = ("v", "e")

    def __init__(self, v, e=None):
        self.v = np.asarray(v, np.float64)
        self.e = np.zeros_like(self.v) if e is None else np.asarray(e, np.float64)


def const(c, shape, exact=False):
    v = np.full(shape, c, np.float64)
    return V(v, np.zeros(shape) if exact else U * np.abs(v))


def rnd(v, e):
    return V(v, e + U * np.abs(v))


def add(a, b):
    return rnd(a.v + b.v, a.e + b.e)


def sub(a, b):
    return rnd(a.v - b.v, a.e + b.e)


def mul(a, b):
    return rnd(a.v * b.v, np.abs(a.v) * b.e + np.abs(b.v) * a.e)


def div(a, b):
    q = a.v / b.v
    return rnd(q, a.e / np.abs(b.v) + np.abs(q) * b.e / np.abs(b.v))


def fma(a, b, c):
    """a·b + c, one rounding."""
    return rnd(a.v * b.v + c.v, np.abs(a.v) * b.e + np.abs(b.v) * a.e + c.e)


def neg(a):
    return V(-a.v, a.e)


def exact_mul(a, k):
    """a·k for a power of 2 k (exact)."""
    return V(a.v * k, a.e * abs(k))


def horner(x, coefs, exact_coefs=False):
    """poly(x, c) as the crate's poly: s = c[n-1]; s = fma(s, x, c[i])."""
    sh = x.v.shape
    s = const(coefs[-1], sh, exact_coefs)
    for c in reversed(coefs[:-1]):
        s = fma(s, x, const(c, sh, exact_coefs))
    return s


def cmul(a, b):
    """cmul of complex.rs: (fma(ar, br, -(ai*bi)), fma(ar, bi, ai*br))."""
    ar, ai = a
    br, bi = b
    return (fma(ar, br, neg(mul(ai, bi))), fma(ar, bi, mul(ai, br)))


# ---- the trapezoidal rule, x > 0, y > 0, |z| < 10 ----

GAUSS = [0.159_154_943_091_895_35, 0.123_949_994_309_652_96, 0.058_549_831_524_319_16, 0.016_774_807_587_073_417,
         0.002_915_024_465_028_193_5, 0.000_307_241_318_192_835, 1.964_128_034_639_743_7e-5, 7.615_750_862_323_311e-7,
         1.791_052_932_828_018_5e-8, 2.554_799_797_725_798e-10, 2.210_334_915_491_785_8e-12, 1.159_877_313_739_617_6e-14,
         3.691_635_240_477_673e-17, 7.126_532_519_424_097e-20]
E0C = [1.0, -1.0, 0.5, -1.0 / 6.0, 1.0 / 24.0, -1.0 / 120.0, 1.0 / 720.0, -1.0 / 5040.0, 1.0 / 40320.0, -1.0 / 362_880.0]
CHC = [1.0, 0.5, 1.0 / 24.0, 1.0 / 720.0, 1.0 / 40320.0, 1.0 / 3_628_800.0, 1.0 / 479_001_600.0]
SHC = [1.0, 1.0 / 6.0, 1.0 / 120.0, 1.0 / 5040.0, 1.0 / 362_880.0, 1.0 / 39_916_800.0, 1.0 / 6_227_020_800.0]


def w_trapezoid(x, y):
    """(re, im) as V for x > 0, y > 0, |z| < 10, without the pole correction's own errors
    (added by pole_correction below where it applies)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    sh_ = x.shape
    n0 = np.round(-2.0 * x - 0.5)  # Rust's round: half away from zero (np.round is half-even; the
    n0 = np.where(np.abs(-2.0 * x - 0.5 - n0) == 0.5, np.sign(-2.0 * x - 0.5) * np.ceil(np.abs(-2.0 * x - 0.5)), n0)
    s0 = (n0 + 0.5) * 0.5
    t0v = x + s0
    # t0 = x + s0 is exact where Sterbenz's lemma applies (x >= 1/8); else rounded
    t0 = V(t0v, np.where(x >= 0.125, 0.0, U * np.abs(t0v)))
    u_ = mul(t0, t0)
    e0 = horner(u_, E0C)
    ch = horner(u_, CHC)
    sh = mul(t0, horner(u_, SHC))
    p = sub(ch, sh)
    ip = add(ch, sh)
    S0 = V(s0)
    yv = V(y)
    y2 = mul(yv, yv)
    re = div(mul(e0, const(GAUSS[0], sh_)), fma(S0, S0, y2))
    im = mul(re, S0)
    pk, mk = e0, e0
    for k in range(1, 14):
        pk = mul(pk, p)
        mk = mul(mk, ip)
        g = const(GAUSS[k], sh_)
        sp = V(s0 + 0.5 * k)
        sm = V(s0 - 0.5 * k)
        dp = fma(sp, sp, y2)
        dm = fma(sm, sm, y2)
        ep = mul(pk, g)
        em = mul(mk, g)
        a = mul(ep, dm)
        b = mul(em, dp)
        r = div(const(1.0, sh_, True), mul(dp, dm))
        re = fma(add(a, b), r, re)
        im = fma(fma(sp, a, mul(sm, b)), r, im)
    wre = mul(yv, re)
    wim = neg(im)
    # the left-out parts, each relative to its own part (the rule and the series are real on the
    # real axis, so the part of the size of y left out of Re w is of the size of y too)
    extra = TRAP_ALIAS + 13 * TRAP_TAYLOR
    return V(wre.v, wre.e + extra * np.abs(wre.v)), V(wim.v, wim.e + extra * np.abs(wim.v))


def pole_applies(x, y):
    ex = y * (y - 4 * np.pi) - x * x
    with np.errstate(divide="ignore"):
        lnf = np.floor(np.log2(np.where(y > 0, y, 1.0))) * np.log(2.0)
    return (y < 2 * np.pi) & ((ex > -45.0) | ((y < 1.0) & (ex > lnf - 46.0)))


def pole_correction(x, y, wre, wim):
    """Adds the pole correction 2e^(-z^2)/(1 + e^(4 pi y))·(cos 2xy, -sin 2xy) where it applies:
    its value, and its error (exp_dd twice, sin_cos, the division and the products: 6u of g)."""
    ap = pole_applies(x, y)
    z2r = y * y - x * x - 4 * np.pi * y
    with np.errstate(over="ignore", under="ignore"):
        g = 2 * np.exp(z2r) / (1 + np.exp(-4 * np.pi * y))
    g = np.where(ap, g, 0.0)
    co, s = np.cos(2 * x * y), np.sin(2 * x * y)
    ge = g * (6 * U + 2 * EXP_DD + SINCOS)
    re = V(wre.v + g * co, wre.e + ge * np.abs(co) + U * np.abs(wre.v + g * co))
    im = V(wim.v - g * s, wim.e + ge * np.abs(s) + U * np.abs(wim.v - g * s))
    return re, im


# ---- the asymptotic series, |z| >= 10, x >= 0, y >= 0, max < 2^1020 ----

ASYM = [1.0, 0.5, 0.75, 1.875, 6.5625, 29.53125, 162.421875, 1055.7421875, 7918.06640625, 67303.564453125,
        639383.8623046875, 6713530.554199219, 77205601.37329102, 965070017.1661377]
FRAC_1_SQRT_PI = 0.5641895835477563


def recip(x, y):
    """Smith's 1/z as complex.rs: (re, im) as V."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    big = np.abs(x) >= np.abs(y)
    a = np.where(big, x, y)
    b = np.where(big, y, x)
    A, B = V(a), V(b)
    r = div(B, A)
    d = div(const(1.0, a.shape, True), fma(B, r, A))
    rd = mul(r, d)
    # big: (d, -r d); else (r d, -d)
    re = V(np.where(big, d.v, rd.v), np.where(big, d.e, rd.e))
    im = V(np.where(big, -rd.v, -d.v), np.where(big, rd.e, d.e))
    return re, im


def w_asymptotic(x, y):
    q = recip(x, y)
    v = cmul(q, q)
    sh_ = np.shape(x)
    s = (V(np.full(sh_, ASYM[13])), V(np.zeros(sh_)))
    for i in range(12, -1, -1):
        s = cmul(s, v)
        s = (add(s[0], V(np.full(sh_, ASYM[i]))), s[1])  # ASYM are exact (dyadic)
    p = cmul(q, s)
    c = const(FRAC_1_SQRT_PI, sh_)
    wre = neg(mul(c, p[1]))
    wim = mul(c, p[0])
    wre = V(wre.v, wre.e + ASYM_TRUNC * np.abs(wre.v))
    wim = V(wim.v, wim.e + ASYM_TRUNC * np.abs(wim.v))
    # Within 1e-20 of the real axis, below x = 27.3, e^(-x^2) (to half an ulp and 2^-60) is added
    # to the real part, rounded.
    near = (np.asarray(y) < 1e-20) & (np.asarray(x) < 27.3)
    if np.any(near):
        with np.errstate(under="ignore"):
            ex = np.where(near, np.exp(-np.asarray(x, np.float64) ** 2), 0.0)
        v = wre.v + ex
        wre = V(v, wre.e + ex * (U + 2.0 ** -60) + np.where(near, U * np.abs(v), 0.0))
    return wre, wim


def w_quadrant(x, y):
    """x > 0, y > 0, finite, below 2^1020 (the main domain of the upper half-plane)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    asy = (x >= 10) | (y >= 10) | (x * x + y * y >= 100)
    out_re = V(np.zeros_like(x), np.zeros_like(x))
    out_im = V(np.zeros_like(x), np.zeros_like(x))
    if (~asy).any():
        xs, ys = x[~asy], y[~asy]
        r, i = w_trapezoid(xs, ys)
        r, i = pole_correction(xs, ys, r, i)
        out_re.v[~asy], out_re.e[~asy] = r.v, r.e
        out_im.v[~asy], out_im.e[~asy] = i.v, i.e
    if asy.any():
        r, i = w_asymptotic(x[asy], y[asy])
        out_re.v[asy], out_re.e[asy] = r.v, r.e
        out_im.v[asy], out_im.e[asy] = i.v, i.e
    return out_re, out_im


def w_upper(x, y):
    """y > 0, x != 0: by w(-x + iy) = conj w(x + iy)."""
    x = np.asarray(x, np.float64)
    r, i = w_quadrant(np.abs(x), y)
    return r, V(np.where(x < 0, -i.v, i.v), i.e)


# ---- e^(-z^2) and the compositions ----


def exp_neg_sq(x, y):
    """e^(-z^2) = (m co, m s) as V (no power of 2 split: the magnitudes are what matter); y^2 - x^2
    as (y - x)(y + x), which keeps its size next to the diagonals at a large |z|."""
    a = (y - x) * (y + x)
    with np.errstate(over="ignore", under="ignore"):
        m = np.exp(a)
    th = -2 * x * y
    co, s = np.cos(th), np.sin(th)
    # m.value(): EXP_DD + one rounding; the kernels SINCOS (relative) each; m*co, m*s: one rounding
    re = V(m * co, np.abs(m * co) * (EXP_DD + 2 * U + SINCOS))
    im = V(m * s, np.abs(m * s) * (EXP_DD + 2 * U + SINCOS))
    return re, im


def faddeeva_lower(x, y):
    """w(z) = 2e^(-z^2) - w(-z), y < 0, x != 0 (general path)."""
    er, ei = exp_neg_sq(x, y)
    wr, wi = w_upper(-x, -y)
    re = sub(exact_mul(er, 2.0), wr)
    im = sub(exact_mul(ei, 2.0), wi)
    return re, im


def erfc_right(x, y):
    """e^(-z^2)·w(iz), x > 0 (general path)."""
    wr, wi = w_upper(-np.asarray(y, np.float64), x)
    er, ei = exp_neg_sq(x, y)
    return cmul((er, ei), (wr, wi))


TAYLOR = [1.1283791670955126, -0.37612638903183754, 0.11283791670955126, -0.026866170645131252, 0.005223977625442188,
          -0.0008548327023450853, 0.00012055332981789664, -1.492565035840625e-05, 1.6462114365889248e-06,
          -1.6365844691234924e-07, 1.4807192815879218e-08, -1.2290555301717928e-09, 9.422759064650411e-11,
          -6.7113668551641105e-12, 4.4632242632864775e-13, -2.7835162072109215e-14, 1.6342614095367152e-15,
          -9.063970842808673e-17, 4.763348040515068e-18]
TAYLOR_TRUNC = 2.0 ** -60


def erf_small(x, y):
    """|z| < 1: z·S(z^2) by Horner in complex arithmetic (complex.rs)."""
    X, Y = V(x), V(y)
    ur = mul(sub(X, Y), add(X, Y))
    ui = mul(exact_mul(X, 2.0), Y)
    sh_ = np.shape(x)
    s = (const(TAYLOR[18], sh_), V(np.zeros(sh_)))
    for i in range(17, -1, -1):
        s = cmul(s, (ur, ui))
        s = (add(s[0], const(TAYLOR[i], sh_)), s[1])
    re, im = cmul((X, Y), s)
    return V(re.v, re.e + TAYLOR_TRUNC * np.abs(re.v)), V(im.v, im.e + TAYLOR_TRUNC * np.abs(im.v))


def bound(fn, x, y):
    """B(z): the bound on |computed - exact| (a complex modulus) for the general paths, and the
    value's modulus; NaN where the point is outside what this module covers."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if fn == "faddeeva":
        upper = y > 0
        re = V(np.zeros_like(x), np.zeros_like(x))
        im = V(np.zeros_like(x), np.zeros_like(x))
        if upper.any():
            r, i = w_upper(x[upper], y[upper])
            re.v[upper], re.e[upper], im.v[upper], im.e[upper] = r.v, r.e, i.v, i.e
        lo = ~upper
        if lo.any():
            r, i = faddeeva_lower(x[lo], y[lo])
            re.v[lo], re.e[lo], im.v[lo], im.e[lo] = r.v, r.e, i.v, i.e
    elif fn == "erfcx":
        return bound("faddeeva", -y, x)
    elif fn in ("erfc", "erf"):
        right = x > 0
        xr = np.where(right, x, -x)
        yr = np.where(right, y, -y)
        er, ei = erfc_right(xr, yr)
        if fn == "erfc":
            # x < 0: 2 - erfc(-z)
            re = V(np.where(right, er.v, 2 - er.v), np.where(right, er.e, er.e + U * np.abs(2 - er.v)))
            im = V(np.where(right, ei.v, -ei.v), ei.e)
        else:
            # |z| >= 1: +-(1 - erfc(+-z)); |z| < 1: the Taylor series
            r1 = V(1 - er.v, er.e + U * np.abs(1 - er.v))
            re = V(np.where(right, r1.v, -r1.v), r1.e)
            im = V(np.where(right, -ei.v, ei.v), ei.e)
            small = x * x + y * y < 1
            if small.any():
                sr, si = erf_small(x[small], y[small])
                re.v[small], re.e[small], im.v[small], im.e[small] = sr.v, sr.e, si.v, si.e
    return np.hypot(re.e, im.e), np.hypot(re.v, im.v)


# ---- real
HERE = os.path.dirname(os.path.abspath(__file__))
TXT = open(os.path.join(HERE, "..", "..", "..", "..", "src", "math", "special", "erf", "tables.rs")).read()


def table(name):
    m = re.search(r"static " + name + r": [^=]*= \[(.*?)\n\];", TXT, re.S)
    body = m.group(1)
    rows = re.findall(r"\[([^\[\]]*)\]", body)
    num = lambda s: [float(v.replace("_", "")) for v in re.findall(r"-?[0-9][0-9_.]*(?:e-?[0-9]+)?", s)]
    if rows:
        return [num(r) for r in rows]
    return num(body)


ERF_SMALL = table("ERF_SMALL")
DAWSON_SMALL = table("DAWSON_SMALL")
FRESNEL_C_SMALL = table("FRESNEL_C_SMALL")
FRESNEL_S_SMALL = table("FRESNEL_S_SMALL")
ERFCX_S = table("ERFCX_S")
assert len(ERF_SMALL) == 10 and len(DAWSON_SMALL) == 9 and len(FRESNEL_C_SMALL) == 9 and len(FRESNEL_S_SMALL) == 10
assert len(ERFCX_S) == 8 and all(len(r) == 13 for r in ERFCX_S)


def rhorner(x, ex, c):
    """Value and error bound of the crate's Horner (fma) at x with |error of x| <= ex."""
    s = np.full_like(x, c[-1])
    es = np.abs(s) * U
    ds = np.zeros_like(x)  # derivative, for the error of x
    for ci in reversed(c[:-1]):
        ds = ds * x + s
        v = s * x + ci
        es = es * np.abs(x) + U * np.abs(v)
        s = v
    return s, es + np.abs(ds) * ex


def ulp(v):
    v = np.abs(v)
    e = np.floor(np.log2(np.maximum(v, 2.0 ** -1022)))
    return 2.0 ** (np.maximum(e, -1022) - 52)


def erf_real_small(x):
    t = x * x
    q, eq = rhorner(t, U * t, ERF_SMALL)
    r = x + x * q
    fit = FIT['erf'] * np.abs(r)
    return (np.abs(x) * eq + fit) / ulp(r) + 0.5


def erfc_real_small(x):
    t = x * x
    q, eq = rhorner(t, U * t, ERF_SMALL)
    f = (1.0 - x) - (1 - x) + (-x * q)  # e = 0 for the magnitudes; f = -x q + e
    r = 1.0 - x - x * q
    fit = FIT['erf'] * np.abs(x + x * q)
    return (np.abs(x) * eq + U * np.abs(f) + fit) / ulp(r) + 0.5


def dawson_small(x):
    t = x * x
    p, ep = rhorner(t, U * t, DAWSON_SMALL)
    xt = x * t
    ext = np.abs(x) * U * t + U * np.abs(xt)
    r = x + xt * p
    fit = FIT['dawson'] * np.abs(r)
    return (np.abs(xt) * ep + np.abs(p) * ext + fit) / ulp(r) + 0.5


def erfcx_small(x):
    i = np.minimum(((x + 0.5) * 8.0).astype(int), 7)
    out = np.zeros_like(x)
    for j in range(8):
        m = i == j
        if not m.any():
            continue
        row = ERFCX_S[j]
        r = x[m] - (-0.5 + 0.0625 + 0.125 * j)
        p, ep = rhorner(r, 0.0 * r, row[2:])
        q = r * p
        eq = np.abs(r) * ep + U * np.abs(q)
        v = row[0] + row[1] + q
        lo = (row[0] + q) - row[0] - q + row[1]
        fit = FIT['erfcx_s'] * np.abs(v)
        out[m] = (eq + U * np.abs(lo) + fit) / ulp(v) + 0.5
    return out


def fresnel_small(x):
    a = np.abs(x)
    x2 = a * a
    w = x2 * x2
    ew = U * w * (1 + 4 * U)  # w = fma(x2, x2, 2 x2 x2l): x^4 rounded once
    # C = fma(a w, Pc(w), a)
    pc, epc = rhorner(w, ew, FRESNEL_C_SMALL)
    aw = a * w
    eaw = a * ew + U * aw
    c = a + aw * pc
    fitc = FIT['fresnel_c'] * np.abs(c)
    bc = (aw * epc + np.abs(pc) * eaw + fitc) / ulp(c) + 0.5
    # S = x^3 (double-double) times head(S_SMALL, w), rounded once
    row = FRESNEL_S_SMALL
    pt, ept = rhorner(w, ew, row[2:])
    q = w * pt
    eq = w * ept + np.abs(pt) * ew + U * np.abs(q)
    Q = row[0] + row[1] + q
    x3 = x2 * a
    s = x3 * Q
    fits = FIT['fresnel_s'] * np.abs(s)
    bs = (x3 * eq + 2.0 ** -100 * np.abs(s) + fits) / ulp(s) + 0.5
    return bs, bc


ERFCX_T = table("ERFCX_T")
DAWSON_T = table("DAWSON_T")


def head_table(x, T, fit):
    """head(&T[i], r).value() with the piece of eighths of binades from 1/2 (piece(x, B_HALF))."""
    b = x.view(np.uint64) >> np.uint64(49)
    c = ((b << np.uint64(49)) | np.uint64(1 << 48)).view(np.float64)
    i = (b - np.uint64(0x3FE0_0000_0000_0000 >> 49)).astype(np.int64)
    r = x - c
    out = np.zeros_like(x)
    for j in range(len(T)):
        m = i == j
        if not m.any():
            continue
        row = T[j]
        p, ep = rhorner(r[m], 0.0 * r[m], row[2:])
        q = r[m] * p
        eq = np.abs(r[m]) * ep + U * np.abs(q)
        v = row[0] + row[1] + q
        out[m] = (eq + U * 2.0 ** -55 * np.abs(v) + fit * np.abs(v)) / ulp(v) + 0.5
    return out




# ---- the Fresnel integrals from 1: C = 1/2 + f sin - g cos, S = 1/2 - f cos - g sin

FRESNEL_F_T = table("FRESNEL_F_T")
FRESNEL_G_T = table("FRESNEL_G_T")
SINQ = table("SINQ")
COSQ = table("COSQ")
F_ASYM_1 = (lambda c: [c[0] + c[1]] + c[2:])(table("FRESNEL_F_ASYM"))  # the fold of erf/mod.rs
G_ASYM_1 = (lambda c: [c[0] + c[1]] + c[2:])(table("FRESNEL_G_ASYM"))
assert len(FRESNEL_F_T) == 24 and len(FRESNEL_G_T) == 24 and len(SINQ) == 9 and len(COSQ) == 7


def hu(v):
    """Half an ulp of v: the largest error of rounding to nearest with v the result."""
    return 0.5 * np.spacing(np.abs(v))


def two_prod_np(a, b):
    """a*b = p + e exactly (Dekker's product)."""
    p = a * b
    t = 134217729.0 * a
    ah = t - (t - a)
    al = a - ah
    t = 134217729.0 * b
    bh = t - (t - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def rust_round(v):
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


def hhorner(x, ex, c):
    """rhorner with every rounding as half an ulp."""
    s = np.full_like(x, c[-1])
    es = np.zeros_like(x)
    ds = np.zeros_like(x)
    for ci in reversed(c[:-1]):
        ds = ds * x + s
        v = s * x + ci
        es = es * np.abs(x) + hu(v)
        s = v
    return s, es + np.abs(ds) * ex


def fresnel_sin_cos(x):
    """sin_cos_half_pi_sq of erf/mod.rs: x^2 exact as h + l, reduced modulo 4 exactly but for the
    one rounding of r = (h4 - n1) + (l4 - n2); sin(pi r/2) = r * head(SINQ, r^2), cos by one fma.
    The values (sin pi x^2/2, cos pi x^2/2) and their error bounds."""
    h, l = two_prod_np(x, x)
    h4 = h - 4.0 * rust_round(h * 0.25)
    l4 = l - 4.0 * rust_round(l * 0.25)
    n1, n2 = rust_round(h4), rust_round(l4)
    r = (h4 - n1) + (l4 - n2)
    er = hu(r)
    n3 = rust_round(r)
    r = r - n3
    q = (n1 + n2 + n3).astype(np.int64) & 3
    s2 = r * r
    es2 = 2 * np.abs(r) * er + hu(s2)
    p, ep = hhorner(s2, es2, SINQ[2:])
    qq = s2 * p
    eqq = s2 * ep + np.abs(p) * es2 + hu(qq)
    hi = SINQ[0] + qq
    lo = ((SINQ[0] - hi) + qq) + SINQ[1]
    hv = hi + lo
    ehv = eqq + hu(lo) + hu(hv)
    sn = r * hv
    esn = np.abs(hv) * er + np.abs(r) * ehv + hu(sn) + FIT["sinq"] * np.abs(sn)
    pc, epc = hhorner(s2, es2, COSQ)
    cs = s2 * pc + 1.0
    ecs = s2 * epc + np.abs(pc) * es2 + hu(cs) + FIT["cosq"] * np.abs(s2 * pc)
    swap = (q & 1) == 1
    sv, cv = np.where(swap, cs, sn), np.where(swap, sn, cs)
    se, ce = np.where(swap, ecs, esn), np.where(swap, esn, ecs)
    return np.where(q >= 2, -sv, sv), se, np.where((q == 1) | (q == 2), -cv, cv), ce


def fresnel_comb(f, ef, g, eg, x):
    """comb of erf/mod.rs for C and S: the products exact, the low words and their sums below
    2^-100 of the result; then the one rounding. Bounds in ulps of S and C."""
    sn, esn, cs, ecs = fresnel_sin_cos(x)
    c = 0.5 + f * sn - g * cs
    s = 0.5 - f * cs - g * sn
    ec = np.abs(sn) * ef + np.abs(f) * esn + np.abs(cs) * eg + np.abs(g) * ecs + 2.0 ** -100 * np.abs(c)
    es = np.abs(cs) * ef + np.abs(f) * ecs + np.abs(sn) * eg + np.abs(g) * esn + 2.0 ** -100 * np.abs(s)
    return es / ulp(s) + 0.5, ec / ulp(c) + 0.5


def fresnel_head(x, T, fit):
    """head(&T[i], r) with the pieces of eighths of binades from 1: the value and its error bound."""
    b = x.view(np.uint64) >> np.uint64(49)
    c = ((b << np.uint64(49)) | np.uint64(1 << 48)).view(np.float64)
    i = (b - np.uint64(0x3FF0_0000_0000_0000 >> 49)).astype(np.int64)
    r = x - c
    val, err = np.zeros_like(x), np.zeros_like(x)
    for j in range(len(T)):
        m = i == j
        if not m.any():
            continue
        row = T[j]
        p, ep = hhorner(r[m], 0.0 * r[m], row[2:])
        q = r[m] * p
        hi = row[0] + q
        lo = ((row[0] - hi) + q) + row[1]
        val[m] = hi + lo
        err[m] = np.abs(r[m]) * ep + hu(q) + hu(lo) + fit * np.abs(hi + lo)
    return val, err


def fresnel_mid(x):
    """1 <= x < 8: f and g from the pieces."""
    f, ef = fresnel_head(x, FRESNEL_F_T, FIT["fresnel_f_t"])
    g, eg = fresnel_head(x, FRESNEL_G_T, FIT["fresnel_g_t"])
    return fresnel_comb(f, ef, g, eg, x)


def fresnel_far(x):
    """8 <= x < 2^54: x f and x^3 g in v = 1/x^4, in plain doubles."""
    ix = 1.0 / x
    eix = hu(ix)
    ix2 = ix * ix
    eix2 = 2 * ix * eix + hu(ix2)
    v = ix2 * ix2
    ev = 2 * ix2 * eix2 + hu(v)
    pf, epf = hhorner(v, ev, F_ASYM_1)
    f = ix * pf
    ef = ix * epf + np.abs(pf) * eix + hu(f) + FIT["fresnel_f_asym"] * np.abs(f)
    ix3 = ix2 * ix
    eix3 = ix2 * eix + ix * eix2 + hu(ix3)
    pg, epg = hhorner(v, ev, G_ASYM_1)
    g = ix3 * pg
    eg = ix3 * epg + np.abs(pg) * eix3 + hu(g) + FIT["fresnel_g_asym"] * np.abs(g)
    return fresnel_comb(f, ef, g, eg, x)


# ---- the polynomials' own errors, with their coefficients as doubles (mpmath)

# Measured by fits() below (the comments of tables.rs give the fits before the rounding of the
# coefficients for some of them: 2^-57.4 for erf's, which is 2^-55.70 with its coefficients).
FIT = {"erf": 2.0 ** -55.69, "dawson": 2.0 ** -56.49, "fresnel_c": 2.0 ** -55.67, "fresnel_s": 2.0 ** -56.25,
       "erfcx_s": 2.0 ** -56.39, "erfcx_t": 2.0 ** -56.84, "dawson_t": 2.0 ** -55.01,
       # the auxiliary f and g of the Fresnel integrals on [1, 8) and from 8 (relative to f, g; the
       # head of the asymptotic ones folded into one double), sin(pi r/2) and (cos(pi r/2) - 1)/r^2
       "fresnel_f_t": 2.0 ** -57.8, "fresnel_g_t": 2.0 ** -54.65, "fresnel_f_asym": 2.0 ** -53.35,
       "fresnel_g_asym": 2.0 ** -54.4, "sinq": 2.0 ** -57.4, "cosq": 2.0 ** -53.8}


def _poly(c, t):
    s = mp.mpf(0)
    for ci in reversed(c):
        s = s * t + mp.mpf(ci)
    return s


def fits(n=600):
    """The largest relative errors of the polynomials on a grid of their intervals (each below
    its constant in FIT)."""
    mp.mp.prec = 200
    F = lambda x: mp.sqrt(mp.pi) / 2 * mp.exp(-x * x) * mp.erfi(x)
    xs = [mp.mpf(i) / n * 0.5 for i in range(1, n + 1)]
    xs1 = [mp.mpf(i) / n for i in range(1, n + 1)]
    got = {
        "erf": max(abs((1 + _poly(ERF_SMALL, x * x)) / (mp.erf(x) / x) - 1) for x in xs),
        "dawson": max(abs((1 + x * x * _poly(DAWSON_SMALL, x * x)) / (F(x) / x) - 1) for x in xs),
        "fresnel_c": max(abs((1 + x ** 4 * _poly(FRESNEL_C_SMALL, x ** 4)) / (mp.fresnelc(x) / x) - 1) for x in xs1),
        "fresnel_s": max(abs((mp.mpf(FRESNEL_S_SMALL[0]) + FRESNEL_S_SMALL[1] + x ** 4 * _poly(FRESNEL_S_SMALL[2:], x ** 4))
                             / (mp.fresnels(x) / x ** 3) - 1) for x in xs1),
    }
    w = 0
    for j, row in enumerate(ERFCX_S):
        for i in range(1, n // 8):
            x = mp.mpf(-0.5) + mp.mpf(j) / 8 + mp.mpf(i) / (n // 8) / 8
            r = x - (mp.mpf(-0.5) + mp.mpf(0.0625) + mp.mpf(0.125) * j)
            w = max(w, abs((mp.mpf(row[0]) + row[1] + r * _poly(row[2:], r)) / (mp.exp(x * x) * mp.erfc(x)) - 1))
    got["erfcx_s"] = w
    for name, T, f in [("erfcx_t", ERFCX_T, lambda x: mp.exp(x * x) * mp.erfc(x)), ("dawson_t", DAWSON_T, F)]:
        w = 0
        for j in range(32):
            a = 0.5 * 2 ** (j // 8) * (1 + (j % 8) / 8.0)
            b = 0.5 * 2 ** (j // 8) * (1 + (j % 8 + 1) / 8.0)
            c = float(((np.float64(a).view(np.uint64) >> np.uint64(49) << np.uint64(49)) | np.uint64(1 << 48)).view(np.float64))
            for i in range(1, 60):
                x = mp.mpf(a) + (mp.mpf(b) - a) * i / 60
                r = x - mp.mpf(c)
                w = max(w, abs((mp.mpf(T[j][0]) + T[j][1] + r * _poly(T[j][2:], r)) / f(x) - 1))
        got[name] = w
    # The Fresnel integrals from 1: f = (1/2 - S) cos - (1/2 - C) sin, g = (1/2 - C) cos + (1/2 - S) sin
    # (of pi x^2/2), so that C = 1/2 + f sin - g cos and S = 1/2 - f cos - g sin.
    def aux(x):
        s, c = mp.fresnels(x), mp.fresnelc(x)
        sn, cs = mp.sinpi(x * x / 2), mp.cospi(x * x / 2)
        return (mp.mpf(0.5) - s) * cs - (mp.mpf(0.5) - c) * sn, (mp.mpf(0.5) - c) * cs + (mp.mpf(0.5) - s) * sn
    wf = wg = 0
    for j in range(24):
        a = 2 ** (j // 8) * (1 + (j % 8) / 8.0)
        b = 2 ** (j // 8) * (1 + (j % 8 + 1) / 8.0)
        c = float(((np.float64(a).view(np.uint64) >> np.uint64(49) << np.uint64(49)) | np.uint64(1 << 48)).view(np.float64))
        for i in range(0, 61):
            x = mp.mpf(a) + (mp.mpf(b) - a) * i / 60
            if x >= 8:
                continue
            f, g = aux(x)
            r = x - mp.mpf(c)
            F, G = FRESNEL_F_T[j], FRESNEL_G_T[j]
            wf = max(wf, abs((mp.mpf(F[0]) + F[1] + r * _poly(F[2:], r)) / f - 1))
            wg = max(wg, abs((mp.mpf(G[0]) + G[1] + r * _poly(G[2:], r)) / g - 1))
    got["fresnel_f_t"], got["fresnel_g_t"] = wf, wg
    wf = wg = 0
    for i in range(0, 241):
        x = mp.mpf(8) * mp.mpf(2) ** (mp.mpf(i) / 20)
        f, g = aux(x)
        wf = max(wf, abs(_poly(F_ASYM_1, 1 / x ** 4) / (x * f) - 1))
        wg = max(wg, abs(_poly(G_ASYM_1, 1 / x ** 4) / (x ** 3 * g) - 1))
    got["fresnel_f_asym"], got["fresnel_g_asym"] = wf, wg
    ws = wc = 0
    for i in range(1, n + 1):
        r = mp.mpf(i) / (2 * n)
        s2 = r * r
        ws = max(ws, abs(r * (mp.mpf(SINQ[0]) + SINQ[1] + s2 * _poly(SINQ[2:], s2)) / mp.sinpi(r / 2) - 1))
        wc = max(wc, abs(_poly(COSQ, s2) / ((mp.cospi(r / 2) - 1) / s2) - 1))
    got["sinq"], got["cosq"] = ws, wc
    for k, v in got.items():
        assert v <= FIT[k], (k, float(v), FIT[k])
        print(f"fit {k:<10} 2^{float(mp.log(v, 2)):.2f} (taken as 2^{np.log2(FIT[k]):.2f})")


# ---- the inverses: the error of the Newton or Halley step from its residual


def inverses():
    """erf_inv for |x| <= 1/2 (and erfc_inv on (1/2, 3/2)), and erfc_inv_tail with y0 below and
    above 1/2: the last rounding plus the error of the correction, which carries the error of the
    residual erf(y0) - x (or erfc(y0) - c, (erfc(y0) - c) e^(y0^2)) times 1/erf'(y0). The first
    guess's own error enters squared (2^-62) and is left out."""
    out = {}
    y = np.linspace(1e-6, float(mp.erfinv(0.5)), 2_000_001)
    t = y * y
    q, eq = rhorner(t, U * t, ERF_SMALL)
    res = np.abs(y) * eq + FIT["erf"] * np.abs(y + y * q) + U * 2.0 ** -33 * np.abs(y)
    b = 0.5 + res * 0.886226925452758 * np.exp(t) * (1 + 4 * U) / ulp(y)
    out["erf_inv |x| <= 1/2"] = b.max()
    y = np.linspace(float(mp.erfinv(0.5)), 0.5, 2_000_001)[:-1]
    t = y * y
    q, eq = rhorner(t, U * t, ERF_SMALL)
    err = np.abs(y) * eq + U * np.abs(y * q) + FIT["erf"] * np.abs(y + y * q)
    out["erfc_inv tail, y0 < 1/2"] = (0.5 + err * 0.886226925452758 * np.exp(t) * (1 + 4 * U) / ulp(y)).max()
    y = np.linspace(0.5, 8.0, 200_001)[:-1]
    ex = np.array([float(mp.exp(v * v) * mp.erfc(v)) for v in y[::100]]).repeat(100)[: len(y)]
    erfcx_err = (head_table(y, ERFCX_T, FIT["erfcx_t"]) - 0.5) * ulp(ex) + 2.0 ** -58 * ex
    out["erfc_inv tail, y0 >= 1/2"] = (0.5 + erfcx_err * 0.886226925452758 * (1 + 4 * U) / ulp(y)).max()
    return out


# ---- the complex regions


def _scale(fn, x, y, mod):
    """The scale of the docs at x + iy (mod = |w| of the shadow)."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if fn == "faddeeva":
            s = mod.copy()
            lo = (y < 0) & (x != 0)
            if lo.any():
                _, mw = bound("faddeeva", -x[lo], -y[lo])
                s[lo] = np.maximum(np.maximum(2 * np.exp((y[lo] - x[lo]) * (y[lo] + x[lo])), mw), mod[lo])
            return s
        if fn == "erfcx":
            return _scale("faddeeva", -y, x, mod)
        s = mod.copy()
        if fn == "erfc":
            le = (x < 0) & (y != 0)
            if le.any():
                _, m2 = bound("erfc", -x[le], -y[le])
                s[le] = np.maximum(np.maximum(2.0, m2), mod[le])
            return s
        big = (x * x + y * y >= 1) & (x != 0) & (y != 0)
        if big.any():
            xs = np.abs(x[big])
            ys = np.where(x[big] < 0, -y[big], y[big])
            _, m2 = bound("erfc", xs, ys)
            s[big] = np.maximum(np.maximum(1.0, m2), mod[big])
        return s


def _general(fn, x, y):
    """Off the axes and the near-axis paths (their parts come from the real functions)."""
    if fn in ("erf", "erfc"):
        k = np.abs(x) * np.maximum(np.abs(y), 1) >= 2.0 ** -14
    elif fn == "faddeeva":
        k = ~((y < 0) & (np.abs(x) * np.maximum(-y, 1) < 2.0 ** -30))
    else:
        k = ~((x < 0) & (np.abs(y) * np.maximum(-x, 1) < 2.0 ** -30))
    return k & (x != 0) & (y != 0)


def complex_ratio(fn, x, y):
    B, mod = bound(fn, x, y)
    return B / _scale(fn, x, y, mod) / EPS


def maximise(f, gen, ok, rng, n=200_000, starts=3, rounds=5, top=30, per=300):
    """The largest f over gen's arguments that pass ok, by random sampling and a local search."""
    best = []
    for _ in range(starts):
        x, y = gen(n)
        k = ok(x, y)
        x, y = x[k], y[k]
        with np.errstate(all="ignore"):
            v = f(x, y)
        v = np.where(np.isfinite(v), v, -1)
        best += [(v[i], x[i], y[i]) for i in np.argsort(-v)[:top]]
    best.sort(key=lambda t: -t[0])
    best = best[:top]
    for _ in range(rounds):
        xs, ys = [], []
        for b in best:
            sc = 10.0 ** rng.uniform(-9, -1, per)
            xs.append(b[1] + sc * rng.standard_normal(per) * max(abs(b[1]), 1e-3))
            ys.append(b[2] + sc * rng.standard_normal(per) * max(abs(b[2]), 1e-3))
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        k = ok(xs, ys)
        xs, ys = xs[k], ys[k]
        with np.errstate(all="ignore"):
            v = f(xs, ys)
        v = np.where(np.isfinite(v), v, -1)
        best += [(v[i], xs[i], ys[i]) for i in np.argsort(-v)[:top]]
        best.sort(key=lambda t: -t[0])
        best = best[:top]
    return best[0]


def complex_regions(rng):
    lu = lambda lo, hi, m: 10.0 ** rng.uniform(lo, hi, m)

    def polar(lo, hi, m, a0=-np.pi, a1=np.pi, log=True):
        r = lu(np.log10(lo), np.log10(hi), m) if log else rng.uniform(0, 1, m) ** 0.5 * hi
        t = rng.uniform(a0, a1, m)
        return r * np.cos(t), r * np.sin(t)

    box = lambda m: (rng.uniform(-10, 10, m), rng.uniform(-10, 10, m))
    inbox = lambda x, y: (np.abs(x) <= 10) & (np.abs(y) <= 10)
    out = lambda x, y: ~inbox(x, y)

    def diag(m, sx, sy):
        """Next to the diagonals, where e^(-+z^2) has a size (|y^2 - x^2| < 700), |z| to 1e150."""
        x = lu(1, 150, m)
        y = np.sqrt(np.maximum(x * x + rng.uniform(-700, 700, m), 0.0))
        return sx(m) * x, sy(m) * y

    pm = lambda m: np.where(rng.random(m) < 0.5, -1.0, 1.0)
    one = lambda m: np.ones(m)
    near = lambda x, y: out(x, y) & (np.abs((np.abs(y) - np.abs(x)) * (np.abs(y) + np.abs(x))) < 700)
    return [
        ("faddeeva", "upper half-plane, |z| < 10", lambda m: polar(0, 10, m, 0, np.pi, False), lambda x, y: (y > 0) & (x * x + y * y < 100)),
        ("faddeeva", "upper half-plane, |z| >= 10", lambda m: polar(10, 1e300, m, 0, np.pi), lambda x, y: (y > 0) & (x * x + y * y >= 100)),
        ("faddeeva", "lower half-plane, [-10, 10] x [-10, 0)", lambda m: (rng.uniform(-10, 10, m), -rng.uniform(0, 10, m)),
         lambda x, y: (y < 0) & inbox(x, y)),
        ("faddeeva", "lower half-plane, elsewhere (|z| to 1e4)", lambda m: polar(10, 1e4, m, -np.pi, 0), lambda x, y: (y < 0) & out(x, y)),
        ("faddeeva", "lower half-plane, next to the diagonals (|z| to 1e150)", lambda m: diag(m, pm, lambda k: -one(k)),
         lambda x, y: (y < 0) & near(x, y)),
        ("erf", "|z| < 1", lambda m: polar(0, 1, m, log=False), lambda x, y: x * x + y * y < 1),
        ("erf", "[-10, 10]^2, |z| >= 1", box, lambda x, y: inbox(x, y) & (x * x + y * y >= 1)),
        ("erf", "elsewhere (|z| to 1e4)", lambda m: polar(10, 1e4, m), out),
        ("erf", "next to the diagonals (|z| to 1e150)", lambda m: diag(m, pm, pm), near),
        ("erfc", "[-10, 10]^2", box, inbox),
        ("erfc", "elsewhere (|z| to 1e4)", lambda m: polar(10, 1e4, m), out),
        ("erfc", "elsewhere, Re z > 0 (|z| to 1e4)", lambda m: polar(10, 1e4, m, -np.pi / 2, np.pi / 2), lambda x, y: out(x, y) & (x > 0)),
        ("erfc", "next to the diagonals (|z| to 1e150)", lambda m: diag(m, pm, pm), near),
        ("erfcx", "[-10, 10]^2", box, inbox),
        ("erfcx", "elsewhere, Re z > 0", lambda m: polar(10, 1e300, m, -np.pi / 2, np.pi / 2), lambda x, y: out(x, y) & (x > 0)),
        ("erfcx", "elsewhere, Re z < 0 (|z| to 1e4)", lambda m: polar(10, 1e4, m, np.pi / 2, 3 * np.pi / 2), lambda x, y: out(x, y) & (x < 0)),
        ("erfcx", "Re z < 0, next to the diagonals (|z| to 1e150)", lambda m: diag(m, lambda k: -one(k), pm), lambda x, y: (x < 0) & near(x, y)),
    ]


def re_w_ratio(x, y):
    r, _ = w_upper(x, y)
    return r.e / ulp(r.v)


def main():
    rng = np.random.default_rng(2026_11_21)
    fits()
    print("\nReal functions, ulps (the last rounding as half an ulp):")
    for name, f, lo, hi in [("erf |x| < 1/2", erf_real_small, 0, 0.5), ("erfc |x| < 1/2", erfc_real_small, -0.5, 0.5),
                            ("erfcx |x| < 1/2", erfcx_small, -0.5, 0.5), ("dawson |x| < 1/2", dawson_small, 0, 0.5)]:
        x = np.concatenate([rng.uniform(lo, hi, 2_000_000), np.linspace(lo, hi, 2_000_001)[1:-1]])
        x = x[x != 0]
        print(f"  {name:<28} {f(x).max():.4f}")
    x = np.concatenate([rng.uniform(1e-3, 1, 2_000_000), np.linspace(1e-3, 1, 2_000_001)[:-1]])
    bs, bc = fresnel_small(x)
    print(f"  {'fresnel S |x| < 1':<28} {bs.max():.4f}")
    print(f"  {'fresnel C |x| < 1':<28} {bc.max():.4f}")
    x = np.concatenate([rng.uniform(0.5, 8, 2_000_000), np.linspace(0.5, 8, 2_000_001)[:-1]])
    print(f"  {'erfcx [1/2, 8)':<28} {head_table(x, ERFCX_T, FIT['erfcx_t']).max():.4f}")
    print(f"  {'dawson [1/2, 8)':<28} {head_table(x, DAWSON_T, FIT['dawson_t']).max():.4f}")
    x = np.concatenate([rng.uniform(1, 8, 4_000_000), np.linspace(1, 8, 4_000_001)[:-1]])
    bs, bc = fresnel_mid(x)
    print(f"  {'fresnel S [1, 8)':<28} {bs.max():.4f}")
    print(f"  {'fresnel C [1, 8)':<28} {bc.max():.4f}")
    x = np.concatenate([rng.uniform(8, 64, 3_000_000), np.linspace(8, 64, 3_000_001)[:-1], 2.0 ** rng.uniform(6, 53.99, 2_000_000)])
    bs, bc = fresnel_far(x)
    print(f"  {'fresnel S [8, 2^54)':<28} {bs.max():.4f}")
    print(f"  {'fresnel C [8, 2^54)':<28} {bc.max():.4f}")
    for k, v in inverses().items():
        print(f"  {k:<28} {v:.4f}")
    print("\nComplex functions, eps of the scale:")
    for fn, name, gen, ok in complex_regions(rng):
        b = maximise(lambda x, y: complex_ratio(fn, x, y), gen, lambda x, y: ok(x, y) & _general(fn, x, y), rng)
        print(f"  {fn:<8} {name:<42} {b[0]:.3f} at ({b[1]!r}, {b[2]!r})", flush=True)
    print("\nRe w on its own next to the real axis (0 < y <= 1e-2), ulps of Re w:")
    for name, gx, okx in [("|x| < 10", lambda m: rng.uniform(-10, 10, m), lambda x: np.abs(x) < 10),
                          ("|x| >= 10 (to 1e4)", lambda m: np.where(rng.random(m) < .5, -1, 1) * 10.0 ** rng.uniform(1, 4, m), lambda x: np.abs(x) >= 10)]:
        gen = lambda m, gx=gx: (gx(m), 10.0 ** rng.uniform(-300, -2, m))
        b = maximise(re_w_ratio, gen, lambda x, y, okx=okx: okx(x) & (y > 0) & (y <= 1e-2), rng)
        print(f"  {name:<28} {b[0]:.2f} at ({b[1]!r}, {b[2]!r})", flush=True)


if __name__ == "__main__":
    with np.errstate(all="ignore"):
        main()
