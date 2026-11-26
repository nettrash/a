"""The reference data of the gamma area (mpmath, 256 bits): Γ, ln|Γ|, the complex Γ and ln Γ,
ψ and ψ⁽ⁿ⁾, B and ln B, P and Q with their inverses, I and its complement with their inverses,
and factorials, binomials and Pochhammer's symbol of floats.

Run from this directory: `python3 gen_gamma.py` (the repository's files), or
`python3 gen_gamma.py --big K --out DIR` for K times as many random cases in DIR (the area's
larger harness, not in the repository).

Each file ends with the regression cases of the November 2026 audit (`audit_*` below: the
arguments that exposed a defect and points around them), drawn from their own generators so that
the earlier cases are unchanged.
"""

import cmath
import math
import random
import struct
import sys

sys.dont_write_bytecode = True  # no __pycache__ beside the data

import mpmath

from gen_common import log_uniform, mp_arg, to_f32, uniform, write

BIG = 1
OUT = "."
if "--big" in sys.argv:
    BIG = int(sys.argv[sys.argv.index("--big") + 1])
if "--out" in sys.argv:
    OUT = sys.argv[sys.argv.index("--out") + 1]
ONLY = None
if "--only" in sys.argv:
    ONLY = sys.argv[sys.argv.index("--only") + 1].split(",")

rng = random.Random(2020)
mpf = mpmath.mpf


def n(k):
    return k * BIG


def wanted(name):
    return ONLY is None or name in ONLY


def emit(name, header, rows, keep=1.0, extra=()):
    """Writes `gamma_<name>.txt`; `keep` < 1 thins the repository's file evenly (not the larger
    harness's), keeping the last ten rows (the special cases); the `extra` rows (the audit's
    regression cases) follow, never thinned."""
    if not wanted(name):
        return
    if BIG == 1 and keep < 1.0:
        k = len(rows)
        rows = [r for i, r in enumerate(rows) if int((i + 1) * keep) > int(i * keep) or i >= k - 10]
    write(f"{OUT}/gamma_{name}.txt", header, list(rows) + list(extra))


def signed(xs):
    return [x if rng.random() < 0.5 else -x for x in xs]


def near(c, lo, hi, k, single=False):
    """`k` points c ± δ with δ log-uniform in [lo, hi] (f32 values if `single`)."""
    out = []
    for d in log_uniform(rng, lo, hi, k):
        x = c + d if rng.random() < 0.5 else c - d
        out.append(to_f32(x) if single else x)
    return out


def not_pole(x):
    return x > 0 or x != math.floor(x)


def ra(k):
    """The generator of the audit's regression cases of file `k` (independent of `rng`)."""
    return random.Random(20261121 + k)


def h(w):
    """The f64 of 16 hexadecimal digits."""
    return struct.unpack("<d", struct.pack("<Q", int(w, 16)))[0]


def lu(r, a, b):
    """A float with its logarithm uniform in [ln a, ln b]."""
    return math.exp(r.uniform(math.log(a), math.log(b)))


def near_pole_sums(r, k):
    """`k` pairs (a, b) of non-integers whose exact sum lies beside a non-positive integer -m
    (by 1e-17 to 1e-2, often within an ulp, so that the rounded sum is -m itself)."""
    out = []
    while len(out) < k:
        m = r.randint(1, 30)
        d = lu(r, 1e-17, 1e-2) * r.choice((-1, 1))
        a = -r.uniform(0.0, m)
        b = -m + d - a
        if a == math.floor(a) or b == math.floor(b):
            continue
        s = mpf(a) + mpf(b)
        if s <= 0 and s == mpmath.floor(s):
            continue
        out.append((a, b))
    return out


# ---------------------------------------------------------------- Γ and ln|Γ| of reals
def gamma_args(single):
    s = single
    xs = []
    xs += signed(log_uniform(rng, 1e-300 if not s else 1e-37, 1e-5, n(40), s))
    xs += uniform(rng, 1e-5, 0.5, n(40), s)
    xs += uniform(rng, 0.5, 1.5, n(60), s)
    xs += uniform(rng, 1.5, 8.0, n(80), s)
    top = 171.62 if not s else 35.04
    xs += uniform(rng, 8.0, top, n(120), s)
    xs += uniform(rng, top - 1.0, top, n(20), s)
    xs += uniform(rng, -8.0, 0.0, n(120), s)
    xs += uniform(rng, -190.0 if not s else -42.0, -8.0, n(120), s)
    for k in range(1, 40):
        c = -float(rng.randint(1, 180 if not s else 40))
        xs += near(c, 1e-14 if not s else 1e-6, 0.3, 1 * BIG, s)
    xs += [float(k) for k in range(1, 172 if not s else 36)]
    xs += [k + 0.5 for k in range(-20, 40)]
    xs += [2.0 ** -1074, -(2.0 ** -1074), 2.0 ** -1022, 5.5e-17, 171.6243769563027]
    xs = [to_f32(x) if s else x for x in xs]
    return [x for x in xs if not_pole(x) and x != 0]


def lgamma_args(single):
    s = single
    xs = []
    xs += log_uniform(rng, 1e-300 if not s else 1e-37, 0.5, n(60), s)
    for c in (1.0, 2.0):
        xs += near(c, 1e-17 if not s else 1e-7, 0.5, n(80), s)
    xs += uniform(rng, 0.5, 3.0, n(60), s)
    xs += uniform(rng, 3.0, 8.0, n(60), s)
    xs += log_uniform(rng, 8.0, 1e4, n(60), s)
    xs += log_uniform(rng, 1e4, 2e305 if not s else 3e36, n(60), s)
    if not s:
        xs += [2.5563479031161946e305, 2.6e305, 1e307, 1.7976931348623157e308, 4503599627370495.5, 4503599627370496.0]
    else:
        xs += [to_f32(4.0850e36), to_f32(4.09e36), 3.4028234663852886e38]
    xs += uniform(rng, -0.5, 0.0, n(40), s)
    xs += signed(log_uniform(rng, 1e-300 if not s else 1e-37, 1e-5, n(20), s))
    xs += uniform(rng, -2.0, -0.5, n(40), s)
    xs += [x for x in log_uniform(rng, 10.0, 1e15 if not s else 1e7, n(60), s)]
    xs = [-x if i % 3 == 0 and x > 10 else x for i, x in enumerate(xs)]
    for k in range(30):
        c = -float(rng.randint(1, 1000 if not s else 100))
        xs += near(c, 1e-13 if not s else 1e-5, 0.3, BIG, s)
    xs = [to_f32(x) if s else x for x in xs]
    return [x for x in xs if not_pole(x) and x != 0 and not (-2.0 < x < -0.5 and False)]


def lgamma_ref(x):
    x = mp_arg(x)
    if x > 0:
        return mpmath.loggamma(x)
    return mpmath.log(abs(mpmath.gamma(x)))


# The zeros of ln|Γ| on the negative axis, where the relative error is meaningless: there the
# absolute error is tested (file gamma_lgamma_negzero_f64).
def neg_zeros():
    zs = []
    for k in range(2, 8):
        f = lambda x: mpmath.log(abs(mpmath.gamma(x)))
        for g in (-k - 0.02, -k - 0.98, -k - 0.5):
            try:
                z = mpmath.findroot(f, mpf(g))
            except Exception:
                continue
            if -k - 1 < z < -k and all(abs(z - w) > 1e-6 for w in zs):
                zs.append(z)
    return zs


def is_near_neg_zero(x, zs):
    return any(abs(mpf(x) - z) < 0.06 for z in zs)


ZS = neg_zeros()

def audit_gamma():
    r = ra(1)
    xs = [h("bc823ab0e4735f8d"), h("3c8fffffffffffff"), 5.5e-17, -5.5e-17, 2.0 ** -55, -(2.0 ** -55)]
    xs += [lu(r, 1e-300, 5.551115123125783e-17) * r.choice((-1, 1)) for _ in range(60)]
    # 1/x just below the overflow, and subnormal results (rounded once now).
    xs += [2.0 ** -1024 * (1 + k * 2.0 ** -50) for k in (1, 2, 3, 1000)]
    xs += [x for x in (r.uniform(-184.0, -171.7) for _ in range(60)) if x != math.floor(x)]
    return [((x,), [mpmath.gamma(mp_arg(x))]) for x in xs]


xs = gamma_args(False)
emit("gamma_f64", "gamma(x), f64", [((x,), [mpmath.gamma(mp_arg(x))]) for x in xs], keep=0.75,
     extra=audit_gamma())
xs = gamma_args(True)
emit("gamma_f32", "gamma(x), f32 arguments", [((x,), [mpmath.gamma(mp_arg(x))]) for x in xs], keep=0.5)
def audit_lgamma():
    r = ra(2)
    xs = [h("3ff801e4d2520600"), 1.5000000000000018, 1.9259124119279338, 3.054995468014856,
          h("c00f9453a30cfea0"), -7.000312999690003, -3.1856353198971856, -3.2183877662725573]
    for c in (1.5, 2.0, 3.0):
        xs += [c + lu(r, 1e-16, 0.5) * r.choice((-1, 1)) for _ in range(40)]
    xs += [r.uniform(3.0, 3.5) for _ in range(40)]
    # The negative axis, the zeros included (tested by the combined measure).
    xs += [r.uniform(-8.0, -0.5) for _ in range(300)]
    for z in ZS:
        xs += [float(z) + lu(r, 1e-6, 0.06) * r.choice((-1, 1)) for _ in range(8)]
    for k in range(1, 8):
        xs += [-k - lu(r, 1e-9, 1e-2) * r.choice((-1, 1)) for _ in range(6)]
    xs = [x for x in xs if not_pole(x)]
    return [((x,), [lgamma_ref(x)]) for x in xs]


xs = lgamma_args(False)
emit("lgamma_f64", "ln|gamma(x)|, f64 (away from the zeros on the negative axis, but the audit's cases)",
     [((x,), [lgamma_ref(x)]) for x in xs if not is_near_neg_zero(x, ZS)], extra=audit_lgamma())
xs = lgamma_args(True)
emit("lgamma_f32", "ln|gamma(x)|, f32 arguments (away from the zeros on the negative axis)",
     [((x,), [lgamma_ref(x)]) for x in xs if not is_near_neg_zero(x, ZS)], keep=0.5)
xs = []
for z in ZS:
    xs += [float(z)] + [float(z) + d for d in signed(log_uniform(rng, 1e-16, 0.06, n(12)))]
emit("lgamma_negzero_f64", "ln|gamma(x)| beside its zeros on the negative axis (absolute error)",
     [((x,), [lgamma_ref(x)]) for x in xs])


# ---------------------------------------------------------------- the complex Γ and ln Γ
def cplx_args(k_rect, big):
    zs = []
    zs += [(rng.uniform(-30, 40), rng.uniform(-30, 30)) for _ in range(n(k_rect))]
    for c in (1.0, 2.0):
        for _ in range(n(30)):
            r = math.exp(rng.uniform(math.log(1e-12), math.log(0.3)))
            t = rng.uniform(-math.pi, math.pi)
            zs.append((c + r * math.cos(t), r * math.sin(t)))
    for _ in range(n(30)):
        r = math.exp(rng.uniform(math.log(1e-200), 0.0))
        t = rng.uniform(-math.pi, math.pi)
        zs.append((r * math.cos(t), r * math.sin(t)))
    zs += [(rng.uniform(-60, 0), signed([math.exp(rng.uniform(math.log(1e-10), math.log(2)))])[0]) for _ in range(n(40))]
    zs += [(rng.uniform(-3, 3), rng.uniform(-big, big)) for _ in range(n(20))]
    for _ in range(n(30)):
        r = math.exp(rng.uniform(math.log(10), math.log(big)))
        t = rng.uniform(-math.pi, math.pi)
        zs.append((r * math.cos(t), r * math.sin(t)))
    return zs


def stirling_re(z):
    """Re ln Γ(z) by Stirling's leading terms (complex floats, |z| ≥ 10; the reflection for
    Re z < 0): only to place points where |Γ| is representable."""
    if z.real < 0:
        return math.log(math.pi) - math.pi * abs(z.imag) + math.log(2) - stirling_re(1 - z)
    return ((z - 0.5) * cmath.log(z) - z).real + 0.5 * math.log(2 * math.pi)


def band_point(r, m):
    """A z with |z| = m on the right where ln|Γ(z)| (by `stirling_re`) is uniform in
    [−650, 650] (the band beside the imaginary axis where Γ is representable), or None."""
    target = r.uniform(-650.0, 650.0)
    a, b = 0.0, math.pi / 2
    fa = stirling_re(cmath.rect(m, a)) - target
    if fa * (stirling_re(cmath.rect(m, b)) - target) > 0:
        return None
    for _ in range(80):
        c = 0.5 * (a + b)
        fc = stirling_re(cmath.rect(m, c)) - target
        if (fc > 0) == (fa > 0):
            a, fa = c, fc
        else:
            b = c
    z = cmath.rect(m, 0.5 * (a + b))
    return (z.real, z.imag if r.random() < 0.5 else -z.imag)


def review_cplx():
    """The review of the fixes (round 1): Re z < 0 with a large Im z and Re z > 0 beside the
    diagonal (the argument of the logarithm, good to 2⁻⁵⁷, times |z| in Stirling's series: up to
    2.24 ε), and |z| from 10³ to 10¹⁶ in the band where |Γ| is representable (the logarithm's
    2⁻⁷² and the first-order sine of a large Im ln Γ, times |z|: to 10¹⁶ ε). Beyond 10¹⁴ the
    documented limit applies (the test's allowance |z| ln |z| · 2⁻¹⁰⁴)."""
    r = ra(24)
    zs = [(-96.51744326627312, 92.16244998915758), (-54.346539262398984, 51.91555998499905)]
    zs += [(r.uniform(-120.0, -20.0), r.uniform(20.0, 120.0)) for _ in range(60)]
    for _ in range(40):
        m, t = lu(r, 30.0, 300.0), r.uniform(math.pi / 4 - 0.15, math.pi / 4 + 0.15)
        zs.append((m * math.cos(t), m * math.sin(t) * r.choice((-1, 1))))
    while len(zs) < 102 + 80:
        p = band_point(r, lu(r, 1e3, 1e14))
        if p is not None:
            zs.append(p)
    while len(zs) < 102 + 100:
        p = band_point(r, lu(r, 1e14, 1e16))
        if p is not None:
            zs.append(p)
    g_rows, l_rows = [], []
    for a, b in zs:
        z = mpmath.mpc(a, b)
        with mpmath.workprec(mpmath.mp.prec + 2 * int(math.log2(abs(complex(a, b)))) + 64):
            g, lg = mpmath.gamma(z), mpmath.loggamma(z)
        if mpf(10) ** -300 < abs(g) < mpf(10) ** 300:
            g_rows.append(((a, b), [+g]))
        l_rows.append(((a, b), [+lg]))
    return g_rows, l_rows


REVIEW_GAMMA_C64, REVIEW_LGAMMA_C64 = review_cplx()
zs = cplx_args(150, 1e6)
emit("lgamma_c64", "principal ln gamma(z), Complex<f64>", [((a, b), [mpmath.loggamma(mpmath.mpc(a, b))]) for a, b in zs], keep=0.8,
     extra=REVIEW_LGAMMA_C64)
zs = [(a, b) for a, b in cplx_args(150, 400.0) if abs(a) < 180]
zs += [(rng.uniform(-180, 175), rng.uniform(-5, 5)) for _ in range(n(40))]
rows = [((a, b), [mpmath.gamma(mpmath.mpc(a, b))]) for a, b in zs if b != 0]
emit("gamma_c64", "gamma(z), Complex<f64> (|gamma(z)| above 1e-300)", [r for r in rows if abs(r[1][0]) > 1e-300], keep=0.8,
     extra=REVIEW_GAMMA_C64)


# ---------------------------------------------------------------- ψ and ψ⁽ⁿ⁾
X0 = mpmath.findroot(mpmath.digamma, 1.46)


def psi_neg_zeros(k_max):
    zs = []
    for k in range(0, k_max):
        # one zero in each (−k − 1, −k)
        f = mpmath.digamma
        a, b = mpf(-k - 1) + mpf("1e-12"), mpf(-k) - mpf("1e-12")
        zs.append(mpmath.findroot(f, (a, b), solver="anderson"))
    return zs


PZ = psi_neg_zeros(12)


def digamma_args(single):
    s = single
    xs = near(float(X0), 1e-17 if not s else 1e-7, 0.5, n(60), s)
    xs += log_uniform(rng, 1e-300 if not s else 1e-37, 1.0, n(40), s)
    xs += uniform(rng, 1.0, 8.0, n(60), s)
    xs += log_uniform(rng, 8.0, 1e300 if not s else 1e37, n(60), s)
    xs += uniform(rng, -12.0, 0.0, n(80), s)
    xs += [-x for x in log_uniform(rng, 12.0, 1e15 if not s else 1e6, n(40), s)]
    for k in range(25):
        c = -float(rng.randint(1, 1000 if not s else 100))
        xs += near(c, 1e-13 if not s else 1e-5, 0.3, BIG, s)
    xs += [-0.5, -1.5, 0.5, 1.0, 2.0, 10.0]
    xs = [to_f32(x) if s else x for x in xs]
    return [x for x in xs if not_pole(x) and x != 0 and all(abs(mpf(x) - z) > 0.02 for z in PZ)]


def audit_digamma():
    r = ra(3)
    xs = [5e-324, 1e-320, 1.09215821018e-312, 2.0 ** -1024, h("0004000000000001"), 1e-310,
          2.0 ** -1022, h("0004003a54da2fbf"), h("00115c79a2e6aa9a")]
    xs += [-x for x in xs]
    xs += [lu(r, 5e-324, 2.3e-308) * r.choice((-1, 1)) for _ in range(40)]
    xs += [lu(r, 2.3e-308, 1e-15) * r.choice((-1, 1)) for _ in range(20)]
    return [((x,), [mpmath.digamma(mp_arg(x))]) for x in xs if x != 0]


emit("digamma_f64", "digamma(x), f64 (away from the zeros on the negative axis)",
     [((x,), [mpmath.digamma(mp_arg(x))]) for x in digamma_args(False)], extra=audit_digamma())
emit("digamma_f32", "digamma(x), f32 arguments (away from the zeros on the negative axis)",
     [((x,), [mpmath.digamma(mp_arg(x))]) for x in digamma_args(True)])
xs = []
for z in PZ:
    xs += [float(z)] + [float(z) + d for d in signed(log_uniform(rng, 1e-16, 0.02, n(8)))]
emit("digamma_negzero_f64", "digamma(x) beside its zeros on the negative axis (absolute error)",
     [((x,), [mpmath.digamma(mp_arg(x))]) for x in xs])


ORDERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 25, 40, 70, 120, 170, 250, 500]


def polygamma_rows(neg):
    rows = []
    for m in ORDERS:
        k = 1 if m > 100 else 2
        if not neg:
            xs = log_uniform(rng, 1e-6, 1.0, n(4 * k)) + uniform(rng, 1.0, 40.0, n(4 * k))
            xs += log_uniform(rng, 40.0, 1e12, n(3 * k))
            if m <= 8:
                xs += log_uniform(rng, 1e-300 / 10 ** (0), 1e-6, n(2))
        else:
            xs = [x for x in uniform(rng, -12.0, 0.0, n(6 * k)) if abs(x - round(x)) > 1e-3]
            xs += [-x for x in log_uniform(rng, 12.0, 1e6, n(2 * k)) if abs(x - round(x)) > 1e-3]
        for x in xs:
            rows.append(((float(m), x), [mpmath.polygamma(m, mp_arg(x))]))
    return rows


def audit_polygamma():
    r = ra(4)
    xs = [2.0 ** -512 * f for f in (1 - 2.0 ** -53, 1.0, 1 + 2.0 ** -52, 1 + 2.0 ** -51)]
    xs += [h("4003e29293f4d58e"), 2.4832957441179175, 3.000000000864111]
    edges = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for _ in range(300):
        e = r.choice(edges)
        x = e + lu(r, 1e-16, 0.03) * r.choice((-1, 1))
        xs.append(x)
    return [((1.0, x), [mpmath.polygamma(1, mp_arg(x))]) for x in xs]


emit("polygamma_f64", "polygamma(n, x), f64, x > 0 (args: n as a float, then x)", polygamma_rows(False),
     extra=audit_polygamma())


def neg_scale(m, x):
    x = mp_arg(x)
    k = mpmath.ceil(-x)
    r, p = -(x + k - 1), x + k
    return mpmath.factorial(m) * (r ** -(m + 1) + p ** -(m + 1))


emit("polygamma_neg_f64",
     "polygamma(n, x), f64, x < 0: the value, then the size of its largest terms, n! (r^-(n+1) + p^-(n+1))",
     [(a, v + [neg_scale(int(a[0]), a[1])]) for a, v in polygamma_rows(True)])


# ---------------------------------------------------------------- B, ln B, x!, C(n, k), (x)ₙ
def beta_rows(neg):
    rows = []
    if not neg:
        pairs = [(a, b) for a, b in zip(log_uniform(rng, 1e-5, 1e5, n(160)), log_uniform(rng, 1e-5, 1e5, n(160)))]
        pairs += [(a, b) for a, b in zip(log_uniform(rng, 1e-300, 1e-5, n(30)), log_uniform(rng, 1e-3, 1e3, n(30)))]
        pairs += [(a, b) for a, b in zip(log_uniform(rng, 1e5, 1e300, n(40)), log_uniform(rng, 1e-3, 1e300, n(40)))]
        pairs += [(a, b) for a, b in zip(uniform(rng, 0.5, 10.0, n(40)), uniform(rng, 0.5, 10.0, n(40)))]
        pairs += [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (8.0, 8.0), (1e10, 1e10), (1.0, 1e300)]
    else:
        pairs = [(a, b) for a, b in zip(uniform(rng, -10.0, 10.0, n(80)), uniform(rng, -10.0, 10.0, n(80)))]
        pairs = [(a, b) for a, b in pairs if (a < 0 or b < 0) and a != int(a) and b != int(b) and a + b != int(a + b)]
    for a, b in pairs:
        # ln Γ(a) + ln Γ(b) − ln Γ(a + b) cancels about log₂ max(a, b) + 10 bits.
        extra = int(math.log2(max(abs(a), abs(b), 2.0))) + 40
        with mpmath.workprec(256 + extra):
            A, B_ = mp_arg(a), mp_arg(b)
            if A > 0 and B_ > 0:
                lv = mpmath.loggamma(A) + mpmath.loggamma(B_) - mpmath.loggamma(A + B_)
                v = mpmath.exp(lv)
            else:
                v = mpmath.beta(A, B_)
                lv = mpmath.log(abs(v))
            rows.append(((a, b), [+v, +lv]))
    return rows


def beta_pairs_rows(pairs):
    """beta_rows' references for given pairs (precision as there, with room for the cancellation
    of the logarithms of large negative arguments)."""
    rows = []
    for a, b in pairs:
        extra = int(math.log2(max(abs(a), abs(b), 2.0))) + 40
        with mpmath.workprec(256 + 2 * extra):
            A, B_ = mp_arg(a), mp_arg(b)
            if A > 0 and B_ > 0:
                lv = mpmath.loggamma(A) + mpmath.loggamma(B_) - mpmath.loggamma(A + B_)
                v = mpmath.exp(lv)
            else:
                v = mpmath.gamma(A) * mpmath.gamma(B_) / mpmath.gamma(A + B_)
                lv = mpmath.log(abs(v))
            rows.append(((a, b), [+v, +lv]))
    return rows


def audit_beta():
    r = ra(5)
    big = 1.7976931348623157e308
    pairs = [(1.3e308, 1.3e308), (1.4e308, 1.4e308), (big, big), (big, 1e308), (1.2e308, 1.5e308),
             (10.0, big), (20.5, big), (9.5, 1e308), (5.5, 1e20), (1e20, 3.75), (2.5, 3e17), (7.5, 9e15)]
    # One argument more than 2^53 times the other (ln(1 + b/a) from ln1p).
    for _ in range(40):
        a = lu(r, 0.05, 40.0)
        pairs.append((a, a * lu(r, 2.0 ** 53, 1e18)) if r.random() < 0.5 else (a * lu(r, 2.0 ** 53, 1e18), a))
    return beta_pairs_rows(pairs)


def audit_beta_neg():
    r = ra(6)
    pairs = [(-0.3, -0.7), (-0.1, -0.9), (-1.1, -0.9), (-2.3, -0.7), (0.3, -1.3), (-6.3, -18.7), (0.1, -2.1),
             (-6.772712835730017, -18.227287164270408), (-0.3, -0.7000000000000002), (-10.3, -14.700000000000005),
             (-0.25, -0.7500000000000002), (-131467.00173281308, 0.0018932383038010971),
             (-1e6 - 0.3, 0.7), (-1e9 - 0.3, 0.7), (-1e12 - 0.3, 0.7), (-1e15 - 0.5, 0.7)]
    pairs += near_pole_sums(r, 120)
    # Large negative non-integers beside small arguments of either sign.
    for _ in range(40):
        a = -lu(r, 1e3, 4e15)
        b = r.uniform(-5.0, 5.0)
        if a != math.floor(a) and b != math.floor(b):
            pairs.append((a, b) if r.random() < 0.5 else (b, a))
    # (the test reads ln|B| in plain ulps here: only where it is not small)
    return [row for row in beta_pairs_rows(pairs) if abs(row[1][1]) >= 0.5]


emit("beta_f64", "beta(a, b) and ln|beta(a, b)|, f64, a, b > 0", beta_rows(False), extra=audit_beta())
emit("beta_neg_f64", "beta(a, b) and ln|beta(a, b)|, f64, a or b < 0", beta_rows(True), extra=audit_beta_neg())
rows = []
for a, b in zip(log_uniform(rng, 1e-3, 1e3, n(80), True), log_uniform(rng, 1e-3, 1e3, n(80), True)):
    v = mpmath.beta(mp_arg(a), mp_arg(b))
    rows.append(((a, b), [v, mpmath.log(v)]))
emit("beta_f32", "beta(a, b) and ln beta(a, b), f32 arguments", rows)

xs = uniform(rng, -0.999, 8.0, n(60)) + uniform(rng, 8.0, 170.6, n(60)) + signed(log_uniform(rng, 1e-300, 1e-3, n(20)))
xs += [x for x in uniform(rng, -190.0, -1.0, n(60)) if x != int(x)]
xs += [float(k) for k in range(0, 171, 7)] + [k + 0.5 for k in range(-10, 10)]
def audit_factorial():
    r = ra(7)
    xs = [-171.76081656443694, -171.93752631038475, -171.75068879583918, -171.94740836946832]
    xs += [x for x in (r.uniform(-185.0, -171.0) for _ in range(80)) if x != math.floor(x)]
    return [((x,), [mpmath.gamma(mp_arg(x) + 1)]) for x in xs]


emit("factorial_f64", "x! = gamma(x + 1), f64", [((x,), [mpmath.gamma(mp_arg(x) + 1)]) for x in xs],
     extra=audit_factorial())


def binomial_rows():
    rows = []
    for _ in range(n(80)):
        nn = float(rng.randint(0, 1100))
        k = float(rng.randint(0, int(nn)))
        rows.append((nn, k))
    for _ in range(n(40)):
        nn = float(rng.randint(10 ** 4, 10 ** 15))
        k = float(rng.randint(0, 40))
        rows.append((nn, k))
    for _ in range(n(30)):
        nn = -float(rng.randint(1, 200))
        k = float(rng.randint(0, 60))
        rows.append((nn, k))
    rows += [(a, b) for a, b in zip(uniform(rng, 0.0, 60.0, n(40)), uniform(rng, 0.0, 60.0, n(40))) if b < a + 0.9]
    rows += [(a, float(rng.randint(0, 25))) for a in uniform(rng, -30.0, 30.0, n(40))]
    out = []
    for nn, k in rows:
        v = mpmath.binomial(mp_arg(nn), mp_arg(k))
        out.append(((nn, k), [v]))
    return out


def audit_binomial():
    r = ra(8)
    rows = [(0.1, 19.1), (0.1, 5.1), (8.66611728504263, 26.666117285042642), (-5.558525121058192, 12.441474878940786),
            (-1e6 - 0.3, 0.7), (-1e9 - 0.3, 0.7), (-1e12 - 0.3, 0.7), (-1e15 - 0.5, 0.7),
            (2.5240532531942624e26, 5.740889315234043), (1e20, 2.5), (1e20, 10.5), (1e15, 5.123456789)]
    # n − k + 1 beside a non-positive integer.
    while len(rows) < 12 + 100:
        m = r.randint(1, 30)
        d = lu(r, 1e-17, 1e-2) * r.choice((-1, 1))
        k = r.uniform(0.0, 30.0)
        nn = -m - 1 + k + d
        w = mpf(nn) - mpf(k) + 1
        if k != math.floor(k) and not (w <= 0 and w == mpmath.floor(w)):
            rows.append((nn, k))
    # Huge n beside a real k.
    rows += [(lu(r, 1e16, 1e30), r.uniform(0.0, 40.0)) for _ in range(40)]
    out = []
    for nn, k in rows:
        with mpmath.workprec(256 + 2 * (int(math.log2(max(abs(nn), 2.0))) + 40)):
            out.append(((nn, k), [+mpmath.binomial(mp_arg(nn), mp_arg(k))]))
    return out


emit("binomial_f64", "binomial(n, k), f64", binomial_rows(), extra=audit_binomial())


def pochhammer_rows():
    rows = []
    for x in log_uniform(rng, 1e-5, 1e5, n(60)):
        rows.append((x, float(rng.randint(-32, 32))))
    for x in log_uniform(rng, 1e-3, 1e5, n(60)):
        m = rng.uniform(-min(x, 50.0) * 0.999, 300.0)
        rows.append((x, m))
    rows += [(x, float(rng.randint(-10, 30))) for x in uniform(rng, -20.0, 0.0, n(30)) if x != int(x)]
    rows += [(x, rng.uniform(-5.0, 5.0)) for x in uniform(rng, -10.0, 0.0, n(20)) if x != int(x)]
    rows += [(x, float(rng.randint(33, 150))) for x in log_uniform(rng, 1e-2, 1e3, n(20))]
    out = []
    for x, m in rows:
        X, M = mp_arg(x), mp_arg(m)
        v = mpmath.rf(X, M)
        out.append(((x, m), [v]))
    return out


def audit_pochhammer():
    r = ra(9)
    rows = [(-0.3, -0.7), (0.3, -1.3), (-0.1, -0.9), (-5.930450501356775, -24.06954949864211),
            (-1e6 - 0.3, 0.7), (-1e9 - 0.3, 0.7), (-1e12 - 0.3, 0.7), (-1e15 - 0.5, 0.7),
            (5.147890760783697e19, 11.787629701987186), (1e20, 11.5), (1e20, 0.75), (7882697183662165.0, 2.481149340422057)]
    rows += near_pole_sums(r, 100)
    rows += [(-lu(r, 1e3, 4e15), r.uniform(-5.0, 5.0)) for _ in range(30)]
    rows += [(lu(r, 1e16, 1e25), r.uniform(-30.0, 30.0)) for _ in range(40)]
    rows = [(x, m) for x, m in rows if x != math.floor(x) or x > 0]
    out = []
    for x, m in rows:
        with mpmath.workprec(256 + 2 * (int(math.log2(max(abs(x), 2.0))) + 40)):
            out.append(((x, m), [+mpmath.rf(mp_arg(x), mp_arg(m))]))
    return out


emit("pochhammer_f64", "pochhammer(x, n) = gamma(x + n)/gamma(x), f64", pochhammer_rows(), extra=audit_pochhammer())


# ---------------------------------------------------------------- P, Q and their inverses
# (Q's references from the series of P below a + 1 and Legendre's continued fraction above, at
# 320 bits and more: mpmath's gammainc does not converge for large arguments.)
rng_ig = random.Random(2021)


def _pq(a, x):
    """(P, Q) at 256 bits (plus guard bits): the series of P below a + 1, Legendre's continued
    fraction of Q above; the other as the complement."""
    a, x = mp_arg(a), mp_arg(x)
    if x == 0:
        return mpf(0), mpf(1)
    # Q = 1 − P cancels for tiny a: enough guard bits for it.
    extra = int(max(0, -mpmath.log(a, 2))) + 64 if a < 1 else 0
    wp = 320 + extra
    with mpmath.workprec(wp):
        lpre = a * mpmath.log(x) - x - mpmath.loggamma(a)
        if x < a + 1:
            t = 1 / a
            s = t
            k = 1
            while True:
                t *= x / (a + k)
                s += t
                if t < s * mpf(2) ** -(wp + 10):
                    break
                k += 1
            p = mpmath.exp(lpre) * s
            return +p, +(1 - p)
        if lpre < -2000:
            # Q < e^-2000 (the fraction is below 1): zero in any float.
            return mpf(1), mpf(0)
        # Q = pre / (b0 + a1/(b1 + ...)), Lentz at high precision.
        tiny = mpf(2) ** -2000
        b0 = x - a + 1
        f = b0
        c = f
        d = mpf(0)
        k = 1
        while True:
            ak = k * (a - k)
            bk = b0 + 2 * k
            d = bk + ak * d
            if d == 0:
                d = tiny
            c = bk + ak / c
            if c == 0:
                c = tiny
            d = 1 / d
            delta = c * d
            f *= delta
            if abs(delta - 1) < mpf(2) ** -330:
                break
            k += 1
        q = mpmath.exp(lpre) / f
        return +(1 - q), +q


def P(a, x):
    return _pq(a, x)[0]


def Q(a, x):
    return _pq(a, x)[1]


def pq_args(single):
    s = single
    f = (lambda v: to_f32(v)) if s else (lambda v: v)
    out = []
    # small a, any x
    for a in log_uniform(rng_ig, 1e-30 if not s else 1e-30, 1.0, n(60), s):
        out.append((a, f(math.exp(rng_ig.uniform(math.log(1e-8), math.log(40.0))))))
    for a in log_uniform(rng_ig, 1e-300 if not s else 1e-38, 1e-30, n(8), s):
        out.append((a, f(math.exp(rng_ig.uniform(math.log(1e-30), math.log(10.0))))))
    # x < 1.1 crossover band
    for _ in range(n(60)):
        out.append((f(rng_ig.uniform(0.0, 1.5)), f(rng_ig.uniform(0.0, 1.1))))
    # moderate a, x relative to a
    for a in uniform(rng_ig, 1.0, 20.0, n(120), s):
        out.append((a, f(a * math.exp(rng_ig.uniform(math.log(0.01), math.log(8.0))))))
    for a in uniform(rng_ig, 1.0, 20.0, n(60), s):
        out.append((a, f(a + rng_ig.uniform(-3.0, 3.0) * math.sqrt(a))))
    # integers and half-integers
    for _ in range(n(40)):
        a = float(rng_ig.randint(1, 40)) + (0.5 if rng_ig.random() < 0.4 else 0.0)
        out.append((a, f(a * math.exp(rng_ig.uniform(math.log(0.05), math.log(4.0))))))
    # large a near the transition, and further out
    for lo, hi in ((20.0, 200.0), (200.0, 1e4), (1e4, 1e6)):
        for a in log_uniform(rng_ig, lo, hi, n(50), s):
            k = rng_ig.choice([0.5, 2.0, 6.0, 12.0])
            out.append((a, f(a + rng_ig.uniform(-1.0, 1.0) * k * math.sqrt(a))))
        for a in log_uniform(rng_ig, lo, hi, n(20), s):
            out.append((a, f(a * math.exp(rng_ig.uniform(math.log(0.1), math.log(5.0))))))
    # tiny and huge x
    for a in log_uniform(rng_ig, 1e-3, 50.0, n(15), s):
        out.append((a, f(math.exp(rng_ig.uniform(math.log(1e-300 if not s else 1e-38), math.log(1e-8))))))
    for a in log_uniform(rng_ig, 1e-3, 1e3, n(15), s):
        out.append((a, f(math.exp(rng_ig.uniform(math.log(1e3), math.log(1e300 if not s else 1e38))))))
    return [(a, x) for a, x in out if a > 0 and x > 0 and math.isfinite(x)]


def ig_inverse(target, a, y, upper):
    """The x with P(a, x) = y (Q if `upper`), at 256 bits; `target` a starting point."""
    a = mp_arg(a)
    y = mp_arg(y)
    # in the log domain: g = ln S(a, t) − ln y, monotone in t
    ly = mpmath.log(y)
    if upper:
        g = lambda t: mpmath.log(_pq(a, t)[1]) - ly
    else:
        g = lambda t: mpmath.log(_pq(a, t)[0]) - ly
    # log-domain bisection then secant (robust for tiny y)
    lo, hi = mpf(target) / 4, mpf(target) * 4 + 1
    while g(lo) * g(hi) > 0:
        lo /= 16
        hi *= 4
    for _ in range(40):
        mid = (lo + hi) / 2 if lo > hi / 1e3 else mpmath.sqrt(lo * hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    r = mpmath.findroot(g, (lo, hi), solver="anderson", tol=mpf(10) ** -70)
    assert abs(g(r)) < mpf(10) ** -60, (a, y, r)
    return r


def inv_rows(single, upper):
    s = single
    rows = []
    args = pq_args(single)
    rng_ig.shuffle(args)
    for a, x in args[: n(160)]:
        v = (Q if upper else P)(a, x)
        if not (mpf("1e-300") < v < 1):
            continue
        y = float(v)
        if s:
            y = to_f32(y)
            if y == 0 or y == 1:
                continue
        if y == 1.0:
            continue
        try:
            r = ig_inverse(x, a, y, upper)
        except Exception:
            continue
        rows.append(((a, y), [r]))
    return rows



def audit_pq():
    """Q for small a beyond x = 1.1 (q_small_a's cancelling region, now the fraction's), a = 1/2
    there, and the half-integer sums of Q."""
    r = ra(10)
    out = [(0.5098632371190795, 2.9697154035088498), (0.501910562302807, 2.968502696438861),
           (0.5, 2.5928251148347505), (0.32978368227014376, 2.99287185805912), (0.4337510048171571, 2.9348105344643463),
           (19.5, 73.0924306446073)]
    out += [(r.uniform(0.001, 0.75), r.uniform(1.1, 3.0)) for _ in range(150)]
    out += [(lu(r, 1e-4, 0.75), r.uniform(1.1, 3.0)) for _ in range(60)]
    out += [(0.5, r.uniform(1.1, 3.0)) for _ in range(40)]
    for _ in range(80):
        a = r.randint(0, 19) + 0.5
        out.append((a, a * math.exp(r.uniform(0.0, math.log(4.0)))))
    return [((a, x), list(_pq(a, x))) for a, x in out]


ps = pq_args(False)
emit("gamma_pq_f64", "P(a, x) and Q(a, x), f64", [((a, x), list(_pq(a, x))) for a, x in ps], keep=0.75,
     extra=audit_pq())
ps = pq_args(True)
emit("gamma_pq_f32", "P(a, x) and Q(a, x), f32 arguments", [((a, x), list(_pq(a, x))) for a, x in ps], keep=0.5)
def ig_root(a, y, upper, x0):
    """The x with P(a, x) = y (Q if `upper`) by Newton's method on ln S in ln x from x0 (for the
    audit's small a, whose roots are tiny: ig_inverse's bisection would not reach them)."""
    A, Y = mp_arg(a), mp_arg(y)
    w = mpmath.log(mpf(x0))
    ly = mpmath.log(Y)
    for _ in range(200):
        x = mpmath.exp(w)
        P_, Q_ = _pq(A, x)
        S = Q_ if upper else P_
        dens = mpmath.exp(A * w - x - mpmath.loggamma(A))  # x times the density
        step = (mpmath.log(S) - ly) / ((-dens if upper else dens) / S)
        w -= step
        if abs(step) < mpf(2) ** -230:
            break
    x = mpmath.exp(w)
    S = _pq(A, x)[1 if upper else 0]
    assert abs(mpmath.log(S) - ly) < mpf(10) ** -65, (a, y, x)
    return x


def audit_ginv(upper):
    """Small a, where the root is badly conditioned (kappa = 1/a), and roots in the subnormals
    (P only: P = x^a/Gamma(a + 1) (1 + O(x)) there, so its leading term gives the root to 1e-300)."""
    r = ra(11 + upper)
    cases = [(0.002088698151875441, 0.2949075783164681)] if upper else [(0.0011354924213005213, 0.6984679583797635)]
    while len(cases) < 17:
        a, t = lu(r, 1e-3, 0.04), r.uniform(0.05, 0.95)
        # (only roots well inside the normal range: P = x^a/Gamma(1 + a) to O(x))
        p = 1 - t if upper else t
        if math.log(p * math.gamma(1 + a)) / a > -660:
            cases.append((a, t))
    out = []
    for a, t in cases:
        p = 1 - t if upper else t
        x0 = math.exp(math.log(p * math.gamma(1 + a)) / a)
        out.append(((a, t), [ig_root(a, t, upper, x0)]))
    if not upper:
        sub = [(0.013045050775225536, 6.570938176088643e-05), (0.1563125004492042, 3.9938215932935026e-51)]
        while len(sub) < 2 + 30:
            a = lu(r, 0.01, 5.0)
            xt = mp_arg(lu(r, 5e-324, 2.2e-308))
            t = float(xt ** a / mpmath.gamma(1 + a))
            if 0 < t < 0.5:
                sub.append((a, t))
        for a, t in sub:
            out.append(((a, t), [(mp_arg(t) * mpmath.gamma(1 + mp_arg(a))) ** (1 / mp_arg(a))]))
    return out


for name, single, upper, head in [
    ("gamma_p_inv_f64", False, False, "the x with P(a, x) = p, f64"),
    ("gamma_q_inv_f64", False, True, "the x with Q(a, x) = q, f64"),
    ("gamma_p_inv_f32", True, False, "the x with P(a, x) = p, f32 arguments"),
    ("gamma_q_inv_f32", True, True, "the x with Q(a, x) = q, f32 arguments"),
]:
    # The inverses take minutes: only when wanted.
    if wanted(name):
        emit(name, head, inv_rows(single, upper), extra=() if single else audit_ginv(upper))


# ---------------------------------------------------------------- I, its complement, inverses
# (The references from the continued fraction at 320 bits on the side where it converges,
# checked against mpmath.betainc where that converges: betainc fails for large parameters.)
rng_ib = random.Random(2021)


def _cf(a, b, x, eps):
    # Lentz, d_{2m+1} = -(a+m)(a+b+m)x/((a+2m)(a+2m+1)), d_{2m} = m(b-m)x/((a+2m-1)(a+2m))
    tiny = mpf(10) ** (-mpmath.mp.dps * 3)
    f = mpf(1); C = mpf(1); D = mpf(0)
    m = 0
    n = 0
    while True:
        # odd step
        for k in (1, 2):
            if k == 1:
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            else:
                m1 = m + 1
                num = m1 * (b - m1) * x / ((a + 2 * m1 - 1) * (a + 2 * m1))
            D = 1 + num * D
            if D == 0: D = tiny
            C = 1 + num / C
            if C == 0: C = tiny
            D = 1 / D
            delta = C * D
            f *= delta
            n += 1
        m += 1
        if abs(delta - 1) < eps or n > 4000000:
            break
    return f, n

def ibeta_pair(a, b, x):
    """(I, 1 - I) for exact mpf a, b, x in (0, 1)."""
    a, b, x = mpf(a), mpf(b), mpf(x)
    y = 1 - x
    with mpmath.workprec(mpmath.mp.prec + 64):
        eps = mpf(2) ** (-mpmath.mp.prec - 32)
        lpre = a * mpmath.log(x) + b * mpmath.log(y) + mpmath.loggamma(a + b) - mpmath.loggamma(a) - mpmath.loggamma(b)
        if x < (a + 1) / (a + b + 2):
            f, n = _cf(a, b, x, eps)
            I = mpmath.exp(lpre) / (a * f)
            return +I, +(1 - I)
        f, n = _cf(b, a, y, eps)
        Ic = mpmath.exp(lpre) / (b * f)
        return +(1 - Ic), +Ic


SCALE = 1


def m(k):
    return n(k) * SCALE


def around_mean(a, b, k):
    """x = mean + t sigma with t in [-k, k], clamped into (0, 1)."""
    m = a / (a + b)
    s = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
    x = m + rng_ib.uniform(-k, k) * s
    return min(max(x, 1e-300), 1 - 2 ** -53)


def triples(single, scale=1):
    global SCALE
    SCALE = scale
    s = single
    r = []
    # Both parameters small.
    for _ in range(m(60)):
        a, b = log_uniform(rng_ib, 1e-6, 1.0, 2, s)
        x = rng_ib.choice([rng_ib.uniform(0, 1), math.exp(rng_ib.uniform(math.log(1e-12), 0)), 1 - math.exp(rng_ib.uniform(math.log(1e-12), 0))])
        r.append((a, b, x))
    # One small, one moderate or large.
    for _ in range(m(80)):
        a = log_uniform(rng_ib, 1e-5, 1.0, 1, s)[0]
        b = log_uniform(rng_ib, 1.0, 1e5 if not s else 1e4, 1, s)[0]
        x = rng_ib.choice([rng_ib.uniform(0, 1), math.exp(rng_ib.uniform(math.log(1e-10), 0)), 1 - math.exp(rng_ib.uniform(math.log(1e-14), -0.01))])
        r.append((a, b, x) if rng_ib.random() < 0.5 else (b, a, x))
    # Moderate parameters.
    for _ in range(m(100)):
        a, b = log_uniform(rng_ib, 1.0, 40.0, 2, s)
        x = rng_ib.choice([rng_ib.uniform(0, 1), around_mean(a, b, 4), math.exp(rng_ib.uniform(math.log(1e-8), 0))])
        r.append((a, b, x))
    # Integers and half-integers.
    for _ in range(m(30)):
        a = rng_ib.randint(1, 60) / rng_ib.choice([1, 2])
        b = rng_ib.randint(1, 60) / rng_ib.choice([1, 2])
        r.append((a, b, rng_ib.uniform(0, 1)))
    # Small b beside a large a, x near 1 (BGRAT).
    for _ in range(m(50)):
        a = log_uniform(rng_ib, 15.0, 1e5, 1, s)[0]
        b = log_uniform(rng_ib, 1e-4, 3.0, 1, s)[0]
        x = around_mean(a, b, 3) if rng_ib.random() < 0.5 else 1 - math.exp(rng_ib.uniform(math.log(1e-6), math.log(0.5))) / a * 10
        r.append((a, b, x) if rng_ib.random() < 0.7 else (b, a, 1 - x))
    # Both large, around the mean and in the tails.
    for _ in range(m(100)):
        a, b = log_uniform(rng_ib, 40.0, 1e6 if not s else 1e5, 2, s)
        x = around_mean(a, b, rng_ib.choice([1, 4, 12, 30]))
        r.append((a, b, x))
    # Very large.
    for _ in range(m(25)):
        a, b = log_uniform(rng_ib, 1e6, 1e10 if not s else 1e7, 2, s)
        r.append((a, b, around_mean(a, b, 6)))
    out = []
    for a, b, x in r:
        if s:
            a, b, x = to_f32(a), to_f32(b), to_f32(x)
        if 0 < x < 1 and a > 0 and b > 0:
            out.append((a, b, x))
    return out


def rows_of(ts):
    rows = []
    for a, b, x in ts:
        I, Ic = ibeta_pair(a, b, x)
        rows.append(((a, b, x), [I, Ic]))
    return rows


def audit_beta_inc():
    """A small a beside 8 <= b < 15 at a tiny x (a NaN), b >= 40 with b x tiny (the continued
    fraction's power terms), and a parameter of at least 2^995: the incomplete gamma limit
    I_x(a, b) = P(a, -b ln(1 - x)) (relative error O((a/b) sqrt(a) + b x^2), below 1e-280 here) with
    the exact product, and a point mass beside a parameter of at least 2^600."""
    r = ra(13)
    ts = [(0.001805913590426878, 13.552467641142972, 7.838550122480618e-50)]
    ts += [(lu(r, 1e-4, 0.6), r.uniform(8.0, 15.0), lu(r, 1e-300, 1e-35)) for _ in range(60)]
    ts += [(14.42888823517585, 704.2065003728154, 1.0907857743066827e-22),
           (18.84630933217607, 691.6984757730617, 8.434119298179883e-19),
           (7.301401765493768, 1692.4287300950919, 8.332876343100199e-34),
           (12.736719509100567, 129.29604854940385, 1.464578923062878e-21)]
    for _ in range(60):
        b = lu(r, 40.0, 1e4)
        ts.append((r.uniform(1.0, 40.0), b, lu(r, 1e-300, 1e-10 / b)))
    rows = rows_of(ts)
    huge = [(1.0, 1e300, 7.000000000000001e-298), (2.5, 1e300, 6.123456789e-298), (0.5, 1e300, 4.4444444444444e-298),
            (3.0, 1e300, 3.3333333333333e-298), (100.0, 1e300, 1.0000000000000001e-299), (300.0, 1e300, 2.0000000000000005e-299)]
    for _ in range(30):
        a = r.choice([0.5, 1.0, 2.5]) if r.random() < 0.3 else lu(r, 0.1, 1e3)
        b = lu(r, 2.0 ** 995, 1e306)
        u = max(a / 10, a + r.uniform(-5.0, 10.0) * math.sqrt(a) + r.uniform(0.0, 20.0))
        huge.append((a, b, u / b))
    for a, b, x in huge:
        with mpmath.workprec(600):
            P, Q = _pq(a, -mp_arg(b) * mpmath.log1p(-mp_arg(x)))
        rows.append(((a, b, x), [P, Q]))
        # The same with the parameters and x, 1 − x exchanged: I_y(b, a) = 1 − I_x(a, b).
        y = 1.0 - x
        if y != 1.0:
            with mpmath.workprec(600):
                P, Q = _pq(a, -mp_arg(b) * mpmath.log(mp_arg(y)))
            rows.append(((b, a, y), [Q, P]))
    for _ in range(10):
        a, b = lu(r, 2.0 ** 600, 1e299), lu(r, 2.0 ** 995, 1e307)
        m = a / (a + b)
        x = m * (1 + r.choice((-1, 1)) * lu(r, 1e-12, 1e-3))
        below = mp_arg(x) < mp_arg(a) / (mp_arg(a) + mp_arg(b))
        rows.append(((a, b, x), [mpf(0), mpf(1)] if below else [mpf(1), mpf(0)]))
    return rows


def wide_prec(a, b):
    """A working precision that covers a + b: the continued fraction's coefficients and the
    logarithms of the gamma functions of huge parameters need it."""
    return mpmath.mp.prec + int(2 * math.log2(a + b)) + 64


def wide_rows(ts):
    """`rows_of` with the precision of each case widened by `wide_prec`."""
    rows = []
    for a, b, x in ts:
        with mpmath.workprec(wide_prec(a, b)):
            I, Ic = ibeta_pair(a, b, x)
        rows.append(((a, b, x), [I, Ic]))
    return rows


def review_beta_inc():
    """The review of the fixes (round 1): one parameter from 1e14 on beside x = u/b, u about the
    other (the power terms' xc − a formed from an x near 1, hundreds of ulps), up to 2^996 (the
    continued fraction's coefficients overflowed: NaN), the exchanged arguments, x within 2^-53 of
    1 beside a huge a in BGRAT (its ln x), and a subnormal x beside b >= 8 or a tiny a (products in
    the subnormals). The references by the continued fraction at the precision of wide_prec
    (checked against mpmath.hyp2f1 and the incomplete gamma limit)."""
    r = ra(20)
    ts = [(32.07660897607658, 7.378928205788206e+17, 7.214634660029751e-17),
          (729.1989080840139, 1.1245223953321167e+19, 9.316746533268602e-17),
          (147.0184414975602, 2.1477604423873044e+19, 7.75331148381479e-18),
          (2.0, 1e50, 5e-50), (8.0, 1e100, 3.53e-99), (50.0, 1e200, 6e-199),
          (7.0, 3.797050995194295e299, 1.1902087629910181e-298),
          (0.32276792471668025, 5.501885493696907e+17, 4.955175376232243e-17),
          (0.09623715162088611, 2.61811408329382e+17, 6.864820321672101e-17),
          (0.7157280877724658, 11.269352670023864, 5e-324),
          (0.8853117542444165, 25.24220672995262, 1.5e-323)]
    for _ in range(40):
        a, b = lu(r, 1.0, 1e3), lu(r, 1e14, 1e22)
        u = max(a / 20, a + r.uniform(-6.0, 12.0) * math.sqrt(a))
        ts.append((a, b, u / b))
    for _ in range(40):
        a, b = lu(r, 0.5, 1e3), lu(r, 5e24, 2.0 ** 996)
        u = max(a / 20, a + r.uniform(-6.0, 12.0) * math.sqrt(a))
        ts.append((a, b, u / b))
    # (Exchanged: x = 1 − u/a is 1 in f64 once u/a is below 2⁻⁵⁴, so a up to 1e17.)
    for _ in range(20):
        a, b = lu(r, 1e14, 1e17), lu(r, 1.0, 1e3)
        u = max(b / 20, b + r.uniform(-6.0, 12.0) * math.sqrt(b))
        if 1.0 - u / a < 1.0:
            ts.append((a, b, 1.0 - u / a))
    for _ in range(20):
        a, b = lu(r, 0.05, 1.0), lu(r, 1e14, 1e30)
        ts.append((a, b, r.uniform(10.0, 40.0) / b))
    for _ in range(30):
        ts.append((r.uniform(0.05, 1.0), lu(r, 8.0, 1e4), lu(r, 5e-324, 2.2e-308)))
    for _ in range(20):
        ts.append((lu(r, 1e-6, 5e-4), lu(r, 1.5, 1e4), lu(r, 5e-324, 2.2e-308)))
    for _ in range(10):
        ts.append((lu(r, 1e-6, 0.15), lu(r, 0.1, 1.0), lu(r, 5e-324, 2.2e-308)))
    return wide_rows(ts)


def ibeta_quad(a, b, x):
    """(I, 1 − I) of exact a, b > 1 by quadrature (tanh-sinh) of the density in u = (t − m)/s, m
    the mode and s the width, normalised by its integral over the line (beyond |u| = 60 it is below
    e^-1800), at the precision of wide_prec: the continued fraction would need about √min(a, b)
    terms beside the mean. The tail beyond u_x, where |u_x| > 2, in the variable y = |u − u_x|·|u_x|
    relative to the density at u_x (there it falls like e^-y). (Checked against Arb on parameters
    to 5e4, and in the far tails against the leading term erfc(z₀)/2 of the uniform expansion,
    whose next term is smaller by about z₀/√a.)"""
    with mpmath.workprec(wide_prec(a, b)):
        A, B, X = mp_arg(a), mp_arg(b), mp_arg(x)
        m = (A - 1) / (A + B - 2)
        s = mpmath.sqrt(m * (1 - m) / (A + B - 2))
        g0 = (A - 1) * mpmath.log(m) + (B - 1) * mpmath.log1p(-m)

        def lf(u):
            t = m + s * u
            return (A - 1) * mpmath.log(t) + (B - 1) * mpmath.log1p(-t) - g0

        def f(u):
            t = m + s * u
            return mpf(0) if t <= 0 or t >= 1 else mpmath.exp(lf(u))

        ux = (X - m) / s
        assert abs(ux) < 55, (a, b, x)
        if abs(ux) <= 2:
            lo = mpmath.quad(f, [-60, -20, -5, ux])
            hi = mpmath.quad(f, [ux, 5, 20, 60])
        else:
            sg = 1 if ux > 0 else -1
            L = abs(ux)
            l0 = lf(ux)
            ymax = (60 - L) * L
            g = lambda y: f(ux + sg * y / L) / mpmath.exp(l0)
            tail = mpmath.exp(l0) / L * mpmath.quad(g, [0, 0.5, 2, 8, 32, 128, ymax])
            # (the side with the mode in u directly)
            if ux > 0:
                near = mpmath.quad(f, [-60, -20, -5, 0, ux])
                lo, hi = near, tail
            else:
                near = mpmath.quad(f, [ux, 0, 5, 20, 60])
                lo, hi = tail, near
        return +(lo / (lo + hi)), +(hi / (lo + hi))


def review_both_huge():
    """The review of the fixes (round 1): both parameters beyond 10¹⁵ beside the mean, where the
    side was chosen by a − (a + b)x in f64 (an ulp of a off: from 10³⁰ on more than λ, and BASYM
    then gave the other tail, I and 1 − I both 1) and d = xc − a kept 2⁻¹⁰⁶ of a. x is the mean
    plus t widths where the width is above the spacing of the floats, else the nearest floats to
    the mean; references by `ibeta_quad`."""
    r = ra(25)
    rows = []
    while len(rows) < 30:
        a = lu(r, 1e15, 1e60)
        b = a * lu(r, 1.0, 30.0) if r.random() < 0.5 else lu(r, 1e15, 1e60)
        A, B = mp_arg(a), mp_arg(b)
        m = A / (A + B)
        sd = mpmath.sqrt(A * B / ((A + B) ** 2 * (A + B + 1)))
        if sd / m > mpf(2) ** -45:
            x = float(m + r.uniform(-6.0, 6.0) * sd)
        else:
            x = h("%016x" % (int(bits_of(float(m)), 16) + r.randint(-6, 6)))
        mode = (A - 1) / (A + B - 2)
        sm = mpmath.sqrt(mode * (1 - mode) / (A + B - 2))
        if 0 < x < 1 and abs((mp_arg(x) - mode) / sm) < 40:
            rows.append(((a, b, x), list(ibeta_quad(a, b, x))))
    return rows


def bits_of(x):
    """The 16 hexadecimal digits of the f64 `x`."""
    return "%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


def review2_basym():
    """The second review of the fixes: BASYM beside the mean of both parameters from 1e5 to 1e8,
    3 to 6 widths out (erfcx was taken at the rounded √f, whose rounding it carried into the result:
    up to 1.67 ulps), the review's three cases first; references by `ibeta_quad`."""
    r = ra(30)
    ts = [(21037838.962722816, 14714726.966045273, 0.58794218883753),
          (105715.2734074789, 35663.10436784449, 0.7542565124102395),
          (4475266.469623705, 183200.9468870443, 0.9609385975502966)]
    while len(ts) < 43:
        a, b = lu(r, 1e5, 1e8), lu(r, 1e5, 1e8)
        m = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        x = m + r.choice((-1, 1)) * r.uniform(3.0, 6.0) * sd
        if 0 < x < 1:
            ts.append((a, b, x))
    return [((a, b, x), list(ibeta_quad(a, b, x))) for a, b, x in ts]


emit("beta_inc_f64", "beta_inc(a, b, x) and beta_inc_c(a, b, x), f64", rows_of(triples(False, 2)), keep=0.55,
     extra=audit_beta_inc() + review_beta_inc() + review_both_huge() + review2_basym())
emit("beta_inc_f32", "beta_inc(a, b, x) and beta_inc_c(a, b, x), f32 arguments", rows_of(triples(True)[: n(320)]), keep=0.5)


def inverse_rows(ts, complement, single=False):
    """(a, b, p) with p = I_x(a, b) (or 1 − I) rounded, the exact inverse x and its condition
    number |p/(x dx/dp)|... stored as the second reference."""
    rows = []
    for a, b, x in ts:
        I, Ic = ibeta_pair(a, b, x)
        p = float(Ic if complement else I)
        if single:
            p = to_f32(p)
        if not (0 < p < 1):
            continue
        # Solve for the exact x of the rounded p by Newton on the side that is accurate.
        A, B, P = mpf(a), mpf(b), mpf(p)
        lb = mpmath.loggamma(A) + mpmath.loggamma(B) - mpmath.loggamma(A + B)
        xe = mpf(x)
        for _ in range(200):
            I2, Ic2 = ibeta_pair(A, B, xe)
            f = (Ic2 - P) if complement else (I2 - P)
            d = mpmath.exp((A - 1) * mpmath.log(xe) + (B - 1) * mpmath.log(1 - xe) - lb)
            step = f / d if not complement else -f / d
            xn = xe - step
            if xn <= 0:
                xn = xe / 2
            if xn >= 1:
                xn = (1 + xe) / 2
            if abs(xn - xe) <= abs(xe) * mpf(2) ** -240:
                xe = xn
                break
            xe = xn
        d = mpmath.exp((A - 1) * mpmath.log(xe) + (B - 1) * mpmath.log(1 - xe) - lb)
        kappa = abs(P / (xe * d))
        rows.append(((a, b, p), [xe, kappa]))
    return rows


def audit_binv(complement):
    """Roots in the subnormals: I_x(a, b) = x^a/(a B(a, b)) (1 + O(b x)) there, so its leading term
    gives the root of the rounded p (or of 1 − q, exact) to 1e-300, and kappa from the density."""
    r = ra(14 + complement)
    if complement:
        cases = [(0.015051674884659758, 0.68609527838892, 0.9999825508925646)]
    else:
        cases = [(0.6931097059489624, 1283.9036731128458, 7.138512485316517e-212),
                 (0.4146331898231363, 4734.756688589159, 1.4816897758239047e-127)]
    while len(cases) < 40:
        if complement:
            a, b = lu(r, 0.005, 0.05), lu(r, 0.5, 5.0)
        else:
            a, b = lu(r, 0.01, 5.0), lu(r, 0.5, 1e4)
        xt = mp_arg(lu(r, 5e-324, 2.2e-308))
        A, B = mp_arg(a), mp_arg(b)
        i = xt ** A / (A * mpmath.beta(A, B))
        t = float(1 - i) if complement else float(i)
        if 0 < t < (1 if complement else 0.5) and (not complement or t > 0.5):
            cases.append((a, b, t))
    rows = []
    for a, b, t in cases:
        A, B, T = mp_arg(a), mp_arg(b), mp_arg(t)
        i = 1 - T if complement else T
        lb = mpmath.log(mpmath.beta(A, B))
        x = mpmath.exp((mpmath.log(i * A) + lb) / A)
        d = mpmath.exp((A - 1) * mpmath.log(x) + (B - 1) * mpmath.log1p(-x) - lb)
        rows.append(((a, b, t), [x, abs(T / (x * d))]))
    return rows


inv_ts = triples(False, 2)
rng_ib.shuffle(inv_ts)
def review_binv(complement):
    """The review of the fixes (round 1): a tiny p beside a large a (the iterate was the larger of
    x and 1 − x, whose grid is coarser: up to 44 ulps), and one parameter from 5e24 on (wrong
    roots: the function was NaN there). Each case from an x, its t = I_x (or 1 − I_x) rounded,
    then the exact root of t by Newton's method at the precision of wide_prec."""
    r = ra(22 + complement)
    ts = []
    for _ in range(30):
        a, b = lu(r, 10.0, 1e3), lu(r, 0.5, 20.0)
        A, B = mp_arg(a), mp_arg(b)
        # x from the leading term I = x^a/(a B(a, b)) of a target p in [1e-300, 1e-10].
        lp = mpmath.log(lu(r, 1e-300, 1e-10))
        x = float(mpmath.exp((lp + mpmath.log(A) + mpmath.log(mpmath.beta(A, B))) / A))
        ts.append((a, b, x) if not complement else (b, a, 1.0 - x))
    for _ in range(20):
        a, b = lu(r, 0.5, 1e3), lu(r, 5e24, 1e299)
        u = max(a / 20, a + r.uniform(-4.0, 8.0) * math.sqrt(a))
        ts.append((a, b, u / b))
    rows = []
    for a, b, x in ts:
        if 0.0 < x < 1.0:
            with mpmath.workprec(wide_prec(a, b)):
                rows += inverse_rows([(a, b, x)], complement)
    # The review's own cases (a target t given): 27.8 and 43.8 ulps, wrong roots beside a huge b,
    # and the root 5e-324 of t = I at 5e-324 (a subnormal x beside b >= 8: 0 was returned).
    if complement:
        cases = [(8.0, 1e100, 0.3)]
    else:
        cases = [(95.24536450060963, 2.2025008747537638, 1.9073464690980808e-174),
                 (165.50418496543588, 2.6279058120876093, 2.5216873264520006e-274),
                 (50.0, 1e200, 0.9)]
        a, b = 0.7157280877724658, 11.269352670023864
        with mpmath.workprec(wide_prec(a, b)):
            cases.append((a, b, float(ibeta_pair(a, b, 5e-324)[0])))
    return rows + target_rows(cases, complement)


def target_rows(cases, complement):
    """Rows of `inverse_rows`' form for given targets (a, b, t): the root of I_x(a, b) = t (or of
    1 − I_x(a, b) = t) by bisection in ln x below 1/2 (in ln(1 − x) above), 160 steps, at the
    precision of wide_prec, and its condition number from the density."""
    rows = []
    for a, b, t in cases:
        with mpmath.workprec(wide_prec(a, b)):
            A, B, T = mp_arg(a), mp_arg(b), mp_arg(t)

            def g(x):
                I, Ic = ibeta_pair(A, B, x)
                return (T - Ic) if complement else (I - T)  # increasing in x

            low = g(mpf(0.5)) > 0
            lo, hi = (mpf(-1100), mpmath.log(mpf(0.5))) if low else (mpmath.log(mpf(0.5)), mpf(-200))
            for _ in range(160):
                mid = (lo + hi) / 2
                x = mpmath.exp(mid) if low else 1 - mpmath.exp(mid)
                if g(x) > 0:
                    hi = mid
                else:
                    lo = mid
            mid = (lo + hi) / 2
            x = mpmath.exp(mid) if low else 1 - mpmath.exp(mid)
            lb = mpmath.loggamma(A) + mpmath.loggamma(B) - mpmath.loggamma(A + B)
            d = mpmath.exp((A - 1) * mpmath.log(x) + (B - 1) * mpmath.log1p(-x) - lb)
            rows.append(((a, b, t), [+x, abs(T / (x * d))]))
    return rows


def normal_quantile_mp(p):
    """The standard normal quantile of 0 < p ≤ 1/2 (an mpf): Newton's method on ln Φ(z) − ln p
    from z = −√(−2 ln p), at the working precision (mpmath's erfinv gives −∞ for p of 1e-200)."""
    z = -mpmath.sqrt(-2 * mpmath.log(p))
    lp = mpmath.log(p)
    for _ in range(200):
        phi = mpmath.erfc(-z / mpmath.sqrt(2)) / 2
        dens = mpmath.exp(-z * z / 2) / mpmath.sqrt(2 * mpmath.pi)
        step = (mpmath.log(phi) - lp) * phi / dens
        z -= step
        if abs(step) <= mpf(2) ** (-mpmath.mp.prec + 16) * (1 + abs(z)):
            return z
    raise RuntimeError("normal_quantile_mp: no convergence at %s" % p)


def wide_root(a, b, t, complement):
    """The x with I_x(a, b) = t (or 1 − I_x(a, b) = t) for both parameters large, and its condition
    number: Newton's method in x on `ibeta_quad` from the normal approximation with Cornish and
    Fisher's skewness term, at the precision of wide_prec, until the step is below 2^-100 widths."""
    with mpmath.workprec(wide_prec(a, b)):
        A, B, T = mp_arg(a), mp_arg(b), mp_arg(t)
        c = A + B
        sd = mpmath.sqrt(A * B / (c ** 2 * (c + 1)))
        g1 = 2 * (B - A) * mpmath.sqrt(c + 1) / ((c + 2) * mpmath.sqrt(A * B))
        # z = Φ⁻¹(I) at the root, from t itself (1 − t of a tiny t would round to 1).
        z = normal_quantile_mp(T) if T <= 0.5 else -normal_quantile_mp(1 - T)
        if complement:
            z = -z
        x = A / c + sd * (z + g1 * (z * z - 1) / 6)
        lb = mpmath.loggamma(A) + mpmath.loggamma(B) - mpmath.loggamma(c)
        for _ in range(40):
            I, Ic = ibeta_quad(A, B, x)
            d = mpmath.exp((A - 1) * mpmath.log(x) + (B - 1) * mpmath.log1p(-x) - lb)
            step = (T - Ic) / d if complement else (I - T) / d
            x -= step
            if abs(step) <= sd * mpf(2) ** -100:
                break
        else:
            raise RuntimeError("wide_root: no convergence at %r" % ((a, b, t, complement),))
        d = mpmath.exp((A - 1) * mpmath.log(x) + (B - 1) * mpmath.log1p(-x) - lb)
        return +x, abs(T / (x * d))


def review2_binv(complement):
    """The second review of the fixes: both parameters from 1e20 on. The iteration accepted steps
    made from slopes without digits (the difference of two exponents of size |ln T|) far from the
    root, and Newton's step from where T is near 1 overshot into the far tail (wrong roots, up to
    1e15 ulps). The review's cases, then random ones (the larger parameter log-uniform in [1e20,
    1e200], the other up to 1e8 times smaller and at least 1e6, t uniform or log-uniform in [1e-300,
    1e-3]), roots by `wide_root`; then four point masses (both parameters from 2^600 on, or a + b
    overflowing), whose root is the mean to far below an ulp (the distribution is sub-Gaussian
    with variance 1/(4(a + b + 1))), so the mean is the reference."""
    r = ra(31 + complement)
    if complement:
        cases = [(2.5511891997651948e+45, 1.7038334124882498e+84, 0.551862181195602),
                 (2.745627768412921e+63, 2.8248648711537885e+98, 0.2727846452892402),
                 (5.111774859066366e+36, 1.263896121842748e+37, 0.946992785979806)]
    else:
        cases = [(1.9817856541964746e+56, 4.8992926582625025e+66, 0.4646068011980492),
                 (2.3317116251758114e+37, 4.624054220909504e+37, 0.10454860975003455),
                 (1.6591863844074735e+35, 1.1078011732795632e+35, 0.6591641169902301),
                 (6.154909085917088e+36, 5.3858084737813284e+35, 8.551300633084796e-204),
                 (3.3037381324109974e+34, 1.0678666444508326e+33, 0.9954398912722061),
                 (7.365393010816306e+63, 1.242941527279095e+68, 0.5812519701423424),
                 (3.6173451480613984e+27, 1.019538296662128e+21, 0.4579154680940464),
                 (2.6651344127691475e+35, 7.005693486623686e+32, 0.173333489918984),
                 (4.649641777878605e+25, 5.1237108340164646e+17, 0.976897880705228)]
    while len(cases) < (33 if complement else 39):
        big = lu(r, 1e20, 1e200)
        other = max(big * 10 ** r.uniform(-8.0, 0.0), 1e6)
        a, b = (big, other) if r.random() < 0.5 else (other, big)
        t = r.random() if r.random() < 0.7 else lu(r, 1e-300, 1e-3)
        if 0 < t < 1:
            cases.append((a, b, t))
    rows = []
    for a, b, t in cases:
        x, k = wide_root(a, b, t, complement)
        rows.append(((a, b, t), [x, k]))
    for a, b, t in [(4.2e180, 7.5e250, 0.3), (1e299, 3.1e298, 1e-200), (1.7e308, 1.7e308, 0.6),
                    (6e299, 5.5e181, 0.999)]:
        A, B = mp_arg(a), mp_arg(b)
        rows.append(((a, b, t), [A / (A + B), mpf(0)]))
    return rows


emit("beta_inc_inv_f64", "beta_inc_inv(a, b, p) = x, then the condition number |p/(x I'(x))|",
     inverse_rows(inv_ts[: n(330)], False), keep=0.7, extra=audit_binv(False) + review_binv(False) + review2_binv(False))
emit("beta_inc_c_inv_f64", "beta_inc_c_inv(a, b, q) = x, then the condition number |q/(x I'(x))|",
     inverse_rows(inv_ts[n(330): n(660)], True), keep=0.7, extra=audit_binv(True) + review_binv(True) + review2_binv(True))
inv32 = triples(True)
rng_ib.shuffle(inv32)
emit("beta_inc_inv_f32", "beta_inc_inv(a, b, p) = x, f32 arguments, then the condition number",
     inverse_rows(inv32[: n(150)], False, True))
