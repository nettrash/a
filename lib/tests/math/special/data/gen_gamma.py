"""The reference data of the gamma area (mpmath, 256 bits): Γ, ln|Γ|, the complex Γ and ln Γ,
ψ and ψ⁽ⁿ⁾, B and ln B, P and Q with their inverses, I and its complement with their inverses,
and factorials, binomials and Pochhammer's symbol of floats.

Run from this directory: `python3 gen_gamma.py` (the repository's files), or
`python3 gen_gamma.py --big K --out DIR` for K times as many random cases in DIR (the area's
larger harness, not in the repository).
"""

import math
import random
import sys

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


def emit(name, header, rows, keep=1.0):
    """Writes `gamma_<name>.txt`; `keep` < 1 thins the repository's file evenly (not the larger
    harness's), keeping the last ten rows (the special cases)."""
    if not wanted(name):
        return
    if BIG == 1 and keep < 1.0:
        k = len(rows)
        rows = [r for i, r in enumerate(rows) if int((i + 1) * keep) > int(i * keep) or i >= k - 10]
    write(f"{OUT}/gamma_{name}.txt", header, rows)


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

xs = gamma_args(False)
emit("gamma_f64", "gamma(x), f64", [((x,), [mpmath.gamma(mp_arg(x))]) for x in xs], keep=0.75)
xs = gamma_args(True)
emit("gamma_f32", "gamma(x), f32 arguments", [((x,), [mpmath.gamma(mp_arg(x))]) for x in xs], keep=0.5)
xs = lgamma_args(False)
emit("lgamma_f64", "ln|gamma(x)|, f64 (away from the zeros on the negative axis)",
     [((x,), [lgamma_ref(x)]) for x in xs if not is_near_neg_zero(x, ZS)])
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


zs = cplx_args(150, 1e6)
emit("lgamma_c64", "principal ln gamma(z), Complex<f64>", [((a, b), [mpmath.loggamma(mpmath.mpc(a, b))]) for a, b in zs], keep=0.8)
zs = [(a, b) for a, b in cplx_args(150, 400.0) if abs(a) < 180]
zs += [(rng.uniform(-180, 175), rng.uniform(-5, 5)) for _ in range(n(40))]
rows = [((a, b), [mpmath.gamma(mpmath.mpc(a, b))]) for a, b in zs if b != 0]
emit("gamma_c64", "gamma(z), Complex<f64> (|gamma(z)| above 1e-300)", [r for r in rows if abs(r[1][0]) > 1e-300], keep=0.8)


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


emit("digamma_f64", "digamma(x), f64 (away from the zeros on the negative axis)",
     [((x,), [mpmath.digamma(mp_arg(x))]) for x in digamma_args(False)])
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


emit("polygamma_f64", "polygamma(n, x), f64, x > 0 (args: n as a float, then x)", polygamma_rows(False))


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


emit("beta_f64", "beta(a, b) and ln|beta(a, b)|, f64, a, b > 0", beta_rows(False))
emit("beta_neg_f64", "beta(a, b) and ln|beta(a, b)|, f64, a or b < 0", beta_rows(True))
rows = []
for a, b in zip(log_uniform(rng, 1e-3, 1e3, n(80), True), log_uniform(rng, 1e-3, 1e3, n(80), True)):
    v = mpmath.beta(mp_arg(a), mp_arg(b))
    rows.append(((a, b), [v, mpmath.log(v)]))
emit("beta_f32", "beta(a, b) and ln beta(a, b), f32 arguments", rows)

xs = uniform(rng, -0.999, 8.0, n(60)) + uniform(rng, 8.0, 170.6, n(60)) + signed(log_uniform(rng, 1e-300, 1e-3, n(20)))
xs += [x for x in uniform(rng, -190.0, -1.0, n(60)) if x != int(x)]
xs += [float(k) for k in range(0, 171, 7)] + [k + 0.5 for k in range(-10, 10)]
emit("factorial_f64", "x! = gamma(x + 1), f64", [((x,), [mpmath.gamma(mp_arg(x) + 1)]) for x in xs])


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


emit("binomial_f64", "binomial(n, k), f64", binomial_rows())


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


emit("pochhammer_f64", "pochhammer(x, n) = gamma(x + n)/gamma(x), f64", pochhammer_rows())


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



ps = pq_args(False)
emit("gamma_pq_f64", "P(a, x) and Q(a, x), f64", [((a, x), list(_pq(a, x))) for a, x in ps], keep=0.75)
ps = pq_args(True)
emit("gamma_pq_f32", "P(a, x) and Q(a, x), f32 arguments", [((a, x), list(_pq(a, x))) for a, x in ps], keep=0.5)
for name, single, upper, head in [
    ("gamma_p_inv_f64", False, False, "the x with P(a, x) = p, f64"),
    ("gamma_q_inv_f64", False, True, "the x with Q(a, x) = q, f64"),
    ("gamma_p_inv_f32", True, False, "the x with P(a, x) = p, f32 arguments"),
    ("gamma_q_inv_f32", True, True, "the x with Q(a, x) = q, f32 arguments"),
]:
    # The inverses take minutes: only when wanted.
    if wanted(name):
        emit(name, head, inv_rows(single, upper))


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


emit("beta_inc_f64", "beta_inc(a, b, x) and beta_inc_c(a, b, x), f64", rows_of(triples(False, 2)), keep=0.55)
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


inv_ts = triples(False, 2)
rng_ib.shuffle(inv_ts)
emit("beta_inc_inv_f64", "beta_inc_inv(a, b, p) = x, then the condition number |p/(x I'(x))|",
     inverse_rows(inv_ts[: n(330)], False), keep=0.7)
emit("beta_inc_c_inv_f64", "beta_inc_c_inv(a, b, q) = x, then the condition number |q/(x I'(x))|",
     inverse_rows(inv_ts[n(330): n(660)], True), keep=0.7)
inv32 = triples(True)
rng_ib.shuffle(inv32)
emit("beta_inc_inv_f32", "beta_inc_inv(a, b, p) = x, f32 arguments, then the condition number",
     inverse_rows(inv32[: n(150)], False, True))
