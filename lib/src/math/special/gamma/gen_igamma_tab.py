"""Generates igamma_tab.rs, the tables of the incomplete gamma functions (`igamma.rs`):

* `erfcx` in eight minimax pieces `Y + R(s)` of width 1/2 on [0, 4), then `t·erfcx(t)` in
  `u = 1/t²` on [4, 8) and beyond 8 (Remez, mpmath at 160 bits);
* Temme's coefficients `c_k(z)`, k = 0..12, of the uniform expansion of `Q(a, x)` for large `a`
  (Temme 1979): their power series in `z` derived anew by exact truncated series arithmetic
  (mpmath at 400 bits, Lagrange inversion of `z²/2 = λ − 1 − ln λ`), then fitted minimax on
  |z| ≤ 1/2;
* the Taylor coefficients of `expm1`.

Run from this directory: `python3 gen_igamma_tab.py`, then `rustfmt igamma_tab.rs`; the result is
the committed file byte for byte. The derived and fitted coefficients are cached (pickle) in
`tempfile.gettempdir()`; delete the cache to recompute them.
"""

import sys

sys.dont_write_bytecode = True  # no __pycache__ beside the sources
import os
import pickle
import tempfile

import mpmath
from mpmath import mp, mpf

from gen_remez import fit

CACHE = os.path.join(tempfile.gettempdir(), "nettrash_a_igamma_temme")
os.makedirs(CACHE, exist_ok=True)


def temme_series():
    """The power series in z of c_0..c_13 (lists of mpf strings), cached."""
    path = os.path.join(CACHE, "temme_coeffs.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    mp.prec = 400
    N = 70
    def mul(a, b):
        n = min(len(a), len(b)); return [sum(a[i] * b[k - i] for i in range(k + 1)) for k in range(n)]
    def inv(a):
        r = [1 / a[0]]
        for k in range(1, len(a)):
            r.append(-sum(a[i] * r[k - i] for i in range(1, k + 1)) / a[0])
        return r
    def log_s(a):  # a[0] = 1
        # (log a)' = a'/a
        da = [(i + 1) * a[i + 1] for i in range(len(a) - 1)]
        q = mul(da, inv(a[:-1]))
        return [mpf(0)] + [q[i] / (i + 1) for i in range(len(q))]
    def exp_s(a):  # a[0] = 0
        r = [mpf(1)] + [mpf(0)] * (len(a) - 1)
        da = [(i + 1) * a[i + 1] for i in range(len(a) - 1)]
        for k in range(1, len(a)):
            r[k] = sum(da[i] * r[k - 1 - i] for i in range(k)) / k
        return r
    # h(mu) = 2 (mu - ln(1+mu)) / mu^2 = sum_j 2 (-1)^j mu^j/(j+2)
    h = [2 * mpf(-1) ** j / (j + 2) for j in range(N + 2)]
    lh = log_s(h)
    # Lagrange inversion: [eta^n] mu = (1/n) [mu^(n-1)] h^(-n/2)
    mu = [mpf(0)]
    for n in range(1, N + 1):
        e = exp_s([-mpf(n) / 2 * c for c in lh[:n]])
        mu.append(e[n - 1] / n)
    # 1/mu = (1/eta) inv(mu/eta)
    m1 = inv(mu[1:])          # (eta/mu) series, length N
    c = [m1[i + 1] for i in range(len(m1) - 1)]   # c0 = 1/mu - 1/eta regular part
    cs = [c]
    gam = []
    for k in range(1, 14):
        prev = cs[-1]
        d = [(i + 1) * prev[i + 1] for i in range(len(prev) - 1)]   # c'_{k-1}
        # (1/eta) d: pole d[0]/eta, regular d[i+1] eta^i; (-1)^k g_k/mu = (-1)^k g_k (1/eta) m1
        gk = -d[0]   # (-1)^k gamma_k
        gam.append(gk)
        reg = [d[i + 1] + gk * m1[i + 1] for i in range(len(d) - 1)]
        cs.append(reg)
    res = [[str(v) for v in s] for s in cs]
    pickle.dump(res, open(path, "wb"))
    return res


def temme_fits():
    """Minimax fits of c_0..c_12 on |z| <= 1/2 (lists of floats), cached."""
    path = os.path.join(CACHE, "temme_fit.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    ser = temme_series()
    mp.prec = 160
    cs = [[mpf(v) for v in s] for s in ser]
    ZM = mpf("0.5")
    res = []
    for k in range(13):
        s = cs[k]
        f = lambda z, s=s: mpmath.polyval(list(reversed(s)), z)
        tol = mpf("1e-19") * mpf(20) ** k
        for n in range(2, 16):
            c, e = fit(f, -ZM, ZM, n, grid=400)
            if e < tol:
                break
        print(k, n, mpmath.nstr(e, 3), mpmath.nstr(tol, 3))
        res.append([float(v) for v in c])
    pickle.dump(res, open(path, "wb"))
    return res


mp.prec = 160
out = []
def lit(x):
    x = float(x)
    if x == 0: return "0.0"
    r = repr(x)
    if 'e' not in r and '.' not in r: r += ".0"
    return r
def arr(name, vals, doc, t="f64"):
    out.append(f"/// {doc}")
    out.append(f"pub(super) static {name}: [f64; {len(vals)}] = [")
    for v in vals: out.append(f"    {lit(v)},")
    out.append("];")

f = lambda t: mpmath.erfc(t) * mpmath.exp(t * t)
# pieces of width 1/2 on [0, 4): erfcx(c + s) = Y + R(s), |s| <= 1/4; R padded to 15 coefficients
def fitYR(func, lo, hi, n):
    c0 = (lo + hi) / 2
    Y = float(func(c0))
    basis = [(lambda k: (lambda s: s ** k))(k) for k in range(n + 1)]
    c, e = fit(lambda s: func(c0 + s) - Y, lo - c0, hi - c0, w=lambda s: 1 / func(c0 + s), basis=basis, grid=500)
    return Y, [float(x) for x in c], e
ys, pieces = [], []
for k in range(8):
    a = mpf(k) / 2
    Y, c, e = fitYR(f, a, a + mpf(1) / 2, 14 if k < 2 else 13)
    print("erfcx piece", k, mpmath.nstr(e, 3))
    ys.append(Y)
    pieces.append(c + [0.0] * (15 - len(c)))
arr("ERFCX_Y", ys, "`erfcx(k/2 + 1/4)` rounded, `k = 0..8`.")
out.append("/// `erfcx(k/2 + 1/4 + s) − Y_k`, `|s| ≤ 1/4` (minimax, relative to `erfcx`; zero padded).")
out.append("pub(super) static ERFCX_R: [[f64; 15]; 8] = [")
for cf in pieces:
    out.append("    [" + ", ".join(lit(v) for v in cf) + "],")
out.append("];")
g = lambda u: (1 / mpmath.sqrt(u)) * f(1 / mpmath.sqrt(u)) if u > 0 else 1 / mpmath.sqrt(mpmath.pi)
Y, c, e = fitYR(g, mpf(1) / 64, mpf(1) / 16, 12)
print("erfcx u mid", mpmath.nstr(e, 3))
out.append(f"pub(super) const ERFCX_U_MID_Y: f64 = {lit(Y)};")
arr("ERFCX_U_MID", c, "`t·erfcx(t) − Y` at `u = 1/t² = 5/128 + s`, `|s| ≤ 3/128` (`4 ≤ t ≤ 8`; minimax, relative).")
Y = float(g(mpf(0)))
basis = [(lambda k: (lambda u: u ** k))(k) for k in range(10)]
c, e = fit(lambda u: g(u) - Y, 0, mpf(1) / 64, w=lambda u: 1 / g(u), basis=basis, grid=500)
print("erfcx u tail", mpmath.nstr(e, 3))
out.append(f"pub(super) const ERFCX_U_TAIL_Y: f64 = {lit(Y)};")
arr("ERFCX_U_TAIL", [float(x) for x in c], "`t·erfcx(t) − 1/√π` in `u = 1/t²` on `t ≥ 8` (minimax, relative).")

fits = temme_fits()
mp.prec = 160
for k, c in enumerate(fits):
    arr(f"TEMME_C{k}", c, f"Temme's `c_{k}(z)` of `Q(a, x)` on `|z| ≤ 1/2` (minimax, absolute error below `10⁻¹⁹·20^{k}`).")
arr("EXPM1_TAIL", [1 / mpmath.factorial(k) for k in range(3, 17)], "`(eˣ − 1 − x − x²/2)/x³` (Taylor).")
hdr = ["//! Tables of the incomplete gamma functions, generated by `gen_igamma_tab.py` (beside this file).", "",
       "#![allow(clippy::excessive_precision, clippy::unreadable_literal)]", ""]
open("igamma_tab.rs", "w").write("\n".join(hdr + out) + "\n")
print("ok")
