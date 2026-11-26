"""Generates `fd_tables.rs` beside this file: the fits of the polylogarithm of the orders 1/2, 3/2,
5/2 and 3 on −7.3·10²⁶ ≤ x < 0 (the Fermi–Dirac integrals). For each order, Chebyshev interpolation
at 45 digits by mpmath, cut where its coefficients fall below 10⁻²¹ of the value, in powers of the
variable: `R_s(μ) = −Γ(s + 1)·Li_s(−eᵘ)/μˢ` on nine pieces of `ln μ` past x = −2 (48 nodes),
`−Li_s(−eᵘ)` in μ on −2 ≤ x < −3/4 (40 nodes), `Li_s(x)/x` on −3/4 ≤ x < 0 (44 nodes, in x, then
re-expanded about −3/8); the coefficients kept to 40 digits between the steps.

`python3 gen_fd.py > fd_tables.rs` (mpmath 1.3; a few minutes) gives the file byte for byte.
"""
import sys

sys.dont_write_bytecode = True
import mpmath as mp

mp.mp.dps = 45

def fit_pieces(S):
    N = 48
    def R(s, mu):
        return mp.re(-mp.gamma(s + 1) * mp.polylog(s, -mp.exp(mu)) / mu**s)
    def tomono(cheb, d):
        T = [[mp.mpf(1)], [mp.mpf(0), mp.mpf(1)]]
        for k in range(2, d + 1):
            prev, prev2 = T[-1], T[-2]
            nxt = [mp.mpf(0)] * (k + 1)
            for m_, v in enumerate(prev): nxt[m_ + 1] += 2 * v
            for m_, v in enumerate(prev2): nxt[m_] -= v
            T.append(nxt)
        mono = [mp.mpf(0)] * (d + 1)
        for k in range(d + 1):
            for m_, v in enumerate(T[k]): mono[m_] += cheb[k] * v
        return mono
    out = {}
    for s in S:
        pieces = []
        for i in range(9):
            c, h = mp.mpf(-0.125) + mp.mpf(0.5) * i, mp.mpf(0.25)
            xs = [mp.cos(mp.pi * (k + mp.mpf(1)/2) / N) for k in range(N)]
            fs = [R(s, mp.exp(c + h * t)) for t in xs]
            cheb = [2 * mp.fsum(fs[k] * mp.cos(mp.pi * j * (k + mp.mpf(1)/2) / N) for k in range(N)) / N for j in range(N)]
            cheb[0] /= 2
            r0 = min(abs(f) for f in fs)
            d = N - 1
            while d > 0 and abs(cheb[d]) < mp.mpf(10)**-21 * r0:
                d -= 1
            mono = tomono(cheb, d)
            pieces.append({"c": float(c), "deg": d, "mono": [mp.nstr(v, 40) for v in mono]})
        out[str(float(s))] = pieces
    return out


def fit_near_m1(S):
    a, b = mp.log(mp.mpf(3) / 4) - mp.mpf('0.002'), mp.log(2) + mp.mpf('0.002')
    c, h = (a + b) / 2, (b - a) / 2
    N = 40
    out = {}
    for s in S:
        F = lambda mu: mp.re(-mp.polylog(s, -mp.exp(mu)))
        xs = [mp.cos(mp.pi * (k + mp.mpf(1)/2) / N) for k in range(N)]
        fs = [F(c + h * t) for t in xs]
        cheb = [2 * mp.fsum(fs[k] * mp.cos(mp.pi * j * (k + mp.mpf(1)/2) / N) for k in range(N)) / N for j in range(N)]
        cheb[0] /= 2
        r0 = min(abs(f) for f in fs)
        d = N - 1
        while abs(cheb[d]) < mp.mpf(10)**-21 * r0: d -= 1
        # chebyshev in t = (mu - c)/h -> polynomial in mu: evaluate via mp polynomial arithmetic
        # build T_k(t) as polys in mu
        lin = [-c / h, 1 / h]  # t = lin0 + lin1 mu
        def pmul(p, q):
            r = [mp.mpf(0)] * (len(p) + len(q) - 1)
            for i, x in enumerate(p):
                for j, y in enumerate(q): r[i + j] += x * y
            return r
        def padd(p, q, k=1):
            n = max(len(p), len(q)); r = [mp.mpf(0)] * n
            for i, x in enumerate(p): r[i] += x
            for i, x in enumerate(q): r[i] += k * x
            return r
        T = [[mp.mpf(1)], lin]
        for k in range(2, d + 1):
            T.append(padd(pmul([2 * lin[0], 2 * lin[1]], T[-1]), T[-2], -1))
        mono = [mp.mpf(0)] * (d + 1)
        for k in range(d + 1):
            for i, v in enumerate(T[k]): mono[i] += cheb[k] * v
        # check in f64-ish (mp) evaluation
        err = 0
        for j in range(401):
            mu = a + (b - a) * j / 400
            p = mp.mpf(0)
            for v in reversed(mono): p = p * mu + v
            err = max(err, abs(p / F(mu) - 1))
        out[str(float(s))] = [mp.nstr(v, 40) for v in mono]
    return out


def fit_neg(S):
    a, b = mp.mpf('-0.752'), mp.mpf(0)
    c, h = (a + b) / 2, (b - a) / 2
    N = 44
    out = {}
    def pmul(p, q):
        r = [mp.mpf(0)] * (len(p) + len(q) - 1)
        for i, x in enumerate(p):
            for j, y in enumerate(q): r[i + j] += x * y
        return r
    def padd(p, q, k=1):
        n = max(len(p), len(q)); r = [mp.mpf(0)] * n
        for i, x in enumerate(p): r[i] += x
        for i, x in enumerate(q): r[i] += k * x
        return r
    for s in S:
        G = lambda x: mp.re(mp.polylog(s, x) / x) if x != 0 else mp.mpf(1)
        xs = [mp.cos(mp.pi * (k + mp.mpf(1)/2) / N) for k in range(N)]
        fs = [G(c + h * t) for t in xs]
        cheb = [2 * mp.fsum(fs[k] * mp.cos(mp.pi * j * (k + mp.mpf(1)/2) / N) for k in range(N)) / N for j in range(N)]
        cheb[0] /= 2
        r0 = min(abs(f) for f in fs)
        d = N - 1
        while abs(cheb[d]) < mp.mpf(10)**-21 * r0: d -= 1
        lin = [-c / h, 1 / h]
        T = [[mp.mpf(1)], lin]
        for k in range(2, d + 1):
            T.append(padd(pmul([2 * lin[0], 2 * lin[1]], T[-1]), T[-2], -1))
        mono = [mp.mpf(0)] * (d + 1)
        for k in range(d + 1):
            for i, v in enumerate(T[k]): mono[i] += cheb[k] * v
        err = 0
        for j in range(401):
            x = a + (b - a) * j / 400
            p = mp.mpf(0)
            for v in reversed(mono): p = p * x + v
            err = max(err, abs(p / G(x) - 1))
        out[str(float(s))] = [mp.nstr(v, 40) for v in mono]
    return out



def dd(v):
    v = mp.mpf(v); hi = float(v); lo = float(v - hi); return hi, lo
def r(x):
    return repr(float(x)) if x != 0 else "0.0"
def row(m, width, ind):
    h0, l0 = dd(m[0]); h1, l1 = dd(m[1])
    rest = [float(mp.mpf(v)) for v in m[2:]]
    rest += [0.0] * (width - 4 - len(rest))
    return [f"{ind}{r(v)}," for v in [h0, l0, h1, l1] + rest]
def run(J):
    out = ["//! The fits of `R_s(μ) = −Γ(s + 1)·Li_s(−eᵘ)/μˢ` for `s = 1/2, 3/2, 5/2` on `ln 2 < μ ≤ 61.9`, by",
           "//! pieces of `v = ln μ`: nine of width 1/2 about `cₖ = −1/8 + k/2`, each a polynomial in",
           "//! `t = 4(v − cₖ)` on `[−1, 1]` (Chebyshev interpolation at 48 nodes of the values at 45",
           "//! digits by mpmath, cut where its coefficients fall below `10⁻²¹` of the value, then in",
           "//! powers of `t`): `[c₀ hi, c₀ lo, c₁ hi, c₁ lo, c₂, …, c₁₈]` (zeros past the degree).",
           "", "#![allow(clippy::excessive_precision, clippy::unreadable_literal)]", ""]
    names = {"0.5": "FD_HALF_0", "1.5": "FD_HALF_1", "2.5": "FD_HALF_2"}
    def pieces(name, doc, ps):
        out.append(doc)
        out.append(f"pub(super) const {name}: [[f64; 21]; 9] = [")
        for p in ps:
            out.append("    [")
            out.extend(row(p["mono"], 21, "        "))
            out.append("    ],")
        out.append("];")
        out.append("")
    for key in ["0.5", "1.5", "2.5"]:
        pieces(names[key], f"/// `s = {key}`.", J["fits9"][key])
    def rows3(name, doc, width, d):
        out.extend(doc)
        out.append(f"pub(super) const {name}: [[f64; {width}]; 3] = [")
        for key in ["0.5", "1.5", "2.5"]:
            out.append("    [")
            out.extend(row(d[key], width, "        "))
            out.append("    ],")
        out.append("];")
        out.append("")
    rows3("FD_NEAR_M1", ["/// `−Li_s(−eᵘ)` for `s = 1/2, 3/2, 5/2` on `ln(3/4) − 0.002 ≤ μ ≤ ln 2 + 0.002` (`−2 ≤ x < −3/4`):",
        "/// polynomials in `μ` (Chebyshev interpolation at 40 nodes of the values at 45 digits by",
        "/// mpmath, cut where its coefficients fall below `10⁻²¹` of the value, then in powers of `μ`;",
        "/// relative error at most `8·10⁻²²`), `[c₀ hi, c₀ lo, c₁ hi, c₁ lo, c₂, …, c₁₈]`."], 21, J["fit_m1"])
    rows3("FD_NEG", ["/// `Li_s(x)/x` for `s = 1/2, 3/2, 5/2` on `−0.752 ≤ x ≤ 0`: polynomials in `u = x + 3/8`",
        "/// (Chebyshev interpolation at 44 nodes of the values at 45 digits by mpmath, cut where its",
        "/// coefficients fall below `10⁻²¹` of the value, then in powers of `u`, whose terms fall as",
        "/// `0.27ᵏ` on the interval; relative error at most `5·10⁻²²`), `[c₀ hi, c₀ lo, c₁ hi, c₁ lo,",
        "/// c₂, …, c₂₄]`."], 27, J["fit_neg"])
    pieces("FD_HALF_3", "/// `s = 3`, the same fits as `FD_HALF_0` (`R₃(μ) = −6 Li₃(−eᵘ)/μ³`).", J["s3_fits9"]["3.0"])
    def one(name, doc, width, m):
        out.append(doc)
        out.append(f"pub(super) const {name}: [f64; {width}] = [")
        out.extend(row(m, width, "    "))
        out.append("];")
        out.append("")
    one("FD_NEAR_M1_3", "/// `s = 3` for `FD_NEAR_M1` (`−2 ≤ x < −3/4`, in `μ`).", 21, J["s3_fit_m1"]["3.0"])
    one("FD_NEG_3", "/// `s = 3` for `FD_NEG` (`−3/4 ≤ x < 0`, in `x + 3/8`).", 27, J["s3_fit_neg"]["3.0"])
    return "\n".join(out)

def shift(m, x0):
    """The polynomial Σ mᵢ xⁱ in powers of u = x − x0 (exact in mpmath)."""
    m = [mp.mpf(v) for v in m]
    n = len(m)
    return [mp.fsum(mp.binomial(i, k) * m[i] * mp.mpf(x0) ** (i - k) for i in range(k, n)) for k in range(n)]


if __name__ == "__main__":
    half = [mp.mpf(1) / 2, mp.mpf(3) / 2, mp.mpf(5) / 2]
    three = [mp.mpf(3)]
    J = {"fits9": fit_pieces(half), "s3_fits9": fit_pieces(three),
         "fit_m1": fit_near_m1(half), "s3_fit_m1": fit_near_m1(three)}
    neg, neg3 = fit_neg(half), fit_neg(three)
    J["fit_neg"] = {k: [mp.nstr(v, 40) for v in shift(m, mp.mpf(-3) / 8)] for k, m in neg.items()}
    J["s3_fit_neg"] = {k: [mp.nstr(v, 40) for v in shift(m, mp.mpf(-3) / 8)] for k, m in neg3.items()}
    sys.stdout.write(run(J))
