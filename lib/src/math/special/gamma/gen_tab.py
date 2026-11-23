"""Generates `tab.rs` beside this file: the tables and minimax coefficients of the gamma area.

`python3 gen_tab.py` (mpmath 1.3), then `rustfmt --edition 2024 tab.rs` gives the file byte for
byte. The minimax fits are by the Remez exchange of `gen_remez.py`.
"""
import sys

sys.dont_write_bytecode = True  # no __pycache__ beside the sources
import mpmath, struct, math
from mpmath import mp, mpf
import os
from gen_remez import fit, check

TAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tab.rs")
mp.prec = 200

out = []
def f64s(v): return float(v)
def lit(x):
    x = float(x)
    if x == 0: return "0.0"
    r = repr(x)
    if 'e' not in r and '.' not in r: r += ".0"
    return r
def arr(name, vals, doc):
    out.append(f"/// {doc}")
    out.append(f"pub(super) static {name}: [f64; {len(vals)}] = [")
    for v in vals: out.append(f"    {lit(v)},")
    out.append("];")
def dd(v):
    hi = float(v); lo = float(v - mpf(hi)); return hi, lo

# ---- log table: z in [OFF, 2 OFF), 64 subintervals indexed by the top 6 mantissa bits of (ix - OFF)
OFF = 0x3fe6955500000000
N = 64
inv, lh, ll = [], [], []
for i in range(N):
    # z range for index i: bits OFF + i<<46 .. OFF + (i+1)<<46, normalised to exponent of OFF's or next
    def zof(b):
        tmp = b - OFF
        k = tmp >> 52
        iz = b - (tmp & (0xfff << 52))
        return struct.unpack('<d', struct.pack('<Q', iz))[0]
    lo_b = OFF + (i << 46); hi_b = OFF + ((i + 1) << 46) - 1
    z0, z1 = zof(lo_b), zof(hi_b)
    zm = (mpf(z0) + mpf(z1)) / 2
    if z0 <= 1.0 <= z1 or abs(zm - 1) < mpf(1)/256:
        c = mpf(1)
    else:
        c0 = 1 / zm
        e = mpmath.floor(mpmath.log(c0, 2))
        best = None
        for cand in [mpmath.floor(c0 * 2 ** (6 - e)) / 2 ** (6 - e), mpmath.ceil(c0 * 2 ** (6 - e)) / 2 ** (6 - e)]:
            m = max(abs(mpf(z0) * cand - 1), abs(mpf(z1) * cand - 1))
            if best is None or m < best[0]: best = (m, cand)
        c = best[1]
        # exactness of fma(z, c, -1) on the subinterval: z c - 1 must fit in 53 bits
        import random
        rr = random.Random(i)
        for zz in [z0, z1] + [z0 + (z1 - z0) * rr.random() for _ in range(2000)]:
            v = mpf(zz) * c - 1
            assert mpf(float(v)) == v, (i, zz, c)
    inv.append(float(c))
    h, l = dd(-mpmath.log(c)); lh.append(h); ll.append(l)
    r0 = mpf(z0) * c - 1; r1 = mpf(z1) * c - 1
    assert max(abs(r0), abs(r1)) < 0.0116, (i, z0, z1, r0, r1)
arr("LOG_INVC", inv, "`1/c` of the subintervals of the logarithm (7 significant bits).")
arr("LOG_LOGC_HI", lh, "`ln c`, high parts.")
arr("LOG_LOGC_LO", ll, "`ln c`, low parts.")
# log1p tail: log1p(r) = r - r^2/2 + r^3 P(r), |r| < 0.0081
P = [mpf((-1) ** j) / (j + 3) for j in range(8)]
arr("LOG1P_TAIL", [float(p) for p in P], "`(ln(1 + r) − r + r²/2)/r³` (Taylor).")
# ln 2 split: hi with 42 bits
ln2 = mpmath.log(2)
ln2hi = float(mpmath.floor(ln2 * 2 ** 42) / 2 ** 42); ln2lo = float(ln2 - ln2hi); ln2lo2 = float(ln2 - ln2hi - mpf(ln2lo))
out.append(f"pub(super) const LN2_HI: f64 = {lit(ln2hi)};\npub(super) const LN2_LO: f64 = {lit(ln2lo)};")
# ---- exp table 2^(j/64)
th, tl = [], []
for j in range(64):
    h, l = dd(mpf(2) ** (mpf(j) / 64)); th.append(h); tl.append(l)
arr("EXP_T_HI", th, "`2^(j/64)`, high parts.")
arr("EXP_T_LO", tl, "`2^(j/64)`, low parts.")
c = ln2 / 64
chi = float(mpmath.floor(c * 2 ** 42) / 2 ** 42)
clo = float(c - chi)
out.append(f"pub(super) const EXP_C_HI: f64 = {lit(chi)};\npub(super) const EXP_C_LO: f64 = {lit(clo)};\npub(super) const EXP_INV_C: f64 = {lit(64 / ln2)};")
arr("EXP_TAIL", [float(1 / mpmath.factorial(k)) for k in range(3, 8)], "`(e^r − 1 − r − r²/2)/r³` (Taylor).")

# ---- sin(pi r), cos(pi r) on |r| <= 1/4
pi = mpmath.pi
P3 = float(pi ** 3 / 6)
def fs(u):
    r = mpmath.sqrt(u); return mpmath.sin(pi * r) / r - pi + pi ** 3 / 6 * u
basis = [(lambda k: (lambda u: u ** k))(k) for k in range(1, 8)]
cs, e = fit(lambda u: fs(u) if u > 0 else fs(mpf('1e-60')), mpf('1e-60'), mpf(1) / 16, w=lambda u: mpmath.sqrt(u) / mpmath.sin(pi * mpmath.sqrt(u)), basis=basis, grid=600)
print("sin", mpmath.nstr(e, 4))
arr("SINPI", [float(x) for x in cs], "`(sin(πr)/r − π + π³r²/6)/r²` in `r²` on `|r| ≤ 1/4` (minimax, relative to `sin(πr)`; the first coefficient takes up the rounding of `π³/6`).")
h, l = dd(pi ** 3 / 6); out.append(f"pub(super) const PI3_6_HI: f64 = {lit(h)};\npub(super) const PI3_6_LO: f64 = {lit(l)};")
def fc(u):
    if u == 0: return pi ** 4 / 24
    r = mpmath.sqrt(u); return (mpmath.cos(pi * r) - 1 + pi ** 2 * u / 2) / u ** 2
cc, e = fit(fc, 0, mpf(1) / 16, 6, w=lambda u: u ** 2 / mpmath.cos(pi * mpmath.sqrt(u)) + mpf('1e-30'), grid=600)
print("cos", mpmath.nstr(e, 4))
arr("COSPI", [float(x) for x in cc], "`(cos(πr) − 1 + π²r²/2)/r⁴` in `r²` on `|r| ≤ 1/4` (minimax).")
h, l = dd(pi); out.append(f"pub(super) const PI_HI: f64 = {lit(h)};\npub(super) const PI_LO: f64 = {lit(l)};")
h, l = dd(pi ** 2 / 2); out.append(f"pub(super) const PI2_2_HI: f64 = {lit(h)};\npub(super) const PI2_2_LO: f64 = {lit(l)};")

# ---- 1/Gamma(1+t) = 1 + g t + t^2 S(t), |t| <= 1/2
g = mpmath.euler
def D(t): return mpmath.rgamma(1 + t)
s2 = mpmath.taylor(mpmath.rgamma, 1, 3)
C2 = float(s2[2])
fR = lambda t: D(t) - 1 - g * t - mpf(C2) * t ** 2
basis = [(lambda k: (lambda t: t ** k))(k) for k in range(2, 17)]
cr, e = fit(fR, -0.5, 0.5, w=lambda t: 1 / D(t), basis=basis, grid=1500)
print("rgamma", mpmath.nstr(e, 4))
out.append(f"pub(super) const RGAMMA_C2: f64 = {lit(C2)};")
arr("RGAMMA", [float(x) for x in cr], "`(1/Γ(1 + t) − 1 − γt − c₂t²)/t²` on `|t| ≤ 1/2` (minimax, relative to `1/Γ(1 + t)`).")
h, l = dd(g); out.append(f"pub(super) const EULER_HI: f64 = {lit(h)};\npub(super) const EULER_LO: f64 = {lit(l)};")

# ---- lgamma(2 + t) = t (1 + t) F2(t), pieces centred at -1/4, 1/4, 3/4 of width 1/2
def F2(t):
    w = 2 + t
    if abs(t) < mpf('1e-40'): return mpmath.psi(0, 2)
    return mpmath.loggamma(w) / ((w - 1) * (w - 2))
for name, tc, n in [("LGAM_A", -0.25, 14), ("LGAM_B", 0.25, 13), ("LGAM_C", 0.75, 12)]:
    Y = float(F2(mpf(tc)))
    f = lambda s: F2(mpf(tc) + s) - Y
    c2, e = fit(f, -0.25, 0.25, n, w=lambda s: 1 / F2(mpf(tc) + s), grid=1000)
    print(name, mpmath.nstr(e, 4))
    arr(name, [Y] + [float(x) for x in c2], f"`ln Γ(2 + t)/(t(1 + t))` at `t = {tc} + s`, `|s| ≤ 1/4`: `Y` then `R(s)` (minimax, relative).")

# ---- Stirling: x S(x) in u = 1/x^2 on x >= 8
def xS(u):
    if u == 0: return mpf(1) / 12
    x = 1 / mpmath.sqrt(u)
    return x * (mpmath.loggamma(x) - ((x - mpf(1) / 2) * mpmath.log(x) - x + mpmath.log(2 * pi) / 2))
cst, e = fit(xS, 0, mpf(1) / 64, 7, grid=800)
print("stirling", mpmath.nstr(e, 4))
c0h, c0l = dd(cst[0])
arr("STIRLING", [c0h] + [float(x) for x in cst[1:]], "`x·(ln Γ(x) − (x − 1/2) ln x + x − ln √(2π))` in `u = 1/x²` on `x ≥ 8` (minimax, absolute).")
out.append(f"pub(super) const STIRLING_C0_LO: f64 = {lit(float(cst[0] - mpf(c0h)))};")
h, l = dd(mpmath.log(2 * pi) / 2); out.append(f"pub(super) const LN_SQRT_2PI_HI: f64 = {lit(h)};\npub(super) const LN_SQRT_2PI_LO: f64 = {lit(l)};")
h, l = dd(mpmath.log(pi)); out.append(f"pub(super) const LN_PI_HI: f64 = {lit(h)};\npub(super) const LN_PI_LO: f64 = {lit(l)};")

hdr = ["//! Tables and minimax coefficients of the gamma area, generated by `gen_tab.py` (beside this",
       "//! file; mpmath at 200 bits, the minimax fits by the Remez exchange of `gen_remez.py`).", "",
       "#![allow(clippy::excessive_precision, clippy::unreadable_literal, clippy::approx_constant)]", ""]
open(TAB, "w").write("\n".join(hdr + out) + "\n")
print("ok", f"OFF={OFF:#x}")

# ---- appended: complex ln Γ (Taylor at 1 and 2, Stirling's Bernoulli terms), ψ, ψ⁽ⁿ⁾
extra = []
def arr2(name, vals, doc):
    extra.append(f"/// {doc}")
    extra.append(f"pub(super) static {name}: [f64; {len(vals)}] = [")
    for v in vals: extra.append(f"    {lit(v)},")
    extra.append("];")
# ln Γ(1 + w) = Σ_{k≥1} c_k w^k: c_1 = −γ, c_k = (−1)^k ζ(k)/k
arr2("LGAMMA1_TAYLOR", [float(-mpmath.euler)] + [float((-1) ** k * mpmath.zeta(k) / k) for k in range(2, 28)], "`ln Γ(1 + w) = Σ c_k w^k`, `k = 1, 2, …` (Taylor).")
arr2("LGAMMA2_TAYLOR", [float(1 - mpmath.euler)] + [float((-1) ** k * (mpmath.zeta(k) - 1) / k) for k in range(2, 22)], "`ln Γ(2 + w) = Σ c_k w^k`, `k = 1, 2, …` (Taylor).")
arr2("STIRLING_B", [float(mpmath.bernoulli(2 * k) / (2 * k * (2 * k - 1))) for k in range(1, 13)], "`B₂ₖ/(2k(2k − 1))`, `k = 1, 2, …`: Stirling's series of `ln Γ`.")
arr2("BERN_FACT", [float(mpmath.bernoulli(2 * k) / mpmath.factorial(2 * k)) for k in range(1, 41)], "`B₂ₖ/(2k)!`, `k = 1, 2, …`: the Euler–Maclaurin tail of the Hurwitz zeta function.")
# ψ(x) − ln x + 1/(2x) = −Σ B_2k/(2k x^2k): minimax in u = 1/x² on x ≥ 8 of −(that)/u
def psitail(u):
    if u == 0: return mpf(1) / 12
    x = 1 / mpmath.sqrt(u)
    return -(mpmath.digamma(x) - mpmath.log(x) + 1 / (2 * x)) / u
cpt, e = fit(psitail, 0, mpf(1) / 64, 7, grid=800)
print("psi tail", mpmath.nstr(e, 4))
arr2("PSI_TAIL", [float(c) for c in cpt], "`(ln x − 1/(2x) − ψ(x))·x²` in `u = 1/x²` on `x ≥ 8` (minimax, absolute).")
x0 = mpmath.findroot(mpmath.digamma, 1.46)
h, l = dd(x0); extra.append(f"pub(super) const PSI_ZERO_HI: f64 = {lit(h)};\npub(super) const PSI_ZERO_LO: f64 = {lit(l)};")
# ψ(x)/(x − x0) on [1, 2] in two pieces (s = x − 1.25, x − 1.75), and ψ(2 + t) on t in [0, 1) in two pieces
def psiq(x):
    if abs(x - x0) < mpf('1e-40'): return mpmath.psi(1, x0)
    return mpmath.digamma(x) / (x - x0)
rows = []
for c, n in [(1.125, 13), (1.375, 12), (1.625, 12), (1.875, 11)]:
    Y = float(psiq(mpf(c)))
    f = lambda s: psiq(mpf(c) + s) - Y
    cc2, e = fit(f, -0.125, 0.125, n, w=lambda s: 1 / psiq(mpf(c) + s), grid=1000)
    print("PSI_Q", c, mpmath.nstr(e, 4))
    rows.append([Y] + [float(v) for v in cc2] + [0.0] * (15 - n))
extra.append("/// `ψ(x)/(x − x₀)` at `x = c + s` on the quarters of `[1, 2]` (`c` the centre, `|s| ≤ 1/8`):")
extra.append("/// `Y` then `R(s)`, minimax relative, padded with zeros to 16 coefficients.")
extra.append(f"pub(super) static PSI_Q: [[f64; 17]; 4] = [")
for r in rows:
    extra.append("    [" + ", ".join(lit(v) for v in r) + "],")
extra.append("];")
open(TAB, "a").write("\n".join(extra) + "\n")
print("extra ok")

# ---- appended: trigamma
extra2 = []
def arr3(name, vals, doc):
    extra2.append(f"/// {doc}")
    extra2.append(f"pub(super) static {name}: [f64; {len(vals)}] = [")
    for v in vals: extra2.append(f"    {lit(v)},")
    extra2.append("];")
# ψ′(x) = 1/x + 1/(2x²) + T(u)/x³, u = 1/x², x ≥ 10
def trit(u):
    if u == 0: return mpf(1) / 6
    x = 1 / mpmath.sqrt(u)
    return (mpmath.psi(1, x) - 1 / x - 1 / (2 * x * x)) * x ** 3
ct, e = fit(trit, 0, mpf(1) / 100, 6, grid=800)
print("trigamma tail", mpmath.nstr(e, 4))
arr3("TRIG_TAIL", [float(c) for c in ct], "`(ψ′(x) − 1/x − 1/(2x²))·x³` in `u = 1/x²` on `x ≥ 10` (minimax, absolute).")
pieces = [(2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)] + [(float(k), float(k + 1)) for k in range(4, 10)]
rows = []
for lo, hi in pieces:
    c = (lo + hi) / 2
    h = (hi - lo) / 2
    Y = float(mpmath.psi(1, mpf(c)))
    f = lambda s: mpmath.psi(1, mpf(c) + s) - Y
    best = None
    for n in range(8, 16):
        cc, e = fit(f, -h, h, n, w=lambda s: 1 / mpmath.psi(1, mpf(c) + s), grid=600)
        if e < 2e-18:
            best = (n, cc, e); break
    n, cc, e = best
    print("trig piece", lo, hi, n, mpmath.nstr(e, 3))
    rows.append([Y] + [float(v) for v in cc] + [0.0] * (15 - n))
extra2.append("/// `ψ′(c + s)` on ten pieces of `[2, 10)` (halves below 4, units above; `c` the centre):")
extra2.append("/// `Y` then `R(s)`, minimax relative, padded with zeros to 16 coefficients.")
extra2.append(f"pub(super) static TRIG_PIECES: [[f64; 17]; {len(rows)}] = [")
for r in rows:
    extra2.append("    [" + ", ".join(lit(v) for v in r) + "],")
extra2.append("];")
open(TAB, "a").write("\n".join(extra2) + "\n")
print("trig ok")

# ---- appended: n! for n = 0..170 as double-doubles
extra3 = []
hs, ls = [], []
for k in range(171):
    v = mpmath.factorial(k)
    h = float(v); hs.append(h); ls.append(float(v - mpf(h)))
extra3.append("/// `n!` for `n = 0, 1, …, 170`, high parts (correctly rounded).")
extra3.append("pub(super) static FACT_HI: [f64; 171] = [" + ", ".join(lit(v) for v in hs) + "];")
extra3.append("/// `n!` for `n = 0, 1, …, 170`, low parts.")
extra3.append("pub(super) static FACT_LO: [f64; 171] = [" + ", ".join(lit(v) for v in ls) + "];")
open(TAB, "a").write("\n".join(extra3) + "\n")
print("fact ok")


# ---- the speed pass: ψ on [2, 8), ln Γ on [0.5, 1.5) and [3, 8)
extra4 = []


def pieces_fit(name, doc, f, pieces, w=None):
    rows = []
    for c, h in pieces:
        Y = float(f(mpf(c)))
        g = lambda s: f(mpf(c) + s) - Y
        ww = w if w else (lambda s: 1 / f(mpf(c) + s))
        best = None
        for n in range(8, 16):
            cc, e = fit(g, -h, h, n, w=lambda s: ww(mpf(c) + s) if w else 1 / f(mpf(c) + s), grid=600)
            if e < 2e-18:
                best = (n, cc, e)
                break
        assert best, (name, c)
        n, cc, e = best
        # The code takes c₁s without error: refit the rest with c₁ fixed to its double.
        c1 = float(cc[1])
        g2 = lambda s: f(mpf(c) + s) - Y - mpf(c1) * s
        while True:
            basis = [lambda s: mpf(1)] + [(lambda k: (lambda s: s ** k))(k) for k in range(2, n + 1)]
            cc2, e = fit(g2, -h, h, w=lambda s: ww(mpf(c) + s) if w else 1 / f(mpf(c) + s), basis=basis, grid=600)
            if e < 2.5e-18 or n == 15:
                break
            n += 1
        cc = [cc2[0], mpf(c1)] + list(cc2[1:])
        print(name, c, n, mpmath.nstr(e, 3))
        rows.append([Y] + [float(v) for v in cc] + [0.0] * (15 - n))
    extra4.append(doc)
    extra4.append(f"pub(super) static {name}: [[f64; 17]; {len(rows)}] = [")
    for r in rows:
        extra4.append("    [" + ", ".join(lit(v) for v in r) + "],")
    extra4.append("];")


pieces_fit("PSI_PIECES",
           "/// `ψ(c + s)` on eight pieces of `[2, 8)` (halves below 4, units above; `c` the centre):\n"
           "/// `Y` then `R(s)`, minimax relative, padded with zeros to 16 coefficients.",
           lambda x: mpmath.digamma(x),
           [(2.25, 0.25), (2.75, 0.25), (3.25, 0.25), (3.75, 0.25)] + [(k + 0.5, 0.5) for k in range(4, 8)])


def H(t):
    if abs(t) < mpf("1e-40"):
        return -mpmath.euler
    return mpmath.loggamma(1 + t) / t


pieces_fit("LGAM_H",
           "/// `ln Γ(1 + t)/t` on five pieces of `t ∈ [−1/2, 1/2)` (`c` the centre): `Y` then `R(s)`,\n"
           "/// minimax relative, padded with zeros to 16 coefficients.",
           H, [(-0.4375, 0.0625), (-0.3125, 0.0625), (-0.125, 0.125), (0.125, 0.125), (0.375, 0.125)])
pieces_fit("LGAM_MID",
           "/// `ln Γ(c + s)` on the units of `[3, 8)` (`c` the centre): `Y` then `R(s)`, minimax\n"
           "/// relative, padded with zeros to 16 coefficients.",
           lambda x: mpmath.loggamma(x), [(k + 0.5, 0.5) for k in range(3, 8)])
open(TAB, "a").write("\n".join(extra4) + "\n")
print("speed pass ok")

# ---- appended (November 2026): the accurate argument and logarithm of the complex logarithm
# (cplx.rs, ddm.rs ln_acc_of_dd): Stirling's series multiplies their errors by |z|.
extra5 = []
def arr5(name, vals, doc):
    extra5.append(f"/// {doc}")
    extra5.append(f"pub(super) static {name}: [f64; {len(vals)}] = [")
    for v in vals: extra5.append(f"    {lit(v)},")
    extra5.append("];")
def dd5(name, v, doc):
    h, l = dd(v)
    extra5.append(f"/// {doc}")
    extra5.append(f"pub(super) const {name}_HI: f64 = {lit(h)};\npub(super) const {name}_LO: f64 = {lit(l)};")
ah, al = [], []
for j in range(65):
    v = mpmath.atan(mpf(j) / 64) / pi
    h, l = dd(v)
    # (checked: tan(π(h + l)) is j/64 to 2⁻¹⁰⁴ relative, at 400 bits)
    with mpmath.workprec(400):
        t = mpmath.tan(pi * (mpf(h) + mpf(l)))
        assert j == 0 or abs(t * 64 / j - 1) < mpf(2) ** -104, (j, t)
    ah.append(h); al.append(l)
arr5("ATAN_J64_HI", ah, "`atan(j/64)/π`, `j = 0, 1, …, 64`, high parts.")
arr5("ATAN_J64_LO", al, "`atan(j/64)/π`, low parts.")
arr5("ATAN_TAIL", [float(mpf((-1) ** (k + 1)) / (2 * k + 7)) for k in range(4)], "`(atan(u) − u + u³/3 − u⁵/5)/u⁷` in `u²` (Taylor, `|u| ≤ 1/128`).")
arr5("ATANH_TAIL", [float(mpf(2) / (2 * k + 7)) for k in range(4)], "`(2 atanh(s) − 2s − 2s³/3 − 2s⁵/5)/s⁷` in `s²` (Taylor, `|s| ≤ 0.0059`).")
dd5("INV_PI", 1 / pi, "`1/π`.")
dd5("THIRD", mpf(1) / 3, "`1/3`.")
dd5("FIFTH", mpf(1) / 5, "`1/5`.")
dd5("TWO_THIRDS", mpf(2) / 3, "`2/3`.")
dd5("TWO_FIFTHS", mpf(2) / 5, "`2/5`.")
dd5("LN2_DD", ln2, "`ln 2` (to `2⁻¹⁰⁶`; `LN2_HI` has 42 bits for exact multiples).")
open(TAB, "a").write("\n".join(extra5) + "\n")
print("argument ok")
