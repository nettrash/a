"""Prints `ERF_F32` of `mod.rs` (the `f32` form of `erf` on `1/2 ≤ |x| < 4`): Chebyshev
interpolants of degree 12 on the halves of `[1/2, 4)`, mpmath at 40 digits. `python3 gen_f32.py`
(seconds) prints the table as it stands in `mod.rs`, byte for byte."""
import sys

sys.dont_write_bytecode = True
import mpmath as mp
mp.mp.dps = 40
def fit(a, b, D):
    c, h = (a + b) / 2, (b - a) / 2
    N = D + 1
    xs = [mp.cos(mp.pi * (j + mp.mpf(1) / 2) / N) for j in range(N)]
    fs = [mp.erf(c + h * t) for t in xs]
    ch = [2 * mp.fsum(fs[j] * mp.cos(mp.pi * i * (j + mp.mpf(1) / 2) / N) for j in range(N)) / N for i in range(N)]
    ch[0] /= 2
    T = [[mp.mpf(1)], [mp.mpf(0), mp.mpf(1)]]
    for i in range(2, N):
        nx = [mp.mpf(0)] * (i + 1)
        for m_, v in enumerate(T[-1]): nx[m_ + 1] += 2 * v
        for m_, v in enumerate(T[-2]): nx[m_] -= v
        T.append(nx)
    mono = [mp.mpf(0)] * N
    for i in range(N):
        for m_, v in enumerate(T[i]): mono[m_] += ch[i] * v
    cf = [float(mono[i] / h**i) for i in range(N)]
    err = 0
    for j in range(401):
        x = a + (b - a) * j / 400
        u = float(x - c); v = 0.0
        for k in reversed(cf): v = v * u + k
        err = max(err, abs(v / mp.erf(x) - 1))
    return cf, err
rows = []; worst = 0
for i in range(7):
    cf, e = fit(mp.mpf(1)/2 + mp.mpf(i)/2, 1 + mp.mpf(i)/2, 12); rows.append(cf); worst = max(worst, e)
if True:
    o = sys.stdout
    o.write("/// `erf(x)` for the `f32` form on `1/2 ≤ x < 4`: Chebyshev interpolants of degree 12 in\n")
    o.write("/// `x − (3/4 + i/2)` on the pieces `[1/2 + i/2, 1 + i/2]`, `i = 0, …, 6` (mpmath at 40 digits),\n")
    o.write(f"/// relative error at most {mp.nstr(worst, 2)} (an `f32` needs about `2⁻³⁴`).\n")
    o.write("const ERF_F32: [[f64; 13]; 7] = [\n")
    for r in rows:
        o.write("    [\n" + "".join(f"        {repr(v)},\n" for v in r) + "    ],\n")
    o.write("];\n")
print("worst", worst, file=sys.stderr)
