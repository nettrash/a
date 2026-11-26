"""Prints `PSI_F32` and `PSI_X0_F32` of `psi.rs` (the `f32` form of `ψ` on `1 ≤ x < 8`):
Chebyshev interpolants of degree 15 on the halves of `[1, 8)`, on the first two of `ψ(x)/(x − x₀)`,
mpmath at 40 digits. `python3 gen_psi_f32.py` (seconds) prints the tables as they stand in
`psi.rs`, byte for byte."""
import sys

sys.dont_write_bytecode = True
import mpmath as mp
mp.mp.dps = 40
x0 = mp.findroot(mp.digamma, 1.46)
x0h = float(x0); x0l = float(x0 - x0h)
def fit(f, a, b, D):
    c, h = (a + b) / 2, (b - a) / 2
    N = D + 1
    xs = [mp.cos(mp.pi * (j + mp.mpf(1) / 2) / N) for j in range(N)]
    fs = [f(c + h * t) for t in xs]
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
        err = max(err, abs(v / f(x) - 1))
    return cf, err
Q = lambda x: mp.digamma(x) / (x - x0) if abs(x - x0) > mp.mpf(10)**-20 else mp.psi(1, x0)
D = 15
rows = []; worst = 0
for i in range(14):
    a, b = 1 + mp.mpf(i) / 2, mp.mpf(3) / 2 + mp.mpf(i) / 2
    cf, e = fit(Q if i < 2 else mp.digamma, a, b, D)
    rows.append(cf); worst = max(worst, e); print(i, float(a), mp.nstr(e, 2), file=sys.stderr)
if True:
    o = sys.stdout
    o.write("/// `ψ(x)` for the `f32` form on `1 ≤ x < 8`: Chebyshev interpolants of degree 15 in\n")
    o.write("/// `x − (5/4 + i/2)` on the pieces `[1 + i/2, 3/2 + i/2]`, `i = 0, …, 13` (mpmath at 40\n")
    o.write("/// digits), on the first two of `ψ(x)/(x − x₀)` about the zero `x₀`; relative error at most\n")
    o.write(f"/// {mp.nstr(worst, 2)} (an `f32` needs about `2⁻³⁴`).\n")
    o.write("const PSI_F32: [[f64; 16]; 14] = [\n")
    for r in rows:
        o.write("    [\n" + "".join(f"        {repr(v)},\n" for v in r) + "    ],\n")
    o.write("];\n")
    o.write(f"/// The zero `x₀` of `ψ` as a double-double.\nconst PSI_X0_F32: (f64, f64) = ({x0h!r}, {x0l!r});\n")
print("worst", mp.nstr(worst, 3), x0h, x0l, file=sys.stderr)
