"""The stability angles and the gains of Bdf's formulas, which its documentation states.

A(alpha) of the BDFs and of ode15s's NDFs of orders 1 to 6 from the boundary locus, and the
error constants kappa*gamma_k + 1/(k+1) against the BDFs' 1/(k+1) with the step-size gain
(C_BDF/C_NDF)^(1/(k+1)) they allow (Shampine and Reichelt, "The MATLAB ODE Suite", 1997,
Table 1: 26%, 26%, 26%, 12%, 0 and alpha 90, 90, 80, 66, 51 against 90, 90, 86, 73, 51).

The NDF of order k is sum_{j=1..k} (1/j) nabla^j y_{n+1} - kappa*gamma_k*nabla^{k+1} y_{n+1}
= h f_{n+1}; on y = zeta^n, nabla = 1 - 1/zeta, so the boundary of the stability region is
z(theta) = rho(zeta)/sigma(zeta) at zeta = e^{i theta}. alpha is the least angle between the
negative real axis and the locus (capped at 90 degrees). Plain floats and cmath, 2e6 points on
the half circle; runs in about a minute:

    python3 lib/tests/math/numerics/ode/data/check_bdf_angles.py
"""
import cmath, math

KAPPA = [0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0]

def z_of(theta, k, kappa):
    d = 1 - cmath.exp(-1j * theta)
    g = sum(1.0 / j for j in range(1, k + 1))
    return sum(d ** j / j for j in range(1, k + 1)) - kappa * g * d ** (k + 1)

def alpha(k, kappa):
    best = math.pi / 2
    n = 2_000_000
    for i in range(1, n):
        th = math.pi * i / n
        z = z_of(th, k, kappa)
        if z.real < 0:
            ang = math.pi - abs(cmath.phase(z))
            best = min(best, ang)
    return math.degrees(best)

if __name__ == "__main__":
    print("order  BDF alpha  NDF alpha  C_BDF  C_NDF  error-constant reduction  step gain")
    for k in range(1, 7):
        kap = KAPPA[k] if k < len(KAPPA) else 0.0
        g = sum(1.0 / j for j in range(1, k + 1))
        cb, cn = 1.0 / (k + 1), kap * g + 1.0 / (k + 1)
        print(f"{k:5d}  {alpha(k, 0.0):9.2f}  {alpha(k, kap):9.2f}  {cb:.4f} {cn:.4f}"
              f"  {100 * (1 - cn / cb):5.1f}%  {100 * ((cb / cn) ** (1 / (k + 1)) - 1):5.1f}%")
