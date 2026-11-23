"""The battery of one-dimensional integrals of lib/tests/math/numerics/quad/battery.rs.

For every integrand (by the name the Rust test gives it): the interval, the exact value from mpmath
(a closed form, else mpmath.quad at 40 digits with the singular points as breakpoints), and what
scipy.integrate.quad (QUADPACK's QAGS on a finite interval, QAGI on an infinite one) returns at
epsabs = epsrel = 1e-12 and limit = 200: value, error estimate, evaluations and ier. A divergent
integral has the exact value nan.

Writes battery.txt next to this script: `name a b exact scipy_value scipy_error scipy_neval
scipy_ier`.

Run: python3 gen_quad.py   (numpy, scipy, mpmath; a few seconds)
"""
import os
import warnings
import numpy as np
import mpmath as mp
from scipy.integrate import quad

mp.mp.dps = 40
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "battery.txt")
inf = np.inf
pi = mp.pi


def mq(f, *pts):
    return mp.quad(f, list(pts), maxdegree=12)


# name, a, b, numpy integrand, exact (mpmath)
BATTERY = [
    # smooth
    ("exp", 0, 1, np.exp, mp.e - 1),
    ("four_over_one_plus_x2", 0, 1, lambda x: 4 / (1 + x * x), pi),
    ("sin", 0, float(np.pi), np.sin, mq(mp.sin, 0, mp.mpf(float(np.pi)))),
    ("x10", 0, 1, lambda x: x**10, mp.mpf(1) / 11),
    ("runge", -1, 1, lambda x: 1 / (1 + 25 * x * x), 2 * mp.atan(5) / 5),
    ("gauss3", -3, 3, lambda x: np.exp(-x * x), mp.sqrt(pi) * mp.erf(3)),
    ("log1p", 0, 1, np.log1p, 2 * mp.log(2) - 1),
    ("x_exp_x", 0, 2, lambda x: x * np.exp(x), mp.e**2 + 1),
    ("poly5", -2, 3, lambda x: x**5 - 3 * x**2, mp.mpf(729) / 6 - 27 - (mp.mpf(64) / 6 + 8)),
    ("sqrt", 0, 1, np.sqrt, mp.mpf(2) / 3),
    # oscillatory
    ("sin30", 0, 1, lambda x: np.sin(30 * x), (1 - mp.cos(30)) / 30),
    ("cos200", 0, 1, lambda x: np.cos(200 * x), mp.sin(200) / 200),
    ("x_sin50", 0, 1, lambda x: x * np.sin(50 * x), mq(lambda x: x * mp.sin(50 * x), 0, 1)),
    ("exp_cos1000", 0, 1, lambda x: np.exp(-x) * np.cos(1000 * x),
     mp.re((mp.exp(mp.mpc(-1, 1000)) - 1) / mp.mpc(-1, 1000))),
    ("sinc", 0.001, 100, lambda x: np.sin(x) / x, mp.si(100) - mp.si(mp.mpf("0.001"))),
    # peaked
    ("lorentz", 0, 1, lambda x: 1 / (1e-4 + (x - 0.3) ** 2),
     100 * (mp.atan(70) + mp.atan(30))),
    ("gauss_peak", 0, 1, lambda x: np.exp(-100 * (x - 0.5) ** 2), mp.sqrt(pi) / 10 * mp.erf(5)),
    ("gauss13", 0, 40, lambda x: np.exp(-((x - 13) ** 2)),
     mp.sqrt(pi) / 2 * (mp.erf(27) + mp.erf(13))),
    ("sech2", 0, 1, lambda x: 1 / np.cosh(10 * (x - 0.2)) ** 2, (mp.tanh(8) + mp.tanh(2)) / 10),
    # singular at an end
    ("inv_sqrt", 0, 1, lambda x: 1 / np.sqrt(x), mp.mpf(2)),
    ("log", 0, 1, np.log, mp.mpf(-1)),
    ("x_m09", 0, 1, lambda x: x**-0.9, mp.mpf(10)),
    ("inv_sqrt_right", 0, 1, lambda x: 1 / np.sqrt(1 - x), mp.mpf(2)),
    ("log_log", 0, 1, lambda x: np.log(x) * np.log(1 - x), 2 - pi**2 / 6),
    ("arcsine", 0, 1, lambda x: 1 / np.sqrt(x * (1 - x)), pi),
    ("cos_over_sqrt", 0, 1, lambda x: np.cos(x) / np.sqrt(x),
     mq(lambda x: mp.cos(x) / mp.sqrt(x), 0, 1)),
    ("x_m09_right", 0, 1, lambda x: (1 - x) ** -0.9, mp.mpf(10)),
    ("log_right", 0, 1, lambda x: np.log(1 - x), mp.mpf(-1)),
    # inside the interval
    ("kink", 0, 1, lambda x: abs(x - 1 / 3), mp.mpf(5) / 18),
    ("inner_sqrt", 0, 1, lambda x: abs(x - 0.4) ** -0.5,
     2 * (mp.sqrt(mp.mpf("0.4")) + mp.sqrt(mp.mpf("0.6")))),
    ("step", 0, 1, lambda x: 1.0 if x > 0.7 else 0.0, mp.mpf("0.3")),
    ("floor4", 0, 1, lambda x: np.floor(4 * x), mp.mpf(3) / 2),
    ("sqrt_abs", 0, 1, lambda x: np.sqrt(abs(x - 0.3)),
     mp.mpf(2) / 3 * (mp.mpf("0.3") ** 1.5 + mp.mpf("0.7") ** 1.5)),
    # wide
    ("exp_decay50", 0, 50, lambda x: np.exp(-x), 1 - mp.exp(-50)),
    ("inv_x_wide", 1, 1e6, lambda x: 1 / x, mp.log(mp.mpf(10) ** 6)),
    # infinite
    ("exp_half", 0, inf, lambda x: np.exp(-x), mp.mpf(1)),
    ("gauss_line", -inf, inf, lambda x: np.exp(-x * x), mp.sqrt(pi)),
    ("cauchy_line", -inf, inf, lambda x: 1 / (1 + x * x), pi),
    ("exp_cos_half", 0, inf, lambda x: np.exp(-x) * np.cos(x), mp.mpf(1) / 2),
    ("power_tail", 1, inf, lambda x: x**-1.1, mp.mpf(10)),
    ("sqrt_cauchy_half", 0, inf, lambda x: 1 / ((1 + x) * np.sqrt(x)), pi),
    ("exp_lower", -inf, 0, np.exp, mp.mpf(1)),
    ("x2_exp_half", 0, inf, lambda x: x * x * np.exp(-x), mp.mpf(2)),
    ("abs_tail_line", -inf, inf, lambda x: (1 + abs(x)) ** -1.5, mp.mpf(4)),
    ("normal30_line", -inf, inf, lambda x: np.exp(-((x - 30) ** 2) / 2) / np.sqrt(2 * np.pi),
     mp.mpf(1)),
    ("gauss30_half", 0, inf, lambda x: np.exp(-((x - 30) ** 2)), mp.sqrt(pi) / 2 * (1 + mp.erf(30))),
    # divergent
    ("inv_x_tail", 1, inf, lambda x: 1 / x, mp.nan),
    ("inv_x_zero", 0, 1, lambda x: 1 / x, mp.nan),
]


def main():
    lines = ["# The one-dimensional battery: see gen_quad.py",
             "# name a b exact scipy_value scipy_error scipy_neval scipy_ier"]
    for name, a, b, f, exact in BATTERY:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = quad(f, a, b, epsabs=1e-12, epsrel=1e-12, limit=200, full_output=1)
        value, error, info = r[0], r[1], r[2]
        ier = 0 if len(r) == 3 else r[3] if isinstance(r[3], int) else 1
        if len(r) > 3:
            ier = 1
        ex = "nan" if mp.isnan(exact) else mp.nstr(exact, 25)
        lines.append(f"{name} {a!r} {b!r} {ex} {value!r} {error!r} {info['neval']} {ier}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
