"""Exact values of the integrals of the review-round-2 regression tests, by mpmath at 40 digits.

    python3 gen_review2.py

prints each integral, as the tests embed it (rounded to the nearest f64). The integrands are
Gaussians e^(-((x-c)/w)^2) and Lorentzians 1/(1+((x-c)/w)^2) with the closed forms below; the
parameters are those of the second review's sweep (its bumps.csv) that the tests take.
"""
import mpmath as mp

mp.mp.dps = 40


def gauss(a, b, c, w):
    a, b, c, w = (mp.mpf(v) if v not in (mp.inf, -mp.inf) else v for v in (a, b, c, w))
    lo = mp.erf((a - c) / w) if a != -mp.inf else -1
    hi = mp.erf((b - c) / w) if b != mp.inf else 1
    return w * mp.sqrt(mp.pi) / 2 * (hi - lo)


def lorentz(a, b, c, w):
    a, b, c, w = mp.mpf(a), mp.mpf(b), mp.mpf(c), mp.mpf(w)
    hi = mp.atan((b - c) / w) if b != mp.inf else mp.pi / 2
    return w * (hi - mp.atan((a - c) / w))


CASES = [
    # clenshaw.rs, the_rounding_of_the_points_is_in_the_error_estimate
    ("cc peak about 101", gauss(100.0, 167.66011795304195, 101.01364004955065, 0.05986618496275109)),
    ("cc peak about 119", gauss(100.0, 124.35747911845812, 118.9773779740595, 0.06948934569848714)),
    ("cc peak about 8.5", gauss(2.0, 10.123824066840644, 8.480353527792523, 0.00839085192224972)),
    ("cc peak about 1.16", gauss(0.0, 1.4653368690885704, 1.1560060288518472, 0.0006237717139641789)),
    # tanh_sinh.rs
    ("ts Lorentzian on [1, inf)", lorentz(1.0, mp.inf, 2.9593111465700366, 0.00011608163360711831)),
    ("ts Gaussian on [10, inf)", gauss(10.0, mp.inf, 21.415596906922598, 0.06093840166231403)),
    # kronrod.rs
    ("QAGI Lorentzian on [-3, inf)", lorentz(-3.0, mp.inf, 77.21891235287957, 0.00013152740782116954)),
    (
        "two bumps on [0, inf)",
        gauss(0.0, mp.inf, 28.46139236549884, 0.8254181151991499)
        + gauss(0.0, mp.inf, 2 * 28.46139236549884, 0.8254181151991499),
    ),
]

for name, value in CASES:
    print(f"{name:32} {mp.nstr(value, 25)}  f64 {float(value)!r}")
