#!/usr/bin/env python3
"""Writes ../reference.rs: the reference data of the ODE tests of a::math::numerics.

    python3 gen_reference.py > ../reference.rs

Needs numpy, scipy (1.18 was used) and mpmath. Four kinds of data, for the six non-stiff
problems of problems.rs at the tolerances rtol = atol listed below:

* EXACT: y(t1) of each problem, from its closed form or from mpmath's Taylor integrator at 30
  digits (odefun), printed to 17 significant digits.
* DOPRI5: Hairer's DOPRI5 as SciPy bundles it (scipy.integrate.ode, "dopri5"), with the
  defaults of the code (safe 0.9, fac1 0.2, fac2 10, beta 0.04, hinit): accepted and rejected
  steps, evaluations, final state.
* DOP853: the step loop of Hairer's DOP853 (dp86co) transcribed literally below in numpy, with
  the coefficients SciPy prints from dop853.f, and the defaults of the code (safe 0.9, fac1 0.333,
  fac2 6, beta 0, hinit). SciPy's own "dop853" integrator is not used for this: in SciPy 1.18 it
  multiplies every rejected step by its dfactor, 0.3 by default, whatever the error (that rule
  reproduces its counts in all 24 runs), where DOP853 divides it by min(1/fac1, err^(1/8)/safe),
  so its step counts are not those of the published code.
* SCIPY_RK45 and SCIPY_DOP853: scipy.integrate.solve_ivp with method "RK45" and "DOP853".
"""
import math
import sys

import mpmath
import numpy as np
import scipy
from scipy.integrate import ode, solve_ivp
import scipy.integrate._ivp.dop853_coefficients as dc

mpmath.mp.dps = 30
MU = 0.012277471
ARENSTORF_Y0 = [0.994, 0.0, 0.0, -2.00158510637908252240537862224]
ARENSTORF_T = 17.0652165601579625588917206249
KEPLER_E = 0.9

def decay(t, y):
    return [-y[0]]

def oscillator(t, y):
    return [y[1], -y[0]]

def kepler(t, y):
    r3 = (y[0] * y[0] + y[1] * y[1]) ** 1.5
    return [y[2], y[3], -y[0] / r3, -y[1] / r3]

def lotka(t, y):
    return [1.5 * y[0] - y[0] * y[1], -3.0 * y[1] + y[0] * y[1]]

def arenstorf(t, y, sqrt=math.sqrt):
    mup = 1 - MU
    r1 = (y[0] + MU) ** 2 + y[1] ** 2
    r1 = r1 * sqrt(r1)
    r2 = (y[0] - mup) ** 2 + y[1] ** 2
    r2 = r2 * sqrt(r2)
    return [y[2], y[3],
            y[0] + 2 * y[3] - mup * (y[0] + MU) / r1 - MU * (y[0] - mup) / r2,
            y[1] - 2 * y[2] - mup * y[1] / r1 - MU * y[1] / r2]

def brusselator(t, y):
    return [1 + y[0] * y[0] * y[1] - 4 * y[0], 3 * y[0] - y[0] * y[0] * y[1]]

PROBLEMS = [
    # name in problems.rs, f, y0, t1
    ("Decay", decay, [1.0], 10.0),
    ("Oscillator", oscillator, [1.0, 0.0], 20 * math.pi),
    ("Kepler", kepler, [1 - KEPLER_E, 0.0, 0.0, math.sqrt((1 + KEPLER_E) / (1 - KEPLER_E))],
     6 * math.pi),
    ("Lotka", lotka, [10.0, 5.0], 15.0),
    ("Arenstorf", arenstorf, ARENSTORF_Y0, ARENSTORF_T),
    ("Brusselator", brusselator, [1.5, 3.0], 20.0),
]

def exact(name, f, y0, t1):
    """y(t1) to 30 digits, at the double t1."""
    T = mpmath.mpf(t1)
    if name == "Decay":
        return [mpmath.exp(-T)]
    if name == "Oscillator":
        return [mpmath.cos(T), -mpmath.sin(T)]
    if name == "Kepler":
        # The orbit of the double initial state, which starts at the perihelion: its semi-major
        # axis and eccentricity from the energy, then Kepler's equation E − e sin E = n·t. (The
        # rounding of the initial velocity moves the period by a few parts in 10¹⁴, which three
        # periods at e = 0.9 make visible.)
        r0, v0 = mpmath.mpf(y0[0]), mpmath.mpf(y0[3])
        a = -1 / (2 * (v0 * v0 / 2 - 1 / r0))
        e = 1 - r0 / a
        n = a ** mpmath.mpf(-1.5)
        M = n * T
        E = mpmath.findroot(lambda E: E - e * mpmath.sin(E) - M, M)
        d = 1 - e * mpmath.cos(E)
        w = mpmath.sqrt(1 - e * e)
        return [a * (mpmath.cos(E) - e), a * w * mpmath.sin(E), -a * n * mpmath.sin(E) / d,
                a * n * w * mpmath.cos(E) / d]
    # The rest by the Taylor method, at the decimal initial values the doubles stand for.
    mf = {
        "Lotka": lambda t, y: [mpmath.mpf("1.5") * y[0] - y[0] * y[1], -3 * y[1] + y[0] * y[1]],
        "Arenstorf": lambda t, y: arenstorf(t, y, mpmath.sqrt),
        "Brusselator": lambda t, y: [1 + y[0] ** 2 * y[1] - 4 * y[0], 3 * y[0] - y[0] ** 2 * y[1]],
    }[name]
    F = mpmath.odefun(mf, 0, [mpmath.mpf(v) for v in y0])
    return list(F(T))

# ---- DOP853 transcribed (dp86co and hinit of dop853.f) ------------------------------------------
A, B, C, E3, E5 = dc.A, dc.B, dc.C, dc.E3, dc.E5

def hinit(f, x, y, f0, iord, hmax, rtol, atol):
    sk = atol + rtol * np.abs(y)
    dnf = np.sum((f0 / sk) ** 2)
    dny = np.sum((y / sk) ** 2)
    h = 1e-6 if (dnf <= 1e-10 or dny <= 1e-10) else math.sqrt(dny / dnf) * 0.01
    h = min(h, hmax)
    f1 = np.array(f(x + h, y + h * f0))
    der2 = math.sqrt(np.sum(((f1 - f0) / sk) ** 2)) / h
    der12 = max(abs(der2), math.sqrt(dnf))
    h1 = max(1e-6, abs(h) * 1e-3) if der12 <= 1e-15 else (0.01 / der12) ** (1.0 / iord)
    return min(100 * abs(h), h1, hmax)

def dop853(f, y0, t1, rtol, atol):
    x, y = 0.0, np.array(y0, float)
    n = len(y)
    k1 = np.array(f(x, y))
    nfev = 1
    h = hinit(f, x, y, k1, 8, t1, rtol, atol)
    nfev += 1
    uround = 2.3e-16
    facold, beta, safe, facc1, facc2 = 1e-4, 0.0, 0.9, 1 / 0.333, 1 / 6.0
    expo1 = 1 / 8 - beta * 0.2
    reject = last = False
    acc = rej = 0
    while True:
        if 0.1 * abs(h) <= abs(x) * uround:
            raise RuntimeError("step size too small")
        if x + 1.01 * h - t1 > 0:
            h = t1 - x
            last = True
        K = np.zeros((12, n))
        K[0] = k1
        for s in range(1, 12):
            K[s] = f(x + C[s] * h, y + h * (K[:s].T @ A[s, :s]))
        nfev += 11
        k4 = K.T @ B
        k5 = y + h * k4
        sk = atol + rtol * np.maximum(np.abs(y), np.abs(k5))
        err2 = np.sum(((K.T @ E3[:12]) / sk) ** 2)
        err = np.sum(((K.T @ E5[:12]) / sk) ** 2)
        deno = err + 0.01 * err2
        if deno <= 0:
            deno = 1.0
        err = abs(h) * err * math.sqrt(1.0 / (n * deno))
        fac11 = err ** expo1
        fac = fac11 / facold ** beta
        fac = max(facc2, min(facc1, fac / safe))
        hnew = h / fac
        if err <= 1:
            facold = max(err, 1e-4)
            acc += 1
            k1 = np.array(f(x + h, k5))
            nfev += 1
            y = k5
            x = x + h
            if last:
                return acc, rej, nfev, y
            if abs(hnew) > t1:
                hnew = t1
            if reject:
                hnew = min(abs(hnew), abs(h))
            reject = False
        else:
            hnew = h / min(facc1, fac11 / safe)
            reject = True
            last = False
            rej += 1
        h = hnew

# ---- Hairer's DOPRI5 as SciPy bundles it ----------------------------------------------------------
def dopri5(f, y0, t1, rtol, atol):
    calls = [0]
    accepted = [0]

    def g(t, y):
        calls[0] += 1
        return f(t, y)

    def solout(t, y):
        accepted[0] += 1

    r = ode(g).set_integrator("dopri5", rtol=rtol, atol=atol, nsteps=10**8)
    r.set_solout(solout)
    r.set_initial_value(y0, 0.0)
    y = r.integrate(t1)
    assert r.successful()
    acc = accepted[0] - 1  # solout is called at the initial point too
    tried = (calls[0] - 2) // 6
    assert 2 + 6 * tried == calls[0]
    return acc, tried - acc, calls[0], y

def ivp(method, f, y0, t1, rtol, atol):
    s = solve_ivp(f, (0.0, t1), y0, method=method, rtol=rtol, atol=atol)
    assert s.success
    return len(s.t) - 1, 0, s.nfev, s.y[:, -1]

def rust_f64(v):
    s = repr(float(v))
    if "e" not in s and "." not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s

def rust_list(vs):
    return "&[" + ", ".join(rust_f64(v) for v in vs) + "]"

def main():
    out = sys.stdout
    out.write("//! Reference data of the ODE tests, written by `data/gen_reference.py` with SciPy "
              f"{scipy.__version__}, NumPy {np.__version__}\n//! and mpmath {mpmath.__version__}: "
              "do not edit; run the script again.\n\n")
    out.write("use super::problems::Problem;\n\n")
    out.write("/// One integration by a reference code: the problem, `rtol = atol`, the accepted and\n"
              "/// rejected steps, the evaluations of the right-hand side and the state at `t1`.\n")
    out.write("pub struct Run {\n    pub problem: Problem,\n    pub tol: f64,\n    pub steps: usize,\n"
              "    pub rejected: usize,\n    pub nfev: usize,\n    pub y: &'static [f64],\n}\n\n")
    out.write("/// `y(t1)` of each problem, from its closed form or from mpmath's Taylor method at 30\n"
              "/// digits.\n")
    out.write("pub const EXACT: &[(Problem, &[f64])] = &[\n")
    for name, f, y0, t1 in PROBLEMS:
        ex = exact(name, f, y0, t1)
        out.write(f"    (Problem::{name}, {rust_list(ex)}),\n")
        sys.stderr.write(f"exact {name} done\n")
    out.write("];\n\n")
    codes = [
        ("DOPRI5", "Hairer's DOPRI5 as bundled by SciPy (`scipy.integrate.ode`).", dopri5,
         [1e-4, 1e-7, 1e-10]),
        ("DOP853", "The step loop of Hairer's DOP853, transcribed in the script.", dop853,
         [1e-4, 1e-7, 1e-10, 1e-13]),
        ("SCIPY_RK45", "`scipy.integrate.solve_ivp(method = \"RK45\")`; `rejected` is not reported.",
         lambda *a: ivp("RK45", *a), [1e-4, 1e-7, 1e-10]),
        ("SCIPY_DOP853", "`scipy.integrate.solve_ivp(method = \"DOP853\")`; `rejected` is not "
         "reported.", lambda *a: ivp("DOP853", *a), [1e-4, 1e-7, 1e-10, 1e-13]),
    ]
    for const, doc, run, tols in codes:
        out.write(f"/// {doc}\n")
        out.write(f"pub const {const}: &[Run] = &[\n")
        for name, f, y0, t1 in PROBLEMS:
            for tol in tols:
                acc, rej, nfev, y = run(f, y0, t1, tol, tol)
                out.write(f"    Run {{ problem: Problem::{name}, tol: {rust_f64(tol)}, steps: {acc}, "
                          f"rejected: {rej}, nfev: {nfev}, y: {rust_list(y)} }},\n")
        out.write("];\n\n")
        sys.stderr.write(f"{const} done\n")

if __name__ == "__main__":
    main()
