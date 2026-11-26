#!/usr/bin/env python3
"""Writes ../stiff_reference.rs: the reference data of the tests of the stiff methods.

    python3 gen_stiff.py > ../stiff_reference.rs

Needs numpy and scipy (1.18 was used). Two kinds of data:

* REFERENCE: y(t1) of each stiff problem of stiff_problems.rs, by scipy.integrate.solve_ivp
  with method "Radau" at rtol 1e-13 (the linear system: its closed form). Each is checked here
  against LSODA at the same tolerance (asserted to agree within 1e-9 relatively) and printed to
  stderr beside the values of the Test Set for IVP Solvers (Mazzia and Magherini, Bari) where it
  has the problem; the non-stiff problems use EXACT of reference.rs.
* SCIPY_RADAU and SCIPY_BDF: scipy.integrate.solve_ivp with "Radau" and "BDF" at the listed
  tolerances: accepted steps, evaluations of f, Jacobians, LU decompositions, final state. The
  Jacobian is SciPy's finite differences, but for the linear system, which is given its matrix.
  SciPy's nfev does not count the evaluations of its differenced Jacobians (num_jac calls the
  function itself); the tests add n per Jacobian before they compare.
* ROBERTSON_40 and VAN_DER_POL_10: y(40) of Robertson's problem and y(20) of the Van der Pol
  oscillator at mu = 10 from (2, 0), by "Radau" at rtol 1e-13, checked against LSODA, for the
  tests of the orders and of the tolerances.
* VDP_ZEROS and ROBERTSON_HALF: the zeros of y0 of the Van der Pol problem on [0, 3000], and the
  time at which Robertson's y2 (the third component) reaches 1/2, by solve_ivp's events with
  "Radau" at rtol 1e-12, checked against LSODA.
"""
import os
import sys

import numpy as np
import scipy
from scipy.integrate import solve_ivp

# The non-stiff problems are those of gen_reference.py, next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_reference as gr  # noqa: E402


def rober(t, y):
    a, b, c = y
    return [-0.04 * a + 1e4 * b * c, 0.04 * a - 1e4 * b * c - 3e7 * b * b, 3e7 * b * b]


def rober_jac(t, y):
    a, b, c = y
    return [[-0.04, 1e4 * c, 1e4 * b], [0.04, -1e4 * c - 6e7 * b, -1e4 * b], [0.0, 6e7 * b, 0.0]]


def vdp(t, y):
    return [y[1], 1000.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]]


def vdp10(t, y):
    return [y[1], 10.0 * (1.0 - y[0] * y[0]) * y[1] - y[0]]


def hires(t, y):
    y1, y2, y3, y4, y5, y6, y7, y8 = y
    return [-1.71 * y1 + 0.43 * y2 + 8.32 * y3 + 0.0007,
            1.71 * y1 - 8.75 * y2,
            -10.03 * y3 + 0.43 * y4 + 0.035 * y5,
            8.32 * y2 + 1.71 * y3 - 1.12 * y4,
            -1.745 * y5 + 0.43 * y6 + 0.43 * y7,
            -280.0 * y6 * y8 + 0.69 * y4 + 1.71 * y5 - 0.43 * y6 + 0.69 * y7,
            280.0 * y6 * y8 - 1.81 * y7,
            -280.0 * y6 * y8 + 1.81 * y7]


def orego(t, y):
    y1, y2, y3 = y
    return [77.27 * (y2 + y1 * (1.0 - 8.375e-6 * y1 - y2)),
            (y3 - (1.0 + y1) * y2) / 77.27,
            0.161 * (y1 - y3)]


# The linear system of dimension 50: A = Q diag(lambda) Q with the reflection
# Q = I - 2 v v^T / (v^T v), v_i = ((37 i) mod 50 + 1) / 50, and lambda_i = -10^(6 i / 49):
# eigenvalues from -1 to -10^6, spread evenly in their logarithms.
N50 = 50
V50 = np.array([((37 * i) % 50 + 1) / 50.0 for i in range(N50)])
Q50 = np.eye(N50) - 2.0 * np.outer(V50, V50) / V50.dot(V50)
L50 = np.array([-(10.0 ** (6.0 * i / 49.0)) for i in range(N50)])
A50 = Q50 @ np.diag(L50) @ Q50


def linear(t, y):
    return A50 @ y


def linear_jac(t, y):
    return A50


def linear_exact(t):
    return Q50 @ (np.exp(L50 * t) * (Q50 @ np.ones(N50)))


# Name in stiff_problems.rs, f, y0, t1, analytic Jacobian given to SciPy, atol / rtol, and the
# values of the Test Set where it has the problem.
STIFF = [
    ("Robertson", rober, [1.0, 0.0, 0.0], 1e11, None, 1e-8,
     [0.2083340149701255e-07, 0.8333360770334713e-13, 0.9999999791665050]),
    ("VanDerPol", vdp, [2.0, 0.0], 3000.0, None, 1e-4, None),
    ("Hires", hires, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057], 321.8122, None, 1e-4,
     [0.7371312573325668e-3, 0.1442485726316185e-3, 0.5888729740967575e-4,
      0.1175651343283149e-2, 0.2386356198831331e-2, 0.6238968252742796e-2,
      0.2849998395185769e-2, 0.2850001604814231e-2]),
    ("Orego", orego, [1.0, 2.0, 3.0], 360.0, None, 1e-4,
     [0.1000814870318523e1, 0.1228178521549917e4, 0.1320554942846706e3]),
    ("Linear", linear, [1.0] * N50, 10.0, linear_jac, 1e-6, None),
]

# The non-stiff problems of problems.rs, as reference.rs has them.
NONSTIFF = gr.PROBLEMS


def reference(name, f, y0, t1, jac):
    if name == "Linear":
        return linear_exact(t1)
    kw = {} if jac is None else {"jac": jac}
    atol = 1e-22 if name.startswith("Robertson") or name.startswith("ROBERTSON") else 1e-16
    a = solve_ivp(f, (0.0, t1), y0, method="Radau", rtol=1e-13, atol=atol, **kw)
    b = solve_ivp(f, (0.0, t1), y0, method="LSODA", rtol=1e-13, atol=atol, **kw)
    assert a.success and b.success, name
    ya, yb = a.y[:, -1], b.y[:, -1]
    spread = np.max(np.abs(ya - yb) / np.abs(ya))
    assert spread < 1e-9, (name, spread)
    sys.stderr.write(f"reference {name}: Radau and LSODA agree to {spread:.1e}\n")
    return ya


def events():
    def zero(t, y):
        return y[0]

    def half(t, y):
        return y[2] - 0.5
    half.terminal = True
    out = []
    for f, y0, t1, g, kw in [(vdp, [2.0, 0.0], 3000.0, zero, {"atol": 1e-14}),
                             (rober, [1.0, 0.0, 0.0], 1e11, half, {"atol": 1e-22, "jac": rober_jac})]:
        a = solve_ivp(f, (0.0, t1), y0, method="Radau", rtol=1e-12, events=g, **kw)
        b = solve_ivp(f, (0.0, t1), y0, method="LSODA", rtol=1e-12, events=g, **kw)
        ta, tb = a.t_events[0], b.t_events[0]
        assert len(ta) == len(tb), (ta, tb)
        spread = np.max(np.abs(ta - tb) / np.abs(ta))
        assert spread < 1e-8, spread
        sys.stderr.write(f"events: {len(ta)}, Radau and LSODA agree to {spread:.1e}\n")
        out.append(ta)
    return out


def run(method, f, y0, t1, rtol, atol, jac):
    kw = {} if jac is None else {"jac": jac}
    s = solve_ivp(f, (0.0, t1), y0, method=method, rtol=rtol, atol=atol, **kw)
    assert s.success, (method, rtol)
    return len(s.t) - 1, s.nfev, s.njev, s.nlu, s.y[:, -1]


def rust_f64(v):
    s = repr(float(v))
    if "e" not in s and "." not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return s


def rust_list(vs):
    return "&[" + ", ".join(rust_f64(v) for v in vs) + "]"


def main():
    out = sys.stdout
    out.write("//! Reference data of the tests of the stiff methods, written by `data/gen_stiff.py` "
              f"with SciPy {scipy.__version__}\n//! and NumPy {np.__version__}: do not edit; run "
              "the script again.\n\n")
    out.write("use super::problems::Problem;\nuse super::stiff_problems::Stiff;\n\n")
    out.write("/// One integration by SciPy: the problem, the tolerances, the accepted steps, the "
              "evaluations\n/// of the right-hand side, the Jacobians, the LU decompositions and "
              "the state at `t1`.\npub struct Run<P: 'static> {\n    pub problem: P,\n"
              "    pub rtol: f64,\n    pub atol: f64,\n    pub steps: usize,\n    pub nfev: usize,\n"
              "    pub njev: usize,\n    pub nlu: usize,\n    pub y: &'static [f64],\n}\n\n")
    out.write("/// `y(t1)` of each stiff problem: `solve_ivp(method = \"Radau\", rtol = 1e-13)`, "
              "checked against\n/// LSODA, and the closed form of the linear system.\n")
    out.write("pub const REFERENCE: &[(Stiff, &[f64])] = &[\n")
    for name, f, y0, t1, jac, _, testset in STIFF:
        y = reference(name, f, y0, t1, jac)
        if testset is not None:
            d = np.max(np.abs(np.array(testset) - y) / np.abs(y))
            sys.stderr.write(f"  {name}: against the Test Set {d:.1e}\n")
        out.write(f"    (Stiff::{name}, {rust_list(y)}),\n")
    out.write("];\n\n")
    for const, f, y0, t1, jac in [("ROBERTSON_40", rober, [1.0, 0.0, 0.0], 40.0, rober_jac),
                                   ("VAN_DER_POL_10", vdp10, [2.0, 0.0], 20.0, None)]:
        y = reference(const, f, y0, t1, jac)
        out.write(f"/// `{const}`: see the script.\n")
        out.write(f"pub const {const}: &[f64] = {rust_list(y)};\n\n")
    zeros, half = events()
    out.write("/// The zeros of `y₀` of the Van der Pol problem on `[0, 3000]`.\n")
    out.write(f"pub const VDP_ZEROS: &[f64] = {rust_list(zeros)};\n\n")
    out.write("/// The time at which the third component of Robertson's problem reaches 1/2.\n")
    out.write(f"pub const ROBERTSON_HALF: f64 = {rust_f64(half[0])};\n\n")
    tols = [1e-4, 1e-6, 1e-8]
    for method in ["Radau", "BDF"]:
        out.write(f"/// `solve_ivp(method = \"{method}\")` on the stiff problems, `atol` the given "
                  "multiple of `rtol`.\n")
        out.write(f"pub const SCIPY_{method.upper()}: &[Run<Stiff>] = &[\n")
        for name, f, y0, t1, jac, ratio, _ in STIFF:
            for rtol in tols:
                atol = rtol * ratio
                steps, nfev, njev, nlu, y = run(method, f, y0, t1, rtol, atol, jac)
                out.write(f"    Run {{ problem: Stiff::{name}, rtol: {rust_f64(rtol)}, atol: "
                          f"{rust_f64(atol)}, steps: {steps}, nfev: {nfev}, njev: {njev}, nlu: {nlu}, "
                          f"y: {rust_list(y)} }},\n")
        out.write("];\n\n")
        sys.stderr.write(f"{method} stiff done\n")
        out.write(f"/// `solve_ivp(method = \"{method}\")` on the non-stiff problems, `rtol = atol`.\n")
        out.write(f"pub const SCIPY_{method.upper()}_NONSTIFF: &[Run<Problem>] = &[\n")
        for name, f, y0, t1 in NONSTIFF:
            for tol in [1e-4, 1e-7]:
                steps, nfev, njev, nlu, y = run(method, f, y0, t1, tol, tol, None)
                out.write(f"    Run {{ problem: Problem::{name}, rtol: {rust_f64(tol)}, atol: "
                          f"{rust_f64(tol)}, steps: {steps}, nfev: {nfev}, njev: {njev}, nlu: {nlu}, "
                          f"y: {rust_list(y)} }},\n")
        out.write("];\n\n")
        sys.stderr.write(f"{method} non-stiff done\n")


if __name__ == "__main__":
    main()
