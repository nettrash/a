"""SciPy's Radau and BDF on the two problems of radau.rs whose stiff component follows a slow one
through a nonlinear coupling, over the end times the tests use, in the norm the tests assert,
max |y_i - exact_i| / (atol + rtol |exact_i|); and the closed form of the second problem against
SciPy's Radau at rtol = 1e-13.

    python3 lib/tests/math/numerics/ode/data/check_stiff_coupling.py

Kaps's problem: y1' = -(1/eps + 2) y1 + y2^2/eps, y2' = y1 - y2 - y2^2, y(0) = (1, 1), solution
(e^-2t, e^-t). The cube: y1' = (y2^3 - y1)/eps, y2' = -y2 + y3, y3' = -2 y3 + cos(t)/2,
y(0) = (0, 1.5, 1); y3 and y2 in closed form, y1 by the outer expansion u - eps u' + eps^2 u''
with u = y2^3 (error O(eps^3) past the initial layer).
"""
import numpy as np
from scipy.integrate import solve_ivp


def norm(y, exact, rtol, atol):
    exact = np.asarray(exact)
    return float(np.max(np.abs(np.asarray(y) - exact) / (atol + rtol * np.abs(exact))))


def kaps(eps):
    def f(t, y):
        return [-(1 / eps + 2) * y[0] + y[1] ** 2 / eps, y[0] - y[1] - y[1] ** 2]

    def jac(t, y):
        return [[-(1 / eps + 2), 2 * y[1] / eps], [1.0, -1 - 2 * y[1]]]

    return f, jac


def cube(eps):
    def f(t, y):
        return [(y[1] ** 3 - y[0]) / eps, -y[1] + y[2], -2 * y[2] + 0.5 * np.cos(t)]

    def jac(t, y):
        return [[-1 / eps, 3 * y[1] ** 2 / eps, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, -2.0]]

    return f, jac


def cube_solution(eps, t):
    c, s, e1, e2 = np.cos(t), np.sin(t), np.exp(-t), np.exp(-2 * t)
    y3 = 0.2 * c + 0.1 * s + 0.8 * e2
    y2 = 0.05 * c + 0.15 * s - 0.8 * e2 + 2.25 * e1
    d2 = -0.05 * s + 0.15 * c + 1.6 * e2 - 2.25 * e1
    dd2 = -0.05 * c - 0.15 * s - 3.2 * e2 + 2.25 * e1
    u, du, ddu = y2**3, 3 * y2**2 * d2, 6 * y2 * d2**2 + 3 * y2**2 * dd2
    return np.array([u - eps * du + eps**2 * ddu, y2, y3])


def main():
    print("The closed form of the cube against SciPy's Radau at rtol 1e-13, atol 1e-15, on [0.5, 10]:")
    for eps in [1e-5, 1e-8]:
        f, jac = cube(eps)
        ref = solve_ivp(f, (0, 10), [0.0, 1.5, 1.0], method="Radau", rtol=1e-13, atol=1e-15,
                        jac=jac, dense_output=True).sol
        worst = max(np.max(np.abs(cube_solution(eps, t) - ref(t))) for t in np.linspace(0.5, 10, 96))
        print(f"  eps {eps:g}: largest absolute difference {worst:.1e}")

    print("Kaps, 25 end times in [0.5, 5] (as kaps_problem_ends_within_its_tolerance_at_every_end_time):")
    for eps in [1e-6, 1e-9]:
        f, jac = kaps(eps)
        for rtol, atol in [(1e-3, 1e-6), (1e-4, 1e-7)]:
            for method in ["Radau", "BDF"]:
                worst = 0.0
                for k in range(25):
                    t1 = 0.5 + 4.5 * k / 24
                    s = solve_ivp(f, (0, t1), [1.0, 1.0], method=method, rtol=rtol, atol=atol)
                    worst = max(worst, norm(s.y[:, -1], [np.exp(-2 * t1), np.exp(-t1)], rtol, atol))
                print(f"  {method:5} eps {eps:g} rtol {rtol:g}: largest error {worst:.3f} tolerances")

    print("The cube, 20 end times in [0.5, 10] (as a_stiff_component_that_follows_a_cube_ends_within_its_tolerance):")
    for eps in [1e-5, 1e-8]:
        f, jac = cube(eps)
        for rtol, atol in [(1e-3, 1e-6), (1e-5, 1e-8)]:
            for method in ["Radau", "BDF"]:
                worst = 0.0
                for k in range(20):
                    t1 = 0.5 + 9.5 * k / 19
                    s = solve_ivp(f, (0, t1), [0.0, 1.5, 1.0], method=method, rtol=rtol, atol=atol)
                    worst = max(worst, norm(s.y[:, -1], cube_solution(eps, t1), rtol, atol))
                print(f"  {method:5} eps {eps:g} rtol {rtol:g}: largest error {worst:.3f} tolerances")


main()
