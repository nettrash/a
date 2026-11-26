"""Reference values of the named linear programmes of the tests (`../lp.rs`): the textbook
examples, the examples of cycling, the cubes of Klee and Minty, an assignment and a transportation
problem.

Run from anywhere with `python3 gen_lp_examples.py`; it prints, for each programme, what
`scipy.optimize.linprog(method="highs")` says of it (the point, the optimum and the dual values in
the sign convention of the tests: the rates of change of the optimum, so that a maximisation has
the signs of the minimisation reversed), and what the simplex method does with it in exact
rational arithmetic under the pivoting rules the tests name. Nothing is written; the tests carry
the values as constants.

The exact simplex method below works on the tableau of `minimise c.x subject to A x <= b, x >= 0`
with `b >= 0`, the slack variables the first basis. Its rules:

* 'first': the entering variable of the most negative reduced cost, the lowest index on ties;
  of the rows of the minimum ratio the lowest (the textbook rule);
* 'basisindex': the same entering variable; of the tied rows, the one whose basic variable has
  the lowest index;
* 'largest': the same entering variable; of the tied rows, the one with the largest pivot, the
  lowest basic index on equal pivots (what a Harris ratio test does with exact ties);
* 'bland': the eligible variable of the lowest index enters, the tied row of the lowest basic
  index leaves (Bland, 1977).

It reports 'cycle' when a basis comes back, with the length of the cycle.

The last two sections hold the programmes of the first and the second review of the solver, on
which it once went wrong: their exact optima, by the rational simplex method of `gen_lp_ill.py`
(`exact_optimum`), with what HiGHS makes of them.

Tools: Python 3.13, NumPy 2.3, SciPy 1.18, SymPy 1.14.
"""

import sys
from fractions import Fraction as F

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog

# The sibling module is imported for its exact solver; it leaves no bytecode beside the data.
sys.dont_write_bytecode = True
from gen_lp_ill import exact_optimum  # noqa: E402


def exact_simplex(c, a, b, rule, limit=20000):
    """(outcome, pivots, cycle length or None, optimum or None) of the tableau method."""
    m, n = len(a), len(c)
    rows = [[F(v) for v in a[i]] + [F(int(i == k)) for k in range(m)] + [F(b[i])] for i in range(m)]
    z = [F(v) for v in c] + [F(0)] * (m + 1)
    basis = list(range(n, n + m))
    seen = {}
    for pivots in range(limit):
        key = tuple(basis)
        if key in seen:
            return "cycle", pivots, pivots - seen[key], None
        seen[key] = pivots
        eligible = [j for j in range(n + m) if z[j] < 0]
        if not eligible:
            return "optimal", pivots, None, -z[-1]
        q = min(eligible) if rule == "bland" else min(eligible, key=lambda j: (z[j], j))
        candidates = [i for i in range(m) if rows[i][q] > 0]
        if not candidates:
            return "unbounded", pivots, None, None
        least = min(rows[i][-1] / rows[i][q] for i in candidates)
        ties = [i for i in candidates if rows[i][-1] / rows[i][q] == least]
        if rule == "first":
            r = min(ties)
        elif rule in ("basisindex", "bland"):
            r = min(ties, key=lambda i: basis[i])
        else:
            r = min(ties, key=lambda i: (-rows[i][q], basis[i]))
        pivot = rows[r][q]
        rows[r] = [v / pivot for v in rows[r]]
        for i in range(m):
            if i != r and rows[i][q] != 0:
                f = rows[i][q]
                rows[i] = [v - f * w for v, w in zip(rows[i], rows[r])]
        f = z[q]
        z = [v - f * w for v, w in zip(z, rows[r])]
        basis[r] = q
    return "limit", limit, None, None


def highs(name, c, a_ub=None, b_ub=None, a_eq=None, b_eq=None, bounds=(0, None), maximize=False):
    """Prints HiGHS's answer with the dual values in the tests' convention."""
    sign = -1.0 if maximize else 1.0
    r = linprog([sign * v for v in c], A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds,
                method="highs")
    status = {0: "optimal", 2: "infeasible", 3: "unbounded"}.get(r.status, r.message)
    print(f"{name}: {status}")
    if r.status == 0:
        print("   x         ", [float(v) for v in r.x])
        print("   objective ", float(sign * r.fun))
        if a_ub is not None:
            print("   dual_le   ", [float(sign * v) for v in r.ineqlin.marginals])
        if a_eq is not None:
            print("   dual_eq   ", [float(sign * v) for v in r.eqlin.marginals])
        reduced = sign * (r.lower.marginals + r.upper.marginals)
        print("   reduced   ", [float(v) for v in reduced])


def cycling(name, c, a, b):
    """The exact simplex method under each rule."""
    for rule in ("first", "basisindex", "largest", "bland"):
        print(f"   exact {rule:10s}", exact_simplex(c, a, b, rule))


def klee_minty(n, beta, gamma):
    """The cube as a minimisation of the negated objective, in exact numbers."""
    c = [-F(beta) ** (n - j) for j in range(1, n + 1)]
    a = [[2 * F(beta) ** (i - j) if j < i else F(int(j == i)) for j in range(1, n + 1)]
         for i in range(1, n + 1)]
    b = [F(gamma) ** (i - 1) for i in range(1, n + 1)]
    return c, a, b


def main():
    print("== Textbook examples")
    highs("Wyndor glass (Hillier and Lieberman)", [3, 5], [[1, 0], [0, 2], [3, 2]], [4, 12, 18],
          maximize=True)
    highs("Chvatal ch. 2", [5, 4, 3], [[2, 3, 1], [4, 1, 2], [3, 4, 2]], [5, 11, 8], maximize=True)
    highs("Taha, two phases", [4, 1], [[-4, -3], [1, 2]], [-6, 4], [[3, 1]], [3])
    highs("Vanderbei ch. 2, infeasible origin", [-2, -1], [[-1, 1], [-1, -2], [0, 1]], [-1, -2, 1], maximize=True)
    highs("diet", [3, 4], [[-2, -1], [-1, -3]], [-8, -10])
    highs("free variables only", [1, 1], [[-1, -2], [-3, -1]], [-4, -6], bounds=(None, None))
    highs("equations only, free", [1, 1, 1], a_eq=[[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],
          b_eq=[-3, 5, 2], bounds=(None, None))
    highs("equations only, x >= 0", [1, 1, 1], a_eq=[[2, 1, -1], [-3, -1, 2], [-2, 1, 2]],
          b_eq=[-3, 5, 2])
    highs("redundant equations", [-1, 0, 1],
          a_eq=[[1, 1, 1], [1, -1, 0], [2, 0, 1], [2, 2, 2], [1, 1, 1]], b_eq=[4, 1, 5, 8, 4])

    print("== Cycling")
    half = F(1, 2)
    c = [-10, 57, 9, 24]
    a = [[half, -F(11, 2), -F(5, 2), 9], [half, -F(3, 2), -half, 1], [1, 0, 0, 0]]
    highs("Chvatal ch. 3 (maximise 10x1 - 57x2 - 9x3 - 24x4)", [10, -57, -9, -24],
          [[float(v) for v in row] for row in a], [0, 0, 1], maximize=True)
    cycling("Chvatal ch. 3", c, a, [0, 0, 1])
    c = [-F(3, 4), 150, -F(1, 50), 6]
    a = [[F(1, 4), -60, -F(1, 25), 9], [half, -90, -F(1, 50), 3], [0, 0, 1, 0]]
    highs("Beale (1955)", [float(v) for v in c], [[float(v) for v in row] for row in a], [0, 0, 1])
    cycling("Beale", c, a, [0, 0, 1])
    c = [-2, -3, 1, 12]
    a = [[-2, -9, 1, 9], [F(1, 3), 1, -F(1, 3), -2], [2, 3, -1, -12]]
    highs("Kuhn", c, [[float(v) for v in row] for row in a], [0, 0, 2])
    cycling("Kuhn", c, a, [0, 0, 2])
    c = [-F(23, 10), -F(215, 100), F(1355, 100), F(4, 10)]
    a = [[F(4, 10), F(2, 10), -F(14, 10), -F(2, 10)], [-F(78, 10), -F(14, 10), F(78, 10), F(4, 10)]]
    highs("Hall and McKinnon (2004)", [float(v) for v in c],
          [[float(v) for v in row] for row in a], [0, 0])
    cycling("Hall and McKinnon", c, a, [0, 0])

    print("== Klee and Minty: pivots of Dantzig's rule ('first') in exact arithmetic")
    for beta, gamma, largest in ((10, 100, 10), (2, 5, 13)):
        counts = []
        for n in range(1, largest + 1):
            outcome, pivots, _, optimum = exact_simplex(*klee_minty(n, beta, gamma), "first")
            assert outcome == "optimal" and optimum == -F(gamma) ** (n - 1)
            counts.append(pivots)
        print(f"   beta {beta}, gamma {gamma}: n = 1..{largest}:", counts,
              "= 2^n - 1:", counts == [2 ** n - 1 for n in range(1, largest + 1)])

    print("== Assignment and transportation")
    cost = np.array([[(7 * i + 3 * j * j + 5) % 17 + 1 for j in range(6)] for i in range(6)],
                    dtype=float)
    rows, cols = linear_sum_assignment(cost)
    print("assignment 6 x 6: cost matrix", cost.astype(int).tolist())
    print("   linear_sum_assignment optimum", float(cost[rows, cols].sum()), "columns", cols.tolist())
    a_eq = [[float(k // 6 == i) for k in range(36)] for i in range(6)]
    a_eq += [[float(k % 6 == j) for k in range(36)] for j in range(6)]
    highs("assignment as a linear programme", cost.flatten().tolist(), a_eq=a_eq, b_eq=[1] * 12)
    tcost = [8, 6, 10, 9, 9, 12, 13, 7, 14, 9, 16, 5]
    a_eq = [[float(k // 4 == i) for k in range(12)] for i in range(3)]
    a_eq += [[float(k % 4 == j) for k in range(12)] for j in range(4)]
    highs("transportation 3 x 4", tcost, a_eq=a_eq, b_eq=[20, 30, 25, 10, 35, 15, 15])

    print("== Review round 1: exact optima")
    review()
    print("== Review round 2: exact optima")
    review2()


def review():
    """The programmes of the first review: the exact optimum of their float data, and HiGHS's."""
    inf = None
    cases = {}
    # Two equations that differ by d in one coefficient make multipliers of about 1/d; x2, whose
    # cost is small, meets only rows whose multipliers are zero, and goes to its lower bound.
    for d, rows, cost in ((1e-4, 40, 3e-4), (1e-7, 1, 0.005)):
        cases[f"unrelated large multipliers, d = {d:g}, {rows} rows"] = (
            [0.0, -1.0, cost], [[0.0, 0.0, 1.0]] * rows, [5.0 + i for i in range(rows)],
            [[1.0, 1.0, 0.0], [1.0, 1.0 + d, 0.0]], [1.0, 1.0 + d / 2.0],
            [0.0, 0.0, -1.0], [10.0, 10.0, inf])
    # ill6-2767 of the review's probes: feasible, once called infeasible at the tolerance 1e-7.
    cases["large multipliers in phase one (ill6-2767)"] = (
        [0.033508399431287784, 4.289652038062165, -0.768514392365316, -1.819291862865434,
         1.493464830699932, 2.7316974948303416],
        [[0.4255195714642922, 0.048212952533530785, 0.0, -0.6172831581747933, 1.4325383293044864,
          0.0],
         [0.9972340243526487, 0.0, 0.0, 0.0, 0.0, -1.5216180946154498]],
        [-1.1029048003867024, -10.898904359367187],
        [[0.0, 0.538456371255046, -0.027411779201491782, 0.0, -0.7179867108968127, 0.0],
         [0.0, 0.761105535061727, 0.0, -0.9667587287579885, 0.7262146292722984, 0.0],
         [-0.6506197927973945, 2.147181031501968, 0.0, -0.34382307390282374, 0.6955285281444558,
          0.5887978417492359],
         [0.0, 1.5069348579334945, 0.0, 0.3433577578115057, 0.0, 0.0],
         [0.0, 1.5069358579334944, 0.0, 0.3433577578115057, 0.0, 0.0]],
        [-0.0015406304548472582, 1.9082869414388917, 6.967941612756002, -0.5728560274815098,
         -0.5728560274815098],
        [-6.728342167855769, 0.0, inf, -5.423615749675798, 0.0, 1.7143356018294535],
        [-6.381355342606776, inf, -8.577349726291384, inf, inf, inf])
    # Three consistent equations of rank two whose only feasible point is (0, s/r).
    p, q, r, t, s, lo1, c0 = (0.8898297483912715, 1.0335765890563122, 0.9507606806068835,
                              0.1082930293028078, 3.192013253322282, 1.6786628425172025,
                              0.29943152228274333)
    cases["rank-deficient equations, optimum 0"] = (
        [c0, 0.0], [], [], [[p, 0.0], [q, r], [q + t, r]], [0.0, s, s], [0.0, lo1], [inf, inf])
    for name, case in cases.items():
        value, x = exact_optimum(case)
        print(f"{name}: exact optimum {float(value)!r}")
        print("   x", [float(v) for v in x])
        c, a_le, b_le, a_eq, b_eq, lower, upper = case
        bounds = list(zip(lower, upper))
        for tol in (None, 1e-6):
            options = {} if tol is None else dict(primal_feasibility_tolerance=tol,
                                                  dual_feasibility_tolerance=tol)
            res = linprog(c, A_ub=a_le or None, b_ub=b_le or None, A_eq=a_eq, b_eq=b_eq,
                          bounds=bounds, method="highs", options=options)
            label = "HiGHS" if tol is None else f"HiGHS with tolerances {tol:g}"
            print(f"   {label}:", float(res.fun) if res.status == 0 else res.message)


def review2():
    """The programmes of the second review: one large number beside rows of moderate size. A
    variable x_new of cost zero carries it, as the slack of x_new <= big, the bound x_new >= big or
    the value fixed by x_new = big; none changes the optimum, which is that of the programme
    alone. Minimise 1000 x2 subject to x0 + x1 + x2 = 1, x0 <= 0.4, x1 <= 0.6 - g: the optimum is
    1000 (1 - 0.4 - fl(0.6 - g)), and in f32 the same of the data rounded to f32."""
    inf = None

    def forced(g, single=False):
        r = (lambda v: float(np.float32(v))) if single else (lambda v: v)
        return ([0.0, 0.0, 1000.0], [], [], [[1.0, 1.0, 1.0]], [1.0], [0.0, 0.0, 0.0],
                [r(0.4), r(0.6 - g), inf])

    def beside(case, big, large):
        c, a_le, b_le, a_eq, b_eq, lower, upper = case
        c = c + [0.0]
        a_le = [row + [0.0] for row in a_le]
        a_eq = [row + [0.0] for row in a_eq]
        lower, upper, b_le, b_eq = lower + [0.0], upper + [inf], list(b_le), list(b_eq)
        unit = [0.0] * (len(c) - 1) + [1.0]
        if large == "slack":
            a_le, b_le = a_le + [unit], b_le + [big]
        elif large == "bound":
            lower[-1] = big
        else:
            a_eq, b_eq = a_eq + [unit], b_eq + [big]
        return c, a_le, b_le, a_eq, b_eq, lower, upper

    for g, single in ((1e-5, False), (1e-8, False), (1e-2, True), (1e-3, True)):
        case = forced(g, single)
        value, x = exact_optimum(case)
        print(f"forced by one equation, g = {g:g}{' (f32 data)' if single else ''}: "
              f"exact optimum {float(value)!r}")
        print("   x", [float(v) for v in x])
        for big in (1e6, 1e12) if not single else (1e5,):
            for large in ("slack", "bound", "basic"):
                c, a_le, b_le, a_eq, b_eq, lower, upper = beside(case, big, large)
                res = linprog(c, A_ub=a_le or None, b_ub=b_le or None, A_eq=a_eq, b_eq=b_eq,
                              bounds=list(zip(lower, upper)), method="highs")
                print(f"   beside {big:g} ({large}): HiGHS",
                      float(res.fun) if res.status == 0 else res.message)
    # Unscaled, a row in numbers of 1e200 beside one in ones; the second equation fixes the
    # objective at 2.
    case = ([1.0, 1.0], [], [], [[1e200, 1e-200], [1.0, 1.0]], [1e200, 2.0], [0.0, 0.0],
            [inf, inf])
    value, x = exact_optimum(case)
    print(f"1e200 beside ones: exact optimum {float(value)!r}, x", [float(v) for v in x])


if __name__ == "__main__":
    main()
