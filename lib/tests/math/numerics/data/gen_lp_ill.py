"""Exact reference data of the ill-conditioned linear programmes of the tests (`../lp.rs`).

Run from this directory with `python3 gen_lp_ill.py`; it writes `lp_ill_exact.txt`. A case is a
seed; the programme of a seed is built by `generate`, which `generate_ill` in `../lp.rs` mirrors
statement for statement. Every number of a programme is a dyadic rational with a few bits (an
integer over 8, 16 or 2^30 at most), so the floats both sides hold are the same exact numbers,
and every sum the construction forms is exact in binary.

Each programme is feasible and bounded by construction, with a pair of equations that differ by
a small power of two `delta` in one coefficient: `e1·x = b1` and `(e1 + delta·u_k)·x = b2`. In
exact arithmetic the pair fixes `x_k = (b2 − b1)/delta` and is otherwise one equation, so the
programme is well posed, but its bases have a condition of about `1/delta` and its multipliers
are as large: the costs carry a pull on `x_k` that only the pair can balance. These are the
programmes on which a tolerance on the reduced costs relative to the largest multiplier took real
reduced costs for zero (an optimum 1e-4 to 1e-2 off), and on which phase one stopped early and a
Farkas certificate dropped terms that were not negligible (a feasible programme called
infeasible).

The reference of a seed is its exact optimum: the programme's numbers as fractions, rewritten in
the form `sympy.solvers.simplex.linprog` takes (every variable at least zero; it keeps that bound
whatever else it is told, so the bounds are handled here), and solved by its rational simplex
method. The optimum is checked here, whatever that solver did, by the exactly feasible point
it returns and by an exactly feasible point of the dual programme with the same objective, the
multipliers complementary slackness with that point asks for (found by phase one of the simplex
method with Bland's rule, in exact arithmetic, `exact_dual_bound`): by weak duality no feasible
point does better. A line of the file is

    seed n m_le m_eq checksum status objective

with `status` `O` (every case is optimal by construction; the script checks it) and `objective`
the 16 hexadecimal digits of the bits of the f64 nearest to the exact optimum.

Tools: Python 3.13, SymPy 1.14.
"""

import struct
from fractions import Fraction as F

from sympy import Matrix, Rational
from sympy.solvers.simplex import linprog

MASK = (1 << 64) - 1
INF_CODE = 1 << 40
SEEDS = range(1, 301)


class Rng:
    """SplitMix64 (Steele, Lea and Flood, 2014), on integers only."""

    def __init__(self, seed):
        self.state = seed & MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, n):
        """An integer in 0..n-1."""
        return self.next() % n

    def between(self, lo, hi):
        """An integer in lo..hi, both included."""
        return lo + self.below(hi - lo + 1)


def generate(seed):
    """The programme of a seed, in exact fractions: (c, a_le, b_le, a_eq, b_eq, lower, upper),
    a bound that is None being infinite. A minimisation."""
    r = Rng(seed)
    n = r.between(3, 8)
    m_le = r.between(0, 5)
    m_eq = r.between(0, 2)
    delta = F(1, 2 ** [14, 20, 26][r.below(3)])
    # The bounds, integers, and a point x0 in halves inside them.
    lower, upper, x0 = [], [], []
    for _ in range(n):
        kind = r.below(5)
        if kind == 0:
            lo, hi = 0, None
        elif kind == 1:
            lo, hi = None, None
        elif kind == 2:
            lo, hi = r.between(-8, 8), None
        elif kind == 3:
            lo, hi = None, r.between(-8, 8)
        else:
            lo = r.between(-8, 8)
            hi = lo + r.between(1, 10)
        if lo is not None and hi is not None:
            x = 2 * lo + r.below(2 * (hi - lo) + 1)
        elif lo is not None:
            x = 2 * lo + r.below(13)
        elif hi is not None:
            x = 2 * hi - r.below(13)
        else:
            x = r.between(-12, 12)
        lower.append(lo)
        upper.append(hi)
        x0.append(F(x, 2))
    # The ordinary rows, in eighths.
    rows = []
    for _ in range(m_le + m_eq):
        rows.append([F(r.between(-16, 16), 8) if r.below(100) < 60 else F(0) for _ in range(n)])
    # The pair: e1 on a support that holds k, e2 = e1 + delta·u_k.
    k = r.below(n)
    e1 = []
    for j in range(n):
        if j == k:
            v = r.between(1, 16) * (1 if r.below(2) == 0 else -1)
            e1.append(F(v, 8))
        elif r.below(2) == 0:
            e1.append(F(r.between(-16, 16), 8))
        else:
            e1.append(F(0))
    e2 = list(e1)
    e2[k] += delta
    # Right-hand sides through x0; the inequalities with a slack in halves, a third of them none.
    def at(row):
        return sum(a * x for a, x in zip(row, x0))
    b_le = []
    for i in range(m_le):
        slack = 0 if r.below(3) == 0 else r.between(1, 8)
        b_le.append(at(rows[i]) + F(slack, 2))
    a_eq = rows[m_le:] + [e1, e2]
    b_eq = [at(row) for row in a_eq]
    a_le = rows[:m_le]
    # The costs: a feasible dual (multipliers in quarters of the right signs, reduced costs of the
    # signs their bounds ask for), and then a pull on x_k that only the pair can balance.
    y = [F(-r.between(0, 8), 4) if r.below(3) != 0 else F(0) for _ in range(m_le)]
    y += [F(r.between(-8, 8), 4) for _ in range(len(a_eq))]
    every = a_le + a_eq
    c = []
    for j in range(n):
        lo, hi = lower[j], upper[j]
        if lo is None and hi is None:
            d = 0
        elif hi is None:
            d = r.between(0, 8)
        elif lo is None:
            d = -r.between(0, 8)
        else:
            d = r.between(-8, 8)
        c.append(F(d, 4) + sum(y[i] * every[i][j] for i in range(len(every))))
    c[k] += F(r.between(1, 16) * (1 if r.below(2) == 0 else -1), 8)
    return c, a_le, b_le, a_eq, b_eq, lower, upper


def checksum(case):
    """The word-wise FNV-1a hash of the bits of the f64 data, as `../lp.rs` computes it."""
    c, a_le, b_le, a_eq, b_eq, lower, upper = case
    h = 0xCBF29CE484222325

    def mix(v):
        nonlocal h
        h = ((h ^ (v & MASK)) * 0x100000001B3) & MASK

    def bits(x):
        return struct.unpack("<Q", struct.pack("<d", float(x)))[0]

    mix(len(c))
    mix(len(b_le))
    mix(len(b_eq))
    for v in c:
        mix(bits(v))
    for row in a_le:
        for v in row:
            mix(bits(v))
    for v in b_le:
        mix(bits(v))
    for row in a_eq:
        for v in row:
            mix(bits(v))
    for v in b_eq:
        mix(bits(v))
    for lo, hi in zip(lower, upper):
        mix(-INF_CODE if lo is None else lo)
        mix(INF_CODE if hi is None else hi)
    return h


def exact_optimum(case):
    """The exact optimum and an optimal point, by sympy's rational simplex method on the
    programme rewritten with every variable at least zero: x = l + x' for a finite lower bound
    (and the row x' <= u − l for a finite upper bound too), x = u − x' for an upper bound only,
    x = x+ − x− for a free variable."""
    c, a_le, b_le, a_eq, b_eq, lower, upper = case
    # Floats, where a caller has them, are taken for the exact numbers they are.
    c = [F(v) for v in c]
    a_le = [[F(v) for v in row] for row in a_le]
    b_le = [F(v) for v in b_le]
    a_eq = [[F(v) for v in row] for row in a_eq]
    b_eq = [F(v) for v in b_eq]
    n = len(c)
    columns, offset, caps = [], [], []
    width = 0
    for j in range(n):
        lo, hi = lower[j], upper[j]
        if lo is not None:
            columns.append([(width, 1)])
            offset.append(F(lo))
            if hi is not None:
                caps.append((width, F(hi) - F(lo)))
            width += 1
        elif hi is not None:
            columns.append([(width, -1)])
            offset.append(F(hi))
            width += 1
        else:
            columns.append([(width, 1), (width + 1, -1)])
            offset.append(F(0))
            width += 2

    def rewrite(rows, rhs):
        out_rows, out_rhs = [], []
        for row, b in zip(rows, rhs):
            new = [F(0)] * width
            for j, a in enumerate(row):
                for idx, sign in columns[j]:
                    new[idx] += a * sign
            out_rows.append(new)
            out_rhs.append(b - sum(a * o for a, o in zip(row, offset)))
        return out_rows, out_rhs

    le_rows, le_rhs = rewrite(a_le, b_le)
    for idx, cap in caps:
        row = [F(0)] * width
        row[idx] = F(1)
        le_rows.append(row)
        le_rhs.append(cap)
    if not le_rows:
        # sympy's simplex method wants an inequality: 0 <= 1.
        le_rows.append([F(0)] * width)
        le_rhs.append(F(1))
    eq_rows, eq_rhs = rewrite(a_eq, b_eq)
    cost = [F(0)] * width
    for j in range(n):
        for idx, sign in columns[j]:
            cost[idx] += c[j] * sign
    constant = sum(cj * o for cj, o in zip(c, offset))

    def rat(v):
        return Rational(v.numerator, v.denominator)

    value, point = linprog(Matrix([rat(v) for v in cost]),
                           Matrix([[rat(v) for v in row] for row in le_rows]),
                           Matrix([rat(v) for v in le_rhs]),
                           Matrix([[rat(v) for v in row] for row in eq_rows]),
                           Matrix([rat(v) for v in eq_rhs]))
    point = [F(int(Rational(v).p), int(Rational(v).q)) for v in point]
    x = [offset[j] + sum(sign * point[idx] for idx, sign in columns[j]) for j in range(n)]
    value = F(int(Rational(value).p), int(Rational(value).q)) + constant
    return value, x


def exactly_feasible(case, x):
    c, a_le, b_le, a_eq, b_eq, lower, upper = case
    inside = all((lo is None or v >= lo) and (hi is None or v <= hi)
                 for v, lo, hi in zip(x, lower, upper))
    rows = all(sum(a * v for a, v in zip(row, x)) <= b for row, b in zip(a_le, b_le))
    equations = all(sum(a * v for a, v in zip(row, x)) == b for row, b in zip(a_eq, b_eq))
    return inside and rows and equations


def nonnegative_solution(matrix, rhs):
    """A vector v >= 0 with matrix·v = rhs, or None when there is none: phase one of the simplex
    method on the tableau, an artificial variable for each row, with Bland's rule (the entering
    column of the smallest index, of the tied rows the basic variable of the smallest index), so
    that it cannot cycle; in exact arithmetic."""
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    tableau = []
    for i in range(rows):
        sign = -1 if rhs[i] < 0 else 1
        tableau.append([sign * a for a in matrix[i]] + [F(int(i == k)) for k in range(rows)]
                       + [sign * rhs[i]])
    basis = [cols + i for i in range(rows)]
    while True:
        # The reduced costs of phase one: cost one on the artificial variables, zero elsewhere.
        def reduced(k):
            own = F(int(k >= cols))
            return own - sum(tableau[i][k] for i in range(rows) if basis[i] >= cols)
        entering = next((k for k in range(cols + rows) if k not in basis and reduced(k) < 0),
                        None)
        if entering is None:
            break
        _, _, row = min((tableau[i][-1] / tableau[i][entering], basis[i], i)
                        for i in range(rows) if tableau[i][entering] > 0)
        pivot = tableau[row][entering]
        tableau[row] = [v / pivot for v in tableau[row]]
        for i in range(rows):
            if i != row and tableau[i][entering] != 0:
                f = tableau[i][entering]
                tableau[i] = [a - f * b for a, b in zip(tableau[i], tableau[row])]
        basis[row] = entering
    if any(basis[i] >= cols and tableau[i][-1] != 0 for i in range(rows)):
        return None
    v = [F(0)] * cols
    for i in range(rows):
        if basis[i] < cols:
            v[basis[i]] = tableau[i][-1]
    return v


def exact_dual_bound(case, x):
    """The objective of an exactly feasible point of the dual programme that is complementary
    to the point x, which certifies that no feasible point has a smaller objective.

    The dual of minimise c.x subject to A_le x <= b_le, A_eq x = b_eq, l <= x <= u is: maximise
    b_le.y + b_eq.w + l.p - u.q subject to A_le^T y + A_eq^T w + p - q = c, y <= 0, p, q >= 0
    (p_j only where l_j is finite, q_j where u_j is). Complementary slackness with x leaves y_i
    only on the inequalities x makes tight, and p_j, q_j only where x_j is at that bound; the
    multipliers are found by `nonnegative_solution` (with y = -y', w = w+ - w-), and the dual
    constraints and objective are then checked here, term by term, in exact arithmetic."""
    c, a_le, b_le, a_eq, b_eq, lower, upper = case
    c = [F(v) for v in c]
    a_le = [[F(v) for v in row] for row in a_le]
    b_le = [F(v) for v in b_le]
    a_eq = [[F(v) for v in row] for row in a_eq]
    b_eq = [F(v) for v in b_eq]
    n = len(c)
    # Each multiplier: its kind, its row or variable, and the sign its column enters with.
    parts = []
    for i, (row, b) in enumerate(zip(a_le, b_le)):
        if sum(a * v for a, v in zip(row, x)) == b:
            parts.append(("y", i, -1))
    for i in range(len(b_eq)):
        parts += [("w", i, 1), ("w", i, -1)]
    for j in range(n):
        if lower[j] is not None and x[j] == lower[j]:
            parts.append(("p", j, 1))
        if upper[j] is not None and x[j] == upper[j]:
            parts.append(("q", j, -1))

    def column(kind, i, sign):
        if kind == "y":
            return [sign * a for a in a_le[i]]
        if kind == "w":
            return [sign * a for a in a_eq[i]]
        return [F(sign * int(k == i)) for k in range(n)]

    columns = [column(*part) for part in parts]
    matrix = [[columns[k][j] for k in range(len(parts))] for j in range(n)]
    v = nonnegative_solution(matrix, c)
    assert v is not None, "no dual point is complementary to the point: it is not optimal"
    y = [F(0)] * len(b_le)
    w = [F(0)] * len(b_eq)
    p = [F(0)] * n
    q = [F(0)] * n
    for (kind, i, sign), value in zip(parts, v):
        if kind == "y":
            y[i] += sign * value
        elif kind == "w":
            w[i] += sign * value
        elif kind == "p":
            p[i] += value
        else:
            q[i] += value
    # The dual constraints, each checked exactly.
    assert all(v <= 0 for v in y) and all(v >= 0 for v in p + q)
    assert all(p[j] == 0 for j in range(n) if lower[j] is None)
    assert all(q[j] == 0 for j in range(n) if upper[j] is None)
    for j in range(n):
        share = (sum(a_le[i][j] * y[i] for i in range(len(y)))
                 + sum(a_eq[i][j] * w[i] for i in range(len(w))) + p[j] - q[j])
        assert share == c[j]
    return (sum(b * v for b, v in zip(b_le, y)) + sum(b * v for b, v in zip(b_eq, w))
            + sum(F(lower[j]) * p[j] for j in range(n) if lower[j] is not None)
            - sum(F(upper[j]) * q[j] for j in range(n) if upper[j] is not None))


def main():
    lines = ["# seed n m_le m_eq checksum status objective (exact optimum, f64 bits)"]
    for seed in SEEDS:
        case = generate(seed)
        # Every number is an exact f64.
        for v in [*case[0], *case[2], *case[4]] + [a for row in case[1] + case[3] for a in row]:
            assert F(float(v)) == v, (seed, v)
        value, x = exact_optimum(case)
        assert exactly_feasible(case, x), seed
        assert exact_dual_bound(case, x) == value, seed
        bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
        lines.append(f"{seed} {len(case[0])} {len(case[2])} {len(case[4])} "
                     f"{checksum(case):016x} O {bits:016x}")
    with open("lp_ill_exact.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines) - 1} cases")


if __name__ == "__main__":
    main()
