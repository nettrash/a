"""Reference data of the linear-programming tests (`../lp.rs`), by `scipy.optimize.linprog`
with `method="highs"`.

Run from this directory with `python3 gen_lp.py`; it writes `lp_highs.txt`. Everything is
deterministic: a case is a seed, and the linear programme of a seed is built by `generate`, which
the test mirrors statement for statement (`generate` in `../lp.rs`). The data of a programme are
integers, so both sides hold exactly the same numbers; a checksum of them is stored with the
reference so that the test notices if the two generators ever drift apart.

A line of the file is

    seed n m_le m_eq checksum status objective

where `checksum` is 16 hexadecimal digits, `status` is `O` (optimal), `I` (infeasible) or `U`
(unbounded) as HiGHS classified the programme, and `objective` is the 16 hexadecimal digits of
the bits of the optimal objective as an f64 (`-` when there is none). For a maximisation the
objective is the maximum.

Checked here before a case is written: HiGHS's dual simplex and its interior point method with
crossover agree on the status, and on the objective to 1e-9 relative; and the status is the one
the construction of the case intends, where it intends one. A seed on which they disagree is not
written but reported. (HiGHS runs with its default tolerances: with its feasibility tolerances
tightened to 1e-10 it misjudged the status of seven of the first six hundred seeds, all of them
programmes whose rows and columns are scaled by powers of ten.)

Tools: Python 3.13, NumPy 2.3, SciPy 1.18.
"""

import struct
import sys

import numpy as np
from scipy.optimize import linprog

MASK = (1 << 64) - 1
INF_CODE = 1 << 40


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


class Problem:
    """minimise (or maximise) c.x subject to a_le.x <= b_le, a_eq.x = b_eq, lower <= x <= upper;
    a bound that is None is infinite. `intent` is the status the construction aims at: 'O', 'I',
    'U' or '?' when it aims at none."""

    def __init__(self):
        self.n = 0
        self.maximize = False
        self.c = []
        self.a_le = []
        self.b_le = []
        self.a_eq = []
        self.b_eq = []
        self.lower = []
        self.upper = []
        self.intent = "?"

    def checksum(self):
        """A word-wise FNV-1a hash of all the data, in a fixed order."""
        h = 0xCBF29CE484222325

        def mix(v):
            nonlocal h
            h = ((h ^ (v & MASK)) * 0x100000001B3) & MASK

        mix(self.n)
        mix(len(self.b_le))
        mix(len(self.b_eq))
        mix(1 if self.maximize else 0)
        for v in self.c:
            mix(v)
        for row in self.a_le:
            for v in row:
                mix(v)
        for v in self.b_le:
            mix(v)
        for row in self.a_eq:
            for v in row:
                mix(v)
        for v in self.b_eq:
            mix(v)
        for lo, hi in zip(self.lower, self.upper):
            mix(-INF_CODE if lo is None else lo)
            mix(INF_CODE if hi is None else hi)
        return h


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def generate(seed):
    """The linear programme of a seed. Mirrored by `generate` in `../lp.rs`: keep the order of
    the draws from the generator identical in both."""
    r = Rng(seed)
    p = Problem()
    # 0..3 are feasible and bounded by construction (1 a degenerate vertex, 2 a degenerate
    # dual), 4 is infeasible by construction, 5 unbounded by construction, 6 and 7 are
    # unstructured and may be anything.
    kind = r.below(8)
    size = r.below(10)
    if size < 6:
        n = r.between(1, 10)
        m_le = r.between(0, 8)
        m_eq = r.between(0, 4)
    elif size < 9:
        n = r.between(8, 40)
        m_le = r.between(0, 25)
        m_eq = r.between(0, 10)
    else:
        n = r.between(40, 120)
        m_le = r.between(10, 45)
        m_eq = r.between(0, 15)
    density = (20, 50, 100)[r.below(3)]
    wide = r.below(5) == 0
    mix = r.below(4)
    p.n = n
    p.maximize = r.below(4) == 0

    # The bounds, and a point x0 inside them.
    x0 = []
    for _ in range(n):
        if mix == 0:
            bound = 0
        elif mix == 2:
            bound = 1
        elif mix == 3:
            bound = 4 if r.below(4) != 0 else r.below(6)
        else:
            bound = r.below(6)
        if bound == 0:
            lo, hi = 0, None
        elif bound == 1:
            lo, hi = None, None
        elif bound == 2:
            lo, hi = r.between(-10, 10), None
        elif bound == 3:
            lo, hi = None, r.between(-10, 10)
        elif bound == 4:
            lo = r.between(-10, 10)
            hi = lo + r.between(1, 12)
        else:
            lo = r.between(-10, 10)
            hi = lo
        if lo is not None and hi is not None:
            t = r.below(4)
            if t == 0 or kind == 1:
                x = lo
            elif t == 1:
                x = hi
            else:
                x = lo + r.below(hi - lo + 1)
        elif lo is not None:
            x = lo if (r.below(2) == 0 or kind == 1) else lo + r.between(1, 6)
        elif hi is not None:
            x = hi if (r.below(2) == 0 or kind == 1) else hi - r.between(1, 6)
        else:
            x = r.between(-6, 6)
        p.lower.append(lo)
        p.upper.append(hi)
        x0.append(x)

    # The multipliers of the columns and of the rows: a wide problem has some of each a few
    # powers of ten larger than the rest.
    column_scale = [(1, 1, 1, 10, 100)[r.below(5)] if wide else 1 for _ in range(n)]
    row_scale = [(1, 1, 1, 10, 1000)[r.below(5)] if wide else 1 for _ in range(m_le + m_eq)]

    rows = []
    for i in range(m_le + m_eq):
        row = []
        for j in range(n):
            v = r.between(-9, 9) if r.below(100) < density else 0
            row.append(v * column_scale[j] * row_scale[i])
        rows.append(row)

    # An unbounded programme has a ray: a few variables that can run to infinity together, along
    # which no inequality grows, every equation stays put and the cost falls.
    ray = [0] * n
    if kind == 5:
        open_ended = [j for j in range(n) if p.lower[j] is None or p.upper[j] is None]
        if not open_ended:
            p.lower[0] = None
            p.upper[0] = None
            x0[0] = 0
            open_ended = [0]
        count = min(len(open_ended), r.between(1, 3))
        start = r.below(len(open_ended))
        support = [open_ended[(start + k) % len(open_ended)] for k in range(count)]
        for j in support:
            if p.upper[j] is None and p.lower[j] is None:
                ray[j] = 1 if r.below(2) == 0 else -1
            elif p.upper[j] is None:
                ray[j] = 1
            else:
                ray[j] = -1
        first = support[0]
        for i in range(m_le):
            if dot(rows[i], ray) > 0:
                for j in support:
                    rows[i][j] = -rows[i][j]
        for i in range(m_le, m_le + m_eq):
            rest = dot(rows[i], ray) - rows[i][first] * ray[first]
            rows[i][first] = -ray[first] * rest

    # A redundant equation: the last one becomes a combination of the first two.
    if m_eq >= 3 and r.below(4) == 0:
        f1 = r.between(-2, 2)
        f2 = r.between(-2, 2)
        e = m_le
        rows[e + m_eq - 1] = [f1 * u + f2 * v for u, v in zip(rows[e], rows[e + 1])]

    # The right-hand sides: through x0 when the programme is to be feasible.
    rhs = []
    for i in range(m_le + m_eq):
        at = dot(rows[i], x0)
        if kind >= 6:
            if i < m_le:
                rhs.append(r.between(-20, 60) * row_scale[i])
            elif r.below(2) == 0:
                rhs.append(at)
            else:
                rhs.append(r.between(-10, 10) * row_scale[i])
        elif i < m_le:
            slack = 0 if (r.below(3) == 0 or kind == 1) else r.between(1, 20)
            rhs.append(at + slack * row_scale[i])
        else:
            rhs.append(at)

    # The costs. A bounded programme has a feasible dual: multipliers of the right signs.
    cost = []
    if kind <= 4:
        y = []
        for i in range(m_le + m_eq):
            if i < m_le:
                y.append(-r.between(0, 4) if r.below(3) != 0 else 0)
            else:
                y.append(r.between(-4, 4))
        for j in range(n):
            lo, hi = p.lower[j], p.upper[j]
            if kind == 2 and r.below(4) != 0:
                d = 0
            elif lo is None and hi is None:
                d = 0
            elif hi is None:
                d = r.between(0, 4)
            elif lo is None:
                d = -r.between(0, 4)
            else:
                d = r.between(-4, 4)
            cost.append(d * column_scale[j] + sum(y[i] * rows[i][j] for i in range(m_le + m_eq)))
    else:
        for j in range(n):
            cost.append(r.between(-9, 9) * column_scale[j])
        if kind == 5:
            along = dot(cost, ray)
            if along >= 0:
                first = [j for j in range(n) if ray[j] != 0][0]
                cost[first] -= ray[first] * (along + r.between(1, 5))

    p.a_le = rows[:m_le]
    p.b_le = rhs[:m_le]
    p.a_eq = rows[m_le:]
    p.b_eq = rhs[m_le:]

    # An infeasible programme: a feasible one and a constraint that contradicts it.
    if kind == 4:
        how = r.below(3)
        if how == 0 and m_le >= 1:
            weights = [r.between(0, 2) for _ in range(m_le)]
            if sum(weights) == 0:
                weights[0] = 1
            row = [-sum(weights[i] * p.a_le[i][j] for i in range(m_le)) for j in range(n)]
            p.a_le.append(row)
            p.b_le.append(-dot(weights, p.b_le) - r.between(1, 5))
        elif how == 1 and m_eq >= 1:
            i = r.below(m_eq)
            p.a_eq.append([2 * v for v in p.a_eq[i]])
            p.b_eq.append(2 * p.b_eq[i] + r.between(1, 5))
        else:
            j = r.below(n)
            unit = [1 if k == j else 0 for k in range(n)]
            minus = [-v for v in unit]
            if p.lower[j] is not None:
                p.a_le.append(unit)
                p.b_le.append(p.lower[j] - r.between(1, 3))
            elif p.upper[j] is not None:
                p.a_le.append(minus)
                p.b_le.append(-p.upper[j] - r.between(1, 3))
            else:
                p.a_le.append(unit)
                p.b_le.append(0)
                p.a_le.append(minus)
                p.b_le.append(-r.between(1, 3))

    p.c = [-v for v in cost] if p.maximize else cost
    p.intent = "O" if kind <= 3 else "I" if kind == 4 else "U" if kind == 5 else "?"
    return p


STATUS = {0: "O", 2: "I", 3: "U"}


def highs(p, method, presolve=True):
    """HiGHS's verdict on the programme, as a minimisation: (status letter, minimum)."""
    c = [-v for v in p.c] if p.maximize else p.c
    res = linprog(
        c,
        A_ub=np.array(p.a_le, dtype=float).reshape(len(p.b_le), p.n) if p.b_le else None,
        b_ub=p.b_le or None,
        A_eq=np.array(p.a_eq, dtype=float).reshape(len(p.b_eq), p.n) if p.b_eq else None,
        b_eq=p.b_eq or None,
        bounds=list(zip(p.lower, p.upper)),
        method=method,
        options={"presolve": presolve},
    )
    return STATUS.get(res.status, "?"), res.fun


def reference(p):
    """The status and the objective of the programme (the maximum of a maximisation), or the
    reason there is no trustworthy reference."""
    status, fun = highs(p, "highs-ds")
    if status == "?":
        status, fun = highs(p, "highs-ds", presolve=False)
    other, other_fun = highs(p, "highs-ipm")
    if other == "?":
        other, other_fun = highs(p, "highs-ipm", presolve=False)
    if status == "?" or status != other:
        return None, f"dual simplex says {status}, interior point {other}"
    if p.intent != "?" and status != p.intent:
        return None, f"built to be {p.intent}, HiGHS says {status}"
    if status != "O":
        return (status, None), ""
    if abs(fun - other_fun) > 1e-9 * max(1.0, abs(fun)):
        return None, f"objectives differ: {fun!r} against {other_fun!r}"
    return (status, -fun if p.maximize else fun), ""


def bits(x):
    return "%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


def main():
    cases = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    lines = []
    count = {"O": 0, "I": 0, "U": 0}
    skipped = []
    for seed in range(1, cases + 1):
        p = generate(seed)
        ref, why = reference(p)
        if ref is None:
            skipped.append((seed, why))
            continue
        status, objective = ref
        count[status] += 1
        lines.append(
            "%d %d %d %d %016x %s %s"
            % (
                seed,
                p.n,
                len(p.b_le),
                len(p.b_eq),
                p.checksum(),
                status,
                "-" if objective is None else bits(objective),
            )
        )
    with open("lp_highs.txt", "w") as f:
        f.write("# Linear programmes by seed and what scipy.optimize.linprog(method=\"highs\") says of\n")
        f.write("# them; made by gen_lp.py, which describes the columns:\n")
        f.write("# seed n m_le m_eq checksum status objective\n")
        f.write("\n".join(lines) + "\n")
    print("written:", len(lines), count, "skipped:", skipped)


if __name__ == "__main__":
    main()
