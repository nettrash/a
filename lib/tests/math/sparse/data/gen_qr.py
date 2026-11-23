"""Reference least-squares solutions for the sparse QR tests, lib/tests/math/sparse/qr.rs.

The matrices and right-hand sides are built from integer formulas whose values are dyadic, so
they are exact in f32 and f64 alike and the Rust tests rebuild them bit for bit (the formulas
are `entry`, `rhs` and the cases below; qr.rs repeats them). For each case the script takes the
solution of numpy.linalg.lstsq, checks it against the solution of the normal equations in mpmath
at 60 digits (the exact least-squares solution to far more digits than a double holds) and against
scipy.sparse.linalg.lsqr, and prints the constants the test embeds: the solution, the norm of the
residual, the 2-norm condition number and the condition number of the least-squares problem,
kappa + kappa^2 * |r| / (|A| |x|), which bounds how far a backward-stable solution can be from the
exact one in units of the rounding. The rank-deficient cases give numpy.linalg.matrix_rank.

The singular matrices of the tests of the condition estimate come last: a 3 x 3 matrix with a
column 2^8 times the difference of two others (its rank by sympy in exact arithmetic), two
structurally rank-deficient matrices (scipy's structural_rank, numpy's matrix_rank) printed as
the constants the test embeds, and the condition numbers of the bidiagonal matrix of 1 and -2
from mpmath at 50 digits.

Run: python3 gen_qr.py   (numpy 2.3, scipy 1.18, mpmath 1.3, sympy 1.14)
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import mpmath as mp


def entry(case, i, j):
    """The element (i, j) of a case, as (re, im); (0, 0) is not stored."""
    m, n, seed, k = case["m"], case["n"], case["seed"], case["k"]
    if i == (j * m) // n:
        # A strong element in a row of its own for every column: full column rank.
        re = 4.0 + (j % 5) / 4.0
        im = ((j * 3 + seed) % 5 - 2) / 8.0 if case["complex"] else 0.0
        return re, im
    if (i * 73 + j * 151 + seed * 17) % k != 0:
        return 0.0, 0.0
    re = ((i * 31 + j * 57 + seed) % 15 - 7) / 4.0
    im = ((i * 43 + j * 29 + seed * 7) % 13 - 6) / 8.0 if case["complex"] else 0.0
    return re, im


def rhs(case, i):
    """The element i of the right-hand side of a case."""
    seed = case["seed"]
    re = ((i * 19 + seed * 3) % 11 - 5) / 2.0
    im = ((i * 23 + seed) % 7 - 3) / 4.0 if case["complex"] else 0.0
    return re, im


def matrix(case):
    m, n = case["m"], case["n"]
    a = np.zeros((m, n), dtype=complex)
    for i in range(m):
        for j in range(n):
            re, im = entry(case, i, j)
            a[i, j] = complex(re, im)
    # The dependent columns of the rank-deficient cases: `(to, from, other)` makes column `to`
    # the sum of columns `from` and `other`; `(to, from, None)` makes it column `from` times
    # 1 + i (times 1 for a real case); `(to, None, None)` empties it.
    for to, frm, other in case.get("dependent", []):
        if frm is None:
            a[:, to] = 0
        elif other is None:
            a[:, to] = a[:, frm] * (complex(1, 1) if case["complex"] else 1)
        else:
            a[:, to] = a[:, frm] + a[:, other]
    b = np.array([complex(*rhs(case, i)) for i in range(m)])
    if not case["complex"]:
        a, b = a.real.copy(), b.real.copy()
    return a, b


def smoothing(n):
    """[I; 2 D] with D the first differences, and the data c_i = ((37 i) % 9 - 4) / 4 against
    zeros: the least squares of a smoothed curve, m = 2 n - 1."""
    a = np.zeros((2 * n - 1, n))
    for i in range(n):
        a[i, i] = 1.0
    for i in range(n - 1):
        a[n + i, i] = -2.0
        a[n + i, i + 1] = 2.0
    b = np.zeros(2 * n - 1)
    for i in range(n):
        b[i] = ((37 * i) % 9 - 4) / 4.0
    return a, b


CASES = [
    dict(name="SQUARE", m=12, n=12, seed=1, k=4, complex=False),
    dict(name="TALL", m=36, n=12, seed=2, k=5, complex=False),
    dict(name="VERY_TALL", m=300, n=15, seed=3, k=7, complex=False),
    dict(name="COMPLEX_SQUARE", m=9, n=9, seed=4, k=3, complex=True),
    dict(name="COMPLEX_TALL", m=40, n=10, seed=5, k=4, complex=True),
    dict(name="COMPLEX_VERY_TALL", m=200, n=10, seed=6, k=6, complex=True),
]

DEFICIENT = [
    dict(name="DEFICIENT", m=20, n=8, seed=7, k=3, complex=False,
         dependent=[(5, 2, 3), (7, None, None)]),
    dict(name="COMPLEX_DEFICIENT", m=15, n=6, seed=8, k=3, complex=True,
         dependent=[(4, 1, None), (2, 0, None)]),
]


# Singular matrices whose R has no small diagonal element in some column order, found by the
# first review of the sparse QR: the decomposition without a condition estimate solved them.

# Column 2 is 2^8 (column 1 - column 0), exactly: the elements are dyadic.
LARGE_COMBINATION = [
    [-1.0, -1.0 + 2.0 / 256.0, 2.0],
    [-3.0, -3.0 + 3.0 / 256.0, 3.0],
    [2.0, 2.0 + 1.0 / 256.0, 1.0],
]

# 13 x 5; columns 0, 2 and 4 store only in rows 0 and 2.
STRUCTURAL = [
    (0, 0, -0.6446499814983495), (0, 1, 0.8182224590931129), (0, 2, 0.37690383650090253),
    (0, 3, -0.33473394512862487), (0, 4, 0.24606349249284443), (1, 3, -0.4514242697154671),
    (2, 0, -0.025679805834457836), (2, 3, -0.1183901520437265), (2, 4, 0.8645024825922223),
    (3, 1, -0.11385088727414328), (7, 3, -0.6009378343030822), (8, 1, -0.32981445750449323),
    (11, 1, 0.8476842129253435),
]

# 18 x 16 with a dense row.
DENSE_ROW = [
    (0, 8, 0.6023761272419847), (0, 12, 0.3783878659664561), (1, 11, -0.12420937473333371),
    (1, 14, 0.5548381517550194), (2, 10, 0.3257821505055558), (2, 12, 0.2772095944681834),
    (3, 3, -0.47051244568027917), (3, 4, -0.658919213762335), (4, 8, 0.9095475539133266),
    (4, 11, 0.4804826280738781), (5, 1, -0.7661099809350511), (5, 5, 0.5013755561861923),
    (6, 13, 0.7530019359197235), (7, 11, -0.895103597058257), (7, 14, 0.211118259241188),
    (8, 2, -0.6056515058961842), (8, 11, -0.7606311510412471), (9, 0, 0.5576925101091985),
    (9, 1, 0.05849810743998152), (9, 2, 0.5534207791475692), (9, 3, -0.5304217865297336),
    (9, 4, -0.9012995873924932), (9, 5, 0.3533688872442331), (9, 6, -0.9254069713403907),
    (9, 7, 0.4003977909640988), (9, 8, 0.8876276669279255), (9, 9, 0.14141769139310356),
    (9, 10, -0.6468263237456451), (9, 11, 0.18831520758490639), (9, 12, 0.6915343365416913),
    (9, 13, 0.8750872401936423), (9, 14, -0.4414336918845916), (9, 15, 0.5970008266178273),
    (10, 3, -0.9441693808990836), (10, 9, -0.9697131685746025), (11, 5, -0.8438604887428931),
    (11, 7, 0.685998812522667), (12, 8, -0.2254315454999667), (12, 15, -0.7469415976071982),
    (13, 0, 0.451834795156965), (13, 6, 0.3652486899280414), (14, 11, -0.9491415113545476),
    (14, 13, -0.3405577599850993), (15, 5, -0.3093121967887369), (15, 13, 0.49869127341413133),
    (16, 11, 0.7077855345775024), (16, 12, -0.27219671352977604), (17, 2, 0.8167062161548604),
    (17, 15, 0.8501753000233978),
]


def sparse_from(entries, m, n):
    a = np.zeros((m, n))
    for i, j, x in entries:
        a[i, j] = x
    return a


def bidiagonal_cond(n, beta=2.0):
    """The 2-norm condition number of the upper bidiagonal matrix of 1 on the diagonal and -beta
    above it, at 50 digits: its inverse is known, beta^(j - i) for j >= i."""
    mp.mp.dps = 50
    a = mp.matrix(n, n)
    inverse = mp.matrix(n, n)
    for i in range(n):
        a[i, i] = 1
        if i + 1 < n:
            a[i, i + 1] = -beta
        for j in range(i, n):
            inverse[i, j] = mp.mpf(beta) ** (j - i)
    largest = max(mp.svd_r(a, compute_uv=False))
    smallest = 1 / max(mp.svd_r(inverse, compute_uv=False))
    return largest / smallest


def singular_cases():
    import sympy
    from scipy.sparse.csgraph import structural_rank

    exact = sympy.Matrix([[sympy.Rational(x) for x in row] for row in LARGE_COMBINATION])
    a = np.array(LARGE_COMBINATION)
    s = np.linalg.svd(a, compute_uv=False)
    assert exact.rank() == 2 and exact.det() == 0 and np.linalg.matrix_rank(a) == 2
    print("// LARGE_COMBINATION: sympy rank 2, determinant 0; numpy.linalg.matrix_rank 2, "
          f"sigma_3 / sigma_1 = {s[2] / s[0]:.1e}.")
    print()
    for name, entries, m, n in [("STRUCTURAL", STRUCTURAL, 13, 5),
                                ("DENSE_ROW", DENSE_ROW, 18, 16)]:
        a = sparse_from(entries, m, n)
        structural = structural_rank(scipy.sparse.csr_array(a))
        numerical = np.linalg.matrix_rank(a)
        s = np.linalg.svd(a, compute_uv=False)
        assert structural == numerical == n - 1, (name, structural, numerical)
        print(f"// {name}: {m} x {n}, scipy.sparse.csgraph.structural_rank {structural}, "
              f"numpy.linalg.matrix_rank {numerical}, sigma_n / sigma_1 = {s[-1] / s[0]:.1e}.")
        print(f"const {name}_RANK: usize = {structural};")
        items = ", ".join(f"({i}, {j}, {fmt(x)})" for i, j, x in entries)
        print(f"const {name}: [(usize, usize, f64); {len(entries)}] = [{items}];")
        print()
    conds = [(n, bidiagonal_cond(n)) for n in (40, 60)]
    print("// The upper bidiagonal matrix of 1 and -2: condition numbers, mpmath at 50 digits.")
    items = ", ".join(f"({n}, {float(c)!r})" for n, c in conds)
    print(f"const BIDIAGONAL_COND: [(usize, f64); {len(conds)}] = [{items}];")
    print()


def mp_lstsq(a, b):
    """The exact least-squares solution to about 55 digits: the normal equations at 60."""
    mp.mp.dps = 60
    m, n = a.shape
    am = mp.matrix(m, n)
    for i in range(m):
        for j in range(n):
            am[i, j] = mp.mpc(a[i, j].real, a[i, j].imag)
    bm = mp.matrix([mp.mpc(x.real, x.imag) for x in b])
    x = mp.lu_solve(am.H * am, am.H * bm)
    return np.array([complex(x[j]) for j in range(n)]), x


def fmt(x):
    return repr(float(x))


def report(name, a, b):
    m, n = a.shape
    x, _, rank, sv = np.linalg.lstsq(a, b, rcond=None)
    assert rank == n, f"{name}: lstsq rank {rank}"
    exact, _ = mp_lstsq(a, b)
    lsqr = scipy.sparse.linalg.lsqr(scipy.sparse.csr_array(a), b, atol=1e-16, btol=1e-16,
                                    conlim=1e20, iter_lim=100000)[0]
    scale = np.linalg.norm(exact)
    e_numpy = np.linalg.norm(x - exact) / scale
    e_lsqr = np.linalg.norm(lsqr - exact) / scale
    assert e_numpy < 1e-13 and e_lsqr < 1e-9, (name, e_numpy, e_lsqr)
    residual = np.linalg.norm(a @ x - b)
    cond = sv[0] / sv[-1]
    kappa = cond + cond ** 2 * residual / (sv[0] * np.linalg.norm(x))
    print(f"// {name}: {m} x {n}, {np.count_nonzero(a)} stored elements;")
    print(f"// lstsq against mpmath {e_numpy:.1e}, lsqr against mpmath {e_lsqr:.1e}.")
    print(f"const {name}_RESIDUAL: f64 = {fmt(residual)};")
    print(f"const {name}_COND: f64 = {fmt(cond)};")
    print(f"const {name}_KAPPA: f64 = {fmt(kappa)};")
    if np.iscomplexobj(a):
        items = ", ".join(f"[{fmt(v.real)}, {fmt(v.imag)}]" for v in x)
        print(f"const {name}_X: [[f64; 2]; {n}] = [{items}];")
    else:
        items = ", ".join(fmt(v) for v in x)
        print(f"const {name}_X: [f64; {n}] = [{items}];")
    print()


def main():
    print("// Generated by lib/tests/math/sparse/data/gen_qr.py; do not edit by hand.")
    print()
    for case in CASES:
        a, b = matrix(case)
        report(case["name"], a, b)
    a, b = smoothing(40)
    report("SMOOTHING", a, b)
    for case in DEFICIENT:
        a, _ = matrix(case)
        rank = np.linalg.matrix_rank(a)
        print(f"// {case['name']}: {a.shape[0]} x {a.shape[1]}, numpy.linalg.matrix_rank.")
        print(f"const {case['name']}_RANK: usize = {rank};")
        print()
    singular_cases()


if __name__ == "__main__":
    main()
