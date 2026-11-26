"""Reference data of the Krylov solvers and preconditioners of `a::math::sparse` (issue #22):
scipy.sparse.linalg.cg, minres, gmres and bicgstab run on the problems that `solve.rs` builds,
with the same tolerances, starting guesses and preconditioners. Writes `iter_scipy.txt`.

Run from this directory: `python3 gen_iter.py`. Deterministic (no random numbers): every matrix
and vector is given by a formula whose values are exact in `f32` as well as `f64`, so that the
Rust tests build bit for bit the same systems.

What is recorded, one line per run and element type:

    problem type method preconditioner restart start rtol iterations converged low high
        w·x (real part) w·x (imaginary part) ‖b − A·x‖₂ ‖A⁻¹‖₂

* `iterations` is scipy's count in the meaning of the Rust `SolveResult::iterations`:
  - cg: the number of iterations scipy's own test needed (calls of its callback);
  - minres: the first k at which the TRUE residual ‖b − A·xₖ‖₂ of scipy's k-th iterate is
    within rtol·‖b‖₂ (scipy's own test is relative to ‖A‖·‖x‖; the iterates are the method's).
    Where scipy stops by a test of its own first, it begins again from its iterate, as long as
    that improves the residual, and the count goes on (the graded problems in single precision
    and in Complex<f64>);
  - gmres: the inner iterations, of restarted GMRES on `A·M⁻¹` when there is a preconditioner
    (right preconditioning, as the Rust method does; scipy's gmres preconditions on the left);
  - bicgstab: the products with A divided by two, rounded up, which counts a final half step
    as an iteration as the Rust method does (scipy's callback does not).
* `low` and `high` are the fewest and the most iterations of the converged ones among that run
  and six more on the right-hand side times 1 + k·δ, k = 1..6, δ = 2⁻⁴⁴ in double and 2⁻²⁰ in
  single precision: changes at the level of the rounding errors the runs make anyway. They show
  how far rounding alone moves a count. For the methods that minimise something they stay
  together (within 2 in double precision, 5 in single); BiCGSTAB on an indefinite matrix
  spreads by a third and more (130 to 186 on magshift2d10 in double precision).
* `w·x` is Σ wᵢ·xᵢ for wᵢ = ((3i + 1) mod 5) − 2 and scipy's solution x, and `‖b − A·x‖₂` its
  true residual, both in float64 from scipy's numbers. With ‖A⁻¹‖₂ (from a dense SVD; for a
  singular matrix that of its pseudo-inverse) they bound how far another solution of residual
  ‖r‖ can be from scipy's: |w·(x − x_scipy)| ≤ ‖w‖·‖A⁻¹‖·(‖r‖ + ‖r_scipy‖).
* The incomplete factorisations given to scipy are the textbook right-looking (KIJ) IC(0) and
  ILU(0) below, on the pattern of the matrix, computed in the working precision; the Rust code
  uses the row-by-row forms, so the two are independent. Each is checked against its defining
  property, (L·Lᴴ)ᵢⱼ = aᵢⱼ and (L·U)ᵢⱼ = aᵢⱼ on the pattern.
"""

import math

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sla

TYPES = {"real": [("f64", np.float64), ("f32", np.float32)],
         "complex": [("c64", np.complex128), ("c32", np.complex64)]}
RTOL = {"f64": 1e-10, "c64": 1e-10, "f32": 1e-4, "c32": 1e-4}
PERTURB = {"f64": 2.0 ** -44, "c64": 2.0 ** -44, "f32": 2.0 ** -20, "c32": 2.0 ** -20}


# ---------------------------------------------------------------------------------------------
# The problems (the Rust builders in solve.rs mirror these)
# ---------------------------------------------------------------------------------------------

def from_entries(n, entries):
    rows = [e[0] for e in entries]
    cols = [e[1] for e in entries]
    vals = [e[2] for e in entries]
    return sp.coo_matrix((np.array(vals, dtype=complex), (rows, cols)), shape=(n, n)).tocsr()


def grid(k, diagonal, east, west, north, south):
    """The 5-point operator on a k x k grid, node (r, c) numbered r*k + c."""
    e = []
    for r in range(k):
        for c in range(k):
            i = r * k + c
            e.append((i, i, diagonal(r, c)))
            if c + 1 < k:
                e.append((i, i + 1, east(r, c)))
            if c > 0:
                e.append((i, i - 1, west(r, c)))
            if r + 1 < k:
                e.append((i, i + k, north(r, c)))
            if r > 0:
                e.append((i, i - k, south(r, c)))
    return from_entries(k * k, e)


def lap1d(n):
    e = [(i, i, 2.0) for i in range(n)]
    e += [(i, i + 1, -1.0) for i in range(n - 1)] + [(i + 1, i, -1.0) for i in range(n - 1)]
    return from_entries(n, e)


def lap2d(k):
    m = lambda r, c: -1.0
    return grid(k, lambda r, c: 4.0, m, m, m, m)


def scaled2d(k):
    """D·lap2d·D with d_i = 2^((i mod 4) − 1): rows of very different sizes."""
    a = lap2d(k).tocoo()
    d = np.array([2.0 ** ((i % 4) - 1) for i in range(k * k)])
    return from_entries(k * k, [(i, j, d[i] * v * d[j]) for i, j, v in zip(a.row, a.col, a.data)])


def shifted2d(k):
    """lap2d − 2.5·I: symmetric, indefinite (24 negative eigenvalues of 100 for k = 10)."""
    m = lambda r, c: -1.0
    return grid(k, lambda r, c: 1.5, m, m, m, m)


def saddle(k):
    """[[lap1d(k), Bᵀ], [B, 0]] with B (k/2 x k) summing pairs: symmetric indefinite, nonsingular."""
    h = k // 2
    e = [(i, i, 2.0) for i in range(k)]
    e += [(i, i + 1, -1.0) for i in range(k - 1)] + [(i + 1, i, -1.0) for i in range(k - 1)]
    for i in range(h):
        for j in (2 * i, 2 * i + 1):
            e.append((k + i, j, 1.0))
            e.append((j, k + i, 1.0))
    return from_entries(k + h, e)


def neumann2d(k):
    """The graph Laplacian of the grid: singular, its null space the constants."""
    def degree(r, c):
        return float((c + 1 < k) + (c > 0) + (r + 1 < k) + (r > 0))
    m = lambda r, c: -1.0
    return grid(k, degree, m, m, m, m)


def convdiff1d(n):
    e = [(i, i, 2.0) for i in range(n)]
    e += [(i + 1, i, -1.5) for i in range(n - 1)] + [(i, i + 1, -0.5) for i in range(n - 1)]
    return from_entries(n, e)


def convdiff2d(k):
    return grid(k, lambda r, c: 4.0, lambda r, c: -0.5, lambda r, c: -1.5,
                lambda r, c: -0.75, lambda r, c: -1.25)


def magnetic2d(k, d):
    """Hermitian: d on the diagonal, −iʳ on the edge from (r, c) to (r, c + 1) (a flux of a
    quarter turn through every cell), −1 on the vertical edges. Positive definite for d = 3,
    indefinite for d = 1."""
    phase = [1.0, 1j, -1.0, -1j]
    return grid(k, lambda r, c: d, lambda r, c: -phase[r % 4], lambda r, c: -np.conj(phase[r % 4]),
                lambda r, c: -1.0, lambda r, c: -1.0)


def helmholtz2d(k):
    """lap2d − (1.5 + 0.25i)·I: complex symmetric, not Hermitian."""
    m = lambda r, c: -1.0
    return grid(k, lambda r, c: 2.5 - 0.25j, m, m, m, m)


def cconvdiff2d(k):
    return grid(k, lambda r, c: 4.0 + 1.0j, lambda r, c: -0.5 + 0.25j, lambda r, c: -1.5 - 0.25j,
                lambda r, c: -0.75 + 0.5j, lambda r, c: -1.25)


def one(field):
    return from_entries(1, [(0, 0, 3.0)])


def graded(k, phases):
    """−(c·u')' + 2⁻¹⁰·u on k points, c_{i+1/2} = 2^e(i) with e(i) = 8 − ⌊16·|(i mod 24) − 12| / 24⌋,
    which runs from 1 up to 2⁸ and down again every 24 points: symmetric positive definite, with a
    condition number of about 10⁶ and a diagonal far from constant, which Jacobi evens out (the
    review of round 1 found MINRES with Jacobi stopping early on such systems). With `phases` the
    element (i, i + 1) is multiplied by iⁱ and (i + 1, i) by its conjugate: a Hermitian matrix that
    is unitarily similar to the real one."""
    c = lambda i: 2.0 ** (8 - (abs((i % 24) - 12) * 16) // 24)
    phase = [1.0, 1j, -1.0, -1j]
    e = []
    for i in range(k):
        e.append((i, i, c(i) + c(i + 1) + 2.0 ** -10))
        if i + 1 < k:
            ph = phase[i % 4] if phases else 1.0
            e.append((i, i + 1, -c(i + 1) * ph))
            e.append((i + 1, i, -c(i + 1) * np.conj(ph)))
    return from_entries(k, e)


PROBLEMS = {
    # name: (field, builder)
    "lap2d12": ("real", lambda: lap2d(12)),
    "scaled2d12": ("real", lambda: scaled2d(12)),
    "lap1d64": ("real", lambda: lap1d(64)),
    "lap2d30": ("real", lambda: lap2d(30)),
    "shifted2d10": ("real", lambda: shifted2d(10)),
    "saddle20": ("real", lambda: saddle(20)),
    "neumann2d10": ("real", lambda: neumann2d(10)),
    "convdiff2d12": ("real", lambda: convdiff2d(12)),
    "convdiff1d50": ("real", lambda: convdiff1d(50)),
    "one": ("real", lambda: one("real")),
    "graded100": ("real", lambda: graded(100, False)),
    "magnetic2d12": ("complex", lambda: magnetic2d(12, 3.0)),
    "magshift2d10": ("complex", lambda: magnetic2d(10, 1.0)),
    "helmholtz2d12": ("complex", lambda: helmholtz2d(12)),
    "cconvdiff2d10": ("complex", lambda: cconvdiff2d(10)),
    "cone": ("complex", lambda: one("complex")),
    "cgraded100": ("complex", lambda: graded(100, True)),
}


def rhs(n, field, problem):
    re = np.array([((13 * i + 5) % 17) - 8 for i in range(n)], dtype=float)
    if problem == "neumann2d10":
        # In the range of the matrix: the constants are its null space, and this sums to 0.
        c = [((7 * i + 3) % 11) for i in range(n)]
        re = np.array([c[i] - c[n - 1 - i] for i in range(n)], dtype=float)
    graded = problem in ("graded100", "cgraded100")
    if graded:
        # The right-hand side of the review's example.
        re = np.array([((7 * i + 3) % 11) - 5 for i in range(n)], dtype=float)
    if field == "real":
        return re
    im = np.array([((7 * i + 3) % 11) - 5 for i in range(n)], dtype=float)
    if graded:
        im = np.array([((5 * i + 1) % 7) - 3 for i in range(n)], dtype=float)
    return re + 1j * im


def start(n, field):
    re = np.array([(((5 * i + 1) % 7) - 3) / 4 for i in range(n)])
    if field == "real":
        return re
    im = np.array([(((3 * i + 2) % 5) - 2) / 4 for i in range(n)])
    return re + 1j * im


def weights(n):
    return np.array([((3 * i + 1) % 5) - 2 for i in range(n)], dtype=float)


# ---------------------------------------------------------------------------------------------
# Reference preconditioners, in the working precision
# ---------------------------------------------------------------------------------------------

def ic0(a):
    """IC(0), right-looking: column k is divided by its pivot, then its outer product is taken
    from the entries of the lower triangle that are in the pattern."""
    n = a.shape[0]
    low = sp.tril(a).toarray()
    mask = low != 0
    for k in range(n):
        if not (low[k, k].real > 0):
            return None
        low[k, k] = np.sqrt(low[k, k].real)
        rows = [i for i in range(k + 1, n) if mask[i, k]]
        for i in rows:
            low[i, k] = low[i, k] / low[k, k]
        for j in rows:
            for i in rows:
                if i >= j and mask[i, j]:
                    low[i, j] = low[i, j] - low[i, k] * np.conj(low[j, k])
    return sp.csr_matrix(low)


def ilu0(a):
    """ILU(0), right-looking: the multipliers of column k, then the update of the entries of the
    pattern to the right and below."""
    n = a.shape[0]
    m = a.toarray()
    mask = m != 0
    for k in range(n):
        if m[k, k] == 0:
            return None
        rows = [i for i in range(k + 1, n) if mask[i, k]]
        cols = [j for j in range(k + 1, n) if mask[k, j]]
        for i in rows:
            m[i, k] = m[i, k] / m[k, k]
            for j in cols:
                if mask[i, j]:
                    m[i, j] = m[i, j] - m[i, k] * m[k, j]
    lower = sp.csr_matrix(np.tril(m, -1) + np.eye(n, dtype=m.dtype))
    upper = sp.csr_matrix(np.triu(m))
    return lower, upper


def check_ic0(a, low):
    d = low.toarray()
    prod = d @ d.conj().T
    full = a.toarray()
    mask = np.tril(full != 0)
    err = np.abs(prod - full)[mask].max()
    assert err <= 1e3 * np.finfo(a.dtype).eps * np.abs(full).max(), err


def check_ilu0(a, lower, upper):
    prod = lower.toarray() @ upper.toarray()
    full = a.toarray()
    err = np.abs(prod - full)[full != 0].max()
    assert err <= 1e3 * np.finfo(a.dtype).eps * np.abs(full).max(), err


def preconditioner(a, kind):
    """M⁻¹ as a function of a vector, or None for no preconditioner."""
    if kind == "none":
        return None
    if kind == "jacobi":
        d = a.diagonal()
        return lambda r: (r / d).astype(a.dtype)
    if kind == "ic0":
        low = ic0(a)
        check_ic0(a, low)
        up = low.conj().T.tocsr()
        return lambda r: sla.spsolve_triangular(
            up, sla.spsolve_triangular(low, r, lower=True), lower=False).astype(a.dtype)
    if kind == "ilu0":
        lower, upper = ilu0(a)
        check_ilu0(a, lower, upper)
        return lambda r: sla.spsolve_triangular(
            upper, sla.spsolve_triangular(lower, r, lower=True, unit_diagonal=True),
            lower=False).astype(a.dtype)
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------
# The runs
# ---------------------------------------------------------------------------------------------

def operator(a, f=None):
    return sla.LinearOperator(a.shape, matvec=f if f is not None else (lambda v: a @ v),
                              dtype=a.dtype)


def run(a, b, x0, method, pre, restart, rtol, budget):
    minv = preconditioner(a, pre)
    m = None if minv is None else operator(a, minv)
    count = [0]
    def tick(_):
        count[0] += 1
    if method == "cg":
        x, info = sla.cg(a, b, x0=x0, rtol=rtol, atol=0.0, maxiter=budget, M=m, callback=tick)
        return x, info == 0, count[0]
    if method == "minres":
        a64 = a.astype(np.complex128)
        b64 = b.astype(np.complex128)
        tol = rtol * np.linalg.norm(b64)
        found = []
        def watch(xk):
            count[0] += 1
            if not found and np.linalg.norm(b64 - a64 @ xk.astype(np.complex128)) <= tol:
                found.append((count[0], xk.copy()))
        x = x0
        last = np.inf
        while True:
            x, info = sla.minres(a, b, x0=x, rtol=1e-30, maxiter=budget - count[0], M=m,
                                 callback=watch)
            if found:
                return found[0][1], True, found[0][0]
            # scipy's MINRES ends by its own tests too (an estimate of the condition number past
            # 0.1/eps, a residual at the level of rounding); it begins again from its iterate,
            # as the Rust method does when a residual formed afresh contradicts its estimate,
            # while that improves the residual and the budget lasts.
            now = np.linalg.norm(b64 - a64 @ x.astype(np.complex128))
            if info != 0 or count[0] >= budget or not now < last:
                return x, False, count[0]
            last = now
    if method == "gmres":
        cycles = math.ceil(budget / min(restart, a.shape[0]))
        if minv is None:
            x, info = sla.gmres(a, b, x0=x0, rtol=rtol, atol=0.0, restart=restart,
                                maxiter=cycles, callback=tick, callback_type="pr_norm")
        else:
            assert x0 is None
            right = operator(a, lambda v: a @ minv(v))
            u, info = sla.gmres(right, b, rtol=rtol, atol=0.0, restart=restart, maxiter=cycles,
                                callback=tick, callback_type="pr_norm")
            x = minv(u)
        return x, info == 0, count[0]
    if method == "bicgstab":
        products = [0]
        def product(v):
            products[0] += 1
            return a @ v
        x, info = sla.bicgstab(operator(a, product), b, x0=x0, rtol=rtol, atol=0.0,
                               maxiter=budget, M=m)
        k = products[0] - (1 if x0 is not None else 0)
        return x, info == 0, (k + 1) // 2
    raise ValueError(method)


# (method, preconditioner, restart, start) for each problem; restart 0 for the methods without.
RUNS = {
    "lap2d12": [("cg", "none", 0, "zero"), ("cg", "jacobi", 0, "zero"), ("cg", "ic0", 0, "zero"),
                ("cg", "ilu0", 0, "zero"), ("cg", "none", 0, "guess"),
                ("minres", "none", 0, "zero"), ("minres", "jacobi", 0, "zero"),
                ("minres", "ic0", 0, "zero"), ("minres", "none", 0, "guess"),
                ("gmres", "none", 1, "zero"), ("gmres", "none", 5, "zero"),
                ("gmres", "none", 30, "zero"), ("gmres", "none", 200, "zero"),
                ("gmres", "jacobi", 30, "zero"), ("gmres", "ilu0", 30, "zero"),
                ("gmres", "ic0", 30, "zero"), ("gmres", "none", 30, "guess"),
                ("bicgstab", "none", 0, "zero"), ("bicgstab", "jacobi", 0, "zero"),
                ("bicgstab", "ilu0", 0, "zero"), ("bicgstab", "none", 0, "guess")],
    "scaled2d12": [("cg", "none", 0, "zero"), ("cg", "jacobi", 0, "zero"),
                   ("minres", "none", 0, "zero"), ("minres", "jacobi", 0, "zero"),
                   ("gmres", "none", 30, "zero"), ("gmres", "jacobi", 30, "zero"),
                   ("bicgstab", "none", 0, "zero"), ("bicgstab", "jacobi", 0, "zero")],
    "lap1d64": [("cg", "none", 0, "guess"), ("cg", "ic0", 0, "guess"),
                ("minres", "none", 0, "guess"), ("gmres", "none", 10, "guess"),
                ("bicgstab", "none", 0, "guess"), ("bicgstab", "ilu0", 0, "guess")],
    "lap2d30": [("cg", "none", 0, "zero"), ("cg", "jacobi", 0, "zero"), ("cg", "ic0", 0, "zero"),
                ("minres", "none", 0, "zero"), ("minres", "ic0", 0, "zero"),
                ("gmres", "none", 30, "zero"), ("gmres", "ilu0", 30, "zero"),
                ("bicgstab", "none", 0, "zero"), ("bicgstab", "ilu0", 0, "zero")],
    "shifted2d10": [("minres", "none", 0, "zero"), ("minres", "jacobi", 0, "zero"),
                    ("gmres", "none", 30, "zero"), ("gmres", "none", 100, "zero"),
                    ("bicgstab", "none", 0, "zero")],
    "saddle20": [("minres", "none", 0, "zero"), ("gmres", "none", 30, "zero")],
    "neumann2d10": [("cg", "none", 0, "zero"), ("minres", "none", 0, "zero"),
                    ("gmres", "none", 100, "zero")],
    "convdiff2d12": [("gmres", "none", 1, "zero"), ("gmres", "none", 5, "zero"),
                     ("gmres", "none", 30, "zero"), ("gmres", "none", 144, "zero"),
                     ("gmres", "jacobi", 30, "zero"), ("gmres", "ilu0", 30, "zero"),
                     ("gmres", "ilu0", 5, "zero"), ("bicgstab", "none", 0, "zero"),
                     ("bicgstab", "jacobi", 0, "zero"), ("bicgstab", "ilu0", 0, "zero")],
    "convdiff1d50": [("gmres", "none", 10, "guess"), ("gmres", "none", 50, "zero"),
                     ("bicgstab", "none", 0, "guess")],
    "one": [("cg", "none", 0, "zero"), ("minres", "none", 0, "zero"),
            ("gmres", "none", 1, "zero"), ("bicgstab", "none", 0, "zero")],
    "magnetic2d12": [("cg", "none", 0, "zero"), ("cg", "jacobi", 0, "zero"),
                     ("cg", "ic0", 0, "zero"), ("cg", "none", 0, "guess"),
                     ("minres", "none", 0, "zero"), ("minres", "ic0", 0, "zero"),
                     ("gmres", "none", 30, "zero"), ("gmres", "ilu0", 30, "zero"),
                     ("bicgstab", "none", 0, "zero"), ("bicgstab", "ilu0", 0, "zero")],
    "magshift2d10": [("minres", "none", 0, "zero"), ("gmres", "none", 30, "zero"),
                     ("gmres", "none", 100, "zero"), ("bicgstab", "none", 0, "zero")],
    "helmholtz2d12": [("gmres", "none", 30, "zero"), ("gmres", "none", 144, "zero"),
                      ("gmres", "ilu0", 30, "zero"), ("bicgstab", "none", 0, "zero"),
                      ("bicgstab", "ilu0", 0, "zero")],
    "cconvdiff2d10": [("gmres", "none", 1, "zero"), ("gmres", "none", 5, "zero"),
                      ("gmres", "none", 30, "zero"), ("gmres", "ilu0", 5, "zero"),
                      ("bicgstab", "none", 0, "zero"), ("bicgstab", "ilu0", 0, "zero")],
    "cone": [("cg", "none", 0, "zero"), ("minres", "none", 0, "zero"),
             ("gmres", "none", 1, "zero"), ("bicgstab", "none", 0, "zero")],
    "graded100": [("minres", "jacobi", 0, "zero"), ("minres", "jacobi", 0, "guess"),
                  ("minres", "none", 0, "zero"), ("cg", "jacobi", 0, "zero")],
    "cgraded100": [("minres", "jacobi", 0, "zero"), ("minres", "jacobi", 0, "guess"),
                   ("minres", "none", 0, "zero"), ("cg", "jacobi", 0, "zero")],
}


def inverse_norm(a):
    s = np.linalg.svd(a.toarray(), compute_uv=False)
    s = s[s > 1e-10 * s.max()]
    return 1.0 / s.min()


def main():
    lines = ["# Made by gen_iter.py with numpy %s and scipy %s; see that file for the columns."
             % (np.__version__, scipy.__version__)]
    for name, (field, build) in PROBLEMS.items():
        a128 = build()
        if field == "real":
            a128 = a128.real.astype(np.float64)
        n = a128.shape[0]
        b128 = rhs(n, field, name)
        g128 = start(n, field)
        w = weights(n)
        inv = inverse_norm(a128)
        budget = 20 * n
        for tname, dtype in TYPES[field]:
            a = a128.astype(dtype)
            b = b128.astype(dtype)
            rtol = RTOL[tname]
            for method, pre, restart, guess in RUNS[name]:
                x0 = None if guess == "zero" else g128.astype(dtype)
                x, ok, its = run(a, b, x0, method, pre, restart, rtol, budget)
                spread = [its] if ok else []
                for k in range(1, 7):
                    bk = (b128 * (1 + k * PERTURB[tname])).astype(dtype)
                    _, okk, itk = run(a, bk, x0, method, pre, restart, rtol, budget)
                    if okk:
                        spread.append(itk)
                low, high = (min(spread), max(spread)) if spread else (its, its)
                x = np.asarray(x).astype(np.complex128)
                res = np.linalg.norm(b128.astype(np.complex128) - a128.astype(np.complex128) @ x)
                wx = complex(w @ x)
                lines.append("%s %s %s %s %d %s %.1e %d %d %d %d %.17e %.17e %.17e %.17e" % (
                    name, tname, method, pre, restart, guess, rtol, its, int(ok), low, high,
                    wx.real, wx.imag, res, inv))
    with open("iter_scipy.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %d runs" % (len(lines) - 1))


if __name__ == "__main__":
    main()
