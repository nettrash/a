#!/usr/bin/env python3
"""Reference data for the tests of the sparse Cholesky and L·D·Lᴴ factorisations.

Writes `cholesky_reference.rs` and `ldl_reference.rs` next to itself, which `../cholesky.rs`
and `../ldl.rs` include:

    python3 gen_cholesky.py

The matrices are drawn from a seeded generator, so a run reproduces the files exactly. The
solutions, the factors, the pivots and the determinants are computed with mpmath in 50 digits
from the matrices as the doubles the tests read, and rounded to doubles; they are then checked
here against NumPy (`numpy.linalg.solve`, `cholesky`, `slogdet`, `eigvalsh`) and SciPy
(`scipy.sparse.linalg.spsolve`), which must agree with them to rounding, and the run stops if
they don't. The numbers of nonzeros of the factors come from the elimination game on the
pattern.

Needs numpy, scipy and mpmath.
"""

import os

import mpmath as mp
import numpy as np
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

mp.mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))


def hermitian(n, pairs, rng, complex_values):
    """A random Hermitian pattern of the pairs given, with values in [-1, 1]."""
    a = np.zeros((n, n), dtype=complex)
    for i, j in pairs:
        x = rng.uniform(-1, 1)
        if complex_values:
            x = x + 1j * rng.uniform(-1, 1)
        a[i, j] = x
        a[j, i] = np.conj(x)
    return a


def dominant(a, rng):
    """The matrix with a diagonal that dominates its rows: positive definite."""
    a = a.copy()
    for i in range(a.shape[0]):
        a[i, i] = np.abs(a[i]).sum() - abs(a[i, i]) + rng.uniform(0.1, 1.0)
    return a


def grid_pairs(nx, ny):
    pairs = []
    for y in range(ny):
        for x in range(nx):
            i = x + nx * y
            if x + 1 < nx:
                pairs.append((i + 1, i))
            if y + 1 < ny:
                pairs.append((i + nx, i))
    return pairs


def random_pairs(n, count, rng):
    pairs = set()
    while len(pairs) < count:
        i, j = rng.integers(0, n, 2)
        if i != j:
            pairs.add((max(i, j), min(i, j)))
    return sorted(pairs)


def to_mp(a):
    n = a.shape[0]
    return mp.matrix([[mp.mpc(a[i, j].real, a[i, j].imag) for j in range(n)] for i in range(n)])


def fill(a, p):
    """The pattern of the factor of P·A·Pᵀ by the elimination game: the rows of each column."""
    n = len(p)
    columns = [set(i for i in range(j + 1, n) if a[p[i], p[j]] != 0) for j in range(n)]
    for j in range(n):
        rows = sorted(columns[j])
        for at, k in enumerate(rows):
            columns[k].update(rows[at + 1:])
    return columns


def ldl(a):
    """L and D of A = L·D·Lᴴ without pivoting, in mpmath."""
    n = a.rows
    l = mp.eye(n)
    d = [mp.mpf(0)] * n
    for j in range(n):
        s = a[j, j]
        for k in range(j):
            s -= l[j, k] * mp.conj(l[j, k]) * d[k]
        assert abs(mp.im(s)) < mp.mpf(10) ** -40
        d[j] = mp.re(s)
        assert d[j] != 0
        for i in range(j + 1, n):
            s = a[i, j]
            for k in range(j):
                s -= l[i, k] * mp.conj(l[j, k]) * d[k]
            l[i, j] = s / d[j]
    return l, d


def solve(a, b):
    return mp.lu_solve(a, mp.matrix([mp.mpc(x.real, x.imag) for x in b]))


def slogdet(a):
    """NumPy's sign and logarithm of the determinant, without the floating-point flags that
    the LAPACK of some platforms raises on the way to a correct result."""
    with np.errstate(all="ignore"):
        return np.linalg.slogdet(a)


def close(x, y, tolerance):
    return abs(x - y) <= tolerance * max(1.0, abs(y))


def number(x):
    return repr(float(x))


def floats(xs):
    return "&[" + ", ".join(number(x) for x in xs) + "]"


def triplets(entries):
    lines = ["&["]
    for i, j, x in entries:
        lines.append(f"        ({i}, {j}, {number(x.real)}, {number(x.imag)}),")
    lines.append("    ]")
    return "\n".join(lines)


def lower_of(a):
    n = a.shape[0]
    return [(i, j, complex(a[i, j])) for i in range(n) for j in range(i + 1) if a[i, j] != 0]


def common(a, rng, complex_values):
    """What the two kinds of case share: the right-hand side, the solution, a permutation."""
    n = a.shape[0]
    b = rng.uniform(-1, 1, n) + (1j * rng.uniform(-1, 1, n) if complex_values else 0)
    x = solve(to_mp(a), b)
    x = np.array([complex(v) for v in x])
    # NumPy and SciPy must give the same solution to rounding.
    for other in (np.linalg.solve(a, b), splinalg.spsolve(sparse.csc_matrix(a), b)):
        assert all(close(u, v, 1e-12) for u, v in zip(other, x)), "the solutions differ"
    p = [int(v) for v in rng.permutation(n)]
    return b, x, p


def cholesky_case(name, a, rng, complex_values):
    n = a.shape[0]
    b, x, p = common(a, rng, complex_values)
    am = to_mp(a)
    l, d = ldl(am)
    assert all(v > 0 for v in d)
    # The Cholesky factor is L·√D.
    factor = [[l[i, j] * mp.sqrt(d[j]) for j in range(n)] for i in range(n)]
    reference = np.linalg.cholesky(a)
    pattern = fill(a, list(range(n)))
    entries = []
    for j in range(n):
        for i in [j] + sorted(pattern[j]):
            value = complex(factor[i][j])
            assert close(value, reference[i, j], 1e-12), "the factors differ"
            assert value != 0
            entries.append((i, j, value))
    # Nothing outside the pattern, and nothing cancelled inside it.
    assert np.count_nonzero(np.abs(np.tril(reference)) > 1e-13) == len(entries)
    entries.sort()
    log_det = mp.fsum(mp.log(v) for v in d)
    sign, numpy_log_det = slogdet(a)
    assert close(sign.real, 1.0, 1e-12) and close(float(log_det), numpy_log_det, 1e-12)
    assert np.all(np.linalg.eigvalsh(a) > 0)
    nnz_natural = len(entries)
    nnz_permuted = n + sum(len(c) for c in fill(a, p))
    return f"""
/// {name.lower()}: order {n}, condition number {np.linalg.cond(a):.1f}.
pub(super) const {name}: Case = Case {{
    n: {n},
    lower: {triplets(lower_of(a))},
    permutation: &{p},
    nnz_natural: {nnz_natural},
    nnz_permuted: {nnz_permuted},
    b_re: {floats(b.real)},
    b_im: {floats(b.imag)},
    x_re: {floats(x.real)},
    x_im: {floats(x.imag)},
    log_det: {number(log_det)},
    det: {number(mp.exp(log_det))},
    l_natural: {triplets(entries)},
}};
"""


def ldl_case(name, a, positive, rng, complex_values):
    n = a.shape[0]
    b, x, p = common(a, rng, complex_values)
    l, d = ldl(to_mp(a))
    permuted = a[np.ix_(p, p)]
    _, d_permuted = ldl(to_mp(permuted))
    eigenvalues = np.linalg.eigvalsh(a)
    inertia = (int((eigenvalues > 0).sum()), int((eigenvalues < 0).sum()), 0)
    assert inertia == (positive, n - positive, 0)
    for pivots in (d, d_permuted):
        assert sum(1 for v in pivots if v > 0) == positive, "Sylvester's law"
    det = mp.fprod(d)
    sign, log_det = slogdet(a)
    assert close(float(det), (sign * np.exp(log_det)).real, 1e-11)
    pattern = fill(a, list(range(n)))
    entries = []
    for j in range(n):
        for i in [j] + sorted(pattern[j]):
            value = complex(l[i, j])
            assert value != 0
            entries.append((i, j, value))
    entries.sort()
    # L·D·Lᴴ is the matrix.
    lf = np.array([[complex(l[i, j]) for j in range(n)] for i in range(n)])
    df = np.array([float(v) for v in d])
    assert np.allclose((lf * df) @ lf.conj().T, a, rtol=0, atol=1e-13)
    nnz_permuted = n + sum(len(c) for c in fill(a, p))
    growth = np.abs(lf).max()
    return f"""
/// {name.lower()}: order {n}, condition number {np.linalg.cond(a):.1f}, largest element of `L` {growth:.2f}.
pub(super) const {name}: Case = Case {{
    n: {n},
    lower: {triplets(lower_of(a))},
    permutation: &{p},
    nnz_natural: {len(entries)},
    nnz_permuted: {nnz_permuted},
    b_re: {floats(b.real)},
    b_im: {floats(b.imag)},
    x_re: {floats(x.real)},
    x_im: {floats(x.imag)},
    d_natural: {floats(d)},
    d_permuted: {floats(d_permuted)},
    inertia: {inertia},
    det: {number(det)},
    l_natural: {triplets(entries)},
}};
"""


def saddle(n, m, rng, complex_values):
    """[[H, Bᴴ], [B, −C]] with H and C positive definite and B sparse."""
    h = dominant(hermitian(n, random_pairs(n, 2 * n, rng), rng, complex_values), rng)
    c = dominant(hermitian(m, random_pairs(m, m, rng), rng, complex_values), rng)
    a = np.zeros((n + m, n + m), dtype=complex)
    a[:n, :n] = h
    a[n:, n:] = -c
    for i in range(m):
        for j in rng.choice(n, 3, replace=False):
            x = rng.uniform(-1, 1) + (1j * rng.uniform(-1, 1) if complex_values else 0)
            a[n + i, j] = x
            a[j, n + i] = np.conj(x)
    return a


HEADER = """// Written by gen_cholesky.py with NumPy {numpy}, SciPy {scipy} and mpmath {mpmath}: not to be
// edited by hand. Each element is `(row, column, real part, imaginary part)`.

/// A matrix with what its factorisation must give.
pub(super) struct Case {{
    /// The order of the matrix.
    pub n: usize,
    /// The lower triangle of the matrix.
    pub lower: &'static [(usize, usize, f64, f64)],
    /// A permutation to factor it under, `p[new] = old`.
    pub permutation: &'static [usize],
    /// The nonzeros of the factor in the natural order.
    pub nnz_natural: usize,
    /// The nonzeros of the factor under the permutation.
    pub nnz_permuted: usize,
    /// A right-hand side, its real parts.
    pub b_re: &'static [f64],
    /// Its imaginary parts.
    pub b_im: &'static [f64],
    /// The solution, its real parts.
    pub x_re: &'static [f64],
    /// Its imaginary parts.
    pub x_im: &'static [f64],
{fields}
    /// The factor `L` of the natural order.
    pub l_natural: &'static [(usize, usize, f64, f64)],
}}
"""

CHOLESKY_FIELDS = """    /// The natural logarithm of the determinant.
    pub log_det: f64,
    /// The determinant.
    pub det: f64,"""

LDL_FIELDS = """    /// The pivots `D` in the natural order.
    pub d_natural: &'static [f64],
    /// The pivots under the permutation.
    pub d_permuted: &'static [f64],
    /// The numbers of positive, negative and zero eigenvalues.
    pub inertia: (usize, usize, usize),
    /// The determinant.
    pub det: f64,"""


def write(name, fields, cases):
    versions = dict(numpy=np.__version__, scipy=scipy.__version__, mpmath=mp.__version__)
    text = HEADER.format(fields=fields, **versions) + "".join(cases)
    with open(os.path.join(HERE, name), "w") as out:
        out.write(text)
    print(f"wrote {name}: {len(text)} bytes")


def main():
    rng = np.random.default_rng(20261120)
    # A 6 × 5 grid with a dozen more couplings, and a ring with more of them.
    real = dominant(hermitian(30, grid_pairs(6, 5) + random_pairs(30, 12, rng), rng, False), rng)
    ring = [((i + 1) % 18, i) for i in range(18)]
    complex_ = dominant(hermitian(18, ring + random_pairs(18, 14, rng), rng, True), rng)
    assert np.all(real.imag == 0)
    write(
        "cholesky_reference.rs",
        CHOLESKY_FIELDS,
        [cholesky_case("REAL", real, rng, False), cholesky_case("COMPLEX", complex_, rng, True)],
    )
    write(
        "ldl_reference.rs",
        LDL_FIELDS,
        [
            ldl_case("REAL", saddle(16, 8, rng, False), 16, rng, False),
            ldl_case("COMPLEX", saddle(10, 6, rng, True), 10, rng, True),
        ],
    )


if __name__ == "__main__":
    main()
