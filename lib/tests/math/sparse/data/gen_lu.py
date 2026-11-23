#!/usr/bin/env python3
"""Reference data for the tests of the sparse LU decomposition (lib/tests/math/sparse/lu.rs).

Writes lu_reference.rs, next to this script, which the test file includes. Every case holds a
sparse matrix, a right-hand side, and what SciPy and NumPy make of them:

* the solutions of A x = b, A^T x = b and A^H x = b by scipy.sparse.linalg.spsolve and splu,
  which is SuperLU; each is checked here against numpy.linalg.solve, which is LAPACK;
* the determinant, by numpy.linalg.det;
* the reciprocal condition number in the 1-norm, by numpy.linalg.cond;
* the row permutation SuperLU chooses with the columns in their natural order, partial pivoting
  and no equilibration, in the convention p[new] = old;
* the fill of the elimination with those pivots: the elements of L below its diagonal and of U
  on and above it that it reaches, counted by a symbolic elimination with no cancellation (the
  counts of nonzero values depend on the rounding and are not compared: SuperLU leaves values of
  1e-17 where the exact one is zero, and cancels some that are exactly 4e-22).

It also writes STRUCTURAL, thin random patterns with their structural rank by
scipy.sparse.csgraph.structural_rank: a matrix whose structural rank is below its order is
singular whatever its elements, and one of full structural rank has, with these elements, a
condition number below 1e6 (checked by numpy.linalg.cond).

Run: python3 gen_lu.py      (NumPy 2.3, SciPy 1.18 when the file was last written)
"""

import os

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import structural_rank


def cond1(dense):
    """numpy.linalg.cond in the 1-norm, without the warnings of numpy's determinant, which it
    takes along the way, about a singular candidate that is then drawn again."""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return np.linalg.cond(dense, 1)


def random_unsymmetric(rng):
    """A random pattern with a few nonzeros a row, some of the diagonal missing."""
    n = 30
    while True:
        a = sp.random(n, n, density=0.1, random_state=rng, data_rvs=lambda k: rng.uniform(-1, 1, k))
        d = rng.uniform(-1, 1, n)
        d[rng.choice(n, 10, replace=False)] = 0.0  # zeros on the diagonal: pivoting is needed
        a = sp.csc_matrix(a + sp.diags(d))
        a.eliminate_zeros()
        dense = a.toarray()
        if np.linalg.matrix_rank(dense) == n and cond1(dense) < 2e3:
            return a


def convection_diffusion(k, cx, cy):
    """Five-point convection-diffusion on a k x k grid with the cell Peclet numbers cx, cy."""
    n = k * k
    rows, cols, vals = [], [], []
    for i in range(k):
        for j in range(k):
            me = i * k + j
            for (di, dj, v) in ((-1, 0, -1 - cy), (0, -1, -1 - cx), (0, 0, 4.0), (0, 1, -1 + cx), (1, 0, -1 + cy)):
                ii, jj = i + di, j + dj
                if 0 <= ii < k and 0 <= jj < k:
                    rows.append(me)
                    cols.append(ii * k + jj)
                    vals.append(v)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))


def tiny_diagonal(rng):
    """A diagonal of 1e-10 under off-diagonal elements of order one."""
    n = 24
    while True:
        a = sp.random(n, n, density=0.1, random_state=rng, data_rvs=lambda k: rng.uniform(0.5, 1.5, k) * rng.choice([-1, 1], k))
        a = sp.lil_matrix(a)
        for i in range(n):
            a[i, i] = 1e-10
            a[i, (i + 1) % n] = rng.uniform(1, 2)
        a = sp.csc_matrix(a)
        dense = a.toarray()
        if cond1(dense) < 2e3:
            return a


def block_triangular(rng):
    """Five diagonal blocks of order 6 with a few elements above them, rows and columns shuffled."""
    n, block = 30, 6
    while True:
        dense = np.zeros((n, n))
        for bi in range(n // block):
            lo = bi * block
            dense[lo:lo + block, lo:lo + block] = np.where(rng.random((block, block)) < 0.6, rng.uniform(-1, 1, (block, block)), 0.0)
            for bj in range(bi + 1, n // block):
                mask = rng.random((block, block)) < 0.12
                dense[lo:lo + block, bj * block:(bj + 1) * block] = np.where(mask, rng.uniform(-1, 1, (block, block)), 0.0)
        dense = dense[rng.permutation(n)][:, rng.permutation(n)]
        if np.linalg.matrix_rank(dense) == n and cond1(dense) < 5e3:
            return sp.csc_matrix(dense)


def complex_random(rng):
    """A random complex pattern, neither symmetric nor Hermitian."""
    n = 24
    while True:
        k = int(0.12 * n * n)
        rows, cols = rng.integers(0, n, k), rng.integers(0, n, k)
        vals = rng.uniform(-1, 1, k) + 1j * rng.uniform(-1, 1, k)
        a = sp.csc_matrix(sp.coo_matrix((vals, (rows, cols)), shape=(n, n)))
        a.sum_duplicates()
        dense = a.toarray()
        if np.linalg.matrix_rank(dense) == n and cond1(dense) < 2e3:
            return a


def complex_helmholtz(k):
    """A damped Helmholtz operator with a complex convection term on a k x k grid."""
    n = k * k
    rows, cols, vals = [], [], []
    for i in range(k):
        for j in range(k):
            me = i * k + j
            for (di, dj, v) in ((-1, 0, -1 - 0.5j), (0, -1, -1 + 0.25j), (0, 0, 4.0 - (1.2 + 0.8j)), (0, 1, -1 - 0.75j), (1, 0, -1 + 1.5j)):
                ii, jj = i + di, j + dj
                if 0 <= ii < k and 0 <= jj < k:
                    rows.append(me)
                    cols.append(ii * k + jj)
                    vals.append(v)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))


def symbolic_fill(a, perm):
    """The elements of L below its diagonal plus those of U that the elimination of the rows
    of `a` in the order `perm`, columns in place, reaches when nothing cancels."""
    pattern = (a.toarray() != 0)[perm]
    n = pattern.shape[0]
    for k in range(n):
        assert pattern[k, k], "a pivot of SuperLU outside the pattern"
        below = np.nonzero(pattern[k + 1:, k])[0] + k + 1
        right = np.nonzero(pattern[k, k + 1:])[0] + k + 1
        pattern[np.ix_(below, right)] = True
    return int(np.count_nonzero(pattern))


def pair(z):
    z = complex(z)
    return "(%r, %r)" % (z.real, z.imag)


def case(name, a, rng, out):
    a = sp.csc_matrix(a)
    a.sort_indices()
    n = a.shape[0]
    is_complex = np.iscomplexobj(a.data)
    dense = a.toarray()
    b = rng.uniform(-1, 1, n) + (1j * rng.uniform(-1, 1, n) if is_complex else 0)
    lu = spla.splu(a)
    x = spla.spsolve(a, b)
    xt = lu.solve(b, trans="T")
    xh = lu.solve(b, trans="H")
    # SuperLU against LAPACK: the references must be right before they are embedded.
    for (got, m) in ((x, dense), (xt, dense.T), (xh, dense.conj().T)):
        want = np.linalg.solve(m, b)
        assert np.abs(got - want).max() <= 1e-11 * np.abs(want).max(), name
    # The LAPACK under numpy raises floating-point flags in the determinant of some of these
    # regular matrices, whose value is right (the tests hold it to the sparse determinant).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        det = np.linalg.det(dense)
    rcond = 1.0 / np.linalg.cond(dense, 1)
    natural = spla.splu(a, permc_spec="NATURAL", diag_pivot_thresh=1.0,
                        options=dict(Equil=False, SymmetricMode=False))
    assert (natural.perm_c == np.arange(n)).all()
    perm = np.empty(n, dtype=int)
    perm[natural.perm_r] = np.arange(n)
    lower, upper = natural.L, natural.U
    assert np.abs(dense[perm] - (lower @ upper).toarray()).max() < 1e-12
    fill = symbolic_fill(a, perm)
    # The nonzero values are within the reach.
    assert lower.count_nonzero() - n + upper.count_nonzero() <= fill
    coo = sp.coo_matrix(a)
    out.append("    Reference {")
    out.append('        name: "%s",' % name)
    out.append("        n: %d," % n)
    out.append("        complex: %s," % ("true" if is_complex else "false"))
    out.append("        entries: &[")
    for (i, j, v) in sorted(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist())):
        z = complex(v)
        out.append("            (%d, %d, %r, %r)," % (i, j, z.real, z.imag))
    out.append("        ],")
    for (field, vec) in (("rhs", b), ("x", x), ("xt", xt), ("xh", xh)):
        out.append("        %s: &[" % field)
        for v in vec:
            out.append("            %s," % pair(v))
        out.append("        ],")
    out.append("        det: %s," % pair(det))
    out.append("        rcond: %r," % float(rcond))
    out.append("        perm: &[%s]," % ", ".join(str(int(v)) for v in perm))
    out.append("        fill: %d," % fill)
    out.append("    },")
    print("%-22s n=%3d nnz=%4d cond1=%9.2f det=%s  fill=%d (nonzero values of SuperLU: %d) moved rows=%d" % (
        name, n, a.nnz, 1 / rcond, pair(det), fill, lower.count_nonzero() - n + upper.count_nonzero(),
        int((perm != np.arange(n)).sum())))


def structural(rng, out):
    """Thin random patterns, about one and a half elements a row, and their structural ranks."""
    out.append("const STRUCTURAL: &[Structural] = &[")
    singular = regular = 0
    while singular < 6 or regular < 6:
        n = int(rng.integers(4, 31))
        k = int(1.5 * n)
        rows, cols = rng.integers(0, n, k), rng.integers(0, n, k)
        vals = rng.uniform(0.5, 1.5, k) * rng.choice([-1.0, 1.0], k)
        a = sp.csr_matrix(sp.coo_matrix((vals, (rows, cols)), shape=(n, n)))
        a.sum_duplicates()
        a.eliminate_zeros()
        rank = int(structural_rank(a))
        if rank < n:
            if singular >= 6:
                continue
            singular += 1
        else:
            if regular >= 6 or cond1(a.toarray()) > 1e6:
                continue
            regular += 1
        coo = a.tocoo()
        out.append("    Structural {")
        out.append("        n: %d," % n)
        out.append("        entries: &[%s]," % ", ".join(
            "(%d, %d, %r)" % t for t in sorted(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()))))
        out.append("        rank: %d," % rank)
        out.append("    },")
        print("structural n=%2d nnz=%3d rank=%2d" % (n, a.nnz, rank))
    out.append("];")


def main():
    rng = np.random.default_rng(20261120)
    out = [
        "// Written by gen_lu.py with NumPy %s and SciPy %s: do not edit, run the script." % (np.__version__, scipy.__version__),
        "// The fields are described on `Reference` in lu.rs and in the script.",
        "",
        "const REFERENCES: &[Reference] = &[",
    ]
    case("random_unsymmetric", random_unsymmetric(rng), rng, out)
    case("convection_diffusion", convection_diffusion(6, 1.5, 3.5), rng, out)
    case("laplacian", convection_diffusion(6, 0.0, 0.0), rng, out)
    case("tiny_diagonal", tiny_diagonal(rng), rng, out)
    case("block_triangular", block_triangular(rng), rng, out)
    case("complex_random", complex_random(rng), rng, out)
    case("complex_helmholtz", complex_helmholtz(5), rng, out)
    out.append("];")
    out.append("")
    structural(rng, out)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lu_reference.rs")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
