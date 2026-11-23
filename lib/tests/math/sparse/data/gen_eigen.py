"""Reference eigenvalues for lib/tests/math/sparse/eigen.rs, from ARPACK through SciPy.

Run with `python3 gen_eigen.py` (numpy 2.3, scipy 1.18). It builds the same matrices as the
Rust tests, from the same xorshift generator as tests/math/sparse/common.rs (`Rng`), computes the
wanted eigenvalues with scipy.sparse.linalg.eigsh / eigs (ARPACK, tol=0, i.e. machine precision),
checks each against the dense solvers of numpy (LAPACK), and prints the Rust constants that the
tests embed. The matrices are summed from triplets exactly as `SparseMatrix::from_coo` does.
ARPACK starts from a random vector of its own, so the last digit or two of its values change from
run to run; the tests compare within 1e-9 relative, far above that.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

MASK = (1 << 64) - 1


class Rng:
    """The xorshift64 generator of common.rs: next(), unit() in [-1, 1), below(n)."""

    def __init__(self, seed):
        self.s = seed

    def next(self):
        s = self.s
        s ^= (s << 13) & MASK
        s ^= s >> 7
        s ^= (s << 17) & MASK
        self.s = s
        return s

    def unit(self):
        return (self.next() >> 11) / float(1 << 52) - 1.0

    def below(self, n):
        return self.next() % n


def build(n, entries, dtype):
    rows = [e[0] for e in entries]
    cols = [e[1] for e in entries]
    vals = np.array([e[2] for e in entries], dtype=dtype)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def symmetric(seed, n, per_row):
    """`symmetric` of eigen.rs: a diagonal in [-3, 3) and per_row symmetric pairs a row."""
    r = Rng(seed)
    e = []
    for i in range(n):
        e.append((i, i, 3.0 * r.unit()))
        for _ in range(per_row):
            j = r.below(n)
            x = r.unit()
            if j != i:
                e.append((i, j, x))
                e.append((j, i, x))
    return build(n, e, np.float64)


def hermitian(seed, n, per_row):
    """`hermitian` of eigen.rs: as `symmetric`, the pairs complex conjugates."""
    r = Rng(seed)
    e = []
    for i in range(n):
        e.append((i, i, complex(3.0 * r.unit(), 0.0)))
        for _ in range(per_row):
            j = r.below(n)
            x = complex(r.unit(), r.unit())
            if j != i:
                e.append((i, j, x))
                e.append((j, i, x.conjugate()))
    return build(n, e, np.complex128)


def unsymmetric(seed, n, per_row):
    """`unsymmetric` of eigen.rs: a diagonal in [-3, 3) and per_row more elements a row."""
    r = Rng(seed)
    e = []
    for i in range(n):
        e.append((i, i, 3.0 * r.unit()))
        for _ in range(per_row):
            j = r.below(n)
            e.append((i, j, r.unit()))
    return build(n, e, np.float64)


def complex_unsymmetric(seed, n, per_row):
    """`complex_unsymmetric` of eigen.rs: as `unsymmetric`, the elements complex."""
    r = Rng(seed)
    e = []
    for i in range(n):
        e.append((i, i, complex(3.0 * r.unit(), 3.0 * r.unit())))
        for _ in range(per_row):
            j = r.below(n)
            e.append((i, j, complex(r.unit(), r.unit())))
    return build(n, e, np.complex128)


def rotations(seed, n):
    """`rotations` of eigen.rs: 2 x 2 blocks a·[[1, -b], [b, 1]] down the diagonal, so complex
    pairs a ± a·b·i, coupled by an upper triangle of random elements, which leaves them."""
    r = Rng(seed)
    e = []
    for p in range(n // 2):
        a = 1.0 + 2.0 * (r.unit() + 1.0)
        b = r.unit()
        i = 2 * p
        e += [(i, i, a), (i, i + 1, -a * b), (i + 1, i, a * b), (i + 1, i + 1, a)]
        for _ in range(2):
            j = r.below(n)
            if j > i + 1:
                e.append((i, j, r.unit()))
    return build(n, e, np.float64)


def lap1(n):
    return sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]).tocsr()


def lap2(nx, ny):
    return (sp.kron(sp.identity(ny), lap1(nx)) + sp.kron(lap1(ny), sp.identity(nx))).tocsr()


def clusters():
    """`clusters` of eigen.rs: a diagonal matrix of 200 with clusters at both ends."""
    d = [1.0 + j * 1e-3 for j in range(5)] + [5.0 + j * 1e-8 for j in range(3)]
    d += [10.0 + 0.05 * j for j in range(184)]
    d += [30.0, 30.0 + 1e-6, 30.0 + 2e-6, 31.0, 31.0 + 1e-4, 40.0, 40.0, 40.0]
    return sp.diags([np.array(d)], [0]).tocsr(), d


def lanczos_pick(w, k, which):
    w = np.sort(w)
    if which == "LA":
        return w[-k:]
    if which == "SA":
        return w[:k]
    if which == "LM":
        return np.sort(w[np.argsort(-np.abs(w), kind="stable")[:k]])
    if which == "BE":
        low = k // 2
        return np.sort(np.concatenate([w[:low], w[len(w) - (k - low):]]))
    raise ValueError(which)


def arnoldi_score(z, which, real):
    if which == "LM":
        return np.abs(z)
    if which == "LR":
        return z.real
    if which == "SR":
        return -z.real
    if which == "LI":
        return np.abs(z.imag) if real else z.imag
    if which == "SI":
        return -np.abs(z.imag) if real else -z.imag
    raise ValueError(which)


def check(name, got, want, tol):
    got = np.sort_complex(np.asarray(got, dtype=complex))
    want = np.sort_complex(np.asarray(want, dtype=complex))
    err = np.max(np.abs(got - want)) if len(got) else 0.0
    assert len(got) == len(want) and err <= tol, (name, got, want, err)


def real_const(name, values):
    body = ", ".join(repr(float(v)) for v in values)
    print(f"const {name}: [f64; {len(values)}] = [{body}];")


def complex_const(name, values):
    body = ", ".join(f"({float(z.real)!r}, {float(z.imag)!r})" for z in values)
    print(f"const {name}: [(f64, f64); {len(values)}] = [{body}];")


def main():
    print("// Generated by lib/tests/math/sparse/data/gen_eigen.py: ARPACK through SciPy, each")
    print("// value checked against LAPACK through numpy; for Arnoldi the LAPACK values, which")
    print("// ARPACK's were checked to be among (a tie of a pair at the k-th place allowed for).")
    # Lanczos: every selection on a random symmetric and a complex Hermitian matrix.
    for label, a in (("SYMMETRIC", symmetric(7, 150, 3)), ("HERMITIAN", hermitian(11, 120, 3))):
        dense = np.linalg.eigvalsh(a.toarray())
        for which in ("LA", "SA", "LM", "BE"):
            k = 5
            if label == "HERMITIAN" and which == "BE":
                # SciPy refuses `BE` for a complex matrix: the dense solver (LAPACK) instead.
                print("// HERMITIAN_BE from numpy.linalg.eigvalsh: eigsh has no `BE` for complex.")
                real_const(f"{label}_{which}", lanczos_pick(dense, k, which))
                continue
            w = np.sort(sla.eigsh(a, k=k, which=which, tol=0, return_eigenvectors=False))
            check(label + which, w, lanczos_pick(dense, k, which), 1e-10)
            real_const(f"{label}_{which}", w)
    # The 1-D and 2-D Laplacians: ARPACK against the closed forms.
    a = lap1(100)
    exact = np.sort([2 - 2 * np.cos(j * np.pi / 101) for j in range(1, 101)])
    for which in ("LA", "SA"):
        w = np.sort(sla.eigsh(a, k=4, which=which, tol=0, return_eigenvectors=False))
        check("LAP1" + which, w, lanczos_pick(exact, 4, which), 1e-12)
        real_const(f"LAPLACIAN_1D_{which}", w)
    nx = ny = 12
    a = lap2(nx, ny)
    exact = np.sort([4 - 2 * np.cos(i * np.pi / (nx + 1)) - 2 * np.cos(j * np.pi / (ny + 1))
                     for i in range(1, nx + 1) for j in range(1, ny + 1)])
    for which in ("LA", "SA"):
        w = np.sort(sla.eigsh(a, k=8, which=which, tol=0, ncv=40, return_eigenvectors=False))
        try:
            check("LAP2" + which, w, lanczos_pick(exact, 8, which), 1e-10)
            real_const(f"LAPLACIAN_2D_{which}", w)
        except AssertionError:
            # ARPACK missed a copy: the closed form is the reference, and the miss is recorded.
            print(f"// ARPACK (ncv 40, tol 0) missed a copy for LAPLACIAN_2D_{which}: {w!r}")
            real_const(f"LAPLACIAN_2D_{which}", lanczos_pick(exact, 8, which))
    # Shift-and-invert around an interior target of the 2-D Laplacian.
    w = np.sort(sla.eigsh(a, k=6, sigma=2.5, which="LM", tol=0, return_eigenvectors=False))
    check("LAP2SI", w, exact[np.argsort(np.abs(exact - 2.5), kind="stable")[:6]], 1e-10)
    real_const("LAPLACIAN_2D_NEAR_2_5", w)
    # Clusters.
    a, d = clusters()
    for which in ("LA", "SA"):
        w = np.sort(sla.eigsh(a, k=6, which=which, tol=0, ncv=30, return_eigenvectors=False))
        check("CLUSTER" + which, w, lanczos_pick(np.array(d), 6, which), 1e-10)
        real_const(f"CLUSTERS_{which}", w)
    # Shift-and-invert of the random symmetric and Hermitian matrices, interior targets.
    for label, a, sigma in (("SYMMETRIC", symmetric(7, 150, 3), 0.3),
                            ("HERMITIAN", hermitian(11, 120, 3), -0.5)):
        dense = np.linalg.eigvalsh(a.toarray())
        w = np.sort(sla.eigsh(a, k=4, sigma=sigma, which="LM", tol=0, return_eigenvectors=False))
        check(label + "SI", w, np.sort(dense[np.argsort(np.abs(dense - sigma))[:4]]), 1e-10)
        real_const(f"{label}_NEAR", w)
    # Arnoldi: every selection on a real and a complex unsymmetric matrix, and on real pairs.
    for label, a, real in (("UNSYMMETRIC", unsymmetric(13, 150, 3), True),
                           ("COMPLEX", complex_unsymmetric(17, 120, 3), False),
                           ("ROTATIONS", rotations(19, 100), True)):
        dense = np.linalg.eigvals(a.toarray())
        for which in ("LM", "LR", "SR", "LI", "SI"):
            if label == "UNSYMMETRIC" and which == "SI":
                # Every real eigenvalue ties at |Im| = 0: the selection is not defined there.
                continue
            k = 6
            try:
                w = sla.eigs(a, k=k, which=which, tol=0, maxiter=20000,
                             return_eigenvectors=False)
            except sla.ArpackNoConvergence as e:
                print(f"// ARPACK did not converge for {label}_{which}: {e}")
                w = e.eigenvalues
            score = arnoldi_score(dense, which, real)
            order = np.argsort(-score, kind="stable")
            # Ties (the two of a pair) make the k-th ambiguous: the reference takes the k best by
            # the dense solver, and ARPACK's are checked to be among the k + 1 best.
            want = dense[order[:k]]
            best = dense[order[:k + 1]]
            missed = [z for z in w if np.min(np.abs(best - z)) >= 1e-10]
            if missed:
                # Nearest the real axis is interior for a real matrix: ARPACK can settle on the
                # wrong ones. The LAPACK values stay the reference, and the miss is recorded.
                print(f"// ARPACK's {label}_{which} holds values not among the wanted: {missed}")
            complex_const(f"{label}_{which}", want)
    for label, a, sigma in (("UNSYMMETRIC", unsymmetric(13, 150, 3), 0.2),
                            ("COMPLEX", complex_unsymmetric(17, 120, 3), 0.1 + 0.2j)):
        dense = np.linalg.eigvals(a.toarray())
        w = sla.eigs(a, k=4, sigma=sigma, which="LM", tol=0, return_eigenvectors=False)
        near = dense[np.argsort(np.abs(dense - sigma), kind="stable")[:5]]
        for z in w:
            assert np.min(np.abs(near - z)) < 1e-10, (label, z)
        complex_const(f"{label}_NEAR", dense[np.argsort(np.abs(dense - sigma), kind="stable")[:4]])


if __name__ == "__main__":
    main()
