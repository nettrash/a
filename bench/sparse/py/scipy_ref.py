#!/usr/bin/env python3
"""SciPy references for the sparse scoreboard of nettrash-a.

Ours runs in this same process through ctypes (bench/pyffi, a C ABI over a::math::sparse whose
functions time `reps` repetitions with Rust's Instant), alternating sample by sample with the SciPy
call (perf_counter_ns around `reps` repetitions). The matrices are written by
`sparse-bench dump <name> <base>`, so both sides read the same arrays.

Usage: scipy_ref.py <bench binary> <libsparse_ours_ffi> <scratch dir> <group> [--quick]
Groups: storage io ordering lu iter eigen. Prints TSV rows (no header).
Run single-threaded: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 (ARPACK's dense work uses BLAS).
"""

import ctypes as C
import os
import subprocess
import sys
import time

import numpy as np
import scipy
import scipy.io
import scipy.io._fast_matrix_market as fmm
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import scipy.sparse.linalg as spla

fmm.PARALLELISM = 1  # fast_matrix_market uses every core by default

BIN, LIB, SCRATCH, GROUP = sys.argv[1:5]
QUICK = "--quick" in sys.argv
SAMPLES = int(os.environ.get("BENCH_SAMPLES", "15"))
BUDGET = float(os.environ.get("BENCH_ROW_BUDGET_S", "60"))
MIN_BATCH = float(os.environ.get("BENCH_MIN_BATCH_US", "1000")) * 1e3  # ns
SCIPY = f"SciPy {scipy.__version__}"

lib = C.CDLL(LIB)
P = C.c_void_p
D = C.c_double
U = C.c_uint64
I = C.c_int32
DP = C.POINTER(C.c_double)
UP = C.POINTER(C.c_uint64)
for name, res, args in [
    ("ours_new", P, [U, U, U, UP, UP, DP]),
    ("ours_free", None, [P]),
    ("t_spmv", D, [P, DP, DP, I, U]),
    ("t_spgemm", D, [P, P, U, UP]),
    ("t_add", D, [P, P, U]),
    ("t_transpose", D, [P, U]),
    ("t_coo", D, [U, U, UP, UP, DP, U]),
    ("t_permute", D, [P, UP, U]),
    ("t_submatrix", D, [P, U, U, U, U, U]),
    ("t_select", D, [P, UP, U, UP, U, U]),
    ("t_triangle", D, [P, I, U]),
    ("t_norm", D, [P, I, U, DP]),
    ("t_diagonal", D, [P, U]),
    ("t_scale", D, [P, U]),
    ("t_mmread", D, [C.c_char_p, U, I, U]),
    ("t_mmwrite", D, [P, C.c_char_p, U, I, U]),
    ("t_rcm", D, [P, U, UP]),
    ("t_components", D, [P, U, UP]),
    ("t_lu", D, [P, U, DP]),
    ("ours_lu_new", P, [P]),
    ("ours_lu_free", None, [P]),
    ("t_lu_solve", D, [P, U, DP, DP, U]),
    ("t_krylov", D, [P, I, I, D, U, U, DP, DP, U, DP]),
    ("t_eigen", D, [P, I, I, U, U, D, D, U, DP, U, DP, DP]),
]:
    f = getattr(lib, name)
    f.restype = res
    f.argtypes = args


def dp(a):
    return a.ctypes.data_as(DP)


def up(a):
    return a.ctypes.data_as(UP)


_cache = {}


def load(name):
    """The matrix `name` of the suite, as a canonical scipy CSR and our handle."""
    if name in _cache:
        return _cache[name]
    base = os.path.join(SCRATCH, "mat_" + name)
    if not os.path.exists(base + ".val"):
        subprocess.run([BIN, "dump", name, base], check=True)
    m, n, nnz = map(int, open(base + ".shape").read().split())
    ptr = np.fromfile(base + ".ptr", dtype=np.uint64)
    idx = np.fromfile(base + ".idx", dtype=np.uint64)
    val = np.fromfile(base + ".val", dtype=np.float64)
    A = sp.csr_matrix((val, idx.astype(np.int32), ptr.astype(np.int32)), shape=(m, n))
    A.has_canonical_format = True
    h = lib.ours_new(m, n, nnz, up(ptr), up(idx), dp(val))
    _cache[name] = (A, h)
    return A, h


def handle_of(A):
    A = A.tocsr()
    A.sort_indices()
    A.sum_duplicates()
    A.eliminate_zeros()
    ptr = A.indptr.astype(np.uint64)
    idx = A.indices.astype(np.uint64)
    val = np.ascontiguousarray(A.data, dtype=np.float64)
    return lib.ours_new(A.shape[0], A.shape[1], A.nnz, up(ptr), up(idx), dp(val))


def timed(f):
    """A Python callable f() turned into reps -> ns per op."""

    def run(reps):
        t = time.perf_counter_ns()
        for _ in range(reps):
            f()
        return (time.perf_counter_ns() - t) / reps

    return run


def race(cs):
    """cs: list of (name, run(reps) -> ns/op); the first is ours. Returns [(name, median, lo, hi, n)]."""
    reps = []
    for _, run in cs:
        t = run(1)
        r = 1
        if t < MIN_BATCH:
            est = max(1, int(MIN_BATCH / max(t, 1.0)))
            t = run(est)
            r = max(1, int(np.ceil(MIN_BATCH / max(t, 1.0))))
        reps.append((r, t))
    rnd = sum(r * t for r, t in reps) / 1e9
    samples = SAMPLES
    if rnd * SAMPLES > BUDGET:
        floor = 3 if rnd > 20 else 5
        samples = int(min(SAMPLES, max(floor, BUDGET // max(rnd, 1e-9))))
    times = [[] for _ in cs]
    k = len(cs)
    for s in range(samples):
        for off in range(k):
            ci = (s + off) % k
            times[ci].append(cs[ci][1](reps[ci][0]))
    out = []
    for (name, _), v in zip(cs, times):
        v = sorted(v)
        q = lambda p: v[int(round((len(v) - 1) * p))]
        out.append((name, q(0.5), q(0.1), q(0.9), len(v)))
    return out


def sig3(x):
    if x == 0 or not np.isfinite(x):
        return str(x)
    d = 2 - int(np.floor(np.log10(abs(x))))
    return f"{x:.{d}f}" if d > 0 else f"{x:.0f}"


def spread(st):
    name, med, lo, hi, n = st
    s = []
    if (hi - lo) / med > 0.10:
        s.append(f"spread p10-p90 {100 * (hi - lo) / med:.0f}%")
    if n < 15:
        s.append(f"{n} samples")
    return "; ".join(s)


def rows(group, op, params, stats, notes=None):
    notes = notes or {}
    ours = stats[0]
    for i, st in enumerate(stats[1:], 1):
        n = [x for x in [notes.get(0, ""), notes.get(i, "")] if x]
        n.append("same process via ctypes; SciPy timed with perf_counter_ns")
        if spread(ours):
            n.append("ours " + spread(ours))
        if spread(st):
            n.append("ref " + spread(st))
        print(f"sparse\t{group}\t{op}\t{params}\t{ours[1]:.0f}\t{st[0]}\t{st[1]:.0f}\t{sig3(ours[1] / st[1])}\t{'; '.join(n)}", flush=True)


def raw(group, op, params, ours, ref, refv, note):
    print(f"sparse\t{group}\t{op}\t{params}\t{ours:.0f}\t{ref}\t{refv:.0f}\t{sig3(ours / refv)}\t{note}", flush=True)


def note(s):
    print(f"[scipy_ref] {s}", file=sys.stderr, flush=True)


def params(name, A):
    return f"{name} n={A.shape[0]} nnz={A.nnz}"


def want(case):
    """PY_CASES (comma-separated case keys such as bicgstab:upwind2d-1000) selects cases; unset:
    all of them."""
    sel = os.environ.get("PY_CASES", "")
    return not sel or case in sel.split(",")


def rs_e(x):
    """x as Rust's {:e} writes it (1e-8, not 1.000000e-08), so that the params of the rows of
    both sides match."""
    m, e = f"{x:e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e{int(e)}"


def rand_vec(n, seed):
    return np.random.default_rng(seed).uniform(-1, 1, n)


# ------------------------------------------------------------------------------------------------
# storage


def g_storage():
    from scipy.sparse import _sparsetools as st

    spmv_set = ["lap2d-100", "randunsym-100000"] if QUICK else ["lap2d-100", "lap2d-1000", "lap3d-100", "elast3d-30", "randunsym-1000000", "circuit-1000000"]
    for name in spmv_set:
        note(f"spmv {name}")
        A, h = load(name)
        m, n = A.shape
        x = rand_vec(n, 1)
        y = np.zeros(m)
        lib.t_spmv(h, dp(x), dp(y), 0, 1)
        assert np.allclose(y, A @ x, rtol=1e-13, atol=1e-13)
        y2 = np.zeros(m)

        def kernel():
            y2[:] = 0
            st.csr_matvec(m, n, A.indptr, A.indices, A.data, x, y2)

        stats = race([("ours", lambda r: lib.t_spmv(h, dp(x), dp(y), 0, r)), (f"{SCIPY} A @ x", timed(lambda: A @ x)), (f"{SCIPY} _sparsetools.csr_matvec", timed(kernel))])
        rows("storage", "mul_vec (A·x)", params(name, A), stats, {1: "allocates the result", 2: "int32; y zeroed then y += A·x; no allocation"})
    for name in ["randunsym-100000"] if QUICK else ["randunsym-1000000", "elast3d-30", "lap2d-1000"]:
        note(f"spmv_t {name}")
        A, h = load(name)
        m, n = A.shape
        x = rand_vec(m, 2)
        y = np.zeros(n)
        lib.t_spmv(h, dp(x), dp(y), 1, 1)
        assert np.allclose(y, A.T @ x, rtol=1e-12, atol=1e-12)
        At = A.T  # a CSC view, no copy
        stats = race([("ours", lambda r: lib.t_spmv(h, dp(x), dp(y), 1, r)), (f"{SCIPY} A.T @ x", timed(lambda: At @ x))])
        rows("storage", "mul_vec_transpose (Aᵀ·x)", params(name, A), stats, {1: "csc_matvec on the transpose view; allocates"})
    for name in ["lap2d-316"] if QUICK else ["lap2d-316", "lap3d-46", "randunsym-100000", "elast3d-16"]:
        note(f"spgemm {name}")
        A, h = load(name)
        nn = U(0)
        lib.t_spgemm(h, h, 1, C.byref(nn))
        Cm = A @ A
        stats = race([("ours", lambda r: lib.t_spgemm(h, h, r, C.byref(nn))), (f"{SCIPY} A @ A", timed(lambda: A @ A))])
        rows("storage", "sparse × sparse (A·A)", params(name, A), stats, {0: f"nnz(A·A)={nn.value}", 1: f"SMMP; result nnz {Cm.nnz}, has_sorted_indices={Cm.has_sorted_indices}"})
    for name in ["randunsym-100000"] if QUICK else ["randunsym-100000", "randunsym-1000000"]:
        note(f"add {name}")
        A, h = load(name)
        B = A.T.tocsr()
        hb = handle_of(B)
        stats = race([("ours", lambda r: lib.t_add(h, hb, r)), (f"{SCIPY} A + B", timed(lambda: A + B))])
        rows("storage", "add (A + B)", params(name, A), stats, {0: "B = Aᵀ"})
        lib.ours_free(hb)
        note(f"scale {name}")
        stats = race([("ours", lambda r: lib.t_scale(h, r)), (f"{SCIPY} A * 1.5", timed(lambda: A * 1.5))])
        rows("storage", "scale (A·α)", params(name, A), stats)
    for name in ["randunsym-100000"] if QUICK else ["randunsym-1000000", "lap3d-100", "elast3d-30"]:
        note(f"transpose {name}")
        A, h = load(name)
        stats = race([("ours", lambda r: lib.t_transpose(h, r)), (f"{SCIPY} A.T.tocsr()", timed(lambda: A.T.tocsr())), (f"{SCIPY} A.tocsc()", timed(lambda: A.tocsc()))])
        rows("storage", "transpose (Aᵀ)", params(name, A), stats)
    # Assembly: the same triplets as the Rust side's FE assembly (k × k Q1) and random ones.
    for k in [300] if QUICK else [300, 1000]:
        note(f"assembly fe {k}")
        nodes = (k + 1) * (k + 1)
        ke = np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]], dtype=float) / 6.0
        ey, ex = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        n0 = (ey * (k + 1) + ex).ravel()
        ids = np.stack([n0, n0 + 1, n0 + k + 2, n0 + k + 1], axis=1)
        ri = np.repeat(ids, 4, axis=1).ravel().astype(np.uint64)
        ci = np.tile(ids, (1, 4)).ravel().astype(np.uint64)
        v = np.tile(ke.ravel(), k * k)
        coo = sp.coo_matrix((v, (ri.astype(np.int32), ci.astype(np.int32))), shape=(nodes, nodes))
        stats = race([("ours", lambda r: lib.t_coo(nodes, len(v), up(ri), up(ci), dp(v), r)), (f"{SCIPY} coo.tocsr()", timed(lambda: coo.tocsr()))])
        rows("storage", "assembly (triplets → CSR)", f"FE assembly {k}×{k} Q1, {len(v)} triplets", stats, {1: "sums repeats; int32"})
    nr, nz = (100_000, 1_000_000) if QUICK else (1_000_000, 10_000_000)
    note("assembly random")
    rng = np.random.default_rng(9)
    ri = rng.integers(0, nr, nz).astype(np.uint64)
    ci = rng.integers(0, nr, nz).astype(np.uint64)
    v = rng.uniform(0, 1, nz)
    coo = sp.coo_matrix((v, (ri.astype(np.int32), ci.astype(np.int32))), shape=(nr, nr))
    stats = race([("ours", lambda r: lib.t_coo(nr, nz, up(ri), up(ci), dp(v), r)), (f"{SCIPY} coo.tocsr()", timed(lambda: coo.tocsr()))])
    rows("storage", "assembly (triplets → CSR)", f"random triplets n={nr}, {nz} triplets", stats, {1: "sums repeats; int32; triplets from numpy's generator (not the Rust side's)"})
    for name in ["lap2d-316"] if QUICK else ["lap2d-1000", "elast3d-30"]:
        note(f"permute {name}")
        A, h = load(name)
        n = A.shape[0]
        p = np.random.default_rng(10).permutation(n).astype(np.uint64)
        pi = p.astype(np.int64)
        stats = race([("ours", lambda r: lib.t_permute(h, up(p), r)), (f"{SCIPY} A[p][:, p]", timed(lambda: A[pi][:, pi]))])
        rows("storage", "permute (P·A·Pᵀ)", params(name, A), stats, {1: "fancy indexing, two passes"})
    name = "lap2d-316" if QUICK else "lap2d-1000"
    note(f"submatrix {name}")
    A, h = load(name)
    n = A.shape[0]
    r0, r1 = n // 4, 3 * n // 4
    stats = race([("ours", lambda r: lib.t_submatrix(h, r0, r1, r0, r1, r)), (f"{SCIPY} A[r0:r1, c0:c1]", timed(lambda: A[r0:r1, r0:r1]))])
    rows("storage", "submatrix (ranges, middle half)", params(name, A), stats)
    rr = np.random.default_rng(12).permutation(n)[: n // 2].astype(np.uint64)
    cc = np.random.default_rng(13).permutation(n)[: n // 2].astype(np.uint64)
    ri64, ci64 = rr.astype(np.int64), cc.astype(np.int64)
    stats = race([("ours", lambda r: lib.t_select(h, up(rr), len(rr), up(cc), len(cc), r)), (f"{SCIPY} A[rows][:, cols]", timed(lambda: A[ri64][:, ci64]))])
    rows("storage", "select (random half of rows and columns)", params(name, A), stats, {1: "numpy's permutation, not the Rust side's"})
    name = "randunsym-100000" if QUICK else "randunsym-1000000"
    note(f"triangles/norms {name}")
    A, h = load(name)
    stats = race([("ours", lambda r: lib.t_triangle(h, 1, r)), (f"{SCIPY} sp.triu(A, format='csr')", timed(lambda: sp.triu(A, format="csr")))])
    rows("storage", "upper_triangle", params(name, A), stats)
    stats = race([("ours", lambda r: lib.t_triangle(h, 0, r)), (f"{SCIPY} sp.tril(A, format='csr')", timed(lambda: sp.tril(A, format="csr")))])
    rows("storage", "lower_triangle", params(name, A), stats)
    out = D(0)
    for kind, op, ref in [(0, "norm(One)", 1), (1, "norm(Infinity)", np.inf), (2, "norm(Frobenius)", "fro")]:
        lib.t_norm(h, kind, 1, C.byref(out))
        w = spla.norm(A, ref)
        assert abs(out.value - w) <= 1e-12 * w, (op, out.value, w)
        stats = race([("ours", lambda r, kind=kind: lib.t_norm(h, kind, r, C.byref(out))), (f"{SCIPY} sparse.linalg.norm", timed(lambda ref=ref: spla.norm(A, ref)))])
        rows("storage", op, params(name, A), stats, {0: "ours compensated sums (and scaled, Frobenius)", 1: "plain sums"})
    stats = race([("ours", lambda r: lib.t_norm(h, 3, r, C.byref(out))), (f"{SCIPY} abs(A).max()", timed(lambda: abs(A).max()))])
    rows("storage", "norm(Max)", params(name, A), stats)
    name = "lap3d-30" if QUICK else "lap3d-100"
    A, h = load(name)
    stats = race([("ours", lambda r: lib.t_diagonal(h, r)), (f"{SCIPY} A.diagonal()", timed(lambda: A.diagonal()))])
    rows("storage", "diagonal", params(name, A), stats)


# ------------------------------------------------------------------------------------------------
# io


def g_io():
    for name in ["lap2d-316"] if QUICK else ["lap2d-316", "lap2d-1000", "randunsym-1000000", "elast3d-30"]:
        A, h = load(name)
        for sym in [False, True] if not name.startswith("randunsym") else [False]:
            note(f"io {name} sym={sym}")
            path = os.path.join(SCRATCH, f"mm_{name}_{'sym' if sym else 'gen'}.mtx")
            pb = path.encode()
            lib.t_mmwrite(h, pb, len(pb), int(sym), 1)
            mb = os.path.getsize(path) / 1e6
            B = scipy.io.mmread(path, spmatrix=True).tocsr()
            assert (abs(B - A)).max() == 0, "scipy reads our file differently"
            pp = f"{params(name, A)} {'symmetric' if sym else 'general'} file {mb:.1f} MB"
            stats = race([("ours", lambda r: lib.t_mmread(pb, len(pb), 0, r)), (f"{SCIPY} io.mmread (fast_matrix_market, 1 thread)", timed(lambda: scipy.io.mmread(path, spmatrix=True)))])
            rows("io", "read_matrix_market (to COO)", pp, stats, {0: "from a file in the page cache", 1: "to coo_matrix; PARALLELISM = 1"})
            stats = race([("ours", lambda r: lib.t_mmread(pb, len(pb), 1, r)), (f"{SCIPY} io.mmread(...).tocsr()", timed(lambda: scipy.io.mmread(path, spmatrix=True).tocsr()))])
            rows("io", "SparseMatrix::read_matrix_market (to CSR)", pp, stats)
            wpath = os.path.join(SCRATCH, f"mmw_{name}.mtx")
            wb = wpath.encode()
            symm = "symmetric" if sym else "general"
            stats = race([("ours", lambda r: lib.t_mmwrite(h, wb, len(wb), int(sym), r)), (f"{SCIPY} io.mmwrite (fast_matrix_market, 1 thread)", timed(lambda: scipy.io.mmwrite(wpath, A, symmetry=symm)))])
            rows("io", "write_matrix_market", pp, stats, {0: "to a file; shortest round-trip digits", 1: "17 significant digits by default"})


# ------------------------------------------------------------------------------------------------
# ordering


def band_profile(A, p):
    S = (A + A.T).tocsr()
    S.setdiag(0)
    S.eliminate_zeros()
    pinv = np.empty_like(p)
    pinv[p] = np.arange(len(p))
    Sp = S.tocoo()
    i, j = pinv[Sp.row], pinv[Sp.col]
    band = int(np.max(np.abs(i - j))) if len(i) else 0
    first = np.arange(len(p))
    np.minimum.at(first, i, j)
    return band, int(np.sum(np.arange(len(p)) - first + 1))


def scrambled(name, seed):
    """The same scrambling as the Rust side's mats::scrambled is not reproducible here: a numpy
    permutation is used, and both sides order the same scrambled matrix."""
    A, _ = load(name)
    p = np.random.default_rng(seed).permutation(A.shape[0])
    B = A[p][:, p].tocsr()
    return B, handle_of(B)


def g_ordering():
    for name in ["lap2d-100"] if QUICK else ["lap2d-316", "lap3d-46", "circuit-100000", "lap2d-1000"]:
        note(f"rcm {name}")
        A, h = scrambled(name, 6)
        n = A.shape[0]
        po = np.zeros(n, dtype=np.uint64)
        lib.t_rcm(h, 1, up(po))
        ps = csg.reverse_cuthill_mckee(A, symmetric_mode=True)
        bo, pro = band_profile(A, po.astype(np.int64))
        bs, prs = band_profile(A, ps.astype(np.int64))
        pp = f"{name} scrambled n={n} nnz={A.nnz}"
        raw("ordering", "reverse_cuthill_mckee band", pp + " [values: band width, not ns]", bo, f"{SCIPY} csgraph.reverse_cuthill_mckee", bs, "quality: lower is better")
        raw("ordering", "reverse_cuthill_mckee profile", pp + " [values: profile, not ns]", pro, f"{SCIPY} csgraph.reverse_cuthill_mckee", prs, "quality: lower is better")
        stats = race([("ours", lambda r: lib.t_rcm(h, r, up(po))), (f"{SCIPY} csgraph.reverse_cuthill_mckee", timed(lambda: csg.reverse_cuthill_mckee(A, symmetric_mode=True)))])
        rows("ordering", "reverse_cuthill_mckee", pp, stats, {0: "scrambled by numpy's permutation", 1: "symmetric_mode=True (reads the pattern as given)"})
        lib.ours_free(h)
    for name in ["circuit-10000"] if QUICK else ["circuit-1000000", "lap3d-100"]:
        note(f"components {name}")
        A, h = load(name)
        cnt = U(0)
        lib.t_components(h, 1, C.byref(cnt))
        ncs, _ = csg.connected_components(A, directed=False)
        assert ncs == cnt.value
        stats = race([("ours", lambda r: lib.t_components(h, r, C.byref(cnt))), (f"{SCIPY} csgraph.connected_components", timed(lambda: csg.connected_components(A, directed=False)))])
        rows("ordering", "connected_components", params(name, A), stats, {1: "directed=False"})


# ------------------------------------------------------------------------------------------------
# lu


def g_lu():
    for name in ["convdiff2d-100", "circuit-10000"] if QUICK else ["convdiff2d-100", "convdiff2d-316", "upwind2d-316", "convdiff3d-22", "circuit-10000", "tridiag-100000"]:
        note(f"splu {name}")
        A, h = load(name)
        Ac = A.tocsc()
        n = A.shape[0]
        out = (D * 2)()
        lib.t_lu(h, 1, out)
        lu = spla.splu(Ac)
        b = rand_vec(n, 51)
        x = lu.solve(b)
        assert np.linalg.norm(A @ x - b) <= 1e-8 * np.linalg.norm(b)
        stats = race([("ours", lambda r: lib.t_lu(h, r, out)), (f"{SCIPY} splu (SuperLU, COLAMD)", timed(lambda: spla.splu(Ac)))])
        rows("lu", "SparseLu::new", params(name, A), stats, {0: f"nnz(L+U) ours {int(out[0])}", 1: f"diag_pivot_thresh 1, SciPy defaults; nnz(L+U) {lu.L.nnz + lu.U.nnz}"})
        f = lib.ours_lu_new(h)
        xs = np.zeros(n)
        stats = race([("ours", lambda r: lib.t_lu_solve(f, n, dp(b), dp(xs), r)), (f"{SCIPY} SuperLU.solve", timed(lambda: lu.solve(b)))])
        rows("lu", "SparseLu::solve_vec", params(name, A), stats)
        lib.ours_lu_free(f)


# ------------------------------------------------------------------------------------------------
# iterative

RTOL = 1e-8


def counted(solver, A, b, **kw):
    it = [0]

    def cb(_):
        it[0] += 1

    x, info = solver(A, b, callback=cb, **kw)
    return x, info, it[0]


def g_iter():
    spd = ["lap2d-100"] if QUICK else ["lap2d-316", "lap2d-1000", "lap3d-46", "lap3d-100"]
    uns = ["upwind2d-100"] if QUICK else ["upwind2d-316", "upwind2d-1000", "convdiff3d-46"]
    ind = ["lap2d-100"] if QUICK else ["lap2d-316", "lap3d-46"]
    cases = [(nm, 0, "cg", spla.cg) for nm in spd] + [(nm, 2, "gmres", spla.gmres) for nm in uns] + [(nm, 3, "bicgstab", spla.bicgstab) for nm in uns] + [(nm, 1, "minres", spla.minres) for nm in ind]
    for name, method, mname, solver in cases:
        if not want(f"{mname}:{name}"):
            continue
        A, h = load(name)
        label = name
        if method == 1:
            A = (A - 0.5 * sp.identity(A.shape[0], format="csr")).tocsr()
            h = handle_of(A)
            label = f"{name} − 0.5·I"
        n = A.shape[0]
        b = rand_vec(n, 70 + method)
        dinv = 1.0 / A.diagonal()
        for pc in [0, 1]:
            note(f"{mname} {label} precond={pc}")
            M = spla.LinearOperator((n, n), matvec=lambda r: dinv * r) if pc else None
            kw = dict(rtol=RTOL, atol=0.0, maxiter=100_000, M=M)
            if method == 2:
                kw["restart"] = 30
            if method == 1:
                kw = dict(rtol=RTOL, maxiter=100_000, M=M)
            out = (D * 4)()
            x = np.zeros(n)
            lib.t_krylov(h, method, pc, RTOL, 100_000, 30, dp(b), dp(x), 1, out)
            ours_true = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
            xs, info, its = counted(solver, A, b, **({**kw, "callback_type": "pr_norm"} if method == 2 else kw))
            ref_true = np.linalg.norm(A @ xs - b) / np.linalg.norm(b)
            stats = race([("ours", lambda r: lib.t_krylov(h, method, pc, RTOL, 100_000, 30, dp(b), dp(x), r, out)), (f"{SCIPY} sparse.linalg.{mname}", timed(lambda: solver(A, b, **kw)))])
            pre = "Jacobi" if pc else "none"
            pp = f"{label} n={n} nnz={A.nnz} rtol={rs_e(RTOL)}"
            extra = "restart 30; " if method == 2 else ""
            rows("iterative", f"{mname}_with ({pre})", pp, stats, {0: f"{extra}ours {int(out[0])} iterations ({int(out[1])} products), true residual {ours_true:.1e}", 1: f"{its} iterations (callbacks), info {info}, true residual {ref_true:.1e}; Python loop around compiled kernels; Jacobi as a LinearOperator"})
        if method == 1:
            lib.ours_free(h)


# ------------------------------------------------------------------------------------------------
# eigen

K, NCV, TOL = 6, 20, 1e-10


def exact_ani(name):
    """The eigenvalues of the anisotropic grids (mats.rs ani2d/ani3d), ascending, or None."""
    fam, k = name.split("-")
    k = int(k)
    c = np.cos(np.arange(1, k + 1) * np.pi / (k + 1))
    if fam == "ani2d":
        w = 1.37
        return np.sort(((2 + 2 * w) - 2 * c[:, None] - 2 * w * c[None, :]).ravel())
    if fam == "cd2d":
        gx, gy = 2 * np.sqrt(1 - 0.05**2), 2 * np.sqrt(1 - 0.03**2)
        return np.sort((4 + gx * c[:, None] + gy * c[None, :]).ravel())
    if fam == "ani3d":
        wx, wy, wz = 1.71, 1.37, 1.0
        return np.sort((2 * (wx + wy + wz) - 2 * wx * c[:, None, None] - 2 * wy * c[None, :, None] - 2 * wz * c[None, None, :]).ravel())
    return None


def g_eigen():
    v0s = {}

    def v0(n):
        if n not in v0s:
            v0s[n] = 1.0 + (np.arange(n) * 7919 % 101) / 101.0
        return v0s[n]

    # Anisotropic grids: no multiple eigenvalues (see g_eigen.rs).
    # (Grids of 10⁶ unknowns take minutes a call for both: they stop at 10⁵, see g_eigen.rs.)
    sym = [("ani2d-100", "LA"), ("ani2d-100", "SA")] if QUICK else [("ani2d-316", "LA"), ("ani3d-22", "LA"), ("randspd-100000", "LA"), ("ani2d-100", "SA"), ("ani3d-22", "SA")]
    for name, which in sym:
        if not want(f"eigsh:{name}:{which}"):
            continue
        note(f"eigsh {name} {which}")
        A, h = load(name)
        n = A.shape[0]
        vals = (D * (4 * K))()
        out = (D * 3)()
        w = 0 if which == "LA" else 1
        lib.t_eigen(h, 0, w, K, NCV, TOL, 0.0, 10_000, dp(v0(n)), 1, vals, out)
        ev = np.linalg.norm
        ref = spla.eigsh(A, k=K, which=which, ncv=NCV, tol=TOL, v0=v0(n), maxiter=100_000, return_eigenvectors=True)
        ours_v = np.sort(np.array(vals[:K]))
        agree = np.max(np.abs(ours_v - np.sort(ref[0]))) / spla.norm(A, 1)
        cnt = [0]

        def mv(v):
            cnt[0] += 1
            return A @ v

        spla.eigsh(spla.LinearOperator(A.shape, matvec=mv, dtype=float), k=K, which=which, ncv=NCV, tol=TOL, v0=v0(n), maxiter=100_000)
        stats = race([("ours", lambda r: lib.t_eigen(h, 0, w, K, NCV, TOL, 0.0, 10_000, dp(v0(n)), r, vals, out)), (f"{SCIPY} eigsh (ARPACK)", timed(lambda: spla.eigsh(A, k=K, which=which, ncv=NCV, tol=TOL, v0=v0(n), maxiter=100_000)))])
        pp = f"{name} n={n} nnz={A.nnz} k={K} basis={NCV} tol={rs_e(TOL)} which={which}"
        ex = exact_ani(name)
        exn = ""
        if ex is not None:
            target = ex[-K:] if which == "LA" else ex[:K]
            eo = np.max(np.abs(ours_v - np.sort(target))) / spla.norm(A, 1)
            er = np.max(np.abs(np.sort(ref[0]) - np.sort(target))) / spla.norm(A, 1)
            exn = f"; against the closed form: ours {eo:.1e}, ARPACK {er:.1e} (·‖A‖₁){' — ARPACK returned a wrong set' if er > 1e-8 else ''}"
        rows("eigen", "lanczos_with", pp, stats, {0: f"ours {int(out[0])} converged, {int(out[1])} restarts, {int(out[2])} products; eigenvalues agree to {agree:.1e}·‖A‖₁{exn}", 1: f"ARPACK {cnt[0]} products; with eigenvectors; ARPACK tests ‖r‖ ≤ tol·|λ|, ours ≤ tol·‖A‖; same start vector"})
    for name in ["ani2d-100"] if QUICK else ["ani2d-316", "ani3d-22"]:
        if not want(f"eigsh-si:{name}"):
            continue
        note(f"eigsh shift-invert {name}")
        A, h = load(name)
        n = A.shape[0]
        vals = (D * (4 * K))()
        out = (D * 3)()
        lib.t_eigen(h, 2, 0, K, NCV, TOL, 0.0, 10_000, dp(v0(n)), 1, vals, out)
        assert out[0] >= K, "ours shift-invert failed"
        stats = race([("ours", lambda r: lib.t_eigen(h, 2, 0, K, NCV, TOL, 0.0, 10_000, dp(v0(n)), r, vals, out)), (f"{SCIPY} eigsh(sigma=0) (ARPACK + SuperLU)", timed(lambda: spla.eigsh(A, k=K, sigma=0.0, which="LM", ncv=NCV, tol=TOL, v0=v0(n))))])
        pp = f"{name} n={n} nnz={A.nnz} k={K} basis={NCV} tol={rs_e(TOL)} sigma=0"
        rows("eigen", "lanczos_shift_invert_with", pp, stats, {0: f"ours {int(out[0])} converged, {int(out[1])} restarts, {int(out[2])} products; SparseLu of A − σI", 1: "splu of A − σI inside"})
    # cd2d: mild convection with a closed form (see g_eigen.rs).
    uns = [("cd2d-100", "LM")] if QUICK else [("cd2d-316", "LM"), ("circuit-100000", "LM"), ("randunsym-100000", "LM"), ("cd2d-100", "LR")]
    for name, which in uns:
        if not want(f"eigs:{name}:{which}"):
            continue
        note(f"eigs {name} {which}")
        A, h = load(name)
        n = A.shape[0]
        vals = (D * (4 * K))()
        out = (D * 3)()
        w = 0 if which == "LM" else 1
        lib.t_eigen(h, 1, w, K, NCV, TOL, 0.0, 10_000, dp(v0(n)), 1, vals, out)
        m = int(out[0]); ov = np.array(vals[: 2 * m]).reshape(m, 2)
        ov = ov[:, 0] + 1j * ov[:, 1]
        cnt = [0]

        def mv(v):
            cnt[0] += 1
            return A @ v

        rv = spla.eigs(spla.LinearOperator(A.shape, matvec=mv, dtype=float), k=K, which=which, ncv=NCV, tol=TOL, v0=v0(n), maxiter=100_000, return_eigenvectors=False)
        key = (lambda z: -abs(z)) if which == "LM" else (lambda z: -z.real)
        agree = max(abs(a - b) for a, b in zip(sorted(ov[: int(out[0])], key=key), sorted(rv, key=key))) / spla.norm(A, 1)
        ex = exact_ani(name)
        if ex is not None:
            target = ex[::-1][:K]
            eo = max(abs(abs(a) - b) for a, b in zip(sorted(ov[:K], key=key), target)) / spla.norm(A, 1)
            er = max(abs(abs(a) - b) for a, b in zip(sorted(rv, key=key), target)) / spla.norm(A, 1)
            agree_s = f"{agree:.1e}·‖A‖₁; against the closed form: ours {eo:.1e}, ARPACK {er:.1e}{' — ARPACK returned a wrong set' if er > 1e-8 else ''}"
        else:
            agree_s = f"{agree:.1e}·‖A‖₁"
        stats = race([("ours", lambda r: lib.t_eigen(h, 1, w, K, NCV, TOL, 0.0, 10_000, dp(v0(n)), r, vals, out)), (f"{SCIPY} eigs (ARPACK)", timed(lambda: spla.eigs(A, k=K, which=which, ncv=NCV, tol=TOL, v0=v0(n), maxiter=100_000)))])
        pp = f"{name} n={n} nnz={A.nnz} k={K} basis={NCV} tol={rs_e(TOL)} which={which}"
        rows("eigen", "arnoldi_with", pp, stats, {0: f"ours {int(out[0])} converged, {int(out[1])} restarts, {int(out[2])} products; eigenvalues agree to {agree_s}", 1: f"ARPACK {cnt[0]} products; with eigenvectors; same start vector"})
    for name in ["cd2d-100"] if QUICK else ["cd2d-316"]:
        if not want(f"eigs-si:{name}"):
            continue
        note(f"eigs shift-invert {name}")
        A, h = load(name)
        n = A.shape[0]
        vals = (D * (4 * K))()
        out = (D * 3)()
        # σ = −0.5, left of the spectrum: the eigenvalues of smallest real part. (A shift inside
        # the spectrum of this non-normal operator makes A − σI singular to working precision,
        # cond₁ ~ 1e24, which ours refuses and ARPACK + SuperLU solves into noise.)
        lib.t_eigen(h, 3, 0, K, NCV, TOL, -0.5, 10_000, dp(v0(n)), 1, vals, out)
        assert out[0] >= K, "ours shift-invert failed"
        stats = race([("ours", lambda r: lib.t_eigen(h, 3, 0, K, NCV, TOL, -0.5, 10_000, dp(v0(n)), r, vals, out)), (f"{SCIPY} eigs(sigma=-0.5) (ARPACK + SuperLU)", timed(lambda: spla.eigs(A, k=K, sigma=-0.5, which="LM", ncv=NCV, tol=TOL, v0=v0(n))))])
        pp = f"{name} n={n} nnz={A.nnz} k={K} basis={NCV} tol={rs_e(TOL)} sigma=-0.5"
        rows("eigen", "arnoldi_shift_invert_with", pp, stats, {0: f"ours {int(out[0])} converged, {int(out[1])} restarts, {int(out[2])} products"})


{"storage": g_storage, "io": g_io, "ordering": g_ordering, "lu": g_lu, "iter": g_iter, "eigen": g_eigen}[GROUP]()
