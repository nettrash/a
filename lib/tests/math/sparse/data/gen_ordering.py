"""Reference values of the ordering tests of issue #22 (`../ordering.rs`).

The graphs and matrices are built here exactly as the tests build them, from the same xorshift
generator as `../common.rs`, and for each this prints what SciPy and an exact minimum degree
ordering make of it:

* for a symmetric pattern: the band width and the profile after
  `scipy.sparse.csgraph.reverse_cuthill_mckee`, and nnz(L) of the Cholesky factor (the diagonal
  counted) in the natural order, after SuperLU's `MMD_AT_PLUS_A` ordering
  (`scipy.sparse.linalg.splu(..., permc_spec='MMD_AT_PLUS_A')`, its `perm_c` read as
  `perm_c[old] = new`, which reproduces SuperLU's own `L.nnz`) and after a true minimum degree
  ordering done here on the elimination graph (ties to the lowest index);
* for a column ordering of a matrix `A` of any shape: nnz(L) of the Cholesky factor of
  `(A·Q)ᵀ·(A·Q)` for the natural `Q`, for SuperLU's `COLAMD` (square matrices only: SuperLU
  orders nothing else) and for a true minimum degree ordering of the pattern of `AᵀA`.

The counts are symbolic: they depend on the pattern alone. Run from this directory:
`python3 gen_ordering.py`. Deterministic; prints the two Rust tables of `ordering.rs`.
"""

import heapq

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import scipy.sparse.linalg as spla

MASK = (1 << 64) - 1


class Rng:
    """The xorshift generator of `common.rs`."""

    def __init__(self, seed):
        self.s = seed & MASK

    def next(self):
        s = self.s
        s ^= (s << 13) & MASK
        s ^= s >> 7
        s ^= (s << 17) & MASK
        self.s = s
        return s

    def below(self, n):
        return self.next() % n


# The families of graphs, each as (n, edges); an edge may repeat or be a loop.


def grid(kx, ky):
    e = []
    for y in range(ky):
        for x in range(kx):
            v = y * kx + x
            if x + 1 < kx:
                e.append((v, v + 1))
            if y + 1 < ky:
                e.append((v, v + kx))
    return kx * ky, e


def grid9(kx, ky):
    n, e = grid(kx, ky)
    for y in range(ky - 1):
        for x in range(kx):
            v = y * kx + x
            if x + 1 < kx:
                e.append((v, v + kx + 1))
            if x > 0:
                e.append((v, v + kx - 1))
    return n, e


def grid3(k):
    e = []
    for z in range(k):
        for y in range(k):
            for x in range(k):
                v = (z * k + y) * k + x
                if x + 1 < k:
                    e.append((v, v + 1))
                if y + 1 < k:
                    e.append((v, v + k))
                if z + 1 < k:
                    e.append((v, v + k * k))
    return k * k * k, e


def band(n, w):
    return n, [(i, j) for i in range(n) for j in range(i + 1, min(n, i + w + 1))]


def arrow(n, hubs):
    return n, [(h, v) for h in range(hubs) for v in range(n) if v != h]


def tree(n):
    return n, [(v, (v - 1) // 2) for v in range(1, n)]


def path(n):
    return n, [(v, v + 1) for v in range(n - 1)]


def random_graph(n, m, seed):
    r = Rng(seed)
    e = []
    for _ in range(m):
        u = r.below(n)
        v = r.below(n)
        e.append((u, v))
    return n, e


def union(*parts):
    n, e = 0, []
    for m, f in parts:
        e += [(u + n, v + n) for u, v in f]
        n += m
    return n, e


def scrambled(graph, mult, add):
    n, e = graph
    assert np.gcd(mult, n) == 1
    label = lambda v: (v * mult + add) % n
    return n, [(label(u), label(v)) for u, v in e]


SYMMETRIC = [
    ("the 30 x 30 grid", grid(30, 30)),
    ("the band of half-width 9, n = 120", band(120, 9)),
    ("the 40 x 25 grid, scrambled", scrambled(grid(40, 25), 7, 3)),
    ("the 20 x 20 nine-point grid, scrambled", scrambled(grid9(20, 20), 37, 11)),
    ("the 8 x 8 x 8 grid, scrambled", scrambled(grid3(8), 5, 1)),
    ("the 12 x 12 x 12 grid", grid3(12)),
    ("a random graph, n = 500, m = 1000", random_graph(500, 1000, 1)),
    ("a random graph, n = 2000, m = 8000", random_graph(2000, 8000, 2)),
    (
        "a grid, a path, a random graph and isolated vertices, scrambled",
        scrambled(union(grid(12, 12), path(60), random_graph(100, 250, 3), (20, [])), 11, 5),
    ),
    ("the arrow of one hub, n = 300", arrow(300, 1)),
    ("the binary tree of 511 vertices", tree(511)),
    ("the 64 x 64 grid", grid(64, 64)),
    ("the 48 x 48 grid, scrambled", scrambled(grid(48, 48), 13, 7)),
    ("the arrow of three hubs, n = 400, scrambled", scrambled(arrow(400, 3), 3, 2)),
    ("the band of half-width 4, n = 500, scrambled", scrambled(band(500, 4), 3, 0)),
]


# The matrices of the column orderings, each as (rows, cols, the columns of every row).


def random_pattern(rows, cols, count, seed, diagonal):
    r = Rng(seed)
    lists = [set() for _ in range(rows)]
    for _ in range(count):
        i = r.below(rows)
        j = r.below(cols)
        lists[i].add(j)
    if diagonal:
        for i in range(min(rows, cols)):
            lists[i].add(i)
    return rows, cols, [sorted(s) for s in lists]


def upwind(k):
    """An unsymmetric stencil on the k x k grid: the cell, its left and lower neighbours, and the
    one up and to the right."""
    lists = []
    for y in range(k):
        for x in range(k):
            v = y * k + x
            row = {v}
            if x > 0:
                row.add(v - 1)
            if y > 0:
                row.add(v - k)
            if x + 1 < k and y + 1 < k:
                row.add(v + k + 1)
            lists.append(sorted(row))
    return k * k, k * k, lists


COLUMNS = [
    ("a random 300 x 300 pattern with its diagonal", random_pattern(300, 300, 1200, 11, True)),
    ("a random 1000 x 1000 pattern with its diagonal", random_pattern(1000, 1000, 4000, 12, True)),
    ("the upwind stencil of the 30 x 30 grid", upwind(30)),
    ("a random 600 x 300 pattern", random_pattern(600, 300, 1500, 13, False)),
    ("a random 200 x 500 pattern", random_pattern(200, 500, 1200, 14, False)),
    ("a random 800 x 400 pattern with its diagonal", random_pattern(800, 400, 2400, 15, True)),
]


def adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return [sorted(s) for s in adj]


def fill(adj, perm):
    """nnz(L), the diagonal counted, of the Cholesky factor of the graph renumbered by `perm`
    (perm[new] = old): Liu's elimination tree, then the row subtrees counted."""
    n = len(adj)
    inv = [0] * n
    for new, old in enumerate(perm):
        inv[old] = new
    parent = [-1] * n
    ancestor = [-1] * n
    lower = []
    for i in range(n):
        low = sorted(inv[w] for w in adj[perm[i]] if inv[w] < i)
        lower.append(low)
        for k in low:
            r = k
            while ancestor[r] != -1 and ancestor[r] != i:
                nxt = ancestor[r]
                ancestor[r] = i
                r = nxt
            if ancestor[r] == -1:
                ancestor[r] = i
                parent[r] = i
    mark = [-1] * n
    count = n
    for i in range(n):
        mark[i] = i
        for k in lower[i]:
            j = k
            while mark[j] != i:
                mark[j] = i
                count += 1
                j = parent[j]
    return count


def exact_minimum_degree(adj):
    """Minimum degree on the elimination graph itself, with true degrees, ties to the lowest
    index: the ordering the approximate methods approximate."""
    n = len(adj)
    nb = [set(a) for a in adj]
    alive = [True] * n
    heap = [(len(nb[v]), v) for v in range(n)]
    heapq.heapify(heap)
    order = []
    while heap:
        d, v = heapq.heappop(heap)
        if not alive[v] or d != len(nb[v]):
            continue
        alive[v] = False
        order.append(v)
        ns = nb[v]
        for u in ns:
            nb[u].discard(v)
        for u in ns:
            nb[u] |= ns
            nb[u].discard(u)
            heapq.heappush(heap, (len(nb[u]), u))
        nb[v] = set()
    return order


def band_profile(adj, perm):
    n = len(adj)
    inv = [0] * n
    for new, old in enumerate(perm):
        inv[old] = new
    width, profile = 0, 0
    for i in range(n):
        first = i
        for w in adj[perm[i]]:
            j = inv[w]
            width = max(width, abs(i - j))
            first = min(first, j)
        profile += i - first + 1
    return width, profile


def spd(adj):
    """A diagonally dominant matrix of the pattern, for SuperLU."""
    n = len(adj)
    rows, cols, vals = [], [], []
    for i, nb in enumerate(adj):
        rows.append(i)
        cols.append(i)
        vals.append(len(nb) + 1.0)
        for j in nb:
            rows.append(i)
            cols.append(j)
            vals.append(-1.0)
    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))


def superlu(a, spec):
    """The column permutation SuperLU chooses, as perm[new] = old."""
    options = dict(SymmetricMode=True) if spec == "MMD_AT_PLUS_A" else {}
    thresh = 0.0 if spec == "MMD_AT_PLUS_A" else 1.0
    lu = spla.splu(a, permc_spec=spec, diag_pivot_thresh=thresh, options=options)
    return [int(v) for v in np.argsort(lu.perm_c)], lu


def normal_adjacency(cols, lists):
    adj = [set() for _ in range(cols)]
    for row in lists:
        for a in row:
            for b in row:
                if a != b:
                    adj[a].add(b)
    return [sorted(s) for s in adj]


def main():
    print("// Made by data/gen_ordering.py: n, the edges, then scipy's reverse_cuthill_mckee (band,")
    print("// profile) and nnz(L) natural, after SuperLU's MMD_AT_PLUS_A and after a true minimum degree.")
    print("const SYMMETRIC: [[usize; 7]; %d] = [" % len(SYMMETRIC))
    for name, (n, edges) in SYMMETRIC:
        adj = adjacency(n, edges)
        m = sum(len(a) for a in adj) // 2
        a = spd(adj)
        rcm = [int(v) for v in csgraph.reverse_cuthill_mckee(a.tocsr(), symmetric_mode=True)]
        width, profile = band_profile(adj, rcm)
        natural = fill(adj, list(range(n)))
        mmd, lu = superlu(a, "MMD_AT_PLUS_A")
        f_mmd = fill(adj, mmd)
        assert f_mmd == lu.L.nnz, (name, f_mmd, lu.L.nnz)
        f_exact = fill(adj, exact_minimum_degree(adj))
        print("    // %s" % name)
        print("    [%d, %d, %d, %d, %d, %d, %d]," % (n, m, width, profile, natural, f_mmd, f_exact))
    print("];")
    print()
    print("// Made by data/gen_ordering.py: rows, columns, stored elements, then nnz(L) of the Cholesky")
    print("// factor of (AQ)ᵀ(AQ) for the natural Q, SuperLU's COLAMD (0: not square) and a true minimum")
    print("// degree ordering of AᵀA.")
    print("const COLUMNS: [[usize; 6]; %d] = [" % len(COLUMNS))
    for name, (rows, cols, lists) in COLUMNS:
        nnz = sum(len(r) for r in lists)
        adj = normal_adjacency(cols, lists)
        natural = fill(adj, list(range(cols)))
        f_colamd = 0
        if rows == cols:
            r, c, v = [], [], []
            for i, row in enumerate(lists):
                for j in row:
                    r.append(i)
                    c.append(j)
                    v.append(len(row) + 1.0 if i == j else -1.0)
            a = sp.csc_matrix((v, (r, c)), shape=(rows, cols))
            q, _ = superlu(a, "COLAMD")
            f_colamd = fill(adj, q)
        f_exact = fill(adj, exact_minimum_degree(adj))
        print("    // %s" % name)
        print("    [%d, %d, %d, %d, %d, %d]," % (rows, cols, nnz, natural, f_colamd, f_exact))
    print("];")


main()
