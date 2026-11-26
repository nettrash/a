//! The orderings of issue #22 and the pattern utilities around them: reverse Cuthill-McKee, the
//! approximate minimum degree, the column approximate minimum degree, nested dissection, the
//! elimination tree, postorders and permutations.
//!
//! Everything here reads the pattern of a matrix and nothing else, so most tests build a graph,
//! make a matrix of it and check what comes back against a definition: that it is a permutation,
//! that the Cuthill-McKee order is one, that the separators of a dissection separate, that the
//! elimination tree is the tree of the factor built here column by column. The fill of the
//! orderings is compared with SciPy's and with an exact minimum degree ordering, whose numbers
//! `data/gen_ordering.py` made for the same graphs, built there in the same way.

use super::common::{Num, Rng, assert_csr, each_type, size};
use a::math::matrix::Scalar;
use a::math::sparse::{
    Ordering, SparseMatrix, amd, band_width, colamd, connected_components, elimination_tree,
    inverse_permutation, nested_dissection, postorder, profile, reverse_cuthill_mckee,
    symmetric_permutation,
};
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::time::{Duration, Instant};

// The graphs, built exactly as data/gen_ordering.py builds them.

/// An undirected graph: the number of vertices and the edges, which may repeat or be loops.
#[derive(Clone)]
struct Graph {
    n: usize,
    edges: Vec<(usize, usize)>,
}

/// The grid of `kx × ky` vertices of the five-point stencil, vertex `y·kx + x`.
fn grid(kx: usize, ky: usize) -> Graph {
    let mut edges = Vec::new();
    for y in 0..ky {
        for x in 0..kx {
            let v = y * kx + x;
            if x + 1 < kx {
                edges.push((v, v + 1));
            }
            if y + 1 < ky {
                edges.push((v, v + kx));
            }
        }
    }
    Graph { n: kx * ky, edges }
}

/// The grid of the nine-point stencil: the five-point one and the diagonals.
fn grid9(kx: usize, ky: usize) -> Graph {
    let mut g = grid(kx, ky);
    for y in 0..ky.saturating_sub(1) {
        for x in 0..kx {
            let v = y * kx + x;
            if x + 1 < kx {
                g.edges.push((v, v + kx + 1));
            }
            if x > 0 {
                g.edges.push((v, v + kx - 1));
            }
        }
    }
    g
}

/// The cube of `k³` vertices of the seven-point stencil.
fn grid3(k: usize) -> Graph {
    let mut edges = Vec::new();
    for z in 0..k {
        for y in 0..k {
            for x in 0..k {
                let v = (z * k + y) * k + x;
                if x + 1 < k {
                    edges.push((v, v + 1));
                }
                if y + 1 < k {
                    edges.push((v, v + k));
                }
                if z + 1 < k {
                    edges.push((v, v + k * k));
                }
            }
        }
    }
    Graph {
        n: k * k * k,
        edges,
    }
}

/// The cube of `k³` vertices of the 27-point stencil: every vertex joined to the 26 around it.
fn grid27(k: usize) -> Graph {
    let side = k as isize;
    let at = |x: isize, y: isize, z: isize| ((z * side + y) * side + x) as usize;
    let inside = |c: isize| (0..side).contains(&c);
    let mut edges = Vec::new();
    for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                // Each edge once: to the neighbours whose offset (dz, dy, dx) comes after zero.
                for dz in 0..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let (u, w, s) = (x + dx, y + dy, z + dz);
                            if (dz, dy, dx) > (0, 0, 0) && inside(u) && inside(w) && inside(s) {
                                edges.push((at(x, y, z), at(u, w, s)));
                            }
                        }
                    }
                }
            }
        }
    }
    Graph {
        n: k * k * k,
        edges,
    }
}

/// The graph with `u` unknowns at every vertex of `g`, each joined to every other unknown of its
/// vertex and of the vertices next to it.
fn nodes(g: &Graph, u: usize) -> Graph {
    let mut edges = Vec::new();
    for v in 0..g.n {
        for a in 0..u {
            for b in a + 1..u {
                edges.push((v * u + a, v * u + b));
            }
        }
    }
    for &(x, y) in &g.edges {
        for a in 0..u {
            for b in 0..u {
                edges.push((x * u + a, y * u + b));
            }
        }
    }
    Graph { n: g.n * u, edges }
}

/// The full band of half-width `w`.
fn band(n: usize, w: usize) -> Graph {
    let mut edges = Vec::new();
    for i in 0..n {
        for j in i + 1..n.min(i + w + 1) {
            edges.push((i, j));
        }
    }
    Graph { n, edges }
}

/// The arrow whose first `hubs` vertices are joined to every vertex.
fn arrow(n: usize, hubs: usize) -> Graph {
    let mut edges = Vec::new();
    for h in 0..hubs {
        for v in 0..n {
            if v != h {
                edges.push((h, v));
            }
        }
    }
    Graph { n, edges }
}

/// The binary tree in heap order: the parent of `v` is `(v − 1) / 2`.
fn tree(n: usize) -> Graph {
    Graph {
        n,
        edges: (1..n).map(|v| (v, (v - 1) / 2)).collect(),
    }
}

/// The path `0 — 1 — … — n − 1`.
fn path(n: usize) -> Graph {
    Graph {
        n,
        edges: (1..n).map(|v| (v - 1, v)).collect(),
    }
}

/// `n` vertices and no edge.
fn isolated(n: usize) -> Graph {
    Graph {
        n,
        edges: Vec::new(),
    }
}

/// `m` random pairs of vertices, loops and repeats included.
fn random_graph(n: usize, m: usize, seed: u64) -> Graph {
    let mut r = Rng(seed);
    let edges = (0..m)
        .map(|_| {
            let u = r.below(n);
            let v = r.below(n);
            (u, v)
        })
        .collect();
    Graph { n, edges }
}

/// `n` points uniform in the unit square (`dim` = 2) or cube (`dim` = 3), each joined to the
/// others nearer than the radius that gives about `degree` neighbours: an irregular mesh.
fn geometric(n: usize, dim: usize, degree: f64, seed: u64) -> Graph {
    let mut r = Rng(seed);
    let points: Vec<[f64; 3]> = (0..n)
        .map(|_| {
            let mut p = [0.0; 3];
            for x in p.iter_mut().take(dim) {
                *x = (r.next() >> 11) as f64 / (1u64 << 53) as f64;
            }
            p
        })
        .collect();
    let ball = if dim == 2 {
        core::f64::consts::PI
    } else {
        4.0 / 3.0 * core::f64::consts::PI
    };
    let radius = (degree / (n as f64 * ball)).powf(1.0 / dim as f64);
    // Buckets as wide as the radius: a point's neighbours are in its bucket and the next ones.
    let cells = ((1.0 / radius) as usize).max(1);
    let cell = |x: f64| ((x * cells as f64) as usize).min(cells - 1);
    let key = |p: &[f64; 3]| {
        (
            cell(p[0]),
            cell(p[1]),
            if dim == 3 { cell(p[2]) } else { 0 },
        )
    };
    let mut buckets = std::collections::BTreeMap::<(usize, usize, usize), Vec<usize>>::new();
    for (v, p) in points.iter().enumerate() {
        buckets.entry(key(p)).or_default().push(v);
    }
    let depth = if dim == 3 { 1 } else { 0 };
    let mut edges = Vec::new();
    for (v, p) in points.iter().enumerate() {
        let (x, y, z) = key(p);
        for dx in x.saturating_sub(1)..=(x + 1).min(cells - 1) {
            for dy in y.saturating_sub(1)..=(y + 1).min(cells - 1) {
                for dz in z.saturating_sub(depth)..=(z + depth).min(cells - 1) {
                    for &w in buckets.get(&(dx, dy, dz)).map_or(&[][..], Vec::as_slice) {
                        let d2: f64 = (0..dim).map(|k| (p[k] - points[w][k]).powi(2)).sum();
                        if w > v && d2 < radius * radius {
                            edges.push((v, w));
                        }
                    }
                }
            }
        }
    }
    Graph { n, edges }
}

/// The graphs side by side, the vertices of each after those of the ones before.
fn union(parts: &[Graph]) -> Graph {
    let mut g = isolated(0);
    for part in parts {
        let offset = g.n;
        g.edges
            .extend(part.edges.iter().map(|&(u, v)| (u + offset, v + offset)));
        g.n += part.n;
    }
    g
}

/// The graph relabelled by `v ↦ (v·mult + add) mod n`, a permutation when `mult` and `n` are
/// coprime: the same graph numbered badly.
fn scrambled(g: Graph, mult: usize, add: usize) -> Graph {
    let n = g.n;
    let label = |v: usize| (v * mult + add) % n;
    Graph {
        n,
        edges: g.edges.iter().map(|&(u, v)| (label(u), label(v))).collect(),
    }
}

/// The symmetric matrix of the graph: −1 for every edge, so that repeated edges add up rather
/// than cancel, and on the diagonal one more than the edges at the vertex, which makes it
/// positive definite.
fn matrix<T: Num>(g: &Graph) -> SparseMatrix<T> {
    let mut entries = Vec::with_capacity(g.n + 2 * g.edges.len());
    let mut diagonal = vec![1.0; g.n];
    for &(u, v) in &g.edges {
        if u != v {
            entries.push((u, v, T::real(-1.0)));
            entries.push((v, u, T::real(-1.0)));
            diagonal[u] += 1.0;
            diagonal[v] += 1.0;
        }
    }
    for (i, &d) in diagonal.iter().enumerate() {
        entries.push((i, i, T::real(d)));
    }
    SparseMatrix::from_coo(g.n, g.n, &entries)
}

/// The neighbours of every vertex, ascending, without loops or repeats.
fn adjacency(g: &Graph) -> Vec<Vec<usize>> {
    let mut sets = vec![BTreeSet::new(); g.n];
    for &(u, v) in &g.edges {
        if u != v {
            sets[u].insert(v);
            sets[v].insert(u);
        }
    }
    sets.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// The graphs of the table [`SYMMETRIC`], in its order.
fn symmetric_cases() -> Vec<(&'static str, Graph)> {
    vec![
        ("the 30 × 30 grid", grid(30, 30)),
        ("the band of half-width 9, n = 120", band(120, 9)),
        ("the 40 × 25 grid, scrambled", scrambled(grid(40, 25), 7, 3)),
        (
            "the 20 × 20 nine-point grid, scrambled",
            scrambled(grid9(20, 20), 37, 11),
        ),
        ("the 8 × 8 × 8 grid, scrambled", scrambled(grid3(8), 5, 1)),
        ("the 12 × 12 × 12 grid", grid3(12)),
        (
            "a random graph, n = 500, m = 1000",
            random_graph(500, 1000, 1),
        ),
        (
            "a random graph, n = 2000, m = 8000",
            random_graph(2000, 8000, 2),
        ),
        (
            "a grid, a path, a random graph and isolated vertices, scrambled",
            scrambled(
                union(&[
                    grid(12, 12),
                    path(60),
                    random_graph(100, 250, 3),
                    isolated(20),
                ]),
                11,
                5,
            ),
        ),
        ("the arrow of one hub, n = 300", arrow(300, 1)),
        ("the binary tree of 511 vertices", tree(511)),
        ("the 64 × 64 grid", grid(64, 64)),
        (
            "the 48 × 48 grid, scrambled",
            scrambled(grid(48, 48), 13, 7),
        ),
        (
            "the arrow of three hubs, n = 400, scrambled",
            scrambled(arrow(400, 3), 3, 2),
        ),
        (
            "the band of half-width 4, n = 500, scrambled",
            scrambled(band(500, 4), 3, 0),
        ),
    ]
}

// Made by data/gen_ordering.py: n, the edges, then scipy's reverse_cuthill_mckee (band,
// profile) and nnz(L) natural, after SuperLU's MMD_AT_PLUS_A and after a true minimum degree.
const SYMMETRIC: [[usize; 7]; 15] = [
    // the 30 x 30 grid
    [900, 1740, 30, 19315, 27029, 10098, 10351],
    // the band of half-width 9, n = 120
    [120, 1035, 9, 1155, 1155, 1158, 1155],
    // the 40 x 25 grid, scrambled
    [1000, 1935, 25, 21075, 79345, 11688, 11831],
    // the 20 x 20 nine-point grid, scrambled
    [400, 1482, 39, 10147, 17527, 6061, 6213],
    // the 8 x 8 x 8 grid, scrambled
    [512, 1344, 52, 19566, 38484, 12117, 12274],
    // the 12 x 12 x 12 grid
    [1728, 4752, 114, 143595, 231419, 74273, 79517],
    // a random graph, n = 500, m = 1000
    [500, 996, 194, 46460, 37729, 9446, 9467],
    // a random graph, n = 2000, m = 8000
    [2000, 7971, 1219, 1228526, 1157080, 404484, 412097],
    // a grid, a path, a random graph and isolated vertices, scrambled
    [324, 566, 55, 4073, 5159, 2108, 2090],
    // the arrow of one hub, n = 300
    [300, 299, 298, 599, 45150, 599, 599],
    // the binary tree of 511 vertices
    [511, 510, 128, 17818, 66046, 1021, 1021],
    // the 64 x 64 grid
    [4096, 8064, 64, 180832, 262207, 63266, 70587],
    // the 48 x 48 grid, scrambled
    [2304, 4512, 48, 77128, 309592, 34606, 35829],
    // the arrow of three hubs, n = 400, scrambled
    [400, 1194, 398, 1594, 79409, 1594, 1594],
    // the band of half-width 4, n = 500, scrambled
    [500, 1990, 4, 2490, 3811, 2490, 2490],
];

/// A pattern of `rows × cols`, as the columns of every row.
struct Pattern {
    rows: usize,
    cols: usize,
    lists: Vec<Vec<usize>>,
}

/// `count` random cells, repeats merged, and the diagonal if asked.
fn random_pattern(rows: usize, cols: usize, count: usize, seed: u64, diagonal: bool) -> Pattern {
    let mut r = Rng(seed);
    let mut sets = vec![BTreeSet::new(); rows];
    for _ in 0..count {
        let i = r.below(rows);
        let j = r.below(cols);
        sets[i].insert(j);
    }
    if diagonal {
        for (i, set) in sets.iter_mut().enumerate().take(rows.min(cols)) {
            set.insert(i);
        }
    }
    Pattern {
        rows,
        cols,
        lists: sets.into_iter().map(|s| s.into_iter().collect()).collect(),
    }
}

/// An unsymmetric stencil on the `k × k` grid: the cell, its left and lower neighbours, and the
/// one up and to the right.
fn upwind(k: usize) -> Pattern {
    let mut lists = Vec::new();
    for y in 0..k {
        for x in 0..k {
            let v = y * k + x;
            let mut row = BTreeSet::from([v]);
            if x > 0 {
                row.insert(v - 1);
            }
            if y > 0 {
                row.insert(v - k);
            }
            if x + 1 < k && y + 1 < k {
                row.insert(v + k + 1);
            }
            lists.push(row.into_iter().collect());
        }
    }
    Pattern {
        rows: k * k,
        cols: k * k,
        lists,
    }
}

/// The matrices of the table [`COLUMNS`], in its order.
fn column_cases() -> Vec<(&'static str, Pattern)> {
    vec![
        (
            "a random 300 × 300 pattern with its diagonal",
            random_pattern(300, 300, 1200, 11, true),
        ),
        (
            "a random 1000 × 1000 pattern with its diagonal",
            random_pattern(1000, 1000, 4000, 12, true),
        ),
        ("the upwind stencil of the 30 × 30 grid", upwind(30)),
        (
            "a random 600 × 300 pattern",
            random_pattern(600, 300, 1500, 13, false),
        ),
        (
            "a random 200 × 500 pattern",
            random_pattern(200, 500, 1200, 14, false),
        ),
        (
            "a random 800 × 400 pattern with its diagonal",
            random_pattern(800, 400, 2400, 15, true),
        ),
    ]
}

// Made by data/gen_ordering.py: rows, columns, stored elements, then nnz(L) of the Cholesky
// factor of (AQ)ᵀ(AQ) for the natural Q, SuperLU's COLAMD (0: not square) and a true minimum
// degree ordering of AᵀA.
const COLUMNS: [[usize; 6]; 6] = [
    // a random 300 x 300 pattern with its diagonal
    [300, 300, 1488, 37088, 20426, 20329],
    // a random 1000 x 1000 pattern with its diagonal
    [1000, 1000, 4989, 404885, 209665, 207607],
    // the upwind stencil of the 30 x 30 grid
    [900, 900, 3481, 52258, 29314, 30581],
    // a random 600 x 300 pattern
    [600, 300, 1491, 28125, 0, 12562],
    // a random 200 x 500 pattern
    [200, 500, 1197, 73777, 0, 21524],
    // a random 800 x 400 pattern with its diagonal
    [800, 400, 2786, 64906, 0, 37909],
];

/// The matrix of the pattern, with the value `x` in every cell.
fn pattern_matrix<T: Num>(p: &Pattern, x: T) -> SparseMatrix<T> {
    let mut row_ptrs = vec![0];
    let mut cols = Vec::new();
    for list in &p.lists {
        cols.extend_from_slice(list);
        row_ptrs.push(cols.len());
    }
    let values = vec![x; cols.len()];
    SparseMatrix::from_csr(p.rows, p.cols, row_ptrs, cols, values)
}

// What the tests check against.

/// Whether `p` holds every index of `0..n` once.
fn is_permutation(p: &[usize], n: usize) -> bool {
    let mut seen = vec![false; n];
    p.len() == n
        && p.iter()
            .all(|&v| v < n && !core::mem::replace(&mut seen[v], true))
}

/// A random permutation of `0..n`.
fn shuffled(r: &mut Rng, n: usize) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        p.swap(i, r.below(i + 1));
    }
    p
}

/// Every kind of ordering, the given one being `given`.
fn orderings(given: Vec<usize>) -> [Ordering; 6] {
    [
        Ordering::Natural,
        Ordering::ReverseCuthillMckee,
        Ordering::Amd,
        Ordering::Colamd,
        Ordering::NestedDissection,
        Ordering::Given(given),
    ]
}

/// The pattern of `A`, with 1 in every stored cell.
fn ones<T: Scalar>(a: &SparseMatrix<T>) -> SparseMatrix<f64> {
    SparseMatrix::from_csr(
        a.rows(),
        a.cols(),
        a.row_pointers().to_vec(),
        a.col_indices().to_vec(),
        vec![1.0; a.nnz()],
    )
}

/// The pattern of `AᵀA`, with positive elements so that nothing cancels: what a column ordering
/// is judged on.
fn normal_pattern<T: Scalar>(a: &SparseMatrix<T>) -> SparseMatrix<f64> {
    let ones = ones(a);
    &ones.transpose() * &ones
}

/// The Cholesky factor of the pattern of `P·(A + Aᵀ)·Pᵀ` from its definition: column `j` holds
/// the rows below `j` that the matrix stores in it, and the rows of the columns of the children
/// of `j` other than `j`; the parent of `j` is the first of them. Returns nnz(L), the diagonal
/// counted, and the parents: a reference that shares nothing with the library's elimination tree.
fn factor_columns<T: Scalar>(a: &SparseMatrix<T>, p: &[usize]) -> (usize, Vec<Option<usize>>) {
    let n = a.rows();
    let mut new = vec![0; n];
    for (k, &old) in p.iter().enumerate() {
        new[old] = k;
    }
    let mut below: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for &j in &a.col_indices()[a.row_pointers()[i]..a.row_pointers()[i + 1]] {
            let (x, y) = (new[i], new[j]);
            if x != y {
                below[x.min(y)].push(x.max(y));
            }
        }
    }
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut parent = vec![None; n];
    let mut columns: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut count = n;
    for j in 0..n {
        let mut column = core::mem::take(&mut below[j]);
        for &c in &children[j] {
            column.extend(columns[c].iter().copied().filter(|&i| i != j));
            columns[c] = Vec::new();
        }
        column.sort_unstable();
        column.dedup();
        count += column.len();
        if let Some(&first) = column.first() {
            parent[j] = Some(first);
            children[first].push(j);
        }
        columns[j] = column;
    }
    (count, parent)
}

/// nnz(L) of the Cholesky factor of `P·(A + Aᵀ)·Pᵀ`.
fn fill<T: Scalar>(a: &SparseMatrix<T>, p: &[usize]) -> usize {
    factor_columns(a, p).0
}

/// Checks that the numbering of the forest `parent`, in which every parent comes after its
/// children, is a postorder of it: that the subtree of every vertex is one run of indices, the
/// run of its size that ends at the vertex.
#[track_caller]
fn assert_postordered(parent: &[Option<usize>], what: &str) {
    let n = parent.len();
    let mut size = vec![1usize; n];
    for v in 0..n {
        if let Some(p) = parent[v] {
            assert!(p > v, "{what}: the parent of {v} comes before it");
            size[p] += size[v];
        }
    }
    // Every child's run lies inside its parent's: then, from the leaves up, the runs of the
    // children of a vertex fill the run of the vertex below it, and each subtree is its run.
    for v in 0..n {
        if let Some(p) = parent[v] {
            assert!(
                p + 1 - size[p] <= v + 1 - size[v],
                "{what}: the subtree of {p} is not one run, its child {v} reaches below it"
            );
        }
    }
}

/// `A·Q`: the columns of `a` in the order `q`, `q[new] = old`.
fn columns_permuted(a: &SparseMatrix<f64>, q: &[usize]) -> SparseMatrix<f64> {
    let place = inverse_permutation(q);
    let mut entries = Vec::with_capacity(a.nnz());
    for i in 0..a.rows() {
        entries.extend(a.row(i).map(|(j, x)| (i, place[j], x)));
    }
    SparseMatrix::from_coo(a.rows(), a.cols(), &entries)
}

/// The column elimination tree of `a`, the elimination tree of `AᵀA`, from the factor of the
/// formed `AᵀA` built by its definition.
fn column_elimination_tree(a: &SparseMatrix<f64>) -> Vec<Option<usize>> {
    let identity: Vec<usize> = (0..a.cols()).collect();
    factor_columns(&normal_pattern(a), &identity).1
}

/// nnz(L) of the Cholesky factor of `P·(A + Aᵀ)·Pᵀ` counted along the elimination tree: the
/// columns that row `i` reaches are the paths up the tree from its elements below the diagonal,
/// each walked until it meets one already counted for the row. Linear in nnz(L) and in need of
/// no more memory than the matrix, so it counts the factors of the large measurements; the
/// elimination tree it walks is checked against [`factor_columns`] by the other tests.
fn fill_along_the_tree(a: &SparseMatrix<f64>, p: &[usize]) -> usize {
    let b = symmetric_permutation(&(a + &a.transpose()), p);
    let parent = elimination_tree(&b);
    let n = b.rows();
    let mut mark = vec![usize::MAX; n];
    let mut count = n;
    for i in 0..n {
        mark[i] = i;
        for &k in &b.col_indices()[b.row_pointers()[i]..b.row_pointers()[i + 1]] {
            let mut j = k;
            while j < i && mark[j] != i {
                mark[j] = i;
                count += 1;
                j = parent[j].expect("an ancestor above every element below the diagonal");
            }
        }
    }
    count
}

/// Checks that `cm` is a Cuthill-McKee order: the first vertex of every block touches nothing
/// numbered before it, and the neighbours of each vertex in turn that are not numbered yet come
/// next, together, by increasing degree and then by index.
#[track_caller]
fn assert_cuthill_mckee(adj: &[Vec<usize>], cm: &[usize], name: &str) {
    let n = adj.len();
    let mut position = vec![0; n];
    for (k, &v) in cm.iter().enumerate() {
        position[v] = k;
    }
    let mut numbered = vec![false; n];
    let mut next = 0;
    for (k, &v) in cm.iter().enumerate() {
        if k == next {
            assert!(
                adj[v].iter().all(|&w| !numbered[w]),
                "{name}: the block of position {k} touches one numbered before"
            );
            numbered[v] = true;
            next += 1;
        }
        let mut fresh: Vec<usize> = adj[v].iter().copied().filter(|&w| !numbered[w]).collect();
        fresh.sort_unstable_by_key(|&w| (adj[w].len(), w));
        for (offset, &w) in fresh.iter().enumerate() {
            assert_eq!(
                position[w],
                next + offset,
                "{name}: a neighbour of the vertex at position {k} is out of its Cuthill-McKee place"
            );
            numbered[w] = true;
        }
        next += fresh.len();
    }
}

/// The eccentricity of `v`: the number of levels after the first of a breadth-first search.
fn eccentricity(adj: &[Vec<usize>], v: usize) -> usize {
    let mut depth = vec![usize::MAX; adj.len()];
    depth[v] = 0;
    let mut queue = vec![v];
    let mut head = 0;
    let mut deepest = 0;
    while head < queue.len() {
        let u = queue[head];
        head += 1;
        deepest = deepest.max(depth[u]);
        for &w in &adj[u] {
            if depth[w] == usize::MAX {
                depth[w] = depth[u] + 1;
                queue.push(w);
            }
        }
    }
    deepest
}

/// A cut found in an order: the positions `lo..hi` hold a part, a second part from `split`, and
/// the separator from `hi − separator`, with no edge between the two parts.
#[derive(Debug, Clone, Copy)]
struct Cut {
    lo: usize,
    hi: usize,
    split: usize,
    separator: usize,
}

/// The dissection an order holds, found from the order alone: a piece of the positions `lo..hi`
/// that has more than `leaf` of them is cut when its last `s` positions, at most a quarter of
/// the piece, leave the rest in two runs of positions that no edge joins, each at least an eighth
/// of the rest; of the cuts, one with the fewest `s` and then the most even parts is taken, and
/// the parts are searched in the same way. `adj` is the graph renumbered by the order.
fn cuts(adj: &[Vec<usize>], lo: usize, hi: usize, leaf: usize, found: &mut Vec<Cut>) {
    let m = hi - lo;
    if m <= leaf {
        return;
    }
    // For every split point, the first position at or after it that one before it reaches: the
    // run between the two touches nothing before the split point.
    let mut reached = BinaryHeap::new();
    let mut best: Option<(usize, Reverse<usize>, usize)> = None;
    for split in lo + 1..hi {
        reached.extend(
            adj[split - 1]
                .iter()
                .copied()
                .filter(|&j| j >= split && j < hi)
                .map(Reverse),
        );
        while reached.peek().is_some_and(|&Reverse(j)| j < split) {
            reached.pop();
        }
        let end = reached.peek().map_or(hi, |&Reverse(j)| j);
        let separator = hi - end;
        let (left, right) = (split - lo, end.saturating_sub(split));
        let rest = m - separator;
        if right == 0 || 4 * separator > m || 8 * left.min(right) < rest {
            continue;
        }
        let key = (separator, Reverse(left.abs_diff(right)), split);
        if best.is_none_or(|b| (key.0, key.1) < (b.0, b.1)) {
            best = Some(key);
        }
    }
    if let Some((separator, _, split)) = best {
        found.push(Cut {
            lo,
            hi,
            split,
            separator,
        });
        cuts(adj, lo, split, leaf, found);
        cuts(adj, split, hi - separator, leaf, found);
    }
}

/// The graph renumbered by `p`, `p[new] = old`.
fn renumbered(adj: &[Vec<usize>], p: &[usize]) -> Vec<Vec<usize>> {
    let inverse = inverse_permutation(p);
    p.iter()
        .map(|&old| {
            let mut list: Vec<usize> = adj[old].iter().map(|&w| inverse[w]).collect();
            list.sort_unstable();
            list
        })
        .collect()
}

/// Runs `f` and fails when it takes longer than `limit`: a guard on the complexity class, with
/// room for a slow, shared machine many times over and none for a quadratic method.
#[track_caller]
fn within<R>(limit: Duration, what: &str, f: impl FnOnce() -> R) -> R {
    let start = Instant::now();
    let out = f();
    let spent = start.elapsed();
    assert!(spent <= limit, "{what} took {spent:?}, more than {limit:?}");
    out
}

/// The arrow of `n` vertices whose hub is vertex 0, built straight in compressed form.
fn arrow_matrix(n: usize) -> SparseMatrix<f64> {
    let mut row_ptrs = vec![0];
    let mut cols: Vec<usize> = (0..n).collect();
    row_ptrs.push(n);
    for i in 1..n {
        cols.extend([0, i]);
        row_ptrs.push(cols.len());
    }
    let values = vec![1.0; cols.len()];
    SparseMatrix::from_csr(n, n, row_ptrs, cols, values)
}

/// A random unsymmetric pattern of `n × n` with about `count` distinct cells, each holding a
/// random element of `T` (distinct cells, so that nothing cancels).
fn random_square<T: Num>(r: &mut Rng, n: usize, count: usize) -> SparseMatrix<T> {
    let mut cells = BTreeSet::new();
    for _ in 0..count {
        cells.insert((r.below(n), r.below(n)));
    }
    let entries: Vec<(usize, usize, T)> = cells
        .into_iter()
        .map(|(i, j)| (i, j, T::random(r)))
        .collect();
    SparseMatrix::from_coo(n, n, &entries)
}

// Every ordering, on every kind of input.

#[test]
fn every_ordering_gives_a_permutation_of_random_patterns_of_every_size() {
    let mut r = Rng(0x2545_F491_4F6C_DD1D);
    for n in 0..size(34, 6) {
        for trial in 0..size(6, 2) {
            // Unsymmetric, with or without a diagonal, of every density up to nearly full.
            let count = r.below(n * n / 2 + 2 * n + 1);
            let entries: Vec<(usize, usize, f64)> =
                (0..count).map(|_| (r.below(n), r.below(n), 1.0)).collect();
            let a = SparseMatrix::from_coo(n, n, &entries);
            let case = format!("n = {n}, trial {trial}");
            for (what, p) in [
                ("reverse_cuthill_mckee", reverse_cuthill_mckee(&a)),
                ("amd", amd(&a)),
                ("colamd", colamd(&a)),
                ("nested_dissection", nested_dissection(&a)),
            ] {
                assert!(is_permutation(&p, n), "{what} of {case}: {p:?}");
            }
            let rows = r.below(2 * n + 2);
            let entries: Vec<(usize, usize, f64)> = (0..r.below(rows * n + 1))
                .map(|_| (r.below(rows), r.below(n), 1.0))
                .collect();
            let rectangular = SparseMatrix::from_coo(rows, n, &entries);
            for ordering in orderings(shuffled(&mut r, n)) {
                let p = ordering.symmetric(&a);
                assert!(is_permutation(&p, n), "{ordering:?} of {case}: {p:?}");
                let q = ordering.columns(&a);
                assert!(
                    is_permutation(&q, n),
                    "{ordering:?} columns of {case}: {q:?}"
                );
                let q = ordering.columns(&rectangular);
                assert!(
                    is_permutation(&q, n),
                    "{ordering:?} columns of the {rows} × {n} matrix of {case}: {q:?}"
                );
            }
        }
    }
}

/// The orderings of a random pattern of elements of `T` against those of the same pattern of
/// ones, and of the transpose and the symmetric part against those of the matrix.
fn pattern_alone<T: Num>() {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    for n in [0, 1, 2, 7, 40, 150, size(300, 60)] {
        let a = random_square::<T>(&mut r, n, 3 * n);
        let ones = ones(&a);
        let sum = &ones + &ones.transpose();
        let t = a.transpose();
        assert_eq!(reverse_cuthill_mckee(&a), reverse_cuthill_mckee(&ones));
        assert_eq!(amd(&a), amd(&ones));
        assert_eq!(colamd(&a), colamd(&ones));
        assert_eq!(nested_dissection(&a), nested_dissection(&ones));
        assert_eq!(elimination_tree(&a), elimination_tree(&ones));
        assert_eq!(connected_components(&a), connected_components(&ones));
        assert_eq!(band_width(&a), band_width(&ones));
        assert_eq!(profile(&a), profile(&ones));
        for ordering in orderings(shuffled(&mut r, n)) {
            assert_eq!(ordering.symmetric(&a), ordering.symmetric(&ones));
            assert_eq!(ordering.columns(&a), ordering.columns(&ones));
        }
        // Read as the pattern of A + Aᵀ, the matrix, its transpose and its symmetric part are
        // the same graph.
        assert_same_graph(&a, &t);
        assert_same_graph(&a, &ones.transpose());
        assert_same_graph(&a, &sum);
    }
}

/// Checks that the symmetric orderings and measures of `a` and `b` agree, as they must when the
/// two have the same pattern of `A + Aᵀ`.
#[track_caller]
fn assert_same_graph<T: Scalar, U: Scalar>(a: &SparseMatrix<T>, b: &SparseMatrix<U>) {
    assert_eq!(reverse_cuthill_mckee(a), reverse_cuthill_mckee(b));
    assert_eq!(amd(a), amd(b));
    assert_eq!(nested_dissection(a), nested_dissection(b));
    assert_eq!(elimination_tree(a), elimination_tree(b));
    assert_eq!(connected_components(a), connected_components(b));
    assert_eq!(profile(a), profile(b));
    assert_eq!(band_width(a), band_width(b));
    assert_eq!(Ordering::Colamd.symmetric(a), Ordering::Colamd.symmetric(b));
}

#[test]
fn the_orderings_read_the_pattern_and_nothing_else() {
    each_type!(pattern_alone);
}

/// The orderings of matrices with no rows, one row, and no stored element.
fn trivial<T: Num>() {
    let empty = SparseMatrix::<T>::zeros(0, 0);
    assert!(reverse_cuthill_mckee(&empty).is_empty());
    assert!(amd(&empty).is_empty());
    assert!(colamd(&empty).is_empty());
    assert!(nested_dissection(&empty).is_empty());
    assert!(elimination_tree(&empty).is_empty());
    assert!(postorder(&[]).is_empty());
    assert!(connected_components(&empty).is_empty());
    assert!(inverse_permutation(&[]).is_empty());
    assert_eq!((band_width(&empty), profile(&empty)), (0, 0));
    assert_eq!(symmetric_permutation(&empty, &[]), empty);
    for ordering in orderings(Vec::new()) {
        assert!(ordering.symmetric(&empty).is_empty());
        assert!(ordering.columns(&empty).is_empty());
    }
    for a in [
        SparseMatrix::<T>::zeros(1, 1),
        SparseMatrix::from_diagonal(&[T::real(2.0)]),
    ] {
        assert_eq!(reverse_cuthill_mckee(&a), [0]);
        assert_eq!(amd(&a), [0]);
        assert_eq!(colamd(&a), [0]);
        assert_eq!(nested_dissection(&a), [0]);
        assert_eq!(elimination_tree(&a), [None]);
        assert_eq!(connected_components(&a), [0]);
        assert_eq!((band_width(&a), profile(&a)), (0, 1));
        for ordering in orderings(vec![0]) {
            assert_eq!(ordering.symmetric(&a), [0]);
            assert_eq!(ordering.columns(&a), [0]);
        }
    }
    // A column ordering needs no rows, and orders no columns when there are none.
    let wide = SparseMatrix::<T>::zeros(0, 4);
    let tall = SparseMatrix::<T>::zeros(5, 0);
    for ordering in orderings(vec![3, 1, 0, 2]) {
        assert!(is_permutation(&ordering.columns(&wide), 4));
    }
    for ordering in orderings(Vec::new()) {
        assert!(ordering.columns(&tall).is_empty());
    }
    // A diagonal matrix is as many blocks as rows, each a root of the elimination forest.
    let n = 50;
    let diagonal = SparseMatrix::from_diagonal(&vec![T::real(3.0); n]);
    for p in [
        reverse_cuthill_mckee(&diagonal),
        amd(&diagonal),
        colamd(&diagonal),
        nested_dissection(&diagonal),
    ] {
        assert!(is_permutation(&p, n));
    }
    assert_eq!(elimination_tree(&diagonal), vec![None; n]);
    assert_eq!(connected_components(&diagonal), (0..n).collect::<Vec<_>>());
    assert_eq!(postorder(&vec![None; n]), (0..n).collect::<Vec<_>>());
}

#[test]
fn empty_one_by_one_and_diagonal_matrices_have_the_trivial_orderings() {
    each_type!(trivial);
}

#[test]
fn a_full_matrix_is_ordered_as_the_clique_it_is() {
    for n in [2, 17, size(40, 5), size(200, 6)] {
        let mut edges = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                edges.push((i, j));
            }
        }
        let a = matrix::<f64>(&Graph { n, edges });
        for ordering in orderings((0..n).rev().collect()) {
            let p = ordering.symmetric(&a);
            assert!(is_permutation(&p, n), "{ordering:?}");
            // Whatever the order, the factor of a clique is full.
            assert_eq!(fill(&a, &p), n * (n + 1) / 2, "{ordering:?}");
        }
    }
}

// Reverse Cuthill-McKee.

#[test]
fn reverse_cuthill_mckee_keeps_the_band_of_a_naturally_numbered_grid_and_of_a_full_band() {
    // p_direct-6: the old level sort made a band of 57 and a profile of 19721 of this grid, and
    // 16 and 1191 of the band; SciPy's reverse Cuthill-McKee makes 30 and 19315, and 9 and 1155.
    let a = matrix::<f64>(&grid(30, 30));
    let b = symmetric_permutation(&a, &reverse_cuthill_mckee(&a));
    assert_eq!((band_width(&a), band_width(&b)), (30, 30));
    assert_eq!(profile(&b), 19315);
    let a = matrix::<f64>(&band(120, 9));
    let b = symmetric_permutation(&a, &reverse_cuthill_mckee(&a));
    assert_eq!((band_width(&b), profile(&b)), (9, 1155));
}

#[test]
fn reverse_cuthill_mckee_is_a_cuthill_mckee_numbering_taken_backwards() {
    let mut r = Rng(0xD1B5_4A32_D192_ED03);
    let mut graphs = symmetric_cases();
    graphs.truncate(size(15, 2));
    for n in 0..size(60, 8) {
        graphs.push((
            "a small random graph",
            random_graph(n.max(1), 2 * n, r.next()),
        ));
    }
    for (name, g) in &graphs {
        let adj = adjacency(g);
        let mut cm = reverse_cuthill_mckee(&matrix::<f64>(g));
        assert!(is_permutation(&cm, g.n), "{name}");
        cm.reverse();
        assert_cuthill_mckee(&adj, &cm, name);
        // The search for a far vertex only ever moves to a deeper level structure, so each block
        // starts at a vertex at least as eccentric as the one the search began from.
        let labels = connected_components(&matrix::<f64>(g));
        let mut start_of = vec![usize::MAX; g.n];
        for &v in &cm {
            if start_of[labels[v]] == usize::MAX {
                start_of[labels[v]] = v;
            }
        }
        let mut first_of = vec![usize::MAX; g.n];
        for v in (0..g.n).rev() {
            first_of[labels[v]] = v;
        }
        for v in 0..g.n {
            if first_of[labels[v]] == v {
                let start = start_of[labels[v]];
                assert!(
                    eccentricity(&adj, start) >= eccentricity(&adj, v),
                    "{name}: block of vertex {v} starts at {start}"
                );
            }
        }
    }
    // On a path the start is an end, whatever the numbering.
    let g = scrambled(path(101), 7, 4);
    let mut cm = reverse_cuthill_mckee(&matrix::<f64>(&g));
    cm.reverse();
    assert_eq!(adjacency(&g)[cm[0]].len(), 1);
}

#[test]
fn reverse_cuthill_mckee_narrows_bands_and_profiles_as_much_as_scipys() {
    for ((name, g), reference) in symmetric_cases().iter().zip(SYMMETRIC).take(size(15, 2)) {
        let [n, edges, scipy_band, scipy_profile, ..] = reference;
        let a = matrix::<f64>(g);
        assert_eq!(
            (a.rows(), (a.nnz() - n) / 2),
            (n, edges),
            "{name}: not the graph of the script"
        );
        let b = symmetric_permutation(&a, &reverse_cuthill_mckee(&a));
        let (width, length) = (band_width(&b), profile(&b));
        eprintln!(
            "reverse_cuthill_mckee, {name}: band {width} against SciPy's {scipy_band}, profile \
             {length} against {scipy_profile} ({:.3})",
            length as f64 / scipy_profile as f64
        );
        assert!(
            width as f64 <= 1.1 * scipy_band as f64 + 1.0,
            "{name}: band {width}, SciPy's {scipy_band}"
        );
        assert!(
            length as f64 <= 1.1 * scipy_profile as f64,
            "{name}: profile {length}, SciPy's {scipy_profile}"
        );
    }
}

#[test]
fn reverse_cuthill_mckee_orders_a_disconnected_matrix_block_by_block() {
    let g = scrambled(
        union(&[grid(10, 10), path(30), tree(63), isolated(7), grid9(5, 5)]),
        17,
        3,
    );
    let a = matrix::<f64>(&g);
    let labels = connected_components(&a);
    let p = reverse_cuthill_mckee(&a);
    // Each block takes one run of the order.
    let mut finished = BTreeSet::new();
    for k in 1..p.len() {
        if labels[p[k]] != labels[p[k - 1]] {
            assert!(finished.insert(labels[p[k - 1]]), "a block is split");
        }
    }
    assert!(
        !finished.contains(&labels[p[p.len() - 1]]),
        "a block is split"
    );
}

#[test]
fn reverse_cuthill_mckee_and_the_elimination_tree_stay_near_linear_where_the_old_code_was_quadratic()
 {
    // p_direct-9: the old start-vertex search ran a breadth-first search from every leaf of an
    // arrow and allocated a marker per search, and the old elimination tree built the columns of
    // the factor: about n² on an arrow with its hub first, a full binary tree and a diagonal.
    let n = size(100_000, 300);
    let limit = Duration::from_secs(20);
    let a = arrow_matrix(n);
    let p = within(limit, "reverse_cuthill_mckee of an arrow", || {
        reverse_cuthill_mckee(&a)
    });
    assert!(is_permutation(&p, n));
    // The hub is next to everything, so the band is n − 1 whatever the order; the profile is
    // the least an arrow can have, with the hub at the end but for one leaf: 2n − 1.
    assert_eq!(profile(&symmetric_permutation(&a, &p)), 2 * n - 1);
    let parent = within(limit, "elimination_tree of an arrow", || {
        elimination_tree(&a)
    });
    // Eliminating the hub first joins every vertex to every other: the tree is a path.
    let chain: Vec<Option<usize>> = (1..n).map(Some).chain([None]).collect();
    assert_eq!(parent, chain);
    // A binary tree in heap order: its root first, which fills in a quarter of the factor.
    let g = tree((1 << size(17, 8)) - 1);
    let t = matrix::<f64>(&g);
    let p = within(limit, "reverse_cuthill_mckee of a binary tree", || {
        reverse_cuthill_mckee(&t)
    });
    assert!(is_permutation(&p, g.n));
    let parent = within(limit, "elimination_tree of a binary tree", || {
        elimination_tree(&t)
    });
    // One tree, since the graph is connected, rooted at the last vertex, every parent after its
    // children.
    assert_eq!(parent[g.n - 1], None);
    assert!((0..g.n - 1).all(|v| parent[v].is_some_and(|p| p > v)));
    // Leaves first, the elimination makes no fill and the tree is the graph itself.
    let leaves_first: Vec<usize> = (0..g.n).rev().collect();
    let parent = within(
        limit,
        "elimination_tree of a binary tree, leaves first",
        || elimination_tree(&symmetric_permutation(&t, &leaves_first)),
    );
    for v in 1..g.n {
        assert_eq!(parent[g.n - 1 - v], Some(g.n - 1 - (v - 1) / 2));
    }
    let d = SparseMatrix::from_diagonal(&vec![1.0f64; n]);
    let p = within(limit, "reverse_cuthill_mckee of a diagonal", || {
        reverse_cuthill_mckee(&d)
    });
    assert!(is_permutation(&p, n));
    let blocks = union(&vec![path(2); n / 2]);
    let b = matrix::<f64>(&blocks);
    let p = within(limit, "reverse_cuthill_mckee of 2 × 2 blocks", || {
        reverse_cuthill_mckee(&b)
    });
    assert!(is_permutation(&p, blocks.n));
}

// The elimination tree and postorders.

/// The elimination tree against the factor built from its definition.
fn tree_of_the_factor<T: Num>() {
    let mut r = Rng(0xBF58_476D_1CE4_E5B9);
    for n in 0..size(40, 8) {
        for _ in 0..size(5, 1) {
            let count = r.below(2 * n * n / 3 + 1) + n;
            let a = random_square::<T>(&mut r, n, count);
            // A triangle of it too: the tree reads the pattern either way.
            let lower = a.map(|i, j, x| if j <= i { x } else { T::zero() });
            for b in [&a, &lower] {
                let identity: Vec<usize> = (0..n).collect();
                let (_, parent) = factor_columns(b, &identity);
                assert_eq!(elimination_tree(b), parent, "n = {n}");
                for (j, &p) in parent.iter().enumerate() {
                    assert!(p.is_none_or(|p| p > j));
                }
            }
        }
    }
    for g in [
        grid(9, 7),
        scrambled(grid3(5), 3, 1),
        tree(40),
        arrow(30, 2),
    ] {
        let a = matrix::<T>(&g);
        let identity: Vec<usize> = (0..g.n).collect();
        assert_eq!(elimination_tree(&a), factor_columns(&a, &identity).1);
        let b = symmetric_permutation(&a, &amd(&a));
        assert_eq!(elimination_tree(&b), factor_columns(&b, &identity).1);
    }
}

#[test]
fn the_elimination_tree_is_where_each_column_of_the_factor_first_reaches_below_the_diagonal() {
    each_type!(tree_of_the_factor);
}

#[test]
fn the_elimination_tree_of_independent_blocks_is_a_forest_with_a_root_per_block() {
    let g = scrambled(
        union(&[grid(8, 8), path(20), tree(31), isolated(5), arrow(12, 2)]),
        13,
        1,
    );
    let a = matrix::<f64>(&g);
    let labels = connected_components(&a);
    let parent = elimination_tree(&a);
    // A connected block has one tree, whose root is the block's last vertex.
    let blocks = labels.iter().max().map_or(0, |&b| b + 1);
    let mut last = vec![0; blocks];
    for (v, &b) in labels.iter().enumerate() {
        last[b] = v;
        if let Some(p) = parent[v] {
            assert_eq!(labels[p], b, "a parent outside the block");
        }
    }
    let roots: Vec<usize> = (0..g.n).filter(|&v| parent[v].is_none()).collect();
    assert_eq!(
        roots,
        last.iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    );
    assert_eq!(roots.len(), 5 + 4);
}

#[test]
fn a_postorder_puts_each_subtree_in_one_piece_after_its_descendants() {
    let mut r = Rng(0x94D0_49BB_1331_11EB);
    for n in 0..size(80, 10) {
        // A random forest: every vertex hangs below one of higher rank, or is a root.
        let rank = shuffled(&mut r, n);
        let mut by_rank = vec![0; n];
        for (v, &k) in rank.iter().enumerate() {
            by_rank[k] = v;
        }
        let parent: Vec<Option<usize>> = (0..n)
            .map(|v| {
                let above = n - 1 - rank[v];
                (above > 0 && r.below(5) > 0).then(|| by_rank[rank[v] + 1 + r.below(above)])
            })
            .collect();
        let order = postorder(&parent);
        assert!(is_permutation(&order, n));
        let place = inverse_permutation(&order);
        // The size of every subtree, children before parents.
        let mut sizes = vec![1; n];
        for &v in &order {
            if let Some(p) = parent[v] {
                sizes[p] += sizes[v];
            }
        }
        for v in 0..n {
            // The subtree of v is the run of its size that ends at v.
            let begin = place[v] + 1 - sizes[v];
            for &w in &order[begin..=place[v]] {
                let mut up = w;
                while up != v {
                    up = parent[up].expect("a vertex of the run outside the subtree");
                }
            }
            if let Some(p) = parent[v] {
                assert!(place[p] > place[v]);
            }
        }
        // Siblings, and roots, in ascending order.
        for v in 0..n {
            for w in v + 1..n {
                if parent[v] == parent[w] {
                    assert!(place[v] < place[w]);
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "the parents close a cycle")]
fn postorder_refuses_parents_that_close_a_cycle() {
    let _ = postorder(&[Some(1), Some(2), Some(0), None]);
}

#[test]
#[should_panic(expected = "a parent is out of range")]
fn postorder_refuses_a_parent_outside_the_forest() {
    let _ = postorder(&[Some(5), None]);
}

#[test]
fn a_postorder_of_the_elimination_tree_is_an_equivalent_ordering() {
    let mut r = Rng(0x2127_599B_F432_5C37);
    let mut graphs = vec![
        grid(20, 20),
        scrambled(grid3(7), 5, 2),
        random_graph(300, 700, 17),
    ];
    for n in 1..size(30, 6) {
        graphs.push(random_graph(n, 2 * n, r.next()));
    }
    for g in &graphs {
        let a = matrix::<f64>(g);
        for p in [amd(&a), reverse_cuthill_mckee(&a), (0..g.n).collect()] {
            let b = symmetric_permutation(&a, &p);
            let post = postorder(&elimination_tree(&b));
            let composed: Vec<usize> = post.iter().map(|&k| p[k]).collect();
            assert_eq!(fill(&a, &composed), fill(&a, &p));
            // Renumbered so, the tree has every subtree as one run of columns.
            let c = symmetric_permutation(&a, &composed);
            assert_postordered(&elimination_tree(&c), "the renumbered matrix");
        }
    }
}

// Permutations.

#[test]
fn inverse_permutation_undoes_a_permutation() {
    let mut r = Rng(0x5851_F42D_4C95_7F2D);
    for n in 0..size(60, 8) {
        let p = shuffled(&mut r, n);
        let q = inverse_permutation(&p);
        for (new, &old) in p.iter().enumerate() {
            assert_eq!(q[old], new);
        }
        assert_eq!(inverse_permutation(&q), p);
    }
}

#[test]
#[should_panic(expected = "the permutation repeats an index")]
fn inverse_permutation_refuses_a_repeated_index() {
    let _ = inverse_permutation(&[0, 2, 0]);
}

#[test]
#[should_panic(expected = "the permutation is out of range")]
fn inverse_permutation_refuses_an_index_out_of_range() {
    let _ = inverse_permutation(&[0, 3, 1]);
}

/// `P·A·Pᵀ` against the definition, for random unsymmetric matrices of `T`.
fn permutes_rows_and_columns<T: Num>() {
    let mut r = Rng(0x6C62_272E_07BB_0142);
    for n in 0..size(25, 6) {
        let count = r.below(n * n + 1);
        let a = random_square::<T>(&mut r, n, count);
        let p = shuffled(&mut r, n);
        let b = symmetric_permutation(&a, &p);
        assert_csr(&b);
        assert_eq!(b.nnz(), a.nnz());
        for i in 0..n {
            for j in 0..n {
                assert_eq!(b[(i, j)], a[(p[i], p[j])]);
            }
        }
        assert_eq!(symmetric_permutation(&b, &inverse_permutation(&p)), a);
    }
}

#[test]
fn symmetric_permutation_puts_element_p_i_p_j_at_i_j() {
    each_type!(permutes_rows_and_columns);
}

#[test]
#[should_panic(expected = "the permutation repeats an index")]
fn symmetric_permutation_refuses_a_repeated_index() {
    let a = SparseMatrix::<f64>::identity(3);
    let _ = symmetric_permutation(&a, &[1, 1, 0]);
}

#[test]
#[should_panic(expected = "the permutation isn't of the right length")]
fn symmetric_permutation_refuses_a_permutation_of_the_wrong_length() {
    let a = SparseMatrix::<f64>::identity(3);
    let _ = symmetric_permutation(&a, &[1, 0]);
}

#[test]
#[should_panic(expected = "the matrix isn't square")]
fn symmetric_permutation_refuses_a_rectangular_matrix() {
    let a = SparseMatrix::<f64>::zeros(2, 3);
    let _ = symmetric_permutation(&a, &[1, 0]);
}

// The measures of a pattern.

#[test]
fn connected_components_numbers_the_blocks_in_the_order_they_are_met() {
    let g = scrambled(
        union(&[
            grid(6, 6),
            path(9),
            isolated(3),
            tree(15),
            random_graph(40, 30, 5),
        ]),
        7,
        2,
    );
    let n = g.n;
    let adj = adjacency(&g);
    let mut want = vec![usize::MAX; n];
    let mut count = 0;
    for seed in 0..n {
        if want[seed] != usize::MAX {
            continue;
        }
        let mut stack = vec![seed];
        want[seed] = count;
        while let Some(v) = stack.pop() {
            for &w in &adj[v] {
                if want[w] == usize::MAX {
                    want[w] = count;
                    stack.push(w);
                }
            }
        }
        count += 1;
    }
    assert_eq!(connected_components(&matrix::<f64>(&g)), want);
    // The pattern is read both ways: a chain stored below the diagonal only is one block.
    let chain = SparseMatrix::from_coo(5, 5, &[(1, 0, 1.0), (2, 1, 1.0), (4, 3, 1.0), (3, 2, 1.0)]);
    assert_eq!(connected_components(&chain), [0; 5]);
}

#[test]
fn band_width_and_profile_follow_their_definitions() {
    let mut r = Rng(0xA076_1D64_78BD_642F);
    for n in 0..size(30, 6) {
        let count = r.below(n * n / 3 + 1);
        let a = random_square::<f64>(&mut r, n, count);
        let stored: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| a.row(i).map(move |(j, _)| (i, j)))
            .collect();
        let width = stored
            .iter()
            .map(|&(i, j)| i.abs_diff(j))
            .max()
            .unwrap_or(0);
        assert_eq!(band_width(&a), width);
        // Row by row, from the first column of the symmetric pattern to the diagonal.
        let envelope: usize = (0..n)
            .map(|i| {
                let first = stored
                    .iter()
                    .filter(|&&(x, y)| (x == i && y <= i) || (y == i && x <= i))
                    .map(|&(x, y)| x.min(y))
                    .min()
                    .unwrap_or(i);
                i - first + 1
            })
            .sum();
        assert_eq!(profile(&a), envelope);
    }
    // An arrow stored in its upper triangle alone has the profile of the full arrow.
    let upper = SparseMatrix::from_coo(4, 4, &[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]);
    assert_eq!((band_width(&upper), profile(&upper)), (3, 1 + 2 + 3 + 4));
}

// The approximate minimum degree.

#[test]
fn amd_fills_in_about_as_little_as_superlus_minimum_degree_and_an_exact_one() {
    for ((name, g), reference) in symmetric_cases().iter().zip(SYMMETRIC).take(size(15, 2)) {
        let [n, _, _, _, natural, mmd, exact] = reference;
        let a = matrix::<f64>(g);
        let p = amd(&a);
        assert!(is_permutation(&p, n), "{name}");
        let ours = fill(&a, &p);
        let best = mmd.min(exact);
        eprintln!(
            "amd, {name}: nnz(L) {ours}, {:.3} of SuperLU's MMD {mmd}, {:.3} of an exact minimum \
             degree {exact}, {:.3} of the natural order",
            ours as f64 / mmd as f64,
            ours as f64 / exact as f64,
            ours as f64 / natural as f64
        );
        assert!(
            ours as f64 <= 1.1 * best as f64,
            "{name}: nnz(L) {ours}, SuperLU's MMD {mmd}, exact minimum degree {exact}"
        );
    }
}

#[test]
fn amd_cuts_the_fill_of_a_badly_numbered_grid_far_below_the_natural_order() {
    for g in [scrambled(grid(48, 48), 13, 7), scrambled(grid3(10), 7, 5)]
        .into_iter()
        .take(size(2, 0))
    {
        let a = matrix::<f64>(&g);
        let natural = fill(&a, &(0..g.n).collect::<Vec<_>>());
        let ours = fill(&a, &amd(&a));
        assert!(3 * ours < natural, "nnz(L) {ours}, natural {natural}");
    }
}

#[test]
fn amd_eliminates_arrows_and_trees_without_fill() {
    for g in [
        arrow(300, 1),
        scrambled(arrow(200, 1), 7, 3),
        tree(511),
        scrambled(tree(1000), 3, 1),
        path(77),
    ]
    .into_iter()
    .take(size(5, 2))
    {
        let a = matrix::<f64>(&g);
        // A graph without cycles has an elimination order without fill: the leaves first.
        assert_eq!(fill(&a, &amd(&a)), 2 * g.n - 1);
    }
}

#[test]
fn amd_orders_dense_rows_last() {
    // n = 400 sets aside the rows of more than 10·√400 = 200 elements: the three hubs, which
    // the scrambling put at 2, 5 and 8.
    let g = scrambled(arrow(400, 3), 3, 2);
    let a = matrix::<f64>(&g);
    let p = amd(&a);
    assert_eq!(p[397..], [2, 5, 8]);
    assert_eq!(fill(&a, &p), 1594);
    let p = nested_dissection(&a);
    assert_eq!(p[397..], [2, 5, 8]);
}

#[test]
fn amd_puts_out_a_postorder_of_its_elimination_tree() {
    // The pivots come out in a postorder of the assembly tree, which is not the elimination tree:
    // before the final postorder the elimination tree of P·A·Pᵀ had a subtree split by a vertex
    // of another branch on about one small random pattern in six, and on most larger graphs.
    let mut r = Rng(0x3C6E_F372_FE94_F82B);
    for n in 0..size(40, 8) {
        for trial in 0..size(25, 2) {
            let count = r.below(2 * n + 1) + r.below(n * n / 4 + 1);
            let a = random_square::<f64>(&mut r, n, count);
            for (what, p) in [
                ("amd", amd(&a)),
                ("Ordering::Colamd.symmetric", Ordering::Colamd.symmetric(&a)),
            ] {
                // The tree from the factor built by its definition, not from the library.
                let (_, parent) = factor_columns(&a, &p);
                assert_postordered(&parent, &format!("{what} of a {n} × {n} pattern, {trial}"));
            }
        }
    }
    // Grids, cubes, trees, random graphs, a disconnected union and rows set aside as dense.
    for (name, g) in symmetric_cases().into_iter().take(size(15, 2)) {
        let a = matrix::<f64>(&g);
        for (what, p) in [
            ("amd", amd(&a)),
            ("Ordering::Colamd.symmetric", Ordering::Colamd.symmetric(&a)),
        ] {
            let parent = elimination_tree(&symmetric_permutation(&a, &p));
            assert_postordered(&parent, &format!("{what} of {name}"));
        }
    }
}

#[test]
fn amd_takes_time_about_linear_in_the_stored_elements() {
    // Debug builds are slow, the machine may be shared: the limit is far above what a method of
    // the right class takes and far below what a quadratic one would.
    let k = size(300, 12);
    let g = scrambled(grid(k, k), 7, 11);
    let a = matrix::<f64>(&g);
    let limit = Duration::from_secs(60);
    let p = within(limit, "amd of a scrambled grid", || amd(&a));
    assert!(is_permutation(&p, g.n));
    let p = within(limit, "nested_dissection of a scrambled grid", || {
        nested_dissection(&a)
    });
    assert!(is_permutation(&p, g.n));
}

// The column approximate minimum degree.

#[test]
fn colamd_fills_in_about_as_little_as_superlus_colamd_and_amd_of_the_normal_matrix() {
    for ((name, pattern), reference) in column_cases().iter().zip(COLUMNS).take(size(6, 1)) {
        let [rows, cols, nnz, natural, superlu, exact] = reference;
        let a = pattern_matrix(pattern, 1.0f64);
        assert_eq!((a.rows(), a.cols(), a.nnz()), (rows, cols, nnz), "{name}");
        let normal = normal_pattern(&a);
        let q = colamd(&a);
        assert!(is_permutation(&q, cols), "{name}");
        let ours = fill(&normal, &q);
        let with_amd = fill(&normal, &amd(&normal));
        let mut best = exact.min(with_amd);
        if superlu > 0 {
            best = best.min(superlu);
        }
        eprintln!(
            "colamd, {name}: nnz(L) {ours}, {} of SuperLU's COLAMD {superlu}, {:.3} of amd of AᵀA \
             {with_amd}, {:.3} of an exact minimum degree {exact}, {:.3} of the natural order",
            if superlu > 0 {
                format!("{:.3}", ours as f64 / superlu as f64)
            } else {
                "none".to_string()
            },
            ours as f64 / with_amd as f64,
            ours as f64 / exact as f64,
            ours as f64 / natural as f64
        );
        assert!(
            ours as f64 <= 1.15 * best as f64,
            "{name}: nnz(L) {ours}, SuperLU's COLAMD {superlu}, amd of AᵀA {with_amd}, exact \
             minimum degree {exact}"
        );
        assert!(ours < natural, "{name}: nnz(L) {ours}, natural {natural}");
    }
}

/// A 400 × 300 pattern with something of every kind COLAMD sets aside. A column of more than
/// 10·√300 = 173 elements is dense, and so is a row of more than 173 elements in the columns that
/// are not: row 0 holds 249 columns, column 7 is in every row, column 11 in none, and column 23
/// in the dense row alone.
fn pattern_with_dense_parts() -> Pattern {
    let mut pattern = random_pattern(400, 300, 1600, 21, false);
    for (i, list) in pattern.lists.iter_mut().enumerate() {
        list.retain(|&j| j != 11 && j != 23);
        list.push(7);
        // Every other column has an element in a row that is not dense.
        if (1..=300).contains(&i) && i - 1 != 11 && i - 1 != 23 {
            list.push(i - 1);
        }
    }
    pattern.lists[0] = (0..250).filter(|&j| j != 11).collect();
    for list in pattern.lists.iter_mut() {
        list.sort_unstable();
        list.dedup();
    }
    pattern
}

#[test]
fn colamd_sets_dense_and_empty_columns_aside() {
    let pattern = pattern_with_dense_parts();
    let a = pattern_matrix(&pattern, 1.0f64);
    let q = colamd(&a);
    assert!(is_permutation(&q, 300));
    // Last of all the empty column 11, before it the dense column 7, and before that column 23,
    // which only the dense row holds.
    assert_eq!(q[297..], [23, 7, 11]);
    // Without the dense row, column 23 is empty too.
    let mut thin = pattern;
    thin.lists[0].clear();
    let b = pattern_matrix(&thin, 1.0f64);
    let r = colamd(&b);
    assert_eq!(r[297..], [7, 11, 23]);
    // The dense row takes no part in the elimination, so it orders the other columns as well as
    // if it were not there. It is part of the column elimination tree, though, which the final
    // postorder follows, so the two orders are not the same: they are orders of the same fill.
    let normal = normal_pattern(&b);
    assert_eq!(fill(&normal, &q), fill(&normal, &r));
}

#[test]
fn colamd_puts_out_a_postorder_of_the_column_elimination_tree() {
    // The pattern of the review: in the order of the assembly tree alone, q = [0, 5, 6, 1, 3, 4,
    // 2], the subtree of the column numbered 4 in the column elimination tree of A·Q held the
    // columns 0, 2, 3 and 4, and column 1, of another branch, lay inside it.
    let lists: [&[usize]; 7] = [&[], &[0, 3], &[5], &[4, 5], &[0], &[1, 6], &[1, 3, 4]];
    let review = Pattern {
        rows: 7,
        cols: 7,
        lists: lists.iter().map(|list| list.to_vec()).collect(),
    };
    let mut cases = vec![(
        "the pattern of the review".to_string(),
        pattern_matrix(&review, 1.0f64),
    )];
    // Small patterns of every shape, where about one in twenty had a subtree split before.
    let mut r = Rng(0x7FB5_D329_728E_A185);
    for trial in 0..size(2000, 40) {
        let cols = 1 + r.below(16);
        let rows = 1 + r.below(cols + 6);
        let entries: Vec<(usize, usize, f64)> = (0..r.below(3 * cols + 1))
            .map(|_| (r.below(rows), r.below(cols), 1.0))
            .collect();
        cases.push((
            format!("a random {rows} × {cols} pattern, {trial}"),
            SparseMatrix::from_coo(rows, cols, &entries),
        ));
    }
    // The matrices of the fill table, and dense, bare and empty columns with an ignored row.
    for (name, pattern) in column_cases().into_iter().take(size(6, 1)) {
        cases.push((name.to_string(), pattern_matrix(&pattern, 1.0f64)));
    }
    cases.push((
        "the pattern with dense parts".to_string(),
        pattern_matrix(&pattern_with_dense_parts(), 1.0f64),
    ));
    for (name, a) in &cases {
        for (what, q) in [
            ("colamd", colamd(a)),
            ("Ordering::Amd.columns", Ordering::Amd.columns(a)),
        ] {
            let tree = column_elimination_tree(&columns_permuted(a, &q));
            assert_postordered(&tree, &format!("{what} of {name}"));
        }
    }
}

#[test]
fn colamd_orders_the_columns_of_matrices_of_every_shape() {
    let mut r = Rng(0xE703_7ED1_A0B4_28DB);
    for (rows, cols) in [
        (1, 30),
        (30, 1),
        (7, 50),
        (50, 7),
        (120, 80),
        (80, 120),
        (3, 3),
    ] {
        for _ in 0..size(4, 1) {
            let entries: Vec<(usize, usize, f64)> = (0..r.below(3 * (rows + cols)) + 1)
                .map(|_| (r.below(rows), r.below(cols), 1.0))
                .collect();
            let a = SparseMatrix::from_coo(rows, cols, &entries);
            let q = colamd(&a);
            assert!(is_permutation(&q, cols), "{rows} × {cols}");
            assert_eq!(Ordering::Colamd.columns(&a), q);
        }
    }
}

// Nested dissection.

#[test]
fn nested_dissection_numbers_real_separators_after_the_parts_they_separate() {
    // The graph, how many separators its dissection must hold at least, and how large the first
    // may be: about a side of a square grid, a face of a cube.
    for (name, g, at_least, largest) in [
        ("the 40 × 40 grid", grid(40, 40), 7, 80),
        ("the 30 × 30 nine-point grid", grid9(30, 30), 3, 60),
        ("the 11 × 11 × 11 grid", grid3(11), 3, 2 * 121),
        (
            "the 50 × 30 grid, scrambled",
            scrambled(grid(50, 30), 7, 3),
            3,
            60,
        ),
        (
            "two grids side by side",
            scrambled(union(&[grid(30, 30), grid(20, 20)]), 11, 1),
            3,
            0,
        ),
        (
            "a random geometric graph of 3000 points in the cube",
            geometric(3000, 3, 16.0, 0x5DEE_CE66_D1CE_4E5B),
            3,
            400,
        ),
        (
            "a random geometric graph of 4000 points in the square",
            geometric(4000, 2, 12.0, 0x2F69_3A1B_77C2_09E5),
            3,
            200,
        ),
    ]
    .into_iter()
    .take(size(7, 1))
    {
        let a = matrix::<f64>(&g);
        let p = nested_dissection(&a);
        assert!(is_permutation(&p, g.n), "{name}");
        let adj = renumbered(&adjacency(&g), &p);
        let mut found = Vec::new();
        cuts(&adj, 0, g.n, 160, &mut found);
        // No edge joins the two runs before the separator of a cut.
        for cut in &found {
            for neighbours in &adj[cut.lo..cut.split] {
                assert!(
                    neighbours
                        .iter()
                        .all(|&j| j < cut.split || j >= cut.hi - cut.separator),
                    "{name}: an edge across {cut:?}"
                );
            }
        }
        let real = found.iter().filter(|c| c.separator > 0).count();
        assert!(
            real >= at_least,
            "{name}: only {real} separators in {found:?}"
        );
        // The first one splits the whole, with a separator of the order of the side of a grid.
        let top = found.first().expect("no cut at all");
        assert_eq!((top.lo, top.hi), (0, g.n), "{name}");
        assert!(top.separator <= largest, "{name}: {top:?}");
    }
}

#[test]
fn nested_dissection_fill_grows_like_n_log_n_on_grids_and_stays_close_to_amd() {
    let mut ratios = Vec::new();
    for k in [16, 32, 64, 128].into_iter().take(size(4, 1)) {
        let a = matrix::<f64>(&grid(k, k));
        let n = (k * k) as f64;
        let ours = fill(&a, &nested_dissection(&a)) as f64;
        let with_amd = fill(&a, &amd(&a)) as f64;
        eprintln!(
            "nested_dissection, the {k} × {k} grid: nnz(L) {ours}, {:.3}·n·log₂ n, {:.3} of amd's",
            ours / (n * n.log2()),
            ours / with_amd
        );
        assert!(
            ours <= 1.25 * with_amd,
            "k = {k}: nnz(L) {ours}, amd's {with_amd}"
        );
        ratios.push(ours / (n * n.log2()));
    }
    // nnz(L) / (n·log₂ n) settles to a constant: it grows by less and less, and stays small.
    for w in ratios.windows(2) {
        assert!(w[1] < w[0] + 0.25, "{ratios:?}");
    }
    assert!(ratios.iter().all(|&x| x < 2.0), "{ratios:?}");
}

/// The number of vertices on the longest path from a leaf of the forest to its root.
fn height(parent: &[Option<usize>]) -> usize {
    let mut depth = vec![1usize; parent.len()];
    for v in (0..parent.len()).rev() {
        if let Some(p) = parent[v] {
            depth[v] = depth[p] + 1;
        }
    }
    depth.into_iter().max().unwrap_or(0)
}

#[test]
fn nested_dissection_leaves_a_short_elimination_tree_where_minimum_degree_leaves_a_tall_one() {
    // A path from a leaf of the tree to its root passes through one separator of every level of
    // the dissection of the k × k grid, of about k, k/2, k/2, k/4, k/4, … vertices, and through a
    // leaf of at most 128. That is an estimate, not a bound: the heights measured are 51, 113,
    // 221 and 441 at k = 16 to 128 and 926 at k = 256, about 3.5·k, so the assertion below holds
    // for the sizes it runs. Minimum degree's tree is nearly twice as tall at k = 128 (787 against
    // 441).
    for k in [32, 64, 128].into_iter().take(size(3, 1)) {
        let a = matrix::<f64>(&grid(k, k));
        let ours = height(&elimination_tree(&symmetric_permutation(
            &a,
            &nested_dissection(&a),
        )));
        let with_amd = height(&elimination_tree(&symmetric_permutation(&a, &amd(&a))));
        eprintln!(
            "nested_dissection, the {k} × {k} grid: elimination tree {ours} high, amd's {with_amd}"
        );
        assert!(ours <= 3 * k + 128, "k = {k}: a tree {ours} high");
        if k == 128 {
            assert!(
                10 * ours < 7 * with_amd,
                "k = {k}: a tree {ours} high, amd's {with_amd}"
            );
        }
    }
}

#[test]
fn nested_dissection_cuts_irregular_meshes_by_their_small_separators() {
    // In three dimensions the dissection fills in clearly less than minimum degree; in the plane,
    // at these sizes, minimum degree is the better of the two, and the dissection stays near it.
    for (dim, n, degree, bound) in [(3, 4000, 16.0, 0.95), (2, 5000, 12.0, 1.3)]
        .into_iter()
        .take(size(2, 0))
    {
        let g = geometric(n, dim, degree, 0x9FB2_1C65_1E98_DF25);
        let a = matrix::<f64>(&g);
        let ours = fill(&a, &nested_dissection(&a));
        let with_amd = fill(&a, &amd(&a));
        eprintln!(
            "nested_dissection, a random geometric graph of {n} points in {dim} dimensions: \
             nnz(L) {ours}, {:.3} of amd's",
            ours as f64 / with_amd as f64
        );
        assert!(
            ours as f64 <= bound * with_amd as f64,
            "{dim} dimensions: nnz(L) {ours}, amd's {with_amd}"
        );
    }
}

#[test]
fn nested_dissection_leaves_small_and_separatorless_graphs_to_minimum_degree() {
    let mut r = Rng(0x8CB9_2BA7_2F3D_8DD7);
    for n in [1, 10, 60, 128] {
        let a = matrix::<f64>(&random_graph(n, 2 * n, r.next()));
        assert_eq!(nested_dissection(&a), amd(&a));
    }
    // A random graph has no small separators, and dissecting it fills in more than minimum
    // degree does.
    for (n, m) in [(1000, 1500), (3000, 6000), (10_000, 15_000)]
        .into_iter()
        .take(size(3, 0))
    {
        let a = matrix::<f64>(&random_graph(n, m, r.next()));
        let ours = fill(&a, &nested_dissection(&a));
        let with_amd = fill(&a, &amd(&a));
        eprintln!(
            "nested_dissection, a random graph of n = {n}, m = {m}: {:.3} of amd's nnz(L)",
            ours as f64 / with_amd as f64
        );
        assert!(
            ours as f64 <= 1.05 * with_amd as f64,
            "n = {n}, m = {m}: nnz(L) {ours}, amd's {with_amd}"
        );
    }
}

// The orderings as an Ordering names them.

#[test]
fn ordering_symmetric_gives_the_permutation_of_each_kind() {
    let g = scrambled(grid(20, 15), 7, 1);
    let a = matrix::<f64>(&g);
    let n = g.n;
    let given = shuffled(&mut Rng(77), n);
    assert_eq!(Ordering::default(), Ordering::Amd);
    assert_eq!(Ordering::Natural.symmetric(&a), (0..n).collect::<Vec<_>>());
    assert_eq!(
        Ordering::ReverseCuthillMckee.symmetric(&a),
        reverse_cuthill_mckee(&a)
    );
    assert_eq!(Ordering::Amd.symmetric(&a), amd(&a));
    assert_eq!(
        Ordering::NestedDissection.symmetric(&a),
        nested_dissection(&a)
    );
    assert_eq!(Ordering::Given(given.clone()).symmetric(&a), given);
    // Colamd has no symmetric form of its own: it orders the matrix of the edges, whose normal
    // matrix has the pattern of A, and comes out close to amd.
    let p = Ordering::Colamd.symmetric(&a);
    assert!(is_permutation(&p, n));
    assert!(fill(&a, &p) as f64 <= 1.1 * fill(&a, &amd(&a)) as f64);
    for ((name, g), reference) in symmetric_cases().iter().zip(SYMMETRIC).take(size(15, 2)) {
        let [_, _, _, _, _, mmd, exact] = reference;
        let a = matrix::<f64>(g);
        let ours = fill(&a, &Ordering::Colamd.symmetric(&a));
        eprintln!(
            "Ordering::Colamd.symmetric, {name}: nnz(L) {ours}, {:.3} of SuperLU's MMD {mmd}",
            ours as f64 / mmd as f64
        );
        assert!(
            ours as f64 <= 1.1 * mmd.min(exact) as f64,
            "{name}: nnz(L) {ours}, SuperLU's MMD {mmd}, exact {exact}"
        );
    }
}

#[test]
fn ordering_columns_orders_by_the_normal_matrix_or_by_colamd() {
    let pattern = random_pattern(120, 80, 400, 31, false);
    let a = pattern_matrix(&pattern, 2.0f64);
    let normal = normal_pattern(&a);
    let given = shuffled(&mut Rng(78), 80);
    assert_eq!(Ordering::Natural.columns(&a), (0..80).collect::<Vec<_>>());
    assert_eq!(Ordering::Colamd.columns(&a), colamd(&a));
    assert_eq!(Ordering::Amd.columns(&a), amd(&normal));
    assert_eq!(
        Ordering::ReverseCuthillMckee.columns(&a),
        reverse_cuthill_mckee(&normal)
    );
    assert_eq!(
        Ordering::NestedDissection.columns(&a),
        nested_dissection(&normal)
    );
    assert_eq!(Ordering::Given(given.clone()).columns(&a), given);
}

#[test]
#[should_panic(expected = "the permutation repeats an index")]
fn ordering_symmetric_refuses_a_given_permutation_with_a_repeat() {
    let a = SparseMatrix::<f64>::identity(3);
    let _ = Ordering::Given(vec![0, 0, 1]).symmetric(&a);
}

#[test]
#[should_panic(expected = "the permutation isn't of the right length")]
fn ordering_columns_refuses_a_given_permutation_of_the_wrong_length() {
    let a = SparseMatrix::<f64>::zeros(4, 3);
    let _ = Ordering::Given(vec![0, 1, 2, 3]).columns(&a);
}

#[test]
#[should_panic(expected = "the matrix isn't square")]
fn ordering_symmetric_refuses_a_rectangular_matrix() {
    let a = SparseMatrix::<f64>::zeros(4, 3);
    let _ = Ordering::Amd.symmetric(&a);
}

#[test]
#[should_panic(expected = "the matrix isn't square")]
fn the_symmetric_orderings_refuse_a_rectangular_matrix() {
    let a = SparseMatrix::<f64>::zeros(2, 5);
    let _ = nested_dissection(&a);
}

// Measurements.

#[test]
#[ignore = "a measurement of a minute or two in a release build: run it with `cargo test \
            --release --test math -- --ignored sparse::ordering --nocapture`"]
fn the_orderings_measured_on_grids_of_up_to_a_million_unknowns() {
    // The fill of nested dissection against amd on k × k grids, against n·log₂ n.
    for k in [16, 32, 64, 128, 256, 512] {
        let a = matrix::<f64>(&grid(k, k));
        let n = (k * k) as f64;
        let nd = fill_along_the_tree(&a, &nested_dissection(&a)) as f64;
        let with_amd = fill_along_the_tree(&a, &amd(&a)) as f64;
        eprintln!(
            "the {k} × {k} grid: nnz(L) by nested_dissection {nd} ({:.3}·n·log₂ n), by amd \
             {with_amd} ({:.3}·n·log₂ n), nested_dissection / amd {:.3}",
            nd / (n * n.log2()),
            with_amd / (n * n.log2()),
            nd / with_amd
        );
        // From some ten thousand unknowns on, the dissection is the better ordering, by more the
        // larger the grid: 0.846 of amd's fill at k = 256 and 0.762 at k = 512 when measured.
        if k >= 256 {
            assert!(
                nd < 0.9 * with_amd,
                "k = {k}: {nd} against amd's {with_amd}"
            );
        }
    }
    // Several unknowns at every node, the case supervariables are made for: the time of amd and
    // colamd per stored element should stay of the order of that of the grids below.
    for (name, g) in [
        (
            "the 120 × 120 grid with six unknowns at every node",
            nodes(&grid(120, 120), 6),
        ),
        ("the 40 × 40 × 40 grid of the 27-point stencil", grid27(40)),
    ] {
        let a = matrix::<f64>(&g);
        let nnz = a.nnz() as f64;
        let start = Instant::now();
        assert!(is_permutation(&amd(&a), g.n));
        let with_amd = start.elapsed().as_secs_f64();
        let start = Instant::now();
        assert!(is_permutation(&colamd(&a), g.n));
        let with_colamd = start.elapsed().as_secs_f64();
        eprintln!(
            "{name}, {} stored: amd {with_amd:.3} s ({:.0} ns per stored element), colamd \
             {with_colamd:.3} s ({:.0} ns)",
            a.nnz(),
            with_amd / nnz * 1e9,
            with_colamd / nnz * 1e9
        );
    }
    // The time of each ordering on scrambled grids from 10⁴ to 10⁶ unknowns, and per stored
    // element: what a method linear in the stored elements keeps about level.
    for k in [100, 200, 400, 1000] {
        let g = scrambled(grid(k, k), 7, 11);
        let a = matrix::<f64>(&g);
        let nnz = a.nnz() as f64;
        let mut line = format!("the scrambled {k} × {k} grid, {} stored:", a.nnz());
        let mut time = |what: &str, f: &dyn Fn() -> usize| {
            let start = Instant::now();
            let size = f();
            let spent = start.elapsed().as_secs_f64();
            assert!(size > 0);
            line += &format!(
                " {what} {:.3} s ({:.0} ns per stored element);",
                spent,
                spent / nnz * 1e9
            );
        };
        time("reverse_cuthill_mckee", &|| reverse_cuthill_mckee(&a).len());
        time("amd", &|| amd(&a).len());
        time("Ordering::Colamd.symmetric", &|| {
            Ordering::Colamd.symmetric(&a).len()
        });
        time("colamd", &|| colamd(&a).len());
        time("nested_dissection", &|| nested_dissection(&a).len());
        time("elimination_tree", &|| elimination_tree(&a).len());
        eprintln!("{line}");
    }
}
