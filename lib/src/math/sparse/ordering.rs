//! The pattern of a sparse matrix, seen as a graph, and what is done to it.
//!
//! A square matrix is the adjacency matrix of a graph whose vertices are its rows and columns, and
//! everything expensive about factoring it — how much fill-in appears, how wide the bands are, what
//! a column of the factor can reach — is a fact about that graph rather than about the elements.
//! The permutations here relabel the vertices, which means replacing `A` by `P·A·Pᵀ` for a
//! permutation `P`: the same matrix in another order, with the same elements, the same determinant
//! and the same condition number, and often a factor a small fraction of the size.
//!
//! Nothing here reads an element: only which cells store one, and, for a square matrix, read
//! symmetrically, as the pattern of `A + Aᵀ`. A permutation is a vector `p` with `p[new] = old`
//! everywhere, as [`Lu::permutation`](crate::math::matrix::Lu::permutation) of the dense module.
//!
//! * [`reverse_cuthill_mckee`] narrows the band and shortens the profile, which
//!   [`band_width`] and [`profile`] measure.
//! * [`amd`](super::amd), [`colamd`](super::colamd) and
//!   [`nested_dissection`](super::nested_dissection) keep the fill of a factorisation low, and an
//!   [`Ordering`] names one of them for a direct method.
//! * [`elimination_tree`] says which column of a Cholesky factor depends on which, and
//!   [`postorder`] lays a tree out with every subtree in one piece.
//! * [`symmetric_permutation`] applies a permutation, [`inverse_permutation`] turns one round, and
//!   [`connected_components`] finds the independent blocks.

use super::SparseMatrix;
use super::graph::{Graph, Levels, NONE};
use crate::math::matrix::Scalar;

/// The order in which a direct method eliminates the rows and columns of a matrix, which decides
/// how much its factors fill in.
///
/// A symmetric factorisation (Cholesky, `L·D·Lᴴ`) applies it to rows and columns alike, as
/// `P·A·Pᵀ`, and takes its permutation from [`symmetric`](Self::symmetric). LU and QR apply it to
/// the columns, as `A·Q`, choose their rows themselves, and take it from
/// [`columns`](Self::columns); what bounds their fill is the Cholesky factor of `(A·Q)ᴴ·(A·Q)`,
/// so that is the matrix a column ordering is judged on.
///
/// | Variant | [`symmetric`](Self::symmetric), on the pattern of `A + Aᵀ` | [`columns`](Self::columns), for LU and QR |
/// | --- | --- | --- |
/// | `Natural` | the order the matrix is in | the order the columns are in |
/// | `ReverseCuthillMckee` | [`reverse_cuthill_mckee`] | the same, on the pattern of `Aᴴ·A` |
/// | `Amd` | [`amd`](super::amd) | the same, on the pattern of `Aᴴ·A` |
/// | `Colamd` | `colamd` of the matrix with one row for each edge, holding its two ends | [`colamd`](super::colamd) of `A` itself |
/// | `NestedDissection` | [`nested_dissection`](super::nested_dissection) | the same, on the pattern of `Aᴴ·A` |
/// | `Given(p)` | `p`, checked | `p`, checked |
///
/// The pattern of `Aᴴ·A` is formed from the pattern of `A` alone, so no cancellation can thin it;
/// it costs the sum of the squares of the row lengths, and one full row makes it complete.
/// `Colamd` never forms it and sets such rows aside, which is why it is the one to give LU and QR.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{Ordering, SparseMatrix};
///
/// assert_eq!(Ordering::default(), Ordering::Amd);
///
/// // An arrow pointing the wrong way: eliminating the hub first fills the whole matrix in, so
/// // the approximate minimum degree keeps it until one other row is left at most.
/// let a = SparseMatrix::from_fn(5, 5, |i, j| if i == j || i == 0 || j == 0 { 1.0 } else { 0.0 });
/// let p = Ordering::Amd.symmetric(&a);
/// assert!(p[3..].contains(&0));
/// assert_eq!(Ordering::Natural.columns(&a), [0, 1, 2, 3, 4]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Ordering {
    /// The order the matrix is in.
    Natural,
    /// [`reverse_cuthill_mckee`], which narrows the band.
    ReverseCuthillMckee,
    /// [`amd`](super::amd), the approximate minimum degree: the default.
    #[default]
    Amd,
    /// [`colamd`](super::colamd), the column approximate minimum degree, for LU and QR.
    Colamd,
    /// [`nested_dissection`](super::nested_dissection).
    NestedDissection,
    /// The permutation given, `p[new] = old`.
    Given(Vec<usize>),
}

impl Ordering {
    /// The permutation `p`, `p[new] = old`, this ordering gives the rows and columns of the square
    /// `a`, read as the pattern of `A + Aᵀ`: the one a factorisation of `P·A·Pᵀ` uses.
    ///
    /// `Colamd` has no symmetric form of its own. Here it is given the matrix `M` with one row for
    /// every edge of the graph and that edge's two ends in it, for which `Mᵀ·M` has the pattern of
    /// `A + Aᵀ`: the construction the paper of COLAMD calls SYMAMD.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square, or if a given permutation isn't one of `0..n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseMatrix};
    ///
    /// let a = SparseMatrix::tridiagonal(&[-1.0; 3], &[2.0; 4], &[-1.0; 3]);
    /// assert_eq!(Ordering::Natural.symmetric(&a), [0, 1, 2, 3]);
    /// assert_eq!(Ordering::Given(vec![3, 2, 1, 0]).symmetric(&a), [3, 2, 1, 0]);
    /// let mut p = Ordering::NestedDissection.symmetric(&a);
    /// p.sort_unstable();
    /// assert_eq!(p, [0, 1, 2, 3]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn symmetric<T: Scalar>(&self, a: &SparseMatrix<T>) -> Vec<usize> {
        assert!(a.rows() == a.cols(), "the matrix isn't square");
        match self {
            Ordering::Natural => (0..a.rows()).collect(),
            Ordering::ReverseCuthillMckee => reverse_cuthill_mckee(a),
            Ordering::Amd => super::amd(a),
            Ordering::Colamd => super::colamd::colamd_of_edges(&Graph::of_symmetric_pattern(a)),
            Ordering::NestedDissection => super::nested_dissection(a),
            Ordering::Given(p) => {
                check_permutation(p, a.rows());
                p.clone()
            }
        }
    }

    /// The permutation `q`, `q[new] = old`, this ordering gives the columns of `a`, of any shape,
    /// for a factorisation of `A·Q` that chooses its rows itself.
    ///
    /// `Colamd` works on `a` as it is. The symmetric orderings are applied to the pattern of
    /// `Aᴴ·A`, which is formed for them: that costs the sum of the squares of the row lengths, and
    /// a matrix with a full row has a complete one.
    ///
    /// # Panics
    ///
    /// Panics if a given permutation isn't one of `0..cols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseMatrix};
    ///
    /// // Three rows, four columns: a column ordering needs no square matrix.
    /// let a = SparseMatrix::from_coo(3, 4, &[(0, 0, 1.0), (0, 3, 1.0), (1, 1, 1.0), (2, 2, 1.0)]);
    /// for ordering in [Ordering::Colamd, Ordering::Amd, Ordering::ReverseCuthillMckee] {
    ///     let mut q = ordering.columns(&a);
    ///     q.sort_unstable();
    ///     assert_eq!(q, [0, 1, 2, 3]);
    /// }
    /// ```
    #[must_use]
    #[track_caller]
    pub fn columns<T: Scalar>(&self, a: &SparseMatrix<T>) -> Vec<usize> {
        match self {
            Ordering::Natural => (0..a.cols()).collect(),
            Ordering::Colamd => super::colamd(a),
            Ordering::Given(q) => {
                check_permutation(q, a.cols());
                q.clone()
            }
            Ordering::ReverseCuthillMckee => {
                reverse_cuthill_mckee_of(&Graph::of_normal_equations(a))
            }
            Ordering::Amd => super::amd::amd_of(&Graph::of_normal_equations(a)),
            Ordering::NestedDissection => {
                super::nd::nested_dissection_of(&Graph::of_normal_equations(a))
            }
        }
    }
}

/// Checks that `p` is a permutation of `0..n`.
///
/// # Panics
///
/// Panics if it isn't.
#[track_caller]
pub(crate) fn check_permutation(p: &[usize], n: usize) {
    assert_eq!(p.len(), n, "the permutation isn't of the right length");
    let mut seen = vec![false; n];
    for &old in p {
        assert!(old < n, "the permutation is out of range");
        assert!(!seen[old], "the permutation repeats an index");
        seen[old] = true;
    }
}

/// The inverse of the permutation `p`: where `p[new] = old`, the result has `inverse[old] = new`.
///
/// The orderings give `p[new] = old`, which is what gathers a permuted vector, `y[new] =
/// x[p[new]]`; the inverse is what scatters one back, and what relabels a column index.
///
/// # Panics
///
/// Panics if `p` isn't a permutation of `0..p.len()`.
///
/// # Examples
///
/// ```
/// use a::math::sparse::inverse_permutation;
///
/// let p = [2, 0, 3, 1];
/// let inverse = inverse_permutation(&p);
/// assert_eq!(inverse, [1, 3, 0, 2]);
/// assert_eq!(inverse_permutation(&inverse), p);
/// ```
#[must_use]
#[track_caller]
pub fn inverse_permutation(p: &[usize]) -> Vec<usize> {
    let n = p.len();
    let mut inverse = vec![NONE; n];
    for (new, &old) in p.iter().enumerate() {
        assert!(old < n, "the permutation is out of range");
        assert!(inverse[old] == NONE, "the permutation repeats an index");
        inverse[old] = new;
    }
    inverse
}

/// The independent blocks of the matrix, as the graph of its pattern: `labels[i]` is the block of
/// row `i`, numbered from 0 in the order the blocks are first met. Two rows share a label exactly
/// when a path of stored elements joins them, reading the pattern in either direction, so a
/// triangular matrix — whose pattern is connected one way but not the other — is one block.
///
/// A matrix of several blocks can be solved block by block, each with its own ordering and its own
/// iterations, which is what [`reverse_cuthill_mckee`] exploits. The cost is the stored elements.
///
/// # Panics
///
/// Panics if the matrix isn't square.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, connected_components};
///
/// // Two 2 × 2 blocks on the diagonal and nothing between them.
/// let a = SparseMatrix::from_coo(4, 4, &[
///     (0, 0, 2.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0),
///     (2, 2, 3.0), (2, 3, -1.0), (3, 2, -1.0), (3, 3, 3.0),
/// ]);
/// assert_eq!(connected_components(&a), vec![0, 0, 1, 1]);
/// ```
#[must_use]
#[track_caller]
pub fn connected_components<T: Scalar>(a: &SparseMatrix<T>) -> Vec<usize> {
    assert!(a.rows() == a.cols(), "the matrix isn't square");
    Graph::of_symmetric_pattern(a).components().0
}

/// The elimination tree of the pattern of `A + Aᵀ`: `parent[j]` is the row of the first element
/// below the diagonal in column `j` of the Cholesky factor, and `None` where the column has none,
/// which makes `j` a root. A matrix of several independent blocks has a forest, one tree to a
/// block.
///
/// Eliminating column `j` of a symmetric matrix changes exactly the columns of its ancestors, so
/// the tree is the dependency graph of the factorisation: a column is finished only after every
/// column of its subtree is, subtrees that share no vertex can be factored in any order or at once,
/// and the nonzeros of `L` can be counted before any arithmetic is done. A parent always has a
/// larger index than its children.
///
/// This is Liu's algorithm: the rows are taken in order, and from every element `(i, k)` with
/// `k < i` the path of ancestors of `k` is followed to its end, which becomes a child of `i`. The
/// paths are compressed as they are walked, which keeps the whole within a logarithmic factor of
/// the stored elements; the factor itself is never formed.
///
/// The pattern is read symmetrically, so a triangular matrix and its symmetric part have the same
/// tree, and the diagonal plays no part.
///
/// # Panics
///
/// Panics if the matrix isn't square.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, elimination_tree};
///
/// // The path 0—1—2—3, whose tree is a chain to the last vertex.
/// let a = SparseMatrix::tridiagonal(&[-1.0; 3], &[2.0; 4], &[-1.0; 3]);
/// assert_eq!(elimination_tree(&a), [Some(1), Some(2), Some(3), None]);
///
/// // An arrow: every column reaches the last one directly.
/// let arrow = SparseMatrix::from_fn(4, 4, |i, j| if i == j || i == 3 || j == 3 { 1.0 } else { 0.0 });
/// assert_eq!(elimination_tree(&arrow), [Some(3), Some(3), Some(3), None]);
/// ```
#[must_use]
#[track_caller]
pub fn elimination_tree<T: Scalar>(a: &SparseMatrix<T>) -> Vec<Option<usize>> {
    assert!(a.rows() == a.cols(), "the matrix isn't square");
    let graph = Graph::of_symmetric_pattern(a);
    let n = graph.len();
    let mut tree = Forest::new(n);
    for i in 0..n {
        for &k in graph.neighbours(i) {
            if k > i {
                break;
            }
            tree.join(k, i);
        }
    }
    tree.parent
}

/// An elimination forest being built by Liu's algorithm, its vertices taken in order.
struct Forest {
    /// The parent of every vertex, `None` while it is a root.
    parent: Vec<Option<usize>>,
    /// Where the walk from a vertex may jump to: an ancestor of it, the root of its tree so far
    /// once a walk has passed it.
    ancestor: Vec<usize>,
}

impl Forest {
    /// The forest of `n` vertices and no edge.
    fn new(n: usize) -> Self {
        Self {
            parent: vec![None; n],
            ancestor: vec![NONE; n],
        }
    }

    /// Records that the vertex `k`, numbered before `i`, reaches `i` in the factor: the root of
    /// the tree of `k` so far becomes a child of `i`, unless it is `i` already. The walk to that
    /// root points every vertex it passes straight at `i`, which keeps the walks short.
    fn join(&mut self, k: usize, i: usize) {
        debug_assert!(k < i);
        let mut r = k;
        while self.ancestor[r] != NONE && self.ancestor[r] != i {
            let next = self.ancestor[r];
            self.ancestor[r] = i;
            r = next;
        }
        if self.ancestor[r] == NONE {
            self.ancestor[r] = i;
            self.parent[r] = Some(i);
        }
    }

    /// The order `p` that the forest was built on, `p[new] = old`, renumbered by a postorder of
    /// the forest.
    fn postordered(&self, p: &[usize]) -> Vec<usize> {
        postorder(&self.parent).into_iter().map(|k| p[k]).collect()
    }
}

/// The symmetric order `p`, `p[new] = old`, renumbered by a postorder of the elimination tree of
/// the graph in that order.
///
/// Every order that keeps each vertex of the elimination tree after its descendants gives the
/// same fill (Liu, "The role of elimination trees in sparse factorization", 1990, on equivalent
/// orderings), and a postorder is the one of them that puts every subtree in one run of columns.
/// The tree is built by Liu's algorithm from the lists of the graph, without forming the factor.
pub(super) fn in_postorder(graph: &Graph, p: &[usize]) -> Vec<usize> {
    let n = graph.len();
    debug_assert_eq!(p.len(), n);
    let place = inverse_permutation(p);
    let mut tree = Forest::new(n);
    for (i, &v) in p.iter().enumerate() {
        for &w in graph.neighbours(v) {
            if place[w] < i {
                tree.join(place[w], i);
            }
        }
    }
    tree.postordered(p)
}

/// The column order `q`, `q[new] = old`, of a pattern of `rows` rows renumbered by a postorder of
/// its column elimination tree: the elimination tree of `(A·Q)ᵀ·(A·Q)`.
///
/// The pattern is given by its columns: `col_ptrs` says where each column's rows begin in
/// `row_indices`. A row joins all its columns to each other in `Aᵀ·A`, but for the tree it is
/// enough to join each of its columns to the one of the row met just before it, the columns taken
/// in their new order: the earlier ones are in that column's tree already. So `Aᵀ·A` is never
/// formed, and the cost is within a logarithmic factor of the stored elements of `A`, as that of
/// [`elimination_tree`] is of its own.
pub(super) fn columns_in_postorder(
    rows: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    q: &[usize],
) -> Vec<usize> {
    let mut tree = Forest::new(q.len());
    // The column met last in every row, in the new numbering.
    let mut last = vec![NONE; rows];
    for (k, &j) in q.iter().enumerate() {
        for &i in &row_indices[col_ptrs[j]..col_ptrs[j + 1]] {
            let before = core::mem::replace(&mut last[i], k);
            if before != NONE {
                tree.join(before, k);
            }
        }
    }
    tree.postordered(q)
}

/// A postorder of the forest `parent`, in which `parent[v]` is the parent of the vertex `v` and
/// `None` marks a root: the vertices in an order that puts every vertex after all of its
/// descendants and keeps every subtree in one piece. The result has `order[new] = old`, so it is a
/// permutation in the form the orderings give.
///
/// Children are taken in ascending order, and so are the roots. Renumbering a matrix by a
/// postorder of its [`elimination_tree`] leaves the fill of its Cholesky factor as it was and puts
/// the columns of every subtree next to each other, which is what the supernodal and multifrontal
/// methods want.
///
/// # Panics
///
/// Panics if a parent is out of range or if the parents close a cycle, so that `parent` isn't a
/// forest.
///
/// # Examples
///
/// ```
/// use a::math::sparse::postorder;
///
/// // 4 is the root, with the children 1 and 3; 0 hangs below 3, and 2 below 1.
/// let parent = [Some(3), Some(4), Some(1), Some(4), None];
/// assert_eq!(postorder(&parent), [2, 1, 0, 3, 4]);
/// ```
#[must_use]
#[track_caller]
pub fn postorder(parent: &[Option<usize>]) -> Vec<usize> {
    let n = parent.len();
    let mut first_child = vec![NONE; n];
    let mut next_sibling = vec![NONE; n];
    // Taken backwards, so that every list of children ascends.
    for v in (0..n).rev() {
        if let Some(p) = parent[v] {
            assert!(p < n, "a parent is out of range");
            next_sibling[v] = first_child[p];
            first_child[p] = v;
        }
    }
    let mut order = Vec::with_capacity(n);
    let mut stack = Vec::new();
    for (root, up) in parent.iter().enumerate() {
        if up.is_some() {
            continue;
        }
        stack.push(root);
        while let Some(&top) = stack.last() {
            let child = first_child[top];
            if child == NONE {
                order.push(top);
                stack.pop();
            } else {
                // The child is taken off the list as it is entered, so it is entered once.
                first_child[top] = next_sibling[child];
                stack.push(child);
            }
        }
    }
    // A vertex on a cycle hangs below no root, and is never reached.
    assert!(order.len() == n, "the parents close a cycle");
    order
}

/// The bandwidth of the matrix: the largest distance from the diagonal among its stored elements, so
/// that `a[(i, j)]` is 0 whenever `(i - j).abs()` is larger. A diagonal matrix has bandwidth 0, and
/// the tridiagonal one 1.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, band_width};
///
/// let a = SparseMatrix::from_coo(4, 4, &[(0, 0, 1.0), (0, 3, 1.0), (3, 0, 1.0)]);
/// assert_eq!(band_width(&a), 3);
/// ```
#[must_use]
pub fn band_width<T: Scalar>(a: &SparseMatrix<T>) -> usize {
    let (row_ptrs, cols) = (a.row_pointers(), a.col_indices());
    let mut width = 0;
    for i in 0..a.rows() {
        // The columns of a row ascend, so its two ends are its farthest elements.
        if row_ptrs[i] < row_ptrs[i + 1] {
            let (first, last) = (cols[row_ptrs[i]], cols[row_ptrs[i + 1] - 1]);
            width = width.max(i.abs_diff(first)).max(i.abs_diff(last));
        }
    }
    width
}

/// The profile, or envelope, of the square matrix, read as the pattern of `A + Aᵀ`: over the rows,
/// the distance from the first stored column of the row to the diagonal, the diagonal counted.
/// A row with nothing left of the diagonal counts 1 for itself; a matrix with no rows has profile 0.
///
/// A Cholesky factorisation fills in nothing outside the envelope, so the profile is the storage
/// of a skyline solver, and the work it does is of the order of the sum of the squares of the row
/// lengths: the profile says what a band solver will cost before it runs.
///
/// # Panics
///
/// Panics if the matrix isn't square.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, profile};
///
/// let a = SparseMatrix::tridiagonal(&[-1.0; 3], &[2.0; 4], &[-1.0; 3]);
/// assert_eq!(profile(&a), 7);
///
/// // The upper triangle is read as well: this is the profile of the full arrow.
/// let upper = SparseMatrix::from_coo(3, 3, &[(0, 2, 1.0)]);
/// assert_eq!(profile(&upper), 1 + 1 + 3);
/// ```
#[must_use]
#[track_caller]
pub fn profile<T: Scalar>(a: &SparseMatrix<T>) -> usize {
    assert!(a.rows() == a.cols(), "the matrix isn't square");
    let n = a.rows();
    let (row_ptrs, cols) = (a.row_pointers(), a.col_indices());
    // The first column of every row of the symmetrised pattern, no farther right than the diagonal.
    let mut first: Vec<usize> = (0..n).collect();
    for i in 0..n {
        for &j in &cols[row_ptrs[i]..row_ptrs[i + 1]] {
            if j < i {
                first[i] = first[i].min(j);
            } else {
                first[j] = first[j].min(i);
            }
        }
    }
    first.iter().enumerate().map(|(i, &f)| i - f + 1).sum()
}

/// The reverse Cuthill-McKee ordering: a permutation `p` with `p[new] = old` such that
/// [`symmetric_permutation`] puts the matrix into a narrow band and a short profile.
///
/// The matrix is read as the graph of the pattern of `A + Aᵀ`, and each connected component is
/// numbered by itself:
///
/// 1. A starting vertex far from everything is found by the iteration of George and Liu: from the
///    block's first vertex, the levels of a breadth-first search are built; a vertex of least
///    degree (and then of least index) in the last level becomes the new start, and the search
///    goes on while its level structure is deeper than the one before. It ends by itself, as a
///    rule after two to four searches, and is cut off after twelve.
/// 2. The Cuthill-McKee numbering takes the vertices in the order they were numbered and appends
///    the unnumbered neighbours of each in order of increasing degree, then of index.
/// 3. The whole order is reversed, which leaves the band as it is and never lengthens the
///    profile (George's observation, proved by Liu and Sherman, SIAM J. Numer. Anal. 13, 1976).
///
/// The cost is the stored elements times the logarithm of the largest degree, for the sorts. The
/// result is a heuristic: on a matrix already numbered well it can be slightly worse than the
/// order it was given. Its gain shows on one numbered badly — by a mesh generator, say.
///
/// # Panics
///
/// Panics if the matrix isn't square.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{
///     SparseMatrix, band_width, profile, reverse_cuthill_mckee, symmetric_permutation,
/// };
///
/// // A path of nine vertices, numbered so that neighbours are far apart.
/// let label = |k: usize| (k * 4) % 9;
/// let mut entries = Vec::new();
/// for k in 0..9 {
///     entries.push((label(k), label(k), 2.0));
///     if k + 1 < 9 {
///         entries.push((label(k), label(k + 1), -1.0));
///         entries.push((label(k + 1), label(k), -1.0));
///     }
/// }
/// let a = SparseMatrix::from_coo(9, 9, &entries);
/// assert_eq!(band_width(&a), 5);
///
/// let p = reverse_cuthill_mckee(&a);
/// let b = symmetric_permutation(&a, &p);
/// assert_eq!(band_width(&b), 1); // tridiagonal again
/// assert_eq!(profile(&b), 17);
/// ```
#[must_use]
#[track_caller]
pub fn reverse_cuthill_mckee<T: Scalar>(a: &SparseMatrix<T>) -> Vec<usize> {
    assert!(a.rows() == a.cols(), "the matrix isn't square");
    reverse_cuthill_mckee_of(&Graph::of_symmetric_pattern(a))
}

/// [`reverse_cuthill_mckee`] of a graph.
pub(super) fn reverse_cuthill_mckee_of(graph: &Graph) -> Vec<usize> {
    let n = graph.len();
    let mut order = Vec::with_capacity(n);
    let mut numbered = vec![false; n];
    let mut levels = Levels::new(n);
    for seed in 0..n {
        if numbered[seed] {
            continue;
        }
        let root = levels.far(graph, seed, &|_| true);
        // The order is its own queue: the vertices are visited in the order they were numbered.
        let mut next = order.len();
        numbered[root] = true;
        order.push(root);
        while next < order.len() {
            let v = order[next];
            next += 1;
            let from = order.len();
            for &w in graph.neighbours(v) {
                if !numbered[w] {
                    numbered[w] = true;
                    order.push(w);
                }
            }
            order[from..].sort_unstable_by_key(|&w| (graph.degree(w), w));
        }
    }
    order.reverse();
    order
}

/// `P·A·Pᵀ` for the permutation `p`, which is `p[new] = old`: the element `(i, j)` of the result is
/// the element `(p[i], p[j])` of `a`. The same permutation of rows and columns, which is what keeps
/// a symmetric matrix symmetric and a Cholesky factorisation of the permuted matrix a permutation of
/// the factorisation of the original.
///
/// The elements are moved by two counting passes, so the cost is the stored elements and nothing
/// is sorted.
///
/// # Panics
///
/// Panics if the matrix isn't square or `p` isn't a permutation of its rows.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, symmetric_permutation};
///
/// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 3.0)]);
/// let b = symmetric_permutation(&a, &[1, 0]);
/// assert_eq!(b[(0, 0)], 3.0); // the old (1, 1), which the permutation moved to the front
/// assert_eq!(b[(0, 1)], 2.0);
/// assert!(b.is_symmetric()); // the same permutation of rows and columns keeps it symmetric
/// ```
#[must_use]
#[track_caller]
pub fn symmetric_permutation<T: Scalar>(a: &SparseMatrix<T>, p: &[usize]) -> SparseMatrix<T> {
    assert!(a.rows() == a.cols(), "the matrix isn't square");
    let n = a.rows();
    assert_eq!(p.len(), n, "the permutation isn't of the right length");
    let inverse = inverse_permutation(p);
    let (row_ptrs, cols, values) = (a.row_pointers(), a.col_indices(), a.values());
    // First by new column, the new rows taken in order: every column then holds its rows
    // ascending.
    let mut col_ptrs = vec![0usize; n + 1];
    for &j in cols {
        col_ptrs[inverse[j] + 1] += 1;
    }
    for j in 0..n {
        col_ptrs[j + 1] += col_ptrs[j];
    }
    let mut cursor = col_ptrs.clone();
    let mut by_column: Vec<(usize, T)> = vec![(0, T::zero()); values.len()];
    let mut new_row_ptrs = Vec::with_capacity(n + 1);
    new_row_ptrs.push(0);
    for (new, &old) in p.iter().enumerate() {
        for k in row_ptrs[old]..row_ptrs[old + 1] {
            let at = &mut cursor[inverse[cols[k]]];
            by_column[*at] = (new, values[k]);
            *at += 1;
        }
        new_row_ptrs.push(new_row_ptrs[new] + row_ptrs[old + 1] - row_ptrs[old]);
    }
    // Then by new row, the new columns taken in order: every row holds its columns ascending.
    let mut cursor = new_row_ptrs.clone();
    let mut new_cols = vec![0usize; values.len()];
    let mut new_values = vec![T::zero(); values.len()];
    for j in 0..n {
        for &(i, x) in &by_column[col_ptrs[j]..col_ptrs[j + 1]] {
            let at = &mut cursor[i];
            new_cols[*at] = j;
            new_values[*at] = x;
            *at += 1;
        }
    }
    SparseMatrix::from_csr(n, n, new_row_ptrs, new_cols, new_values)
}
