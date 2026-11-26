//! The watch that the automatic strategy of [`SparseLu`](super::SparseLu) keeps over its
//! symmetric elimination, and the fill it predicts.
//!
//! The symmetric strategy orders the columns by [`amd`](crate::math::sparse::amd) of the pattern
//! of `A + Aᵀ`, and that order bounds the fill only while the pivots stay on the diagonal: with
//! every pivot there, the factors of `P·A·Pᵀ` have their elements within the pattern of the
//! Cholesky factor of `P·(A + Aᵀ)·Pᵀ` and of its transpose, column `j` of `L` in column `j` of
//! that factor and column `j` of `U` in its row `j`. A pivot off the diagonal breaks the bound,
//! and many of them — a diagonal that is weak, or becomes so as the elimination runs — can fill
//! the factors far beyond what partial pivoting on a column ordering would. So the elimination is
//! watched: after every step, the elements it has made are compared with what the first `j + 1`
//! columns of that Cholesky factor and its transpose hold, and the largest element of `U` with
//! the largest of `A`. When the one goes beyond [`FILL`] times the prediction and [`FILL_SLACK`]
//! times the elements of `A` more, or the other beyond [`GROWTH`] times, the elimination stops,
//! and the matrix is factored again from the start by the unsymmetric strategy (as it is when
//! the symmetric factors come out singular, which `SparseLu::by_strategy` decides).
//!
//! The prediction is made only once a pivot has left the diagonal — until then the bound holds
//! without a look — by the elimination tree of the pattern in the order, after Liu, and the
//! counts of the rows and columns of the factor by the algorithm of Gilbert, Ng and Peyton (*An
//! efficient algorithm to compute row and column counts for sparse Cholesky factorization*,
//! SIAM J. Matrix Anal. Appl. 15, 1994), in time nearly linear in the elements of `A`.

use super::NONE;
use crate::math::complex::Float;
use crate::math::sparse::graph::Graph;

/// How many times the elements that the symmetric analysis predicts the elimination may make
/// before it is stopped.
// COLAMD with partial pivoting fills 1.5 to 2.7 times the symmetric prediction on the 2-D and
// 3-D grids of the scoreboard. Measured on an Apple M4 Max (sweep 1.25 to 4): `SparseLu::new` on
// the weak-diagonal grids with convection of 50² to 200², whose pivots leave the diagonal, as fast
// at every value (within 0.4 %), and the 450 systems refuse master's 64 at every value with the
// same worst backward errors; 100 to 108 of them are refactored.
pub(super) const FILL: f64 = 2.0;

/// How many times the elements of `A` the elimination may make beyond [`FILL`] times the
/// prediction: a few steps off the diagonal in a small matrix are not worth a second
/// factorisation.
// Measured on an Apple M4 Max (sweep 0.25 to 4, the matrices of [`FILL`]): flat, within 0.4 %,
// with the same refusals and worst backward errors on the 450 systems.
pub(super) const FILL_SLACK: f64 = 1.0;

/// How many times the largest absolute value of an element of `A` an element of `U` may reach under
/// the symmetric strategy before it is stopped. A diagonal pivot down to a ten-thousandth of its
/// column makes multipliers of up to ten thousand, and growth that partial pivoting would not have
/// costs accuracy: the backward error of the solves and the test of singularity to working
/// precision scale with it, and shift-and-invert Lanczos next to an eigenvalue (the tests' 120 ×
/// 120 real matrix 10⁻⁶·‖A‖ from its 61st eigenvalue) converges nothing with a growth of 130, where
/// partial pivoting grows 1.1 and the same order with a threshold of 0.003 grows 15.
// The owner's choice, with `SYMMETRIC_DIAGONAL` at 10⁻⁴, from the quiet sweep (sweep 16 to 10⁴,
// the diagonal kept down to 10⁻³): the area's eigen and LU tests pass from 16 to 128 and fail from
// 256 (shift-and-invert of a complex Hermitian matrix next to an eigenvalue), and the shift-and-
// invert near the 61st eigenvalue above converges nothing from 256 on. On the 450 systems every
// value refuses master's 64; the nearly symmetric ones are refactored 133 times at 16, 124 at
// 32, 104 at 64, 84 at 128 and 22 at 10⁴, with the worst backward error per family 23, 103, 197,
// 197 and 7328 `ε·‖A‖∞·‖x‖∞` (master 14.5). `SparseLu::new` on the weak-diagonal grids, timed on
// an Apple M4 Max, is as fast from 16 to 64 (within 1.6 %) and up to 7 % slower at 128. Measured
// again with both of the owner's values: on the 450 systems master's 64 refused, 135 refactored,
// the worst backward error 23; the matrices of the scoreboard keep every pivot on the diagonal and
// a growth of at most 2.3, and their factors; shift-and-invert Arnoldi at 48 random interior shifts
// of grid operators finds the six nearest eigenvalues every time (47 at 64), Lanczos at 24 as
// master does.
pub(super) const GROWTH: f64 = 16.0;

/// The watch over one symmetric elimination.
pub(in crate::math::sparse) struct Watch<'g> {
    /// The pattern of `A + Aᵀ` without its diagonal, which the order was found for.
    graph: &'g Graph,
    /// The elements of `A`.
    elements: usize,
    /// For each step `j`, the elements of `L` and `U`, pivots included, that the first `j + 1`
    /// steps make with every pivot on the diagonal; empty until a pivot has left it.
    predicted: Vec<usize>,
}

impl<'g> Watch<'g> {
    /// The watch over the elimination of a matrix with `elements` elements, whose pattern
    /// `A + Aᵀ` is `graph`.
    pub(in crate::math::sparse) fn new(graph: &'g Graph, elements: usize) -> Self {
        Self {
            graph,
            elements,
            predicted: Vec::new(),
        }
    }

    /// Whether the elimination in the order `order` must stop after step `j`: it has made
    /// `made` elements of `L` and `U` in its steps so far, pivots included, `off_diagonal` says
    /// whether any of their pivots was off the diagonal, and `largest_u` and `largest_a` are the
    /// largest absolute values of an element of `U` and of `A`.
    pub(super) fn exceeded<R: Float>(
        &mut self,
        order: &[usize],
        j: usize,
        off_diagonal: bool,
        made: usize,
        largest_u: R,
        largest_a: R,
    ) -> bool {
        if largest_u > R::from_f64(GROWTH) * largest_a {
            return true;
        }
        if !off_diagonal && !cfg!(debug_assertions) {
            return false;
        }
        if self.predicted.is_empty() {
            self.predicted = predicted_fill(self.graph, order);
        }
        // With every pivot on the diagonal the elimination stays within the prediction.
        debug_assert!(
            off_diagonal || made <= self.predicted[j],
            "step {j}: {made} elements, {} predicted",
            self.predicted[j]
        );
        off_diagonal
            && made as f64 > FILL * self.predicted[j] as f64 + FILL_SLACK * self.elements as f64
    }
}

/// For each step `j`, the elements of `L` and `U`, pivots included, that the first `j + 1` steps
/// of the elimination in the symmetric order `order`, `order[new] = old`, make when every pivot
/// is on the diagonal, at most: the elements of the first `j + 1` columns and rows of the
/// Cholesky factor of the pattern `graph` in that order, the diagonal counted once.
pub(super) fn predicted_fill(graph: &Graph, order: &[usize]) -> Vec<usize> {
    let n = order.len();
    let mut place = vec![0usize; n];
    for (new, &old) in order.iter().enumerate() {
        place[old] = new;
    }
    // The vertices are the new indices. The neighbours of `j` that come after it are the rows
    // below the diagonal of its column, and those before it the columns of its row.
    let after = |j: usize| {
        graph
            .neighbours(order[j])
            .iter()
            .map(|&w| place[w])
            .filter(move |&i| i > j)
    };
    // The elimination tree, by Liu's algorithm with path compression: row `i` hangs the root of
    // every tree a nonzero of it lies in below itself.
    let mut parent = vec![NONE; n];
    let mut ancestor = vec![NONE; n];
    for (i, &v) in order.iter().enumerate() {
        for &w in graph.neighbours(v) {
            let mut k = place[w];
            if k >= i {
                continue;
            }
            loop {
                let above = ancestor[k];
                if above == i {
                    break;
                }
                ancestor[k] = i;
                if above == NONE {
                    parent[k] = i;
                    break;
                }
                k = above;
            }
        }
    }
    // A postorder of the tree, the size of every subtree, and the depth of every vertex.
    let mut head = vec![NONE; n];
    let mut next = vec![NONE; n];
    for v in (0..n).rev() {
        if parent[v] != NONE {
            next[v] = head[parent[v]];
            head[parent[v]] = v;
        }
    }
    let mut post = Vec::with_capacity(n);
    let mut stack = Vec::new();
    for root in (0..n).filter(|&v| parent[v] == NONE) {
        stack.push(root);
        while let Some(&top) = stack.last() {
            let child = head[top];
            if child == NONE {
                post.push(top);
                stack.pop();
            } else {
                head[top] = next[child];
                stack.push(child);
            }
        }
    }
    let mut size = vec![1usize; n];
    for &v in &post {
        if parent[v] != NONE {
            size[parent[v]] += size[v];
        }
    }
    let mut level = vec![0usize; n];
    for &v in post.iter().rev() {
        if parent[v] != NONE {
            level[v] = level[parent[v]] + 1;
        }
    }
    // The counts. Column `j` of the factor has an element in row `i` when `j` lies in the row
    // subtree of `i`, the union of the paths from the nonzeros of row `i` up to `i`. With the
    // vertices taken in postorder, `j` is a leaf of that subtree when no vertex of the subtree of
    // `j` has had a nonzero in row `i` yet; the paths of two consecutive leaves join at their
    // least common ancestor, the lowest vertex not yet finished above the earlier leaf, which a
    // forest of disjoint sets names. The row count adds the path from each leaf up to that
    // vertex (to `i` from the first leaf); the column counts are sums over the subtrees of
    // weights: `+1` at each leaf, `−1` at each such ancestor and at the parent of each vertex,
    // and `+1` at each leaf of the tree.
    let mut weight = vec![0isize; n];
    for v in 0..n {
        if size[v] == 1 {
            weight[v] += 1;
        }
        if parent[v] != NONE {
            weight[parent[v]] -= 1;
        }
    }
    let mut rows = vec![1usize; n];
    let mut latest = vec![0usize; n];
    let mut leaf = vec![NONE; n];
    let mut set: Vec<usize> = (0..n).collect();
    for (k, &j) in post.iter().enumerate() {
        let first_descendant = k + 1 - size[j];
        for i in after(j) {
            if latest[i] <= first_descendant {
                weight[j] += 1;
                let top = if leaf[i] == NONE {
                    i
                } else {
                    let q = find(&mut set, leaf[i]);
                    weight[q] -= 1;
                    q
                };
                rows[i] += level[j] - level[top];
                leaf[i] = j;
            }
            latest[i] = k + 1;
        }
        if parent[j] != NONE {
            set[j] = parent[j];
        }
    }
    for &v in &post {
        if parent[v] != NONE {
            weight[parent[v]] += weight[v];
        }
    }
    // Column `j` of `L` below its diagonal, column `j` of `U` above it, and the pivot.
    let mut predicted = Vec::with_capacity(n);
    let mut total = 0usize;
    for j in 0..n {
        debug_assert!(
            weight[j] >= 1,
            "a column of the factor has its diagonal at least"
        );
        total += weight[j] as usize + rows[j] - 1;
        predicted.push(total);
    }
    predicted
}

/// The vertex that names the set of `x`, with every vertex passed pointed nearer to it.
fn find(set: &mut [usize], mut x: usize) -> usize {
    while set[x] != x {
        set[x] = set[set[x]];
        x = set[x];
    }
    x
}
