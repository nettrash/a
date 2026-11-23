//! The column approximate minimum degree ordering.
//!
//! An LU or QR factorisation of `A·Q` that chooses its rows as it goes cannot know its fill
//! beforehand, but whatever rows it chooses its factors fit inside the Cholesky factor of
//! `(A·Q)ᵀ·(A·Q)`. So the columns are ordered to keep *that* factor sparse, which is a symmetric
//! ordering of `Aᵀ·A` — a matrix far denser than `A` and not worth forming.
//!
//! It need not be formed. Every row of `A` makes a clique of the columns it has an element in, and
//! a clique of variables is what an element of a quotient graph stands for: `A` itself, its rows
//! read as elements and its columns as variables, *is* a quotient graph of `Aᵀ·A`. Minimum degree
//! starts from it and runs as it does for a symmetric matrix.
//!
//! This is the algorithm of Davis, Gilbert, Larimore and Ng, "A column approximate minimum degree
//! ordering algorithm", ACM Trans. Math. Softw. 30 (2004). The elimination is the one of
//! [`amd`](super::amd), in the same code: columns that become indistinguishable are merged into
//! supercolumns, a row whose columns all lie in the pivot row is absorbed into it, and the degree
//! of a column is bounded by the sum of the set differences of its rows with the pivot row. The
//! order is then renumbered by a postorder of the column elimination tree, which keeps the fill
//! and puts every subtree of that tree in one run of columns.

use super::SparseMatrix;
use super::amd::QuotientGraph;
use super::graph::{Graph, dense_threshold, transposed_lists};
use super::ordering::columns_in_postorder;
use crate::math::matrix::Scalar;

/// The column approximate minimum degree ordering of a matrix of any shape: a permutation `q` of
/// its columns with `q[new] = old` that keeps the fill of the Cholesky factor of `(A·Q)ᴴ·(A·Q)`
/// low, and with it the fill of an LU or QR factorisation of `A·Q`, without forming `Aᴴ·A`.
///
/// The method is that of Davis, Gilbert, Larimore and Ng (2004): minimum degree on `A` itself,
/// whose rows are the elements of a quotient graph of `Aᴴ·A` and whose columns are its variables,
/// with supercolumns, aggressive absorption of rows and approximate degrees from set differences.
/// What would make `Aᴴ·A` dense is set aside first, with the thresholds of the paper, and takes no
/// part in the elimination:
///
/// * a column with more than `10·√min(rows, cols)` elements (and more than 16) is ordered after
///   all the others, before the final postorder places it after what it shares rows with;
/// * a row with more than `10·√cols` elements (and more than 16) in the columns that are left is
///   ignored;
/// * a column with no element, or with elements in ignored rows only, has nothing to order it by,
///   and is put last of all or just before the dense columns.
///
/// The pivots come out in a postorder of the assembly tree, the tree of the rows and of the pivot
/// rows that absorbed them. That is not the column elimination tree of `A·Q`, the elimination tree
/// of `(A·Q)ᴴ·(A·Q)`, so last the whole order is renumbered by a postorder of that tree, found from
/// the rows of `A` (the ignored ones too) without forming `Aᴴ·A`. The postorder leaves the fill as
/// it is and puts the columns of every subtree of the column elimination tree next to each other,
/// as supernodal and multifrontal LU and QR want. It keeps every column after all the columns it
/// shares a row with that came before it, so a column set aside still follows everything it
/// touches, and the empty columns stay last.
///
/// Only the pattern is read. The working memory is a small multiple of the stored elements.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, colamd};
///
/// // Column 0 is full; every other column holds its diagonal element alone.
/// let n = 6;
/// let a = SparseMatrix::from_fn(n, n, |i, j| if i == j || j == 0 { 1.0 } else { 0.0 });
/// // The nonzeros of the Cholesky factor of (A·Q)ᵀ·(A·Q), the columns taken in the order q.
/// let fill = |q: &[usize]| {
///     let aq = SparseMatrix::from_fn(n, n, |i, j| a[(i, q[j])]);
///     let l = (&aq.transpose() * &aq).to_matrix().cholesky().unwrap().l();
///     l.as_slice().iter().filter(|&&x| x != 0.0).count()
/// };
/// // In the natural order Aᵀ·A has a full first row and column, and its factor fills in.
/// assert_eq!(fill(&[0, 1, 2, 3, 4, 5]), n * (n + 1) / 2);
/// // colamd takes the full column late, where it fills nothing in.
/// assert_eq!(fill(&colamd(&a)), 2 * n - 1);
///
/// // Any shape will do, and every column is in the result once.
/// let wide = SparseMatrix::from_coo(2, 5, &[(0, 0, 1.0), (0, 4, 1.0), (1, 2, 1.0)]);
/// let mut q = colamd(&wide);
/// q.sort_unstable();
/// assert_eq!(q, [0, 1, 2, 3, 4]);
/// ```
#[must_use]
pub fn colamd<T: Scalar>(a: &SparseMatrix<T>) -> Vec<usize> {
    colamd_of(a.cols(), a.row_pointers(), a.col_indices())
}

/// [`colamd`] of the pattern with one row for each edge of the graph, holding the two ends of the
/// edge: for that pattern `M`, `Mᵀ·M` has the pattern of the graph, so this is a symmetric
/// ordering of the graph computed by the column method (the SYMAMD of the paper).
pub(super) fn colamd_of_edges(graph: &Graph) -> Vec<usize> {
    let n = graph.len();
    let mut ends = Vec::with_capacity(graph.entries());
    for v in 0..n {
        for &w in graph.neighbours(v) {
            if w > v {
                ends.push(v);
                ends.push(w);
            }
        }
    }
    let row_ptrs: Vec<usize> = (0..=ends.len() / 2).map(|row| 2 * row).collect();
    colamd_of(n, &row_ptrs, &ends)
}

/// What becomes of a column before the elimination starts.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Column {
    /// Ordered by the elimination.
    Ordered,
    /// It has elements, but only in rows that are ignored.
    Bare,
    /// Too many elements: set aside.
    Dense,
    /// No element at all.
    Empty,
}

/// [`colamd`] of a pattern held by rows: `row_ptrs` gives where each row begins in `col_indices`.
pub(super) fn colamd_of(cols: usize, row_ptrs: &[usize], col_indices: &[usize]) -> Vec<usize> {
    let rows = row_ptrs.len() - 1;
    let (col_ptrs, row_indices) = transposed_lists(cols, row_ptrs, col_indices);
    let count = |j: usize| col_ptrs[j + 1] - col_ptrs[j];

    // The columns that are too full, and the ones that are empty.
    let dense_column = dense_threshold(rows.min(cols));
    let mut column: Vec<Column> = (0..cols)
        .map(|j| match count(j) {
            0 => Column::Empty,
            c if c > dense_column => Column::Dense,
            _ => Column::Ordered,
        })
        .collect();
    // The rows, by what they hold in the columns that are left: a row that is too full is
    // ignored, and a row with nothing left has nothing to say.
    let dense_row = dense_threshold(cols);
    let row_length: Vec<usize> = (0..rows)
        .map(|i| {
            let row = &col_indices[row_ptrs[i]..row_ptrs[i + 1]];
            let kept = row
                .iter()
                .filter(|&&j| column[j] == Column::Ordered)
                .count();
            if kept > dense_row { 0 } else { kept }
        })
        .collect();
    // A column all of whose rows are ignored is left with nothing.
    let mut entries = 0;
    for j in 0..cols {
        if column[j] != Column::Ordered {
            continue;
        }
        let kept = row_indices[col_ptrs[j]..col_ptrs[j + 1]]
            .iter()
            .filter(|&&i| row_length[i] > 0)
            .count();
        if kept == 0 {
            column[j] = Column::Bare;
        }
        entries += kept;
    }
    let ordered = column.iter().filter(|&&c| c == Column::Ordered).count();

    // The quotient graph: the columns are the variables, and the rows, numbered after them, are
    // elements from the start.
    let mut quotient = QuotientGraph::new(cols + rows, cols, 2 * entries);
    for j in 0..cols {
        if column[j] != Column::Ordered {
            continue;
        }
        let in_rows = row_indices[col_ptrs[j]..col_ptrs[j + 1]]
            .iter()
            .copied()
            .filter(|&i| row_length[i] > 0);
        // Every row gives the column the others it holds: a bound of the degree in `AᵀA`, where
        // a column reached through two rows counts once.
        let degree: usize = in_rows.clone().map(|i| row_length[i] - 1).sum();
        quotient.variable(
            j,
            in_rows.map(|i| cols + i),
            [],
            Some(degree.min(ordered - 1)),
        );
    }
    for i in 0..rows {
        if row_length[i] > 0 {
            let row = &col_indices[row_ptrs[i]..row_ptrs[i + 1]];
            quotient.element(
                cols + i,
                row.iter()
                    .copied()
                    .filter(|&j| column[j] == Column::Ordered),
            );
        }
    }
    let mut order = quotient.eliminate();
    order.extend((0..cols).filter(|&j| column[j] == Column::Bare));
    let mut dense: Vec<usize> = (0..cols).filter(|&j| column[j] == Column::Dense).collect();
    dense.sort_unstable_by_key(|&j| (count(j), j));
    order.extend(dense);
    order.extend((0..cols).filter(|&j| column[j] == Column::Empty));
    // The assembly tree of the elimination is not the column elimination tree, and the rows
    // ignored by the one are part of the other: the tree that bounds the factors is that of all of
    // `Aᵀ·A`, so the ignored rows are read for it too.
    columns_in_postorder(rows, &col_ptrs, &row_indices, &order)
}
