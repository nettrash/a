//! Sparse matrices: rectangular arrays of `f32`, `f64`,
//! [`Complex<f32>`](crate::math::complex::Complex) or
//! [`Complex<f64>`](crate::math::complex::Complex) elements that store only the nonzero ones, with
//! the direct and the iterative methods that solve their systems and find a few of their
//! eigenvalues.
//!
//! # Storage
//!
//! Three forms hold the same matrix, and each converts to the other two and to a dense
//! [`Matrix`] in the time it takes to read the stored elements and the dimensions:
//!
//! * [`CooMatrix`], the coordinate form, is a list of `(row, column, value)` triplets in any
//!   order, the triplets of one cell summed when it is converted: the form a matrix is assembled
//!   in.
//! * [`SparseMatrix`], or [`CsrMatrix`], is the compressed sparse row form: the values in row
//!   order, the column of each, and where each row begins. It is the form every method of this
//!   module takes. A row is walked in the time it takes to read it, and a product by a vector
//!   costs what the nonzeros cost rather than what the shape costs.
//! * [`CscMatrix`], the compressed sparse column form, is the same by columns.
//!
//! The two compressed forms are canonical: the indices of a row or column strictly ascend and no
//! zero is stored, which every constructor restores, so two equal matrices have equal arrays. The
//! arrays of every form are reachable, and `into_parts` hands them over without a copy.
//!
//! [`read_matrix_market`] and [`write_matrix_market`] read and write the Matrix Market exchange
//! format: the coordinate and array formats, real, integer, complex and pattern fields, and
//! general, symmetric, skew-symmetric and Hermitian matrices.
//!
//! # Operations
//!
//! * The products of a sparse matrix by a vector ([`mul_vec`](SparseMatrix::mul_vec), and by its
//!   transpose and conjugate transpose without forming either), by a dense matrix and by another
//!   sparse matrix; `+`, `-`, and `*` and `/` by an element.
//! * [`transpose`](SparseMatrix::transpose), [`submatrix`](SparseMatrix::submatrix),
//!   [`select`](SparseMatrix::select), the triangles, [`permute`](SparseMatrix::permute), the
//!   norms and the trace.
//!
//! # Orderings
//!
//! A direct method fills its factors in more or less according to the order it eliminates in. An
//! [`Ordering`] names one: [`reverse_cuthill_mckee`], which narrows the band; [`amd`], the
//! approximate minimum degree; [`colamd`], its form for the columns of an unsymmetric or
//! rectangular matrix; and [`nested_dissection`]. All of them are permutations `p` with
//! `p[new] = old`. [`elimination_tree`] and [`postorder`] give the dependencies between the
//! columns of a factor and an order that keeps every subtree of them together.
//!
//! # Direct methods
//!
//! * The Cholesky factorisation `P·A·Pᵀ = L·Lᴴ` of a Hermitian positive definite matrix,
//!   [`SparseCholesky`], and the `L·D·Lᴴ` factorisation without pivoting of a Hermitian matrix
//!   that need not be definite, [`SparseLdl`], both supernodal: the columns of the factor that
//!   share a pattern are factored together as dense blocks.
//! * The LU factorisation with partial pivoting `P·A·Q = L·U`, [`SparseLu`].
//! * The multifrontal QR decomposition `A·Q_c = Q·R`, [`SparseQr`], for least squares problems.
//!
//! # Iterative methods
//!
//! * [`cg`] for Hermitian positive definite matrices, [`minres`] for Hermitian indefinite ones,
//!   and restarted [`gmres`] and [`bicgstab`] for the others. Each takes a [`LinearOperator`], so
//!   that it runs on a matrix that is never formed.
//! * The preconditioners: [`Jacobi`], the incomplete Cholesky factorisation [`Ic0`] and the
//!   incomplete LU factorisation [`Ilu0`].
//!
//! # Eigenvalues
//!
//! A few eigenvalues of a large matrix, and their vectors: the Lanczos method for Hermitian
//! matrices ([`lanczos`]) and the Arnoldi method for the others ([`arnoldi`]), both with implicit
//! restarts, and the eigenvalues nearest a target by shift-and-invert, through a [`SparseLu`]
//! ([`lanczos_shift_invert`], [`arnoldi_shift_invert`]) or an inverse of the caller's own
//! ([`lanczos_inverse_with`], [`arnoldi_inverse_with`]).
//!
//! Everything is generic over [`Scalar`], so a matrix of `f32` is stored and computed in `f32` and
//! one of complex numbers needs no conversion. No method passes off an answer it has not
//! verified: a Cholesky or `L·D·Lᴴ` factorisation that does not exist is `None`; LU and QR report
//! a matrix singular to working precision and refuse to solve it; an iterative method that cannot
//! reach its tolerance returns its best point with `converged` false and the reason; and the
//! eigenvalue methods return only the pairs whose residuals pass their check. A call with inputs
//! of the wrong shape panics.
//!
//! # Examples
//!
//! ```
//! use a::math::sparse::{CooMatrix, SparseMatrix};
//!
//! // The 1-D Laplacian of five points: tridiagonal, so four of every five entries are zero.
//! let a = SparseMatrix::tridiagonal(&[-1.0; 4], &[2.0; 5], &[-1.0; 4]);
//! assert_eq!(a.nnz(), 13);
//! assert_eq!(a[(2, 1)], -1.0);
//! assert_eq!(a[(2, 4)], 0.0);
//!
//! // The same matrix, assembled from the element matrices of its four intervals: the triplets of
//! // one cell are summed.
//! let mut triplets = CooMatrix::new(5, 5);
//! for k in 0..4 {
//!     triplets.push(k, k, 1.0);
//!     triplets.push(k, k + 1, -1.0);
//!     triplets.push(k + 1, k, -1.0);
//!     triplets.push(k + 1, k + 1, 1.0);
//! }
//! triplets.push(0, 0, 1.0);
//! triplets.push(4, 4, 1.0);
//! assert_eq!(triplets.to_csr(), a);
//!
//! // By columns, and back.
//! let by_columns = a.to_csc();
//! assert_eq!(by_columns.column(0).collect::<Vec<_>>(), [(0, 2.0), (1, -1.0)]);
//! assert_eq!(by_columns.to_csr(), a);
//! ```

mod amd;
mod cholesky;
mod colamd;
mod coo;
mod csc;
mod eigen;
mod graph;
mod io;
mod ldl;
mod lu;
mod nd;
mod operator;
mod ops;
mod ordering;
mod precond;
mod qr;
mod solve;
mod supernodal;
mod symbolic;

pub use amd::amd;
pub use cholesky::SparseCholesky;
pub use colamd::colamd;
pub use coo::CooMatrix;
pub use csc::CscMatrix;
pub use eigen::{
    ArnoldiResult, ArnoldiWhich, EigenOptions, LanczosResult, LanczosWhich, arnoldi,
    arnoldi_inverse_with, arnoldi_shift_invert, arnoldi_shift_invert_with, arnoldi_with, lanczos,
    lanczos_inverse_with, lanczos_shift_invert, lanczos_shift_invert_with, lanczos_with,
};
pub use io::{
    MatrixMarketError, MatrixMarketField, MatrixMarketOptions, MatrixMarketSource,
    MatrixMarketSymmetry, parse_matrix_market, read_matrix_market, to_matrix_market,
    to_matrix_market_with, write_matrix_market, write_matrix_market_with,
};
pub use ldl::SparseLdl;
pub use lu::{LuStrategy, SparseLu, SparseLuOptions};
pub use nd::nested_dissection;
pub use operator::{FnOperator, LinearOperator};
pub use ordering::{
    Ordering, band_width, connected_components, elimination_tree, inverse_permutation, postorder,
    profile, reverse_cuthill_mckee, symmetric_permutation,
};
pub use precond::{Ic0, Ilu0, Jacobi, Preconditioner};
pub use qr::SparseQr;
pub use solve::{
    SolveOptions, SolveResult, StopReason, bicgstab, bicgstab_with, cg, cg_with, gmres, gmres_with,
    minres, minres_with,
};

/// The compressed sparse row form, which is what a [`SparseMatrix`] is.
pub type CsrMatrix<T> = SparseMatrix<T>;

use crate::math::matrix::{Matrix, Scalar};
use core::fmt;
use core::ops::Index;

/// A matrix of `rows × cols` elements of the type `T` that stores only its nonzero elements, in
/// compressed sparse row form: the values in row order, the column index of each, and the offset in
/// those two where each row begins.
///
/// Within a row the column indices strictly ascend and no zero is stored, which is the invariant
/// every constructor restores and every algorithm here relies on. A NaN is not a zero, and is
/// stored.
///
/// Two matrices compare with `==` as the dense matrices of the same elements do: equal shapes and
/// equal elements. Like [`Matrix`] of floats, the type is neither `Eq` (a matrix holding a NaN is
/// not equal to itself) nor `Hash`.
///
/// # Examples
///
/// ```
/// use a::math::sparse::SparseMatrix;
///
/// let a = SparseMatrix::from_coo(2, 3, &[(0, 2, 1.5), (1, 0, -2.0), (1, 2, 4.0)]);
/// assert_eq!(a.shape(), (2, 3));
/// assert_eq!(a.nnz(), 3);
/// assert_eq!(a.row(1).collect::<Vec<_>>(), [(0, -2.0), (2, 4.0)]);
/// assert_eq!(a.row_pointers(), [0, 1, 3]);
/// assert_eq!(a.col_indices(), [2, 0, 2]);
/// assert_eq!(a.values(), [1.5, -2.0, 4.0]);
/// ```
#[derive(Clone)]
pub struct SparseMatrix<T> {
    /// The number of rows.
    rows: usize,
    /// The number of columns.
    cols: usize,
    /// The offset in `col_indices` and `values` where each row begins, of length `rows + 1` and
    /// nondecreasing, the last entry the number of stored elements.
    row_ptrs: Vec<usize>,
    /// The column index of each stored element, ascending within a row.
    col_indices: Vec<usize>,
    /// The stored elements, in the same order as the column indices.
    values: Vec<T>,
    /// The zero of the type, which an index into a cell that stores nothing points at: the element
    /// a sparse matrix has without storing it.
    zero: T,
}

/// The number of offsets a matrix of `n` rows or columns needs, `n + 1`, which a release build
/// would otherwise let wrap around to none.
///
/// # Panics
///
/// Panics if it overflows `usize`.
#[track_caller]
pub(crate) fn offsets(n: usize) -> usize {
    n.checked_add(1)
        .expect("the matrix has more rows or columns than a vector of offsets can hold")
}

/// The positions `0..len` in ascending order of `key(position)`, the positions of equal keys in
/// ascending order: a stable sort of keys below `bound`, in time and memory proportional to `len`.
///
/// Where the keys are few — no more than there are positions, or than 256 — it is one counting
/// pass with a bucket for each key. Otherwise it is a radix sort from the lowest digit, each digit
/// as wide as that many buckets allow, so that neither the time nor the memory grows with `bound`
/// beyond the eight digits a `usize` can hold: a matrix with far more columns than elements is
/// sorted in the time its elements take.
pub(crate) fn sorted_order(len: usize, bound: usize, key: impl Fn(usize) -> usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    if len < 2 || bound < 2 {
        return order;
    }
    let limit = len.max(256);
    // The width of a digit in bits and the number of buckets it takes; one digit as wide as the
    // word when a bucket for each key is affordable.
    let (bits, buckets) = if bound <= limit {
        (usize::BITS, bound)
    } else {
        (limit.ilog2(), 1usize << limit.ilog2())
    };
    let mask = if bits == usize::BITS {
        usize::MAX
    } else {
        buckets - 1
    };
    let top = bound - 1;
    let mut scratch = vec![0usize; len];
    let mut starts = vec![0usize; buckets + 1];
    let mut shift = 0u32;
    while shift < usize::BITS && (top >> shift) != 0 {
        starts.fill(0);
        for position in 0..len {
            starts[((key(position) >> shift) & mask) + 1] += 1;
        }
        for digit in 0..buckets {
            starts[digit + 1] += starts[digit];
        }
        // In the order of the pass before, so that equal digits keep it: the sort is stable.
        for &position in &order {
            let digit = (key(position) >> shift) & mask;
            scratch[starts[digit]] = position;
            starts[digit] += 1;
        }
        core::mem::swap(&mut order, &mut scratch);
        shift = shift.saturating_add(bits);
    }
    order
}

/// The longest row that [`sort_rows`] sorts by insertion, and that a permutation gathers rather
/// than deals out by counting.
// Measured on an Apple M4 Max (`permute` and `from_csr` of rows given in reverse, rows of 8 to
// 64 random columns, a Laplacian and an elasticity mesh; sweep 8 to 96): the two uses cross at
// different lengths. A permutation deals out by counting 21 to 35 % faster from rows of 16 on,
// while sorting by insertion stays the faster up to rows of 40; at 16 the sort of rows of 24 to
// 32 took 2.1 % longer, and from 48 on the permutations of rows of 40 to 64 1.5 to 2.1 times
// as long. Kept at 32 for the sort.
pub(crate) const SHORT_ROW: usize = 32;

/// Puts the columns of every row in ascending order, the values with them, equal columns staying
/// in the order they were given in. A read when the rows are in order already, and otherwise one
/// stable sort of all the elements by column, dealt out to the rows: linear in the elements, where
/// a sort row by row would be quadratic in the longest row or carry a logarithm — except when no row
/// is longer than [`SHORT_ROW`], where each row is sorted by insertion in place.
pub(crate) fn sort_rows<T: Copy>(
    row_ptrs: &[usize],
    bound: usize,
    col_indices: &mut Vec<usize>,
    values: &mut Vec<T>,
) {
    let rows = row_ptrs.len() - 1;
    if (0..rows).all(|i| col_indices[row_ptrs[i]..row_ptrs[i + 1]].is_sorted()) {
        return;
    }
    // Short rows, as permutations and selections of meshes give, are sorted where they are, by
    // insertion, which keeps equal columns in their order as the global pass below does.
    if (0..rows).all(|i| row_ptrs[i + 1] - row_ptrs[i] <= SHORT_ROW) {
        for i in 0..rows {
            let (begin, end) = (row_ptrs[i], row_ptrs[i + 1]);
            let (cols, vals) = (&mut col_indices[begin..end], &mut values[begin..end]);
            for k in 1..cols.len() {
                let (c, v) = (cols[k], vals[k]);
                let mut at = k;
                while at > 0 && cols[at - 1] > c {
                    cols[at] = cols[at - 1];
                    vals[at] = vals[at - 1];
                    at -= 1;
                }
                cols[at] = c;
                vals[at] = v;
            }
        }
        return;
    }
    let mut row_of = vec![0usize; col_indices.len()];
    for i in 0..rows {
        row_of[row_ptrs[i]..row_ptrs[i + 1]].fill(i);
    }
    let order = sorted_order(col_indices.len(), bound, |k| col_indices[k]);
    let mut cursor = row_ptrs[..rows].to_vec();
    let mut sorted_indices = vec![0usize; col_indices.len()];
    let mut sorted_values = values.clone();
    for k in order {
        let at = cursor[row_of[k]];
        cursor[row_of[k]] += 1;
        sorted_indices[at] = col_indices[k];
        sorted_values[at] = values[k];
    }
    *col_indices = sorted_indices;
    *values = sorted_values;
}

/// Sums the elements of a row that share a column and sit next to each other, in the order they
/// are in, and drops the zeros, in place: what makes sorted rows canonical.
fn compact<T: Scalar>(row_ptrs: &mut [usize], col_indices: &mut Vec<usize>, values: &mut Vec<T>) {
    let rows = row_ptrs.len() - 1;
    let (mut read, mut write) = (0, 0);
    for i in 0..rows {
        // The row begins where the one before ended, at `read`; its new beginning is `write`.
        let end = row_ptrs[i + 1];
        row_ptrs[i] = write;
        while read < end {
            let j = col_indices[read];
            let mut sum = values[read];
            read += 1;
            while read < end && col_indices[read] == j {
                sum += values[read];
                read += 1;
            }
            if !sum.is_zero() {
                col_indices[write] = j;
                values[write] = sum;
                write += 1;
            }
        }
    }
    row_ptrs[rows] = write;
    col_indices.truncate(write);
    values.truncate(write);
}

impl<T: Scalar> SparseMatrix<T> {
    /// The empty matrix of `rows × cols`, which stores nothing.
    ///
    /// # Panics
    ///
    /// Panics if `rows + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::<f64>::zeros(4, 7);
    /// assert_eq!(a.nnz(), 0);
    /// assert_eq!(a.density(), 0.0);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptrs: vec![0; offsets(rows)],
            col_indices: Vec::new(),
            values: Vec::new(),
            zero: T::zero(),
        }
    }

    /// The `n × n` matrix with 1 on the diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let i = SparseMatrix::<f64>::identity(3);
    /// assert_eq!(i.nnz(), 3);
    /// assert_eq!(i.trace(), 3.0);
    /// ```
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self::from_diagonal(&vec![T::one(); n])
    }

    /// The square matrix with `d` on its diagonal, of whatever length `d` is.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let d = SparseMatrix::from_diagonal(&[1.0, 0.0, 4.0]);
    /// assert_eq!(d.nnz(), 2);
    /// assert_eq!(d[(2, 2)], 4.0);
    /// ```
    #[must_use]
    pub fn from_diagonal(d: &[T]) -> Self {
        let rows = d.len();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut row_ptrs = Vec::with_capacity(rows + 1);
        for (k, &x) in d.iter().enumerate() {
            row_ptrs.push(col_indices.len());
            if !x.is_zero() {
                col_indices.push(k);
                values.push(x);
            }
        }
        row_ptrs.push(col_indices.len());
        Self {
            rows,
            cols: rows,
            row_ptrs,
            col_indices,
            values,
            zero: T::zero(),
        }
    }

    /// The matrix with `lower` on the subdiagonal, `diagonal` on the diagonal and `upper` on the
    /// superdiagonal, the three of lengths `n − 1`, `n` and `n − 1`. A zero of the three is not
    /// stored, so this builds a tridiagonal, bidiagonal or diagonal matrix as it is given one.
    ///
    /// # Panics
    ///
    /// Panics if `lower` or `upper` isn't one shorter than `diagonal`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// // The second-difference matrix of four intervals.
    /// let a = SparseMatrix::tridiagonal(&[-1.0; 3], &[2.0; 4], &[-1.0; 3]);
    /// assert_eq!(a.nnz(), 10);
    /// assert_eq!(a[(1, 0)], -1.0);
    /// assert_eq!(a[(1, 3)], 0.0);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn tridiagonal(lower: &[T], diagonal: &[T], upper: &[T]) -> Self {
        let n = diagonal.len();
        assert_eq!(
            lower.len(),
            n.saturating_sub(1),
            "the subdiagonal is the wrong length"
        );
        assert_eq!(
            upper.len(),
            n.saturating_sub(1),
            "the superdiagonal is the wrong length"
        );
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut row_ptrs = Vec::with_capacity(n + 1);
        for i in 0..n {
            row_ptrs.push(col_indices.len());
            for (j, x) in [
                (
                    i.saturating_sub(1),
                    if i > 0 { lower[i - 1] } else { T::zero() },
                ),
                (i, diagonal[i]),
                (i + 1, if i + 1 < n { upper[i] } else { T::zero() }),
            ] {
                if j < n && !x.is_zero() {
                    col_indices.push(j);
                    values.push(x);
                }
            }
        }
        row_ptrs.push(col_indices.len());
        Self {
            rows: n,
            cols: n,
            row_ptrs,
            col_indices,
            values,
            zero: T::zero(),
        }
    }

    /// The matrix of `rows × cols` holding the entries `(i, j, x)`, where the entries of one cell
    /// are summed, in the order they are given in, and a zero is not stored. The order of the
    /// entries doesn't matter otherwise. A [`CooMatrix`] holds such entries as a matrix of its own.
    ///
    /// The time and the memory are proportional to the entries and the rows, however long a row
    /// is and however many columns there are.
    ///
    /// # Panics
    ///
    /// Panics if an entry lies outside the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 1.0), (0, 1, 2.0), (1, 0, 3.0)]);
    /// assert_eq!(a[(0, 1)], 3.0);
    /// assert_eq!(a.nnz(), 2);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn from_coo(rows: usize, cols: usize, entries: &[(usize, usize, T)]) -> Self {
        for &(i, j, _) in entries {
            assert!(i < rows && j < cols, "an entry lies outside the matrix");
        }
        Self::assemble(
            rows,
            cols,
            entries.len(),
            |k| entries[k].0,
            |k| entries[k].1,
            |k| entries[k].2,
        )
    }

    /// The matrix of `rows × cols` holding the `len` entries `(row(k), col(k), value(k))`, all of
    /// them inside it: the entries of one cell summed in the order of `k`, and no zero stored.
    ///
    /// The entries are sorted by column, stably, and then dealt out to their rows in that order,
    /// which leaves every row in ascending order of column with equal columns in the order given:
    /// two linear passes rather than a sort of each row.
    #[track_caller]
    pub(crate) fn assemble(
        rows: usize,
        cols: usize,
        len: usize,
        row: impl Fn(usize) -> usize,
        col: impl Fn(usize) -> usize,
        value: impl Fn(usize) -> T,
    ) -> Self {
        let mut row_ptrs = vec![0usize; offsets(rows)];
        for k in 0..len {
            row_ptrs[row(k) + 1] += 1;
        }
        for i in 0..rows {
            row_ptrs[i + 1] += row_ptrs[i];
        }
        let mut cursor = row_ptrs[..rows].to_vec();
        let mut col_indices = vec![0usize; len];
        let mut values = vec![T::zero(); len];
        for k in sorted_order(len, cols, &col) {
            let i = row(k);
            let at = cursor[i];
            cursor[i] += 1;
            col_indices[at] = col(k);
            values[at] = value(k);
        }
        compact(&mut row_ptrs, &mut col_indices, &mut values);
        Self {
            rows,
            cols,
            row_ptrs,
            col_indices,
            values,
            zero: T::zero(),
        }
    }

    /// The matrix whose element in row `i` and column `j` is `f(i, j)`, of which only the nonzeros
    /// are kept. The function is called `rows · cols` times, so this is how a matrix is built when
    /// its elements are cheap and its pattern isn't known beforehand.
    ///
    /// The calls go row after row, the order the matrix is held in, where
    /// [`Matrix::from_fn`] goes column after column: a function with a state of its own sees
    /// the cells in another order.
    ///
    /// # Panics
    ///
    /// Panics if `rows + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// // The pattern of the off-diagonal of a matrix: n(n − 1) nonzeros.
    /// let a = SparseMatrix::from_fn(4, 4, |i, j| if i == j { 0.0 } else { 1.0 });
    /// assert_eq!(a.nnz(), 12);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        let mut row_ptrs = Vec::with_capacity(offsets(rows));
        for i in 0..rows {
            row_ptrs.push(col_indices.len());
            for j in 0..cols {
                let x = f(i, j);
                if !x.is_zero() {
                    col_indices.push(j);
                    values.push(x);
                }
            }
        }
        row_ptrs.push(col_indices.len());
        Self {
            rows,
            cols,
            row_ptrs,
            col_indices,
            values,
            zero: T::zero(),
        }
    }

    /// The sparse matrix of the dense `a`, its zeros left out, in the `rows · cols` steps it takes
    /// to read `a`.
    ///
    /// # Panics
    ///
    /// Panics if `rows + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let dense = Matrix::from_rows(&[[1.0, 0.0], [0.0, 2.0]]);
    /// let a = SparseMatrix::from_matrix(&dense);
    /// assert_eq!(a.nnz(), 2);
    /// assert_eq!(a.to_matrix(), dense);
    /// ```
    #[must_use]
    pub fn from_matrix(a: &Matrix<T>) -> Self {
        let (rows, cols) = a.shape();
        // The dense matrix is held by columns: one pass over them counts the nonzeros of each row,
        // and a second deals them out, the columns of a row ascending as they are met.
        let mut row_ptrs = vec![0usize; offsets(rows)];
        for j in 0..cols {
            for (i, x) in a.column(j).iter().enumerate() {
                if !x.is_zero() {
                    row_ptrs[i + 1] += 1;
                }
            }
        }
        for i in 0..rows {
            row_ptrs[i + 1] += row_ptrs[i];
        }
        let mut cursor = row_ptrs[..rows].to_vec();
        let mut col_indices = vec![0usize; row_ptrs[rows]];
        let mut values = vec![T::zero(); row_ptrs[rows]];
        for j in 0..cols {
            for (i, &x) in a.column(j).iter().enumerate() {
                if !x.is_zero() {
                    let at = cursor[i];
                    cursor[i] += 1;
                    col_indices[at] = j;
                    values[at] = x;
                }
            }
        }
        Self {
            rows,
            cols,
            row_ptrs,
            col_indices,
            values,
            zero: T::zero(),
        }
    }

    /// The dense matrix of the same elements, in `rows · cols` steps and one for each stored
    /// element.
    ///
    /// # Panics
    ///
    /// Panics if `rows · cols` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(1, 3, &[(0, 1, 7.0)]);
    /// assert_eq!(a.to_matrix().as_slice(), &[0.0, 7.0, 0.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn to_matrix(&self) -> Matrix<T> {
        let mut dense = Matrix::zeros(self.rows, self.cols);
        let data = dense.as_mut_slice();
        for i in 0..self.rows {
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                data[i + self.col_indices[k] * self.rows] = self.values[k];
            }
        }
        dense
    }

    /// The same matrix in compressed sparse column form, in time proportional to the stored
    /// elements, the rows and the columns.
    ///
    /// # Panics
    ///
    /// Panics if `cols + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 3, &[(0, 2, 1.5), (1, 0, -2.0), (1, 2, 4.0)]);
    /// let c = a.to_csc();
    /// assert_eq!(c.col_pointers(), [0, 1, 1, 3]);
    /// assert_eq!(c.row_indices(), [1, 0, 1]);
    /// assert_eq!(c.values(), [-2.0, 1.5, 4.0]);
    /// assert_eq!(c.to_csr(), a);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn to_csc(&self) -> CscMatrix<T> {
        CscMatrix::from_transposed(self.transpose())
    }

    /// The same matrix in coordinate form: its stored elements as triplets, in row order, in time
    /// proportional to them and to the rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 3, &[(1, 2, 4.0), (0, 2, 1.5), (1, 0, -2.0)]);
    /// let c = a.to_coo();
    /// assert_eq!(c.iter().collect::<Vec<_>>(), [(0, 2, 1.5), (1, 0, -2.0), (1, 2, 4.0)]);
    /// assert_eq!(c.to_csr(), a);
    /// ```
    #[must_use]
    pub fn to_coo(&self) -> CooMatrix<T> {
        CooMatrix::from_parts_unchecked(
            self.rows,
            self.cols,
            self.expanded_rows(),
            self.col_indices.clone(),
            self.values.clone(),
        )
    }

    /// The row of each stored element, in the order they are stored in: the offsets of the rows,
    /// expanded.
    fn expanded_rows(&self) -> Vec<usize> {
        let mut row_indices = vec![0usize; self.values.len()];
        for i in 0..self.rows {
            row_indices[self.row_ptrs[i]..self.row_ptrs[i + 1]].fill(i);
        }
        row_indices
    }

    /// The number of rows.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    #[inline]
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `(rows, cols)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// assert_eq!(SparseMatrix::<f64>::zeros(2, 5).shape(), (2, 5));
    /// ```
    #[inline]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Whether it has as many rows as columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// assert!(SparseMatrix::<f64>::identity(3).is_square());
    /// assert!(!SparseMatrix::<f64>::zeros(2, 5).is_square());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The number of stored elements, which is the number of nonzero ones.
    #[inline]
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Whether the matrix stores nothing: it has no nonzero element, whatever its shape. This is
    /// not the question [`Matrix::is_empty`] answers, whether there are no cells at all; that is
    /// `rows() == 0 || cols() == 0` here.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// assert!(SparseMatrix::<f64>::zeros(3, 3).is_empty());
    /// assert!(!SparseMatrix::<f64>::identity(3).is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// The fraction of the elements that are stored, in `[0, 1]`, and 0 of a matrix with no rows
    /// or no columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::<f64>::tridiagonal(&[-1.0; 3], &[2.0; 4], &[-1.0; 3]);
    /// assert!((a.density() - 0.625).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn density(&self) -> T::Real {
        use crate::math::complex::Float;
        // The number of cells, exactly: a `usize` product of the dimensions of a very sparse
        // matrix can overflow, on a 32-bit machine at 65 536 × 65 536 already, and one in floats
        // would round twice.
        let cells = self.rows as u128 * self.cols as u128;
        if cells == 0 {
            return T::Real::ZERO;
        }
        T::Real::from_f64(self.values.len() as f64 / cells as f64)
    }

    /// The offsets of the rows in the stored elements, of length `rows + 1`: row `i` is the
    /// elements from `row_pointers()[i]` up to `row_pointers()[i + 1]`.
    #[must_use]
    pub fn row_pointers(&self) -> &[usize] {
        &self.row_ptrs
    }

    /// The column indices of the stored elements, ascending within a row.
    #[must_use]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// The stored elements, in the order of the column indices.
    #[must_use]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// The matrix taken apart, without a copy: `(rows, cols, row_pointers, col_indices, values)`,
    /// which [`from_csr`](Self::from_csr) puts together again.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 3, &[(0, 2, 1.5), (1, 0, -2.0), (1, 2, 4.0)]);
    /// let (rows, cols, row_pointers, col_indices, values) = a.clone().into_parts();
    /// assert_eq!((rows, cols), (2, 3));
    /// assert_eq!(row_pointers, [0, 1, 3]);
    /// assert_eq!(SparseMatrix::from_csr(rows, cols, row_pointers, col_indices, values), a);
    /// ```
    #[must_use]
    pub fn into_parts(self) -> (usize, usize, Vec<usize>, Vec<usize>, Vec<T>) {
        (
            self.rows,
            self.cols,
            self.row_ptrs,
            self.col_indices,
            self.values,
        )
    }

    /// The stored elements of row `i`, paired with their columns, which ascend.
    ///
    /// # Panics
    ///
    /// Panics if `i` isn't a row of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 2.0)]);
    /// assert_eq!(a.row(0).collect::<Vec<_>>(), [(0, 1.0), (1, 2.0)]);
    /// assert_eq!(a.row(1).count(), 0);
    /// ```
    #[track_caller]
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        assert!(i < self.rows, "the row is outside the matrix");
        let (begin, end) = (self.row_ptrs[i], self.row_ptrs[i + 1]);
        (begin..end).map(move |k| (self.col_indices[k], self.values[k]))
    }

    /// The stored elements of column `j`, paired with their rows, which ascend.
    ///
    /// The matrix is held by rows, so nothing says where a column is: every row is searched for
    /// it, a binary search each, `O(rows · log(nnz / rows))`. Where more than a few columns are
    /// wanted, [`to_csc`](Self::to_csc) is the way to have them all, in the time one pass over the
    /// matrix takes, and [`CscMatrix::column`] then reads one in the time its elements take.
    ///
    /// # Panics
    ///
    /// Panics if `j` isn't a column of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(3, 2, &[(0, 1, 1.0), (2, 1, 2.0), (2, 0, 3.0)]);
    /// assert_eq!(a.column(1), [(0, 1.0), (2, 2.0)]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn column(&self, j: usize) -> Vec<(usize, T)> {
        assert!(j < self.cols, "the column is outside the matrix");
        let mut out = Vec::new();
        for i in 0..self.rows {
            if let Some(k) = self.find(i, j) {
                out.push((i, self.values[k]));
            }
        }
        out
    }

    /// The diagonal, with a zero where the matrix stores none: `min(rows, cols)` elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 3, &[(0, 0, 5.0), (1, 2, 4.0)]);
    /// assert_eq!(a.diagonal(), [5.0, 0.0]);
    /// ```
    #[must_use]
    pub fn diagonal(&self) -> Vec<T> {
        (0..self.rows.min(self.cols))
            .map(|i| self[(i, i)])
            .collect()
    }

    /// The transpose, in time proportional to the stored elements, the rows and the columns.
    ///
    /// # Panics
    ///
    /// Panics if `cols + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 3, &[(0, 1, 1.0), (1, 0, 2.0), (1, 2, 3.0)]);
    /// let t = a.transpose();
    /// assert_eq!((t.rows(), t.cols(), t.nnz()), (3, 2, 3));
    /// assert_eq!(t[(1, 0)], 1.0);
    /// assert_eq!(t.transpose(), a);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn transpose(&self) -> Self {
        // A counting sort by column: the offsets of the columns, then every element dealt to its
        // column in row order, which leaves the rows of a column ascending.
        let mut row_ptrs = vec![0usize; offsets(self.cols)];
        for &j in &self.col_indices {
            row_ptrs[j + 1] += 1;
        }
        for j in 0..self.cols {
            row_ptrs[j + 1] += row_ptrs[j];
        }
        let mut cursor = row_ptrs[..self.cols].to_vec();
        let mut col_indices = vec![0usize; self.values.len()];
        let mut values = vec![T::zero(); self.values.len()];
        for i in 0..self.rows {
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                let j = self.col_indices[k];
                let at = cursor[j];
                cursor[j] += 1;
                col_indices[at] = i;
                values[at] = self.values[k];
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            row_ptrs,
            col_indices,
            values,
            zero: T::zero(),
        }
    }

    /// The matrix of the same pattern whose elements are `f(i, j, x)`; a zero the function returns
    /// leaves the pattern, which is smaller then.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 1.0), (1, 0, 4.0)]);
    /// let root = a.map(|_, _, x: f64| x.sqrt());
    /// assert_eq!(root[(1, 0)], 2.0);
    /// let dropped = a.map(|i, _, x| if i == 0 { x } else { 0.0 });
    /// assert_eq!(dropped.nnz(), 1);
    /// ```
    #[must_use]
    pub fn map(&self, mut f: impl FnMut(usize, usize, T) -> T) -> Self {
        let mut col_indices = Vec::with_capacity(self.values.len());
        let mut values = Vec::with_capacity(self.values.len());
        let mut row_ptrs = Vec::with_capacity(self.rows + 1);
        for i in 0..self.rows {
            row_ptrs.push(col_indices.len());
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                let j = self.col_indices[k];
                let x = f(i, j, self.values[k]);
                if !x.is_zero() {
                    col_indices.push(j);
                    values.push(x);
                }
            }
        }
        row_ptrs.push(col_indices.len());
        Self {
            rows: self.rows,
            cols: self.cols,
            row_ptrs,
            col_indices,
            values,
            zero: T::zero(),
        }
    }

    /// Whether the matrix is square and equal to its transpose.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 3.0), (1, 0, 3.0), (1, 1, 1.0)]);
    /// assert!(a.is_symmetric());
    /// assert!(!a.map(|i, _, x| if i == 0 { -x } else { x }).is_symmetric());
    /// ```
    #[must_use]
    pub fn is_symmetric(&self) -> bool {
        self.mirrored(|x, y| x == y)
    }

    /// Whether the matrix is square and equal to its conjugate transpose, which for a real matrix
    /// is being symmetric.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(
    ///     2,
    ///     2,
    ///     &[(0, 1, Complex::new(1.0, 2.0)), (1, 0, Complex::new(1.0, -2.0))],
    /// );
    /// assert!(a.is_hermitian());
    /// assert!(!a.is_symmetric());
    /// ```
    #[must_use]
    pub fn is_hermitian(&self) -> bool {
        self.mirrored(|x, y| x == y.conj())
    }

    /// Whether the pattern is symmetric: the matrix is square and `(i, j)` is stored exactly when
    /// `(j, i)` is. The elements themselves may still differ —
    /// [`is_symmetric`](Self::is_symmetric) asks about those.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 1, 3.0), (1, 0, -7.0)]);
    /// assert!(a.is_symmetric_pattern());
    /// assert!(!a.is_symmetric());
    /// ```
    #[must_use]
    pub fn is_symmetric_pattern(&self) -> bool {
        self.mirrored(|_, _| true)
    }

    /// Whether the matrix is square and every stored element `x` at `(i, j)` has a stored mirror
    /// `y` at `(j, i)` with `agree(x, y)`: one pass over the stored elements and an offset for
    /// each row, without a transpose or a search.
    pub(crate) fn mirrored(&self, agree: impl Fn(T, T) -> bool) -> bool {
        if self.rows != self.cols {
            return false;
        }
        // `next[j]` is the first element of row j that no earlier row has claimed as its mirror.
        // The rows are visited in ascending order and so are the columns within them, so the
        // mirror of `(i, j)` is exactly that element when the pattern is symmetric: the rows
        // before `i` have claimed the elements of row j in the columns before `i`, one each.
        let mut next = self.row_ptrs[..self.rows].to_vec();
        for i in 0..self.rows {
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                let j = self.col_indices[k];
                let at = next[j];
                if at >= self.row_ptrs[j + 1]
                    || self.col_indices[at] != i
                    || !agree(self.values[k], self.values[at])
                {
                    return false;
                }
                next[j] = at + 1;
            }
        }
        true
    }

    /// The rows as `(column, value)` lists, for the methods that want a whole row at a time as a
    /// vector they can index.
    #[must_use]
    #[allow(dead_code)] // The factorisations of the other files call it.
    pub(crate) fn row_lists(&self) -> Vec<Vec<(usize, T)>> {
        (0..self.rows).map(|i| self.row(i).collect()).collect()
    }

    /// The matrix of the compressed sparse row arrays: `row_ptrs` of length `rows + 1` gives where
    /// each row begins in `col_indices` and `values`. The columns of a row may come in any order,
    /// equal ones are summed in the order given, and zeros are left out, so the result has the
    /// canonical form whatever it was given.
    ///
    /// Arrays that are canonical already are taken as they are, after one pass that checks them;
    /// the others are sorted and summed in time proportional to the elements and the rows.
    /// [`into_parts`](Self::into_parts) is the way back.
    ///
    /// # Panics
    ///
    /// Panics if `row_ptrs` isn't `rows + 1` nondecreasing offsets from 0 to the number of
    /// elements, if `col_indices` and `values` differ in length, or if a column is outside the
    /// matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_csr(2, 3, vec![0, 2, 3], vec![2, 0, 1], vec![1.0, 2.0, 3.0]);
    /// assert_eq!(a.row(0).collect::<Vec<_>>(), [(0, 2.0), (2, 1.0)]);
    /// assert_eq!(a[(1, 1)], 3.0);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn from_csr(
        rows: usize,
        cols: usize,
        mut row_ptrs: Vec<usize>,
        mut col_indices: Vec<usize>,
        mut values: Vec<T>,
    ) -> Self {
        assert_eq!(
            Some(row_ptrs.len()),
            rows.checked_add(1),
            "there isn't an offset for each row and the end"
        );
        assert_eq!(
            col_indices.len(),
            values.len(),
            "the columns and the values differ in length"
        );
        assert_eq!(row_ptrs[0], 0, "the first offset isn't 0");
        assert_eq!(
            row_ptrs[rows],
            values.len(),
            "the last offset isn't the number of elements"
        );
        assert!(
            row_ptrs.windows(2).all(|w| w[0] <= w[1]),
            "the offsets decrease"
        );
        // One pass checks the columns and finds whether the arrays are canonical already: every
        // row strictly ascending and no zero stored. Arrays that are need nothing more.
        let mut canonical = true;
        for w in row_ptrs.windows(2) {
            let (row, vals) = (&col_indices[w[0]..w[1]], &values[w[0]..w[1]]);
            let mut previous = None;
            for (&j, x) in row.iter().zip(vals) {
                assert!(j < cols, "a column lies outside the matrix");
                canonical &= previous.is_none_or(|p| p < j) && !x.is_zero();
                previous = Some(j);
            }
        }
        if !canonical {
            sort_rows(&row_ptrs, cols, &mut col_indices, &mut values);
            compact(&mut row_ptrs, &mut col_indices, &mut values);
        }
        Self {
            rows,
            cols,
            row_ptrs,
            col_indices,
            values,
            zero: T::zero(),
        }
    }

    /// The conjugate transpose, which is the transpose of a real matrix.
    ///
    /// # Panics
    ///
    /// Panics if `cols + 1` overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::complex::Complex;
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(1, 2, &[(0, 1, Complex::new(1.0, 2.0))]);
    /// assert_eq!(a.conjugate_transpose()[(1, 0)], Complex::new(1.0, -2.0));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn conjugate_transpose(&self) -> Self {
        let mut t = self.transpose();
        if T::IS_COMPLEX {
            for x in &mut t.values {
                *x = x.conj();
            }
        }
        t
    }

    /// The offset of the element `(i, j)` in the stored elements, or `None` where the matrix stores
    /// none, or has no such cell: a binary search of row `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::SparseMatrix;
    ///
    /// let a = SparseMatrix::from_coo(2, 3, &[(0, 2, 1.5), (1, 0, -2.0), (1, 2, 4.0)]);
    /// assert_eq!(a.find(1, 2), Some(2));
    /// assert_eq!(a.values()[2], 4.0);
    /// assert_eq!(a.find(0, 0), None);
    /// assert_eq!(a.find(5, 0), None);
    /// ```
    #[must_use]
    pub fn find(&self, i: usize, j: usize) -> Option<usize> {
        if i >= self.rows || j >= self.cols {
            return None;
        }
        let (begin, end) = (self.row_ptrs[i], self.row_ptrs[i + 1]);
        let found = self.col_indices[begin..end].binary_search(&j);
        found.ok().map(|k| begin + k)
    }

    /// The matrix with each row's elements of equal column, which must sit next to each other,
    /// summed and its zeros left out, which is what every constructor returns. A row built in
    /// ascending order of column needs nothing else to be canonical.
    #[must_use]
    pub(crate) fn compress(mut self) -> Self {
        compact(&mut self.row_ptrs, &mut self.col_indices, &mut self.values);
        self
    }

    /// The matrix of arrays that are canonical already: the indices of each row strictly ascending
    /// and no zero among the values. Nothing is checked in a release build.
    pub(crate) fn from_canonical(
        rows: usize,
        cols: usize,
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        debug_assert_eq!(row_ptrs.len(), rows + 1);
        debug_assert_eq!(row_ptrs[rows], values.len());
        debug_assert_eq!(col_indices.len(), values.len());
        debug_assert!((0..rows).all(|i| {
            let row = &col_indices[row_ptrs[i]..row_ptrs[i + 1]];
            row.windows(2).all(|w| w[0] < w[1]) && row.last().is_none_or(|&j| j < cols)
        }));
        debug_assert!(values.iter().all(|x| !x.is_zero()));
        Self {
            rows,
            cols,
            row_ptrs,
            col_indices,
            values,
            zero: T::zero(),
        }
    }

    /// The matrix whose rows are the given ones, each a list of `(column, value)` pairs whose
    /// columns ascend; the internal way of building a factor without the sort of `from_coo`.
    #[must_use]
    #[allow(dead_code)] // The factorisations of the other files call it.
    pub(crate) fn from_row_lists(rows: usize, cols: usize, row_lists: &[Vec<(usize, T)>]) -> Self {
        let mut row_ptrs = Vec::with_capacity(rows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for list in row_lists {
            row_ptrs.push(col_indices.len());
            for &(j, x) in list {
                if !x.is_zero() {
                    col_indices.push(j);
                    values.push(x);
                }
            }
        }
        row_ptrs.push(col_indices.len());
        Self {
            rows,
            cols,
            row_ptrs,
            col_indices,
            values,
            zero: T::zero(),
        }
        .compress()
    }
}

impl<T: Scalar> Index<(usize, usize)> for SparseMatrix<T> {
    type Output = T;

    /// The element in row `i` and column `j`, the zero of the type where the matrix stores none.
    ///
    /// # Panics
    ///
    /// Panics if the pair is outside the matrix.
    #[track_caller]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        assert!(
            i < self.rows && j < self.cols,
            "the element is outside the matrix"
        );
        match self.find(i, j) {
            Some(k) => &self.values[k],
            None => &self.zero,
        }
    }
}

impl<T: PartialEq> PartialEq for SparseMatrix<T> {
    /// Whether the two have the same shape and the same element in every cell, which their stored
    /// elements decide, both being canonical. A matrix that holds a NaN is not equal to itself, as
    /// a dense matrix that holds one isn't.
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && self.row_ptrs == other.row_ptrs
            && self.col_indices == other.col_indices
            && self.values == other.values
    }
}

impl<T: Scalar> Default for SparseMatrix<T> {
    /// The `0 × 0` matrix, which stores nothing.
    fn default() -> Self {
        Self::zeros(0, 0)
    }
}

impl<T: fmt::Display> fmt::Display for SparseMatrix<T> {
    /// Writes the stored elements, one to a line, as `(i, j): value`, in row and then column
    /// order; nothing for a matrix that stores none.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                writeln!(f, "({}, {}): {}", i, self.col_indices[k], self.values[k])?;
            }
        }
        Ok(())
    }
}

impl<T: fmt::Debug> fmt::Debug for SparseMatrix<T> {
    /// Writes the shape and the stored elements in row order, each under its row and column:
    /// `2 × 3 {(0, 2): 1.5, (1, 0): -2.0}`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} × {} ", self.rows, self.cols)?;
        let mut map = f.debug_map();
        for i in 0..self.rows {
            for k in self.row_ptrs[i]..self.row_ptrs[i + 1] {
                map.entry(&(i, self.col_indices[k]), &self.values[k]);
            }
        }
        map.finish()
    }
}

impl<T: Scalar> From<&SparseMatrix<T>> for CscMatrix<T> {
    /// The same matrix by columns: [`SparseMatrix::to_csc`].
    fn from(a: &SparseMatrix<T>) -> Self {
        a.to_csc()
    }
}

impl<T: Scalar> From<SparseMatrix<T>> for CscMatrix<T> {
    /// The same matrix by columns: [`SparseMatrix::to_csc`].
    fn from(a: SparseMatrix<T>) -> Self {
        a.to_csc()
    }
}

impl<T: Scalar> From<&SparseMatrix<T>> for CooMatrix<T> {
    /// The stored elements as triplets: [`SparseMatrix::to_coo`].
    fn from(a: &SparseMatrix<T>) -> Self {
        a.to_coo()
    }
}

impl<T: Scalar> From<SparseMatrix<T>> for CooMatrix<T> {
    /// The stored elements as triplets, as [`SparseMatrix::to_coo`] gives them, the columns and
    /// the values taken over without a copy.
    fn from(a: SparseMatrix<T>) -> Self {
        let row_indices = a.expanded_rows();
        CooMatrix::from_parts_unchecked(a.rows, a.cols, row_indices, a.col_indices, a.values)
    }
}
