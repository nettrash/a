//! The sparse QR decomposition, by the multifrontal method with Householder reflections.
//!
//! # The decomposition
//!
//! [`SparseQr`] factors an `m × n` matrix with `m ≥ n` as `A·Q_c = Q·R`: `Q_c` a permutation of
//! the columns chosen to keep the factors sparse, `Q` of `m × n` with orthonormal columns, kept
//! as the Householder reflections that make it, and `R` of `n × n`, upper triangular and sparse.
//!
//! # The method
//!
//! The rows of `R` depend on each other along the column elimination tree, the elimination tree
//! of `(A·Q_c)ᴴ·(A·Q_c)`: row `k` of `R` comes from the rows of the matrix that begin in column
//! `k` and from what the children of `k` in the tree left over (George and Heath, "Solution of
//! sparse linear least squares problems using Givens rotations", Linear Algebra Appl. 34, 1980;
//! Liu, "On general row merging schemes for sparse Givens transformations", SIAM J. Sci. Stat.
//! Comput. 7, 1986). The multifrontal method does that merging on small dense matrices, the
//! fronts (Matstoms, "Sparse QR factorization in MATLAB", ACM Trans. Math. Softw. 20, 1994;
//! Amestoy, Duff and Puglisi, "Multifrontal QR factorization in a multiprocessor environment",
//! Numer. Linear Algebra Appl. 3, 1996; Davis, "Algorithm 915, SuiteSparseQR", ACM Trans. Math.
//! Softw. 38, 2011):
//!
//! * A front belongs to a chain of columns of the tree whose rows of `R` have the same pattern
//!   but for their leading columns, a fundamental supernode: its pivot columns. Its columns are
//!   those and the others its rows reach; its rows are the rows of the matrix that begin in a
//!   pivot column, and the contribution blocks of its children. In a matrix of 128 columns or
//!   more, small fronts are then merged into their parents, the relaxed amalgamation of Ashcraft
//!   and Grimes ("The influence of relaxed supernode partitions on the multifrontal method", ACM
//!   Trans. Math. Softw. 15, 1989) by the rule of the Cholesky factorisation's supernodes on the
//!   explicit zeros that adds to `R`: any merge into a front of at most 4 pivot columns, and up
//!   to 30 % zeros for 16, 10 % for 48 and 5 % for more. A merged child's rows of `R` take all
//!   the columns of its parent, zeros where its own rows do not reach.
//! * The front is reduced to an upper triangle by Householder reflections, LAPACK's `larfg`: a
//!   front of at most 64 columns one reflection after another, each applied to the columns
//!   after it, and a wider one in panels of 16 columns ([`front`]). A front of 128 reflections
//!   or more whose staircase fills at least half of its rectangle is held dense, and the columns
//!   right of each panel take the panel's reflections as one block, `I − V·T·Vᴴ` (the compact
//!   WY form of Schreiber and Van Loan, SIAM J. Sci. Stat. Comput. 10, 1989), by products of
//!   dense blocks; the others take them one reflection at a time, four columns at a time. Its first rows, one for each
//!   pivot column, are rows of `R`; the rest of the triangle is its contribution block, which
//!   its parent takes in; the rows below the triangle are zero and are dropped.
//! * The rows of a front are sorted by their first columns, so that the reflection of a column
//!   works on the rows that have reached that column and on no others: the staircase of the
//!   front. A block that a child has made triangular already costs its parent only the rows
//!   that are new.
//!
//! Everything about the fronts follows from the pattern of the matrix, and the symbolic analysis
//! in [`symbolic`] finds it first: the tree, the fronts and their columns, the staircases, and
//! with them the length of every row of `R` and of every Householder vector, so that the numeric
//! phase allocates its results once and works in two buffers, the largest front and the stack of
//! contribution blocks.
//!
//! `Q` is never formed. It is kept as the reflections of the fronts, each a sparse vector on the
//! rows of its front (Lu and Barlow, "Multifrontal computation with the orthogonal factors of
//! sparse matrices", SIAM J. Matrix Anal. Appl. 17, 1996), and applied front by front.
//!
//! # The cost
//!
//! The work is that of the reflections inside the staircases, and the memory that of `R`, of the
//! vectors and of the two buffers: nothing is of the order of `m·n` unless the factors are. A
//! column Householder method that keeps every row in its vectors until the row is a pivot
//! (George and Ng's, as `cs_qr` of Davis's "Direct Methods for Sparse Linear Systems", ch. 5)
//! has simpler vectors, but they grow with the depth of the tree times the rows that are never
//! pivots, which for a matrix of many more rows than columns is most of them; reducing each
//! front to a triangle is what removes those rows as soon as they are spent.
//!
//! # Singular matrices
//!
//! There is no pivoting for rank. A column that depends on the columns before it leaves a zero,
//! or a rounding error, on the diagonal of `R`, and the rounding error can be of any size: where
//! the dependency has large coefficients they amplify it, and a triangle can be singular to
//! working precision with no small diagonal element at all. So the diagonal is not trusted
//! alone: [`SparseQr::solve`] refuses `R` when a diagonal element is small, or when a lower
//! bound of its condition number, from a few solves with `R` and `Rᴴ` ([`condition`]), passes
//! `1/(max(m, 32)·ε)`; and [`SparseQr::rank`] counts the diagonal elements that are not small,
//! one fewer when every one of them is but the bound passes the limit all the same — an
//! estimate.
//!
//! # What is not done
//!
//! A matrix with fewer rows than columns is refused.

mod condition;
mod front;
mod symbolic;

use super::{Ordering, SparseMatrix};
use crate::math::complex::Float;
use crate::math::matrix::householder::{reflect_left, reflector};
use crate::math::matrix::{Matrix, Scalar};
use condition::singular_to_working_precision;
use core::fmt;
use symbolic::{Symbolic, analyse};

/// The QR decomposition `A·Q_c = Q·R` of a sparse `m × n` matrix with `m ≥ n`, by the
/// multifrontal method with Householder reflections: `Q_c` a permutation of the columns, `Q` of
/// `m × n` with orthonormal columns (unitary for complex elements when `m = n`), kept as its
/// reflections, and `R` of `n × n`, upper triangular and sparse.
///
/// The diagonal of `R` is real, of the signs the reflections leave, which are not made positive
/// afterwards. The reflection of a column works on the rows that reach it, those that store an
/// element in it or have taken one from an earlier reflection: for `α` the first of their
/// elements and `x` the others, `R_jj = −sign(Re α)·‖(α, x)‖`, the sign of zero taken as
/// positive, as LAPACK's `geqrf`; where `x` is zero and `α` real there is nothing to reflect, and
/// `R_jj = α`, of either sign. So a real diagonal matrix is its own `R`, and so is any real
/// `A·Q_c` whose rows can be put in an order that makes it upper triangular with no zero on its
/// diagonal, up to that order;
/// a complex `α` with nothing below it is still turned real, `diag(i, 3 − 4i)` having the `R` of
/// `diag(−1, −5)`. Up to the signs of its rows, `R` is the conjugate transpose of the Cholesky
/// factor of `(A·Q_c)ᴴ·(A·Q_c)`.
///
/// A column that depends on those before it leaves a zero on the diagonal, or a rounding error
/// where the arithmetic is not exact, which large coefficients in the dependency amplify: such
/// a matrix has a decomposition all the same, which [`rank`](Self::rank) counts and
/// [`solve`](Self::solve) refuses, from the diagonal and from an estimate of the condition
/// number of `R`.
///
/// The decomposition solves the least-squares problem `min ‖A·x − b‖₂` without forming `Aᴴ·A`,
/// whose condition number is the square of that of `A`.
///
/// # Examples
///
/// ```
/// use a::math::matrix::{Matrix, Norm};
/// use a::math::sparse::{SparseMatrix, SparseQr};
///
/// // The line through four points, three of which are on y = 1 + 2x.
/// let a = SparseMatrix::from_coo(4, 2, &[
///     (0, 0, 1.0f64),
///     (1, 0, 1.0), (1, 1, 1.0),
///     (2, 0, 1.0), (2, 1, 2.0),
///     (3, 0, 1.0), (3, 1, 3.0),
/// ]);
/// let b = Matrix::from_rows(&[[1.0], [3.0], [5.0], [6.0]]);
/// let qr = SparseQr::new(&a);
/// let x = qr.solve(&b).unwrap();
/// assert!((x[(0, 0)] - 1.2).abs() < 1e-14 && (x[(1, 0)] - 1.7).abs() < 1e-14);
///
/// // The residual of the least squares is orthogonal to the columns of the matrix.
/// let residual = &a * &x - &b;
/// assert!((&a.transpose() * &residual).norm(Norm::Max) < 1e-14);
///
/// // And the factors are those of the matrix with its columns reordered.
/// let (q, r) = (qr.q(), qr.r());
/// let permuted = Matrix::from_fn(4, 2, |i, j| a[(i, qr.column_permutation()[j])]);
/// assert!((&q * &r.to_matrix() - &permuted).norm(Norm::Max) < 1e-14);
/// assert!((&q.transpose() * &q - Matrix::identity(2)).norm(Norm::Max) < 1e-14);
/// ```
#[derive(Clone)]
pub struct SparseQr<T: Scalar> {
    /// The number of rows of the matrix.
    rows: usize,
    /// The number of columns of the matrix.
    cols: usize,
    /// The column permutation, `permutation[new] = old`.
    permutation: Vec<usize>,
    /// Where each row of `R` begins in `r_cols` and `r_values`, and where the last one ends.
    r_ptr: Vec<usize>,
    /// The columns of the stored elements of `R`, ascending within a row, the diagonal first.
    r_cols: Vec<usize>,
    /// The stored elements of `R`: the whole pattern, the elements that came out zero too.
    r_values: Vec<T>,
    /// Where the rows of each front begin in `coordinates`, and where the last ones end. The
    /// fronts are in the order they were factored, each after its children.
    row_ptr: Vec<usize>,
    /// The rows of the matrix each front works on, in the order of its staircase.
    coordinates: Vec<usize>,
    /// The first reflection of each front, and the number of all of them.
    reflection_ptr: Vec<usize>,
    /// Where the vector of each reflection begins in `vectors`, and where the last one ends.
    /// Reflection `t` of a front works on the rows `t..t + length` of the front.
    vector_ptr: Vec<usize>,
    /// The Householder vectors, each with its leading 1.
    vectors: Vec<T>,
    /// The factors `τ` of the reflections `I − τ·v·vᴴ`.
    taus: Vec<T>,
    /// The row of the matrix in whose place each row of `R` stands once every reflection has
    /// been applied: column `j` of `Q` is the image of that unit vector.
    pivot_row: Vec<usize>,
    /// The most rows of a front, the working memory a product with `Q` needs for each column.
    tallest: usize,
    /// The number of diagonal elements of `R` above the tolerance of [`rank`](Self::rank), one
    /// fewer when that is all of them but the condition estimate finds `R` singular.
    rank: usize,
    /// Whether every element of `R` is finite.
    finite: bool,
}

impl<T: Scalar> SparseQr<T> {
    /// The decomposition of `a` with its columns in the order [`Ordering::Colamd`] gives them.
    ///
    /// # Panics
    ///
    /// Panics if `a` has fewer rows than columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseMatrix, SparseQr};
    ///
    /// let entries = [(0, 0, 3.0f64), (1, 0, 4.0), (1, 1, 1.0), (2, 1, 2.0)];
    /// let a = SparseMatrix::from_coo(3, 2, &entries);
    /// let qr = SparseQr::new(&a);
    /// assert_eq!(qr.rank(), 2);
    /// // The first diagonal element of R is the norm of the first column, up to its sign.
    /// let first = qr.column_permutation()[0];
    /// let norm = if first == 0 { 5.0 } else { 5.0f64.sqrt() };
    /// assert!((qr.r()[(0, 0)].abs() - norm).abs() < 1e-14);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn new(a: &SparseMatrix<T>) -> Self {
        Self::with_ordering(a, &Ordering::Colamd)
    }

    /// The decomposition of `a` with its columns in the order `ordering` gives them:
    /// [`Ordering::Natural`] leaves them where they are, [`Ordering::Given`] puts column
    /// `q[k]` of `a` in place `k`, [`Ordering::Colamd`] works on the pattern of `a` and the
    /// symmetric orderings on that of `Aᴴ·A`. The decomposition is right for any order; what
    /// the order changes is how many elements `R` and the reflections have.
    ///
    /// # Panics
    ///
    /// Panics if `a` has fewer rows than columns, or if a given permutation isn't one of its
    /// columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseMatrix, SparseQr};
    ///
    /// // An arrow: the first column is full, the rest of the matrix is diagonal.
    /// let n = 6;
    /// let a = SparseMatrix::from_fn(n, n, |i, j| {
    ///     if j == 0 || i == j { 1.0 + i as f64 } else { 0.0 }
    /// });
    /// // Eliminated first, the full column fills R; eliminated last, it fills nothing.
    /// let first = SparseQr::with_ordering(&a, &Ordering::Natural);
    /// let last = SparseQr::with_ordering(&a, &Ordering::Given(vec![1, 2, 3, 4, 5, 0]));
    /// assert_eq!(first.r().nnz(), n * (n + 1) / 2);
    /// assert_eq!(last.r().nnz(), 2 * n - 1);
    /// assert_eq!(last.column_permutation(), [1, 2, 3, 4, 5, 0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn with_ordering(a: &SparseMatrix<T>, ordering: &Ordering) -> Self {
        assert!(
            a.rows() >= a.cols(),
            "the matrix has fewer rows than columns"
        );
        let permutation = ordering.columns(a);
        let symbolic = analyse(a, &permutation);
        factor(a, symbolic, permutation)
    }

    /// `R`, `n × n` and upper triangular, with a real diagonal of the signs described at
    /// [`SparseQr`]. The elements of its pattern that came out zero are not stored in the matrix
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseMatrix, SparseQr};
    ///
    /// let entries = [(0, 0, 3.0f64), (1, 0, 4.0), (1, 1, 5.0), (2, 1, 12.0)];
    /// let a = SparseMatrix::from_coo(3, 2, &entries);
    /// let r = SparseQr::with_ordering(&a, &Ordering::Natural).r();
    /// assert_eq!((r.rows(), r.cols()), (2, 2));
    /// assert_eq!(r[(1, 0)], 0.0);
    /// // Rᴴ·R is Aᴴ·A: the first column has norm 5, and its product with the second is 20.
    /// assert!((r[(0, 0)].abs() - 5.0).abs() < 1e-14);
    /// assert!((r[(0, 0)] * r[(0, 1)] - 20.0).abs() < 1e-13);
    /// ```
    #[must_use]
    pub fn r(&self) -> SparseMatrix<T> {
        SparseMatrix::from_csr(
            self.cols,
            self.cols,
            self.r_ptr.clone(),
            self.r_cols.clone(),
            self.r_values.clone(),
        )
    }

    /// `Q`, the thin `m × n` factor with orthonormal columns, as a dense matrix: for small
    /// matrices and for tests. It costs a product with `Q` for each column;
    /// [`apply_q`](Self::apply_q) and [`apply_qh`](Self::apply_qh) do without it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::{Matrix, Norm};
    /// use a::math::sparse::{SparseMatrix, SparseQr};
    ///
    /// let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0), (1, 0, 1.0), (1, 1, 1.0), (2, 1, 1.0)]);
    /// let q = SparseQr::new(&a).q();
    /// assert_eq!((q.rows(), q.cols()), (3, 2));
    /// assert!((&q.transpose() * &q - Matrix::identity(2)).norm(Norm::Max) < 1e-15);
    /// ```
    #[must_use]
    pub fn q(&self) -> Matrix<T> {
        self.apply_q(&Matrix::identity(self.cols))
    }

    /// `Q·x` for `x` of `n` rows, a matrix of `m` rows, without forming `Q`: each column costs
    /// the stored elements of the reflections.
    ///
    /// # Panics
    ///
    /// Panics if `x` hasn't as many rows as the matrix has columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::{Matrix, Norm};
    /// use a::math::sparse::{SparseMatrix, SparseQr};
    ///
    /// let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0), (1, 0, 2.0), (1, 1, 1.0), (2, 1, 3.0)]);
    /// let qr = SparseQr::new(&a);
    /// let x = Matrix::from_rows(&[[1.0], [-2.0]]);
    /// assert!((qr.apply_q(&x) - &qr.q() * &x).norm(Norm::Max) < 1e-15);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn apply_q(&self, x: &Matrix<T>) -> Matrix<T> {
        assert_eq!(
            x.rows(),
            self.cols,
            "the matrix hasn't a row for each column of Q"
        );
        let (m, count) = (self.rows, x.cols());
        let mut y = Matrix::zeros(m, count);
        for c in 0..count {
            let column = y.column_mut(c);
            for (&row, &element) in self.pivot_row.iter().zip(x.column(c)) {
                column[row] = element;
            }
        }
        self.reflect(false, y.as_mut_slice(), count);
        y
    }

    /// `Qᴴ·b` for `b` of `m` rows, a matrix of `n` rows, without forming `Q`: each column costs
    /// the stored elements of the reflections.
    ///
    /// # Panics
    ///
    /// Panics if `b` hasn't as many rows as the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::{Matrix, Norm};
    /// use a::math::sparse::{SparseMatrix, SparseQr};
    ///
    /// let a = SparseMatrix::from_coo(3, 2, &[(0, 0, 1.0), (1, 0, 2.0), (1, 1, 1.0), (2, 1, 3.0)]);
    /// let qr = SparseQr::new(&a);
    /// let b = Matrix::from_rows(&[[1.0], [0.0], [-1.0]]);
    /// assert!((qr.apply_qh(&b) - &qr.q().transpose() * &b).norm(Norm::Max) < 1e-15);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn apply_qh(&self, b: &Matrix<T>) -> Matrix<T> {
        assert_eq!(
            b.rows(),
            self.rows,
            "the matrix hasn't a row for each row of Q"
        );
        let (m, count) = (self.rows, b.cols());
        let mut y = b.clone();
        self.reflect(true, y.as_mut_slice(), count);
        let y = y.as_slice();
        Matrix::from_fn(self.cols, count, |j, c| y[self.pivot_row[j] + c * m])
    }

    /// The column permutation `q` of `A·Q_c = Q·R`, `q[new] = old`: column `k` of `A·Q_c` is
    /// column `q[k]` of `A`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseMatrix, SparseQr};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 1, 3.0)]);
    /// let qr = SparseQr::with_ordering(&a, &Ordering::Given(vec![1, 0]));
    /// assert_eq!(qr.column_permutation(), [1, 0]);
    /// ```
    #[must_use]
    pub fn column_permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// The least-squares solution: the `x` that makes `‖A·x − b‖₂` the smallest, for each
    /// column of `b`, which for a square matrix is the solution of `A·x = b`. It is
    /// `Q_c·R⁻¹·Qᴴ·b`: the reflections applied to `b`, a back substitution, and the columns put
    /// back in their order.
    ///
    /// `None` when `R` is singular to working precision, that is when [`rank`](Self::rank) is
    /// below the number of columns: when a diagonal element of `R` is below the tolerance of
    /// `rank`, or when a lower bound of the condition number of `R` passes `1/(max(m, 32)·ε)`.
    /// The matrix is then rank deficient, or too near it for the solution to mean anything, and
    /// has no single least-squares solution. The condition number of `R` is that of the matrix,
    /// and the bound is LINPACK's estimate followed by five steps of the power method on
    /// `(Rᴴ·R)⁻¹`, six solves with `R` and `Rᴴ` that cost the elements of `R` each: it can fall
    /// short of the condition number, but not pass it, so a matrix whose condition number is
    /// below the limit is not refused, up to the rounding errors of the decomposition. Of
    /// thousands of random matrices around the limit, of every element type, none below it was
    /// refused and none beyond twice it was solved. `None`
    /// also when the matrix held a NaN or an infinity, or overflowed, and when an element of the
    /// solution isn't finite: `b` held a NaN or an infinity, or the solution is beyond the range
    /// of the type.
    ///
    /// # Panics
    ///
    /// Panics if `b` hasn't as many rows as the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::{Ordering, SparseMatrix, SparseQr};
    ///
    /// let entries = [(0, 0, 1.0f64), (1, 1, 1.0), (2, 0, 1.0), (2, 1, 1.0)];
    /// let a = SparseMatrix::from_coo(3, 2, &entries);
    /// let b = Matrix::from_rows(&[[1.0, 0.0], [2.0, 0.0], [3.0, 3.0]]);
    /// let x = SparseQr::new(&a).solve(&b).unwrap();
    /// // The first right-hand side is A·(1, 2), the second isn't in the range of A.
    /// assert!((x[(0, 0)] - 1.0).abs() < 1e-14 && (x[(1, 0)] - 2.0).abs() < 1e-14);
    /// assert!((x[(0, 1)] - 1.0).abs() < 1e-14 && (x[(1, 1)] - 1.0).abs() < 1e-14);
    ///
    /// // Two equal columns: no single solution.
    /// let entries = [(0, 0, 1.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 2.0)];
    /// let twice = SparseMatrix::from_coo(2, 2, &entries);
    /// assert!(SparseQr::new(&twice).solve(&Matrix::from_rows(&[[1.0], [2.0]])).is_none());
    ///
    /// // A triangle of ones on the diagonal and −2 above it, its own R in the natural order: no
    /// // diagonal element is small, but the condition number is about 2⁶¹, and the solution
    /// // would mean nothing.
    /// let n = 60;
    /// let a = SparseMatrix::from_fn(n, n, |i, j| {
    ///     if i == j { 1.0 } else if j == i + 1 { -2.0 } else { 0.0 }
    /// });
    /// let qr = SparseQr::with_ordering(&a, &Ordering::Natural);
    /// assert!(qr.solve_vec(&vec![1.0; n]).is_none());
    /// assert_eq!(qr.rank(), n - 1);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve(&self, b: &Matrix<T>) -> Option<Matrix<T>> {
        assert_eq!(
            b.rows(),
            self.rows,
            "the right-hand side has the wrong number of rows"
        );
        if !self.finite || self.rank < self.cols {
            return None;
        }
        let n = self.cols;
        let mut z = self.apply_qh(b);
        let mut x = Matrix::zeros(n, b.cols());
        for c in 0..b.cols() {
            let z = z.column_mut(c);
            // `R·z = Qᴴ·b` from the last row up: a row of `R` begins with its diagonal, which
            // is real.
            for j in (0..n).rev() {
                let (begin, end) = (self.r_ptr[j], self.r_ptr[j + 1]);
                let mut sum = z[j];
                for (&k, &r) in self.r_cols[begin + 1..end]
                    .iter()
                    .zip(&self.r_values[begin + 1..end])
                {
                    sum -= r * z[k];
                }
                let d = self.r_values[begin].re();
                z[j] = T::from_parts(sum.re() / d, sum.im() / d);
            }
            let x = x.column_mut(c);
            for (&old, &element) in self.permutation.iter().zip(z.iter()) {
                x[old] = element;
            }
        }
        // A solution beyond the range of the type, or from a right-hand side that held a NaN or
        // an infinity, isn't one.
        x.as_slice().iter().all(|x| x.is_finite()).then_some(x)
    }

    /// The least-squares solution for one right-hand side, as [`solve`](Self::solve) gives it.
    ///
    /// # Panics
    ///
    /// Panics if `b` isn't as long as the matrix has rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseMatrix, SparseQr};
    ///
    /// // The mean of three measurements of one unknown.
    /// let a = SparseMatrix::from_coo(3, 1, &[(0, 0, 1.0f64), (1, 0, 1.0), (2, 0, 1.0)]);
    /// let x = SparseQr::new(&a).solve_vec(&[1.0, 2.0, 6.0]).unwrap();
    /// assert!((x[0] - 3.0).abs() < 1e-15);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve_vec(&self, b: &[T]) -> Option<Vec<T>> {
        assert_eq!(
            b.len(),
            self.rows,
            "the right-hand side has the wrong length"
        );
        let b = Matrix::from_column_major(self.rows, 1, b.to_vec());
        self.solve(&b).map(Matrix::into_vec)
    }

    /// An estimate of the rank of the matrix: the number of diagonal elements of `R` larger in
    /// absolute value than `max(m, 32)·ε·c`, `c` the largest 2-norm of a column of the matrix,
    /// which no diagonal element of `R` exceeds; one fewer when that is every one of them but
    /// the condition estimate of [`solve`](Self::solve) finds `R` singular to working precision
    /// all the same. So the count is below the number of columns exactly when `solve` refuses a
    /// matrix of finite elements.
    ///
    /// The tolerance is NumPy's `matrix_rank`, which counts the singular values above
    /// `max(m, n)·ε·σ₁`, with the diagonal of `R` in place of the singular values and `c`, which
    /// lies between `σ₁/√n` and `σ₁`, in place of `σ₁`. A diagonal element below it says that
    /// the matrix is rank deficient to working precision, since the smallest singular value of a
    /// triangular matrix is no larger than its smallest diagonal element, and the condition
    /// estimate never passes the condition number: no matrix whose condition number is below
    /// `1/(max(m, 32)·ε)` is counted short, up to the rounding errors of the decomposition.
    ///
    /// The count can be above the rank. A column that depends exactly on others leaves a
    /// rounding error on the diagonal, which for a copy, a small multiple or a sum of other
    /// columns stays below the tolerance (up to about `7·ε·c` in millions of small random
    /// matrices, `0.07·m·ε·c` at thousands of rows), but which large coefficients in the
    /// dependency amplify: with a column `2ᵖ` times the difference of two others, every diagonal
    /// element passed the tolerance in up to a third of 10 000 random matrices of each type, the
    /// smallest by up to about `2ᵖ/10` times (30 times for `2⁸`, 210 times for `2¹¹`). Such a
    /// matrix is refused by the condition estimate and counted one short of full, but two or more
    /// columns so dependent are counted as one.
    ///
    /// The count can be below the rank. Without pivoting for rank, a dependent column eliminated
    /// before an independent one still takes a row of `R`, as in the dense decomposition, and the
    /// independent column can find too little left in the rows below it. Over 2000 random
    /// matrices of each element type with dependencies of small coefficients, the count was the
    /// rank whenever the dependent columns came after the others; under random orders, it was the
    /// rank in every matrix of five or twenty times as many rows as columns, in 99 to 99.8 % of
    /// those of twice as many and in 83 to 88 % of the square ones, which it missed by up to 3.
    /// An empty column takes no row. A column whose own norm is below the tolerance counts as
    /// dependent whatever its direction, as it would in the singular values. Where the rank
    /// matters, the singular values of the dense matrix decide it. For a matrix that holds a NaN
    /// or an infinity the count means nothing.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseMatrix, SparseQr};
    ///
    /// // The third column is the sum of the first two, and the fourth is empty.
    /// let a = SparseMatrix::from_coo(4, 4, &[
    ///     (0, 0, 1.0), (0, 2, 1.0),
    ///     (1, 1, 1.0), (1, 2, 1.0),
    ///     (2, 0, 1.0), (2, 1, 1.0), (2, 2, 2.0),
    ///     (3, 0, 1.0), (3, 2, 1.0),
    /// ]);
    /// assert_eq!(SparseQr::new(&a).rank(), 2);
    ///
    /// // The third column is 2⁸ times the second minus the first. In the natural order the
    /// // rounding error it leaves on the diagonal is above the tolerance, 32·ε times the largest
    /// // column norm √14, and the condition estimate finds the dependency.
    /// let a = SparseMatrix::from_coo(3, 3, &[
    ///     (0, 0, -1.0f64), (0, 1, -1.0 + 2.0 / 256.0), (0, 2, 2.0),
    ///     (1, 0, -3.0), (1, 1, -3.0 + 3.0 / 256.0), (1, 2, 3.0),
    ///     (2, 0, 2.0), (2, 1, 2.0 + 1.0 / 256.0), (2, 2, 1.0),
    /// ]);
    /// let qr = SparseQr::with_ordering(&a, &Ordering::Natural);
    /// let (r, tolerance) = (qr.r(), 32.0 * f64::EPSILON * 14f64.sqrt());
    /// assert!((0..3).all(|j| r[(j, j)].abs() > tolerance));
    /// assert_eq!(qr.rank(), 2);
    /// assert!(qr.solve_vec(&[1.0, 2.0, 3.0]).is_none());
    /// ```
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The number of elements the factors store: those of the pattern of `R`, and those of the
    /// Householder vectors that stand for `Q`, each counted with its leading 1. This is what a
    /// column ordering is judged on. An element of a pattern that came out zero counts, so the
    /// part of `R` can be above the count of [`r`](Self::r)'s nonzeros; and the patterns are
    /// those of the fronts after small ones were merged into their parents, which adds explicit
    /// zeros: a merged child's rows of `R` take all its parent's columns, and its rows are
    /// reflected with the parent's pivot columns rather than reduced to its own triangle first,
    /// which lengthens the parent's vectors (see the module's documentation).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseMatrix, SparseQr};
    ///
    /// // A full 4 × 2 matrix: 3 elements of R, and vectors of 4 and of 3 elements.
    /// let a = SparseMatrix::from_fn(4, 2, |i, j| 1.0 + (i * (j + 1)) as f64);
    /// assert_eq!(SparseQr::with_ordering(&a, &Ordering::Natural).nnz(), 3 + 4 + 3);
    /// ```
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.r_values.len() + self.vectors.len()
    }

    /// `y ← Qᴴ·y` when `adjoint` is set, `y ← Q·y` otherwise, for the whole `m × m` product of
    /// the reflections and the `m × count` block `y`: the fronts in the order they were
    /// factored for `Qᴴ`, and in the opposite order for `Q`.
    fn reflect(&self, adjoint: bool, y: &mut [T], count: usize) {
        let fronts = self.row_ptr.len() - 1;
        let mut block = vec![T::zero(); self.tallest * count];
        if adjoint {
            for f in 0..fronts {
                self.reflect_front(f, true, y, count, &mut block);
            }
        } else {
            for f in (0..fronts).rev() {
                self.reflect_front(f, false, y, count, &mut block);
            }
        }
    }

    /// Applies the reflections of front `f` to the block `y`, or their conjugate transposes in
    /// the opposite order when `adjoint` is set: the rows of the front are gathered into the
    /// dense `block`, reflected there, and put back.
    fn reflect_front(&self, f: usize, adjoint: bool, y: &mut [T], count: usize, block: &mut [T]) {
        let (first, last) = (self.reflection_ptr[f], self.reflection_ptr[f + 1]);
        if first == last || count == 0 {
            return;
        }
        let rows = &self.coordinates[self.row_ptr[f]..self.row_ptr[f + 1]];
        let (m, height) = (self.rows, rows.len());
        let block = &mut block[..height * count];
        for (gathered, column) in block.chunks_exact_mut(height).zip(y.chunks_exact(m)) {
            for (element, &row) in gathered.iter_mut().zip(rows) {
                *element = column[row];
            }
        }
        let mut one = |g: usize| {
            let vector = &self.vectors[self.vector_ptr[g]..self.vector_ptr[g + 1]];
            let t = g - first;
            reflect_left(
                self.taus[g],
                adjoint,
                vector,
                vector.len(),
                count,
                &mut block[t..],
                height,
            );
        };
        if adjoint {
            (first..last).for_each(&mut one);
        } else {
            (first..last).rev().for_each(&mut one);
        }
        for (gathered, column) in block.chunks_exact(height).zip(y.chunks_exact_mut(m)) {
            for (&element, &row) in gathered.iter().zip(rows) {
                column[row] = element;
            }
        }
    }
}

impl<T: Scalar> fmt::Debug for SparseQr<T> {
    /// Writes the shape and the sizes of the factors rather than their elements.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseQr")
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("rank", &self.rank)
            .field("r_elements", &self.r_values.len())
            .field("reflections", &self.taus.len())
            .field("vector_elements", &self.vectors.len())
            .finish()
    }
}

/// The fewest rows the tolerance of [`SparseQr::rank`] is reckoned for. A column that is a copy,
/// a small multiple or a sum of earlier columns leaves up to about `7·ε·c` on the diagonal of a
/// small matrix, `c` the largest column norm (the worst of 2.6 million random matrices of each
/// element type from `2 × 2` to `32 × 32`), more than `m·ε·c` for `m` below 8. A combination of
/// large coefficients can leave much more, which is what the condition estimate is for.
const RANK_FLOOR: usize = 32;

/// `max(m, 32)·ε`, the tolerance of [`SparseQr::rank`] relative to `c`, and the reciprocal of
/// the largest condition number [`SparseQr::solve`] accepts.
fn relative_tolerance<T: Scalar>(m: usize) -> T::Real {
    T::Real::from_f64(m.max(RANK_FLOOR) as f64) * T::Real::EPSILON
}

/// `c`, the largest 2-norm of a column of `a`, which no diagonal element of `R` exceeds, since
/// `|R_jj| ≤ ‖R·e_j‖ = ‖A·Q_c·e_j‖`, and which is no more than `‖R‖₂`. Unlike the largest
/// diagonal element, `c` keeps the tolerance of the rank above the rounding error a column
/// leaves that is a large multiple of an earlier one.
///
/// The squares are summed in units of the largest part of an element, which no column norm is
/// below, so that they cannot overflow and only those too small to matter can underflow. An
/// infinite element makes the norm infinite; a NaN is passed over, its column's sum being a NaN,
/// which `max` ignores.
fn largest_column_norm<T: Scalar>(a: &SparseMatrix<T>) -> T::Real {
    let mut scale = T::Real::ZERO;
    for x in a.values() {
        scale = scale.max(x.re().abs()).max(x.im().abs());
    }
    if scale == T::Real::ZERO || !scale.is_finite() {
        return scale;
    }
    let mut sums = vec![T::Real::ZERO; a.cols()];
    let unit = T::Real::ONE / scale;
    for (&j, &x) in a.col_indices().iter().zip(a.values()) {
        sums[j] += if unit.is_finite() {
            x.scale(unit).abs2()
        } else {
            // A subnormal scale, whose reciprocal overflows: a division for each part.
            let (re, im) = (x.re() / scale, x.im() / scale);
            re * re + im * im
        };
    }
    let largest = sums.into_iter().fold(T::Real::ZERO, T::Real::max);
    largest.sqrt() * scale
}

/// The numeric phase: the fronts of `symbolic` assembled from `a` and from their children,
/// reduced, and taken apart into rows of `R`, Householder vectors and contribution blocks.
fn factor<T: Scalar>(
    a: &SparseMatrix<T>,
    symbolic: Symbolic,
    permutation: Vec<usize>,
) -> SparseQr<T> {
    let s = symbolic;
    let (m, n) = (a.rows(), a.cols());
    let (pointers, indices, elements) = (a.row_pointers(), a.col_indices(), a.values());
    let fronts = s.pivots.len();
    let reflections = s.reflection_ptr[fronts];
    let mut r_values = vec![T::zero(); s.r_cols.len()];
    let mut vectors = vec![T::zero(); s.vector_ptr[reflections]];
    let mut taus = vec![T::zero(); reflections];
    // The front being reduced, each column as long as its step of the staircase or, for a
    // front held dense, as the front is tall; and the stack of the contribution blocks not yet
    // taken in.
    let mut work = vec![T::zero(); s.work];
    let mut stack = vec![T::zero(); s.stack];
    // Where each column of the front begins in `work`, and where its step of the staircase
    // ends: the rows above it are those it holds.
    let mut start = vec![0usize; s.widest + 1];
    let mut end = vec![0usize; s.widest];
    // The vectors of a block of reflections.
    let mut scratch: Vec<T> = Vec::new();
    // The place in the front of each of its columns and of each of its rows.
    let mut local = vec![0usize; n];
    let mut slot = vec![0usize; m];
    for f in 0..fronts {
        let cols = &s.columns[s.column_ptr[f]..s.column_ptr[f + 1]];
        let rows = &s.coordinates[s.row_ptr[f]..s.row_ptr[f + 1]];
        let (width, height, pivots) = (cols.len(), rows.len(), s.pivots[f]);
        if height == 0 {
            continue;
        }
        let first = s.reflection_ptr[f];
        let count = s.reflection_ptr[f + 1] - first;
        // Column `t` holds the rows above its step: those its reflection works on and the
        // finished ones above them, and all the rows of a front that has run out of reflections.
        // Held by its staircase, the front has each column as long as its step.
        for t in 0..width {
            let end = if t < count {
                t + s.vector_ptr[first + t + 1] - s.vector_ptr[first + t]
            } else {
                height
            };
            start[t + 1] = start[t] + end;
        }
        // A front held dense has its columns `height` apart instead, the elements below each
        // step unused; and a front reduced by panels is told where the steps end.
        let dense = front::dense(height, width, count, start[width]);
        // No front held dense is small: it has a hundred reflections or more.
        let small = width <= front::SMALL;
        debug_assert!(!(small && dense), "a small front is held dense");
        if dense || !small {
            for t in 0..width {
                end[t] = start[t + 1] - start[t];
            }
        }
        let end = &end[..width];
        if dense {
            for t in 0..width {
                start[t + 1] = (t + 1) * height;
            }
        }
        // The rows above its step of column `l`.
        let step = |l: usize| {
            if dense {
                end[l]
            } else {
                start[l + 1] - start[l]
            }
        };
        let front = &mut work[..start[width]];
        if dense {
            for (t, &e) in end.iter().enumerate() {
                front[start[t]..start[t] + e].fill(T::zero());
            }
        } else {
            front.fill(T::zero());
        }
        for (l, &col) in cols.iter().enumerate() {
            local[col] = l;
        }
        for (at, &row) in rows.iter().enumerate() {
            slot[row] = at;
        }

        // The rows of the matrix that begin in this front.
        for (at, &row) in rows.iter().enumerate() {
            if s.entering[row] != f {
                continue;
            }
            for p in pointers[row]..pointers[row + 1] {
                let l = local[s.position[indices[p]]];
                debug_assert!(at < step(l), "a row is below its staircase");
                front[start[l] + at] = elements[p];
            }
        }
        // The contribution blocks of the children, each an upper triangle kept by columns.
        for &child in &s.children[s.child_ptr[f]..s.child_ptr[f + 1]] {
            let taken = s.pivots[child];
            let made = s.reflection_ptr[child + 1] - s.reflection_ptr[child];
            if made <= taken {
                continue;
            }
            let passed_cols = &s.columns[s.column_ptr[child] + taken..s.column_ptr[child + 1]];
            let passed_rows = &s.coordinates[s.row_ptr[child] + taken..s.row_ptr[child] + made];
            let mut block = &stack[s.block_ptr[child]..];
            for (v, &col) in passed_cols.iter().enumerate() {
                let l = local[col];
                let (column, rest) = block.split_at((v + 1).min(passed_rows.len()));
                for (&element, &row) in column.iter().zip(passed_rows) {
                    debug_assert!(slot[row] < step(l), "a row is below its staircase");
                    front[start[l] + slot[row]] = element;
                }
                block = rest;
            }
        }

        // The reflections: that of column `t` takes the column's elements from row `t` down to
        // its step to `(β, 0, …)` and is applied to the columns after it, which are all at
        // least as long.
        if small {
            // A front of a few columns: each reflection applied to the columns after it one at a
            // time, as they come, which spares it the panels' bookkeeping. The fronts of a curve
            // fit or a band are mostly such.
            for t in 0..count {
                let g = first + t;
                let vector = &mut vectors[s.vector_ptr[g]..s.vector_ptr[g + 1]];
                let length = vector.len();
                let (reduced, rest) = front.split_at_mut(start[t + 1]);
                let column = &mut reduced[start[t] + t..start[t] + t + length];
                let (tau, beta) = reflector(column[0], &mut column[1..]);
                vector[0] = T::one();
                vector[1..].copy_from_slice(&column[1..]);
                column[0] = beta;
                taus[g] = tau;
                if tau == T::zero() {
                    continue;
                }
                for u in t + 1..width {
                    let at = start[u] - start[t + 1] + t;
                    reflect_left(
                        tau,
                        true,
                        vector,
                        length,
                        1,
                        &mut rest[at..at + length],
                        length,
                    );
                }
            }
        } else {
            front::reduce(
                front,
                &start[..=width],
                end,
                dense.then_some(height),
                count,
                &mut vectors,
                &s.vector_ptr[first..=first + count],
                &mut taus[first..first + count],
                &mut scratch,
            );
        }

        // The first rows of the triangle are the rows of `R` of the pivot columns.
        for (t, &col) in cols[..pivots.min(count)].iter().enumerate() {
            let row = &mut r_values[s.r_ptr[col]..s.r_ptr[col + 1]];
            for (element, u) in row.iter_mut().zip(t..width) {
                *element = front[start[u] + t];
            }
        }
        // The rest of it is the contribution block, which takes the place of the children's.
        if count > pivots {
            let passed = count - pivots;
            let mut block = &mut stack[s.block_ptr[f]..];
            for v in 0..width - pivots {
                let (column, rest) = block.split_at_mut((v + 1).min(passed));
                let from = start[pivots + v] + pivots;
                column.copy_from_slice(&front[from..from + column.len()]);
                block = rest;
            }
        }
    }

    // The diagonal of `R` begins its rows; a row of no elements has a zero there.
    let diagonal = |j: usize| {
        if s.r_ptr[j] < s.r_ptr[j + 1] {
            r_values[s.r_ptr[j]].abs()
        } else {
            T::Real::ZERO
        }
    };
    let c = largest_column_norm(a);
    let relative = relative_tolerance::<T>(m);
    let tolerance = relative * c;
    // A comparison with a NaN is false, so a diagonal element that isn't a number isn't counted.
    let mut rank = (0..n).filter(|&j| diagonal(j) > tolerance).count();
    let finite = r_values.iter().all(|x| x.is_finite());
    // No diagonal element is small, which is not enough: see the module's documentation. Every
    // one is above the tolerance, so `c` is positive, and with `R` finite it is finite.
    if finite
        && rank == n
        && n > 0
        && singular_to_working_precision(&s.r_ptr, &s.r_cols, &r_values, c, T::Real::ONE / relative)
    {
        rank = n - 1;
    }
    SparseQr {
        rows: m,
        cols: n,
        permutation,
        r_ptr: s.r_ptr,
        r_cols: s.r_cols,
        r_values,
        row_ptr: s.row_ptr,
        coordinates: s.coordinates,
        reflection_ptr: s.reflection_ptr,
        vector_ptr: s.vector_ptr,
        vectors,
        taus,
        pivot_row: s.pivot_row,
        tallest: s.tallest,
        rank,
        finite,
    }
}
