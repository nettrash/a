//! Linear operators: what an iterative method needs to know of a matrix.

use super::SparseMatrix;
use crate::math::matrix::{Matrix, Scalar};

/// A linear map from vectors of length [`cols`](Self::cols) to vectors of length
/// [`rows`](Self::rows), known through its action alone.
///
/// The Krylov solvers and the eigenvalue methods ask nothing else of a matrix, so they run on
/// anything that implements this: a [`SparseMatrix`], a dense [`Matrix`], or a function wrapped
/// in an [`FnOperator`] — a matrix that is never formed, such as the inverse a factorisation
/// applies.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{FnOperator, LinearOperator, SparseMatrix};
///
/// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (1, 1, 3.0)]);
/// let mut y = [0.0; 2];
/// a.apply(&[1.0, 1.0], &mut y);
/// assert_eq!(y, [2.0, 3.0]);
///
/// // The same map, given as a function.
/// let f = FnOperator::new(2, 2, |x: &[f64], y: &mut [f64]| {
///     y[0] = 2.0 * x[0];
///     y[1] = 3.0 * x[1];
/// });
/// f.apply(&[1.0, 1.0], &mut y);
/// assert_eq!(y, [2.0, 3.0]);
/// ```
pub trait LinearOperator<T: Scalar> {
    /// The length of the vectors the operator gives.
    fn rows(&self) -> usize;

    /// The length of the vectors the operator takes.
    fn cols(&self) -> usize;

    /// Writes the image of `x` into `y`.
    ///
    /// # Panics
    ///
    /// Panics if `x` isn't of length `cols` or `y` of length `rows`.
    fn apply(&self, x: &[T], y: &mut [T]);

    /// Writes the residual `b − A·x` into `r`, for `x` of length [`cols`](Self::cols) and `b`
    /// and `r` of length [`rows`](Self::rows).
    ///
    /// This is the vector the iterative solvers decide convergence on. The provided method
    /// applies the operator and subtracts, so it carries the rounding error of the product, of
    /// the order of `ε·|A|·|x|`, which is no smaller for a residual that is. [`SparseMatrix`]
    /// and [`Matrix`] instead accumulate each element in twice the working precision, which
    /// makes theirs the exact residual, rounded once.
    ///
    /// # Panics
    ///
    /// Panics if `x` isn't of length `cols`, or `b` or `r` of length `rows`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{LinearOperator, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (1, 1, 3.0)]);
    /// let mut r = [0.0; 2];
    /// a.residual(&[1.0, 1.0], &[5.0, 5.0], &mut r);
    /// assert_eq!(r, [3.0, 2.0]);
    /// ```
    #[track_caller]
    fn residual(&self, x: &[T], b: &[T], r: &mut [T]) {
        assert_eq!(
            b.len(),
            r.len(),
            "the right-hand side and the result differ in length"
        );
        self.apply(x, r);
        for (r, &b) in r.iter_mut().zip(b) {
            *r = b - *r;
        }
    }
}

impl<T: Scalar> LinearOperator<T> for SparseMatrix<T> {
    #[inline]
    fn rows(&self) -> usize {
        SparseMatrix::rows(self)
    }

    #[inline]
    fn cols(&self) -> usize {
        SparseMatrix::cols(self)
    }

    #[inline]
    #[track_caller]
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.mul_vec_into(x, y);
    }

    #[track_caller]
    fn residual(&self, x: &[T], b: &[T], r: &mut [T]) {
        super::solve::residual_sparse(self, x, b, r);
    }
}

impl<T: Scalar> LinearOperator<T> for Matrix<T> {
    #[inline]
    fn rows(&self) -> usize {
        Matrix::rows(self)
    }

    #[inline]
    fn cols(&self) -> usize {
        Matrix::cols(self)
    }

    #[track_caller]
    fn apply(&self, x: &[T], y: &mut [T]) {
        assert_eq!(
            x.len(),
            Matrix::cols(self),
            "the vector has the wrong length for the matrix"
        );
        assert_eq!(
            y.len(),
            Matrix::rows(self),
            "the result has the wrong length for the matrix"
        );
        y.fill(T::zero());
        // Column after column, which is how the matrix is held.
        for (j, &xj) in x.iter().enumerate() {
            for (y, &a) in y.iter_mut().zip(self.column(j)) {
                *y += a * xj;
            }
        }
    }

    #[track_caller]
    fn residual(&self, x: &[T], b: &[T], r: &mut [T]) {
        super::solve::residual_dense(self, x, b, r);
    }
}

/// A [`LinearOperator`] given by a function `f(x, y)` that writes the image of `x` into `y`.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{FnOperator, LinearOperator};
///
/// // The cyclic shift of three elements.
/// let shift = FnOperator::new(3, 3, |x: &[f64], y: &mut [f64]| {
///     y[0] = x[2];
///     y[1] = x[0];
///     y[2] = x[1];
/// });
/// let mut y = [0.0; 3];
/// shift.apply(&[1.0, 2.0, 3.0], &mut y);
/// assert_eq!(y, [3.0, 1.0, 2.0]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FnOperator<F> {
    /// The length of the vectors the function gives.
    rows: usize,
    /// The length of the vectors the function takes.
    cols: usize,
    /// The function.
    f: F,
}

impl<F> FnOperator<F> {
    /// The operator from vectors of length `cols` to vectors of length `rows` that `f` applies.
    #[must_use]
    pub fn new(rows: usize, cols: usize, f: F) -> Self {
        Self { rows, cols, f }
    }
}

impl<T: Scalar, F: Fn(&[T], &mut [T])> LinearOperator<T> for FnOperator<F> {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }

    #[track_caller]
    fn apply(&self, x: &[T], y: &mut [T]) {
        assert_eq!(
            x.len(),
            self.cols,
            "the vector has the wrong length for the operator"
        );
        assert_eq!(
            y.len(),
            self.rows,
            "the result has the wrong length for the operator"
        );
        (self.f)(x, y);
    }
}
