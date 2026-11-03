//! Eigenvalues and eigenvectors: of any square matrix from its Schur form, and of a Hermitian
//! one by a reduction to a real tridiagonal matrix and divide and conquer.

use super::hermitian::hermitian;
use super::schur::{eigenvectors, schur};
use super::{Matrix, Scalar};
use crate::math::complex::{Complex, Float};

/// The eigendecomposition `A·V = V·Λ` of a square matrix: its eigenvalues, the diagonal of `Λ`,
/// and its eigenvectors, the columns of `V`, complex in general, each of length 1. When `A` has
/// a full set of independent eigenvectors, `A = V·Λ·V⁻¹`. From [`Matrix::eigen`].
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::matrix::Matrix;
///
/// let a: Matrix<f64> = Matrix::from_rows(&[[2.0, 1.0], [1.0, 2.0]]);
/// let e = a.eigen();
/// let (v, w) = (e.vectors(), e.values());
/// // Each column is an eigenvector: A·v = λ·v.
/// let a = a.map(|&x| Complex::from(x));
/// for (j, &w) in w.iter().enumerate() {
///     let v = Matrix::from_columns(&[v.column(j)]);
///     let d = &(&a * &v) - &(&v * w);
///     assert!(d.as_slice().iter().all(|x| x.abs() < 1e-14));
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Eigen<R> {
    /// The eigenvalues.
    values: Vec<Complex<R>>,
    /// The eigenvectors, as columns.
    vectors: Matrix<Complex<R>>,
}

impl<R: Float> Eigen<R> {
    /// The eigenvalues, in the order of the columns of [`vectors`](Eigen::vectors).
    pub fn values(&self) -> &[Complex<R>] {
        &self.values
    }

    /// The eigenvectors, each a column of length 1 in the 2-norm.
    pub fn vectors(&self) -> &Matrix<Complex<R>> {
        &self.vectors
    }
}

/// The eigendecomposition `A = V·Λ·V*` of a Hermitian (for real elements, symmetric) matrix:
/// real eigenvalues in ascending order, and orthonormal eigenvectors, the columns of `V`, of the
/// matrix's own type. From [`Matrix::hermitian_eigen`].
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
///
/// let a: Matrix<f64> = Matrix::from_rows(&[[2.0, 1.0], [1.0, 2.0]]);
/// let e = a.hermitian_eigen();
/// assert!((e.values()[0] - 1.0).abs() < 1e-15 && (e.values()[1] - 3.0).abs() < 1e-15);
/// let v = e.vectors();
/// let back = &(v * &Matrix::from_diagonal(e.values())) * &v.transpose();
/// assert!((&back - &a).as_slice().iter().all(|x| x.abs() < 1e-14));
/// ```
#[derive(Clone, Debug)]
pub struct HermitianEigen<T: Scalar> {
    /// The eigenvalues, ascending.
    values: Vec<T::Real>,
    /// The eigenvectors, as columns.
    vectors: Matrix<T>,
}

impl<T: Scalar> HermitianEigen<T> {
    /// The eigenvalues, real, in ascending order.
    pub fn values(&self) -> &[T::Real] {
        &self.values
    }

    /// The eigenvectors, orthonormal columns in the order of the eigenvalues.
    pub fn vectors(&self) -> &Matrix<T> {
        &self.vectors
    }
}

impl<T: Scalar> Matrix<T> {
    /// The eigenvalues and eigenvectors: see [`Eigen`]. From the Schur form `A = Q·T·Q*`, the
    /// eigenvectors of `T` by back substitution, in real arithmetic for real elements, and `Q`
    /// times them. A matrix with a NaN or an infinity gives NaNs.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn eigen(&self) -> Eigen<T::Real> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        let (t, q, values) = schur(self, true, true);
        let vectors = eigenvectors(&t.data, &q.unwrap_or_default().data, n);
        Eigen {
            values,
            vectors: Matrix::from_column_major(n, n, vectors),
        }
    }

    /// The eigenvalues and orthonormal eigenvectors of a Hermitian (for real elements,
    /// symmetric) matrix: see [`HermitianEigen`]. A blocked reduction to a real symmetric
    /// tridiagonal matrix by Householder reflections, its eigensystem by Cuppen's divide and
    /// conquer, then the reflections applied to it in blocks; up to about a hundred rows (80 to
    /// 160 by the type), where that takes longer, the reduction one column at a time, its
    /// reflections multiplied out into `Q`, and the implicit QL algorithm's rotations applied
    /// to `Q`. Only the lower triangle is read, the upper one taken to be its conjugate. A
    /// matrix with a NaN or an infinity gives NaNs.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn hermitian_eigen(&self) -> HermitianEigen<T> {
        assert!(self.is_square(), "the matrix is not square");
        let (values, vectors) = hermitian(self, true);
        HermitianEigen {
            values,
            vectors: vectors.unwrap_or_default(),
        }
    }

    /// The eigenvalues of a Hermitian (for real elements, symmetric) matrix, real and
    /// ascending, without the eigenvectors: the same reduction, then the QL and QR algorithm
    /// without square roots, from about a hundred rows less than half the time of
    /// [`hermitian_eigen`](Matrix::hermitian_eigen). Only the lower triangle is read.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    #[track_caller]
    pub fn hermitian_eigenvalues(&self) -> Vec<T::Real> {
        assert!(self.is_square(), "the matrix is not square");
        hermitian(self, false).0
    }
}

/// Turns the real Schur form `t`, held as complex numbers, into a complex upper triangular one,
/// a unitary rotation for each 2 × 2 block, applied to `q` too, as MATLAB's `rsf2csf`.
pub(crate) fn real_to_complex_schur<C: Scalar>(t: &mut [C], q: &mut [C], n: usize) {
    let at = |i: usize, j: usize| i + j * n;
    let zero = C::Real::ZERO;
    for m in (1..n).rev() {
        let sub = t[at(m, m - 1)];
        if sub == C::zero() {
            continue;
        }
        // The block's eigenvalue with a positive imaginary part, less `t[m, m]`.
        let (a, b, c, d) = (t[at(m - 1, m - 1)], t[at(m - 1, m)], sub, t[at(m, m)]);
        let half = (a - d).scale(C::Real::HALF);
        let disc = (half * half + b * c).sqrt();
        let mu = half + disc;
        let r = mu.abs().hypot(c.abs());
        if r == zero {
            continue;
        }
        let cs = mu.scale(C::Real::ONE / r);
        let sn = c.scale(C::Real::ONE / r);
        // `G = [[cs*, sn], [-sn, cs]]` from the left on rows `m - 1, m`, `G*` from the right on
        // columns `m - 1, m`.
        for j in m - 1..n {
            let (x, y) = (t[at(m - 1, j)], t[at(m, j)]);
            t[at(m - 1, j)] = cs.conj() * x + sn * y;
            t[at(m, j)] = -sn.conj() * x + cs * y;
        }
        for i in 0..=m {
            let (x, y) = (t[at(i, m - 1)], t[at(i, m)]);
            t[at(i, m - 1)] = x * cs + y * sn.conj();
            t[at(i, m)] = -(x * sn) + y * cs.conj();
        }
        for i in 0..n {
            let (x, y) = (q[at(i, m - 1)], q[at(i, m)]);
            q[at(i, m - 1)] = x * cs + y * sn.conj();
            q[at(i, m)] = -(x * sn) + y * cs.conj();
        }
        t[at(m, m - 1)] = C::zero();
    }
}
