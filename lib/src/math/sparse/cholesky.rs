//! The sparse Cholesky factorisation of a Hermitian positive definite matrix.

use super::supernodal::Factor;
use super::symbolic::Symbolic;
use super::{Ordering, SparseMatrix};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};
use core::fmt;

/// The Cholesky factorisation `P·A·Pᵀ = L·Lᴴ` of a sparse Hermitian (for real elements,
/// symmetric) positive definite matrix: `P` a permutation that keeps `L` sparse, `L` lower
/// triangular with a positive real diagonal.
///
/// The matrix must be square. **Only its lower triangle is read**; the upper one is taken to be
/// its conjugate transpose and may hold anything, as for the dense
/// [`cholesky`](crate::math::matrix::Matrix::cholesky). Of a diagonal element the real part is
/// read.
///
/// The work is done in two parts. The *symbolic analysis* reads the pattern alone: it orders the
/// matrix, finds the elimination tree (Liu's algorithm) and the number of nonzeros of every
/// column of `L` (the algorithm of Gilbert, Ng and Peyton), and gathers the columns into
/// *supernodes*, runs of columns with the same rows below a dense triangle. The *numeric
/// factorisation* then goes supernode by supernode, left-looking as in Ng and Peyton's method
/// and CHOLMOD's: a supernode is a dense block, updated by products of the blocks of its
/// descendants, factored by the dense Cholesky method and solved against its triangle, so that
/// all the arithmetic is that of the dense kernels. The time is that of the nonzeros of `L` and
/// the operations on them, not of `n²`, and [`solve`](Self::solve) costs what the nonzeros of
/// `L` cost.
///
/// # Supernodes
///
/// A *fundamental* supernode is a longest run of columns in which each is the only child of the
/// next in the elimination tree and has one nonzero more than it: columns that are one dense
/// block exactly. A supernode is then merged into its parent where that costs few explicit
/// zeros, the *relaxed amalgamation* of Ashcraft and Grimes: always when the merged block has
/// at most 4 columns or gains no zero, and otherwise when its zeros, those of earlier merges
/// counted, stay under 80 % of a block of up to 16 columns, under 10 % of one of up to 48 and
/// under 5 % of any. These are CHOLMOD's defaults, not tuned here. A larger block makes the
/// dense kernels faster, and its zeros cost memory and operations; whatever the merges, the
/// zeros stay under four fifths of what the blocks hold on and below their diagonals, so those
/// parts hold fewer than five times the nonzeros of `L`, and the blocks in all, whose squares
/// on the diagonal are stored whole, fewer than ten times. A tridiagonal matrix, whose blocks
/// of 16 columns are mostly zeros, comes near that: 8.5 times.
/// [`supernodes`](Self::supernodes) tells how many blocks there are.
///
/// # Not done
///
/// The factorisation runs on one thread. A factor cannot be updated or downdated for a change
/// of the matrix, which is factored again instead, and the symbolic analysis is not kept apart
/// for a second matrix of the same pattern. A Hermitian matrix that is not positive definite is
/// for [`SparseLdl`](super::SparseLdl).
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::sparse::{SparseCholesky, SparseMatrix};
///
/// // The 1-D Laplacian of six points.
/// let a = SparseMatrix::tridiagonal(&[-1.0; 5], &[2.0; 6], &[-1.0; 5]);
/// let c = SparseCholesky::new(&a).unwrap();
///
/// // A·x = b for a right-hand side with a known solution.
/// let x = Matrix::from_fn(6, 1, |i, _| (i + 1) as f64);
/// let b = &a.to_matrix() * &x;
/// let solved = c.solve(&b);
/// assert!((0..6).all(|i| (solved[(i, 0)] - x[(i, 0)]).abs() < 1e-13));
///
/// // det(A) = 7, and its logarithm, which does not overflow for a large matrix.
/// assert!((c.det() - 7.0).abs() < 1e-13);
/// assert!((c.log_det() - 7.0f64.ln()).abs() < 1e-13);
///
/// // A matrix that is not positive definite has no factorisation.
/// let indefinite = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (1, 0, 2.0), (1, 1, 1.0)]);
/// assert!(SparseCholesky::new(&indefinite).is_none());
/// ```
#[derive(Clone)]
pub struct SparseCholesky<T: Scalar> {
    /// The supernodal factor `L`.
    factor: Factor<T>,
}

impl<T: Scalar> SparseCholesky<T> {
    /// The factorisation of the Hermitian positive definite matrix whose lower triangle is that
    /// of `a`, under the default [`Ordering`], or `None` when the matrix is not positive
    /// definite: a pivot is not positive, or not finite.
    ///
    /// A singular positive semidefinite matrix has a zero pivot and gives `None` when the
    /// rounding leaves that pivot at zero or below it, but a factor with a tiny pivot when the
    /// rounding leaves it above, as the dense method does.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square, or if its factor would hold more elements than a vector can.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseCholesky, SparseMatrix};
    ///
    /// // Only the lower triangle is given: the upper one is its transpose.
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 4.0f64), (1, 0, 2.0), (1, 1, 5.0)]);
    /// let c = SparseCholesky::new(&a).unwrap();
    /// let x = c.solve_vec(&[8.0, 12.0]);
    /// assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14);
    ///
    /// // The identity with a negative element on its diagonal is not positive definite.
    /// let b = SparseMatrix::from_diagonal(&[1.0, -1.0]);
    /// assert!(SparseCholesky::new(&b).is_none());
    /// ```
    #[must_use]
    #[track_caller]
    pub fn new(a: &SparseMatrix<T>) -> Option<Self> {
        Self::with_ordering(a, &Ordering::default())
    }

    /// The factorisation `P·A·Pᵀ = L·Lᴴ` with the permutation the `ordering` gives, or `None`
    /// when the matrix is not positive definite. Any permutation gives a factorisation; what the
    /// ordering decides is how many nonzeros `L` has.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square, if a permutation given isn't one of its rows, or if the
    /// factor would hold more elements than a vector can.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseCholesky, SparseMatrix};
    ///
    /// // An arrow pointing up and to the left: eliminating its tip first fills the factor in.
    /// let a = SparseMatrix::from_coo(4, 4, &[
    ///     (0, 0, 4.0), (1, 0, 1.0), (2, 0, 1.0), (3, 0, 1.0),
    ///     (1, 1, 4.0), (2, 2, 4.0), (3, 3, 4.0),
    /// ]);
    /// let natural = SparseCholesky::with_ordering(&a, &Ordering::Natural).unwrap();
    /// assert_eq!(natural.nnz(), 10);
    ///
    /// // With the tip eliminated last there is no fill at all.
    /// let reversed = Ordering::Given(vec![3, 2, 1, 0]);
    /// let c = SparseCholesky::with_ordering(&a, &reversed).unwrap();
    /// assert_eq!(c.nnz(), 7);
    /// assert_eq!(c.permutation(), [3, 2, 1, 0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn with_ordering(a: &SparseMatrix<T>, ordering: &Ordering) -> Option<Self> {
        let symbolic = Symbolic::analyse(a, ordering);
        let lower = symbolic.lower(a);
        Factor::new(symbolic, &lower, false).map(|factor| Self { factor })
    }

    /// The solution `X` of `A·X = B`, by the two triangular solves, supernode by supernode. `b`
    /// may have several columns.
    ///
    /// # Panics
    ///
    /// Panics if `b` has not as many rows as the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::{SparseCholesky, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_diagonal(&[4.0, 16.0]);
    /// let c = SparseCholesky::new(&a).unwrap();
    /// let b = Matrix::from_rows(&[[4.0, 8.0], [16.0, 32.0]]);
    /// assert_eq!(c.solve(&b), Matrix::from_rows(&[[1.0, 2.0], [1.0, 2.0]]));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve(&self, b: &Matrix<T>) -> Matrix<T> {
        self.factor.solve(b)
    }

    /// The solution `x` of `A·x = b`, for one right-hand side.
    ///
    /// # Panics
    ///
    /// Panics if `b` is not as long as the matrix has rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseCholesky, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_diagonal(&[4.0, 16.0]);
    /// let c = SparseCholesky::new(&a).unwrap();
    /// assert_eq!(c.solve_vec(&[4.0, 16.0]), [1.0, 1.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve_vec(&self, b: &[T]) -> Vec<T> {
        self.factor.solve_vec(b)
    }

    /// The factor `L` of the permuted matrix, `P·A·Pᵀ = L·Lᴴ`, as a sparse matrix: lower
    /// triangular with a positive real diagonal. An element the arithmetic made exactly zero is
    /// not stored, so it can have fewer nonzeros than [`nnz`](Self::nnz) tells.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseCholesky, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 4.0), (1, 0, 2.0), (1, 1, 5.0)]);
    /// let c = SparseCholesky::with_ordering(&a, &Ordering::Natural).unwrap();
    /// let l = c.l();
    /// assert_eq!((l[(0, 0)], l[(1, 0)], l[(1, 1)]), (2.0, 1.0, 2.0));
    /// assert_eq!(l[(0, 1)], 0.0);
    /// ```
    #[must_use]
    pub fn l(&self) -> SparseMatrix<T> {
        self.factor.l()
    }

    /// The permutation `P` as the ordering gave it: row `i` of `P·A·Pᵀ` is row
    /// `permutation()[i]` of `A`, and so with the columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseCholesky, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_diagonal(&[1.0, 2.0, 3.0]);
    /// let c = SparseCholesky::with_ordering(&a, &Ordering::Natural).unwrap();
    /// assert_eq!(c.permutation(), [0, 1, 2]);
    /// ```
    #[must_use]
    pub fn permutation(&self) -> &[usize] {
        &self.factor.symbolic.permutation
    }

    /// The determinant of the matrix, the square of the product of the diagonal of `L`: real
    /// and positive. It overflows to infinity, or underflows to zero, long before the matrix is
    /// large: [`log_det`](Self::log_det) does not.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseCholesky, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 4.0), (1, 0, 2.0), (1, 1, 5.0)]);
    /// let det: f64 = SparseCholesky::new(&a).unwrap().det();
    /// assert!((det - 16.0).abs() < 1e-13);
    /// ```
    #[must_use]
    pub fn det(&self) -> T::Real {
        self.factor
            .diagonal()
            .fold(T::Real::ONE, |product, l| product * (l * l))
    }

    /// The natural logarithm of the determinant, twice the sum of the logarithms of the diagonal
    /// of `L`, added with compensation so that the sum of many loses no more than a rounding or
    /// two of the total.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseCholesky, SparseMatrix};
    ///
    /// // The determinant of this matrix is 10⁴⁰⁰, beyond the largest `f64`.
    /// let a = SparseMatrix::from_diagonal(&[1e4f64; 100]);
    /// let c = SparseCholesky::new(&a).unwrap();
    /// assert!(c.det().is_infinite());
    /// assert!((c.log_det() - 400.0 * 10.0f64.ln()).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn log_det(&self) -> T::Real {
        // A compensated sum, Neumaier's form of Kahan's: the plain one of `n` logarithms loses
        // up to `n` roundings of the total, which in `f32` is the fourth digit at `n = 10⁴`.
        let (mut sum, mut carry) = (T::Real::ZERO, T::Real::ZERO);
        for l in self.factor.diagonal() {
            let term = l.ln();
            let next = sum + term;
            carry += if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sum = next;
        }
        T::Real::TWO * (sum + carry)
    }

    /// The number of nonzeros of `L`, its diagonal counted, as the pattern of the matrix and the
    /// ordering decide it: the fill the factorisation has, whatever its elements are.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseCholesky, SparseMatrix};
    ///
    /// // A tridiagonal matrix in its own order has a bidiagonal factor.
    /// let a = SparseMatrix::tridiagonal(&[-1.0; 4], &[2.0; 5], &[-1.0; 4]);
    /// let c = SparseCholesky::with_ordering(&a, &Ordering::Natural).unwrap();
    /// assert_eq!(c.nnz(), 9);
    /// assert_eq!(c.l().nnz(), 9);
    /// ```
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.factor.symbolic.nnz
    }

    /// The number of supernodes the columns of `L` were gathered into: the dense blocks the
    /// factorisation worked on.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseCholesky, SparseMatrix};
    ///
    /// // Nothing joins the columns of a diagonal matrix; a full matrix is one block.
    /// let diagonal = SparseMatrix::from_diagonal(&[1.0, 2.0, 3.0]);
    /// assert_eq!(SparseCholesky::new(&diagonal).unwrap().supernodes(), 3);
    /// let full = SparseMatrix::from_fn(3, 3, |i, j| if i == j { 4.0 } else { 1.0 });
    /// assert_eq!(SparseCholesky::new(&full).unwrap().supernodes(), 1);
    /// ```
    #[must_use]
    pub fn supernodes(&self) -> usize {
        self.factor.symbolic.supernodes()
    }
}

impl<T: Scalar> fmt::Debug for SparseCholesky<T> {
    /// Writes the order, the nonzeros of the factor and the supernodes, as
    /// `SparseCholesky { order: 6, nnz: 11, supernodes: 1 }`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseCholesky")
            .field("order", &self.factor.symbolic.n)
            .field("nnz", &self.nnz())
            .field("supernodes", &self.supernodes())
            .finish()
    }
}
