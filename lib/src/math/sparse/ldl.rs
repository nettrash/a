//! The sparse `L·D·Lᴴ` factorisation of a Hermitian matrix that need not be positive definite.

use super::supernodal::Factor;
use super::symbolic::Symbolic;
use super::{Ordering, SparseMatrix};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};
use core::fmt;

/// The root-free factorisation `P·A·Pᵀ = L·D·Lᴴ` of a sparse Hermitian (for real elements,
/// symmetric) matrix: `P` a permutation that keeps `L` sparse, `L` lower triangular with ones on
/// its diagonal, `D` diagonal and real, with the signs the matrix calls for.
///
/// The matrix must be square. **Only its lower triangle is read**; the upper one is taken to be
/// its conjugate transpose and may hold anything. Of a diagonal element the real part is read.
///
/// It is computed as [`SparseCholesky`](super::SparseCholesky) is — the same symbolic analysis,
/// the same supernodes, the same dense kernels on their blocks — without the square roots, so
/// the pivots may be negative.
///
/// # When it can be relied on
///
/// **There is no numerical pivoting**: the pivots are taken in the order the ordering fixed
/// before any arithmetic, as CHOLMOD's and Eigen's `LDLᵀ` take them. The factorisation exists
/// exactly when no pivot is zero, that is, when every leading block of `P·A·Pᵀ` is nonsingular,
/// and `None` is returned when a pivot is zero or not finite. That is assured, for every
/// permutation, for:
///
/// * positive definite and negative definite matrices, where it is as stable as the Cholesky
///   factorisation;
/// * quasi-definite matrices, `[[H, Bᴴ], [B, −C]]` with `H` and `C` positive definite, the
///   matrices of regularised saddle-point and interior-point systems (Vanderbei, "Symmetric
///   quasidefinite matrices", SIAM J. Optim. 5, 1995). The factors are then accurate when the
///   coupling `B` is not large against the two definite blocks (Gill, Saunders and Shinnerl,
///   "On the stability of Cholesky factorization for symmetric quasidefinite systems", SIAM J.
///   Matrix Anal. Appl. 17, 1996): the Schur complement `C + B·H⁻¹·Bᴴ` shows that they can grow
///   by about `‖B‖²/(λ(H)·λ(C))`, `λ` the smallest eigenvalues.
///
/// A general indefinite matrix is another matter. It can have a zero pivot although it is
/// nonsingular — `[[0, 1], [1, 0]]` does — and then gives `None`; worse, it can have a pivot
/// that is tiny without being zero, and then gives factors whose elements are huge and whose
/// solutions are wrong in most of their digits, with nothing to tell. A stable factorisation of
/// such a matrix needs `2 × 2` pivots chosen as it goes (Bunch and Kaufman), which this one does
/// not use: take [`SparseLu`](super::SparseLu), which pivots, or check the residual of what
/// [`solve`](Self::solve) returns.
///
/// # Not done
///
/// No `2 × 2` pivots, as said, and no pivoting of any kind after the ordering. The
/// factorisation runs on one thread, and a factor cannot be updated or downdated for a change
/// of the matrix, which is factored again instead.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseLdl, SparseMatrix};
///
/// // A saddle-point matrix [[H, Bᵀ], [B, −C]] with H = 2·I and C = 1: its lower triangle.
/// let a = SparseMatrix::from_coo(3, 3, &[
///     (0, 0, 2.0f64),
///     (1, 1, 2.0),
///     (2, 0, 1.0), (2, 1, 1.0), (2, 2, -1.0),
/// ]);
/// let f = SparseLdl::new(&a).unwrap();
///
/// // Two positive pivots and a negative one: by Sylvester's law, the signs of the eigenvalues.
/// assert_eq!(f.inertia(), (2, 1, 0));
/// assert!((f.det() + 8.0).abs() < 1e-14);
///
/// // A·x = b for x = (1, 1, 1).
/// let x = f.solve_vec(&[3.0, 3.0, 1.0]);
/// assert!(x.iter().all(|&x| (x - 1.0f64).abs() < 1e-14));
///
/// // A zero pivot: no factorisation without pivoting, though the matrix is nonsingular.
/// let swap = SparseMatrix::from_coo(2, 2, &[(1, 0, 1.0)]);
/// assert!(SparseLdl::new(&swap).is_none());
/// ```
#[derive(Clone)]
pub struct SparseLdl<T: Scalar> {
    /// The supernodal factor: the unit `L`, and `D`.
    factor: Factor<T>,
    /// `D` in the numbering of `P·A·Pᵀ`.
    d: Vec<T::Real>,
}

impl<T: Scalar> SparseLdl<T> {
    /// The factorisation of the Hermitian matrix whose lower triangle is that of `a`, under the
    /// default [`Ordering`], or `None` when a pivot is zero or not finite: see the
    /// [type](Self) for which matrices have none.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square, or if its factor would hold more elements than a vector can.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseLdl, SparseMatrix};
    ///
    /// // Negative definite: the Cholesky factorisation has nothing to offer, this one has.
    /// let a = SparseMatrix::tridiagonal(&[1.0; 3], &[-2.0; 4], &[1.0; 3]);
    /// let f = SparseLdl::new(&a).unwrap();
    /// assert_eq!(f.inertia(), (0, 4, 0));
    /// assert!(f.d().iter().all(|&d| d < 0.0));
    /// ```
    #[must_use]
    #[track_caller]
    pub fn new(a: &SparseMatrix<T>) -> Option<Self> {
        Self::with_ordering(a, &Ordering::default())
    }

    /// The factorisation `P·A·Pᵀ = L·D·Lᴴ` with the permutation the `ordering` gives, or `None`
    /// when a pivot is zero or not finite. Unlike the Cholesky factorisation of a positive
    /// definite matrix, that of an indefinite one can exist for one permutation and not for
    /// another.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square, if a permutation given isn't one of its rows, or if the
    /// factor would hold more elements than a vector can.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLdl, SparseMatrix};
    ///
    /// // [[0, 1], [1, 2]]: the first pivot is zero, so there is no factorisation as it stands.
    /// let a = SparseMatrix::from_coo(2, 2, &[(1, 0, 1.0), (1, 1, 2.0)]);
    /// assert!(SparseLdl::with_ordering(&a, &Ordering::Natural).is_none());
    ///
    /// // Taken the other way round it is [[2, 1], [1, 0]], with the pivots 2 and −1/2.
    /// let f = SparseLdl::with_ordering(&a, &Ordering::Given(vec![1, 0])).unwrap();
    /// assert_eq!(f.d(), [2.0, -0.5]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn with_ordering(a: &SparseMatrix<T>, ordering: &Ordering) -> Option<Self> {
        let symbolic = Symbolic::analyse(a, ordering);
        let lower = symbolic.lower(a);
        let factor = Factor::new(symbolic, &lower, true)?;
        let d = factor.pivots();
        Some(Self { factor, d })
    }

    /// The diagonal of `D`, the pivots, in the order of `P·A·Pᵀ`: real, none of them zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLdl, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 4.0), (1, 0, 2.0), (1, 1, -3.0)]);
    /// let f = SparseLdl::with_ordering(&a, &Ordering::Natural).unwrap();
    /// assert_eq!(f.d(), [4.0, -4.0]);
    /// ```
    #[must_use]
    pub fn d(&self) -> &[T::Real] {
        &self.d
    }

    /// The factor `L` of the permuted matrix, `P·A·Pᵀ = L·D·Lᴴ`, as a sparse matrix: lower
    /// triangular, with its ones on the diagonal. An element the arithmetic made exactly zero is
    /// not stored, so it can have fewer nonzeros than [`nnz`](Self::nnz) tells.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLdl, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 4.0), (1, 0, 2.0), (1, 1, -3.0)]);
    /// let f = SparseLdl::with_ordering(&a, &Ordering::Natural).unwrap();
    /// let l = f.l();
    /// assert_eq!((l[(0, 0)], l[(1, 0)], l[(1, 1)]), (1.0, 0.5, 1.0));
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
    /// use a::math::sparse::{Ordering, SparseLdl, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_diagonal(&[1.0, -2.0, 3.0]);
    /// let f = SparseLdl::with_ordering(&a, &Ordering::Given(vec![2, 0, 1])).unwrap();
    /// assert_eq!(f.permutation(), [2, 0, 1]);
    /// assert_eq!(f.d(), [3.0, 1.0, -2.0]);
    /// ```
    #[must_use]
    pub fn permutation(&self) -> &[usize] {
        &self.factor.symbolic.permutation
    }

    /// The solution `X` of `A·X = B`: a triangular solve, a division by the pivots and a second
    /// triangular solve, supernode by supernode. `b` may have several columns.
    ///
    /// # Panics
    ///
    /// Panics if `b` has not as many rows as the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::sparse::{SparseLdl, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_diagonal(&[2.0, -4.0]);
    /// let f = SparseLdl::new(&a).unwrap();
    /// let b = Matrix::from_rows(&[[2.0, 4.0], [4.0, 8.0]]);
    /// assert_eq!(f.solve(&b), Matrix::from_rows(&[[1.0, 2.0], [-1.0, -2.0]]));
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
    /// use a::math::sparse::{SparseLdl, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_diagonal(&[2.0, -4.0]);
    /// let f = SparseLdl::new(&a).unwrap();
    /// assert_eq!(f.solve_vec(&[2.0, 4.0]), [1.0, -1.0]);
    /// ```
    #[must_use]
    #[track_caller]
    pub fn solve_vec(&self, b: &[T]) -> Vec<T> {
        self.factor.solve_vec(b)
    }

    /// The numbers of positive, negative and zero pivots. By Sylvester's law of inertia these
    /// are the numbers of positive, negative and zero eigenvalues of the matrix, whatever the
    /// permutation. A factorisation that exists has no zero pivot, so the third is 0.
    ///
    /// The count is as good as the pivots: a pivot that rounding has carried across zero is
    /// counted with the wrong sign, which can happen only to one that is tiny beside the
    /// elements of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseLdl, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_diagonal(&[3.0, -1.0, 2.0, -5.0, 1.0]);
    /// assert_eq!(SparseLdl::new(&a).unwrap().inertia(), (3, 2, 0));
    /// ```
    #[must_use]
    pub fn inertia(&self) -> (usize, usize, usize) {
        let positive = self.d.iter().filter(|&&d| d > T::Real::ZERO).count();
        let negative = self.d.iter().filter(|&&d| d < T::Real::ZERO).count();
        (positive, negative, self.d.len() - positive - negative)
    }

    /// The determinant of the matrix, the product of the pivots: real, with the sign of the
    /// matrix's. Like any determinant of a large matrix it overflows or underflows easily.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{SparseLdl, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_diagonal(&[3.0, -1.0, 2.0]);
    /// assert_eq!(SparseLdl::new(&a).unwrap().det(), -6.0);
    /// ```
    #[must_use]
    pub fn det(&self) -> T::Real {
        self.factor
            .diagonal()
            .fold(T::Real::ONE, |product, d| product * d)
    }

    /// The number of nonzeros of `L`, the ones of its diagonal counted, as the pattern of the
    /// matrix and the ordering decide it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ordering, SparseLdl, SparseMatrix};
    ///
    /// let a = SparseMatrix::tridiagonal(&[1.0; 4], &[-2.0; 5], &[1.0; 4]);
    /// let f = SparseLdl::with_ordering(&a, &Ordering::Natural).unwrap();
    /// assert_eq!(f.nnz(), 9);
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
    /// use a::math::sparse::{SparseLdl, SparseMatrix};
    ///
    /// let diagonal = SparseMatrix::from_diagonal(&[1.0, -2.0, 3.0]);
    /// assert_eq!(SparseLdl::new(&diagonal).unwrap().supernodes(), 3);
    /// ```
    #[must_use]
    pub fn supernodes(&self) -> usize {
        self.factor.symbolic.supernodes()
    }
}

impl<T: Scalar> fmt::Debug for SparseLdl<T> {
    /// Writes the order, the nonzeros of the factor, the supernodes and the inertia, as
    /// `SparseLdl { order: 3, nnz: 5, supernodes: 1, inertia: (2, 1, 0) }`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseLdl")
            .field("order", &self.factor.symbolic.n)
            .field("nnz", &self.nnz())
            .field("supernodes", &self.supernodes())
            .field("inertia", &self.inertia())
            .finish()
    }
}
