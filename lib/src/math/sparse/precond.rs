//! The preconditioners of the iterative solvers: the diagonal one, and the incomplete
//! factorisations that keep the pattern of the matrix.
//!
//! A preconditioner is a matrix `M` near `A` whose systems are cheap to solve. A Krylov method
//! run on the preconditioned system needs as many iterations as the spectrum of `M⁻¹·A` asks for
//! rather than that of `A`, and all it ever does with `M` is solve with it, once or twice an
//! iteration: that is [`Preconditioner::apply`].
//!
//! | Preconditioner | `M` | For | Cost of building | Cost of applying |
//! | --- | --- | --- | --- | --- |
//! | [`Jacobi`] | the diagonal of `A` | any matrix with a nonzero diagonal | `n` divisions | `n` products |
//! | [`Ilu0`] | `L·U`, with the pattern of `A` | any matrix with nonzero pivots | the sum over the rows of the lengths of the rows each one uses | the nonzeros of `A` |
//! | [`Ic0`] | `L·Lᴴ`, with the pattern of the lower triangle | Hermitian positive definite matrices | the same, on the lower triangle | twice the nonzeros of the lower triangle |
//!
//! A constructor gives `None` when the preconditioner does not exist: a diagonal entry or a pivot
//! is missing, zero, not positive where it has to be, or not a number.
//!
//! # Examples
//!
//! ```
//! use a::math::sparse::{Ic0, SolveOptions, SparseMatrix, cg_with};
//!
//! // The 1-D Laplacian of thirty points. Its factor has no fill, so the incomplete
//! // factorisation is the complete one and one step of the method is enough.
//! const N: usize = 30;
//! let a = SparseMatrix::tridiagonal(&[-1.0; N - 1], &[2.0; N], &[-1.0; N - 1]);
//! let b = vec![1.0; N];
//! let m = Ic0::new(&a).unwrap();
//! let plain = cg_with(&a, &b, &SolveOptions::new(), None).unwrap();
//! let preconditioned = cg_with(&a, &b, &SolveOptions::new(), Some(&m)).unwrap();
//! assert!(plain.converged && preconditioned.converged);
//! assert_eq!((plain.iterations, preconditioned.iterations), (15, 1));
//! ```

use super::SparseMatrix;
use crate::math::complex::Float;
use crate::math::matrix::Scalar;

/// What an iterative solver asks of a preconditioner `M`: the solution of `M·z = r`.
///
/// [`Jacobi`], [`Ilu0`] and [`Ic0`] implement it, and so can anything else that maps a residual
/// to a correction. The conjugate gradient method and MINRES need `M` Hermitian positive
/// definite; GMRES and BiCGSTAB take any nonsingular `M`.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{Preconditioner, SolveOptions, SparseMatrix, gmres_with};
///
/// // A preconditioner of one's own: the reciprocals of the row sums.
/// struct RowSums(Vec<f64>);
///
/// impl Preconditioner<f64> for RowSums {
///     fn apply(&self, r: &[f64], z: &mut [f64]) {
///         for ((z, &r), &s) in z.iter_mut().zip(r).zip(&self.0) {
///             *z = r / s;
///         }
///     }
/// }
///
/// let a = SparseMatrix::from_coo(3, 3, &[
///     (0, 0, 4.0), (0, 1, 1.0),
///     (1, 1, 30.0), (1, 2, 2.0),
///     (2, 0, 100.0), (2, 2, 500.0),
/// ]);
/// let m = RowSums(vec![5.0, 32.0, 600.0]);
/// let x = gmres_with(&a, &[5.0, 32.0, 600.0], &SolveOptions::new(), Some(&m)).unwrap();
/// assert!(x.converged);
/// assert!(x.x.iter().all(|&v| (v - 1.0).abs() < 1e-7));
/// ```
pub trait Preconditioner<T: Scalar> {
    /// Writes `M⁻¹·r` into `z`.
    ///
    /// # Panics
    ///
    /// The preconditioners of this module panic if `r` and `z` are not as long as the matrix
    /// they were made from has rows.
    fn apply(&self, r: &[T], z: &mut [T]);
}

/// The offset of the diagonal element of every row in the stored elements of the square `a`, or
/// `None` if a row stores none.
fn diagonal_offsets<T: Scalar>(a: &SparseMatrix<T>) -> Option<Vec<usize>> {
    let (pointers, columns) = (a.row_pointers(), a.col_indices());
    (0..a.rows())
        .map(|i| {
            let (begin, end) = (pointers[i], pointers[i + 1]);
            let found = columns[begin..end].binary_search(&i);
            found.ok().map(|k| begin + k)
        })
        .collect()
}

/// The checks every `apply` starts with.
#[track_caller]
fn check_lengths<T>(n: usize, r: &[T], z: &[T]) {
    assert_eq!(
        r.len(),
        n,
        "the vector has the wrong length for the preconditioner"
    );
    assert_eq!(
        z.len(),
        n,
        "the result has the wrong length for the preconditioner"
    );
}

/// The diagonal preconditioner `M = diag(A)`, Jacobi's: the cheapest there is, and a real help
/// when the rows of the matrix are of very different sizes. On a matrix whose diagonal is
/// constant it changes nothing, since the Krylov space is the same.
///
/// It is Hermitian positive definite, as the conjugate gradient method and MINRES need, exactly
/// when the diagonal is real and positive.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{Jacobi, Preconditioner, SparseMatrix};
///
/// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 4.0), (0, 1, 1.0), (1, 1, 2.0)]);
/// let jacobi = Jacobi::new(&a).unwrap();
/// let mut z = [0.0; 2];
/// jacobi.apply(&[4.0, 2.0], &mut z);
/// assert_eq!(z, [1.0, 1.0]);
///
/// // No diagonal entry in the second row, so no preconditioner.
/// let singular = SparseMatrix::from_coo(2, 2, &[(0, 0, 4.0), (1, 0, 1.0)]);
/// assert!(Jacobi::new(&singular).is_none());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Jacobi<T> {
    /// The reciprocals of the diagonal.
    inverse: Vec<T>,
}

impl<T: Scalar> Jacobi<T> {
    /// The preconditioner of the square `a`, or `None` if a diagonal entry is missing, zero or not
    /// finite.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square.
    #[must_use]
    #[track_caller]
    pub fn new(a: &SparseMatrix<T>) -> Option<Self> {
        assert!(a.rows() == a.cols(), "the matrix isn't square");
        let values = a.values();
        let mut inverse = Vec::with_capacity(a.rows());
        for at in diagonal_offsets(a)? {
            let reciprocal = T::one() / values[at];
            // A reciprocal that is zero is that of an infinite entry.
            if !reciprocal.is_finite() || reciprocal.is_zero() {
                return None;
            }
            inverse.push(reciprocal);
        }
        Some(Self { inverse })
    }
}

impl<T: Scalar> Preconditioner<T> for Jacobi<T> {
    #[track_caller]
    fn apply(&self, r: &[T], z: &mut [T]) {
        check_lengths(self.inverse.len(), r, z);
        for ((z, &w), &r) in z.iter_mut().zip(&self.inverse).zip(r) {
            *z = w * r;
        }
    }
}

/// The incomplete LU factorisation with no fill, ILU(0): `M = L·U` with `L` unit lower
/// triangular, `U` upper triangular, and both confined to the pattern of `A`.
///
/// It is Gaussian elimination without pivoting in which every element outside the pattern of `A`
/// is thrown away as it appears, which leaves `(L·U)ᵢⱼ = aᵢⱼ` on every stored `(i, j)` and an
/// error only where the complete factors would have filled in. It follows the row-by-row (IKJ)
/// form of Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., algorithm 10.4: a row is
/// reduced by the rows above it that it has an element under, each of them read once, and the
/// place of a column in the row being reduced is looked up in a table rather than searched for.
/// A tridiagonal matrix has no fill, so there the factorisation is the complete one.
///
/// There is no pivoting: the factorisation needs every diagonal entry stored and every pivot
/// nonzero, and it can be unstable where the complete factorisation without pivoting would be.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{Ilu0, Preconditioner, SparseMatrix};
///
/// let a = SparseMatrix::tridiagonal(&[-1.0; 3], &[2.0; 4], &[-1.0; 3]);
/// let ilu = Ilu0::new(&a).unwrap();
/// assert_eq!(ilu.nnz(), a.nnz());
/// // No fill, so L·U is the matrix to rounding and the preconditioner solves the system.
/// let product = &ilu.l() * &ilu.u();
/// assert!((&product - &a).values().iter().all(|x: &f64| x.abs() < 1e-15));
/// let mut z = [0.0; 4];
/// ilu.apply(&[1.0, 0.0, 0.0, 1.0], &mut z);
/// assert!(z.iter().all(|v: &f64| (v - 1.0).abs() < 1e-15));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Ilu0<T: Scalar> {
    /// Where each row begins in `columns` and `values`: the pattern of `A`.
    row_ptrs: Vec<usize>,
    /// The column of each stored element, ascending within a row.
    columns: Vec<usize>,
    /// The factors, merged: left of the diagonal the elements of `L`, whose diagonal of ones is
    /// not stored, and from the diagonal on those of `U`.
    values: Vec<T>,
    /// The offset of the diagonal element of each row.
    diagonal: Vec<usize>,
}

impl<T: Scalar> Ilu0<T> {
    /// The incomplete factorisation of the square `a`, or `None` if a diagonal entry is not
    /// stored, a pivot comes out zero, or an element of the factors is not finite.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square.
    #[must_use]
    #[track_caller]
    pub fn new(a: &SparseMatrix<T>) -> Option<Self> {
        assert!(a.rows() == a.cols(), "the matrix isn't square");
        let n = a.rows();
        let row_ptrs = a.row_pointers().to_vec();
        let columns = a.col_indices().to_vec();
        let mut values = a.values().to_vec();
        let diagonal = diagonal_offsets(a)?;
        // `position[j]` is the offset of column j in the row being reduced, for the columns the
        // row stores, so that an update finds its target in one step or learns that it is fill.
        const ABSENT: usize = usize::MAX;
        let mut position = vec![ABSENT; n];
        for i in 0..n {
            let (begin, end) = (row_ptrs[i], row_ptrs[i + 1]);
            for at in begin..end {
                position[columns[at]] = at;
            }
            // The elements left of the diagonal, in ascending order of column: each becomes a
            // multiplier, and its row of U is subtracted from what the row still holds to the
            // right of it.
            for at in begin..diagonal[i] {
                let k = columns[at];
                let multiplier = values[at] / values[diagonal[k]];
                values[at] = multiplier;
                for above in diagonal[k] + 1..row_ptrs[k + 1] {
                    let target = position[columns[above]];
                    if target != ABSENT {
                        values[target] = values[target].mul_sub(multiplier, values[above]);
                    }
                }
            }
            for at in begin..end {
                position[columns[at]] = ABSENT;
            }
            let pivot = values[diagonal[i]];
            if pivot.is_zero() || !pivot.is_finite() {
                return None;
            }
        }
        if !values.iter().all(|x| x.is_finite()) {
            return None;
        }
        Some(Self {
            row_ptrs,
            columns,
            values,
            diagonal,
        })
    }

    /// The number of stored elements of the two factors together, the unit diagonal of `L` not
    /// counted: the number of stored elements of the matrix, since there is no fill.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// `L`, unit lower triangular. An element that came out exactly zero is not stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ilu0, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 4.0), (1, 1, 5.0)]);
    /// let l = Ilu0::new(&a).unwrap().l();
    /// assert_eq!((l[(0, 0)], l[(1, 0)], l[(1, 1)]), (1.0, 2.0, 1.0));
    /// ```
    #[must_use]
    pub fn l(&self) -> SparseMatrix<T> {
        let n = self.diagonal.len();
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            row_ptrs.push(columns.len());
            for at in self.row_ptrs[i]..self.diagonal[i] {
                columns.push(self.columns[at]);
                values.push(self.values[at]);
            }
            columns.push(i);
            values.push(T::one());
        }
        row_ptrs.push(columns.len());
        SparseMatrix::from_csr(n, n, row_ptrs, columns, values)
    }

    /// `U`, upper triangular. An element that came out exactly zero is not stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ilu0, SparseMatrix};
    ///
    /// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 2.0), (0, 1, 1.0), (1, 0, 4.0), (1, 1, 5.0)]);
    /// let u = Ilu0::new(&a).unwrap().u();
    /// assert_eq!((u[(0, 0)], u[(0, 1)], u[(1, 1)]), (2.0, 1.0, 3.0));
    /// ```
    #[must_use]
    pub fn u(&self) -> SparseMatrix<T> {
        let n = self.diagonal.len();
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            row_ptrs.push(columns.len());
            for at in self.diagonal[i]..self.row_ptrs[i + 1] {
                columns.push(self.columns[at]);
                values.push(self.values[at]);
            }
        }
        row_ptrs.push(columns.len());
        SparseMatrix::from_csr(n, n, row_ptrs, columns, values)
    }
}

impl<T: Scalar> Preconditioner<T> for Ilu0<T> {
    /// `z = U⁻¹·L⁻¹·r`: a forward and a back substitution over the rows of the factors, with
    /// nothing allocated.
    #[track_caller]
    fn apply(&self, r: &[T], z: &mut [T]) {
        let n = self.diagonal.len();
        check_lengths(n, r, z);
        for i in 0..n {
            let mut sum = r[i];
            for at in self.row_ptrs[i]..self.diagonal[i] {
                sum = sum.mul_sub(self.values[at], z[self.columns[at]]);
            }
            z[i] = sum;
        }
        for i in (0..n).rev() {
            let mut sum = z[i];
            for at in self.diagonal[i] + 1..self.row_ptrs[i + 1] {
                sum = sum.mul_sub(self.values[at], z[self.columns[at]]);
            }
            z[i] = sum / self.values[self.diagonal[i]];
        }
    }
}

/// The incomplete Cholesky factorisation with no fill, IC(0), of a Hermitian positive definite
/// matrix: `M = L·Lᴴ` with `L` lower triangular, its diagonal real and positive, and confined to
/// the pattern of the lower triangle of `A`.
///
/// Every element of `L` is the one the complete factorisation computes, `lᵢₖ = (aᵢₖ − Σⱼ lᵢⱼ·l̄ₖⱼ)
/// / lₖₖ` and `lᵢᵢ² = aᵢᵢ − Σⱼ |lᵢⱼ|²`, with the sums taken over the columns `j < k` that both
/// rows store, so that `(L·Lᴴ)ᵢⱼ = aᵢⱼ` on every stored `(i, j)` of the lower triangle; what is
/// thrown away is the fill. `M` is Hermitian positive definite, which makes it the
/// preconditioner of choice for the conjugate gradient method.
///
/// Only the lower triangle of `a` is read; the upper one is taken to be its conjugate transpose,
/// and the imaginary part of the diagonal is dropped, as [`Matrix::cholesky`] does.
///
/// The incomplete factorisation of a positive definite matrix need not exist: throwing fill away
/// can make a pivot negative. It always exists for an M-matrix (Meijerink and van der Vorst,
/// 1977) and for a diagonally dominant matrix. Where it does not, [`with_shift`](Self::with_shift)
/// factors `A + α·diag(A)` instead (Manteuffel, 1980), for an `α` large enough.
///
/// [`Matrix::cholesky`]: crate::math::matrix::Matrix::cholesky
///
/// # Examples
///
/// ```
/// use a::math::sparse::{Ic0, Preconditioner, SparseMatrix};
///
/// // A full matrix has no fill to lose: the factor is the Cholesky factor [[2, 0], [1, 2]].
/// let a = SparseMatrix::from_coo(2, 2, &[(0, 0, 4.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 5.0)]);
/// let ic = Ic0::new(&a).unwrap();
/// let l = ic.l();
/// assert_eq!((l[(0, 0)], l[(1, 0)], l[(1, 1)]), (2.0, 1.0, 2.0));
/// let mut z = [0.0; 2];
/// ic.apply(&[4.0, 2.0], &mut z);
/// assert_eq!(z, [1.0, 0.0]);
///
/// // An indefinite matrix has no factor.
/// let b = SparseMatrix::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 1.0)]);
/// assert!(Ic0::new(&b).is_none());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Ic0<T: Scalar> {
    /// Where each row of `L` begins in `columns` and `values`: the pattern of the lower triangle
    /// of `A`.
    row_ptrs: Vec<usize>,
    /// The column of each stored element, ascending within a row, so that the diagonal is the
    /// last element of its row.
    columns: Vec<usize>,
    /// The elements of `L`.
    values: Vec<T>,
}

impl<T: Scalar> Ic0<T> {
    /// The incomplete factorisation of the Hermitian positive definite `a`, or `None` if a
    /// diagonal entry is not stored or a pivot is not positive or not finite: the matrix is not
    /// positive definite, or it is and the factorisation without fill breaks down all the same.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square.
    #[must_use]
    #[track_caller]
    pub fn new(a: &SparseMatrix<T>) -> Option<Self> {
        Self::with_shift(a, T::Real::ZERO)
    }

    /// The incomplete factorisation of `A + α·diag(A)`: the diagonal is multiplied by `1 + α`
    /// before it is factored, which makes the pivots larger and the factorisation exist once `α`
    /// is large enough, at the price of a preconditioner further from the matrix. The usual way
    /// is to try [`new`](Self::new) first and then `α = 10⁻³, 10⁻², …` until a factor comes
    /// back.
    ///
    /// `None` as for [`new`](Self::new), and when `α` is negative or not finite.
    ///
    /// # Panics
    ///
    /// Panics if `a` isn't square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ic0, SparseMatrix};
    ///
    /// // Kershaw's matrix: positive definite, and yet the last pivot of IC(0) is negative.
    /// let a = SparseMatrix::from_coo(4, 4, &[
    ///     (0, 0, 3.0), (0, 1, -2.0), (0, 3, 2.0),
    ///     (1, 0, -2.0), (1, 1, 3.0), (1, 2, -2.0),
    ///     (2, 1, -2.0), (2, 2, 3.0), (2, 3, -2.0),
    ///     (3, 0, 2.0), (3, 2, -2.0), (3, 3, 3.0),
    /// ]);
    /// assert!(Ic0::new(&a).is_none());
    /// assert!(Ic0::with_shift(&a, 0.5).is_some());
    /// ```
    #[must_use]
    #[track_caller]
    pub fn with_shift(a: &SparseMatrix<T>, alpha: T::Real) -> Option<Self> {
        assert!(a.rows() == a.cols(), "the matrix isn't square");
        if !(alpha >= T::Real::ZERO && alpha.is_finite()) {
            return None;
        }
        let n = a.rows();
        // The lower triangle of the matrix, the diagonal last in each row.
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut columns = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            row_ptrs.push(columns.len());
            for (j, x) in a.row(i) {
                if j <= i {
                    columns.push(j);
                    values.push(x);
                }
            }
            if columns.len() == row_ptrs[i] || columns[columns.len() - 1] != i {
                return None;
            }
        }
        row_ptrs.push(columns.len());
        // `position[j]` is the offset of column j in the row being computed, as in `Ilu0`.
        const ABSENT: usize = usize::MAX;
        let mut position = vec![ABSENT; n];
        for i in 0..n {
            let (begin, last) = (row_ptrs[i], row_ptrs[i + 1] - 1);
            for at in begin..last {
                position[columns[at]] = at;
            }
            // In ascending order of column, so that the elements of this row which a sum reads,
            // those in the columns before k, are already final.
            for at in begin..last {
                let k = columns[at];
                let (k_begin, k_last) = (row_ptrs[k], row_ptrs[k + 1] - 1);
                let mut sum = values[at];
                for other in k_begin..k_last {
                    let mine = position[columns[other]];
                    if mine != ABSENT {
                        sum = sum.mul_sub(values[mine], values[other].conj());
                    }
                }
                let pivot = values[k_last].re();
                values[at] = T::from_parts(sum.re() / pivot, sum.im() / pivot);
            }
            for at in begin..last {
                position[columns[at]] = ABSENT;
            }
            let mut pivot = values[last].re();
            pivot += alpha * pivot;
            for x in &values[begin..last] {
                pivot -= x.abs2();
            }
            // Written so that a pivot that is not a number is refused too.
            if !(pivot > T::Real::ZERO && pivot.is_finite()) {
                return None;
            }
            values[last] = T::from_real(pivot.sqrt());
        }
        if !values.iter().all(|x| x.is_finite()) {
            return None;
        }
        Some(Self {
            row_ptrs,
            columns,
            values,
        })
    }

    /// The number of stored elements of `L`: that of the lower triangle of the matrix, the
    /// diagonal counted, since there is no fill.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// `L`, lower triangular with a real positive diagonal. An element that came out exactly zero
    /// is not stored.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::sparse::{Ic0, SparseMatrix};
    ///
    /// let a = SparseMatrix::tridiagonal(&[-1.0; 3], &[2.0; 4], &[-1.0; 3]);
    /// let l = Ic0::new(&a).unwrap().l();
    /// assert_eq!(l.nnz(), 7);
    /// // No fill in a tridiagonal matrix, so L·Lᵀ is the matrix to rounding.
    /// let product = &l * &l.transpose();
    /// assert!((&product - &a).values().iter().all(|x: &f64| x.abs() < 1e-15));
    /// ```
    #[must_use]
    pub fn l(&self) -> SparseMatrix<T> {
        let n = self.row_ptrs.len() - 1;
        SparseMatrix::from_csr(
            n,
            n,
            self.row_ptrs.clone(),
            self.columns.clone(),
            self.values.clone(),
        )
    }
}

impl<T: Scalar> Preconditioner<T> for Ic0<T> {
    /// `z = L⁻ᴴ·L⁻¹·r`: a forward substitution over the rows of `L`, then a back substitution
    /// with `Lᴴ` in which each element, once known, is taken out of the ones above it through
    /// its row of `L`. Nothing is allocated.
    #[track_caller]
    fn apply(&self, r: &[T], z: &mut [T]) {
        let n = self.row_ptrs.len() - 1;
        check_lengths(n, r, z);
        for i in 0..n {
            let (begin, last) = (self.row_ptrs[i], self.row_ptrs[i + 1] - 1);
            let mut sum = r[i];
            for at in begin..last {
                sum = sum.mul_sub(self.values[at], z[self.columns[at]]);
            }
            let pivot = self.values[last].re();
            z[i] = T::from_parts(sum.re() / pivot, sum.im() / pivot);
        }
        for i in (0..n).rev() {
            let (begin, last) = (self.row_ptrs[i], self.row_ptrs[i + 1] - 1);
            let pivot = self.values[last].re();
            let solved = T::from_parts(z[i].re() / pivot, z[i].im() / pivot);
            z[i] = solved;
            for at in begin..last {
                let j = self.columns[at];
                z[j] = z[j].mul_sub(self.values[at].conj(), solved);
            }
        }
    }
}
