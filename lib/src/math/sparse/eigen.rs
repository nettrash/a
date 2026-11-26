//! A few eigenvalues of a large sparse matrix: the Lanczos and Arnoldi methods with implicit
//! restarts.
//!
//! Both build an orthonormal basis `V` of a Krylov space `v, A·v, A²·v, …` one product with the
//! operator at a time and take the eigenvalues of the small matrix `Vᴴ·A·V` — the Ritz values —
//! as approximations of those of `A`. The extreme ones converge first, so `k` of them are found
//! with a basis of a few times `k` vectors, at the cost of products with `A` and of keeping the
//! basis orthogonal, never of the size of `A` squared. The memory the methods take is the basis
//! of `m` vectors, the vectors they have found and locked (see below) and, for the answer, the
//! products of the operator with those; once the basis is full, an *implicit restart* (Sorensen,
//! SIAM J. Matrix Anal. Appl. 13, 1992) applies the `m − k` unwanted Ritz values as the shifts of
//! implicitly shifted QR
//! steps on the small matrix, which compresses the basis to `k` vectors that keep what was wanted
//! of it, and the basis is grown again from there. The methods follow Lehoucq, Sorensen and
//! Yang's *ARPACK Users' Guide* (SIAM, 1998), but the code is this crate's own.
//!
//! * [`lanczos`] — for a Hermitian (real symmetric, or complex Hermitian) operator: the small
//!   matrix is real, symmetric and tridiagonal, the eigenvalues real, the eigenvectors
//!   orthonormal. Selections [`LanczosWhich`]: the largest, the smallest, the largest in
//!   absolute value, or some of each end.
//! * [`arnoldi`] — for any square operator: the small matrix is upper Hessenberg and the
//!   eigenvalues complex. Selections [`ArnoldiWhich`]: the largest in absolute value, the
//!   largest or smallest real part, the largest or smallest imaginary part. For a real operator
//!   the basis stays real and a pair of complex conjugate eigenvalues is found, or not, as a
//!   whole.
//! * [`lanczos_shift_invert`] and [`arnoldi_shift_invert`] — the eigenvalues nearest a target
//!   `σ`, by either method on `(A − σI)⁻¹` through a [`SparseLu`] of `A − σI`: they are its
//!   largest, and well separated, where the interior or the small end of the spectrum of `A`
//!   converges slowly or not at all. [`lanczos_inverse_with`] and [`arnoldi_inverse_with`]
//!   take the inverse as an operator of the caller's own. A pair is judged by its residual with
//!   `A` itself, not with the inverse, whose size `1/δ` for a target `δ` from an eigenvalue
//!   would set the scale of the test. For complex elements the shift-and-invert Lanczos forms
//!   carry an Arnoldi factorisation, as [`lanczos_inverse_with`] explains: the solves of a
//!   complex Hermitian `A − σI` near an eigenvalue are not Hermitian enough for a real
//!   tridiagonal one. For a matrix that is not normal, the rounding of the solves limits the
//!   pairs other than the nearest once `σ` is near an eigenvalue: see
//!   [`arnoldi_inverse_with`].
//!
//! Each new basis vector is made orthogonal to all the others by classical Gram–Schmidt applied
//! twice (with a further pass when the second still removes much, after Daniel, Gragg, Kaufman
//! and Stewart), so the basis stays orthonormal to working precision. The rotations of the
//! restarts move the columns they keep by an `ε` or so each time, a drift that adds up over
//! hundreds of restarts; what a run locks, and the Ritz vectors it ends with, are made
//! orthonormal again by Gram–Schmidt before they are passed on, so that the eigenvectors
//! returned are of length 1 to working precision however many restarts were made. A Ritz pair
//! `(θ, x)` has converged when the estimate of its residual from the factorisation passes
//! ARPACK's test `‖A·x − θ·x‖ ≤ tol·max(ε^(2/3)·‖A‖, |θ|)`, with ARPACK's floor `ε^(2/3)` made
//! relative to the size of the operator — or when the estimate is at most `ε·‖A‖`, the rounding
//! of the operator's own products, below which no residual can be told from zero, which matters
//! for a tolerance below `ε^(1/3)`. Within a run, converged pairs are kept by the exact-shift
//! strategy — they are never shifts, and a restart keeps more Ritz values than were asked for.
//! A block of the small matrix that splits off — the factorisation broke down and went on from a
//! fresh vector, or a coupling became negligible — is *locked*: its vectors leave the
//! factorisation, which goes on with all of its columns in their complement, and its
//! eigenvalues, exact as far as the arithmetic can tell, stand beside the Ritz values in every
//! decision. An operator with few distinct eigenvalues splits off such blocks again and again,
//! and kept in the basis they would fill it with values that no shift can move.
//!
//! A Krylov space of one starting vector holds only one vector of each eigenspace, so a run
//! sees one copy of an eigenvalue of multiplicity two or more at first. The other copies come
//! from two sources within the run: from rounding, which the full reorthogonalisation keeps at
//! the level of `ε` but which seeds the directions of the other copies, so that the restarts
//! grow them and a run usually converges to every copy its basis has room for (the 2-D
//! Laplacian); and, when the operator has few distinct eigenvalues, from breakdowns: the Krylov
//! space is then an invariant subspace, the run locks it and goes on from a fresh random vector
//! of its complement, which holds the next copy. To catch what a run missed, the methods then
//! lock what it found and start another run from a fresh random vector, the operator restricted
//! to the complement, until a run finds nothing that is wanted more than the least wanted
//! eigenvalue found: these runs find the copies a run had no room for and an extreme eigenvalue
//! it passed over. Last, the vectors of all the runs are put through a Rayleigh–Ritz step with
//! the operator — the eigenpairs of `Zᴴ·A·Z` give the answer, which also turns the locked blocks
//! and the vectors found in a complement into eigenvectors — and each pair is checked by its
//! true residual, from products with the operator, before it is reported.
//!
//! A method never presents a Ritz value that has not converged as an eigenvalue, nor an
//! eigenvalue less wanted than one it has not found: what it reports is, at each end of the
//! selection, the eigenvalues from the most wanted inward up to the first that had not
//! converged, or failed its check, when the budget of restarts ran out. What it cannot do, any
//! more than ARPACK can, is make a Krylov space reach the inside of a spectrum early: a
//! selection of eigenvalues that are interior to it — those of a real matrix nearest the real
//! axis, between the pairs above and below it — may converge to genuine eigenvalues that are not
//! the ones wanted, and so may the selections by the imaginary part of any matrix, or by the
//! real part of a complex one whose eigenvalues crowd the edge of its spectrum, with a small
//! basis or a loose tolerance; shift-and-invert is the method for those. Starting vectors come
//! from a fixed pseudo-random generator, so a call is repeatable and asks nothing of the system.
//!
//! # Examples
//!
//! ```
//! use a::math::sparse::{LanczosWhich, SparseMatrix, lanczos};
//!
//! // The 1-D Laplacian of 100 points: eigenvalues 2 − 2·cos(jπ/101).
//! const N: usize = 100;
//! let a = SparseMatrix::tridiagonal(&[-1.0f64; N - 1], &[2.0; N], &[-1.0; N - 1]);
//! let r = lanczos(&a, 3, LanczosWhich::LargestAlgebraic).unwrap();
//! assert_eq!(r.converged, 3);
//! for (i, &value) in r.values.iter().enumerate() {
//!     let j = (N - 2 + i) as f64;
//!     let exact = 2.0 - 2.0 * (j * core::f64::consts::PI / (N + 1) as f64).cos();
//!     assert!((value - exact).abs() < 1e-12);
//!     assert!(r.residuals[i] < 1e-12);
//! }
//! ```

mod arnoldi;
mod basis;
mod lanczos;

use super::{FnOperator, LinearOperator, SparseLu, SparseMatrix};
use crate::math::complex::{Complex, Float};
use crate::math::matrix::blas::{Op, gemm};
use crate::math::matrix::{Matrix, Scalar};
use basis::{
    Candidate, Original, Outcome, Random, Rule, Run, Score, State, Stop, Verdict, combine,
    combine_complex, complex_norm, eps23, finite, fresh, general_eigen, hermitian_eigen, norm,
    parts, pick, scale_down, unit,
};

/// Which eigenvalues of a Hermitian operator [`lanczos`] looks for.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{LanczosWhich, SparseMatrix, lanczos};
///
/// let a = SparseMatrix::from_diagonal(&[1.0f64, -5.0, 3.0, 2.0, 0.5]);
/// let r = lanczos(&a, 1, LanczosWhich::LargestMagnitude).unwrap();
/// assert!((r.values[0] + 5.0).abs() < 1e-14);
/// let r = lanczos(&a, 3, LanczosWhich::BothEnds).unwrap();
/// assert_eq!(r.values.len(), 3);
/// assert!((r.values[0] + 5.0).abs() < 1e-14 && (r.values[2] - 3.0).abs() < 1e-14);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LanczosWhich {
    /// The largest, towards `+∞` (ARPACK's `LA`).
    LargestAlgebraic,
    /// The smallest, towards `−∞` (`SA`).
    SmallestAlgebraic,
    /// The largest in absolute value (`LM`).
    LargestMagnitude,
    /// Half of them from each end of the spectrum, one more from the top when `k` is odd
    /// (`BE`).
    BothEnds,
}

/// Which eigenvalues of a general operator [`arnoldi`] looks for.
///
/// For a real operator, whose eigenvalues that are not real come in conjugate pairs,
/// [`LargestImaginary`](Self::LargestImaginary) and [`SmallestImaginary`](Self::SmallestImaginary)
/// rank by the absolute value of the imaginary part, as ARPACK's `dnaupd` does: the pairs
/// farthest from the real axis, or the eigenvalues nearest to it. For a complex operator they
/// rank by the imaginary part itself.
///
/// The selections by the imaginary part, and those by the real part of a complex operator
/// whose eigenvalues crowd the edge of its spectrum, can converge to genuine eigenvalues that
/// are not the most wanted, as ARPACK's can, most often with a small basis or a loose
/// tolerance: [`SmallestImaginary`](Self::SmallestImaginary) of a real operator lies inside the
/// spectrum, between the pairs above and below the real axis, and a real Ritz value of a real
/// operator ties with every other at `|Im| = 0`. A larger basis, or shift-and-invert at a target
/// near the wanted eigenvalues, is the remedy.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{ArnoldiWhich, SparseMatrix, arnoldi};
///
/// let a = SparseMatrix::from_diagonal(&[1.0f64, -5.0, 3.0, 2.0, 0.5]);
/// let r = arnoldi(&a, 2, ArnoldiWhich::SmallestReal).unwrap();
/// assert!((r.values[0].re + 5.0).abs() < 1e-14 && (r.values[1].re - 0.5).abs() < 1e-14);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArnoldiWhich {
    /// The largest in absolute value (`LM`).
    LargestMagnitude,
    /// The largest real part (`LR`).
    LargestReal,
    /// The smallest real part (`SR`).
    SmallestReal,
    /// The largest imaginary part (`LI`).
    LargestImaginary,
    /// The smallest imaginary part (`SI`).
    SmallestImaginary,
}

/// The options of the eigenvalue methods: the size of the basis, the tolerance, the budget of
/// restarts and the starting vector.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{EigenOptions, LanczosWhich, SparseMatrix, lanczos_with};
///
/// const N: usize = 200;
/// let a = SparseMatrix::tridiagonal(&[-1.0; N - 1], &[2.0; N], &[-1.0; N - 1]);
/// let opts = EigenOptions::new()
///     .with_basis(30)
///     .with_tolerance(1e-10)
///     .with_max_restarts(500)
///     .with_start(vec![1.0; N]);
/// let r = lanczos_with(&a, 4, LanczosWhich::LargestAlgebraic, &opts).unwrap();
/// assert_eq!(r.converged, 4);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct EigenOptions<T: Scalar> {
    /// The number `m` of vectors in the basis, `None` for `max(2k + 1, 20)`. It is raised to
    /// `k + 1` for Lanczos and `k + 2` for Arnoldi if it is smaller (a restart needs a shift,
    /// and Arnoldi room for a conjugate pair), and lowered to `n`.
    pub basis: Option<usize>,
    /// The relative tolerance of the convergence test, `ε` of the type by default. It should not
    /// be below the relative accuracy of the operator's products — of a solve, for an inverse
    /// given as an operator — which no residual can beat.
    pub tolerance: T::Real,
    /// The most restarts a call may make, over all its runs; 300 by default.
    pub max_restarts: usize,
    /// The vector the first run starts from, `None` for one of the deterministic generator.
    pub start: Option<Vec<T>>,
}

impl<T: Scalar> EigenOptions<T> {
    /// The default options: the basis of `max(2k + 1, 20)` vectors, the tolerance `ε`, 300
    /// restarts, and a pseudo-random starting vector from a fixed seed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            basis: None,
            tolerance: T::Real::EPSILON,
            max_restarts: 300,
            start: None,
        }
    }

    /// Sets [`basis`](Self::basis).
    #[must_use]
    pub fn with_basis(mut self, basis: usize) -> Self {
        self.basis = Some(basis);
        self
    }

    /// Sets [`tolerance`](Self::tolerance).
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: T::Real) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets [`max_restarts`](Self::max_restarts).
    #[must_use]
    pub fn with_max_restarts(mut self, max_restarts: usize) -> Self {
        self.max_restarts = max_restarts;
        self
    }

    /// Sets [`start`](Self::start).
    #[must_use]
    pub fn with_start(mut self, start: Vec<T>) -> Self {
        self.start = Some(start);
        self
    }
}

impl<T: Scalar> Default for EigenOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// What [`lanczos`] found: the eigenvalues that converged, ascending, their orthonormal
/// eigenvectors, the residual of each pair, and what it cost.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{LanczosWhich, SparseMatrix, lanczos};
///
/// let a = SparseMatrix::from_diagonal(&[4.0f64, 1.0, 3.0, 2.0]);
/// let r = lanczos(&a, 2, LanczosWhich::SmallestAlgebraic).unwrap();
/// assert_eq!(r.converged, 2);
/// assert!((r.values[0] - 1.0).abs() < 1e-14 && (r.values[1] - 2.0).abs() < 1e-14);
/// // The eigenvector of 1 is the second unit vector, up to its sign.
/// assert!((r.vectors[(1, 0)].abs() - 1.0).abs() < 1e-14);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LanczosResult<T: Scalar> {
    /// The eigenvalues that converged, ascending.
    pub values: Vec<T::Real>,
    /// Their eigenvectors, orthonormal, as the columns of an `n × converged` matrix, in the order
    /// of `values`.
    pub vectors: Matrix<T>,
    /// `‖A·v − λ·v‖` of each pair, from a product with the operator after the iteration (for the
    /// shift-and-invert forms, a product with `A` itself).
    pub residuals: Vec<T::Real>,
    /// How many pairs converged, the length of `values`. Fewer than `min(k, n)` when the budget
    /// of restarts ran out before every pair asked for had converged — or before the search for
    /// copies had confirmed that no eigenvalue was missed —, when the operator gave values that
    /// are not finite, or when a pair failed the check of its true residual after the
    /// iteration. For the shift-and-invert forms also, with restarts left, when the pairs still
    /// open do not pass the test with `A` and no restart can change that: either the basis spans
    /// all of the space that is left, so that the factorisation is exact, or every pair still
    /// open has converged on the inverse to the rounding of its own size, so that no restart can
    /// move its vector (near an eigenvalue of an `A` that is not normal; see
    /// [`arnoldi_inverse_with`]). The pairs reported are then the most wanted ones: at each end of the
    /// selection, from the most wanted inward, up to the first that had not converged or not
    /// passed, so that no eigenvalue less wanted than one not found is reported.
    pub converged: usize,
    /// The restarts made, over all the runs.
    pub restarts: usize,
    /// The applications of the operator — for the shift-and-invert forms, of `(A − σI)⁻¹`, the
    /// products with `A` that judge and check the answer not counted.
    pub applications: usize,
}

/// What [`arnoldi`] found: the eigenvalues that converged, the most wanted first, their
/// eigenvectors of length 1, the residual of each pair, and what it cost. For a real operator a
/// pair of complex conjugate eigenvalues is never split: when the `k`-th eigenvalue is one of a
/// pair, its conjugate is there too, and `k + 1` are returned.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{ArnoldiWhich, SparseMatrix, arnoldi};
///
/// // A rotation by a quarter turn and a stretch: eigenvalues ±i and 2.
/// let a = SparseMatrix::from_coo(3, 3, &[(0, 1, -1.0f64), (1, 0, 1.0), (2, 2, 2.0)]);
/// let r = arnoldi(&a, 1, ArnoldiWhich::LargestImaginary).unwrap();
/// // The pair of ±i together.
/// assert_eq!(r.converged, 2);
/// assert!((r.values[0].im - 1.0).abs() < 1e-14 && (r.values[1].im + 1.0).abs() < 1e-14);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ArnoldiResult<T: Scalar> {
    /// The eigenvalues that converged, the most wanted first; a conjugate pair of a real
    /// operator together, the member with the positive imaginary part first.
    pub values: Vec<Complex<T::Real>>,
    /// Their eigenvectors, each of length 1, as the columns of an `n × converged` matrix, in the
    /// order of `values`.
    pub vectors: Matrix<Complex<T::Real>>,
    /// `‖A·v − λ·v‖` of each pair, from a product with the operator after the iteration (for the
    /// shift-and-invert forms, a product with `A` itself).
    pub residuals: Vec<T::Real>,
    /// How many pairs converged, the length of `values`: as in [`LanczosResult::converged`],
    /// fewer than `min(k, n)` — counting the conjugate that completes a pair — when the budget
    /// of restarts ran out before every pair asked for had converged or the search for copies
    /// had confirmed what was found, when the operator gave values that are not finite, when
    /// a pair failed its check, or, for the shift-and-invert forms, when the pairs still open do
    /// not pass the test with `A` and no restart can change that — the basis spans all of the
    /// space that is left, or every open pair has converged on the inverse to the rounding of
    /// its own size — so that the call ends with restarts left (see [`arnoldi_inverse_with`] for
    /// why such pairs fail near an eigenvalue); the pairs reported
    /// are then the most wanted ones, up to the first that had not converged or not passed.
    pub converged: usize,
    /// The restarts made, over all the runs.
    pub restarts: usize,
    /// The applications of the operator — for the shift-and-invert forms, of `(A − σI)⁻¹`, the
    /// products with `A` that judge and check the answer not counted.
    pub applications: usize,
}

/// The `k` eigenvalues of the Hermitian operator `a` that `which` asks for, and their
/// eigenvectors, by the implicitly restarted Lanczos method with the default
/// [`EigenOptions`]: see [`lanczos_with`].
///
/// # Examples
///
/// ```
/// use a::math::sparse::{LanczosWhich, SparseMatrix, lanczos};
///
/// let a = SparseMatrix::tridiagonal(&[-1.0; 49], &[2.0; 50], &[-1.0; 49]);
/// let r = lanczos(&a, 2, LanczosWhich::SmallestAlgebraic).unwrap();
/// assert_eq!(r.converged, 2);
/// assert!(r.residuals.iter().all(|&x| x < 1e-12));
/// ```
///
/// # Panics
///
/// Panics if the operator is not square.
#[must_use]
#[track_caller]
pub fn lanczos<T: Scalar, A: LinearOperator<T> + ?Sized>(
    a: &A,
    k: usize,
    which: LanczosWhich,
) -> Option<LanczosResult<T>> {
    lanczos_with(a, k, which, &EigenOptions::new())
}

/// The `k` eigenvalues of the Hermitian operator `a` that `which` asks for, and their
/// orthonormal eigenvectors, by the implicitly restarted Lanczos method with the options
/// `opts`.
///
/// The operator must be Hermitian (for real elements, symmetric); it is known only through its
/// products, so this is not checked, but the true residual of every pair is, and a pair the
/// method made of an operator that is not Hermitian fails that check rather than being
/// reported. `k` larger than `n` is taken as `n`: the basis is then the whole space, and every
/// eigenvalue is found. `k = 0` and an empty operator give an empty result.
///
/// `None` if the tolerance is not a positive finite number, or the starting vector is zero or
/// not finite. An operator that gives values that are not finite ends the iteration: the pairs
/// that earlier runs had found and that check out are reported, the most wanted of them up to
/// the line the failed run was to confirm — none if the first run failed.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::sparse::{EigenOptions, LanczosWhich, SparseMatrix, lanczos_with};
///
/// // A complex Hermitian tridiagonal matrix: its eigenvalues are real.
/// let n = 60;
/// let off = vec![Complex::new(0.0, -1.0); n - 1];
/// let up: Vec<_> = off.iter().map(|z: &Complex<f64>| z.conj()).collect();
/// let a = SparseMatrix::tridiagonal(&off, &vec![Complex::new(2.0, 0.0); n], &up);
/// let opts = EigenOptions::new().with_tolerance(1e-12);
/// let r = lanczos_with(&a, 2, LanczosWhich::LargestAlgebraic, &opts).unwrap();
/// let exact = 2.0 + 2.0 * (core::f64::consts::PI / (n + 1) as f64).cos();
/// assert!((r.values[1] - exact).abs() < 1e-10);
/// ```
///
/// # Panics
///
/// Panics if the operator is not square, or the starting vector is not of length `n`.
#[must_use]
#[track_caller]
pub fn lanczos_with<T: Scalar, A: LinearOperator<T> + ?Sized>(
    a: &A,
    k: usize,
    which: LanczosWhich,
    opts: &EigenOptions<T>,
) -> Option<LanczosResult<T>> {
    let op = Borrowed(a);
    let (call, start) = Call::prepare(&op, Method::Lanczos, k, lanczos_rule(which), opts, None)?;
    let found = search(&call, opts.max_restarts, start);
    Some(finish_lanczos(&call, found))
}

/// The `k` eigenvalues of the square operator `a` that `which` asks for, and their eigenvectors,
/// by the implicitly restarted Arnoldi method with the default [`EigenOptions`]: see
/// [`arnoldi_with`].
///
/// # Examples
///
/// ```
/// use a::math::sparse::{ArnoldiWhich, SparseMatrix, arnoldi};
///
/// // An upper bidiagonal matrix: its eigenvalues are its diagonal.
/// let n = 40;
/// let diagonal: Vec<f64> = (1..=n).map(|i| i as f64).collect();
/// let a = SparseMatrix::tridiagonal(&vec![0.0; n - 1], &diagonal, &vec![1.0; n - 1]);
/// let r = arnoldi(&a, 2, ArnoldiWhich::LargestReal).unwrap();
/// assert!((r.values[0].re - 40.0).abs() < 1e-10 && (r.values[1].re - 39.0).abs() < 1e-10);
/// ```
///
/// # Panics
///
/// Panics if the operator is not square.
#[must_use]
#[track_caller]
pub fn arnoldi<T: Scalar, A: LinearOperator<T> + ?Sized>(
    a: &A,
    k: usize,
    which: ArnoldiWhich,
) -> Option<ArnoldiResult<T>> {
    arnoldi_with(a, k, which, &EigenOptions::new())
}

/// The `k` eigenvalues of the square operator `a` that `which` asks for, and their eigenvectors,
/// by the implicitly restarted Arnoldi method with the options `opts`.
///
/// For a real operator the factorisation is carried in real arithmetic: a pair of conjugate
/// shifts is one Francis double-shift step, and a conjugate pair of eigenvalues is wanted,
/// shifted, converged and returned as a whole, so `k + 1` values come back when the `k`-th is
/// one of a pair. `k` larger than `n` is taken as `n`; `k = 0` and an empty operator give an
/// empty result.
///
/// `None` if the tolerance is not a positive finite number, or the starting vector is zero or
/// not finite. An operator that gives values that are not finite ends the iteration: the pairs
/// that earlier runs had found and that check out are reported, the most wanted of them up to
/// the line the failed run was to confirm — none if the first run failed.
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::sparse::{ArnoldiWhich, EigenOptions, SparseMatrix, arnoldi_with};
///
/// // A complex diagonal matrix with a superdiagonal: the eigenvalues are the diagonal.
/// let n = 30;
/// let diagonal: Vec<Complex<f64>> = (0..n).map(|j| Complex::new(1.0, j as f64)).collect();
/// let a = SparseMatrix::tridiagonal(
///     &vec![Complex::new(0.0, 0.0); n - 1],
///     &diagonal,
///     &vec![Complex::new(0.5, 0.0); n - 1],
/// );
/// let opts = EigenOptions::new().with_tolerance(1e-12);
/// let r = arnoldi_with(&a, 1, ArnoldiWhich::LargestImaginary, &opts).unwrap();
/// assert!((r.values[0] - Complex::new(1.0, 29.0)).abs() < 1e-9);
/// ```
///
/// # Panics
///
/// Panics if the operator is not square, or the starting vector is not of length `n`.
#[must_use]
#[track_caller]
pub fn arnoldi_with<T: Scalar, A: LinearOperator<T> + ?Sized>(
    a: &A,
    k: usize,
    which: ArnoldiWhich,
    opts: &EigenOptions<T>,
) -> Option<ArnoldiResult<T>> {
    let op = Borrowed(a);
    let rule = arnoldi_rule::<T>(which);
    let (call, start) = Call::prepare(&op, Method::Arnoldi, k, rule, opts, None)?;
    let found = search(&call, opts.max_restarts, start);
    Some(finish_arnoldi(&call, found))
}

/// The `k` eigenvalues of the Hermitian matrix `a` nearest the real target `sigma`, and their
/// eigenvectors, by shift-and-invert Lanczos with the default [`EigenOptions`]: see
/// [`lanczos_shift_invert_with`].
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, lanczos_shift_invert};
///
/// // The smallest eigenvalues of a positive definite matrix: the target 0.
/// let a = SparseMatrix::tridiagonal(&[-1.0f64; 99], &[2.0; 100], &[-1.0; 99]);
/// let r = lanczos_shift_invert(&a, 0.0, 2).unwrap();
/// let smallest = 2.0 - 2.0 * (core::f64::consts::PI / 101.0).cos();
/// assert!((r.values[0] - smallest).abs() < 1e-14);
/// ```
///
/// # Panics
///
/// Panics if the matrix is not square.
#[must_use]
#[track_caller]
pub fn lanczos_shift_invert<T: Scalar>(
    a: &SparseMatrix<T>,
    sigma: T::Real,
    k: usize,
) -> Option<LanczosResult<T>> {
    lanczos_shift_invert_with(a, sigma, k, &EigenOptions::new())
}

/// The `k` eigenvalues of the Hermitian matrix `a` nearest the real target `sigma`, and their
/// orthonormal eigenvectors: the Lanczos method on `(A − σI)⁻¹`, applied by a [`SparseLu`] of
/// `A − σI`, whose largest eigenvalues `θ = 1/(λ − σ)` in absolute value belong to the
/// eigenvalues `λ` of `A` nearest `σ`, on either side. See [`lanczos_inverse_with`] for how
/// convergence is judged — by the residual of each pair with `A` — what is reported, and why
/// for complex elements the factorisation is Arnoldi's; here `‖A‖` in that test is bounded by
/// `√(‖A‖₁·‖A‖∞)`.
///
/// `None` if `sigma` is not finite, if `A − σI` is singular — `σ` an eigenvalue to working
/// precision — or for the reasons of [`lanczos_with`]. A target so near an eigenvalue that the
/// factorisation is singular to working precision but not exactly so gives the pairs whose
/// residual with `A` passes, and `converged < k` if the others do not.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{EigenOptions, SparseMatrix, lanczos_shift_invert_with};
///
/// let a = SparseMatrix::from_diagonal(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
/// let opts = EigenOptions::new().with_basis(6);
/// let r = lanczos_shift_invert_with(&a, 4.4, 2, &opts).unwrap();
/// assert!((r.values[0] - 4.0).abs() < 1e-14 && (r.values[1] - 5.0).abs() < 1e-14);
/// // An eigenvalue as the target: `A − σI` is singular.
/// assert!(lanczos_shift_invert_with(&a, 3.0, 2, &opts).is_none());
/// ```
///
/// # Panics
///
/// Panics if the matrix is not square, or the starting vector is not of length `n`.
#[must_use]
#[track_caller]
pub fn lanczos_shift_invert_with<T: Scalar>(
    a: &SparseMatrix<T>,
    sigma: T::Real,
    k: usize,
    opts: &EigenOptions<T>,
) -> Option<LanczosResult<T>> {
    assert_eq!(a.rows(), a.cols(), "the matrix is not square");
    if !sigma.is_finite() {
        return None;
    }
    let size = matrix_norm(a);
    let lu = shifted_lu(a, T::from_real(sigma))?;
    let inverse = inverse_operator(&lu, a.rows());
    lanczos_inverse(a, &inverse, k, opts, Some(size))
}

/// The eigenvalues of the Hermitian operator `a` nearest a target `σ`, and their orthonormal
/// eigenvectors, by the Lanczos method on the operator `inverse`, which applies `(A − σI)⁻¹` —
/// by a factorisation of the caller's choice, a direct solver or an iterative one solved
/// tightly enough.
///
/// The method finds the `k` largest eigenvalues in absolute value of `inverse` and their
/// eigenvectors, which are those of `A`. A Ritz pair has converged when its vector `x` is an
/// eigenvector of `A` to the tolerance: its eigenvalue is the Rayleigh quotient `λ = xᴴ·A·x`,
/// and its residual with `A` must pass `‖A·x − λ·x‖ ≤ tol·max(|λ|, ε^(2/3)·‖A‖)` with room for
/// the rounding of that product, `8·√n·ε·‖A‖` — the test of [`lanczos_with`], made with `A`
/// itself, so that what the tolerance asks of the answer does not depend on how near `σ` an
/// eigenvalue lies, and a pair of the inverse that is not one of `A`, from a solve that is
/// wrong, never converges. A Ritz value `1/δ` of an eigenvalue `δ` from the target is of a size
/// that would spoil the others through the rotations of every restart: once converged it is
/// locked, and the run goes on in its complement. The Rayleigh–Ritz step at the end is made with
/// `A` too, whose small matrix is well scaled where that of the inverse is not, and each pair is
/// checked with `A`. `‖A‖` here comes from a few steps of the power method with `a`, which
/// bound it from below. `σ` itself is not needed: any operator with the eigenvectors of `A` and
/// its wanted eigenvalues the largest serves. The result reports `λ`, `x` and that residual.
/// The eigenvalues reported are eigenvalues of `A`, by their residuals, whatever `inverse`
/// does; that they are the nearest `σ` rests on `inverse` applying `(A − σI)⁻¹`.
///
/// For complex elements the factorisation is an Arnoldi one, `B·V = V·H + f·eₘᵀ` with `H` full
/// and `B` the operator `inverse`, restarted by the single-shift QR steps of [`arnoldi_with`];
/// the search for what was missed and the answer are those of a Hermitian operator, the last a
/// Rayleigh–Ritz step with `A` made Hermitian, so the eigenvalues are real and the vectors
/// orthonormal as for real elements. The solves of a complex Hermitian `A − σI` are not
/// Hermitian: the rounding of the small pivot of a nearly singular `A − σI` is a complex number,
/// and the largest eigenvalue of the computed inverse, `1/(δ + η)` with `η` of the order of
/// `ε·‖A‖`, has an imaginary part of relative size about `ε·‖A‖/δ` (measured: 0.005 to 0.13
/// times it, for `δ` from `10⁻⁴` to `10⁻¹⁰·‖A‖`). The real tridiagonal `T` of a Lanczos
/// factorisation has no room for what that does to the products: with it, even the nearest
/// pair never passed the test with `A` once `δ` was below about `10⁻⁵·‖A‖`. In real arithmetic
/// the rounding of a pivot is real, the solves stay symmetric to a few `ε`, and the real forms
/// keep the Lanczos factorisation. `A` being normal, the pairs other than the nearest are not
/// limited near an eigenvalue as those of [`arnoldi_inverse_with`] are for a matrix that is not.
///
/// `None` for the reasons of [`lanczos_with`].
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::sparse::{EigenOptions, FnOperator, SparseMatrix, lanczos_inverse_with};
///
/// let n = 50;
/// let a = SparseMatrix::tridiagonal(&vec![-1.0; n - 1], &vec![2.0; n], &vec![-1.0; n - 1]);
/// // `(A − σI)⁻¹` through the dense LU, for σ = 1.1.
/// let shifted = SparseMatrix::tridiagonal(&vec![-1.0; n - 1], &vec![0.9; n], &vec![-1.0; n - 1]);
/// let lu = shifted.to_matrix().lu();
/// let inverse = FnOperator::new(n, n, |x: &[f64], y: &mut [f64]| {
///     let b = Matrix::from_column_major(n, 1, x.to_vec());
///     y.copy_from_slice(lu.solve(&b).unwrap().as_slice());
/// });
/// let r = lanczos_inverse_with(&a, &inverse, 1, &EigenOptions::new()).unwrap();
/// // The eigenvalue nearest 1.1: 2 − 2·cos(jπ/51) for j = 18.
/// let nearest = 2.0 - 2.0 * (18.0 * core::f64::consts::PI / 51.0).cos();
/// assert!((r.values[0] - nearest).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if the operators are not square and of the same size, or the starting vector is not
/// of length `n`.
#[must_use]
#[track_caller]
pub fn lanczos_inverse_with<T: Scalar, A, B>(
    a: &A,
    inverse: &B,
    k: usize,
    opts: &EigenOptions<T>,
) -> Option<LanczosResult<T>>
where
    A: LinearOperator<T> + ?Sized,
    B: LinearOperator<T> + ?Sized,
{
    lanczos_inverse(a, inverse, k, opts, None)
}

/// [`lanczos_inverse_with`], with a bound of `‖A‖` when it is known, and the estimate of the
/// power method when it is not.
#[track_caller]
fn lanczos_inverse<T: Scalar, A, B>(
    a: &A,
    inverse: &B,
    k: usize,
    opts: &EigenOptions<T>,
    size: Option<T::Real>,
) -> Option<LanczosResult<T>>
where
    A: LinearOperator<T> + ?Sized,
    B: LinearOperator<T> + ?Sized,
{
    same_size(a, inverse);
    let op = Borrowed(inverse);
    let a = Borrowed(a);
    let rule = Rule::One(Score::Magnitude);
    let (call, start) = Call::prepare(&op, Method::Lanczos, k, rule, opts, None)?;
    let original = Original {
        operator: &a,
        norm: size.unwrap_or_else(|| norm_estimate(&a)),
    };
    let call = Call {
        // The solves of a complex Hermitian `A − σI` near an eigenvalue are not Hermitian, and
        // the real tridiagonal `T` cannot hold what they do: see `lanczos_inverse_with`. The
        // Arnoldi factorisation can; the search and the answer stay those of a Hermitian
        // operator.
        method: if T::IS_COMPLEX {
            Method::Arnoldi
        } else {
            Method::Lanczos
        },
        original: Some(&original),
        ..call
    };
    let found = search(&call, opts.max_restarts, start);
    Some(finish_lanczos(&call, found))
}

/// The `k` eigenvalues of the square matrix `a` nearest the target `sigma`, and their
/// eigenvectors, by shift-and-invert Arnoldi with the default [`EigenOptions`]: see
/// [`arnoldi_shift_invert_with`].
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, arnoldi_shift_invert};
///
/// let n = 60;
/// let diagonal: Vec<f64> = (1..=n).map(|i| i as f64).collect();
/// let a = SparseMatrix::tridiagonal(&vec![0.0; n - 1], &diagonal, &vec![1.0; n - 1]);
/// let r = arnoldi_shift_invert(&a, 20.3, 1).unwrap();
/// assert!((r.values[0].re - 20.0).abs() < 1e-10);
/// ```
///
/// # Panics
///
/// Panics if the matrix is not square.
#[must_use]
#[track_caller]
pub fn arnoldi_shift_invert<T: Scalar>(
    a: &SparseMatrix<T>,
    sigma: T,
    k: usize,
) -> Option<ArnoldiResult<T>> {
    arnoldi_shift_invert_with(a, sigma, k, &EigenOptions::new())
}

/// The `k` eigenvalues of the square matrix `a` nearest the target `sigma`, and their
/// eigenvectors: the Arnoldi method on `(A − σI)⁻¹`, applied by a [`SparseLu`] of `A − σI`. See
/// [`arnoldi_inverse_with`] for how convergence is judged — by the residual of each pair with
/// `A` — what is reported, and why near an eigenvalue of a matrix that is not normal only the
/// nearest pair may come back at the default tolerance; here `‖A‖` in that test is bounded by
/// `√(‖A‖₁·‖A‖∞)`. The target
/// is of the element type: for a real matrix and a target off the real axis, convert the matrix
/// to complex elements first.
///
/// `None` if `sigma` is not finite, if `A − σI` is singular, or for the reasons of
/// [`arnoldi_with`].
///
/// # Examples
///
/// ```
/// use a::math::complex::Complex;
/// use a::math::sparse::{EigenOptions, SparseMatrix, arnoldi_shift_invert_with};
///
/// // A real matrix with the eigenvalues 3 ± 0.5i and 5, 6, 7.
/// let a = SparseMatrix::from_coo(5, 5, &[
///     (0, 0, 3.0), (0, 1, -0.5), (1, 0, 0.5), (1, 1, 3.0),
///     (2, 2, 5.0), (3, 3, 6.0), (4, 4, 7.0), (0, 4, 1.0),
/// ]);
/// let r = arnoldi_shift_invert_with(&a, 0.0, 1, &EigenOptions::new()).unwrap();
/// // The nearest to 0 is a pair, which comes as a whole.
/// assert_eq!(r.converged, 2);
/// assert!((r.values[0] - Complex::new(3.0, 0.5)).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if the matrix is not square, or the starting vector is not of length `n`.
#[must_use]
#[track_caller]
pub fn arnoldi_shift_invert_with<T: Scalar>(
    a: &SparseMatrix<T>,
    sigma: T,
    k: usize,
    opts: &EigenOptions<T>,
) -> Option<ArnoldiResult<T>> {
    assert_eq!(a.rows(), a.cols(), "the matrix is not square");
    if !sigma.is_finite() {
        return None;
    }
    let size = matrix_norm(a);
    let lu = shifted_lu(a, sigma)?;
    let inverse = inverse_operator(&lu, a.rows());
    arnoldi_inverse(a, &inverse, k, opts, Some(size))
}

/// The eigenvalues of the square operator `a` nearest a target `σ`, and their eigenvectors, by
/// the Arnoldi method on the operator `inverse`, which applies `(A − σI)⁻¹` by a factorisation
/// of the caller's choice.
///
/// The method finds the `k` largest eigenvalues in absolute value of `inverse` and their
/// eigenvectors, which are those of `A`. A Ritz pair has converged when its vector `x` of
/// length 1 is an eigenvector of `A` to the tolerance, as in [`lanczos_inverse_with`]: its
/// eigenvalue is the Rayleigh quotient `λ = xᴴ·A·x`, and its residual with `A` must pass
/// `‖A·x − λ·x‖ ≤ tol·max(|λ|, ε^(2/3)·‖A‖)` with room for the rounding of that product,
/// `8·√n·ε·‖A‖`, `‖A‖` from a few steps of the power method with `a`. In a run that works in the
/// complement of vectors found before, the residual is that of the operator restricted to the
/// complement: for an `A` that is not normal its eigenvectors there are Schur vectors, which
/// the Rayleigh–Ritz step with `A` at the end turns into eigenvectors. For a real operator a
/// conjugate pair stays together, as in [`arnoldi_with`]. The result reports `λ`, `x` and that
/// residual; that the eigenvalues are the nearest `σ` rests on `inverse` applying
/// `(A − σI)⁻¹`.
///
/// Near an eigenvalue, for an `A` that is not normal, the rounding of the solves limits the
/// pairs other than the nearest, in every precision. A vector orthogonal to the eigenvector
/// nearest `σ` still has a component along its left eigenvector, which differs from it, so a
/// solve gives it a component `1/δ` times larger along the nearest eigenvector, `δ` the
/// distance of `σ` from that eigenvalue, and the rounding of that component leaves the pair
/// `j` with a residual with `A` of about `ε·‖A‖·|λⱼ − σ|/δ`: no method that applies the inverse
/// in floating point removes it, and ARPACK's pairs carry it too. Once it is above the room of
/// the test — from about `δ = 10⁻³·‖A‖` on — only the nearest pair comes back at the default
/// tolerance `ε`, `converged` is less than `k`, and the call ends, with restarts left, once the
/// others have converged on the inverse as far as they can. On 20 random sparse matrices of
/// order 80 and `k = 3`, the calls came back complete 18, 9 and 0 times in `f64` at
/// `δ = 10⁻³`, `10⁻⁴` and `10⁻⁶·‖A‖` (15, 3, 0 in `Complex<f64>`; 20, 10, 0 in `f32`; 19, 3, 0
/// in `Complex<f32>`), with the nearest eigenvalue first, and right, every time. A tolerance of
/// `10·ε·‖A‖/δ` — `δ = |λ₁ − σ|`, from the nearest pair of a first call — brings the others
/// back within it: every call complete on the same matrices, for `δ` from `10⁻³` to
/// `10⁻⁸·‖A‖` in double precision and to `10⁻⁵·‖A‖` in single, below which that tolerance is
/// no longer small. A normal `A` — a Hermitian one among them — has no such limit: a vector
/// orthogonal to its nearest eigenvector has no component along it.
///
/// `None` for the reasons of [`arnoldi_with`].
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::sparse::{ArnoldiWhich, EigenOptions, FnOperator, SparseMatrix, arnoldi_inverse_with};
///
/// let n = 40;
/// let diagonal: Vec<f64> = (1..=n).map(|i| i as f64).collect();
/// let a = SparseMatrix::tridiagonal(&vec![0.0; n - 1], &diagonal, &vec![1.0; n - 1]);
/// // `(A − 10.2·I)⁻¹` by the dense LU.
/// let shifted: Vec<f64> = diagonal.iter().map(|d| d - 10.2).collect();
/// let s = SparseMatrix::tridiagonal(&vec![0.0; n - 1], &shifted, &vec![1.0; n - 1]);
/// let lu = s.to_matrix().lu();
/// let inverse = FnOperator::new(n, n, |x: &[f64], y: &mut [f64]| {
///     let b = Matrix::from_column_major(n, 1, x.to_vec());
///     y.copy_from_slice(lu.solve(&b).unwrap().as_slice());
/// });
/// let r = arnoldi_inverse_with(&a, &inverse, 2, &EigenOptions::new()).unwrap();
/// assert!((r.values[0].re - 10.0).abs() < 1e-10 && (r.values[1].re - 11.0).abs() < 1e-10);
/// ```
///
/// # Panics
///
/// Panics if the operators are not square and of the same size, or the starting vector is not
/// of length `n`.
#[must_use]
#[track_caller]
pub fn arnoldi_inverse_with<T: Scalar, A, B>(
    a: &A,
    inverse: &B,
    k: usize,
    opts: &EigenOptions<T>,
) -> Option<ArnoldiResult<T>>
where
    A: LinearOperator<T> + ?Sized,
    B: LinearOperator<T> + ?Sized,
{
    arnoldi_inverse(a, inverse, k, opts, None)
}

/// [`arnoldi_inverse_with`], with a bound of `‖A‖` when it is known, and the estimate of the
/// power method when it is not.
#[track_caller]
fn arnoldi_inverse<T: Scalar, A, B>(
    a: &A,
    inverse: &B,
    k: usize,
    opts: &EigenOptions<T>,
    size: Option<T::Real>,
) -> Option<ArnoldiResult<T>>
where
    A: LinearOperator<T> + ?Sized,
    B: LinearOperator<T> + ?Sized,
{
    same_size(a, inverse);
    let op = Borrowed(inverse);
    let a = Borrowed(a);
    let rule = Rule::One(Score::Magnitude);
    let (call, start) = Call::prepare(&op, Method::Arnoldi, k, rule, opts, None)?;
    let original = Original {
        operator: &a,
        norm: size.unwrap_or_else(|| norm_estimate(&a)),
    };
    let call = Call {
        original: Some(&original),
        ..call
    };
    let found = search(&call, opts.max_restarts, start);
    Some(finish_arnoldi(&call, found))
}

/// How many steps of the power method estimate `‖A‖` for an inverse given as an operator.
const POWER_STEPS: usize = 8;

/// A lower bound of `‖A‖`: the largest `‖A·v‖` of a unit `v` met in a few steps of the power
/// method, from a vector of a generator of its own. It is what the convergence test of
/// shift-and-invert scales its floor and its room for rounding by, where nothing else is known
/// of `A`; a matrix's own bound is used where there is one.
fn norm_estimate<T: Scalar>(a: &dyn LinearOperator<T>) -> T::Real {
    let n = a.rows();
    let mut largest = T::Real::ZERO;
    if n == 0 {
        return largest;
    }
    let mut random = Random::other();
    let mut v: Vec<T> = random.vector(n);
    let mut y = vec![T::zero(); n];
    for _ in 0..POWER_STEPS {
        let length = norm(&v);
        if length == T::Real::ZERO || !length.is_finite() {
            break;
        }
        scale_down(&mut v, length);
        a.apply(&v, &mut y);
        let size = norm(&y);
        if !size.is_finite() {
            break;
        }
        largest = largest.max(size);
        core::mem::swap(&mut v, &mut y);
    }
    largest
}

/// `√(‖A‖₁·‖A‖∞)` of a sparse matrix, which bounds `‖A‖₂` from above, or `None` if an element
/// or the bound is not finite. The sums are of the elements divided by the largest absolute
/// value, so that neither they nor their product overflow or underflow whatever the size of the
/// elements: formed from the elements themselves, the product overflowed once `‖A‖` passed
/// `√MAX` (about `10¹⁹` in `f32`, `10¹⁵⁴` in `f64`) and underflowed to zero below
/// `√MIN_POSITIVE`, and the convergence test of shift-and-invert, which it scales, then passed
/// every pair at once or none ever. The bound is zero only for the zero
/// matrix: the largest element adds one to its row and to its column.
fn norm_bound<T: Scalar>(a: &SparseMatrix<T>) -> Option<T::Real> {
    if !finite(a.values()) {
        return None;
    }
    let largest = a.values().iter().fold(T::Real::ZERO, |m, x| m.max(x.abs()));
    if !largest.is_finite() {
        return None;
    }
    if largest == T::Real::ZERO {
        return Some(largest);
    }
    let mut rows = vec![T::Real::ZERO; a.rows()];
    let mut cols = vec![T::Real::ZERO; a.cols()];
    for (i, row) in rows.iter_mut().enumerate() {
        for (j, x) in a.row(i) {
            let v = x.abs() / largest;
            *row += v;
            cols[j] += v;
        }
    }
    let widest = |v: &[T::Real]| v.iter().fold(T::Real::ZERO, |m, &x| m.max(x));
    let bound = largest * (widest(&rows) * widest(&cols)).sqrt();
    bound.is_finite().then_some(bound)
}

/// The `‖A‖` the shift-and-invert drivers scale their test by: [`norm_bound`], or where that is
/// not finite the estimate of the power method — a matrix with elements that are not finite gives
/// products that are not, and no pair passes with it.
fn matrix_norm<T: Scalar>(a: &SparseMatrix<T>) -> T::Real {
    norm_bound(a).unwrap_or_else(|| norm_estimate(a))
}

/// A reference to an operator, sized whatever the operator is, so that it can be used as a
/// `dyn LinearOperator`.
struct Borrowed<'a, A: ?Sized>(&'a A);

impl<T: Scalar, A: LinearOperator<T> + ?Sized> LinearOperator<T> for Borrowed<'_, A> {
    fn rows(&self) -> usize {
        self.0.rows()
    }

    fn cols(&self) -> usize {
        self.0.cols()
    }

    fn apply(&self, x: &[T], y: &mut [T]) {
        self.0.apply(x, y);
    }
}

/// Panics unless `a` and `b` are square and of the same size.
#[track_caller]
fn same_size<T: Scalar, A, B>(a: &A, b: &B)
where
    A: LinearOperator<T> + ?Sized,
    B: LinearOperator<T> + ?Sized,
{
    assert_eq!(a.rows(), a.cols(), "the operator is not square");
    assert!(
        b.rows() == a.rows() && b.cols() == a.cols(),
        "the inverse is not of the size of the operator"
    );
}

/// The LU decomposition of `A − σI`, `None` when it is singular.
fn shifted_lu<T: Scalar>(a: &SparseMatrix<T>, sigma: T) -> Option<SparseLu<T>> {
    let n = a.rows();
    let mut entries: Vec<(usize, usize, T)> = Vec::with_capacity(a.nnz() + n);
    for i in 0..n {
        entries.extend(a.row(i).map(|(j, x)| (i, j, x)));
        entries.push((i, i, -sigma));
    }
    let shifted = SparseMatrix::from_coo(n, n, &entries);
    let lu = SparseLu::new(&shifted);
    (!lu.is_singular()).then_some(lu)
}

/// `x ↦ (A − σI)⁻¹·x` by the decomposition, NaNs when a solve is refused, which ends the
/// iteration.
fn inverse_operator<T: Scalar>(
    lu: &SparseLu<T>,
    n: usize,
) -> FnOperator<impl Fn(&[T], &mut [T]) + '_> {
    FnOperator::new(n, n, move |x: &[T], y: &mut [T]| {
        let b = Matrix::from_column_major(n, 1, x.to_vec());
        match lu.solve(&b) {
            Some(s) => y.copy_from_slice(s.as_slice()),
            None => y.fill(T::from_real(T::Real::NAN)),
        }
    })
}

/// The method of a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    /// Lanczos, for a Hermitian operator.
    Lanczos,
    /// Arnoldi, for any operator.
    Arnoldi,
}

impl Method {
    /// One run of the method.
    fn run<T: Scalar>(self, run: &mut Run<'_, T>, start: &[T]) -> Outcome<T> {
        match self {
            Method::Lanczos => lanczos::run(run, start),
            Method::Arnoldi => arnoldi::run(run, start),
        }
    }
}

/// The rule of a [`LanczosWhich`].
fn lanczos_rule(which: LanczosWhich) -> Rule {
    match which {
        LanczosWhich::LargestAlgebraic => Rule::One(Score::Real),
        LanczosWhich::SmallestAlgebraic => Rule::One(Score::NegReal),
        LanczosWhich::LargestMagnitude => Rule::One(Score::Magnitude),
        LanczosWhich::BothEnds => Rule::BothEnds,
    }
}

/// The rule of an [`ArnoldiWhich`], for elements of the type `T`.
fn arnoldi_rule<T: Scalar>(which: ArnoldiWhich) -> Rule {
    Rule::One(match which {
        ArnoldiWhich::LargestMagnitude => Score::Magnitude,
        ArnoldiWhich::LargestReal => Score::Real,
        ArnoldiWhich::SmallestReal => Score::NegReal,
        ArnoldiWhich::LargestImaginary if T::IS_COMPLEX => Score::Imaginary,
        ArnoldiWhich::LargestImaginary => Score::AbsImaginary,
        ArnoldiWhich::SmallestImaginary if T::IS_COMPLEX => Score::NegImaginary,
        ArnoldiWhich::SmallestImaginary => Score::NegAbsImaginary,
    })
}

/// A call with its inputs checked.
struct Call<'a, T: Scalar> {
    /// The operator the Krylov spaces are built with.
    operator: &'a dyn LinearOperator<T>,
    /// Lanczos or Arnoldi.
    method: Method,
    /// The order of the operator.
    n: usize,
    /// How many eigenvalues are wanted, at most `n`.
    k: usize,
    /// The size of the basis, `m`.
    basis: usize,
    /// Which.
    rule: Rule,
    /// Whether conjugate pairs are kept together: Arnoldi on a real operator.
    pairs: bool,
    /// Whether the operator is Hermitian, which the search for what was missed uses: true for
    /// Lanczos, and for the complex shift-and-invert Lanczos, which runs the Arnoldi
    /// factorisation on a Hermitian operator.
    hermitian: bool,
    /// The tolerance.
    tolerance: T::Real,
    /// For shift-and-invert, `A`, which the operator of the call is `(A − σI)⁻¹` of: every pair
    /// is judged by its residual with `A`.
    original: Option<&'a Original<'a, T>>,
}

impl<'a, T: Scalar> Call<'a, T> {
    /// The call and its starting vector, of length 1, if one was given; `None` for a tolerance
    /// that is not a positive finite number or a starting vector that is not finite or is zero.
    #[track_caller]
    fn prepare(
        operator: &'a dyn LinearOperator<T>,
        method: Method,
        k: usize,
        rule: Rule,
        opts: &EigenOptions<T>,
        original: Option<&'a Original<'a, T>>,
    ) -> Option<(Self, Option<Vec<T>>)> {
        let n = operator.rows();
        assert_eq!(n, operator.cols(), "the operator is not square");
        let tolerance = opts.tolerance;
        if !(tolerance > T::Real::ZERO && tolerance.is_finite()) {
            return None;
        }
        let start = match &opts.start {
            Some(s) => {
                assert_eq!(s.len(), n, "the starting vector has the wrong length");
                let mut w = s.clone();
                if !finite(&w) {
                    return None;
                }
                // Only the direction matters: divided by its largest part first, a vector whose
                // elements are finite but whose norm overflows (every element the largest number
                // of the type) is still a starting vector.
                let largest = w
                    .iter()
                    .fold(T::Real::ZERO, |m, x| m.max(x.re().abs()).max(x.im().abs()));
                if largest > T::Real::ZERO {
                    scale_down(&mut w, largest);
                }
                if n > 0 && !unit(&[], &[], &mut w) {
                    return None;
                }
                Some(w)
            }
            None => None,
        };
        let k = k.min(n);
        let least = k + if method == Method::Lanczos { 1 } else { 2 };
        let basis = opts.basis.unwrap_or((2 * k + 1).max(20)).max(least).min(n);
        let call = Call {
            operator,
            method,
            n,
            k,
            basis,
            rule,
            pairs: method == Method::Arnoldi && !T::IS_COMPLEX,
            hermitian: method == Method::Lanczos,
            tolerance,
            original,
        };
        Some((call, start))
    }
}

/// What the runs of a call found: orthonormal vectors spanning an invariant subspace (to the
/// tolerance), the eigenvalues of the operator on it, the frontier, and the counters.
struct Found<T: Scalar> {
    /// The vectors, column after column.
    vectors: Vec<T>,
    /// The eigenvalues of the operator on their span.
    values: Vec<Complex<T::Real>>,
    /// Ritz values of runs that ended before they had converged or confirmed what was found: a
    /// value found that is not more wanted than one of them is not reported, since an eigenvalue
    /// not found could lie between.
    frontier: Vec<Complex<T::Real>>,
    /// The counters.
    state: State<T::Real>,
}

/// `tol·max(|t|, ε^(2/3)·size) + ε·size`: how much more wanted than the threshold `t` a value
/// must be to count as a new eigenvalue rather than another approximation of the threshold's.
fn margin<R: Float>(tolerance: R, t: R, size: R) -> R {
    tolerance * t.abs().max(eps23::<R>() * size) + R::EPSILON * size
}

/// For each end of the rule, its score and the score of the least wanted of the `k` values found
/// that it takes, which a new value must beat to be wanted; with the rule and the number of
/// values asked for of the runs that look for what was missed. For the largest absolute values of
/// a Hermitian operator, whose Ritz values approach the two ends of the spectrum from inside,
/// those runs watch both ends.
fn thresholds<T: Scalar>(
    call: &Call<'_, T>,
    values: &[Complex<T::Real>],
) -> (Vec<(Score, T::Real)>, Rule, usize) {
    let k = call.k;
    let least = |score: Score, picked: &[usize]| {
        picked
            .iter()
            .map(|&i| score.of(values[i]))
            .fold(T::Real::INFINITY, |a, b| a.min(b))
    };
    match call.rule {
        Rule::One(Score::Magnitude) if call.hermitian => {
            let t = least(Score::Magnitude, &pick(call.rule, values, k, false));
            (
                vec![(Score::Real, t), (Score::NegReal, t)],
                Rule::BothEnds,
                2 * k,
            )
        }
        Rule::One(score) => {
            let t = least(score, &pick(call.rule, values, k, call.pairs));
            (vec![(score, t)], call.rule, k)
        }
        Rule::BothEnds => {
            let picked = pick(Rule::BothEnds, values, k, false);
            let low = k / 2;
            let mut ends = Vec::with_capacity(2);
            if low > 0 {
                ends.push((Score::NegReal, least(Score::NegReal, &picked[..low])));
            }
            ends.push((Score::Real, least(Score::Real, &picked[low..])));
            (ends, Rule::BothEnds, k)
        }
    }
}

/// Whether a run looking for what was missed has seen enough: at every end, the Ritz values from
/// the most wanted inward have converged up to one that is not wanted more than the end's
/// threshold — past that, under the premise every Krylov method rests on, that the most wanted
/// Ritz value approximates the most wanted eigenvalue there is, nothing is. For a Hermitian
/// operator an eigenvalue lies within the estimate of the residual of a Ritz value, so a Ritz
/// value that is not wanted by four times that settles the end before it has converged; the
/// factor keeps a Ritz value that is still on its way, with a large estimate, from settling it.
fn settled<R: Float>(
    ends: &[(Score, R)],
    shown: &[Candidate<R>],
    verdict: Verdict<'_>,
    tolerance: R,
    size: R,
    hermitian: bool,
) -> bool {
    ends.iter().all(|&(score, t)| {
        let mut order: Vec<usize> = (0..shown.len()).collect();
        order.sort_by(|&a, &b| {
            score
                .of(shown[b].0)
                .partial_cmp(&score.of(shown[a].0))
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        let line = t + margin(tolerance, t, size);
        for i in order {
            let (z, bound) = shown[i];
            let s = score.of(z);
            if hermitian && s + R::from_f64(4.0) * bound <= line {
                return true;
            }
            if !verdict(i) {
                return false;
            }
            if s <= line {
                return true;
            }
        }
        false
    })
}

/// The runs of a call: the first from the starting vector, then, when it has converged and
/// while the last found something wanted more than what was found before it, another from a
/// fresh random vector, orthogonal to everything found, which the operator is restricted to the
/// complement of. A run that ends before it has converged, or before it has confirmed what was
/// found, leaves the Ritz values it had not converged as the frontier of what is known.
fn search<T: Scalar>(call: &Call<'_, T>, max_restarts: usize, start: Option<Vec<T>>) -> Found<T> {
    let n = call.n;
    let mut found = Found {
        vectors: Vec::new(),
        values: Vec::new(),
        frontier: Vec::new(),
        state: State {
            restarts_left: max_restarts,
            restarts: 0,
            applications: 0,
            norm: T::Real::ZERO,
            random: Random::new(),
        },
    };
    if n == 0 || call.k == 0 {
        return found;
    }
    let start = match start {
        Some(s) => s,
        None => match fresh(&mut found.state.random, n, &[], &[]) {
            Some(s) => s,
            None => return found,
        },
    };
    let outcome = {
        let mut run = Run {
            operator: call.operator,
            n,
            locked: &[],
            basis: call.basis,
            want: call.k,
            rule: call.rule,
            pairs: call.pairs,
            tolerance: call.tolerance,
            state: &mut found.state,
            judge: None,
            original: call.original,
        };
        call.method.run(&mut run, &start)
    };
    found.vectors.extend(outcome.vectors);
    found.values.extend(outcome.values);
    if outcome.stop != Stop::Converged {
        found.frontier = outcome.frontier;
        return found;
    }
    loop {
        let count = found.vectors.len() / n;
        if count >= n {
            break;
        }
        let (ends, rule, want) = thresholds(call, &found.values);
        let space = n - count;
        let extra = usize::from(call.pairs);
        let basis = call.basis.max(want + 1 + extra).min(space);
        let want = if basis == space {
            want.min(basis)
        } else {
            want.min(basis.saturating_sub(1 + extra))
        };
        if want == 0 {
            break;
        }
        let Some(start) = fresh(&mut found.state.random, n, &[found.vectors.as_slice()], &[])
        else {
            break;
        };
        let tolerance = call.tolerance;
        let hermitian = call.hermitian;
        let judge = |shown: &[Candidate<T::Real>], verdict: Verdict<'_>, size: T::Real| {
            settled(&ends, shown, verdict, tolerance, size, hermitian)
        };
        let outcome = {
            let mut run = Run {
                operator: call.operator,
                n,
                locked: &found.vectors,
                basis,
                want,
                rule,
                pairs: call.pairs,
                tolerance,
                state: &mut found.state,
                judge: Some(&judge),
                original: call.original,
            };
            call.method.run(&mut run, &start)
        };
        if outcome.stop == Stop::Failed {
            // The check could not be made: nothing found at or below the line it was to watch
            // is certain.
            found
                .frontier
                .extend(ends.iter().map(|&(score, t)| score.point(t)));
            break;
        }
        let size = found.state.norm;
        let new = outcome.values.iter().any(|&z| {
            ends.iter()
                .any(|&(score, t)| score.of(z) > t + margin(tolerance, t, size))
        });
        let exhausted = outcome.stop == Stop::Exhausted;
        if exhausted {
            found.frontier.extend(outcome.frontier);
        }
        if !new {
            break;
        }
        found.vectors.extend(outcome.vectors);
        found.values.extend(outcome.values);
        if exhausted {
            break;
        }
    }
    found
}

/// Whether `z`, taken from the end of the selection whose score is `score`, is more wanted than
/// every value of the frontier by more than the margin: then no eigenvalue that a run had not
/// converged to when it ended can be more wanted than `z`.
fn trusted<R: Float>(
    score: Score,
    z: Complex<R>,
    frontier: &[Complex<R>],
    tolerance: R,
    size: R,
) -> bool {
    let s = score.of(z);
    frontier.iter().all(|&u| {
        let t = score.of(u);
        s > t + margin(tolerance, t, size)
    })
}

/// The score of the part `part` of the selection of `rule`: its own score, or for both ends the
/// left end's and the right end's.
fn part_score(rule: Rule, part: usize) -> Score {
    match rule {
        Rule::One(score) => score,
        Rule::BothEnds if part == 0 => Score::NegReal,
        Rule::BothEnds => Score::Real,
    }
}

/// Whether the true residual `residual` of an eigenpair of absolute value `size` of an operator
/// of norm `operator` confirms the convergence test the iteration passed: within four times the
/// tolerance of that test, with room for rounding. The room is `64·√w·ε·‖A‖`, with `w` the
/// larger of the order `n` of the operator and `m·(r + 1)` for a basis of `m` vectors restarted
/// `r` times: what a factorisation attains grows with the length of its inner products and
/// with every rotation `V·Q` of its basis — on the 2-D Laplacian of 10⁵ unknowns, at the
/// tolerance `ε`, past `64ε·‖A‖` after a hundred restarts — and the check is there to catch a
/// method gone wrong or an operator that is not Hermitian, not to second-guess the last digits
/// of one that converged. Of an operator that is not linear it catches only what its residuals
/// show: one whose products grow without bound raises the estimate of `‖A‖` the check is
/// relative to, and can pass it.
fn verified<R: Float>(tolerance: R, work: usize, operator: R, size: R, residual: R) -> bool {
    let slack = R::from_f64(4.0);
    let room = R::from_f64(64.0 * (work.max(1) as f64).sqrt());
    residual <= slack * tolerance * size.max(eps23::<R>() * operator) + room * R::EPSILON * operator
}

/// The `w` of [`verified`] for a call that made `restarts` restarts.
fn work<T: Scalar>(call: &Call<'_, T>, restarts: usize) -> usize {
    call.n
        .max(call.basis.saturating_mul(restarts.saturating_add(1)))
}

/// `A·Z` for the columns `z` (of length `n`) and `Zᴴ·A·Z`, the products counted in `state` when
/// it is given; `None` if a product is not finite.
fn rayleigh<T: Scalar>(
    op: &dyn LinearOperator<T>,
    n: usize,
    z: &[T],
    mut state: Option<&mut State<T::Real>>,
) -> Option<(Vec<T>, Vec<T>)> {
    let c = z.len() / n;
    let mut w = vec![T::zero(); n * c];
    for (x, y) in z.chunks_exact(n).zip(w.chunks_exact_mut(n)) {
        op.apply(x, y);
        if !finite(y) {
            return None;
        }
        if let Some(state) = state.as_mut() {
            state.applications += 1;
            let size = norm(y);
            if size > state.norm {
                state.norm = size;
            }
        }
    }
    let mut r = vec![T::zero(); c * c];
    gemm(
        c,
        c,
        n,
        T::one(),
        z,
        n,
        Op::C,
        &w,
        n,
        Op::N,
        T::zero(),
        &mut r,
        c,
    );
    Some((w, r))
}

/// The Hermitian part `(R + Rᴴ)/2` of the square matrix `r` of order `c`: `Zᴴ·A·Z` of a
/// Hermitian `A` is Hermitian but for rounding.
fn hermitian_part<T: Scalar>(r: &[T], c: usize) -> Vec<T> {
    (0..c * c)
        .map(|at| {
            let (i, j) = (at % c, at / c);
            (r[i + j * c] + r[j + i * c].conj()).scale(T::Real::HALF)
        })
        .collect()
}

/// `sᴴ·R·s` for the square matrix `r` of order `c` and the complex coefficients `s`.
fn quadratic<T: Scalar>(r: &[T], c: usize, s: &[Complex<T::Real>]) -> Complex<T::Real> {
    let zero = Complex::new(T::Real::ZERO, T::Real::ZERO);
    let mut sum = zero;
    for (j, &sj) in s.iter().enumerate() {
        let column = r[j * c..(j + 1) * c]
            .iter()
            .zip(s)
            .fold(zero, |acc, (&x, &si)| acc + si.conj() * x.to_complex());
        sum += column * sj;
    }
    sum
}

/// The Rayleigh–Ritz step on the found vectors `Z` for the answer: the eigenpairs of `Zᴴ·A·Z`,
/// as the values that rank them, the eigenvalues, the coefficients of the vectors in `Z`, and
/// `A·Z`. Without shift-and-invert, `A` is the operator of the call, and the eigenvalues rank
/// themselves. With it, the step is made with `A` itself, whose small matrix is well scaled, where
/// that of `(A − σI)⁻¹` holds `1/(λ − σ)` for an eigenvalue `λ` near `σ`, so that a dense
/// eigensolver, accurate to `ε` times its largest eigenvalue, would lose the others; the values
/// that rank the pairs are then their Rayleigh quotients `sᴴ·Zᴴ·(A − σI)⁻¹·Z·s` with the
/// operator of the call, which needs no `σ`.
type Answer<T> = (
    Vec<Complex<<T as Scalar>::Real>>,
    Vec<Complex<<T as Scalar>::Real>>,
    Matrix<Complex<<T as Scalar>::Real>>,
    Vec<T>,
);

/// [`Answer`] of a call from its found vectors, `None` if a product or the eigensolver gives a
/// value that is not finite; `hermitian` for the Hermitian eigensolver.
fn answer<T: Scalar>(
    call: &Call<'_, T>,
    z: &[T],
    state: &mut State<T::Real>,
    hermitian: bool,
) -> Option<Answer<T>> {
    let n = call.n;
    let c = z.len() / n;
    let (w, r) = rayleigh(call.operator, n, z, Some(state))?;
    let (wa, ra) = match call.original {
        None => (w, r.clone()),
        Some(original) => rayleigh(original.operator, n, z, None)?,
    };
    let (lambda, s): basis::ComplexEigenpairs<T::Real> = if hermitian {
        let (values, vectors) = hermitian_eigen(&hermitian_part(&ra, c), c)?;
        let vectors = vectors.as_slice().iter().map(|x| x.to_complex()).collect();
        (
            values
                .iter()
                .map(|&x| Complex::new(x, T::Real::ZERO))
                .collect(),
            Matrix::from_column_major(c, c, vectors),
        )
    } else {
        general_eigen(&ra, c)?
    };
    let rank = match call.original {
        None => lambda.clone(),
        Some(_) => {
            let r = if hermitian { hermitian_part(&r, c) } else { r };
            (0..c)
                .map(|i| {
                    let q = quadratic(&r, c, s.column(i));
                    if hermitian {
                        Complex::new(q.re, T::Real::ZERO)
                    } else {
                        q
                    }
                })
                .collect()
        }
    };
    for z in &rank {
        let size = z.re.hypot(z.im);
        if size > state.norm {
            state.norm = size;
        }
    }
    Some((rank, lambda, s, wa))
}

/// The answer of a Lanczos call from what its runs found: the Rayleigh–Ritz step on the found
/// vectors (see [`answer`]) — the eigenpairs of `Zᴴ·A·Z` taken back by `Z`, which turns the
/// blocks the runs locked into eigenvectors — and, at each end of the selection, its pairs from
/// the most wanted inward, each more wanted than the frontier and checked by its true residual,
/// up to the first that is not.
fn finish_lanczos<T: Scalar>(call: &Call<'_, T>, found: Found<T>) -> LanczosResult<T> {
    let n = call.n;
    let mut state = found.state;
    let mut pairs: Vec<(T::Real, Vec<T>, T::Real)> = Vec::new();
    if let Some((rank, lambda, s, w)) = (n > 0 && !found.vectors.is_empty())
        .then(|| answer(call, &found.vectors, &mut state, true))
        .flatten()
    {
        let c = found.vectors.len() / n;
        let effort = work(call, state.restarts);
        let size = match call.original {
            None => state.norm,
            Some(original) => original.norm,
        };
        for (p, part) in parts(call.rule, &rank, call.k, false).iter().enumerate() {
            let score = part_score(call.rule, p);
            for unit in part {
                let i = unit[0];
                if !trusted(score, rank[i], &found.frontier, call.tolerance, state.norm) {
                    break;
                }
                let coefficients: Vec<T> = s
                    .column(i)
                    .iter()
                    .map(|z| T::from_parts(z.re, z.im))
                    .collect();
                let x = combine(n, c, &found.vectors, &coefficients, 1);
                let ax = combine(n, c, &w, &coefficients, 1);
                let l = lambda[i].re;
                let d: Vec<T> = ax.iter().zip(&x).map(|(&y, &v)| y - v.scale(l)).collect();
                let residual = norm(&d);
                if !verified(call.tolerance, effort, size.max(l.abs()), l.abs(), residual) {
                    break;
                }
                pairs.push((l, x, residual));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    let converged = pairs.len();
    let mut data = Vec::with_capacity(n * converged);
    for (_, x, _) in &pairs {
        data.extend_from_slice(x);
    }
    LanczosResult {
        values: pairs.iter().map(|p| p.0).collect(),
        vectors: Matrix::from_column_major(n, converged, data),
        residuals: pairs.iter().map(|p| p.2).collect(),
        converged,
        restarts: state.restarts,
        applications: state.applications,
    }
}

/// An eigenpair of an Arnoldi result as it is assembled: the value, the vector and its residual.
type ComplexPair<R> = (Complex<R>, Vec<Complex<R>>, R);

/// The answer of an Arnoldi call from what its runs found: the Rayleigh–Ritz step on the found
/// vectors (see [`answer`]) — the eigenvectors of `Zᴴ·A·Z` taken back by `Z`, which turns the
/// blocks the runs locked and the vectors of the complement found by later runs into
/// eigenvectors of `A` — and its pairs from the most wanted on, each more wanted than the
/// frontier and checked by its true residual, a conjugate pair as a whole, up to the first that
/// is not.
fn finish_arnoldi<T: Scalar>(call: &Call<'_, T>, found: Found<T>) -> ArnoldiResult<T> {
    let n = call.n;
    let mut state = found.state;
    let mut pairs: Vec<ComplexPair<T::Real>> = Vec::new();
    if let Some((rank, lambda, s, w)) = (n > 0 && !found.vectors.is_empty())
        .then(|| answer(call, &found.vectors, &mut state, false))
        .flatten()
    {
        let c = found.vectors.len() / n;
        let effort = work(call, state.restarts);
        let size = match call.original {
            None => state.norm,
            Some(original) => original.norm,
        };
        'parts: for (p, part) in parts(call.rule, &rank, call.k, call.pairs)
            .iter()
            .enumerate()
        {
            let score = part_score(call.rule, p);
            for unit in part {
                // A pair stands by its member with the positive imaginary part, the first.
                let i = unit[0];
                if !trusted(score, rank[i], &found.frontier, call.tolerance, state.norm) {
                    break;
                }
                let x = combine_complex(n, c, &found.vectors, s.column(i));
                let ax = combine_complex(n, c, &w, s.column(i));
                let l = lambda[i];
                let d: Vec<Complex<T::Real>> =
                    ax.iter().zip(&x).map(|(&y, &v)| y - l * v).collect();
                let residual = complex_norm(&d);
                let modulus = l.re.hypot(l.im);
                if !verified(call.tolerance, effort, size.max(modulus), modulus, residual) {
                    break 'parts;
                }
                if unit.len() == 2 {
                    // The member with the positive imaginary part first; through an inversion
                    // that may be the conjugate of the one the ranking value stood for.
                    let conj: Vec<Complex<T::Real>> = x.iter().map(|v| v.conj()).collect();
                    let (first, second) = if l.im < T::Real::ZERO {
                        ((l.conj(), conj), (l, x))
                    } else {
                        ((l, x), (l.conj(), conj))
                    };
                    pairs.push((first.0, first.1, residual));
                    pairs.push((second.0, second.1, residual));
                } else {
                    pairs.push((l, x, residual));
                }
            }
        }
    }
    let converged = pairs.len();
    let mut data = Vec::with_capacity(n * converged);
    for (_, x, _) in &pairs {
        data.extend_from_slice(x);
    }
    ArnoldiResult {
        values: pairs.iter().map(|p| p.0).collect(),
        vectors: Matrix::from_column_major(n, converged, data),
        residuals: pairs.iter().map(|p| p.2).collect(),
        converged,
        restarts: state.restarts,
        applications: state.applications,
    }
}
