//! The Krylov methods: `A·x = b` solved by improving a guess rather than by factoring.
//!
//! A method of this family touches the matrix only through products with vectors, so it runs on
//! anything that is a [`LinearOperator`]: a [`SparseMatrix`], a dense [`Matrix`], or a function
//! wrapped in an [`FnOperator`](super::FnOperator). What an iteration costs is fixed, a product
//! or two and a few vector operations; how many are needed depends on the spectrum, which is
//! what a [`Preconditioner`] is for.
//!
//! | Method | For | Preconditioner `M` | Its recurrence follows | An iteration is |
//! | --- | --- | --- | --- | --- |
//! | [`cg`] | Hermitian positive definite `A` | symmetric: `M` Hermitian positive definite | the residual `b − A·x` | one product with `A` |
//! | [`minres`] | Hermitian `A`, indefinite or singular too | symmetric: `M` Hermitian positive definite | the residual in the norm of `M⁻¹`, and `b − A·x` beside it | one product |
//! | [`gmres`] | any square `A` | on the right: `A·M⁻¹·u = b`, `x = M⁻¹·u` | the norm of `b − A·x` | one product |
//! | [`bicgstab`] | any square `A` | on the right | the residual `b − A·x` | two products, or one when it ends halfway |
//!
//! Every method comes as `name(a, b)`, with the default options and no preconditioner, and as
//! `name_with(a, b, &opts, pre)`. Complex systems use the inner product `Σ ūᵢ·vᵢ`.
//!
//! # When a method stops
//!
//! All four stop on the same test, on the residual of the system as it was given whatever the
//! preconditioner:
//!
//! ```text
//! ‖b − A·x‖₂ ≤ max(rtol·‖b‖₂, atol)
//! ```
//!
//! The residual a recurrence carries drifts away from `b − A·x` as rounding errors accumulate,
//! so it only says when to look: before a result is marked `converged` the residual is formed
//! afresh from `x` by [`LinearOperator::residual`], and the test is made on that. If it fails,
//! the method starts again from that residual and goes on; if a second look finds no
//! improvement, the tolerance is beyond what the arithmetic can verify and the result says
//! [`StopReason::Stagnated`]. A tolerance below `ε·‖b‖₂` is looked at once the recurrence
//! comes down to that level, which is as far as a residual formed afresh can follow it. The
//! `residual` of a [`SolveResult`] is always the one formed afresh from the `x` returned, and
//! `converged` is true exactly when it passes the test.
//!
//! [`SparseMatrix`] and [`Matrix`] form that residual in twice the working precision, so for
//! them it is the exact residual, rounded once. An operator known only through its products
//! forms it with one of them, and then carries that product's rounding error, about
//! `ε·‖|A|·|x| + |b|‖₂`: a tolerance below that level cannot be verified by anyone.
//!
//! A run that does not converge returns the best point it reached, not merely its last, and the
//! starting guess counts among the points reached, so that no result is worse than where the
//! run began: for [`cg`] and [`bicgstab`], whose residuals rise and fall, the iterate of the
//! smallest residual the recurrence carried, its residual formed afresh and compared with the
//! last one's; for [`gmres`] the better end of its last cycle; for [`minres`] the best of the
//! points it verified on the way (see there) and the last.
//!
//! The inner products are summed in `f64` for every element type, and the scalar recurrences of
//! [`cg`] and [`bicgstab`] are carried in `f64` too. The inner products that square the scale of
//! the right-hand side are taken of vectors brought near a norm of one by a power of two, which
//! is exact, and the factor is renewed when the residual has come far from it: a right-hand
//! side of any size the type holds is solved, from `10³⁰⁰` down to the subnormal numbers in
//! `f64`, and from `10²⁰` down to them in `f32`, and so is a system whose starting guess is
//! `10³⁰` times its solution in `f32`. [`minres`] and [`gmres`] normalise their bases instead.
//!
//! Nothing here returns an answer it has not checked, or runs for ever on its default budget: a
//! zero right-hand side gives `x = 0` at once, a starting guess that passes the test is returned
//! at once, and a matrix that is not what the method needs, a breakdown of the recurrence, the
//! iteration limit and data that are not finite each end in a result with `converged` false and
//! the reason. With an unlimited budget (`max_iter` of `usize::MAX`), [`cg`] on a matrix that is
//! not positive definite, such as a singular one whose range the right-hand side is not in, can
//! go on until the arithmetic stops it: its residual neither comes down nor shows that it
//! cannot.
//!
//! # Examples
//!
//! ```
//! use a::math::sparse::{Ilu0, SolveOptions, SparseMatrix, StopReason, cg, gmres_with};
//!
//! // The 1-D Laplacian of twenty points, positive definite.
//! const N: usize = 20;
//! let a = SparseMatrix::tridiagonal(&[-1.0; N - 1], &[2.0; N], &[-1.0; N - 1]);
//! let b: Vec<f64> = (0..N).map(|i| (i + 1) as f64).collect();
//! let x = cg(&a, &b).unwrap();
//! assert!(x.converged && x.reason == StopReason::Converged);
//! let norm_b = b.iter().map(|v| v * v).sum::<f64>().sqrt();
//! assert!(x.residual <= f64::EPSILON.sqrt() * norm_b);
//!
//! // Convection and diffusion in one dimension: not symmetric, so GMRES, here restarted every
//! // ten iterations and preconditioned by an incomplete LU factorisation.
//! let c = SparseMatrix::tridiagonal(&[-1.5; N - 1], &[2.0; N], &[-0.5; N - 1]);
//! let m = Ilu0::new(&c).unwrap();
//! let opts = SolveOptions::new().with_restart(10).with_rtol(1e-12);
//! let y = gmres_with(&c, &b, &opts, Some(&m)).unwrap();
//! assert!(y.converged && y.iterations <= 2);
//! ```
//!
//! [`Matrix`]: crate::math::matrix::Matrix

use super::SparseMatrix;
use super::operator::LinearOperator;
use super::precond::Preconditioner;
use crate::math::complex::{Complex, Float};
use crate::math::matrix::{Matrix, Scalar};
use crate::math::numerics::ordered;

/// When an iterative method stops, and where it starts.
///
/// A method stops with `converged` true once `‖b − A·x‖₂ ≤ max(rtol·‖b‖₂, atol)`, and gives up
/// after `max_iter` iterations. The defaults are `rtol` `√ε` of the element type (`1.5e-8` for
/// `f64`, `3.5e-4` for `f32`), `atol` 0, `max_iter` ten times the number of unknowns, a restart
/// length of 30 and the zero vector as the starting guess.
///
/// A tolerance of zero is allowed: with both at zero only an exactly zero residual converges.
/// A tolerance that is negative or not finite, or a restart length of zero for [`gmres`], is
/// not a request a method can serve, and it returns `None`.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SolveOptions, SparseMatrix, cg_with};
///
/// let a = SparseMatrix::tridiagonal(&[-1.0; 3], &[2.0; 4], &[-1.0; 3]);
/// let b = [1.0, 0.0, 0.0, 1.0];
/// let opts = SolveOptions::new().with_rtol(1e-12).with_max_iter(50);
/// assert_eq!((opts.rtol, opts.atol, opts.max_iter, opts.restart), (1e-12, 0.0, Some(50), 30));
/// let x = cg_with(&a, &b, &opts, None).unwrap();
/// assert!(x.converged);
///
/// // From the solution itself nothing is left to do.
/// let warm = opts.clone().with_initial_guess(&[1.0, 1.0, 1.0, 1.0]);
/// let y = cg_with(&a, &b, &warm, None).unwrap();
/// assert!(y.converged && y.iterations == 0);
///
/// // A negative tolerance is no tolerance.
/// assert!(cg_with(&a, &b, &opts.with_rtol(-1.0), None).is_none());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SolveOptions<T: Scalar> {
    /// The relative tolerance, on the residual against `‖b‖₂`.
    pub rtol: T::Real,
    /// The absolute tolerance, on the residual itself.
    pub atol: T::Real,
    /// The largest number of iterations; `None`, the default, stands for ten times the number
    /// of unknowns.
    pub max_iter: Option<usize>,
    /// The number of iterations after which [`gmres`] restarts, clamped to the number of
    /// unknowns; the other methods ignore it.
    pub restart: usize,
    /// The starting guess; `None`, the default, stands for the zero vector.
    pub initial_guess: Option<Vec<T>>,
}

impl<T: Scalar> SolveOptions<T> {
    /// The defaults: `rtol` `√ε`, `atol` 0, `max_iter` ten times the number of unknowns,
    /// `restart` 30, and the zero vector to start from.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rtol: T::Real::EPSILON.sqrt(),
            atol: T::Real::ZERO,
            max_iter: None,
            restart: 30,
            initial_guess: None,
        }
    }

    /// Sets the relative tolerance.
    #[must_use]
    pub fn with_rtol(mut self, rtol: T::Real) -> Self {
        self.rtol = rtol;
        self
    }

    /// Sets the absolute tolerance.
    #[must_use]
    pub fn with_atol(mut self, atol: T::Real) -> Self {
        self.atol = atol;
        self
    }

    /// Sets the largest number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = Some(max_iter);
        self
    }

    /// Sets the restart length of [`gmres`]: the dimension of the Krylov space it builds before
    /// it starts again from the solution reached.
    #[must_use]
    pub fn with_restart(mut self, restart: usize) -> Self {
        self.restart = restart;
        self
    }

    /// Sets the starting guess, of which a copy is kept.
    #[must_use]
    pub fn with_initial_guess(mut self, x0: &[T]) -> Self {
        self.initial_guess = Some(x0.to_vec());
        self
    }
}

impl<T: Scalar> Default for SolveOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Why an iterative method stopped.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SolveOptions, SparseMatrix, StopReason, cg, cg_with};
///
/// let a = SparseMatrix::tridiagonal(&[-1.0; 9], &[2.0; 10], &[-1.0; 9]);
/// let b = [1.0; 10];
/// assert_eq!(cg(&a, &b).unwrap().reason, StopReason::Converged);
///
/// let short = cg_with(&a, &b, &SolveOptions::new().with_max_iter(2), None).unwrap();
/// assert_eq!(short.reason, StopReason::IterationLimit);
///
/// // The conjugate gradient method on an indefinite matrix: found out, not answered.
/// let indefinite = SparseMatrix::from_diagonal(&[1.0, -1.0]);
/// let wrong = cg(&indefinite, &[1.0, 1.0]).unwrap();
/// assert_eq!(wrong.reason, StopReason::NotPositiveDefinite);
/// assert!(!wrong.converged);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The residual of the solution returned passes the test.
    Converged,
    /// The method ran its `max_iter` iterations.
    IterationLimit,
    /// The recurrence cannot go on: an inner product it divides by vanished (`ρ`, `r̂ᴴ·A·p̂` or
    /// `ω` of [`bicgstab`]), or the matrix is singular on the Krylov space with the residual
    /// still above the tolerance, as for a singular matrix whose range the right-hand side is
    /// not in: the least squares problem of [`gmres`] lost a column, or [`minres`] came to a
    /// residual that the matrix annihilates to working precision, or verified one that it
    /// annihilates to half of it and then could not improve on its best point. `x` is then the
    /// best point verified, for [`minres`] a least squares solution to the accuracy the
    /// arithmetic allows.
    Breakdown,
    /// [`cg`] met a direction `p` with `pᴴ·A·p ≤ 0`, or [`cg`] or [`minres`] a residual `r` with
    /// `rᴴ·M⁻¹·r ≤ 0`: the matrix, or the preconditioner, is not Hermitian positive definite.
    NotPositiveDefinite,
    /// A NaN or an infinity turned up, in the data or in the course of the method.
    NonFinite,
    /// The method believed it had converged, the residual formed afresh said otherwise, and
    /// going on did not improve it; or a whole cycle of [`gmres`] left the residual where it
    /// was. The tolerance is beyond what the arithmetic, the restart length, or the system can
    /// reach.
    Stagnated,
}

/// What an iterative method reached.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, StopReason, bicgstab};
///
/// let a = SparseMatrix::<f64>::from_coo(2, 2, &[(0, 0, 4.0), (0, 1, 1.0), (1, 0, 2.0), (1, 1, 3.0)]);
/// let r = bicgstab(&a, &[5.0, 5.0]).unwrap();
/// assert!(r.converged && r.reason == StopReason::Converged);
/// assert!((r.x[0] - 1.0).abs() < 1e-7 && (r.x[1] - 1.0).abs() < 1e-7);
/// // The residual is the one of the x returned, formed afresh.
/// let again = ((5.0 - 4.0 * r.x[0] - r.x[1]).powi(2) + (5.0 - 2.0 * r.x[0] - 3.0 * r.x[1]).powi(2)).sqrt();
/// assert!((r.residual - again).abs() <= 1e-15);
/// assert!(r.matvecs >= r.iterations);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SolveResult<T: Scalar> {
    /// The solution, or the best point reached when the method did not converge.
    pub x: Vec<T>,
    /// `‖b − A·x‖₂` for the `x` returned, formed afresh rather than taken from the recurrence.
    /// It is not finite when the data are not.
    pub residual: T::Real,
    /// The number of iterations: for [`cg`], [`minres`] and [`gmres`] one product with `A`
    /// each, for [`bicgstab`] two, or one for a step that ended halfway.
    pub iterations: usize,
    /// The number of products with `A` in all: those of the iterations, the one of the first
    /// residual when there is a starting guess, and one for each residual formed afresh.
    pub matvecs: usize,
    /// Whether `residual ≤ max(rtol·‖b‖₂, atol)`.
    pub converged: bool,
    /// Why the method stopped; [`StopReason::Converged`] exactly when `converged` is true.
    pub reason: StopReason,
}

// ---------------------------------------------------------------------------------------------
// Vectors
// ---------------------------------------------------------------------------------------------

/// The inner product `Σ ūᵢ·vᵢ`, conjugate linear in its first argument, of two vectors whose
/// size does not depend on the scale of the system: those of [`minres`] and [`gmres`], which
/// normalise their bases.
///
/// The sum is taken in `f64` whatever the element type, so that the inner products of `f32`
/// vectors are exact to the last bit or two however long the vectors are.
fn dot<T: Scalar>(u: &[T], v: &[T]) -> T {
    let (re, im) = dot_parts(u, v, 1.0, 1.0);
    T::from_parts(T::Real::from_f64(re), T::Real::from_f64(im))
}

/// A number of the scalar recurrences of [`cg`] and [`bicgstab`], which are carried in `f64`
/// whatever the element type: their inner products are summed in `f64` anyway, and kept there a
/// quotient of two of them cannot underflow or overflow on the way, however far the residual
/// has come from the one the factor of the run was taken from (see [`Session::renew`]). For
/// `f64` and `Complex<f64>` this is the arithmetic of the type itself.
type Wide = Complex<f64>;

/// The element nearest to `x`, its real part for a real type.
#[inline]
fn narrow<T: Scalar>(x: Wide) -> T {
    T::from_parts(T::Real::from_f64(x.re), T::Real::from_f64(x.im))
}

/// `x` as a [`Wide`] number, exactly.
#[inline]
fn widen<T: Scalar>(x: T) -> Wide {
    Complex::new(x.re().to_f64(), x.im().to_f64())
}

/// The norm of `M⁻¹` of the residual `r`, whose 2-norm is `norm`: `√(rᴴ·M⁻¹·r)`, which is `norm`
/// itself without a preconditioner. `scratch` is overwritten.
fn m_norm<T: Scalar>(
    pre: Option<&dyn Preconditioner<T>>,
    r: &[T],
    norm: T::Real,
    scratch: &mut [T],
) -> T::Real {
    match pre {
        Some(m) => {
            m.apply(r, scratch);
            inner_root(r, scratch)
        }
        None => norm,
    }
}

/// The two parts of `Σ conj(s·uᵢ)·(t·vᵢ)` in `f64`, in four running sums: the term of the index
/// `i` to the sum `i mod 4` (the last `n mod 4` to the first), added as `(s₀ + s₁) + (s₂ + s₃)`
/// — four independent chains of additions, in the same order every time.
fn dot_parts<T: Scalar>(u: &[T], v: &[T], s: f64, t: f64) -> (f64, f64) {
    let n = u.len().min(v.len());
    let (u, v) = (&u[..n], &v[..n]);
    let (mut re, mut im) = ([0.0f64; 4], [0.0f64; 4]);
    let (uc, uc_rest) = u.as_chunks::<4>();
    let (vc, vc_rest) = v.as_chunks::<4>();
    if T::IS_COMPLEX {
        let term = |re: &mut f64, im: &mut f64, u: T, v: T| {
            let (a, b) = (u.re().to_f64() * s, u.im().to_f64() * s);
            let (c, d) = (v.re().to_f64() * t, v.im().to_f64() * t);
            *re += a * c + b * d;
            *im += a * d - b * c;
        };
        for (u4, v4) in uc.iter().zip(vc.iter()) {
            for q in 0..4 {
                term(&mut re[q], &mut im[q], u4[q], v4[q]);
            }
        }
        for (&u, &v) in uc_rest.iter().zip(vc_rest) {
            term(&mut re[0], &mut im[0], u, v);
        }
    } else {
        for (u4, v4) in uc.iter().zip(vc.iter()) {
            for q in 0..4 {
                re[q] += (u4[q].re().to_f64() * s) * (v4[q].re().to_f64() * t);
            }
        }
        for (&u, &v) in uc_rest.iter().zip(vc_rest) {
            re[0] += (u.re().to_f64() * s) * (v.re().to_f64() * t);
        }
    }
    (
        (re[0] + re[1]) + (re[2] + re[3]),
        (im[0] + im[1]) + (im[2] + im[3]),
    )
}

/// `Σ |xᵢ|²` in `f64`, in four running sums as [`dot_parts`] has them; the elements are divided
/// by `scale` first (1 for none).
fn sum_squares<T: Scalar>(x: &[T], scale: f64) -> f64 {
    let mut lanes = [0.0f64; 4];
    let square = |x: T| {
        let (a, b) = (x.re().to_f64() / scale, x.im().to_f64() / scale);
        a * a + b * b
    };
    let (chunks, chunks_rest) = x.as_chunks::<4>();
    for x4 in chunks.iter() {
        for q in 0..4 {
            lanes[q] += square(x4[q]);
        }
    }
    for &x in chunks_rest {
        lanes[0] += square(x);
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// The largest part of an element of `x` in absolute value, in `f64`.
fn largest_part<T: Scalar>(x: &[T]) -> f64 {
    x.iter().fold(0.0f64, |top, x| {
        top.max(x.re().to_f64().abs()).max(x.im().to_f64().abs())
    })
}

/// `√(Re(uᴴ·v))` for `v = M⁻¹·u`: the norm of `u` in the inner product of `M⁻¹`, which MINRES
/// normalises its vectors in. A negative square, which only a preconditioner that is not
/// positive definite gives, comes back negative, and one that is not a number as NaN.
///
/// The square is summed in `f64`, which holds the squares of any `f32` elements. Where it
/// overflows or underflows so far that it loses its digits, which `f64` elements of the order
/// of `10¹⁶⁰` or `10⁻¹⁷⁰` make it do, the sum is taken again on the two vectors brought to a
/// largest element near one, and the root scaled back.
fn inner_root<T: Scalar>(u: &[T], v: &[T]) -> T::Real {
    let signed_root = |square: f64| square.abs().sqrt().copysign(square);
    let square = dot_parts(u, v, 1.0, 1.0).0;
    if square.is_nan() || (square.is_finite() && square.abs() >= f64::MIN_POSITIVE / f64::EPSILON) {
        return T::Real::from_f64(signed_root(square));
    }
    // Overflowed, or so small that it may have underflowed, a zero among them.
    let (top_u, top_v) = (largest_part(u), largest_part(v));
    if top_u == 0.0 || top_v == 0.0 {
        return T::Real::ZERO;
    }
    let (s, t) = (reciprocal_scale(top_u), reciprocal_scale(top_v));
    // √s and √t are powers of two or √2 times one, each within the range of `f64`.
    let root = signed_root(dot_parts(u, v, s, t).0) / (s.sqrt() * t.sqrt());
    T::Real::from_f64(root)
}

/// A power of two near `1 / x` for a positive `x`, exact and within the normal range of `f64`.
fn reciprocal_scale(x: f64) -> f64 {
    let exponent = if x > 0.0 && x.is_finite() {
        // The exponent of x, from its bits; a subnormal x counts as the smallest normal one,
        // which leaves its scaled square small but clear of underflow.
        ((x.to_bits() >> 52) & 0x7ff).max(1) as i64 - 1023
    } else {
        0
    };
    // 2^(-exponent), built from its bits: the exponent field 1023 - exponent lies in 1..=2045.
    f64::from_bits(((1023 - exponent.clamp(-1022, 1022)) as u64) << 52)
}

/// The Euclidean norm, with the squares summed in `f64`. Where they overflow, or underflow so
/// far that they lose their digits, the sum is taken again on the elements divided by the
/// largest of them. A NaN among the elements gives NaN, and otherwise an infinity gives infinity.
fn norm2<T: Scalar>(x: &[T]) -> T::Real {
    norm2_from(x, sum_squares(x, 1.0))
}

/// [`norm2`] with the squares summed one after another: the norm of `b` and of every residual
/// formed afresh, which decide what a run returns and against what, measured as they always
/// were (the recurrences' estimates take the four-sum [`norm2`]).
fn norm2_ordered<T: Scalar>(x: &[T]) -> T::Real {
    let mut sum = 0.0f64;
    for x in x {
        let (a, b) = (x.re().to_f64(), x.im().to_f64());
        sum += a * a + b * b;
    }
    if sum.is_nan() {
        return T::Real::NAN;
    }
    if sum.is_finite() && sum >= f64::MIN_POSITIVE / f64::EPSILON {
        return T::Real::from_f64(sum.sqrt());
    }
    let mut top = 0.0f64;
    for x in x {
        top = top.max(x.re().to_f64().abs()).max(x.im().to_f64().abs());
    }
    if top == 0.0 || top.is_infinite() {
        return T::Real::from_f64(top);
    }
    let mut sum = 0.0f64;
    for x in x {
        let (a, b) = (x.re().to_f64() / top, x.im().to_f64() / top);
        sum += a * a + b * b;
    }
    T::Real::from_f64(top * sum.sqrt())
}

/// [`norm2`] of `x` given the sum of its squares, `sum_squares(x, 1.0)`, already taken (as a
/// fused update takes it): the same result.
fn norm2_from<T: Scalar>(x: &[T], sum: f64) -> T::Real {
    if sum.is_nan() {
        return T::Real::NAN;
    }
    if sum.is_finite() && sum >= f64::MIN_POSITIVE / f64::EPSILON {
        return T::Real::from_f64(sum.sqrt());
    }
    let mut top = 0.0f64;
    for x in x {
        top = top.max(x.re().to_f64().abs()).max(x.im().to_f64().abs());
    }
    if top == 0.0 || top.is_infinite() {
        return T::Real::from_f64(top);
    }
    T::Real::from_f64(top * sum_squares(x, top).sqrt())
}

/// `x ← x + α·p` and `r ← r − α·q` in one pass, returning the sum of the squares of the new `r`
/// as [`sum_squares`] takes it (the same bits), for [`norm2_from`].
fn advance<T: Scalar>(x: &mut [T], p: &[T], r: &mut [T], q: &[T], alpha: T::Real) -> f64 {
    let n = x.len();
    debug_assert!(p.len() == n && r.len() == n && q.len() == n);
    let mut lanes = [0.0f64; 4];
    let (xc, xc_rest) = x.as_chunks_mut::<4>();
    let (pc, pc_rest) = p.as_chunks::<4>();
    let (rc, rc_rest) = r.as_chunks_mut::<4>();
    let (qc, qc_rest) = q.as_chunks::<4>();
    for (((x4, p4), r4), q4) in xc
        .iter_mut()
        .zip(pc.iter())
        .zip(rc.iter_mut())
        .zip(qc.iter())
    {
        for k in 0..4 {
            x4[k] += p4[k].scale(alpha);
            r4[k] -= q4[k].scale(alpha);
            let (a, b) = (r4[k].re().to_f64(), r4[k].im().to_f64());
            lanes[k] += a * a + b * b;
        }
    }
    let rest = xc_rest.iter_mut().zip(pc_rest);
    for ((x, &p), (r, &q)) in rest.zip(rc_rest.iter_mut().zip(qc_rest)) {
        *x += p.scale(alpha);
        *r -= q.scale(alpha);
        let (a, b) = (r.re().to_f64(), r.im().to_f64());
        lanes[0] += a * a + b * b;
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// The elements of the vectors a pass of [`multi_dot`] and [`multi_subtract`] reads at a time,
/// so that the piece of `w` stays in the first-level cache while every basis vector streams by:
/// 16 KiB of `f64`, 32 KiB of `Complex<f64>`. A multiple of 4, which keeps the sums as they are.
// Measured on an Apple M4 Max (`gmres_with` on convection-diffusion grids of 10⁴ and 10⁵
// unknowns and a random matrix of 10⁶, restarts of 30 and 50, `f64` and `Complex<f64>`; sweep
// 128 to 8192): 2048 against 512 takes 3 to 11 % off every one; 4096 is as fast, 8192 loses on
// 10⁴ unknowns and in `f64`, 128 and 256 lose 1 to 8 %. On x86-64, whose first-level data cache
// is 32 to 48 KiB against 128 KiB here, the complex block fills 32 KiB: the Actions run measures
// it.
const GS_BLOCK: usize = 2048;

/// `h[i] = basis[i]ᴴ·w` for every basis vector, in one pass over the basis, each inner product
/// summed in `f64` in four running sums (the term of the index `e` to the sum `e mod 4`), the
/// same order every time.
fn multi_dot<T: Scalar>(basis: &[Vec<T>], w: &[T], h: &mut Vec<T>) {
    let n = w.len();
    let mut re = vec![[0.0f64; 4]; basis.len()];
    let mut im = vec![[0.0f64; 4]; basis.len()];
    let mut start = 0;
    while start < n {
        // The blocks begin at multiples of 4, so that an element's sum is its index mod 4.
        let end = (start + GS_BLOCK).min(n);
        let wb = &w[start..end];
        for (v, (re, im)) in basis.iter().zip(re.iter_mut().zip(im.iter_mut())) {
            let (mut r, mut i) = (*re, *im);
            let (vc, vc_rest) = v[start..end].as_chunks::<4>();
            let (wc, wc_rest) = wb.as_chunks::<4>();
            for (v4, w4) in vc.iter().zip(wc.iter()) {
                for q in 0..4 {
                    let (a, b) = (v4[q].re().to_f64(), v4[q].im().to_f64());
                    let (c, d) = (w4[q].re().to_f64(), w4[q].im().to_f64());
                    if T::IS_COMPLEX {
                        r[q] += a * c + b * d;
                        i[q] += a * d - b * c;
                    } else {
                        r[q] += a * c;
                    }
                }
            }
            for (&v, &x) in vc_rest.iter().zip(wc_rest) {
                let (a, b) = (v.re().to_f64(), v.im().to_f64());
                let (c, d) = (x.re().to_f64(), x.im().to_f64());
                if T::IS_COMPLEX {
                    r[0] += a * c + b * d;
                    i[0] += a * d - b * c;
                } else {
                    r[0] += a * c;
                }
            }
            *re = r;
            *im = i;
        }
        start = end;
    }
    h.clear();
    for (re, im) in re.iter().zip(&im) {
        let (r, i) = (
            (re[0] + re[1]) + (re[2] + re[3]),
            (im[0] + im[1]) + (im[2] + im[3]),
        );
        h.push(T::from_parts(T::Real::from_f64(r), T::Real::from_f64(i)));
    }
}

/// `w ← w − Σ h[i]·basis[i]`, in one pass over the basis, the terms of each element taken out
/// in the order of the basis.
fn multi_subtract<T: Scalar>(w: &mut [T], basis: &[Vec<T>], h: &[T]) {
    let n = w.len();
    let mut start = 0;
    while start < n {
        let end = (start + GS_BLOCK).min(n);
        let wb = &mut w[start..end];
        for (v, &c) in basis.iter().zip(h) {
            for (w, &v) in wb.iter_mut().zip(&v[start..end]) {
                *w = w.mul_sub(c, v);
            }
        }
        start = end;
    }
}

/// `y ← y + α·x`.
fn add_scaled<T: Scalar>(y: &mut [T], alpha: T, x: &[T]) {
    for (y, &x) in y.iter_mut().zip(x) {
        *y += alpha * x;
    }
}

/// `y ← y − α·x`.
fn subtract_scaled<T: Scalar>(y: &mut [T], alpha: T, x: &[T]) {
    for (y, &x) in y.iter_mut().zip(x) {
        *y = y.mul_sub(alpha, x);
    }
}

/// `x / r` for a real `r`: a division of each part, which a product by `1 / r` would not match
/// where `r` is so small that its reciprocal overflows.
#[inline]
fn divide<T: Scalar>(x: T, r: T::Real) -> T {
    T::from_parts(x.re() / r, x.im() / r)
}

// ---------------------------------------------------------------------------------------------
// The residual in twice the working precision
// ---------------------------------------------------------------------------------------------

/// `a + b` as the float nearest to it and the rounding error of that, exactly (Knuth's
/// two-sum).
#[inline]
fn two_sum<R: Float>(a: R, b: R) -> (R, R) {
    let sum = a + b;
    let b_part = sum - a;
    (sum, (a - (sum - b_part)) + (b - b_part))
}

/// A sum of products carried as a float and the errors it has dropped so far, which is the
/// arithmetic of twice the working precision that Ogita, Rump and Oishi's `Dot2` does ("Accurate
/// sum and dot product", SIAM J. Sci. Comput. 26, 2005): the result is as accurate as if the
/// whole sum had been taken in doubled precision and rounded at the end.
#[derive(Clone, Copy)]
struct Doubled<R> {
    /// The sum so far, rounded.
    high: R,
    /// What the rounding dropped.
    low: R,
}

impl<R: Float> Doubled<R> {
    /// The sum that holds `x` alone.
    #[inline]
    fn new(x: R) -> Self {
        Self {
            high: x,
            low: R::ZERO,
        }
    }

    /// Subtracts `p·q`, whose own rounding error a fused multiply-add gives exactly.
    #[inline]
    fn subtract_product(&mut self, p: R, q: R) {
        let product = p * q;
        let dropped = p.mul_add(q, -product);
        let (sum, error) = two_sum(self.high, -product);
        self.high = sum;
        self.low += error - dropped;
    }

    /// The sum, rounded once.
    #[inline]
    fn value(self) -> R {
        self.high + self.low
    }
}

/// One element of a residual, `b − Σ aₖ·xₖ` over the pairs `(aₖ, xₖ)` of `terms`, in twice the
/// working precision.
#[inline]
fn residual_element<T: Scalar>(b: T, terms: impl Iterator<Item = (T, T)>) -> T {
    let mut re = Doubled::new(b.re());
    let mut im = Doubled::new(b.im());
    for (a, x) in terms {
        re.subtract_product(a.re(), x.re());
        if T::IS_COMPLEX {
            re.subtract_product(-a.im(), x.im());
            im.subtract_product(a.re(), x.im());
            im.subtract_product(a.im(), x.re());
        }
    }
    T::from_parts(re.value(), im.value())
}

/// The checks of a residual's arguments.
#[track_caller]
fn check_residual<T>(rows: usize, cols: usize, x: &[T], b: &[T], r: &[T]) {
    assert_eq!(
        x.len(),
        cols,
        "the vector has the wrong length for the matrix"
    );
    assert_eq!(
        b.len(),
        rows,
        "the right-hand side has the wrong length for the matrix"
    );
    assert_eq!(
        r.len(),
        rows,
        "the result has the wrong length for the matrix"
    );
}

/// `r = b − A·x` for a sparse matrix, each element in twice the working precision:
/// [`LinearOperator::residual`] of a [`SparseMatrix`].
#[track_caller]
pub(super) fn residual_sparse<T: Scalar>(a: &SparseMatrix<T>, x: &[T], b: &[T], r: &mut [T]) {
    check_residual(a.rows(), a.cols(), x, b, r);
    let (pointers, columns, values) = (a.row_pointers(), a.col_indices(), a.values());
    for (i, r) in r.iter_mut().enumerate() {
        let row = pointers[i]..pointers[i + 1];
        *r = residual_element(b[i], row.map(|k| (values[k], x[columns[k]])));
    }
}

/// `r = b − A·x` for a dense matrix, each element in twice the working precision:
/// [`LinearOperator::residual`] of a [`Matrix`].
#[track_caller]
pub(super) fn residual_dense<T: Scalar>(a: &Matrix<T>, x: &[T], b: &[T], r: &mut [T]) {
    let (rows, cols) = a.shape();
    check_residual(rows, cols, x, b, r);
    let elements = a.as_slice();
    for (i, r) in r.iter_mut().enumerate() {
        *r = residual_element(b[i], (0..cols).map(|j| (elements[i + j * rows], x[j])));
    }
}

// ---------------------------------------------------------------------------------------------
// What the four methods share
// ---------------------------------------------------------------------------------------------

/// A system being solved: the operator, the right-hand side, the preconditioner, the tolerance
/// and the budget, and the count of the products done so far.
struct Session<'a, T: Scalar, A: ?Sized> {
    /// The operator `A`.
    a: &'a A,
    /// The right-hand side.
    b: &'a [T],
    /// The preconditioner, or `None` for the identity.
    pre: Option<&'a dyn Preconditioner<T>>,
    /// `max(rtol·‖b‖₂, atol)`, which the residual has to come down to.
    threshold: T::Real,
    /// `max(threshold, ε·‖b‖₂)`: where a residual carried by a recurrence is worth checking.
    /// Below `ε·‖b‖₂` the recurrence goes on falling while the residual formed afresh cannot
    /// follow it, so a tolerance smaller than that is checked there, and found stalled rather
    /// than chased until the recurrence underflows.
    trigger: T::Real,
    /// The factor the vectors of the inner products of [`cg`] and [`bicgstab`] are multiplied
    /// by, a power of two near the reciprocal of the norm of the first residual (see
    /// [`reciprocal_scale`]) and renewed when the residual has come far from that (see
    /// [`renew`](Self::renew)). The methods divide one inner product by another, so `scale²`
    /// cancels; it keeps the squares of a right-hand side of `10¹⁶⁰` or `10⁻¹⁷⁰` within the
    /// range of `f64`, and as a product by a power of two is exact, the quotients are those of
    /// the unscaled vectors to the last bit.
    scale: f64,
    /// The largest number of iterations.
    max_iter: usize,
    /// The number of products with `A` done so far.
    matvecs: usize,
}

/// How a run begins.
struct Opening<'a, T: Scalar, A: ?Sized> {
    /// The system.
    session: Session<'a, T, A>,
    /// The starting guess.
    x: Vec<T>,
    /// Its residual `b − A·x`, formed afresh.
    r: Vec<T>,
    /// The norm of that residual, which does not pass the test.
    norm: T::Real,
}

impl<'a, T: Scalar, A: LinearOperator<T> + ?Sized> Session<'a, T, A> {
    /// The checks and the first residual every method starts with. `Err` holds what the method
    /// returns at once: `None` for options it cannot serve (`valid` is what the method asks of
    /// them beyond the tolerances), the zero vector for a zero right-hand side, the starting
    /// guess when that already passes the test or when the data are not finite.
    #[track_caller]
    fn open(
        a: &'a A,
        b: &'a [T],
        opts: &SolveOptions<T>,
        pre: Option<&'a dyn Preconditioner<T>>,
        valid: bool,
    ) -> Result<Opening<'a, T, A>, Option<SolveResult<T>>> {
        let n = b.len();
        assert!(a.rows() == a.cols(), "the operator isn't square");
        assert_eq!(a.rows(), n, "the right-hand side has the wrong length");
        if let Some(x0) = &opts.initial_guess {
            assert_eq!(x0.len(), n, "the starting guess has the wrong length");
        }
        let usable = |t: T::Real| t >= T::Real::ZERO && t.is_finite();
        if !valid || !usable(opts.rtol) || !usable(opts.atol) {
            return Err(None);
        }
        let norm_b = norm2_ordered(b);
        if norm_b == T::Real::ZERO {
            return Err(Some(SolveResult {
                x: vec![T::zero(); n],
                residual: T::Real::ZERO,
                iterations: 0,
                matvecs: 0,
                converged: true,
                reason: StopReason::Converged,
            }));
        }
        let threshold = (opts.rtol * norm_b).max(opts.atol);
        let mut session = Session {
            a,
            b,
            pre,
            threshold,
            trigger: threshold.max(T::Real::EPSILON * norm_b),
            scale: 1.0,
            max_iter: opts.max_iter.unwrap_or(n.saturating_mul(10)),
            matvecs: 0,
        };
        let (x, r, norm) = match &opts.initial_guess {
            Some(x0) => {
                let mut r = vec![T::zero(); n];
                let norm = session.residual(x0, &mut r);
                (x0.clone(), r, norm)
            }
            None => (vec![T::zero(); n], b.to_vec(), norm_b),
        };
        if !norm.is_finite() {
            return Err(Some(session.report(x, norm, 0, StopReason::NonFinite)));
        }
        if session.met(norm) {
            return Err(Some(session.report(x, norm, 0, StopReason::Converged)));
        }
        session.scale = reciprocal_scale(norm.to_f64());
        Ok(Opening {
            session,
            x,
            r,
            norm,
        })
    }

    /// `y = A·x`, counted.
    fn apply(&mut self, x: &[T], y: &mut [T]) {
        self.a.apply(x, y);
        self.matvecs += 1;
    }

    /// `r = b − A·x` formed afresh, counted, and its norm.
    fn residual(&mut self, x: &[T], r: &mut [T]) -> T::Real {
        self.a.residual(x, self.b, r);
        self.matvecs += 1;
        norm2_ordered(r)
    }

    /// `z = M⁻¹·r`, a copy of `r` when there is no preconditioner.
    fn precondition(&self, r: &[T], z: &mut [T]) {
        match self.pre {
            Some(m) => m.apply(r, z),
            None => z.copy_from_slice(r),
        }
    }

    /// The inner product `uᴴ·v` times `scale²`, in `f64`; it only ever enters a quotient.
    fn dot(&self, u: &[T], v: &[T]) -> Wide {
        let (re, im) = dot_parts(u, v, self.scale, self.scale);
        Complex::new(re, im)
    }

    /// `x` times `scale`, in `f64`: the size an inner product of the run is compared with.
    fn scaled(&self, x: T::Real) -> f64 {
        x.to_f64() * self.scale
    }

    /// Renews `scale` from the norm `norm` of the residual the recurrence has come to, once
    /// that norm times `scale` has left the range `2⁻²⁰⁰` to `2²⁰⁰`, beyond which the squares
    /// of the scaled vectors could leave the range of `f64` before long. Returns the factor that
    /// turns an inner product taken with the old `scale` into one of the new: the square of
    /// their ratio, an exact power of two, or 1 when nothing changed.
    fn renew(&mut self, norm: T::Real) -> f64 {
        let scaled = norm.to_f64() * self.scale;
        let far = 2f64.powi(200);
        if !(scaled > 0.0 && scaled.is_finite()) || (1.0 / far..=far).contains(&scaled) {
            return 1.0;
        }
        let renewed = reciprocal_scale(norm.to_f64());
        let ratio = renewed / self.scale;
        self.scale = renewed;
        ratio * ratio
    }

    /// Whether a residual of the norm `norm` passes the test. One that is not a number never
    /// does, however large the tolerance.
    fn met(&self, norm: T::Real) -> bool {
        norm.is_finite() && norm <= self.threshold
    }

    /// Whether the norm `norm` of a residual carried by a recurrence calls for the residual to
    /// be formed afresh: it is within the tolerance, or at the level of rounding.
    fn suggests(&self, norm: T::Real) -> bool {
        norm.is_finite() && norm <= self.trigger
    }

    /// The result for the point `x`, whose residual formed afresh has the norm `residual`: marked
    /// converged if that passes the test, whatever stopped the recurrence, and with `reason`
    /// otherwise.
    fn report(
        &self,
        x: Vec<T>,
        residual: T::Real,
        iterations: usize,
        reason: StopReason,
    ) -> SolveResult<T> {
        let converged = self.met(residual);
        debug_assert!(converged || reason != StopReason::Converged);
        SolveResult {
            x,
            residual,
            iterations,
            matvecs: self.matvecs,
            converged,
            reason: if converged {
                StopReason::Converged
            } else {
                reason
            },
        }
    }

    /// The result of a run that stopped at `x`, whose residual formed afresh has the norm
    /// `residual`, unless a point verified earlier, the starting guess among them, did better:
    /// then that one is returned.
    fn conclude(
        &self,
        x: Vec<T>,
        residual: T::Real,
        iterations: usize,
        reason: StopReason,
        best: Best<T>,
    ) -> SolveResult<T> {
        // A residual that is not a number loses to any point verified.
        if residual.is_nan() || residual > best.norm {
            let (point, norm) = best.into_point(x.len());
            return self.report(point, norm, iterations, reason);
        }
        self.report(x, residual, iterations, reason)
    }

    /// [`conclude`](Self::conclude) for a run that stopped short of the tolerance at `x`, whose
    /// recurrence put its residual at `estimate`: the candidate of `best`, if the recurrence
    /// ranked it better, has its residual formed too, and the best of the points is returned.
    /// `work` is overwritten.
    fn finish(
        &mut self,
        x: Vec<T>,
        estimate: T::Real,
        work: &mut [T],
        iterations: usize,
        reason: StopReason,
        mut best: Best<T>,
    ) -> SolveResult<T> {
        let residual = self.residual(&x, work);
        if let Some((norm, point)) = best.candidate.take()
            && !ordered(estimate, norm)
        {
            let truth = self.residual(&point, work);
            best.keep(truth, &point);
        }
        self.conclude(x, residual, iterations, reason, best)
    }
}

/// The points a run may return instead of its last one: the best whose residual was formed
/// afresh, which is the starting guess until another does better, and, since the last such
/// point, the one whose recurrence carried the smallest residual.
///
/// A run that fails returns the best point it reached, as MATLAB's solvers return the iterate of
/// the smallest residual: the residual of a conjugate gradient or BiCGSTAB recurrence is not
/// monotone, and the iterates of MINRES on a singular system that is not consistent drift away
/// once the residual has come down to its least squares value. The starting guess counts as a
/// point reached, so that no method returns a point whose residual is above its.
struct Best<'a, T: Scalar> {
    /// The norm of the residual of the best point verified so far, the starting guess's to
    /// begin with.
    norm: T::Real,
    /// That point, `None` while it is the starting guess, which is not copied.
    point: Option<Vec<T>>,
    /// The starting guess, `None` for the zero vector.
    start: Option<&'a [T]>,
    /// The point of the smallest residual a recurrence carried since the last point verified,
    /// and that estimate.
    candidate: Option<(T::Real, Vec<T>)>,
}

impl<'a, T: Scalar> Best<'a, T> {
    /// The starting guess `start`, `None` for the zero vector, whose residual has the norm
    /// `norm`.
    fn new(norm: T::Real, start: Option<&'a [T]>) -> Self {
        Self {
            norm,
            point: None,
            start,
            candidate: None,
        }
    }

    /// Notes the point `x`, whose residual a recurrence puts at `estimate`: it is kept if no
    /// point noted since the last residual formed afresh had a smaller one.
    fn note(&mut self, estimate: T::Real, x: &[T]) {
        match &mut self.candidate {
            Some((norm, point)) => {
                if ordered(estimate, *norm) {
                    *norm = estimate;
                    point.copy_from_slice(x);
                }
            }
            None => self.candidate = Some((estimate, x.to_vec())),
        }
    }

    /// Notes that the residual of `x`, formed afresh, has the norm `truth`: `x` is kept if that
    /// is below the best so far, and the answer is whether it was. The candidates of the
    /// recurrence are forgotten: they were no better.
    fn keep(&mut self, truth: T::Real, x: &[T]) -> bool {
        self.candidate = None;
        if !ordered(truth, self.norm) {
            return false;
        }
        self.norm = truth;
        match &mut self.point {
            Some(point) => point.copy_from_slice(x),
            None => self.point = Some(x.to_vec()),
        }
        true
    }

    /// [`keep`](Self::keep) for a point looked at as converged: `true` if it did not improve on
    /// the best so far, which means the method has stalled.
    fn stalled(&mut self, truth: T::Real, x: &[T]) -> bool {
        !self.keep(truth, x)
    }

    /// The best point verified, of `n` elements, and the norm of its residual.
    fn into_point(self, n: usize) -> (Vec<T>, T::Real) {
        let point = match (self.point, self.start) {
            (Some(point), _) => point,
            (None, Some(start)) => start.to_vec(),
            (None, None) => vec![T::zero(); n],
        };
        (point, self.norm)
    }
}

// ---------------------------------------------------------------------------------------------
// Conjugate gradients
// ---------------------------------------------------------------------------------------------

/// `A·x = b` by the conjugate gradient method, for a Hermitian positive definite `a`.
///
/// Hestenes and Stiefel's method (1952): the iterate minimises the error in the norm of `A` over
/// the Krylov space, with one product by `A` and two inner products an iteration and six
/// vectors of working memory, the solution and the iterate of the smallest residual so far
/// among them, and a seventh once a residual formed afresh has been kept as the best. It
/// converges in about `√κ` iterations to a fixed accuracy, `κ` the condition number, and is the
/// method to use whenever the matrix is positive definite.
///
/// Only a positive definite matrix makes `pᴴ·A·p` positive for every direction `p`; a direction
/// for which it is not ends the method with [`StopReason::NotPositiveDefinite`]. An indefinite
/// matrix whose first directions happen to have positive curvature is not found out before one
/// of the other kind turns up: [`minres`] is the method for those. For a complex matrix the real
/// part of `pᴴ·A·p` is used, which is all of it when `A` is Hermitian.
///
/// Returns `None` only for options that are not usable (see [`SolveOptions`]).
///
/// # Panics
///
/// Panics if the operator isn't square or `b` is of the wrong length.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, cg};
///
/// let a = SparseMatrix::tridiagonal(&[-1.0; 3], &[2.0; 4], &[-1.0; 3]);
/// let r = cg(&a, &[1.0, 0.0, 0.0, 1.0]).unwrap();
/// assert!(r.converged);
/// assert!(r.x.iter().all(|&x: &f64| (x - 1.0).abs() < 1e-7));
/// ```
#[must_use]
#[track_caller]
pub fn cg<T, A>(a: &A, b: &[T]) -> Option<SolveResult<T>>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    cg_with(a, b, &SolveOptions::new(), None)
}

/// [`cg`] with the options `opts` and the preconditioner `pre`, `None` for no preconditioner.
///
/// The preconditioner `M` has to be Hermitian positive definite, as [`Jacobi`](super::Jacobi)
/// of a positive diagonal and [`Ic0`](super::Ic0) are. The method is then the conjugate gradient
/// method on `L⁻¹·A·L⁻ᴴ` for `M = L·Lᴴ`, arranged so that only `M⁻¹` is applied and the residual
/// carried is still `b − A·x`. A residual `r` with `rᴴ·M⁻¹·r ≤ 0` shows that `M` is not positive
/// definite and ends the method with [`StopReason::NotPositiveDefinite`].
///
/// # Panics
///
/// Panics if the operator isn't square, or if `b`, the starting guess or the vectors the
/// preconditioner takes are of the wrong length.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{Jacobi, SolveOptions, SparseMatrix, cg_with};
///
/// // Rows of very different sizes, which the diagonal preconditioner evens out.
/// let a = SparseMatrix::from_coo(3, 3, &[
///     (0, 0, 1000.0), (0, 1, 1.0),
///     (1, 0, 1.0), (1, 1, 2.0), (1, 2, 0.001),
///     (2, 1, 0.001), (2, 2, 0.003),
/// ]);
/// let b = [1001.0, 3.001, 0.004];
/// let m = Jacobi::new(&a).unwrap();
/// let r = cg_with(&a, &b, &SolveOptions::new().with_rtol(1e-12), Some(&m)).unwrap();
/// assert!(r.converged);
/// assert!(r.x.iter().all(|&x: &f64| (x - 1.0).abs() < 1e-9));
/// ```
#[must_use]
#[track_caller]
pub fn cg_with<T, A>(
    a: &A,
    b: &[T],
    opts: &SolveOptions<T>,
    pre: Option<&dyn Preconditioner<T>>,
) -> Option<SolveResult<T>>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    let (mut s, mut x, mut r, opening_norm) = match Session::open(a, b, opts, pre, true) {
        Ok(opening) => (opening.session, opening.x, opening.r, opening.norm),
        Err(done) => return done,
    };
    let n = b.len();
    // Without a preconditioner `z` is `r` itself, not a copy of it.
    let mut z = if pre.is_some() {
        vec![T::zero(); n]
    } else {
        Vec::new()
    };
    let mut p = vec![T::zero(); n];
    let mut q = vec![T::zero(); n];
    let mut best = Best::new(opening_norm, opts.initial_guess.as_deref());
    let mut iterations = 0;
    // Whether the next direction starts a recurrence, from the residual formed afresh in `r`.
    let mut fresh = true;
    // `rᴴ·M⁻¹·r` of the residual the recurrence is at, times `scale²` (see `Wide`).
    let mut rz = 0.0f64;
    // The norm of the residual the recurrence carries for `x`.
    let mut estimate = opening_norm;
    let reason = loop {
        let zr: &[T] = if pre.is_some() {
            s.precondition(&r, &mut z);
            &z
        } else {
            &r
        };
        let rz_next = s.dot(&r, zr).re;
        if !rz_next.is_finite() {
            break StopReason::NonFinite;
        }
        // The residual is not zero here, so only a preconditioner that is not positive definite
        // gives this.
        if rz_next <= 0.0 {
            break StopReason::NotPositiveDefinite;
        }
        if fresh {
            p.copy_from_slice(zr);
        } else {
            let beta = T::Real::from_f64(rz_next / rz);
            for (p, &z) in p.iter_mut().zip(zr) {
                *p = z + p.scale(beta);
            }
        }
        fresh = false;
        rz = rz_next;
        if iterations == s.max_iter {
            break StopReason::IterationLimit;
        }
        s.apply(&p, &mut q);
        iterations += 1;
        let curvature = s.dot(&p, &q).re;
        if !curvature.is_finite() {
            break StopReason::NonFinite;
        }
        if curvature <= 0.0 {
            break StopReason::NotPositiveDefinite;
        }
        let alpha = T::Real::from_f64(rz / curvature);
        if !alpha.is_finite() {
            break StopReason::NonFinite;
        }
        let squares = advance(&mut x, &p, &mut r, &q, alpha);
        let norm = norm2_from(&r, squares);
        if !norm.is_finite() {
            break StopReason::NonFinite;
        }
        estimate = norm;
        best.note(norm, &x);
        // The next inner product is divided by this one, so both are to be of the same factor.
        rz *= s.renew(norm);
        if s.suggests(norm) {
            let truth = s.residual(&x, &mut r);
            if s.met(truth) || !truth.is_finite() || best.stalled(truth, &x) {
                let reason = if truth.is_finite() {
                    StopReason::Stagnated
                } else {
                    StopReason::NonFinite
                };
                return Some(s.conclude(x, truth, iterations, reason, best));
            }
            s.renew(truth);
            fresh = true;
        }
    };
    Some(s.finish(x, estimate, &mut q, iterations, reason, best))
}

// ---------------------------------------------------------------------------------------------
// MINRES
// ---------------------------------------------------------------------------------------------

/// `A·x = b` by the minimal residual method, for a Hermitian `a` that need not be definite.
///
/// Paige and Saunders' MINRES ("Solution of sparse indefinite systems of linear equations",
/// SIAM J. Numer. Anal. 12, 1975): the Lanczos process turns `A` into a tridiagonal matrix on
/// the Krylov space, whose QR factorisation by one Givens rotation an iteration gives the
/// iterate of the smallest residual in that space and, from the rotations alone, the norm of
/// that residual. One product by `A` an iteration and eight vectors of working memory, the
/// solution among them (nine with a preconditioner), and one more once a point it verified is
/// kept as the best; the residual never grows.
///
/// A singular matrix is fine as long as the system is consistent: from the zero starting guess
/// the method converges to the solution of the smallest norm. Where the right-hand side is not
/// in the range there is no solution to converge to. The residual then comes down to its least
/// squares value, after which, in floating point arithmetic, the iterates drift along the null
/// space while the estimate of the residual goes on falling. The method keeps watch with Choi,
/// Paige and Saunders' estimate of `‖A·r‖` for the residual `r` of its iterate ("MINRES-QLP",
/// SIAM J. Sci. Comput. 33, 2011), which its rotations give for nothing, measured against
/// `‖T‖·‖r‖` for the tridiagonal matrices `T` of the run:
///
/// * where that is at most `4·ε`, `r` is a null vector of `A` to working precision and the
///   iterate a least squares solution: the method stops there, with [`StopReason::Breakdown`];
/// * where it is below `64·√ε`, and again at each level an eighth of the last, the method
///   verifies the iterate, forming its residual afresh, and keeps it if it is the best so far.
///   Should the estimate then claim a residual a sixty-fourth below the smallest one formed
///   afresh so far, the residual is formed afresh again. Once a residual verified in the run
///   was a null vector to half the working precision (below `√ε`), one that does not come out
///   below that smallest residual means the iterates are drifting along the null space, and one
///   that comes out below it by less than half the decrease claimed means the estimate has lost
///   track of the residual: its next claim is checked against the new residual in turn.
///
/// A drift so found, or an estimate of convergence that the residual formed afresh does not
/// bear out, begins a new Lanczos process from the iterate, which holds what the last one found
/// of the hard part of the solution even where rounding errors have spoilt its residual: on an
/// ill-conditioned matrix in single precision that is what takes the method on. That goes on as
/// long as each process improves on the best point verified, by a sixty-fourth after a drift.
/// When one does not, the method stops with the best point: with [`StopReason::Breakdown`] if a
/// residual it verified was a null vector to half the working precision, the matrix singular to
/// that precision and the right-hand side not in its range, and with
/// [`StopReason::Stagnated`] otherwise.
///
/// On a system that is not consistent, that point is a least squares solution to the accuracy
/// the arithmetic allows (on the singular Laplacians of the tests within 1 % of the least
/// squares residual, mostly to all its digits), but not the one of least norm, which is
/// MINRES-QLP's. A system that is merely close to singular is not stopped: the step past a near
/// null vector is the one that solves it, and the point verified before it is only kept in case
/// it does not. Between the two, a matrix that is singular only up to the rounding of its
/// elements is singular to working precision; the method mostly ends there as on a singular
/// one, but its iterates may also run to the iteration limit, which then returns the best point
/// verified.
///
/// Returns `None` only for options that are not usable (see [`SolveOptions`]).
///
/// # Panics
///
/// Panics if the operator isn't square or `b` is of the wrong length.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, StopReason, minres};
///
/// // Symmetric and indefinite: the eigenvalues are 3 and −1.
/// let a = SparseMatrix::<f64>::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 2.0), (1, 1, 1.0)]);
/// let r = minres(&a, &[5.0, 4.0]).unwrap();
/// assert!(r.converged);
/// assert!((r.x[0] - 1.0).abs() < 1e-7 && (r.x[1] - 2.0).abs() < 1e-7);
///
/// // Singular but consistent: the solution of the smallest norm.
/// let s = SparseMatrix::<f64>::from_coo(2, 2, &[(0, 0, 1.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)]);
/// let r = minres(&s, &[2.0, 2.0]).unwrap();
/// assert!(r.converged);
/// assert!((r.x[0] - 1.0).abs() < 1e-7 && (r.x[1] - 1.0).abs() < 1e-7);
///
/// // The Laplacian of a path with free ends is singular, its null space the constants, and a
/// // ramp on top of a constant is not in its range: a least squares solution, and the reason.
/// let mut diagonal = [2.0; 30];
/// (diagonal[0], diagonal[29]) = (1.0, 1.0);
/// let path = SparseMatrix::tridiagonal(&[-1.0; 29], &diagonal, &[-1.0; 29]);
/// let b: Vec<f64> = (0..30).map(|i| 1.0 + i as f64 / 100.0).collect();
/// let r = minres(&path, &b).unwrap();
/// assert_eq!(r.reason, StopReason::Breakdown);
/// // The least squares residual is the part of b along the constants, |mean(b)|·√30.
/// let least = b.iter().sum::<f64>() / 30.0 * 30f64.sqrt();
/// assert!((r.residual - least).abs() <= 1e-6 * least);
/// ```
#[must_use]
#[track_caller]
pub fn minres<T, A>(a: &A, b: &[T]) -> Option<SolveResult<T>>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    minres_with(a, b, &SolveOptions::new(), None)
}

/// [`minres`] with the options `opts` and the preconditioner `pre`, `None` for no
/// preconditioner.
///
/// The preconditioner `M` has to be Hermitian positive definite, even though `A` need not be:
/// the method is MINRES on `L⁻¹·A·L⁻ᴴ` for `M = L·Lᴴ`, arranged so that only `M⁻¹` is applied.
/// What it then minimises, and what its rotations estimate, is the residual in the norm of
/// `M⁻¹`, `√(rᴴ·M⁻¹·r)`. The stopping test is on `‖b − A·x‖₂` all the same: with a
/// preconditioner the residual vector is carried by a recurrence of its own, at the cost of one
/// more vector, and its norm taken at every iteration. The checks of [`minres`] are made in the
/// norms of `M⁻¹` (`‖A·r‖` there means `‖L⁻¹·A·M⁻¹·r‖₂`), so the least squares solution of a
/// system that is not consistent is one in the norm of `M⁻¹`; only the best point is chosen by
/// `‖b − A·x‖₂`. A residual `r` with `rᴴ·M⁻¹·r < 0` ends the method with
/// [`StopReason::NotPositiveDefinite`].
///
/// # Panics
///
/// Panics if the operator isn't square, or if `b`, the starting guess or the vectors the
/// preconditioner takes are of the wrong length.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{Jacobi, SolveOptions, SparseMatrix, minres_with};
///
/// let a = SparseMatrix::from_coo(3, 3, &[
///     (0, 0, 100.0), (0, 1, 1.0),
///     (1, 0, 1.0), (1, 1, 2.0), (1, 2, 3.0),
///     (2, 1, 3.0), (2, 2, 1.0),
/// ]);
/// let b = [101.0, 6.0, 4.0];
/// let m = Jacobi::new(&a).unwrap();
/// let r = minres_with(&a, &b, &SolveOptions::new().with_rtol(1e-12), Some(&m)).unwrap();
/// assert!(r.converged && r.iterations <= 3);
/// assert!(r.x.iter().all(|&x: &f64| (x - 1.0).abs() < 1e-10));
/// ```
#[must_use]
#[track_caller]
pub fn minres_with<T, A>(
    a: &A,
    b: &[T],
    opts: &SolveOptions<T>,
    pre: Option<&dyn Preconditioner<T>>,
) -> Option<SolveResult<T>>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    // The Lanczos vectors are carried unnormalised, as `current = βₖ·vₖ`, and with them
    // `z = M⁻¹·current`, so that `w = z / βₖ` is the vector the product is taken of. The first
    // vector of a process is divided by its norm instead, so that the scale of the right-hand
    // side enters only `φ`, the right-hand side of the least squares problem, and no quotient of
    // the recurrence can overflow on its account (a subnormal right-hand side made `α / β` do
    // that).
    let (mut s, mut x, mut current, opening_norm) = match Session::open(a, b, opts, pre, true) {
        Ok(opening) => (opening.session, opening.x, opening.r, opening.norm),
        Err(done) => return done,
    };
    let n = b.len();
    let (zero, one) = (T::Real::ZERO, T::Real::ONE);
    let eps = T::Real::EPSILON;
    let mut previous = vec![T::zero(); n];
    let mut next = vec![T::zero(); n];
    let mut z = vec![T::zero(); n];
    let mut w = vec![T::zero(); n];
    // The last two columns of `W·R⁻¹`, of which the iterate is a combination.
    let mut direction = vec![T::zero(); n];
    let mut direction_before = vec![T::zero(); n];
    // The residual, by its recurrence; only a preconditioner makes it necessary.
    let mut residual = if pre.is_some() {
        vec![T::zero(); n]
    } else {
        Vec::new()
    };
    let mut best = Best::new(opening_norm, opts.initial_guess.as_deref());
    let mut iterations = 0;
    // The constants of the watch (see `minres`): the level of `‖A·r‖ / (‖T‖·‖r‖)` below which
    // it verifies points, each time an eighth of the last; the part of a verified residual that
    // a decrease the recurrence claims beyond it has to be for the residual formed afresh to be
    // asked to confirm it, which is also the part by which a process that ended in a drift has
    // to have improved on the best point for another to be worth beginning; the level of
    // `‖A·r‖ / (‖T‖·‖r‖)` at which a residual counts as a null vector of `A`; and the part of a
    // claimed decrease that the residual formed afresh has to show for the claim to be settled.
    let watch_from = T::Real::from_f64(64.0) * eps.sqrt();
    let part = T::Real::from_f64(1.0 / 64.0);
    let eighth = T::Real::from_f64(0.125);
    let null = eps.sqrt();
    let half = T::Real::from_f64(0.5);
    // The norm of the residual of the best point when the last process began, the smallest
    // `‖A·r‖ / (‖T‖·‖r‖)` of a point verified, and the smallest residual formed afresh so far in
    // the norm of `M⁻¹` (the 2-norm without a preconditioner), below which a claim of the
    // recurrence has to come to be borne out.
    let mut since = opening_norm;
    let mut least = T::Real::INFINITY;
    let mut lowest = T::Real::INFINITY;
    // The largest column sum of the tridiagonal matrices `T` so far, at least the norm of each
    // and at most √3 times that of `A` (of `L⁻¹·A·L⁻ᴴ` with a preconditioner): the size a pivot
    // is measured against. It is kept from one process to the next, so that a process that
    // begins at a near null vector measures it against what is known of `A`.
    let mut scale = zero;
    // Where a pivot is taken for zero.
    let rounding = T::Real::from_f64(4.0) * eps;
    let reason = 'method: loop {
        // A Lanczos process begins, from the residual formed afresh in `current`.
        s.precondition(&current, &mut z);
        let norm = inner_root(&current, &z);
        if !norm.is_finite() {
            break StopReason::NonFinite;
        }
        // The residual is not zero here, so only a preconditioner that is not positive definite
        // gives this.
        if norm <= zero {
            break StopReason::NotPositiveDefinite;
        }
        lowest = lowest.min(norm);
        if pre.is_some() {
            residual.copy_from_slice(&current);
        }
        for (current, z) in current.iter_mut().zip(z.iter_mut()) {
            *current = divide(*current, norm);
            *z = divide(*z, norm);
        }
        // The norm of `current` as it is stored.
        let mut beta = one;
        let mut beta_before = zero;
        // The rotations of the two columns before this one, the identity to begin with.
        let (mut c1, mut s1, mut c2, mut s2) = (one, zero, one, zero);
        // What the rotations have left of the right-hand side `β₁·e₁` below the triangle: the
        // norm of the residual, up to its sign.
        let mut phi = norm;
        previous.fill(T::zero());
        direction.fill(T::zero());
        direction_before.fill(T::zero());
        // The level of `‖A·r‖ / (‖T‖·‖r‖)` at which the watch verifies the next point, and, while
        // a claim beyond a point it verified is pending, the residual in the norm of `M⁻¹` that
        // the claim is measured against.
        let mut watch = watch_from;
        let mut claimed_below: Option<T::Real> = None;
        let mut first = true;
        loop {
            if iterations == s.max_iter {
                break 'method StopReason::IterationLimit;
            }
            for (w, &z) in w.iter_mut().zip(&z) {
                *w = divide(z, beta);
            }
            s.apply(&w, &mut next);
            iterations += 1;
            // The next Lanczos vector, `A·w − βₖ·vₖ₋₁ − αₖ·vₖ`, with `αₖ` taken after the first
            // subtraction: Paige's arrangement ("Error analysis of the Lanczos algorithm for
            // tridiagonalizing a symmetric matrix", 1976), the one that keeps the process stable.
            if !first {
                let from_previous = beta / beta_before;
                for (next, &previous) in next.iter_mut().zip(&previous) {
                    *next -= previous.scale(from_previous);
                }
            }
            let alpha = dot(&w, &next).re();
            if !alpha.is_finite() {
                break 'method StopReason::NonFinite;
            }
            let from_current = alpha / beta;
            for (next, &current) in next.iter_mut().zip(&current) {
                *next -= current.scale(from_current);
            }
            core::mem::swap(&mut previous, &mut current);
            core::mem::swap(&mut current, &mut next);
            s.precondition(&current, &mut z);
            let beta_next = inner_root(&current, &z);
            if !beta_next.is_finite() {
                break 'method StopReason::NonFinite;
            }
            if beta_next < zero {
                break 'method StopReason::NotPositiveDefinite;
            }
            // The new column of the tridiagonal matrix is `β` (not in the first), `α`,
            // `β_next`: the two rotations before it are applied, and a third one made that
            // takes `β_next` out.
            let above = if first { zero } else { beta };
            let epsilon = s2 * above;
            let carried = c2 * above;
            let delta = c1 * carried + s1 * alpha;
            let pivot = c1 * alpha - s1 * carried;
            let gamma = pivot.hypot(beta_next);
            scale = scale.max(alpha.abs() + above + beta_next);
            if !gamma.is_finite() {
                break 'method StopReason::NonFinite;
            }
            // `‖A·rₖ₋₁‖ = |φ|·√(pivot² + (c₁·β_next)²)` for the residual of the iterate before
            // this step (in the norms of `M⁻¹` with a preconditioner), from the rotations alone:
            // `rₖ₋₁` is `φ·Vₖ·q` for the last column `q` of the rotations so far, and the
            // Hermitian `T` turns `q` into `pivot·eₖ` with `c₁·β_next` below it. This is Choi,
            // Paige and Saunders' estimate ("MINRES-QLP", SIAM J. Sci. Comput. 33, 2011). Where
            // it vanishes to working precision, `rₖ₋₁` is a null vector of `A`: the system has no
            // solution, the iterate is a least squares solution, and the step would divide by a
            // pivot of rounding errors.
            let normal = pivot.hypot(c1 * beta_next);
            if normal <= rounding * scale {
                let truth = s.residual(&x, &mut next);
                let reason = if truth.is_finite() {
                    StopReason::Breakdown
                } else {
                    StopReason::NonFinite
                };
                return Some(s.conclude(x, truth, iterations, reason, best));
            }
            let (c, sn) = (pivot / gamma, beta_next / gamma);
            let tau = c * phi;
            let phi_next = -sn * phi;
            for ((before, &direction), &w) in direction_before.iter_mut().zip(&direction).zip(&w) {
                *before = divide(w - direction.scale(delta) - before.scale(epsilon), gamma);
            }
            core::mem::swap(&mut direction, &mut direction_before);
            // The watch: a residual nearer a null vector than any before in this process. The
            // iterate before the step is verified and kept if it is the best, and what the
            // recurrence claims beyond its residual, or beyond the smallest residual formed
            // afresh before it where that is lower, is to be checked: a process begun from a
            // point worse than the best one has to come below the best for a claim to count.
            if normal <= watch * scale {
                watch = normal / scale * eighth;
                least = least.min(normal / scale);
                let truth = s.residual(&x, &mut next);
                if s.met(truth) || !truth.is_finite() {
                    let reason = if truth.is_finite() {
                        StopReason::Stagnated
                    } else {
                        StopReason::NonFinite
                    };
                    return Some(s.conclude(x, truth, iterations, reason, best));
                }
                best.keep(truth, &x);
                lowest = lowest.min(m_norm(pre, &next, truth, &mut w));
                claimed_below = Some(lowest);
            }
            for (x, &d) in x.iter_mut().zip(&direction) {
                *x += d.scale(tau);
            }
            let estimate = if pre.is_some() {
                // rₖ = sₖ²·rₖ₋₁ + cₖ·φₖ·vₖ₊₁, which is zero when the process has ended.
                let keep = sn * sn;
                let add = if beta_next > zero {
                    c * phi_next / beta_next
                } else {
                    zero
                };
                for (r, &v) in residual.iter_mut().zip(&current) {
                    *r = r.scale(keep) + v.scale(add);
                }
                norm2(&residual)
            } else {
                phi_next.abs()
            };
            if !estimate.is_finite() {
                break 'method StopReason::NonFinite;
            }
            (c2, s2, c1, s1) = (c1, s1, c, sn);
            phi = phi_next;
            beta_before = beta;
            beta = beta_next;
            first = false;
            // The estimate is checked when it says the method has converged (a process that has
            // ended, `β_next = 0`, leaves an estimate of zero, so it is looked at here too and
            // never divided by), and when it claims a decrease beyond a point the watch verified.
            let converging = s.suggests(estimate);
            let claimed = claimed_below.is_some_and(|level| phi.abs() <= (one - part) * level);
            if converging || claimed {
                let truth = s.residual(&x, &mut next);
                if s.met(truth) || !truth.is_finite() {
                    let reason = if truth.is_finite() {
                        StopReason::Stagnated
                    } else {
                        StopReason::NonFinite
                    };
                    return Some(s.conclude(x, truth, iterations, reason, best));
                }
                best.keep(truth, &x);
                let fresh = m_norm(pre, &next, truth, &mut w);
                lowest = lowest.min(fresh);
                // In exact arithmetic `|φ|` is the norm of the residual. An estimate of
                // convergence the residual formed afresh does not bear out has lost track of it,
                // by rounding errors, which an ill-conditioned matrix makes grow. So has a claimed
                // decrease the residual formed afresh does not bear out once a residual verified
                // in the run was a null vector to half the working precision: past a least
                // squares solution the iterates drift along the null space while the estimate
                // goes on falling. That evidence is the run's, not the process's: a process begun
                // from a drifting iterate starts far from the least squares point, and its own
                // residuals may never come that near a null vector again. A claim the residual
                // bears out lets the process go on; but where the residual came down by less
                // than half the decrease claimed, the estimate has lost track of it all the same,
                // and its next claim is checked against this residual in turn, since a decrease
                // that happens to come with the first steps of a drift would otherwise leave the
                // rest of it unwatched. Without such evidence a claim only makes the method look.
                let mut drifting = false;
                if let Some(level) = claimed_below.take()
                    && least <= null
                {
                    if !ordered(fresh, level) {
                        drifting = true;
                    } else if fresh > level - (level - phi.abs()) * half {
                        claimed_below = Some(fresh);
                    }
                }
                if converging || drifting {
                    // A new process begins from the iterate, which holds what the last one
                    // found of the hard part of the solution even where rounding errors have
                    // spoilt its residual, as long as the last process improved on the best
                    // point: by a sixty-fourth where the iterates drift, since along the null
                    // space they change the residual by rounding errors alone. Otherwise the
                    // tolerance is beyond what the arithmetic can reach, or, where a point
                    // verified had a residual that is a null vector to half the working
                    // precision, the matrix is singular to that precision and the right-hand
                    // side not in its range.
                    let required = if drifting { one - part } else { one };
                    if !ordered(best.norm, required * since) {
                        let reason = if least <= null {
                            StopReason::Breakdown
                        } else {
                            StopReason::Stagnated
                        };
                        return Some(s.conclude(x, truth, iterations, reason, best));
                    }
                    since = best.norm;
                    current.copy_from_slice(&next);
                    continue 'method;
                }
            }
        }
    };
    Some(s.finish(x, zero, &mut next, iterations, reason, best))
}

// ---------------------------------------------------------------------------------------------
// GMRES
// ---------------------------------------------------------------------------------------------

/// The rotation `[c, s; −s̄, c]`, `c` real, that takes `(a, b)` to `(ρ, 0)` for a real `b > 0`:
/// returns `(c, s, ρ)`, with `ρ` of the argument of `a` and of the modulus `√(|a|² + b²)`.
fn rotation<T: Scalar>(a: T, b: T::Real) -> (T::Real, T, T) {
    let modulus = a.abs();
    if modulus == T::Real::ZERO {
        return (T::Real::ZERO, T::one(), T::from_real(b));
    }
    let length = modulus.hypot(b);
    let phase = divide(a, modulus);
    (
        modulus / length,
        phase.scale(b / length),
        phase.scale(length),
    )
}

/// Writes `v / length` as the vector `k` of the basis, which grows by one vector when it has to
/// and otherwise reuses the one of an earlier cycle.
fn set_basis<T: Scalar>(basis: &mut Vec<Vec<T>>, k: usize, v: &[T], length: T::Real) {
    if k == basis.len() {
        basis.push(v.iter().map(|&x| divide(x, length)).collect());
    } else {
        for (target, &x) in basis[k].iter_mut().zip(v) {
            *target = divide(x, length);
        }
    }
}

/// `A·x = b` by the generalised minimal residual method, restarted, for any square `a`.
///
/// Saad and Schultz's GMRES(m) ("GMRES: a generalized minimal residual algorithm for solving
/// nonsymmetric linear systems", SIAM J. Sci. Stat. Comput. 7, 1986): the Arnoldi process builds
/// an orthonormal basis of the Krylov space and the Hessenberg matrix of `A` on it, whose QR
/// factorisation by one Givens rotation an iteration gives the norm of the smallest residual in
/// the space without forming the iterate. After `m` iterations, the restart length, the iterate
/// is formed and the method starts again from it, which bounds the memory at the `m` vectors
/// of the basis and four more (five with a preconditioner), and the work of an iteration at
/// `m` inner products. The residual never grows.
///
/// * The basis is orthogonalised by classical Gram–Schmidt, every coefficient from the same
///   vector, and once more when the first pass cancelled more than a factor `√2` of the
///   vector, which is when it leaves the basis short of orthogonal.
/// * A vector that vanishes against the basis means the Krylov space is invariant, and then
///   the solution lies in it: the iterate is formed from the columns found so far, which is
///   exact but for rounding. This is the lucky breakdown, and it is how the identity matrix is
///   solved in one iteration. If the new column is itself a combination of the others, the
///   matrix is singular on the space and the right-hand side not in its range: the column is
///   dropped, the least squares solution taken, and the method ends with
///   [`StopReason::Breakdown`] once a cycle cannot improve on it.
/// * A restart loses what the space had learnt, and for a small `m` the method may stop
///   converging altogether. A cycle that does not reduce the residual at all would be repeated
///   for ever, so it ends the method with [`StopReason::Stagnated`]: ask for a longer restart
///   or a preconditioner.
/// * The restart length is clamped to the number of unknowns, where the method is the full
///   one, and the memory is taken as the basis grows, not for `m` vectors beforehand.
///
/// Returns `None` only for options that are not usable (see [`SolveOptions`]), which here
/// includes a restart length of zero.
///
/// # Panics
///
/// Panics if the operator isn't square or `b` is of the wrong length.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, gmres};
///
/// let a = SparseMatrix::from_coo(3, 3, &[
///     (0, 0, 3.0), (0, 1, 1.0),
///     (1, 1, 4.0), (1, 2, 1.0),
///     (2, 0, 2.0), (2, 2, 5.0),
/// ]);
/// let r = gmres(&a, &[4.0, 5.0, 7.0]).unwrap();
/// assert!(r.converged && r.iterations <= 3);
/// assert!(r.x.iter().all(|&x: &f64| (x - 1.0).abs() < 1e-7));
///
/// // The identity: the first Krylov vector is the solution.
/// let i = SparseMatrix::<f64>::identity(3);
/// let r = gmres(&i, &[1.0, 2.0, 3.0]).unwrap();
/// assert!(r.converged && r.iterations == 1);
/// assert!(r.x.iter().zip([1.0, 2.0, 3.0]).all(|(x, b)| (x - b).abs() < 1e-15));
/// ```
#[must_use]
#[track_caller]
pub fn gmres<T, A>(a: &A, b: &[T]) -> Option<SolveResult<T>>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    gmres_with(a, b, &SolveOptions::new(), None)
}

/// [`gmres`] with the options `opts`, the restart length among them, and the preconditioner
/// `pre`, `None` for no preconditioner.
///
/// The preconditioner `M`, any nonsingular one, is applied on the right: the method solves
/// `A·M⁻¹·u = b` and returns `x = M⁻¹·u`. The residual of that system is the residual of the one
/// given, so what the method minimises and estimates is `‖b − A·x‖₂` itself, whatever the
/// preconditioner does to the scale of the matrix. It costs one application of `M⁻¹` an
/// iteration and one more a cycle.
///
/// # Panics
///
/// Panics if the operator isn't square, or if `b`, the starting guess or the vectors the
/// preconditioner takes are of the wrong length.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{Jacobi, SolveOptions, SparseMatrix, StopReason, gmres_with};
///
/// let a = SparseMatrix::from_coo(3, 3, &[
///     (0, 0, 3.0), (0, 1, 1.0),
///     (1, 1, 4.0), (1, 2, 1.0),
///     (2, 0, 2.0), (2, 2, 5.0),
/// ]);
/// let b = [4.0, 5.0, 7.0];
/// let m = Jacobi::new(&a).unwrap();
/// let opts = SolveOptions::new().with_restart(2).with_rtol(1e-12);
/// let r = gmres_with(&a, &b, &opts, Some(&m)).unwrap();
/// assert!(r.converged);
/// assert!(r.x.iter().all(|&x: &f64| (x - 1.0).abs() < 1e-10));
///
/// // The cyclic shift of three elements gives restarted GMRES nothing until the third
/// // iteration: restarted every two, it is where it started, and says so.
/// let shift = SparseMatrix::from_coo(3, 3, &[(1, 0, 1.0), (2, 1, 1.0), (0, 2, 1.0)]);
/// let stuck = gmres_with(&shift, &[1.0, 0.0, 0.0], &opts, None).unwrap();
/// assert_eq!(stuck.reason, StopReason::Stagnated);
/// let full = gmres_with(&shift, &[1.0, 0.0, 0.0], &SolveOptions::new(), None).unwrap();
/// assert!(full.converged && full.iterations == 3);
/// assert_eq!(full.x, [0.0, 0.0, 1.0]);
/// ```
#[must_use]
#[track_caller]
pub fn gmres_with<T, A>(
    a: &A,
    b: &[T],
    opts: &SolveOptions<T>,
    pre: Option<&dyn Preconditioner<T>>,
) -> Option<SolveResult<T>>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    let (mut s, mut x, mut r, mut norm) = match Session::open(a, b, opts, pre, opts.restart > 0) {
        Ok(opening) => (opening.session, opening.x, opening.r, opening.norm),
        Err(done) => return done,
    };
    let n = b.len();
    let restart = opts.restart.min(n);
    let eps = T::Real::EPSILON;
    let half_root = T::Real::from_f64(core::f64::consts::FRAC_1_SQRT_2);
    let mut basis: Vec<Vec<T>> = Vec::new();
    // The triangular factor of the Hessenberg matrix, column after column: column `j` holds
    // its `j + 1` elements from offset `j·(j + 1)/2`.
    let mut triangle: Vec<T> = Vec::new();
    let mut rotations: Vec<(T::Real, T)> = Vec::new();
    // The right-hand side `‖r‖·e₁` of the least squares problem, rotated as the matrix is.
    let mut rhs: Vec<T> = Vec::new();
    let mut column: Vec<T> = Vec::new();
    let mut correction: Vec<T> = Vec::new();
    let mut y: Vec<T> = Vec::new();
    let mut w = vec![T::zero(); n];
    let mut z = if pre.is_some() {
        vec![T::zero(); n]
    } else {
        Vec::new()
    };
    let mut before_cycle = vec![T::zero(); n];
    let mut iterations = 0;
    loop {
        // A cycle begins, from the residual formed afresh in `r`, of the norm `norm`.
        if iterations >= s.max_iter {
            return Some(s.report(x, norm, iterations, StopReason::IterationLimit));
        }
        set_basis(&mut basis, 0, &r, norm);
        triangle.clear();
        rotations.clear();
        rhs.clear();
        rhs.push(T::from_real(norm));
        let mut k = 0;
        // Whether a column of this cycle was dropped as a combination of the others: the
        // matrix, preconditioned, is singular on the Krylov space.
        let mut singular = false;
        while k < restart && iterations < s.max_iter {
            match pre {
                Some(m) => {
                    m.apply(&basis[k], &mut z);
                    s.apply(&z, &mut w);
                }
                None => s.apply(&basis[k], &mut w),
            }
            iterations += 1;
            let length = norm2(&w);
            if !length.is_finite() {
                return Some(s.report(x, norm, iterations, StopReason::NonFinite));
            }
            // Classical Gram–Schmidt with the test of Daniel, Gragg, Kaufman and Stewart: all
            // the coefficients from one pass over the basis, the combination taken out in a
            // second, and both again when the first shrank the vector by more than 1 − 1/√2.
            multi_dot(&basis[..=k], &w, &mut column);
            multi_subtract(&mut w, &basis[..=k], &column);
            let mut remainder = norm2(&w);
            if remainder < length * half_root {
                multi_dot(&basis[..=k], &w, &mut correction);
                multi_subtract(&mut w, &basis[..=k], &correction);
                for (h, &c) in column.iter_mut().zip(&correction) {
                    *h += c;
                }
                let again = norm2(&w);
                // Twice is enough: a vector that shrinks as much a second time has nothing
                // outside the space but rounding errors.
                remainder = if again < remainder * half_root {
                    T::Real::ZERO
                } else {
                    again
                };
            }
            if !remainder.is_finite() {
                return Some(s.report(x, norm, iterations, StopReason::NonFinite));
            }
            for (i, &(c, sn)) in rotations.iter().enumerate() {
                let (upper, lower) = (column[i], column[i + 1]);
                column[i] = upper.scale(c) + sn * lower;
                column[i + 1] = lower.scale(c) - sn.conj() * upper;
            }
            if remainder <= eps * length {
                // The space is invariant: the least squares problem is a square system, solved
                // exactly by the columns found, this one included unless it is itself a
                // combination of the others, which is the mark of a singular matrix.
                if column[k].abs() > eps * length {
                    triangle.extend_from_slice(&column);
                    k += 1;
                } else {
                    singular = true;
                }
                break;
            }
            let (c, sn, rho) = rotation(column[k], remainder);
            column[k] = rho;
            triangle.extend_from_slice(&column);
            rotations.push((c, sn));
            let top = rhs[k];
            rhs[k] = top.scale(c);
            rhs.push(-(sn.conj() * top));
            k += 1;
            if s.suggests(rhs[k].abs()) {
                break;
            }
            if k < restart {
                set_basis(&mut basis, k, &w, remainder);
            }
        }
        if k == 0 {
            // The one column of this cycle was dropped: `A·M⁻¹·r` is zero to rounding.
            return Some(s.report(x, norm, iterations, StopReason::Breakdown));
        }
        // The combination `y` of the basis by back substitution, and the correction `M⁻¹·V·y`.
        y.clear();
        y.resize(k, T::zero());
        for i in (0..k).rev() {
            let mut sum = rhs[i];
            for j in i + 1..k {
                sum = sum.mul_sub(triangle[j * (j + 1) / 2 + i], y[j]);
            }
            y[i] = sum / triangle[i * (i + 1) / 2 + i];
        }
        w.fill(T::zero());
        for (v, &c) in basis.iter().zip(&y) {
            add_scaled(&mut w, c, v);
        }
        before_cycle.copy_from_slice(&x);
        match pre {
            Some(m) => {
                m.apply(&w, &mut z);
                add_scaled(&mut x, T::one(), &z);
            }
            None => add_scaled(&mut x, T::one(), &w),
        }
        let truth = s.residual(&x, &mut r);
        if s.met(truth) {
            return Some(s.report(x, truth, iterations, StopReason::Converged));
        }
        // In exact arithmetic a cycle never increases the residual, and leaves it as it was only
        // if it leaves the iterate as it was, in which case every further cycle would do the
        // same: a singular system that is not consistent, or a restart too short to reduce it,
        // or a tolerance below what the arithmetic resolves. A residual that is not a number is
        // not below anything, and ends the method too.
        if !ordered(truth, norm) {
            let reason = if !truth.is_finite() {
                StopReason::NonFinite
            } else if singular {
                StopReason::Breakdown
            } else {
                StopReason::Stagnated
            };
            return Some(s.report(before_cycle, norm, iterations, reason));
        }
        norm = truth;
    }
}

// ---------------------------------------------------------------------------------------------
// BiCGSTAB
// ---------------------------------------------------------------------------------------------

/// `A·x = b` by the stabilised biconjugate gradient method, for any square `a`.
///
/// Van der Vorst's Bi-CGSTAB ("Bi-CGSTAB: a fast and smoothly converging variant of Bi-CG for
/// the solution of nonsymmetric linear systems", SIAM J. Sci. Stat. Comput. 13, 1992): a step of
/// the biconjugate gradient method, taken against a fixed shadow residual and without the
/// transpose, followed by one step of residual minimisation that smooths it. Two products by `A`
/// an iteration and nine vectors of working memory whatever the number of iterations (the
/// solution and the iterate of the smallest residual so far among them, and a tenth once a
/// residual formed afresh has been kept as the best), which is what it has over [`gmres`]; in
/// exchange the residual does not decrease monotonically and the method can break down.
///
/// * When the first half of an iteration already brings the residual within the tolerance, the
///   method stops there, with one product spent on the iteration.
/// * The recurrence divides by three inner products: `ρ = r̂ᴴ·r`, `r̂ᴴ·A·p̂` and, through `ω`,
///   `tᴴ·s`. One of them below `ε²` times the norms of its two vectors ends the method with
///   [`StopReason::Breakdown`] at the last iterate; a skew-symmetric matrix gives that at once.
///   One that is small but not that small is not a breakdown, and the method goes on, as van
///   der Vorst's does: near convergence `ρ` falls faster than the residual. The residual formed
///   afresh before any verdict keeps a recurrence that has lost its accuracy from being taken
///   for converged.
///
/// Returns `None` only for options that are not usable (see [`SolveOptions`]).
///
/// # Panics
///
/// Panics if the operator isn't square or `b` is of the wrong length.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{SparseMatrix, StopReason, bicgstab};
///
/// let a = SparseMatrix::from_coo(3, 3, &[
///     (0, 0, 3.0), (0, 1, 1.0),
///     (1, 1, 4.0), (1, 2, 1.0),
///     (2, 0, 2.0), (2, 2, 5.0),
/// ]);
/// let r = bicgstab(&a, &[4.0, 5.0, 7.0]).unwrap();
/// assert!(r.converged);
/// assert!(r.x.iter().all(|&x: &f64| (x - 1.0).abs() < 1e-7));
///
/// // A rotation by a quarter turn is skew-symmetric, so that r̂ᴴ·A·r vanishes at the first
/// // step: the method cannot take it, and says so.
/// let turn = SparseMatrix::from_coo(2, 2, &[(0, 1, -1.0), (1, 0, 1.0)]);
/// let r = bicgstab(&turn, &[1.0, 2.0]).unwrap();
/// assert_eq!(r.reason, StopReason::Breakdown);
/// assert!(!r.converged);
/// ```
#[must_use]
#[track_caller]
pub fn bicgstab<T, A>(a: &A, b: &[T]) -> Option<SolveResult<T>>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    bicgstab_with(a, b, &SolveOptions::new(), None)
}

/// [`bicgstab`] with the options `opts` and the preconditioner `pre`, `None` for no
/// preconditioner.
///
/// The preconditioner `M`, any nonsingular one, is applied on the right, in both halves of an
/// iteration: the search direction and the intermediate residual each go through `M⁻¹` before
/// their product by `A`. The method is then Bi-CGSTAB on `A·M⁻¹`, and the residual it carries
/// is `b − A·x` itself.
///
/// # Panics
///
/// Panics if the operator isn't square, or if `b`, the starting guess or the vectors the
/// preconditioner takes are of the wrong length.
///
/// # Examples
///
/// ```
/// use a::math::sparse::{Ilu0, SolveOptions, SparseMatrix, bicgstab_with};
///
/// // Convection and diffusion on forty points.
/// const N: usize = 40;
/// let a = SparseMatrix::tridiagonal(&[-1.5; N - 1], &[2.0; N], &[-0.5; N - 1]);
/// let b = vec![1.0; N];
/// let opts = SolveOptions::new().with_rtol(1e-10);
/// let plain = bicgstab_with(&a, &b, &opts, None).unwrap();
/// let m = Ilu0::new(&a).unwrap();
/// let fast = bicgstab_with(&a, &b, &opts, Some(&m)).unwrap();
/// assert!(plain.converged && fast.converged);
/// // A tridiagonal matrix has no fill, so its incomplete factors are the complete ones.
/// assert!(fast.iterations == 1 && plain.iterations > 10);
/// ```
#[must_use]
#[track_caller]
pub fn bicgstab_with<T, A>(
    a: &A,
    b: &[T],
    opts: &SolveOptions<T>,
    pre: Option<&dyn Preconditioner<T>>,
) -> Option<SolveResult<T>>
where
    T: Scalar,
    A: LinearOperator<T> + ?Sized,
{
    let (mut s, mut x, mut r, mut norm) = match Session::open(a, b, opts, pre, true) {
        Ok(opening) => (opening.session, opening.x, opening.r, opening.norm),
        Err(done) => return done,
    };
    let n = b.len();
    // An inner product is taken for zero when it is below `ε²` times the norms of its two vectors,
    // Eigen's measure of a breakdown. A small one that is not that small is still information: near
    // convergence `ρ = r̂ᴴ·r` falls faster than `‖r‖`, to a cosine of about 4·10⁻¹⁴ on a Helmholtz
    // problem that the method then solves in one more step. A recurrence that has lost its accuracy
    // is not taken for converged all the same, since every verdict is made on the residual formed
    // afresh.
    let negligible = (T::Real::EPSILON * T::Real::EPSILON).to_f64();
    // The shadow residual, kept at the residual the recurrence started from.
    let mut shadow = r.clone();
    let mut shadow_norm = norm;
    let mut p = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut t = vec![T::zero(); n];
    // `M⁻¹·p` and `M⁻¹·s`.
    let mut p_hat = vec![T::zero(); n];
    let mut s_hat = vec![T::zero(); n];
    // The scalars of the recurrence, in `f64` (see `Wide`).
    let unit = Wide::new(1.0, 0.0);
    let (mut rho_before, mut alpha, mut omega) = (unit, unit, unit);
    let mut best = Best::new(norm, opts.initial_guess.as_deref());
    let mut iterations = 0;
    // Whether the next direction starts a recurrence, from the residual formed afresh in `r`.
    let mut fresh = true;
    // The norm of the residual the recurrence carries for `x`.
    let mut estimate = norm;
    let reason = loop {
        // The inner products are those of the run's factor (see `Session::scale`), and so are
        // the norms they are measured against.
        let rho = s.dot(&shadow, &r);
        if !rho.is_finite() {
            break StopReason::NonFinite;
        }
        if rho.abs() <= negligible * s.scaled(shadow_norm) * s.scaled(norm) {
            break StopReason::Breakdown;
        }
        if fresh {
            p.copy_from_slice(&r);
        } else {
            let beta: T = narrow((rho / rho_before) * (alpha / omega));
            let smooth: T = narrow(omega);
            for ((p, &r), &v) in p.iter_mut().zip(&r).zip(&v) {
                *p = r + beta * p.mul_sub(smooth, v);
            }
        }
        fresh = false;
        if iterations == s.max_iter {
            break StopReason::IterationLimit;
        }
        s.precondition(&p, &mut p_hat);
        s.apply(&p_hat, &mut v);
        iterations += 1;
        let sigma = s.dot(&shadow, &v);
        if !sigma.is_finite() {
            break StopReason::NonFinite;
        }
        if sigma.abs() <= negligible * s.scaled(shadow_norm) * s.scaled(norm2(&v)) {
            break StopReason::Breakdown;
        }
        // The step lengths enter the next direction as they were applied to `x` and `r`, which
        // is what the relations the direction rests on hold for.
        let step: T = narrow(rho / sigma);
        if !step.is_finite() {
            break StopReason::NonFinite;
        }
        alpha = widen(step);
        // The residual of the first half, `s = r − α·v`, takes the place of `r`.
        subtract_scaled(&mut r, step, &v);
        let half = norm2(&r);
        if !half.is_finite() {
            break StopReason::NonFinite;
        }
        let mut looked = None;
        if s.suggests(half) {
            add_scaled(&mut x, step, &p_hat);
            looked = Some(s.residual(&x, &mut r));
        } else {
            s.precondition(&r, &mut s_hat);
            s.apply(&s_hat, &mut t);
            let tt = s.dot(&t, &t).re;
            let ts = s.dot(&t, &r);
            // Whatever the second half does, the first is a step of its own.
            add_scaled(&mut x, step, &p_hat);
            estimate = half;
            best.note(half, &x);
            if !tt.is_finite() || !ts.is_finite() {
                break StopReason::NonFinite;
            }
            // `A·M⁻¹·s = 0` for an `s` that is not: the matrix is singular.
            if tt <= 0.0 {
                break StopReason::Breakdown;
            }
            let smooth: T = narrow(Wide::new(ts.re / tt, ts.im / tt));
            omega = widen(smooth);
            add_scaled(&mut x, smooth, &s_hat);
            subtract_scaled(&mut r, smooth, &t);
            norm = norm2(&r);
            if !norm.is_finite() {
                break StopReason::NonFinite;
            }
            estimate = norm;
            best.note(norm, &x);
            if s.suggests(norm) {
                looked = Some(s.residual(&x, &mut r));
            } else if ts.abs() <= negligible * tt.sqrt() * s.scaled(half) {
                break StopReason::Breakdown;
            }
        }
        if let Some(truth) = looked {
            if s.met(truth) || !truth.is_finite() || best.stalled(truth, &x) {
                let reason = if truth.is_finite() {
                    StopReason::Stagnated
                } else {
                    StopReason::NonFinite
                };
                return Some(s.conclude(x, truth, iterations, reason, best));
            }
            shadow.copy_from_slice(&r);
            shadow_norm = truth;
            norm = truth;
            fresh = true;
        }
        // The next `ρ` is divided by this one, so both are to be of the same factor.
        rho_before = rho * s.renew(norm);
    };
    Some(s.finish(x, estimate, &mut t, iterations, reason, best))
}
