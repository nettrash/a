//! The zeros of a system of equations, by Newton's method with a Jacobian and by Broyden's
//! method.
//!
//! A system of `n` equations in `n` unknowns is a function `f` from `Rⁿ` to `Rⁿ`, its unknowns
//! and its values each held as a [`Matrix`] of one column. [`newton`] takes the Jacobian `J` of `f`
//! too and solves the correction equation `J·δ = −f(x)` on [`Matrix::lu`]; [`broyden`] needs `f`
//! alone, starts from a Jacobian by forward differences (or the caller's) and improves it at every
//! step by Broyden's rank-one update, carried on its QR factorisation.
//!
//! Both are globalised by the backtracking line search of Dennis and Schnabel (*Numerical Methods
//! for Unconstrained Optimization and Nonlinear Equations*, 1983, algorithm A6.3.1) on the merit
//! function `½‖f‖²`: the full step is tried first, and a shorter one, from a quadratic and then
//! cubic model of the merit along the step, until the merit falls by at least `10⁻⁴` of what the
//! linear model promises. That brings the methods in from starting points where plain Newton
//! diverges; [`SystemOptions::with_line_search`] turns it off.
//!
//! Whether a point is a root is decided by three tests, each independent of the units of `x` and
//! of `f`, which [`SystemStop`] describes; a method that stops without one of them holding reports
//! why, with the best point it reached and `converged` false.
//!
//! The result is `None` only for inputs no method can start from: a starting point or a first
//! residual that is not finite, or options that are negative, not finite or zero where they must
//! not be. Inputs of the wrong shape panic.
//!
//! ```
//! use a::math::matrix::Matrix;
//! use a::math::numerics::{SystemStop, broyden, newton_system};
//!
//! // Rosenbrock's function as a system, 10(y − x²) = 0 and 1 − x = 0, from the classic start.
//! let f = |v: &Matrix<f64>| {
//!     let (x, y) = (v[(0, 0)], v[(1, 0)]);
//!     Matrix::from_column_major(2, 1, vec![10.0 * (y - x * x), 1.0 - x])
//! };
//! let j = |v: &Matrix<f64>| Matrix::from_rows(&[[-20.0 * v[(0, 0)], 10.0], [-1.0, 0.0]]);
//! let x0 = Matrix::from_column_major(2, 1, vec![-1.2, 1.0]);
//! let r = newton_system(&x0, f, j).unwrap();
//! assert!(r.converged && r.stop == SystemStop::Residual);
//! assert_eq!(r.x.as_slice(), &[1.0, 1.0]);
//! let b = broyden(&x0, f).unwrap();
//! assert!(b.converged && (b.x[(0, 0)] - 1.0).abs() < 1e-15);
//! ```

use super::{DEFAULT_MAX_ITER, Real};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};

/// The fraction of the decrease of the merit `½‖f‖²` that its linear model predicts which a step
/// must deliver: the sufficient decrease (Armijo) condition of Dennis and Schnabel's line search.
const ARMIJO: f64 = 1e-4;

/// The number of epsilons by which a residual may exceed the size of its linear terms and still
/// be rounding: a few operations' worth.
const BACKWARD: f64 = 8.0;

/// The most times a line search shortens its step before it gives up.
const BACKTRACKS: usize = 64;

/// How closely the difference quotients of a restart at two steps must agree, relative to the
/// larger of the two, beyond the rounding of the values they are formed from.
const AGREE: f64 = 0.1;

/// The most times a restart quarters the step of a column of differences to make the quotients
/// at two steps agree: enough to take a step of `√ε·|xₖ|` down to the spacing of the floats.
const QUARTERS: usize = 16;

/// How closely `f` must follow its linear model along a correction for the rounding test, as a
/// fraction of the change the model predicts there: a sixteenth, between the largest deviation
/// measured at a root where the methods stop on the test (0.023, on the battery of Moré, Garbow
/// and Hillstrom's systems) and the smallest at a point that is no root (0.20, a line passing the
/// unit circle at `10⁻⁴` from it, at an unknown of `1.8·10¹⁴`).
const LINEAR: f64 = 0.0625;

/// The number of steps, `4δ, 16δ, …, 4096δ`, at which the rounding test compares `f` with its
/// linear model along the correction `δ`.
const SCALES: usize = 6;

/// The options of [`newton`] and [`broyden`]: the tolerances, how many steps are run, how the
/// finite differences of [`broyden`] are formed, and whether the line search is used.
///
/// # Examples
///
/// ```
/// use a::math::numerics::SystemOptions;
///
/// let opts = SystemOptions::<f64>::new().with_ftol(1e-10).with_max_iter(50).with_line_search(false);
/// assert_eq!((opts.ftol, opts.max_iter, opts.line_search), (1e-10, 50, false));
/// assert_eq!(opts.xtol, 4.0 * f64::EPSILON);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SystemOptions<T> {
    /// The tolerance of the step test, relative to each unknown: a Newton correction with
    /// `|δᵢ| ≤ xtol·|xᵢ|` for every `i` ends the iteration.
    pub xtol: T,
    /// The absolute tolerance on the residual: `‖f(x)‖∞ ≤ ftol` ends the iteration. Zero, the
    /// default, asks for none, and leaves the residual to the test against its linear terms.
    pub ftol: T,
    /// The largest number of iterations, each one Newton or Broyden step.
    pub max_iter: usize,
    /// The relative step of the differences of [`broyden`] (forward for its first Jacobian,
    /// central for a restart): `h·|xₖ|`, or `h` where that is zero or subnormal; `√ε` by default.
    /// A restart checks its differences against steps a quarter as long, or four times as long,
    /// and takes another step where this one does not hold.
    pub h: T,
    /// Whether the steps are shortened by the line search where the full step does not lower
    /// `½‖f‖²` enough; `true` by default.
    pub line_search: bool,
}

impl<T: Real> Default for SystemOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> SystemOptions<T> {
    /// The default options: `xtol` 4ε, `ftol` 0, `max_iter` 100, the difference step `√ε`, and
    /// the line search on.
    #[must_use]
    pub fn new() -> Self {
        let eps = T::EPSILON;
        Self {
            xtol: T::from_f64(4.0) * eps,
            ftol: T::ZERO,
            max_iter: DEFAULT_MAX_ITER,
            h: Float::sqrt(eps),
            line_search: true,
        }
    }

    /// Sets [`xtol`](Self::xtol).
    #[must_use]
    pub fn with_xtol(mut self, xtol: T) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets [`ftol`](Self::ftol).
    #[must_use]
    pub fn with_ftol(mut self, ftol: T) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets [`max_iter`](Self::max_iter).
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets [`h`](Self::h).
    #[must_use]
    pub fn with_finite_difference_step(mut self, h: T) -> Self {
        self.h = h;
        self
    }

    /// Sets [`line_search`](Self::line_search).
    #[must_use]
    pub fn with_line_search(mut self, on: bool) -> Self {
        self.line_search = on;
        self
    }

    /// Whether the tolerances are numbers a method can work with: `xtol` and `ftol` neither
    /// negative nor infinite, `h` positive and finite.
    fn valid(&self) -> bool {
        let ok = |t: T| t >= T::ZERO && Float::is_finite(t);
        ok(self.xtol) && ok(self.ftol) && self.h > T::ZERO && Float::is_finite(self.h)
    }
}

/// Why [`newton`] or [`broyden`] stopped, and the tests of a root they stop on.
///
/// No test on the residual alone can be right in every unit, and no test on the step alone can
/// tell a root from a point where the step is small for another reason. A point `x` is accepted as
/// a root by one of three tests, each independent of the units of `x` and of `f`, which the first
/// three variants name:
///
/// * [`Residual`](Self::Residual): `f(x)` exactly zero, `‖f(x)‖∞ ≤ ftol` when the caller gives an
///   `ftol`, or the residual down to rounding and one more step to prove it. The residual is down
///   to rounding when every equation is satisfied to within `8ε` of the size of its own linear
///   terms, `|fᵢ(x)| ≤ 8ε·(|J|·|x|)ᵢ` (the backward error of `x` as an exact root of the system
///   with its Jacobian perturbed by `8ε` relatively), and the Newton correction `δ` changes no
///   unknown by more than `ε^(2/3)` of itself. Neither is enough alone: the size of the linear
///   terms overstates the rounding of an equation whose terms cancel or whose coupling `f`
///   evaluates exactly, as `0.01·(x₁ − 10¹⁶)` at `x₁ = 10¹⁶`, and near a root that is nearly
///   double a short correction is not a small error. So `δ` is taken, to `y`, and the next
///   correction is solved for there on the same factorisation, `δ̄ = −J⁻¹·f(y)`, the simplified
///   Newton correction of Deuflhard's natural monotonicity test. `y` is a root when `δ̄` changes
///   no unknown by more than `xtol` of itself. While `δ̄` is shorter than `δ`, relative to the
///   unknowns, the iteration is still converging, and goes on from `y`. When it is no shorter,
///   the iteration has stopped converging: at the floor of the rounding of `f`, or because
///   Newton's step overshoots where the size of the linear terms was a loose bound — with
///   `arctan(x₂ − 10¹²)` coupled to `x₁ = 10¹⁶` as above, the step from `x₂ = 10¹² + 2` lands at
///   `10¹² − 3.5`. That is no proof of a root; the rounding test decides, by the line search
///   along `δ`.
/// * [`Step`](Self::Step): the Newton correction at the point changes no unknown by more than
///   `xtol` of itself, `|δᵢ| ≤ xtol·|xᵢ|`; the step is taken, and the better point returned.
/// * [`Rounding`](Self::Rounding): the line search finds no point along a Newton correction `δ`
///   below `ε^(2/3)` of every unknown that lowers `½‖f‖²` enough, and `f` is seen to follow its
///   linear model along `δ` at longer steps. It is tried where the line search fails, and where
///   the residual test found that the iteration had stopped converging (with the line search off
///   too: there the iteration goes on from the point the line search found, if it found one). To
///   first order `f(x + λ·δ) = (1 − λ)·f(x)`, and the search would succeed at once; what stops it
///   is a deviation from that model as large as `f(x)`, which is either the rounding of `f` — as
///   at the root of the trigonometric test function of Moré, Garbow and Hillstrom, whose terms
///   cancel to `10⁻¹⁶` — and of the points to floats, or the shape of `f`: a local minimum of
///   `‖f‖` that is no root, where the floats of an unknown are too coarse for the steps that would
///   find a lower value (`(u − 1/3)² + 1` at `x = 10¹⁴ + u`, where the floats are `1/64` apart),
///   or a kink. A rounding does not grow with the step, and a curvature grows as its square: so
///   `f` is evaluated at `x + k·δ` for `k = 4, 16, …, 4096`, and the point is a root, to rounding,
///   when at two successive `k` every equation is within a sixteenth of its linear model, measured
///   by the smaller of the change the model predicts for the residual and the size of the
///   equation's own terms along the step: `|fᵢ(x + k·δ) − (1 − k)·fᵢ(x)| ≤ (k − 1)·min(‖f(x)‖∞,
///   Σⱼ |Jᵢⱼ·δⱼ|)/16`. A curvature that was as large as `f(x)` at `δ` is `k²` times larger at
///   `k·δ`, and fails the test at every `k`. For one equation both sizes are `|f(x)|`, and a step
///   that passes is a proof of a root between `x` and `x + k·δ`, where `f` has the sign opposite
///   to `f(x)`; for a system the test is a check of the model, not a proof, and presumes that no
///   feature of `f` between the length of `δ` and 4096 times it mimics the model at two steps in
///   every equation. Like the line search, its first size compares the equations with one
///   another: it does not depend on the units of `f`, but on how its equations are scaled
///   against each other. Where `f` does not follow its model the method stops with `Rounding` and
///   `converged` false, at the best point: so at the local minimum above, and at the kink
///   `|x − 1| + 10⁻¹²`.
///
/// Short of an `ftol`, which accepts what it is told to, a point these tests accept has a Newton
/// correction below `xtol` (the step test) or `ε^(2/3)` (the others) of every unknown: to first
/// order it is that close to a root of `f` as evaluated, and at a regular root, with `f` evaluated
/// accurately, the step taken after the test leaves it within a few roundings. Where the rounding
/// of `f` itself leaves the root less certain, as in an ill-conditioned problem, the point is as
/// near as that rounding allows and no test can tell more: Powell's badly scaled system in `f32`,
/// whose second unknown the rounding of `f` fixes only to about `10⁻⁴`, is answered to `4·10⁻⁵`.
/// The step test and the residual test are of the first order, and accept a point that is no
/// root, or one further from a root than the tolerance, where `f` curves within the distances
/// they rest on, `xtol·|xᵢ|` and, for the bound of the residual test, `8ε·|xᵢ|`:
/// * at a root where the Jacobian is singular, approached only linearly, the correction
///   understates the error by about the multiplicity `m`, and the confirming step accepts a point
///   up to `m/(1 − 1/m)^(m−1)·xtol` from a root of multiplicity `m` (4 at `m = 2`, 6.75 at 3, 17.6
///   at 7) — measured, `2(m − 1)·xtol` on `(x − 1)ᵐ`, `m` from 2 to 7 in `f64` and `f32`, and
///   `4·xtol` on a double root coupled to an unknown of `10¹⁶`;
/// * at a local minimum of `|f|` that is no root, but whose least value `c` is below about
///   `f″·(xtol·|x|)²/2`: a tolerance on the step cannot tell it from a double root, which it is of
///   `f − c`. Measured on `(u − 1/3)² + c` at `x = T + u`: the step test accepted such points up
///   to 0.8 times that bound (`f32`: 0.7), the residual test up to 3.2 times (2.8); so in `f64`
///   from `T = 6.3·10¹²` for `c = 10⁻⁴`, `6.3·10¹³` for `c = 10⁻²` and `6.3·10¹⁴` for `c = 1`.
///   On systems of two equations without a root whose least `‖f‖` is as small (a line passing
///   the unit circle closely, `uv + c = 0` on `u = v`) they did from `T = 10¹²`;
/// * where an unknown is so large that these distances are not small against the scale on which
///   `f` varies in it, at the resolution of the format. On `u³ − 2u − 5` and `u³ − 2u + 2` at
///   `x = T + u`, with a root and a local minimum of `|f|` a unit or so apart, every converged
///   answer was within `0.47·xtol·|x|` of the root up to `T = 2.5·10¹⁴`; from `T = 3.2·10¹⁴`, where
///   `xtol·|x| = 0.28`, the step and residual tests accepted points up to `10.8·xtol·|x|` from it,
///   as `u = 1.25` for the root `−1.77` (in `f32` from `T = 10⁶`, up to 7.7).
///
/// The rounding test accepted none of these points: where it was tried on them, the method ended
/// with `Rounding` and `converged` false.
///
/// [`broyden`] believes none of these on its approximation alone: when one holds, or its line
/// search fails, with an approximation that was updated, or that is its first one by forward
/// differences, it forms the Jacobian afresh there by checked central differences and goes on, and
/// its confirming step stays within the steps over which those were seen to hold. A difference
/// quotient is wrong in its leading digit where `f` curves within the step more than it rises
/// (a forward one, near two roots closer than about `√ε` of their size) or where the third
/// derivative dominates within it (a central one, where the step spans the scale on which `f`
/// varies, as `e^(x − 10⁹)` with a step of 15, or near a root of multiplicity three or more): so
/// each column is checked against the quotients at a step a quarter as long, and the step
/// quartered until two agree to a tenth (or lengthened, where rounding keeps them apart); a
/// Jacobian that cannot be checked so is reported [`Singular`](Self::Singular). The variants after
/// `Rounding` are failures, reported with the best point reached and `converged` false, and so is
/// `Rounding` where its test fails. A root whose unknowns
/// are zero cannot pass a relative step test, and one at which the Jacobian is singular may not be
/// reached within `max_iter` steps: give such problems an `ftol`, in the units of `f`. The methods
/// trust the residual and the Jacobian they are given; [`newton`] cannot tell a wrong Jacobian
/// from a right one.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{SystemStop, newton_system};
///
/// // x² + y² + 1 = 0 and x = y have no real solution. Newton's steps descend the residual towards
/// // its least value, 1 at the origin, where the Jacobian is singular, and no test of a root holds.
/// let f = |v: &Matrix<f64>| {
///     let (x, y) = (v[(0, 0)], v[(1, 0)]);
///     Matrix::from_column_major(2, 1, vec![x * x + y * y + 1.0, x - y])
/// };
/// let j = |v: &Matrix<f64>| Matrix::from_rows(&[[2.0 * v[(0, 0)], 2.0 * v[(1, 0)]], [1.0, -1.0]]);
/// let r = newton_system(&Matrix::from_column_major(2, 1, vec![1.0, 2.0]), f, j).unwrap();
/// assert!(!r.converged && r.stop == SystemStop::Iterations);
/// assert!(r.residual >= 1.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SystemStop {
    /// The residual test held: `f(x)` exactly zero, `‖f(x)‖∞ ≤ ftol`, or every `|fᵢ(x)|` within
    /// `8ε` of the size of its linear terms, and the step after it confirmed the root. Converged.
    Residual,
    /// The Newton correction at the point changed no unknown by more than `xtol` of itself.
    /// Converged.
    Step,
    /// No step along a Newton correction below `ε^(2/3)` of every unknown lowered the residual.
    /// Converged when `f` follows its linear model along the correction at longer steps: the
    /// residual is down to the rounding of `f`. Not converged, with the best point, when it does
    /// not: the shape of `f` stopped the line search — a local minimum of `‖f‖` that is no root,
    /// at an unknown whose floats are too coarse to go further, or a kink.
    Rounding,
    /// The Jacobian was singular or not finite, or a step could not be solved for: the method
    /// does not divide by it. For [`broyden`], the Jacobian of a restart was singular too, or its
    /// differences could not be checked: no two steps gave quotients that agree.
    Singular,
    /// No point along the Newton direction lowered `½‖f‖²` enough: a local minimum of `‖f‖`
    /// that is not a root, or a function that is not smooth there. For [`broyden`], with the
    /// Jacobian of a restart too.
    LineSearch,
    /// `max_iter` iterations were made.
    Iterations,
    /// With the line search off, a step landed where `f` is not finite.
    NotFinite,
}

/// What [`newton`] or [`broyden`] found.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{SystemStop, broyden};
///
/// // e^x = 2 and y² = 4 from (0, 1): one equation in each unknown.
/// let f = |v: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![v[(0, 0)].exp() - 2.0, v[(1, 0)] * v[(1, 0)] - 4.0]);
/// let r = broyden(&Matrix::from_column_major(2, 1, vec![0.0, 1.0]), f).unwrap();
/// assert!(r.converged && matches!(r.stop, SystemStop::Residual | SystemStop::Step));
/// assert!((r.x[(0, 0)] - core::f64::consts::LN_2).abs() < 1e-15 && (r.x[(1, 0)] - 2.0).abs() < 1e-15);
/// // The residual reported is the one at the point reported, and its norm.
/// assert_eq!(r.f.as_slice(), f(&r.x).as_slice());
/// assert_eq!(r.residual, r.f.as_slice().iter().fold(0.0f64, |m, v| m.max(v.abs())));
/// // The first Jacobian by differences cost two evaluations of f, of the total counted.
/// assert!(r.jacobians >= 1 && r.evaluations > r.iterations + 2);
/// ```
#[derive(Debug, Clone)]
pub struct SystemResult<T> {
    /// The root when the method converged; otherwise the iterate of the smallest residual.
    pub x: Matrix<T>,
    /// The residual `f(x)`.
    pub f: Matrix<T>,
    /// Its infinity norm, `‖f(x)‖∞`.
    pub residual: T,
    /// The number of Newton or Broyden steps taken.
    pub iterations: usize,
    /// The number of evaluations of `f`, those of the line search, of the finite differences and
    /// of the rounding test (at most six) included.
    pub evaluations: usize,
    /// The number of Jacobians formed: evaluations of `j` for [`newton`]; for [`broyden`], the
    /// first Jacobian and every one of a restart (`n` evaluations of `f` by forward differences,
    /// at least `4n` by checked central ones).
    pub jacobians: usize,
    /// Whether the method stopped on one of its tests of a root: always with
    /// [`SystemStop::Residual`] and [`SystemStop::Step`], never with the failures, and with
    /// [`SystemStop::Rounding`] where the rounding test held.
    pub converged: bool,
    /// Why the method stopped.
    pub stop: SystemStop,
}

/// The infinity norm of a column, or NaN if an entry is NaN (`Float::max` would pass over it).
fn norm_inf<T: Real + Scalar>(a: &Matrix<T>) -> T {
    let mut m = T::ZERO;
    for &x in a.as_slice() {
        let v = Float::abs(x);
        if v.is_nan() {
            return T::NAN;
        }
        m = m.max(v);
    }
    m
}

/// The 2-norm of a column, scaled by its largest entry so that the squares neither overflow nor
/// underflow; infinite or NaN when an entry is.
fn norm2<T: Real + Scalar>(a: &Matrix<T>) -> T {
    let big = norm_inf(a);
    if big == T::ZERO || !Float::is_finite(big) {
        return big;
    }
    let mut s = T::ZERO;
    for &x in a.as_slice() {
        let t = x / big;
        s += t * t;
    }
    big * Float::sqrt(s)
}

/// Whether every entry is finite.
fn finite<T: Real + Scalar>(a: &Matrix<T>) -> bool {
    a.as_slice().iter().all(|&v| Float::is_finite(v))
}

/// The backward error test: every `|fᵢ|` at most `8ε·(|J|·|x|)ᵢ`. A bound that overflows fails
/// it: an infinite one would pass any finite residual, as `10¹⁰·(x − 10²⁹⁹)` at `1.005·10²⁹⁹`,
/// where the residual is `6·10³⁰⁶`.
fn backward<T: Real + Scalar>(fx: &Matrix<T>, j: &Matrix<T>, x: &Matrix<T>) -> bool {
    let n = x.rows();
    let limit = T::from_f64(BACKWARD) * T::EPSILON;
    (0..n).all(|i| {
        let mut scale = T::ZERO;
        for k in 0..n {
            scale += Float::abs(j[(i, k)]) * Float::abs(x[(k, 0)]);
        }
        Float::is_finite(scale) && Float::abs(fx[(i, 0)]) <= limit * scale
    })
}

/// The length of the correction `v` relative to the point `x`, unit by unit: the largest
/// `|vᵢ|/|xᵢ|`, infinite where an `xᵢ` is zero and its `vᵢ` is not. Both are finite.
fn relative<T: Real + Scalar>(v: &Matrix<T>, x: &Matrix<T>) -> T {
    let mut m = T::ZERO;
    for (&d, &p) in v.as_slice().iter().zip(x.as_slice()) {
        if d != T::ZERO {
            m = m.max(Float::abs(d) / Float::abs(p));
        }
    }
    m
}

/// The step test: every `|δᵢ| ≤ tol·|xᵢ|`.
fn small_step<T: Real + Scalar>(delta: &Matrix<T>, x: &Matrix<T>, tol: T) -> bool {
    delta
        .as_slice()
        .iter()
        .zip(x.as_slice())
        .all(|(&d, &v)| Float::abs(d) <= tol * Float::abs(v))
}

/// `x + λ·p`.
fn along<T: Real + Scalar>(x: &Matrix<T>, p: &Matrix<T>, lambda: T) -> Matrix<T> {
    let mut y = x.clone();
    for (v, &d) in y.as_mut_slice().iter_mut().zip(p.as_slice()) {
        *v += lambda * d;
    }
    y
}

/// Evaluates `f` and checks the shape of what it returns.
#[track_caller]
fn evaluate<T: Real + Scalar>(
    f: &mut impl FnMut(&Matrix<T>) -> Matrix<T>,
    x: &Matrix<T>,
    evaluations: &mut usize,
) -> Matrix<T> {
    let fx = f(x);
    *evaluations += 1;
    assert!(
        fx.shape() == x.shape(),
        "the residual is {}×{}, not a column of {} rows",
        fx.rows(),
        fx.cols(),
        x.rows()
    );
    fx
}

/// The point and the residual where a line search ended, or nothing when it found no point low
/// enough.
type Found<T> = Option<(Matrix<T>, Matrix<T>, bool)>;

/// Dennis and Schnabel's backtracking line search (algorithm A6.3.1) along the step `p` from `x`,
/// where the residual is `fx`, on the merit `½‖f‖²` normalised by its value at `x` (so that it
/// cannot overflow), whose slope along `p` is then `−1`, as it is for a Newton step. The full step
/// is tried first; a shorter one comes from the quadratic through the merit, its slope and the
/// first trial, then from the cubic through the last two trials, kept within `[0.1, 0.5]` of the
/// step before; a trial where `f` is not finite is cut to a tenth. It gives up when the step would
/// change no unknown by more than `xtol` of itself, or after 64 trials. Returns the point, its
/// residual, and whether the full step was taken.
fn line_search<T: Real + Scalar>(
    f: &mut impl FnMut(&Matrix<T>) -> Matrix<T>,
    x: &Matrix<T>,
    fx: &Matrix<T>,
    p: &Matrix<T>,
    xtol: T,
    evaluations: &mut usize,
) -> Found<T> {
    let norm0 = norm2(fx);
    let alpha = T::from_f64(ARMIJO);
    let (half, slope) = (T::HALF, -T::ONE);
    // The relative length of the step, and the shortest that still changes something.
    let mut length = T::ZERO;
    for (&d, &v) in p.as_slice().iter().zip(x.as_slice()) {
        if v != T::ZERO {
            length = length.max(Float::abs(d) / Float::abs(v));
        }
    }
    let shortest = if length > T::ZERO {
        xtol / length
    } else {
        T::ZERO
    };
    let mut lambda = T::ONE;
    let (mut previous, mut merit_previous) = (T::ZERO, T::ZERO);
    for trial in 0..BACKTRACKS {
        let y = along(x, p, lambda);
        let fy = evaluate(f, &y, evaluations);
        let merit = if finite(&fy) {
            let r = norm2(&fy) / norm0;
            half * r * r
        } else {
            T::INFINITY
        };
        if merit <= half + alpha * lambda * slope {
            return Some((y, fy, trial == 0));
        }
        if lambda < shortest {
            return None;
        }
        let next = if !Float::is_finite(merit) {
            lambda * T::from_f64(0.1)
        } else if trial == 0 {
            // The minimum of the quadratic through the merit at 0 and 1 with the slope at 0.
            -slope / (T::TWO * (merit - half - slope))
        } else {
            // The minimum of the cubic through the merit at 0, the slope there, and the last two
            // trials.
            let r1 = merit - half - lambda * slope;
            let r2 = merit_previous - half - previous * slope;
            let d = lambda - previous;
            let a = (r1 / (lambda * lambda) - r2 / (previous * previous)) / d;
            let b = (-previous * r1 / (lambda * lambda) + lambda * r2 / (previous * previous)) / d;
            let t = if a == T::ZERO {
                -slope / (T::TWO * b)
            } else {
                let disc = b * b - T::from_f64(3.0) * a * slope;
                if disc < T::ZERO {
                    lambda * half
                } else {
                    (-b + Float::sqrt(disc)) / (T::from_f64(3.0) * a)
                }
            };
            if t > lambda * half || !Float::is_finite(t) {
                lambda * half
            } else {
                t
            }
        };
        previous = lambda;
        merit_previous = merit;
        let tenth = lambda * T::from_f64(0.1);
        lambda = if Float::is_finite(next) && next > tenth {
            next
        } else {
            tenth
        };
    }
    None
}

/// The certificate of the rounding test at `x`, where the residual is `fx`, the correction `d`
/// and the Jacobian (or the approximation of it that gave `d`) `jac`: whether `f` follows its
/// linear model along `d`, `f(x + k·d) = (1 − k)·f(x)`, at two successive steps `k` of
/// `4, 16, …, 4096`, in every equation to within [`LINEAR`] (a sixteenth) of the smaller of two
/// sizes — the change the model predicts for the residual, `(k − 1)·‖f(x)‖∞`, and the size of the
/// equation's own terms along the step, `(k − 1)·Σⱼ |Jᵢⱼ·dⱼ|`:
/// `|fᵢ(x + k·d) − (1 − k)·fᵢ(x)| ≤ (k − 1)·min(‖f(x)‖∞, Σⱼ |Jᵢⱼ·dⱼ|)/16`.
///
/// The rounding test is tried where the line search has found no point along `d` that lowers
/// `½‖f‖²` enough. To first order `f(x + λ·d) = (1 − λ)·f(x)` and the search succeeds at once, so
/// something else is as large as `f(x)` at `λ = 1`: either a deviation from the model that does
/// not grow with the step — the rounding of `f`, and of the points `x + λ·d` to floats — and then
/// `x` is a root to rounding, or the shape of `f` — a curvature, a local minimum of `‖f‖` that is
/// no root, the floats of an unknown too coarse for the steps the search would need — whose
/// deviation grows at least as the square of the step. Both are tried at longer steps: there a
/// rounding falls to a small part of the change the model predicts, and a curvature that was as
/// large as `f(x)` at `d` is `k²` times larger. Two successive steps that agree with the model
/// rule out a coincidence at one. Each of the two sizes alone can be fooled in a system: the
/// terms of an equation along `d` can cancel to far less than their size (a line that passes the
/// unit circle closely: the correction runs along the tangent, and in the circle's equation
/// `2u·δu + 2v·δv` cancels), and an equation can have a residual far below the largest one (beside
/// another at its floor of rounding). For one equation both are `|f(x)|`, and a step that agrees
/// is a proof of a root between `x` and `x + k·d`, where `f` has the sign opposite to `f(x)`.
fn follows_linear_model<T: Real + Scalar>(
    f: &mut impl FnMut(&Matrix<T>) -> Matrix<T>,
    x: &Matrix<T>,
    fx: &Matrix<T>,
    d: &Matrix<T>,
    jac: &Matrix<T>,
    evaluations: &mut usize,
) -> bool {
    let r = norm_inf(fx);
    if r == T::ZERO {
        return true;
    }
    let n = x.rows();
    // The size of the terms of the change of each equation along `d`, `Σⱼ |Jᵢⱼ·dⱼ|`.
    let terms: Vec<T> = (0..n)
        .map(|i| (0..n).fold(T::ZERO, |s, j| s + Float::abs(jac[(i, j)] * d[(j, 0)])))
        .collect();
    if !terms.iter().all(|&t| Float::is_finite(t)) {
        return false;
    }
    let (limit, four) = (T::from_f64(LINEAR), T::from_f64(4.0));
    let mut k = four;
    let mut held = false;
    for _ in 0..SCALES {
        let fz = evaluate(f, &along(x, d, k), evaluations);
        if !finite(&fz) {
            return false;
        }
        let holds = (0..n).all(|i| {
            let deviation = Float::abs(fz[(i, 0)] - (T::ONE - k) * fx[(i, 0)]);
            deviation <= limit * (k - T::ONE) * r.min(terms[i])
        });
        if holds && held {
            return true;
        }
        held = holds;
        k *= four;
    }
    false
}

/// What came of [`Run::settle`].
enum Settled {
    /// A root by the residual test: the run is at the point to return.
    Root,
    /// The iteration had stopped converging and no point along the correction lowered the
    /// residual: the rounding test, passed (`true`) or not. The run is at the point of the smaller
    /// residual of the two.
    Rounding(bool),
    /// Not yet: the run has moved along the correction and goes on from there.
    Moved,
    /// `f` was not finite at the end of the correction, which says nothing; the run is where it
    /// was, and goes on as if the test had not held.
    Declined,
}

/// The state of an iteration that the two methods share: the iterate, its residual, the best
/// point so far and the counts.
struct Run<T> {
    x: Matrix<T>,
    fx: Matrix<T>,
    best: (Matrix<T>, Matrix<T>, T),
    iterations: usize,
    evaluations: usize,
    jacobians: usize,
}

impl<T: Real + Scalar> Run<T> {
    /// Moves to the point `x` with the residual `fx`, and remembers it if it is the best so far.
    fn move_to(&mut self, x: Matrix<T>, fx: Matrix<T>) {
        let r = norm_inf(&fx);
        if r < self.best.2 {
            self.best = (x.clone(), fx.clone(), r);
        }
        self.x = x;
        self.fx = fx;
    }

    /// The result of a converged iteration, at the current point.
    fn converged(self, stop: SystemStop) -> SystemResult<T> {
        let residual = norm_inf(&self.fx);
        SystemResult {
            x: self.x,
            f: self.fx,
            residual,
            iterations: self.iterations,
            evaluations: self.evaluations,
            jacobians: self.jacobians,
            converged: true,
            stop,
        }
    }

    /// The result of an iteration that did not converge, at the best point it reached.
    fn failed(self, stop: SystemStop) -> SystemResult<T> {
        let (x, f, residual) = self.best;
        SystemResult {
            x,
            f,
            residual,
            iterations: self.iterations,
            evaluations: self.evaluations,
            jacobians: self.jacobians,
            converged: false,
            stop,
        }
    }

    /// The result of the rounding test at the current point, whose correction is `d`: converged
    /// where [`follows_linear_model`] holds, and otherwise a failure at the best point, with the
    /// same stop.
    fn rounding(
        mut self,
        f: &mut impl FnMut(&Matrix<T>) -> Matrix<T>,
        d: &Matrix<T>,
        jac: &Matrix<T>,
    ) -> SystemResult<T> {
        if follows_linear_model(f, &self.x, &self.fx, d, jac, &mut self.evaluations) {
            self.converged(SystemStop::Rounding)
        } else {
            self.failed(SystemStop::Rounding)
        }
    }

    /// Whether the residual test without a Jacobian holds: exactly zero, or within `ftol`.
    fn residual_small(&self, opts: &SystemOptions<T>) -> bool {
        let r = norm_inf(&self.fx);
        r == T::ZERO || r <= opts.ftol
    }

    /// Takes the full correction `delta` from a point that has passed a test, and keeps whichever
    /// of the two points has the smaller residual: at a regular root the step squares the error
    /// once more, and at the floor of the rounding it changes nothing that matters.
    fn polish(&mut self, f: &mut impl FnMut(&Matrix<T>) -> Matrix<T>, delta: &Matrix<T>) {
        let y = along(&self.x, delta, T::ONE);
        let fy = evaluate(f, &y, &mut self.evaluations);
        self.iterations += 1;
        if finite(&fy) && norm_inf(&fy) <= norm_inf(&self.fx) {
            self.move_to(y, fy);
        }
    }

    /// The residual test at a point whose residual is down to the size of its rounding (the
    /// backward test held there) and whose correction `delta` changes no unknown by more than
    /// `rounding` (`ε^(2/3)`) of itself. Neither says by itself that the point is a root: the
    /// size of the linear terms overstates the rounding of an equation whose terms cancel or
    /// whose coupling `f` evaluates exactly, as `0.01·(x₁ − 10¹⁶)` at `x₁ = 10¹⁶`, and a short
    /// correction is not a small error near a root that is nearly double, where Newton's method
    /// only halves the error. So the correction is taken, to `y = x + δ`, and the next one is
    /// solved for there on the same factors, `δ̄ = −J⁻¹·f(y)` (`correct`): the simplified Newton
    /// correction, the measure of the error left at `y` of Deuflhard's natural monotonicity test
    /// (P. Deuflhard, *Newton Methods for Nonlinear Problems*, 2004). The point is a
    /// root when `f(y)` is zero or within `ftol`, or when `δ̄` changes no unknown by more than
    /// `xtol` of itself. While `δ̄` is shorter than `δ`, relative to the unknowns, the iteration is
    /// still converging, and goes on from `y`.
    ///
    /// When `δ̄` is no shorter than `δ` (and below `rounding` too), the iteration has stopped
    /// converging. That happens at the floor of the rounding of `f`, but also where Newton's step
    /// overshoots and the backward bound was loose: a coupling that `f` evaluates exactly beside
    /// an unknown whose features are much finer than its size, as `10¹² + u` with `arctan u`,
    /// where from `u = 2` the step lands at `u = −3.5`. The line search along `δ` from `x` tells the
    /// two apart, as in the rounding test: where it finds a point, the iteration goes on from
    /// there; where it finds none, the point of the smaller residual is a root if `f` follows its
    /// linear model along that point's own correction ([`follows_linear_model`]), and the stop is
    /// `Rounding` either way. It does so without the line search too, since this
    /// correction is short against the unknowns and the search is the test, not a damping of the
    /// method: going on from `y` instead, at the floor, where the search succeeds by chance about
    /// as often as not, plain Newton wandered there to its last step.
    fn settle(
        &mut self,
        f: &mut impl FnMut(&Matrix<T>) -> Matrix<T>,
        delta: &Matrix<T>,
        correct: impl Fn(&Matrix<T>) -> Option<Matrix<T>>,
        jac: &Matrix<T>,
        opts: &SystemOptions<T>,
        rounding: T,
    ) -> Settled {
        let y = along(&self.x, delta, T::ONE);
        let fy = evaluate(f, &y, &mut self.evaluations);
        if !finite(&fy) {
            return Settled::Declined;
        }
        self.iterations += 1;
        let r = norm_inf(&fy);
        if r == T::ZERO || r <= opts.ftol {
            self.move_to(y, fy);
            return Settled::Root;
        }
        let Some(next) = correct(&fy) else {
            self.move_to(y, fy);
            return Settled::Moved;
        };
        if small_step(&next, &y, opts.xtol) {
            self.move_to(y, fy);
            return Settled::Root;
        }
        if relative(&next, &y) >= relative(delta, &self.x) && small_step(&next, &y, rounding) {
            return match line_search(
                f,
                &self.x,
                &self.fx,
                delta,
                opts.xtol,
                &mut self.evaluations,
            ) {
                None => {
                    // The rounding test at the point returned, along its own correction.
                    let d = if r <= norm_inf(&self.fx) {
                        self.move_to(y, fy);
                        next
                    } else {
                        delta.clone()
                    };
                    let (x, fx) = (&self.x, &self.fx);
                    let evaluations = &mut self.evaluations;
                    Settled::Rounding(follows_linear_model(f, x, fx, &d, jac, evaluations))
                }
                Some((z, fz, _)) => {
                    self.move_to(z, fz);
                    Settled::Moved
                }
            };
        }
        self.move_to(y, fy);
        Settled::Moved
    }
}

/// Checks the starting point and the options, and evaluates `f` there: the run, or `None` when
/// it cannot start. Panics if `x0` is not a column.
#[track_caller]
fn start<T: Real + Scalar>(
    x0: &Matrix<T>,
    f: &mut impl FnMut(&Matrix<T>) -> Matrix<T>,
    opts: &SystemOptions<T>,
) -> Option<Run<T>> {
    assert!(
        x0.cols() == 1,
        "the starting point is {}×{}, not a column",
        x0.rows(),
        x0.cols()
    );
    if !opts.valid() || !finite(x0) {
        return None;
    }
    let mut evaluations = 0;
    let fx = if x0.rows() == 0 {
        Matrix::zeros(0, 1)
    } else {
        evaluate(f, x0, &mut evaluations)
    };
    if !finite(&fx) {
        return None;
    }
    let r = norm_inf(&fx);
    Some(Run {
        x: x0.clone(),
        fx: fx.clone(),
        best: (x0.clone(), fx, r),
        iterations: 0,
        evaluations,
        jacobians: 0,
    })
}

/// The solution of `f(x) = 0` from the starting point `x0`, by Newton's method with the line
/// search, and the default [`SystemOptions`].
///
/// At every step the Jacobian `j(x)` is formed and the correction `δ` solving `j(x)·δ = −f(x)` is
/// found by [`Matrix::lu`]; the step along it is the full one where that lowers `½‖f‖²` enough,
/// and a shorter one from the line search otherwise. `f` and `j` take the iterate as a column and
/// return the residual as a column of as many rows and the Jacobian as a square of that order. The
/// tests for a root, and what is reported when there is none, are those of [`SystemStop`]: near a
/// regular root the residual squares at every step, and the method stops on the step test, or on
/// the residual test once the residual is down to rounding and one more step confirms the root.
///
/// Returns `None` when `x0` or `f(x0)` is not finite, or an option is invalid. An empty system is
/// solved at once.
///
/// # Panics
///
/// Panics if `x0` is not a column, or `f` or `j` returns a matrix of the wrong shape.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::newton_system;
///
/// // The intersection of the circle of radius 2 centred on the origin with the line y = x − 1
/// // that lies nearest the starting point, at ((1 + √7)/2, (√7 − 1)/2).
/// let f = |x: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)] - 4.0, x[(0, 0)] - x[(1, 0)] - 1.0]);
/// let j = |x: &Matrix<f64>| Matrix::from_rows(&[[2.0 * x[(0, 0)], 2.0 * x[(1, 0)]], [1.0, -1.0]]);
/// let r = newton_system(&Matrix::from_column_major(2, 1, vec![1.5, 0.5]), f, j).unwrap();
/// assert!(r.converged);
/// assert!((r.x[(0, 0)] - 1.822_875_655_532_295_4).abs() < 1e-15);
/// assert!((r.x[(1, 0)] - 0.822_875_655_532_295_2).abs() < 1e-15);
/// ```
#[track_caller]
pub fn newton<T>(
    x0: &Matrix<T>,
    f: impl FnMut(&Matrix<T>) -> Matrix<T>,
    j: impl FnMut(&Matrix<T>) -> Matrix<T>,
) -> Option<SystemResult<T>>
where
    T: Real + Scalar,
{
    newton_with(x0, f, j, &SystemOptions::new())
}

/// [`newton`] with the options `opts`.
///
/// # Panics
///
/// Panics if `x0` is not a column, or `f` or `j` returns a matrix of the wrong shape.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{SystemOptions, SystemStop, newton_system_with};
///
/// // A Jacobian that is singular at the start is reported, not divided by.
/// let f = |v: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![v[(0, 0)] * v[(0, 0)] - v[(1, 0)], v[(1, 0)] - 1.0]);
/// let j = |v: &Matrix<f64>| Matrix::from_rows(&[[2.0 * v[(0, 0)], -1.0], [0.0, 1.0]]);
/// let r = newton_system_with(&Matrix::from_column_major(2, 1, vec![0.0, 0.5]), f, j, &SystemOptions::new()).unwrap();
/// assert_eq!(r.stop, SystemStop::Singular);
/// assert!(!r.converged && r.iterations == 0);
/// ```
#[track_caller]
pub fn newton_with<T>(
    x0: &Matrix<T>,
    mut f: impl FnMut(&Matrix<T>) -> Matrix<T>,
    mut j: impl FnMut(&Matrix<T>) -> Matrix<T>,
    opts: &SystemOptions<T>,
) -> Option<SystemResult<T>>
where
    T: Real + Scalar,
{
    let mut run = start(x0, &mut f, opts)?;
    let n = x0.rows();
    let rounding = (T::EPSILON * T::EPSILON).cbrt();
    loop {
        if run.residual_small(opts) {
            return Some(run.converged(SystemStop::Residual));
        }
        if run.iterations >= opts.max_iter {
            return Some(run.failed(SystemStop::Iterations));
        }
        let jx = j(&run.x);
        run.jacobians += 1;
        assert!(
            jx.shape() == (n, n),
            "the Jacobian is {}×{}, not {n}×{n}",
            jx.rows(),
            jx.cols()
        );
        if !finite(&jx) {
            return Some(run.failed(SystemStop::Singular));
        }
        let lu = jx.lu();
        let correct = |r: &Matrix<T>| lu.solve(&-r).filter(finite);
        let Some(delta) = correct(&run.fx) else {
            return Some(run.failed(SystemStop::Singular));
        };
        if small_step(&delta, &run.x, opts.xtol) {
            run.polish(&mut f, &delta);
            return Some(run.converged(SystemStop::Step));
        }
        if backward(&run.fx, &jx, &run.x) && small_step(&delta, &run.x, rounding) {
            match run.settle(&mut f, &delta, correct, &jx, opts, rounding) {
                Settled::Root => return Some(run.converged(SystemStop::Residual)),
                Settled::Rounding(true) => return Some(run.converged(SystemStop::Rounding)),
                Settled::Rounding(false) => return Some(run.failed(SystemStop::Rounding)),
                Settled::Moved => continue,
                Settled::Declined => {}
            }
        }
        if opts.line_search {
            match line_search(
                &mut f,
                &run.x,
                &run.fx,
                &delta,
                opts.xtol,
                &mut run.evaluations,
            ) {
                Some((y, fy, _)) => run.move_to(y, fy),
                None if small_step(&delta, &run.x, rounding) => {
                    return Some(run.rounding(&mut f, &delta, &jx));
                }
                None => return Some(run.failed(SystemStop::LineSearch)),
            }
        } else {
            let y = along(&run.x, &delta, T::ONE);
            let fy = evaluate(&mut f, &y, &mut run.evaluations);
            if !finite(&fy) {
                run.iterations += 1;
                return Some(run.failed(SystemStop::NotFinite));
            }
            run.move_to(y, fy);
        }
        run.iterations += 1;
    }
}

/// The most times a restart lengthens the step of a column of differences fourfold when
/// quartering it found no two steps that agree: rounding, which grows as the step shrinks, may be
/// what keeps them apart.
const LENGTHENINGS: usize = 2;

/// The difference quotients of one column of a Jacobian, at one step.
#[derive(Clone)]
struct Column<T> {
    /// The quotient of each equation.
    values: Vec<T>,
    /// An estimate of the rounding of each: `8ε` of each value the quotient is formed from,
    /// over the step.
    noise: Vec<T>,
}

impl<T: Real + Scalar> Column<T> {
    /// Whether the column has an entry that is not zero.
    fn nonzero(&self) -> bool {
        self.values.iter().any(|&d| d != T::ZERO)
    }

    /// Whether every quotient agrees with that of `other` to [`AGREE`] of the larger of the two,
    /// beyond the rounding of both. Two zeros agree only in an equation marked false in `seen`,
    /// one whose quotient was zero at every step tried: a zero after a quotient that was not is
    /// a change of `f` below the spacing of its values, not a derivative. The allowance for
    /// rounding cannot make a correction look short: a quotient `|f′| + 8ε·|f|/s` at the step `s`
    /// gives a correction no shorter than about `s/(8ε)` where the rounding dominates, far above
    /// any tolerance, and one no shorter than half the true correction where it does not.
    fn agrees(&self, other: &Self, seen: &[bool]) -> bool {
        let agree = T::from_f64(AGREE);
        self.values
            .iter()
            .zip(&other.values)
            .zip(self.noise.iter().zip(&other.noise))
            .zip(seen)
            .all(|(((&p, &q), (&np, &nq)), &seen)| {
                if p == T::ZERO && q == T::ZERO {
                    !seen
                } else {
                    Float::abs(p - q) <= agree * Float::abs(p).max(Float::abs(q)) + np + nq
                }
            })
    }

    /// Marks in `seen` the equations whose quotient is not zero.
    fn mark(&self, seen: &mut [bool]) {
        for (s, &d) in seen.iter_mut().zip(&self.values) {
            *s |= d != T::ZERO;
        }
    }
}

/// Checks the central differences `column`, formed with the step `hk` in the unknown `k`, against
/// those at other steps: quartered first, up to [`QUARTERS`] times, until the quotients at two
/// steps agree; and if none do, lengthened fourfold from `hk`, up to [`LENGTHENINGS`] times. The
/// column at the longer of the two steps that agree, and the shorter step, or `None`.
fn verify<T: Real + Scalar>(
    f: &mut impl FnMut(&Matrix<T>) -> Matrix<T>,
    y: &mut Matrix<T>,
    fx: &Matrix<T>,
    k: usize,
    column: Column<T>,
    hk: T,
    evaluations: &mut usize,
) -> Option<(Column<T>, T)> {
    let mut seen = vec![false; column.values.len()];
    column.mark(&mut seen);
    let quarter = T::from_f64(0.25);
    let (mut longer, mut step) = (column.clone(), hk);
    for _ in 0..QUARTERS {
        let Some(shorter) = quotients(f, y, fx, k, step * quarter, true, evaluations) else {
            break;
        };
        step *= quarter;
        if longer.agrees(&shorter, &seen) {
            return Some((longer, step));
        }
        shorter.mark(&mut seen);
        longer = shorter;
    }
    let (mut shorter, mut step) = (column, hk);
    for _ in 0..LENGTHENINGS {
        let Some(longer) = quotients(f, y, fx, k, step / quarter, true, evaluations) else {
            break;
        };
        if longer.agrees(&shorter, &seen) {
            return Some((longer, step));
        }
        longer.mark(&mut seen);
        shorter = longer;
        step /= quarter;
    }
    None
}

/// The difference quotients in the unknown `k` at `x`, where the residual is `fx`, with the step
/// `hk` ahead and behind as the float makes it: central where `f` is finite on both sides and
/// `central` asks for it, one-sided otherwise. `y` is a copy of `x`, used and restored. `None`
/// when neither side gives a step and a finite value.
fn quotients<T: Real + Scalar>(
    f: &mut impl FnMut(&Matrix<T>) -> Matrix<T>,
    y: &mut Matrix<T>,
    fx: &Matrix<T>,
    k: usize,
    hk: T,
    central: bool,
    evaluations: &mut usize,
) -> Option<Column<T>> {
    let n = y.rows();
    let xk = y[(k, 0)];
    // The values and the steps as the float makes them, ahead and behind.
    let mut sides: [Option<(Matrix<T>, T)>; 2] = [None, None];
    for (side, sign) in [T::ONE, -T::ONE].into_iter().enumerate() {
        if side == 1 && !central && sides[0].is_some() {
            break;
        }
        y[(k, 0)] = xk + sign * hk;
        let step = y[(k, 0)] - xk;
        if step == T::ZERO || !Float::is_finite(step) {
            continue;
        }
        let fy = evaluate(f, y, evaluations);
        if finite(&fy) {
            sides[side] = Some((fy, step));
        }
    }
    y[(k, 0)] = xk;
    let limit = T::from_f64(BACKWARD) * T::EPSILON;
    let (values, noise) = match &sides {
        [Some((ahead, s)), Some((behind, t))] => (0..n)
            .map(|i| {
                let (a, b, d) = (ahead[(i, 0)], behind[(i, 0)], *s - *t);
                (
                    (a - b) / d,
                    limit * (Float::abs(a) + Float::abs(b)) / Float::abs(d),
                )
            })
            .unzip(),
        [Some((fy, s)), None] | [None, Some((fy, s))] => (0..n)
            .map(|i| {
                let (a, b) = (fy[(i, 0)], fx[(i, 0)]);
                (
                    (a - b) / *s,
                    limit * (Float::abs(a) + Float::abs(b)) / Float::abs(*s),
                )
            })
            .unzip(),
        [None, None] => return None,
    };
    Some(Column { values, noise })
}

/// The Jacobian by differences at `x`, where the residual is `fx`, and the step of each column:
/// column `k` from the step `h·|xₖ|` (or `h` where that is zero or subnormal), rounded to a step
/// the float can make exactly; forward differences (A5.4.1), or with `central` the difference of
/// the values on both sides, whose error is of the second order in the step rather than the
/// first. A column that comes out zero is tried again with steps 2²⁰ times longer, up to three
/// times, for an unknown whose step changes nothing at the size of the residual (as `x − 10¹⁶` at
/// `x = 0`). Where `f` is not finite on one side the difference is taken on the other.
///
/// Neither kind is right whatever `f`: a forward difference is wrong in its leading digit where
/// `f` curves within the step more than it rises, as `(x − 2)(1 + 10⁹(x − 2))` near its root
/// `2 − 10⁻⁹`, where the step of `√ε` adds 30 to a slope of −1, and a central one where the third
/// derivative dominates within the step — where the step spans the scale on which `f` varies, as
/// in `x = 10⁹ + u` with `e^u` (the step is 15, and the quotient `10⁵` for a slope of 1), or near
/// a root of multiplicity three or more (`(x − 1)³` from `1 + 10⁻¹¹`: `2·10⁻¹⁶` for `3·10⁻²²`).
/// So the central differences, on which [`broyden`]'s tests of a root rest, are checked by
/// [`verify`]: each column is formed again with a step a quarter as long, and the step is
/// quartered, up to [`QUARTERS`] times, until the quotients at two steps agree entry by entry to
/// [`AGREE`] of the larger, beyond the rounding of the values (`8ε` of each, over the step). Where
/// `f` is smooth the error of the second-order term then falls sixteenfold with each quarter, and
/// the longer of the two is kept, accurate to about a tenth; the step recorded is the shorter, the
/// distance over which the quotients were seen to hold. Where rounding rather than the third
/// derivative keeps them apart (it grows as the step shrinks: an equation whose terms of size one
/// cancel, so that the change over the step is a few of their roundings), steps four and sixteen
/// times longer are tried instead. `None` when a column cannot be formed from finite values, or,
/// with `central`, when no two steps agree.
fn differences<T: Real + Scalar>(
    f: &mut impl FnMut(&Matrix<T>) -> Matrix<T>,
    x: &Matrix<T>,
    fx: &Matrix<T>,
    h: T,
    central: bool,
    evaluations: &mut usize,
) -> Option<(Matrix<T>, Vec<T>)> {
    let n = x.rows();
    let mut jac = Matrix::zeros(n, n);
    let mut steps = vec![T::ZERO; n];
    let mut y = x.clone();
    let longer = T::TWO.powi(20);
    for k in 0..n {
        // An unknown so small that its relative step is not a normal float is taken for zero:
        // an iterate whose unknown tends to a zero of the root, as the helical valley's, reaches
        // the subnormals, where that step would vanish.
        let relative = h * Float::abs(x[(k, 0)]);
        let mut hk = if relative < T::MIN_POSITIVE {
            h
        } else {
            relative
        };
        let mut column = quotients(f, &mut y, fx, k, hk, central, evaluations)?;
        for _ in 0..3 {
            if column.nonzero() {
                break;
            }
            hk *= longer;
            column = quotients(f, &mut y, fx, k, hk, central, evaluations)?;
        }
        if central && column.nonzero() {
            (column, hk) = verify(f, &mut y, fx, k, column, hk, evaluations)?;
        }
        for (i, &d) in column.values.iter().enumerate() {
            jac[(i, k)] = d;
        }
        steps[k] = hk;
    }
    finite(&jac).then_some((jac, steps))
}

/// Whether the correction `delta` changes no unknown by more than its entry in `reach`, the
/// distance over which a restart saw its differences hold.
fn within<T: Real + Scalar>(delta: &Matrix<T>, reach: &[T]) -> bool {
    delta
        .as_slice()
        .iter()
        .zip(reach)
        .all(|(&d, &s)| Float::abs(d) <= s)
}

/// Broyden's approximation of the Jacobian, `B = Q·R`, with both factors kept so that a step
/// costs one triangular solve and an update two sweeps of plane rotations: `O(n²)`, not the
/// `O(n³)` of a new factorisation. `B` itself is kept too, for the residual test.
struct Approximation<T> {
    /// `B`.
    b: Matrix<T>,
    /// `Qᵀ`.
    qt: Matrix<T>,
    /// `R`, upper triangular.
    r: Matrix<T>,
}

impl<T: Real + Scalar> Approximation<T> {
    /// The factorisation of `b`.
    fn new(b: Matrix<T>) -> Self {
        let qr = b.qr();
        Approximation {
            qt: qr.q().transpose(),
            r: qr.r(),
            b,
        }
    }

    /// The solution of `B·δ = −f`, or `None` when `R` has a zero on its diagonal.
    fn step(&self, f: &Matrix<T>) -> Option<Matrix<T>> {
        let n = self.r.rows();
        let mut y = Matrix::zeros(n, 1);
        for i in 0..n {
            let mut s = T::ZERO;
            for k in 0..n {
                s -= self.qt[(i, k)] * f[(k, 0)];
            }
            y[(i, 0)] = s;
        }
        for i in (0..n).rev() {
            let d = self.r[(i, i)];
            if d == T::ZERO {
                return None;
            }
            let mut s = y[(i, 0)];
            for k in i + 1..n {
                s -= self.r[(i, k)] * y[(k, 0)];
            }
            y[(i, 0)] = s / d;
        }
        Some(y)
    }

    /// Applies the plane rotation `(c, s)` to rows `i` and `i + 1` of `R` and of `Qᵀ`.
    fn rotate(&mut self, i: usize, c: T, s: T) {
        let n = self.r.rows();
        for m in [&mut self.r, &mut self.qt] {
            for k in 0..n {
                let (a, b) = (m[(i, k)], m[(i + 1, k)]);
                m[(i, k)] = c * a + s * b;
                m[(i + 1, k)] = c * b - s * a;
            }
        }
    }

    /// The rotation that takes `(a, b)` to `(√(a² + b²), 0)`.
    fn givens(a: T, b: T) -> (T, T) {
        if b == T::ZERO {
            return (T::ONE, T::ZERO);
        }
        let h = Float::hypot(a, b);
        (a / h, b / h)
    }

    /// Broyden's update for the step `s` and the change `y` of the residual: `B ← B + (y − B·s)·sᵀ
    /// / (sᵀs)`, the least change of `B` that makes `B·s = y`. On the factors it is Dennis and
    /// Schnabel's algorithm A3.4.1: with `t = Qᵀ·u`, rotations from the bottom take `t` to a
    /// multiple of the first axis and `R` to upper Hessenberg, the rank-one term is added to the
    /// first row, and rotations from the top take `R` back to triangular.
    fn update(&mut self, s: &Matrix<T>, y: &Matrix<T>) {
        let n = s.rows();
        let ss = s.as_slice().iter().fold(T::ZERO, |acc, &v| acc + v * v);
        if ss == T::ZERO || !Float::is_finite(ss) {
            return;
        }
        let mut u = Matrix::zeros(n, 1);
        for i in 0..n {
            let mut bs = T::ZERO;
            for k in 0..n {
                bs += self.b[(i, k)] * s[(k, 0)];
            }
            u[(i, 0)] = (y[(i, 0)] - bs) / ss;
        }
        if !finite(&u) {
            return;
        }
        for i in 0..n {
            for k in 0..n {
                let v = self.b[(i, k)] + u[(i, 0)] * s[(k, 0)];
                self.b[(i, k)] = v;
            }
        }
        let mut t = vec![T::ZERO; n];
        for (i, ti) in t.iter_mut().enumerate() {
            let mut acc = T::ZERO;
            for k in 0..n {
                acc += self.qt[(i, k)] * u[(k, 0)];
            }
            *ti = acc;
        }
        for k in (1..n).rev() {
            let (c, sn) = Self::givens(t[k - 1], t[k]);
            self.rotate(k - 1, c, sn);
            t[k - 1] = c * t[k - 1] + sn * t[k];
            t[k] = T::ZERO;
        }
        for k in 0..n {
            let v = self.r[(0, k)] + t[0] * s[(k, 0)];
            self.r[(0, k)] = v;
        }
        for k in 0..n.saturating_sub(1) {
            let (c, sn) = Self::givens(self.r[(k, k)], self.r[(k + 1, k)]);
            self.rotate(k, c, sn);
            self.r[(k + 1, k)] = T::ZERO;
        }
    }
}

/// The solution of `f(x) = 0` from `x0`, by Broyden's method with the line search, and the
/// default [`SystemOptions`].
///
/// This is Broyden's first ("good") method (C. G. Broyden, *A class of methods for solving
/// nonlinear simultaneous equations*, Math. Comp. 19, 1965) in the form of Dennis and Schnabel
/// (algorithms A8.3.2 and A3.4.1): the Jacobian is approximated by forward differences once, at
/// `x0` (n evaluations of `f`), and after every step `s` that changed the residual by `y` it is
/// replaced by the nearest matrix that maps `s` to `y`, `B + (y − B·s)·sᵀ/(sᵀs)`, an update made on
/// its QR factorisation in `O(n²)`. The steps solve `B·δ = −f(x)` and go through the same line
/// search as [`newton`]. The approximation is formed afresh — a restart — when it is singular,
/// when the line search fails along its step, when two steps in a row needed the line search to
/// shorten them, and before any test of a root is believed, so that no answer rests on an
/// approximation alone, nor on a forward difference. A restart forms central differences, accurate
/// to the second order in the step, and checks them: each column against the quotients at a step
/// a quarter as long, the step quartered until two agree to a tenth entry by entry (or lengthened
/// fourfold, up to twice, where rounding keeps them apart), at least `4n` evaluations. That finds
/// the scale on which `f` varies where the step `√ε·|xₖ|` spans it — an unknown of `10⁹` whose
/// features are of size one — and near a root of multiplicity three or more, where the third
/// derivative dominates within the step; a restart whose differences cannot be checked ends the
/// iteration, as [`SystemStop::Singular`]. Near a regular root the convergence is superlinear.
/// Near a root at which the Jacobian is singular it is linear, and stops short of the root where
/// the condition of the approximation passes `1/ε`: its QR factors then lose the rows that vanish
/// at the root to the rounding of the others, and the run ends unconverged, at its best point, as
/// [`SystemStop::Singular`] where that rounding leaves a zero on the diagonal of `R`, or as
/// [`SystemStop::LineSearch`] or [`SystemStop::Iterations`] where it leaves a step of rounding.
/// Which one depends on the order of the sums of the dense kernels, so on the platform (Powell's
/// singular system: unknowns of about `10⁻¹⁷`, a residual of `3·10⁻³⁴` to `3·10⁻³²`, where
/// [`newton`], whose LU keeps those rows, goes on).
///
/// Returns `None` when `x0` or `f(x0)` is not finite, or an option is invalid.
///
/// # Panics
///
/// Panics if `x0` is not a column, or `f` returns a matrix of the wrong shape.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::broyden;
///
/// let f = |x: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![x[(0, 0)] * x[(0, 0)] + x[(1, 0)] * x[(1, 0)] - 4.0, x[(0, 0)] - x[(1, 0)] - 1.0]);
/// let r = broyden(&Matrix::from_column_major(2, 1, vec![1.5, 0.5]), f).unwrap();
/// assert!(r.converged);
/// assert!((r.x[(0, 0)] - 1.822_875_655_532_295_4).abs() < 1e-15);
/// ```
#[track_caller]
pub fn broyden<T>(x0: &Matrix<T>, f: impl FnMut(&Matrix<T>) -> Matrix<T>) -> Option<SystemResult<T>>
where
    T: Real + Scalar,
{
    broyden_with(x0, f, &SystemOptions::new())
}

/// [`broyden`] with the options `opts`.
///
/// # Panics
///
/// Panics if `x0` is not a column, or `f` returns a matrix of the wrong shape.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{SystemOptions, broyden_with};
///
/// // Coarser differences for a function too noisy for √ε.
/// let opts = SystemOptions::new().with_finite_difference_step(1e-4);
/// let f = |x: &Matrix<f64>| Matrix::from_column_major(1, 1, vec![x[(0, 0)].exp() - 2.0]);
/// let r = broyden_with(&Matrix::from_column_major(1, 1, vec![0.0]), f, &opts).unwrap();
/// assert!(r.converged && (r.x[(0, 0)] - core::f64::consts::LN_2).abs() < 1e-15);
/// ```
#[track_caller]
pub fn broyden_with<T>(
    x0: &Matrix<T>,
    mut f: impl FnMut(&Matrix<T>) -> Matrix<T>,
    opts: &SystemOptions<T>,
) -> Option<SystemResult<T>>
where
    T: Real + Scalar,
{
    let run = start(x0, &mut f, opts)?;
    broyden_from(run, f, None, opts)
}

/// [`broyden`] from the caller's first Jacobian `j0` rather than one by differences, with the
/// options `opts`; the restarts still form theirs by differences.
///
/// # Panics
///
/// Panics if `x0` is not a column, `j0` is not square of its order, or `f` returns a matrix of the
/// wrong shape.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{SystemOptions, broyden_with_jacobian};
///
/// // A linear system from its exact Jacobian: one step solves it, without the two evaluations of
/// // f that a first Jacobian by differences would cost.
/// let a = Matrix::from_rows(&[[4.0, 1.0], [1.0, 3.0]]);
/// let f = |x: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![4.0 * x[(0, 0)] + x[(1, 0)] - 1.0, x[(0, 0)] + 3.0 * x[(1, 0)] - 2.0]);
/// let x0 = Matrix::from_column_major(2, 1, vec![0.0, 0.0]);
/// let opts = SystemOptions::new();
/// let r = broyden_with_jacobian(&x0, f, &a, &opts).unwrap();
/// assert!(r.converged && r.iterations == 1);
/// assert!(r.evaluations + 2 <= a::math::numerics::broyden_with(&x0, f, &opts).unwrap().evaluations);
/// assert!((r.x[(0, 0)] - 1.0 / 11.0).abs() < 1e-15 && (r.x[(1, 0)] - 7.0 / 11.0).abs() < 1e-15);
/// ```
#[track_caller]
pub fn broyden_with_jacobian<T>(
    x0: &Matrix<T>,
    mut f: impl FnMut(&Matrix<T>) -> Matrix<T>,
    j0: &Matrix<T>,
    opts: &SystemOptions<T>,
) -> Option<SystemResult<T>>
where
    T: Real + Scalar,
{
    let n = x0.rows();
    assert!(
        j0.shape() == (n, n) || x0.cols() != 1,
        "the Jacobian is {}×{}, not {n}×{n}",
        j0.rows(),
        j0.cols()
    );
    let run = start(x0, &mut f, opts)?;
    broyden_from(run, f, Some(j0), opts)
}

/// The iteration of Broyden's method from the started `run`, with the first Jacobian `j0` or one
/// by differences.
fn broyden_from<T>(
    mut run: Run<T>,
    mut f: impl FnMut(&Matrix<T>) -> Matrix<T>,
    j0: Option<&Matrix<T>>,
    opts: &SystemOptions<T>,
) -> Option<SystemResult<T>>
where
    T: Real + Scalar,
{
    if run.x.rows() == 0 || run.residual_small(opts) {
        return Some(run.converged(SystemStop::Residual));
    }
    let rounding = (T::EPSILON * T::EPSILON).cbrt();
    let n = run.x.rows();
    let first = match j0 {
        Some(j) if finite(j) => Some(j.clone()),
        Some(_) => None,
        None => differences(&mut f, &run.x, &run.fx, opts.h, false, &mut run.evaluations)
            .map(|(b, _)| b),
    };
    run.jacobians += 1;
    let Some(first) = first else {
        return Some(run.failed(SystemStop::Singular));
    };
    let mut approx = Approximation::new(first);
    // Whether the approximation is a Jacobian that the tests of a root may rest on: the caller's,
    // or one by checked central differences, formed at the current point or within `reach` of it
    // (not an update, and not the first one by forward differences).
    let mut trusted = j0.is_some();
    // How much further, unknown by unknown, the confirming steps of the residual test may take
    // the iterate before the differences of the last restart are no longer known to hold there:
    // the steps over which they were seen to agree, less the moves made since. The caller's
    // Jacobian is taken to hold as Newton's method takes its own to.
    let mut reach = vec![T::INFINITY; n];
    // Steps in a row that the line search had to shorten.
    let mut shortened = 0;
    // Whether the last step halved the residual.
    let mut progress = true;
    loop {
        if run.residual_small(opts) {
            return Some(run.converged(SystemStop::Residual));
        }
        if run.iterations >= opts.max_iter {
            return Some(run.failed(SystemStop::Iterations));
        }
        // Forms the approximation afresh at the current point by checked central differences,
        // and goes round again; a point at which no Jacobian can be formed is the end.
        macro_rules! restart {
            () => {{
                run.jacobians += 1;
                match differences(&mut f, &run.x, &run.fx, opts.h, true, &mut run.evaluations) {
                    Some((b, held)) => {
                        approx = Approximation::new(b);
                        reach = held;
                        trusted = true;
                        shortened = 0;
                        continue;
                    }
                    None => return Some(run.failed(SystemStop::Singular)),
                }
            }};
        }
        let delta = match approx.step(&run.fx).filter(finite) {
            Some(d) => d,
            None if !trusted => restart!(),
            None => return Some(run.failed(SystemStop::Singular)),
        };
        if small_step(&delta, &run.x, opts.xtol) {
            if !trusted {
                restart!();
            }
            run.polish(&mut f, &delta);
            return Some(run.converged(SystemStop::Step));
        }
        // A residual down to rounding by the approximation is reason enough to form the Jacobian
        // afresh when the correction is short, to test the point, and when the last step did not
        // halve the residual: an approximation that has stopped improving the iterate is renewed
        // that way. While the steps still converge, as towards a root with unknowns at zero,
        // whose residual is at rounding against the other unknowns long before them, it goes on.
        if backward(&run.fx, &approx.b, &run.x) {
            let short = small_step(&delta, &run.x, rounding);
            if !trusted && (short || !progress) {
                restart!();
            }
            // The confirming step solves for its correction on this approximation at the end of
            // the step, which it may do only where the differences were seen to hold: a step
            // beyond them, along an unknown whose features are much finer than its size, goes on
            // as any other, and a restart there tests the next point.
            if short && trusted && within(&delta, &reach) {
                let correct = |r: &Matrix<T>| approx.step(r).filter(finite);
                match run.settle(&mut f, &delta, correct, &approx.b, opts, rounding) {
                    Settled::Root => return Some(run.converged(SystemStop::Residual)),
                    Settled::Rounding(true) => return Some(run.converged(SystemStop::Rounding)),
                    Settled::Rounding(false) => return Some(run.failed(SystemStop::Rounding)),
                    Settled::Moved => {
                        // A step this short is no secant to update the approximation by (its
                        // change of the residual is rounding), and the approximation is as good
                        // a Jacobian at the new point as one formed there, within what is left
                        // of its reach.
                        for (r, &d) in reach.iter_mut().zip(delta.as_slice()) {
                            *r -= Float::abs(d);
                        }
                        shortened = 0;
                        continue;
                    }
                    Settled::Declined => {}
                }
            }
        }
        let (y, fy, full) = if opts.line_search {
            match line_search(
                &mut f,
                &run.x,
                &run.fx,
                &delta,
                opts.xtol,
                &mut run.evaluations,
            ) {
                Some(found) => found,
                None if !trusted => restart!(),
                None if small_step(&delta, &run.x, rounding) => {
                    return Some(run.rounding(&mut f, &delta, &approx.b));
                }
                None => return Some(run.failed(SystemStop::LineSearch)),
            }
        } else {
            let y = along(&run.x, &delta, T::ONE);
            let fy = evaluate(&mut f, &y, &mut run.evaluations);
            if !finite(&fy) {
                if !trusted {
                    restart!();
                }
                run.iterations += 1;
                return Some(run.failed(SystemStop::NotFinite));
            }
            (y, fy, true)
        };
        let s = &y - &run.x;
        let dy = &fy - &run.fx;
        approx.update(&s, &dy);
        progress = norm_inf(&fy) <= T::HALF * norm_inf(&run.fx);
        run.move_to(y, fy);
        run.iterations += 1;
        trusted = false;
        shortened = if full { 0 } else { shortened + 1 };
        if shortened >= 2 {
            restart!();
        }
    }
}
