//! Ordinary differential equations: the initial value problem `y′ = f(t, y)`, `y(t₀) = y₀`.
//!
//! The state `y` is a column [`Matrix`], and the right-hand side writes its value into a matrix
//! the solver owns, so a caller integrates without allocating anything itself; the stages of a
//! step live in work arrays the method keeps between steps.
//!
//! Three methods are provided, each a type implementing [`OdeMethod`]:
//!
//! * [`Rk4`] — the classical explicit Runge–Kutta method of the fourth order, whose error is
//!   estimated by step doubling. Twelve right-hand side evaluations a step.
//! * [`DormandPrince5`] — Dormand and Prince's explicit 5(4) pair of order 5, judged by its
//!   embedded fourth order formula. Seven stages a step, the last of them at the end of it, as in
//!   [^3].
//! * [`RadauIia`] — the implicit Runge–Kutta method of the Radau IIA family of three stages, of
//!   order 5, L-stable, with Newton's method on the collocation system solved on
//!   [`Matrix::lu`], and its error judged, like [`Rk4`]'s, by step doubling.
//!
//! The explicit methods suit non-stiff problems; [`RadauIia`] suits those whose Jacobian has
//! eigenvalues of large negative real part, where an explicit method would be forced to a step far
//! below the accuracy it is being asked for.
//!
//! Every step is accepted or rejected by the usual controller `h′ = h · S · err^(−β)`, where `err`
//! is the local error of the step divided by the tolerance scale `atol + rtol · |y|`, `S` is a
//! safety factor of 0.9 and `β` is `1/(p + 1)` for a method of order `p`; the factor is clamped to
//! `[0.2, 5]`, so a step never grows more than five-fold nor shrinks more than five-fold. A step
//! whose stages are not all finite is rejected and the step size shrunk; after enough consecutive
//! failures the integration gives up and returns `None` rather than reporting a truncated answer.
//!
//! # Examples
//!
//! ```
//! use a::math::numerics::{DormandPrince5, RkOptions, integrate};
//! use a::math::matrix::Matrix;
//!
//! // The harmonic oscillator y′ = [y₁, −y₀] over a quarter of its period, from rest at one.
//! let f = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| {
//!     out[(0, 0)] = y[(1, 0)];
//!     out[(1, 0)] = -y[(0, 0)];
//! };
//! let y0 = Matrix::from_rows(&[[1.0], [0.0]]);
//! let r = integrate(f, &y0, 0.0, core::f64::consts::FRAC_PI_2, DormandPrince5::new(),
//!                   RkOptions::new(1e-10, 1e-12)).unwrap();
//! // y(π/2) = [cos(π/2), −sin(π/2)] = [0, −1] to the requested tolerance.
//! assert!((r.y[(0, 0)]).abs() < 1e-9);
//! assert!((r.y[(1, 0)] + 1.0).abs() < 1e-9);
//! ```
//!
//! [^3]: E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I*,
//!     Springer, 1993.

use super::{Real, ordered};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};

/// The largest factor a step size grows by, and its reciprocal the largest it shrinks by.
const MAX_FACTOR: f64 = 5.0;
const MIN_FACTOR: f64 = 1.0 / MAX_FACTOR;

/// The safety factor `S` of the step-size controller.
const SAFETY: f64 = 0.9;

/// How many times a step may fail in a row before the integration gives up.
const MAX_FAILURES: usize = 24;

/// The largest number of Newton iterations a stiff corrector runs before it gives up on a step.
const NEWTON_MAX_ITER: usize = 8;

/// How much of the tolerance the Newton iteration is content with: it converges to a twentieth of
/// what the step is judged by, so that the corrector is never what limits the accuracy.
const NEWTON_TOLERANCE: f64 = 0.05;

/// How far short of the round-off of the stage equations themselves the Newton iteration is still
/// content to stop, as a fraction of the size of the terms they subtract. A caller who asks for a
/// loose step tolerance is not owed an accurate solution of the step, but is owed the collocation
/// solution the method is, so the corrector converges to this as well however wide the tolerance.
const NEWTON_FLOOR: f64 = 1e-10;

// ---------------------------------------------------------------------------------------------
// Options and results
// ---------------------------------------------------------------------------------------------

/// The tolerances and step limits of an integration, for every method.
///
/// The local error of an accepted step is kept below `atol + rtol · |y|`, elementwise. `rtol`
/// controls the number of correct digits and `atol` the number of correct decimal places, so
/// `atol` should be set below the smallest value `rtol · |y|` is expected to reach, or `rtol` is
/// not achieved.
///
/// # Examples
///
/// ```
/// use a::math::numerics::RkOptions;
///
/// let opts = RkOptions::new(1e-9, 1e-12).with_hmax(0.01).with_max_steps(10_000);
/// assert_eq!(opts.rtol, 1e-9);
/// assert_eq!(opts.hmax, 0.01);
/// assert_eq!(opts.max_steps, 10_000);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RkOptions<T> {
    /// The relative tolerance.
    pub rtol: T,
    /// The absolute tolerance.
    pub atol: T,
    /// The first step size to try, or `T::ZERO` to let the solver choose one.
    pub h0: T,
    /// The largest step size to take, or `T::INFINITY` for no bound.
    pub hmax: T,
    /// The largest number of accepted steps before the integration gives up.
    pub max_steps: usize,
    /// The largest number of Newton iterations a stiff corrector runs before it rejects a step.
    pub newton_max_iter: usize,
    /// The spacing of the finite differences that approximate a Jacobian, as a fraction of the
    /// step size.
    pub jac_step: T,
}

impl<T: Float> RkOptions<T> {
    /// The options with relative tolerance `rtol`, absolute tolerance `atol`, an automatically
    /// chosen first step, no bound on the step size, and the default limits.
    ///
    /// # Panics
    ///
    /// If a tolerance is not positive and finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::RkOptions;
    ///
    /// let opts = RkOptions::new(1e-6, 1e-9);
    /// assert_eq!(opts.rtol, 1e-6);
    /// assert_eq!(opts.atol, 1e-9);
    /// assert_eq!(opts.h0, 0.0);
    /// assert_eq!(opts.max_steps, 100_000);
    /// ```
    #[must_use]
    pub fn new(rtol: T, atol: T) -> Self {
        assert!(rtol > T::ZERO && rtol.is_finite(), "rtol must be positive");
        assert!(atol > T::ZERO && atol.is_finite(), "atol must be positive");
        Self {
            rtol,
            atol,
            h0: T::ZERO,
            hmax: T::INFINITY,
            max_steps: 100_000,
            newton_max_iter: NEWTON_MAX_ITER,
            jac_step: T::from_f64(1e-8),
        }
    }

    /// `self`, with first step size `h0`. Pass `T::ZERO` to choose one automatically.
    #[must_use]
    pub fn with_h0(mut self, h0: T) -> Self {
        self.h0 = h0;
        self
    }

    /// `self`, with no step larger than `hmax`.
    #[must_use]
    pub fn with_hmax(mut self, hmax: T) -> Self {
        self.hmax = hmax;
        self
    }

    /// `self`, which gives up after `max_steps` accepted steps.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// `self`, whose stiff correctors run at most `newton_max_iter` Newton iterations.
    #[must_use]
    pub fn with_newton_max_iter(mut self, newton_max_iter: usize) -> Self {
        assert!(newton_max_iter > 0, "need at least one Newton iteration");
        self.newton_max_iter = newton_max_iter;
        self
    }

    /// `self`, whose finite-difference Jacobians perturb a component by `jac_step · |h|`.
    #[must_use]
    pub fn with_jac_step(mut self, jac_step: T) -> Self {
        assert!(
            jac_step > T::ZERO && jac_step.is_finite(),
            "jac_step must be positive"
        );
        self.jac_step = jac_step;
        self
    }
}

impl<T: Float> Default for RkOptions<T> {
    /// The options of [`RkOptions::new`] with relative tolerance `1e-6` and absolute tolerance
    /// `1e-9`.
    fn default() -> Self {
        Self::new(T::from_f64(1e-6), T::from_f64(1e-9))
    }
}

/// What an integration reached: the state and time of the last accepted step, and what it cost.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{Rk4, RkOptions, integrate};
/// use a::math::matrix::Matrix;
///
/// let f = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = -y[(0, 0)];
/// let r = integrate(f, &Matrix::from_rows(&[[1.0]]), 0.0, 1.0, Rk4::new(),
///                   RkOptions::new(1e-10, 1e-12)).unwrap();
/// assert_eq!(r.t, 1.0);
/// assert!(r.steps >= 1);
/// assert!(r.nfev >= 4 * r.steps);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OdeResult<T> {
    /// The state at `t`.
    pub y: Matrix<T>,
    /// The time reached, always `t1` when an integration succeeds.
    pub t: T,
    /// The number of accepted steps.
    pub steps: usize,
    /// The number of times the right-hand side was evaluated.
    pub nfev: usize,
}

// ---------------------------------------------------------------------------------------------
// The method trait
// ---------------------------------------------------------------------------------------------

/// What [`integrate`] needs from a method: the order the controller uses, the number of
/// right-hand side evaluations a step costs, and the step itself.
///
/// A step either succeeds, returning the error estimate of the step divided by the tolerance scale
/// — which must be at most 1 for the step to be accepted — or fails, returning `None`, in which
/// case the integrator shrinks the step and tries again.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{DormandPrince5, OdeMethod};
///
/// let m = DormandPrince5::<f64>::new();
/// assert_eq!(m.order(), 5);
/// assert_eq!(m.stages(), 7);
/// ```
pub trait OdeMethod<T> {
    /// The order of the method's main formula.
    fn order(&self) -> usize;

    /// The number of right-hand side evaluations a step costs, not counting the ones for a
    /// Jacobian.
    fn stages(&self) -> usize;

    /// Takes one step of size `h` from `(t, y)`, writing the new state to `y_new` and returning
    /// the local error of the step divided by `atol + rtol · max(|y|, |y_new|)`.
    ///
    /// `f` is called as `f(t, y, out)`, which must leave `out` holding `f(t, y)`. `scratch` is a
    /// matrix of the same shape as `y`, free for the method to use. `jac`, when given, is called
    /// as `jac(t, y, f_y, out)` and must leave `out` holding the `n × n` Jacobian of `f` with
    /// respect to `y` at `(t, y)`, where `f_y` is `f` evaluated at `(t, y)`; `opts` carries the
    /// tolerances to judge the step by, the spacing of the finite differences that stand in for a
    /// Jacobian, and how long a corrector may iterate.
    ///
    /// Returns `None` when the step cannot be taken — a singular corrector matrix, a non-finite
    /// stage, a corrector that will not converge — leaving the integrator to shrink `h`.
    #[allow(clippy::too_many_arguments)]
    fn step<F, J>(
        &mut self,
        f: &mut F,
        jac: Option<&mut J>,
        t: T,
        h: T,
        y: &Matrix<T>,
        y_new: &mut Matrix<T>,
        scratch: &mut Matrix<T>,
        opts: &RkOptions<T>,
    ) -> Option<T>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
        J: FnMut(T, &Matrix<T>, &Matrix<T>, &mut Matrix<T>);
}

// ---------------------------------------------------------------------------------------------
// Explicit Runge–Kutta methods
// ---------------------------------------------------------------------------------------------

/// The classical explicit Runge–Kutta method of the fourth order.
///
/// Four stages a step. It has no embedded formula, so the error a step is judged by is the
/// difference of the step and two half-steps — step doubling — which costs eight evaluations a
/// step and makes it the slowest of the explicit methods here; prefer [`DormandPrince5`] unless
/// the classical coefficients matter.
///
/// The value a step returns is the Richardson extrapolation of the two, and so is of the fifth
/// order wherever the classical method is of the fourth.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{Rk4, RkOptions, solve};
/// use a::math::matrix::Matrix;
///
/// // y′ = −2y, y(0) = 1, whose solution at t = 3 is e⁻⁶.
/// let f = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = -2.0 * y[(0, 0)];
/// let y = solve(f, &Matrix::from_rows(&[[1.0]]), 0.0, 3.0, Rk4::new(),
///               RkOptions::new(1e-10, 1e-12)).unwrap();
/// assert!((y[(0, 0)] - (-6.0f64).exp()).abs() < 1e-9);
/// ```
pub struct Rk4<T> {
    /// The four stages of the step in hand, and the perturbed state the stages are evaluated at.
    k: Vec<Matrix<T>>,
    p: Matrix<T>,
}

impl<T: Real + Scalar> Rk4<T> {
    /// The classical fourth order Runge–Kutta method.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::{OdeMethod, Rk4};
    ///
    /// let m = Rk4::<f64>::new();
    /// assert_eq!(m.order(), 4);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            k: Vec::new(),
            p: Matrix::zeros(0, 0),
        }
    }
}

/// Dormand and Prince's explicit Runge–Kutta pair of orders 5 and 4.
///
/// Seven stages a step, the last of them at the end of it, where the next step begins — an
/// implementation that carried that stage over would need six evaluations a step, and this one
/// does not. The step is judged by the embedded fourth order formula, the difference of the two
/// weight vectors of [^3].
///
/// [^3]: E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I*,
///     Springer, 1993.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{DormandPrince5, RkOptions, solve};
/// use a::math::matrix::Matrix;
///
/// // The harmonic oscillator y₀′ = y₁, y₁′ = −y₀ round a quarter of its period.
/// let f = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| {
///     out[(0, 0)] = y[(1, 0)];
///     out[(1, 0)] = -y[(0, 0)];
/// };
/// let y = solve(f, &Matrix::from_rows(&[[1.0], [0.0]]), 0.0, core::f64::consts::FRAC_PI_2,
///               DormandPrince5::new(), RkOptions::new(1e-11, 1e-13)).unwrap();
/// assert!(y[(0, 0)].abs() < 1e-10);
/// assert!((y[(1, 0)] + 1.0).abs() < 1e-10);
/// ```
pub struct DormandPrince5<T> {
    /// The seven stages of the step in hand, and the perturbed state they are evaluated at.
    k: Vec<Matrix<T>>,
    p: Matrix<T>,
}

impl<T: Real + Scalar> DormandPrince5<T> {
    /// Dormand and Prince's 5(4) pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::{DormandPrince5, OdeMethod};
    ///
    /// assert_eq!(DormandPrince5::<f64>::new().stages(), 7);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            k: Vec::new(),
            p: Matrix::zeros(0, 0),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The integration
// ---------------------------------------------------------------------------------------------

/// The Jacobian of `f` with respect to `y` at `(t, y)`, by forward differences of step
/// `jac_step · max(|y|, 1)` down each column, written to `out`. `fy` is `f` at `(t, y)`, which the
/// first column does not need again. Whether every entry came out finite.
fn difference_jacobian<T, F>(
    f: &mut F,
    t: T,
    y: &Matrix<T>,
    fy: &Matrix<T>,
    jac_step: T,
    out: &mut Matrix<T>,
) -> bool
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
{
    let n = y.rows();
    let mut perturbed = y.clone();
    let mut column = Matrix::<T>::zeros(n, 1);
    for j in 0..n {
        let yj = y[(j, 0)];
        let d = jac_step * magnitude(yj).max(T::ONE);
        perturbed[(j, 0)] = yj + d;
        f(t, &perturbed, &mut column);
        perturbed[(j, 0)] = yj;
        for i in 0..n {
            out[(i, j)] = (column[(i, 0)] - fy[(i, 0)]) / d;
        }
    }
    out.as_slice().iter().all(|v| finite(*v))
}

/// Whether `x` is finite, named through [`Float`], as [`Scalar`] names the same test and both are
/// in scope wherever a matrix element is a float.
fn finite<T: Real>(x: T) -> bool {
    Float::is_finite(x)
}

/// The magnitude of `x`, named through [`Float`] for the same reason.
fn magnitude<T: Real>(x: T) -> T {
    Float::abs(x)
}

/// The scale a local error is measured against: `atol + rtol · max(|y|, |y₁|)`, elementwise, as
/// every accepted or rejected step of this module is.
fn tolerance_scale<T: Real>(y: T, y_new: T, rtol: T, atol: T) -> T {
    atol + rtol * magnitude(y).max(magnitude(y_new))
}

/// The largest of the quotients `|d[i]| / (atol + rtol · max(|y[i]|, |y_new[i]|))`, which is the
/// number the controller compares with one.
fn error_norm<T: Real>(d: &Matrix<T>, y: &Matrix<T>, y_new: &Matrix<T>, rtol: T, atol: T) -> T {
    let mut err = T::ZERO;
    for i in 0..d.rows() {
        let scale = tolerance_scale(y[(i, 0)], y_new[(i, 0)], rtol, atol);
        err = err.max(Float::abs(d[(i, 0)]) / scale);
    }
    err
}

/// The first step of an integration whose `h0` was left for the solver: `1%` of the time the state
/// takes to change by its own size, which is where a method of the first order would first notice
/// the solution moving.
fn first_step<T, F>(f: &mut F, y0: &Matrix<T>, t0: T) -> T
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
{
    let n = y0.rows();
    let mut fy = Matrix::<T>::zeros(n, 1);
    f(t0, y0, &mut fy);
    let count = T::from_f64(n as f64);
    let mut d0 = T::ZERO;
    let mut d1 = T::ZERO;
    for i in 0..n {
        d0 += y0[(i, 0)] * y0[(i, 0)];
        d1 += fy[(i, 0)] * fy[(i, 0)];
    }
    let (d0, d1) = (Float::sqrt(d0 / count), Float::sqrt(d1 / count));
    let small = T::from_f64(1e-6);
    if d0 < small || d1 < small || !ordered(T::ZERO, d1) {
        small
    } else {
        T::from_f64(0.01) * d0 / d1
    }
}

/// The integration of `y′ = f(t, y)` from `(t₀, y₀)` to `t₁` by `method` under `opts`, reporting
/// what it reached.
///
/// `f` is called as `f(t, y, out)`, which must leave `out` holding `f(t, y)`; `out` is a matrix
/// the solver owns and may reuse for the next call. `t₁` may be before `t₀`, which integrates
/// backwards.
///
/// Returns `None` when a step cannot be taken, when the right-hand side is not finite, or when
/// `max_steps` accepted steps do not reach `t₁` — never a truncated answer.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{DormandPrince5, RkOptions, integrate};
/// use a::math::matrix::Matrix;
///
/// // The growth y′ = y from 1, whose value at 1 is e.
/// let f = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = y[(0, 0)];
/// let r = integrate(f, &Matrix::from_rows(&[[1.0]]), 0.0, 1.0, DormandPrince5::new(),
///                   RkOptions::new(1e-12, 1e-14)).unwrap();
/// assert!((r.y[(0, 0)] - core::f64::consts::E).abs() < 1e-11);
/// assert_eq!(r.t, 1.0);
/// ```
pub fn integrate<T, F, M>(
    f: F,
    y0: &Matrix<T>,
    t0: T,
    t1: T,
    method: M,
    opts: RkOptions<T>,
) -> Option<OdeResult<T>>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    integrate_with(
        f,
        y0,
        t0,
        t1,
        method,
        opts,
        None::<fn(T, &Matrix<T>, &Matrix<T>, &mut Matrix<T>)>,
    )
}

/// [`integrate`] with the Jacobian `jac` of `f` with respect to `y`, called as
/// `jac(t, y, fy, out)`, which must leave the `n × n` matrix `out` holding it. Explicit methods
/// ignore it; [`RadauIia`] needs it at every Newton iteration and falls back on finite differences
/// — of `opts.jac_step` — when it is `None`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{RadauIia, RkOptions, integrate_with};
/// use a::math::matrix::Matrix;
///
/// // The stiff decay y′ = −50(y − cos t), which an explicit method can only follow by a step
/// // far below what the tolerance asks for.
/// let f = |t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| {
///     out[(0, 0)] = 50.0 * (t.cos() - y[(0, 0)]);
/// };
/// let jac = |t: f64, _: &Matrix<f64>, _: &Matrix<f64>, out: &mut Matrix<f64>| {
///     let _ = t;
///     out[(0, 0)] = -50.0;
/// };
/// let r = integrate_with(f, &Matrix::from_rows(&[[0.0]]), 0.0, 2.0, RadauIia::new(),
///                        RkOptions::new(1e-8, 1e-10), Some(jac)).unwrap();
/// // The exponential has long died; what is left is the forced response, which lags the
/// // forcing by 1/50 of a radian.
/// let want = (2500.0 * 2.0f64.cos() + 50.0 * 2.0f64.sin()) / 2501.0;
/// assert!((r.y[(0, 0)] - want).abs() < 1e-7);
/// ```
pub fn integrate_with<T, F, J, M>(
    mut f: F,
    y0: &Matrix<T>,
    t0: T,
    t1: T,
    mut method: M,
    opts: RkOptions<T>,
    mut jac: Option<J>,
) -> Option<OdeResult<T>>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    J: FnMut(T, &Matrix<T>, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    let n = y0.rows();
    if n == 0 || y0.cols() != 1 || !finite(t0) || !finite(t1) {
        return None;
    }
    if !ordered(T::ZERO, opts.rtol) || !ordered(T::ZERO, opts.atol) {
        return None;
    }
    if t0 == t1 {
        return Some(OdeResult {
            y: y0.clone(),
            t: t0,
            steps: 0,
            nfev: 0,
        });
    }

    let direction = if t1 > t0 { T::ONE } else { -T::ONE };
    let mut y = y0.clone();
    let mut t = t0;
    let mut scratch = Matrix::<T>::zeros(n, 1);
    let mut y_new = Matrix::<T>::zeros(n, 1);
    let mut h = if opts.h0 > T::ZERO && finite(opts.h0) {
        opts.h0
    } else {
        first_step(&mut f, y0, t0)
    };
    let beta = T::from_f64(1.0 / (method.order() as f64 + 1.0));
    let safety = T::from_f64(SAFETY);
    let growth = T::from_f64(MAX_FACTOR);
    let shrink = T::from_f64(MIN_FACTOR);
    let mut steps = 0;
    let mut nfev = 0;
    let mut failures = 0;

    while t != t1 {
        if steps >= opts.max_steps {
            return None;
        }
        let remaining = magnitude(t1 - t);
        let size = h.min(opts.hmax).min(remaining);
        if !ordered(T::ZERO, size) || !finite(size) {
            return None;
        }
        // The step the method is given is signed, so that its stages lie between `t` and `t₁`
        // whether the integration runs forwards or backwards; the controller keeps the size.
        let step = direction * size;
        let estimate = method.step(
            &mut f,
            jac.as_mut(),
            t,
            step,
            &y,
            &mut y_new,
            &mut scratch,
            &opts,
        );
        match estimate {
            None => {
                failures += 1;
                if failures >= MAX_FAILURES {
                    return None;
                }
                h = (h * shrink).max(T::MIN_POSITIVE);
            }
            Some(err) => {
                nfev += method.stages();
                if !finite(err) || err > T::ONE {
                    failures += 1;
                    if failures >= MAX_FAILURES {
                        return None;
                    }
                    let factor = if finite(err) {
                        (safety * err.powf(T::ZERO - beta)).max(shrink)
                    } else {
                        shrink
                    };
                    h = (h * factor).max(T::MIN_POSITIVE);
                } else {
                    core::mem::swap(&mut y, &mut y_new);
                    t = if size == remaining { t1 } else { t + step };
                    steps += 1;
                    failures = 0;
                    let factor = if err > T::ZERO {
                        (safety * err.powf(T::ZERO - beta)).min(growth)
                    } else {
                        growth
                    };
                    h *= factor.max(shrink);
                }
            }
        }
    }

    Some(OdeResult {
        y,
        t: t1,
        steps,
        nfev,
    })
}

/// The state at `t₁` of the initial value problem [`integrate`] solves, or `None` when the
/// integration could not reach it.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{DormandPrince5, RkOptions, solve};
/// use a::math::matrix::Matrix;
///
/// // The harmonic oscillator, a quarter of its period from rest.
/// let f = |_t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| {
///     out[(0, 0)] = y[(1, 0)];
///     out[(1, 0)] = -y[(0, 0)];
/// };
/// let y = solve(f, &Matrix::from_rows(&[[1.0], [0.0]]), 0.0, core::f64::consts::FRAC_PI_2,
///               DormandPrince5::new(), RkOptions::new(1e-11, 1e-13)).unwrap();
/// assert!(y[(0, 0)].abs() < 1e-10);
/// ```
pub fn solve<T, F, M>(
    f: F,
    y0: &Matrix<T>,
    t0: T,
    t1: T,
    method: M,
    opts: RkOptions<T>,
) -> Option<Matrix<T>>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    integrate(f, y0, t0, t1, method, opts).map(|r| r.y)
}

/// [`solve`] with the Jacobian `jac`, as [`integrate_with`] has it.
pub fn solve_with<T, F, J, M>(
    f: F,
    y0: &Matrix<T>,
    t0: T,
    t1: T,
    method: M,
    opts: RkOptions<T>,
    jac: Option<J>,
) -> Option<Matrix<T>>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    J: FnMut(T, &Matrix<T>, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    integrate_with(f, y0, t0, t1, method, opts, jac).map(|r| r.y)
}

// ---------------------------------------------------------------------------------------------
// The methods
// ---------------------------------------------------------------------------------------------

impl<T: Real + Scalar> Default for Rk4<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real + Scalar> Rk4<T> {
    /// One classical step of size `h` from `y`, left in `out`: `k₁` at the start, `k₂` and `k₃` at
    /// the midpoint and `k₄` at the end, combined with the weights `[1, 4, 4, 1]/6` of Simpson's
    /// rule — which is what makes the method fourth order. Whether the result is finite.
    fn advance<F>(&mut self, f: &mut F, t: T, h: T, y: &Matrix<T>, out: &mut Matrix<T>) -> bool
    where
        T: Real + Scalar,
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let n = y.rows();
        let half = h * T::HALF;
        f(t, y, &mut self.k[0]);
        for i in 0..n {
            self.p[(i, 0)] = y[(i, 0)] + half * self.k[0][(i, 0)];
        }
        f(t + half, &self.p, &mut self.k[1]);
        for i in 0..n {
            self.p[(i, 0)] = y[(i, 0)] + half * self.k[1][(i, 0)];
        }
        f(t + half, &self.p, &mut self.k[2]);
        for i in 0..n {
            self.p[(i, 0)] = y[(i, 0)] + h * self.k[2][(i, 0)];
        }
        f(t + h, &self.p, &mut self.k[3]);
        let sixth = T::from_f64(1.0 / 6.0);
        for i in 0..n {
            out[(i, 0)] = y[(i, 0)]
                + h * sixth
                    * (self.k[0][(i, 0)]
                        + self.k[3][(i, 0)]
                        + T::TWO * (self.k[1][(i, 0)] + self.k[2][(i, 0)]));
        }
        out.as_slice().iter().all(|v| finite(*v))
    }
}

impl<T: Real + Scalar> OdeMethod<T> for Rk4<T> {
    fn order(&self) -> usize {
        4
    }

    fn stages(&self) -> usize {
        12
    }

    fn step<F, J>(
        &mut self,
        f: &mut F,
        jac: Option<&mut J>,
        t: T,
        h: T,
        y: &Matrix<T>,
        y_new: &mut Matrix<T>,
        scratch: &mut Matrix<T>,
        opts: &RkOptions<T>,
    ) -> Option<T>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
        J: FnMut(T, &Matrix<T>, &Matrix<T>, &mut Matrix<T>),
    {
        let _ = (jac, scratch);
        let rtol = opts.rtol;
        let atol = opts.atol;
        let n = y.rows();
        if self.k.len() != 4 || self.k[0].rows() != n {
            self.k = (0..4).map(|_| Matrix::zeros(n, 1)).collect();
            self.p = Matrix::zeros(n, 1);
        }
        let mut full = Matrix::<T>::zeros(n, 1);
        let mut half = Matrix::<T>::zeros(n, 1);
        if !self.advance(f, t, h, y, &mut full) {
            return None;
        }
        if !self.advance(f, t, h * T::HALF, y, &mut half) {
            return None;
        }
        if !self.advance(f, t + h * T::HALF, h * T::HALF, &half, y_new) {
            return None;
        }
        // The two half-steps are of the fourth order too, but their error is a fifteenth of the
        // full step's and of the opposite sign, so the Richardson extrapolation of the two is of
        // the fifth and their difference is the error of either.
        let fifteenth = T::from_f64(1.0 / 15.0);
        let mut d = Matrix::<T>::zeros(n, 1);
        for i in 0..n {
            let correction = (y_new[(i, 0)] - full[(i, 0)]) * fifteenth;
            y_new[(i, 0)] += correction;
            d[(i, 0)] = Float::abs(correction);
        }
        if !y_new.as_slice().iter().all(|v| finite(*v)) {
            return None;
        }
        Some(error_norm(&d, y, y_new, rtol, atol))
    }
}

/// The abscissae of Dormand and Prince's 5(4) pair.
const DP5_C: [f64; 7] = [0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0];

/// The rows of the Dormand and Prince tableau, the last of them the fifth order weights, so that
/// the seventh stage is `f` at the end of the step.
const DP5_A: [&[f64]; 7] = [
    &[],
    &[1.0 / 5.0],
    &[3.0 / 40.0, 9.0 / 40.0],
    &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
    &[
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    ],
    &[
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    ],
    &[
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];

/// The fifth order weights of the Dormand and Prince pair.
const DP5_B: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];

/// Its fourth order weights, whose difference from [`DP5_B`] estimates the error of a step.
const DP5_B_HAT: [f64; 7] = [
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];

impl<T: Real + Scalar> Default for DormandPrince5<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real + Scalar> DormandPrince5<T> {
    /// The `i`th stage, the abscissa `c[i]` of the pair and the rows of `A` below it.
    fn stage<F>(&mut self, f: &mut F, t: T, h: T, y: &Matrix<T>, i: usize) -> bool
    where
        T: Real + Scalar,
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let row = DP5_A[i];
        let n = y.rows();
        for r in 0..n {
            let mut sum = y[(r, 0)];
            for (j, &a) in row.iter().enumerate() {
                if a != 0.0 {
                    sum += T::from_f64(a) * h * self.k[j][(r, 0)];
                }
            }
            self.p[(r, 0)] = sum;
        }
        f(t + T::from_f64(DP5_C[i]) * h, &self.p, &mut self.k[i]);
        self.k[i].as_slice().iter().all(|v| finite(*v))
    }
}

impl<T: Real + Scalar> OdeMethod<T> for DormandPrince5<T> {
    fn order(&self) -> usize {
        5
    }

    fn stages(&self) -> usize {
        7
    }

    fn step<F, J>(
        &mut self,
        f: &mut F,
        jac: Option<&mut J>,
        t: T,
        h: T,
        y: &Matrix<T>,
        y_new: &mut Matrix<T>,
        scratch: &mut Matrix<T>,
        opts: &RkOptions<T>,
    ) -> Option<T>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
        J: FnMut(T, &Matrix<T>, &Matrix<T>, &mut Matrix<T>),
    {
        let _ = (jac, scratch);
        let rtol = opts.rtol;
        let atol = opts.atol;
        let n = y.rows();
        if self.k.len() != 7 || self.k[0].rows() != n {
            self.k = (0..7).map(|_| Matrix::zeros(n, 1)).collect();
            self.p = Matrix::zeros(n, 1);
        }
        for i in 0..7 {
            if !self.stage(f, t, h, y, i) {
                return None;
            }
        }
        let mut d = Matrix::<T>::zeros(n, 1);
        for r in 0..n {
            let mut value = y[(r, 0)];
            let mut error = T::ZERO;
            for j in 0..7 {
                let w = T::from_f64(DP5_B[j] - DP5_B_HAT[j]);
                if w != T::ZERO {
                    error += w * h * self.k[j][(r, 0)];
                }
                value += T::from_f64(DP5_B[j]) * h * self.k[j][(r, 0)];
            }
            y_new[(r, 0)] = value;
            d[(r, 0)] = Float::abs(error);
        }
        if !y_new.as_slice().iter().all(|v| finite(*v)) {
            return None;
        }
        Some(error_norm(&d, y, y_new, rtol, atol))
    }
}

/// √6, of which the two interior abscissae below are rational functions.
const SQRT_6: f64 = 2.449_489_742_783_178;

/// The abscissae of the three-stage Radau IIA method: the two roots of `5x² − 8x + 2` in `(0, 1)`
/// and the end of the step, which is what makes the method stiffly accurate.
const RADAU_C: [f64; 3] = [(4.0 - SQRT_6) / 10.0, (4.0 + SQRT_6) / 10.0, 1.0];

/// The coefficients of the three-stage Radau IIA method, `a[i][j] = ∫₀^{cᵢ} ℓⱼ` and
/// `b[j] = ∫₀¹ ℓⱼ`, where `ℓⱼ` is the Lagrange basis polynomial at [`RADAU_C`] — the collocation
/// conditions themselves, rather than the closed forms they simplify to, so that the rows sum to
/// the abscissae and the weights to one by construction.
fn radau_coefficients() -> ([[f64; 3]; 3], [f64; 3]) {
    let integral = |j: usize, u: f64| {
        let (p, q) = match j {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let sum = RADAU_C[p] + RADAU_C[q];
        let product = RADAU_C[p] * RADAU_C[q];
        let denominator = (RADAU_C[j] - RADAU_C[p]) * (RADAU_C[j] - RADAU_C[q]);
        (u * u * u / 3.0 - sum * u * u / 2.0 + product * u) / denominator
    };
    let mut a = [[0.0; 3]; 3];
    for (i, &ci) in RADAU_C.iter().enumerate() {
        for (j, cell) in a[i].iter_mut().enumerate() {
            *cell = integral(j, ci);
        }
    }
    // The last abscissa is the end of the step, so the last row is already the quadrature formula:
    // the method is stiffly accurate, and its value is one of its own stages.
    let b = a[2];
    (a, b)
}

/// The implicit Runge–Kutta method of the Radau IIA family of three stages: order 5, L-stable, and
/// the stiff method of this module.
///
/// A step is the collocation polynomial whose three stages satisfy
///
/// ```text
/// Yᵢ = y + h · Σⱼ a[i][j] · f(t + c[j]·h, Yⱼ),     y₁ = y + h · Σⱼ b[j] · f(t + c[j]·h, Yⱼ)
/// ```
///
/// which is solved by Newton's method on the `3n` equations `Kᵢ = h·f(t + cᵢh, Yᵢ)` for the stages
/// `Kᵢ`, each iteration correcting them by the solution of `(I − h·A ⊗ J)·δ = −G` on
/// [`Matrix::lu`], `J` being the Jacobian of `f` at the end of the step. The stages of the last
/// solve start the next iteration, so a step of unchanging size usually needs two or three of them;
/// a corrector that will not settle fails the step, and the integrator shrinks `h` and tries
/// again.
///
/// The method has no embedded formula of a lower order, and the estimator Hairer and Wanner build
/// from the interior stages measures the stages rather than the solution — dependable only where
/// the problem is stiff — so a step is judged as in [`Rk4`], by step doubling. The difference of
/// one step and two half-steps is the local error of either, and the value a step returns is their
/// Richardson extrapolation, which is of the sixth order. Nine stage evaluations a step, on top of
/// the ones a Jacobian costs.
///
/// L-stability is what it is for: a component decaying at rate `λ` with `h·|λ|` past the reach of
/// an explicit method is damped by a factor tending to nought as `h·λ` grows, rather than by one
/// tending to a limit of the stability polynomial.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{RadauIia, RkOptions, solve};
/// use a::math::matrix::Matrix;
///
/// // The stiff decay to a slowly moving equilibrium. An explicit method needs a step of a
/// // few thousandths here for stability alone; this takes no more than forty of them.
/// let f = |t: f64, y: &Matrix<f64>, out: &mut Matrix<f64>| {
///     out[(0, 0)] = 1000.0 * (t.sin() - y[(0, 0)]);
/// };
/// let y = solve(f, &Matrix::from_rows(&[[0.0]]), 0.0, 5.0, RadauIia::new(),
///               RkOptions::new(1e-8, 1e-10)).unwrap();
/// // The forced response of y′ + 1000y = 1000 sin t, the transient e⁻⁵⁰⁰⁰ having vanished.
/// let want = (1e6 * 5.0f64.sin() - 1000.0 * 5.0f64.cos()) / (1e6 + 1.0);
/// assert!((y[(0, 0)] - want).abs() < 1e-7);
/// ```
#[derive(Debug, Clone)]
pub struct RadauIia<T> {
    /// The stages `Kᵢ = h·f(Yᵢ)` of the last solve, which start the Newton iteration of the next.
    k: Vec<Matrix<T>>,
    /// The collocation points `Yᵢ` and the values of `f` there.
    y: Vec<Matrix<T>>,
    fy: Vec<Matrix<T>>,
    /// The state the stages are evaluated at, and the Jacobian of `f` at the end of the step.
    p: Matrix<T>,
    j: Matrix<T>,
    /// The residuals of the `3n` collocation equations, and the `3n × 3n` matrix of each Newton
    /// iteration.
    g: Matrix<T>,
    w: Matrix<T>,
    /// The collocation coefficients, computed once from the abscissae.
    a: [[f64; 3]; 3],
    b: [f64; 3],
}

impl<T: Real + Scalar> RadauIia<T> {
    /// The three-stage Radau IIA method.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::{OdeMethod, RadauIia};
    ///
    /// assert_eq!(RadauIia::<f64>::new().order(), 5);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            k: Vec::new(),
            y: Vec::new(),
            fy: Vec::new(),
            p: Matrix::zeros(0, 0),
            j: Matrix::zeros(0, 0),
            g: Matrix::zeros(0, 0),
            w: Matrix::zeros(0, 0),
            a: [[0.0; 3]; 3],
            b: [0.0; 3],
        }
    }

    /// The stages and their derivatives, sized for a state of `n` equations.
    fn prepare(&mut self, n: usize) {
        if self.b[2] == 0.0 {
            let (a, b) = radau_coefficients();
            self.a = a;
            self.b = b;
        }
        if self.k.len() != 3 || self.k[0].rows() != n {
            self.k = (0..3).map(|_| Matrix::zeros(n, 1)).collect();
            self.y = (0..3).map(|_| Matrix::zeros(n, 1)).collect();
            self.fy = (0..3).map(|_| Matrix::zeros(n, 1)).collect();
            self.p = Matrix::zeros(n, 1);
            self.j = Matrix::zeros(n, n);
            self.g = Matrix::zeros(3 * n, 1);
            self.w = Matrix::zeros(3 * n, 3 * n);
        }
    }

    /// The collocation points and the values of `f` there, for the stages in hand, then the
    /// residual `Gᵢ = Kᵢ − h·f(Yᵢ)` of the collocation equations, left in [`Self::g`] as `3n`
    /// rows. Whether every value of `f` came out finite.
    fn residual<F>(&mut self, f: &mut F, t: T, h: T, y: &Matrix<T>) -> bool
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let n = y.rows();
        for (i, &c) in RADAU_C.iter().enumerate() {
            for r in 0..n {
                let mut sum = y[(r, 0)];
                for (j, row) in self.a[i].iter().enumerate() {
                    sum += T::from_f64(*row) * self.k[j][(r, 0)];
                }
                self.p[(r, 0)] = sum;
            }
            for r in 0..n {
                self.y[i][(r, 0)] = self.p[(r, 0)];
            }
            f(t + T::from_f64(c) * h, &self.p, &mut self.fy[i]);
            if !self.fy[i].as_slice().iter().all(|v| finite(*v)) {
                return false;
            }
            for r in 0..n {
                self.g[(i * n + r, 0)] = self.k[i][(r, 0)] - h * self.fy[i][(r, 0)];
            }
        }
        true
    }

    /// The collocation solution of size `h` from `y`, left in `out`: Newton's method on the `3n`
    /// stage equations, started from the stages of the last solve. Whether it converged, which is
    /// what makes the value in `out` trustworthy.
    #[allow(clippy::too_many_arguments)]
    fn advance<F, J>(
        &mut self,
        f: &mut F,
        mut jac: Option<&mut J>,
        t: T,
        h: T,
        y: &Matrix<T>,
        out: &mut Matrix<T>,
        opts: &RkOptions<T>,
    ) -> bool
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
        J: FnMut(T, &Matrix<T>, &Matrix<T>, &mut Matrix<T>),
    {
        let n = y.rows();
        let mut settled = false;
        for _ in 0..opts.newton_max_iter {
            if !self.residual(f, t, h, y) {
                return false;
            }
            // How far the collocation equations are from being satisfied, against the tighter of
            // the tolerance the step is judged by and the round-off of the equations themselves.
            let mut err = T::ZERO;
            for i in 0..3 {
                for r in 0..n {
                    let scale = tolerance_scale(y[(r, 0)], self.y[i][(r, 0)], opts.rtol, opts.atol);
                    let terms = magnitude(y[(r, 0)]) + magnitude(h) * magnitude(self.fy[i][(r, 0)]);
                    let want = (T::from_f64(NEWTON_TOLERANCE) * scale)
                        .min(T::from_f64(NEWTON_FLOOR) * terms.max(T::ONE));
                    err = err.max(magnitude(self.g[(i * n + r, 0)]) / want);
                }
            }
            if err <= T::ONE {
                settled = true;
                break;
            }
            // The Jacobian at the end of the step, which every iteration of the corrector shares.
            let last = &self.y[2];
            f(t + h, last, &mut self.p);
            let formed = match jac.as_mut() {
                Some(j) => {
                    j(t + h, last, &self.p, &mut self.j);
                    self.j.as_slice().iter().all(|v| finite(*v))
                }
                None => difference_jacobian(f, t + h, last, &self.p, opts.jac_step, &mut self.j),
            };
            if !formed {
                return false;
            }
            // `I − h·(A ⊗ J)`, the `n × n` block `(i, j)` of which is `δ[i][j]·I − h·a[i][j]·J`.
            for i in 0..3 {
                for j in 0..3 {
                    for r in 0..n {
                        for c in 0..n {
                            // The unknowns are the stages `K = h·f(Y)` and `∂Yᵢ/∂Kⱼ = a[i][j]`,
                            // so each off-diagonal block is `−h·a[i][j]·J`.
                            let mut entry = -T::from_f64(self.a[i][j]) * h * self.j[(r, c)];
                            if i == j && r == c {
                                entry += T::ONE;
                            }
                            self.w[(i * n + r, j * n + c)] = entry;
                        }
                    }
                }
            }
            for v in self.g.as_mut_slice() {
                *v = -*v;
            }
            let correction = match self.w.lu().solve(&self.g) {
                Some(correction) => correction,
                None => return false,
            };
            for i in 0..3 {
                for r in 0..n {
                    self.k[i][(r, 0)] += correction[(i * n + r, 0)];
                }
            }
        }
        if !settled {
            return false;
        }
        // The value at the end of the step, which for a stiffly accurate method is the last stage.
        for r in 0..n {
            let mut value = y[(r, 0)];
            for j in 0..3 {
                value += T::from_f64(self.b[j]) * h * self.fy[j][(r, 0)];
            }
            out[(r, 0)] = value;
        }
        out.as_slice().iter().all(|v| finite(*v))
    }
}

impl<T: Real + Scalar> Default for RadauIia<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real + Scalar> OdeMethod<T> for RadauIia<T> {
    fn order(&self) -> usize {
        5
    }

    fn stages(&self) -> usize {
        9
    }

    fn step<F, J>(
        &mut self,
        f: &mut F,
        mut jac: Option<&mut J>,
        t: T,
        h: T,
        y: &Matrix<T>,
        y_new: &mut Matrix<T>,
        scratch: &mut Matrix<T>,
        opts: &RkOptions<T>,
    ) -> Option<T>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
        J: FnMut(T, &Matrix<T>, &Matrix<T>, &mut Matrix<T>),
    {
        let _ = scratch;
        let n = y.rows();
        self.prepare(n);
        let half = h * T::HALF;
        let mut full = Matrix::<T>::zeros(n, 1);
        let mut mid = Matrix::<T>::zeros(n, 1);
        if !self.advance(f, jac.as_deref_mut(), t, h, y, &mut full, opts) {
            return None;
        }
        if !self.advance(f, jac.as_deref_mut(), t, half, y, &mut mid, opts) {
            return None;
        }
        if !self.advance(f, jac, t + half, half, &mid, y_new, opts) {
            return None;
        }
        // Two half-steps carry a sixty-fourth of the error of a whole one, so their difference is
        // the local error of either; a thirty-first of it corrects the two half-steps to the sixth
        // order.
        let sixty_thirds = T::from_f64(1.0 / 31.0);
        let mut d = Matrix::<T>::zeros(n, 1);
        for r in 0..n {
            let correction = (y_new[(r, 0)] - full[(r, 0)]) * sixty_thirds;
            y_new[(r, 0)] += correction;
            d[(r, 0)] = magnitude(correction);
        }
        if !y_new.as_slice().iter().all(|v| finite(*v)) {
            return None;
        }
        Some(error_norm(&d, y, y_new, opts.rtol, opts.atol))
    }
}
