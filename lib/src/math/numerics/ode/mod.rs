//! Ordinary differential equations: the initial value problem `y′ = f(t, y)`, `y(t₀) = y₀`.
//!
//! The state `y` is a column [`Matrix`], and the right-hand side is a closure `f(t, y, dy)` that
//! writes `f(t, y)` into a matrix the solver owns, so a system of any size is written the same way
//! as a scalar equation and a caller allocates nothing itself.
//!
//! # The methods
//!
//! Each is a type implementing [`OdeMethod`], which owns the state of an integration, takes one
//! accepted step at a time under its own error control, and gives the continuous extension of
//! that step:
//!
//! | Method | Order | Error estimate | Dense output | For |
//! | --- | --- | --- | --- | --- |
//! | [`Rk4`] | 4 | step doubling, or none with [`Rk4::fixed`] | Hermite cubic | the classical method itself |
//! | [`DormandPrince5`] | 5 | embedded formula of order 4 | order 4 | non-stiff problems at moderate tolerances |
//! | [`DormandPrince853`] | 8 | embedded formulas of orders 5 and 3 | order 7 | non-stiff problems at stringent tolerances |
//! | [`RadauIia`] | 5 | embedded formula of order 3, filtered | collocation polynomial | stiff problems, stringent tolerances |
//! | [`Bdf`] | 1 to 5 | the next backward difference | interpolating polynomial | stiff problems, large systems |
//!
//! The explicit methods suit non-stiff problems. A stiff one — whose Jacobian has eigenvalues of
//! large negative real part — holds an explicit method to a step far below what the accuracy asks
//! for, and is what the implicit [`RadauIia`] and [`Bdf`] are for: each step solves its implicit
//! equations by a Newton iteration on LU decompositions ([`Matrix::lu`]) of matrices built from
//! the Jacobian `∂f/∂y`, which is the caller's (`with_jacobian`) or formed by finite
//! differences.
//!
//! # The entry points
//!
//! * [`solve`] returns the state at `t₁`, and [`integrate`] the state with what it cost.
//! * [`solve_at`] returns the states at a list of times, sampled from the dense output without
//!   keeping the steps.
//! * [`solve_ivp`] returns an [`OdeSolution`]: the solution as a function of time
//!   ([`DenseOutput`]), the [`Event`]s located along it, and how the integration ended.
//!
//! Each has a `_with` form that takes an [`OdeOptions`].
//!
//! # Step-size control
//!
//! A step is accepted when the method's estimate of its local error, `err`, satisfies
//!
//! ```text
//! √( (1/n) · Σᵢ ( errᵢ / (atol + rtol · max(|yᵢ|, |y₁ᵢ|)) )² )  ≤  1
//! ```
//!
//! — the root mean square norm of Hairer's codes, with `y` and `y₁` the states at the two ends of
//! the step — and is otherwise rejected and retried with a smaller one. The size of the next step
//! is the method's own business: see each method. What they share:
//!
//! * `h0`, the first step, is chosen by Hairer's `hinit` unless the caller gives it; no step is
//!   larger than `hmax`.
//! * No step but the last is shorter than the floor — `hmin`, and never less than ten units in
//!   the last place of `t` — to the rounding of the time (for [`Rk4::new`], no pair of its
//!   half-steps). A shorter proposal is raised to it, as LSODE raises one to its `hmin`. A
//!   rejected step is retried shorter, at `hmin` itself before anything below it, and when the
//!   retry would be below the floor the integration fails with [`OdeFailure::StepTooSmall`]
//!   rather than loop: this is how a solution that blows up in finite time ends, and a problem
//!   too stiff for an explicit method.
//! * The last step, what is left of the interval, is taken whatever its size, so an interval
//!   shorter than the floor is tried as one step; a method that rejects it fails with
//!   [`OdeFailure::StepTooSmall`], since nothing shorter may be tried. [`Rk4::new`] needs two
//!   units in the last place of `t` for the midpoint of its pair.
//! * A right-hand side that is not finite at a stage rejects the step, as an error that is too
//!   large does; the integration goes on with a smaller one if it can.
//! * Tolerances below the rounding error of the state fail with
//!   [`OdeFailure::ToleranceTooSmall`] instead of reporting an accuracy that was not reached.
//! * The last step ends on `t₁` exactly. `t₁ < t₀` integrates backwards in time, and `t₁ = t₀`
//!   returns the initial state.
//!
//! A failed integration is never passed off as the solution at `t₁`: [`solve`], [`integrate`] and
//! [`solve_at`] return `None`, and [`solve_ivp`] an [`OdeStatus::Failed`] with the point it
//! reached.
//!
//! # Examples
//!
//! ```
//! use a::math::matrix::Matrix;
//! use a::math::numerics::{DormandPrince5, OdeOptions, solve_with};
//!
//! // The harmonic oscillator y′ = [y₁, −y₀] over a quarter of its period, from rest at one.
//! let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
//!     dy[(0, 0)] = y[(1, 0)];
//!     dy[(1, 0)] = -y[(0, 0)];
//! };
//! let y0 = Matrix::from_rows(&[[1.0], [0.0]]);
//! let opts = OdeOptions::new().with_tolerances(1e-10, 1e-12);
//! let y = solve_with(f, &y0, 0.0, core::f64::consts::FRAC_PI_2, DormandPrince5::new(), &opts)
//!     .unwrap();
//! // y(π/2) = [cos(π/2), −sin(π/2)] = [0, −1].
//! assert!(y[(0, 0)].abs() < 1e-9);
//! assert!((y[(1, 0)] + 1.0).abs() < 1e-9);
//! ```

mod bdf;
mod control;
mod dense;
mod dop853;
mod dopri5;
mod events;
mod radau;
mod rk4;
mod stiff;
mod tableau;

pub use bdf::Bdf;
pub use dense::{DenseOutput, DenseStep};
pub use dop853::DormandPrince853;
pub use dopri5::DormandPrince5;
pub use events::{Event, EventDirection, EventPoint};
pub use radau::RadauIia;
pub use rk4::Rk4;
pub use stiff::NoJacobian;
pub use tableau::ButcherTableau;

use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};
use crate::math::numerics::Real;

// ---------------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------------

/// The tolerances and the limits of an integration, for every method.
///
/// The local error of an accepted step is kept below `atol + rtol · |y|` in each component, in
/// the root mean square over the components. `rtol` sets the number of correct digits and `atol`
/// the size below which a component no longer matters; either may be zero, but not both, and
/// `atol` zero leaves a component that passes through zero with nothing to be measured against
/// — and the Newton iteration of a stiff method cannot converge where it is exactly zero at the
/// start of the step ([`RadauIia`]) or in the prediction ([`Bdf`]).
///
/// | Field | Default | |
/// | --- | --- | --- |
/// | `rtol` | `1e-6` | the relative tolerance |
/// | `atol` | `1e-9` | the absolute tolerance |
/// | `h0` | `0` | the size of the first step; `0` lets the method choose |
/// | `hmax` | `∞` | the largest size of a step, to the rounding of the time |
/// | `hmin` | `0` | the smallest size of a step but the last, to the rounding of the time: a shorter proposal is raised to it, and a rejected step of that size ends the integration |
/// | `max_steps` | `100 000` | the number of accepted steps after which the integration fails |
/// | `dense_output` | `true` | whether [`solve_ivp`] keeps the steps for [`DenseOutput`] |
///
/// The defaults are those of `f64`. For `f32` a relative tolerance of `1e-6` is eight units in
/// the last place: ask for `1e-4` or so.
///
/// # Examples
///
/// ```
/// use a::math::numerics::OdeOptions;
///
/// let opts = OdeOptions::<f64>::new()
///     .with_tolerances(1e-9, 1e-12)
///     .with_hmax(0.01)
///     .with_max_steps(10_000);
/// assert_eq!((opts.rtol, opts.atol), (1e-9, 1e-12));
/// assert_eq!(opts.hmax, 0.01);
/// assert_eq!(opts.max_steps, 10_000);
/// assert_eq!(opts.h0, 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdeOptions<T> {
    /// The relative tolerance.
    pub rtol: T,
    /// The absolute tolerance.
    pub atol: T,
    /// The size of the first step to try, or zero to let the method choose one.
    pub h0: T,
    /// The largest size of a step, or infinity for no bound but the interval.
    pub hmax: T,
    /// The smallest size of a step, or zero for no bound but the resolution of the time.
    pub hmin: T,
    /// The largest number of accepted steps before the integration gives up.
    pub max_steps: usize,
    /// Whether [`solve_ivp`] keeps the continuous extension of every step.
    pub dense_output: bool,
}

impl<T: Real> Default for OdeOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> OdeOptions<T> {
    /// The defaults: `rtol` `1e-6`, `atol` `1e-9`, the first step chosen by the method, no bound
    /// on the step, `max_steps` 100 000, and the dense output kept.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::OdeOptions;
    ///
    /// let opts = OdeOptions::<f64>::new();
    /// assert_eq!((opts.rtol, opts.atol), (1e-6, 1e-9));
    /// assert_eq!(opts.hmax, f64::INFINITY);
    /// assert_eq!(opts.max_steps, 100_000);
    /// assert!(opts.dense_output);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        OdeOptions {
            rtol: T::from_f64(1e-6),
            atol: T::from_f64(1e-9),
            h0: T::ZERO,
            hmax: T::INFINITY,
            hmin: T::ZERO,
            max_steps: 100_000,
            dense_output: true,
        }
    }

    /// Sets the relative tolerance.
    #[must_use]
    pub fn with_rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    /// Sets the absolute tolerance.
    #[must_use]
    pub fn with_atol(mut self, atol: T) -> Self {
        self.atol = atol;
        self
    }

    /// Sets the relative and the absolute tolerance.
    #[must_use]
    pub fn with_tolerances(mut self, rtol: T, atol: T) -> Self {
        self.rtol = rtol;
        self.atol = atol;
        self
    }

    /// Sets the size of the first step; zero lets the method choose it.
    #[must_use]
    pub fn with_h0(mut self, h0: T) -> Self {
        self.h0 = h0;
        self
    }

    /// Sets the largest size of a step.
    #[must_use]
    pub fn with_hmax(mut self, hmax: T) -> Self {
        self.hmax = hmax;
        self
    }

    /// Sets the smallest size of a step. A shorter proposal of the method's controller is raised
    /// to it, and a step of that size that is rejected ends the integration with
    /// [`OdeFailure::StepTooSmall`], since no shorter one may be tried. The last step, what is
    /// left of the interval, may be shorter.
    #[must_use]
    pub fn with_hmin(mut self, hmin: T) -> Self {
        self.hmin = hmin;
        self
    }

    /// Sets the number of accepted steps after which the integration gives up.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Sets whether [`solve_ivp`] keeps the continuous extension of every step. Without it the
    /// events are still located, and the solution's [`dense`](OdeSolution::dense) is `None`.
    #[must_use]
    pub fn with_dense_output(mut self, dense_output: bool) -> Self {
        self.dense_output = dense_output;
        self
    }

    /// Whether the options are ones an integration can run under: tolerances that are finite,
    /// not negative and not both zero, and step limits that are numbers and do not contradict
    /// each other. A NaN fails every one of the comparisons.
    fn valid(&self) -> bool {
        let tolerances = self.rtol >= T::ZERO
            && self.atol >= T::ZERO
            && self.rtol.is_finite()
            && self.atol.is_finite()
            && (self.rtol > T::ZERO || self.atol > T::ZERO);
        let limits = self.h0 >= T::ZERO
            && self.h0.is_finite()
            && self.hmax > T::ZERO
            && self.hmin >= T::ZERO
            && self.hmin.is_finite()
            && self.hmin <= self.hmax
            && (self.h0 == T::ZERO || self.h0 >= self.hmin);
        tolerances && limits
    }
}

// ---------------------------------------------------------------------------------------------
// How an integration ends
// ---------------------------------------------------------------------------------------------

/// Why an integration stopped short of its end.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince5, OdeFailure, OdeStatus, solve_ivp};
///
/// // y′ = y² from 1 is 1/(1 − t), which has left the real line by t = 2.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = y[(0, 0)] * y[(0, 0)];
/// let y0 = Matrix::from_rows(&[[1.0]]);
/// let solution = solve_ivp(f, &y0, 0.0, 2.0, DormandPrince5::new(), &mut []);
/// assert_eq!(solution.status, OdeStatus::Failed(OdeFailure::StepTooSmall));
/// // What it reached is where its solution ends — the pole, moved by the error of the
/// // integration — and not the answer at t = 2.
/// assert!((solution.t - 1.0).abs() < 1e-5 && solution.y[(0, 0)] > 1e10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OdeFailure {
    /// The problem is not one an integration can start on: a time or a component of the initial
    /// state that is not finite, tolerances that are negative, not finite or both zero, step
    /// limits that are not numbers or contradict each other, output times out of order, or a
    /// method that was not started.
    InvalidInput,
    /// The right-hand side, or the state, is not finite where no smaller step could avoid it: at
    /// the initial point, in a method of fixed steps, or in the extra stages of a dense output.
    NonFinite,
    /// After a rejection the method needs a step below the smallest one allowed — `hmin`, which
    /// is tried itself first, or ten units in the last place of the time: a solution leaving the
    /// reals, a discontinuity the tolerances cannot be met across, a problem too stiff for the
    /// method, or an interval shorter than that floor whose one step the method rejected.
    StepTooSmall,
    /// The tolerances ask for less than the rounding error of the state.
    ToleranceTooSmall,
    /// `max_steps` accepted steps did not reach the end.
    MaxSteps,
    /// An event function returned NaN.
    EventNotFinite,
}

/// How an integration ended.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince5, OdeStatus, solve_ivp};
///
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
/// let y0 = Matrix::from_rows(&[[1.0]]);
/// let solution = solve_ivp(f, &y0, 0.0, 1.0, DormandPrince5::new(), &mut []);
/// assert_eq!(solution.status, OdeStatus::Finished);
/// assert!(solution.status.is_success());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OdeStatus {
    /// The integration reached `t₁`.
    Finished,
    /// A terminal event stopped the integration before `t₁`.
    Terminated,
    /// The integration failed; what it returns is the last point it reached, not the solution at
    /// `t₁`.
    Failed(OdeFailure),
}

impl OdeStatus {
    /// Whether the integration did what was asked: reached `t₁`, or stopped at a terminal event.
    pub fn is_success(self) -> bool {
        !matches!(self, OdeStatus::Failed(_))
    }
}

/// What [`integrate`] returns: the state at `t₁` and what the integration cost.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince5, integrate};
///
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
/// let r = integrate(f, &Matrix::from_rows(&[[1.0]]), 0.0, 1.0, DormandPrince5::new()).unwrap();
/// assert_eq!(r.t, 1.0);
/// assert!((r.y[(0, 0)] - (-1.0f64).exp()).abs() < 1e-6);
/// // Two evaluations to start, and six for every step tried.
/// assert_eq!(r.nfev, 2 + 6 * (r.steps + r.rejected));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OdeResult<T> {
    /// The state at `t`.
    pub y: Matrix<T>,
    /// The time reached, which is `t₁`.
    pub t: T,
    /// The number of accepted steps.
    pub steps: usize,
    /// The number of steps that were tried and rejected.
    pub rejected: usize,
    /// The number of evaluations of the right-hand side, every one of them counted.
    pub nfev: usize,
    /// The number of Jacobians formed; zero for an explicit method.
    pub njev: usize,
    /// The number of LU decompositions; zero for an explicit method.
    pub nlu: usize,
}

/// What [`solve_ivp`] returns: where the integration ended and why, the events along the way,
/// the solution as a function of time, and what it cost.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince853, OdeStatus, solve_ivp};
///
/// // Lotka and Volterra's predators and prey.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = 1.5 * y[(0, 0)] - y[(0, 0)] * y[(1, 0)];
///     dy[(1, 0)] = -3.0 * y[(1, 0)] + y[(0, 0)] * y[(1, 0)];
/// };
/// let y0 = Matrix::from_rows(&[[10.0], [5.0]]);
/// let solution = solve_ivp(f, &y0, 0.0, 15.0, DormandPrince853::new(), &mut []);
/// assert_eq!(solution.status, OdeStatus::Finished);
/// assert_eq!(solution.t, 15.0);
/// // The solution anywhere in between, and its end.
/// let y = solution.eval(7.5).unwrap();
/// assert!(y[(0, 0)] > 0.0 && y[(1, 0)] > 0.0);
/// assert_eq!(solution.eval(15.0).unwrap(), solution.y);
/// // The invariant x − 3 ln x + y − 1.5 ln y is kept to the tolerance.
/// let invariant = |y: &Matrix<f64>| {
///     y[(0, 0)] - 3.0 * y[(0, 0)].ln() + y[(1, 0)] - 1.5 * y[(1, 0)].ln()
/// };
/// assert!((invariant(&y) - invariant(&y0)).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OdeSolution<T> {
    /// How the integration ended.
    pub status: OdeStatus,
    /// The time the integration ended at: `t₁`, the time of the terminal event, or the last
    /// accepted step before a failure.
    pub t: T,
    /// The state at `t`.
    pub y: Matrix<T>,
    /// The events that occurred, in the order of their times.
    pub events: Vec<EventPoint<T>>,
    /// The solution from `t₀` to `t`, when [`OdeOptions::dense_output`] asked for it.
    pub dense: Option<DenseOutput<T>>,
    /// The number of accepted steps.
    pub steps: usize,
    /// The number of steps that were tried and rejected.
    pub rejected: usize,
    /// The number of evaluations of the right-hand side, every one of them counted.
    pub nfev: usize,
    /// The number of Jacobians formed; zero for an explicit method.
    pub njev: usize,
    /// The number of LU decompositions; zero for an explicit method.
    pub nlu: usize,
}

impl<T: Real> OdeSolution<T> {
    /// The solution at the time `t`, from the dense output: `None` when `t` is outside the
    /// interval that was integrated, or when the dense output was not kept.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{DormandPrince5, OdeOptions, solve_ivp_with};
    ///
    /// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
    /// let y0 = Matrix::from_rows(&[[1.0]]);
    /// let kept = solve_ivp_with(f, &y0, 0.0, 2.0, DormandPrince5::new(), &mut [],
    ///                           &OdeOptions::new());
    /// assert!((kept.eval(0.7).unwrap()[(0, 0)] - (-0.7f64).exp()).abs() < 1e-6);
    /// assert!(kept.eval(2.1).is_none());
    ///
    /// let dropped = solve_ivp_with(f, &y0, 0.0, 2.0, DormandPrince5::new(), &mut [],
    ///                              &OdeOptions::new().with_dense_output(false));
    /// assert!(dropped.eval(0.7).is_none());
    /// ```
    #[must_use]
    pub fn eval(&self, t: T) -> Option<Matrix<T>> {
        let dense = self.dense.as_ref()?;
        // The end of the integration is a state the method computed, not one to interpolate.
        if t == self.t {
            return Some(self.y.clone());
        }
        let mut y = self.y.clone();
        dense.eval_into(t, y.as_mut_slice()).then_some(y)
    }
}

// ---------------------------------------------------------------------------------------------
// The method trait
// ---------------------------------------------------------------------------------------------

/// A method for the initial value problem, as the integration drives it: one accepted step at a
/// time.
///
/// A method owns the state of an integration — the time, the solution, the size of the next
/// step, its work arrays and whatever history it needs — so that a one-step method and a
/// multistep one look alike from outside. [`start`](Self::start) places it at the initial point;
/// each call of [`step`](Self::step) then advances it by one step that it has itself accepted,
/// rejecting and retrying inside as its error estimate demands and choosing the size of the next
/// one; and [`dense`](Self::dense) gives the continuous extension over the step just taken. The
/// step-size control belongs to the method, because each has its own estimate and its own
/// controller.
///
/// The right-hand side is passed to every call rather than kept, as `f(t, y, dy)`, which must
/// leave `dy` holding `f(t, y)`.
///
/// The entry points of the module ([`solve`], [`solve_ivp`], …) are all one loop over this
/// trait, and a method written outside the crate can be handed to them.
///
/// # Examples
///
/// Driving a method by hand:
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DenseStep, DormandPrince5, OdeMethod, OdeOptions};
///
/// let mut f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
/// let mut method = DormandPrince5::new();
/// assert_eq!(method.order(), 5);
/// method.start(&mut f, 0.0, &Matrix::from_rows(&[[1.0]]), 1.0, &OdeOptions::new()).unwrap();
/// let mut step = DenseStep::new();
/// while method.t() != 1.0 {
///     method.step(&mut f).unwrap();
///     // The continuous extension follows e⁻ᵗ between the ends of the step.
///     method.dense(&mut f, &mut step).unwrap();
///     let middle = 0.5 * (method.t_old() + method.t());
///     assert!((step.eval(middle)[(0, 0)] - (-middle).exp()).abs() < 1e-6);
/// }
/// assert!((method.y()[(0, 0)] - (-1.0f64).exp()).abs() < 1e-6);
/// ```
///
/// A method of one's own — Euler's, with `n` equal steps and a straight line between them:
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DenseStep, OdeFailure, OdeMethod, OdeOptions, solve};
///
/// struct Euler {
///     n: usize,
///     taken: usize,
///     ends: (f64, f64),
///     t: f64,
///     t_old: f64,
///     y: Matrix<f64>,
///     y_old: Matrix<f64>,
///     dy: Matrix<f64>,
/// }
///
/// impl OdeMethod<f64> for Euler {
///     fn order(&self) -> usize {
///         1
///     }
///     fn start<F>(&mut self, _f: &mut F, t0: f64, y0: &Matrix<f64>, t1: f64,
///                 _opts: &OdeOptions<f64>) -> Result<(), OdeFailure>
///     where
///         F: FnMut(f64, &Matrix<f64>, &mut Matrix<f64>),
///     {
///         (self.ends, self.t, self.t_old, self.taken) = ((t0, t1), t0, t0, 0);
///         (self.y, self.y_old, self.dy) = (y0.clone(), y0.clone(), y0.clone());
///         Ok(())
///     }
///     fn step<F>(&mut self, f: &mut F) -> Result<(), OdeFailure>
///     where
///         F: FnMut(f64, &Matrix<f64>, &mut Matrix<f64>),
///     {
///         f(self.t, &self.y, &mut self.dy);
///         self.taken += 1;
///         // The last step lands on t₁ itself.
///         let (t0, t1) = self.ends;
///         let t = if self.taken == self.n {
///             t1
///         } else {
///             t0 + (t1 - t0) * self.taken as f64 / self.n as f64
///         };
///         let h = t - self.t;
///         self.y_old.as_mut_slice().copy_from_slice(self.y.as_slice());
///         for (y, dy) in self.y.as_mut_slice().iter_mut().zip(self.dy.as_slice()) {
///             *y += h * dy;
///         }
///         (self.t_old, self.t) = (self.t, t);
///         Ok(())
///     }
///     fn t(&self) -> f64 {
///         self.t
///     }
///     fn t_old(&self) -> f64 {
///         self.t_old
///     }
///     fn y(&self) -> &Matrix<f64> {
///         &self.y
///     }
///     fn dense<F>(&mut self, _f: &mut F, out: &mut DenseStep<f64>) -> Result<(), OdeFailure>
///     where
///         F: FnMut(f64, &Matrix<f64>, &mut Matrix<f64>),
///     {
///         // y_old + θ·(y − y_old): the Newton form on the single node 0.
///         out.reset(self.t_old, self.t, self.y.as_slice().len(), &[0.0]);
///         out.coefficient_mut(0).copy_from_slice(self.y_old.as_slice());
///         let difference = out.coefficient_mut(1).iter_mut();
///         for ((c, y), y_old) in difference.zip(self.y.as_slice()).zip(self.y_old.as_slice()) {
///             *c = y - y_old;
///         }
///         Ok(())
///     }
/// }
///
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
/// let empty = Matrix::zeros(0, 1);
/// let euler = Euler {
///     n: 1000,
///     taken: 0,
///     ends: (0.0, 0.0),
///     t: 0.0,
///     t_old: 0.0,
///     y: empty.clone(),
///     y_old: empty.clone(),
///     dy: empty,
/// };
/// let y = solve(f, &Matrix::from_rows(&[[1.0]]), 0.0, 1.0, euler).unwrap();
/// // (1 − 1/1000)¹⁰⁰⁰, which is e⁻¹ to three digits.
/// assert!((y[(0, 0)] - 0.999f64.powi(1000)).abs() < 1e-12);
/// assert!((y[(0, 0)] - (-1.0f64).exp()).abs() < 2e-4);
/// ```
pub trait OdeMethod<T: Real + Scalar> {
    /// The order of the solution the method propagates; for a method of variable order, the
    /// order of its last step.
    fn order(&self) -> usize;

    /// Places the method at `(t0, y0)` for an integration to `t1` under `opts`, forgetting
    /// whatever integration it did before. `t0`, `t1` and `y0` are finite and `opts` has been
    /// checked when the entry points of the module call this.
    ///
    /// # Errors
    ///
    /// [`OdeFailure::NonFinite`] if the right-hand side is not finite at the initial point.
    fn start<F>(
        &mut self,
        f: &mut F,
        t0: T,
        y0: &Matrix<T>,
        t1: T,
        opts: &OdeOptions<T>,
    ) -> Result<(), OdeFailure>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>);

    /// Takes one step and accepts it: on return [`t`](Self::t) has moved towards `t1` and never
    /// past it, and the step that ends the integration ends on `t1` exactly. Rejected attempts
    /// are retried inside. Does nothing once `t1` is reached.
    ///
    /// # Errors
    ///
    /// The reason no step could be accepted, after which the method stays at the last point it
    /// had reached.
    fn step<F>(&mut self, f: &mut F) -> Result<(), OdeFailure>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>);

    /// The time reached.
    fn t(&self) -> T;

    /// The time the last accepted step started from; `t0` before the first.
    fn t_old(&self) -> T;

    /// The state at [`t`](Self::t), of the shape of the initial state.
    fn y(&self) -> &Matrix<T>;

    /// Writes the continuous extension of the last accepted step to `out`: a polynomial that
    /// takes the method's values at [`t_old`](Self::t_old) and [`t`](Self::t) and approximates
    /// the solution between them. It is only called between a successful [`step`](Self::step)
    /// and the next, and may evaluate `f` when the method needs more stages for it.
    ///
    /// # Errors
    ///
    /// [`OdeFailure::NonFinite`] if an evaluation it needs is not finite. The methods of this
    /// module answer a call before [`start`](Self::start) with [`OdeFailure::InvalidInput`], as
    /// they answer [`step`](Self::step).
    fn dense<F>(&mut self, f: &mut F, out: &mut DenseStep<T>) -> Result<(), OdeFailure>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>);

    /// The number of steps tried and rejected since [`start`](Self::start): zero unless the method
    /// controls its error.
    fn rejected(&self) -> usize {
        0
    }

    /// The number of Jacobians formed since [`start`](Self::start): zero unless the method is
    /// implicit.
    fn jacobians(&self) -> usize {
        0
    }

    /// The number of LU decompositions since [`start`](Self::start): zero unless the method is
    /// implicit.
    fn factorisations(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------------------------
// The integration
// ---------------------------------------------------------------------------------------------

/// What one run of the driver produced; the entry points each return a part of it.
struct Run<T> {
    status: OdeStatus,
    /// Whether the method was started, and so holds a point of the solution.
    started: bool,
    t: T,
    y: Matrix<T>,
    events: Vec<EventPoint<T>>,
    samples: Vec<Matrix<T>>,
    dense: Option<DenseOutput<T>>,
    steps: usize,
    rejected: usize,
    nfev: usize,
    njev: usize,
    nlu: usize,
}

/// Panics unless `y0` is a column, or empty.
#[track_caller]
fn check_shape<T>(y0: &Matrix<T>) {
    assert!(
        y0.cols() == 1 || y0.is_empty(),
        "the initial state must be a column"
    );
}

/// Whether the times `ts` can be sampled by one integration from `t0`: all finite, and never
/// going back towards `t0`, in whichever direction the last of them lies.
fn ordered_from<T: Real>(t0: T, ts: &[T]) -> bool {
    let Some(&last) = ts.last() else {
        return true;
    };
    if !ts.iter().all(|t| t.is_finite()) {
        return false;
    }
    let forwards = last >= t0;
    let mut previous = t0;
    for &t in ts {
        if (forwards && t < previous) || (!forwards && t > previous) {
            return false;
        }
        previous = t;
    }
    true
}

/// The integration behind every entry point: `method` from `(t0, y0)` to `t1` under `opts`,
/// watching `events`, sampling the solution at `samples` and keeping the dense output if `keep`.
#[allow(clippy::too_many_arguments)]
fn run<T, F, M>(
    mut f: F,
    y0: &Matrix<T>,
    t0: T,
    t1: T,
    mut method: M,
    events: &mut [Event<'_, T>],
    opts: &OdeOptions<T>,
    samples: &[T],
    keep: bool,
) -> Run<T>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    let mut out = Run {
        status: OdeStatus::Failed(OdeFailure::InvalidInput),
        started: false,
        t: t0,
        y: y0.clone(),
        events: Vec::new(),
        samples: Vec::new(),
        dense: None,
        steps: 0,
        rejected: 0,
        nfev: 0,
        njev: 0,
        nlu: 0,
    };
    if !opts.valid()
        || !Float::is_finite(t0)
        || !Float::is_finite(t1)
        || !control::all_finite(y0.as_slice())
        || !ordered_from(t0, samples)
        || samples.last().is_some_and(|&last| last != t1)
    {
        return out;
    }
    if keep {
        out.dense = Some(DenseOutput::new(t0, y0.as_slice()));
    }
    let mut nfev = 0usize;
    let mut counted = |t: T, y: &Matrix<T>, dy: &mut Matrix<T>| {
        nfev += 1;
        f(t, y, dy);
    };
    let result = advance(
        &mut counted,
        y0,
        t0,
        t1,
        &mut method,
        events,
        opts,
        samples,
        &mut out,
    );
    out.status = match result {
        // A terminal event has left its own time and state.
        Ok(OdeStatus::Terminated) => OdeStatus::Terminated,
        Ok(status) => status,
        Err(failure) => OdeStatus::Failed(failure),
    };
    if out.started && out.status != OdeStatus::Terminated {
        // Where the method stands: `t1`, or the last step it accepted before it failed.
        out.t = method.t();
        out.y = method.y().clone();
    }
    out.nfev = nfev;
    out.rejected = method.rejected();
    out.njev = method.jacobians();
    out.nlu = method.factorisations();
    out
}

/// The loop of [`run`]: steps until `t1`, a terminal event or a failure, leaving in `out` the
/// point reached and everything gathered on the way.
#[allow(clippy::too_many_arguments)]
fn advance<T, F, M>(
    f: &mut F,
    y0: &Matrix<T>,
    t0: T,
    t1: T,
    method: &mut M,
    events: &mut [Event<'_, T>],
    opts: &OdeOptions<T>,
    samples: &[T],
    out: &mut Run<T>,
) -> Result<OdeStatus, OdeFailure>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    let forwards = t1 >= t0;
    // Whether `a` lies beyond `b` in the direction of the integration.
    let beyond = |a: T, b: T| if forwards { a > b } else { a < b };

    // The samples at the initial time are the initial state itself.
    let mut next = 0;
    while next < samples.len() && samples[next] == t0 {
        out.samples.push(y0.clone());
        next += 1;
    }
    if t0 == t1 {
        return Ok(OdeStatus::Finished);
    }

    method.start(f, t0, y0, t1, opts)?;
    out.started = true;
    let mut before = Vec::with_capacity(events.len());
    for event in events.iter_mut() {
        let g = event.value(t0, y0);
        if g.is_nan() {
            return Err(OdeFailure::EventNotFinite);
        }
        before.push(g);
    }
    let mut after = before.clone();
    let mut step = DenseStep::new();
    let mut scratch = y0.clone();
    // The events of one step: when, which, and whether the function rose.
    let mut hits: Vec<(T, usize, bool)> = Vec::new();

    while method.t() != t1 {
        if out.steps >= opts.max_steps {
            return Err(OdeFailure::MaxSteps);
        }
        method.step(f)?;
        out.steps += 1;
        let t_new = method.t();
        // The continuous extension of the step, formed the first time something needs it.
        let mut formed = false;

        hits.clear();
        let t_old = method.t_old();
        for (i, event) in events.iter_mut().enumerate() {
            let g = event.value(t_new, method.y());
            if g.is_nan() {
                return Err(OdeFailure::EventNotFinite);
            }
            after[i] = g;
            let (mut was, is) = (events::sign(before[i]), events::sign(g));
            // Where the search for a crossing begins: the start of the step, unless the function
            // is zero there.
            let mut from = t_old;
            if was == 0 {
                // Zero where the step starts — at `t0`, or where the last step ended on an
                // arrival. Leaving zero is not an event, but a crossing after it is, so the sign
                // the function leaves with is read just after the start, on the continuous
                // extension, as CVODE does. A function that is zero at both ends shows nothing.
                if is == 0 {
                    continue;
                }
                let span = t_new - t_old;
                let nudge =
                    T::from_f64(100.0) * T::EPSILON * (Float::abs(t_old) + Float::abs(span));
                if nudge >= Float::abs(span) {
                    continue;
                }
                if !formed {
                    method.dense(f, &mut step)?;
                    formed = true;
                }
                from = if forwards {
                    t_old + nudge
                } else {
                    t_old - nudge
                };
                was = events::sign(
                    events::value_on(event, &step, from, &mut scratch)
                        .ok_or(OdeFailure::EventNotFinite)?,
                );
                if was == 0 {
                    continue;
                }
            }
            // Arriving at zero, or passing through it, is an event.
            if is == was || !event.admits(was < 0) {
                continue;
            }
            let time = if is == 0 {
                t_new
            } else {
                if !formed {
                    method.dense(f, &mut step)?;
                    formed = true;
                }
                events::locate(event, &step, from, was, &mut scratch)
                    .ok_or(OdeFailure::EventNotFinite)?
            };
            hits.push((time, i, was < 0));
        }
        // In the order they happened; the first terminal one ends the step, and the
        // integration, at its time.
        hits.sort_by(|a, b| {
            let order = a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal);
            let order = if forwards { order } else { order.reverse() };
            order.then(a.1.cmp(&b.1))
        });
        let stop = hits
            .iter()
            .find(|hit| events[hit.1].terminal)
            .map(|hit| hit.0);
        let t_end = stop.unwrap_or(t_new);
        for &(time, index, rising) in &hits {
            if beyond(time, t_end) {
                break;
            }
            let y = if time == t_new {
                method.y().clone()
            } else {
                step.eval_into(time, scratch.as_mut_slice());
                scratch.clone()
            };
            out.events.push(EventPoint {
                index,
                t: time,
                y,
                rising,
            });
        }

        // The samples that fall in the step, up to where it was cut.
        while next < samples.len() && !beyond(samples[next], t_end) {
            if samples[next] == t_new {
                out.samples.push(method.y().clone());
            } else {
                if !formed {
                    method.dense(f, &mut step)?;
                    formed = true;
                }
                step.eval_into(samples[next], scratch.as_mut_slice());
                out.samples.push(scratch.clone());
            }
            next += 1;
        }

        if stop.is_some() {
            out.t = t_end;
            out.y = if t_end == t_new {
                method.y().clone()
            } else {
                step.eval_into(t_end, scratch.as_mut_slice());
                scratch.clone()
            };
        }
        if let Some(dense) = out.dense.as_mut() {
            if !formed {
                method.dense(f, &mut step)?;
            }
            dense.push(core::mem::take(&mut step));
            if stop.is_some() {
                dense.end_at(t_end);
            }
        }
        if stop.is_some() {
            return Ok(OdeStatus::Terminated);
        }
        before.copy_from_slice(&after);
    }
    Ok(OdeStatus::Finished)
}

/// The state at `t₁` of the initial value problem `y′ = f(t, y)`, `y(t₀) = y₀`, by `method` under
/// the default [`OdeOptions`]; `None` if the integration could not reach `t₁`.
///
/// `f` is called as `f(t, y, dy)` and must leave `dy` holding `f(t, y)`; `y` and `dy` have the
/// shape of `y0`, which is a column. `t₁` may be before `t₀`.
///
/// # Panics
///
/// Panics if `y0` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince853, solve};
///
/// // The logistic equation y′ = y(1 − y) from 1/2, whose solution is 1/(1 + e⁻ᵗ).
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = y[(0, 0)] * (1.0 - y[(0, 0)]);
/// };
/// let y = solve(f, &Matrix::from_rows(&[[0.5]]), 0.0, 3.0, DormandPrince853::new()).unwrap();
/// assert!((y[(0, 0)] - 1.0 / (1.0 + (-3.0f64).exp())).abs() < 1e-7);
/// ```
#[track_caller]
pub fn solve<T, F, M>(f: F, y0: &Matrix<T>, t0: T, t1: T, method: M) -> Option<Matrix<T>>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    solve_with(f, y0, t0, t1, method, &OdeOptions::new())
}

/// [`solve`] under the options `opts`.
///
/// # Panics
///
/// Panics if `y0` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince853, OdeOptions, solve_with};
///
/// // Kepler's problem on an orbit of eccentricity 0.5, once round: the planet comes back.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     let r3 = (y[(0, 0)] * y[(0, 0)] + y[(1, 0)] * y[(1, 0)]).powf(1.5);
///     dy[(0, 0)] = y[(2, 0)];
///     dy[(1, 0)] = y[(3, 0)];
///     dy[(2, 0)] = -y[(0, 0)] / r3;
///     dy[(3, 0)] = -y[(1, 0)] / r3;
/// };
/// let y0 = Matrix::from_rows(&[[0.5], [0.0], [0.0], [3.0f64.sqrt()]]);
/// let opts = OdeOptions::new().with_tolerances(1e-12, 1e-14);
/// let y = solve_with(f, &y0, 0.0, 2.0 * core::f64::consts::PI, DormandPrince853::new(), &opts)
///     .unwrap();
/// for i in 0..4 {
///     assert!((y[(i, 0)] - y0[(i, 0)]).abs() < 1e-10);
/// }
/// ```
#[track_caller]
pub fn solve_with<T, F, M>(
    f: F,
    y0: &Matrix<T>,
    t0: T,
    t1: T,
    method: M,
    opts: &OdeOptions<T>,
) -> Option<Matrix<T>>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    integrate_with(f, y0, t0, t1, method, opts).map(|r| r.y)
}

/// The integration of `y′ = f(t, y)` from `(t₀, y₀)` to `t₁` by `method` under the default
/// [`OdeOptions`]: the state at `t₁` with the numbers of steps and evaluations it took, or `None`
/// if the integration could not reach `t₁`.
///
/// # Panics
///
/// Panics if `y0` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince5, integrate};
///
/// // The growth y′ = y from 1, whose value at 1 is e.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = y[(0, 0)];
/// let r = integrate(f, &Matrix::from_rows(&[[1.0]]), 0.0, 1.0, DormandPrince5::new()).unwrap();
/// assert!((r.y[(0, 0)] - core::f64::consts::E).abs() < 1e-6);
/// assert_eq!(r.t, 1.0);
/// assert!(r.steps >= 1 && r.nfev > 6 * r.steps);
/// ```
#[track_caller]
pub fn integrate<T, F, M>(f: F, y0: &Matrix<T>, t0: T, t1: T, method: M) -> Option<OdeResult<T>>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    integrate_with(f, y0, t0, t1, method, &OdeOptions::new())
}

/// [`integrate`] under the options `opts`.
///
/// # Panics
///
/// Panics if `y0` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince5, DormandPrince853, OdeOptions, integrate_with};
///
/// // At a stringent tolerance the method of order 8 does the same work in far fewer steps.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = y[(1, 0)];
///     dy[(1, 0)] = -y[(0, 0)];
/// };
/// let y0 = Matrix::from_rows(&[[1.0], [0.0]]);
/// let opts = OdeOptions::new().with_tolerances(1e-12, 1e-14);
/// let five = integrate_with(f, &y0, 0.0, 20.0, DormandPrince5::new(), &opts).unwrap();
/// let eight = integrate_with(f, &y0, 0.0, 20.0, DormandPrince853::new(), &opts).unwrap();
/// assert!((five.y[(0, 0)] - 20.0f64.cos()).abs() < 1e-10);
/// assert!((eight.y[(0, 0)] - 20.0f64.cos()).abs() < 1e-10);
/// assert!(eight.nfev * 3 < five.nfev);
/// ```
#[track_caller]
pub fn integrate_with<T, F, M>(
    f: F,
    y0: &Matrix<T>,
    t0: T,
    t1: T,
    method: M,
    opts: &OdeOptions<T>,
) -> Option<OdeResult<T>>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    check_shape(y0);
    let run = run(f, y0, t0, t1, method, &mut [], opts, &[], false);
    (run.status == OdeStatus::Finished).then_some(OdeResult {
        y: run.y,
        t: run.t,
        steps: run.steps,
        rejected: run.rejected,
        nfev: run.nfev,
        njev: run.njev,
        nlu: run.nlu,
    })
}

/// The states of the initial value problem `y′ = f(t, y)`, `y(t₀) = y₀`, at the times `ts`, by
/// `method` under the default [`OdeOptions`]; `None` if the integration could not reach the last
/// of them.
///
/// The times must be in the order of an integration from `t₀`: never decreasing if the last of
/// them is after `t₀`, never increasing if it is before; repeated times and `t₀` itself are
/// allowed, and times out of order give `None`. The integration takes the steps it would take
/// without them and reads the states off the continuous extension of each step, so the output
/// times cost no steps and no step is kept.
///
/// # Panics
///
/// Panics if `y0` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince5, solve_at};
///
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
/// let ts = [0.0, 0.5, 1.0, 1.5, 2.0];
/// let ys = solve_at(f, &Matrix::from_rows(&[[1.0]]), 0.0, &ts, DormandPrince5::new()).unwrap();
/// assert_eq!(ys.len(), 5);
/// for (t, y) in ts.iter().zip(&ys) {
///     assert!((y[(0, 0)] - (-t).exp()).abs() < 1e-6);
/// }
/// ```
#[track_caller]
pub fn solve_at<T, F, M>(f: F, y0: &Matrix<T>, t0: T, ts: &[T], method: M) -> Option<Vec<Matrix<T>>>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    solve_at_with(f, y0, t0, ts, method, &OdeOptions::new())
}

/// [`solve_at`] under the options `opts`.
///
/// # Panics
///
/// Panics if `y0` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince853, OdeOptions, solve_at_with};
///
/// // Backwards in time from t = 0, at a tolerance of twelve digits.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = y[(0, 0)];
/// let opts = OdeOptions::new().with_tolerances(1e-12, 1e-14);
/// let ts = [-0.25, -0.5, -1.0];
/// let ys = solve_at_with(f, &Matrix::from_rows(&[[1.0]]), 0.0, &ts, DormandPrince853::new(),
///                        &opts).unwrap();
/// for (t, y) in ts.iter().zip(&ys) {
///     assert!((y[(0, 0)] - t.exp()).abs() < 1e-11);
/// }
/// ```
#[track_caller]
pub fn solve_at_with<T, F, M>(
    f: F,
    y0: &Matrix<T>,
    t0: T,
    ts: &[T],
    method: M,
    opts: &OdeOptions<T>,
) -> Option<Vec<Matrix<T>>>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    check_shape(y0);
    let Some(&t1) = ts.last() else {
        return Some(Vec::new());
    };
    let run = run(f, y0, t0, t1, method, &mut [], opts, ts, false);
    (run.status == OdeStatus::Finished).then_some(run.samples)
}

/// The solution of the initial value problem `y′ = f(t, y)`, `y(t₀) = y₀`, from `t₀` to `t₁` by
/// `method` under the default [`OdeOptions`], with the `events` located along it.
///
/// The [`OdeSolution`] returned says how the integration ended — at `t₁`, at a terminal event, or
/// in a failure, with the point it reached — and holds the events that occurred and the solution
/// as a function of time. Pass `&mut []` for no events.
///
/// # Panics
///
/// Panics if `y0` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince853, Event, OdeStatus, solve_ivp};
///
/// // The pendulum θ″ = −sin θ released from 2 radians: the times it passes through the
/// // vertical, and the first time it comes to rest again, which is half its period.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = y[(1, 0)];
///     dy[(1, 0)] = -y[(0, 0)].sin();
/// };
/// let vertical = Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)]);
/// let at_rest = Event::new(|_t: f64, y: &Matrix<f64>| y[(1, 0)]).with_terminal(true);
/// let y0 = Matrix::from_rows(&[[2.0], [0.0]]);
/// let mut events = [vertical, at_rest];
/// let solution = solve_ivp(f, &y0, 0.0, 100.0, DormandPrince853::new(), &mut events);
///
/// assert_eq!(solution.status, OdeStatus::Terminated);
/// assert_eq!(solution.events.len(), 2);
/// let (crossing, rest) = (&solution.events[0], &solution.events[1]);
/// assert_eq!((crossing.index, rest.index), (0, 1));
/// // The swing is symmetric: the vertical is passed at half the time it takes to stop, and
/// // it stops as far on the other side as it started.
/// assert!((2.0 * crossing.t - rest.t).abs() < 1e-5);
/// assert_eq!(solution.t, rest.t);
/// assert!((solution.y[(0, 0)] + 2.0).abs() < 1e-5);
/// // The solution is there to be read at any time up to the stop.
/// assert!((solution.eval(crossing.t).unwrap()[(0, 0)]).abs() < 1e-9);
/// assert!(solution.eval(rest.t + 1.0).is_none());
/// ```
#[track_caller]
pub fn solve_ivp<T, F, M>(
    f: F,
    y0: &Matrix<T>,
    t0: T,
    t1: T,
    method: M,
    events: &mut [Event<'_, T>],
) -> OdeSolution<T>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    solve_ivp_with(f, y0, t0, t1, method, events, &OdeOptions::new())
}

/// [`solve_ivp`] under the options `opts`.
///
/// # Panics
///
/// Panics if `y0` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince5, Event, EventDirection, OdeOptions, solve_ivp_with};
///
/// // The zeros of sin t from below, counted over a hundred periods without keeping a step.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = y[(1, 0)];
///     dy[(1, 0)] = -y[(0, 0)];
/// };
/// let up = Event::new(|_t: f64, y: &Matrix<f64>| y[(0, 0)])
///     .with_direction(EventDirection::Rising);
/// let y0 = Matrix::from_rows(&[[0.0], [1.0]]);
/// let opts = OdeOptions::new().with_tolerances(1e-9, 1e-12).with_dense_output(false);
/// let end = 200.5 * core::f64::consts::PI;
/// let solution = solve_ivp_with(f, &y0, 0.0, end, DormandPrince5::new(), &mut [up], &opts);
/// // The zero at the start is not an event; those at 2π, 4π, …, 200π are.
/// assert_eq!(solution.events.len(), 100);
/// for (k, event) in solution.events.iter().enumerate() {
///     let exact = 2.0 * core::f64::consts::PI * (k + 1) as f64;
///     assert!((event.t - exact).abs() < 1e-6);
/// }
/// assert!(solution.dense.is_none());
/// ```
#[track_caller]
pub fn solve_ivp_with<T, F, M>(
    f: F,
    y0: &Matrix<T>,
    t0: T,
    t1: T,
    method: M,
    events: &mut [Event<'_, T>],
    opts: &OdeOptions<T>,
) -> OdeSolution<T>
where
    T: Real + Scalar,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    M: OdeMethod<T>,
{
    check_shape(y0);
    let run = run(f, y0, t0, t1, method, events, opts, &[], opts.dense_output);
    OdeSolution {
        status: run.status,
        t: run.t,
        y: run.y,
        events: run.events,
        dense: run.dense,
        steps: run.steps,
        rejected: run.rejected,
        nfev: run.nfev,
        njev: run.njev,
        nlu: run.nlu,
    }
}
