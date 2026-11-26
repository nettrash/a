//! The minima of functions: of one variable, of several, and of a sum of squares.
//!
//! | Method | Needs | Rate | `converged` says |
//! | --- | --- | --- | --- |
//! | [`golden_section`] | a bracket, unimodality | linear, 0.618 of the bracket a step | the bracket is within the tolerance on the point |
//! | [`minimize`] | a bracket, unimodality | superlinear on a smooth function | the same |
//! | [`nelder_mead`] | `f`, a start and a first step | linear | the simplex is within the tolerances and no neighbour of its best corner is better |
//! | [`conjugate_gradient`] | `f` and ∇`f` | linear | `‖∇f‖∞ ≤ gtol` |
//! | [`bfgs`] | `f` and ∇`f` | superlinear | `‖∇f‖∞ ≤ gtol` |
//! | [`gauss_newton`] | the residual and its Jacobian | quadratic on a small residual | the fit is stationary, or exact to rounding, by one of four tests |
//! | [`levenberg_marquardt`] | the residual and its Jacobian | between 1 and 2 | the same |
//!
//! Minimisation within bounds is [`lbfgsb`](super::lbfgsb()), in a file of its own.
//!
//! A bracketed method is told where the minimum is and cannot miss it, but it only works on a
//! function that falls and then rises once inside the bracket; an open method is told only where
//! to start, so it converges to whichever stationary point the starting point leads it to.
//!
//! None of them reports an answer it has not verified. `converged` is true only when the test of
//! the table held at the point returned: a method that runs out of budget, or that can make no
//! further progress, returns the best point it reached with `converged` false, and one that has
//! nothing to return — a start it cannot evaluate, tolerances that are not numbers — returns
//! `None`. A step that has become short, or a value that has stopped improving, is never taken
//! for convergence where it may be the method stalling, which looks the same as arriving: the
//! gradient methods stop on the gradient alone, and the two fits judge a step by its length or
//! its gain only when it solves the linearised problem, and its length parameter by parameter,
//! so that the scale of one parameter cannot hide the step of another. A short step proves
//! something only where the linearised problem it solved is the problem, so the fits believe
//! one, and a residual at the level of rounding, only where the Jacobian at the end of the step
//! is still the one the step was made with: beside a pole of the model a step that is short
//! against its parameter quarters the cost and is followed by another like it.
//!
//! The tolerances and the budget are the same [`OptOptions`] for all of them; which of its fields
//! a method reads is stated there. The gradient methods stop on the infinity norm of the
//! gradient, [`OptOptions::gtol`], which is why the default is `√ε` rather than `ε`: at a minimum
//! the function is flat, so the gradient cannot be trusted further than the square root of the
//! precision the values are computed in.
//!
//! # Examples
//!
//! ```
//! use a::math::numerics::{OptOptions, bfgs, minimize};
//! use a::math::matrix::Matrix;
//!
//! // The minimum of (x² − 2)², which is √2, by Brent on a bracket.
//! let r = minimize(|x: f64| (x * x - 2.0).powi(2), 1.0, 2.0).unwrap();
//! assert!((r.x - core::f64::consts::SQRT_2).abs() < 1e-7);
//! assert!(r.converged);
//!
//! // Rosenbrock's valley, by Broyden–Fletcher–Goldfarb–Shanno on the gradient.
//! let f = |x: &Matrix<f64>| {
//!     let (a, b) = (x[(0, 0)], x[(1, 0)]);
//!     (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
//! };
//! let g = |x: &Matrix<f64>| {
//!     let (a, b) = (x[(0, 0)], x[(1, 0)]);
//!     Matrix::from_column_major(2, 1, vec![
//!         -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
//!         200.0 * (b - a * a),
//!     ])
//! };
//! let x = bfgs(f, g, &Matrix::from_column_major(2, 1, vec![-1.2, 1.0])).unwrap();
//! assert!((x.x[(0, 0)] - 1.0).abs() < 1e-6 && (x.x[(1, 0)] - 1.0).abs() < 1e-6);
//! assert!(x.converged);
//! ```

use super::{Real, ordered};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};

/// The fraction of the decrease its tangent promises that a step has to deliver to be accepted:
/// the `μ` of the sufficient decrease condition of the line search.
const DECREASE: f64 = 1e-4;
/// How much of the starting slope a step of a quasi-Newton method may keep: the `η` of the
/// curvature condition. A loose search suits a method whose unit step is nearly right.
const CURVATURE_QUASI_NEWTON: f64 = 0.9;
/// The same for the conjugate gradient, whose directions are conjugate only as far as each
/// search is exact.
const CURVATURE_CONJUGATE: f64 = 0.1;
/// The relative width of the bracket of a line search below which it is not narrowed further.
const LINE_SEARCH_NARROW: f64 = 1e-10;
/// How many times a line search evaluates the function before it settles for the best step it
/// has seen.
const LINE_SEARCH_EVALUATIONS: usize = 40;
/// The longest step a line search of the unbounded methods tries.
const LONGEST_STEP: f64 = 1e20;

/// Before it has bracketed a minimum the line search of Moré and Thuente lengthens its trial step
/// by between these two multiples of the distance from the best step so far (their section 2).
const EXTRAPOLATE_LEAST: f64 = 1.1;
const EXTRAPOLATE_MOST: f64 = 4.0;
/// The fraction a bracket has to shrink to over two trials, or the next one bisects it.
const BRACKET_SHRINK: f64 = 0.66;

/// The first trust region of the Levenberg and Marquardt method, in multiples of the length of
/// the scaled starting point (Moré's `factor`).
const REGION_FACTOR: f64 = 100.0;
/// How close to the edge of the trust region a damped step has to come for the damping to be
/// accepted: within this fraction of the radius (Moré's `σ`).
const REGION_FIT: f64 = 0.1;
/// The most trials of the damping for one step (Moré found two enough on average).
const DAMPING_TRIALS: usize = 10;
/// The share of the reduction the linearised problem predicts that a step has to deliver to be
/// taken.
const ACCEPT: f64 = 1e-4;
/// How far the Jacobian may move over a step for a test made on that step to be believed, as a
/// fraction of its size (see [`jacobian_change`]). A test made on a step reads the linearised
/// problem as the problem; over a step that the tolerances call short, or that only refines the
/// rounding, the Jacobian of a smooth model moves by the tolerance times its sensitivity, while a
/// whole step beside a pole of the model cuts the column of the parameter it moves to a quarter
/// (a change of ¾), and one beside a logarithmic singularity to a half at most.
const JACOBIAN_HOLDS: f64 = 0.25;

/// Whether `x` is finite, named through [`Float`], as [`Scalar`] names the same test and both are
/// in scope wherever a matrix element is a float.
pub(super) fn finite<T: Real>(x: T) -> bool {
    Float::is_finite(x)
}

/// The magnitude of `x`, named through [`Float`] for the same reason.
pub(super) fn magnitude<T: Real>(x: T) -> T {
    Float::abs(x)
}

/// The square root of `x`, named through [`Float`] for the same reason.
pub(super) fn root<T: Real>(x: T) -> T {
    Float::sqrt(x)
}

/// Whether `a` is the better value, a value that is not finite never being better than one that
/// is. A function may be undefined where an open method wanders; such a point is then simply the
/// worst the method has seen rather than an error.
fn better<T: Real>(a: T, b: T) -> bool {
    finite(a) && (!finite(b) || a < b)
}

/// Whether `a` is no worse than `b`, which is the test a step is accepted by where a step of the
/// very same value is worth taking.
fn not_worse<T: Real>(a: T, b: T) -> bool {
    finite(a) && (!finite(b) || a <= b)
}

/// The largest of the absolute values of a vector, the norm of the stopping tests on a gradient.
pub(super) fn norm_inf<T: Real>(v: &[T]) -> T {
    v.iter()
        .fold(T::ZERO, |largest, x| largest.max(magnitude(*x)))
}

/// The Euclidean length of a vector, scaled by its largest element so that the squares neither
/// overflow nor vanish.
pub(super) fn norm2<T: Real>(v: &[T]) -> T {
    let largest = norm_inf(v);
    if largest == T::ZERO || !finite(largest) {
        return largest;
    }
    let squares: T = v
        .iter()
        .map(|x| {
            let scaled = *x / largest;
            scaled * scaled
        })
        .sum();
    largest * root(squares)
}

/// The inner product `⟨a, b⟩` of two vectors.
pub(super) fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b).map(|(x, y)| *x * *y).sum()
}

/// The length `‖D·x‖` of `x` scaled element by element by the diagonal `scale` of `D`.
fn scaled_norm<T: Real>(x: &[T], scale: &[T]) -> T {
    let scaled: Vec<T> = x.iter().zip(scale).map(|(x, d)| *x * *d).collect();
    norm2(&scaled)
}

/// Whether every element of a vector is finite.
pub(super) fn all_finite<T: Real>(v: &[T]) -> bool {
    v.iter().all(|x| finite(*x))
}

/// The number of elements of the column `x`.
///
/// # Panics
///
/// Panics if `x` has not exactly one column.
#[track_caller]
pub(super) fn column_length<T>(x: &Matrix<T>, what: &str) -> usize {
    assert!(
        x.cols() == 1,
        "{what} is not a column: it has {} columns",
        x.cols()
    );
    x.rows()
}

/// When an optimisation method stops, and when it gives up.
///
/// Each method reads the fields that mean something to it:
///
/// | Method | `xtol`, `rtol` | `ftol` | `gtol` | `max_iter` counts |
/// | --- | --- | --- | --- | --- |
/// | [`golden_section`], [`minimize`] | the width of the bracket, `xtol + rtol·max(\|x\|, 1)` | not read | not read | evaluations after the first |
/// | [`nelder_mead`] | the size of the simplex, `xtol + rtol·max(‖x‖∞, 1)` | the spread of its values, `ftol + rtol·max(\|f\|, 1)` | not read | reflections and polls |
/// | [`conjugate_gradient`], [`bfgs`] | not read | not read | `‖∇f‖∞` | steps |
/// | [`gauss_newton`], [`levenberg_marquardt`] | the step of each parameter, `xtol + rtol·\|xⱼ\|` | the reduction of the cost, `ftol + rtol·cost` | the cosine between the residual and the columns of the Jacobian | trial steps |
///
/// The gradient methods read neither tolerance on the point nor on the value: a short step or a
/// small gain is what a stalled method shows as well as a finished one, and only the gradient
/// tells the two apart.
///
/// The defaults are `xtol` 0, `rtol` 4√ε, `ftol` 0, `gtol` √ε and `max_iter` 1000, with ε that of
/// the float type, so that they suit `f32` as they suit `f64`. A tolerance that is negative or not
/// a number is refused by every method, which then returns `None`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::OptOptions;
///
/// let opts = OptOptions::new().with_gtol(1e-10).with_max_iter(200);
/// assert_eq!(opts.gtol, 1e-10);
/// assert_eq!(opts.max_iter, 200);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptOptions<T> {
    /// The absolute part of the tolerance on the point.
    pub xtol: T,
    /// The relative part of the tolerance on the point, and on the value.
    pub rtol: T,
    /// The absolute part of the tolerance on the value; 0 asks for none.
    pub ftol: T,
    /// The size of the gradient at which a stationary point is accepted.
    pub gtol: T,
    /// The largest number of iterations.
    pub max_iter: usize,
}

impl<T: Real> Default for OptOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> OptOptions<T> {
    /// The defaults: `xtol` 0, `rtol` 4√ε, `ftol` 0, `gtol` √ε, `max_iter` 1000.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::OptOptions;
    ///
    /// let opts = OptOptions::<f64>::new();
    /// assert_eq!(opts.gtol, f64::EPSILON.sqrt());
    /// assert_eq!(opts.rtol, 4.0 * f64::EPSILON.sqrt());
    /// assert_eq!((opts.xtol, opts.ftol, opts.max_iter), (0.0, 0.0, 1000));
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            xtol: T::ZERO,
            rtol: T::EPSILON.sqrt() * T::from_f64(4.0),
            ftol: T::ZERO,
            gtol: T::EPSILON.sqrt(),
            max_iter: 10 * super::DEFAULT_MAX_ITER,
        }
    }

    /// Sets the absolute tolerance on the point.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::OptOptions;
    ///
    /// assert_eq!(OptOptions::<f64>::new().with_xtol(1e-9).xtol, 1e-9);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_xtol(mut self, xtol: T) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the relative tolerance on the point and on the value.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::OptOptions;
    ///
    /// assert_eq!(OptOptions::<f64>::new().with_rtol(1e-12).rtol, 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    /// Sets the absolute tolerance on the value.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::OptOptions;
    ///
    /// assert_eq!(OptOptions::<f64>::new().with_ftol(1e-6).ftol, 1e-6);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_ftol(mut self, ftol: T) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets the tolerance on the gradient.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::OptOptions;
    ///
    /// assert_eq!(OptOptions::<f64>::new().with_gtol(1e-5).gtol, 1e-5);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_gtol(mut self, gtol: T) -> Self {
        self.gtol = gtol;
        self
    }

    /// Sets the largest number of iterations.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::OptOptions;
    ///
    /// assert_eq!(OptOptions::<f64>::new().with_max_iter(50).max_iter, 50);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// The tolerance on a point at `x`: `xtol + rtol · max(|x|, 1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::OptOptions;
    ///
    /// let opts = OptOptions::<f64>::new().with_xtol(1e-3).with_rtol(1e-6);
    /// assert_eq!(opts.tol_at(0.5), 1e-3 + 1e-6);
    /// assert_eq!(opts.tol_at(-200.0), 1e-3 + 2e-4);
    /// ```
    #[must_use]
    pub fn tol_at(&self, x: T) -> T {
        self.xtol + self.rtol * magnitude(x).max(T::ONE)
    }

    /// The same for a whole point, by the infinity norm of the column.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::OptOptions;
    ///
    /// let opts = OptOptions::<f64>::new().with_xtol(0.0).with_rtol(1e-6);
    /// let x = Matrix::from_column_major(2, 1, vec![3.0, -40.0]);
    /// assert!((opts.tol_of(&x) - 4e-5).abs() < 1e-18);
    /// ```
    #[must_use]
    pub fn tol_of(&self, x: &Matrix<T>) -> T {
        self.xtol + self.rtol * norm_inf(x.as_slice()).max(T::ONE)
    }

    /// Whether the tolerances are ones a method can work to: none negative, none not a number.
    fn sane(&self) -> bool {
        self.xtol >= T::ZERO && self.rtol >= T::ZERO && self.ftol >= T::ZERO && self.gtol >= T::ZERO
    }
}

/// The point a one-dimensional method found, and how hard it worked for it.
///
/// # Examples
///
/// ```
/// use a::math::numerics::minimize;
///
/// let r = minimize(|x: f64| (x - 3.0).powi(2) + 1.0, 0.0, 10.0).unwrap();
/// assert!((r.x - 3.0).abs() < 1e-7);
/// assert!((r.value - 1.0).abs() < 1e-14);
/// assert!(r.converged);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneDimResult<T> {
    /// The point the method found.
    pub x: T,
    /// The value of the function there.
    pub value: T,
    /// The number of times the function was evaluated.
    pub evaluations: usize,
    /// Whether the bracket had become as narrow as the tolerance asks.
    pub converged: bool,
}

/// The point a method of several variables found, and how hard it worked for it.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::nelder_mead;
///
/// let f = |x: &Matrix<f64>| (x[(0, 0)] - 1.0).powi(2) + (x[(1, 0)] + 2.0).powi(2);
/// let r = nelder_mead(f, &Matrix::from_column_major(2, 1, vec![0.0, 0.0]), 1.0).unwrap();
/// assert!((r.x[(0, 0)] - 1.0).abs() < 1e-6 && (r.x[(1, 0)] + 2.0).abs() < 1e-6);
/// assert!(r.value < 1e-12 && r.converged && r.iterations > 0);
/// ```
#[derive(Debug, Clone)]
pub struct MinResult<T> {
    /// The point the method found.
    pub x: Matrix<T>,
    /// The value of the function there.
    pub value: T,
    /// The number of iterations: steps taken for the gradient methods, each one that lowered the
    /// value or, where the value was level within its rounding, the largest element of the
    /// gradient; for the simplex its moves, each a reflection and all that came with it, and its
    /// polls.
    pub iterations: usize,
    /// Whether the test of the method held at `x`, rather than the budget or the progress running
    /// out.
    pub converged: bool,
}

/// The parameters a least squares method found.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::gauss_newton;
///
/// // The mean of three numbers is the constant that fits them best.
/// let r = |p: &Matrix<f64>| Matrix::from_column_major(3, 1, vec![p[(0, 0)] - 1.0, p[(0, 0)] - 2.0, p[(0, 0)] - 6.0]);
/// let j = |_: &Matrix<f64>| Matrix::from_column_major(3, 1, vec![1.0, 1.0, 1.0]);
/// let fit = gauss_newton(r, j, &Matrix::from_column_major(1, 1, vec![0.0])).unwrap();
/// assert!((fit.x[(0, 0)] - 3.0).abs() < 1e-14);
/// assert!((fit.cost - 7.0).abs() < 1e-13);
/// assert!(fit.converged);
/// ```
#[derive(Debug, Clone)]
pub struct FitResult<T> {
    /// The parameters.
    pub x: Matrix<T>,
    /// Half the sum of the squares of the residuals there, the quantity the method minimises.
    pub cost: T,
    /// The number of trial steps, whether or not each was taken.
    pub iterations: usize,
    /// Whether the fit is stationary by one of the tests of the method, rather than the budget or
    /// the progress having run out.
    pub converged: bool,
}

// ---------------------------------------------------------------------------------------------
// One variable
// ---------------------------------------------------------------------------------------------

/// The reciprocal of the golden ratio, `(√5 − 1)/2`: the fraction of a bracket that a step of the
/// golden section keeps.
fn golden<T: Real>() -> T {
    (root(T::from_f64(5.0)) - T::ONE) * T::HALF
}

/// The point the fraction `t` of the way from `a` to `b`, by the form whose terms cannot
/// overflow where `b − a` would, held inside `[a, b]` against the last rounding.
fn section<T: Real>(a: T, b: T, t: T) -> T {
    (a * (T::ONE - t) + b * t).max(a.min(b)).min(a.max(b))
}

/// Whether `[a, b]` is a bracket a search can divide: `a < b`, both finite, and `b − a` a float,
/// so that the bracket and every part of it can be measured.
fn usable_bracket<T: Real>(a: T, b: T) -> bool {
    ordered(a, b) && finite(a) && finite(b) && finite(b - a)
}

/// The narrowest a bracket about `x` can be asked to become: the tolerance on the point, and no
/// less than four units in the last place of `x`, below which the floats inside it run out.
fn bracket_tolerance<T: Real>(opts: &OptOptions<T>, x: T) -> T {
    opts.tol_at(x)
        .max(T::from_f64(4.0) * T::EPSILON * magnitude(x))
        .max(T::MIN_POSITIVE)
}

/// Where a bracketed search starts.
enum Begin<T> {
    /// A bracket, the two points that divide it in the golden ratio, and the values at them: that
    /// at `low` always, that at `high` only if `low` is where the function is not defined. The
    /// function is finite at one of the two.
    Inside {
        left: T,
        right: T,
        low: T,
        f_low: T,
        high: T,
        f_high: Option<T>,
        evaluations: usize,
    },
    /// The function is defined at an end of the bracket and at no point the search found short of
    /// it: that end is the answer.
    End(OneDimResult<T>),
}

/// The start of both bracketed methods: the two points of the golden section, and, where the
/// function is defined at neither, the bracket closed in on the end at which it is defined until
/// one of them is.
///
/// A function that is undefined on part of its bracket is taken to be `+∞` there, and has to fall
/// and rise once in that sense: the set on which it is finite is then an interval, and when both
/// points lie outside it and exactly one end of the bracket inside, that end tells on which side
/// the interval is. With both ends finite or neither there is no telling, and `None` is all that
/// can be said.
fn begin<T: Real>(
    f: &mut impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &OptOptions<T>,
) -> Option<Begin<T>> {
    let ratio = golden::<T>();
    let (mut left, mut right) = (a, b);
    let mut evaluations = 0;
    let mut end: Option<(bool, T)> = None;
    let mut rounds = 0;
    loop {
        let low = section(left, right, T::ONE - ratio);
        let high = section(left, right, ratio);
        let f_low = f(low);
        evaluations += 1;
        if finite(f_low) {
            return Some(Begin::Inside {
                left,
                right,
                low,
                f_low,
                high,
                f_high: None,
                evaluations,
            });
        }
        let f_high = f(high);
        evaluations += 1;
        if finite(f_high) {
            return Some(Begin::Inside {
                left,
                right,
                low,
                f_low,
                high,
                f_high: Some(f_high),
                evaluations,
            });
        }
        let (at_left, value) = match end {
            Some(known) => known,
            None => {
                let (f_a, f_b) = (f(a), f(b));
                evaluations += 2;
                let known = match (finite(f_a), finite(f_b)) {
                    (true, false) => (true, f_a),
                    (false, true) => (false, f_b),
                    _ => return None,
                };
                end = Some(known);
                known
            }
        };
        if at_left {
            right = low;
        } else {
            left = high;
        }
        rounds += 1;
        let x = if at_left { left } else { right };
        let narrow = right - left <= bracket_tolerance(opts, x);
        if narrow || rounds >= opts.max_iter {
            return Some(Begin::End(OneDimResult {
                x,
                value,
                evaluations,
                converged: narrow,
            }));
        }
    }
}

/// The minimum of `f` within `[a, b]`, by the golden section.
///
/// The function has to fall and then rise once inside the bracket, which is what lets the method
/// throw away 0.382 of the bracket for one evaluation a step; it is the slowest method here and
/// the only one that cannot be misled by a flat or a noisy function. Use [`minimize`] unless the
/// function is one parabolic interpolation cannot be trusted on.
///
/// Returns `None` when the bracket is empty, not finite or wider than the largest float, when a
/// tolerance is negative or not a number, when no point at which `f` is finite can be found, or
/// when the bracket has not closed within the default budget: see [`golden_section_with`], which
/// returns the best point it has in that last case.
///
/// # Examples
///
/// ```
/// use a::math::numerics::golden_section;
///
/// // The minimum of x ↦ x⁴ − 2x² on [0, 2] is at one, with value −1.
/// let x = golden_section(|x: f64| x.powi(4) - 2.0 * x * x, 0.0, 2.0).unwrap();
/// assert!((x - 1.0).abs() < 1e-7);
/// ```
pub fn golden_section<T: Real>(f: impl FnMut(T) -> T, a: T, b: T) -> Option<T> {
    golden_section_with(f, a, b, &OptOptions::new())
        .filter(|r| r.converged)
        .map(|r| r.x)
}

/// [`golden_section`] under `opts`, reporting the value, the work and whether the bracket closed.
///
/// The bracket is accepted, and `converged` is true, once it is no wider than
/// `xtol + rtol · max(|x|, 1)` about the best point `x` — or than four units in the last place of
/// `x`, which no bracket can be narrowed below. `max_iter` is the number of evaluations after the
/// first two; when it runs out the best point is returned with `converged` false. `ftol` and
/// `gtol` are not read.
///
/// Where `f` is not finite it is taken to be `+∞`: a function defined on part of the bracket is
/// minimised over that part, provided it is defined at one of the two points the search starts
/// from, or at exactly one end of the bracket. Otherwise the result is `None`.
///
/// Falling and then rising has to hold for the values as computed: where they are equal over a
/// stretch of the bracket — `atan(x)²` beyond `|x| = 10¹⁶`, where the arctangent is `π/2` to the
/// last bit — two points on that stretch cannot tell the method which side the minimum is on,
/// and it can close the bracket on the stretch. A bracket whose width `b − a` is not a float is
/// refused for the same reason: its parts cannot be measured.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{OptOptions, golden_section_with};
///
/// let r = golden_section_with(|x: f64| (x - 0.25).abs(), 0.0, 1.0, &OptOptions::new()).unwrap();
/// assert!((r.x - 0.25).abs() < 1e-7 && r.converged);
/// // Every step keeps 0.618 of the bracket, so a relative width of 6·10⁻⁸ takes some 35.
/// assert!(r.evaluations > 30 && r.evaluations < 45);
/// // x ln x is defined right of zero only; its minimum is at 1/e.
/// let r = golden_section_with(|x: f64| x * x.ln(), -3.0, 1.0, &OptOptions::new()).unwrap();
/// assert!((r.x - (-1.0f64).exp()).abs() < 1e-7);
/// ```
pub fn golden_section_with<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &OptOptions<T>,
) -> Option<OneDimResult<T>> {
    if !usable_bracket(a, b) || !opts.sane() {
        return None;
    }
    let ratio = golden::<T>();
    let (mut left, mut right, mut low, mut f_low, mut high, mut f_high, mut evaluations) =
        match begin(&mut f, a, b, opts)? {
            Begin::End(result) => return Some(result),
            Begin::Inside {
                left,
                right,
                low,
                f_low,
                high,
                f_high,
                evaluations,
            } => {
                let (f_high, evaluations) = match f_high {
                    Some(value) => (value, evaluations),
                    None => (f(high), evaluations + 1),
                };
                (left, right, low, f_low, high, f_high, evaluations)
            }
        };

    let mut steps = 0;
    let mut converged = false;
    loop {
        let x = if better(f_low, f_high) { low } else { high };
        if right - left <= bracket_tolerance(opts, x) {
            converged = true;
            break;
        }
        if steps >= opts.max_iter {
            break;
        }
        // Whichever part of the bracket is thrown away, one of the two points is where the new
        // bracket wants one of its own, so that only the other is new. The new one is placed
        // 0.382 of the way from the point kept to the near end, not from the ends: from the ends
        // the rounding of each step moves the pair off the golden ratio by 1.6 times as much
        // again the next, and a long search (a bracket of 10³⁰⁰ takes 1,500 steps) loses it.
        if better(f_low, f_high) {
            let point = section(low, left, T::ONE - ratio);
            if !(left < point && point < low) {
                // No float is left between the two: the bracket is as narrow as it can be.
                converged = true;
                break;
            }
            right = high;
            high = low;
            f_high = f_low;
            low = point;
            f_low = f(low);
        } else {
            let point = section(high, right, T::ONE - ratio);
            if !(high < point && point < right) {
                converged = true;
                break;
            }
            left = low;
            low = high;
            f_low = f_high;
            high = point;
            f_high = f(high);
        }
        evaluations += 1;
        steps += 1;
    }

    let (x, value) = if better(f_low, f_high) {
        (low, f_low)
    } else {
        (high, f_high)
    };
    Some(OneDimResult {
        x,
        value,
        evaluations,
        converged,
    })
}

/// The minimum of `f` within `[a, b]`, by Brent's method: a step to the minimum of the parabola
/// through the three best points whenever that step can be trusted, and a step of the golden
/// section whenever it cannot.
///
/// The function has to fall and then rise once inside the bracket. Where it is smooth the
/// parabola is nearly right and the method converges superlinearly; where it is not, every step is
/// a golden one and it is as slow and as sure as [`golden_section`]. This is the method to reach
/// for in one variable.
///
/// Returns `None` when the bracket is empty, not finite or wider than the largest float, when a
/// tolerance is negative or not a number, or when no point at which `f` is finite can be found:
/// see [`minimize_with`].
///
/// # Examples
///
/// ```
/// use a::math::numerics::minimize;
///
/// // The quartic x⁴ − 2x² has its minimum on [0, 2] at one, with value −1.
/// let r = minimize(|x: f64| x.powi(4) - 2.0 * x * x, 0.0, 2.0).unwrap();
/// assert!((r.x - 1.0).abs() < 1e-7);
/// assert!((r.value + 1.0).abs() < 1e-14);
/// // A third of the evaluations the golden section needs for the same bracket.
/// assert!(r.evaluations < 16);
/// ```
pub fn minimize<T: Real>(f: impl FnMut(T) -> T, a: T, b: T) -> Option<OneDimResult<T>> {
    minimize_with(f, a, b, &OptOptions::new())
}

/// [`minimize`] under `opts`.
///
/// This is `localmin` of Brent's *Algorithms for Minimization without Derivatives* (1973,
/// chapter 5), with his tolerance `tol` a quarter of `xtol + rtol · max(|x|, 1)`: the method
/// stops, and `converged` is true, once the minimum is bracketed within `2·tol` on either side of
/// the best point `x`, a bracket no wider than the tolerance on the point, and it never evaluates
/// `f` at two points closer than `tol`. A tolerance below two units in the last place of `x` is
/// raised to that. `max_iter` is the number of evaluations after the first; when it runs out the
/// best point is returned with `converged` false. `ftol` and `gtol` are not read.
///
/// The default `rtol` of `4√ε` makes `tol` `√ε·max(|x|, 1)`, which is as close as the values of
/// a smooth function can tell two points near its minimum apart.
///
/// Where `f` is not finite it is taken to be `+∞`, as [`golden_section_with`] takes it.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{OptOptions, minimize_with};
///
/// // A parabola is found by the first interpolation: six evaluations in all.
/// let r = minimize_with(|x: f64| (x - 3.0).powi(2) + 1.0, 0.0, 10.0, &OptOptions::new()).unwrap();
/// assert!((r.x - 3.0).abs() < 1e-7 && r.converged);
/// assert!(r.evaluations <= 8);
/// // A kink has no parabola, and takes about as long as the golden section does.
/// let r = minimize_with(|x: f64| (x - 0.25).abs(), 0.0, 1.0, &OptOptions::new()).unwrap();
/// assert!((r.x - 0.25).abs() < 1e-7 && r.converged);
/// assert!(r.evaluations > 20);
/// ```
pub fn minimize_with<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &OptOptions<T>,
) -> Option<OneDimResult<T>> {
    if !usable_bracket(a, b) || !opts.sane() {
        return None;
    }
    // The fraction of the bracket a golden step leaves behind it, (3 − √5)/2.
    let short = T::ONE - golden::<T>();
    let (mut left, mut right, mut x, mut f_x, mut evaluations) = match begin(&mut f, a, b, opts)? {
        Begin::End(result) => return Some(result),
        Begin::Inside {
            left,
            right,
            low,
            f_low,
            high,
            f_high,
            evaluations,
        } => match f_high {
            // The function is not defined at the first point, and is at the second.
            Some(value) => (low, right, high, value, evaluations),
            None => (left, right, low, f_low, evaluations),
        },
    };
    // `w` is the second best point and `v` the one `w` was before; `step` is the last step and
    // `before` the one before it, which a parabolic step has to be shorter than half of.
    let (mut w, mut f_w, mut v, mut f_v) = (x, f_x, x, f_x);
    let mut step = T::ZERO;
    let mut before = T::ZERO;

    let mut steps = 0;
    let mut converged = false;
    loop {
        let middle = left * T::HALF + right * T::HALF;
        let tol = (opts.tol_at(x) * T::from_f64(0.25))
            .max(T::TWO * T::EPSILON * magnitude(x))
            .max(T::MIN_POSITIVE);
        let two = T::TWO * tol;
        if (x - left).max(right - x) <= two {
            converged = true;
            break;
        }
        if steps >= opts.max_iter {
            break;
        }

        let mut parabolic = false;
        if magnitude(before) > tol {
            // The minimum of the parabola through `x`, `w` and `v` is `x + p/q`.
            let r = (x - w) * (f_x - f_v);
            let mut q = (x - v) * (f_x - f_w);
            let mut p = (x - v) * q - (x - w) * r;
            q = T::TWO * (q - r);
            if q > T::ZERO {
                p = -p;
            } else {
                q = -q;
            }
            let last_but_one = before;
            before = step;
            // It is taken when it falls inside the bracket and is shorter than half the step
            // before last, which is what makes the steps shrink whatever the parabolas do. A
            // parabola through values that are not finite passes neither test.
            if magnitude(p) < magnitude(T::HALF * q * last_but_one)
                && p > q * (left - x)
                && p < q * (right - x)
            {
                step = p / q;
                let u = x + step;
                // The function is not to be evaluated too close to an end of the bracket.
                if u - left < two || right - u < two {
                    step = if x < middle { tol } else { -tol };
                }
                parabolic = true;
            }
        }
        if !parabolic {
            // A golden section step into the larger part of the bracket.
            let far = if x < middle { right } else { left };
            before = far - x;
            step = section(x, far, short) - x;
        }

        // The function is not evaluated closer to `x` than `tol`.
        let u = if magnitude(step) >= tol {
            x + step
        } else if step > T::ZERO {
            x + tol
        } else {
            x - tol
        };
        let f_u = f(u);
        evaluations += 1;
        steps += 1;

        if not_worse(f_u, f_x) {
            if u < x {
                right = x;
            } else {
                left = x;
            }
            v = w;
            f_v = f_w;
            w = x;
            f_w = f_x;
            x = u;
            f_x = f_u;
        } else {
            if u < x {
                left = u;
            } else {
                right = u;
            }
            if not_worse(f_u, f_w) || w == x {
                v = w;
                f_v = f_w;
                w = u;
                f_w = f_u;
            } else if not_worse(f_u, f_v) || v == x || v == w {
                v = u;
                f_v = f_u;
            }
        }
    }

    Some(OneDimResult {
        x,
        value: f_x,
        evaluations,
        converged,
    })
}

// ---------------------------------------------------------------------------------------------
// Several variables, without derivatives
// ---------------------------------------------------------------------------------------------

/// The minimum of `f`, by the simplex of Nelder and Mead: `n + 1` points, the worst of them
/// reflected through the centroid of the others, the reflection expanded when it is the best
/// point yet and contracted when it is not good enough, the whole simplex shrunk towards its best
/// point when the contraction fails too.
///
/// It needs only values, which is what makes it the method for a function whose gradient cannot be
/// written down — a simulation, a schedule, a landscape with a kink in it. It is not a global
/// method, it gains nothing from a smooth function that a gradient would tell more about, and it
/// slows down badly as the number of variables grows: beyond ten or so it seldom reaches a minimum
/// within the default budget, and says so.
///
/// `step` is how far from `x0` the other corners of the starting simplex are; it should be the
/// distance over which `f` changes appreciably. Returns `None` when `x0` is empty or not finite,
/// when `step` is not a positive number, when a tolerance is negative or not a number, or when
/// `f` is not finite at `x0`.
///
/// # Panics
///
/// Panics if `x0` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::numerics::nelder_mead;
/// use a::math::matrix::Matrix;
///
/// // Rosenbrock's valley, whose minimum is (1, 1) with value 0.
/// let f = |x: &Matrix<f64>| {
///     let (a, b) = (x[(0, 0)], x[(1, 0)]);
///     (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
/// };
/// let r = nelder_mead(f, &Matrix::from_column_major(2, 1, vec![-1.2, 1.0]), 1.0).unwrap();
/// assert!((r.x[(0, 0)] - 1.0).abs() < 1e-6 && (r.x[(1, 0)] - 1.0).abs() < 1e-6);
/// assert!(r.value < 1e-12);
/// assert!(r.converged);
/// ```
#[track_caller]
pub fn nelder_mead<T: Real + Scalar>(
    f: impl FnMut(&Matrix<T>) -> T,
    x0: &Matrix<T>,
    step: T,
) -> Option<MinResult<T>> {
    nelder_mead_with(f, x0, step, &OptOptions::new())
}

/// [`nelder_mead`] under `opts`.
///
/// The moves are those of Nelder and Mead (1965) as Lagarias, Reeds, Wright and Wright (1998) set
/// them out: reflection, expansion to twice as far, contraction half way — outside the simplex
/// when the reflected point beats the worst corner, inside it when it does not — and a shrink by
/// half towards the best corner.
///
/// The simplex has closed when no corner is further than `xtol + rtol · max(‖x‖∞, 1)` from the
/// best one and no value differs from the best by more than `ftol + rtol · max(|f|, 1)`. A
/// simplex can close like that on a point that is no minimum — it can flatten itself against a
/// slope, and one that starts smaller than the tolerance has closed before it has moved — so
/// before saying `converged` the method polls: it evaluates `f` at the `2n` points a tolerance
/// away from the best corner along each axis. If none of them is better the point is returned
/// with `converged` true; if one is, they become the corners of a new simplex and the search
/// goes on. `max_iter` counts moves and polls alike; when it runs out the best corner is
/// returned with `converged` false. `gtol` is not read.
///
/// A corner at which `f` is not finite is the worst the simplex has, whatever else it holds.
///
/// # Panics
///
/// Panics if `x0` is not a column.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{OptOptions, nelder_mead_with};
/// use a::math::matrix::Matrix;
///
/// // A kink is no obstacle to a method that never asks for a slope.
/// let f = |x: &Matrix<f64>| (x[(0, 0)] - 1.0).abs() + 2.0 * (x[(1, 0)] + 0.5).abs();
/// let x0 = Matrix::from_column_major(2, 1, vec![3.0, 2.0]);
/// let r = nelder_mead_with(f, &x0, 1.0, &OptOptions::new()).unwrap();
/// assert!((r.x[(0, 0)] - 1.0).abs() < 1e-6 && (r.x[(1, 0)] + 0.5).abs() < 1e-6);
/// assert!(r.converged);
/// // A budget of ten moves is not enough, and the answer says so.
/// let r = nelder_mead_with(f, &x0, 1.0, &OptOptions::new().with_max_iter(10)).unwrap();
/// assert!(!r.converged && r.iterations == 10);
/// ```
#[track_caller]
pub fn nelder_mead_with<T: Real + Scalar>(
    mut f: impl FnMut(&Matrix<T>) -> T,
    x0: &Matrix<T>,
    step: T,
    opts: &OptOptions<T>,
) -> Option<MinResult<T>> {
    let n = column_length(x0, "the starting point");
    if n == 0
        || !ordered(T::ZERO, step)
        || !finite(step)
        || !all_finite(x0.as_slice())
        || !opts.sane()
    {
        return None;
    }
    let value0 = f(x0);
    if !finite(value0) {
        return None;
    }

    // The corners, best first, and the values at them.
    let mut corners: Vec<Matrix<T>> = Vec::with_capacity(n + 1);
    let mut values: Vec<T> = Vec::with_capacity(n + 1);
    corners.push(x0.clone());
    values.push(value0);
    for i in 0..n {
        let mut corner = x0.clone();
        corner[(i, 0)] = x0[(i, 0)] + step;
        values.push(f(&corner));
        corners.push(corner);
    }
    sort_simplex(&mut corners, &mut values);

    let mut centroid = vec![T::ZERO; n];
    let mut reflected = Matrix::<T>::zeros(n, 1);
    let mut moved = Matrix::<T>::zeros(n, 1);
    let mut iterations = 0;
    let mut converged = false;
    // Whether the simplex has just been rebuilt by a poll, and so has to move before it is
    // looked at again: it is as small as the tolerance by construction.
    let mut rebuilt = false;
    loop {
        if !rebuilt && simplex_closed(&corners, &values, opts) {
            if iterations >= opts.max_iter {
                break;
            }
            iterations += 1;
            if !poll(&mut f, &mut corners, &mut values, opts) {
                converged = true;
                break;
            }
            sort_simplex(&mut corners, &mut values);
            rebuilt = true;
            continue;
        }
        rebuilt = false;
        if iterations >= opts.max_iter {
            break;
        }
        iterations += 1;

        // The centroid of all the corners but the worst.
        centroid.fill(T::ZERO);
        for corner in &corners[..n] {
            for (sum, x) in centroid.iter_mut().zip(corner.as_slice()) {
                *sum += *x;
            }
        }
        let count = T::from_f64(n as f64);
        for sum in &mut centroid {
            *sum /= count;
        }
        // The point as far beyond the centroid as `t` times the worst corner is short of it.
        let along = |target: &mut Matrix<T>, worst: &Matrix<T>, t: T| {
            for ((out, c), w) in target
                .as_mut_slice()
                .iter_mut()
                .zip(&centroid)
                .zip(worst.as_slice())
            {
                *out = *c + t * (*c - *w);
            }
        };

        along(&mut reflected, &corners[n], T::ONE);
        let f_reflected = f(&reflected);
        let second_worst = values[n.saturating_sub(1)];
        if better(f_reflected, values[0]) {
            along(&mut moved, &corners[n], T::TWO);
            let f_expanded = f(&moved);
            if better(f_expanded, f_reflected) {
                replace_worst(&mut corners, &mut values, &moved, f_expanded);
            } else {
                replace_worst(&mut corners, &mut values, &reflected, f_reflected);
            }
        } else if better(f_reflected, second_worst) {
            replace_worst(&mut corners, &mut values, &reflected, f_reflected);
        } else {
            // A contraction: outside the simplex, towards a reflected point that at least beats
            // the worst corner, or inside it, towards the worst corner itself.
            let outside = better(f_reflected, values[n]);
            along(
                &mut moved,
                &corners[n],
                if outside { T::HALF } else { -T::HALF },
            );
            let f_moved = f(&moved);
            let accepted = if outside {
                not_worse(f_moved, f_reflected)
            } else {
                better(f_moved, values[n])
            };
            if accepted {
                replace_worst(&mut corners, &mut values, &moved, f_moved);
            } else {
                let (best, rest) = corners.split_at_mut(1);
                for (corner, value) in rest.iter_mut().zip(&mut values[1..]) {
                    for (x, b) in corner.as_mut_slice().iter_mut().zip(best[0].as_slice()) {
                        *x = *b + T::HALF * (*x - *b);
                    }
                    *value = f(corner);
                }
                sort_simplex(&mut corners, &mut values);
            }
        }
    }

    Some(MinResult {
        x: corners.swap_remove(0),
        value: values[0],
        iterations,
        converged,
    })
}

/// Sorts the corners of a simplex by their values, the best first; corners of equal value keep
/// their order.
fn sort_simplex<T: Real>(corners: &mut Vec<Matrix<T>>, values: &mut Vec<T>) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| {
        if better(values[*a], values[*b]) {
            core::cmp::Ordering::Less
        } else if better(values[*b], values[*a]) {
            core::cmp::Ordering::Greater
        } else {
            core::cmp::Ordering::Equal
        }
    });
    let mut slots: Vec<Option<Matrix<T>>> = corners.drain(..).map(Some).collect();
    corners.extend(order.iter().map(|&i| slots[i].take().unwrap()));
    let sorted: Vec<T> = order.iter().map(|&i| values[i]).collect();
    *values = sorted;
}

/// Puts `point` in the place of the worst corner of a sorted simplex and moves it up to where its
/// value belongs, behind the corners that are no worse.
fn replace_worst<T: Real>(
    corners: &mut [Matrix<T>],
    values: &mut [T],
    point: &Matrix<T>,
    value: T,
) {
    let mut at = values.len() - 1;
    corners[at].as_mut_slice().copy_from_slice(point.as_slice());
    values[at] = value;
    while at > 0 && better(values[at], values[at - 1]) {
        corners.swap(at, at - 1);
        values.swap(at, at - 1);
        at -= 1;
    }
}

/// Whether a sorted simplex is within the tolerances: no corner further from the best than the
/// tolerance on the point, no value further from the best than the tolerance on the value.
fn simplex_closed<T: Real>(corners: &[Matrix<T>], values: &[T], opts: &OptOptions<T>) -> bool {
    let best = corners[0].as_slice();
    let size = corners[1..].iter().fold(T::ZERO, |far, corner| {
        corner
            .as_slice()
            .iter()
            .zip(best)
            .fold(far, |far, (x, b)| far.max(magnitude(*x - *b)))
    });
    let spread = magnitude(values[values.len() - 1] - values[0]);
    size <= opts.tol_of(&corners[0])
        && spread <= opts.ftol + opts.rtol * magnitude(values[0]).max(T::ONE)
}

/// The poll that stands between a closed simplex and `converged`: the function at the `2n`
/// points a tolerance away from the best corner along each axis. Returns whether any of them is
/// better than the best corner, and in that case leaves the best corner and the better of each
/// pair as the simplex, unsorted.
fn poll<T: Real + Scalar>(
    f: &mut impl FnMut(&Matrix<T>) -> T,
    corners: &mut [Matrix<T>],
    values: &mut [T],
    opts: &OptOptions<T>,
) -> bool {
    let n = corners.len() - 1;
    let best_value = values[0];
    // The distance of the poll: the tolerance on the point, and no less than a few units in the
    // last place, a step shorter than which need not move a coordinate at all.
    let reach = opts
        .tol_of(&corners[0])
        .max(T::from_f64(8.0) * T::EPSILON * norm_inf(corners[0].as_slice()).max(T::ONE));
    let (best, rest) = corners.split_at_mut(1);
    let best = &best[0];
    let mut improved = false;
    let mut probe = best.clone();
    for (i, (corner, value)) in rest.iter_mut().zip(&mut values[1..]).enumerate() {
        probe[(i, 0)] = best[(i, 0)] + reach;
        let ahead = f(&probe);
        probe[(i, 0)] = best[(i, 0)] - reach;
        let behind = f(&probe);
        probe[(i, 0)] = best[(i, 0)];
        let (side, found) = if better(behind, ahead) {
            (-reach, behind)
        } else {
            (reach, ahead)
        };
        corner.as_mut_slice().copy_from_slice(best.as_slice());
        corner[(i, 0)] = best[(i, 0)] + side;
        *value = found;
        improved |= better(found, best_value);
    }
    debug_assert_eq!(rest.len(), n);
    improved
}

// ---------------------------------------------------------------------------------------------
// The line search
// ---------------------------------------------------------------------------------------------

/// What a line search asks of the function `φ(α) = f(x + α·d)` it searches along.
pub(super) trait Line<T> {
    /// `φ(α)` and `φ′(α)`, or `None` where either is not finite.
    fn at(&mut self, step: T) -> Option<(T, T)>;
    /// Remembers the point of the last call of [`Line::at`] as the best so far, in place of the
    /// one remembered before.
    fn keep(&mut self);
}

/// The conditions a line search works to.
pub(super) struct LineSearch<T> {
    /// `μ`: a step is acceptable when `φ(α) ≤ φ(0) + μ·α·φ′(0)`.
    pub(super) decrease: T,
    /// `η`: and when `|φ′(α)| ≤ η·|φ′(0)|`.
    pub(super) curvature: T,
    /// The relative width of a bracket that is narrow enough to stop at.
    pub(super) narrow: T,
    /// The longest step that may be tried.
    pub(super) longest: T,
    /// The most evaluations of the function.
    pub(super) evaluations: usize,
}

/// What a line search found.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Found {
    /// A step that satisfies both conditions, the strong conditions of Wolfe.
    Wolfe,
    /// A step that keeps the decrease condition, and the least value among those tried that do;
    /// the curvature condition could not be met.
    Decrease,
    /// No step that keeps even the decrease condition.
    Nothing,
}

/// The step of a line search.
pub(super) struct LineOutcome<T> {
    /// What kind of step it is.
    pub(super) found: Found,
    /// `φ(α)`.
    pub(super) value: T,
    /// Whether the point of the step is the one last kept by [`Line::keep`], rather than the one
    /// last evaluated by [`Line::at`].
    pub(super) kept: bool,
}

/// The minimiser of the cubic that takes the values `fa`, `fb` and the slopes `ga`, `gb` at `a`
/// and `b` (Nocedal and Wright, *Numerical Optimization*, 3.59), or `None` when the cubic has no
/// minimum or the arithmetic fails.
fn cubic_minimiser<T: Real>(a: T, fa: T, ga: T, b: T, fb: T, gb: T) -> Option<T> {
    let theta = T::from_f64(3.0) * (fa - fb) / (b - a) + ga + gb;
    // Scaled, so that the square of a large slope does not overflow.
    let scale = magnitude(theta).max(magnitude(ga)).max(magnitude(gb));
    if !ordered(T::ZERO, scale) || !finite(scale) {
        return None;
    }
    let under = (theta / scale) * (theta / scale) - (ga / scale) * (gb / scale);
    if !ordered(T::ZERO, under) {
        return None;
    }
    let mut gamma = scale * root(under);
    if b < a {
        gamma = -gamma;
    }
    let p = (gamma - ga) + theta;
    let q = ((gamma - ga) + gamma) + gb;
    let at = a + (p / q) * (b - a);
    finite(at).then_some(at)
}

/// The interval of a line search: the best step so far, with the value and the slope there, and
/// the other end, which brackets a minimum with it once `bracketed` is true.
struct Interval<T> {
    best: T,
    f_best: T,
    g_best: T,
    other: T,
    f_other: T,
    g_other: T,
    bracketed: bool,
}

impl<T: Real> Interval<T> {
    /// Takes the trial `step`, with the value `f` and the slope `g` there, into the interval, and
    /// returns the next trial: the safeguarded interpolation of section 4 of Moré and Thuente,
    /// its four cases told apart by whether the value went up, whether the slope changed sign,
    /// and whether it grew. `least` and `most` are the limits of a trial outside a bracket.
    /// Returns as well whether the trial became the best step.
    fn advance(&mut self, step: T, f: T, g: T, least: T, most: T) -> (T, bool) {
        let opposite =
            (g < T::ZERO && self.g_best > T::ZERO) || (g > T::ZERO && self.g_best < T::ZERO);
        let ahead = step > self.best;
        let secant = step + (g / (g - self.g_best)) * (self.best - step);
        let next = if f > self.f_best {
            // A higher value: the minimum is bracketed. The cubic through both ends if it is
            // nearer the best step than the parabola, their mean otherwise.
            self.bracketed = true;
            let width = step - self.best;
            let parabola = self.best
                + (self.g_best / ((self.f_best - f) / width + self.g_best)) * T::HALF * width;
            match cubic_minimiser(self.best, self.f_best, self.g_best, step, f, g) {
                Some(cubic) if magnitude(cubic - self.best) < magnitude(parabola - self.best) => {
                    cubic
                }
                Some(cubic) => cubic + (parabola - cubic) * T::HALF,
                None => parabola,
            }
        } else if opposite {
            // A lower value and a slope of the other sign: bracketed again. Whichever of the
            // cubic and the secant step is further from the trial.
            self.bracketed = true;
            match cubic_minimiser(step, f, g, self.best, self.f_best, self.g_best) {
                Some(cubic) if magnitude(cubic - step) > magnitude(secant - step) => cubic,
                _ => secant,
            }
        } else if magnitude(g) < magnitude(self.g_best) {
            // A lower value, a slope of the same sign and smaller: the cubic may have no minimum
            // beyond the trial, and then it stands for the limit on that side.
            let cubic = cubic_minimiser(step, f, g, self.best, self.f_best, self.g_best)
                .filter(|c| (*c > step) == ahead && *c != step)
                .unwrap_or(if ahead { most } else { least });
            if self.bracketed {
                let nearer = if magnitude(cubic - step) < magnitude(secant - step) {
                    cubic
                } else {
                    secant
                };
                // No further than the fraction 0.66 of the way to the other end.
                let limit = step + T::from_f64(BRACKET_SHRINK) * (self.other - step);
                if ahead {
                    nearer.min(limit)
                } else {
                    nearer.max(limit)
                }
            } else {
                let further = if magnitude(cubic - step) > magnitude(secant - step) {
                    cubic
                } else {
                    secant
                };
                further.max(least).min(most)
            }
        } else if self.bracketed {
            // A lower value and a slope that has not shrunk: the cubic through the trial and the
            // other end.
            cubic_minimiser(step, f, g, self.other, self.f_other, self.g_other)
                .unwrap_or(step + (self.other - step) * T::HALF)
        } else if ahead {
            most
        } else {
            least
        };

        let improved = !ordered(self.f_best, f);
        if improved {
            if opposite {
                self.other = self.best;
                self.f_other = self.f_best;
                self.g_other = self.g_best;
            }
            self.best = step;
            self.f_best = f;
            self.g_best = g;
        } else {
            self.other = step;
            self.f_other = f;
            self.g_other = g;
        }
        // An interpolation that came to nothing is replaced by the plainest safe step.
        let next = if finite(next) {
            next
        } else if self.bracketed {
            self.best + (self.other - self.best) * T::HALF
        } else {
            most
        };
        (next, improved)
    }
}

/// The line search of Moré and Thuente (ACM Transactions on Mathematical Software 20, 1994): a
/// step along `line` that satisfies the strong conditions of Wolfe,
/// `φ(α) ≤ φ(0) + μ·α·φ′(0)` and `|φ′(α)| ≤ η·|φ′(0)|`, from the value `value0` and the slope
/// `slope0` at zero and the first trial `first`.
///
/// The search keeps an interval that it either lengthens, by a factor between 1.1 and 4 of the
/// distance from the best step, or — once a minimum is bracketed — narrows, each trial the
/// minimiser of a cubic or of a parabola through the ends, replaced by a bisection when two
/// trials have not shrunk the bracket to 0.66 of itself. Until a trial both keeps the decrease
/// condition and has a slope that is not negative, the interval is updated on
/// `ψ(α) = φ(α) − φ(0) − μ·α·φ′(0)` in place of `φ`, which is what guarantees that the interval
/// holds a point satisfying both conditions.
///
/// Where the function is not finite the longest step allowed comes in to half way between there
/// and the best step. When the conditions cannot both be met within the evaluations allowed —
/// the step is cut short by `longest`, the bracket has become too narrow to interpolate in, the
/// function has no such point — the result is the step of least value that keeps the decrease
/// condition alone, and when there is none, [`Found::Nothing`].
pub(super) fn more_thuente<T: Real>(
    line: &mut impl Line<T>,
    value0: T,
    slope0: T,
    first: T,
    search: &LineSearch<T>,
) -> LineOutcome<T> {
    let nothing = LineOutcome {
        found: Found::Nothing,
        value: value0,
        kept: false,
    };
    // Not a direction that leads downhill, or not a step: there is nothing to search along.
    if !ordered(slope0, T::ZERO)
        || !ordered(T::ZERO, first)
        || !ordered(T::ZERO, search.longest)
        || !finite(value0)
    {
        return nothing;
    }
    let test = search.decrease * slope0;
    let mut longest = search.longest;
    let mut interval = Interval {
        best: T::ZERO,
        f_best: value0,
        g_best: slope0,
        other: T::ZERO,
        f_other: value0,
        g_other: slope0,
        bracketed: false,
    };
    let mut first_stage = true;
    let mut step = first.min(longest);
    let mut least = T::ZERO;
    let mut most = step + step * T::from_f64(EXTRAPOLATE_MOST);
    let mut width = longest;
    let mut width_before = width * T::TWO;
    // The least value of a step that keeps the decrease condition; the point is in the line's
    // keeping.
    let mut kept: Option<T> = None;

    for _ in 0..search.evaluations {
        let Some((f, g)) = line.at(step) else {
            // Inside a bracket whose ends the function is defined at there is no telling where
            // it stops being so; outside one, the far limit comes in.
            if interval.bracketed {
                break;
            }
            longest = interval.best + (step - interval.best) * T::HALF;
            if !ordered(interval.best, longest) {
                break;
            }
            most = most.min(longest);
            least = least.min(longest);
            step = longest;
            continue;
        };
        let bound = value0 + step * test;
        let decreased = f <= bound;
        if decreased && magnitude(g) <= search.curvature * -slope0 {
            return LineOutcome {
                found: Found::Wolfe,
                value: f,
                kept: false,
            };
        }
        if decreased && kept.is_none_or(|least_value| f < least_value) {
            line.keep();
            kept = Some(f);
        }
        if first_stage && decreased && g >= T::ZERO {
            first_stage = false;
        }
        // Still falling as steeply at the longest step allowed: there is no further to go.
        if step >= longest && decreased && g <= test {
            break;
        }

        // In the first stage a trial that lowers the value without keeping the decrease
        // condition is judged on ψ, whose values and slopes are those of φ less the line of the
        // decrease condition.
        if first_stage && f <= interval.f_best && f > bound {
            let line_at = |at: T| at * test;
            interval.f_best -= line_at(interval.best);
            interval.f_other -= line_at(interval.other);
            interval.g_best -= test;
            interval.g_other -= test;
            let (next, _) = interval.advance(step, f - line_at(step), g - test, least, most);
            interval.f_best += line_at(interval.best);
            interval.f_other += line_at(interval.other);
            interval.g_best += test;
            interval.g_other += test;
            step = next;
        } else {
            let (next, _) = interval.advance(step, f, g, least, most);
            step = next;
        }

        if interval.bracketed {
            let apart = magnitude(interval.other - interval.best);
            if apart >= T::from_f64(BRACKET_SHRINK) * width_before {
                step = interval.best + (interval.other - interval.best) * T::HALF;
            }
            width_before = width;
            width = apart;
            least = interval.best.min(interval.other);
            most = interval.best.max(interval.other);
        } else {
            least = step + T::from_f64(EXTRAPOLATE_LEAST) * (step - interval.best);
            most = step + T::from_f64(EXTRAPOLATE_MOST) * (step - interval.best);
        }
        step = step.max(T::ZERO).min(longest);
        // A bracket too narrow to place a trial inside, or to be worth one, ends the search.
        if interval.bracketed
            && (step <= least || step >= most || most - least <= search.narrow * most)
        {
            break;
        }
        if !ordered(T::ZERO, step) {
            break;
        }
    }

    match kept {
        Some(value) => LineOutcome {
            found: Found::Decrease,
            value,
            kept: true,
        },
        None => nothing,
    }
}

/// The points and gradients a line search of the gradient methods moves between: those of the
/// last trial and those of the trial last kept.
struct Workspace<T> {
    point: Matrix<T>,
    gradient: Matrix<T>,
    kept_point: Matrix<T>,
    kept_gradient: Matrix<T>,
}

impl<T: Real + Scalar> Workspace<T> {
    fn new(n: usize) -> Self {
        Self {
            point: Matrix::zeros(n, 1),
            gradient: Matrix::zeros(n, 1),
            kept_point: Matrix::zeros(n, 1),
            kept_gradient: Matrix::zeros(n, 1),
        }
    }

    /// The point and the gradient of the step a search came back with.
    fn found(&self, outcome: &LineOutcome<T>) -> (&Matrix<T>, &Matrix<T>) {
        if outcome.kept {
            (&self.kept_point, &self.kept_gradient)
        } else {
            (&self.point, &self.gradient)
        }
    }
}

/// The gradient `g` returned for a point of `n` variables.
///
/// # Panics
///
/// Panics if it is not a column of `n` elements.
pub(super) fn checked_gradient<T>(gradient: Matrix<T>, n: usize) -> Matrix<T> {
    assert!(
        gradient.rows() == n && gradient.cols() == 1,
        "the gradient is not a column of as many elements as the point: it is {} × {} for {} variables",
        gradient.rows(),
        gradient.cols(),
        n
    );
    gradient
}

/// The ray `x + α·d` of a gradient method, for the line search.
struct Ray<'a, T, F, G> {
    f: &'a mut F,
    g: &'a mut G,
    x: &'a [T],
    d: &'a [T],
    work: &'a mut Workspace<T>,
}

impl<T, F, G> Line<T> for Ray<'_, T, F, G>
where
    T: Real + Scalar,
    F: FnMut(&Matrix<T>) -> T,
    G: FnMut(&Matrix<T>) -> Matrix<T>,
{
    fn at(&mut self, step: T) -> Option<(T, T)> {
        for ((p, x), d) in self
            .work
            .point
            .as_mut_slice()
            .iter_mut()
            .zip(self.x)
            .zip(self.d)
        {
            *p = *x + step * *d;
        }
        let value = (self.f)(&self.work.point);
        if !finite(value) {
            return None;
        }
        let gradient = checked_gradient((self.g)(&self.work.point), self.x.len());
        if !all_finite(gradient.as_slice()) {
            return None;
        }
        let slope = dot(gradient.as_slice(), self.d);
        self.work.gradient = gradient;
        Some((value, slope))
    }

    fn keep(&mut self) {
        self.work
            .kept_point
            .as_mut_slice()
            .copy_from_slice(self.work.point.as_slice());
        self.work
            .kept_gradient
            .as_mut_slice()
            .copy_from_slice(self.work.gradient.as_slice());
    }
}

/// The search of the unbounded gradient methods, with the curvature condition `curvature`.
fn unbounded_search<T: Real>(curvature: f64) -> LineSearch<T> {
    LineSearch {
        decrease: T::from_f64(DECREASE),
        curvature: T::from_f64(curvature),
        narrow: T::from_f64(LINE_SEARCH_NARROW),
        longest: T::from_f64(LONGEST_STEP),
        evaluations: LINE_SEARCH_EVALUATIONS,
    }
}

/// Whether the step a line search of the gradient methods came back with is progress from the
/// point of value `value` and gradient `gradient`: a step to a lower value or, where the change
/// of `f` along it is lost in the rounding of `f`, a step that satisfies the strong conditions
/// of Wolfe and lowers the largest element of the gradient, which is what the stopping test
/// reads. At the limit of the arithmetic the decrease condition holds in rounding for a step that
/// does not move the point at all; a search that has nothing better than that to show has found
/// nothing.
fn progress<T: Real + Scalar>(
    outcome: &LineOutcome<T>,
    value: T,
    work: &Workspace<T>,
    gradient: &[T],
) -> bool {
    match outcome.found {
        Found::Nothing => false,
        found => {
            ordered(outcome.value, value)
                || (found == Found::Wolfe && {
                    let (_, new_gradient) = work.found(outcome);
                    ordered(norm_inf(new_gradient.as_slice()), norm_inf(gradient))
                })
        }
    }
}

/// The first trial step down the gradient `g`, where nothing is known of the scale of the
/// function: the step that moves the point by one, and no longer than the gradient itself.
fn first_step<T: Real>(g: &[T]) -> T {
    let length = norm2(g);
    if length > T::ONE {
        T::ONE / length
    } else {
        T::ONE
    }
}

// ---------------------------------------------------------------------------------------------
// Several variables, with gradients
// ---------------------------------------------------------------------------------------------

/// The minimum of `f`, by the conjugate gradient method of Polak, Ribière and Polyak: a
/// steepest descent step, then each new direction the old one corrected by how much the gradient
/// has turned, and a restart down the gradient every `n` steps and whenever a direction stops
/// pointing downhill.
///
/// It needs the gradient and keeps only the last one, so where `n` is large enough that BFGS's
/// `n × n` matrix is unwelcome this is the method to use; [`lbfgsb`](super::lbfgsb()) is the
/// other, and usually takes fewer evaluations. On a quadratic it would arrive in `n` steps if
/// each search along a direction were exact; the searches here are not, and it takes somewhat
/// more, and on a badly conditioned function a great many more.
///
/// Returns `None` when `x0` is empty or not finite, when a tolerance is negative or not a number,
/// or when `f` or ∇`f` is not finite at `x0`.
///
/// # Panics
///
/// Panics if `x0` is not a column, or if `g` returns anything but a column of as many elements.
///
/// # Examples
///
/// ```
/// use a::math::numerics::conjugate_gradient;
/// use a::math::matrix::Matrix;
///
/// // Rosenbrock's valley, from the usual start.
/// let f = |x: &Matrix<f64>| {
///     let (a, b) = (x[(0, 0)], x[(1, 0)]);
///     (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
/// };
/// let g = |x: &Matrix<f64>| {
///     let (a, b) = (x[(0, 0)], x[(1, 0)]);
///     Matrix::from_column_major(2, 1, vec![
///         -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
///         200.0 * (b - a * a),
///     ])
/// };
/// let r = conjugate_gradient(f, g, &Matrix::from_column_major(2, 1, vec![-1.2, 1.0])).unwrap();
/// assert!((r.x[(0, 0)] - 1.0).abs() < 1e-7 && (r.x[(1, 0)] - 1.0).abs() < 1e-7);
/// assert!(r.converged);
/// ```
#[track_caller]
pub fn conjugate_gradient<T: Real + Scalar>(
    f: impl FnMut(&Matrix<T>) -> T,
    g: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
) -> Option<MinResult<T>> {
    conjugate_gradient_with(f, g, x0, &OptOptions::new())
}

/// [`conjugate_gradient`] under `opts`.
///
/// The method stops, and `converged` is true, when `‖∇f‖∞ ≤ gtol` at the point it is at; nothing
/// else makes `converged` true. It stops without that when `max_iter` steps have been taken, or
/// when no step along the gradient itself lowers the function any more, which is how the limit of
/// the arithmetic shows. `xtol`, `rtol` and `ftol` are not read.
///
/// A step is one to a lower value of `f` or, where the change of `f` along it is lost in the
/// rounding of `f`, one that satisfies the strong conditions of Wolfe and lowers `‖∇f‖∞`. At the
/// limit of the arithmetic the decrease condition holds in rounding for a step that does not move
/// the point at all, and that is not a step: a direction along which the search finds none is
/// tried again down the gradient, and when that finds none either the method stops, after at most
/// two more searches of at most 40 evaluations each.
///
/// Each step is found by the line search of Moré and Thuente under the strong conditions of
/// Wolfe with `μ = 10⁻⁴` and `η = 0.1`; the new direction is `−∇f + β·d` with the `β` of Polak
/// and Ribière held at zero or above. A search that could not meet the curvature condition, as
/// on a function that falls for ever, is followed by a restart.
///
/// # Panics
///
/// Panics if `x0` is not a column, or if `g` returns anything but a column of as many elements.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{OptOptions, conjugate_gradient_with};
/// use a::math::matrix::Matrix;
///
/// // A quadratic bowl ten times as steep one way as the other.
/// let f = |x: &Matrix<f64>| x[(0, 0)].powi(2) + 10.0 * x[(1, 0)].powi(2);
/// let g = |x: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![2.0 * x[(0, 0)], 20.0 * x[(1, 0)]]);
/// let x0 = Matrix::from_column_major(2, 1, vec![3.0, -1.0]);
/// let r = conjugate_gradient_with(f, g, &x0, &OptOptions::new().with_gtol(1e-10)).unwrap();
/// assert!(r.converged && r.x[(0, 0)].abs() < 1e-10 && r.x[(1, 0)].abs() < 1e-10);
/// // One step is not enough, and the answer says so.
/// let r = conjugate_gradient_with(f, g, &x0, &OptOptions::new().with_max_iter(1)).unwrap();
/// assert!(!r.converged && r.iterations == 1 && r.value < f(&x0));
/// ```
#[track_caller]
pub fn conjugate_gradient_with<T: Real + Scalar>(
    mut f: impl FnMut(&Matrix<T>) -> T,
    mut g: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
) -> Option<MinResult<T>> {
    let n = column_length(x0, "the starting point");
    if n == 0 || !all_finite(x0.as_slice()) || !opts.sane() {
        return None;
    }
    let mut x = x0.clone();
    let mut value = f(&x);
    if !finite(value) {
        return None;
    }
    let mut gradient = checked_gradient(g(&x), n);
    if !all_finite(gradient.as_slice()) {
        return None;
    }

    let search = unbounded_search::<T>(CURVATURE_CONJUGATE);
    let mut work = Workspace::new(n);
    let mut direction: Vec<T> = gradient.as_slice().iter().map(|v| -*v).collect();
    // The value before the last step, from which the first trial of the next search is guessed.
    let mut value_before: Option<T> = None;
    let mut since_restart = 0;
    let mut iterations = 0;
    let mut converged;
    loop {
        converged = norm_inf(gradient.as_slice()) <= opts.gtol;
        if converged || iterations >= opts.max_iter {
            break;
        }
        let mut slope = dot(&direction, gradient.as_slice());
        let mut steepest = since_restart == 0;
        if !ordered(slope, T::ZERO) {
            steepest = true;
        }
        let mut outcome = None;
        for attempt in 0..2 {
            if steepest {
                for (d, v) in direction.iter_mut().zip(gradient.as_slice()) {
                    *d = -*v;
                }
                slope = -dot(gradient.as_slice(), gradient.as_slice());
                since_restart = 0;
            }
            // The step that would repeat the last decrease, were the function a parabola along
            // the direction (Nocedal and Wright, 3.60); down the gradient with nothing behind
            // it, the step of unit length.
            let guess = match value_before {
                Some(before) if !steepest || attempt == 0 => {
                    let guess = T::TWO * (value - before) / slope;
                    Some(guess).filter(|guess| *guess > T::ZERO && finite(*guess))
                }
                _ => None,
            };
            let first = guess.unwrap_or_else(|| first_step(&direction));
            let mut ray = Ray {
                f: &mut f,
                g: &mut g,
                x: x.as_slice(),
                d: &direction,
                work: &mut work,
            };
            let found = more_thuente(&mut ray, value, slope, first, &search);
            if progress(&found, value, &work, gradient.as_slice()) {
                outcome = Some(found);
                break;
            }
            // A direction that led nowhere is tried again as the gradient itself, which cannot
            // fail to lead down while there is anything to gain; and the gradient, when the
            // search along it began from the guess, from the step of unit length. The guess can
            // be wild after a step that gained much where the gradient is now small (10¹⁴ on a
            // quadratic of condition 10³), and the search then gives up long before it has come
            // down to the scale of the function.
            if steepest && guess.is_none() {
                break;
            }
            steepest = true;
        }
        let Some(outcome) = outcome else {
            break;
        };

        let (point, new_gradient) = work.found(&outcome);
        x.as_mut_slice().copy_from_slice(point.as_slice());
        // Polak and Ribière's β measures how much the gradient has turned since the last step;
        // a negative one means the new gradient undoes the old direction, so it is held at zero,
        // which restarts the method down the gradient.
        let square = dot(gradient.as_slice(), gradient.as_slice());
        let turned: T = new_gradient
            .as_slice()
            .iter()
            .zip(gradient.as_slice())
            .map(|(new, old)| *new * (*new - *old))
            .sum();
        gradient
            .as_mut_slice()
            .copy_from_slice(new_gradient.as_slice());
        value_before = Some(value);
        value = outcome.value;
        iterations += 1;
        since_restart += 1;

        let beta = if square > T::ZERO && finite(turned / square) {
            (turned / square).max(T::ZERO)
        } else {
            T::ZERO
        };
        if since_restart >= n || outcome.found != Found::Wolfe || beta == T::ZERO {
            since_restart = 0;
            for (d, v) in direction.iter_mut().zip(gradient.as_slice()) {
                *d = -*v;
            }
        } else {
            for (d, v) in direction.iter_mut().zip(gradient.as_slice()) {
                *d = beta * *d - *v;
            }
        }
    }

    Some(MinResult {
        x,
        value,
        iterations,
        converged,
    })
}

/// The minimum of `f`, by BFGS: a quasi-Newton method that builds up the inverse Hessian from the
/// gradients it has seen, one rank-two correction a step.
///
/// Each step is along `−H∇f` and the inverse Hessian `H` is corrected to be exactly right about
/// the change of the gradient along the step just taken, so after `n` independent steps it knows
/// a quadratic as well as if it had been given one. It keeps an `n × n` matrix and spends `n²`
/// operations a step on it; where that is too much use [`lbfgsb`](super::lbfgsb()), which keeps
/// a few vectors in its place, or [`conjugate_gradient`].
///
/// Returns `None` when `x0` is empty or not finite, when a tolerance is negative or not a number,
/// or when `f` or ∇`f` is not finite at `x0`.
///
/// # Panics
///
/// Panics if `x0` is not a column, or if `g` returns anything but a column of as many elements.
///
/// # Examples
///
/// ```
/// use a::math::numerics::bfgs;
/// use a::math::matrix::Matrix;
///
/// // The quadratic x₀² + 2x₁² − 2x₀ − 8x₁ + 9, which is (x₀ − 1)² + 2(x₁ − 2)²: its minimum
/// // is (1, 2), with value 0.
/// let f = |x: &Matrix<f64>| {
///     let (a, b) = (x[(0, 0)], x[(1, 0)]);
///     a * a + 2.0 * b * b - 2.0 * a - 8.0 * b + 9.0
/// };
/// let g = |x: &Matrix<f64>| {
///     let (a, b) = (x[(0, 0)], x[(1, 0)]);
///     Matrix::from_column_major(2, 1, vec![2.0 * a - 2.0, 4.0 * b - 8.0])
/// };
/// let r = bfgs(f, g, &Matrix::from_column_major(2, 1, vec![3.0, -4.0])).unwrap();
/// assert!((r.x[(0, 0)] - 1.0).abs() < 1e-7 && (r.x[(1, 0)] - 2.0).abs() < 1e-7);
/// assert!(r.value.abs() < 1e-14);
/// assert!(r.converged);
/// ```
#[track_caller]
pub fn bfgs<T: Real + Scalar>(
    f: impl FnMut(&Matrix<T>) -> T,
    g: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
) -> Option<MinResult<T>> {
    bfgs_with(f, g, x0, &OptOptions::new())
}

/// [`bfgs`] under `opts`.
///
/// The method stops, and `converged` is true, when `‖∇f‖∞ ≤ gtol` at the point it is at; nothing
/// else makes `converged` true. It stops without that when `max_iter` steps have been taken, or
/// when no step along the gradient itself lowers the function any more, which is how the limit of
/// the arithmetic shows. `xtol`, `rtol` and `ftol` are not read. A step is what it is for
/// [`conjugate_gradient_with`]: one to a lower value, or a step of the strong conditions of Wolfe
/// that lowers `‖∇f‖∞` where the change of `f` is lost in its rounding, and never one that does
/// not move the point.
///
/// Each step is found by the line search of Moré and Thuente under the strong conditions of
/// Wolfe with `μ = 10⁻⁴` and `η = 0.9`, the unit step tried first. The inverse Hessian starts as
/// the identity, unscaled as in SciPy, and takes the correction
/// `H ← (I − ρsyᵀ) H (I − ρysᵀ) + ρssᵀ`, `ρ = 1/sᵀy`, in its expanded form. A step whose `sᵀy` is
/// not safely positive is passed over, which keeps the matrix positive definite; a direction
/// along which the search finds no step discards the matrix for the gradient, and when the
/// gradient leads to none either the method stops.
///
/// # Panics
///
/// Panics if `x0` is not a column, or if `g` returns anything but a column of as many elements.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{OptOptions, bfgs_with};
/// use a::math::matrix::Matrix;
///
/// let f = |x: &Matrix<f64>| (x[(0, 0)] - 1.0).powi(4) + (x[(1, 0)] + 2.0).powi(2);
/// let g = |x: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![
///     4.0 * (x[(0, 0)] - 1.0).powi(3),
///     2.0 * (x[(1, 0)] + 2.0),
/// ]);
/// let x0 = Matrix::from_column_major(2, 1, vec![0.0, 0.0]);
/// let loose = bfgs_with(f, g, &x0, &OptOptions::new().with_gtol(1e-3)).unwrap();
/// let tight = bfgs_with(f, g, &x0, &OptOptions::new().with_gtol(1e-9)).unwrap();
/// assert!(loose.converged && tight.converged && loose.iterations < tight.iterations);
/// assert!(g(&tight.x).as_slice().iter().all(|v| v.abs() <= 1e-9));
/// ```
#[track_caller]
pub fn bfgs_with<T: Real + Scalar>(
    mut f: impl FnMut(&Matrix<T>) -> T,
    mut g: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
) -> Option<MinResult<T>> {
    let n = column_length(x0, "the starting point");
    if n == 0 || !all_finite(x0.as_slice()) || !opts.sane() {
        return None;
    }
    let mut x = x0.clone();
    let mut value = f(&x);
    if !finite(value) {
        return None;
    }
    let mut gradient = checked_gradient(g(&x), n);
    if !all_finite(gradient.as_slice()) {
        return None;
    }

    let search = unbounded_search::<T>(CURVATURE_QUASI_NEWTON);
    let mut work = Workspace::new(n);
    // The inverse Hessian, row after row, and whether any step has gone into it: until one has
    // it is the identity, and the direction the gradient.
    let mut inverse = vec![T::ZERO; n * n];
    let mut learnt = false;
    let mut direction = vec![T::ZERO; n];
    let mut s = vec![T::ZERO; n];
    let mut y = vec![T::ZERO; n];
    let mut hy = vec![T::ZERO; n];
    let mut iterations = 0;
    let mut converged;
    loop {
        converged = norm_inf(gradient.as_slice()) <= opts.gtol;
        if converged || iterations >= opts.max_iter {
            break;
        }

        let mut outcome = None;
        for _ in 0..2 {
            if learnt {
                for (d, row) in direction.iter_mut().zip(inverse.chunks_exact(n)) {
                    *d = -dot(row, gradient.as_slice());
                }
            } else {
                for (d, v) in direction.iter_mut().zip(gradient.as_slice()) {
                    *d = -*v;
                }
            }
            let slope = dot(&direction, gradient.as_slice());
            if !ordered(slope, T::ZERO) && learnt {
                // A direction that is not downhill means the curvature gathered so far is not
                // to be trusted: forget it and go down the gradient.
                learnt = false;
                continue;
            }
            let first = if learnt {
                T::ONE
            } else {
                first_step(&direction)
            };
            let mut ray = Ray {
                f: &mut f,
                g: &mut g,
                x: x.as_slice(),
                d: &direction,
                work: &mut work,
            };
            let found = more_thuente(&mut ray, value, slope, first, &search);
            if progress(&found, value, &work, gradient.as_slice()) {
                outcome = Some(found);
                break;
            }
            if !learnt {
                break;
            }
            learnt = false;
        }
        let Some(outcome) = outcome else {
            break;
        };

        let (point, new_gradient) = work.found(&outcome);
        for (((s, y), (new_x, old_x)), (new_g, old_g)) in s
            .iter_mut()
            .zip(&mut y)
            .zip(point.as_slice().iter().zip(x.as_slice()))
            .zip(new_gradient.as_slice().iter().zip(gradient.as_slice()))
        {
            *s = *new_x - *old_x;
            *y = *new_g - *old_g;
        }
        x.as_mut_slice().copy_from_slice(point.as_slice());
        gradient
            .as_mut_slice()
            .copy_from_slice(new_gradient.as_slice());
        value = outcome.value;
        iterations += 1;

        // A step with sᵀy ≤ 0 says nothing about curvature that a positive definite H can agree
        // with, and one with sᵀy lost in the rounding of its two factors says nothing at all.
        let sy = dot(&s, &y);
        if !ordered(T::EPSILON * norm2(&s) * norm2(&y), sy) {
            continue;
        }
        if !learnt {
            // The identity, as SciPy starts. Scaled by yᵀs/yᵀy, the curvature along the first
            // step (Nocedal and Wright, 6.20), it took 3 to 9 times the iterations on badly
            // conditioned quadratics: a first step down the gradient sees the largest
            // curvatures, and the scale then shortens every step along the smallest.
            inverse.fill(T::ZERO);
            for i in 0..n {
                inverse[i * n + i] = T::ONE;
            }
            learnt = true;
        }
        // H ← H − ρ(s·uᵀ + u·sᵀ) + ρ(1 + ρ·yᵀu)·s·sᵀ with u = H·y: the correction multiplied
        // out, n² operations where the product of the three matrices would take n³.
        for (u, row) in hy.iter_mut().zip(inverse.chunks_exact(n)) {
            *u = dot(row, &y);
        }
        let rho = T::ONE / sy;
        let factor = rho * (T::ONE + rho * dot(&y, &hy));
        for (i, row) in inverse.chunks_exact_mut(n).enumerate() {
            for (j, h) in row.iter_mut().enumerate() {
                *h += factor * s[i] * s[j] - rho * (s[i] * hy[j] + hy[i] * s[j]);
            }
        }
    }

    Some(MinResult {
        x,
        value,
        iterations,
        converged,
    })
}

// ---------------------------------------------------------------------------------------------
// Least squares
// ---------------------------------------------------------------------------------------------

/// The least squares of `residual`, by Gauss and Newton: each step solves the linearised problem
/// `J·δ = −r` in the least squares sense, by [`qr`](crate::math::matrix::Matrix::qr), and takes
/// it whole.
///
/// Where the residual is nearly linear in the parameters it converges quadratically and often in a
/// handful of steps; where it is not, a whole step can overshoot and the method has nothing to
/// fall back on, so it stops without improving. Use [`levenberg_marquardt`] unless the residual is
/// well behaved near its zero.
///
/// `residual` maps the `n` parameters to a column of `m ≥ n` residuals and `jacobian` to the
/// `m × n` matrix of their derivatives. Returns `None` when `x0` is empty or not finite, when
/// there are fewer residuals than parameters, when a tolerance is negative or not a number, when
/// the residual or the Jacobian is not finite at `x0`, or when the columns of the Jacobian are
/// dependent at `x0` and it is neither stationary nor exact to rounding, where the linearised
/// problem has no one solution. Where the columns become dependent at a point the steps have led
/// to, or at a start that is exact to rounding by the residual test, that point is returned
/// instead, with `converged` false: no step can be made from it, and only a step can confirm
/// the residual test.
///
/// # Panics
///
/// Panics if `x0` or the residual is not a column, or if the Jacobian has not as many rows as the
/// residual and as many columns as there are parameters.
///
/// # Examples
///
/// ```
/// use a::math::numerics::gauss_newton;
/// use a::math::matrix::Matrix;
///
/// // Fit y = a + b·t to four points that lie exactly on the line y = 1 + 2t: the linear case,
/// // which Gauss and Newton settles in one step.
/// let t = [0.0, 1.0, 2.0, 3.0];
/// let y = [1.0, 3.0, 5.0, 7.0];
/// let r = move |p: &Matrix<f64>| {
///     let (a, b) = (p[(0, 0)], p[(1, 0)]);
///     Matrix::from_column_major(4, 1, t.iter().zip(&y).map(|(t, yi)| a + b * t - yi).collect())
/// };
/// let j = move |_: &Matrix<f64>| Matrix::from_rows(&[&[1.0, 0.0], &[1.0, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
/// let fit = gauss_newton(r, j, &Matrix::from_column_major(2, 1, vec![0.0, 0.0])).unwrap();
/// assert!((fit.x[(0, 0)] - 1.0).abs() < 1e-12 && (fit.x[(1, 0)] - 2.0).abs() < 1e-12);
/// assert!(fit.cost < 1e-24);
/// assert!(fit.converged);
/// ```
#[track_caller]
pub fn gauss_newton<T: Real + Scalar>(
    residual: impl FnMut(&Matrix<T>) -> Matrix<T>,
    jacobian: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
) -> Option<FitResult<T>> {
    gauss_newton_with(residual, jacobian, x0, &OptOptions::new())
}

/// [`gauss_newton`] under `opts`.
///
/// The tests that make `converged` true are those of [`levenberg_marquardt_with`], every step
/// here being the Gauss and Newton step they are made on. A step that does not lower the cost is
/// not taken, and ends the method with `converged` false unless one of the tests holds.
///
/// # Panics
///
/// Panics if `x0` or the residual is not a column, or if the Jacobian has not as many rows as the
/// residual and as many columns as there are parameters.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{OptOptions, gauss_newton_with};
/// use a::math::matrix::Matrix;
///
/// // The square root of two, as the least squares of x² − 2.
/// let r = |p: &Matrix<f64>| Matrix::from_column_major(1, 1, vec![p[(0, 0)] * p[(0, 0)] - 2.0]);
/// let j = |p: &Matrix<f64>| Matrix::from_column_major(1, 1, vec![2.0 * p[(0, 0)]]);
/// let x0 = Matrix::from_column_major(1, 1, vec![1.0]);
/// let fit = gauss_newton_with(r, j, &x0, &OptOptions::new()).unwrap();
/// assert!(fit.converged && (fit.x[(0, 0)] - core::f64::consts::SQRT_2).abs() < 1e-15);
/// let fit = gauss_newton_with(r, j, &x0, &OptOptions::new().with_max_iter(2)).unwrap();
/// assert!(!fit.converged && fit.iterations == 2);
/// ```
#[track_caller]
pub fn gauss_newton_with<T: Real + Scalar>(
    residual: impl FnMut(&Matrix<T>) -> Matrix<T>,
    jacobian: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
) -> Option<FitResult<T>> {
    fit(residual, jacobian, x0, opts, false)
}

/// The least squares of `residual`, by Levenberg and Marquardt: the Gauss and Newton step held
/// inside a region in which the linearised problem can be trusted, the region widened when a step
/// pays off as the linearisation promised and narrowed when it does not.
///
/// A narrow region makes the step a short one down the gradient, which always improves the cost,
/// and a wide one makes it the Gauss and Newton step, which converges quadratically — the method
/// moves between the two and is the one to reach for when the starting parameters are far out.
/// The region is measured in the parameters scaled by the lengths of the columns of the Jacobian,
/// so the method does not depend on the units of the parameters or of the data.
///
/// `residual` maps the `n` parameters to a column of `m ≥ n` residuals and `jacobian` to the
/// `m × n` matrix of their derivatives. Returns `None` when `x0` is empty or not finite, when
/// there are fewer residuals than parameters, when a tolerance is negative or not a number, or
/// when the residual or the Jacobian is not finite at `x0`.
///
/// # Panics
///
/// Panics if `x0` or the residual is not a column, or if the Jacobian has not as many rows as the
/// residual and as many columns as there are parameters.
///
/// # Examples
///
/// ```
/// use a::math::numerics::levenberg_marquardt;
/// use a::math::matrix::Matrix;
///
/// // Fit y = a·e^(−λt) to five points generated from a = 2, λ = 1.5.
/// let t: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];
/// let y: Vec<f64> = t.iter().map(|t| 2.0 * (-1.5 * *t).exp()).collect();
/// let r = move |p: &Matrix<f64>| {
///     let (a, lambda) = (p[(0, 0)], p[(1, 0)]);
///     Matrix::from_column_major(5, 1, t.iter().zip(&y).map(|(t, yi)| a * (-lambda * *t).exp() - yi).collect())
/// };
/// let j = move |p: &Matrix<f64>| {
///     let (a, lambda) = (p[(0, 0)], p[(1, 0)]);
///     Matrix::from_fn(5, 2, |i, k| match k {
///         0 => (-lambda * t[i]).exp(),
///         _ => -a * t[i] * (-lambda * t[i]).exp(),
///     })
/// };
/// let fit = levenberg_marquardt(r, j, &Matrix::from_column_major(2, 1, vec![1.0, 1.0])).unwrap();
/// assert!((fit.x[(0, 0)] - 2.0).abs() < 1e-8 && (fit.x[(1, 0)] - 1.5).abs() < 1e-8);
/// assert!(fit.cost < 1e-20);
/// assert!(fit.converged);
/// ```
#[track_caller]
pub fn levenberg_marquardt<T: Real + Scalar>(
    residual: impl FnMut(&Matrix<T>) -> Matrix<T>,
    jacobian: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
) -> Option<FitResult<T>> {
    levenberg_marquardt_with(residual, jacobian, x0, &OptOptions::new())
}

/// [`levenberg_marquardt`] under `opts`.
///
/// This is the method as Moré set it out (*The Levenberg-Marquardt algorithm: implementation and
/// theory*, 1978), the algorithm of MINPACK's `lmder`. With `D` the diagonal matrix of the
/// largest length each column of the Jacobian `J` has had, the step `p` minimises
/// `‖J·p + r‖² + λ‖D·p‖²`: it is the least squares solution of the stacked system
/// `[J; √λ·D]·p = [−r; 0]`, found by [`qr`](crate::math::matrix::Matrix::qr) — one decomposition
/// of `J` a Jacobian, then one of the `2n × n` matrix `[R; √λ·D]` a trial of `λ` — and never by
/// the normal equations, which would square the condition number. `λ` is zero when the Gauss and
/// Newton step lies inside the region `‖D·p‖ ≤ Δ`, and otherwise the root of `‖D·p(λ)‖ = Δ`,
/// found within a tenth of `Δ` by Moré's safeguarded Newton iteration. The ratio `ρ` of the
/// reduction of the cost to the reduction the linearised problem predicted decides the rest: the
/// step is taken when `ρ ≥ 10⁻⁴`; `Δ` shrinks, by a factor between a tenth and a half, when
/// `ρ ≤ ¼`, and becomes twice the length of the step when `ρ ≥ ¾`.
///
/// `converged` is true when one of four tests holds, and only then:
///
/// * the gradient test, at any point: the cosine of the angle between the residual and every
///   column of the Jacobian is at most `gtol`, `|Jₖᵀr| ≤ gtol·‖Jₖ‖·‖r‖`, which no change of units
///   alters; a residual of zero passes it;
/// * the residual test, at any point: `‖r‖ ≤ 2(1 + √n)·ε·‖D_J·x‖`, `D_J` the lengths the
///   columns of the Jacobian have there. Rounding each parameter to a float moves the model by
///   up to `ε/2·|xₖ|·‖Jₖ‖`, so the exact minimiser of a fit that is exact, rounded and evaluated,
///   has a residual of about that size, and a point whose residual is no larger cannot be told
///   from it — where the model is linear over the rounding of the parameters. The point takes
///   one more step, which solves the linearised problem and can only refine it, and the method
///   stops after that step, with the better of the two points, if the Jacobian holds over it
///   (below). Where no such step can be made — the budget is spent, the damping fails, the
///   columns of a Gauss and Newton Jacobian are dependent — the point is returned with
///   `converged` false. Exact data with a parameter whose answer is zero converge by this test:
///   the step of that parameter is rounding about zero, never short against its own size;
/// * the step test, on a step that solves the linearised problem: the step of every parameter is
///   at most `xtol + rtol·|xⱼ|`, and the Jacobian holds over it (below), so that the linearised
///   problem wants each parameter where it is to within the tolerance on that parameter;
/// * the reduction test, on such a step too: the reduction of the cost, and the reduction the
///   linearised problem predicted, both at most `ftol + rtol·cost`. This says that the cost
///   cannot be lowered by more than that fraction of itself, not that the gradient is small: the
///   cosines of the point the step was made from are then at most about `√(rtol + ftol/cost)`,
///   `2.4·10⁻⁴` with the defaults, where the gradient test asks for `gtol`.
///
/// A step solves the linearised problem when `λ` is zero, or when `λ` is at its floor (below)
/// and the step leaves the linearised residual `r + J·p` with no cosine against a column above
/// `ε^¼`; with `D` the current lengths of the columns the floor leaves at most half of that, so
/// a step that leaves more was held back by a length its column no longer has. A step that is
/// short because the region is small, or because it does not solve the linearised problem,
/// proves nothing, and is not taken for convergence: when the region has shrunk below the
/// resolution of the point the method stops with `converged` false. The step is judged
/// parameter by parameter because in one norm over all of them a parameter whose column is long,
/// or once was, hides a step that changes another one wholly. The step and reduction tests have
/// a floor of one unit of rounding, `ε·|xⱼ|` and `ε·cost`, so that tolerances of zero ask for all
/// the arithmetic can give. `max_iter` counts trial steps.
///
/// The Jacobian holds over a step when, evaluated at the end of the step, it is the one the step
/// was made with to within a quarter: the change of each column weighted by its parameter, as
/// the residual test weighs them, `‖(|xₖ|·‖ΔJₖ‖)ₖ‖ ≤ ¼·‖D_J·x‖`. The residual test and the step
/// test read the linearised problem as the problem, which it is not over a step across which the
/// Jacobian moves further. Beside a pole of the model — `a·t/(b + t)` with `b` near `−tₖ` — the
/// Gauss and Newton step moves `b` by about its distance from the pole: a step short against
/// `b`, and within an ulp of the pole the residual is huge and below the resolution of the
/// residual test all the same, as rounding `b` moves the model as much; yet the step doubles the
/// distance, quarters the cost and cuts the column of `b` to a quarter, and the next one does the
/// same. Both tests took such points for fits, as MINPACK does; with the check such a point is
/// not taken for a fit: the method goes on, and either leaves the pole or, from a start within an
/// ulp of it, stops with `converged` false. It costs one evaluation of the Jacobian, at the point
/// the method stops at.
///
/// Where the columns of the Jacobian are dependent there is no Gauss and Newton step; the step is
/// then the damped one with `λ` no smaller than `√ε` in the scaled problem, which is close to
/// the shortest solution of the linearised problem.
///
/// A point at which a column of the Jacobian that has had a length has none passes none of the
/// tests, unless the residual is zero or exact to rounding: its parameter has gone where the
/// residual no longer depends on it — the rate of `a·e^(−λt)` run off to where the exponential
/// underflows — and the cost is flat there because the model has saturated, not because it
/// fits. The method goes on and, unable to make progress, stops with `converged` false. MINPACK
/// reports success at such points.
///
/// # Panics
///
/// Panics if `x0` or the residual is not a column, or if the Jacobian has not as many rows as the
/// residual and as many columns as there are parameters.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{OptOptions, levenberg_marquardt_with};
/// use a::math::matrix::Matrix;
///
/// // Rosenbrock's function as two residuals, 10(x₁ − x₀²) and 1 − x₀.
/// let r = |p: &Matrix<f64>| Matrix::from_column_major(2, 1, vec![
///     10.0 * (p[(1, 0)] - p[(0, 0)] * p[(0, 0)]),
///     1.0 - p[(0, 0)],
/// ]);
/// let j = |p: &Matrix<f64>| Matrix::from_rows(&[[-20.0 * p[(0, 0)], 10.0], [-1.0, 0.0]]);
/// let x0 = Matrix::from_column_major(2, 1, vec![-1.2, 1.0]);
/// let fit = levenberg_marquardt_with(r, j, &x0, &OptOptions::new()).unwrap();
/// assert!(fit.converged && fit.cost < 1e-24);
/// assert!((fit.x[(0, 0)] - 1.0).abs() < 1e-12 && (fit.x[(1, 0)] - 1.0).abs() < 1e-12);
/// let cut = levenberg_marquardt_with(r, j, &x0, &OptOptions::new().with_max_iter(3)).unwrap();
/// assert!(!cut.converged && cut.iterations == 3 && cut.cost > fit.cost);
/// ```
#[track_caller]
pub fn levenberg_marquardt_with<T: Real + Scalar>(
    residual: impl FnMut(&Matrix<T>) -> Matrix<T>,
    jacobian: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
) -> Option<FitResult<T>> {
    fit(residual, jacobian, x0, opts, true)
}

/// The solution `w` of `Rᵀ·w = u` for the upper triangular `r`, or `None` when a diagonal element
/// of `r` is zero.
fn solve_transposed<T: Real + Scalar>(r: &Matrix<T>, u: &[T]) -> Option<Vec<T>> {
    let n = u.len();
    let mut w = vec![T::ZERO; n];
    for i in 0..n {
        let column = r.column(i);
        if column[i] == T::ZERO {
            return None;
        }
        w[i] = (u[i] - dot(&column[..i], &w[..i])) / column[i];
    }
    Some(w)
}

/// The step of Levenberg and Marquardt for the trust region `‖D·p‖ ≤ delta`: Moré's algorithm
/// (5.5), on the triangular factor `r` of the Jacobian and `qtr = Qᵀ·residual`.
///
/// `gauss_newton` is the Gauss and Newton step, where the Jacobian is of full rank; `damping` is
/// the `λ` the last step ended with, which the iteration starts from. Returns the step, its `λ`,
/// and whether the region cut the step short — whether `λ` is what holds `‖D·p‖` at `delta`,
/// rather than zero or its floor. `None` when the arithmetic fails.
fn damped_step<T: Real + Scalar>(
    r: &Matrix<T>,
    qtr: &[T],
    scale: &[T],
    gauss_newton: Option<&[T]>,
    delta: T,
    damping: T,
) -> Option<(Vec<T>, T, bool)> {
    let n = scale.len();
    let fit = T::from_f64(REGION_FIT);
    let scaled_length = |p: &[T]| {
        let scaled: Vec<T> = p.iter().zip(scale).map(|(p, d)| *p * *d).collect();
        norm2(&scaled)
    };
    // D²·p/‖D·p‖, the right-hand side of the equation for the slope of ‖D·p(λ)‖.
    let slope_side = |p: &[T], length: T| -> Vec<T> {
        p.iter()
            .zip(scale)
            .map(|(p, d)| *d * (*d * *p / length))
            .collect()
    };

    // The lower end of the interval that holds λ: the Newton step from zero, which lies below
    // the root because ‖D·p(λ)‖ is convex in λ; zero where there is no Gauss and Newton step.
    let mut lower = T::ZERO;
    let mut excess = T::INFINITY;
    if let Some(p) = gauss_newton {
        let length = scaled_length(p);
        excess = length - delta;
        if excess <= fit * delta {
            return Some((p.to_vec(), T::ZERO, false));
        }
        if let Some(w) = solve_transposed(r, &slope_side(p, length)) {
            let w = norm2(&w);
            let bound = excess / delta / w / w;
            if bound > T::ZERO && finite(bound) {
                lower = bound;
            }
        }
    }
    // Where the Jacobian has dependent columns λ has a floor, below which the stacked system is
    // as ill conditioned as the Jacobian is rank deficient and the step is rounding and no more.
    let floor = if gauss_newton.is_some() {
        T::ZERO
    } else {
        root(T::EPSILON)
    };
    // The upper end: ‖D⁻¹·Jᵀr‖/Δ, since ‖D·p(λ)‖ ≤ ‖D⁻¹·Jᵀr‖/λ. Jᵀr is Rᵀ·Qᵀr.
    let gradient: Vec<T> = (0..n)
        .map(|k| dot(&r.column(k)[..=k], &qtr[..=k]) / scale[k])
        .collect();
    let steepest = norm2(&gradient);
    let mut upper = steepest / delta;
    if !ordered(T::ZERO, upper) {
        upper = T::MIN_POSITIVE / delta.min(fit);
    }
    if !finite(upper) {
        return None;
    }

    let mut lambda = damping.max(lower).min(upper);
    if lambda == T::ZERO {
        lambda = T::from_f64(1e-3) * upper;
    }
    let mut stacked = Matrix::<T>::zeros(2 * n, n);
    let mut side = Matrix::<T>::zeros(2 * n, 1);
    for (b, q) in side.as_mut_slice().iter_mut().zip(qtr) {
        *b = -*q;
    }
    let mut found: Option<(Vec<T>, T, T)> = None;
    for trial in 1..=DAMPING_TRIALS {
        lambda = lambda.max(floor).max(T::MIN_POSITIVE);
        // The least squares of [R; √λ·D]·p = [−Qᵀr; 0].
        let weight = root(lambda);
        for k in 0..n {
            let column = stacked.column_mut(k);
            column[..=k].copy_from_slice(&r.column(k)[..=k]);
            column[n + k] = weight * scale[k];
        }
        let qr = stacked.qr();
        let p = qr.solve(&side)?.into_vec();
        if !all_finite(&p) {
            return None;
        }
        let length = scaled_length(&p);
        let before = excess;
        excess = length - delta;
        let at_floor = lambda <= floor;
        let done = magnitude(excess) <= fit * delta
            || (lower == T::ZERO && excess <= before && before < T::ZERO)
            || (at_floor && excess < T::ZERO)
            || trial == DAMPING_TRIALS;
        if done {
            found = Some((p, lambda, excess));
            break;
        }
        // The Newton step on 1/Δ − 1/‖D·p(λ)‖, which is nearly straight where ‖D·p(λ)‖ is not.
        let w = solve_transposed(&qr.r(), &slope_side(&p, length))?;
        let w = norm2(&w);
        let correction = excess / delta / w / w;
        if excess > T::ZERO {
            lower = lower.max(lambda);
        } else {
            upper = upper.min(lambda);
        }
        lambda = (lambda + correction).max(lower);
        if !finite(lambda) {
            return None;
        }
    }
    let (p, lambda, _) = found?;
    // The region holds the step whenever λ is above its floor, whether or not the iteration
    // brought the step all the way to the edge: a step that ten trials of λ left well inside
    // the region (a column of the Jacobian of 10⁻⁴⁶ beside one of order one does that) is still
    // a damped one, and its being short proves nothing.
    let cut = ordered(floor, lambda);
    Some((p, lambda, cut))
}

/// The Jacobian `j` returned for `m` residuals and `n` parameters.
///
/// # Panics
///
/// Panics if it is not `m × n`.
#[track_caller]
fn checked_jacobian<T>(j: Matrix<T>, m: usize, n: usize) -> Matrix<T> {
    assert!(
        j.shape() == (m, n),
        "the Jacobian is not of as many rows as the residual and as many columns as the point: it is {} × {} for {} residuals and {} parameters",
        j.rows(),
        j.cols(),
        m,
        n
    );
    j
}

/// How far the Jacobian `end` at the end of a step has moved from `start`, the one the step was
/// made with at the point `x`, as a fraction of its size: the change of each column weighted by
/// its parameter, as the residual test weighs the columns,
/// `‖(|xₖ|·‖endₖ − startₖ‖)ₖ‖ / ‖(|xₖ|·‖startₖ‖)ₖ‖`, and infinite where that cannot be measured.
/// A parameter of zero contributes nothing to the model through its column, and its column's
/// change is not counted.
fn jacobian_change<T: Real + Scalar>(x: &[T], start: &Matrix<T>, end: &Matrix<T>) -> T {
    let mut moved = Vec::with_capacity(x.len());
    let mut size = Vec::with_capacity(x.len());
    let mut difference = vec![T::ZERO; start.rows()];
    for (k, x) in x.iter().enumerate() {
        for ((d, a), b) in difference
            .iter_mut()
            .zip(start.column(k))
            .zip(end.column(k))
        {
            *d = *b - *a;
        }
        moved.push(magnitude(*x) * norm2(&difference));
        size.push(magnitude(*x) * norm2(start.column(k)));
    }
    let (moved, size) = (norm2(&moved), norm2(&size));
    if !finite(moved) || !finite(size) {
        T::INFINITY
    } else if size == T::ZERO {
        if moved == T::ZERO {
            T::ZERO
        } else {
            T::INFINITY
        }
    } else {
        moved / size
    }
}

/// Both least squares methods: [`gauss_newton`] with `damped` false, [`levenberg_marquardt`] with
/// it true. The region that holds the step is what the one has and the other does without.
#[track_caller]
fn fit<T: Real + Scalar>(
    mut residual: impl FnMut(&Matrix<T>) -> Matrix<T>,
    mut jacobian: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
    damped: bool,
) -> Option<FitResult<T>> {
    let n = column_length(x0, "the starting point");
    if n == 0 || !all_finite(x0.as_slice()) || !opts.sane() {
        return None;
    }
    let mut x = x0.clone();
    let mut r = residual(&x);
    let m = column_length(&r, "the residual");
    if m < n || !all_finite(r.as_slice()) {
        return None;
    }
    let mut r_norm = norm2(r.as_slice());
    if !finite(r_norm * r_norm) {
        return None;
    }

    let eps = T::EPSILON;
    let tenth = T::from_f64(0.1);
    let quarter = T::from_f64(0.25);
    // The scaling D, the radius Δ of the region and the damping λ of the last step.
    let mut scale = vec![T::ZERO; n];
    let mut delta = T::ZERO;
    let mut lambda = T::ZERO;
    let mut trial = Matrix::<T>::zeros(n, 1);
    let mut started = false;
    // Which columns of the Jacobian have ever had a length.
    let mut seen = vec![false; n];
    let mut taken = 0usize;
    let mut iterations = 0;
    let mut converged = false;
    // The Jacobian at the point a step was taken to, where a test made on the step evaluated it.
    let mut next_jacobian: Option<Matrix<T>> = None;
    'jacobians: loop {
        let j = match next_jacobian.take() {
            Some(j) => j,
            None => checked_jacobian(jacobian(&x), m, n),
        };
        if !all_finite(j.as_slice()) {
            if started {
                break;
            }
            return None;
        }
        let lengths: Vec<T> = (0..n).map(|k| norm2(j.column(k))).collect();
        // A column that has had a length and has none now belongs to a parameter that has gone
        // where the residual no longer depends on it — a rate of decay run off to where the
        // exponential underflows. The cost is flat there because the model has saturated, which
        // every test would take for a fit, and none may.
        for (seen, length) in seen.iter_mut().zip(&lengths) {
            *seen |= *length > T::ZERO;
        }
        let vanished = seen
            .iter()
            .zip(&lengths)
            .any(|(seen, length)| *seen && *length == T::ZERO);

        // The gradient test: the largest cosine between the residual and a column.
        let cosine = if r_norm == T::ZERO {
            T::ZERO
        } else {
            (0..n).fold(T::ZERO, |largest, k| {
                if lengths[k] > T::ZERO {
                    // Divided one at a time, so that neither product overflows.
                    let along = dot(j.column(k), r.as_slice()) / lengths[k] / r_norm;
                    largest.max(magnitude(along))
                } else {
                    largest
                }
            })
        };
        if r_norm == T::ZERO || (!vanished && cosine <= opts.gtol) {
            converged = true;
            break;
        }
        // The residual test: rounding each parameter to a float moves the model by up to
        // ε/2·|xₖ|·‖Jₖ‖, and the residual of the exact minimiser of a fit that is exact, rounded
        // and evaluated, is a few of those; once the residual is no larger, no point can be told
        // from that minimiser by its residual. The lengths are those the columns have now: a
        // length a column had once and lost says nothing of the model here. The argument holds
        // only where the model is linear over the rounding of the parameters, which within an
        // ulp of a pole it is not: there the residual is huge and below this resolution all the
        // same. So the point is a fit only once one more step from it, which solves the
        // linearised problem and can only refine the point, has been made and the Jacobian at
        // its end is found to be the one the step was made with; where no step can be made, the
        // point is returned unconfirmed.
        let resolution = T::from_f64(2.0 * (1.0 + (n as f64).sqrt()))
            * eps
            * scaled_norm(x.as_slice(), &lengths);
        let exact = finite(resolution) && r_norm <= resolution;
        if iterations >= opts.max_iter {
            break;
        }

        // D: the length of each column, and never less than it has been (Moré, section 6).
        for (d, length) in scale.iter_mut().zip(&lengths) {
            *d = if started {
                d.max(*length)
            } else if *length > T::ZERO {
                *length
            } else {
                T::ONE
            };
        }
        if !started {
            let length = scaled_norm(x.as_slice(), &scale);
            delta = T::from_f64(REGION_FACTOR) * if length > T::ZERO { length } else { T::ONE };
            started = true;
        }

        // J = Q·R. The Gauss and Newton step is the least squares solution of J·p = −r, which
        // exists where no column of R has shrunk to the rounding of the column of J it is the
        // rest of — where none depends on those before it.
        let qr = j.qr();
        let upper = qr.r();
        let rank_tolerance = eps * T::from_f64(m.max(n) as f64);
        let full_rank = (0..n).all(|k| magnitude(upper[(k, k)]) > rank_tolerance * lengths[k]);
        let gauss_newton: Option<Vec<T>> = if full_rank {
            qr.solve(&-&r)
                .map(Matrix::into_vec)
                .filter(|p| all_finite(p))
        } else {
            None
        };
        let qtr: Vec<T> = if damped {
            let q = qr.q();
            (0..n).map(|k| dot(q.column(k), r.as_slice())).collect()
        } else {
            Vec::new()
        };

        loop {
            let (p, cut) = if damped {
                let Some((p, damping, cut)) =
                    damped_step(&upper, &qtr, &scale, gauss_newton.as_deref(), delta, lambda)
                else {
                    break 'jacobians;
                };
                lambda = damping;
                (p, cut)
            } else {
                match gauss_newton.clone() {
                    Some(p) => (p, false),
                    // Dependent columns: the linearised problem has no one solution, and there is
                    // no step to take. A point that steps have led to, or that the residual test
                    // passed, is returned as the best point reached, unconfirmed; at a start that
                    // is neither there is nothing to return.
                    None if exact || taken > 0 => break 'jacobians,
                    None => return None,
                }
            };
            let scaled: Vec<T> = p.iter().zip(&scale).map(|(p, d)| *p * *d).collect();
            let p_norm = norm2(&scaled);
            if damped && taken == 0 {
                delta = delta.min(p_norm);
            }

            for ((t, x), p) in trial.as_mut_slice().iter_mut().zip(x.as_slice()).zip(&p) {
                *t = *x + *p;
            }
            let r_trial = residual(&trial);
            assert!(
                r_trial.rows() == m && r_trial.cols() == 1,
                "the residual changed its shape: it was a column of {} and is {} × {}",
                m,
                r_trial.rows(),
                r_trial.cols()
            );
            let trial_norm = if all_finite(r_trial.as_slice()) {
                norm2(r_trial.as_slice())
            } else {
                T::INFINITY
            };
            iterations += 1;

            // The reduction of the sum of squares, and the one the linearised problem predicted,
            // ‖J·p‖² + 2λ‖D·p‖², both as fractions of the sum of squares; and its slope along p.
            let actual = if tenth * trial_norm < r_norm {
                T::ONE - (trial_norm / r_norm) * (trial_norm / r_norm)
            } else {
                -T::ONE
            };
            let jp: Vec<T> = (0..m)
                .map(|i| j.row(i).zip(&p).map(|(a, p)| *a * *p).sum::<T>())
                .collect();
            // Whether the step solves the linearised problem, so that its length and its gain
            // speak of the point: always where λ is zero; at the floor of λ only when it leaves
            // the linearised residual r + J·p with no cosine against a column above ε^¼. With D
            // the current lengths of the columns the floor λ = √ε leaves at most ε^¼/2 (the
            // damped step leaves λσ/(σ² + λ) ≤ √λ/2 of each singular direction), so a step that
            // leaves more has been held back by a length its column no longer has.
            let solves = lambda == T::ZERO || {
                let linearised: Vec<T> =
                    r.as_slice().iter().zip(&jp).map(|(r, q)| *r + *q).collect();
                let most = root(root(eps));
                (0..n).all(|k| {
                    lengths[k] == T::ZERO
                        || magnitude(dot(j.column(k), &linearised)) / lengths[k] / r_norm <= most
                })
            };
            let linear = norm2(&jp) / r_norm;
            let penalty = root(lambda) * p_norm / r_norm;
            let predicted = linear * linear + T::TWO * penalty * penalty;
            let slope = -(linear * linear + penalty * penalty);
            let ratio = if predicted > T::ZERO {
                actual / predicted
            } else {
                T::ZERO
            };

            let accepted = if damped {
                // The region: narrowed, by the factor that the parabola through the two values
                // and the slope puts the minimum at, when the step paid a quarter or less of
                // what was predicted; twice the step when it paid three quarters or more.
                if ratio <= quarter {
                    let mut factor = if actual >= T::ZERO {
                        T::HALF
                    } else {
                        T::HALF * slope / (slope + T::HALF * actual)
                    };
                    if tenth * trial_norm >= r_norm || !(ordered(tenth, factor) || factor == tenth)
                    {
                        factor = tenth;
                    }
                    delta = factor * delta.min(p_norm / tenth);
                    lambda /= factor;
                } else if lambda == T::ZERO || ratio >= T::from_f64(0.75) {
                    delta = p_norm / T::HALF;
                    lambda *= T::HALF;
                }
                ratio >= T::from_f64(ACCEPT)
            } else {
                trial_norm < r_norm
            };
            let cost_before = r_norm;
            if accepted {
                core::mem::swap(&mut x, &mut trial);
                r = r_trial;
                r_norm = trial_norm;
                taken += 1;
            }
            // The point the step was made from and the one it led to, whether it was taken or not.
            let (from, to) = if accepted { (&trial, &x) } else { (&x, &trial) };
            // A step the region did not cut short and that solves the linearised problem: the
            // only kind a test may be made on.
            let full = !cut && solves;
            // The Jacobian at the end of the step, once a test made on the step has asked for it,
            // and whether it is the one the step was made with, to within JACOBIAN_HOLDS. The
            // residual test and the step test read the linearised problem as the problem; over a
            // step across which the Jacobian moves further they would read it wrong — beside a
            // pole of the model a step short against the parameter doubles its distance from the
            // pole, quarters the cost and cuts its column to a quarter, and the next does the same.
            let mut end: Option<(Matrix<T>, bool)> = None;
            let mut holds = |end: &mut Option<(Matrix<T>, bool)>| -> bool {
                if let Some((_, holds)) = end {
                    return *holds;
                }
                let at_end = checked_jacobian(jacobian(to), m, n);
                let holds =
                    jacobian_change(from.as_slice(), &j, &at_end) <= T::from_f64(JACOBIAN_HOLDS);
                *end = Some((at_end, holds));
                holds
            };
            // The step that refines a point the residual test passed, taken or refused: the
            // point kept is the better of the two, and passed the test or improved on one that
            // did.
            if exact && full && holds(&mut end) {
                converged = true;
                break 'jacobians;
            }

            // The step and reduction tests, on a full step: its being short, or its gain small,
            // then says that the point is where the linearised problem wants it. The step is
            // judged parameter by parameter, each against its own size: measured in one norm over
            // all of them, a parameter whose column is long, or has once been, would hide a step
            // that changes another one wholly. A short step is believed where the Jacobian holds
            // over it; a gain as small as the reduction test asks for is not one a step beside a
            // pole makes.
            let x_norm = scaled_norm(x.as_slice(), &scale);
            if full && !vanished {
                let short = p
                    .iter()
                    .zip(x.as_slice())
                    .all(|(p, x)| magnitude(*p) <= opts.xtol + (opts.rtol + eps) * magnitude(*x));
                let half = T::HALF * cost_before * cost_before;
                let reduction_tolerance = if half > T::ZERO {
                    opts.rtol + opts.ftol / half + eps
                } else {
                    T::INFINITY
                };
                if (short && holds(&mut end))
                    || (magnitude(actual) <= reduction_tolerance
                        && predicted <= reduction_tolerance
                        && ratio <= T::TWO)
                {
                    converged = true;
                    break 'jacobians;
                }
            }
            if iterations >= opts.max_iter {
                break 'jacobians;
            }
            if !damped && !accepted {
                // The whole step overshot, and there is no shorter one to fall back on.
                break 'jacobians;
            }
            // A region below the resolution of the point leaves nothing to try.
            if !ordered(eps * x_norm, delta) || !ordered(T::MIN_POSITIVE, delta) {
                break 'jacobians;
            }
            if accepted {
                // The Jacobian a test evaluated at the new point is the one to go on with.
                next_jacobian = end.map(|(j, _)| j);
                break;
            }
        }
    }

    Some(FitResult {
        x,
        cost: T::HALF * r_norm * r_norm,
        iterations,
        converged,
    })
}
