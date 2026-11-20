//! The minima of functions: of one variable, of several, and of a sum of squares.
//!
//! | Method | Needs | Rate | Misses the minimum |
//! | --- | --- | --- | --- |
//! | [`golden_section`] | a bracket, unimodality | linear, 0.618 of the interval a step | no |
//! | [`minimize`] | a bracket, unimodality | superlinear in practice | no |
//! | [`nelder_mead`] | `f`, a starting point and a first step | linear | yes |
//! | [`conjugate_gradient`] | `f` and ∇`f` | superlinear | yes |
//! | [`bfgs`] | `f` and ∇`f` | superlinear | yes |
//! | [`gauss_newton`] | the residual and its Jacobian | quadratic | yes |
//! | [`levenberg_marquardt`] | the residual and its Jacobian | between 1 and 2 | yes |
//!
//! A bracketed method is told where the minimum is and cannot miss it, but it only works on a
//! function that falls and then rises once inside the bracket; an open method is told only where
//! to start, so it converges to whichever stationary point the starting point leads it to. None of
//! them returns an answer it has not verified: a method that runs out of budget reports the best
//! point it reached with [`MinResult::converged`] false, and one whose iterates stop being finite
//! returns `None`. The tolerances and the budget are the same [`OptOptions`] for all of them.
//!
//! The gradient methods stop on the infinity norm of the gradient, [`OptOptions::gtol`], which is
//! why the default is `√ε` rather than `ε`: at a minimum the function is flat, so the gradient
//! cannot be trusted further than the square root of the precision the values are computed in.
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

/// The fraction of the predicted decrease a step has to deliver to be accepted: the sufficient
/// decrease, or Armijo, condition of the line search.
const WOLFE_C1: f64 = 1e-4;
/// How much of the starting slope a step may keep and still be accepted: the curvature condition,
/// which is what stops the search taking a step that has already passed the valley floor.
const WOLFE_C2: f64 = 0.9;
/// How much the search multiplies its trial step while it has yet to bracket the minimum.
const WOLFE_WIDEN: f64 = 2.1;
/// How many times the search evaluates the function before it gives up on the direction.
const LINE_SEARCH_MAX: usize = 40;
/// The fraction of the widest step the search will accept, and the smallest it will try.
const LINE_SEARCH_MAX_STEP: f64 = 1e10;
const LINE_SEARCH_MIN_STEP: f64 = 1e-16;

/// The damping the Levenberg and Marquardt corrector starts with, and the factors it is scaled by
/// as steps succeed and fail.
const MARQUARDT_LAMBDA: f64 = 1e-3;
const MARQUARDT_UP: f64 = 10.0;
const MARQUARDT_DOWN: f64 = 0.1;
/// How large the damping may grow before the corrector is given up on.
const MARQUARDT_LAMBDA_MAX: f64 = 1e20;

/// Whether `x` is finite, named through [`Float`], as [`Scalar`] names the same test and both are
/// in scope wherever a matrix element is a float.
fn finite<T: Real>(x: T) -> bool {
    Float::is_finite(x)
}

/// The magnitude of `x`, named through [`Float`] for the same reason.
fn magnitude<T: Real>(x: T) -> T {
    Float::abs(x)
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

/// The total ordering of values that [`better`] compares them by, for the sorts that rank a
/// simplex.
fn compare<T: Real>(a: T, b: T) -> core::cmp::Ordering {
    if better(a, b) {
        core::cmp::Ordering::Less
    } else if better(b, a) {
        core::cmp::Ordering::Greater
    } else {
        core::cmp::Ordering::Equal
    }
}

/// The largest of the absolute values of a column, the norm every stopping test on a gradient or a
/// step uses.
fn norm_inf<T: Real>(v: &Matrix<T>) -> T {
    v.as_slice()
        .iter()
        .fold(T::ZERO, |largest, x| largest.max(magnitude(*x)))
}

/// The inner product `⟨a, b⟩` of two columns.
fn dot<T: Real>(a: &Matrix<T>, b: &Matrix<T>) -> T {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| *x * *y)
        .sum()
}

/// Whether every element of a matrix is finite.
fn all_finite<T: Real>(m: &Matrix<T>) -> bool {
    m.as_slice().iter().all(|v| finite(*v))
}

/// When an optimisation method stops, and when it gives up.
///
/// A method of several variables accepts the point it is at once the gradient has stopped being
/// one — its infinity norm is at most [`Self::gtol`] — or once it has stopped moving, by
/// `xtol + rtol · max(‖x‖∞, 1)`. A bracketed method accepts the bracket once it is that narrow.
///
/// The defaults suit `f64`: `xtol` 0, `rtol` 4√ε, `ftol` 0, `gtol` √ε and `max_iter` 1000. They
/// suit `f32` as they stand, being scaled to its precision.
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
    /// The relative part of the tolerance on the point, and on the value a step improves by.
    pub rtol: T,
    /// The absolute part of the tolerance on the value; 0 asks for none.
    pub ftol: T,
    /// The infinity norm of the gradient at which a stationary point is accepted.
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
    #[inline]
    #[must_use]
    pub fn with_xtol(mut self, xtol: T) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the relative tolerance on the point.
    #[inline]
    #[must_use]
    pub fn with_rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    /// Sets the tolerance on the value.
    #[inline]
    #[must_use]
    pub fn with_ftol(mut self, ftol: T) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets the tolerance on the gradient.
    #[inline]
    #[must_use]
    pub fn with_gtol(mut self, gtol: T) -> Self {
        self.gtol = gtol;
        self
    }

    /// Sets the largest number of iterations.
    #[inline]
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// The distance a point at `x` still has to travel to be accepted:
    /// `xtol + rtol · max(|x|, 1)`.
    #[must_use]
    pub fn tol_at(&self, x: T) -> T {
        self.xtol + self.rtol * magnitude(x).max(T::ONE)
    }

    /// The same for a whole state, by the infinity norm of the column.
    #[must_use]
    pub fn tol_of(&self, x: &Matrix<T>) -> T {
        self.xtol + self.rtol * norm_inf(x).max(T::ONE)
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
#[derive(Debug, Clone)]
pub struct MinResult<T> {
    /// The point the method found.
    pub x: Matrix<T>,
    /// The value of the function there.
    pub value: T,
    /// The number of iterations, which for the simplex is a reflection and all that came with it
    /// rather than one evaluation.
    pub iterations: usize,
    /// Whether the method reached a stationary point rather than merely running out of budget.
    pub converged: bool,
}

/// The parameters a least squares method found.
#[derive(Debug, Clone)]
pub struct FitResult<T> {
    /// The parameters.
    pub x: Matrix<T>,
    /// Half the sum of the squares of the residuals there, the quantity the method minimises.
    pub cost: T,
    /// The number of corrector steps taken, whether or not each was accepted.
    pub iterations: usize,
    /// Whether the method reached a stationary point rather than merely running out of budget.
    pub converged: bool,
}

// ---------------------------------------------------------------------------------------------
// One variable
// ---------------------------------------------------------------------------------------------

/// The minimum of `f` within `[a, b]`, by the golden section.
///
/// The function has to fall and then rise once inside the bracket, which is what lets the method
/// throw away 0.382 of the interval for one evaluation a step; it is the slowest method here and
/// the only one that cannot be misled by a flat or a noisy function. Use [`minimize`] unless the
/// function is one parabolic interpolation cannot be trusted on.
///
/// Returns `None` when the bracket is empty or not finite, or when `f` is nowhere finite in it.
///
/// # Examples
///
/// ```
/// use a::math::numerics::golden_section;
///
/// // The minimum of x ↦ x⁴ − 2x² on [0, 2] is at one, with value −1.
/// let x = golden_section(|x: f64| x.powi(4) - 2.0 * x * x, 0.0, 2.0).unwrap();
/// assert!((x - 1.0).abs() < 1e-6);
/// ```
pub fn golden_section<T: Real>(f: impl FnMut(T) -> T, a: T, b: T) -> Option<T> {
    golden_section_with(f, a, b, &OptOptions::new()).map(|r| r.x)
}

/// [`golden_section`] under `opts`, reporting the bracket it settled on.
pub fn golden_section_with<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &OptOptions<T>,
) -> Option<OneDimResult<T>> {
    if !ordered(a, b) || !finite(a) || !finite(b) {
        return None;
    }
    // The reciprocal of the golden ratio, (3 − √5)/2: the two interior points are placed so that
    // whichever half of the interval is thrown away, one of them is still in its right place for
    // the next step and only one evaluation is new.
    let cut = (T::from_f64(5.0).sqrt() - T::ONE) / T::TWO;
    let mut left = a;
    let mut right = b;
    let mut low = right - cut * (right - left);
    let mut high = left + cut * (right - left);
    let (mut f_low, mut f_high) = (f(low), f(high));
    let mut evaluations = 2;

    let tol = opts.xtol + opts.rtol * magnitude(b - a);
    while right - left > tol && evaluations < opts.max_iter {
        if better(f_low, f_high) {
            right = high;
            high = low;
            f_high = f_low;
            low = right - cut * (right - left);
            f_low = f(low);
        } else {
            left = low;
            low = high;
            f_low = f_high;
            high = left + cut * (right - left);
            f_high = f(high);
        }
        evaluations += 1;
    }

    let (x, value) = if better(f_low, f_high) {
        (low, f_low)
    } else {
        (high, f_high)
    };
    if !finite(value) {
        return None;
    }
    Some(OneDimResult {
        x,
        value,
        evaluations,
        converged: right - left <= tol,
    })
}

/// The minimum of `f` within `[a, b]`, by Brent's method: golden section until it can afford a
/// parabola, then parabolic interpolation, then back again.
///
/// The function has to fall and then rise once inside the bracket. Where it is smooth the
/// parabola is nearly right and the method converges superlinearly; where it is not, every step is
/// a golden one and it is as slow and as sure as [`golden_section`]. This is the method to reach
/// for in one variable.
///
/// Returns `None` when the bracket is empty or not finite, or when `f` is nowhere finite in it.
///
/// # Examples
///
/// ```
/// use a::math::numerics::minimize;
///
/// // The same quartic, and the same answer in half the evaluations.
/// let r = minimize(|x: f64| x.powi(4) - 2.0 * x * x, 0.0, 2.0).unwrap();
/// assert!((r.x - 1.0).abs() < 1e-7);
/// assert!((r.value + 1.0).abs() < 1e-14);
/// assert!(r.evaluations < 40);
/// ```
pub fn minimize<T: Real>(f: impl FnMut(T) -> T, a: T, b: T) -> Option<OneDimResult<T>> {
    minimize_with(f, a, b, &OptOptions::new())
}

/// [`minimize`] under `opts`.
pub fn minimize_with<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &OptOptions<T>,
) -> Option<OneDimResult<T>> {
    if !ordered(a, b) || !finite(a) || !finite(b) {
        return None;
    }
    const CGOLD: f64 = 0.381_966_011_250_105_15;
    let gold = T::from_f64(CGOLD);
    let eps = T::EPSILON;

    let mut left = a;
    let mut right = b;
    let mut x = a + gold * (b - a);
    let mut v = x;
    let mut w = x;
    let (mut f_x, mut f_v, mut f_w) = (f(x), f(x), f(x));
    let mut evaluations = 1;
    if !finite(f_x) {
        return None;
    }
    // The length and direction of the step before the last, which is what tells a parabolic step
    // whether it is repeating a golden one too closely to be worth the name.
    let tol = opts.xtol + opts.rtol * magnitude(b - a);

    // The length of the last step and of the one before it, which is what tells a parabolic step
    // whether it repeats a golden one too closely to be worth the name.
    let mut last = T::ZERO;
    let mut before = T::ZERO;
    let mut converged = false;
    for _ in 0..opts.max_iter {
        let mut step;
        let middle = (left + right) * T::HALF;
        let tolerance_one = T::TWO * eps * magnitude(x).max(T::ONE) + tol / T::from_f64(3.0);
        let tolerance_two = T::TWO * tolerance_one;
        if magnitude(x - middle) <= tolerance_two - (right - left) * T::HALF {
            converged = true;
            break;
        }

        if magnitude(last) > tolerance_one {
            // The parabola through the three best points of the last two steps, whose minimum is
            // taken for a step when it lies inside the bracket and is not a repeat of the golden
            // step the last one was.
            let one = (x - w) * (f_x - f_v);
            let two = (x - v) * (f_x - f_w);
            let mut p = (x - v) * two - (x - w) * one;
            let mut q = T::TWO * (two - one);
            if q > T::ZERO {
                p = -p;
            } else {
                q = -q;
            }
            let before_length = magnitude(q * before) * T::HALF;
            before = last;
            if magnitude(p) < before_length
                && p > q * (left - x + tolerance_one)
                && p < q * (right - x - tolerance_one)
            {
                step = p / q;
                let toward = x + step;
                if toward - left < tolerance_one || right - toward < tolerance_one {
                    step = if middle >= x {
                        tolerance_one
                    } else {
                        -tolerance_one
                    };
                }
            } else {
                last = if x >= middle { left - x } else { right - x };
                step = gold * last;
            }
        } else {
            last = if x >= middle { left - x } else { right - x };
            step = gold * last;
        }
        last = before;

        let away = if magnitude(step) >= tolerance_one {
            step
        } else if step >= T::ZERO {
            tolerance_one
        } else {
            -tolerance_one
        };
        let toward = x + away;
        let f_toward = f(toward);
        evaluations += 1;

        if not_worse(f_toward, f_x) {
            if toward >= x {
                left = x;
            } else {
                right = x;
            }
            v = w;
            f_v = f_w;
            w = x;
            f_w = f_x;
            x = toward;
            f_x = f_toward;
        } else {
            if toward < x {
                left = toward;
            } else {
                right = toward;
            }
            if not_worse(f_toward, f_w) || w == x {
                v = w;
                f_v = f_w;
                w = toward;
                f_w = f_toward;
            } else if not_worse(f_toward, f_v) || v == x || v == w {
                v = toward;
                f_v = f_toward;
            }
        }
    }

    if !finite(f_x) {
        return None;
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

/// The minimum of `f`, by the simplex of Nelder and Mead: `n + 1` points, reflected through their
/// centroid, expanded when the reflection looks good and contracted when it does not, the whole
/// simplex shrunk towards its best point when neither works.
///
/// It needs only values, which is what makes it the method for a function whose gradient cannot be
/// written down — a simulation, a schedule, a landscape with a kink in it. It is not a global
/// method, it gains nothing from a smooth function that a gradient would tell more about, and it
/// stalls on a badly scaled one: start it near the answer, or scale the variables first.
///
/// `step` is how far from `x0` the other corners of the starting simplex are; it should be the
/// distance over which `f` changes appreciably. Returns `None` when `x0` is not a column of
/// finite values, when `step` is not positive, or when `f` is not finite at the start.
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
/// assert!(r.value < 1e-12);
/// assert!(r.converged);
/// ```
pub fn nelder_mead<T: Real + Scalar>(
    f: impl FnMut(&Matrix<T>) -> T,
    x0: &Matrix<T>,
    step: T,
) -> Option<MinResult<T>> {
    nelder_mead_with(f, x0, step, &OptOptions::new())
}

/// [`nelder_mead`] under `opts`.
pub fn nelder_mead_with<T: Real + Scalar>(
    mut f: impl FnMut(&Matrix<T>) -> T,
    x0: &Matrix<T>,
    step: T,
    opts: &OptOptions<T>,
) -> Option<MinResult<T>> {
    let n = x0.rows();
    if n == 0 || x0.cols() != 1 || !ordered(T::ZERO, step) || !finite(step) || !all_finite(x0) {
        return None;
    }

    let mut simplex: Vec<Matrix<T>> = (0..=n)
        .map(|i| {
            let mut point = x0.clone();
            if i > 0 {
                point[(i - 1, 0)] = x0[(i - 1, 0)] + step;
            }
            point
        })
        .collect();
    let mut values: Vec<T> = Vec::with_capacity(n + 1);
    for point in &simplex {
        values.push(f(point));
    }
    if !values.iter().any(|v| finite(*v)) {
        return None;
    }

    let mut order: Vec<usize> = (0..=n).collect();
    let mut iterations = 0;
    let mut converged = false;
    while iterations < opts.max_iter {
        iterations += 1;
        order.sort_by(|a, b| compare(values[*a], values[*b]));
        let new_simplex: Vec<Matrix<T>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let new_values: Vec<T> = order.iter().map(|&i| values[i]).collect();
        simplex = new_simplex;
        values = new_values;

        // The simplex is as converged as its corners are together: no two of them are further
        // apart than the tolerance, and no two values differ by more than it.
        let spread = |points: &[Matrix<T>]| -> T {
            (1..=n).fold(T::ZERO, |far, i| {
                far.max(norm_inf(&(&points[i] - &points[0])))
            })
        };
        let far = spread(&simplex);
        let value_gap = magnitude(values[n] - values[0]);
        if far <= opts.tol_of(&simplex[0])
            && value_gap <= opts.ftol + opts.rtol * magnitude(values[0]).max(T::ONE)
        {
            converged = true;
            break;
        }

        // The centroid of all but the worst corner.
        let mut centroid = Matrix::<T>::zeros(n, 1);
        for point in simplex.iter().take(n) {
            for r in 0..n {
                centroid[(r, 0)] += point[(r, 0)];
            }
        }
        for r in 0..n {
            centroid[(r, 0)] /= T::from_f64(n as f64);
        }
        let worst = &simplex[n];

        // Reflect the worst corner through the centroid, then take whichever of the four moves
        // the values ask for: reflect, expand, contract, or shrink the whole simplex.
        let away = &centroid - worst;
        let reflected = &centroid + &away;
        let f_reflected = f(&reflected);
        if better(f_reflected, values[0]) {
            let expanded = &centroid + &(&away * T::TWO);
            let f_expanded = f(&expanded);
            if better(f_expanded, f_reflected) {
                simplex[n] = expanded;
                values[n] = f_expanded;
            } else {
                simplex[n] = reflected;
                values[n] = f_reflected;
            }
        } else if better(f_reflected, values[n - 1]) {
            simplex[n] = reflected;
            values[n] = f_reflected;
        } else {
            let contracted = &centroid + &(&reflected - &centroid) * T::HALF;
            let f_contracted = f(&contracted);
            if better(f_contracted, values[n]) {
                simplex[n] = contracted;
                values[n] = f_contracted;
            } else {
                for i in 1..=n {
                    simplex[i] = &simplex[0] + &(&simplex[i] - &simplex[0]) * T::HALF;
                    values[i] = f(&simplex[i]);
                }
            }
        }
    }

    let best = values
        .iter()
        .enumerate()
        .fold((0usize, values[0]), |acc, (i, v)| {
            if better(*v, acc.1) { (i, *v) } else { acc }
        });
    if !finite(best.1) {
        return None;
    }
    Some(MinResult {
        x: simplex[best.0].clone(),
        value: best.1,
        iterations,
        converged,
    })
}

// ---------------------------------------------------------------------------------------------
// Several variables, with gradients
// ---------------------------------------------------------------------------------------------

/// The value, the gradient and the slope of `φ(α) = f(x + α·d)` at one trial step.
fn probe<T, F, G>(
    f: &mut F,
    g: &mut G,
    x: &Matrix<T>,
    d: &Matrix<T>,
    alpha: T,
) -> (Matrix<T>, T, Matrix<T>, T)
where
    T: Real + Scalar,
    F: FnMut(&Matrix<T>) -> T,
    G: FnMut(&Matrix<T>) -> Matrix<T>,
{
    let point = x + &(d * alpha);
    let value = f(&point);
    let gradient = g(&point);
    let slope = dot(&gradient, d);
    (point, value, gradient, slope)
}

/// The line search of both gradient methods: the largest step along `d` that keeps enough of the
/// decrease its tangent promised (the Armijo condition) and is not so long that the derivative has
/// already turned back up past it (the curvature condition) — together the strong Wolfe conditions,
/// algorithm 3.5 of Nocedal and Wright's *Numerical Optimization*.
///
/// The search widens a trial step until the valley floor is bracketed, then closes the bracket,
/// guessing where along it the derivative vanishes by the secant rule and falling back on
/// bisection whenever that guess is not safely inside. It returns the step, the point it reaches,
/// and the value and the gradient there — the gradient comes out of the search, so the caller does
/// not evaluate it again — or `None` when no step along `d` even keeps the decrease condition,
/// which means `d` does not lead downhill after all.
#[allow(clippy::too_many_arguments)]
fn wolfe<T, F, G>(
    f: &mut F,
    g: &mut G,
    x: &Matrix<T>,
    d: &Matrix<T>,
    value: T,
    slope: T,
    first: T,
) -> Option<(T, Matrix<T>, T, Matrix<T>)>
where
    T: Real + Scalar,
    F: FnMut(&Matrix<T>) -> T,
    G: FnMut(&Matrix<T>) -> Matrix<T>,
{
    let c1 = T::from_f64(WOLFE_C1);
    let c2 = T::from_f64(WOLFE_C2);
    let widest = T::from_f64(LINE_SEARCH_MAX_STEP);
    let narrowest = T::from_f64(LINE_SEARCH_MIN_STEP);
    // The step to fall back on: the longest one so far that kept the decrease condition, which is
    // the answer when the curvature condition is never quite met.
    let mut best: Option<(T, Matrix<T>, T, Matrix<T>)> = None;

    // The bracket of the valley floor: `low` is a step that kept the decrease condition and was
    // still going down, `high` one that either lost the decrease condition or had its derivative
    // turn back. The slope at `high` is not always known, since a step outside the domain of `f`
    // has none.
    let mut low = T::ZERO;
    let mut low_value = value;
    let mut low_slope = slope;
    let mut high = T::ZERO;
    let mut high_slope: Option<T> = None;
    let mut bracketed = false;

    let mut alpha = first.min(widest);
    for _ in 0..LINE_SEARCH_MAX {
        let (point, new_value, new_gradient, new_slope) = probe(f, g, x, d, alpha);
        let usable = finite(new_value) && all_finite(&new_gradient);
        if !usable || new_value > value + c1 * alpha * slope {
            high = alpha;
            high_slope = if usable { Some(new_slope) } else { None };
            bracketed = true;
            break;
        }
        if magnitude(new_slope) <= -c2 * slope {
            return Some((alpha, point, new_value, new_gradient));
        }
        if new_slope >= T::ZERO {
            high = alpha;
            high_slope = Some(new_slope);
            bracketed = true;
            break;
        }
        low = alpha;
        low_value = new_value;
        low_slope = new_slope;
        high = T::INFINITY;
        high_slope = None;
        best = Some((alpha, point, new_value, new_gradient));
        alpha = if bracketed {
            midpoint(low, high)
        } else {
            (alpha * T::from_f64(WOLFE_WIDEN)).min(widest)
        };
        if !ordered(narrowest, alpha) || !finite(alpha) {
            break;
        }
    }

    // The closing of the bracket, which the widening may have left as far apart as it likes.
    if bracketed {
        for _ in 0..LINE_SEARCH_MAX {
            let mid = midpoint(low, high);
            if high - low <= narrowest * magnitude(high).max(T::ONE) || mid <= low || mid >= high {
                break;
            }
            // The secant rule places the next trial where the straight line between the two
            // derivatives crosses zero; when it cannot, or places it within a tenth of an end of
            // the bracket, bisection takes its place.
            let margin = (high - low) * T::from_f64(0.1);
            let secant = match high_slope {
                Some(high_slope) => {
                    let denominator = high_slope - low_slope;
                    if denominator != T::ZERO {
                        Some(high - high_slope * (high - low) / denominator)
                    } else {
                        None
                    }
                }
                None => None,
            };
            let trial = match secant {
                Some(guess) if guess > low + margin && guess < high - margin => guess,
                _ => mid,
            };

            let (point, new_value, new_gradient, new_slope) = probe(f, g, x, d, trial);
            if !finite(new_value)
                || !all_finite(&new_gradient)
                || new_value > value + c1 * trial * slope
                || new_value >= low_value
            {
                high = trial;
                high_slope = if finite(new_value) {
                    Some(new_slope)
                } else {
                    None
                };
                continue;
            }
            if magnitude(new_slope) <= -c2 * slope {
                return Some((trial, point, new_value, new_gradient));
            }
            if new_slope * (high - trial) >= T::ZERO {
                high = trial;
                high_slope = Some(new_slope);
            } else {
                low = trial;
                low_value = new_value;
                low_slope = new_slope;
                best = Some((trial, point, new_value, new_gradient));
            }
        }
    }
    best
}

/// The midpoint of two steps, which is a finite one even when the upper end is the infinity the
/// search has not yet bracketed past.
fn midpoint<T: Real>(low: T, high: T) -> T {
    if finite(high) {
        (low + high) * T::HALF
    } else {
        low * T::TWO
    }
}

/// The minimum of `f`, by the conjugate gradient method of Polak, Ribière and Polyak: a
/// steepest descent step, then each new direction the old one corrected by how much the gradient
/// has turned, and a restart down the gradient whenever a direction stops pointing downhill.
///
/// It needs the gradient and keeps only the last one, so where `n` is large enough that BFGS's
/// `n × n` matrix is unwelcome this is the method to use. It converges in `n` steps on a quadratic
/// and takes more on anything else.
///
/// Returns `None` when `x0` is not a column of finite values, when `f` or ∇`f` is not finite at
/// it, or when the very first direction leads nowhere.
///
/// # Examples
///
/// ```
/// use a::math::numerics::conjugate_gradient;
/// use a::math::matrix::Matrix;
///
/// // Rosenbrock's valley again, from the same start.
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
/// assert!((r.x[(0, 0)] - 1.0).abs() < 1e-4 && (r.x[(1, 0)] - 1.0).abs() < 1e-4);
/// assert!(r.converged);
/// ```
pub fn conjugate_gradient<T: Real + Scalar>(
    f: impl FnMut(&Matrix<T>) -> T,
    g: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
) -> Option<MinResult<T>> {
    conjugate_gradient_with(f, g, x0, &OptOptions::new())
}

/// [`conjugate_gradient`] under `opts`.
pub fn conjugate_gradient_with<T: Real + Scalar>(
    mut f: impl FnMut(&Matrix<T>) -> T,
    mut g: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
) -> Option<MinResult<T>> {
    let n = x0.rows();
    if n == 0 || x0.cols() != 1 || !all_finite(x0) {
        return None;
    }
    let mut x = x0.clone();
    let mut value = f(&x);
    let mut gradient = g(&x);
    if !finite(value) || !all_finite(&gradient) {
        return None;
    }

    let mut direction = -gradient.clone();
    let mut iterations = 0;
    let mut converged = false;
    while iterations < opts.max_iter {
        iterations += 1;
        if norm_inf(&gradient) <= opts.gtol {
            converged = true;
            break;
        }
        let mut slope = dot(&direction, &gradient);
        if !ordered(slope, T::ZERO) {
            direction = -gradient.clone();
            slope = -dot(&gradient, &gradient);
        }
        let found = wolfe(&mut f, &mut g, &x, &direction, value, slope, T::ONE);
        let (_step, moved, moved_value, moved_gradient) = match found {
            Some(found) => found,
            None => break,
        };
        let delta = &moved - &x;
        x = moved;
        value = moved_value;
        let before = std::mem::replace(&mut gradient, moved_gradient);
        if !all_finite(&gradient) {
            return None;
        }

        if norm_inf(&delta) <= opts.tol_of(&x) {
            converged = true;
            break;
        }

        // Polak and Ribière's β, which measures how much the gradient has turned since the last
        // step; negative β means the new gradient undoes the old direction, so it is floored at
        // zero and the direction restarts down the gradient.
        let square = dot(&before, &before);
        let beta = if square > T::ZERO {
            let turned = dot(&gradient, &(&gradient - &before));
            (turned / square).max(T::ZERO)
        } else {
            T::ZERO
        };
        direction = &direction * beta - &gradient;
        let every = iterations % (n + 1) == 0;
        if every || dot(&direction, &gradient) >= T::ZERO {
            direction = -gradient.clone();
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
/// gradients it has seen, one rank-two correction per step.
///
/// Each step is `−H∇f` and the inverse Hessian is updated to be exactly right about the gradient
/// along the step just taken, so after `n` independent steps it knows a quadratic as well as if
/// it had been given one. It keeps an `n × n` matrix, so where that is too much memory use
/// [`conjugate_gradient`], which keeps only the last gradient.
///
/// The update is skipped when a step's `s·y` is not positive, which leaves the matrix positive
/// definite and the next direction pointing downhill. Returns `None` when `x0` is not a column of
/// finite values, or when `f` or ∇`f` is not finite at it.
///
/// # Examples
///
/// ```
/// use a::math::numerics::bfgs;
/// use a::math::matrix::Matrix;
///
/// // The quadratic x₀² + 2x₁² − 2x₀ − 8x₁ + 9, which is (x₀ − 1)² + 2(x₁ − 2)² − 9: its
/// // minimum is (1, 2), with value 0.
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
pub fn bfgs<T: Real + Scalar>(
    f: impl FnMut(&Matrix<T>) -> T,
    g: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
) -> Option<MinResult<T>> {
    bfgs_with(f, g, x0, &OptOptions::new())
}

/// [`bfgs`] under `opts`.
pub fn bfgs_with<T: Real + Scalar>(
    mut f: impl FnMut(&Matrix<T>) -> T,
    mut g: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
) -> Option<MinResult<T>> {
    let n = x0.rows();
    if n == 0 || x0.cols() != 1 || !all_finite(x0) {
        return None;
    }
    let mut x = x0.clone();
    let mut value = f(&x);
    let mut gradient = g(&x);
    if !finite(value) || !all_finite(&gradient) {
        return None;
    }

    // The inverse Hessian, which starts as the identity and learns the curvature one step at a
    // time. It is never inverted, only corrected, so it stays symmetric by construction.
    let mut inverse = Matrix::<T>::identity(n);
    let mut iterations = 0;
    let mut converged = false;
    while iterations < opts.max_iter {
        iterations += 1;
        if norm_inf(&gradient) <= opts.gtol {
            converged = true;
            break;
        }

        let mut direction = -(&inverse * &gradient);
        let mut slope = dot(&direction, &gradient);
        if !ordered(slope, T::ZERO) {
            // A direction that is not downhill means the accumulated curvature is untrustworthy;
            // forget it and go down the gradient, which cannot fail to be downhill.
            inverse = Matrix::identity(n);
            direction = -gradient.clone();
            slope = -dot(&gradient, &gradient);
            if !ordered(slope, T::ZERO) {
                converged = true;
                break;
            }
        }

        // A direction the curvature led uphill is tried again as the gradient itself, which
        // cannot fail to lead down.
        let mut found = wolfe(&mut f, &mut g, &x, &direction, value, slope, T::ONE);
        if found.is_none() {
            inverse = Matrix::identity(n);
            let steep = -gradient.clone();
            let downhill = dot(&steep, &gradient);
            if downhill < T::ZERO {
                found = wolfe(&mut f, &mut g, &x, &steep, value, downhill, T::ONE);
            }
        }
        let (_step, moved, moved_value, moved_gradient) = match found {
            Some(found) => found,
            None => break,
        };
        let delta = &moved - &x;
        x = moved;
        value = moved_value;
        let before = std::mem::replace(&mut gradient, moved_gradient);
        if !all_finite(&gradient) {
            return None;
        }
        if norm_inf(&delta) <= opts.tol_of(&x) {
            converged = true;
            break;
        }

        // The inverse BFGS correction, `H ← (I − ρsyᵀ) H (I − ρysᵀ) + ρssᵀ`, which is the unique
        // symmetric update that satisfies the secant equation Hy = s and leaves the matrix
        // positive definite. A step with s·y ≤ 0 says nothing about curvature that a positive
        // definite H can agree with, so it is passed over.
        let curvature = dot(&delta, &(&gradient - &before));
        if curvature > T::ZERO {
            let y = &gradient - &before;
            let rho = T::ONE / curvature;
            let mut correction = Matrix::<T>::identity(n);
            for i in 0..n {
                for j in 0..n {
                    correction[(i, j)] -= rho * delta[(i, 0)] * y[(j, 0)];
                }
            }
            let transpose = correction.transpose();
            inverse = &correction * &inverse * &transpose + &(&delta * &delta.transpose()) * rho;
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
/// `J·δ = −r` for the correction, by [`qr`](crate::math::matrix::Matrix::qr), and takes it whole.
///
/// Where the residual is nearly linear in the parameters it converges quadratically and often in a
/// handful of steps; where it is not, a whole step can overshoot and the method has nothing to
/// fall back on, so it stops without improving. Use [`levenberg_marquardt`] unless the residual is
/// well behaved near its zero.
///
/// `residual` maps the `n` parameters to a column of `m ≥ n` residuals and `jacobian` to the
/// `m × n` matrix of their derivatives. Returns `None` when either is not finite, when the shapes
/// are not those, or when the Jacobian's columns are dependent.
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
/// assert_eq!(fit.cost, 0.0);
/// assert!(fit.converged);
/// ```
pub fn gauss_newton<T: Real + Scalar>(
    residual: impl FnMut(&Matrix<T>) -> Matrix<T>,
    jacobian: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
) -> Option<FitResult<T>> {
    gauss_newton_with(residual, jacobian, x0, &OptOptions::new())
}

/// [`gauss_newton`] under `opts`.
pub fn gauss_newton_with<T: Real + Scalar>(
    residual: impl FnMut(&Matrix<T>) -> Matrix<T>,
    jacobian: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
) -> Option<FitResult<T>> {
    fit(residual, jacobian, x0, opts, false)
}

/// The least squares of `residual`, by Levenberg and Marquardt: the Gauss and Newton step damped
/// by a multiple of the curvature it disagrees with, the damping shrunk when a step pays off and
/// grown when it does not.
///
/// A large damping makes the step a small one down the gradient, which always improves the cost,
/// and a small one makes it the Gauss and Newton step, which converges quadratically — the method
/// interpolates between the two and is the one to reach for when the starting parameters are far
/// out. The damping is scaled by the diagonal of `JᵀJ`, so it is a fraction of each parameter's own
/// curvature rather than of all of them together.
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
pub fn levenberg_marquardt<T: Real + Scalar>(
    residual: impl FnMut(&Matrix<T>) -> Matrix<T>,
    jacobian: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
) -> Option<FitResult<T>> {
    levenberg_marquardt_with(residual, jacobian, x0, &OptOptions::new())
}

/// [`levenberg_marquardt`] under `opts`.
pub fn levenberg_marquardt_with<T: Real + Scalar>(
    residual: impl FnMut(&Matrix<T>) -> Matrix<T>,
    jacobian: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
) -> Option<FitResult<T>> {
    fit(residual, jacobian, x0, opts, true)
}

/// Both least squares methods: [`gauss_newton`] with `damped` false, [`levenberg_marquardt`] with
/// it true. The damping is what the one has and the other does without.
fn fit<T: Real + Scalar>(
    mut residual: impl FnMut(&Matrix<T>) -> Matrix<T>,
    mut jacobian: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    opts: &OptOptions<T>,
    damped: bool,
) -> Option<FitResult<T>> {
    let n = x0.rows();
    if n == 0 || x0.cols() != 1 || !all_finite(x0) {
        return None;
    }
    let mut x = x0.clone();
    let mut r = residual(&x);
    let m = r.rows();
    if m < n || r.cols() != 1 || !all_finite(&r) {
        return None;
    }
    let half = T::HALF;
    let mut cost = dot(&r, &r) * half;
    if !finite(cost) {
        return None;
    }

    let mut lambda = T::from_f64(MARQUARDT_LAMBDA);
    let mut iterations = 0;
    let mut converged = false;
    while iterations < opts.max_iter {
        iterations += 1;
        let j = jacobian(&x);
        if j.rows() != m || j.cols() != n || !all_finite(&j) {
            return None;
        }
        let transpose = j.transpose();
        let gradient = &transpose * &r;
        let curvature = &transpose * &j;
        if norm_inf(&gradient) <= opts.gtol {
            converged = true;
            break;
        }

        // The correction: Gauss and Newton solves `Jδ = −r` as a least squares, which needs no
        // normal equations and so no squaring of the condition number; Levenberg and Marquardt
        // solves the damped `（JᵀJ + λ·diag(JᵀJ)）δ = −Jᵀr`, which is what makes the step short
        // where the linearisation cannot be trusted.
        let correction = if damped {
            let mut system = curvature.clone();
            for i in 0..n {
                let diagonal = magnitude(curvature[(i, i)]).max(T::MIN_POSITIVE);
                system[(i, i)] += lambda * diagonal;
            }
            match system.lu().solve(&-(gradient.clone())) {
                Some(correction) => correction,
                None => {
                    lambda *= T::from_f64(MARQUARDT_UP);
                    if lambda > T::from_f64(MARQUARDT_LAMBDA_MAX) {
                        break;
                    }
                    continue;
                }
            }
        } else {
            j.qr().solve(&-(r.clone()))?
        };
        if !all_finite(&correction) {
            if !damped {
                return None;
            }
            lambda *= T::from_f64(MARQUARDT_UP);
            continue;
        }

        let moved = &x + &correction;
        let r_moved = residual(&moved);
        if r_moved.rows() != m || r_moved.cols() != 1 {
            return None;
        }
        let moved_cost = if all_finite(&r_moved) {
            dot(&r_moved, &r_moved) * half
        } else {
            T::INFINITY
        };
        if !better(moved_cost, cost) {
            // The step made the cost worse. Undamped there is nothing to do but stop; damped,
            // trust the linearisation less and take a shorter step from where we are.
            if !damped {
                break;
            }
            lambda *= T::from_f64(MARQUARDT_UP);
            if lambda > T::from_f64(MARQUARDT_LAMBDA_MAX) {
                break;
            }
            continue;
        }

        let gain = cost - moved_cost;
        x = moved;
        r = r_moved;
        cost = moved_cost;
        if damped {
            lambda = (lambda * T::from_f64(MARQUARDT_DOWN)).max(T::MIN_POSITIVE);
        }
        if norm_inf(&correction) <= opts.tol_of(&x) || gain <= opts.ftol + opts.rtol * cost {
            converged = true;
            break;
        }
    }

    Some(FitResult {
        x,
        cost,
        iterations,
        converged,
    })
}
