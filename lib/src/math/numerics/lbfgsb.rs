//! The minimum of a function within bounds on each of its variables, by L-BFGS-B.
//!
//! The method is that of Byrd, Lu, Nocedal and Zhu (*A limited memory algorithm for bound
//! constrained optimization*, SIAM Journal on Scientific Computing 16, 1995), in the form of the
//! reference code of Zhu, Byrd, Lu and Nocedal (ACM Transactions on Mathematical Software 23,
//! 1997) with the change to its subspace step that Morales and Nocedal made in 2011 (ACM TOMS
//! 38, the version 3.0 of the code). It minimises `f(x)` subject to `l ≤ x ≤ u`, any bound of
//! which may be infinite, and needs `f` and its gradient. Every iteration does four things:
//!
//! 1. **The model.** The Hessian is approximated by the limited-memory BFGS matrix of the last
//!    `m` steps `s` and the changes of the gradient `y` along them, held in the compact form of
//!    Byrd, Nocedal and Schnabel (1994), `B = θI − W·M·Wᵀ` with `W = [Y, θS]` and the `2m × 2m`
//!    matrix `M` given through the Cholesky factor of `θSᵀS + L·D⁻¹·Lᵀ`, `D` the diagonal and `L`
//!    the strict lower triangle of `SᵀY`. Nothing of size `n × n` is ever formed.
//! 2. **The generalised Cauchy point.** The first minimiser of the quadratic model along the path
//!    `P(x − t·∇f)`, `P` the projection onto the box: the path bends at the points where a
//!    variable reaches its bound, which are taken in order from a heap, and on each straight
//!    piece the model is a parabola whose two derivatives are updated in `O(m²)` (Algorithm CP of
//!    the 1995 paper). The variables at a bound there are the active set of the iteration.
//! 3. **The subspace step.** The model is minimised over the free variables with the active ones
//!    held, by the direct primal method of section 5.1 of the paper: the reduced matrix is
//!    inverted by the Sherman–Morrison–Woodbury formula through a symmetric indefinite `2m × 2m`
//!    matrix `K`, factored as `K = L·E·Lᵀ` from two Cholesky factorisations. The minimiser is
//!    projected onto the box; when that projection turns it into a direction that no longer
//!    leads downhill it is instead cut back to the box along the line from the Cauchy point
//!    (Morales and Nocedal).
//! 4. **The line search.** Along the line from the iterate to that point, by the search of Moré
//!    and Thuente under the strong conditions of Wolfe with the constants of the reference code,
//!    `μ = 10⁻³` and `η = 0.9`, and the step held to the box. The pair `(s, y)` of the step goes
//!    into the memory, unless `sᵀy ≤ ε·(−∇fᵀs)`, the curvature test of the reference code, which
//!    keeps `B` positive definite.
//!
//! Each of these is `O(m·n)` or `O(m²·n)`, so an iteration costs time linear in the number of
//! variables. The method stops when the projected gradient, `‖P(x − ∇f(x)) − x‖∞`, is at most
//! `pgtol` — the first-order conditions for a minimum within the box — or when a step has lowered
//! `f` by no more than `ftol·max(|f|, 1)`, the two tests of the reference code; which of the two
//! it was is in [`LbfgsbResult::stop`]. When the line search finds no lower value (nor, where `f`
//! is level within its rounding, a lower projected gradient), or the model breaks down, the
//! memory is thrown away and the iteration done again from the gradient alone, as the reference
//! code does; a second failure in a row ends the method with `converged` false.
//!
//! # Examples
//!
//! ```
//! use a::math::matrix::Matrix;
//! use a::math::numerics::{Bound, lbfgsb};
//!
//! // Rosenbrock's valley with the first variable held to at most 0.5: the minimum moves from
//! // (1, 1) to (0.5, 0.25), on that bound.
//! let f = |x: &Matrix<f64>| (1.0 - x[(0, 0)]).powi(2) + 100.0 * (x[(1, 0)] - x[(0, 0)].powi(2)).powi(2);
//! let g = |x: &Matrix<f64>| {
//!     let (a, b) = (x[(0, 0)], x[(1, 0)]);
//!     Matrix::from_column_major(2, 1, vec![-2.0 * (1.0 - a) - 400.0 * a * (b - a * a), 200.0 * (b - a * a)])
//! };
//! let lower = Matrix::from_column_major(2, 1, vec![-2.0, -2.0]);
//! let upper = Matrix::from_column_major(2, 1, vec![0.5, 2.0]);
//! let x0 = Matrix::from_column_major(2, 1, vec![-1.2, 1.0]);
//! let r = lbfgsb(f, g, &x0, &lower, &upper).unwrap();
//! assert!(r.converged);
//! assert_eq!(r.x[(0, 0)], 0.5);
//! assert!((r.x[(1, 0)] - 0.25).abs() < 1e-6);
//! assert_eq!(r.active, vec![Bound::Upper, Bound::Free]);
//! ```

use super::opt::{
    Found, Line, LineSearch, all_finite, checked_gradient, column_length, dot, finite, magnitude,
    more_thuente, norm2, root,
};
use super::{Real, ordered};
use crate::math::matrix::{Matrix, Scalar};
use core::cmp::Ordering;
use std::collections::BinaryHeap;

/// `μ` of the sufficient decrease condition of the line search, as in the reference code.
const DECREASE: f64 = 1e-3;
/// `η` of its curvature condition, loose as suits a quasi-Newton step.
const CURVATURE: f64 = 0.9;
/// The relative width of the bracket of the line search below which it stops narrowing it.
const NARROW: f64 = 0.1;
/// The most evaluations one line search makes, the twenty of the reference code.
const SEARCH_EVALUATIONS: usize = 20;
/// The longest step along a direction that no bound limits.
const LONGEST: f64 = 1e10;

/// When [`lbfgsb`] stops, and when it gives up.
///
/// The defaults are those of the reference code as SciPy calls it: a memory of 10 pairs,
/// `pgtol` `10⁻⁵`, `ftol` `10⁷·ε` (its `factr` of `10⁷`, `2.2·10⁻⁹` for `f64`) and at most
/// 15,000 iterations and as many evaluations. For `f32`, where `10⁷·ε` would be more than one,
/// `ftol` is `10·ε`, the `factr` of 10 that the reference code suggests for high accuracy: seven
/// digits leave nothing to spare.
/// A tolerance that is negative or not a number, or a memory of no pairs, makes [`lbfgsb_with`]
/// return `None`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::LbfgsbOptions;
///
/// let opts = LbfgsbOptions::<f64>::new().with_memory(5).with_pgtol(1e-8).with_ftol(0.0);
/// assert_eq!((opts.memory, opts.pgtol, opts.ftol), (5, 1e-8, 0.0));
/// assert_eq!(LbfgsbOptions::<f64>::new().ftol, 1e7 * f64::EPSILON);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LbfgsbOptions<T> {
    /// The number of pairs `(s, y)` the limited-memory matrix is built from.
    pub memory: usize,
    /// The method has converged when the infinity norm of the projected gradient is at most this.
    pub pgtol: T,
    /// And when a step lowers `f` by no more than `ftol · max(|f_old|, |f|, 1)`; 0 turns this
    /// test off.
    pub ftol: T,
    /// The largest number of iterations, each one line search.
    pub max_iter: usize,
    /// The largest number of evaluations of `f` and its gradient, which are made together.
    pub max_evaluations: usize,
}

impl<T: Real> Default for LbfgsbOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> LbfgsbOptions<T> {
    /// The defaults: memory 10, `pgtol` `10⁻⁵`, `ftol` `10⁷·ε` for `f64` and `10·ε` for `f32`,
    /// 15,000 iterations and 15,000 evaluations.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LbfgsbOptions;
    ///
    /// let opts = LbfgsbOptions::<f32>::new();
    /// assert_eq!(opts.memory, 10);
    /// assert_eq!(opts.ftol, 10.0 * f32::EPSILON);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            memory: 10,
            pgtol: T::from_f64(1e-5),
            ftol: {
                let reference = T::from_f64(1e7) * T::EPSILON;
                if reference <= T::from_f64(1e-6) {
                    reference
                } else {
                    T::from_f64(10.0) * T::EPSILON
                }
            },
            max_iter: 15_000,
            max_evaluations: 15_000,
        }
    }

    /// Sets the number of pairs kept.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LbfgsbOptions;
    ///
    /// assert_eq!(LbfgsbOptions::<f64>::new().with_memory(3).memory, 3);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_memory(mut self, memory: usize) -> Self {
        self.memory = memory;
        self
    }

    /// Sets the tolerance on the projected gradient.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LbfgsbOptions;
    ///
    /// assert_eq!(LbfgsbOptions::<f64>::new().with_pgtol(1e-9).pgtol, 1e-9);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_pgtol(mut self, pgtol: T) -> Self {
        self.pgtol = pgtol;
        self
    }

    /// Sets the tolerance on the relative reduction of `f`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LbfgsbOptions;
    ///
    /// assert_eq!(LbfgsbOptions::<f64>::new().with_ftol(1e-12).ftol, 1e-12);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_ftol(mut self, ftol: T) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets the largest number of iterations.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LbfgsbOptions;
    ///
    /// assert_eq!(LbfgsbOptions::<f64>::new().with_max_iter(50).max_iter, 50);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the largest number of evaluations.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::LbfgsbOptions;
    ///
    /// assert_eq!(LbfgsbOptions::<f64>::new().with_max_evaluations(80).max_evaluations, 80);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_max_evaluations(mut self, max_evaluations: usize) -> Self {
        self.max_evaluations = max_evaluations;
        self
    }

    /// Whether the options are ones the method can work to.
    fn sane(&self) -> bool {
        self.memory > 0 && self.pgtol >= T::ZERO && self.ftol >= T::ZERO
    }
}

/// Where a variable of the answer of [`lbfgsb`] lies with respect to its bounds.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{Bound, lbfgsb};
///
/// // (x₀ − 2)² + (x₁ + 2)² + (x₂ − 0.5)² in [0, 1]³, the last variable fixed at 0.25.
/// let f = |x: &Matrix<f64>| (x[(0, 0)] - 2.0).powi(2) + (x[(1, 0)] + 2.0).powi(2) + (x[(2, 0)] - 0.5).powi(2);
/// let g = |x: &Matrix<f64>| Matrix::from_column_major(3, 1, vec![
///     2.0 * (x[(0, 0)] - 2.0), 2.0 * (x[(1, 0)] + 2.0), 2.0 * (x[(2, 0)] - 0.5),
/// ]);
/// let lower = Matrix::from_column_major(3, 1, vec![0.0, 0.0, 0.25]);
/// let upper = Matrix::from_column_major(3, 1, vec![1.0, 1.0, 0.25]);
/// let r = lbfgsb(f, g, &Matrix::zeros(3, 1), &lower, &upper).unwrap();
/// assert_eq!(r.active, vec![Bound::Upper, Bound::Lower, Bound::Fixed]);
/// assert_eq!(r.x.as_slice(), &[1.0, 0.0, 0.25]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bound {
    /// Strictly between its bounds.
    Free,
    /// At its lower bound.
    Lower,
    /// At its upper bound.
    Upper,
    /// Its two bounds are the same number, which it was held at throughout.
    Fixed,
}

/// Why [`lbfgsb`] stopped.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{LbfgsbOptions, LbfgsbStop, lbfgsb_with};
///
/// let f = |x: &Matrix<f64>| x[(0, 0)].powi(4);
/// let g = |x: &Matrix<f64>| Matrix::from_column_major(1, 1, vec![4.0 * x[(0, 0)].powi(3)]);
/// let (x0, lower, upper) = (Matrix::from_column_major(1, 1, vec![1.0]), Matrix::from_column_major(1, 1, vec![-5.0]), Matrix::from_column_major(1, 1, vec![5.0]));
/// let r = lbfgsb_with(f, g, &x0, &lower, &upper, &LbfgsbOptions::new().with_max_iter(2)).unwrap();
/// assert_eq!(r.stop, LbfgsbStop::Iterations);
/// assert!(!r.converged && r.iterations == 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LbfgsbStop {
    /// The projected gradient is at most `pgtol`: the first-order conditions hold.
    ProjectedGradient,
    /// The last step lowered `f` by no more than `ftol · max(|f_old|, |f|, 1)`.
    Reduction,
    /// `max_iter` iterations were made.
    Iterations,
    /// `max_evaluations` evaluations were made.
    Evaluations,
    /// No lower value could be found, along the direction of the model nor down the projected
    /// gradient itself: the limit of the arithmetic, or a function that is not smooth there.
    LineSearch,
}

/// The point [`lbfgsb`] found, and how it stands with respect to the bounds.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{LbfgsbStop, lbfgsb};
///
/// // The minimum of x² + y² over x ≥ 1 is at (1, 0), where the gradient (2, 0) presses against
/// // the bound and the projected gradient is zero.
/// let f = |x: &Matrix<f64>| x[(0, 0)].powi(2) + x[(1, 0)].powi(2);
/// let g = |x: &Matrix<f64>| x * 2.0;
/// let lower = Matrix::from_column_major(2, 1, vec![1.0, f64::NEG_INFINITY]);
/// let upper = Matrix::from_column_major(2, 1, vec![f64::INFINITY, f64::INFINITY]);
/// let r = lbfgsb(f, g, &Matrix::from_column_major(2, 1, vec![3.0, 4.0]), &lower, &upper).unwrap();
/// assert!(r.converged && r.stop == LbfgsbStop::ProjectedGradient);
/// assert_eq!(r.x[(0, 0)], 1.0);
/// assert!(r.x[(1, 0)].abs() < 1e-6 && r.projected_gradient <= 1e-5);
/// assert_eq!(r.gradient[(0, 0)], 2.0);
/// ```
#[derive(Debug, Clone)]
pub struct LbfgsbResult<T> {
    /// The point the method found, inside the box.
    pub x: Matrix<T>,
    /// The value of the function there.
    pub value: T,
    /// The gradient there.
    pub gradient: Matrix<T>,
    /// The infinity norm of the projected gradient there, `‖P(x − ∇f(x)) − x‖∞`: zero exactly at
    /// a point that satisfies the first-order conditions within the box.
    pub projected_gradient: T,
    /// Where each variable lies with respect to its bounds.
    pub active: Vec<Bound>,
    /// The number of iterations, each a step that lowered `f` or, where `f` did not change within
    /// its rounding, the projected gradient.
    pub iterations: usize,
    /// The number of evaluations of `f` and its gradient, the one at the start included.
    pub evaluations: usize,
    /// Whether one of the two tests of the method held at `x`: the projected gradient or the
    /// reduction of `f`, as [`LbfgsbResult::stop`] says.
    pub converged: bool,
    /// Why the method stopped.
    pub stop: LbfgsbStop,
}

/// The minimum of `f` within the box `lower ≤ x ≤ upper`, by L-BFGS-B, from `x0`, with the
/// default [`LbfgsbOptions`].
///
/// `g` is the gradient of `f`. A bound of `−∞` or `+∞` is no bound on that side, and a variable
/// whose two bounds are equal is held there. `x0` is projected into the box before anything else,
/// so it may lie outside it. Returns `None` when `x0` is empty or not finite, when a bound is not
/// a number, when `lower > upper` for some variable or a lower bound is `+∞` or an upper bound
/// `−∞` (the box is empty), or when `f` or its gradient is not finite at the projected start. See
/// [`lbfgsb_with`] for the tests that stop it.
///
/// # Panics
///
/// Panics if `x0`, `lower` or `upper` is not a column, if they are not of the same length, or if
/// `g` returns anything but a column of that length.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::lbfgsb;
///
/// // The bounded quadratic ½xᵀAx − bᵀx with A tridiagonal (2, −1), b = (1, −2, 1) and x ≥ 0:
/// // the second variable is held at zero, where the gradient pushes against its bound, and
/// // the other two solve their own 1 × 1 systems, 2x = 1.
/// let f = |x: &Matrix<f64>| {
///     let (p, q, r) = (x[(0, 0)], x[(1, 0)], x[(2, 0)]);
///     p * p + q * q + r * r - p * q - q * r - p + 2.0 * q - r
/// };
/// let g = |x: &Matrix<f64>| {
///     let (p, q, r) = (x[(0, 0)], x[(1, 0)], x[(2, 0)]);
///     Matrix::from_column_major(3, 1, vec![2.0 * p - q - 1.0, 2.0 * q - p - r + 2.0, 2.0 * r - q - 1.0])
/// };
/// let lower = Matrix::zeros(3, 1);
/// let upper = Matrix::from_column_major(3, 1, vec![f64::INFINITY; 3]);
/// let r = lbfgsb(f, g, &Matrix::from_column_major(3, 1, vec![1.0; 3]), &lower, &upper).unwrap();
/// assert!(r.converged);
/// assert!((r.x[(0, 0)] - 0.5).abs() < 1e-6 && r.x[(1, 0)] == 0.0 && (r.x[(2, 0)] - 0.5).abs() < 1e-6);
/// ```
#[track_caller]
pub fn lbfgsb<T: Real + Scalar>(
    f: impl FnMut(&Matrix<T>) -> T,
    g: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    lower: &Matrix<T>,
    upper: &Matrix<T>,
) -> Option<LbfgsbResult<T>> {
    lbfgsb_with(f, g, x0, lower, upper, &LbfgsbOptions::new())
}

/// [`lbfgsb`] under `opts`.
///
/// The method stops, and `converged` is true, when one of the two tests of the reference code
/// holds at the point it is at:
///
/// * the projected gradient test, `‖P(x − ∇f(x)) − x‖∞ ≤ pgtol`, made at the start and after
///   every step, which is the first-order condition for a minimum within the box to the
///   tolerance: each free variable has a small derivative, each variable at a bound a derivative
///   that presses it against the bound or is small;
/// * the reduction test, after a step that lowered `f`: `f_old − f ≤ ftol · max(|f_old|, |f|, 1)`.
///   This says that `f` has stopped improving, not that the gradient is small, and
///   [`LbfgsbResult::projected_gradient`] says how small it is; an `ftol` of zero turns the test
///   off.
///
/// A step is progress when it lowers `f`, or, where the change of `f` along it is lost in the
/// rounding of `f`, when it satisfies the strong Wolfe conditions and lowers the projected
/// gradient; such a level step never makes the reduction test pass. A line search that finds no
/// progress is not counted: the memory is discarded and the iteration made again down the
/// projected gradient, and when that finds none either the method stops with `converged` false
/// and [`LbfgsbStop::LineSearch`]. Running out of `max_iter` iterations or of `max_evaluations`
/// evaluations stops it the same way; a line search is never allowed more evaluations than are
/// left. In every case the point returned has the lowest value of the points the method has
/// stepped to, since no step raises `f`; a trial of a line search that was not taken, as it did
/// not meet the conditions, can have had a lower one.
///
/// # Panics
///
/// Panics if `x0`, `lower` or `upper` is not a column, if they are not of the same length, or if
/// `g` returns anything but a column of that length.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{LbfgsbOptions, LbfgsbStop, lbfgsb_with};
///
/// // A quadratic with curvatures from 1 to 10⁶, held to [−0.5, 0.5]⁴; its unconstrained minimum
/// // is (0.8, −0.9, 0.3, 0.1).
/// let d = [1.0, 1e2, 1e4, 1e6];
/// let c = [0.8, -0.9, 0.3, 0.1];
/// let f = |x: &Matrix<f64>| (0..4).map(|i| 0.5 * d[i] * (x[(i, 0)] - c[i]).powi(2)).sum::<f64>();
/// let g = |x: &Matrix<f64>| Matrix::from_fn(4, 1, |i, _| d[i] * (x[(i, 0)] - c[i]));
/// let lower = Matrix::from_column_major(4, 1, vec![-0.5; 4]);
/// let upper = Matrix::from_column_major(4, 1, vec![0.5; 4]);
/// let opts = LbfgsbOptions::new().with_pgtol(1e-10).with_ftol(0.0);
/// let r = lbfgsb_with(f, g, &Matrix::zeros(4, 1), &lower, &upper, &opts).unwrap();
/// assert_eq!(r.stop, LbfgsbStop::ProjectedGradient);
/// assert_eq!(&r.x.as_slice()[..2], &[0.5, -0.5]);
/// assert!((r.x[(2, 0)] - 0.3).abs() < 1e-12 && (r.x[(3, 0)] - 0.1).abs() < 1e-12);
/// ```
#[track_caller]
pub fn lbfgsb_with<T: Real + Scalar>(
    mut f: impl FnMut(&Matrix<T>) -> T,
    mut g: impl FnMut(&Matrix<T>) -> Matrix<T>,
    x0: &Matrix<T>,
    lower: &Matrix<T>,
    upper: &Matrix<T>,
    opts: &LbfgsbOptions<T>,
) -> Option<LbfgsbResult<T>> {
    let n = column_length(x0, "the starting point");
    let lengths = (
        column_length(lower, "the lower bound"),
        column_length(upper, "the upper bound"),
    );
    assert!(
        lengths == (n, n),
        "the bounds are not of the length of the starting point: {} and {} for {} variables",
        lengths.0,
        lengths.1,
        n
    );
    if n == 0 || !opts.sane() || !all_finite(x0.as_slice()) {
        return None;
    }
    let (lo, hi) = (lower.as_slice(), upper.as_slice());
    // A NaN fails the first test; an infinite lower bound may only be −∞ and an upper one +∞.
    let empty = lo
        .iter()
        .zip(hi)
        .any(|(l, u)| !(ordered(*l, *u) || *l == *u) || *l == T::INFINITY || *u == -T::INFINITY);
    if empty {
        return None;
    }
    // Whether any variable has a bound, and whether every variable has two.
    let constrained = lo.iter().chain(hi).any(|b| finite(*b));
    let boxed = lo.iter().chain(hi).all(|b| finite(*b));

    let mut x = Matrix::from_column_major(
        n,
        1,
        x0.as_slice()
            .iter()
            .zip(lo.iter().zip(hi))
            .map(|(v, (l, u))| v.max(*l).min(*u))
            .collect(),
    );
    let mut value = f(&x);
    let mut evaluations = 1;
    if !finite(value) {
        return None;
    }
    let mut gradient = checked_gradient(g(&x), n);
    if !all_finite(gradient.as_slice()) {
        return None;
    }

    let mut memory = Memory::new(opts.memory, n);
    // The last step and the change of the gradient along it.
    let (mut s, mut y) = (vec![T::ZERO; n], vec![T::ZERO; n]);
    let mut work = Work::new(n);
    let mut projected = projected_norm(x.as_slice(), gradient.as_slice(), lo, hi);
    let mut iterations = 0;
    let stop = loop {
        if projected <= opts.pgtol {
            break LbfgsbStop::ProjectedGradient;
        }
        if iterations >= opts.max_iter {
            break LbfgsbStop::Iterations;
        }
        if evaluations >= opts.max_evaluations {
            break LbfgsbStop::Evaluations;
        }

        // The point the model leads to, and the line towards it. A model that cannot be
        // minimised, or a direction that does not lead down, is the memory's fault while there
        // is a memory, and the end of the method once there is not.
        let skip_cauchy = !constrained && !memory.is_empty();
        let target = trial_point(
            x.as_slice(),
            gradient.as_slice(),
            lo,
            hi,
            &memory,
            skip_cauchy,
        );
        let direction: Option<Vec<T>> = target
            .as_ref()
            .map(|t| t.iter().zip(x.as_slice()).map(|(t, x)| *t - *x).collect());
        let slope = direction
            .as_ref()
            .map_or(T::NAN, |d| dot(gradient.as_slice(), d));
        let (Some(target), Some(direction)) = (target, direction) else {
            if memory.is_empty() {
                break LbfgsbStop::LineSearch;
            }
            memory.reset();
            continue;
        };
        if !ordered(slope, T::ZERO) || !finite(slope) {
            if memory.is_empty() {
                break LbfgsbStop::LineSearch;
            }
            memory.reset();
            continue;
        }

        // The longest step that stays in the box, and the first one tried: on the very first
        // iteration of a problem with bounds the reference code goes no further than the point
        // of the model, and where some variable is unbounded it tries a step of unit length.
        let (cap, binding) = if iterations == 0 && constrained {
            (T::ONE, Vec::new())
        } else {
            longest_step(x.as_slice(), &direction, lo, hi)
        };
        let first = if iterations == 0 && !boxed {
            (T::ONE / norm2(&direction)).min(cap)
        } else {
            T::ONE.min(cap)
        };
        let search = LineSearch {
            decrease: T::from_f64(DECREASE),
            curvature: T::from_f64(CURVATURE),
            narrow: T::from_f64(NARROW),
            longest: cap,
            evaluations: SEARCH_EVALUATIONS.min(opts.max_evaluations - evaluations),
        };
        let mut segment = Segment {
            f: &mut f,
            g: &mut g,
            x: x.as_slice(),
            direction: &direction,
            target: &target,
            lower: lo,
            upper: hi,
            cap,
            binding: &binding,
            work: &mut work,
            calls: 0,
        };
        let outcome = more_thuente(&mut segment, value, slope, first, &search);
        evaluations += segment.calls;
        // A step is progress if the value it reaches is below the one it left. Near a minimum
        // the decrease can be lost in the rounding of `f` while the gradient still tells the
        // points apart; such a level step is progress when it satisfies the strong Wolfe
        // conditions and lowers the projected gradient, and it never counts towards the
        // reduction test. Where the gradient too is down to its rounding the projected gradient
        // stops falling, and so does the method.
        let decreased = ordered(outcome.value, value);
        let level = !decreased && outcome.found == Found::Wolfe && {
            let (point, new_gradient) = work.found(outcome.kept);
            let after = projected_norm(point.as_slice(), new_gradient.as_slice(), lo, hi);
            ordered(after, projected)
        };
        if outcome.found == Found::Nothing || !(decreased || level) {
            if evaluations >= opts.max_evaluations {
                break LbfgsbStop::Evaluations;
            }
            if memory.is_empty() {
                break LbfgsbStop::LineSearch;
            }
            memory.reset();
            continue;
        }

        let (point, new_gradient) = work.found(outcome.kept);
        for (s, (new, old)) in s.iter_mut().zip(point.as_slice().iter().zip(x.as_slice())) {
            *s = *new - *old;
        }
        for (y, (new, old)) in y
            .iter_mut()
            .zip(new_gradient.as_slice().iter().zip(gradient.as_slice()))
        {
            *y = *new - *old;
        }
        let descent = -dot(gradient.as_slice(), &s);
        x.as_mut_slice().copy_from_slice(point.as_slice());
        gradient
            .as_mut_slice()
            .copy_from_slice(new_gradient.as_slice());
        let value_before = value;
        value = outcome.value;
        iterations += 1;
        projected = projected_norm(x.as_slice(), gradient.as_slice(), lo, hi);
        if projected <= opts.pgtol {
            break LbfgsbStop::ProjectedGradient;
        }
        let scale = magnitude(value_before).max(magnitude(value)).max(T::ONE);
        if decreased && value_before - value <= opts.ftol * scale {
            break LbfgsbStop::Reduction;
        }

        // The curvature test: a pair whose sᵀy is not safely positive would make B indefinite,
        // and is left out. A memory whose matrix cannot be factored is started afresh.
        let sy = dot(&s, &y);
        if ordered(T::ZERO, sy) && sy > T::EPSILON * descent && !memory.push(&s, &y, sy) {
            memory.reset();
        }
    };

    let active = x
        .as_slice()
        .iter()
        .zip(lo.iter().zip(hi))
        .map(|(v, (l, u))| {
            if l == u {
                Bound::Fixed
            } else if v == l {
                Bound::Lower
            } else if v == u {
                Bound::Upper
            } else {
                Bound::Free
            }
        })
        .collect();
    Some(LbfgsbResult {
        x,
        value,
        gradient,
        projected_gradient: projected,
        active,
        iterations,
        evaluations,
        converged: matches!(stop, LbfgsbStop::ProjectedGradient | LbfgsbStop::Reduction),
        stop,
    })
}

/// The infinity norm of the projected gradient, `‖P(x − g) − x‖∞`: for each variable the
/// derivative, unless the variable is at a bound the derivative presses it against, in which case
/// as much of it as the room left on that side.
fn projected_norm<T: Real>(x: &[T], g: &[T], lower: &[T], upper: &[T]) -> T {
    x.iter()
        .zip(g)
        .zip(lower.iter().zip(upper))
        .fold(T::ZERO, |largest, ((x, g), (l, u))| {
            let p = if *g < T::ZERO {
                (*x - *u).max(*g)
            } else {
                (*x - *l).min(*g)
            };
            largest.max(magnitude(p))
        })
}

/// The longest step `α` for which `x + α·d` stays in the box, no longer than [`LONGEST`], and the
/// variables that reach their bound at it.
fn longest_step<T: Real>(x: &[T], d: &[T], lower: &[T], upper: &[T]) -> (T, Vec<usize>) {
    let mut cap = T::from_f64(LONGEST);
    let mut binding = Vec::new();
    for (i, (((x, d), l), u)) in x.iter().zip(d).zip(lower).zip(upper).enumerate() {
        let room = if *d < T::ZERO {
            *l - *x
        } else if *d > T::ZERO {
            *u - *x
        } else {
            continue;
        };
        if !finite(room) {
            continue;
        }
        let reach = (room / *d).max(T::ZERO);
        if reach < cap {
            cap = reach;
            binding.clear();
            binding.push(i);
        } else if reach == cap {
            binding.push(i);
        }
    }
    (cap, binding)
}

// ---------------------------------------------------------------------------------------------
// Small dense matrices
// ---------------------------------------------------------------------------------------------

/// The lower Cholesky factor of the symmetric matrix `a`, or `None` when a pivot is not
/// positive: the matrix is not positive definite, or not enough so for the arithmetic to tell.
/// Only the lower triangle is read, so row `i` of `a` needs no more than `i + 1` elements.
fn cholesky<T: Real>(a: &[Vec<T>]) -> Option<Vec<Vec<T>>> {
    let n = a.len();
    let mut l = vec![vec![T::ZERO; n]; n];
    for j in 0..n {
        let pivot = a[j][j] - dot(&l[j][..j], &l[j][..j]);
        if !ordered(T::ZERO, pivot) || !finite(pivot) {
            return None;
        }
        let diagonal = root(pivot);
        l[j][j] = diagonal;
        for i in j + 1..n {
            let below = (a[i][j] - dot(&l[i][..j], &l[j][..j])) / diagonal;
            l[i][j] = below;
        }
    }
    Some(l)
}

/// Solves `L·z = b` in place for the lower triangular `l`.
fn forward<T: Real>(l: &[Vec<T>], b: &mut [T]) {
    for i in 0..b.len() {
        let known = dot(&l[i][..i], &b[..i]);
        b[i] = (b[i] - known) / l[i][i];
    }
}

/// Solves `Lᵀ·z = b` in place for the lower triangular `l`.
fn backward<T: Real>(l: &[Vec<T>], b: &mut [T]) {
    for i in (0..b.len()).rev() {
        let mut known = T::ZERO;
        for (k, row) in l.iter().enumerate().skip(i + 1) {
            known += row[i] * b[k];
        }
        b[i] = (b[i] - known) / l[i][i];
    }
}

// ---------------------------------------------------------------------------------------------
// The limited memory
// ---------------------------------------------------------------------------------------------

/// The pairs `(s, y)` of the last steps, oldest first, and what the compact form of the matrix
/// they define needs of them.
struct Memory<T> {
    /// The most pairs kept.
    capacity: usize,
    /// The number of variables, the length of each `s` and `y`.
    n: usize,
    /// The pairs kept, and the slot of the oldest: pair `k` is in slot `(head + k) % capacity`.
    len: usize,
    head: usize,
    /// The steps and the changes of the gradient along them, a slot of `n` elements each, in
    /// two allocations made as the memory first fills, written over from then on.
    s_data: Vec<T>,
    y_data: Vec<T>,
    /// `SᵀY`: row `a` holds `sₐᵀ·y_b` for every `b`. Its diagonal is `D`, its strict lower
    /// triangle `L`.
    sy: Vec<Vec<T>>,
    /// `SᵀS`.
    ss: Vec<Vec<T>>,
    /// `YᵀY`, for the reduced matrix while every variable is free.
    yy: Vec<Vec<T>>,
    /// `θ`, the scale of the identity the matrix starts from: `yᵀy/sᵀy` of the last pair.
    theta: T,
    /// The lower Cholesky factor `J` of `θSᵀS + L·D⁻¹·Lᵀ`, through which `M` is applied.
    factor: Vec<Vec<T>>,
}

impl<T: Real> Memory<T> {
    fn new(capacity: usize, n: usize) -> Self {
        Self {
            capacity,
            n,
            len: 0,
            head: 0,
            s_data: Vec::new(),
            y_data: Vec::new(),
            sy: Vec::new(),
            ss: Vec::new(),
            yy: Vec::new(),
            theta: T::ONE,
            factor: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The step of pair `k`, oldest first.
    fn s(&self, k: usize) -> &[T] {
        let slot = (self.head + k) % self.capacity;
        &self.s_data[slot * self.n..(slot + 1) * self.n]
    }

    /// The change of the gradient of pair `k`, oldest first.
    fn y(&self, k: usize) -> &[T] {
        let slot = (self.head + k) % self.capacity;
        &self.y_data[slot * self.n..(slot + 1) * self.n]
    }

    /// Forgets every pair: the matrix is the identity again. The slots are kept.
    fn reset(&mut self) {
        self.len = 0;
        self.head = 0;
        self.sy.clear();
        self.ss.clear();
        self.yy.clear();
        self.factor.clear();
        self.theta = T::ONE;
    }

    /// Takes the pair `(s, y)` with `sᵀy = sy > 0`, in place of the oldest if the memory is
    /// full. Returns false when the matrix the pairs make can no longer be factored, which the
    /// caller answers by starting afresh.
    fn push(&mut self, s: &[T], y: &[T], sy: T) -> bool {
        let y_y = dot(y, y);
        let theta = y_y / sy;
        if !ordered(T::ZERO, theta) || !finite(theta) {
            // A pair whose curvature the arithmetic cannot express teaches nothing.
            return true;
        }
        if self.len == self.capacity {
            self.head = (self.head + 1) % self.capacity;
            self.len -= 1;
            self.sy.remove(0);
            self.ss.remove(0);
            self.yy.remove(0);
            for row in self
                .sy
                .iter_mut()
                .chain(self.ss.iter_mut())
                .chain(self.yy.iter_mut())
            {
                row.remove(0);
            }
        }
        // The new row and column of SᵀY, SᵀS and YᵀY, O(m·n): for each pair kept, its four
        // products with the new pair in one pass, each summed as `dot` sums it (from −0, over
        // the variables in their order); `sₖᵀs` and `yₖᵀy` stand in both the row and the column.
        let mut new_sy = Vec::with_capacity(self.len + 1);
        let mut new_ss = Vec::with_capacity(self.len + 1);
        let mut new_yy = Vec::with_capacity(self.len + 1);
        for k in 0..self.len {
            let (sk, yk) = (self.s(k), self.y(k));
            let (mut sk_y, mut s_yk, mut sk_s, mut yk_y) = (-T::ZERO, -T::ZERO, -T::ZERO, -T::ZERO);
            for (((&a, &b), &c), &d) in sk.iter().zip(yk).zip(s).zip(y) {
                sk_y += a * d;
                s_yk += c * b;
                sk_s += a * c;
                yk_y += b * d;
            }
            self.sy[k].push(sk_y);
            new_sy.push(s_yk);
            self.ss[k].push(sk_s);
            new_ss.push(sk_s);
            self.yy[k].push(yk_y);
            new_yy.push(yk_y);
        }
        new_sy.push(sy);
        self.sy.push(new_sy);
        new_ss.push(dot(s, s));
        self.ss.push(new_ss);
        new_yy.push(y_y);
        self.yy.push(new_yy);
        // The new pair into the next slot, the oldest one's when the memory was full.
        let slot = (self.head + self.len) % self.capacity;
        let n = self.n;
        if self.s_data.len() < (slot + 1) * n {
            self.s_data.extend_from_slice(s);
            self.y_data.extend_from_slice(y);
        } else {
            self.s_data[slot * n..(slot + 1) * n].copy_from_slice(s);
            self.y_data[slot * n..(slot + 1) * n].copy_from_slice(y);
        }
        self.len += 1;
        self.theta = theta;
        self.refactor()
    }

    /// Factors `θSᵀS + L·D⁻¹·Lᵀ`, which is positive definite while every `sᵢᵀyᵢ` is positive and
    /// the steps are independent.
    fn refactor(&mut self) -> bool {
        let col = self.len();
        // Its lower triangle, which is all the factorisation reads; (L·D⁻¹·Lᵀ)ₐ_b is
        // Σ_{k < b} Lₐₖ·L_bₖ/Dₖ, over the k below both a and b.
        let t: Vec<Vec<T>> = (0..col)
            .map(|a| {
                (0..=a)
                    .map(|b| {
                        self.theta * self.ss[a][b]
                            + (0..b)
                                .map(|k| self.sy[a][k] * self.sy[b][k] / self.sy[k][k])
                                .sum::<T>()
                    })
                    .collect()
            })
            .collect();
        match cholesky(&t) {
            Some(factor) => {
                self.factor = factor;
                true
            }
            None => false,
        }
    }

    /// `M·v` for a vector `v = (v₁, v₂)` of `2m` elements, where `M⁻¹ = [−D, Lᵀ; L, θSᵀS]`. With
    /// `J·Jᵀ = θSᵀS + L·D⁻¹·Lᵀ`, `M⁻¹` is the product of `[−D^½, 0; L·D^-½, J]` and
    /// `[D^½, −D^-½·Lᵀ; 0, Jᵀ]` (section 3 of the 1995 paper), and its inverse is applied by
    /// four triangular solves: `w₂ = J⁻ᵀ·J⁻¹·(v₂ + L·D⁻¹·v₁)` and `w₁ = D⁻¹·(Lᵀ·w₂ − v₁)`.
    fn times_m(&self, v: &[T]) -> Vec<T> {
        let col = self.len();
        let (v1, v2) = v.split_at(col);
        let diagonal = |k: usize| self.sy[k][k];
        let mut w2: Vec<T> = (0..col)
            .map(|a| {
                let mut sum = v2[a];
                for (b, v) in v1.iter().enumerate().take(a) {
                    sum += self.sy[a][b] * *v / diagonal(b);
                }
                sum
            })
            .collect();
        forward(&self.factor, &mut w2);
        backward(&self.factor, &mut w2);
        let w1 = (0..col).map(|b| {
            let mut sum = -v1[b];
            for (a, w) in w2.iter().enumerate().skip(b + 1) {
                sum += self.sy[a][b] * *w;
            }
            sum / diagonal(b)
        });
        let mut w: Vec<T> = w1.collect();
        w.extend(w2);
        w
    }

    /// The row `i` of `W = [Y, θS]`.
    fn w_row(&self, i: usize) -> Vec<T> {
        let theta = self.theta;
        (0..self.len)
            .map(|k| self.y(k)[i])
            .chain((0..self.len).map(|k| theta * self.s(k)[i]))
            .collect()
    }

    /// `Wᵀ·v` over the variables `indices`, ascending, `v` holding one element for each of
    /// them: element `k` is `Σ wₖ[i]·v` summed as `dot` sums it, from −0 over `indices` in
    /// their order, times 1 for a column of `Y` and `θ` for one of `θS`. The columns are taken
    /// four at a time, in one pass over `v` each, their sums side by side.
    fn w_transpose_times(&self, indices: &[usize], v: &[T]) -> Vec<T> {
        let col = self.len;
        let column = |c: usize| if c < col { self.y(c) } else { self.s(c - col) };
        let scale = |c: usize| if c < col { T::ONE } else { self.theta };
        // Every variable, in order: the columns read straight through.
        let all = indices.len() == self.n;
        let mut out = Vec::with_capacity(2 * col);
        let mut c = 0;
        while c + 4 <= 2 * col {
            let w = [column(c), column(c + 1), column(c + 2), column(c + 3)];
            let mut sums = [-T::ZERO; 4];
            if all {
                for (j, &vj) in v.iter().enumerate() {
                    for (sum, w) in sums.iter_mut().zip(&w) {
                        *sum += w[j] * vj;
                    }
                }
            } else {
                for (&i, &vj) in indices.iter().zip(v) {
                    for (sum, w) in sums.iter_mut().zip(&w) {
                        *sum += w[i] * vj;
                    }
                }
            }
            for (k, sum) in sums.into_iter().enumerate() {
                out.push(scale(c + k) * sum);
            }
            c += 4;
        }
        for c in c..2 * col {
            let w = column(c);
            let sum: T = indices.iter().zip(v).map(|(i, v)| w[*i] * *v).sum();
            out.push(scale(c) * sum);
        }
        out
    }

    /// The element `i` of `W·v` for a vector `v` of `2m` elements.
    fn w_times_at(&self, i: usize, v: &[T]) -> T {
        let col = self.len;
        let mut sum = T::ZERO;
        for k in 0..col {
            sum += self.y(k)[i] * v[k] + self.theta * self.s(k)[i] * v[col + k];
        }
        sum
    }
}

// ---------------------------------------------------------------------------------------------
// The generalised Cauchy point
// ---------------------------------------------------------------------------------------------

/// A point at which the projected steepest descent path bends: the time `t` at which variable
/// `index` reaches its bound. Ordered so that a `BinaryHeap` gives the earliest first.
struct Breakpoint<T> {
    time: T,
    index: usize,
}

impl<T: Real> Ord for Breakpoint<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Times are never NaN: a finite distance over a nonzero finite slope.
        other
            .time
            .partial_cmp(&self.time)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl<T: Real> PartialOrd for Breakpoint<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Real> PartialEq for Breakpoint<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<T: Real> Eq for Breakpoint<T> {}

/// The generalised Cauchy point `xᶜ`, `c = Wᵀ·(xᶜ − x)`, and where each variable is there.
struct Cauchy<T> {
    point: Vec<T>,
    c: Vec<T>,
    status: Vec<Bound>,
}

/// The first minimiser of the quadratic model `gᵀz + ½zᵀBz` along the projected steepest
/// descent path `z(t) = P(x − t·g) − x`: Algorithm CP of Byrd, Lu, Nocedal and Zhu (1995).
///
/// Along each straight piece of the path the model is a parabola in `t`, `f′` its slope and `f″`
/// its curvature; at each breakpoint the variable that reaches its bound is fixed there and both
/// are updated from the row of `W` for that variable, `O(m²)` a breakpoint. The search stops in
/// the first piece whose parabola has its minimum before the piece ends, or at the last breakpoint
/// when every moving variable has met a bound. `f″` is held above `ε` times its first value, as
/// the reference code holds it, against the rounding that could otherwise make it vanish.
fn cauchy_point<T: Real>(
    x: &[T],
    g: &[T],
    lower: &[T],
    upper: &[T],
    memory: &Memory<T>,
) -> Cauchy<T> {
    let n = x.len();
    let col = memory.len();
    let theta = memory.theta;
    let mut status = vec![Bound::Free; n];
    let mut d = vec![T::ZERO; n];
    let mut breakpoints = Vec::new();
    // Whether some moving variable has no bound ahead of it, so that the path never ends.
    let mut open = false;
    let mut slope = T::ZERO;
    for i in 0..n {
        let (l, u, gi) = (lower[i], upper[i], g[i]);
        if l == u {
            status[i] = Bound::Fixed;
        } else if x[i] <= l && gi >= T::ZERO {
            status[i] = Bound::Lower;
        } else if x[i] >= u && gi <= T::ZERO {
            status[i] = Bound::Upper;
        } else if gi != T::ZERO {
            d[i] = -gi;
            slope -= gi * gi;
            let room = if gi > T::ZERO { x[i] - l } else { u - x[i] };
            if finite(room) {
                breakpoints.push(Breakpoint {
                    time: room / magnitude(gi),
                    index: i,
                });
            } else {
                open = true;
            }
        }
    }
    let mut point = x.to_vec();
    let mut c = vec![T::ZERO; 2 * col];
    if slope == T::ZERO {
        // The projected gradient is zero: x is its own Cauchy point.
        return Cauchy { point, c, status };
    }

    // p = Wᵀd, and the slope and the curvature of the model at t = 0.
    let moving: Vec<usize> = (0..n).filter(|&i| d[i] != T::ZERO).collect();
    let along: Vec<T> = moving.iter().map(|&i| d[i]).collect();
    let mut p = memory.w_transpose_times(&moving, &along);
    let first_curvature = -theta * slope;
    let mut curvature = first_curvature;
    if col > 0 {
        curvature -= dot(&memory.times_m(&p), &p);
    }
    let floor = T::EPSILON * first_curvature;
    curvature = curvature.max(floor);
    let mut minimum = -slope / curvature;

    let total = breakpoints.len();
    let mut heap = BinaryHeap::from(breakpoints);
    let mut time = T::ZERO;
    let mut passed = 0;
    let mut ended = false;
    while let Some(Breakpoint { time: at, index: b }) = heap.pop() {
        let piece = at - time;
        if minimum < piece {
            break;
        }
        // The path reaches the bound of variable b: fix it there.
        time = at;
        passed += 1;
        let db = d[b];
        let (bound, side) = if db > T::ZERO {
            (upper[b], Bound::Upper)
        } else {
            (lower[b], Bound::Lower)
        };
        point[b] = bound;
        status[b] = side;
        d[b] = T::ZERO;
        let zb = bound - x[b];
        for (ci, pi) in c.iter_mut().zip(&p) {
            *ci += piece * *pi;
        }
        if passed == total && !open {
            // Every moving variable is at a bound: the path ends here.
            ended = true;
            break;
        }
        // f′ and f″ on the next piece (the 1995 paper, (4.9) and (4.10) with d_b = −g_b).
        slope = slope + piece * curvature + db * db - theta * db * zb;
        curvature -= theta * db * db;
        if col > 0 {
            let wb = memory.w_row(b);
            let mw = memory.times_m(&wb);
            slope += db * dot(&mw, &c);
            curvature += T::TWO * db * dot(&mw, &p) - db * db * dot(&mw, &wb);
            for (pi, wi) in p.iter_mut().zip(&wb) {
                *pi -= db * *wi;
            }
        }
        curvature = curvature.max(floor);
        minimum = -slope / curvature;
    }
    let rest = if ended { T::ZERO } else { minimum.max(T::ZERO) };
    time += rest;
    for i in 0..n {
        if d[i] != T::ZERO {
            point[i] = (x[i] + time * d[i]).max(lower[i]).min(upper[i]);
        }
    }
    for (ci, pi) in c.iter_mut().zip(&p) {
        *ci += rest * *pi;
    }
    Cauchy { point, c, status }
}

// ---------------------------------------------------------------------------------------------
// The subspace step
// ---------------------------------------------------------------------------------------------

/// The factors of the matrix `K` of the direct primal method,
///
/// `K = [−D − YᵀZZᵀY/θ, L_aᵀ − R_zᵀ; L_a − R_z, θ·SᵀAAᵀS]`,
///
/// `Z` the free variables, `A` the others, `L_a` the strict lower triangle of `SᵀAAᵀY` and `R_z`
/// the upper triangle of `SᵀZZᵀY` with its diagonal. With `P = D + YᵀZZᵀY/θ = J₁·J₁ᵀ`,
/// `Q = L_a − R_z`, `X = J₁⁻¹·Qᵀ` and `J₂·J₂ᵀ = θ·SᵀAAᵀS + XᵀX`, it is `K = L·E·Lᵀ` with
/// `L = [J₁, 0; −Xᵀ, J₂]` and `E = [−I, 0; 0, I]`.
struct Reduced<T> {
    j1: Vec<Vec<T>>,
    j2: Vec<Vec<T>>,
    x: Vec<Vec<T>>,
}

impl<T: Real> Reduced<T> {
    /// Forms and factors `K` for the free variables `free`, or `None` when it is singular.
    fn new(memory: &Memory<T>, free: &[bool]) -> Option<Self> {
        let col = memory.len();
        let theta = memory.theta;
        // YᵀZZᵀY and SᵀZZᵀY over the free variables, SᵀAAᵀY and SᵀAAᵀS over the others:
        // O(m²·n) in all. Of SᵀZZᵀY only the upper triangle (diagonal included) is read below,
        // of SᵀAAᵀY only the strict lower one, and the other two are symmetric: each sum is
        // formed once, over the variables in their order. With every variable free (no bound active, as
        // throughout an unbounded problem) they are `YᵀY` and `SᵀY`, which the memory keeps,
        // summed in the same order, and the others vanish.
        let mut yy = vec![vec![T::ZERO; col]; col];
        let mut sy_free = vec![vec![T::ZERO; col]; col];
        let mut sy_held = vec![vec![T::ZERO; col]; col];
        let mut ss_held = vec![vec![T::ZERO; col]; col];
        if free.iter().all(|f| *f) {
            for a in 0..col {
                yy[a].copy_from_slice(&memory.yy[a]);
                sy_free[a].copy_from_slice(&memory.sy[a]);
            }
        } else {
            for a in 0..col {
                let (sa, ya) = (memory.s(a), memory.y(a));
                for b in a..col {
                    let (yb, sb) = (memory.y(b), memory.s(b));
                    let (mut f_yy, mut f_sy, mut h_ss) = (T::ZERO, T::ZERO, T::ZERO);
                    for (i, is_free) in free.iter().enumerate() {
                        if *is_free {
                            f_yy += ya[i] * yb[i];
                            f_sy += sa[i] * yb[i];
                        } else {
                            h_ss += sa[i] * sb[i];
                        }
                    }
                    yy[a][b] = f_yy;
                    yy[b][a] = f_yy;
                    sy_free[a][b] = f_sy;
                    ss_held[a][b] = h_ss;
                    ss_held[b][a] = h_ss;
                }
                for (b, held) in sy_held[a].iter_mut().enumerate().take(a) {
                    let yb = memory.y(b);
                    let mut h_sy = T::ZERO;
                    for (i, is_free) in free.iter().enumerate() {
                        if !*is_free {
                            h_sy += sa[i] * yb[i];
                        }
                    }
                    *held = h_sy;
                }
            }
        }
        let p: Vec<Vec<T>> = (0..col)
            .map(|a| {
                (0..col)
                    .map(|b| {
                        let d = if a == b { memory.sy[a][a] } else { T::ZERO };
                        d + yy[a][b] / theta
                    })
                    .collect()
            })
            .collect();
        let j1 = cholesky(&p)?;
        // Q = L_a − R_z, and X = J₁⁻¹·Qᵀ a column at a time: column c of X solves J₁·x = row c
        // of Q.
        let q = |a: usize, b: usize| {
            if a > b { sy_held[a][b] } else { -sy_free[a][b] }
        };
        let columns: Vec<Vec<T>> = (0..col)
            .map(|c| {
                let mut column: Vec<T> = (0..col).map(|b| q(c, b)).collect();
                forward(&j1, &mut column);
                column
            })
            .collect();
        let x: Vec<Vec<T>> = (0..col)
            .map(|r| columns.iter().map(|column| column[r]).collect())
            .collect();
        let schur: Vec<Vec<T>> = (0..col)
            .map(|a| {
                (0..col)
                    .map(|b| {
                        let cross: T = x.iter().map(|row| row[a] * row[b]).sum();
                        theta * ss_held[a][b] + cross
                    })
                    .collect()
            })
            .collect();
        let j2 = cholesky(&schur)?;
        Some(Self { j1, j2, x })
    }

    /// Solves `K·v = b` in place: `L·z = b`, `z₁ ← −z₁`, `Lᵀ·v = z`.
    fn solve(&self, b: &mut [T]) {
        let col = self.j1.len();
        let (b1, b2) = b.split_at_mut(col);
        forward(&self.j1, b1);
        for (a, value) in b2.iter_mut().enumerate() {
            let cross: T = self
                .x
                .iter()
                .zip(b1.iter())
                .map(|(row, z)| row[a] * *z)
                .sum();
            *value += cross;
        }
        forward(&self.j2, b2);
        for value in b1.iter_mut() {
            *value = -*value;
        }
        backward(&self.j2, b2);
        for (r, value) in b1.iter_mut().enumerate() {
            *value += dot(&self.x[r], b2);
        }
        backward(&self.j1, b1);
    }
}

/// The point the iteration searches towards: the generalised Cauchy point, moved by the
/// minimiser of the model over the variables that are free there, projected onto the box or
/// cut back to it. `None` when the reduced model cannot be minimised.
fn trial_point<T: Real>(
    x: &[T],
    g: &[T],
    lower: &[T],
    upper: &[T],
    memory: &Memory<T>,
    skip_cauchy: bool,
) -> Option<Vec<T>> {
    let n = x.len();
    let col = memory.len();
    // Without bounds, and with a memory, the Cauchy point would only be moved on to the minimiser
    // of the whole model; the reference code starts from x itself, which is the L-BFGS step.
    let cauchy = if skip_cauchy {
        Cauchy {
            point: x.to_vec(),
            c: vec![T::ZERO; 2 * col],
            status: vec![Bound::Free; n],
        }
    } else {
        cauchy_point(x, g, lower, upper, memory)
    };
    let free_flags: Vec<bool> = cauchy.status.iter().map(|s| *s == Bound::Free).collect();
    let free: Vec<usize> = (0..n).filter(|&i| free_flags[i]).collect();
    let mut target = cauchy.point.clone();
    if free.is_empty() {
        return Some(target);
    }
    let theta = memory.theta;
    // The reduced gradient of the model at the Cauchy point, negated:
    // r = −Zᵀ·(g + θ·(xᶜ − x) − W·M·c).
    // With `c` zero, as when the Cauchy point is x itself, `M·c` is zero and so is every
    // `(W·M·c)ᵢ`: a sum from +0 of products of finite numbers with zeros, which is +0 (the
    // memory holds only finite pairs, and a factor whose elements are finite). It is not formed.
    let wmc_needed = col > 0 && cauchy.c.iter().any(|c| *c != T::ZERO);
    let mc = if wmc_needed {
        memory.times_m(&cauchy.c)
    } else {
        Vec::new()
    };
    let r: Vec<T> = free
        .iter()
        .map(|&i| {
            let wmc = if wmc_needed {
                memory.w_times_at(i, &mc)
            } else {
                T::ZERO
            };
            -(g[i] + theta * (cauchy.point[i] - x[i]) - wmc)
        })
        .collect();
    // The minimiser over the free variables, (ZᵀBZ)⁻¹·r = r/θ + Zᵀ·W·K⁻¹·Wᵀ·Z·r/θ².
    let step: Vec<T> = if col == 0 {
        r.iter().map(|v| *v / theta).collect()
    } else {
        let mut v = memory.w_transpose_times(&free, &r);
        Reduced::new(memory, &free_flags)?.solve(&mut v);
        free.iter()
            .zip(&r)
            .map(|(&i, r)| {
                let mut sum = *r;
                for k in 0..col {
                    sum += memory.y(k)[i] * v[k] / theta + memory.s(k)[i] * v[col + k];
                }
                sum / theta
            })
            .collect()
    };
    if !all_finite(&step) {
        return None;
    }

    // Morales and Nocedal: project the minimiser onto the box, and keep the projection if it
    // still leads downhill from x; otherwise take the longest part of the step from the Cauchy
    // point that stays inside.
    let mut clipped = false;
    for (&i, step) in free.iter().zip(&step) {
        let moved = cauchy.point[i] + *step;
        let kept = moved.max(lower[i]).min(upper[i]);
        clipped |= kept != moved;
        target[i] = kept;
    }
    if clipped {
        let slope: T = (0..n).map(|i| (target[i] - x[i]) * g[i]).sum();
        if slope > T::ZERO {
            let mut fraction = T::ONE;
            let mut binding = None;
            for (j, (&i, step)) in free.iter().zip(&step).enumerate() {
                let room = if *step < T::ZERO {
                    lower[i] - cauchy.point[i]
                } else if *step > T::ZERO {
                    upper[i] - cauchy.point[i]
                } else {
                    continue;
                };
                let reach = room / *step;
                if reach < fraction {
                    fraction = reach.max(T::ZERO);
                    binding = Some(j);
                }
            }
            for (j, (&i, step)) in free.iter().zip(&step).enumerate() {
                target[i] = if binding == Some(j) {
                    if *step > T::ZERO { upper[i] } else { lower[i] }
                } else {
                    (cauchy.point[i] + fraction * *step)
                        .max(lower[i])
                        .min(upper[i])
                };
            }
        }
    }
    Some(target)
}

// ---------------------------------------------------------------------------------------------
// The line search
// ---------------------------------------------------------------------------------------------

/// The points and gradients the line search moves between: those of the last trial and those of
/// the trial last kept.
struct Work<T> {
    point: Matrix<T>,
    gradient: Matrix<T>,
    kept_point: Matrix<T>,
    kept_gradient: Matrix<T>,
}

impl<T: Real + Scalar> Work<T> {
    fn new(n: usize) -> Self {
        Self {
            point: Matrix::zeros(n, 1),
            gradient: Matrix::zeros(n, 1),
            kept_point: Matrix::zeros(n, 1),
            kept_gradient: Matrix::zeros(n, 1),
        }
    }

    /// The point and the gradient of the step a search came back with.
    fn found(&self, kept: bool) -> (&Matrix<T>, &Matrix<T>) {
        if kept {
            (&self.kept_point, &self.kept_gradient)
        } else {
            (&self.point, &self.gradient)
        }
    }
}

/// The segment `x + α·d`, `0 ≤ α ≤ cap`, of an iteration, for the line search. At `α = 1` it is
/// the trial point itself, exactly; elsewhere it is held in the box against rounding, and at the
/// cap the variables that limit it are put on their bounds exactly.
struct Segment<'a, T, F, G> {
    f: &'a mut F,
    g: &'a mut G,
    x: &'a [T],
    direction: &'a [T],
    target: &'a [T],
    lower: &'a [T],
    upper: &'a [T],
    cap: T,
    binding: &'a [usize],
    work: &'a mut Work<T>,
    /// The evaluations made.
    calls: usize,
}

impl<T, F, G> Line<T> for Segment<'_, T, F, G>
where
    T: Real + Scalar,
    F: FnMut(&Matrix<T>) -> T,
    G: FnMut(&Matrix<T>) -> Matrix<T>,
{
    fn at(&mut self, step: T) -> Option<(T, T)> {
        let point = self.work.point.as_mut_slice();
        if step == T::ONE {
            point.copy_from_slice(self.target);
        } else {
            for (((p, x), d), (l, u)) in point
                .iter_mut()
                .zip(self.x)
                .zip(self.direction)
                .zip(self.lower.iter().zip(self.upper))
            {
                *p = (*x + step * *d).max(*l).min(*u);
            }
            if step >= self.cap {
                for &i in self.binding {
                    point[i] = if self.direction[i] < T::ZERO {
                        self.lower[i]
                    } else {
                        self.upper[i]
                    };
                }
            }
        }
        self.calls += 1;
        let value = (self.f)(&self.work.point);
        if !finite(value) {
            return None;
        }
        let gradient = checked_gradient((self.g)(&self.work.point), self.x.len());
        if !all_finite(gradient.as_slice()) {
            return None;
        }
        let slope = dot(gradient.as_slice(), self.direction);
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
