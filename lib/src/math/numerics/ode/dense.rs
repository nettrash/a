//! Dense output: the solution between the steps.
//!
//! Every method of this module has a continuous extension — a polynomial over each accepted step
//! that agrees with the step at its ends and approximates the solution in between to an order
//! that goes with the method's own. A [`DenseStep`] is that polynomial for one step, and a
//! [`DenseOutput`] is the chain of them over an integration, which evaluates the solution at any
//! time the integration covered.
//!
//! The polynomials of the different methods come in different shapes: Hairer's nested products in
//! `θ` and `1 − θ` for the Dormand–Prince pairs, a Hermite cubic for the classical method, the
//! collocation polynomial of an implicit method, the backward differences of a multistep one. All
//! of them are Newton forms on some set of nodes, and that is how a step is kept:
//!
//! ```text
//! y(t₀ + θ·h) = c₀ + (θ − x₀)·(c₁ + (θ − x₁)·(c₂ + … + (θ − x₍ₖ₋₁₎)·cₖ))
//! ```
//!
//! with vectors `c₀ … cₖ` and nodes `x₀ … x₍ₖ₋₁₎` that the method chooses — `0, 1, 0, 1, …` gives
//! the nested products, `0, 0, 1` the Hermite cubic — and evaluated by Horner's rule from the
//! inside out, which is the same arithmetic as the form the method's authors wrote down.

use crate::math::matrix::Matrix;
use crate::math::numerics::Real;

/// The continuous extension of one accepted step: a polynomial in `θ = (t − t₀)/(t₁ − t₀)` with
/// vector coefficients, in Newton form.
///
/// A method fills one in [`OdeMethod::dense`](super::OdeMethod::dense): [`reset`](Self::reset)
/// fixes the step, the size of the state and the nodes, and the coefficients are then written
/// through [`coefficient_mut`](Self::coefficient_mut). A caller only evaluates it.
///
/// # Examples
///
/// ```
/// use a::math::numerics::DenseStep;
///
/// // The parabola through (1, 3), (2, 5) and (3, 11), as a step from t = 1 to t = 3: in θ it
/// // passes through 3, 5 and 11 at 0, ½ and 1, and on the nodes 0 and ½ its divided differences
/// // are 3, 4 and 8.
/// let mut step = DenseStep::<f64>::new();
/// step.reset(1.0, 3.0, 1, &[0.0, 0.5]);
/// step.coefficient_mut(0)[0] = 3.0;
/// step.coefficient_mut(1)[0] = 4.0;
/// step.coefficient_mut(2)[0] = 8.0;
/// assert_eq!(step.degree(), 2);
/// assert_eq!(step.eval(1.0)[(0, 0)], 3.0);
/// assert_eq!(step.eval(2.0)[(0, 0)], 5.0);
/// assert_eq!(step.eval(3.0)[(0, 0)], 11.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DenseStep<T> {
    /// The start of the step.
    t0: T,
    /// The end of the step.
    t1: T,
    /// The number of components of the state.
    n: usize,
    /// The nodes of the Newton form, in units of `θ`: one fewer than there are coefficients.
    nodes: Vec<T>,
    /// The coefficients `c₀ … cₖ`, each of `n` numbers, one after the other.
    coefficients: Vec<T>,
}

impl<T: Real> Default for DenseStep<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> DenseStep<T> {
    /// A step of no length for a state of no components, to be [`reset`](Self::reset) by a
    /// method.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::DenseStep;
    ///
    /// let step = DenseStep::<f64>::new();
    /// assert_eq!(step.dimension(), 0);
    /// assert_eq!(step.degree(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        DenseStep {
            t0: T::ZERO,
            t1: T::ZERO,
            n: 0,
            nodes: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    /// Makes this the polynomial of the step from `t0` to `t1` for a state of `n` components, in
    /// Newton form on `nodes`, with all its `nodes.len() + 1` coefficients zero. The storage of
    /// the step is reused, so that a method writing one step after another allocates once.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::DenseStep;
    ///
    /// // A straight line from (0, [1, 2]) to (2, [3, 6]): one node, two coefficients.
    /// let mut step = DenseStep::<f64>::new();
    /// step.reset(0.0, 2.0, 2, &[0.0]);
    /// step.coefficient_mut(0).copy_from_slice(&[1.0, 2.0]);
    /// step.coefficient_mut(1).copy_from_slice(&[2.0, 4.0]);
    /// assert_eq!(step.eval(1.0).as_slice(), &[2.0, 4.0]);
    /// ```
    pub fn reset(&mut self, t0: T, t1: T, n: usize, nodes: &[T]) {
        self.t0 = t0;
        self.t1 = t1;
        self.n = n;
        self.nodes.clear();
        self.nodes.extend_from_slice(nodes);
        self.coefficients.clear();
        self.coefficients.resize(n * (nodes.len() + 1), T::ZERO);
    }

    /// The time the step starts at.
    pub fn t0(&self) -> T {
        self.t0
    }

    /// The time the step ends at.
    pub fn t1(&self) -> T {
        self.t1
    }

    /// The number of components of the state.
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// The degree of the polynomial: the number of its nodes.
    pub fn degree(&self) -> usize {
        self.nodes.len()
    }

    /// The nodes of the Newton form, in units of `θ`.
    pub fn nodes(&self) -> &[T] {
        &self.nodes
    }

    /// The coefficient `cₖ`, one number for each component of the state.
    ///
    /// # Panics
    ///
    /// Panics if `k` is above the degree.
    #[track_caller]
    pub fn coefficient(&self, k: usize) -> &[T] {
        assert!(k <= self.nodes.len(), "the coefficient is above the degree");
        &self.coefficients[k * self.n..(k + 1) * self.n]
    }

    /// The coefficient `cₖ`, to write.
    ///
    /// # Panics
    ///
    /// Panics if `k` is above the degree.
    #[track_caller]
    pub fn coefficient_mut(&mut self, k: usize) -> &mut [T] {
        assert!(k <= self.nodes.len(), "the coefficient is above the degree");
        &mut self.coefficients[k * self.n..(k + 1) * self.n]
    }

    /// The polynomial at the time `t`, as a column.
    ///
    /// Inside the step this is the method's continuous extension. Outside it the polynomial is
    /// still evaluated, and is an extrapolation whose error grows quickly with the distance.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{DenseStep, DormandPrince853, OdeMethod, OdeOptions};
    ///
    /// // One step of the method of order 8 on y′ = y, and the solution in the middle of it,
    /// // where the continuous extension of order 7 is good to some 1e-11 for this step.
    /// let mut f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = y[(0, 0)];
    /// let mut method = DormandPrince853::new();
    /// let opts = OdeOptions::new().with_h0(0.25);
    /// method.start(&mut f, 0.0, &Matrix::from_rows(&[[1.0]]), 0.25, &opts).unwrap();
    /// method.step(&mut f).unwrap();
    /// let mut step = DenseStep::new();
    /// method.dense(&mut f, &mut step).unwrap();
    /// assert!((step.eval(0.125)[(0, 0)] - 0.125f64.exp()).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn eval(&self, t: T) -> Matrix<T> {
        let mut out = vec![T::ZERO; self.n];
        self.eval_into(t, &mut out);
        Matrix::from_column_major(self.n, 1, out)
    }

    /// The polynomial at the time `t`, written to `out`.
    ///
    /// # Panics
    ///
    /// Panics if `out` does not have one element for each component of the state.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::DenseStep;
    ///
    /// let mut step = DenseStep::<f64>::new();
    /// step.reset(0.0, 1.0, 1, &[0.0]);
    /// step.coefficient_mut(0)[0] = 1.0;
    /// step.coefficient_mut(1)[0] = -3.0;
    /// let mut out = [0.0];
    /// step.eval_into(0.5, &mut out);
    /// assert_eq!(out, [-0.5]);
    /// ```
    #[track_caller]
    pub fn eval_into(&self, t: T, out: &mut [T]) {
        assert_eq!(
            out.len(),
            self.n,
            "the output does not have the size of the state"
        );
        let n = self.n;
        let h = self.t1 - self.t0;
        let theta = if h == T::ZERO {
            T::ZERO
        } else {
            (t - self.t0) / h
        };
        // Horner's rule on the Newton form, from the innermost coefficient outwards.
        let k = self.nodes.len();
        out.copy_from_slice(&self.coefficients[k * n..(k + 1) * n]);
        for j in (0..k).rev() {
            let factor = theta - self.nodes[j];
            let c = &self.coefficients[j * n..(j + 1) * n];
            for (o, &cj) in out.iter_mut().zip(c) {
                *o = cj + factor * *o;
            }
        }
    }
}

/// The solution of an integration as a function of time: the continuous extensions of its steps,
/// one after the other.
///
/// [`eval`](Self::eval) finds the step a time falls in by bisection and evaluates the method's
/// own polynomial there, so its accuracy between the steps is that of the method's continuous
/// extension: of order 4 for [`DormandPrince5`](super::DormandPrince5), 7 for
/// [`DormandPrince853`](super::DormandPrince853), 3 for [`Rk4`](super::Rk4), the stage order 3
/// for the collocation polynomial of [`RadauIia`](super::RadauIia) (whose steps are of order 5),
/// and the order of the formula for [`Bdf`](super::Bdf). The solution is
/// continuous across the steps, to rounding.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince853, OdeOptions, solve_ivp_with};
///
/// // The harmonic oscillator from rest at one, whose solution is (cos t, −sin t).
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = y[(1, 0)];
///     dy[(1, 0)] = -y[(0, 0)];
/// };
/// let y0 = Matrix::from_rows(&[[1.0], [0.0]]);
/// let opts = OdeOptions::new().with_tolerances(1e-10, 1e-12);
/// let solution = solve_ivp_with(f, &y0, 0.0, 10.0, DormandPrince853::new(), &mut [], &opts);
/// let dense = solution.dense.as_ref().unwrap();
/// assert_eq!((dense.t0(), dense.t1()), (0.0, 10.0));
/// for i in 0..=100 {
///     let t = 0.1 * f64::from(i);
///     let y = dense.eval(t).unwrap();
///     assert!((y[(0, 0)] - t.cos()).abs() < 1e-8 && (y[(1, 0)] + t.sin()).abs() < 1e-8);
/// }
/// // Nothing is known outside the interval that was integrated.
/// assert!(dense.eval(10.5).is_none());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DenseOutput<T> {
    /// The time the integration started at.
    start: T,
    /// The time the solution is known up to, which is inside the last step when a terminal event
    /// ended the integration there.
    end: T,
    /// The state at `start`, which is all there is when no step was taken.
    initial: Vec<T>,
    /// The steps, in the order they were taken.
    steps: Vec<DenseStep<T>>,
}

impl<T: Real> DenseOutput<T> {
    /// The solution of an integration that starts at `start` from the state `initial` and has
    /// taken no step yet.
    pub(super) fn new(start: T, initial: &[T]) -> Self {
        DenseOutput {
            start,
            end: start,
            initial: initial.to_vec(),
            steps: Vec::new(),
        }
    }

    /// Adds the next step of the integration.
    pub(super) fn push(&mut self, step: DenseStep<T>) {
        self.end = step.t1();
        self.steps.push(step);
    }

    /// Ends the solution at `end`, which lies in the last step.
    pub(super) fn end_at(&mut self, end: T) {
        self.end = end;
    }

    /// The time the solution starts at.
    pub fn t0(&self) -> T {
        self.start
    }

    /// The time the solution is known up to: the end of the integration, the time of the terminal
    /// event that stopped it, or the last step before it failed.
    pub fn t1(&self) -> T {
        self.end
    }

    /// The steps of the integration, in the order they were taken.
    pub fn steps(&self) -> &[DenseStep<T>] {
        &self.steps
    }

    /// The solution at the time `t`, as a column, or `None` when `t` is outside the interval that
    /// was integrated or is not a number.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{DormandPrince5, solve_ivp};
    ///
    /// // y′ = −y backwards in time, from 1 at t = 0 to e at t = −1.
    /// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
    /// let y0 = Matrix::from_rows(&[[1.0]]);
    /// let solution = solve_ivp(f, &y0, 0.0, -1.0, DormandPrince5::new(), &mut []);
    /// let dense = solution.dense.unwrap();
    /// assert!((dense.eval(-0.3).unwrap()[(0, 0)] - 0.3f64.exp()).abs() < 1e-6);
    /// assert!(dense.eval(0.1).is_none() && dense.eval(f64::NAN).is_none());
    /// ```
    #[must_use]
    pub fn eval(&self, t: T) -> Option<Matrix<T>> {
        let mut out = vec![T::ZERO; self.initial.len()];
        self.eval_into(t, &mut out)
            .then(|| Matrix::from_column_major(self.initial.len(), 1, out))
    }

    /// The solution at the time `t`, written to `out`; `false`, with `out` untouched, when `t` is
    /// outside the interval that was integrated or is not a number.
    ///
    /// # Panics
    ///
    /// Panics if `out` does not have one element for each component of the state.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{Rk4, solve_ivp};
    ///
    /// let f = |t: f64, _y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = 2.0 * t;
    /// let y0 = Matrix::from_rows(&[[0.0]]);
    /// let dense = solve_ivp(f, &y0, 0.0, 2.0, Rk4::fixed(4), &mut []).dense.unwrap();
    /// let mut out = [0.0];
    /// assert!(dense.eval_into(1.25, &mut out));
    /// assert!((out[0] - 1.5625).abs() < 1e-14);
    /// assert!(!dense.eval_into(2.5, &mut out));
    /// ```
    #[track_caller]
    pub fn eval_into(&self, t: T, out: &mut [T]) -> bool {
        assert_eq!(
            out.len(),
            self.initial.len(),
            "the output does not have the size of the state"
        );
        let ascending = self.end >= self.start;
        let inside = if ascending {
            self.start <= t && t <= self.end
        } else {
            self.end <= t && t <= self.start
        };
        if !inside {
            return false;
        }
        if self.steps.is_empty() {
            out.copy_from_slice(&self.initial);
            return true;
        }
        // The first step that does not end before `t`, in the direction of the integration.
        let found = self.steps.partition_point(|step| {
            if ascending {
                step.t1() < t
            } else {
                step.t1() > t
            }
        });
        self.steps[found.min(self.steps.len() - 1)].eval_into(t, out);
        true
    }
}
