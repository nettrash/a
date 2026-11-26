//! The backward differentiation formulas of orders 1 to 5, with variable order and step, in the
//! form Shampine and Reichelt give them for MATLAB's `ode15s`: backward differences, and the step
//! changed by interpolating them.

use super::control::{Limits, all_finite, error_norm, first_step, in_scale};
use super::stiff::{Jacobian, NoJacobian, newton_tolerance};
use super::{DenseStep, OdeFailure, OdeMethod, OdeOptions};
use crate::math::complex::Float;
use crate::math::matrix::{Lu, Matrix, Scalar};
use crate::math::numerics::Real;

/// The highest order of the formulas.
const MAX_ORDER: usize = 5;

/// The most iterates of the Newton iteration of a step.
const NEWTON_MAXITER: usize = 4;

/// The least a rejected step is multiplied by, and the most a step grows when the order is
/// chosen.
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

/// The `κ` of the numerical differentiation formulas of Klopfenstein's family that Shampine and
/// Reichelt chose for `ode15s` (their Table 1), by order: of the same order as the backward
/// differentiation formulas, with error constants `κ·γₖ + 1/(k + 1)` 37%, 50%, 60% and 43%
/// smaller at orders 1 to 4 — steps 26% longer at the same error at orders 1 to 3 and 12% at
/// order 4, the `(C_BDF/C_NDF)^(1/(k+1))` of the table — for a little of their stability; at
/// order 5 the formula is the BDF itself.
const KAPPA: [f64; 6] = [0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0];

/// The variable-order, variable-step backward differentiation formulas of orders 1 to 5, as
/// Shampine and Reichelt implement them in MATLAB's `ode15s`[^sr] and SciPy in its `BDF`: the
/// multistep method of this module for stiff problems, cheaper by the step than
/// [`RadauIia`](super::RadauIia) at moderate tolerances and on large systems.
///
/// * **Formulas.** By default the numerical differentiation formulas (NDFs) of `ode15s`, which
///   are the BDFs with an extra term `−κ·γₖ·(yₙ₊₁ − yₙ₊₁⁽⁰⁾)` that makes their error constants
///   smaller: `κ = −0.1850, −1/9, −0.0823, −0.0415, 0` for the orders 1 to 5. With
///   [`with_ndf(false)`](Self::with_ndf) they are the backward differentiation formulas
///   themselves (`κ = 0`). Both are of order `k` and A-stable to order 2; from order 3 on they
///   are stable on a sector `|arg(−hλ)| < α` that narrows with the order, `α` = 86°, 73° and 51°
///   for the BDFs of orders 3, 4 and 5 and 80°, 66° and 51° for the NDFs (Shampine and
///   Reichelt's Table 1; the boundary locus gives 86.03°, 73.35°, 51.84° and 80.42°, 66.18°,
///   51.84°). [`with_max_order`](Self::with_max_order) caps the order where that matters.
/// * **Representation.** The solution is kept as the backward differences `∇ʲyₙ`,
///   `j = 0 … k`, of the interpolating polynomial at equally spaced points `tₙ − j·h`. The
///   prediction is the value of that polynomial at `tₙ + h`, and a change of the step is an
///   interpolation of it to the new spacing (the matrices `R` and `U` of Shampine and Reichelt),
///   so the formulas are always those of a constant step.
/// * **The Newton iteration.** The modified Newton iteration on `(I − c·h·J)·Δ = …`,
///   `c = 1/((1 − κ)·γₖ)`, through one LU decomposition of `n × n` ([`Matrix::lu`]), formed again
///   whenever `c·h` changes, in the memory of the last
///   ([`Lu::refactor`](crate::math::matrix::Lu::refactor)), each iterate's system solved into
///   memory of its own ([`Lu::solve_into`](crate::math::matrix::Lu::solve_into)); the rate of
///   convergence is watched and an iteration that diverges or would not converge within four
///   iterates is abandoned. `J` is kept from step to step and
///   formed again only when an iteration fails with an old one.
/// * **Error estimate.** `(κ·γₖ + 1/(k + 1))·∇ᵏ⁺¹yₙ₊₁`, from the correction the iteration made
///   to the prediction.
/// * **Step size and order.** A rejected step is retried at `max(0.2, 0.9·err^(−1/(k+1)))` times
///   its size, the safety factor lowered when the iteration needed many iterates. After `k + 1`
///   steps of the same size the errors of the orders `k − 1` and `k + 1`, from `∇ᵏyₙ₊₁` and
///   `∇ᵏ⁺²yₙ₊₁`, are estimated too, and the order that allows the longest step is taken, the step
///   growing at most ten times; until then the step is not changed, which keeps the LU
///   decomposition.
/// * **Dense output.** The interpolating polynomial of the differences, of degree `k`, through
///   the last `k + 1` points of the solution.
/// * **The Jacobian** is the caller's, given by [`with_jacobian`](Self::with_jacobian), or
///   forward differences of `f` (`n + 1` evaluations).
///
/// The integration starts at order 1 with Hairer's `hinit` for an estimate of order 2, unless
/// [`OdeOptions::h0`] gives the first step. [`OdeMethod::order`] is the order of the formula of
/// the last step.
///
/// The Newton iteration measures its corrections against `atol + rtol·|y⁽⁰⁾|`, `y⁽⁰⁾` the
/// prediction, so with `atol = 0` a component predicted to be exactly zero that the step moves
/// has a scale of nought, which no correction meets: at `rtol = 1e-6`, `atol = 0`, `y′ = t` from
/// 0 ends with [`OdeFailure::MaxSteps`] near `t = 0`, where the explicit methods finish. Give a
/// positive `atol`.
///
/// [^sr]: L. F. Shampine, M. W. Reichelt, "The MATLAB ODE Suite", *SIAM Journal on Scientific
///     Computing* 18 (1997), 1–22; G. D. Byrne, A. C. Hindmarsh, "A polyalgorithm for the
///     numerical solution of ordinary differential equations", *ACM Transactions on Mathematical
///     Software* 1 (1975), 71–96.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{Bdf, OdeOptions, integrate_with};
///
/// // Robertson's chemical kinetics, the classic stiff problem, to t = 40.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     let (a, b, c) = (y[(0, 0)], y[(1, 0)], y[(2, 0)]);
///     dy[(0, 0)] = -0.04 * a + 1e4 * b * c;
///     dy[(1, 0)] = 0.04 * a - 1e4 * b * c - 3e7 * b * b;
///     dy[(2, 0)] = 3e7 * b * b;
/// };
/// let y0 = Matrix::from_rows(&[[1.0], [0.0], [0.0]]);
/// let opts = OdeOptions::new().with_tolerances(1e-6, 1e-12);
/// let r = integrate_with(f, &y0, 0.0, 40.0, Bdf::new(), &opts).unwrap();
/// // Hairer and Wanner's values at t = 40, and the mass that is kept.
/// assert!((r.y[(0, 0)] - 0.715_827_07).abs() < 1e-5);
/// assert!((r.y[(1, 0)] - 9.185_535e-6).abs() < 1e-10);
/// assert!((r.y.as_slice().iter().sum::<f64>() - 1.0).abs() < 1e-10);
/// assert!(r.steps < 500 && r.njev < 30);
/// ```
#[derive(Debug, Clone)]
pub struct Bdf<T, J = NoJacobian<T>> {
    /// The caller's Jacobian, when there is one, and the increments of the differences that
    /// stand in for it otherwise.
    jacobian: Option<J>,
    differences: Jacobian<T>,
    /// Whether the formulas are the NDFs, and the highest order allowed.
    ndf: bool,
    max_order: usize,
    /// Whether [`OdeMethod::start`] has placed the method at an initial point.
    started: bool,
    /// The limits of the integration in hand.
    lim: Limits<T>,
    /// The time reached, and the time the last accepted step started from.
    t: T,
    t_old: T,
    /// The step the differences are spaced by, signed: the size of the next step.
    h: T,
    /// The order of the formula, and the number of steps taken with it at the size `h`.
    order: usize,
    equal: usize,
    /// The order and the factor of the step chosen after the last accepted step, which are put
    /// in place when the next one begins: until then the differences are those of the last step,
    /// for its dense output.
    pending: Option<(usize, T)>,
    /// The state at `t` and at `t_old`.
    y: Matrix<T>,
    y_old: Matrix<T>,
    /// The backward differences `∇ʲy`, `j = 0 … MAX_ORDER + 2`, each of `n` numbers, one after
    /// the other.
    d: Vec<T>,
    /// The prediction, the iterate, `f` at it, and the right-hand side of the linear system.
    y_pred: Matrix<T>,
    y_iter: Matrix<T>,
    fy: Matrix<T>,
    rhs: Matrix<T>,
    /// `Σ γⱼ·∇ʲyₙ / ((1 − κ)·γₖ)`, the correction to the prediction, the error estimate, and the
    /// scale of the Newton iteration's norm.
    psi: Vec<T>,
    corr: Vec<T>,
    err: Vec<T>,
    scale: Vec<T>,
    /// The Jacobian; whether it was formed since the last accepted step; whether it is finite.
    jac: Matrix<T>,
    jac_current: bool,
    jac_valid: bool,
    /// The LU decomposition of `I − c·h·J`, whether it is current, and the `c·h` it is for; the
    /// matrix it is formed from, and the solution of its system, each in memory kept from one
    /// decomposition to the next.
    lu: Option<Lu<T>>,
    lu_valid: bool,
    lu_c: T,
    iteration: Matrix<T>,
    dy: Matrix<T>,
    /// `γₖ = Σ 1/j`, `(1 − κₖ)·γₖ` and the error constants `κₖ·γₖ + 1/(k + 1)`, by order.
    gamma: [T; MAX_ORDER + 1],
    alpha: [T; MAX_ORDER + 1],
    error: [T; MAX_ORDER + 1],
    /// The counts.
    rejected: usize,
    njev: usize,
    nlu: usize,
}

impl<T: Real + Scalar> Default for Bdf<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real + Scalar> Bdf<T> {
    /// The numerical differentiation formulas of `ode15s`, of orders 1 to 5, with a Jacobian
    /// formed by finite differences.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::{Bdf, OdeMethod};
    ///
    /// // An integration starts at order 1.
    /// assert_eq!(Bdf::<f64>::new().order(), 1);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::blank(None, true, MAX_ORDER)
    }
}

impl<T: Real + Scalar, J> Bdf<T, J> {
    /// The method in the state of [`Bdf::new`], with the Jacobian `jacobian`, the formulas and
    /// the highest order given.
    fn blank(jacobian: Option<J>, ndf: bool, max_order: usize) -> Self {
        let empty = || Matrix::zeros(0, 1);
        Bdf {
            jacobian,
            differences: Jacobian::new(0),
            ndf,
            max_order,
            started: false,
            lim: Limits::idle(),
            t: T::ZERO,
            t_old: T::ZERO,
            h: T::ZERO,
            order: 1,
            equal: 0,
            pending: None,
            y: empty(),
            y_old: empty(),
            d: Vec::new(),
            y_pred: empty(),
            y_iter: empty(),
            fy: empty(),
            rhs: empty(),
            psi: Vec::new(),
            corr: Vec::new(),
            err: Vec::new(),
            scale: Vec::new(),
            jac: Matrix::zeros(0, 0),
            jac_current: false,
            jac_valid: false,
            lu: None,
            lu_valid: false,
            lu_c: T::ZERO,
            iteration: Matrix::zeros(0, 0),
            dy: empty(),
            gamma: [T::ZERO; MAX_ORDER + 1],
            alpha: [T::ZERO; MAX_ORDER + 1],
            error: [T::ZERO; MAX_ORDER + 1],
            rejected: 0,
            njev: 0,
            nlu: 0,
        }
    }

    /// The method with the Jacobian `jacobian` of the right-hand side, called as
    /// `jacobian(t, y, out)`, which must leave the `n × n` matrix `out` holding `∂f/∂y` at
    /// `(t, y)`; `out` is zero when the call begins. The formulas and the highest order are kept.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{Bdf, OdeOptions, integrate_with};
    ///
    /// // y′ = −1000(y − cos t), whose Jacobian is −1000.
    /// let f = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
    ///     dy[(0, 0)] = -1000.0 * (y[(0, 0)] - t.cos());
    /// };
    /// let jac = |_t: f64, _y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = -1000.0;
    /// let opts = OdeOptions::new().with_tolerances(1e-8, 1e-10);
    /// let r = integrate_with(f, &Matrix::from_rows(&[[0.0]]), 0.0, 2.0,
    ///                        Bdf::new().with_jacobian(jac), &opts).unwrap();
    /// let want = (1e6 * 2.0f64.cos() + 1000.0 * 2.0f64.sin()) / (1e6 + 1.0);
    /// assert!((r.y[(0, 0)] - want).abs() < 1e-7);
    /// // The exact Jacobian of a linear problem never needs forming again.
    /// assert_eq!(r.njev, 1);
    /// ```
    #[must_use]
    pub fn with_jacobian<K>(self, jacobian: K) -> Bdf<T, K>
    where
        K: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        Bdf::blank(Some(jacobian), self.ndf, self.max_order)
    }

    /// `self` with the numerical differentiation formulas if `ndf`, which is the default, and
    /// with the backward differentiation formulas themselves (`κ = 0`) otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{Bdf, OdeOptions, solve_with};
    ///
    /// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
    /// let opts = OdeOptions::new().with_tolerances(1e-8, 1e-10);
    /// let y0 = Matrix::from_rows(&[[1.0]]);
    /// let ndf = solve_with(f, &y0, 0.0, 5.0, Bdf::new(), &opts).unwrap();
    /// let bdf = solve_with(f, &y0, 0.0, 5.0, Bdf::new().with_ndf(false), &opts).unwrap();
    /// for y in [ndf, bdf] {
    ///     assert!((y[(0, 0)] - (-5.0f64).exp()).abs() < 1e-7);
    /// }
    /// ```
    #[must_use]
    pub fn with_ndf(mut self, ndf: bool) -> Self {
        self.ndf = ndf;
        self
    }

    /// `self` with the order held to at most `order`, as `ode15s`'s `MaxOrder`: the formulas of
    /// orders 1 and 2 are A-stable, those above are not, and a problem with eigenvalues near the
    /// imaginary axis may need them.
    ///
    /// # Panics
    ///
    /// Panics unless `order` is from 1 to 5.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{Bdf, OdeMethod, OdeOptions};
    ///
    /// let mut f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
    /// let mut method = Bdf::new().with_max_order(2);
    /// let opts = OdeOptions::new().with_tolerances(1e-10, 1e-12);
    /// method.start(&mut f, 0.0, &Matrix::from_rows(&[[1.0]]), 10.0, &opts).unwrap();
    /// while method.t() != 10.0 {
    ///     method.step(&mut f).unwrap();
    ///     assert!(method.order() <= 2);
    /// }
    /// ```
    #[must_use]
    #[track_caller]
    pub fn with_max_order(mut self, order: usize) -> Self {
        assert!(
            (1..=MAX_ORDER).contains(&order),
            "the order of a backward differentiation formula is from 1 to 5"
        );
        self.max_order = order;
        self
    }

    /// The difference `∇ʲy`, to read.
    fn diff(&self, j: usize) -> &[T] {
        let n = self.y.as_slice().len();
        &self.d[j * n..(j + 1) * n]
    }

    /// Respaces the differences of the formula in use from the step `h` to `factor·h`, which
    /// becomes the step: the interpolating polynomial is the same, its differences are those of
    /// the new spacing, `D ← (R·U)ᵀ·D` with `R[i][j] = Π_{m=1..i} (m − 1 − factor·j)/m` and `U`
    /// the same for `factor = 1`. The count of equal steps starts again.
    fn rescale(&mut self, factor: T) {
        let k = self.order;
        let n = self.y.as_slice().len();
        let mut r = [[T::ZERO; MAX_ORDER + 1]; MAX_ORDER + 1];
        let mut u = [[T::ZERO; MAX_ORDER + 1]; MAX_ORDER + 1];
        for j in 0..=k {
            r[0][j] = T::ONE;
            u[0][j] = T::ONE;
            let jf = T::from_f64(j as f64);
            for i in 1..=k {
                let (im1, i_f) = (T::from_f64((i - 1) as f64), T::from_f64(i as f64));
                r[i][j] = r[i - 1][j] * (im1 - factor * jf) / i_f;
                u[i][j] = u[i - 1][j] * (im1 - jf) / i_f;
            }
        }
        let mut ru = [[T::ZERO; MAX_ORDER + 1]; MAX_ORDER + 1];
        for i in 0..=k {
            for j in 0..=k {
                let mut sum = T::ZERO;
                for (l, row) in u.iter().enumerate().take(k + 1) {
                    sum += r[i][l] * row[j];
                }
                ru[i][j] = sum;
            }
        }
        let mut column = [T::ZERO; MAX_ORDER + 1];
        for col in 0..n {
            for (j, out) in column.iter_mut().enumerate().take(k + 1) {
                let mut sum = T::ZERO;
                for (i, row) in ru.iter().enumerate().take(k + 1) {
                    sum += row[j] * self.d[i * n + col];
                }
                *out = sum;
            }
            for (j, &v) in column.iter().enumerate().take(k + 1) {
                self.d[j * n + col] = v;
            }
        }
        self.h *= factor;
        self.equal = 0;
    }

    /// The prediction `Σⱼ ∇ʲyₙ` (`j ≤ k`), `ψ = Σ γⱼ·∇ʲyₙ / ((1 − κ)·γₖ)` (`1 ≤ j ≤ k`) and the
    /// scale of the iteration's norm at the prediction.
    fn predict(&mut self) {
        let k = self.order;
        let n = self.y.as_slice().len();
        let (rtol, atol) = (self.lim.rtol, self.lim.atol);
        for col in 0..n {
            let mut prediction = self.d[col];
            let mut psi = T::ZERO;
            for j in 1..=k {
                let v = self.d[j * n + col];
                prediction += v;
                psi += self.gamma[j] * v;
            }
            self.y_pred[(col, 0)] = prediction;
            self.psi[col] = psi / self.alpha[k];
            self.scale[col] = atol + rtol * Float::abs(prediction);
        }
    }
}

/// Whether two step sizes differ by no more than the rounding of the times `a` and `b` they are
/// taken between.
fn same_step<T: Real>(h: T, g: T, a: T, b: T) -> bool {
    (h - g).abs() <= T::from_f64(8.0) * T::EPSILON * (a.abs() + b.abs())
}

impl<T, J> Bdf<T, J>
where
    T: Real + Scalar,
    J: FnMut(T, &Matrix<T>, &mut Matrix<T>),
{
    /// The Jacobian at `(t, y)` where `f(t, y)` is `fy`, into `jac`, which is then current.
    /// Whether it is finite.
    fn jacobian_at<F>(&mut self, f: &mut F, t: T, at_prediction: bool) -> bool
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        // Differences need f at the point; the caller's function does not.
        let y = if at_prediction { &self.y_pred } else { &self.y };
        if self.jacobian.is_none() && at_prediction {
            f(t, y, &mut self.fy);
            if !all_finite(self.fy.as_slice()) {
                self.njev += 1;
                self.jac_current = true;
                self.jac_valid = false;
                self.lu_valid = false;
                return false;
            }
        }
        let finite = self.differences.form(
            f,
            self.jacobian.as_mut(),
            t,
            y,
            &self.fy,
            self.lim.atol,
            &mut self.jac,
            &mut self.y_iter,
            &mut self.rhs,
        );
        self.njev += 1;
        self.jac_current = true;
        self.jac_valid = finite;
        self.lu_valid = false;
        finite
    }

    /// The LU decomposition of `I − c·J`, unless it is there for this `c`. Whether it is regular.
    fn factorise(&mut self, c: T) -> bool {
        if let Some(lu) = &self.lu
            && self.lu_valid
            && self.lu_c == c
        {
            return !lu.is_singular();
        }
        let n = self.y.rows();
        for j in 0..n {
            for i in 0..n {
                let x = self.jac[(i, j)];
                self.iteration[(i, j)] = if i == j { T::ONE - c * x } else { -c * x };
            }
        }
        let lu = match &mut self.lu {
            Some(lu) => {
                lu.refactor(&self.iteration);
                lu
            }
            None => self.lu.insert(self.iteration.lu()),
        };
        self.nlu += 1;
        let regular = !lu.is_singular();
        self.lu_valid = true;
        self.lu_c = c;
        regular
    }

    /// The modified Newton iteration of the formula at `t_new` from the prediction: leaves the
    /// iterate in `y_iter` and its distance from the prediction in `corr`, and returns the number
    /// of iterates if it converged to within `tol` in the norm of `scale`.
    fn newton<F>(&mut self, f: &mut F, t_new: T, c: T, tol: T) -> Option<usize>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let n = self.y.rows();
        let count = T::from_f64(n.max(1) as f64);
        self.y_iter
            .as_mut_slice()
            .copy_from_slice(self.y_pred.as_slice());
        self.corr.iter_mut().for_each(|v| *v = T::ZERO);
        let mut old: Option<T> = None;
        for k in 0..NEWTON_MAXITER {
            f(t_new, &self.y_iter, &mut self.fy);
            if !all_finite(self.fy.as_slice()) {
                return None;
            }
            for r in 0..n {
                self.rhs[(r, 0)] = c * self.fy[(r, 0)] - self.psi[r] - self.corr[r];
            }
            let lu = self.lu.as_ref().filter(|_| self.lu_valid)?;
            if !lu.solve_into(&self.rhs, &mut self.dy) {
                return None;
            }
            let dy = &self.dy;
            let mut sum = T::ZERO;
            for (r, &s) in self.scale.iter().enumerate() {
                let q = in_scale(dy[(r, 0)], s);
                sum += q * q;
            }
            let norm = Float::sqrt(sum / count);
            if !Float::is_finite(norm) {
                return None;
            }
            let rate = old.map(|o| norm / o);
            if let Some(rate) = rate {
                let left = rate.powi((NEWTON_MAXITER - k) as i32) / (T::ONE - rate) * norm;
                if rate >= T::ONE || left > tol {
                    return None;
                }
            }
            for r in 0..n {
                self.y_iter[(r, 0)] += dy[(r, 0)];
                self.corr[r] += dy[(r, 0)];
            }
            if norm == T::ZERO || rate.is_some_and(|rate| rate / (T::ONE - rate) * norm < tol) {
                return Some(k + 1);
            }
            old = Some(norm);
        }
        None
    }

    /// Records a rejected attempt of size `h`, the next to be `factor` times it.
    fn rejection(&mut self, h: T, factor: T) {
        let h_new = self.lim.shrink(h, factor);
        self.rescale(h_new / self.h);
        self.h = h_new;
        self.rejected += 1;
    }

    /// Accepts the step to `t_new`, whose end and correction are in `y_iter` and `corr`, with the
    /// error `err` and after `iterations` iterates; updates the differences and chooses the order
    /// and the step of the next one.
    fn accept(&mut self, t_new: T, err: T, iterations: usize) {
        let k = self.order;
        let n = self.y.rows();
        self.equal += 1;
        self.t_old = self.t;
        self.t = t_new;
        core::mem::swap(&mut self.y_old, &mut self.y);
        core::mem::swap(&mut self.y, &mut self.y_iter);
        // ∇ᵏ⁺¹yₙ₊₁ is the correction, ∇ᵏ⁺²yₙ₊₁ its difference from ∇ᵏ⁺¹yₙ, and every lower
        // difference of the new point is that of the old one plus the next higher of the new.
        for r in 0..n {
            self.d[(k + 2) * n + r] = self.corr[r] - self.d[(k + 1) * n + r];
            self.d[(k + 1) * n + r] = self.corr[r];
        }
        for i in (0..=k).rev() {
            for r in 0..n {
                let above = self.d[(i + 1) * n + r];
                self.d[i * n + r] += above;
            }
        }
        self.jac_current = false;
        if self.equal < k + 1 {
            return;
        }
        // The errors the orders k − 1 and k + 1 would have made, and the order with the longest
        // step.
        let (rtol, atol) = (self.lim.rtol, self.lim.atol);
        // The error of the formula of order `order`, from the difference `j` of the new point.
        let estimate = |order: usize, j: usize, this: &mut Self| -> T {
            for r in 0..n {
                this.err[r] = this.error[order] * this.d[j * n + r];
            }
            error_norm(
                &this.err,
                this.y_old.as_slice(),
                this.y.as_slice(),
                rtol,
                atol,
            )
        };
        // The step each order allows, `err^(−1/(order + 1))`: an error of nought allows any.
        let factor_of = |err: T, order: usize| err.powf(-T::ONE / T::from_f64((order + 1) as f64));
        let (mut order, mut best) = (k, factor_of(err, k));
        if k > 1 {
            let down = factor_of(estimate(k - 1, k, self), k - 1);
            // A lower order is preferred when it allows as long a step, as SciPy's argmax does.
            if down >= best {
                (order, best) = (k - 1, down);
            }
        }
        if k < self.max_order {
            let up = factor_of(estimate(k + 1, k + 2, self), k + 1);
            if up > best {
                (order, best) = (k + 1, up);
            }
        }
        let factor = (safety::<T>(iterations) * best).min(T::from_f64(MAX_FACTOR));
        self.pending = Some((order, factor));
    }
}

/// The safety factor of a step after a Newton iteration of `iterations` iterates: 0.9, less the
/// more iterates it took, as in SciPy's `BDF`.
fn safety<T: Real>(iterations: usize) -> T {
    let most = T::from_f64((2 * NEWTON_MAXITER) as f64);
    T::from_f64(0.9) * (most + T::ONE) / (most + T::from_f64(iterations as f64))
}

impl<T, J> OdeMethod<T> for Bdf<T, J>
where
    T: Real + Scalar,
    J: FnMut(T, &Matrix<T>, &mut Matrix<T>),
{
    fn order(&self) -> usize {
        self.order
    }

    fn start<F>(
        &mut self,
        f: &mut F,
        t0: T,
        y0: &Matrix<T>,
        t1: T,
        opts: &OdeOptions<T>,
    ) -> Result<(), OdeFailure>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let n = y0.as_slice().len();
        self.started = false;
        self.lim = Limits::new(t0, t1, opts);
        for (k, &kappa) in KAPPA.iter().enumerate() {
            let kappa = if self.ndf { kappa } else { 0.0 };
            let gamma: f64 = (1..=k).map(|j| 1.0 / j as f64).sum();
            self.gamma[k] = T::from_f64(gamma);
            self.alpha[k] = T::from_f64((1.0 - kappa) * gamma);
            self.error[k] = T::from_f64(kappa * gamma + 1.0 / (k + 1) as f64);
        }
        self.t = t0;
        self.t_old = t0;
        self.y = y0.clone();
        self.y_old = y0.clone();
        self.y_pred = y0.clone();
        self.y_iter = y0.clone();
        self.fy = y0.clone();
        self.rhs = y0.clone();
        self.psi = vec![T::ZERO; n];
        self.corr = vec![T::ZERO; n];
        self.err = vec![T::ZERO; n];
        self.scale = vec![T::ZERO; n];
        self.d = vec![T::ZERO; (MAX_ORDER + 3) * n];
        self.jac = Matrix::zeros(n, n);
        self.differences = Jacobian::new(n);
        self.lu_valid = false;
        self.lu_c = T::ZERO;
        self.iteration = Matrix::zeros(n, n);
        self.dy = y0.clone();
        self.order = 1;
        self.equal = 0;
        self.pending = None;
        self.rejected = 0;
        self.njev = 0;
        self.nlu = 0;
        f(t0, &self.y, &mut self.fy);
        if !all_finite(self.fy.as_slice()) {
            return Err(OdeFailure::NonFinite);
        }
        // The first step of a formula of order 1, whose error shrinks as the square of the step.
        self.h = first_step(
            f,
            t0,
            &self.y,
            &self.fy,
            2,
            opts.h0,
            &self.lim,
            &mut self.y_iter,
            &mut self.rhs,
        );
        self.d[..n].copy_from_slice(y0.as_slice());
        for (d, &v) in self.d[n..2 * n].iter_mut().zip(self.fy.as_slice()) {
            *d = self.h * v;
        }
        if !self.jacobian_at(f, t0, false) {
            return Err(OdeFailure::NonFinite);
        }
        self.started = true;
        Ok(())
    }

    fn step<F>(&mut self, f: &mut F) -> Result<(), OdeFailure>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        if !self.started {
            return Err(OdeFailure::InvalidInput);
        }
        if self.t == self.lim.t1 {
            return Ok(());
        }
        if let Some((order, factor)) = self.pending.take() {
            self.order = order;
            self.rescale(factor);
        }
        // The attempt just rejected from this point, which a retry must be shorter than.
        let mut rejected = None;
        loop {
            if self.lim.out_of_reach(self.y.as_slice()) {
                return Err(OdeFailure::ToleranceTooSmall);
            }
            let (h, t_new, _last) = self.lim.clip(self.t, self.h, rejected)?;
            rejected = Some(h);
            if !same_step(h, self.h, self.t, t_new) {
                self.rescale(h / self.h);
                self.h = h;
            }
            let k = self.order;
            self.predict();
            let c = h / self.alpha[k];
            let tol = newton_tolerance(self.y.as_slice(), self.lim.rtol, self.lim.atol);
            if !self.jac_valid && !self.jacobian_at(f, t_new, true) {
                self.rejection(h, T::HALF);
                continue;
            }
            let iterations = loop {
                if !self.factorise(c) {
                    break None;
                }
                if let Some(iterations) = self.newton(f, t_new, c, tol) {
                    break Some(iterations);
                }
                // An old Jacobian is formed again, at the prediction, and the iteration tried
                // once more; with a current one the step is too long.
                if self.jac_current || !self.jacobian_at(f, t_new, true) {
                    break None;
                }
            };
            let Some(iterations) = iterations else {
                self.rejection(h, T::HALF);
                continue;
            };
            for r in 0..self.y.rows() {
                self.err[r] = self.error[k] * self.corr[r];
            }
            let err = error_norm(
                &self.err,
                self.y.as_slice(),
                self.y_iter.as_slice(),
                self.lim.rtol,
                self.lim.atol,
            );
            if err <= T::ONE {
                self.accept(t_new, err, iterations);
                return Ok(());
            }
            let factor = if Float::is_finite(err) {
                let exponent = -T::ONE / T::from_f64((k + 1) as f64);
                (safety::<T>(iterations) * err.powf(exponent)).max(T::from_f64(MIN_FACTOR))
            } else {
                T::from_f64(MIN_FACTOR)
            };
            self.rejection(h, factor);
        }
    }

    fn t(&self) -> T {
        self.t
    }

    fn t_old(&self) -> T {
        self.t_old
    }

    fn y(&self) -> &Matrix<T> {
        &self.y
    }

    fn dense<F>(&mut self, _f: &mut F, out: &mut DenseStep<T>) -> Result<(), OdeFailure>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        if !self.started {
            return Err(OdeFailure::InvalidInput);
        }
        // The interpolating polynomial Σ ∇ʲyₙ₊₁·Πₘ (s + m)/(m + 1), s = θ − 1: the Newton form on
        // the nodes 1, 0, −1, … in units of the step, with the coefficients ∇ʲyₙ₊₁/j!.
        let k = self.order;
        let n = self.y.as_slice().len();
        let mut nodes = [T::ZERO; MAX_ORDER];
        for (m, node) in nodes.iter_mut().enumerate() {
            *node = T::ONE - T::from_f64(m as f64);
        }
        out.reset(self.t_old, self.t, n, &nodes[..k]);
        let mut factorial = T::ONE;
        for j in 0..=k {
            if j > 0 {
                factorial *= T::from_f64(j as f64);
            }
            let inverse = T::ONE / factorial;
            for (c, &v) in out.coefficient_mut(j).iter_mut().zip(self.diff(j)) {
                *c = v * inverse;
            }
        }
        Ok(())
    }

    fn rejected(&self) -> usize {
        self.rejected
    }

    fn jacobians(&self) -> usize {
        self.njev
    }

    fn factorisations(&self) -> usize {
        self.nlu
    }
}
