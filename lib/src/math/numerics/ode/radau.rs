//! The implicit Runge–Kutta method of the Radau IIA family of three stages, of order 5, as in
//! Hairer and Wanner's code RADAU5.
//!
//! The constants below — the abscissae, the eigenvalues of the inverse of the method's matrix,
//! a real transformation of that inverse to block-diagonal form, and the weights of the error
//! estimate — are derived exactly (sympy) in `tests/math/numerics/ode/data/gen_radau.py`, which
//! prints them to 36 digits and asserts the relations they must satisfy; the tests of the crate
//! check the method they make up against its stability function, its collocation polynomial and
//! its error estimate computed independently.

use super::control::{Limits, all_finite, error_norm, first_step, in_scale};
use super::stiff::{Jacobian, NoJacobian, newton_tolerance, rounding_floor};
use super::{DenseStep, OdeFailure, OdeMethod, OdeOptions};
use crate::math::complex::{Complex, Float};
use crate::math::matrix::{Lu, Matrix, Scalar};
use crate::math::numerics::Real;

/// The abscissae: the zeros `(4 ∓ √6)/10` of `10x² − 8x + 1` and the end of the step, which
/// makes the method stiffly accurate.
#[allow(clippy::excessive_precision)]
const C: [f64; 3] = [
    1.550_510_257_216_821_901_802_715_925_294_108_61e-1,
    6.449_489_742_783_178_098_197_284_074_705_891_39e-1,
    1.0,
];

/// The real eigenvalue of `A⁻¹`, `3 + 3^(2/3) − 3^(1/3)`.
#[allow(clippy::excessive_precision)]
const GAMMA: f64 = 3.637_834_252_744_495_732_208_418_513_577_775_8;

/// The complex pair of eigenvalues of `A⁻¹`, `α ± iβ`: `α = 3 + (3^(1/3) − 3^(2/3))/2` and
/// `β = (3^(5/6) + 3^(7/6))/2`.
#[allow(clippy::excessive_precision)]
const ALPHA: f64 = 2.681_082_873_627_752_133_895_790_743_211_112_1;
#[allow(clippy::excessive_precision)]
const BETA: f64 = 3.050_430_199_247_410_569_426_377_624_787_567_9;

/// `T`, with `T⁻¹·A⁻¹·T = [[γ, 0, 0], [0, α, −β], [0, β, α]]`: its columns are the real
/// eigenvector of `A⁻¹` and the real and imaginary parts of the eigenvector of `α − iβ`, each
/// scaled to end in 1 and 0.
#[allow(clippy::excessive_precision)]
const TRANS: [[f64; 3]; 3] = [
    [
        9.443_876_248_897_524_148_749_007_950_641_658_63e-2,
        -1.412_552_950_209_542_084_279_903_838_077_973_09e-1,
        -3.002_919_410_514_742_449_186_111_708_905_386_67e-2,
    ],
    [
        2.502_131_229_653_333_113_765_090_675_125_016_84e-1,
        2.041_293_522_937_999_319_959_908_102_983_381_74e-1,
        3.829_421_127_572_619_377_954_382_335_998_732_1e-1,
    ],
    [1.0, 1.0, 0.0],
];

/// `T⁻¹`.
#[allow(clippy::excessive_precision)]
const TRANS_INV: [[f64; 3]; 3] = [
    [
        4.178_718_591_551_904_727_346_462_658_512_056_23,
        3.276_828_207_610_623_870_825_332_724_296_162_34e-1,
        5.233_764_454_994_495_480_399_309_159_089_875_02e-1,
    ],
    [
        -4.178_718_591_551_904_727_346_462_658_512_056_23,
        -3.276_828_207_610_623_870_825_332_724_296_162_34e-1,
        4.766_235_545_005_504_519_600_690_840_910_124_98e-1,
    ],
    [
        -5.028_726_349_457_868_759_512_473_431_395_442_93e-1,
        2.571_926_949_855_605_429_186_785_353_601_675_05,
        -5.960_392_048_282_249_249_688_219_110_993_024_03e-1,
    ],
];

/// The weights of the error estimate, `−(13 + 7√6)/3`, `(−13 + 7√6)/3` and `−1/3`: the
/// difference of the solution and the embedded formula of order 3 is
/// `γ₀·(h·f(t, y) + Σ ddⱼ·Zⱼ)` with `γ₀ = 1/γ` and `Zⱼ` the stage increments.
#[allow(clippy::excessive_precision)]
const DD: [f64; 3] = [
    -1.004_880_939_982_741_556_246_032_950_764_707_99e1,
    1.382_142_733_160_748_895_793_662_840_980_413_25,
    -3.333_333_333_333_333_333_333_333_333_333_333_33e-1,
];

/// RADAU5's defaults: the most Newton iterations of a step, the safety factor, the bounds on the
/// ratio of two steps (a step is divided by at most `FACL` and grows at most `1/FACR` times),
/// the rate of convergence below which the Jacobian is kept, and the ratios of two steps within
/// which the step is not changed at all, so that the LU decompositions are kept too.
const NIT: usize = 7;
const SAFE: f64 = 0.9;
const FACL: f64 = 5.0;
const FACR: f64 = 0.125;
const THET: f64 = 0.001;
const QUOT1: f64 = 1.0;
const QUOT2: f64 = 1.2;

/// The implicit Runge–Kutta method of the Radau IIA family of three stages[^hw], as Hairer and
/// Wanner's RADAU5 implements it: order 5, L-stable, and the method of this module for stiff
/// problems at moderate to stringent tolerances.
///
/// A step is the collocation polynomial of degree 3 on the abscissae `(4 ∓ √6)/10` and `1`: the
/// stage increments `Zᵢ = Yᵢ − y` satisfy `Z = h·(A ⊗ I)·F(Z)` with `F(Z)ᵢ = f(t + cᵢ·h, y + Zᵢ)`,
/// and the step ends at `y + Z₃`.
///
/// * **The Newton iteration.** The stages are found by the simplified Newton iteration on
///   `(h⁻¹·A⁻¹ ⊗ I − I ⊗ J)·ΔZ = −h⁻¹·(A⁻¹ ⊗ I)·Z + F(Z)`, with `J` an approximation of the
///   Jacobian `∂f/∂y`. The matrix `A⁻¹` has one real eigenvalue `γ` and a complex pair `α ± iβ`,
///   and in the coordinates `W = (T⁻¹ ⊗ I)·Z` that make it block-diagonal the `3n × 3n` system
///   falls apart into one real system with the matrix `γ/h·I − J` and one complex system with
///   `(α + iβ)/h·I − J`: an LU decomposition of `n × n` of each ([`Matrix::lu`], on `Matrix<T>`
///   and on `Matrix<Complex<T>>`) is all the iteration needs, and they are kept for as long as
///   `h` and `J` are, formed again in the memory of the last
///   ([`Lu::refactor`](crate::math::matrix::Lu::refactor)) and solved into memory of their own
///   ([`Lu::solve_into`](crate::math::matrix::Lu::solve_into)), so that for a system of at most
///   100 equations the steps allocate nothing once the first decompositions have been formed
///   (above that, as `Lu::refactor` says).
/// * **Starting values.** The collocation polynomial of the previous step, extrapolated to the
///   stages of the new one; zero for the first step.
/// * **Convergence.** The rate `θ` of the iteration is watched from its second iterate on: an
///   iteration that diverges (`θ ≥ 0.99`) or would not converge within seven iterates is
///   abandoned and the step retried smaller, and the iteration stops once its error estimated
///   from `θ` is below a fraction `κ = max(10·ε/rtol, min(0.03, √rtol))` of the tolerances
///   (the first term, the rounding of the state, measured in the norm of the tolerances, so that
///   it holds for a purely absolute tolerance too). The rate is always one measured in the step
///   itself, so the iteration takes two iterates at least, unless a correction is already below
///   the rounding of the state (`10·ε` times the state in the norm of the tolerances), which no
///   further iterate could improve on.
/// * **Error estimate.** RADAU5's, the difference from an embedded formula of order 3, filtered
///   through `(I − h·J/γ)⁻¹` so that it stays bounded on the stiff components; for the first
///   step and after a rejection an estimate above one is filtered a second time, through
///   `f(t, y + err)`, which keeps a stiff transient from being judged by an estimate that has
///   not yet been damped.
/// * **Step-size control.** Gustafsson's predictive controller, with RADAU5's constants: the
///   next step is the smaller of `h·fac·err^(−1/4)`, the safety factor `fac` lowered from 0.9
///   when the iteration needed many iterates, and the predictive
///   `h·0.9·(h/h_old)·(err_old/err²)^(1/4)` (`h_old` and `err_old ≥ 0.01` those of the step
///   accepted before), each held to `[h/5, 8h]`. A rejected step is retried at the first of
///   them, a rejected first step at `h/10`. The step after a rejection does not grow, and a step
///   within `[1, 1.2]` times the last one with a Jacobian that served well is not changed at
///   all, so that the decompositions are kept.
/// * **The Jacobian** is formed at the start and then again only when the iteration converged
///   slowly (`θ > 0.001`) or a step failed with an old one. It is the caller's, given by
///   [`with_jacobian`](Self::with_jacobian), or forward differences of `f` (`n` evaluations).
/// * **Dense output.** The collocation polynomial of each step, of degree 3, which agrees with
///   the steps at their ends and is of the stage order 3 between them: its error there falls as
///   `h⁴` where the steps' own falls as `h⁶`, so on a stiff problem, whose steps are long, it is
///   less accurate between the steps than at them, as SciPy's and RADAU5's are.
///
/// Three things differ from RADAU5. The tolerances are the module's: the error estimate is held
/// below one in the norm of [`OdeOptions`]. RADAU5 itself first replaces `rtol` by
/// `0.1·rtol^(2/3)`; this does not, as SciPy's `Radau` does not, so that `rtol` and `atol` mean
/// the same for every method here. The first step, when not given, is Hairer's `hinit` for an
/// estimate of order 4 (RADAU5 starts from `10⁻⁶`). And the first Newton iterate is not
/// accepted on a rate carried over from the step before: RADAU5 takes `η = max(η_old, ε)^0.8`
/// for it, which after a step that converged fast lets a first correction of up to hundreds of
/// tolerances pass as converged while the Jacobian, kept from earlier steps, no longer fits.
/// Where stiff components follow slow ones through a nonlinear coupling, as in Kaps's problem
/// `y₁′ = −(1/ε + 2)·y₁ + y₂²/ε`, `y₂′ = y₁ − y₂ − y₂²`, the stages are then far from converged —
/// a hundred tolerances at `ε = 10⁻⁹`, `rtol = 10⁻³` — and the error estimate, computed from the
/// same stages, does not see it. A rate measured in the step, as SciPy's `Radau` requires, costs
/// one more iterate on the steps RADAU5 would end after one and keeps those errors below the
/// tolerance.
///
/// [`OdeMethod::jacobians`] counts the Jacobians, analytic or differenced, and
/// [`OdeMethod::factorisations`] the LU decompositions, two for each change of `h` or `J`.
///
/// The Newton iteration measures its corrections against `atol + rtol·|y|` at the start of the
/// step, so with `atol = 0` a component that is exactly zero there and that the step moves has a
/// scale of nought, which no correction meets: at `rtol = 1e-6`, `atol = 0` the oscillator
/// `y′ = (y₂, −y₁)` from `(1, 0)` ends with [`OdeFailure::StepTooSmall`] at `t₀`, and `y′ = t`
/// from 0 with [`OdeFailure::MaxSteps`] near `t = 0`, where the explicit methods finish. Give a
/// positive `atol`, as RADAU5 requires.
///
/// [^hw]: E. Hairer, G. Wanner, *Solving Ordinary Differential Equations II: Stiff and
///     Differential-Algebraic Problems*, 2nd edition, Springer, 1996, sections IV.5 and IV.8.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{OdeOptions, RadauIia, integrate_with};
///
/// // The stiff decay to a slowly moving equilibrium, y′ = 1000(sin t − y). An explicit method
/// // needs thousands of steps here for stability alone.
/// let f = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = 1000.0 * (t.sin() - y[(0, 0)]);
/// };
/// let opts = OdeOptions::new().with_tolerances(1e-8, 1e-10);
/// let r = integrate_with(f, &Matrix::from_rows(&[[0.0]]), 0.0, 5.0, RadauIia::new(), &opts)
///     .unwrap();
/// // The forced response of y′ + 1000y = 1000 sin t, the transient e⁻⁵⁰⁰⁰ having vanished.
/// let want = (1e6 * 5.0f64.sin() - 1000.0 * 5.0f64.cos()) / (1e6 + 1.0);
/// assert!((r.y[(0, 0)] - want).abs() < 1e-8);
/// assert!(r.steps < 200);
/// ```
#[derive(Debug, Clone)]
pub struct RadauIia<T, J = NoJacobian<T>> {
    /// The caller's Jacobian, when there is one, and the increments of the differences that
    /// stand in for it otherwise.
    jacobian: Option<J>,
    differences: Jacobian<T>,
    /// Whether [`OdeMethod::start`] has placed the method at an initial point.
    started: bool,
    /// The limits of the integration in hand.
    lim: Limits<T>,
    /// The time reached, and the time the last accepted step started from.
    t: T,
    t_old: T,
    /// The size of the next step to try, signed, and that of the last accepted one.
    h: T,
    h_old: T,
    /// The state at `t` and at `t_old`, and the right-hand side at `t`.
    y: Matrix<T>,
    y_old: Matrix<T>,
    f0: Matrix<T>,
    /// The collocation polynomial of the last accepted step, but for its value `y` at the end:
    /// the divided differences on the nodes `1, c₂, c₁, 0` of the step in units of its length.
    cont: [Vec<T>; 3],
    /// The stage increments `Zᵢ`, the transformed ones `Wᵢ`, and `f` at the stages.
    z: [Vec<T>; 3],
    w: [Vec<T>; 3],
    fz: [Matrix<T>; 3],
    /// The argument of an evaluation, the end of a step before it is accepted, `f` there, and
    /// the error estimate.
    stage: Matrix<T>,
    y_new: Matrix<T>,
    f_new: Matrix<T>,
    err: Vec<T>,
    /// The right-hand sides of the real and the complex system.
    rhs: Matrix<T>,
    rhs_complex: Matrix<Complex<T>>,
    /// The Jacobian, whether it was formed at the point reached, and whether a new one is wanted
    /// before the next attempt.
    jac: Matrix<T>,
    jac_current: bool,
    jac_wanted: bool,
    /// The LU decompositions of `γ/h·I − J` and `(α + iβ)/h·I − J`, whether they are current,
    /// and the `h` they are for; the matrices they are formed from, and the solutions of their
    /// systems, each in memory kept from one decomposition to the next.
    lu_real: Option<Lu<T>>,
    lu_complex: Option<Lu<Complex<T>>>,
    lu_valid: bool,
    lu_h: T,
    iteration_real: Matrix<T>,
    iteration_complex: Matrix<Complex<T>>,
    solution: Matrix<T>,
    solution_complex: Matrix<Complex<T>>,
    /// The rate `θ` of the Newton iteration of the last attempt, which decides whether its
    /// Jacobian is kept (`THET` when no rate was measured: the correction was lost in rounding).
    theta: T,
    /// The size and the error of the last accepted step, for Gustafsson's controller.
    hacc: T,
    erracc: T,
    /// Whether no step has been accepted yet, and whether the last attempt was rejected.
    first: bool,
    reject: bool,
    /// The counts.
    rejected: usize,
    njev: usize,
    nlu: usize,
}

impl<T: Real + Scalar> Default for RadauIia<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real + Scalar> RadauIia<T> {
    /// The three-stage Radau IIA method, with a Jacobian formed by finite differences.
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
        Self::blank(None)
    }
}

impl<T: Real + Scalar, J> RadauIia<T, J> {
    /// The method in the state of [`RadauIia::new`], with the Jacobian `jacobian`.
    fn blank(jacobian: Option<J>) -> Self {
        let empty = || Matrix::zeros(0, 1);
        RadauIia {
            jacobian,
            differences: Jacobian::new(0),
            started: false,
            lim: Limits::idle(),
            t: T::ZERO,
            t_old: T::ZERO,
            h: T::ZERO,
            h_old: T::ZERO,
            y: empty(),
            y_old: empty(),
            f0: empty(),
            cont: core::array::from_fn(|_| Vec::new()),
            z: core::array::from_fn(|_| Vec::new()),
            w: core::array::from_fn(|_| Vec::new()),
            fz: core::array::from_fn(|_| empty()),
            stage: empty(),
            y_new: empty(),
            f_new: empty(),
            err: Vec::new(),
            rhs: empty(),
            rhs_complex: Matrix::from_column_major(0, 1, Vec::new()),
            jac: Matrix::zeros(0, 0),
            jac_current: false,
            jac_wanted: true,
            lu_real: None,
            lu_complex: None,
            lu_valid: false,
            iteration_real: Matrix::zeros(0, 0),
            iteration_complex: Matrix::from_column_major(0, 0, Vec::new()),
            solution: Matrix::zeros(0, 1),
            solution_complex: Matrix::from_column_major(0, 1, Vec::new()),
            lu_h: T::ZERO,
            theta: T::from_f64(THET),
            hacc: T::ZERO,
            erracc: T::from_f64(1e-2),
            first: true,
            reject: false,
            rejected: 0,
            njev: 0,
            nlu: 0,
        }
    }

    /// The method with the Jacobian `jacobian` of the right-hand side, called as
    /// `jacobian(t, y, out)`, which must leave the `n × n` matrix `out` holding `∂f/∂y` at
    /// `(t, y)`. `out` is zero when the call begins, so only the entries that are not need be
    /// written.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{OdeOptions, RadauIia, integrate_with};
    ///
    /// // The stiff decay y′ = −50(y − cos t), with its Jacobian −50.
    /// let f = |t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
    ///     dy[(0, 0)] = 50.0 * (t.cos() - y[(0, 0)]);
    /// };
    /// let jac = |_t: f64, _y: &Matrix<f64>, out: &mut Matrix<f64>| out[(0, 0)] = -50.0;
    /// let opts = OdeOptions::new().with_tolerances(1e-8, 1e-10);
    /// let method = RadauIia::new().with_jacobian(jac);
    /// let r = integrate_with(f, &Matrix::from_rows(&[[0.0]]), 0.0, 2.0, method, &opts).unwrap();
    /// // The transient has long died; what is left is the forced response.
    /// let want = (2500.0 * 2.0f64.cos() + 50.0 * 2.0f64.sin()) / 2501.0;
    /// assert!((r.y[(0, 0)] - want).abs() < 1e-8);
    /// // On a linear problem with its exact Jacobian the first iterate solves the stages and the
    /// // second shows it, so the Jacobian is formed again only after a rejected step, as RADAU5
    /// // does.
    /// assert!(r.njev <= 1 + r.rejected);
    /// ```
    #[must_use]
    pub fn with_jacobian<K>(self, jacobian: K) -> RadauIia<T, K>
    where
        K: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        RadauIia::blank(Some(jacobian))
    }
}

/// Whether two step sizes differ by no more than the rounding of the times `a` and `b` they are
/// taken between: the step a clock really makes from `a` is `(a + h) − a`, which is not `h`.
fn same_step<T: Real>(h: T, g: T, a: T, b: T) -> bool {
    (h - g).abs() <= T::from_f64(8.0) * T::EPSILON * (a.abs() + b.abs())
}

/// What a Newton iteration came to.
enum Newton<T> {
    /// It converged after this many iterates.
    Converged(usize),
    /// It did not, and the step should be retried this many times smaller.
    Failed(T),
}

impl<T, J> RadauIia<T, J>
where
    T: Real + Scalar,
    Complex<T>: Scalar,
    J: FnMut(T, &Matrix<T>, &mut Matrix<T>),
{
    /// The Jacobian at the point reached, which is then current. Whether it is finite.
    fn jacobian_here<F>(&mut self, f: &mut F) -> bool
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let finite = self.differences.form(
            f,
            self.jacobian.as_mut(),
            self.t,
            &self.y,
            &self.f0,
            self.lim.atol,
            &mut self.jac,
            &mut self.stage,
            &mut self.f_new,
        );
        self.njev += 1;
        self.jac_current = true;
        self.jac_wanted = false;
        self.lu_valid = false;
        finite
    }

    /// The LU decompositions of `γ/h·I − J` and `(α + iβ)/h·I − J`, unless they are there for
    /// this `h` already, to the rounding of the times of the step. Whether both are regular.
    fn factorise(&mut self, h: T, t_new: T) -> bool {
        if let (Some(real), Some(complex)) = (&self.lu_real, &self.lu_complex)
            && self.lu_valid
            && same_step(h, self.lu_h, self.t, t_new)
        {
            return !real.is_singular() && !complex.is_singular();
        }
        let n = self.y.rows();
        let g = T::from_f64(GAMMA) / h;
        let (a, b) = (T::from_f64(ALPHA) / h, T::from_f64(BETA) / h);
        for j in 0..n {
            for i in 0..n {
                let x = self.jac[(i, j)];
                self.iteration_real[(i, j)] = if i == j { g - x } else { -x };
                self.iteration_complex[(i, j)] = if i == j {
                    Complex::new(a - x, b)
                } else {
                    Complex::new(-x, T::ZERO)
                };
            }
        }
        let real = match &mut self.lu_real {
            Some(lu) => {
                lu.refactor(&self.iteration_real);
                lu
            }
            None => self.lu_real.insert(self.iteration_real.lu()),
        };
        let real_regular = !real.is_singular();
        let complex = match &mut self.lu_complex {
            Some(lu) => {
                lu.refactor(&self.iteration_complex);
                lu
            }
            None => self.lu_complex.insert(self.iteration_complex.lu()),
        };
        self.nlu += 2;
        let regular = real_regular && !complex.is_singular();
        self.lu_valid = true;
        self.lu_h = h;
        regular
    }

    /// The starting values of the iteration for a step of size `h`: the collocation polynomial
    /// of the last accepted step at the stages of this one, or zero before the first.
    fn start_values(&mut self, h: T) {
        let n = self.y.rows();
        if self.first {
            for i in 0..3 {
                self.z[i].iter_mut().for_each(|v| *v = T::ZERO);
                self.w[i].iter_mut().for_each(|v| *v = T::ZERO);
            }
            return;
        }
        let ratio = h / self.h_old;
        let (c1, c2) = (T::from_f64(C[0]), T::from_f64(C[1]));
        for (z, &c) in self.z.iter_mut().zip(&C) {
            // The stage at θ = 1 + ratio·cᵢ of the last step, less its end.
            let s = ratio * T::from_f64(c);
            for (r, z) in z.iter_mut().enumerate() {
                let inner = self.cont[1][r] + (s + T::ONE - c1) * self.cont[2][r];
                *z = s * (self.cont[0][r] + (s + T::ONE - c2) * inner);
            }
        }
        for r in 0..n {
            let z = [self.z[0][r], self.z[1][r], self.z[2][r]];
            for (i, row) in TRANS_INV.iter().enumerate() {
                self.w[i][r] = T::from_f64(row[0]) * z[0]
                    + T::from_f64(row[1]) * z[1]
                    + T::from_f64(row[2]) * z[2];
            }
        }
    }

    /// The simplified Newton iteration for the stages of the step of size `h` to `t_new`, from
    /// the starting values in `z` and `w`, with the decompositions in place, held to `fnewt`.
    ///
    /// An iterate is accepted only on the rate of convergence measured in this iteration, from
    /// its second iterate on, or once its correction is below `floor`, the rounding of the state,
    /// where the stages are as good as the arithmetic makes them and a rate would be a ratio of
    /// rounding errors. RADAU5 also accepts the first iterate, on `η = max(η_old, ε)^0.8` carried
    /// over from the iteration of the step before; with a Jacobian kept from earlier steps that
    /// passes stages far from converged (see the documentation of the type).
    fn newton<F>(&mut self, f: &mut F, h: T, t_new: T, fnewt: T, floor: T) -> Newton<T>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let n = self.y.rows();
        let (g, a, b) = (
            T::from_f64(GAMMA) / h,
            T::from_f64(ALPHA) / h,
            T::from_f64(BETA) / h,
        );
        let (rtol, atol) = (self.lim.rtol, self.lim.atol);
        // The iteration's own norm: the root mean square of the 3n increments, each component
        // scaled by the tolerance at the start of the step.
        let count = T::from_f64((3 * n.max(1)) as f64);
        self.theta = T::from_f64(THET);
        // `θ/(1 − θ)`, which turns the size of a correction into an estimate of the error left;
        // set from the second iterate on, and only then used.
        let mut faccon = T::ONE;
        let (mut dynold, mut thqold) = (T::ZERO, T::ZERO);
        for newt in 1..=NIT {
            for (i, &c) in C.iter().enumerate() {
                for r in 0..n {
                    self.stage[(r, 0)] = self.y[(r, 0)] + self.z[i][r];
                }
                // The last abscissa is the end of the step, which is `t_new` to the last bit.
                let ti = if i == 2 {
                    t_new
                } else {
                    self.t + T::from_f64(c) * h
                };
                f(ti, &self.stage, &mut self.fz[i]);
                if !all_finite(self.fz[i].as_slice()) {
                    return Newton::Failed(T::HALF);
                }
            }
            // The residual in the transformed coordinates, −h⁻¹·(Λ ⊗ I)·W + (T⁻¹ ⊗ I)·F.
            for r in 0..n {
                let fr = [self.fz[0][(r, 0)], self.fz[1][(r, 0)], self.fz[2][(r, 0)]];
                let mut gr = [T::ZERO; 3];
                for (i, row) in TRANS_INV.iter().enumerate() {
                    gr[i] = T::from_f64(row[0]) * fr[0]
                        + T::from_f64(row[1]) * fr[1]
                        + T::from_f64(row[2]) * fr[2];
                }
                let (w1, w2, w3) = (self.w[0][r], self.w[1][r], self.w[2][r]);
                self.rhs[(r, 0)] = gr[0] - g * w1;
                self.rhs_complex[(r, 0)] =
                    Complex::new(gr[1] - (a * w2 - b * w3), gr[2] - (b * w2 + a * w3));
            }
            let (Some(real), Some(complex), true) =
                (&self.lu_real, &self.lu_complex, self.lu_valid)
            else {
                return Newton::Failed(T::HALF);
            };
            if !real.solve_into(&self.rhs, &mut self.solution)
                || !complex.solve_into(&self.rhs_complex, &mut self.solution_complex)
            {
                return Newton::Failed(T::HALF);
            }
            let (dw1, dwc) = (&self.solution, &self.solution_complex);
            let mut sum = T::ZERO;
            for r in 0..n {
                let scale = atol + rtol * Float::abs(self.y[(r, 0)]);
                let d = [dw1[(r, 0)], dwc[(r, 0)].re, dwc[(r, 0)].im];
                for v in d {
                    let q = in_scale(v, scale);
                    sum += q * q;
                }
            }
            let dyno = Float::sqrt(sum / count);
            if !Float::is_finite(dyno) {
                return Newton::Failed(T::HALF);
            }
            // A correction lost in the rounding of the state ends the iteration whatever the
            // rate: it cannot be improved on, and its rate would be noise.
            let lost = dyno <= floor;
            if !lost && newt > 1 && newt < NIT {
                let thq = dyno / dynold;
                self.theta = if newt == 2 {
                    thq
                } else {
                    Float::sqrt(thq * thqold)
                };
                thqold = thq;
                if self.theta < T::from_f64(0.99) {
                    faccon = self.theta / (T::ONE - self.theta);
                    // The error that is left after the iterates still allowed, if the rate
                    // holds: when it is above the tolerance, the iteration is given up now.
                    let left = self.theta.powi((NIT - 1 - newt) as i32);
                    let dyth = faccon * dyno * left / fnewt;
                    if dyth >= T::ONE {
                        let q = dyth.max(T::from_f64(1e-4)).min(T::from_f64(20.0));
                        let exponent = T::ONE / T::from_f64((4 + NIT - 1 - newt) as f64);
                        return Newton::Failed(T::from_f64(0.8) * q.powf(-exponent));
                    }
                } else {
                    return Newton::Failed(T::HALF);
                }
            }
            dynold = dyno.max(T::EPSILON);
            for r in 0..n {
                self.w[0][r] += dw1[(r, 0)];
                self.w[1][r] += dwc[(r, 0)].re;
                self.w[2][r] += dwc[(r, 0)].im;
                let w = [self.w[0][r], self.w[1][r], self.w[2][r]];
                for (i, row) in TRANS.iter().enumerate() {
                    self.z[i][r] = T::from_f64(row[0]) * w[0]
                        + T::from_f64(row[1]) * w[1]
                        + T::from_f64(row[2]) * w[2];
                }
            }
            // Never on the first iterate alone: a rate of this iteration says how much error is
            // left. (RADAU5 accepts the first on a rate carried from the step before.)
            if lost || (newt > 1 && faccon * dyno <= fnewt) {
                return Newton::Converged(newt);
            }
        }
        Newton::Failed(T::HALF)
    }

    /// RADAU5's estimate of the error of the step of size `h` whose stages are in `z` and whose
    /// end is `y_new`, in the norm of the tolerances: `(γ/h·I − J)⁻¹·(f₀ + Σ ddⱼ·Zⱼ/h)`, filtered a
    /// second time through `f(t, y + err)` when it is above one for the first step or after a
    /// rejection. The estimate itself is left in `err`.
    fn estimate<F>(&mut self, f: &mut F, h: T) -> T
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let n = self.y.rows();
        let dd = [
            T::from_f64(DD[0]) / h,
            T::from_f64(DD[1]) / h,
            T::from_f64(DD[2]) / h,
        ];
        // The part of the estimate that is the same for both filters, kept in `stage`.
        for r in 0..n {
            self.stage[(r, 0)] = dd[0] * self.z[0][r] + dd[1] * self.z[1][r] + dd[2] * self.z[2][r];
            self.rhs[(r, 0)] = self.f0[(r, 0)] + self.stage[(r, 0)];
        }
        let (Some(real), true) = (&self.lu_real, self.lu_valid) else {
            return T::INFINITY;
        };
        if !real.solve_into(&self.rhs, &mut self.solution) {
            return T::INFINITY;
        }
        self.err.copy_from_slice(self.solution.as_slice());
        let (rtol, atol) = (self.lim.rtol, self.lim.atol);
        let floor = T::from_f64(1e-10);
        let mut err = error_norm(
            &self.err,
            self.y.as_slice(),
            self.y_new.as_slice(),
            rtol,
            atol,
        )
        .max(floor);
        if err >= T::ONE && (self.first || self.reject) {
            // `f` at `y + err`, into the second stage's slot, which the step no longer needs.
            for r in 0..n {
                self.rhs[(r, 0)] = self.y[(r, 0)] + self.err[r];
            }
            f(self.t, &self.rhs, &mut self.fz[1]);
            if !all_finite(self.fz[1].as_slice()) {
                return T::INFINITY;
            }
            for r in 0..n {
                self.rhs[(r, 0)] = self.fz[1][(r, 0)] + self.stage[(r, 0)];
            }
            if !real.solve_into(&self.rhs, &mut self.solution) {
                return T::INFINITY;
            }
            self.err.copy_from_slice(self.solution.as_slice());
            err = error_norm(
                &self.err,
                self.y.as_slice(),
                self.y_new.as_slice(),
                rtol,
                atol,
            )
            .max(floor);
        }
        if Float::is_finite(err) {
            err
        } else {
            T::INFINITY
        }
    }

    /// Records a rejected attempt of size `h`, the next to be `factor` times it, with a new
    /// Jacobian before it unless the one there was formed at this point.
    fn rejection(&mut self, h: T, factor: T) {
        self.h = self.lim.shrink(h, factor);
        self.reject = true;
        self.rejected += 1;
        if !self.jac_current {
            self.jac_wanted = true;
        }
    }

    /// Accepts the step of size `h` to `t_new`, whose end and `f` there are in `y_new` and
    /// `f_new`, after `newt` iterates and with the error `err`, and chooses the next step.
    fn accept(&mut self, h: T, t_new: T, newt: usize, err: T) {
        let n = self.y.rows();
        // Gustafsson's predictive controller, as RADAU5 has it.
        let safe = T::from_f64(SAFE);
        let nit = T::from_f64(NIT as f64);
        let cfac = safe * (T::ONE + T::TWO * nit);
        let fac = safe.min(cfac / (T::from_f64(newt as f64) + T::TWO * nit));
        let (facr, facl) = (T::from_f64(FACR), T::from_f64(FACL));
        let quarter = T::from_f64(0.25);
        let mut quot = (err.powf(quarter) / fac).max(facr).min(facl);
        if !self.first {
            let predicted = (self.hacc / h) * (err * err / self.erracc).powf(quarter) / safe;
            quot = quot.max(predicted.max(facr).min(facl));
        }
        let mut h_new = h / quot;
        self.hacc = h;
        self.erracc = err.max(T::from_f64(1e-2));
        if self.reject && Float::abs(h_new) > Float::abs(h) {
            h_new = h;
        }
        // The collocation polynomial of the step: divided differences on 1, c₂, c₁ and 0.
        let (c1, c2) = (T::from_f64(C[0]), T::from_f64(C[1]));
        for r in 0..n {
            let (z1, z2, z3) = (self.z[0][r], self.z[1][r], self.z[2][r]);
            let d1 = (z3 - z2) / (T::ONE - c2);
            let d21 = (z2 - z1) / (c2 - c1);
            let d10 = z1 / c1;
            let d2 = (d1 - d21) / (T::ONE - c1);
            let d210 = (d21 - d10) / c2;
            self.cont[0][r] = d1;
            self.cont[1][r] = d2;
            self.cont[2][r] = d2 - d210;
        }
        core::mem::swap(&mut self.y_old, &mut self.y);
        core::mem::swap(&mut self.y, &mut self.y_new);
        core::mem::swap(&mut self.f0, &mut self.f_new);
        self.t_old = self.t;
        self.t = t_new;
        self.h_old = h;
        self.first = false;
        self.reject = false;
        self.jac_current = false;
        // A Jacobian that let the iteration converge fast is kept, and with it the step and the
        // decompositions when the controller asks for nearly the same step.
        let ratio = h_new / h;
        let theta_small = self.theta <= T::from_f64(THET);
        if theta_small && ratio >= T::from_f64(QUOT1) && ratio <= T::from_f64(QUOT2) {
            self.h = h;
        } else {
            self.h = h_new;
        }
        self.jac_wanted = !theta_small;
    }
}

impl<T, J> OdeMethod<T> for RadauIia<T, J>
where
    T: Real + Scalar,
    Complex<T>: Scalar,
    J: FnMut(T, &Matrix<T>, &mut Matrix<T>),
{
    fn order(&self) -> usize {
        5
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
        self.t = t0;
        self.t_old = t0;
        self.y = y0.clone();
        self.y_old = y0.clone();
        self.f0 = y0.clone();
        self.cont = core::array::from_fn(|_| vec![T::ZERO; n]);
        self.z = core::array::from_fn(|_| vec![T::ZERO; n]);
        self.w = core::array::from_fn(|_| vec![T::ZERO; n]);
        self.fz = core::array::from_fn(|_| y0.clone());
        self.stage = y0.clone();
        self.y_new = y0.clone();
        self.f_new = y0.clone();
        self.err = vec![T::ZERO; n];
        self.rhs = y0.clone();
        self.rhs_complex = Matrix::from_column_major(n, 1, vec![Complex::new(T::ZERO, T::ZERO); n]);
        self.jac = Matrix::zeros(n, n);
        self.differences = Jacobian::new(n);
        self.jac_current = false;
        self.jac_wanted = true;
        self.lu_valid = false;
        self.iteration_real = Matrix::zeros(n, n);
        self.iteration_complex =
            Matrix::from_column_major(n, n, vec![Complex::new(T::ZERO, T::ZERO); n * n]);
        self.solution = y0.clone();
        self.solution_complex = self.rhs_complex.clone();
        self.lu_h = T::ZERO;
        self.theta = T::from_f64(THET);
        self.hacc = T::ZERO;
        self.erracc = T::from_f64(1e-2);
        self.first = true;
        self.reject = false;
        self.rejected = 0;
        self.njev = 0;
        self.nlu = 0;
        f(t0, &self.y, &mut self.f0);
        if !all_finite(self.f0.as_slice()) {
            return Err(OdeFailure::NonFinite);
        }
        // RADAU5's estimate shrinks as the fourth power of the step.
        self.h = first_step(
            f,
            t0,
            &self.y,
            &self.f0,
            4,
            opts.h0,
            &self.lim,
            &mut self.y_new,
            &mut self.f_new,
        );
        self.h_old = self.h;
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
        // The attempt just rejected from this point, which a retry must be shorter than.
        let mut rejected = None;
        loop {
            if self.lim.out_of_reach(self.y.as_slice()) {
                return Err(OdeFailure::ToleranceTooSmall);
            }
            let (h, t_new, _last) = self.lim.clip(self.t, self.h, rejected)?;
            rejected = Some(h);
            if self.jac_wanted && !self.jacobian_here(f) {
                // A Jacobian that is not finite at the point reached: no smaller step changes it.
                return Err(OdeFailure::NonFinite);
            }
            if !self.factorise(h, t_new) {
                self.rejection(h, T::HALF);
                continue;
            }
            self.start_values(h);
            let (rtol, atol) = (self.lim.rtol, self.lim.atol);
            let fnewt = newton_tolerance(self.y.as_slice(), rtol, atol);
            let floor = rounding_floor(self.y.as_slice(), rtol, atol);
            let newt = match self.newton(f, h, t_new, fnewt, floor) {
                Newton::Converged(newt) => newt,
                Newton::Failed(factor) => {
                    self.rejection(h, factor);
                    continue;
                }
            };
            for r in 0..self.y.rows() {
                self.y_new[(r, 0)] = self.y[(r, 0)] + self.z[2][r];
            }
            let mut err = self.estimate(f, h);
            if err < T::ONE {
                // The right-hand side at the end, which the next step and its Jacobian need: a
                // step whose end it is not finite at cannot stand.
                f(t_new, &self.y_new, &mut self.f_new);
                if !all_finite(self.f_new.as_slice()) {
                    err = T::INFINITY;
                }
            }
            if err < T::ONE {
                self.accept(h, t_new, newt, err);
                return Ok(());
            }
            let factor = if Float::is_finite(err) {
                if self.first {
                    T::from_f64(0.1)
                } else {
                    // The controller's step without the predictive part, as RADAU5 has it.
                    let safe = T::from_f64(SAFE);
                    let nit = T::from_f64(NIT as f64);
                    let cfac = safe * (T::ONE + T::TWO * nit);
                    let fac = safe.min(cfac / (T::from_f64(newt as f64) + T::TWO * nit));
                    let quot = (err.powf(T::from_f64(0.25)) / fac)
                        .max(T::from_f64(FACR))
                        .min(T::from_f64(FACL));
                    T::ONE / quot
                }
            } else {
                T::HALF
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
        // The collocation polynomial of the step, in Newton form on the nodes 1, c₂, c₁: its
        // value at the end, then the divided differences.
        let n = self.y.as_slice().len();
        let nodes = [T::ONE, T::from_f64(C[1]), T::from_f64(C[0])];
        out.reset(self.t_old, self.t, n, &nodes);
        out.coefficient_mut(0).copy_from_slice(self.y.as_slice());
        for (k, c) in self.cont.iter().enumerate() {
            out.coefficient_mut(k + 1).copy_from_slice(c);
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
