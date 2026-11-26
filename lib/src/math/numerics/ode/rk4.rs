//! The classical Runge–Kutta method of the fourth order: with steps of a size chosen by step
//! doubling, or with a given number of equal steps.

use super::control::{Limits, all_finite, combine, error_norm, first_step};
use super::tableau::{ButcherTableau, RK4};
use super::{DenseStep, OdeFailure, OdeMethod, OdeOptions};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};
use crate::math::numerics::Real;

/// The safety factor of the step-size rule, and the least and the most a step is multiplied by.
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 5.0;

/// The classical explicit Runge–Kutta method of the fourth order[^kutta], in two forms.
///
/// [`Rk4::fixed`] is the method of the textbooks: `n` equal steps from `t₀` to `t₁`, four
/// evaluations each, and no control of the error at all — the tolerances and the step limits of
/// [`OdeOptions`] are not looked at.
///
/// [`Rk4::new`] chooses its steps. The method has no embedded formula to estimate its error
/// with, so the estimate is by step doubling: every stretch of the solution is covered twice,
/// by one step of size `h` and by two of size `h/2`, and Richardson's rule gives the error of the
/// two half-steps as a fifteenth of the difference. If that is within the tolerances — in the
/// norm of the module — the two half-steps are accepted, and they are what the integration
/// reports as its steps: the solution is a chain of classical steps and nothing else, the
/// extrapolated value is not used. The next pair is of size `h · min(5, max(0.2, 0.9 ·
/// err^(−1/5)))`, and does not grow after a rejection. This costs eleven evaluations for a pair
/// of steps where the four of each step alone would be eight, which is why a pair with an
/// embedded estimate, [`DormandPrince5`](super::DormandPrince5), is the better method wherever
/// the classical coefficients are not the point. `h0`, `hmin` and `hmax` bound the size of a
/// pair, so the steps the integration reports are halves of them; and a pair needs a midpoint,
/// so an interval of one unit in the last place of `t` ends with [`OdeFailure::StepTooSmall`]
/// (two units are enough).
///
/// The tolerance bounds the estimate of the error of each step, which is of the fifth order in
/// the step, while the solution is of the fourth: the global error goes as about `tol^(4/5)` and
/// is many tolerances at stringent ones. At `1e-9` (`rtol = atol`) it ends Lotka and Volterra's
/// problem on `[0, 15]` 96 tolerances out and the Brusselator on `[0, 20]` 3.6 out, against 6.2
/// and 1.1 for `DormandPrince5`, and between the steps its dense output is up to 330 and 120
/// tolerances out.
///
/// The dense output of both forms is the cubic Hermite polynomial through the two ends of a step
/// and the slopes there, which is of the third order: enough for a method of the fourth, and no
/// better than that.
///
/// [^kutta]: W. Kutta, "Beitrag zur näherungsweisen Integration totaler Differentialgleichungen",
///     *Zeitschrift für Mathematik und Physik* 46 (1901), 435–453; E. Hairer, S. P. Nørsett,
///     G. Wanner, *Solving Ordinary Differential Equations I*, Springer, 1993, sections II.1 and
///     II.4.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{OdeOptions, Rk4, integrate, solve_with};
///
/// // y′ = −2y, y(0) = 1, whose solution at t = 3 is e⁻⁶.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -2.0 * y[(0, 0)];
/// let y0 = Matrix::from_rows(&[[1.0]]);
///
/// // A hundred equal steps: four hundred evaluations, and the error of a fourth-order method,
/// // 1.7e-9 (the stability polynomial 1 + z + z²/2 + z³/6 + z⁴/24 at z = −0.06, to the 100th).
/// let r = integrate(f, &y0, 0.0, 3.0, Rk4::fixed(100)).unwrap();
/// assert_eq!((r.steps, r.nfev), (100, 400));
/// assert!((r.y[(0, 0)] - (-6.0f64).exp()).abs() < 2e-9);
///
/// // Steps of its own choosing, to a tolerance.
/// let opts = OdeOptions::new().with_tolerances(1e-10, 1e-12);
/// let y = solve_with(f, &y0, 0.0, 3.0, Rk4::new(), &opts).unwrap();
/// assert!((y[(0, 0)] - (-6.0f64).exp()).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct Rk4<T> {
    /// The number of equal steps, for the fixed form.
    fixed: Option<usize>,
    /// Whether [`OdeMethod::start`] has placed the method at an initial point.
    started: bool,
    /// The limits of the integration in hand.
    lim: Limits<T>,
    /// The time reached, and the time the last accepted step started from.
    t: T,
    t_old: T,
    /// The state at `t` and at `t_old`.
    y: Matrix<T>,
    y_old: Matrix<T>,
    /// The right-hand side at `(t, y)`, when `have_slope` says it has been evaluated, and at
    /// `(t_old, y_old)`.
    slope: Matrix<T>,
    have_slope: bool,
    slope_old: Matrix<T>,
    /// The four stages of a classical step, and the argument of one.
    k: Vec<Matrix<T>>,
    stage: Matrix<T>,
    /// The adaptive form: the size of the next pair of steps, the end of one whole step, and the
    /// second half of an accepted pair — its end and the slope there — waiting to be handed over.
    h: T,
    whole: Matrix<T>,
    pending: bool,
    t_end: T,
    y_end: Matrix<T>,
    slope_end: Matrix<T>,
    err: Vec<T>,
    reject: bool,
    rejected: usize,
    /// The fixed form: where the integration started, and the number of steps taken.
    t0: T,
    taken: usize,
}

impl<T: Real + Scalar> Default for Rk4<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real + Scalar> Rk4<T> {
    /// The classical method with its steps chosen by step doubling, to the tolerances of the
    /// integration.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{OdeMethod, Rk4, integrate};
    ///
    /// assert_eq!(Rk4::<f64>::new().order(), 4);
    /// // The steps come in pairs: ten evaluations for the pair and the step it is checked by,
    /// // and one more for the slope at its end.
    /// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = -y[(0, 0)];
    /// let r = integrate(f, &Matrix::from_rows(&[[1.0]]), 0.0, 1.0, Rk4::new()).unwrap();
    /// assert_eq!(r.steps % 2, 0);
    /// assert_eq!(r.nfev, 2 + 11 * (r.steps / 2 + r.rejected));
    /// ```
    #[must_use]
    pub fn new() -> Self {
        let empty = || Matrix::zeros(0, 1);
        Rk4 {
            fixed: None,
            started: false,
            lim: Limits::idle(),
            t: T::ZERO,
            t_old: T::ZERO,
            y: empty(),
            y_old: empty(),
            slope: empty(),
            have_slope: false,
            slope_old: empty(),
            k: Vec::new(),
            stage: empty(),
            h: T::ZERO,
            whole: empty(),
            pending: false,
            t_end: T::ZERO,
            y_end: empty(),
            slope_end: empty(),
            err: Vec::new(),
            reject: false,
            rejected: 0,
            t0: T::ZERO,
            taken: 0,
        }
    }

    /// The classical method with `n` equal steps and no error control: the step is `(t₁ − t₀)/n`,
    /// whatever the tolerances, `h0`, `hmax` and `hmin` of the options say. Four evaluations a
    /// step.
    ///
    /// With nothing to shrink, a right-hand side that is not finite ends the integration with
    /// [`OdeFailure::NonFinite`]; and `n` steps need [`OdeOptions::max_steps`] to be at least
    /// `n`. An integration over an interval of positive length with `n = 0` fails with
    /// [`OdeFailure::InvalidInput`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numerics::{Rk4, solve};
    ///
    /// // One step of size 1 on y′ = y from 1: 1 + 1 + 1/2 + 1/6 + 1/24, the Taylor polynomial of
    /// // e of the fourth degree, which is what a method of the fourth order gives.
    /// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| dy[(0, 0)] = y[(0, 0)];
    /// let y = solve(f, &Matrix::from_rows(&[[1.0]]), 0.0, 1.0, Rk4::fixed(1)).unwrap();
    /// assert!((y[(0, 0)] - 65.0 / 24.0).abs() < 1e-15);
    /// ```
    #[must_use]
    pub fn fixed(n: usize) -> Self {
        Rk4 {
            fixed: Some(n),
            ..Self::new()
        }
    }

    /// The coefficients of the method: four stages and their weights. It has no embedded formula
    /// and no coefficients of a continuous extension, so `e` and `d` are empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Rk4;
    ///
    /// let tableau = Rk4::<f64>::new().tableau();
    /// assert_eq!(tableau.c, &[0.0, 0.5, 0.5, 1.0]);
    /// assert_eq!(tableau.b, &[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]);
    /// assert!(tableau.e.is_empty() && tableau.d.is_empty());
    /// ```
    #[must_use]
    pub fn tableau(&self) -> ButcherTableau {
        RK4
    }

    /// One classical step from `(t, y)`, where the right-hand side is `slope`, to `t_new` by way
    /// of `t_mid`, left in `out`. Whether everything came out finite.
    #[allow(clippy::too_many_arguments)]
    fn classical<F>(
        f: &mut F,
        k: &mut [Matrix<T>],
        stage: &mut Matrix<T>,
        t: T,
        t_mid: T,
        t_new: T,
        y: &Matrix<T>,
        slope: &Matrix<T>,
        out: &mut Matrix<T>,
    ) -> bool
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let h = t_new - t;
        k[0].as_mut_slice().copy_from_slice(slope.as_slice());
        for i in 1..4 {
            combine(stage.as_mut_slice(), y.as_slice(), h, RK4.a[i], k);
            let ti = if i == 3 { t_new } else { t_mid };
            f(ti, stage, &mut k[i]);
            if !all_finite(k[i].as_slice()) {
                return false;
            }
        }
        combine(out.as_mut_slice(), y.as_slice(), h, RK4.b, k);
        all_finite(out.as_slice())
    }

    /// The right-hand side at the point reached, evaluated if it has not been.
    fn slope_here<F>(&mut self, f: &mut F) -> bool
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        if !self.have_slope {
            f(self.t, &self.y, &mut self.slope);
            self.have_slope = true;
        }
        all_finite(self.slope.as_slice())
    }

    /// Moves the point reached to `(t_new, y_new)`, keeping the old one for the dense output;
    /// `y_new` is left holding a state that is no longer needed.
    fn advance(&mut self, t_new: T, y_new: &mut Matrix<T>) {
        core::mem::swap(&mut self.y, &mut self.y_old);
        core::mem::swap(&mut self.y, y_new);
        core::mem::swap(&mut self.slope, &mut self.slope_old);
        self.have_slope = false;
        self.t_old = self.t;
        self.t = t_new;
    }

    /// One of the `n` equal steps of the fixed form.
    fn fixed_step<F>(&mut self, f: &mut F, n: usize) -> Result<(), OdeFailure>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        if !self.slope_here(f) {
            return Err(OdeFailure::NonFinite);
        }
        // The ends of the steps are computed from the ends of the interval, so that their
        // rounding does not add up, and the last is `t1` itself.
        let end = |i: usize| {
            if i == n {
                self.lim.t1
            } else {
                let fraction = T::from_f64(i as f64) / T::from_f64(n as f64);
                self.t0 + (self.lim.t1 - self.t0) * fraction
            }
        };
        let t_new = end(self.taken + 1);
        let t_mid = self.t + (t_new - self.t) * T::HALF;
        let mut y_new = core::mem::replace(&mut self.whole, Matrix::zeros(0, 1));
        let finite = Self::classical(
            f,
            &mut self.k,
            &mut self.stage,
            self.t,
            t_mid,
            t_new,
            &self.y,
            &self.slope,
            &mut y_new,
        );
        if !finite {
            self.whole = y_new;
            return Err(OdeFailure::NonFinite);
        }
        self.advance(t_new, &mut y_new);
        self.whole = y_new;
        self.taken += 1;
        Ok(())
    }

    /// A pair of steps of the adaptive form: tried, and retried smaller, until the two half-steps
    /// agree with the whole one; then the first of them is the step taken and the second waits.
    fn doubled_step<F>(&mut self, f: &mut F) -> Result<(), OdeFailure>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        // The pair just rejected from this point, which a retry must be shorter than.
        let mut rejected = None;
        loop {
            if self.lim.out_of_reach(self.y.as_slice()) {
                return Err(OdeFailure::ToleranceTooSmall);
            }
            let (h, t_end, _last) = self.lim.clip(self.t, self.h, rejected)?;
            rejected = Some(h);
            let t_mid = self.t + h * T::HALF;
            let err = if t_mid != self.t && t_mid != t_end && self.pair(f, t_mid, t_end) {
                error_norm(
                    &self.err,
                    self.y.as_slice(),
                    self.y_end.as_slice(),
                    self.lim.rtol,
                    self.lim.atol,
                )
            } else {
                T::INFINITY
            };
            let fifth = T::from_f64(0.2);
            if err <= T::ONE {
                let mut factor = if err > T::ZERO {
                    (T::from_f64(SAFETY) * err.powf(-fifth)).min(T::from_f64(MAX_FACTOR))
                } else {
                    T::from_f64(MAX_FACTOR)
                };
                factor = factor.max(T::from_f64(MIN_FACTOR));
                if self.reject && factor > T::ONE {
                    factor = T::ONE;
                }
                self.reject = false;
                self.h = h * factor;
                // The first half is the step taken now. `whole` holds the middle of the pair by
                // this time, and the slope there is the first stage of the second half.
                let mut y_mid = core::mem::replace(&mut self.whole, Matrix::zeros(0, 1));
                self.advance(t_mid, &mut y_mid);
                self.whole = y_mid;
                self.slope
                    .as_mut_slice()
                    .copy_from_slice(self.k[0].as_slice());
                self.have_slope = true;
                self.t_end = t_end;
                self.pending = true;
                return Ok(());
            }
            let factor = if Float::is_finite(err) {
                (T::from_f64(SAFETY) * err.powf(-fifth)).max(T::from_f64(MIN_FACTOR))
            } else {
                T::from_f64(MIN_FACTOR)
            };
            self.h = self.lim.shrink(h, factor);
            self.reject = true;
            self.rejected += 1;
        }
    }

    /// The whole step from the point reached to `t_end` and the two half-steps by way of `t_mid`:
    /// leaves the middle of the pair in `whole`, its end in `y_end`, the slopes at the middle and
    /// at the end in the first stage and in `slope_end`, and Richardson's estimate of the error of
    /// the half-steps in `err`. Whether everything came out finite.
    fn pair<F>(&mut self, f: &mut F, t_mid: T, t_end: T) -> bool
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        let t = self.t;
        let quarter = t + (t_mid - t) * T::HALF;
        let three_quarters = t_mid + (t_end - t_mid) * T::HALF;
        // One step over the pair, whose end is only wanted for the estimate.
        if !Self::classical(
            f,
            &mut self.k,
            &mut self.stage,
            t,
            t_mid,
            t_end,
            &self.y,
            &self.slope,
            &mut self.y_end,
        ) {
            return false;
        }
        for (e, &w) in self.err.iter_mut().zip(self.y_end.as_slice()) {
            *e = w;
        }
        // The first half-step, to the middle.
        if !Self::classical(
            f,
            &mut self.k,
            &mut self.stage,
            t,
            quarter,
            t_mid,
            &self.y,
            &self.slope,
            &mut self.whole,
        ) {
            return false;
        }
        // The second, from the middle, whose slope is evaluated into `slope_end` for the moment.
        f(t_mid, &self.whole, &mut self.slope_end);
        if !all_finite(self.slope_end.as_slice()) {
            return false;
        }
        if !Self::classical(
            f,
            &mut self.k,
            &mut self.stage,
            t_mid,
            three_quarters,
            t_end,
            &self.whole,
            &self.slope_end,
            &mut self.y_end,
        ) {
            return false;
        }
        // The half-steps err by a sixteenth of what the whole step does, with the same sign, so
        // a fifteenth of their difference is the error of the half-steps.
        let fifteenth = T::from_f64(1.0 / 15.0);
        for (e, &two) in self.err.iter_mut().zip(self.y_end.as_slice()) {
            *e = (two - *e) * fifteenth;
        }
        // `classical` has left the slope at the middle in the first stage. The slope at the end
        // is the first stage of the next pair, and a point the solution must be defined at.
        f(t_end, &self.y_end, &mut self.slope_end);
        all_finite(self.slope_end.as_slice())
    }
}

impl<T: Real + Scalar> OdeMethod<T> for Rk4<T> {
    fn order(&self) -> usize {
        4
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
        self.started = false;
        self.lim = Limits::new(t0, t1, opts);
        self.t = t0;
        self.t_old = t0;
        self.t0 = t0;
        self.taken = 0;
        self.y = y0.clone();
        self.y_old = y0.clone();
        self.slope = y0.clone();
        self.slope_old = y0.clone();
        self.have_slope = false;
        self.k = (0..4).map(|_| y0.clone()).collect();
        self.stage = y0.clone();
        self.whole = y0.clone();
        self.y_end = y0.clone();
        self.slope_end = y0.clone();
        self.err = vec![T::ZERO; y0.as_slice().len()];
        self.pending = false;
        self.reject = false;
        self.rejected = 0;
        if let Some(n) = self.fixed {
            if n == 0 && t0 != t1 {
                return Err(OdeFailure::InvalidInput);
            }
        } else {
            if !self.slope_here(f) {
                return Err(OdeFailure::NonFinite);
            }
            // Richardson's estimate shrinks as the fifth power of the step.
            self.h = first_step(
                f,
                t0,
                &self.y,
                &self.slope,
                5,
                opts.h0,
                &self.lim,
                &mut self.stage,
                &mut self.k[0],
            );
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
        if let Some(n) = self.fixed {
            return self.fixed_step(f, n);
        }
        if self.pending {
            // The second half of the pair accepted last time, with the slope at its end.
            let mut y_end = core::mem::replace(&mut self.y_end, Matrix::zeros(0, 1));
            self.advance(self.t_end, &mut y_end);
            self.y_end = y_end;
            core::mem::swap(&mut self.slope, &mut self.slope_end);
            self.have_slope = true;
            self.pending = false;
            return Ok(());
        }
        self.doubled_step(f)
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

    fn dense<F>(&mut self, f: &mut F, out: &mut DenseStep<T>) -> Result<(), OdeFailure>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        if !self.started {
            return Err(OdeFailure::InvalidInput);
        }
        if !self.slope_here(f) {
            return Err(OdeFailure::NonFinite);
        }
        // The cubic through the two ends with the slopes there: the Newton form on the nodes
        // 0, 0, 1, whose coefficients are the divided differences of the Hermite data.
        let n = self.y.as_slice().len();
        let h = self.t - self.t_old;
        out.reset(self.t_old, self.t, n, &[T::ZERO, T::ZERO, T::ONE]);
        out.coefficient_mut(0)
            .copy_from_slice(self.y_old.as_slice());
        let (y_old, y) = (self.y_old.as_slice(), self.y.as_slice());
        let (s_old, s) = (self.slope_old.as_slice(), self.slope.as_slice());
        for r in 0..n {
            let delta = y[r] - y_old[r];
            out.coefficient_mut(1)[r] = h * s_old[r];
            out.coefficient_mut(2)[r] = delta - h * s_old[r];
            out.coefficient_mut(3)[r] = h * (s_old[r] + s[r]) - T::TWO * delta;
        }
        Ok(())
    }

    fn rejected(&self) -> usize {
        self.rejected
    }
}
