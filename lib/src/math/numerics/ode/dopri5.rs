//! Dormand and Prince's explicit Runge–Kutta pair of orders 5 and 4, as in the code DOPRI5 of
//! Hairer, Nørsett and Wanner.

use super::control::{Limits, Lund, all_finite, combine, error_norm, first_step, weighted};
use super::tableau::{ButcherTableau, DOPRI5};
use super::{DenseStep, OdeFailure, OdeMethod, OdeOptions};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};
use crate::math::numerics::Real;

/// The stabilisation of DOPRI5's step-size controller.
const BETA: f64 = 0.04;
/// The least and the most a step is multiplied by, from one to the next.
const FAC1: f64 = 0.2;
const FAC2: f64 = 10.0;

/// Dormand and Prince's explicit Runge–Kutta pair of orders 5 and 4[^dp], as Hairer, Nørsett and
/// Wanner's DOPRI5 implements it[^hnw]: the method to reach for on a non-stiff problem at
/// tolerances from `1e-3` to about `1e-8`.
///
/// * **Stages.** Seven, of which the last is the right-hand side at the end of the step and so
///   the first of the next one: six evaluations for every step tried, accepted or not.
/// * **Local extrapolation.** The solution of order 5 is the one propagated.
/// * **Error estimate.** The difference of the solutions of orders 5 and 4, from the same stages,
///   in the norm of the module.
/// * **Step-size control.** Gustafsson's PI controller with DOPRI5's constants: after a step of
///   size `h` and error `err`, the next is `h · min(10, max(0.2, 0.9 · err^(−0.17) ·
///   errold^(0.04)))`, with `errold` the error of the accepted step before it. A rejected step
///   is retried at `h · max(0.2, 0.9 · err^(−0.17))`, and the step after a rejection does not
///   grow.
/// * **First step.** Hairer's `hinit`, unless [`OdeOptions::h0`] gives one.
/// * **Dense output.** DOPRI5's continuous extension of order 4, which is free: it uses the seven
///   stages of the step and has a continuous first derivative across the steps.
///
/// What DOPRI5 has and this does not is the test for stiffness: a stiff problem is not
/// diagnosed, it makes the method take the many small steps stability forces on it until
/// `max_steps` is reached.
///
/// [^dp]: J. R. Dormand, P. J. Prince, "A family of embedded Runge–Kutta formulae", *Journal of
///     Computational and Applied Mathematics* 6 (1980), 19–26.
///
/// [^hnw]: E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I:
///     Nonstiff Problems*, 2nd edition, Springer, 1993, sections II.4 to II.6 and IV.2.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince5, OdeOptions, integrate_with};
///
/// // The harmonic oscillator y₀′ = y₁, y₁′ = −y₀ round a quarter of its period.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = y[(1, 0)];
///     dy[(1, 0)] = -y[(0, 0)];
/// };
/// let y0 = Matrix::from_rows(&[[1.0], [0.0]]);
/// let opts = OdeOptions::new().with_tolerances(1e-10, 1e-12);
/// let r = integrate_with(f, &y0, 0.0, core::f64::consts::FRAC_PI_2, DormandPrince5::new(), &opts)
///     .unwrap();
/// assert!(r.y[(0, 0)].abs() < 1e-9);
/// assert!((r.y[(1, 0)] + 1.0).abs() < 1e-9);
/// // Six evaluations a step, and two to start with.
/// assert_eq!(r.nfev, 2 + 6 * (r.steps + r.rejected));
/// ```
#[derive(Debug, Clone)]
pub struct DormandPrince5<T> {
    /// Whether [`OdeMethod::start`] has placed the method at an initial point.
    started: bool,
    /// The limits of the integration in hand.
    lim: Limits<T>,
    /// The step-size controller.
    control: Lund<T>,
    /// The time reached, and the time the last accepted step started from.
    t: T,
    t_old: T,
    /// The size of the next step to try, signed, and that of the last accepted one.
    h: T,
    h_used: T,
    /// The state at `t` and at `t_old`.
    y: Matrix<T>,
    y_old: Matrix<T>,
    /// The seven stages of the last step tried. After an accepted step the seventh is the first
    /// of the next one, and is moved there when that step begins — until then the stages are
    /// those of the continuous extension.
    k: Vec<Matrix<T>>,
    carry: bool,
    /// The argument of a stage, and the end of a step before it is accepted.
    stage: Matrix<T>,
    y_new: Matrix<T>,
    /// The error estimate of a step, by component.
    err: Vec<T>,
    /// Whether the last step tried was rejected.
    reject: bool,
    /// The number of steps rejected.
    rejected: usize,
}

impl<T: Real + Scalar> Default for DormandPrince5<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real + Scalar> DormandPrince5<T> {
    /// Dormand and Prince's 5(4) pair.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::{DormandPrince5, OdeMethod};
    ///
    /// assert_eq!(DormandPrince5::<f64>::new().order(), 5);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        DormandPrince5 {
            started: false,
            lim: Limits::idle(),
            control: Lund::new(0.2 - 0.75 * BETA, BETA, FAC1, FAC2),
            t: T::ZERO,
            t_old: T::ZERO,
            h: T::ZERO,
            h_used: T::ZERO,
            y: Matrix::zeros(0, 1),
            y_old: Matrix::zeros(0, 1),
            k: Vec::new(),
            carry: false,
            stage: Matrix::zeros(0, 1),
            y_new: Matrix::zeros(0, 1),
            err: Vec::new(),
            reject: false,
            rejected: 0,
        }
    }

    /// The coefficients of the pair: seven stages, the weights of order 5 in `b`, the difference
    /// of the weights of orders 5 and 4 in the one row of `e`, and in the one row of `d` the
    /// coefficient of the continuous extension that the ends of the step do not fix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::DormandPrince5;
    ///
    /// let tableau = DormandPrince5::<f64>::new().tableau();
    /// assert_eq!(tableau.c.len(), 7);
    /// assert_eq!(tableau.b[0], 35.0 / 384.0);
    /// assert_eq!(tableau.e[0][6], -1.0 / 40.0);
    /// ```
    #[must_use]
    pub fn tableau(&self) -> ButcherTableau {
        DOPRI5
    }

    /// The stages of a step of size `h` from the point reached to `t_new`, the first stage being
    /// in place: leaves the end of the step in `y_new`, the right-hand side there in the seventh
    /// stage and the error estimate in `err`. Whether everything came out finite.
    fn stages<F>(&mut self, f: &mut F, h: T, t_new: T) -> bool
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        for i in 1..6 {
            combine(
                self.stage.as_mut_slice(),
                self.y.as_slice(),
                h,
                DOPRI5.a[i],
                &self.k,
            );
            let ti = if i == 5 {
                t_new
            } else {
                self.t + T::from_f64(DOPRI5.c[i]) * h
            };
            f(ti, &self.stage, &mut self.k[i]);
            if !all_finite(self.k[i].as_slice()) {
                return false;
            }
        }
        // The seventh row of the tableau is the weights: its stage is evaluated at the end of
        // the step itself.
        combine(
            self.y_new.as_mut_slice(),
            self.y.as_slice(),
            h,
            DOPRI5.b,
            &self.k,
        );
        if !all_finite(self.y_new.as_slice()) {
            return false;
        }
        f(t_new, &self.y_new, &mut self.k[6]);
        if !all_finite(self.k[6].as_slice()) {
            return false;
        }
        weighted(&mut self.err, DOPRI5.e[0], &self.k);
        for e in &mut self.err {
            *e *= h;
        }
        true
    }
}

impl<T: Real + Scalar> OdeMethod<T> for DormandPrince5<T> {
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
        self.started = false;
        self.lim = Limits::new(t0, t1, opts);
        self.control.reset();
        self.t = t0;
        self.t_old = t0;
        self.y = y0.clone();
        self.y_old = y0.clone();
        self.stage = y0.clone();
        self.y_new = y0.clone();
        self.k = (0..7).map(|_| y0.clone()).collect();
        self.err = vec![T::ZERO; y0.as_slice().len()];
        self.carry = false;
        self.reject = false;
        self.rejected = 0;
        self.h_used = T::ZERO;
        f(t0, &self.y, &mut self.k[0]);
        if !all_finite(self.k[0].as_slice()) {
            return Err(OdeFailure::NonFinite);
        }
        let (first, rest) = self.k.split_at_mut(1);
        // The error estimate of the pair shrinks as the fifth power of the step.
        self.h = first_step(
            f,
            t0,
            &self.y,
            &first[0],
            5,
            opts.h0,
            &self.lim,
            &mut self.stage,
            &mut rest[0],
        );
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
        if self.carry {
            self.k.swap(0, 6);
            self.carry = false;
        }
        // The attempt just rejected from this point, which a retry must be shorter than.
        let mut rejected = None;
        loop {
            if self.lim.out_of_reach(self.y.as_slice()) {
                return Err(OdeFailure::ToleranceTooSmall);
            }
            let (h, t_new, last) = self.lim.clip(self.t, self.h, rejected)?;
            rejected = Some(h);
            let err = if self.stages(f, h, t_new) {
                error_norm(
                    &self.err,
                    self.y.as_slice(),
                    self.y_new.as_slice(),
                    self.lim.rtol,
                    self.lim.atol,
                )
            } else {
                T::INFINITY
            };
            if err <= T::ONE {
                let mut h_new = self.control.accepted(h, err);
                if self.reject && Float::abs(h_new) > Float::abs(h) {
                    h_new = h;
                }
                self.reject = false;
                core::mem::swap(&mut self.y, &mut self.y_old);
                core::mem::swap(&mut self.y, &mut self.y_new);
                self.t_old = self.t;
                self.t = t_new;
                self.h_used = h;
                self.h = h_new;
                self.carry = true;
                // The flag is only there to say that nothing is left to do.
                let _ = last;
                return Ok(());
            }
            self.h = self.lim.shrink(h, self.control.rejected(err));
            self.reject = true;
            self.rejected += 1;
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
        // DOPRI5's `rcont1 … rcont5`, in the nested form
        //     y_old + θ·(r₂ + (1 − θ)·(r₃ + θ·(r₄ + (1 − θ)·r₅))),
        // which is the Newton form on the nodes 0, 1, 0, 1 with the signs of r₃ and r₄ turned.
        let n = self.y.as_slice().len();
        let h = self.h_used;
        let nodes = [T::ZERO, T::ONE, T::ZERO, T::ONE];
        out.reset(self.t_old, self.t, n, &nodes);
        out.coefficient_mut(0)
            .copy_from_slice(self.y_old.as_slice());
        weighted(out.coefficient_mut(4), DOPRI5.d[0], &self.k);
        let (y_old, y) = (self.y_old.as_slice(), self.y.as_slice());
        let (k1, k7) = (self.k[0].as_slice(), self.k[6].as_slice());
        for r in 0..n {
            let ydiff = y[r] - y_old[r];
            let bspl = h * k1[r] - ydiff;
            out.coefficient_mut(1)[r] = ydiff;
            out.coefficient_mut(2)[r] = -bspl;
            out.coefficient_mut(3)[r] = -(ydiff - h * k7[r] - bspl);
            out.coefficient_mut(4)[r] *= h;
        }
        Ok(())
    }

    fn rejected(&self) -> usize {
        self.rejected
    }
}
