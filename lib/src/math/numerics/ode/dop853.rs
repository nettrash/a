//! The explicit Runge–Kutta method of order 8 of Dormand and Prince with Hairer's embedded
//! formulas of orders 5 and 3, as in the code DOP853.

use super::control::{Limits, Lund, all_finite, combine, first_step, scaled_squares, weighted};
use super::tableau::{ButcherTableau, DOP853};
use super::{DenseStep, OdeFailure, OdeMethod, OdeOptions};
use crate::math::complex::Float;
use crate::math::matrix::{Matrix, Scalar};
use crate::math::numerics::Real;

/// The stabilisation of DOP853's step-size controller, which is none by default.
const BETA: f64 = 0.0;
/// The least and the most a step is multiplied by, from one to the next.
const FAC1: f64 = 0.333;
const FAC2: f64 = 6.0;

/// The explicit Runge–Kutta method of order 8 of Dormand and Prince[^pd] with the embedded
/// formulas of orders 5 and 3, as Hairer, Nørsett and Wanner's DOP853 implements it[^hnw]: the
/// method for a non-stiff problem at stringent tolerances, where its order pays for its twelve
/// stages.
///
/// * **Stages.** Twelve for the step, and the right-hand side at the end of an accepted step,
///   which is the first stage of the next: eleven evaluations for every step tried, and one more
///   for every step accepted.
/// * **Error estimate.** DOP853's: with `err₅` and `err₃` the differences of the solution of
///   order 8 and the embedded ones of orders 5 and 3 divided by `h` — the sums of the stages
///   that DOP853 forms them from, of the orders `h⁵` and `h³` — each in the norm of the module,
///   `err = |h| · err₅² / √(err₅² + 0.01 · err₃²)`. It behaves as `h⁸` and vanishes, as it
///   should, where the formula of order 5 is already exact.
/// * **Step-size control.** After a step of size `h` and error `err`, the next is
///   `h · min(6, max(0.333, 0.9 · err^(−1/8)))`; a rejected step is retried at
///   `h · max(0.333, 0.9 · err^(−1/8))`, and the step after a rejection does not grow.
/// * **First step.** Hairer's `hinit`, unless [`OdeOptions::h0`] gives one.
/// * **Dense output.** DOP853's continuous extension of order 7. It needs three more evaluations
///   for a step, which are made only for the steps whose continuous extension is asked for: all
///   of them when the dense output is kept, and otherwise those an event or an output time falls
///   in.
///
/// What DOP853 has and this does not is the test for stiffness: see [`DormandPrince5`].
///
/// [`DormandPrince5`]: super::DormandPrince5
///
/// [^pd]: P. J. Prince, J. R. Dormand, "High order embedded Runge–Kutta formulae", *Journal of
///     Computational and Applied Mathematics* 7 (1981), 67–75.
///
/// [^hnw]: E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I:
///     Nonstiff Problems*, 2nd edition, Springer, 1993, sections II.5, II.6 and II.10.
///
/// # Examples
///
/// ```
/// use a::math::matrix::Matrix;
/// use a::math::numerics::{DormandPrince853, OdeOptions, integrate_with};
///
/// // The harmonic oscillator through ten periods, to thirteen digits.
/// let f = |_t: f64, y: &Matrix<f64>, dy: &mut Matrix<f64>| {
///     dy[(0, 0)] = y[(1, 0)];
///     dy[(1, 0)] = -y[(0, 0)];
/// };
/// let y0 = Matrix::from_rows(&[[1.0], [0.0]]);
/// let opts = OdeOptions::new().with_tolerances(1e-13, 1e-15);
/// let end = 20.0 * core::f64::consts::PI;
/// let r = integrate_with(f, &y0, 0.0, end, DormandPrince853::new(), &opts).unwrap();
/// assert!((r.y[(0, 0)] - 1.0).abs() < 1e-11 && r.y[(1, 0)].abs() < 1e-11);
/// // Two evaluations to start, eleven a step tried and one more a step accepted.
/// assert_eq!(r.nfev, 2 + 11 * (r.steps + r.rejected) + r.steps);
/// ```
#[derive(Debug, Clone)]
pub struct DormandPrince853<T> {
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
    /// The sixteen stages: twelve of the step, the right-hand side at its end, and the three of
    /// the continuous extension. After an accepted step the thirteenth is the first of the next
    /// one, and is moved there when that step begins.
    k: Vec<Matrix<T>>,
    carry: bool,
    /// Whether the last three stages belong to the last accepted step.
    extended: bool,
    /// The argument of a stage, and the end of a step before it is accepted.
    stage: Matrix<T>,
    y_new: Matrix<T>,
    /// The two error estimates of a step, by component.
    err5: Vec<T>,
    err3: Vec<T>,
    /// Whether the last step tried was rejected.
    reject: bool,
    /// The number of steps rejected.
    rejected: usize,
}

impl<T: Real + Scalar> Default for DormandPrince853<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real + Scalar> DormandPrince853<T> {
    /// The method of order 8 with its embedded formulas of orders 5 and 3.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::{DormandPrince853, OdeMethod};
    ///
    /// assert_eq!(DormandPrince853::<f64>::new().order(), 8);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        DormandPrince853 {
            started: false,
            lim: Limits::idle(),
            control: Lund::new(1.0 / 8.0 - 0.2 * BETA, BETA, FAC1, FAC2),
            t: T::ZERO,
            t_old: T::ZERO,
            h: T::ZERO,
            h_used: T::ZERO,
            y: Matrix::zeros(0, 1),
            y_old: Matrix::zeros(0, 1),
            k: Vec::new(),
            carry: false,
            extended: false,
            stage: Matrix::zeros(0, 1),
            y_new: Matrix::zeros(0, 1),
            err5: Vec::new(),
            err3: Vec::new(),
            reject: false,
            rejected: 0,
        }
    }

    /// The coefficients of the method: sixteen stages, of which the first twelve make the step,
    /// the thirteenth is the right-hand side at its end and the last three serve the continuous
    /// extension; the weights of order 8 in `b`; in `e` the differences of those weights and the
    /// ones of order 5, then of order 3; and in the four rows of `d` the coefficients of the
    /// continuous extension that the ends of the step do not fix.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::DormandPrince853;
    ///
    /// let tableau = DormandPrince853::<f64>::new().tableau();
    /// assert_eq!((tableau.c.len(), tableau.b.len()), (16, 12));
    /// assert_eq!((tableau.e.len(), tableau.d.len()), (2, 4));
    /// // The thirteenth stage is evaluated at the end of the step: its row is the weights.
    /// assert_eq!(tableau.a[12], tableau.b);
    /// ```
    #[must_use]
    pub fn tableau(&self) -> ButcherTableau {
        DOP853
    }

    /// The twelve stages of a step of size `h` from the point reached to `t_new`, the first being
    /// in place: leaves the end of the step in `y_new` and the two error estimates, not yet
    /// multiplied by `h`, in `err5` and `err3`. Whether everything came out finite.
    fn stages<F>(&mut self, f: &mut F, h: T, t_new: T) -> bool
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        for i in 1..12 {
            combine(
                self.stage.as_mut_slice(),
                self.y.as_slice(),
                h,
                DOP853.a[i],
                &self.k,
            );
            let ti = if i == 11 {
                t_new
            } else {
                self.t + T::from_f64(DOP853.c[i]) * h
            };
            f(ti, &self.stage, &mut self.k[i]);
            if !all_finite(self.k[i].as_slice()) {
                return false;
            }
        }
        combine(
            self.y_new.as_mut_slice(),
            self.y.as_slice(),
            h,
            DOP853.b,
            &self.k,
        );
        weighted(&mut self.err5, DOP853.e[0], &self.k);
        weighted(&mut self.err3, DOP853.e[1], &self.k);
        all_finite(self.y_new.as_slice())
    }

    /// DOP853's error of the step of size `h` whose estimates [`stages`](Self::stages) has left:
    /// `|h| · err₅² / √(err₅² + 0.01 · err₃²)`, with both estimates — the stage sums `err5` and
    /// `err3`, the differences of the solutions before the factor `h` — in the root mean square
    /// norm of the tolerances. When both vanish so does the error.
    fn error(&self, h: T) -> T {
        let n = self.err5.len();
        if n == 0 {
            return T::ZERO;
        }
        let (y, y_new) = (self.y.as_slice(), self.y_new.as_slice());
        let (rtol, atol) = (self.lim.rtol, self.lim.atol);
        let err = scaled_squares(&self.err5, y, y_new, rtol, atol);
        let err2 = scaled_squares(&self.err3, y, y_new, rtol, atol);
        let mut deno = err + T::from_f64(0.01) * err2;
        if deno <= T::ZERO {
            deno = T::ONE;
        }
        Float::abs(h) * err * Float::sqrt(T::ONE / (T::from_f64(n as f64) * deno))
    }
}

impl<T: Real + Scalar> OdeMethod<T> for DormandPrince853<T> {
    fn order(&self) -> usize {
        8
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
        self.k = (0..16).map(|_| y0.clone()).collect();
        self.err5 = vec![T::ZERO; y0.as_slice().len()];
        self.err3 = vec![T::ZERO; y0.as_slice().len()];
        self.carry = false;
        self.extended = false;
        self.reject = false;
        self.rejected = 0;
        self.h_used = T::ZERO;
        f(t0, &self.y, &mut self.k[0]);
        if !all_finite(self.k[0].as_slice()) {
            return Err(OdeFailure::NonFinite);
        }
        let (first, rest) = self.k.split_at_mut(1);
        // DOP853 starts as a method whose error shrinks as the eighth power of the step.
        self.h = first_step(
            f,
            t0,
            &self.y,
            &first[0],
            8,
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
            self.k.swap(0, 12);
            self.carry = false;
        }
        self.extended = false;
        // The attempt just rejected from this point, which a retry must be shorter than.
        let mut rejected = None;
        loop {
            if self.lim.out_of_reach(self.y.as_slice()) {
                return Err(OdeFailure::ToleranceTooSmall);
            }
            let (h, t_new, _last) = self.lim.clip(self.t, self.h, rejected)?;
            rejected = Some(h);
            let mut err = if self.stages(f, h, t_new) {
                self.error(h)
            } else {
                T::INFINITY
            };
            if err <= T::ONE {
                // The right-hand side at the end of the step: the first stage of the next one,
                // and a point the solution must be defined at for the step to stand.
                f(t_new, &self.y_new, &mut self.k[12]);
                if !all_finite(self.k[12].as_slice()) {
                    err = T::INFINITY;
                }
            }
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

    fn dense<F>(&mut self, f: &mut F, out: &mut DenseStep<T>) -> Result<(), OdeFailure>
    where
        F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
    {
        // Before `start` there are no stages to extend.
        if !self.started {
            return Err(OdeFailure::InvalidInput);
        }
        let n = self.y.as_slice().len();
        let h = self.h_used;
        if !self.extended {
            // The three stages that only the continuous extension needs, from the thirteen the
            // step has left.
            for i in 13..16 {
                combine(
                    self.stage.as_mut_slice(),
                    self.y_old.as_slice(),
                    h,
                    DOP853.a[i],
                    &self.k,
                );
                let ti = self.t_old + T::from_f64(DOP853.c[i]) * h;
                f(ti, &self.stage, &mut self.k[i]);
                if !all_finite(self.k[i].as_slice()) {
                    return Err(OdeFailure::NonFinite);
                }
            }
            self.extended = true;
        }
        // DOP853's `cont`, in the nested form
        //     y_old + θ·(r₁ + (1 − θ)·(r₂ + θ·(r₃ + (1 − θ)·(r₄ + θ·(r₅ + (1 − θ)·(r₆ + θ·r₇)))))),
        // which is the Newton form on the nodes 0, 1, 0, 1, 0, 1, 0 with the signs of r₂, r₃, r₆
        // and r₇ turned.
        let nodes = [T::ZERO, T::ONE, T::ZERO, T::ONE, T::ZERO, T::ONE, T::ZERO];
        out.reset(self.t_old, self.t, n, &nodes);
        out.coefficient_mut(0)
            .copy_from_slice(self.y_old.as_slice());
        for (row, sign) in [(0, T::ONE), (1, T::ONE), (2, -T::ONE), (3, -T::ONE)] {
            let c = out.coefficient_mut(4 + row);
            weighted(c, DOP853.d[row], &self.k);
            for v in c {
                *v *= sign * h;
            }
        }
        let (y_old, y) = (self.y_old.as_slice(), self.y.as_slice());
        let (k1, k13) = (self.k[0].as_slice(), self.k[12].as_slice());
        for r in 0..n {
            let ydiff = y[r] - y_old[r];
            let bspl = h * k1[r] - ydiff;
            out.coefficient_mut(1)[r] = ydiff;
            out.coefficient_mut(2)[r] = -bspl;
            out.coefficient_mut(3)[r] = -(ydiff - h * k13[r] - bspl);
        }
        Ok(())
    }

    fn rejected(&self) -> usize {
        self.rejected
    }
}
