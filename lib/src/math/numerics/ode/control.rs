//! What the methods share of step-size control: the norm an error is measured in, the limits a
//! step is held to, the first step, and the controller of the two Dormand–Prince codes.
//!
//! Each method keeps its own controller, since each estimates its error in its own way; what is
//! here is the part that has to be the same for all of them, so that `rtol`, `atol`, `h0`, `hmax`
//! and `hmin` mean one thing whichever method is asked.

use super::{OdeFailure, OdeOptions};
use crate::math::matrix::Matrix;
use crate::math::numerics::Real;

/// Whether every element of `v` is finite.
pub(super) fn all_finite<T: Real>(v: &[T]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// `e / sc`, where a scale of nought is met by an error of nought and by nothing else: a
/// component that is exactly zero and exactly right is within any tolerance, a relative one
/// included.
#[inline]
pub(super) fn in_scale<T: Real>(e: T, sc: T) -> T {
    if sc > T::ZERO {
        e / sc
    } else if e == T::ZERO {
        T::ZERO
    } else {
        T::INFINITY
    }
}

/// The sum of the squares of `err[i] / (atol + rtol · max(|y[i]|, |y_new[i]|))`.
pub(super) fn scaled_squares<T: Real>(err: &[T], y: &[T], y_new: &[T], rtol: T, atol: T) -> T {
    let mut sum = T::ZERO;
    for ((&e, &a), &b) in err.iter().zip(y).zip(y_new) {
        let q = in_scale(e, atol + rtol * a.abs().max(b.abs()));
        sum += q * q;
    }
    sum
}

/// The error norm of Hairer's codes, which every method here is controlled by: the root mean
/// square of the components of `err`, each divided by `atol + rtol · max(|y|, |y_new|)`. A step is
/// accepted when it is at most one. An empty state has no error.
pub(super) fn error_norm<T: Real>(err: &[T], y: &[T], y_new: &[T], rtol: T, atol: T) -> T {
    if err.is_empty() {
        return T::ZERO;
    }
    let count = T::from_f64(err.len() as f64);
    (scaled_squares(err, y, y_new, rtol, atol) / count).sqrt()
}

/// `out[r] = y[r] + h · Σⱼ w[j] · k[j][r]`: the argument of a stage, or the end of a step. The sum
/// runs over the stages in their order and skips the zeros of the tableau.
pub(super) fn combine<T: Real>(out: &mut [T], y: &[T], h: T, w: &[f64], k: &[Matrix<T>]) {
    for (r, (o, &yr)) in out.iter_mut().zip(y).enumerate() {
        let mut sum = T::ZERO;
        for (&wj, kj) in w.iter().zip(k) {
            if wj != 0.0 {
                sum += T::from_f64(wj) * kj.as_slice()[r];
            }
        }
        *o = yr + h * sum;
    }
}

/// `out[r] = Σⱼ w[j] · k[j][r]`, in the same order and with the same zeros skipped.
pub(super) fn weighted<T: Real>(out: &mut [T], w: &[f64], k: &[Matrix<T>]) {
    for (r, o) in out.iter_mut().enumerate() {
        let mut sum = T::ZERO;
        for (&wj, kj) in w.iter().zip(k) {
            if wj != 0.0 {
                sum += T::from_f64(wj) * kj.as_slice()[r];
            }
        }
        *o = sum;
    }
}

/// The limits of one integration, fixed when it starts: the tolerances, the end, the direction
/// and the largest and smallest steps.
#[derive(Debug, Clone, Copy)]
pub(super) struct Limits<T> {
    /// The relative tolerance.
    pub rtol: T,
    /// The absolute tolerance.
    pub atol: T,
    /// The time the integration ends at, which no step passes.
    pub t1: T,
    /// `1` for an integration forwards in time, `−1` for one backwards.
    pub dir: T,
    /// The largest size of a step, never more than the interval.
    pub hmax: T,
    /// The smallest size of a step the caller allows.
    pub hmin: T,
}

impl<T: Real> Limits<T> {
    /// The limits of an integration from `t0` to `t1` under `opts`.
    pub fn new(t0: T, t1: T, opts: &OdeOptions<T>) -> Self {
        let span = (t1 - t0).abs();
        Limits {
            rtol: opts.rtol,
            atol: opts.atol,
            t1,
            dir: if t1 < t0 { -T::ONE } else { T::ONE },
            hmax: if opts.hmax < span { opts.hmax } else { span },
            hmin: opts.hmin,
        }
    }

    /// The limits before any integration has started, which no step satisfies.
    pub fn idle() -> Self {
        Limits {
            rtol: T::ZERO,
            atol: T::ZERO,
            t1: T::ZERO,
            dir: T::ONE,
            hmax: T::ZERO,
            hmin: T::ZERO,
        }
    }

    /// The smallest step that can be taken from `t`, but for the last one: the caller's `hmin`,
    /// and never less than ten units in the last place of `t`, below which `t + h` is hardly a
    /// different number — the test of DOPRI5, `0.1·|h| ≤ |t|·ε`.
    fn floor(&self, t: T) -> T {
        let resolution = T::from_f64(10.0) * T::EPSILON * t.abs();
        self.hmin.max(resolution).max(T::MIN_POSITIVE)
    }

    /// The step really taken from `t` when the controller proposes `h`: `(h, t + h, last)`, where
    /// `last` says that the step ends the integration. `rejected` is `None` for the first attempt
    /// of a step, and the step just rejected from `t` for a retry.
    ///
    /// A first attempt is held to the [floor](Self::floor) and to `hmax`: a proposal below the
    /// floor — `hmin`, or ten units in the last place of `t` — is raised to it, as LSODE raises a
    /// step to its `hmin`, and one above `hmax` is cut to it. As in Hairer's codes, a step that
    /// would end within a hundredth of its length of `t1` is stretched or cut to end there
    /// exactly, which spares a last step of next to nothing — unless the stretch would take it
    /// past `hmax`, which is never exceeded: what is left is then covered in two steps of half of
    /// it, or, when a half would be below the floor, in a step of `hmax` and what remains. That
    /// last step is taken whatever its size, so an interval shorter than the floor is tried as
    /// one step.
    ///
    /// A retry is shorter than the step rejected, which ended on `t1` at the latest, so it is
    /// never the last step and is not stretched. It is refused with [`OdeFailure::StepTooSmall`]
    /// when it is below the floor, or when the clock does not make it shorter than the step
    /// rejected: the attempts from one point shrink strictly until one is accepted or the floor
    /// is reached, so no step is retried for ever.
    ///
    /// The size of a step that does not end the integration is recomputed from the time it leads
    /// to, so that the stages are spaced by the step the clock really makes: over a long interval
    /// far from zero the rounding of `t + h` would otherwise add up to a visible error in the
    /// time integrated.
    pub fn clip(&self, t: T, h: T, rejected: Option<T>) -> Result<(T, T, bool), OdeFailure> {
        // A NaN proposal fails the test rather than passing it.
        if h.is_nan() || h * self.dir <= T::ZERO {
            return Err(OdeFailure::StepTooSmall);
        }
        let floor = self.floor(t);
        if let Some(old) = rejected {
            if h.abs() < floor {
                return Err(OdeFailure::StepTooSmall);
            }
            let t_new = t + h;
            let taken = t_new - t;
            if taken.abs() >= old.abs() || t_new == t {
                return Err(OdeFailure::StepTooSmall);
            }
            return Ok((taken, t_new, false));
        }
        let mut h = self.dir * h.abs().max(floor).min(self.hmax);
        let rest = self.t1 - t;
        let stretch = T::from_f64(1.01);
        // Hairer's test, and the same on what is left, for a step of a few units in the last
        // place of `t`, whose hundredth the rounding of `t + 1.01·h` loses.
        if (t + stretch * h - self.t1) * self.dir > T::ZERO || rest.abs() <= stretch * h.abs() {
            if rest.abs() <= self.hmax {
                return Ok((rest, self.t1, true));
            }
            h = rest * T::HALF;
            if h.abs() < floor {
                h = self.dir * self.hmax;
            }
        }
        // Reached only when the caller's `hmax` is below `10·ε·|t|`.
        if h.abs() < floor {
            return Err(OdeFailure::StepTooSmall);
        }
        let t_new = t + h;
        if t_new == t {
            return Err(OdeFailure::StepTooSmall);
        }
        Ok((t_new - t, t_new, false))
    }

    /// The step to try after one of size `h` was rejected, `factor` being what the controller
    /// asks for: `h · factor`, but `hmin` itself is tried before the integration is given up, so
    /// a step is only refused for being too small once the smallest allowed one has failed. A
    /// factor that would not shorten the step, or that is not a number, is taken as a half, so
    /// that every retry is shorter than the attempt before it.
    pub fn shrink(&self, h: T, factor: T) -> T {
        let factor = if factor > T::ZERO && factor < T::ONE {
            factor
        } else {
            T::HALF
        };
        let wanted = h * factor;
        if wanted.abs() < self.hmin && h.abs() > self.hmin {
            self.dir * self.hmin
        } else {
            wanted
        }
    }

    /// Whether the tolerances ask for more than the arithmetic holds at the state `y`: the test
    /// of LSODE and CVODE, `ε · ‖y‖ > 1` in the norm the error is measured in. A step cannot be
    /// more accurate than the rounding of the state it ends in, and an integration that went on
    /// regardless would report a tolerance it had not met.
    pub fn out_of_reach(&self, y: &[T]) -> bool {
        if y.is_empty() {
            return false;
        }
        let count = T::from_f64(y.len() as f64);
        let norm = (scaled_squares(y, y, y, self.rtol, self.atol) / count).sqrt();
        // A NaN norm cannot arise from a finite state; an infinite one is out of reach.
        T::EPSILON * norm > T::ONE
    }
}

/// The first step of an integration whose `h0` was left to the method: Hairer's `hinit`.
///
/// A first guess makes an explicit Euler step change the state by a hundredth of its size, in the
/// norm of the tolerances; one evaluation at the end of that step estimates the second derivative,
/// and the step returned is the one for which `h^order · max(‖f‖, ‖y″‖) = 0.01` — never more than
/// a hundred times the first guess, nor than `hmax`. The norms are the sums of DOPRI5's `hinit`,
/// which does not divide by the number of components as the book's description of it does[^1].
///
/// `f0` is `f(t0, y0)`, `order` the exponent the method's error estimate shrinks by, and `y1` and
/// `f1` are work space. The step is signed by `lim.dir`. One evaluation of `f`.
///
/// [^1]: E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I*,
///     Springer, 1993, section II.4, "Starting Step Size".
#[allow(clippy::too_many_arguments)]
pub(super) fn initial_step<T, F>(
    f: &mut F,
    t0: T,
    y0: &Matrix<T>,
    f0: &Matrix<T>,
    order: usize,
    lim: &Limits<T>,
    y1: &mut Matrix<T>,
    f1: &mut Matrix<T>,
) -> T
where
    T: Real,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
{
    let (rtol, atol) = (lim.rtol, lim.atol);
    let mut dnf = T::ZERO;
    let mut dny = T::ZERO;
    for (&y, &d) in y0.as_slice().iter().zip(f0.as_slice()) {
        let sk = atol + rtol * y.abs();
        let (qf, qy) = (in_scale(d, sk), in_scale(y, sk));
        dnf += qf * qf;
        dny += qy * qy;
    }
    let tiny = T::from_f64(1e-10);
    let mut h = if dnf <= tiny || dny <= tiny {
        T::from_f64(1e-6)
    } else {
        (dny / dnf).sqrt() * T::from_f64(0.01)
    };
    h = h.min(lim.hmax);
    if !h.is_finite() || h <= T::ZERO {
        // A derivative beyond the range of the squares: nothing can be learnt from it, and the
        // controller is left to find its way from the smallest guess.
        return lim.dir * T::from_f64(1e-6).min(lim.hmax);
    }
    // An explicit Euler step, and the second derivative it shows.
    let signed = lim.dir * h;
    for ((o, &y), &d) in y1
        .as_mut_slice()
        .iter_mut()
        .zip(y0.as_slice())
        .zip(f0.as_slice())
    {
        *o = y + signed * d;
    }
    f(t0 + signed, y1, f1);
    if !all_finite(f1.as_slice()) {
        // The right-hand side ends within the first guess: start well inside it.
        return signed * T::from_f64(1e-3);
    }
    let mut der2 = T::ZERO;
    for ((&a, &b), &y) in f1.as_slice().iter().zip(f0.as_slice()).zip(y0.as_slice()) {
        let q = in_scale(a - b, atol + rtol * y.abs());
        der2 += q * q;
    }
    let der2 = der2.sqrt() / h;
    let der12 = der2.max(dnf.sqrt());
    let h1 = if der12 <= T::from_f64(1e-15) {
        T::from_f64(1e-6).max(h * T::from_f64(1e-3))
    } else {
        (T::from_f64(0.01) / der12).powf(T::ONE / T::from_f64(order as f64))
    };
    let mut step = (T::from_f64(100.0) * h).min(lim.hmax);
    if h1 < step {
        step = h1;
    }
    if !step.is_finite() || step <= T::ZERO {
        step = h;
    }
    lim.dir * step
}

/// The first step of an integration: the caller's `h0` when there is one, cut to the interval and
/// to `hmax`, and [`initial_step`] otherwise, raised to `hmin` if it falls below.
#[allow(clippy::too_many_arguments)]
pub(super) fn first_step<T, F>(
    f: &mut F,
    t0: T,
    y0: &Matrix<T>,
    f0: &Matrix<T>,
    order: usize,
    h0: T,
    lim: &Limits<T>,
    y1: &mut Matrix<T>,
    f1: &mut Matrix<T>,
) -> T
where
    T: Real,
    F: FnMut(T, &Matrix<T>, &mut Matrix<T>),
{
    if h0 > T::ZERO {
        return lim.dir * h0.min(lim.hmax);
    }
    let h = initial_step(f, t0, y0, f0, order, lim, y1, f1);
    if h.abs() < lim.hmin {
        lim.dir * lim.hmin.min(lim.hmax)
    } else {
        h
    }
}

/// The step-size controller of DOPRI5 and DOP853: Gustafsson's proportional–integral rule, which
/// Hairer's codes call the Lund stabilisation.
///
/// After a step of size `h` with error `err` in the norm of the tolerances, the next is `h/fac`
/// with `fac = err^expo · errold^(−beta) / safe`, held to `[1/fac2, 1/fac1]`, where `errold` is
/// the error of the last accepted step and `expo = 1/q − 0.75·beta` (DOPRI5, `q = 5`) or
/// `1/q − 0.2·beta` (DOP853, `q = 8`). With `beta = 0` it is the textbook rule
/// `h · safe · err^(−1/q)`; a positive `beta` damps the oscillation of the step size that rule
/// falls into when the step is limited by stability rather than by accuracy.
#[derive(Debug, Clone, Copy)]
pub(super) struct Lund<T> {
    /// The exponent of the error of the step in hand.
    expo: T,
    /// The exponent of the error of the step before it.
    beta: T,
    /// The safety factor.
    safe: T,
    /// `1/fac1`: the most a step is divided by.
    facc1: T,
    /// `1/fac2`: the least a step is divided by, which is the most it grows.
    facc2: T,
    /// The error of the last accepted step, at least `1e-4`.
    errold: T,
}

impl<T: Real> Lund<T> {
    /// The controller with the exponent `expo`, the stabilisation `beta` and a step held between
    /// `fac1` and `fac2` times the one before it.
    pub fn new(expo: f64, beta: f64, fac1: f64, fac2: f64) -> Self {
        Lund {
            expo: T::from_f64(expo),
            beta: T::from_f64(beta),
            safe: T::from_f64(0.9),
            facc1: T::from_f64(1.0 / fac1),
            facc2: T::from_f64(1.0 / fac2),
            errold: T::from_f64(1e-4),
        }
    }

    /// Forgets the last accepted step, as at the start of an integration.
    pub fn reset(&mut self) {
        self.errold = T::from_f64(1e-4);
    }

    /// The step after an accepted one of size `h` and error `err ≤ 1`, which becomes the error
    /// the next proposal is damped by.
    pub fn accepted(&mut self, h: T, err: T) -> T {
        let fac11 = err.powf(self.expo);
        let fac = fac11 / self.errold.powf(self.beta);
        let fac = self.facc2.max(self.facc1.min(fac / self.safe));
        self.errold = err.max(T::from_f64(1e-4));
        h / fac
    }

    /// The factor a rejected step of error `err > 1` is multiplied by: `safe · err^(−expo)`, and
    /// never less than `fac1`, which is also what a step whose error is not a number gets.
    pub fn rejected(&self, err: T) -> T {
        if err.is_finite() {
            T::ONE / self.facc1.min(err.powf(self.expo) / self.safe)
        } else {
            T::ONE / self.facc1
        }
    }
}
