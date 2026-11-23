//! Numerical differentiation, by Richardson extrapolation on central differences.
//!
//! A central difference quotient of step `h` is the derivative plus a series in `h²`, so the
//! quotients of the steps `h`, `h/2`, `h/4`, … can be combined to cancel the even powers of `h` in
//! turn: Neville's scheme, each column of which raises the order of the error by two (C. J. F.
//! Ridders, *Adv. Eng. Software* 4, 1982, whose error estimate is used here). Round-off in the
//! values of `f` grows as the step shrinks, `ε·|f|/h` for the first derivative and `ε·|f|/h²` for
//! the second, so the best answer is somewhere down the ladder of steps, and it is found by
//! watching the table's own error estimate. Exact derivatives through dual numbers are a matter
//! of issue #25 and are not offered here.
//!
//! The step is chosen from the point: the ladder starts at `0.37·max(|x|, 1)` (and, for
//! `0 < |x| < 1`, again at `0.37·|x|` when the first ladder fails or lands near a stationary
//! point), and descends by halves until round-off dominates, the steps reach 16 units in the last
//! place of `x`, or 64 steps have been tried. A function must be smooth on the scale of the steps
//! that give the answer; nothing is assumed about the scale of `x` itself. An answer is returned
//! only when
//!
//! * the table's error estimate (Ridders': the larger of the differences of an entry from its two
//!   neighbours in the table) plus the round-off of the values is at most `tol` times the size of
//!   the derivative or of the one-sided slopes around it (for the second derivative, of the
//!   one-sided second differences), whichever is larger — since a zero has no size of its own, at
//!   or near a stationary point it is the slopes that count, and the error relative to the
//!   derivative itself can be far larger than `tol` (`cosh(0.0103·x)′` at `x = −1.03e−6`: 5.5e−6
//!   at the default `1.5e−8`, the one-sided slopes at the answer's steps, 0.18 and 0.092, being
//!   4.5e4 times the derivative there);
//! * the one-sided quotients agree in the limit (a kink, where they do not, gives `None` rather
//!   than the symmetric quotient's average of two slopes); and
//! * three more quotients, at steps `√2`, `2^(1/3)` and `2^(3/4)` times the finest the answer
//!   used — off the ladder of halvings — agree with what the table's polynomial in `h²` predicts
//!   for them, to the tolerance.
//!
//! A periodic function sampled at steps of many periods can still pass all of that: on the
//! halvings of a step that is a multiple of `2^j` periods it takes the values of a slower
//! sinusoid, whose table settles on the slope of the alias, and at such steps its quotients are of
//! the size of the reciprocal step, small numbers among which a loose tolerance is met by chance.
//! So the ladder does not stop at the first answer that passes. It keeps the one of smallest
//! estimated error relative to its size and goes on down while a finer step could do better; when
//! a finer step shows quotients far larger than the answer kept — the function varying on a scale
//! that answer's steps did not resolve — and no better answer follows, there is no answer. And the
//! answer must agree with the values at the floats next to `x`: one float away they may move by
//! no more than a quarter of what they moved at the answer's finest step, at least 16 floats wide,
//! which a function oscillating on a scale between the two does not satisfy.
//!
//! A function that does not change at all at any step down to where its round-off would hide a
//! derivative of the tolerance's size has the derivative zero. Features narrower than the finest
//! step tried are invisible, as they are to any method that samples. Floats near `x` are what the
//! function is sampled at, and every step is a multiple of their spacing `u`: where `u` is close to
//! a multiple of the function's period (the sine at `1e10` in `f32`, where the floats are 1024
//! apart, 163 periods less 0.03 of one), or many periods long (the sine beyond about `1e17` in
//! `f64`), the function takes at every float there the values of a slower one, of slope about
//! `(u mod period)/u` in units of its amplitude, and is indistinguishable from it: the slope of the
//! sine at 100 points of `[1e234, 1e235]` is given at 11 of them, of size `1e−223`. And the checks
//! see the function as it is computed: one whose own rounding is smooth on the scale of the steps
//! (the phase `w·x + p` of a sine rounded at `w·|x|` near `1e9`) can pass them with that rounding
//! in its slope.
//!
//! No answer is given where the steps are so large beside the values that the round-off estimate
//! of the quotients falls below the smallest subnormal number (the rounding of the values is at
//! least `ε` times the smallest normal one): every quotient is then below the smallest normal
//! number too, and an underflow would pass the relative tests as `0 ≤ 0`. So a derivative smaller
//! than the smallest normal number at such steps gives `None` — the second derivative of a bounded
//! function at `|x|` from about `1e176` (where the steps' squares leave the range of `f64` and the
//! curvature of the sine cannot be known anyway), of `ln` at `1e170` (`−1e−340`, below every
//! subnormal) or of `sin(x/1e160)` at `1e160` (`−8.4e−321`, a subnormal) — while one that is a
//! normal number is found: the round-off `a/h²` is formed as `a/h/h` when `h²` overflows, and the
//! curvature `2e−300` of `1e−300·x²` is found at `1e300`.

use super::Real;

/// The options of the differentiation rules: the first step, the tolerance, and how many columns
/// the table runs to.
#[derive(Debug, Clone, Copy)]
pub struct DiffOptions<T> {
    /// The first step of the ladder, or `T::ZERO` to choose it from the point.
    pub h: T,
    /// The tolerance on the estimated error, relative to the size of the derivative or of the
    /// slopes around a stationary point (see the module's overview). It decides whether an
    /// answer is returned; the answer is the best the table reached, usually far better.
    pub tol: T,
    /// The largest number of columns of the table, at least 2: up to `max_levels − 1`
    /// extrapolations of the quotients.
    pub max_levels: usize,
}

impl<T: Real> Default for DiffOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> DiffOptions<T> {
    /// The default options of [`derivative`]: the step chosen automatically, `tol` `√ε` — about
    /// `1.5e−8` for `f64` and `3.4e−4` for `f32`, which the first derivative of a function
    /// computed to a few units in the last place meets with room to spare — and eight columns.
    /// [`second_derivative`] takes these with `tol` `ε^(1/3)` (`6.1e−6` and `4.9e−3`), since
    /// the round-off of a second difference grows like `1/h²`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::DiffOptions;
    ///
    /// let o = DiffOptions::<f64>::new().with_tol(1e-12).with_max_levels(10);
    /// assert_eq!((o.h, o.tol, o.max_levels), (0.0, 1e-12, 10));
    /// ```
    pub fn new() -> Self {
        Self {
            h: T::ZERO,
            tol: T::EPSILON.sqrt(),
            max_levels: 8,
        }
    }

    /// Sets the first step. Pass `T::ZERO` to have it chosen from the point.
    pub fn with_step(mut self, h: T) -> Self {
        self.h = h;
        self
    }

    /// Sets the tolerance.
    pub fn with_tol(mut self, tol: T) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the largest number of columns.
    pub fn with_max_levels(mut self, max_levels: usize) -> Self {
        self.max_levels = max_levels;
        self
    }
}

/// How many steps one ladder tries at most: from `0.37·max(|x|, 1)` down by a factor of `2⁶³`.
const RUNGS: usize = 64;

/// The first step, a fraction of the scale of the point. Not a power of two, so that the ladder
/// does not share the binary structure of the point and its neighbours.
const START: f64 = 0.367_879_441_171_442_3;

/// The one-sided quotients must agree in the limit within this many times `tol`.
const JUMP: f64 = 16.0;

/// A step whose round-off is this many times the tolerance has nothing left to give.
const STOP: f64 = 1024.0;

/// A rung whose quotients are this many times the size of a verified answer shows the function
/// varying on a finer scale than that answer's steps, and the ladder goes on.
const GROW: f64 = 4.0;

/// The steps of the verification, as multiples of the finest step of an answer: `√2`, `2^(1/3)`
/// and `2^(3/4)`, between that step and the next on the ladder, off its halvings and off each
/// other's.
const RATIOS: &[f64] = &[
    core::f64::consts::SQRT_2,
    1.259_921_049_894_873_2,
    1.681_792_830_507_429,
];

/// The smallest step, in units in the last place of `x`: below it the step is a few floats, on
/// which any function looks smooth.
const GRID: f64 = 16.0;

/// Which derivative.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Order {
    First,
    Second,
}

/// One step of the ladder: the step, the two points and the values there, the central quotient,
/// the difference and the size of the one-sided ones, the round-off of the quotient, and whether
/// the values did not change.
#[derive(Debug, Clone, Copy)]
struct Rung<T> {
    h: T,
    xp: T,
    xm: T,
    fp: T,
    fm: T,
    q: T,
    jump: Option<T>,
    slope: Option<T>,
    noise: T,
    flat: bool,
}

/// An answer of a ladder: the value, its estimated error (table and round-off), the size it was
/// judged against, the quotients `(h, q)` it rests on, and how far the values moved from `f(x)`
/// at the finest of their steps.
#[derive(Debug, Clone)]
struct Answer<T> {
    value: T,
    error: T,
    scale: T,
    window: Vec<(T, T)>,
    swing: T,
}

/// What a ladder found.
enum Found<T> {
    Answer(Answer<T>),
    Kink,
    Nothing,
}

/// The derivative of `f` at `x`, by Richardson extrapolation on central differences, with the
/// default options ([`DiffOptions::new`]).
///
/// An answer is returned only when the table's estimated error and the round-off of the values
/// are within `tol` of the size of the derivative (of the slopes around it at a stationary point),
/// the one-sided quotients agree in the limit, three quotients at steps off the ladder of halvings
/// agree with the table, no finer step shows the function varying on a smaller scale than the
/// steps of the answer, and the values at the floats next to `x` agree with it. So it returns
/// `None` for a function that is not differentiable at `x`, one that varies on a finer scale than
/// the floats near `x` let the steps resolve (unless it takes the values of a slower one at every
/// float there; see the module's overview), usually one computed too roughly for the tolerance
/// (not always: rounding that is smooth on the scale of the steps passes as part of the
/// function), at a kink, when `x` or `f(x)` is not finite, and when the steps are so large beside
/// the values that the quotients and their round-off fall below the smallest normal number (a
/// derivative smaller than that, at such steps).
///
/// # Examples
///
/// ```
/// use a::math::numerics::derivative;
///
/// // The slope of sin at 1 is cos 1, to about fifteen digits.
/// let d = derivative(|x: f64| x.sin(), 1.0).unwrap();
/// assert!((d - 1.0f64.cos()).abs() < 1e-14);
///
/// // A straight line at a point that is not a binary fraction, and the logarithm at 1e-100.
/// assert!((derivative(|x: f64| 7.1 - 3.3 * x, 0.3).unwrap() + 3.3).abs() < 1e-13);
/// assert!((derivative(|x: f64| x.ln(), 1e-100).unwrap() - 1e100).abs() < 1e88);
///
/// // A kink has no derivative, a jump none either.
/// assert_eq!(derivative(|x: f64| x.abs(), 0.0), None);
/// assert_eq!(derivative(|x: f64| x.floor(), 2.0), None);
/// ```
pub fn derivative<T: Real>(f: impl FnMut(T) -> T, x: T) -> Option<T> {
    derivative_with(f, x, &DiffOptions::new())
}

/// [`derivative`] with the options `opts`.
///
/// Returns `None` too when `opts.tol` is not positive and finite, `opts.h` is negative or not
/// finite, or `opts.max_levels` is below 2.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{DiffOptions, derivative_with};
///
/// // A first step of the caller's: the ladder descends from it.
/// let opts = DiffOptions::new().with_step(1e-2);
/// let d = derivative_with(|x: f64| (10.0 * x).sin(), 0.3, &opts).unwrap();
/// assert!((d - 10.0 * 3.0f64.cos()).abs() < 1e-11);
/// // A tolerance no table can meet gives nothing.
/// assert_eq!(derivative_with(|x: f64| x.sin(), 1.0, &DiffOptions::new().with_tol(1e-20)), None);
/// ```
pub fn derivative_with<T: Real>(
    mut f: impl FnMut(T) -> T,
    x: T,
    opts: &DiffOptions<T>,
) -> Option<T> {
    differentiate(&mut f, x, Order::First, opts)
}

/// The second derivative of `f` at `x`, by Richardson extrapolation on central second
/// differences, with the default options and `tol` `ε^(1/3)` (see [`DiffOptions::new`]).
///
/// Returns `None` as [`derivative`] does; a kink of the first derivative is one too.
///
/// # Examples
///
/// ```
/// use a::math::numerics::second_derivative;
///
/// // The curvature of sin at 1 is −sin 1.
/// let d = second_derivative(|x: f64| x.sin(), 1.0).unwrap();
/// assert!((d + 1.0f64.sin()).abs() < 1e-11);
///
/// // x³ − 2x at 1.3 bends by 6·1.3; x·|x| has no curvature at 0, one of −2 and 2 either side.
/// assert!((second_derivative(|x: f64| x * x * x - 2.0 * x, 1.3).unwrap() - 7.8).abs() < 1e-9);
/// assert_eq!(second_derivative(|x: f64| x * x.abs(), 0.0), None);
/// ```
pub fn second_derivative<T: Real>(f: impl FnMut(T) -> T, x: T) -> Option<T> {
    second_derivative_with(f, x, &DiffOptions::new().with_tol(T::EPSILON.cbrt()))
}

/// [`second_derivative`] with the options `opts`; see [`derivative_with`].
///
/// # Examples
///
/// ```
/// use a::math::numerics::{DiffOptions, second_derivative_with};
///
/// let opts = DiffOptions::new().with_step(0.1).with_tol(1e-10);
/// let d = second_derivative_with(|x: f64| x.exp(), 0.0, &opts).unwrap();
/// assert!((d - 1.0).abs() < 1e-10);
/// ```
pub fn second_derivative_with<T: Real>(
    mut f: impl FnMut(T) -> T,
    x: T,
    opts: &DiffOptions<T>,
) -> Option<T> {
    differentiate(&mut f, x, Order::Second, opts)
}

/// Both derivatives: the checks of the arguments, the ladders, and the choice between them.
fn differentiate<T: Real>(
    f: &mut impl FnMut(T) -> T,
    x: T,
    order: Order,
    opts: &DiffOptions<T>,
) -> Option<T> {
    let valid_tol = opts.tol > T::ZERO && opts.tol.is_finite();
    let valid_h = opts.h >= T::ZERO && opts.h.is_finite();
    if !x.is_finite() || !valid_tol || !valid_h || opts.max_levels < 2 {
        return None;
    }
    let fx = f(x);
    if !fx.is_finite() {
        return None;
    }
    let answer = |found: Found<T>| match found {
        Found::Answer(a) => Some(a),
        Found::Kink | Found::Nothing => None,
    };
    if opts.h > T::ZERO {
        let found = answer(ladder(f, x, fx, opts.h, order, opts, false, true));
        return found.and_then(|a| at_the_floats(f, x, fx, &a).then_some(a.value));
    }
    let start = T::from_f64(START);
    let unit = x.abs().max(T::ONE);
    // A point inside (−1, 1) other than 0 has a scale of its own, which a function singular at
    // 0 (the logarithm at 1e-100) needs: a second ladder starts there.
    let near = x != T::ZERO && x.abs() < T::ONE;
    let first = ladder(f, x, fx, start * unit, order, opts, near, true);
    if let Found::Kink = first {
        return None;
    }
    let first = answer(first);
    if near && first.as_ref().is_none_or(|a| a.value.abs() < a.scale) {
        // Nothing settled at the scale of 1, or what did is small beside the slopes around it:
        // the scale of x itself may do better. A function that is flat at that scale is one that
        // varies slowly beside |x|, which the first ladder judges; this one does not.
        if let Some(second) = answer(ladder(f, x, fx, start * x.abs(), order, opts, false, false))
            && first.as_ref().is_none_or(|a| second.error < a.error)
        {
            return at_the_floats(f, x, fx, &second).then_some(second.value);
        }
    }
    first.and_then(|a| at_the_floats(f, x, fx, &a).then_some(a.value))
}

/// Whether the values at the floats next to `x`, `x ± u` for the spacing `u` of the floats
/// there, moved from `f(x)` by at most a quarter of what they moved at the finest step of the
/// answer `a` (and the rounding of the values). That step is at least 16 floats wide, so a
/// function the answer resolves moves an eighth of it or less one float away; one that oscillates
/// on a scale between the spacing of the floats and that step, which can pass the ladder's checks
/// on samples many periods apart, moves by a good part of its amplitude.
fn at_the_floats<T: Real>(f: &mut impl FnMut(T) -> T, x: T, fx: T, a: &Answer<T>) -> bool {
    let Some((xp, xm, _, _, _)) = rung_points(x, T::EPSILON * x.abs()) else {
        // Next to 0 the floats are as dense as the type allows.
        return true;
    };
    let (fp, fm) = (f(xp), f(xm));
    if !(fp.is_finite() && fm.is_finite()) {
        return false;
    }
    let size = fp.abs().max(fm.abs()).max(fx.abs());
    let rounding = T::from_f64(4.0) * (T::EPSILON * size + T::MIN_POSITIVE * T::EPSILON);
    let bound = a.swing * T::from_f64(0.25) + rounding;
    (fp - fx).abs() <= bound && (fx - fm).abs() <= bound
}

/// `x + h` and `x − h` as floats near `x` hold them, and the half sum of the two steps that
/// realises: the step the quotient is taken over. `None` when either step is not positive, or
/// either point not finite.
fn rung_points<T: Real>(x: T, h: T) -> Option<(T, T, T, T, T)> {
    let xp = x + h;
    let hp = xp - x;
    let xm = x - hp;
    let hm = x - xm;
    if !(hp > T::ZERO && hm > T::ZERO) || !xp.is_finite() || !xm.is_finite() {
        return None;
    }
    Some((xp, xm, hp, hm, (hp + hm) * T::HALF))
}

/// `a/h²`, as `a/h/h` when `h²` is beyond the range of the type (steps from about `1e154` in
/// `f64`), where the quotient may still be within it.
fn over_square<T: Real>(a: T, h: T) -> T {
    let square = h * h;
    if square.is_finite() {
        a / square
    } else {
        a / h / h
    }
}

/// One ladder of steps from `h0` down: see the module's overview. `give_way` says that a second
/// ladder will start at the scale of `x`, so that values that are not finite above that scale
/// end this one at once; `may_be_flat` whether a function whose values do not change may be
/// given the derivative zero.
#[allow(clippy::too_many_arguments)]
fn ladder<T: Real>(
    f: &mut impl FnMut(T) -> T,
    x: T,
    fx: T,
    h0: T,
    order: Order,
    opts: &DiffOptions<T>,
    give_way: bool,
    may_be_flat: bool,
) -> Found<T> {
    let eps = T::EPSILON;
    let tol = opts.tol;
    let columns = opts.max_levels;
    let unit = x.abs().max(T::ONE);
    let smallest = T::from_f64(GRID) * eps * x.abs();
    // The derivative of a function that does not change at all is zero to the resolution
    // ε·|f|/h (ε·|f|/h² for the second); below this step that is coarser than the tolerance.
    let flat_floor = match order {
        Order::First => eps * unit / tol,
        Order::Second => unit * (T::from_f64(4.0) * eps / tol).sqrt(),
    };
    let mut rows: Vec<Rung<T>> = Vec::new();
    let mut previous: Vec<T> = Vec::new();
    let mut hopeless = 0;
    let mut kinks = 0;
    let mut last_jump: Option<T> = None;
    let mut all_flat = true;
    let mut best: Option<Answer<T>> = None;
    // The verified answer of smallest error so far: the ladder goes on below it while a finer
    // step could still do better, since an answer of steps many periods of a periodic function
    // wide can pass every test by chance, and the function is only resolved further down.
    let mut verified: Option<Answer<T>> = None;
    // Whether a rung below the verified answer has shown quotients far larger than it: the
    // function varies on a finer scale than that answer's steps resolved.
    let mut finer = false;
    // The largest step of the current run of rungs, whose resolution a zero derivative has.
    let mut top = h0;
    let mut h = h0 * T::TWO;
    for _ in 0..RUNGS {
        h *= T::HALF;
        let Some((xp, xm, hp, hm, step)) = rung_points(x, h) else {
            break;
        };
        if hp < smallest {
            break;
        }
        let (fp, fm) = (f(xp), f(xm));
        let rung = if fp.is_finite() && fm.is_finite() {
            make_rung(x, fx, xp, xm, hp, hm, step, fp, fm, order, rows.last())
        } else {
            None
        };
        let Some(rung) = rung else {
            // A wall: what settled above it may still stand; below it the table starts afresh.
            if let Some(b) = best.take()
                && better(&verified, &b)
                && verify(f, x, fx, &b, order, tol)
            {
                verified = Some(b);
                finer = false;
            }
            rows.clear();
            previous.clear();
            top = h * T::HALF;
            hopeless = 0;
            kinks = 0;
            last_jump = None;
            all_flat = true;
            if give_way && h > x.abs() {
                return settled(verified, finer);
            }
            continue;
        };
        rows.push(rung);
        all_flat = all_flat && rung.flat;
        let n = rows.len();
        // The Neville row of this step: column m cancels the powers h², …, h^(2m).
        let mut current = vec![rung.q];
        for m in 1..=previous.len().min(columns - 1) {
            let ratio = rows[n - 1 - m].h / rung.h;
            let refined =
                current[m - 1] + (current[m - 1] - previous[m - 1]) / (ratio * ratio - T::ONE);
            current.push(refined);
        }
        let mut candidate: Option<(T, T, T, T, usize)> = None;
        if !previous.is_empty() {
            // A round-off estimate of zero has underflowed (the rounding of the values is at
            // least ε times the smallest normal number): the step is so large beside the values
            // that every quotient is below the smallest normal number too, an underflow that the
            // relative tests would read as 0 ≤ tol·0. Such a rung gives no candidate.
            if rung.noise > T::ZERO
                && let Some(slope) = rung.slope
            {
                for m in 1..current.len() {
                    let mut estimate = (current[m] - current[m - 1])
                        .abs()
                        .max((current[m] - previous[m - 1]).abs());
                    if m < previous.len() {
                        estimate = estimate.max((current[m] - previous[m]).abs());
                    } else if m == 1 {
                        // Two quotients alone settle nothing.
                        continue;
                    }
                    if !estimate.is_finite() {
                        continue;
                    }
                    let scale = current[m].abs().max(slope);
                    let total = estimate + rung.noise;
                    if total <= tol * scale && candidate.is_none_or(|c| total < c.0) {
                        candidate = Some((total, current[m], scale, estimate, m));
                    }
                }
            }
            if may_be_flat
                && candidate.is_none()
                && all_flat
                && n >= 3
                && top >= flat_floor
                && rung.h < flat_floor
            {
                // The values have not changed at any step from one where their round-off is
                // below a derivative of the tolerance's size down to one where it would hide
                // such a derivative: the derivative is zero, to the resolution of the largest.
                let resolution = match order {
                    Order::First => T::TWO * eps * fx.abs() / top,
                    Order::Second => over_square(T::from_f64(8.0) * eps * fx.abs(), top),
                };
                let zero = Answer {
                    value: T::ZERO,
                    error: resolution,
                    scale: T::ZERO,
                    window: Vec::new(),
                    swing: T::ZERO,
                };
                return if flat_at(
                    f,
                    x,
                    fx,
                    rung.h * T::from_f64(core::f64::consts::SQRT_2),
                    order,
                ) {
                    Found::Answer(zero)
                } else {
                    Found::Nothing
                };
            }
        }
        previous = current;
        let mut good = false;
        if let Some((_, _, scale, _, _)) = candidate {
            let jumps: Vec<T> = rows[n.saturating_sub(columns)..]
                .iter()
                .filter_map(|r| r.jump)
                .collect();
            if jumps.len() >= 2 {
                let limit = jump_limit(&jumps);
                if limit.abs() <= T::from_f64(JUMP) * tol * scale {
                    good = true;
                } else {
                    // A corner: the one-sided slopes differ and the difference has settled.
                    match last_jump {
                        Some(j) if (limit - j).abs() <= T::from_f64(0.01) * limit.abs() => {
                            kinks += 1;
                            if kinks >= 4 {
                                return Found::Kink;
                            }
                        }
                        _ => kinks = 0,
                    }
                    last_jump = Some(limit);
                }
            }
        }
        let mut done = false;
        if good {
            let (total, value, scale, estimate, m) = candidate.expect("a good candidate");
            if best.as_ref().is_some_and(|b| total >= b.error) {
                // Past the best the table has.
                done = true;
            } else {
                best = Some(Answer {
                    value,
                    error: total,
                    scale,
                    window: rows[n - 1 - m..].iter().map(|r| (r.h, r.q)).collect(),
                    swing: (rung.fp - fx).abs().max((rung.fm - fx).abs()),
                });
                if estimate <= rung.noise {
                    // Only round-off is left, and it grows from here.
                    done = true;
                }
            }
        } else if best.is_some() {
            done = true;
        }
        if done {
            let b = best.take().expect("an answer to check");
            if better(&verified, &b) && verify(f, x, fx, &b, order, tol) {
                verified = Some(b);
                finer = false;
            }
            // The table starts again from this rung: an answer that failed was an alias or a
            // table that settled by chance, and one that passed may still be bettered below.
            rows.drain(..n - 1);
            previous = vec![rung.q];
            kinks = 0;
            last_jump = None;
        }
        let now = rung.q.abs().max(rung.slope.unwrap_or(rung.q.abs()));
        if let Some(v) = verified.as_ref() {
            if now > T::from_f64(GROW) * v.scale {
                // The function varies on a finer scale than the answer's steps resolved.
                finer = true;
            } else if rung.noise > v.error && !finer {
                // No finer step can do better: the round-off alone is larger than the answer's
                // error, and the quotients are of the answer's size.
                break;
            }
        }
        // The round-off alone is beyond the tolerance, and finer steps only add to it.
        if rung.noise > T::from_f64(STOP) * tol * now && !rung.flat {
            hopeless += 1;
            if hopeless >= 3 {
                break;
            }
        } else {
            hopeless = 0;
        }
    }
    if let Some(b) = best
        && better(&verified, &b)
        && verify(f, x, fx, &b, order, tol)
    {
        verified = Some(b);
        finer = false;
    }
    settled(verified, finer)
}

/// What a ladder found: its verified answer, unless the steps of that answer did not resolve the
/// function (`finer`), when what it says is not the function's slope.
fn settled<T: Real>(verified: Option<Answer<T>>, finer: bool) -> Found<T> {
    match verified {
        Some(v) if !finer => Found::Answer(v),
        _ => Found::Nothing,
    }
}

/// Whether `answer` has a smaller estimated error relative to its size than the verified one:
/// relative, since a periodic function sampled many periods wide gives quotients of the size of
/// the reciprocal step, small, whose errors are small too in absolute terms.
fn better<T: Real>(verified: &Option<Answer<T>>, answer: &Answer<T>) -> bool {
    verified
        .as_ref()
        .is_none_or(|v| answer.error / answer.scale < v.error / v.scale)
}

/// The rung of the step `step` from the samples `fp = f(xp)` and `fm = f(xm)`, `None` when a
/// quotient is not finite. The second derivative's one-sided second differences need the
/// samples of the rung above, at twice the step.
#[allow(clippy::too_many_arguments)]
fn make_rung<T: Real>(
    x: T,
    fx: T,
    xp: T,
    xm: T,
    hp: T,
    hm: T,
    step: T,
    fp: T,
    fm: T,
    order: Order,
    above: Option<&Rung<T>>,
) -> Option<Rung<T>> {
    let eps = T::EPSILON;
    let forward = (fp - fx) / hp;
    let backward = (fx - fm) / hm;
    let size = fp.abs().max(fm.abs()).max(fx.abs());
    // An absolute floor of the smallest subnormal's worth for values that are themselves tiny.
    let rounding = eps * size + T::MIN_POSITIVE * eps;
    let mut rung = match order {
        Order::First => Rung {
            h: step,
            xp,
            xm,
            fp,
            fm,
            q: (fp - fm) / (hp + hm),
            jump: Some(forward - backward),
            slope: Some(forward.abs().max(backward.abs())),
            noise: rounding / step,
            flat: (fp - fx).abs() <= T::TWO * rounding && (fm - fx).abs() <= T::TWO * rounding,
        },
        Order::Second => Rung {
            h: step,
            xp,
            xm,
            fp,
            fm,
            q: T::TWO * (forward - backward) / (hp + hm),
            jump: None,
            slope: None,
            noise: over_square(T::from_f64(4.0) * rounding, step),
            flat: ((fp - fx) - (fx - fm)).abs() <= T::from_f64(4.0) * rounding,
        },
    };
    if !(rung.q.is_finite() && forward.is_finite() && backward.is_finite()) {
        return None;
    }
    if order == Order::Second
        && let Some(up) = above
    {
        // The one-sided second differences on x, x + h, x + 2h and on x, x − h, x − 2h.
        let ahead = T::TWO * ((up.fp - fp) / (up.xp - xp) - (fp - fx) / (xp - x)) / (up.xp - x);
        let behind = T::TWO * ((fx - fm) / (x - xm) - (fm - up.fm) / (xm - up.xm)) / (x - up.xm);
        if ahead.is_finite() && behind.is_finite() {
            rung.jump = Some(ahead - behind);
            rung.slope = Some(ahead.abs().max(behind.abs()));
            let outer = up.fp.abs().max(up.fm.abs()).max(size);
            let r = T::from_f64(4.0) * (eps * outer + T::MIN_POSITIVE * eps);
            rung.flat = rung.flat
                && ((up.fp - fp) - (fp - fx)).abs() <= r
                && ((fx - fm) - (fm - up.fm)).abs() <= r;
        } else {
            rung.flat = false;
        }
    }
    Some(rung)
}

/// The limit as `h → 0` of the differences of the one-sided quotients, coarsest first, whose
/// expansion has the odd powers `h, h³, h⁵, …`: Richardson on those exponents, as the steps
/// halve.
fn jump_limit<T: Real>(jumps: &[T]) -> T {
    let mut values = jumps.to_vec();
    let mut power = 1;
    while values.len() > 1 {
        let factor = T::from_f64(2f64.powi(power) - 1.0);
        values = values
            .windows(2)
            .map(|w| w[1] + (w[1] - w[0]) / factor)
            .collect();
        power += 2;
    }
    values[0]
}

/// The quotients at steps off the ladder of halvings, between the finest of the answer's window
/// and twice it, against the values there of the polynomial in `h²` through the window.
fn verify<T: Real>(
    f: &mut impl FnMut(T) -> T,
    x: T,
    fx: T,
    answer: &Answer<T>,
    order: Order,
    tol: T,
) -> bool {
    let Some(&(finest, _)) = answer.window.last() else {
        return false;
    };
    let threshold = tol * answer.scale;
    for &ratio in RATIOS {
        let Some((xp, xm, hp, hm, step)) = rung_points(x, finest * T::from_f64(ratio)) else {
            return false;
        };
        let (fp, fm) = (f(xp), f(xm));
        let Some(rung) = make_rung(x, fx, xp, xm, hp, hm, step, fp, fm, order, None) else {
            return false;
        };
        // Neville's interpolation at step² through the window's (h², q), the steps measured in
        // units of `step`, since their squares leave the range of the type for steps near 1e±154.
        let t = T::ONE;
        let nodes: Vec<T> = answer
            .window
            .iter()
            .map(|&(h, _)| (h / step) * (h / step))
            .collect();
        let mut p: Vec<T> = answer.window.iter().map(|&(_, q)| q).collect();
        for level in 1..p.len() {
            for i in 0..p.len() - level {
                p[i] = ((t - nodes[i + level]) * p[i] + (nodes[i] - t) * p[i + 1])
                    / (nodes[i] - nodes[i + level]);
            }
        }
        // Written so that a deviation that is NaN does not agree.
        let agrees = (rung.q - p[0]).abs() <= threshold + rung.noise;
        if !agrees {
            return false;
        }
    }
    true
}

/// Whether the values at `x ± h`, a step off the ladder, are those at `x` too (for the second
/// derivative: whether they lie on a line through it).
fn flat_at<T: Real>(f: &mut impl FnMut(T) -> T, x: T, fx: T, h: T, order: Order) -> bool {
    let Some((xp, xm, hp, hm, step)) = rung_points(x, h) else {
        return false;
    };
    let (fp, fm) = (f(xp), f(xm));
    if !(fp.is_finite() && fm.is_finite()) {
        return false;
    }
    make_rung(x, fx, xp, xm, hp, hm, step, fp, fm, order, None).is_some_and(|r| r.flat)
}
