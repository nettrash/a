//! Numerical differentiation, by Richardson extrapolation on symmetric differences.
//!
//! A difference quotient of step `h` is a polynomial in `h²` plus a remainder, so several
//! quotients with `h` halving — `h`, `h/2`, `h/4`, … — can be combined to cancel the even powers
//! of `h` in turn. The combination is Neville's, and each column of it squares the order of the
//! accuracy until round-off in the values of `f` bounds it: for `f64` about fourteen digits, from
//! fewer evaluations than a one-sided quotient of the same accuracy needs.
//!
//! Both derivatives are symmetric, which is why they tolerate a function that is only defined on
//! one side of a point being differentiated at the centre: the derivative of `|x|` at `0` is the
//! symmetric one, `0`, and not an error. A method that cannot bring two successive columns within
//! the tolerance returns `None`, as everywhere else here.

use super::{Real, close};

/// The options of the differentiation rules: the first step, the tolerance two columns must come
/// within, and how many columns are run.
#[derive(Debug, Clone, Copy)]
pub struct DiffOptions<T> {
    /// The first step of the difference quotients, or `T::ZERO` to choose one from the point.
    pub h: T,
    /// The relative tolerance two successive columns must satisfy.
    pub tol: T,
    /// The largest number of columns.
    pub max_levels: usize,
}

impl<T: Real> Default for DiffOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> DiffOptions<T> {
    /// The default options: the step chosen automatically, `tol` `ε^(2/3)` — about `3.6e−11` for
    /// `f64` and `2.4e−5` for `f32`, which is as tight as the round-off in the values of `f`
    /// allows — and eight columns.
    pub fn new() -> Self {
        Self {
            h: T::ZERO,
            tol: T::EPSILON.powf(T::from_f64(2.0 / 3.0)),
            max_levels: 8,
        }
    }

    /// Sets the first step. Pass `T::ZERO` to have it chosen from the point.
    pub fn with_step(mut self, h: T) -> Self {
        self.h = h;
        self
    }

    /// Sets the tolerance two successive columns must satisfy.
    pub fn with_tol(mut self, tol: T) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the largest number of columns.
    pub fn with_max_levels(mut self, max_levels: usize) -> Self {
        self.max_levels = max_levels;
        self
    }

    /// The first step: [`h`](Self::h) when it is one, and otherwise the coarsest the quotients of
    /// `q` admit, found by [`asymptotic_step`] from `f` itself.
    fn step(&self, x: T, q: &mut impl FnMut(T) -> T) -> Option<T> {
        if self.h > T::ZERO && self.h.is_finite() {
            Some(self.h)
        } else {
            asymptotic_step(q, x.abs().max(T::ONE))
        }
    }
}

/// How many times the step is halved while the quotients are watched for the asymptotic regime:
/// enough to reach a function varying on a scale of `1e−14` around the origin.
const SCAN: usize = 60;

/// How many successive gaps must quarter before the step is believed to be asymptotic.
const GAPS: usize = 5;

/// How many binary digits the numerator of a difference quotient must keep to be worth dividing.
const SIGNIFICANT: f64 = 256.0;

/// Whether the numerator of a difference quotient still carries information. At a step so fine
/// that the values of `f` differ by a rounding or two the numerator is the round-off and not the
/// derivative, and the table settles on it as readily as on anything else; such a quotient is
/// refused. A numerator that has cancelled to nought exactly is kept: the function is flat at
/// this resolution, whose symmetric derivative is `0`.
fn significant<T: Real>(num: T, scale: T) -> bool {
    num == T::ZERO || num.abs() >= T::from_f64(SIGNIFICANT) * T::EPSILON * scale
}

/// The coarsest step at whose quotients the error is a power of `h²`, which is where — and only
/// where — Richardson's extrapolation is valid. Three successive gaps `|q(h) − q(h/2)|` are
/// watched: within the regime each is about a quarter of the one before, since halving the step
/// cancels two powers at a time, so the ratios of the gaps bracket four — and [`GAPS`] of them
/// must agree, since a quotient that oscillates passes a ratio or two by luck. Above the regime the
/// error is not a power of `h²` at all — the quotient of `sin(1000x)` at `0` still oscillates long
/// after the step is a thousandth of a radian — and below it round-off in the values of `f` has
/// taken over, whose gaps grow as the step shrinks. Quotients that have stopped moving at all are
/// admitted too, the function being then a polynomial of the degree the quotient cannot see past.
///
/// `None` when no such step is found: a function oscillating on a scale finer than any step
/// resolves, or one whose quotients are never finite around the point.
fn asymptotic_step<T: Real>(q: &mut impl FnMut(T) -> T, h_max: T) -> Option<T> {
    // The widest and narrowest ratio of one gap to the next that still says `h²`: four, within a
    // factor of 1.5, so that the round-off in the subtractions does not reject a valid step.
    const LOW: f64 = 4.0 / 1.5;
    const HIGH: f64 = 4.0 * 1.5;

    let mut window = [T::NAN; GAPS + 1];
    for seen in 0..SCAN {
        let h = h_max / T::from_f64(2f64.powi(seen as i32));
        window.copy_within(1.., 0);
        window[GAPS] = q(h);
        if seen < GAPS {
            continue;
        }
        let mut gaps = [T::NAN; GAPS];
        for (k, gap) in gaps.iter_mut().enumerate() {
            *gap = (window[k + 1] - window[k]).abs();
        }
        if !gaps.iter().all(|g| g.is_finite()) {
            continue;
        }
        if gaps
            .iter()
            .all(|g| *g <= T::EPSILON * window[GAPS].abs().max(T::ONE))
        {
            return Some(h * T::from_f64(2.0).powi(GAPS as i32));
        }
        let quartering = |big: T, small: T| {
            small > T::ZERO && big <= small * T::from_f64(HIGH) && small * T::from_f64(LOW) <= big
        };
        if gaps.windows(2).all(|pair| quartering(pair[0], pair[1])) {
            return Some(h * T::from_f64(2.0).powi(GAPS as i32));
        }
    }
    None
}

/// Richardson's table of the quotients `q(h)`, `q(h/2)`, `q(h/4)`, … of one derivative, whose
/// leading error is a power of `h²`. `q` is called as `q(h)` and returns the quotient of that
/// step; the estimate is accepted when two successive columns come within the tolerance, and
/// `None` is returned when they never do.
fn richardson<T: Real>(
    mut q: impl FnMut(T) -> Option<T>,
    opts: &DiffOptions<T>,
    h0: T,
) -> Option<T> {
    let max = opts.max_levels;
    // The table `d[level][column]`, whose first column holds the quotients themselves.
    let mut table: Vec<Vec<T>> = Vec::with_capacity(max);
    let mut previous = T::ZERO;
    for level in 0..max {
        let h = h0 / T::from_f64(2f64.powi(level as i32));
        let mut row = Vec::with_capacity(level + 1);
        row.push(q(h)?);
        for column in 1..=level {
            // Neville on `h²`: the factor is `4^column − 1` because each column cancels one more
            // even power of the step.
            let factor = T::from_f64(4f64.powi(column as i32) - 1.0);
            let refined =
                row[column - 1] + (row[column - 1] - table[level - 1][column - 1]) / factor;
            row.push(refined);
        }
        let estimate = row[level];
        let settled = level > 0 && estimate.is_finite() && close(estimate, previous, opts.tol);
        previous = estimate;
        table.push(row);
        if settled {
            return Some(estimate);
        }
    }
    None
}

/// The derivative of `f` at `x`, by Richardson extrapolation on symmetric quotients.
///
/// # Examples
///
/// ```
/// use a::math::numerics::derivative;
///
/// // The slope of sin at 1 is cos 1.
/// let d = derivative(|x: f64| x.sin(), 1.0).unwrap();
/// assert!((d - 1.0f64.cos()).abs() < 1e-13);
///
/// // A quotient of a function with no slope there is the symmetric one.
/// assert_eq!(derivative(|x: f64| x.abs(), 0.0), Some(0.0));
/// ```
pub fn derivative<T: Real>(f: impl FnMut(T) -> T, x: T) -> Option<T> {
    derivative_with(f, x, &DiffOptions::new())
}

/// [`derivative`] with the options `opts`.
pub fn derivative_with<T: Real>(
    mut f: impl FnMut(T) -> T,
    x: T,
    opts: &DiffOptions<T>,
) -> Option<T> {
    let mut quotient = |h: T| {
        let (up, down) = (f(x + h), f(x - h));
        let num = up - down;
        if !significant(num, up.abs().max(down.abs())) {
            return T::NAN;
        }
        num / (T::TWO * h)
    };
    let h0 = opts.step(x, &mut quotient)?;
    richardson(
        |h| {
            let q = quotient(h);
            q.is_finite().then_some(q)
        },
        opts,
        h0,
    )
}

/// The second derivative of `f` at `x`, by Richardson extrapolation on symmetric quotients.
///
/// # Examples
///
/// ```
/// use a::math::numerics::second_derivative;
///
/// // The curvature of sin at 1 is −sin 1.
/// let d = second_derivative(|x: f64| x.sin(), 1.0).unwrap();
/// assert!((d + 1.0f64.sin()).abs() < 1e-11);
/// ```
pub fn second_derivative<T: Real>(f: impl FnMut(T) -> T, x: T) -> Option<T> {
    second_derivative_with(f, x, &DiffOptions::new())
}

/// [`second_derivative`] with the options `opts`.
pub fn second_derivative_with<T: Real>(
    mut f: impl FnMut(T) -> T,
    x: T,
    opts: &DiffOptions<T>,
) -> Option<T> {
    // The centre value is the same for every step, so it is evaluated once.
    let centre = f(x);
    let mut quotient = |h: T| {
        let (up, down) = (f(x + h), f(x - h));
        let num = up - centre - centre + down;
        let scale = up.abs().max(centre.abs().max(down.abs()));
        if !significant(num, scale) {
            return T::NAN;
        }
        num / (h * h)
    };
    let h0 = opts.step(x, &mut quotient)?;
    richardson(
        |h| {
            let q = quotient(h);
            q.is_finite().then_some(q)
        },
        opts,
        h0,
    )
}
