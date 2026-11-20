//! The zeros of a function of one real variable.
//!
//! Five methods, each generic over [`Real`]:
//!
//! | Method | Needs | Order | Guaranteed |
//! | --- | --- | --- | --- |
//! | [`bisect`] | a bracket | 1 | yes |
//! | [`brent`] | a bracket | 2 to 3 in practice | yes |
//! | [`secant`] | two starting points | 1.618 | no |
//! | [`newton`] | `f` and `f′` | 2 | no |
//! | [`halley`] | `f`, `f′` and `f″` | 3 | no |
//!
//! A bracketing method is told where the root is and cannot miss it; an open method is told where
//! to start and may wander off, so it returns `None` when it does not converge. Every method
//! returns `None` rather than an answer it has not verified, and takes the tolerances and the
//! iteration limit in a [`RootOptions`]:
//!
//! ```
//! use a::math::numerics::{RootOptions, brent, newton_with};
//!
//! // The golden ratio, by Brent on a bracket and by Newton on a starting point.
//! let phi = brent(1.0, 2.0, |x: f64| x * x - x - 1.0).unwrap();
//! assert!((phi - 1.618_033_988_749_895).abs() < 1e-15);
//!
//! let opts = RootOptions::new().with_max_iter(20);
//! let same = newton_with(1.0, |x| x * x - x - 1.0, |x| 2.0 * x - 1.0, &opts).unwrap();
//! assert!((same - phi).abs() < 1e-15);
//! ```

use super::Real;

/// When a root-finding method stops, and when it gives up.
///
/// A method returns its current estimate once the estimate has stopped moving —
/// `|x − x′| ≤ xtol + rtol·|x|` — or once the function is small enough, `|f(x)| ≤ ftol`. It gives
/// up after `max_iter` function evaluations.
///
/// The defaults suit `f64`: `xtol` 0, `rtol` 4ε (`8.88e−16`), `ftol` 0 and `max_iter` 100. For
/// `f32` pass `rtol` of about 4·`f32::EPSILON`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootOptions<T> {
    /// The absolute part of the tolerance on the root.
    pub xtol: T,
    /// The relative part of the tolerance on the root.
    pub rtol: T,
    /// The tolerance on the function value; 0 asks for none.
    pub ftol: T,
    /// The largest number of iterations.
    pub max_iter: usize,
}

impl<T: Real> Default for RootOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> RootOptions<T> {
    /// The defaults: `xtol` 0, `rtol` 4ε, `ftol` 0, `max_iter` 100.
    pub fn new() -> Self {
        RootOptions {
            xtol: T::ZERO,
            rtol: T::EPSILON * T::from_f64(4.0),
            ftol: T::ZERO,
            max_iter: super::DEFAULT_MAX_ITER,
        }
    }

    /// Sets the absolute tolerance on the root.
    #[inline]
    pub fn with_xtol(mut self, xtol: T) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the relative tolerance on the root.
    #[inline]
    pub fn with_rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    /// Sets the tolerance on the function value.
    #[inline]
    pub fn with_ftol(mut self, ftol: T) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets the largest number of iterations.
    #[inline]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// The tolerance on the root at `x`: `xtol + rtol·|x|`.
    #[inline]
    pub fn tol_at(&self, x: T) -> T {
        self.xtol + self.rtol * x.abs()
    }

    /// Whether the value at the current estimate is small enough to accept it. With the default
    /// `ftol` of zero this is never true, and only the movement of the estimate accepts.
    #[inline]
    fn residual(&self, fx: T) -> bool {
        self.ftol > T::ZERO && fx.abs() <= self.ftol
    }

    /// Whether the current estimate is accepted: it is finite, and either the function is small
    /// enough or the estimate has moved no further than its tolerance. `moved` is the distance the
    /// last step travelled, so a method that has not stepped yet passes nothing and is not
    /// accepted by doing nothing.
    #[inline]
    fn accepted(&self, x: T, fx: T, moved: T) -> bool {
        x.is_finite() && (self.residual(fx) || moved <= self.tol_at(x))
    }
}

/// The midpoint of `a` and `b`, which never overflows.
#[inline]
fn midpoint<T: Real>(a: T, b: T) -> T {
    // (a + b)/2 is exact when neither argument is large; a/2 + b/2 cannot overflow.
    if a.abs() <= T::ONE && b.abs() <= T::ONE {
        (a + b) * T::HALF
    } else {
        a * T::HALF + b * T::HALF
    }
}

/// The bracket `a < x < b`, or `b < x < a`.
#[inline]
fn between<T: Real>(x: T, a: T, b: T) -> bool {
    (a < x && x < b) || (b < x && x < a)
}

/// The root of `f` in the bracket `[a, b]`, by bisection.
///
/// `f(a)` and `f(b)` must be finite and of opposite signs, or one of them zero; otherwise the
/// method has no bracket to work on and returns `None`. Each iteration halves the bracket and
/// evaluates `f` once, so `⌈log₂((b − a)/tol)⌉` evaluations suffice: the method always converges,
/// and is the slowest of the five.
///
/// # Examples
///
/// ```
/// use a::math::numerics::bisect;
///
/// // The root of cos x − x, which is the Dottie number.
/// let x = bisect(0.0, 1.0, |x: f64| x.cos() - x).unwrap();
/// assert!((x - 0.739_085_133_215_160_7).abs() < 1e-15);
///
/// // No bracket, so no answer.
/// assert!(bisect(0.0, 1.0, |x: f64| x * x + 1.0).is_none());
/// ```
pub fn bisect<T: Real>(a: T, b: T, f: impl Fn(T) -> T) -> Option<T> {
    bisect_with(a, b, f, &RootOptions::new())
}

/// The root of `f` in `[a, b]` by bisection, with the tolerances and the limit of `opts`.
pub fn bisect_with<T: Real>(
    a: T,
    b: T,
    mut f: impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> Option<T> {
    let (mut lo, mut hi) = (a, b);
    let mut flo = f(lo);
    let fhi = f(hi);
    if flo.is_nan() || fhi.is_nan() {
        return None;
    }
    if flo == T::ZERO {
        return Some(lo);
    }
    if fhi == T::ZERO {
        return Some(hi);
    }
    if !flo.is_finite() || !fhi.is_finite() || flo.signum() == fhi.signum() {
        return None;
    }
    for _ in 0..opts.max_iter {
        let m = midpoint(lo, hi);
        if m == lo || m == hi {
            return Some(m); // no float left between the ends
        }
        let fm = f(m);
        if fm.is_nan() {
            return None;
        }
        if fm == T::ZERO || (hi - lo) * T::HALF <= opts.tol_at(m) || opts.residual(fm) {
            return Some(m);
        }
        if fm.signum() == flo.signum() {
            lo = m;
            flo = fm;
        } else {
            hi = m;
        }
    }
    let m = midpoint(lo, hi);
    if (hi - lo) * T::HALF <= opts.tol_at(m) {
        Some(m)
    } else {
        None
    }
}

/// The root of `f` in the bracket `[a, b]`, by Brent's method.
///
/// Brent's inverse quadratic interpolation with the bisection safeguard: as fast as secant or
/// Newton where the function is smooth, and as safe as bisection where it is not, since the
/// bracket is never lost. Each iteration evaluates `f` once. `f(a)` and `f(b)` must be finite and
/// of opposite signs, or one of them zero.
///
/// This is the method to reach for by default: it needs no derivative and rarely takes more than
/// a few more evaluations than Newton.
///
/// # Examples
///
/// ```
/// use a::math::numerics::brent;
///
/// // The real root of x³ − x − 2.
/// let x = brent(1.0, 2.0, |x: f64| x * x * x - x - 2.0).unwrap();
/// assert!((x - 1.521_379_706_804_567_6).abs() < 1e-15);
///
/// // A bracket with a pole inside it gives no answer rather than the pole.
/// assert!(brent(-1.0, 1.0, |x: f64| 1.0 / x).is_none());
/// ```
pub fn brent<T: Real>(a: T, b: T, f: impl Fn(T) -> T) -> Option<T> {
    brent_with(a, b, f, &RootOptions::new())
}

/// The root of `f` in `[a, b]` by Brent's method, with the tolerances and the limit of `opts`.
pub fn brent_with<T: Real>(
    a: T,
    b: T,
    mut f: impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> Option<T> {
    let (mut x, mut y) = (a, b);
    let (mut fx, mut fy) = (f(a), f(b));
    if fx.is_nan() || fy.is_nan() {
        return None;
    }
    if fx == T::ZERO {
        return Some(x);
    }
    if fy == T::ZERO {
        return Some(y);
    }
    if !fx.is_finite() || !fy.is_finite() || fx.signum() == fy.signum() {
        return None;
    }
    if fx.abs() < fy.abs() {
        std::mem::swap(&mut x, &mut y);
        std::mem::swap(&mut fx, &mut fy);
    }
    // `x` and `y` are the ends of the bracket, |f(x)| ≥ |f(y)|, so `y` is the estimate. `z` is the
    // estimate before the last one, the third point of the quadratic, and `last` the distance that
    // estimate travelled — the step whose half bounds the next one.
    let (mut z, mut fz) = (x, fx);
    let mut last = (x - y).abs();
    for _ in 0..opts.max_iter {
        let tol = opts.tol_at(y);
        if fy == T::ZERO || (x - y).abs() <= tol || opts.residual(fy) {
            return Some(y);
        }
        let mut s = if z != y && fz != fx && fz != fy {
            // Inverse quadratic interpolation through the three points.
            let u = (x * fy * fz) / ((fx - fy) * (fx - fz));
            let v = (y * fx * fz) / ((fy - fx) * (fy - fz));
            let w = (z * fx * fy) / ((fz - fx) * (fz - fy));
            u + v + w
        } else {
            // The secant step, which is what the two points determine.
            (y * fx - x * fy) / (fx - fy)
        };
        // The two safeguards that make the method converge where interpolation alone does not: the
        // step must land between the estimate and the middle of the bracket, so that the bracket
        // left behind is at most half of this one whichever way the root lies; and it must be no
        // longer than half the step that produced the estimate, so that a step that barely moves
        // cannot be followed by another that barely moves. Otherwise, bisect.
        let middle = midpoint(x, y);
        if !between(s, middle, y) || (s - y).abs() >= last * T::HALF {
            s = middle;
        }
        let fs = f(s);
        if fs.is_nan() {
            return None;
        }
        let was = y;
        z = y;
        fz = fy;
        if fx.signum() == fs.signum() {
            x = s;
            fx = fs;
        } else {
            y = s;
            fy = fs;
        }
        if fx.abs() < fy.abs() {
            std::mem::swap(&mut x, &mut y);
            std::mem::swap(&mut fx, &mut fy);
        }
        // Only a step that moved the estimate counts as a step: one that landed on the far side of
        // the root left the estimate where it was, and its bound is the one still to be respected.
        if y != was {
            last = (y - was).abs();
        }
    }
    let tol = opts.tol_at(y);
    if (x - y).abs() <= tol || opts.residual(fy) {
        Some(y)
    } else {
        None
    }
}

/// The root of `f` by the secant method, from the two starting points `x0` and `x1`.
///
/// The next estimate is where the chord through `(x0, f(x0))` and `(x1, f(x1))` meets the axis, so
/// one evaluation of `f` an iteration buys the order `(1 + √5)/2 ≈ 1.618`. Nothing keeps the
/// iterates near the root: the method returns `None` when it does not converge within `max_iter`
/// iterations, when a step is not finite, or when two function values coincide.
///
/// # Examples
///
/// ```
/// use a::math::numerics::secant;
///
/// // The root of x − tan x nearest 4.5 (the second positive one of tan x = x). The starting
/// // points straddle it, since the method has nothing to keep it near the root.
/// let x = secant(4.4, 4.6, |x: f64| x - x.tan()).unwrap();
/// assert!((x - 4.493_409_457_909_064).abs() < 1e-13);
/// ```
pub fn secant<T: Real>(x0: T, x1: T, f: impl Fn(T) -> T) -> Option<T> {
    secant_with(x0, x1, f, &RootOptions::new())
}

/// The root of `f` by the secant method, with the tolerances and the limit of `opts`.
pub fn secant_with<T: Real>(
    x0: T,
    x1: T,
    mut f: impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> Option<T> {
    let (mut x, mut y) = (x0, x1);
    let (mut fx, mut fy) = (f(x), f(y));
    if fx.is_nan() || fy.is_nan() {
        return None;
    }
    if fx == T::ZERO {
        return Some(x);
    }
    if fy == T::ZERO {
        return Some(y);
    }
    for _ in 0..opts.max_iter {
        if fy - fx == T::ZERO {
            return None;
        }
        let s = (x * fy - y * fx) / (fy - fx);
        if !s.is_finite() {
            return None;
        }
        let fs = f(s);
        if fs.is_nan() {
            return None;
        }
        if opts.accepted(s, fs, (s - y).abs()) {
            return Some(s);
        }
        x = y;
        fx = fy;
        y = s;
        fy = fs;
    }
    None
}

/// The root of `f` by Newton's method, from the starting point `x0`, with `f` and its derivative
/// `df`.
///
/// The next estimate is where the tangent at the current one meets the axis: `x − f(x)/f′(x)`.
/// Where the derivative is nonzero and the start close enough, the error squares each iteration,
/// so the digits double; `max_iter` of 20 or so suffices from a rough start. Returns `None` when
/// the derivative vanishes, a step is not finite, or the iteration does not converge.
///
/// # Examples
///
/// ```
/// use a::math::numerics::newton;
///
/// // The root of e^x − 3x² nearest 1, which Newton reaches in four steps from 1.
/// let x = newton(1.0, |x: f64| x.exp() - 3.0 * x * x, |x: f64| x.exp() - 6.0 * x).unwrap();
/// assert!((x - 0.910_007_572_488_709).abs() < 1e-15);
///
/// // A stationary point, so no step and no answer.
/// assert!(newton(0.0, |x: f64| x * x, |x: f64| 2.0 * x).is_none());
/// ```
pub fn newton<T: Real>(x0: T, f: impl Fn(T) -> T, df: impl Fn(T) -> T) -> Option<T> {
    newton_with(x0, f, df, &RootOptions::new())
}

/// The root of `f` by Newton's method, with the tolerances and the limit of `opts`.
pub fn newton_with<T: Real>(
    x0: T,
    mut f: impl FnMut(T) -> T,
    mut df: impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> Option<T> {
    let mut x = x0;
    let mut fx = f(x);
    if fx.is_nan() {
        return None;
    }
    if opts.residual(fx) {
        return Some(x);
    }
    for _ in 0..opts.max_iter {
        let dfx = df(x);
        if !dfx.is_finite() || dfx == T::ZERO {
            return None;
        }
        let s = x - fx / dfx;
        if !s.is_finite() {
            return None;
        }
        let fs = f(s);
        if fs.is_nan() {
            return None;
        }
        if opts.accepted(s, fs, (s - x).abs()) {
            return Some(s);
        }
        x = s;
        fx = fs;
    }
    None
}

/// The root of `f` by Halley's method, from `x0`, with `f`, `df` and the second derivative `ddf`.
///
/// The next estimate is `x − 2f·f′ / (2f′² − f·f″)`, the second-order term of Newton's method:
/// the error cubes each iteration, so the digits triple. Two evaluations of `f`-family an
/// iteration, and the same caveats as [`newton`].
///
/// # Examples
///
/// ```
/// use a::math::numerics::halley;
///
/// // The root of cos x = x again, in three steps.
/// let x = halley(0.5, |x: f64| x.cos() - x, |x: f64| -x.sin() - 1.0, |x: f64| -x.cos()).unwrap();
/// assert!((x - 0.739_085_133_215_160_7).abs() < 1e-15);
/// ```
pub fn halley<T: Real>(
    x0: T,
    f: impl Fn(T) -> T,
    df: impl Fn(T) -> T,
    ddf: impl Fn(T) -> T,
) -> Option<T> {
    halley_with(x0, f, df, ddf, &RootOptions::new())
}

/// The root of `f` by Halley's method, with the tolerances and the limit of `opts`.
pub fn halley_with<T: Real>(
    x0: T,
    mut f: impl FnMut(T) -> T,
    mut df: impl FnMut(T) -> T,
    mut ddf: impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> Option<T> {
    let mut x = x0;
    let mut fx = f(x);
    if fx.is_nan() {
        return None;
    }
    if opts.residual(fx) {
        return Some(x);
    }
    for _ in 0..opts.max_iter {
        let dfx = df(x);
        let ddfx = ddf(x);
        if !dfx.is_finite() || !ddfx.is_finite() {
            return None;
        }
        let den = T::TWO * dfx * dfx - fx * ddfx;
        if den == T::ZERO || !den.is_finite() {
            return None;
        }
        let s = x - T::TWO * fx * dfx / den;
        if !s.is_finite() {
            return None;
        }
        let fs = f(s);
        if fs.is_nan() {
            return None;
        }
        if opts.accepted(s, fs, (s - x).abs()) {
            return Some(s);
        }
        x = s;
        fx = fs;
    }
    None
}
