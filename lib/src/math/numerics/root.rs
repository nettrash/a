//! The zeros of a function of one real variable.
//!
//! Five methods, each generic over [`Real`]:
//!
//! | Method | Needs | Convergence | Guaranteed |
//! | --- | --- | --- | --- |
//! | [`bisect`] | a bracket | linear, a bit an evaluation | yes |
//! | [`brent`] | a bracket | superlinear (order about 1.84) where `f` is smooth | yes |
//! | [`secant`] | two starting points | order `(1 + √5)/2 ≈ 1.618` | no |
//! | [`newton`] | `f` and `f′` | order 2 | no |
//! | [`halley`] | `f`, `f′` and `f″` | order 3 | no |
//!
//! A bracketing method is given an interval on whose ends `f` has opposite signs and keeps such
//! an interval at every step, so it cannot lose the root; an open method is given only where to
//! start, and may wander off. None of them returns an estimate it has not verified by the test of
//! [`RootOptions`]:
//!
//! * an estimate `x` at which `f(x)` is exactly zero is returned whatever the tolerances;
//! * otherwise, with the default `ftol` of zero, `x` is returned when `f` changes sign within
//!   `tol(x) = xtol + rtol·|x|` of it (or within one float of it, where that is finer than the
//!   floats): a bracketing method has closed its bracket to that width, and an open method has
//!   taken a last step no longer than that and found the sign change, against the estimate before
//!   or by evaluating `f` at `x ± tol(x)`, which costs it at most two more evaluations;
//! * with `ftol > 0` the value decides instead: `x` is returned as soon as `|f(x)| ≤ ftol`, and
//!   never where `|f(x)| > ftol`.
//!
//! A sign change is what a method can verify, and it cannot tell a root from a pole or a jump:
//! `1/(x − 0.3)` changes sign at 0.3 too, and with `ftol` zero that is what comes back. Give
//! `ftol` where that matters: `|f|` grows without bound at a pole, so the value test refuses it.
//! A root of even multiplicity, where `f` touches zero without crossing it, has no sign change to
//! find, and is returned only where `f` is exactly zero or, with `ftol`, small enough.
//!
//! Every method returns `None` when there is nothing to verify: no bracket, a NaN, a tolerance
//! that is negative or not finite, a vanishing derivative, a step that is not finite, an
//! iteration that stops moving away from a root, or `max_iter` iterations without an answer.
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

/// The default limit on the iterations: more than [`bisect`] can need on any bracket of finite
/// floats (220 iterations of an `f64` one, 110 of an `f32` one), more than [`brent`] can need
/// where its tolerance is not reachable by halving, as with the default `xtol` of zero (385 and
/// 193, by its safeguards; the most measured is 206), and enough for an open method to come in
/// from far away, as Halley's method from `10¹¹⁰` to `√2`, which takes 234 steps. With a positive
/// `xtol` that halving reaches, [`brent`] is Brent's method unchanged, which his own bound allows
/// more steps than this in theory; the most a search for the worst case found is 249.
const MAX_ITER: usize = 512;

/// When a root-finding method accepts an estimate, and when it gives up.
///
/// The tolerance on the root is `tol(x) = xtol + rtol·|x|`. With `ftol` zero, an estimate is
/// accepted when `f` changes sign within `tol(x)` of it; with `ftol` positive, when `|f(x)| ≤
/// ftol`, and as soon as that holds; and in either case when `f(x)` is exactly zero. A method gives
/// up after `max_iter` iterations, each one evaluation of `f` (and of the derivatives an open method
/// takes), besides the two at the ends of a bracket, and the one or two an open method spends each
/// time it looks for the sign change around a short step.
///
/// The defaults are `xtol` 0, `rtol` 4ε (`8.9·10⁻¹⁶` for `f64`, `4.8·10⁻⁷` for `f32`), `ftol` 0
/// and `max_iter` 512. A purely relative tolerance cannot be met at a root at zero, where the
/// bracketing methods stop instead at the two neighbouring floats around the sign change (or on
/// zero itself); an open method converging there only linearly, at a multiple root, needs an
/// `ftol` to stop — or, at a root of odd multiplicity, where `f` changes sign, an `xtol`; one of
/// even multiplicity, as `x²`, has no sign change for an `xtol` to find. A tolerance that is
/// negative or not finite makes every method return `None`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{RootOptions, bisect_with};
///
/// let opts = RootOptions::new().with_xtol(1e-3);
/// assert_eq!(opts.tol_at(2.0), 1e-3 + 2.0 * 4.0 * f64::EPSILON);
/// // Ten halvings of [1, 2] bring it within 10⁻³ of √2.
/// let x = bisect_with(1.0, 2.0, |x: f64| x * x - 2.0, &opts).unwrap();
/// assert!((x - core::f64::consts::SQRT_2).abs() <= 1e-3);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootOptions<T> {
    /// The absolute part of the tolerance on the root.
    pub xtol: T,
    /// The relative part of the tolerance on the root.
    pub rtol: T,
    /// The tolerance on the value: zero to accept on a sign change, positive to accept only where
    /// `|f(x)| ≤ ftol`.
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
    /// The defaults: `xtol` 0, `rtol` 4ε, `ftol` 0, `max_iter` 512.
    #[must_use]
    pub fn new() -> Self {
        RootOptions {
            xtol: T::ZERO,
            rtol: T::EPSILON * T::from_f64(4.0),
            ftol: T::ZERO,
            max_iter: MAX_ITER,
        }
    }

    /// Sets the absolute tolerance on the root.
    #[inline]
    #[must_use]
    pub fn with_xtol(mut self, xtol: T) -> Self {
        self.xtol = xtol;
        self
    }

    /// Sets the relative tolerance on the root.
    #[inline]
    #[must_use]
    pub fn with_rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    /// Sets the tolerance on the value.
    #[inline]
    #[must_use]
    pub fn with_ftol(mut self, ftol: T) -> Self {
        self.ftol = ftol;
        self
    }

    /// Sets the largest number of iterations.
    #[inline]
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// The tolerance on the root at `x`: `xtol + rtol·|x|`.
    #[inline]
    #[must_use]
    pub fn tol_at(&self, x: T) -> T {
        self.xtol + self.rtol * x.abs()
    }

    /// Whether every tolerance is a number that is neither negative nor infinite.
    fn valid(&self) -> bool {
        let ok = |t: T| t >= T::ZERO && t.is_finite();
        ok(self.xtol) && ok(self.rtol) && ok(self.ftol)
    }

    /// Whether a value is small enough to accept its estimate at once: only with an `ftol`.
    #[inline]
    fn small(&self, fx: T) -> bool {
        self.ftol > T::ZERO && fx.abs() <= self.ftol
    }

    /// Whether an estimate whose location has been verified may be returned with the value `fx`:
    /// always without an `ftol`, and only where `|f| ≤ ftol` with one.
    #[inline]
    fn value_ok(&self, fx: T) -> bool {
        self.ftol == T::ZERO || fx.abs() <= self.ftol
    }

    /// How close to `x` an open method looks for its sign change: the tolerance, or one float
    /// where the tolerance is finer than the floats.
    #[inline]
    fn reach(&self, x: T) -> T {
        self.tol_at(x).max(spacing(x))
    }
}

/// At least the distance from `x` to either float next to it, and less than twice it: `ε·|x|`,
/// or the smallest subnormal where that is below it, so that `x ± spacing(x)` is never `x`.
#[inline]
fn spacing<T: Real>(x: T) -> T {
    (x.abs() * T::EPSILON).max(T::MIN_POSITIVE * T::EPSILON)
}

/// Whether `v` is below zero; the methods only ask it of numbers that are neither zero nor NaN.
#[inline]
fn negative<T: Real>(v: T) -> bool {
    v < T::ZERO
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

/// The number of bits of the type, 32 or 64: halving the count of the floats in a bracket that
/// many times leaves two neighbours, whatever the bracket.
#[inline]
fn bits<T: Real>() -> i32 {
    if T::MANTISSA_DIGITS == f32::MANTISSA_DIGITS {
        32
    } else {
        64
    }
}

/// The place of `x` in the ordered set of the floats of its type: an integer that is 0 for both
/// zeros, increases with `x`, and differs by one between neighbours, so that halving a difference
/// of two of them halves the count of the floats between their floats.
fn ordinal<T: Real>(x: T) -> i64 {
    if T::MANTISSA_DIGITS == f32::MANTISSA_DIGITS {
        // Exact both ways: the value is an f32 to begin with.
        let bits = (x.to_f64() as f32).to_bits();
        let magnitude = i64::from(bits & 0x7fff_ffff);
        if bits >> 31 == 1 {
            -magnitude
        } else {
            magnitude
        }
    } else {
        let bits = x.to_f64().to_bits();
        let magnitude = (bits & 0x7fff_ffff_ffff_ffff) as i64;
        if bits >> 63 == 1 {
            -magnitude
        } else {
            magnitude
        }
    }
}

/// The float of the type at the place `k`, the inverse of [`ordinal`].
fn from_ordinal<T: Real>(k: i64) -> T {
    let magnitude = k.unsigned_abs();
    let x = if T::MANTISSA_DIGITS == f32::MANTISSA_DIGITS {
        f64::from(f32::from_bits(magnitude as u32))
    } else {
        f64::from_bits(magnitude)
    };
    let x = T::from_f64(x);
    if k < 0 { -x } else { x }
}

/// The number of floats from one end of a bracket to the other, as a float, for comparing
/// brackets.
#[inline]
fn count<T: Real>(x: T, y: T) -> f64 {
    (i128::from(ordinal(y)) - i128::from(ordinal(x))).unsigned_abs() as f64
}

/// The float halfway from `lo` to `hi` in the ordered set of the floats: the point that halves the
/// count of the floats between them rather than the length.
fn middle_float<T: Real>(lo: T, hi: T) -> T {
    let k = (i128::from(ordinal(lo)) + i128::from(ordinal(hi))) / 2;
    // Between two i64, so an i64.
    from_ordinal(k as i64)
}

/// Whether the bracket `[lo, hi]` reaches across zero or across more than a factor of two: then
/// halving its length and halving its count of floats are different things.
#[inline]
fn spans<T: Real>(lo: T, hi: T) -> bool {
    if lo <= T::ZERO && hi >= T::ZERO {
        return true;
    }
    let (a, b) = (lo.abs(), hi.abs());
    a.max(b) > T::TWO * a.min(b)
}

/// Whether halving the length of `[lo, hi]` reaches the absolute tolerance within as many
/// halvings as the type has bits: then the textbook bisection is as well bounded as the one on
/// the count of floats, and is what Brent's method and its readers expect.
#[inline]
fn reachable<T: Real>(lo: T, hi: T, opts: &RootOptions<T>) -> bool {
    opts.xtol > T::ZERO && hi * T::HALF - lo * T::HALF <= opts.xtol * T::TWO.powi(bits::<T>() - 1)
}

/// Where a bracketing method bisects. Within a factor of two the midpoint is also the middle
/// float, and with an absolute tolerance that halving the length reaches in time it is the
/// textbook choice. Otherwise, as at a root at zero under a purely relative tolerance, halving the
/// length would take up to 2,100 steps of an `f64` bracket, a binade a step, and halving the count
/// of floats up to 64 — but a bracket such as `[0, 1]` usually holds its root in its upper half,
/// where the middle float, `1.5·10⁻¹⁵⁴`, is of no use. So the two take turns, the midpoint first:
/// the count of floats is at least halved every two bisections, and the length too.
#[derive(Default)]
struct Halver {
    /// Whether the last bisection in turns took the midpoint.
    took_midpoint: bool,
}

impl Halver {
    /// The point that bisects `[lo, hi]`, `lo < hi`.
    fn point<T: Real>(&mut self, lo: T, hi: T, opts: &RootOptions<T>) -> T {
        let middle = midpoint(lo, hi);
        if !spans(lo, hi) || reachable(lo, hi, opts) {
            return middle;
        }
        self.took_midpoint = !self.took_midpoint;
        if self.took_midpoint {
            middle
        } else {
            middle_float(lo, hi)
        }
    }
}

/// The count of floats in Brent's bracket over the steps since its last bisection, for the
/// safeguard that bisects when two steps in a row have not halved it.
#[derive(Default)]
struct Watch {
    /// The counts after the last three steps, the latest last.
    counts: [f64; 3],
    /// How many of them there are.
    len: usize,
}

impl Watch {
    /// Forgets the counts: a bisection is about to be taken.
    fn reset(&mut self) {
        self.len = 0;
    }

    /// Records the count after a step.
    fn push(&mut self, n: f64) {
        if self.len < 3 {
            self.counts[self.len] = n;
            self.len += 1;
        } else {
            self.counts = [self.counts[1], self.counts[2], n];
        }
    }

    /// Whether the last two steps left more than half the floats of the bracket before them.
    fn stalled(&self) -> bool {
        self.len == 3 && self.counts[2] > 0.5 * self.counts[0]
    }
}

/// The root of `f` in the bracket `[a, b]`, by bisection.
///
/// `f(a)` and `f(b)` must be of opposite signs, or one of them zero, which is then the answer; the
/// ends may be given in either order, must be finite, and an infinite value at an end counts by
/// its sign. Each iteration evaluates `f` once at a point that bisects the bracket and keeps the
/// half on which the sign changes, until the root is within the tolerance of the estimate, the
/// value is exactly zero, or no float is left between the ends. A bracket within a factor of two
/// is halved in length, as in the textbook; one reaching across zero or across many binades is
/// halved in length and in its count of floats in turn, unless an absolute tolerance `xtol` is
/// reached by halving the length alone within 64 steps (32 in `f32`). Without such a tolerance, as
/// by default, any two bisections in a row divide the count of floats in the bracket by at least
/// 1.5 (by 2 for the one on the count, and by 1.5 for one within a factor of two, where the floats
/// are at most twice as dense at one end as at the other), so the method finishes whatever the
/// root: within 220 iterations of an `f64` bracket and 110 of an `f32` one, inside the default
/// limit of 512; the most measured, over 200,000 random brackets of each type spanning every
/// magnitude, were 75 and 40.
///
/// The answer always lies inside the bracket. A sign change through a pole is found like a root;
/// see [`RootOptions`] for the `ftol` that refuses it. Returns `None` when there is no bracket,
/// when an end is not finite, when `f` gives a NaN, or when `max_iter` is too small.
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
/// // The ends in either order, and a root at zero, where no relative tolerance can be met: the
/// // estimate is zero itself, or the float next to the sign change.
/// assert!(bisect(2.0, -1.0, |x: f64| x).unwrap().abs() < 1e-300);
///
/// // No bracket, so no answer.
/// assert!(bisect(0.0, 1.0, |x: f64| x * x + 1.0).is_none());
/// ```
pub fn bisect<T: Real>(a: T, b: T, f: impl Fn(T) -> T) -> Option<T> {
    bisect_with(a, b, f, &RootOptions::new())
}

/// The root of `f` in `[a, b]` by bisection, with the tolerances and the limit of `opts`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{RootOptions, bisect_with};
///
/// // With a tolerance on the value the method stops as soon as |f| is that small.
/// let opts = RootOptions::new().with_ftol(1e-3);
/// let x = bisect_with(1.0, 2.0, |x: f64| x * x - 2.0, &opts).unwrap();
/// assert!((x * x - 2.0).abs() <= 1e-3);
/// ```
pub fn bisect_with<T: Real>(
    a: T,
    b: T,
    mut f: impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> Option<T> {
    if !opts.valid() || !a.is_finite() || !b.is_finite() {
        return None;
    }
    let (mut lo, mut hi) = if a <= b { (a, b) } else { (b, a) };
    let mut flo = f(lo);
    if flo.is_nan() {
        return None;
    }
    if flo == T::ZERO {
        return Some(lo);
    }
    if lo == hi {
        return None;
    }
    let mut fhi = f(hi);
    if fhi.is_nan() {
        return None;
    }
    if fhi == T::ZERO {
        return Some(hi);
    }
    if negative(flo) == negative(fhi) {
        return None;
    }
    if opts.small(flo) || opts.small(fhi) {
        return Some(if flo.abs() <= fhi.abs() { lo } else { hi });
    }
    let mut halver = Halver::default();
    for _ in 0..opts.max_iter {
        let m = halver.point(lo, hi, opts);
        if m <= lo || m >= hi {
            // No float between the ends: the sign change is between two neighbours.
            let (x, fx) = if flo.abs() <= fhi.abs() {
                (lo, flo)
            } else {
                (hi, fhi)
            };
            return opts.value_ok(fx).then_some(x);
        }
        let fm = f(m);
        if fm.is_nan() {
            return None;
        }
        if fm == T::ZERO || opts.small(fm) {
            return Some(m);
        }
        if (m - lo).max(hi - m) <= opts.tol_at(m) {
            return opts.value_ok(fm).then_some(m);
        }
        if negative(fm) == negative(flo) {
            lo = m;
            flo = fm;
        } else {
            hi = m;
            fhi = fm;
        }
    }
    None
}

/// The root of `f` in the bracket `[a, b]`, by Brent's method.
///
/// This is Brent's `zero` (R. P. Brent, *Algorithms for Minimization without Derivatives*, 1973,
/// chapter 4), the algorithm of `scipy.optimize.brentq` too: the estimate `b` and the contrapoint
/// `c`, on whose values the sign changes, are kept with `|f(b)| ≤ |f(c)|`; each step is the inverse
/// quadratic interpolation through the last three points, or the secant through two, taken only
/// when it lands inside three quarters of the bracket and is shorter than half the step before
/// last, and the bisection otherwise; and a step shorter than the tolerance `δ = (xtol +
/// rtol·|b|)/2` — `2ε·|b| + xtol/2` with the default `rtol` of 4ε — is lengthened to `δ` towards
/// the contrapoint, so that the far end of the bracket is brought in too. It stops when `|c − b| ≤
/// 2δ`, and returns `b`. Each step evaluates `f` once.
///
/// Brent asks for a positive absolute tolerance, and with one that halving the length reaches
/// within 64 halvings (32 in `f32`) the method is his unchanged. With a smaller one, as with the
/// default `xtol` of zero at a root at zero, two safeguards of the bisection make it finish
/// there: the bisections take the midpoint and the middle float of the bracket in turn, as
/// [`bisect`] does, and a bisection is taken whenever two steps in a row have not halved the
/// count of floats in the bracket. Then two steps that do not halve the count are followed by a
/// bisection, and every other bisection of a bracket across zero or across binades is the one on
/// the count, so the count halves at least every six steps (a bisection within a factor of two
/// divides it by at least 1.5): at most 385 iterations of an `f64` bracket and 193 of an `f32`
/// one, inside the default limit of 512. The most measured, at a multiple root where
/// interpolation converges only linearly, is 206 evaluations, found by a search for the worst
/// case; random brackets spanning every magnitude took at most 136. Smooth problems with a simple
/// root take 6 to 15. That bound is not Brent's: where the method is his unchanged, with a
/// positive `xtol` that halving reaches, his own bound is about `(k + 1)²` evaluations for the `k`
/// bisections the tolerance needs, above the default limit in theory (then the answer is `None`);
/// searches for the worst case found at most 249 evaluations at `xtol = 2·10⁻¹²`, 216 at
/// `10⁻²⁰` and 206 at `10⁻³⁰⁰`, and never `None`.
///
/// `f(a)` and `f(b)` must be of opposite signs, or one of them zero; the ends may be in either
/// order and must be finite; an infinite value counts by its sign, and keeps the method from
/// interpolating through it. The answer lies inside the bracket. A sign change through a pole is
/// found like a root: see [`RootOptions`].
///
/// # Examples
///
/// ```
/// use a::math::numerics::{RootOptions, brent, brent_with};
///
/// // The real root of x³ − x − 2.
/// let x = brent(1.0, 2.0, |x: f64| x * x * x - x - 2.0).unwrap();
/// assert!((x - 1.521_379_706_804_567_6).abs() < 1e-15);
///
/// // 1/(x − 0.3) changes sign at its pole, which is all a bracketing method sees, so without a
/// // tolerance on the value the pole comes back; with one it does not, since |f| only grows there.
/// let pole = brent(0.0, 1.0, |x: f64| 1.0 / (x - 0.3)).unwrap();
/// assert!((pole - 0.3).abs() < 1e-15);
/// let opts = RootOptions::new().with_ftol(1e-10);
/// assert!(brent_with(0.0, 1.0, |x: f64| 1.0 / (x - 0.3), &opts).is_none());
/// ```
pub fn brent<T: Real>(a: T, b: T, f: impl Fn(T) -> T) -> Option<T> {
    brent_with(a, b, f, &RootOptions::new())
}

/// The root of `f` in `[a, b]` by Brent's method, with the tolerances and the limit of `opts`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{RootOptions, brent_with};
///
/// // SciPy's brentq tolerances, xtol 2·10⁻¹², rtol 4ε: e^x − 2 in 8 evaluations, as there.
/// let opts = RootOptions::new().with_xtol(2e-12);
/// let x = brent_with(0.0, 1.0, |x: f64| x.exp() - 2.0, &opts).unwrap();
/// assert!((x - core::f64::consts::LN_2).abs() < 1e-15);
/// ```
pub fn brent_with<T: Real>(
    a: T,
    b: T,
    mut f: impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> Option<T> {
    if !opts.valid() || !a.is_finite() || !b.is_finite() {
        return None;
    }
    let fa = f(a);
    if fa.is_nan() {
        return None;
    }
    if fa == T::ZERO {
        return Some(a);
    }
    if a == b {
        return None;
    }
    let fb = f(b);
    if fb.is_nan() {
        return None;
    }
    if fb == T::ZERO {
        return Some(b);
    }
    if negative(fa) == negative(fb) {
        return None;
    }
    if opts.small(fa) || opts.small(fb) {
        return Some(if fa.abs() <= fb.abs() { a } else { b });
    }
    // Brent's names: `b` the estimate, `c` the contrapoint, `a` the estimate before `b`; `d` the
    // last step and `e` the one before it.
    let (mut a, mut fa, mut b, mut fb) = (a, fa, b, fb);
    let (mut c, mut fc) = (a, fa);
    let mut d = b - a;
    let mut e = d;
    let mut halver = Halver::default();
    let mut watch = Watch::default();
    let three = T::from_f64(3.0);
    let mut iterations = 0;
    loop {
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol = opts.tol_at(b) * T::HALF;
        // Half the bracket, towards the contrapoint: exactly where the ends are small, as
        // `midpoint` forms it (halves of subnormals round away, and two floats apart around zero
        // would pass for none), and by halves where they are large, so that it cannot overflow.
        let m = if b.abs() <= T::ONE && c.abs() <= T::ONE {
            (c - b) * T::HALF
        } else {
            c * T::HALF - b * T::HALF
        };
        if m.abs() <= tol {
            return opts.value_ok(fb).then_some(b);
        }
        let (lo, hi) = if b < c { (b, c) } else { (c, b) };
        let middle = midpoint(lo, hi);
        if middle <= lo || middle >= hi {
            // Neighbours, which no tolerance finer than the floats can separate.
            return opts.value_ok(fb).then_some(b);
        }
        if iterations == opts.max_iter {
            return None;
        }
        iterations += 1;
        // Brent's tolerance in his three other uses, made at least the distance to the next float,
        // so that a step of it always moves the estimate (only a subnormal `b`, or an `rtol` below
        // 2ε, makes a difference).
        let step = tol.max(spacing(b));
        let extended = !reachable(lo, hi, opts);
        let mut next = None;
        if !(extended && watch.stalled())
            && e.abs() >= step
            && fa.abs() > fb.abs()
            && fa.is_finite()
            && fc.is_finite()
        {
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                // The secant through the estimate and the contrapoint.
                p = T::TWO * m * s;
                q = T::ONE - s;
            } else {
                // Inverse quadratic interpolation through the last three points.
                let qa = fa / fc;
                let r = fb / fc;
                p = s * (T::TWO * m * qa * (qa - r) - (b - a) * (r - T::ONE));
                q = (qa - T::ONE) * (r - T::ONE) * (s - T::ONE);
            }
            if p > T::ZERO {
                q = -q;
            } else {
                p = -p;
            }
            let before = e;
            e = d;
            // The step p/q must stay within three quarters of the bracket and be shorter than half
            // the step before last; then it is taken, at least `step` long.
            if T::TWO * p < three * m * q - (step * q).abs() && p < (T::HALF * before * q).abs() {
                d = p / q;
                let x = b + if d.abs() > step { d } else { step.copysign(m) };
                if lo < x && x < hi {
                    next = Some(x);
                }
            }
        }
        let x = match next {
            Some(x) => x,
            None => {
                let x = halver.point(lo, hi, opts);
                d = x - b;
                e = d;
                watch.reset();
                x
            }
        };
        a = b;
        fa = fb;
        b = x;
        fb = f(b);
        if fb.is_nan() {
            return None;
        }
        if fb == T::ZERO || opts.small(fb) {
            return Some(b);
        }
        if negative(fb) == negative(fc) {
            // The sign changes between the new estimate and the old: that is the bracket now.
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        watch.push(count(b, c));
    }
}

/// Whether `f` changes sign within [`RootOptions::reach`] of the estimate `x`, where it is `fx`
/// and not zero: against the estimate `prev` before it, which is that close when the last step was
/// short, or else by evaluating `f` at `x ± reach`, first ahead in the direction the iteration last
/// moved. A NaN probe finds nothing; an infinite one counts by its sign.
fn sign_change<T: Real>(
    x: T,
    fx: T,
    prev: T,
    fprev: T,
    f: &mut impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> bool {
    let reach = opts.reach(x);
    if (x - prev).abs() <= reach && negative(fprev) != negative(fx) {
        return true;
    }
    let ahead = if x < prev { -reach } else { reach };
    for probe in [x + ahead, x - ahead] {
        if probe.is_finite() {
            let v = f(probe);
            if v == T::ZERO || (!v.is_nan() && negative(v) != negative(fx)) {
                return true;
            }
        }
    }
    false
}

/// The root of `f` by the secant method, from the two distinct starting points `x0` and `x1`.
///
/// The next estimate is where the chord through the last two meets the axis, `y − f(y)·(y −
/// x)/(f(y) − f(x))`, its quotient of differences formed first so that nothing in it underflows or
/// overflows before it must. One evaluation of `f` an iteration buys the order `(1 + √5)/2 ≈ 1.618`
/// near a simple root. Nothing keeps the iterates near the root, and a small step proves nothing
/// by itself — a wild step and the two after it can land back on a point where `f` is far from
/// zero — so an estimate is returned only by the test of [`RootOptions`], a sign change within the
/// tolerance, checked with at most two more evaluations. Returns `None` when the starting points
/// coincide, when two values are equal (the chord is flat), when a step is not finite or no longer
/// moves, and after `max_iter` iterations.
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
///
/// // x¹⁰ − 1 from 0.1 and 0.2: the first chord is almost flat and throws the iteration to
/// // 977,517, from where it falls back on 0.2 and stays, f there −0.99999990. That is no root.
/// assert!(secant(0.1, 0.2, |x: f64| x.powi(10) - 1.0).is_none());
/// ```
pub fn secant<T: Real>(x0: T, x1: T, f: impl Fn(T) -> T) -> Option<T> {
    secant_with(x0, x1, f, &RootOptions::new())
}

/// The root of `f` by the secant method, with the tolerances and the limit of `opts`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{RootOptions, secant_with};
///
/// // A linear function is solved by its first chord, exactly.
/// let opts = RootOptions::new().with_max_iter(1);
/// assert_eq!(secant_with(0.0, 1.0, |x: f64| 3.0 * x - 6.0, &opts), Some(2.0));
/// ```
pub fn secant_with<T: Real>(
    x0: T,
    x1: T,
    mut f: impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> Option<T> {
    if !opts.valid() || !x0.is_finite() || !x1.is_finite() {
        return None;
    }
    let fx0 = f(x0);
    if fx0.is_nan() {
        return None;
    }
    if fx0 == T::ZERO || opts.small(fx0) {
        return Some(x0);
    }
    if x0 == x1 {
        return None;
    }
    let fx1 = f(x1);
    if fx1.is_nan() {
        return None;
    }
    if fx1 == T::ZERO || opts.small(fx1) {
        return Some(x1);
    }
    let (mut x, mut fx, mut y, mut fy) = (x0, fx0, x1, fx1);
    for _ in 0..opts.max_iter {
        let (mut dx, mut df) = (y - x, fy - fx);
        if !dx.is_finite() || !df.is_finite() {
            // By halves, which leave the quotient as it is.
            dx = y * T::HALF - x * T::HALF;
            df = fy * T::HALF - fx * T::HALF;
        }
        if df == T::ZERO {
            return None;
        }
        let s = y - fy * (dx / df);
        if !s.is_finite() {
            return None;
        }
        let fs = f(s);
        if fs.is_nan() {
            return None;
        }
        if fs == T::ZERO || opts.small(fs) {
            return Some(s);
        }
        if (s - y).abs() <= opts.reach(s) {
            if opts.ftol == T::ZERO && sign_change(s, fs, y, fy, &mut f, opts) {
                return Some(s);
            }
            if s == y {
                return None;
            }
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
/// so the digits double; at a root of multiplicity `m` the error shrinks only by `(m − 1)/m`. An
/// estimate is returned by the test of [`RootOptions`]: exactly zero, a sign change within the
/// tolerance after a step that short (at most two more evaluations of `f`), or with `ftol` a small
/// enough value. Returns `None` when the derivative vanishes or is not finite where it is needed,
/// when a step is not finite or no longer moves, and after `max_iter` iterations.
///
/// # Examples
///
/// ```
/// use a::math::numerics::newton;
///
/// // The root of e^x − 3x² nearest 1, which Newton's method returns after five iterations from
/// // 1: the fifth step is within the tolerance, and finds the sign change.
/// let x = newton(1.0, |x: f64| x.exp() - 3.0 * x * x, |x: f64| x.exp() - 6.0 * x).unwrap();
/// assert!((x - 0.910_007_572_488_709).abs() < 1e-15);
///
/// // A start on the root is the answer, whatever the derivative does there.
/// assert_eq!(newton(0.0, |x: f64| x * x, |x: f64| 2.0 * x), Some(0.0));
/// // A start on a stationary point that is not a root gives no step and no answer.
/// assert!(newton(0.0, |x: f64| x * x - 1.0, |x: f64| 2.0 * x).is_none());
/// ```
pub fn newton<T: Real>(x0: T, f: impl Fn(T) -> T, df: impl Fn(T) -> T) -> Option<T> {
    newton_with(x0, f, df, &RootOptions::new())
}

/// The root of `f` by Newton's method, with the tolerances and the limit of `opts`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{RootOptions, newton_with};
///
/// // x² has a double root at zero, where Newton halves the error a step and the relative
/// // tolerance cannot be met; a tolerance on the value stops it.
/// let opts = RootOptions::new().with_ftol(1e-20);
/// let x = newton_with(1.0, |x: f64| x * x, |x: f64| 2.0 * x, &opts).unwrap();
/// assert!(x * x <= 1e-20);
/// ```
pub fn newton_with<T: Real>(
    x0: T,
    mut f: impl FnMut(T) -> T,
    mut df: impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> Option<T> {
    if !opts.valid() || !x0.is_finite() {
        return None;
    }
    let mut x = x0;
    let mut fx = f(x);
    if fx.is_nan() {
        return None;
    }
    if fx == T::ZERO || opts.small(fx) {
        return Some(x);
    }
    for _ in 0..opts.max_iter {
        let d = df(x);
        if !d.is_finite() || d == T::ZERO {
            return None;
        }
        let s = x - fx / d;
        if !s.is_finite() {
            return None;
        }
        let fs = f(s);
        if fs.is_nan() {
            return None;
        }
        if fs == T::ZERO || opts.small(fs) {
            return Some(s);
        }
        if (s - x).abs() <= opts.reach(s) {
            if opts.ftol == T::ZERO && sign_change(s, fs, x, fx, &mut f, opts) {
                return Some(s);
            }
            if s == x {
                return None;
            }
        }
        x = s;
        fx = fs;
    }
    None
}

/// The root of `f` by Halley's method, from `x0`, with `f`, `df` and the second derivative `ddf`.
///
/// The next estimate is `x − 2f·f′/(2f′² − f·f″)`, Newton's step `n = f/f′` corrected for the
/// curvature: formed as `n/(1 − n·f″/(2f′))`, so that no product of values or derivatives leaves the
/// range before the quotient does. Near a simple root the error cubes each iteration, so the digits
/// triple, for three evaluations — `f`, `f′` and `f″` — an iteration. Where the correction would
/// reverse the step or make it infinite (`n·f″/(2f′) ≥ 1`), Newton's step is taken instead. The
/// test for an answer and the reasons for `None` are those of [`newton`]; in particular a
/// stationary point, where `f′` is zero and Halley's formula would not move, is no answer.
///
/// # Examples
///
/// ```
/// use a::math::numerics::halley;
///
/// // The root of cos x = x again, returned after four iterations.
/// let x = halley(0.5, |x: f64| x.cos() - x, |x: f64| -x.sin() - 1.0, |x: f64| -x.cos()).unwrap();
/// assert!((x - 0.739_085_133_215_160_7).abs() < 1e-15);
///
/// // From the minimum of x² − 1 there is no step to take.
/// assert!(halley(0.0, |x: f64| x * x - 1.0, |x: f64| 2.0 * x, |_x: f64| 2.0).is_none());
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
///
/// # Examples
///
/// ```
/// use a::math::numerics::{RootOptions, halley_with};
///
/// // √2 from 1 in four steps: the error goes 1.4·10⁻², 3.6·10⁻⁷, 2.2·10⁻¹⁶ and 0, where the sign
/// // change against the step before is the verification.
/// let opts = RootOptions::new().with_max_iter(4);
/// let x = halley_with(1.0, |x: f64| x * x - 2.0, |x: f64| 2.0 * x, |_x: f64| 2.0, &opts).unwrap();
/// assert!((x - core::f64::consts::SQRT_2).abs() <= 4.0 * f64::EPSILON);
/// ```
pub fn halley_with<T: Real>(
    x0: T,
    mut f: impl FnMut(T) -> T,
    mut df: impl FnMut(T) -> T,
    mut ddf: impl FnMut(T) -> T,
    opts: &RootOptions<T>,
) -> Option<T> {
    if !opts.valid() || !x0.is_finite() {
        return None;
    }
    let mut x = x0;
    let mut fx = f(x);
    if fx.is_nan() {
        return None;
    }
    if fx == T::ZERO || opts.small(fx) {
        return Some(x);
    }
    for _ in 0..opts.max_iter {
        let d1 = df(x);
        if !d1.is_finite() || d1 == T::ZERO {
            return None;
        }
        let d2 = ddf(x);
        if !d2.is_finite() {
            return None;
        }
        let n = fx / d1;
        let denominator = T::ONE - n * (d2 / (T::TWO * d1));
        let step = if denominator > T::ZERO && denominator.is_finite() {
            n / denominator
        } else {
            n
        };
        let s = x - step;
        if !s.is_finite() {
            return None;
        }
        let fs = f(s);
        if fs.is_nan() {
            return None;
        }
        if fs == T::ZERO || opts.small(fs) {
            return Some(s);
        }
        if (s - x).abs() <= opts.reach(s) {
            if opts.ftol == T::ZERO && sign_change(s, fs, x, fx, &mut f, opts) {
                return Some(s);
            }
            if s == x {
                return None;
            }
        }
        x = s;
        fx = fs;
    }
    None
}
