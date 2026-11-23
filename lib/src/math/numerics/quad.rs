//! The integral of a function of one real variable, and of several.
//!
//! * [`quad`], [`quad_g10k21`] and [`quad_g7k15`]: adaptive Gauss–Kronrod quadrature as
//!   QUADPACK's QAGS does it — bisection of the interval with the largest error estimate,
//!   QUADPACK's error estimate and round-off detection, and Wynn's epsilon algorithm to
//!   extrapolate towards an end singularity. Infinite limits are mapped onto `(0, 1]` by QAGI's
//!   `x = a + (1 − t)/t`; [`quad_half_infinite`] and [`quad_infinite`] are shorthands.
//! * [`clenshaw_curtis`]: Clenshaw–Curtis quadrature, adaptive both in the degree of each panel
//!   (nested Chebyshev–Lobatto points, from 33 to 257, every value reused) and in the panels
//!   (the worst bisected), for smooth integrands on finite intervals.
//! * [`tanh_sinh`]: the double exponential rule of Takahasi and Mori, for integrands that blow up
//!   at the ends of the interval, with its exp-sinh and sinh-sinh forms on infinite intervals.
//! * [`cubature`]: Genz and Malik's embedded rule of degree 7 on boxes of 2 dimensions and more,
//!   bisected where the error is largest, along the axis of the largest fourth difference.
//! * [`monte_carlo`] and [`quasi_monte_carlo`]: plain Monte Carlo on a seeded xoshiro256**
//!   generator, and quasi-Monte Carlo on the Sobol' sequence of Joe and Kuo with random shifts,
//!   each with the standard error of its estimate; [`Sobol`] is the sequence itself.
//!
//! The one-dimensional rules share [`QuadOptions`] and [`QuadResult`], and a contract: an
//! interval with a NaN for a limit or options that ask for nothing (no tolerance, no room)
//! answer `None`; `a = b` is the integral 0; `a > b` the negated integral over `[b, a]`. A result
//! is `converged` only when its error estimate is within the tolerances, and never when the
//! estimate or its error is not finite in the integrand's type (an `f32` integral above
//! `f32::MAX` comes back infinite, with an infinite error, not converged); otherwise it is the
//! best estimate reached, with its error estimate, and `converged` false. An integrand that
//! returns a NaN or an infinity at a point where it is evaluated ends the integration there,
//! not converged. The rules work in `f64` and call an `f32` integrand at the `f32` points
//! nearest their nodes, its tolerances and round-off measured in `f32`.

mod clenshaw;
mod cubature;
mod epsilon;
mod kronrod;
mod monte_carlo;
mod sobol_table;
mod tanh_sinh;

pub use monte_carlo::{
    MonteCarloOptions, SOBOL_MAX_DIMENSION, Sobol, monte_carlo, monte_carlo_with,
    quasi_monte_carlo, quasi_monte_carlo_with,
};

use super::Real;
use kronrod::Map;

/// The options of the quadrature rules: the tolerances and how much work a rule may do.
///
/// A result is converged only if its error estimate is at most `max(epsabs, epsrel·|value|)`:
/// that is necessary, not sufficient, for every rule also asks of an estimate that it has
/// resolved the integrand — an error below a tenth of the estimate, or at the round-off of the
/// sums — before it accepts it, whatever the absolute tolerance allows. The adaptive rules keep
/// at most `max_subintervals` subintervals (panels, boxes), every rule calls the integrand at
/// most `max_evaluations` times, and a rule stops at the first of the two.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{QuadOptions, quad_g10k21};
///
/// let loose = QuadOptions::new().with_epsabs(1e-6).with_epsrel(1e-6);
/// let r = quad_g10k21(|x: f64| (30.0 * x).sin(), 0.0, 1.0, &loose).unwrap();
/// assert!(r.converged && r.error <= 1e-6);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct QuadOptions<T> {
    /// The absolute tolerance of the error estimate.
    pub epsabs: T,
    /// The relative tolerance of the error estimate.
    pub epsrel: T,
    /// The most subintervals (panels of Clenshaw–Curtis, boxes of the cubature) an adaptive rule
    /// keeps.
    pub max_subintervals: usize,
    /// The most evaluations of the integrand a rule may spend.
    pub max_evaluations: usize,
}

impl<T: Real> Default for QuadOptions<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> QuadOptions<T> {
    /// The default options: `epsabs` and `epsrel` `1e-12` for `f64` and `1e-5` for `f32` (about
    /// 4500 and 80 units of their epsilon, above the floor of the round-off of a rule's sums),
    /// `max_subintervals` 200 and `max_evaluations` 100,000.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::QuadOptions;
    ///
    /// assert_eq!(QuadOptions::<f64>::new().epsrel, 1e-12);
    /// assert_eq!(QuadOptions::<f32>::new().epsrel, 1e-5);
    /// ```
    pub fn new() -> Self {
        let tol = if T::MANTISSA_DIGITS > 24 { 1e-12 } else { 1e-5 };
        Self {
            epsabs: T::from_f64(tol),
            epsrel: T::from_f64(tol),
            max_subintervals: 200,
            max_evaluations: 100_000,
        }
    }

    /// Sets [`epsabs`](Self::epsabs).
    pub fn with_epsabs(mut self, epsabs: T) -> Self {
        self.epsabs = epsabs;
        self
    }

    /// Sets [`epsrel`](Self::epsrel).
    pub fn with_epsrel(mut self, epsrel: T) -> Self {
        self.epsrel = epsrel;
        self
    }

    /// Sets [`max_subintervals`](Self::max_subintervals).
    pub fn with_max_subintervals(mut self, max_subintervals: usize) -> Self {
        self.max_subintervals = max_subintervals;
        self
    }

    /// Sets [`max_evaluations`](Self::max_evaluations).
    pub fn with_max_evaluations(mut self, max_evaluations: usize) -> Self {
        self.max_evaluations = max_evaluations;
        self
    }
}

/// What a quadrature rule returns: the estimate, its error estimate, how many times the integrand
/// was called, and whether the error estimate came within the tolerances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadResult<T> {
    /// The estimate of the integral.
    pub value: T,
    /// The estimate of the absolute error of `value`: for the Monte Carlo rules, one standard
    /// error.
    pub error: T,
    /// The number of evaluations of the integrand.
    pub evaluations: usize,
    /// Whether `error` came within the tolerances; for the Monte Carlo rules, which have none,
    /// whether the estimate and its standard error are finite numbers. Never true for a `value`
    /// or an `error` that is not finite in `T`.
    pub converged: bool,
}

impl<T: Real> QuadResult<T> {
    /// The result as a pair `(value, error)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::quad;
    ///
    /// let r = quad(|x: f64| x, 0.0, 2.0).unwrap();
    /// assert_eq!(r.as_tuple(), (r.value, r.error));
    /// ```
    pub fn as_tuple(&self) -> (T, T) {
        (self.value, self.error)
    }
}

/// The options in `f64`, with the precision of the integrand's type.
#[derive(Clone, Copy)]
struct Settings {
    epsabs: f64,
    epsrel: f64,
    max_subintervals: usize,
    max_evaluations: usize,
    /// The epsilon of the integrand's type.
    eps: f64,
    /// Its smallest normal number.
    tiny: f64,
}

impl Settings {
    /// The options as the rules use them, or `None` for options that ask for nothing: a negative
    /// or NaN tolerance, both tolerances zero, or no room at all.
    fn of<T: Real>(opts: &QuadOptions<T>) -> Option<Self> {
        let (epsabs, epsrel) = (opts.epsabs.to_f64(), opts.epsrel.to_f64());
        let valid = |t: f64| t >= 0.0 && t.is_finite();
        if !valid(epsabs) || !valid(epsrel) || (epsabs == 0.0 && epsrel == 0.0) {
            return None;
        }
        if opts.max_subintervals == 0 || opts.max_evaluations == 0 {
            return None;
        }
        Some(Self {
            epsabs,
            epsrel,
            max_subintervals: opts.max_subintervals,
            max_evaluations: opts.max_evaluations,
            eps: T::EPSILON.to_f64(),
            tiny: T::MIN_POSITIVE.to_f64(),
        })
    }

    /// The tolerance on the error of an estimate `v`.
    fn bound(&self, v: f64) -> f64 {
        self.epsabs.max(self.epsrel * v.abs())
    }

    /// Whether an estimate `value` with the error estimate `error` meets the tolerances: both
    /// finite, and the error within [`bound`](Self::bound). An infinite value would otherwise
    /// pass its own relative tolerance, `∞ ≤ epsrel·∞`.
    fn met(&self, error: f64, value: f64) -> bool {
        value.is_finite() && error <= self.bound(value)
    }
}

/// A result in the type of the integrand: `value` and `error` rounded to `T`, and converged only
/// if both are finite there. An integral beyond the range of `T` (an `f32` integrand whose
/// values all fit, over a wide interval) is no result, however small its error was in the `f64`
/// the rules work in; and a value that is not a finite number carries an infinite error.
pub(super) fn in_type<T: Real>(
    value: f64,
    error: f64,
    evaluations: usize,
    converged: bool,
) -> QuadResult<T> {
    let (value, error) = (T::from_f64(value), T::from_f64(error));
    let finite = value.is_finite() && error.is_finite();
    QuadResult {
        value,
        error: if finite { error } else { T::INFINITY },
        evaluations,
        converged: converged && finite,
    }
}

/// The integrand of a one-dimensional rule, called at the point of its own type nearest the
/// node, and counted.
struct Integrand<'a, T> {
    f: &'a mut dyn FnMut(T) -> T,
    evaluations: usize,
}

impl<T: Real> Integrand<'_, T> {
    /// The value at `x`, in `f64`.
    fn at(&mut self, x: f64) -> f64 {
        self.evaluations += 1;
        (self.f)(T::from_f64(x)).to_f64()
    }
}

/// What a rule found, before the sign of a reversed interval and the conversion to the type.
#[derive(Clone, Copy, Debug)]
struct Outcome {
    value: f64,
    error: f64,
    converged: bool,
}

impl Outcome {
    /// The end of an integration at a value of the integrand that is not a number.
    fn failed(value: f64) -> Self {
        Self {
            value,
            error: f64::INFINITY,
            converged: false,
        }
    }
}

/// The interval of a one-dimensional integral, once the order of its limits is taken out.
#[derive(Clone, Copy, Debug)]
enum Range {
    /// `a = b`.
    Empty,
    /// `[a, b]`, `a < b`, both finite.
    Finite(f64, f64),
    /// `[a, ∞)`.
    Upper(f64),
    /// `(−∞, b]`.
    Lower(f64),
    /// `(−∞, ∞)`.
    Whole,
}

/// The range of `∫ₐᵇ` and the sign to give the result, or `None` for a NaN limit.
fn range<T: Real>(a: T, b: T) -> Option<(Range, f64)> {
    let (a, b) = (a.to_f64(), b.to_f64());
    if a.is_nan() || b.is_nan() {
        return None;
    }
    if a == b {
        return Some((Range::Empty, 1.0));
    }
    let (lo, hi, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
    let range = match (lo.is_finite(), hi.is_finite()) {
        (true, true) => Range::Finite(lo, hi),
        (true, false) => Range::Upper(lo),
        (false, true) => Range::Lower(hi),
        (false, false) => Range::Whole,
    };
    Some((range, sign))
}

/// A one-dimensional rule: the range and the settings checked, the empty range answered, the
/// rule run (`None` from it: a range it does not take), and the result signed and converted.
fn one_dimensional<T: Real>(
    f: &mut dyn FnMut(T) -> T,
    a: T,
    b: T,
    opts: &QuadOptions<T>,
    rule: impl FnOnce(&mut Integrand<'_, T>, Range, &Settings) -> Option<Outcome>,
) -> Option<QuadResult<T>> {
    let (range, sign) = range(a, b)?;
    let settings = Settings::of(opts)?;
    if let Range::Empty = range {
        return Some(QuadResult {
            value: T::ZERO,
            error: T::ZERO,
            evaluations: 0,
            converged: true,
        });
    }
    let mut integrand = Integrand { f, evaluations: 0 };
    let outcome = rule(&mut integrand, range, &settings)?;
    Some(in_type(
        sign * outcome.value,
        outcome.error,
        integrand.evaluations,
        outcome.converged,
    ))
}

/// The Kronrod rules on any range, through QAGI's maps for the infinite ones, where the rule is
/// QAGI's 15-point pair whichever was asked for (as in QUADPACK, whose QAGI has no other); `None`
/// when the budget does not allow one application of the rule.
fn kronrod_rule<T: Real>(
    f: &mut Integrand<'_, T>,
    range: Range,
    s: &Settings,
    table: &kronrod::Table,
) -> Option<Outcome> {
    let (map, a, b, table): (_, _, _, &kronrod::Table) = match range {
        Range::Finite(a, b) => (Map::Identity, a, b, table),
        Range::Upper(a) => (Map::Upper(a), 0.0, 1.0, &kronrod::G7K15),
        Range::Lower(b) => (Map::Lower(b), 0.0, 1.0, &kronrod::G7K15),
        Range::Whole => (Map::Whole, 0.0, 1.0, &kronrod::G7K15),
        Range::Empty => unreachable!("the empty range is answered before"),
    };
    let per_rule = (2 * table.len() - 1) * if let Map::Whole = map { 2 } else { 1 };
    if s.max_evaluations < per_rule {
        return None;
    }
    Some(kronrod::qags(f, map, a, b, table, s))
}

/// The integral of `f` on `[a, b]` by adaptive Gauss–Kronrod quadrature with the G10–K21 rule
/// ([`quad_g10k21`]), to the default tolerances of [`QuadOptions`]. Either limit may be infinite.
///
/// Returns `None` if a limit is NaN.
///
/// # Examples
///
/// ```
/// use a::math::numerics::quad;
///
/// let r = quad(|x: f64| x.exp(), 0.0, 1.0).unwrap();
/// assert!((r.value - (core::f64::consts::E - 1.0)).abs() < 1e-15);
/// assert!(r.converged && r.evaluations == 21);
///
/// // An end singularity, which the extrapolation of QAGS deals with: ∫₀¹ ln x dx = −1.
/// let r = quad(|x: f64| x.ln(), 0.0, 1.0).unwrap();
/// assert!(r.converged && (r.value + 1.0).abs() < 1e-14);
///
/// // A reversed interval is the negated integral, an empty one is 0.
/// assert_eq!(quad(|x: f64| x, 1.0, 1.0).unwrap().value, 0.0);
/// assert!((quad(|x: f64| x, 1.0, 0.0).unwrap().value + 0.5).abs() < 1e-16);
/// ```
pub fn quad<T: Real>(f: impl FnMut(T) -> T, a: T, b: T) -> Option<QuadResult<T>> {
    quad_g10k21(f, a, b, &QuadOptions::new())
}

/// The integral of `f` on `[a, b]` by QUADPACK's QAGS (Piessens et al. 1983) with the 15-point
/// Kronrod rule and its 7-point Gauss rule: the interval with the largest error estimate is
/// bisected; the error of an interval is QUADPACK's, `|K − G|` scaled by `∫|f − mean|` and never
/// below `50 ε` times `∫|f|` over the interval; round-off that stops the error from falling ends
/// the bisection; and the sequence of the estimates as the bisection homes in on an end
/// singularity is extrapolated by Wynn's epsilon algorithm, with QUADPACK's tests for a divergent
/// integral.
///
/// Three departures from QUADPACK, each towards honesty. The totals of the estimates and of
/// their errors are summed afresh over the intervals after every bisection, where QUADPACK keeps
/// running sums that can cancel to an error of exactly 0. A value of a bisected interval that
/// the nodes of the half it lies in do not account for (neither node beside it a tenth as large)
/// is evidence of a feature between them, and stays in the error of the half, and of its halves,
/// as its share of the interval's `∫|f|`, until a node beside it sees a tenth of it — up to four
/// such values an interval, the heaviest, and the value at the centre of a bisected interval,
/// which lies at an end of both halves, for each half: what it can hide there is its value
/// times the gap to the half's outermost node, so that a jump exactly at the centre costs
/// bisections towards it rather than being taken on trust. And a result whose error estimate is
/// a tenth of the result or more is not accepted, unless the error is at the round-off of the
/// sums: QUADPACK accepts it where the absolute tolerance allows, and so passes off as converged
/// a peak it has only grazed.
///
/// So a value the nodes have seen, if among the four heaviest of its interval, is not lost
/// between the nodes of the halves while no node beside it sees a tenth of it. That does not
/// make every feature found that a node has seen. A
/// value seen far down the tail of a narrow feature is a share far below the feature's own, and
/// may be below the tolerance. And a node that sees a tenth of a value accounts for it: if the
/// interval about that node is bisected in turn, the node's own value and share, of the smaller
/// interval, are what stands for the feature, and at a loose tolerance that evidence can shrink
/// below the tolerance before a node finds the feature. On pairs of peaks of widths `10⁻³` and
/// `3·10⁻⁴` on `[0, 1]`, of the runs whose nodes saw both peaks above `10⁻³` of their heights,
/// none was beyond its estimate at `10⁻⁸` or `10⁻¹²`, and at `10⁻⁴` 33 of 246 converged runs of
/// [`quad`] and 24 of 142 of this rule were, up to 120 times. A feature the nodes do not look at
/// is not there for the rule: split the interval at the features where they are known.
///
/// No node is evaluated at or beyond an end of an interval at least 460 units in the last place
/// of its larger end wide (234 for this rule's G7–K15 pair) in `f64`, or 231 (118) in `f32`: the
/// outermost node of the first panel lies `(1 − u)` of the half-width from the end, `u` its
/// abscissa, and the centre's own rounding takes half an ulp of that in `f64` (in `f32` the nodes
/// are formed in `f64` and rounded once). On a narrower interval the first panel's outermost
/// nodes round onto the ends, or beyond them where the centre itself rounds onto one, and an
/// integrand infinite or undefined there makes the result a number that is not finite, not
/// converged. On `[a, ∞)` and `(−∞, a]` the same holds while `|a|` is below `2⁴⁶` (`f64`) or
/// `2¹⁷` (`f32`).
///
/// Either limit may be infinite, as in QUADPACK's QAGI: `[a, ∞)` is mapped onto `(0, 1]` by
/// `x = a + (1 − t)/t`, `(−∞, b]` by `x = b − (1 − t)/t`, and the two halves of the line are
/// folded onto `(0, 1]` together; the rule on the mapped interval is the 15-point pair.
///
/// Returns `None` if a limit is NaN, if the options ask for no tolerance (both zero, or one
/// negative or NaN), or if they leave no room for one application of the rule.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{QuadOptions, quad_g7k15};
///
/// // ∫₀^∞ e^(−x) cos x dx = 1/2.
/// let r = quad_g7k15(|x: f64| (-x).exp() * x.cos(), 0.0, f64::INFINITY, &QuadOptions::new())
///     .unwrap();
/// assert!(r.converged && (r.value - 0.5).abs() < 1e-13);
/// ```
pub fn quad_g7k15<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    one_dimensional(&mut f, a, b, opts, |f, r, s| {
        kronrod_rule(f, r, s, &kronrod::G7K15)
    })
}

/// The integral of `f` on `[a, b]` by QAGS with the 21-point Kronrod rule and its 10-point Gauss
/// rule, which is [`quad`] with options. Either limit may be infinite, as for [`quad_g7k15`]: an
/// infinite interval is integrated as QUADPACK's QAGI does it, with the 15-point pair on the
/// mapped interval.
///
/// Returns `None` if a limit is NaN, if the options ask for no tolerance, or if they leave no
/// room for one application of the rule.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{QuadOptions, quad_g10k21};
///
/// // An algebraic singularity at the right end: ∫₀¹ (1 − x)^(−1/2) dx = 2.
/// let r = quad_g10k21(|x: f64| (1.0 - x).sqrt().recip(), 0.0, 1.0, &QuadOptions::new())
///     .unwrap();
/// assert!(r.converged && (r.value - 2.0).abs() <= r.error && r.error < 1e-12);
///
/// // A divergent integral is reported as such: ∫₁^∞ dx/x.
/// let r = quad_g10k21(|x: f64| 1.0 / x, 1.0, f64::INFINITY, &QuadOptions::new()).unwrap();
/// assert!(!r.converged);
/// ```
pub fn quad_g10k21<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    one_dimensional(&mut f, a, b, opts, |f, r, s| {
        kronrod_rule(f, r, s, &kronrod::G10K21)
    })
}

/// The integral of `f` on `[a, ∞)`: [`quad_g10k21`] with `b = ∞`.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{QuadOptions, quad_half_infinite};
///
/// // ∫₀^∞ e^(−x) dx = 1, and the algebraic tail of ∫₁^∞ x^(−1.1) dx = 10.
/// let r = quad_half_infinite(|x: f64| (-x).exp(), 0.0, &QuadOptions::new()).unwrap();
/// assert!(r.converged && (r.value - 1.0).abs() < 1e-14);
/// let r = quad_half_infinite(|x: f64| x.powf(-1.1), 1.0, &QuadOptions::new()).unwrap();
/// assert!(r.converged && (r.value - 10.0).abs() < 1e-9);
/// ```
pub fn quad_half_infinite<T: Real>(
    f: impl FnMut(T) -> T,
    a: T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    quad_g10k21(f, a, T::INFINITY, opts)
}

/// The integral of `f` on `(−∞, ∞)`: [`quad_g10k21`] with both limits infinite.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{QuadOptions, quad_infinite};
///
/// // ∫ e^(−x²) dx over the line is √π.
/// let r = quad_infinite(|x: f64| (-x * x).exp(), &QuadOptions::new()).unwrap();
/// assert!(r.converged && (r.value - core::f64::consts::PI.sqrt()).abs() < 1e-14);
/// ```
pub fn quad_infinite<T: Real>(
    f: impl FnMut(T) -> T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    quad_g10k21(f, -T::INFINITY, T::INFINITY, opts)
}

/// The integral of `f` on the finite `[a, b]` by Clenshaw–Curtis quadrature, adaptive both in the
/// degree and in the panels.
///
/// Each panel is the interpolant through the Chebyshev–Lobatto points `cos(kπ/N)`, integrated
/// exactly; a panel starts with `N = 32` (33 points) and doubles `N` up to 256, each level
/// reusing every value of the one before. The error estimate of a panel is the difference of
/// its last two levels, enlarged by the geometric tail `r/(1 − r)` where the ratio `r` of the
/// last two differences says that the convergence is slow; it is never below 50 ε times
/// `∫|f|`, and it holds what the rounding of the points could move the sum by, about
/// `ε max(|a|, |b|)` times the variation of the integrand over them, which no difference of the
/// levels sees. The panel whose error is largest is refined — doubled, or bisected once it has
/// 257 points, each half starting again at 33 points with the three shared values reused — and
/// a bisected panel keeps its own estimate wherever it is better than its halves'. A value of a
/// bisected panel that the points of the half it lies in do not account for (neither point
/// beside it a tenth as large) stays in that half's error, its share of the panel's `∫|f|`,
/// until a point beside it sees a tenth of it — up to four such values a panel, the heaviest, as
/// in [`quad_g7k15`]: a value a panel has seen, if among the four heaviest, is not lost by
/// halves that miss it while no point beside it sees a tenth of it. The errors of the panels are
/// summed. As in [`quad_g7k15`], a
/// result whose error estimate is a tenth of the result or more is not accepted, unless the
/// error is at the round-off of the sums.
///
/// That does not make every seen peak found, for the reasons [`quad_g7k15`] gives: a value seen
/// far down a narrow peak's tail is a share far below the peak's, and once a point sees a tenth
/// of a value the evidence is that point's, which at a loose tolerance can shrink below the
/// tolerance before a point finds the peak. On pairs of peaks of widths `10⁻³` and `3·10⁻⁴` on
/// `[0, 1]`, of the converged runs whose points saw both peaks above `10⁻³` of their heights,
/// none was beyond its estimate at `10⁻⁸` or `10⁻¹²`, and 154 of 1009 were at `10⁻⁴`, up to 230
/// times.
///
/// The rule evaluates the integrand at the ends of the interval: an integrand infinite there,
/// or with an infinite limit, is for [`tanh_sinh`] or [`quad`].
///
/// Returns `None` if a limit is NaN or infinite, if the options ask for no tolerance, or if they
/// leave no room for the first 35 evaluations (the 33 points of the first panel and its two off
/// the grid).
///
/// # Examples
///
/// ```
/// use a::math::numerics::{QuadOptions, clenshaw_curtis};
///
/// // The integrand is analytic, so the rule converges geometrically in the degree.
/// let r = clenshaw_curtis(|x: f64| 1.0 / (1.0 + 25.0 * x * x), -1.0, 1.0, &QuadOptions::new())
///     .unwrap();
/// assert!(r.converged && (r.value - 2.0 * 5.0f64.atan() / 5.0).abs() < 1e-14);
/// ```
pub fn clenshaw_curtis<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    one_dimensional(&mut f, a, b, opts, |f, r, s| match r {
        Range::Finite(a, b) => clenshaw::integrate(f, a, b, s),
        _ => None,
    })
}

/// The integral of `f` on `[a, b]` by the tanh-sinh (double exponential) rule of Takahasi and
/// Mori (1974): the substitution `x = c + h·tanh((π/2) sinh t)` pushes the ends of the interval
/// out to infinity, where the trapezoidal rule in `t` converges as fast as the integrand is
/// smooth inside — faster than any algebraic rule where the integrand or one of its derivatives
/// becomes infinite at an end. On `[a, ∞)` the map is exp-sinh, `x = a + e^((π/2) sinh t)`, and
/// on the whole line sinh-sinh, `x = sinh((π/2) sinh t)`.
///
/// Every node is formed from its distance to the nearer end, so that the nodes reach an end at
/// 0 down to the smallest normal number, and no node is evaluated where it would round onto an
/// end: an integrand infinite or NaN at an end is welcome. The step starts at 1 and halves, each
/// level reusing the values of the last; every level is summed over the whole range in which
/// the nodes are distinct from the ends (on an infinite side, until its terms have died away).
/// The error estimate is the difference of the last two levels while the convergence is double
/// exponential (each difference at most the 1.5th power of the one before, two levels running),
/// and otherwise — a kink or a singularity inside the interval, where the levels converge
/// slowly and erratically — the larger of the last two differences times the geometric tail
/// `r/(1 − r)` of their ratio; to it are added an estimate of what lies beyond the last node of
/// each side, the round-off of the sum, and what the rounding of the nodes could move it by, about
/// `Σ (ε_T|xᵢ| + ε dᵢ) |fᵢ − fᵢ₋₁|` over neighbouring nodes (`dᵢ` the distance of the node from
/// the end it is formed from), which every level shares and no difference of levels sees. The
/// double exponential regime is entered only from a difference below `∫|f|`. No result is
/// accepted before the third level, nor while the levels move apart, nor, as in
/// [`quad_g7k15`], with an error estimate a tenth of the result or more unless the error is at
/// the round-off of the sums: levels that disagree in their first digit have only grazed a
/// feature.
///
/// An interval with no number of the type at its centre but its ends (`[1, 1 + ε]`, or `[a, ∞)`
/// with `a ± 1` rounding to `a`) is not evaluated at all, and one whose half-width is below the
/// smallest normal number has no node but its centre: both end unconverged, with an infinite
/// error.
///
/// An end singularity at a nonzero end, as at `x = 1` in `(1 − x)^(−1/2)`, can only be
/// approached to within an ulp of the end: the mass beyond it is in the error estimate, and an
/// integrand whose mass there is above the tolerance is not converged — write it as a function
/// of the distance to that end, on an interval that starts at 0, or use [`quad`]. A kink, a jump
/// or a narrow peak inside the interval belongs to [`quad`], which bisects where the error is:
/// the levels of this rule refine everywhere at once, so such a feature costs evaluations
/// everywhere, a peak between all the nodes (every value exactly 0) is not there for the rule,
/// and on a kink like `|x − c|^p` the levels can agree by chance at a loose tolerance (3 of 60
/// such integrands at `10⁻³`, up to 14 times their estimates).
///
/// Returns `None` if a limit is NaN or if the options ask for nothing (no tolerance, or not one
/// evaluation); a budget too small for the first three levels ends the integration there, not
/// converged.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{QuadOptions, tanh_sinh};
///
/// // The logarithmic singularity at 0: ∫₀¹ ln x dx = −1.
/// let r = tanh_sinh(|x: f64| x.ln(), 0.0, 1.0, &QuadOptions::new()).unwrap();
/// assert!(r.converged && (r.value + 1.0).abs() < 1e-14);
///
/// // An algebraic singularity at 0, and an infinite one there: ∫₀¹ x^(−0.9) dx = 10.
/// let r = tanh_sinh(|x: f64| x.powf(-0.9), 0.0, 1.0, &QuadOptions::new()).unwrap();
/// assert!(r.converged && (r.value - 10.0).abs() < 1e-11);
///
/// // exp-sinh on a half-line: ∫₀^∞ e^(−x)/√x dx = √π.
/// let r = tanh_sinh(|x: f64| (-x).exp() / x.sqrt(), 0.0, f64::INFINITY, &QuadOptions::new())
///     .unwrap();
/// assert!(r.converged && (r.value - core::f64::consts::PI.sqrt()).abs() < 1e-13);
/// ```
pub fn tanh_sinh<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    one_dimensional(&mut f, a, b, opts, tanh_sinh::integrate)
}

/// The integral of `f` over the box `[a₀, b₀] × … × [a_{d−1}, b_{d−1}]` by adaptive cubature: in
/// two dimensions and more, Genz and Malik's (1980) embedded rules of degrees 7 and 5 on
/// `2^d + 2d² + 2d + 1` points of a box, with an error estimate from null rules on the same
/// points in the manner of Berntsen, Espelid and Genz's DCUHRE (1991) — ten times `|I₇ − I₅|`
/// where the null rules say the box resolves the integrand, ten times the largest null rule
/// where they say it does not. The box whose error is largest is halved along the axis where the
/// fourth difference of the integrand is largest. Each box is evaluated once, when it is made. In
/// one dimension it is [`quad_g7k15`].
///
/// As in the one-dimensional rules, a result is accepted only with an error estimate below a
/// tenth of it, or at the round-off of the sums: a box whose points have seen only the tail of a
/// peak is not taken for converged.
///
/// The estimate assumes the integrand smooth on the scale of each box, and no rule on finitely
/// many points certifies a feature narrower than their spacing; the result can then be wrong
/// with `converged` true. A kink or a jump inside a box (a maximum of `|x − c|`, an indicator
/// function) can pass between the rule's points and its faces unseen. A narrow peak fools the
/// rule three ways: between all the points of the first boxes it is not there (every value 0,
/// or a far tail that boxes seeing exactly 0 then replace); its tail beyond a face between two
/// boxes is lost where the box across the face has no point near it (a one-dimensional rule
/// samples the very point it bisects at, a box meets its faces only on a few lines); and a box
/// many widths of the peak long, whose points see the peak along one line only, overestimates
/// it, its null rules agreeing on the same few values. On Gaussian peaks well inside the unit
/// square, of 60 random centres at `10⁻⁴`, `10⁻⁶` and `10⁻⁸` (2000 boxes, 400,000 evaluations),
/// 6, 8 and 10 of width 0.01 came out beyond their estimates (up to 3500 times, 26% off, or 16%
/// too high), and 46, 44 and 45 of width 0.003, 38 of them unseen; in the cube, 6, 6 and 9 of
/// width 0.01 (up to 400 times, 70% too low or twice too high) and 59 of width 0.003, 55
/// unseen. There is no option for a least number of evaluations, and none would certify a
/// narrow feature. Where the features are known, integrate a box a few widths wide about each
/// apart from the rest, and add the results: cut along every axis at `c ± 5w`, every one of
/// those peaks came out within its estimate. For a kink or a jump, split the box at it, or use
/// [`monte_carlo`], whose statistical error assumes no smoothness.
///
/// The work of one box grows as `2^d`: 1245 evaluations in 10 dimensions, a million in 20, where
/// [`quasi_monte_carlo`] is the method. In three dimensions and more the default 200 boxes are few:
/// raise `max_subintervals` and `max_evaluations` with the dimension and the tolerance. A
/// coordinate with `aᵢ > bᵢ` changes the sign of the integral, one with `aᵢ = bᵢ` makes it 0.
///
/// Returns `None` if a limit is NaN or infinite, if the options ask for no tolerance, or if
/// `max_evaluations` does not allow one application of the rule.
///
/// # Panics
///
/// Panics if `a` and `b` differ in length or are empty.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{QuadOptions, cubature};
///
/// // ∫₀¹∫₀¹ e^(x + y) dx dy = (e − 1)², to 10⁻¹⁰.
/// let opts = QuadOptions::new().with_epsabs(1e-10).with_epsrel(1e-10);
/// let r = cubature(|v: &[f64]| (v[0] + v[1]).exp(), &[0.0, 0.0], &[1.0, 1.0], &opts).unwrap();
/// let want = (core::f64::consts::E - 1.0).powi(2);
/// assert!(r.converged && (r.value - want).abs() <= r.error && r.error <= 1e-10 * want);
/// ```
#[track_caller]
pub fn cubature<T: Real>(
    f: impl FnMut(&[T]) -> T,
    a: &[T],
    b: &[T],
    opts: &QuadOptions<T>,
) -> Option<QuadResult<T>> {
    cubature::integrate(f, a, b, opts)
}
