//! Monte Carlo and quasi-Monte Carlo integration over a box.
//!
//! * Plain Monte Carlo: the mean of the integrand at points drawn uniformly from the box by
//!   xoshiro256** (Blackman and Vigna, "Scrambled linear pseudorandom number generators", ACM
//!   TOMS 47, 2021), seeded through SplitMix64, times the volume; its error is the standard error
//!   of that mean, from the sample variance.
//! * Quasi-Monte Carlo: the same mean over the points of the Sobol' sequence (Sobol' 1967, in the
//!   Gray code order of Antonov and Saleev, with Joe and Kuo's direction numbers), whose error
//!   falls nearly as `1/N` for a smooth integrand where the plain mean's falls as `1/√N`. The
//!   sequence itself has no error estimate; one comes from randomising it: each of `R`
//!   replications shifts every point by a uniform random vector modulo 1 (Cranley and Patterson
//!   1976), each shifted rule is an unbiased estimate, and the standard error of their mean is
//!   the error estimate.
//!
//! The sums are kept in `f64` whatever the type of the integrand, the plain one by Welford's
//! update of the mean and the variance, the quasi-random one by compensated summation.

use super::sobol_table::{DIMENSIONS, TABLE};
use super::{QuadResult, in_type};
use crate::math::numerics::Real;

/// The most dimensions [`Sobol`] and [`quasi_monte_carlo`](super::quasi_monte_carlo) take: the
/// dimensions of the table of direction numbers.
pub const SOBOL_MAX_DIMENSION: usize = DIMENSIONS;

/// The bits of the points of the sequence: points are multiples of `2^−32`, and there are `2^32`
/// of them.
const BITS: usize = 32;

/// The SplitMix64 generator of Steele, Lea and Flood (2014), which seeds xoshiro256** as its
/// authors recommend.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The xoshiro256** generator, 256 bits of state, period `2^256 − 1`.
struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        // An all-zero state is the one state the generator cannot leave; SplitMix64 cannot give
        // four zeros in a row, but the guard costs nothing.
        let s = if s == [0; 4] { [1, 0, 0, 0] } else { s };
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// A uniform number in the open `(0, 1)`: the 52 leading bits and half a unit more, which is
    /// exact (with 53 bits the half unit would round, and could give 1), from `2^−53` to
    /// `1 − 2^−53`.
    fn open(&mut self) -> f64 {
        ((self.next_u64() >> 12) as f64 + 0.5) * (1.0 / (1u64 << 52) as f64)
    }

    /// A uniform number in `[0, 1)`, on the grid of `2^−53`.
    fn half_open(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// The Sobol' sequence of up to [`SOBOL_MAX_DIMENSION`] dimensions, unscrambled, in the Gray
/// code order: the points of `scipy.stats.qmc.Sobol(d, scramble=False)`, the first one the
/// origin.
///
/// The points are multiples of `2^−32`; the sequence has `2^32` of them.
///
/// # Examples
///
/// ```
/// use a::math::numerics::Sobol;
///
/// let mut s = Sobol::new(2).unwrap();
/// let mut p = [0.0; 2];
/// let mut points = Vec::new();
/// for _ in 0..4 {
///     assert!(s.next_point(&mut p));
///     points.push(p);
/// }
/// assert_eq!(points, [[0.0, 0.0], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]]);
/// ```
#[derive(Clone, Debug)]
pub struct Sobol {
    dimension: usize,
    /// The direction numbers `v_k = m_k · 2^(32−k)`, `BITS` of them a dimension.
    directions: Vec<u32>,
    /// The last point, as integers.
    x: Vec<u32>,
    /// The index of the next point.
    index: u64,
}

impl Sobol {
    /// The sequence of `dimension` dimensions, or `None` if that is 0 or above
    /// [`SOBOL_MAX_DIMENSION`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::{SOBOL_MAX_DIMENSION, Sobol};
    ///
    /// assert!(Sobol::new(SOBOL_MAX_DIMENSION).is_some());
    /// assert!(Sobol::new(SOBOL_MAX_DIMENSION + 1).is_none());
    /// assert!(Sobol::new(0).is_none());
    /// ```
    pub fn new(dimension: usize) -> Option<Self> {
        if dimension == 0 || dimension > SOBOL_MAX_DIMENSION {
            return None;
        }
        let mut directions = vec![0u32; dimension * BITS];
        // The first dimension: m_k = 1, the van der Corput sequence in base 2.
        for k in 1..=BITS {
            directions[k - 1] = 1u32 << (BITS - k);
        }
        let mut at = 0;
        for j in 1..dimension {
            let poly = u32::from(TABLE[at]);
            let s = (u32::BITS - poly.leading_zeros() - 1) as usize;
            let a = (poly >> 1) & ((1u32 << (s - 1)) - 1);
            let m = &TABLE[at + 1..at + 1 + s];
            at += 1 + s;
            let v = &mut directions[j * BITS..(j + 1) * BITS];
            // Joe and Kuo's recurrence for the direction numbers, scaled by 2^32: the first s
            // from the table, then v_k = v_{k−s} ⊕ (v_{k−s} >> s) ⊕ Σ_{i<s} a_i v_{k−i}, a_i the
            // coefficient of the middle terms of the polynomial, from the highest.
            for k in 1..=s.min(BITS) {
                v[k - 1] = u32::from(m[k - 1]) << (BITS - k);
            }
            for k in s + 1..=BITS {
                let mut x = v[k - s - 1] ^ (v[k - s - 1] >> s);
                for i in 1..s {
                    if (a >> (s - 1 - i)) & 1 == 1 {
                        x ^= v[k - i - 1];
                    }
                }
                v[k - 1] = x;
            }
        }
        Some(Self {
            dimension,
            directions,
            x: vec![0; dimension],
            index: 0,
        })
    }

    /// The number of dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Sobol;
    ///
    /// assert_eq!(Sobol::new(5).unwrap().dimension(), 5);
    /// ```
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Writes the next point into `point` and returns true, or returns false once the `2^32`
    /// points of the sequence are spent.
    ///
    /// # Panics
    ///
    /// Panics if `point` is not of the sequence's dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::Sobol;
    ///
    /// let mut s = Sobol::new(3).unwrap();
    /// let mut p = [1.0; 3];
    /// assert!(s.next_point(&mut p));
    /// assert_eq!(p, [0.0; 3]);
    /// ```
    #[track_caller]
    pub fn next_point(&mut self, point: &mut [f64]) -> bool {
        assert_eq!(
            point.len(),
            self.dimension,
            "the point is not of the dimension of the sequence"
        );
        if !self.advance() {
            return false;
        }
        self.write(point);
        true
    }

    /// Moves to the next point; false once the sequence is spent.
    fn advance(&mut self) -> bool {
        if self.index >= 1u64 << BITS {
            return false;
        }
        if self.index > 0 {
            // Gray code: the next point differs from this one in the direction number of the
            // lowest zero bit of the index of this one.
            let c = (self.index - 1).trailing_ones() as usize;
            for (j, x) in self.x.iter_mut().enumerate() {
                *x ^= self.directions[j * BITS + c];
            }
        }
        self.index += 1;
        true
    }

    /// The current point, in `[0, 1)`.
    fn write(&self, point: &mut [f64]) {
        for (p, &x) in point.iter_mut().zip(&self.x) {
            *p = f64::from(x) / 4_294_967_296.0;
        }
    }
}

/// The options of [`monte_carlo_with`] and [`quasi_monte_carlo_with`]: how many evaluations of the
/// integrand, the seed of the generator, and the number of random shifts of the quasi-random
/// rule.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{MonteCarloOptions, monte_carlo_with};
///
/// let opts = MonteCarloOptions::new().with_samples(1 << 12).with_seed(7);
/// let r = monte_carlo_with(|x: &[f64]| x[0], &[0.0], &[1.0], &opts).unwrap();
/// assert_eq!(r.evaluations, 1 << 12);
/// assert!((r.value - 0.5).abs() < 5.0 * r.error);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MonteCarloOptions {
    /// The number of evaluations of the integrand: for the quasi-random rule, over all the
    /// replications together.
    pub samples: usize,
    /// The seed of the pseudo-random generator, which draws the points of the plain rule and the
    /// shifts of the quasi-random one: the same seed gives the same result.
    pub seed: u64,
    /// The number of randomly shifted copies of the Sobol' points the quasi-random rule averages,
    /// at least 2; its error estimate has `replications − 1` degrees of freedom.
    pub replications: usize,
}

impl Default for MonteCarloOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl MonteCarloOptions {
    /// The default options: `2^16` samples, the seed 1, and 8 replications.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numerics::MonteCarloOptions;
    ///
    /// assert_eq!(MonteCarloOptions::new().samples, 65_536);
    /// ```
    pub fn new() -> Self {
        Self {
            samples: 1 << 16,
            seed: 1,
            replications: 8,
        }
    }

    /// Sets [`samples`](Self::samples).
    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }

    /// Sets [`seed`](Self::seed).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets [`replications`](Self::replications).
    pub fn with_replications(mut self, replications: usize) -> Self {
        self.replications = replications;
        self
    }
}

/// The box, as its lower corner, its sides and the sign of the orientation, or what to answer
/// without a single evaluation: `None` for a limit that is NaN or infinite, the integral 0 for a
/// side of length 0.
enum Box_ {
    Answer(Option<(f64, f64)>),
    Sides {
        lower: Vec<f64>,
        side: Vec<f64>,
        volume: f64,
        sign: f64,
    },
}

#[track_caller]
fn the_box<T: Real>(a: &[T], b: &[T]) -> Box_ {
    assert_eq!(
        a.len(),
        b.len(),
        "the two corners of the box have different dimensions"
    );
    assert!(!a.is_empty(), "the box has no dimensions");
    let mut lower = Vec::with_capacity(a.len());
    let mut side = Vec::with_capacity(a.len());
    let mut sign = 1.0;
    for (&lo, &hi) in a.iter().zip(b) {
        let (lo, hi) = (lo.to_f64(), hi.to_f64());
        if !lo.is_finite() || !hi.is_finite() {
            return Box_::Answer(None);
        }
        if lo > hi {
            sign = -sign;
        }
        lower.push(lo.min(hi));
        side.push((hi - lo).abs());
    }
    if side.contains(&0.0) {
        return Box_::Answer(Some((0.0, 0.0)));
    }
    let volume = side.iter().product();
    Box_::Sides {
        lower,
        side,
        volume,
        sign,
    }
}

/// The integral of `f` over the box `[a₀, b₀] × … × [a_{d−1}, b_{d−1}]` by plain Monte Carlo with
/// `samples` points and the default seed: [`monte_carlo_with`].
///
/// # Panics
///
/// Panics if `a` and `b` differ in length or are empty.
///
/// # Examples
///
/// ```
/// use a::math::numerics::monte_carlo;
///
/// // The volume of the unit ball in 5 dimensions, 8π²/15, from the cube around it.
/// let r = monte_carlo(
///     |x: &[f64]| if x.iter().map(|t| t * t).sum::<f64>() <= 1.0 { 1.0 } else { 0.0 },
///     &[-1.0; 5],
///     &[1.0; 5],
///     1 << 16,
/// )
/// .unwrap();
/// let want = 8.0 * core::f64::consts::PI.powi(2) / 15.0;
/// assert!((r.value - want).abs() < 4.0 * r.error);
/// ```
#[track_caller]
pub fn monte_carlo<T: Real>(
    f: impl FnMut(&[T]) -> T,
    a: &[T],
    b: &[T],
    samples: usize,
) -> Option<QuadResult<T>> {
    monte_carlo_with(f, a, b, &MonteCarloOptions::new().with_samples(samples))
}

/// The integral of `f` over a box by plain Monte Carlo: `V · (1/N) Σᵢ f(xᵢ)` at `N = samples`
/// points drawn uniformly from the open box by xoshiro256** from `seed`, `V` the volume. A
/// coordinate the integrand's type rounds onto a face of the box is drawn again, up to eight
/// times, so that no point lies on a face unless a side holds no number of the type inside it
/// (an `f32` side one ulp long).
///
/// The error is the standard error of the estimate, `V s/√N` with `s²` the sample variance: one
/// standard deviation, which the true error exceeds about a third of the time, twice it about one
/// time in twenty, three times it about one time in 400, for a sample large enough that the mean
/// is nearly normal. `converged` says only that the estimate and its error are finite numbers
/// in the integrand's type: a value of the integrand that is not ends the sampling there, and an
/// estimate beyond the range of the type (an `f32` integrand over a large box) is not converged.
///
/// Returns `None` if a limit is NaN or infinite, or if `samples` is below 2 (there is no variance
/// to estimate the error from). A side of length 0 gives the integral 0, a reversed side the
/// negated one.
///
/// # Panics
///
/// Panics if `a` and `b` differ in length or are empty.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{MonteCarloOptions, monte_carlo_with};
///
/// // ∫ over the unit cube of xyz = 1/8, and the same seed gives the same answer.
/// let opts = MonteCarloOptions::new().with_samples(1 << 16).with_seed(42);
/// let f = |x: &[f64]| x[0] * x[1] * x[2];
/// let r = monte_carlo_with(f, &[0.0; 3], &[1.0; 3], &opts).unwrap();
/// assert!((r.value - 0.125).abs() < 4.0 * r.error && r.error < 1e-3);
/// assert_eq!(r, monte_carlo_with(f, &[0.0; 3], &[1.0; 3], &opts).unwrap());
/// ```
#[track_caller]
pub fn monte_carlo_with<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    a: &[T],
    b: &[T],
    opts: &MonteCarloOptions,
) -> Option<QuadResult<T>> {
    let (lower, side, volume, sign) = match the_box(a, b) {
        Box_::Answer(None) => return None,
        Box_::Answer(Some(_)) => return Some(zero()),
        Box_::Sides {
            lower,
            side,
            volume,
            sign,
        } => (lower, side, volume, sign),
    };
    let n = opts.samples;
    if n < 2 {
        return None;
    }
    let mut rng = Xoshiro256::new(opts.seed);
    let mut x = vec![T::ZERO; lower.len()];
    // The faces as the integrand's type has them: a coordinate that the type rounds onto one is
    // drawn again (a few times at most, should the side hold no number of the type inside it),
    // so that an integrand singular on a face is never evaluated there.
    let faces: Vec<(T, T)> = lower
        .iter()
        .zip(&side)
        .map(|(&lo, &s)| (T::from_f64(lo), T::from_f64(lo + s)))
        .collect();
    // Welford's update of the mean and of the sum of squared deviations, in f64.
    let (mut mean, mut m2) = (0.0f64, 0.0f64);
    for i in 0..n {
        for (((xj, &lo), &s), &(from, to)) in x.iter_mut().zip(&lower).zip(&side).zip(&faces) {
            let mut v = T::from_f64(lo + s * rng.open());
            for _ in 0..8 {
                if v > from && v < to {
                    break;
                }
                v = T::from_f64(lo + s * rng.open());
            }
            *xj = v;
        }
        let v = f(&x).to_f64();
        if !v.is_finite() {
            return Some(QuadResult {
                value: T::from_f64(sign * volume * v),
                error: T::INFINITY,
                evaluations: i + 1,
                converged: false,
            });
        }
        let delta = v - mean;
        mean += delta / (i + 1) as f64;
        m2 += delta * (v - mean);
    }
    let variance = m2 / (n - 1) as f64;
    let value = sign * volume * mean;
    let error = volume * (variance / n as f64).sqrt();
    // Converged only if the estimate and its error are finite in the integrand's type.
    Some(in_type(value, error, n, true))
}

/// The integral 0 of a box with a side of length 0.
fn zero<T: Real>() -> QuadResult<T> {
    QuadResult {
        value: T::ZERO,
        error: T::ZERO,
        evaluations: 0,
        converged: true,
    }
}

/// The integral of `f` over the box `[a₀, b₀] × … × [a_{d−1}, b_{d−1}]` by randomly shifted
/// quasi-Monte Carlo with `samples` evaluations and the default options:
/// [`quasi_monte_carlo_with`].
///
/// # Panics
///
/// Panics if `a` and `b` differ in length or are empty.
///
/// # Examples
///
/// ```
/// use a::math::numerics::quasi_monte_carlo;
///
/// // ∫ over the unit cube in 10 dimensions of Π (1 + (xᵢ − ½)/2)... = 1, nearly to the
/// // accuracy of 1/N.
/// let f = |x: &[f64]| x.iter().map(|t| 1.0 + 0.5 * (t - 0.5)).product::<f64>();
/// let r = quasi_monte_carlo(f, &[0.0; 10], &[1.0; 10], 1 << 16).unwrap();
/// assert!((r.value - 1.0).abs() < 1e-4 && (r.value - 1.0).abs() < 5.0 * r.error);
/// ```
#[track_caller]
pub fn quasi_monte_carlo<T: Real>(
    f: impl FnMut(&[T]) -> T,
    a: &[T],
    b: &[T],
    samples: usize,
) -> Option<QuadResult<T>> {
    quasi_monte_carlo_with(f, a, b, &MonteCarloOptions::new().with_samples(samples))
}

/// The integral of `f` over a box by quasi-Monte Carlo on the Sobol' sequence, randomised by
/// shifts: each of the `R = replications` estimates is the mean of `f` at the first
/// `N = samples / R` points of the sequence, every point shifted by the same uniform random
/// vector modulo 1, times the volume; the result is the mean of the `R` estimates, and its error
/// the standard error of that mean, `s_R/√R`.
///
/// Each shifted estimate is unbiased whatever the integrand, so the error estimate is a
/// statistical one with `R − 1` degrees of freedom: one standard error, which the true error
/// exceeds about 35% of the time for the default `R = 8` (Student's t with 7 degrees of freedom),
/// twice it about 9% and three times it about 2% of the time — where the `R` estimates are near
/// normal. For an integrand singular in the box they are not: on `x^(−1/3)` over `[0, 1]` the
/// true error exceeded twice the estimate 14–15% of the time and three times it 7% (1000 seeds,
/// `2^10` and `2^14` samples). For a smooth integrand the error of `N` points falls nearly as
/// `1/N`, where plain Monte Carlo's falls as `1/√N`; `N` a power of two keeps the points
/// balanced. `converged` says only that the estimate and its error are finite numbers in the
/// integrand's type.
///
/// The shifted points are not kept off the faces of the box: a coordinate can be exactly `aᵢ`,
/// or, rounded to the integrand's type, exactly `bᵢ` (in `f64` about one coordinate in `2^53`,
/// in `f32` about one in `2^24`), and an integrand infinite there ends the integration, reported
/// as not converged.
///
/// Returns `None` if a limit is NaN or infinite, if the box has more than
/// [`SOBOL_MAX_DIMENSION`] dimensions, if `replications` is below 2, or if `samples` is below
/// `replications` or above `2^32 · replications`. A side of length 0 gives the integral 0, a
/// reversed side the negated one.
///
/// # Panics
///
/// Panics if `a` and `b` differ in length or are empty.
///
/// # Examples
///
/// ```
/// use a::math::numerics::{MonteCarloOptions, quasi_monte_carlo_with};
///
/// // ∫ over the unit cube of xyz = 1/8: the Sobol' points come within 10⁻⁴ where plain Monte
/// // Carlo of as many points is within 10⁻³.
/// let opts = MonteCarloOptions::new().with_samples(1 << 16);
/// let r = quasi_monte_carlo_with(|x: &[f64]| x[0] * x[1] * x[2], &[0.0; 3], &[1.0; 3], &opts)
///     .unwrap();
/// assert!((r.value - 0.125).abs() < 1e-4 && r.error < 1e-4);
/// ```
#[track_caller]
pub fn quasi_monte_carlo_with<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    a: &[T],
    b: &[T],
    opts: &MonteCarloOptions,
) -> Option<QuadResult<T>> {
    let (lower, side, volume, sign) = match the_box(a, b) {
        Box_::Answer(None) => return None,
        Box_::Answer(Some(_)) => return Some(zero()),
        Box_::Sides {
            lower,
            side,
            volume,
            sign,
        } => (lower, side, volume, sign),
    };
    let d = lower.len();
    let r = opts.replications;
    if d > SOBOL_MAX_DIMENSION || r < 2 || opts.samples < r {
        return None;
    }
    let n = opts.samples / r;
    if n as u64 > 1u64 << BITS {
        return None;
    }
    let mut rng = Xoshiro256::new(opts.seed);
    // Grown as the replications are made, not reserved for a count that may be vast.
    let mut estimates = Vec::with_capacity(r.min(64));
    let mut u = vec![0.0f64; d];
    let mut x = vec![T::ZERO; d];
    let mut evaluations = 0;
    for _ in 0..r {
        let shift: Vec<f64> = (0..d).map(|_| rng.half_open()).collect();
        let mut sobol = Sobol::new(d).expect("the dimension was checked");
        // Neumaier's compensated sum of the values.
        let (mut sum, mut c) = (0.0f64, 0.0f64);
        for _ in 0..n {
            sobol.advance();
            sobol.write(&mut u);
            for j in 0..d {
                let mut y = u[j] + shift[j];
                if y >= 1.0 {
                    y -= 1.0;
                }
                x[j] = T::from_f64(lower[j] + side[j] * y);
            }
            let v = f(&x).to_f64();
            evaluations += 1;
            if !v.is_finite() {
                return Some(QuadResult {
                    value: T::from_f64(sign * volume * v),
                    error: T::INFINITY,
                    evaluations,
                    converged: false,
                });
            }
            let t = sum + v;
            c += if sum.abs() >= v.abs() {
                (sum - t) + v
            } else {
                (v - t) + sum
            };
            sum = t;
        }
        estimates.push(volume * (sum + c) / n as f64);
    }
    let rf = r as f64;
    let mean = estimates.iter().sum::<f64>() / rf;
    let spread = estimates
        .iter()
        .map(|e| (e - mean) * (e - mean))
        .sum::<f64>();
    let error = (spread / (rf * (rf - 1.0))).sqrt();
    Some(in_type(sign * mean, error, evaluations, true))
}
