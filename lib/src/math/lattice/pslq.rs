//! Integer relations by the PSLQ algorithm of Ferguson and Bailey, as Ferguson, Bailey and Arno
//! analyse it ("Analysis of PSLQ, an integer relation finding algorithm", Math. Comp. 68, 1999):
//! with the integer matrices `A` and `B = A⁻¹` exact, the vector `y` exact, the
//! lower-trapezoidal `H` in fixed point, and every answer checked or proved in exact arithmetic.
//!
//! The numbers are first made integers: each `xᵢ = mᵢ·2^qᵢ` is `Xᵢ = mᵢ·2^(qᵢ − L)` with
//! `L = min qᵢ`, so that `y = X·B` is held exactly and a relation is detected by an exact test on
//! its own entry of `y`, the test that defines a relation (see [`pslq`]). The rounding is all in
//! `H`, which only steers the search; the bound reported when no relation turns up is proved
//! from the exact `A` by [`proved_bound`](super::pslq_bound::proved_bound), not read off the
//! rounded `H`.

use super::pslq_bound::proved_bound;
use crate::math::float::BigFloat;
use crate::math::numbers::Number;

/// `γ` by default: a little above `√(4/3) ≈ 1.1547`, above which the analysis of Ferguson, Bailey
/// and Arno bounds the number of iterations; larger values let PSLQ settle for longer relations
/// (at 1000 bits, `γ` = 1.5 and 2 found relations to the precision among the powers of a number of
/// degree 35, where 1.1548 to 1.3 found its minimal polynomial).
const DEFAULT_GAMMA: f64 = 1.16;

/// The bits that `H` carries beyond the span of the numbers, before the logarithm of `n`.
const GUARD_BITS: u64 = 64;

/// How many bits of `H`'s largest diagonal entry must stand above the size of `A` for `H` to be
/// trusted: below, the rounding errors that the reductions multiply by the entries of `A` reach
/// the diagonal and the precision is exhausted.
const TRUSTED_BITS: u64 = 20;

/// The parameters of [`pslq`].
///
/// # Examples
///
/// ```
/// use a::math::lattice::PslqOptions;
/// use a::math::numbers::Number;
///
/// let o = PslqOptions::new()
///     .with_max_norm(Some(Number::from(1000)))
///     .with_max_iter(Some(500))
///     .with_gamma(1.5)
///     .with_error_bits(4);
/// assert_eq!(o.max_norm, Some(Number::from(1000)));
/// assert_eq!((o.max_iter, o.gamma, o.error_bits), (Some(500), 1.5, 4));
/// assert_eq!(PslqOptions::new().error_bits, 8);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct PslqOptions {
    /// The largest norm of a relation looked for: the search stops, with no relation, once every
    /// relation must be longer; `None` for no limit but the precision. `None` by default.
    ///
    /// A relation that turns up first is returned whatever its norm, and with no limit a vector
    /// that has no relation gets one at the end of its precision: every `n` numbers of `p` bits
    /// have relations to that precision of norm about `2^(p/n)`.
    pub max_norm: Option<Number>,
    /// The most iterations; `None` for a bound from the precision, the number of iterations
    /// within which Ferguson, Bailey and Arno prove that PSLQ finds a relation of norm up to
    /// `2^w` for the working precision `w`. `None` by default.
    pub max_iter: Option<usize>,
    /// `γ`, which weighs the choice of the rows to exchange in each iteration: it must be finite
    /// and exceed `√(4/3) ≈ 1.1547`; `1.16` by default.
    pub gamma: f64,
    /// How many of the last bits of each number may be wrong: each `xᵢ` stands for some real
    /// within `2^error_bits` units in its last place, and a relation is one of such reals. `8` by
    /// default.
    pub error_bits: u32,
}

impl Default for PslqOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl PslqOptions {
    /// No limit but the precision, `γ = 1.16`, and the last 8 bits of each number taken as
    /// possibly wrong.
    #[must_use]
    pub fn new() -> Self {
        PslqOptions {
            max_norm: None,
            max_iter: None,
            gamma: DEFAULT_GAMMA,
            error_bits: 8,
        }
    }

    /// Sets [`max_norm`](Self::max_norm).
    #[must_use]
    pub fn with_max_norm(mut self, max_norm: Option<Number>) -> Self {
        self.max_norm = max_norm;
        self
    }

    /// Sets [`max_iter`](Self::max_iter).
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: Option<usize>) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets [`gamma`](Self::gamma).
    #[must_use]
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Sets [`error_bits`](Self::error_bits).
    #[must_use]
    pub fn with_error_bits(mut self, error_bits: u32) -> Self {
        self.error_bits = error_bits;
        self
    }
}

/// What [`pslq`] found.
///
/// Every bound is proved in exact arithmetic: no integer vector of Euclidean norm below it is a
/// relation of any reals within the errors of the numbers ([`PslqOptions::error_bits`]). The
/// variants other than [`Relation`](Self::Relation) differ in why the search stopped.
///
/// # Examples
///
/// ```
/// use a::math::float::BigFloat;
/// use a::math::lattice::{Pslq, PslqOptions, pslq};
/// use a::math::numbers::Number;
///
/// let r = pslq(&[BigFloat::from(6), BigFloat::from(-4)], &PslqOptions::new());
/// assert_eq!(r, Pslq::Relation(vec![Number::from(2), Number::from(3)]));
/// assert_eq!(r.relation().unwrap().len(), 2);
/// assert_eq!(r.bound(), None);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Pslq {
    /// A relation `m` with `Σ mᵢ·xᵢ = 0` to the precision of the numbers: exactly,
    /// `|Σ mᵢ·xᵢ| ≤ Σ |mᵢ|·2^error_bits·ulp(xᵢ)`. It is primitive (its entries have no common
    /// divisor) and its first nonzero entry is positive.
    Relation(Vec<Number>),
    /// No relation of norm below the bound, which every relation must reach: the bound reached
    /// [`max_norm`](PslqOptions::max_norm), or, with no `max_norm`, the precision ran out.
    NoneBelow(BigFloat),
    /// The precision ran out before the bound reached [`max_norm`](PslqOptions::max_norm): no
    /// relation of norm below the bound, but one between it and `max_norm` is not ruled out;
    /// more precise numbers may settle it.
    PrecisionExhausted(BigFloat),
    /// The iterations ([`max_iter`](PslqOptions::max_iter)) ran out first: no relation of norm
    /// below the bound; more iterations may find one or raise it.
    IterationLimit(BigFloat),
}

impl Pslq {
    /// The relation, if one was found.
    #[must_use]
    pub fn relation(&self) -> Option<&[Number]> {
        match self {
            Pslq::Relation(m) => Some(m),
            _ => None,
        }
    }

    /// The proved lower bound on the norm of every relation, if no relation was found.
    #[must_use]
    pub fn bound(&self) -> Option<&BigFloat> {
        match self {
            Pslq::Relation(_) => None,
            Pslq::NoneBelow(b) | Pslq::PrecisionExhausted(b) | Pslq::IterationLimit(b) => Some(b),
        }
    }
}

/// An integer relation among the numbers `x`: integers `mᵢ`, not all zero, with `Σ mᵢ·xᵢ = 0` to
/// their precision; or a lower bound on the norm of every relation.
///
/// The precision is each number's own: `xᵢ` of precision `pᵢ` stands for a real within
/// `2^error_bits` units in its last place, `δᵢ = 2^error_bits·2^(eᵢ − pᵢ)` for
/// `2^(eᵢ−1) ≤ |xᵢ| < 2^eᵢ`, and `m` is a relation when some such reals satisfy it, which is
/// when `|Σ mᵢ·xᵢ| ≤ Σ |mᵢ|·δᵢ`, tested exactly. A zero is exact (its relation is its unit
/// vector). An integer is no more exact than its precision: `BigFloat::from(1)` has 32 bits, and
/// beside numbers of 1000 bits it is best given as `BigFloat::from(1).with_prec(1000)`.
///
/// Every `n` numbers of `p` bits have relations of norm about `2^(p/n)` to that precision, so a
/// relation is evidence of a true one only when it is much shorter; with
/// [`max_norm`](PslqOptions::max_norm) well below that, numbers with no relation end in
/// [`Pslq::NoneBelow`], and with no limit in such a relation to the precision. A relation that
/// turns up is returned whatever its norm: the shortest of the columns of `B` that pass the test
/// when the first one does, a relation to the precision and not always the shortest relation.
///
/// The search is PSLQ (Ferguson and Bailey; as Ferguson, Bailey and Arno analyse it, Math. Comp.
/// 68, 1999) with the parameter [`gamma`](PslqOptions::gamma), on the numbers made integers
/// exactly (`Xᵢ = xᵢ/2^L`, `2^L` the lowest place of any of them), so that `y = X·B` and the
/// integer matrices `A` and `B = A⁻¹` are exact and a relation is found by the exact test above;
/// only `H` is rounded, in fixed point of the span of the bits of the numbers, 64 bits more and
/// two for each bit of `n`.
/// A bound is proved from the exact `A`: for any real `c`, `aⱼ = c₀·X + Σ_{i<j} cᵢ·aᵢ + rⱼ`
/// gives `|m| ≥ 1/(|c₀|·|δ| + |rⱼ|)` for every relation `m` with `⟨m, aⱼ⟩` the first nonzero
/// `⟨m, aᵢ⟩`, so the coefficients of a Gram–Schmidt orthogonalisation of `X, a₀, …`, taken as the
/// dyadic rationals they are, give the bound `1/max_j(|c₀|·|δ| + |rⱼ|)` in integers; it is
/// Ferguson, Bailey and Arno's `1/max |Hⱼⱼ|` but for the errors of the numbers. An iteration
/// costs `O(n²)` operations and more for each nonzero multiplier of the reduction, each on
/// integers of about the working precision; a proof of the bound `O(n³)`.
///
/// # Examples
///
/// ```
/// use a::math::float::BigFloat;
/// use a::math::lattice::{Pslq, PslqOptions, pslq};
/// use a::math::numbers::Number;
///
/// // 1, √2 + √3 and its powers to 200 bits: the minimal polynomial x⁴ − 10x² + 1.
/// let p = 200;
/// let a = BigFloat::from(2).with_prec(p).sqrt() + BigFloat::from(3).with_prec(p).sqrt();
/// let x: Vec<BigFloat> = (0..5).map(|k| a.powi(k)).collect();
/// let m = [1, 0, -10, 0, 1].map(Number::from).to_vec();
/// assert_eq!(pslq(&x, &PslqOptions::new()), Pslq::Relation(m));
///
/// // π and e have no relation of norm below 10⁶ (at 200 bits).
/// let x = [BigFloat::pi(p), BigFloat::e(p)];
/// let opts = PslqOptions::new().with_max_norm(Some(Number::from(1_000_000)));
/// match pslq(&x, &opts) {
///     Pslq::NoneBelow(bound) => assert!(bound >= 1_000_000),
///     other => panic!("{other:?}"),
/// }
/// ```
///
/// # Panics
///
/// Panics if `x` has fewer than two numbers or one of them is not finite; if
/// [`gamma`](PslqOptions::gamma) is not finite or not above `√(4/3)`, or
/// [`max_norm`](PslqOptions::max_norm) is negative; or if the bits of the numbers span more than
/// `2^40` bits, from the top bit of the largest to the lowest last place of any of them (the
/// smallest ulp), which is what the integers `Xᵢ` must hold.
#[track_caller]
#[must_use]
pub fn pslq(x: &[BigFloat], opts: &PslqOptions) -> Pslq {
    let n = x.len();
    assert!(n >= 2, "PSLQ needs at least two numbers, not {n}");
    for (i, v) in x.iter().enumerate() {
        assert!(
            v.is_finite(),
            "PSLQ needs finite numbers, and number {i} is {v:?}"
        );
    }
    check_gamma(opts.gamma);
    if let Some(m) = &opts.max_norm {
        assert!(!m.is_negative(), "the largest norm {m} is negative");
    }
    // A zero is a relation of norm 1 by itself, the shortest there is.
    if let Some(i) = x.iter().position(BigFloat::is_zero) {
        return Pslq::Relation(unit(n, i));
    }
    // So is a number of no more bits than may be wrong, |xᵢ| < 2^pᵢ·ulp ≤ δᵢ (and the test
    // below would hold numbers of as many bits as error_bits).
    if let Some(i) = x
        .iter()
        .position(|v| v.prec() <= u64::from(opts.error_bits))
    {
        return Pslq::Relation(unit(n, i));
    }
    let input = Input::new(x, opts.error_bits);
    if let Some(i) = (0..n).find(|&i| input.holds(&unit(n, i))) {
        return Pslq::Relation(unit(n, i));
    }
    let mut search = Search::new(input, opts);
    search.run()
}

/// [`pslq`] of `f64`s, each taken at its 53 bits.
///
/// # Examples
///
/// ```
/// use a::math::lattice::{Pslq, PslqOptions, pslq_f64};
/// use a::math::numbers::Number;
///
/// // Machin's formula, π/4 = 4·atan(1/5) − atan(1/239), to the precision of f64.
/// let x = [
///     core::f64::consts::FRAC_PI_4,
///     (1.0f64 / 5.0).atan(),
///     (1.0f64 / 239.0).atan(),
/// ];
/// let m = [1, -4, 1].map(Number::from).to_vec();
/// assert_eq!(pslq_f64(&x, &PslqOptions::new()), Pslq::Relation(m));
/// ```
///
/// # Panics
///
/// As [`pslq`].
#[track_caller]
#[must_use]
pub fn pslq_f64(x: &[f64], opts: &PslqOptions) -> Pslq {
    let x: Vec<BigFloat> = x.iter().map(|&v| BigFloat::from(v)).collect();
    pslq(&x, opts)
}

/// Panics unless `γ` is finite and `γ² > 4/3`, decided exactly on the dyadic rational the `f64`
/// is.
#[track_caller]
fn check_gamma(gamma: f64) {
    assert!(gamma.is_finite(), "γ = {gamma} is not finite");
    let ok = gamma > 1.0 && {
        let g = BigFloat::from(gamma);
        // 3γ² − 4, exact at 112 bits (γ has 53).
        let g2 = g.mul_round(&g, 112, crate::math::float::Round::Nearest).0;
        (g2 * 3u32) > 4u32
    };
    assert!(ok, "γ = {gamma} is not above √(4/3)");
}

/// The unit vector `eᵢ` of length `n`.
fn unit(n: usize, i: usize) -> Vec<Number> {
    (0..n).map(|k| Number::from(u8::from(k == i))).collect()
}

/// The numbers made integers, and their errors.
pub(super) struct Input {
    /// `Xᵢ = xᵢ / 2^L`, an integer, for the lowest place `L` of all the numbers.
    pub(super) x: Vec<Number>,
    /// `log₂` of the error `δᵢ / 2^L` of each `Xᵢ`, a power of two.
    pub(super) err: Vec<u64>,
    /// The bits from `2^L` to the top bit of the largest `Xᵢ`.
    pub(super) span: u64,
}

impl Input {
    /// The integers of the nonzero finite numbers `x`.
    #[track_caller]
    fn new(x: &[BigFloat], error_bits: u32) -> Input {
        let parts: Vec<(Number, i64)> = x
            .iter()
            .map(|v| {
                v.mantissa_exponent()
                    .expect("a finite number that is not zero")
            })
            .collect();
        let low = parts.iter().map(|(_, q)| *q).min().expect("two numbers");
        let top = parts
            .iter()
            .map(|(m, q)| i128::from(*q) + i128::from(m.bits()))
            .max()
            .expect("two numbers");
        let span = top - i128::from(low);
        assert!(
            span <= i128::from(BigFloat::MAX_PREC),
            "the bits of the numbers span {span} bits, more than 2^40"
        );
        let span = span as u64;
        let mut xs = Vec::with_capacity(x.len());
        let mut err = Vec::with_capacity(x.len());
        for (m, q) in parts {
            // Below `span ≤ 2^40`, so the shift fits.
            let shift = (i128::from(q) - i128::from(low)) as u64;
            xs.push(&m << shift as usize);
            err.push(shift + u64::from(error_bits));
        }
        Input { x: xs, err, span }
    }

    /// `Σ |mᵢ|·δᵢ / 2^L`, the most that `Σ mᵢ·Xᵢ` may be for `m` to be a relation.
    pub(super) fn tolerance<'a>(&self, m: impl Iterator<Item = &'a Number>) -> Number {
        let mut t = Number::from(0);
        for (mi, &e) in m.zip(&self.err) {
            if !mi.is_zero() {
                t += mi.abs() << e as usize;
            }
        }
        t
    }

    /// Whether `m` is a relation: `|Σ mᵢ·Xᵢ| ≤ Σ |mᵢ|·δᵢ / 2^L`, exactly.
    pub(super) fn holds(&self, m: &[Number]) -> bool {
        let mut r = Number::from(0);
        for (mi, xi) in m.iter().zip(&self.x) {
            if !mi.is_zero() {
                r += mi * xi;
            }
        }
        r.abs() <= self.tolerance(m.iter())
    }
}

/// The state of a search: the exact `y`, `A` and `B`, and `H` in fixed point.
struct Search {
    input: Input,
    n: usize,
    /// The fraction bits of `H`: an entry `h` stands for `h / 2^frac`.
    frac: usize,
    /// `y = X·B`, exactly.
    y: Vec<Number>,
    /// `H`, `n × (n − 1)`, by rows, lower trapezoidal.
    h: Vec<Number>,
    /// `A`, `n × n` by rows, exact and unimodular.
    a: Vec<Number>,
    /// `B = A⁻¹`, `n × n` by rows, exact.
    b: Vec<Number>,
    gamma_log2: f64,
    max_norm: Option<Number>,
    max_iter: usize,
    /// The best bound proved so far.
    best: BigFloat,
    /// `log₂` of the floating bound `1/max |Hⱼⱼ|` at which a bound was last proved.
    proved_at: f64,
}

/// Why a search stopped without a relation.
enum Stop {
    MaxNorm,
    Precision,
    Iterations,
}

impl Search {
    fn new(input: Input, opts: &PslqOptions) -> Search {
        let n = input.x.len();
        let frac =
            (input.span + GUARD_BITS + 2 * u64::from(usize::BITS - n.leading_zeros())) as usize;
        let gamma_log2 = opts.gamma.log2();
        let max_iter = opts
            .max_iter
            .unwrap_or_else(|| iteration_bound(n, frac as f64, opts.gamma));
        let identity = |i: usize, j: usize| Number::from(u8::from(i == j));
        let mut s = Search {
            n,
            frac,
            y: input.x.clone(),
            h: Vec::new(),
            a: (0..n * n).map(|k| identity(k / n, k % n)).collect(),
            b: (0..n * n).map(|k| identity(k / n, k % n)).collect(),
            input,
            gamma_log2,
            max_norm: opts.max_norm.clone(),
            max_iter,
            best: BigFloat::from(1).with_prec(64),
            proved_at: f64::NEG_INFINITY,
        };
        s.h = s.initial_h();
        s
    }

    /// `H` of the numbers: `Hⱼⱼ = sⱼ₊₁/sⱼ` and `Hᵢⱼ = −XᵢXⱼ/(sⱼsⱼ₊₁)` below the diagonal, for
    /// `sⱼ² = Σ_{k≥j} Xₖ²`; its columns are an orthonormal basis of the plane orthogonal to `X`.
    fn initial_h(&self) -> Vec<Number> {
        let (n, f) = (self.n, self.frac);
        let x = &self.input.x;
        let mut tail = vec![Number::from(0); n + 1];
        for k in (0..n).rev() {
            tail[k] = &tail[k + 1] + &(&x[k] * &x[k]);
        }
        let mut h = vec![Number::from(0); n * (n - 1)];
        for j in 0..n - 1 {
            // Every Xₖ is nonzero, so every tail sum is positive.
            h[j * (n - 1) + j] = ((&tail[j + 1] << (2 * f)) / &tail[j]).isqrt();
            let d = (&tail[j] * &tail[j + 1]).isqrt();
            for i in j + 1..n {
                let p = &x[i] * &x[j];
                h[i * (n - 1) + j] = -round_div(&(p << f), &d);
            }
        }
        h
    }

    fn hh(&self, i: usize, j: usize) -> &Number {
        &self.h[i * (self.n - 1) + j]
    }

    /// Reduces row `i` of `H` by row `j`, the column `j` of `B` by its column `i`: subtracts
    /// `t = nint(Hᵢⱼ/Hⱼⱼ)` times. `None` when `Hⱼⱼ` is zero, which the precision has run out.
    fn reduce(&mut self, i: usize, j: usize) -> Option<()> {
        let n = self.n;
        let w = n - 1;
        let (hij, hjj) = (&self.h[i * w + j], &self.h[j * w + j]);
        if hjj.is_zero() {
            return None;
        }
        // |Hᵢⱼ/Hⱼⱼ| < 2^(bits(Hᵢⱼ) − bits(Hⱼⱼ) + 1), which is below 1/2 and makes t zero when
        // Hᵢⱼ has two bits fewer.
        if hij.bits() + 1 < hjj.bits() {
            return Some(());
        }
        let t = round_div(hij, hjj);
        if t.is_zero() {
            return Some(());
        }
        let ty = &t * &self.y[i];
        self.y[j] += ty;
        for k in 0..=j {
            let d = &t * &self.h[j * w + k];
            self.h[i * w + k] -= d;
        }
        for k in 0..n {
            let d = &t * &self.a[j * n + k];
            self.a[i * n + k] -= d;
            let e = &t * &self.b[k * n + i];
            self.b[k * n + j] += e;
        }
        Some(())
    }

    /// The whole search, from the reduction of the first `H`.
    fn run(&mut self) -> Pslq {
        let n = self.n;
        for i in 1..n {
            for j in (0..i).rev() {
                if self.reduce(i, j).is_none() {
                    return self.finish(Stop::Precision);
                }
            }
        }
        let mut iter = 0;
        loop {
            if let Some(m) = self.relation() {
                return Pslq::Relation(m);
            }
            if let Some(stop) = self.check_stop() {
                return self.finish(stop);
            }
            if iter == self.max_iter {
                return self.finish(Stop::Iterations);
            }
            iter += 1;
            if self.iterate().is_none() {
                return self.finish(Stop::Precision);
            }
        }
    }

    /// One iteration: the exchange of the rows `r` and `r + 1` that maximise `γ^r·|Hᵣᵣ|`, the
    /// rotation that makes `H` lower trapezoidal again, and the reduction of the rows below.
    fn iterate(&mut self) -> Option<()> {
        let n = self.n;
        let w = n - 1;
        let mut r = 0;
        let mut best = f64::NEG_INFINITY;
        for i in 0..w {
            let v = log2_abs(self.hh(i, i)) + (i + 1) as f64 * self.gamma_log2;
            if v > best {
                best = v;
                r = i;
            }
        }
        self.y.swap(r, r + 1);
        for k in 0..w {
            self.h.swap(r * w + k, (r + 1) * w + k);
        }
        for k in 0..n {
            self.a.swap(r * n + k, (r + 1) * n + k);
            self.b.swap(k * n + r, k * n + r + 1);
        }
        if r + 2 < n {
            self.rotate(r)?;
        }
        for i in r + 1..n {
            for j in (0..=(i - 1).min(r + 1)).rev() {
                self.reduce(i, j)?;
            }
        }
        Some(())
    }

    /// Rotates the columns `r` and `r + 1` of `H` so that `Hᵣ,ᵣ₊₁` is zero again.
    fn rotate(&mut self, r: usize) -> Option<()> {
        let (n, f) = (self.n, self.frac);
        let w = n - 1;
        let (p, q) = (&self.h[r * w + r], &self.h[r * w + r + 1]);
        let t0 = (&(p * p) + &(q * q)).isqrt();
        if t0.is_zero() {
            return None;
        }
        let t1 = round_div(&(p << f), &t0);
        let t2 = round_div(&(q << f), &t0);
        for i in r..n {
            let (t3, t4) = (&self.h[i * w + r], &self.h[i * w + r + 1]);
            let u = round_shr(&(&(&t1 * t3) + &(&t2 * t4)), f);
            let v = round_shr(&(&(&t1 * t4) - &(&t2 * t3)), f);
            self.h[i * w + r] = u;
            self.h[i * w + r + 1] = v;
        }
        self.h[r * w + r + 1] = Number::from(0);
        Some(())
    }

    /// The shortest column of `B` that is a relation, normalised, if any is.
    fn relation(&self) -> Option<Vec<Number>> {
        let n = self.n;
        let lg = u64::from(usize::BITS - (n - 1).leading_zeros());
        let mut found: Option<(Number, Vec<Number>)> = None;
        for j in 0..n {
            let yj = &self.y[j];
            if !yj.is_zero() {
                // Σ |Bᵢⱼ|·δᵢ < n·2^top: a |yⱼ| of more bits cannot be within it.
                let top = (0..n)
                    .filter(|&i| !self.b[i * n + j].is_zero())
                    .map(|i| self.b[i * n + j].bits() + self.input.err[i])
                    .max()
                    .unwrap_or(0);
                if yj.bits() > top + lg + 1 {
                    continue;
                }
                let col = (0..n).map(|i| &self.b[i * n + j]);
                if yj.abs() > self.input.tolerance(col) {
                    continue;
                }
            }
            let m: Vec<Number> = (0..n).map(|i| self.b[i * n + j].clone()).collect();
            // y is exact, but the answer is checked again from the numbers themselves.
            if !self.input.holds(&m) {
                continue;
            }
            let norm2 = m.iter().fold(Number::from(0), |s, v| s + v * v);
            if found.as_ref().is_none_or(|(best, _)| norm2 < *best) {
                found = Some((norm2, m));
            }
        }
        found.map(|(_, m)| normalise(m))
    }

    /// Whether to stop: the bound proved beyond `max_norm`, or the precision exhausted.
    fn check_stop(&mut self) -> Option<Stop> {
        let n = self.n;
        let w = n - 1;
        let top = (0..w).map(|j| self.hh(j, j).bits()).max().unwrap_or(0);
        if let Some(max_norm) = self.max_norm.clone() {
            // The floating bound 1/max |Hⱼⱼ|, in log₂.
            let largest = (0..w)
                .map(|j| log2_abs(self.hh(j, j)))
                .fold(f64::NEG_INFINITY, f64::max);
            let floating = self.frac as f64 - largest;
            if max_norm <= 1u32 || self.best >= max_norm {
                return Some(Stop::MaxNorm);
            }
            if floating >= log2_abs(&max_norm) && floating > self.proved_at + 1.0 {
                self.proved_at = floating;
                self.prove();
                if self.best >= max_norm {
                    return Some(Stop::MaxNorm);
                }
            }
        }
        let size = self
            .a
            .iter()
            .chain(&self.b)
            .map(Number::bits)
            .max()
            .unwrap_or(0);
        if size + TRUSTED_BITS >= top {
            return Some(Stop::Precision);
        }
        None
    }

    /// Proves a bound from the current `A`, and keeps it if it is the best so far.
    fn prove(&mut self) {
        let bound = proved_bound(&self.input, &self.a, self.frac as u64);
        if bound > self.best {
            self.best = bound;
        }
    }

    /// The answer when the search stops without a relation.
    fn finish(&mut self, stop: Stop) -> Pslq {
        // A relation may have appeared in the last, interrupted, iteration.
        if let Some(m) = self.relation() {
            return Pslq::Relation(m);
        }
        self.prove();
        let bound = self.best.clone();
        match stop {
            Stop::MaxNorm => Pslq::NoneBelow(bound),
            Stop::Precision => match &self.max_norm {
                Some(m) if bound < *m => Pslq::PrecisionExhausted(bound),
                _ => Pslq::NoneBelow(bound),
            },
            Stop::Iterations => match &self.max_norm {
                Some(m) if bound >= *m => Pslq::NoneBelow(bound),
                _ => Pslq::IterationLimit(bound),
            },
        }
    }
}

/// The bound of Ferguson, Bailey and Arno on the iterations that find a relation of norm up to
/// `2^bits` among `n` numbers: `C(n, 2)·ln(γ^(n−1)·2^bits) / ln τ`, `1/τ² = 1/γ² + 1/4`.
fn iteration_bound(n: usize, bits: f64, gamma: f64) -> usize {
    let tau = 1.0 / (1.0 / (gamma * gamma) + 0.25).sqrt();
    let pairs = (n * (n - 1) / 2) as f64;
    let v = pairs * ((n - 1) as f64 * gamma.ln() + bits * core::f64::consts::LN_2) / tau.ln();
    if v.is_finite() && v < usize::MAX as f64 {
        v.ceil() as usize
    } else {
        usize::MAX
    }
}

/// The relation with its first nonzero entry positive.
fn normalise(mut m: Vec<Number>) -> Vec<Number> {
    if m.iter()
        .find(|v| !v.is_zero())
        .is_some_and(Number::is_negative)
    {
        for v in &mut m {
            *v = -core::mem::take(v);
        }
    }
    m
}

/// `a / b` rounded to the nearest integer, halves away from zero; `b` not zero.
fn round_div(a: &Number, b: &Number) -> Number {
    let (q, r) = a.div_rem(b);
    // |r| < |b|; round away when 2|r| ≥ |b|.
    if (r.abs() << 1) >= b.abs() {
        if a.is_negative() == b.is_negative() {
            q + Number::from(1)
        } else {
            q - Number::from(1)
        }
    } else {
        q
    }
}

/// `a / 2^k` rounded to the nearest integer, halves up.
fn round_shr(a: &Number, k: usize) -> Number {
    if k == 0 {
        return a.clone();
    }
    (a + &(Number::from(1) << (k - 1))) >> k
}

/// `log₂ |x|`, `−∞` for zero, to the precision of an `f64`.
fn log2_abs(x: &Number) -> f64 {
    let b = x.bits();
    if b == 0 {
        return f64::NEG_INFINITY;
    }
    if b <= 1000 {
        x.to_f64().abs().log2()
    } else {
        (x >> (b - 64) as usize).to_f64().abs().log2() + (b - 64) as f64
    }
}
