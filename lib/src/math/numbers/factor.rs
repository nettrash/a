//! Factorisation of numbers: [`Number::factor`] into primes and exponents, the methods it chooses
//! from, and perfect powers.
//!
//! - [`trial`]: trial division by the primes up to a bound.
//! - [`rho`]: Pollard's rho method with Brent's cycle finding.
//! - [`pm1`] and [`pp1`]: Pollard's `p - 1` and Williams's `p + 1`.
//! - [`ecm`]: Lenstra's elliptic curve method on Montgomery's curves.
//! - [`siqs`]: the self-initialising quadratic sieve.
//! - [`Number::perfect_power`]: whether a number is `b^k`, with the largest `k`.
//!
//! The methods share one arithmetic: residues modulo the odd number being factored in Montgomery's
//! form, in registers up to eight limbs and in memory beyond (`field`). [`Number::factor`]
//! chooses among them by size (`driver`).

pub(super) mod driver;
mod ecm;
mod ecmfft;
mod field;
mod mk;
mod pm1;
mod poly;
mod power;
mod primes;
mod rho;
mod siqs;
mod stage1;
mod stage2;
mod tree;
mod trial;

pub use siqs::siqs;

use super::{ByteOrder, Number};
use ecm::{EcmJob, FIRST_SIGMA};
use field::{limbs_of, number_of, with_field};
use pm1::{PP1_SEEDS, Pm1Job, Pp1Job};
use rho::{Rho, RhoJob};
use stage1::Stage1;
use stage2::Plan;

/// The factorisation of a positive number into primes and their exponents, from
/// [`Number::factor`] or [`Factorization::new`]: the primes in increasing order, each once, with an
/// exponent of at least 1. The factorisation of 1 has no factors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Factorization {
    /// The primes, increasing, with their exponents.
    factors: Vec<(Number, u32)>,
    /// The byte order of the numbers it gives.
    order: ByteOrder,
}

impl Factorization {
    /// The factorisation with the given primes and exponents, in any order: the primes are sorted,
    /// equal ones merged by adding their exponents, and those of exponent 0 dropped; each prime
    /// is taken in the byte order `order`. The primes are not tested: a composite one makes every
    /// arithmetic function of the result wrong.
    ///
    /// # Panics
    ///
    /// Panics if a "prime" is below 2, or if the exponents of a prime add up to more than
    /// `u32::MAX`.
    pub fn new(factors: Vec<(Number, u32)>, order: ByteOrder) -> Factorization {
        let two = Number::new_from_u8(2, order);
        let mut f: Vec<(Number, u32)> = factors
            .into_iter()
            .filter(|(_, e)| *e > 0)
            .map(|(p, e)| {
                if p < two {
                    not_a_prime();
                }
                (in_order(p, order), e)
            })
            .collect();
        f.sort_by(|a, b| a.0.cmp(&b.0));
        let mut merged: Vec<(Number, u32)> = Vec::with_capacity(f.len());
        for (p, e) in f {
            match merged.last_mut() {
                Some((q, k)) if *q == p => {
                    *k = k.checked_add(e).unwrap_or_else(|| exponent_overflow());
                }
                _ => merged.push((p, e)),
            }
        }
        Factorization {
            factors: merged,
            order,
        }
    }

    /// The primes, in increasing order, with their exponents.
    pub fn factors(&self) -> &[(Number, u32)] {
        &self.factors
    }

    /// The primes and their exponents, taken.
    pub fn into_factors(self) -> Vec<(Number, u32)> {
        self.factors
    }

    /// The byte order of the numbers it gives.
    pub fn byte_order(&self) -> ByteOrder {
        self.order
    }

    /// The number factored: the product of the primes to their exponents, 1 for no factors, by a
    /// balanced tree of products.
    pub fn value(&self) -> Number {
        let mut level: Vec<Number> = self.factors.iter().map(|(p, e)| p.pow(*e)).collect();
        if level.is_empty() {
            return Number::new_from_u8(1, self.order);
        }
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            let mut it = level.into_iter();
            while let Some(a) = it.next() {
                match it.next() {
                    Some(b) => next.push(a * b),
                    None => next.push(a),
                }
            }
            level = next;
        }
        level.pop().expect("a product")
    }
}

/// Panics for a "prime" below 2 in [`Factorization::new`].
#[cold]
#[inline(never)]
#[track_caller]
fn not_a_prime() -> ! {
    panic!("a prime factor must be at least 2")
}

/// Panics for exponents that add up past `u32::MAX` in [`Factorization::new`].
#[cold]
#[inline(never)]
#[track_caller]
fn exponent_overflow() -> ! {
    panic!("the exponent of a prime factor overflows u32")
}

/// Panics for the factorisation of zero.
#[cold]
#[inline(never)]
#[track_caller]
fn zero_factorisation() -> ! {
    panic!("zero has no factorisation")
}

impl Number {
    /// The factorisation of the absolute value of `self` into primes, in the byte order of `self`:
    /// none for 1 and -1. The primes are probable primes by
    /// [`is_probable_prime`](Number::is_probable_prime), exact below `2^64`.
    ///
    /// The method is chosen by size, as PARI, FLINT and yafu choose it: trial division by the
    /// primes below 8192, then for each cofactor the exact test below `2^64` or Baillie–PSW above,
    /// the test for a perfect power, and the search for a factor: below `2^64`, Hart's one line
    /// method up to 40 bits and the elliptic curve method in registers above; beyond, a short run
    /// of Pollard's rho, `p - 1` from 45 digits, the elliptic curve method by levels of growing
    /// bounds, and the self-initialising quadratic sieve for cofactors of 24 to 110 digits once the
    /// curves have made factors of 30 % of their digits unlikely. The curves go on until they find
    /// a factor, so that the factorisation always ends, in a time that grows with the second
    /// largest prime factor. See the [factorisation page](https://github.com/nettrash/a/blob/HEAD/lib/docs/math/factor.md)
    /// for the methods and their measurements.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 2^64 + 1 = 274177 · 67280421310721.
    /// let n: Number = "18446744073709551617".parse().unwrap();
    /// let f = n.factor();
    /// let primes: Vec<String> = f.factors().iter().map(|(p, _)| p.to_string()).collect();
    /// assert_eq!(primes, ["274177", "67280421310721"]);
    /// assert_eq!(f.value(), n);
    /// ```
    #[track_caller]
    pub fn factor(&self) -> Factorization {
        let x = limbs_of(self);
        if x.is_empty() {
            zero_factorisation();
        }
        let order = self.byte_order;
        let mut words: Vec<(u64, u32)> = Vec::new();
        let mut long = driver::factor_limbs(x, &mut words);
        // The primes in increasing order, each once: the words first, then the longer ones.
        words.sort_unstable_by_key(|w| w.0);
        long.sort_unstable_by(|a, b| {
            a.0.len()
                .cmp(&b.0.len())
                .then_with(|| a.0.iter().rev().cmp(b.0.iter().rev()))
        });
        let mut factors: Vec<(Number, u32)> = Vec::with_capacity(words.len() + long.len());
        let mut last = 0;
        for (p, e) in words {
            if p == last {
                let k = &mut factors.last_mut().expect("the prime before").1;
                *k = k.checked_add(e).unwrap_or_else(|| exponent_overflow());
            } else {
                factors.push((number_of(&[p], order), e));
                last = p;
            }
        }
        let mut prev: Option<Vec<u64>> = None;
        for (p, e) in long {
            if prev.as_ref() == Some(&p) {
                let k = &mut factors.last_mut().expect("the prime before").1;
                *k = k.checked_add(e).unwrap_or_else(|| exponent_overflow());
            } else {
                factors.push((number_of(&p, order), e));
                prev = Some(p);
            }
        }
        Factorization { factors, order }
    }

    /// `Some((b, k))` when the absolute value of `self` is `b^k` for some `k >= 2`, with the largest
    /// such `k`; `b` has the sign of `self`, so `k` is odd for a negative number. `None` for 0, 1
    /// and -1, and for every number that is not a perfect power. `b` is in the byte order of
    /// `self`.
    ///
    /// The largest `k` is the product of the primes `p` for which the number is a `p`-th power,
    /// each as often as its root is one again. The exponents of 2 and of the odd primes to 47 in
    /// the number are multiples of `k`, which settles most numbers at once; the other primes `p`
    /// are tried up to the bits of the number over `log2 53`, by residues modulo primes
    /// `q = 1 mod p`, by the floating-point root when it is short, and otherwise by
    /// [`nth_root`](Number::nth_root). For numbers of more than `2^32` bits, `k` is the largest
    /// that fits in a `u32` among those found.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_u64(1 << 60, ByteOrder::BigEndian);
    /// let (b, k) = n.perfect_power().unwrap();
    /// assert_eq!((b, k), (Number::new_from_u8(2, ByteOrder::BigEndian), 60));
    /// // -64 = (-4)^3, the largest odd exponent.
    /// let (b, k) = Number::new_from_i32(-64, ByteOrder::BigEndian).perfect_power().unwrap();
    /// assert_eq!((b, k), (Number::new_from_i32(-4, ByteOrder::BigEndian), 3));
    /// assert!(Number::new_from_u32(1000001, ByteOrder::BigEndian).perfect_power().is_none());
    /// ```
    pub fn perfect_power(&self) -> Option<(Number, u32)> {
        let negative = self.negative;
        if self.value.len() <= 16 {
            // In registers, without reading the number into limbs.
            let mut x = 0u128;
            match self.byte_order {
                ByteOrder::BigEndian => self
                    .value
                    .iter()
                    .for_each(|&b| x = (x << 8) | u128::from(b)),
                ByteOrder::LittleEndian => self
                    .value
                    .iter()
                    .rev()
                    .for_each(|&b| x = (x << 8) | u128::from(b)),
            }
            if x < 2 {
                return None;
            }
            return power::perfect_power_u128(x, negative).map(|(b, k)| {
                let mut b = Number::new_from_u128(b, self.byte_order);
                b.negative = negative;
                (b, k)
            });
        }
        let x = limbs_of(self);
        if x.is_empty() || x == [1] {
            return None;
        }
        power::perfect_power_limbs(x, negative).map(|(b, k)| {
            let mut b = number_of(&b, self.byte_order);
            b.negative = negative;
            (b, k)
        })
    }

    /// Whether `self` is a perfect power, `b^k` for some `k >= 2`: [`perfect_power`](Self::perfect_power)
    /// is `Some`.
    pub fn is_perfect_power(&self) -> bool {
        self.perfect_power().is_some()
    }
}

/// The primes up to `bound` that divide the absolute value of `n`, with their exponents, in
/// increasing order, and the cofactor left, in the byte order of `n`: the absolute value of `n`
/// divided by those primes to their exponents. For 0, no primes and the cofactor 0.
///
/// The odd primes below 8192 are tried by remainders of `n` modulo products of consecutive
/// primes that fit in a limb, eight products a pass, and a multiplication for each prime, the
/// primes of a product together; primes beyond by a segmented sieve. Once the cofactor is below
/// `2^64`, the primes are tried on it directly, eight at a time, and the division stops when the
/// prime's square passes the cofactor, which is then 1 or a prime, given among the primes when it
/// is at most `bound`.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Number, factor};
///
/// // 1001 · 2^4 · 1000003 = 7 · 11 · 13 · 2^4 · 1000003.
/// let n = Number::new_from_u64(1001 * 16 * 1_000_003, ByteOrder::BigEndian);
/// let (primes, rest) = factor::trial(&n, 100);
/// assert_eq!(primes, [(2, 4), (7, 1), (11, 1), (13, 1)]);
/// assert_eq!(rest, Number::new_from_u32(1_000_003, ByteOrder::BigEndian));
/// // Up to the cofactor's square root, the prime cofactor is found too.
/// let (primes, rest) = factor::trial(&n, 1_000_003);
/// assert_eq!(primes.last(), Some(&(1_000_003, 1)));
/// assert_eq!(rest, Number::new_from_u8(1, ByteOrder::BigEndian));
/// ```
pub fn trial(n: &Number, bound: u64) -> (Vec<(u64, u32)>, Number) {
    let (f, rest) = trial::trial_limbs(limbs_of(n), bound);
    (f, number_of(&rest, n.byte_order))
}

/// The odd absolute value of `n` above 3 for the methods, or what they give at once: `Some(2)`
/// for an even number above 2, `None` for 0 to 3 and for primes.
fn method_input(n: &Number) -> Result<Vec<u64>, Option<Number>> {
    let x = limbs_of(n);
    if x.is_empty() || (x.len() == 1 && x[0] <= 3) {
        return Err(None);
    }
    if x[0] & 1 == 0 {
        return Err(Some(Number::new_from_u8(2, n.byte_order)));
    }
    if is_prime_limbs(&x) {
        return Err(None);
    }
    Ok(x)
}

/// A factor of the absolute value of `n` other than 1 and itself, by Pollard's rho method with
/// Brent's cycle finding, or `None` when `iterations` steps find none (always for a prime, 0 and
/// 1). In the byte order of `n`.
///
/// The sequence is `y -> y^2 + c` from 2 in Montgomery's form, with `c = 1` and then `c = 2, 3,
/// ...` when it cycles modulo every prime at once; the differences of Brent's comparisons are
/// multiplied together, 128 between two greatest common divisors. A prime `p` is found after
/// about `sqrt(p)` steps, each a square and a product modulo `n`. An even number gives 2, and a
/// prime `None` at once, by [`is_probable_prime`](Number::is_probable_prime).
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Number, factor};
///
/// // 10403 = 101 · 103.
/// let n = Number::new_from_u32(10403, ByteOrder::BigEndian);
/// let f = factor::rho(&n, 1000).unwrap();
/// assert!(f == 101u32 || f == 103u32);
/// assert!(factor::rho(&Number::new_from_u32(10007, ByteOrder::BigEndian), 1000).is_none());
/// ```
pub fn rho(n: &Number, iterations: u64) -> Option<Number> {
    let x = match method_input(n) {
        Ok(x) => x,
        Err(r) => return r,
    };
    match with_field(&x, RhoJob { steps: iterations }) {
        Rho::Found(g) => Some(number_of(&g, n.byte_order)),
        Rho::Nothing => None,
    }
}

/// A factor of the absolute value of `n` other than 1 and itself, by Pollard's `p - 1` method with
/// the bounds `b1` (stage 1) and `b2` (stage 2), or `None`. In the byte order of `n`.
///
/// It finds a prime `p` when `p - 1` is a product of prime powers up to `b1` and at most one more
/// prime up to `b2` (or a little past it). Stage 1 computes `3^E` for `E` the product of the
/// largest powers of the primes up to `b1` that are at most `b1`, with a square a bit of `E` (about
/// `1.44 b1` of them); stage 2 takes the primes up to `b2` by the baby-step giant-step continuation
/// with prime pairing, about a product for each prime, or for longer ranges by Montgomery and
/// Kruppa's continuation by fast polynomial arithmetic, which takes every number prime to its `P`
/// up to a little past `b2` in about `sqrt(b2)` products of residues and a few products of
/// polynomials of that degree. The primes up to the baby-step giant-step continuation's `D/2`, at
/// most 30030, are taken in stage 1 when `b1` is below it. An even number gives 2, a prime `None`
/// at once.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Number, factor};
///
/// // 1000003 - 1 = 2 · 3 · 166667, 1000033 - 1 = 2^5 · 3 · 10417: stage 1 to 200 finds neither,
/// // stage 2 to 11000 finds 1000033.
/// let n = Number::new_from_u64(1000003 * 1000033, ByteOrder::BigEndian);
/// assert!(factor::pm1(&n, 200, 0).is_none());
/// assert_eq!(factor::pm1(&n, 200, 11000).unwrap(), 1000033u32);
/// ```
pub fn pm1(n: &Number, b1: u64, b2: u64) -> Option<Number> {
    let x = match method_input(n) {
        Ok(x) => x,
        Err(r) => return r,
    };
    let setup = Setup::new(b1, b2, 1);
    with_field(
        &x,
        Pm1Job {
            stage1: &setup.stage1,
            plan: setup.plan.as_ref(),
        },
    )
    .map(|g| number_of(&g, n.byte_order))
}

/// A factor of the absolute value of `n` other than 1 and itself, by Williams's `p + 1` method
/// with the bounds `b1` and `b2`, or `None`. In the byte order of `n`.
///
/// It finds a prime `p` when `p + 1` (or `p - 1`, depending on the starting value) is a product of
/// prime powers up to `b1` and at most one more prime up to `b2`. Stage 1 computes the Lucas
/// sequence `V_E(P_0)`, `V_0 = 2`, `V_1 = P_0`, for the exponent `E` of [`pm1`], by the ladder, a
/// square and a product a bit, or from six limbs by Montgomery's PRAC chains, about 1.5 products a
/// bit (in registers, to six to eight limbs, for `b1` up to `10^7`); stage 2 is that of [`pm1`], in
/// `Z_n[sqrt(V^2 - 4)]` for the continuation by polynomials.
/// The starting values `2/7`, `6/5` and `6` are tried in turn: for a prime with `p + 1` smooth,
/// one of them works with a chance of 7 in 8. An even number gives 2, a prime `None` at once.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Number, factor};
///
/// // 1000099 + 1 = 2^2 · 5^2 · 73 · 137, 1000117 + 1 = 2 · 3 · 166686 ... has a large prime.
/// let n = Number::new_from_u64(1000099 * 1000117, ByteOrder::BigEndian);
/// assert_eq!(factor::pp1(&n, 200, 0).unwrap(), 1000099u32);
/// ```
pub fn pp1(n: &Number, b1: u64, b2: u64) -> Option<Number> {
    let x = match method_input(n) {
        Ok(x) => x,
        Err(r) => return r,
    };
    let setup = Setup::new(b1, b2, 1);
    PP1_SEEDS.iter().find_map(|&seed| {
        with_field(
            &x,
            Pp1Job {
                seed,
                stage1: &setup.stage1,
                plan: setup.plan.as_ref(),
            },
        )
        .map(|g| number_of(&g, n.byte_order))
    })
}

/// A factor of the absolute value of `n` other than 1 and itself, by Lenstra's elliptic curve
/// method on Montgomery curves with the bounds `b1` and `b2`, on up to `curves` curves, or `None`.
/// In the byte order of `n`.
///
/// The curves are those of GMP-ECM's `-param 2` for `sigma = 2, 3, 4, ...` (`-sigma 2:s`): Suyama's
/// curves, whose group orders are multiples of 12, made from the multiples of a point on
/// `y^2 = x^3 + 36` so that the starting point is `x = 2`; from a `sigma` whose multiple of that
/// point is at infinity modulo every prime of `n` at once, after which the curves would repeat
/// modulo each, GMP-ECM's `-param 1` curves instead (`-sigma 1:s`). A curve finds a prime `p` when
/// its group order modulo `p` is a product of prime powers up to `b1` and at most one more prime up
/// to `b2`. Stage 1 multiplies the point by the exponent `E` of [`pm1`] by Montgomery's ladder, 4
/// squares and 5 products a bit, as the product by the point's `x` is a doubling, taking it to
/// affine after each chunk of `E`, in registers several curves together, step for step; stage 2 is
/// the baby-step giant-step continuation with prime pairing, one product for each pair, or for
/// longer ranges the FFT continuation, the product of the values at the baby steps' `x` of the
/// product of the giant steps' polynomials modulo theirs, which takes every number prime to its `D`
/// up to a little past `b2`. The factor given is the first curve's to find one. The primes up to the
/// continuation's `D/2` are taken in stage 1 when `b1` is below it. An even number gives 2, a prime
/// `None` at once.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Number, factor};
///
/// // (2^61 - 1) · 1000000007.
/// let n = Number::new_from_u128(2305843009213693951 * 1000000007, ByteOrder::BigEndian);
/// let f = factor::ecm(&n, 2000, 200_000, 100).unwrap();
/// assert!(f == 1000000007u32 || f == 2305843009213693951u64);
/// ```
pub fn ecm(n: &Number, b1: u64, b2: u64, curves: u32) -> Option<Number> {
    let x = match method_input(n) {
        Ok(x) => x,
        Err(r) => return r,
    };
    ecm_limbs(&x, b1, b2, FIRST_SIGMA, u64::from(curves)).map(|g| number_of(&g, n.byte_order))
}

/// [`ecm`] on the odd limbs `x`, with the curves of `sigma` from `first`.
fn ecm_limbs(x: &[u64], b1: u64, b2: u64, first: u64, curves: u64) -> Option<Vec<u64>> {
    let setup = Setup::new(b1, b2, ECM_WEIGHT);
    ecm_with(x, &setup, first, curves)
}

/// [`ecm`] on the odd limbs `x` with the stages of `setup`.
fn ecm_with(x: &[u64], setup: &Setup, first: u64, curves: u64) -> Option<Vec<u64>> {
    with_field(
        x,
        EcmJob {
            first,
            count: curves,
            stage1: &setup.stage1,
            plan: setup.plan.as_ref(),
        },
    )
}

/// The exponent of a first stage and the plan of the second, made once for many curves or runs.
struct Setup {
    /// The first stage.
    stage1: Stage1,
    /// The second, if any.
    plan: Option<Plan>,
}

impl Setup {
    /// The stages for the bounds `b1` and `b2`, the plan's `D` for steps of `weight` products.
    fn new(b1: u64, b2: u64, weight: u64) -> Setup {
        let plan = Plan::new(b1, b2, weight);
        let stage1 = Stage1::new(plan.as_ref().map_or(b1, |p| p.b1));
        Setup { stage1, plan }
    }
}

/// The products a baby or giant step of the elliptic curve method costs, against one for a pair:
/// an addition of points, 6, and its share of the inversions, 3.
const ECM_WEIGHT: u64 = 9;

/// Whether the limbs `x` are a probable prime.
fn is_prime_limbs(x: &[u64]) -> bool {
    if x.len() == 1 {
        return super::prime::is_prime_u64(x[0]);
    }
    number_of(x, ByteOrder::LittleEndian).is_probable_prime()
}

/// `x` in the byte order `order`.
fn in_order(x: Number, order: ByteOrder) -> Number {
    if x.byte_order == order {
        return x;
    }
    let mut value = x.value;
    value.reverse();
    Number {
        value,
        negative: x.negative,
        byte_order: order,
    }
}
