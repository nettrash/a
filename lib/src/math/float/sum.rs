//! Sums and dot products of any number of terms, rounded once: the exact sum of the terms, or of
//! the exact products, as MPFR's `mpfr_sum` and `mpfr_dot`.
//!
//! The terms are added exactly into one fixed-point accumulator in two's complement, as long as
//! their bits span a modest range (the usual case): one product into a scratch buffer and one
//! shifted addition for each term, then one rounding. Terms whose bits span a range too wide for
//! that are sorted by their exponents and added exactly from the largest down, until the terms
//! left are too small to matter but through their sign: their sum then lies below a quarter of
//! the last place of the result and below the last bit of the sum so far, and only its sign is
//! needed, which the same accumulation gives at one bit, where the first such stop decides it.

use super::limbs;
use super::round::round_mag;
use super::{BigFloat, Kind, Round};
use crate::math::numbers::nat;
use core::cmp::Ordering;

/// One term of a sum: `±x`, or `±x·y` when `y` is given.
#[derive(Clone, Copy)]
pub(crate) struct Term<'a> {
    pub(crate) x: &'a BigFloat,
    pub(crate) y: Option<&'a BigFloat>,
    pub(crate) negate: bool,
}

impl<'a> Term<'a> {
    /// The term `x`.
    #[inline]
    pub(crate) fn one(x: &'a BigFloat, negate: bool) -> Term<'a> {
        Term { x, y: None, negate }
    }

    /// The term `x·y`.
    #[inline]
    pub(crate) fn product(x: &'a BigFloat, y: &'a BigFloat, negate: bool) -> Term<'a> {
        Term {
            x,
            y: Some(y),
            negate,
        }
    }
}

/// How a sum whose exact value is zero came to be: all of its terms zeros of one sign, zeros of
/// both signs, or terms that are not all zero cancelling exactly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Zeros {
    /// Every term `-0`.
    Negative,
    /// Every term `+0` (or no terms).
    Positive,
    /// Zeros of both signs, or terms that cancel.
    Mixed,
}

impl Zeros {
    /// Whether the zero is `-0` in the mode `round`: as IEEE 754's sums of zeros and MPFR's
    /// `mpfr_sum`, `-0` when every term is `-0`, and for mixed zeros or a cancellation only when
    /// rounding down.
    #[inline]
    pub(crate) fn negative(self, round: Round) -> bool {
        match self {
            Zeros::Negative => true,
            Zeros::Positive => false,
            Zeros::Mixed => round == Round::Down,
        }
    }

    /// The zeros of two sums of zeros together.
    #[inline]
    fn and(self, other: Zeros) -> Zeros {
        if self == other { self } else { Zeros::Mixed }
    }
}

/// The value of a sum before its rounding.
#[derive(Debug)]
pub(crate) enum Summed {
    /// NaN: a NaN term, `0·∞`, or infinities of both signs.
    Nan,
    /// An infinity of the sign.
    Inf(bool),
    /// An exact zero.
    Zero(Zeros),
    /// `±m·2^lsb`, `m` not zero and without zero limbs at the top, exactly; or with `sticky`,
    /// strictly between `±m·2^lsb` and `±(m + 1)·2^lsb`, `m` having more bits than the precision
    /// that the sum was asked for.
    Val {
        negative: bool,
        m: Vec<u64>,
        lsb: i128,
        sticky: bool,
    },
}

/// The number of bits of a bound `2^b ≥ n` of `n ≥ 1`: `ceil(log2 n)`.
#[inline]
fn log2_ceil(n: usize) -> i128 {
    if n <= 1 {
        0
    } else {
        i128::from(usize::BITS - (n - 1).leading_zeros())
    }
}

/// The number of zero limbs at the bottom of a mantissa.
#[inline]
fn lead(m: &[u64]) -> usize {
    m.iter().position(|&x| x != 0).unwrap_or(0)
}

/// The significant limbs of a finite number that is not zero and the exponent of their lowest
/// bit, the zero limbs below the precision left out.
#[inline]
fn significant(x: &BigFloat) -> (bool, &[u64], i128) {
    let (negative, m, lsb) = x.parts().expect("a finite number that is not zero");
    let z = lead(m);
    (negative, &m[z..], lsb + 64 * z as i128)
}

/// The buffers of the fixed-point accumulation, to keep across sums: the product of a term, its
/// shifted copy, and the working memory of long products.
#[derive(Default)]
pub(crate) struct Work {
    product: Vec<u64>,
    shifted: Vec<u64>,
    scratch: Vec<u64>,
}

/// The sum of `terms`, exactly or with a sticky bit below `need` bits: NaN, an infinity, an exact
/// zero, or the value.
pub(crate) fn sum_terms(terms: &[Term<'_>], need: u64) -> Summed {
    sum_terms_in(terms, need, &mut Work::default())
}

/// [`sum_terms`] with the buffers of `work`.
pub(crate) fn sum_terms_in(terms: &[Term<'_>], need: u64, work: &mut Work) -> Summed {
    // The special values first, as `mpfr_mul` and `mpfr_sum` have them.
    let mut inf: Option<bool> = None;
    let mut zeros: Option<Zeros> = None;
    let mut finite = 0usize;
    for t in terms {
        let (x, y) = (t.x, t.y);
        let negative =
            (x.is_sign_negative() != y.is_some_and(|y| y.is_sign_negative())) != t.negate;
        let nan = x.is_nan() || y.is_some_and(|y| y.is_nan());
        let infinite = x.is_infinite() || y.is_some_and(|y| y.is_infinite());
        let zero = x.is_zero() || y.is_some_and(|y| y.is_zero());
        if nan || infinite && zero {
            return Summed::Nan;
        }
        if infinite {
            match inf {
                Some(s) if s != negative => return Summed::Nan,
                _ => inf = Some(negative),
            }
        } else if zero {
            let z = if negative {
                Zeros::Negative
            } else {
                Zeros::Positive
            };
            zeros = Some(zeros.map_or(z, |w| w.and(z)));
        } else {
            finite += 1;
        }
    }
    if let Some(negative) = inf {
        return Summed::Inf(negative);
    }
    if finite == 0 {
        return Summed::Zero(zeros.unwrap_or(Zeros::Positive));
    }
    // The span of the bits: below `2^top` and from `2^low`.
    let mut top = i128::MIN;
    let mut low = i128::MAX;
    let mut longest = 0u64;
    for t in terms {
        if !t.x.is_normal() || t.y.is_some_and(|y| !y.is_normal()) {
            continue;
        }
        let (_, mx, lx) = significant(t.x);
        let (mut e, mut l, mut bits) = (lx + 64 * mx.len() as i128, lx, 64 * mx.len() as u64);
        if let Some(y) = t.y {
            let (_, my, ly) = significant(y);
            e += ly + 64 * my.len() as i128;
            l += ly;
            bits += 64 * my.len() as u64;
        }
        top = top.max(e);
        low = low.min(l);
        longest = longest.max(bits);
    }
    let span = top - low;
    if span <= 4 * i128::from(longest + need) + (1 << 16) {
        fixed(terms, top, low, finite, work)
    } else {
        sorted(terms, need)
    }
}

/// The exact sum by a two's-complement accumulator from `2^low` to past `2^top`, for `finite`
/// terms that are finite and not zero.
fn fixed(terms: &[Term<'_>], top: i128, low: i128, finite: usize, work: &mut Work) -> Summed {
    // `|sum| < finite·2^top`, and a sign bit above.
    let bits = (top - low) + log2_ceil(finite) + 2;
    let n = (bits as usize).div_ceil(64) + 1;
    let mut acc = vec![0u64; n];
    let Work {
        product,
        shifted,
        scratch,
    } = work;
    for t in terms {
        if !t.x.is_normal() || t.y.is_some_and(|y| !y.is_normal()) {
            continue;
        }
        let (nx, mx, lx) = significant(t.x);
        let (m, lsb, negative) = match t.y {
            None => (mx, lx, nx != t.negate),
            Some(y) => {
                let (ny, my, ly) = significant(y);
                if core::ptr::eq(mx, my) {
                    nat::sqr_into(product, mx, scratch);
                } else {
                    nat::mul_into(product, mx, my, scratch);
                }
                (&product[..], lx + ly, (nx != ny) != t.negate)
            }
        };
        let off = (lsb - low) as u64;
        let (q, r) = ((off / 64) as usize, (off % 64) as u32);
        let m = if r == 0 {
            m
        } else {
            shifted.clear();
            let mut carry = 0u64;
            for &x in m {
                shifted.push(x << r | carry);
                carry = x >> (64 - r);
            }
            shifted.push(carry);
            &shifted[..]
        };
        // The carry or borrow out of the top is the two's complement wrapping.
        if negative {
            nat::sub_assign(&mut acc[q..], m);
        } else {
            nat::add_assign(&mut acc[q..], m);
        }
    }
    let negative = acc[n - 1] >> 63 == 1;
    if negative {
        // `-acc = !acc + 1`.
        let mut carry = true;
        for x in &mut acc {
            let (s, c) = (!*x).overflowing_add(u64::from(carry));
            *x = s;
            carry = c;
        }
    }
    limbs::trim(&mut acc);
    if acc.is_empty() {
        return Summed::Zero(Zeros::Mixed);
    }
    Summed::Val {
        negative,
        m: acc,
        lsb: low,
        sticky: false,
    }
}

/// An exact signed magnitude `±m·2^lsb`, `m` not zero and without zero limbs at the top.
struct Val {
    negative: bool,
    m: Vec<u64>,
    lsb: i128,
}

impl Val {
    /// The exponent of a bound: `|v| < 2^top`.
    #[inline]
    fn top(&self) -> i128 {
        self.lsb + i128::from(limbs::bit_len(&self.m))
    }
}

/// `a + b` exactly, `None` when they cancel.
fn add_exact(a: &Val, b: &Val) -> Option<Val> {
    let s = a.lsb.min(b.lsb);
    let x = limbs::shl(&a.m, (a.lsb - s) as u64);
    let y = limbs::shl(&b.m, (b.lsb - s) as u64);
    if a.negative == b.negative {
        return Some(Val {
            negative: a.negative,
            m: limbs::add(&x, &y),
            lsb: s,
        });
    }
    match limbs::cmp(&x, &y) {
        Ordering::Equal => None,
        Ordering::Greater => Some(Val {
            negative: a.negative,
            m: limbs::sub(&x, &y),
            lsb: s,
        }),
        Ordering::Less => Some(Val {
            negative: b.negative,
            m: limbs::sub(&y, &x),
            lsb: s,
        }),
    }
}

/// The scale below which the terms after the sum `s` so far change it only through their sign,
/// for a result of `need` bits: `min(lsb, top - need - 3)`, as [`add_mags`](super::arith::add_mags).
#[inline]
fn negligible_below(s: &Val, need: u64) -> i128 {
    s.lsb.min(s.top() - i128::from(need) - 3)
}

/// The sum of terms spread too widely for [`fixed`]: from the largest down, exactly, until the
/// rest is negligible (see the [module](self)).
fn sorted(terms: &[Term<'_>], need: u64) -> Summed {
    let mut vals: Vec<Val> = terms
        .iter()
        .filter(|t| t.x.is_normal() && t.y.is_none_or(|y| y.is_normal()))
        .map(|t| {
            let (nx, mx, lx) = significant(t.x);
            match t.y {
                None => Val {
                    negative: nx != t.negate,
                    m: mx.to_vec(),
                    lsb: lx,
                },
                Some(y) => {
                    let (ny, my, ly) = significant(y);
                    let mut m = limbs::mul(mx, my);
                    limbs::trim(&mut m);
                    Val {
                        negative: (nx != ny) != t.negate,
                        m,
                        lsb: lx + ly,
                    }
                }
            }
        })
        .collect();
    vals.sort_by_key(|v| core::cmp::Reverse(v.top()));
    let n = vals.len();
    let mut acc: Option<Val> = None;
    for i in 0..n {
        if let Some(s) = &acc {
            let k = negligible_below(s, need);
            // `|Σ_{j ≥ i} v_j| < (n - i)·2^top(v_i)`.
            if vals[i].top() + log2_ceil(n - i) <= k {
                let m = limbs::shl(&s.m, (s.lsb - k) as u64);
                let m = match tail_sign(&vals[i..]) {
                    None => {
                        return Summed::Val {
                            negative: s.negative,
                            m,
                            lsb: k,
                            sticky: false,
                        };
                    }
                    // The sum lies strictly between `m·2^k` and `(m ± 1)·2^k`.
                    Some(negative) if negative == s.negative => m,
                    Some(_) => limbs::sub(&m, &[1]),
                };
                return Summed::Val {
                    negative: s.negative,
                    m,
                    lsb: k,
                    sticky: true,
                };
            }
        }
        acc = match acc {
            None => Some(core::mem::replace(
                &mut vals[i],
                Val {
                    negative: false,
                    m: Vec::new(),
                    lsb: 0,
                },
            )),
            Some(s) => add_exact(&s, &vals[i]),
        };
    }
    match acc {
        None => Summed::Zero(Zeros::Mixed),
        Some(s) => Summed::Val {
            negative: s.negative,
            m: s.m,
            lsb: s.lsb,
            sticky: false,
        },
    }
}

/// The sign of the sum of `vals`, sorted by their exponents from the largest, `None` for zero:
/// once the sum so far `s` makes the rest negligible at one bit, the rest is below a quarter of
/// `|s|` and cannot turn its sign over.
fn tail_sign(vals: &[Val]) -> Option<bool> {
    let n = vals.len();
    let mut acc: Option<Val> = None;
    for (i, v) in vals.iter().enumerate() {
        if let Some(s) = &acc
            && v.top() + log2_ceil(n - i) <= negligible_below(s, 1)
        {
            return Some(s.negative);
        }
        acc = match acc {
            None => Some(Val {
                negative: v.negative,
                m: v.m.clone(),
                lsb: v.lsb,
            }),
            Some(s) => add_exact(&s, v),
        };
    }
    acc.map(|s| s.negative)
}

/// The sum rounded to `prec` bits in the mode `round`, and the direction of the rounding.
pub(crate) fn round_summed(s: Summed, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let kind = match s {
        Summed::Nan => Kind::Nan,
        Summed::Inf(negative) => Kind::Inf { negative },
        Summed::Zero(z) => Kind::Zero {
            negative: z.negative(round),
        },
        Summed::Val {
            negative,
            m,
            lsb,
            sticky,
        } => return round_mag(negative, &m, lsb, sticky, prec, round),
    };
    (BigFloat { prec, kind }, Ordering::Equal)
}

/// The square root of a sum that is not below zero (NaN below), rounded to `prec` bits in the
/// mode `round`; the sum asked for with `2·prec + 4` bits at least.
///
/// A sticky sum lies strictly between `M·2^lsb` and `(M + 1)·2^lsb`, `M` of at least
/// `2·prec + 4` bits: the squares of the numbers of `prec` bits near its root and of the points
/// halfway between them are multiples of `2^(2(e - prec - 1))`, `e` the exponent of the root,
/// which is at least `lsb + 2`, so none lies in that interval, and the root of
/// `(2M + 1)·2^(lsb - 1)` rounds as the root of the sum, inexactly as it does.
pub(crate) fn sqrt_summed(s: Summed, prec: u64, round: Round) -> (BigFloat, Ordering) {
    match s {
        Summed::Val {
            negative: false,
            m,
            lsb,
            sticky,
        } => {
            if sticky {
                let mut m = limbs::shl(&m, 1);
                m[0] |= 1;
                super::arith::sqrt_mag(&m, lsb - 1, prec, round)
            } else {
                super::arith::sqrt_mag(&m, lsb, prec, round)
            }
        }
        Summed::Val { .. } | Summed::Inf(true) => (BigFloat::nan(prec), Ordering::Equal),
        Summed::Zero(z) => (
            BigFloat {
                prec,
                kind: Kind::Zero {
                    negative: z.negative(round),
                },
            },
            Ordering::Equal,
        ),
        s => round_summed(s, prec, round),
    }
}

impl BigFloat {
    /// The sum of `terms` rounded once to `prec` bits in the mode `round`, and the direction of
    /// the rounding, as MPFR's `mpfr_sum`: the exact sum, however many terms and however far
    /// apart their exponents. NaN when a term is NaN or two are infinities of opposite signs; an
    /// exact zero `+0`, or `-0` when every term is `-0`, or when the terms are zeros of both
    /// signs or cancel and the mode rounds down; the sum of no terms `+0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let big = BigFloat::from(1u64 << 60);
    /// let terms = [big.clone(), BigFloat::from(1), -big];
    /// // Added one by one at 53 bits, the 1 would be lost; the sum is exact.
    /// let (s, o) = BigFloat::sum_round(&terms, 53, Round::Nearest);
    /// assert_eq!(s, BigFloat::from(1));
    /// assert_eq!(o, Ordering::Equal);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn sum_round(terms: &[BigFloat], prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        let terms: Vec<Term<'_>> = terms.iter().map(|x| Term::one(x, false)).collect();
        round_summed(sum_terms(&terms, prec), prec, round)
    }

    /// The dot product `Σ xs[i]·ys[i]` rounded once to `prec` bits in the mode `round`, and the
    /// direction of the rounding, as MPFR's `mpfr_dot`: the exact sum of the exact products. The
    /// special values of each product are those of [`mul_round`](BigFloat::mul_round) (`0·∞` is
    /// NaN), and those of the sum those of [`sum_round`](BigFloat::sum_round).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let x = [BigFloat::from(3), BigFloat::from(-5)];
    /// let y = [BigFloat::from(7), BigFloat::from(4)];
    /// assert_eq!(BigFloat::dot_round(&x, &y, 10, Round::Nearest), (BigFloat::from(1), Ordering::Equal));
    /// ```
    ///
    /// # Panics
    ///
    /// When the slices are not of the same length, or `prec` is not in
    /// [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn dot_round(
        xs: &[BigFloat],
        ys: &[BigFloat],
        prec: u64,
        round: Round,
    ) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        assert_eq!(xs.len(), ys.len(), "the slices are not of the same length");
        let terms: Vec<Term<'_>> = xs
            .iter()
            .zip(ys)
            .map(|(x, y)| Term::product(x, y, false))
            .collect();
        round_summed(sum_terms(&terms, prec), prec, round)
    }
}
