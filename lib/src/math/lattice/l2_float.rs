//! The floating-point numbers of the L² algorithm: [`Dpe`], an `f64` mantissa with an exponent
//! of its own (fplll's `dpe`, after Pélissier and Zimmermann), whose 53 bits are the fast first
//! try and whose exponent range takes Gram matrices of entries of any size; and [`Big`], a
//! [`BigFloat`] of a chosen precision, for the precision that Nguyen and Stehlé's proof asks
//! for. Both round each operation to nearest, as the analysis of L² assumes.

use crate::math::float::{BigFloat, Round};
use crate::math::numbers::{ByteOrder, Number};
use crate::math::rational::Rational;

/// The operations of L² on its floating-point numbers, each rounded.
pub(super) trait Fp: Clone {
    /// The nearest number of the precision `prec` to `n` (`prec` is 53 for [`Dpe`]).
    fn of_number(n: &Number, prec: u64) -> Self;
    /// The nearest number of the precision `prec` to `x`.
    fn of_f64(x: f64, prec: u64) -> Self;
    /// `r` rounded in the direction `round` to the precision `prec`, for an `r` in the range of
    /// an `f64`.
    fn of_rational(r: &Rational, prec: u64, round: Round) -> Self;
    /// `self + o`.
    fn add(&self, o: &Self) -> Self;
    /// `self − o`.
    fn sub(&self, o: &Self) -> Self;
    /// `self·o`.
    fn mul(&self, o: &Self) -> Self;
    /// `self / o`.
    fn div(&self, o: &Self) -> Self;
    /// `self − a·b`, rounded twice.
    fn sub_mul(&self, a: &Self, b: &Self) -> Self {
        self.sub(&a.mul(b))
    }
    /// `self < o`; false for NaN.
    fn lt(&self, o: &Self) -> bool;
    /// `|self| ≤ o`; false for NaN.
    fn abs_le(&self, o: &Self) -> bool;
    /// Whether the number is finite and positive.
    fn is_positive(&self) -> bool;
    /// Whether the number is finite.
    fn is_finite(&self) -> bool;
    /// The integer nearest to the number; `None` if it is not finite.
    fn rint(&self) -> Option<Number>;
}

/// `2^e` as an `f64`, for `−1022 ≤ e ≤ 1023`.
fn pow2(e: i64) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// The mantissa in `[1/2, 1)` and the exponent of a finite nonzero `x`.
fn split(x: f64) -> (f64, i64) {
    let bits = x.to_bits();
    let ex = ((bits >> 52) & 0x7ff) as i64;
    if ex == 0 {
        // Subnormal: scaled into the normal range first.
        let (m, e) = split(x * pow2(64));
        return (m, e - 64);
    }
    (
        f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52)),
        ex - 1022,
    )
}

/// A float of 53 bits and an exponent of 64: `m·2^e` with `1/2 ≤ |m| < 1`, or `m = 0` and
/// `e = 0`; a NaN or an infinity in `m` stays there.
#[derive(Clone, Copy, Debug)]
pub(super) struct Dpe {
    m: f64,
    e: i64,
}

impl Dpe {
    /// `x·2^e`, normalised.
    fn new(x: f64, e: i64) -> Dpe {
        if x == 0.0 {
            return Dpe { m: 0.0, e: 0 };
        }
        if !x.is_finite() {
            return Dpe { m: x, e: 0 };
        }
        let (m, k) = split(x);
        Dpe { m, e: e + k }
    }
}

/// The top bits of the magnitude of `n` as an `f64` and the power of two that scales them:
/// `|n| ≈ x·2^e`, rounded once (to nearest of its top 64 bits, then of those to 53).
fn top_of(n: &Number) -> (f64, i64) {
    let v = &n.value;
    let be = n.byte_order == ByteOrder::BigEndian;
    // The bytes from the most significant, past any zero ones.
    let len = v.len();
    let at = |i: usize| if be { v[i] } else { v[len - 1 - i] };
    let mut skip = 0;
    while skip < len && at(skip) == 0 {
        skip += 1;
    }
    let used = len - skip;
    let take = used.min(8);
    let mut w = 0u64;
    for i in 0..take {
        w = (w << 8) | u64::from(at(skip + i));
    }
    (w as f64, 8 * (used - take) as i64)
}

impl Fp for Dpe {
    fn of_number(n: &Number, _: u64) -> Dpe {
        if n.is_zero() {
            return Dpe { m: 0.0, e: 0 };
        }
        let (x, e) = top_of(n);
        Dpe::new(if n.is_negative() { -x } else { x }, e)
    }

    fn of_f64(x: f64, _: u64) -> Dpe {
        Dpe::new(x, 0)
    }

    fn of_rational(r: &Rational, _: u64, round: Round) -> Dpe {
        // Rounded once to 53 bits; the f64 then holds it exactly.
        let x = BigFloat::from_rational(r, 53, round).0;
        Dpe::new(x.to_f64_round(round).0, 0)
    }

    fn add(&self, o: &Dpe) -> Dpe {
        if self.m == 0.0 {
            return *o;
        }
        if o.m == 0.0 {
            return *self;
        }
        let (big, small) = if self.e >= o.e { (self, o) } else { (o, self) };
        let d = big.e - small.e;
        if d > 64 {
            // The smaller is below half an ulp of the larger.
            return *big;
        }
        Dpe::new(big.m + small.m * pow2(-d), big.e)
    }

    fn sub(&self, o: &Dpe) -> Dpe {
        self.add(&Dpe { m: -o.m, e: o.e })
    }

    fn mul(&self, o: &Dpe) -> Dpe {
        Dpe::new(self.m * o.m, self.e + o.e)
    }

    fn div(&self, o: &Dpe) -> Dpe {
        Dpe::new(self.m / o.m, self.e - o.e)
    }

    fn lt(&self, o: &Dpe) -> bool {
        self.sub(o).m < 0.0
    }

    fn abs_le(&self, o: &Dpe) -> bool {
        Dpe {
            m: self.m.abs(),
            e: self.e,
        }
        .sub(o)
        .m <= 0.0
    }

    fn is_positive(&self) -> bool {
        self.m.is_finite() && self.m > 0.0
    }

    fn is_finite(&self) -> bool {
        self.m.is_finite()
    }

    fn rint(&self) -> Option<Number> {
        if !self.m.is_finite() {
            return None;
        }
        if self.e <= 52 {
            // |m·2^e| < 2^52: an f64 holds it exactly, and an i64 its rounding.
            if self.e < -2 {
                return Some(Number::from(0u8));
            }
            return Some(Number::from((self.m * pow2(self.e)).round() as i64));
        }
        // An integer already: the 53 bits of the mantissa, shifted.
        let mi = (self.m * pow2(53)) as i64;
        Some(Number::from(mi) << (self.e - 53) as usize)
    }
}

/// A [`BigFloat`] of a precision fixed when it is made; the operations round to nearest at it.
#[derive(Clone, Debug)]
pub(super) struct Big(BigFloat);

impl Big {
    fn prec(&self, o: &Big) -> u64 {
        self.0.prec().max(o.0.prec())
    }
}

impl Fp for Big {
    fn of_number(n: &Number, prec: u64) -> Big {
        Big(BigFloat::from_number(n, prec, Round::Nearest).0)
    }

    fn of_f64(x: f64, prec: u64) -> Big {
        Big(BigFloat::from_f64_round(x, prec, Round::Nearest).0)
    }

    fn of_rational(r: &Rational, prec: u64, round: Round) -> Big {
        Big(BigFloat::from_rational(r, prec, round).0)
    }

    fn add(&self, o: &Big) -> Big {
        Big(self.0.add_round(&o.0, self.prec(o), Round::Nearest).0)
    }

    fn sub(&self, o: &Big) -> Big {
        Big(self.0.sub_round(&o.0, self.prec(o), Round::Nearest).0)
    }

    fn mul(&self, o: &Big) -> Big {
        Big(self.0.mul_round(&o.0, self.prec(o), Round::Nearest).0)
    }

    fn div(&self, o: &Big) -> Big {
        Big(self.0.div_round(&o.0, self.prec(o), Round::Nearest).0)
    }

    fn lt(&self, o: &Big) -> bool {
        self.0 < o.0
    }

    fn abs_le(&self, o: &Big) -> bool {
        self.0.abs() <= o.0
    }

    fn is_positive(&self) -> bool {
        self.0.is_finite() && self.0 > 0.0
    }

    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    fn rint(&self) -> Option<Number> {
        self.0.to_number_round(Round::Nearest).map(|(n, _)| n)
    }
}
