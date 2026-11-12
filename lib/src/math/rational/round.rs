//! Rounding to integers: [`Rational::floor`], [`Rational::ceil`], [`Rational::round`],
//! [`Rational::round_ties_even`] and [`Rational::trunc`], which give [`Number`]s, and
//! [`Rational::fract`], what `trunc` leaves.

use super::small::{self, Small};
use super::{Rational, is_one};
use crate::math::numbers::{Number, fraction};
use core::cmp::Ordering;

impl Rational {
    /// The largest integer at most the number, `⌊x⌋`, in its byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!((Rational::from(7) / 2u32).floor(), 3);
    /// assert_eq!((Rational::from(-7) / 2u32).floor(), -4);
    /// ```
    pub fn floor(&self) -> Number {
        if let Some(x) = Small::of(self) {
            let (q, r) = (x.num / x.den, x.num % x.den);
            let q = u128::from(q) + u128::from(x.negative && r != 0);
            return small::number(self.byte_order(), x.negative, q);
        }
        if let Some((num, den)) = self.numbers()
            && is_one(den)
        {
            return num.clone();
        }
        if let Some((negative, n, d)) = self.wide() {
            let q = n / d + u128::from(negative);
            return small::number(self.byte_order(), negative, q);
        }
        let (num, den) = big(self);
        // The remainder of a fraction in lowest terms over a denominator above 1 is not zero, so
        // the floor of a negative number is its truncation less one.
        let mut q = num / den;
        if num.is_negative() {
            q -= 1u8;
        }
        q
    }

    /// The smallest integer at least the number, `⌈x⌉`, in its byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!((Rational::from(7) / 2u32).ceil(), 4);
    /// assert_eq!((Rational::from(-7) / 2u32).ceil(), -3);
    /// ```
    pub fn ceil(&self) -> Number {
        if let Some(x) = Small::of(self) {
            let (q, r) = (x.num / x.den, x.num % x.den);
            let q = u128::from(q) + u128::from(!x.negative && r != 0);
            return small::number(self.byte_order(), x.negative, q);
        }
        if let Some((num, den)) = self.numbers()
            && is_one(den)
        {
            return num.clone();
        }
        if let Some((negative, n, d)) = self.wide() {
            let q = n / d + u128::from(!negative);
            return small::number(self.byte_order(), negative, q);
        }
        let (num, den) = big(self);
        let mut q = num / den;
        if !num.is_negative() {
            q += 1u8;
        }
        q
    }

    /// The integer toward zero from the number, in its byte order: `⌊x⌋` for `x ≥ 0` and `⌈x⌉`
    /// below.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!((Rational::from(-7) / 2u32).trunc(), -3);
    /// ```
    pub fn trunc(&self) -> Number {
        if let Some(x) = Small::of(self) {
            return small::number(self.byte_order(), x.negative, u128::from(x.num / x.den));
        }
        if let Some((num, den)) = self.numbers()
            && is_one(den)
        {
            return num.clone();
        }
        if let Some((negative, n, d)) = self.wide() {
            return small::number(self.byte_order(), negative, n / d);
        }
        let (num, den) = big(self);
        num / den
    }

    /// The nearest integer, halves away from zero, as `f64::round` rounds, in its byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!((Rational::from(5) / 2u32).round(), 3);
    /// assert_eq!((Rational::from(-5) / 2u32).round(), -3);
    /// assert_eq!((Rational::from(7) / 3u32).round(), 2);
    /// ```
    pub fn round(&self) -> Number {
        self.nearest(false)
    }

    /// The nearest integer, halves to the even one, as `f64::round_ties_even` rounds, in its byte
    /// order.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!((Rational::from(5) / 2u32).round_ties_even(), 2);
    /// assert_eq!((Rational::from(-7) / 2u32).round_ties_even(), -4);
    /// ```
    pub fn round_ties_even(&self) -> Number {
        self.nearest(true)
    }

    /// The fractional part, `x - trunc(x)`, of the sign of the number, as `f64::fract`: above
    /// `-1` and below 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// assert_eq!((Rational::from(-7) / 2u32).fract().to_string(), "-1/2");
    /// ```
    pub fn fract(&self) -> Rational {
        let order = self.byte_order();
        // `num % den` is prime to `den` as `num` is, and of the sign of `num`.
        if let Some(x) = Small::of(self) {
            let r = x.num % x.den;
            let d = if r == 0 { 1 } else { x.den };
            return Rational::from_word(x.negative && r != 0, r, d, order);
        }
        let (num, den) = big(self);
        if is_one(den) {
            return Rational::from_word(false, 0, 1, order);
        }
        let mut r = num % den;
        super::trim(&mut r);
        Rational::from_reduced(r, den.clone())
    }

    /// The nearest integer, halves away from zero, or to the even one when `even` is set.
    fn nearest(&self, even: bool) -> Number {
        // `|x| = q + r/d` with `0 < r < d`; round `q` up when `2r > d`, or at a half (`d` even)
        // away from zero, or to the even one.
        if let Some(x) = Small::of(self) {
            // Without branches: random fractions round up or down one in two.
            let (q, r) = (x.num / x.den, x.num % x.den);
            let (twice, d) = (u128::from(r) * 2, u128::from(x.den));
            let up = (twice > d) | ((twice == d) & (!even | (q & 1 == 1)));
            return small::number(
                self.byte_order(),
                x.negative,
                u128::from(q) + u128::from(up),
            );
        }
        if let Some((num, den)) = self.numbers()
            && is_one(den)
        {
            return num.clone();
        }
        if let Some((negative, n, d)) = self.wide() {
            let (q, r) = (n / d, n % d);
            // `2r` against `d`, as `r` against `d - r`, which cannot overflow.
            let up = match r.cmp(&(d - r)) {
                Ordering::Greater => true,
                Ordering::Less => false,
                Ordering::Equal => !even || q & 1 == 1,
            };
            return small::number(self.byte_order(), negative, q + u128::from(up));
        }
        let (num, den) = big(self);
        if den.value.len() <= 8 * fraction::NEAREST_LIMBS {
            return fraction::nearest(num, den, even, self.byte_order());
        }
        // The quotient and the remainder, of the sign of the numerator, and `2|r|` against `d`
        // with the remainder doubled in its own vector.
        let negative = num.is_negative();
        let (mut q, mut r) = num.div_rem(den);
        r.negative = false;
        r <<= 1;
        let up = match r.cmp(den) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => !even || q.bit(0),
        };
        if up {
            if negative {
                q -= 1u8;
            } else {
                q += 1u8;
            }
        }
        q
    }

    /// The sign, the magnitude of the numerator and the denominator, when both parts have at most
    /// 16 bytes.
    #[inline(always)]
    fn wide(&self) -> Option<(bool, u128, u128)> {
        let (n, d) = self.magnitudes();
        if n.bytes.len() > 16 || d.bytes.len() > 16 {
            return None;
        }
        Some((n.negative, n.small(), d.small()))
    }
}

/// The numbers of a rational not in words: one the register paths did not take.
#[inline(always)]
fn big(r: &Rational) -> (&Number, &Number) {
    r.numbers().unwrap_or_else(|| unreachable!())
}
