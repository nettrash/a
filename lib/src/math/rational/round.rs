//! Rounding to integers: [`Rational::floor`], [`Rational::ceil`], [`Rational::round`],
//! [`Rational::round_ties_even`] and [`Rational::trunc`], which give [`Number`]s, and
//! [`Rational::fract`], what `trunc` leaves.

use super::small::{self, Small};
use super::{Rational, is_one};
use crate::math::numbers::Number;
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
        if is_one(&self.den) {
            return self.num.clone();
        }
        if let Some(x) = Small::of(self) {
            let (q, r) = (x.num / x.den, x.num % x.den);
            let q = u128::from(q) + u128::from(x.negative && r != 0);
            return small::number(self.byte_order(), x.negative, q);
        }
        self.num.div_floor(&self.den)
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
        if is_one(&self.den) {
            return self.num.clone();
        }
        if let Some(x) = Small::of(self) {
            let (q, r) = (x.num / x.den, x.num % x.den);
            let q = u128::from(q) + u128::from(!x.negative && r != 0);
            return small::number(self.byte_order(), x.negative, q);
        }
        -(-&self.num).div_floor(&self.den)
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
        if is_one(&self.den) {
            return self.num.clone();
        }
        if let Some(x) = Small::of(self) {
            return small::number(self.byte_order(), x.negative, u128::from(x.num / x.den));
        }
        &self.num / &self.den
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
        if is_one(&self.den) {
            return Rational::from_reduced(
                small::number(self.byte_order(), false, 0),
                self.den.clone(),
            );
        }
        // `num % den` is prime to `den` as `num` is, and of the sign of `num`.
        let mut num = &self.num % &self.den;
        super::trim(&mut num);
        Rational::from_reduced(num, self.den.clone())
    }

    /// The nearest integer, halves away from zero, or to the even one when `even` is set.
    fn nearest(&self, even: bool) -> Number {
        if is_one(&self.den) {
            return self.num.clone();
        }
        // `|x| = q + r/d` with `0 < r < d`; round `q` up when `2r > d`, or at a half (`d` even)
        // away from zero, or to the even one.
        if let Some(x) = Small::of(self) {
            let (q, r) = (x.num / x.den, x.num % x.den);
            let twice = u128::from(r) * 2;
            let up = match twice.cmp(&u128::from(x.den)) {
                Ordering::Greater => true,
                Ordering::Less => false,
                Ordering::Equal => !even || q % 2 == 1,
            };
            return small::number(
                self.byte_order(),
                x.negative,
                u128::from(q) + u128::from(up),
            );
        }
        let negative = self.num.is_negative();
        let (mut q, r) = self.num.abs().div_rem(&self.den);
        let up = match (&r + &r).cmp(&self.den) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => !even || q.bit(0),
        };
        if up {
            q += 1u8;
        }
        if negative { -q } else { q }
    }
}
