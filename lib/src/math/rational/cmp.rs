//! Rationals compared by value: `==`, `<` and the rest with each other, with [`Number`]s on
//! either side and with the primitive integers on the right, and a hash that agrees with `==`.
//!
//! As every rational has one form, lowest terms with a positive denominator, two are equal when
//! their parts are, whatever their byte orders. The order compares `a/b` with `c/d` by their signs,
//! then, when the signs are the same, `a·d` with `c·b`: in registers when the parts fit in 64
//! bits, and otherwise first by the lengths of the products, which tell most pairs apart without
//! multiplying.

use super::arith::Primitive;
use super::small::{self, Small};
use super::{Rational, is_one};
use crate::math::numbers::Number;
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};

impl PartialEq for Rational {
    /// Whether the rationals are the same number: whether their parts are equal.
    #[inline]
    fn eq(&self, other: &Rational) -> bool {
        self.num == other.num && self.den == other.den
    }
}

impl Eq for Rational {}

impl Hash for Rational {
    /// Hashes the number by value, so that equal rationals hash alike whatever their byte orders:
    /// the numerator, then the denominator, as [`Number`] hashes them.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.num.hash(state);
        self.den.hash(state);
    }
}

impl PartialOrd for Rational {
    #[inline]
    fn partial_cmp(&self, other: &Rational) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    /// Compares the numbers by value.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::rational::Rational;
    ///
    /// let mut v = vec![Rational::from(1) / 3u32, Rational::from(-1) / 2u32, Rational::from(1) / 4u32];
    /// v.sort();
    /// assert_eq!(v.iter().map(|r| r.to_string()).collect::<Vec<_>>(), ["-1/2", "1/4", "1/3"]);
    /// ```
    fn cmp(&self, other: &Rational) -> Ordering {
        if let (Some(x), Some(y)) = (Small::of(self), Small::of(other)) {
            return small::cmp(x, y);
        }
        cross(&self.num, &self.den, &other.num, &other.den)
    }
}

/// The order of `a/b` and `c/d`, for positive `b` and `d`.
fn cross(a: &Number, b: &Number, c: &Number, d: &Number) -> Ordering {
    let (sa, sc) = (sign(a), sign(c));
    if sa != sc {
        return sa.cmp(&sc);
    }
    if sa == 0 {
        return Ordering::Equal;
    }
    let (b1, d1) = (is_one(b), is_one(d));
    if b1 && d1 {
        return a.cmp(c);
    }
    // `|a|·d` and `|c|·b` have `bits(a) + bits(d) - 1` or as many bits as that and one more:
    // the lengths decide when they differ by two or more.
    let x = a.bits() + d.bits();
    let y = c.bits() + b.bits();
    let magnitudes = if x > y + 1 {
        Ordering::Greater
    } else if y > x + 1 {
        Ordering::Less
    } else {
        let left = if d1 { a.abs() } else { &a.abs() * d };
        let right = if b1 { c.abs() } else { &c.abs() * b };
        left.cmp(&right)
    };
    if sa < 0 {
        magnitudes.reverse()
    } else {
        magnitudes
    }
}

/// The sign of `n` as `-1`, `0` or `1`.
#[inline]
fn sign(n: &Number) -> i8 {
    if n.is_zero() {
        0
    } else if n.is_negative() {
        -1
    } else {
        1
    }
}

impl PartialEq<Number> for Rational {
    /// Whether the rational is the integer `n`.
    #[inline]
    fn eq(&self, n: &Number) -> bool {
        is_one(&self.den) && self.num == *n
    }
}

impl PartialEq<Rational> for Number {
    /// Whether the integer is the rational `r`.
    #[inline]
    fn eq(&self, r: &Rational) -> bool {
        r == self
    }
}

impl PartialOrd<Number> for Rational {
    /// Compares the rational with the integer `n` by value.
    fn partial_cmp(&self, n: &Number) -> Option<Ordering> {
        if is_one(&self.den) {
            return Some(self.num.cmp(n));
        }
        if let (Some(x), Some(y)) = (Small::of(self), Small::integer(n)) {
            return Some(small::cmp(x, y));
        }
        Some(cross(&self.num, &self.den, n, &super::one(n.byte_order())))
    }
}

impl PartialOrd<Rational> for Number {
    /// Compares the integer with the rational `r` by value.
    #[inline]
    fn partial_cmp(&self, r: &Rational) -> Option<Ordering> {
        r.partial_cmp(self).map(Ordering::reverse)
    }
}

macro_rules! primitive_cmp {
    ($($t:ty),*) => {$(
        impl PartialEq<$t> for Rational {
            #[doc = concat!("Whether the rational is the `", stringify!($t), "` `v`.")]
            #[inline]
            fn eq(&self, v: &$t) -> bool {
                is_one(&self.den) && self.num == *v
            }
        }

        impl PartialOrd<$t> for Rational {
            #[doc = concat!("Compares the rational with the `", stringify!($t), "` `v` by value.")]
            fn partial_cmp(&self, v: &$t) -> Option<Ordering> {
                let (negative, m) = v.parts();
                if m >> 64 == 0
                    && let Some(x) = Small::of(self)
                {
                    let y = Small { negative: negative && m != 0, num: m as u64, den: 1 };
                    return Some(small::cmp(x, y));
                }
                self.partial_cmp(&super::arith::primitive_number(negative, m))
            }
        }
    )*};
}

primitive_cmp!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);
