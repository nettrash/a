//! Rationals compared by value: `==`, `<` and the rest with each other, with [`Number`]s on
//! either side and with the primitive integers on the right, and a hash that agrees with `==`.
//!
//! As every rational has one form, lowest terms with a positive denominator, two are equal when
//! their parts are, whatever their byte orders. The order compares `a/b` with `c/d` by their signs,
//! then, when the signs are the same, `a·d` with `c·b`: in registers when the parts fit in 64
//! bits, and otherwise first by bounds of the products from the top 63 bits of each part, which
//! tell apart every pair but those that agree to about 120 bits, in a few dozen cycles whatever
//! the lengths; only those are multiplied out.

use super::arith::Primitive;
use super::small::{self, Small};
use super::{Rational, is_one};
use crate::math::numbers::{Magnitude, Number};
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
        cross(&self.num, &self.den, &other.num, Some(&other.den))
    }
}

/// The order of `a/b` and `c/d`, for positive `b` and `d`, `d` given as `None` for 1.
fn cross(a: &Number, b: &Number, c: &Number, d: Option<&Number>) -> Ordering {
    let (x, y) = (Magnitude::of(a), Magnitude::of(c));
    if x.negative != y.negative {
        return if x.negative {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let magnitudes = match (x.bytes.is_empty(), y.bytes.is_empty()) {
        (true, true) => return Ordering::Equal,
        // Zero against a number of the same sign, which is then positive.
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => cross_magnitudes(a, b, c, d, x, y),
    };
    if x.negative {
        magnitudes.reverse()
    } else {
        magnitudes
    }
}

/// The order of `|a|/b` and `|c|/d` for `a` and `c` not zero, of the magnitudes `x` and `y`.
///
/// The products `|a|·d` and `|c|·b` are first bounded from the top 63 bits of each part: `p` of
/// `L > 63` bits lies in `[h, h + 1]·2^(L - 63)` for `h` its top 63 bits, and a part of at most 63
/// bits is `h` itself; the products of the bounds, in 126 bits, with the sums of the exponents,
/// bound the products. When the bounds are apart, which they are unless the two numbers agree to
/// about 120 bits, they decide, in a few dozen cycles whatever the lengths; otherwise the products
/// are computed. Measured on an Apple M4 Max on random positive rationals with parts of the same
/// length, 2 limbs took 245 cycles a comparison by the products and 1,024 limbs 258,000, and 35 and
/// 53 by the bounds; GMP 6.3's `mpq_cmp` 56 and 524,000.
#[inline]
fn cross_magnitudes(
    a: &Number,
    b: &Number,
    c: &Number,
    d: Option<&Number>,
    x: Magnitude,
    y: Magnitude,
) -> Ordering {
    let (ha, hb, hc, hd) = (
        Head::of(x),
        Head::of(Magnitude::of(b)),
        Head::of(y),
        d.map_or(Head::ONE, |d| Head::of(Magnitude::of(d))),
    );
    let (lo1, hi1, e1) = ha.times(hd);
    let (lo2, hi2, e2) = hc.times(hb);
    if scaled_cmp(lo1, e1, hi2, e2) == Ordering::Greater {
        return Ordering::Greater;
    }
    if scaled_cmp(hi1, e1, lo2, e2) == Ordering::Less {
        return Ordering::Less;
    }
    exact(a, b, c, d)
}

/// The order of `|a|·d` and `|c|·b` by the products, out of line: the bounds decide nearly
/// every comparison.
#[cold]
#[inline(never)]
fn exact(a: &Number, b: &Number, c: &Number, d: Option<&Number>) -> Ordering {
    let d = d.filter(|d| !is_one(d));
    if a == c && d.map_or(is_one(b), |d| b == d) {
        return Ordering::Equal;
    }
    let b1 = is_one(b);
    let left = match d {
        None => a.abs(),
        Some(d) => &a.abs() * d,
    };
    let right = if b1 { c.abs() } else { &c.abs() * b };
    left.cmp(&right)
}

/// The top of a magnitude not zero: the magnitude lies in `[m, m + inexact]·2^exp`, with `m` below
/// `2^63`.
#[derive(Clone, Copy)]
struct Head {
    m: u64,
    exp: usize,
    inexact: bool,
}

impl Head {
    /// The top of 1.
    const ONE: Head = Head {
        m: 1,
        exp: 0,
        inexact: false,
    };

    /// The top of the magnitude `x`, not zero, of significant bytes only.
    #[inline(always)]
    fn of(x: Magnitude) -> Head {
        let (v, n) = (x.bytes, x.bytes.len());
        if n <= 8 {
            let m = x.small() as u64;
            return if m >> 63 == 0 {
                Head {
                    m,
                    exp: 0,
                    inexact: false,
                }
            } else {
                Head {
                    m: m >> 1,
                    exp: 1,
                    inexact: m & 1 != 0,
                }
            };
        }
        // The eight most significant bytes, whose top one is not zero, and the byte below them.
        let (top, next) = if x.big_endian {
            (u64::from_be_bytes(v[..8].try_into().unwrap()), v[8])
        } else {
            (u64::from_le_bytes(v[n - 8..].try_into().unwrap()), v[n - 9])
        };
        let z = top.leading_zeros();
        // The top 64 bits of the magnitude of `8n - z` bits; `z` is below 8.
        let u = (top << z) | (u64::from(next) >> (8 - z));
        Head {
            m: u >> 1,
            exp: 8 * n - z as usize - 63,
            inexact: true,
        }
    }

    /// The bounds of the product of the magnitudes of `self` and `o`: the products of the lower
    /// and the upper bounds, below `2^126`, and their exponent.
    #[inline(always)]
    fn times(self, o: Head) -> (u128, u128, usize) {
        let lo = u128::from(self.m) * u128::from(o.m);
        let hi =
            u128::from(self.m + u64::from(self.inexact)) * u128::from(o.m + u64::from(o.inexact));
        (lo, hi, self.exp + o.exp)
    }
}

/// The order of `m1·2^e1` and `m2·2^e2`, for `m1` and `m2` not zero.
#[inline(always)]
fn scaled_cmp(m1: u128, e1: usize, m2: u128, e2: usize) -> Ordering {
    let l1 = 128 - m1.leading_zeros() as usize + e1;
    let l2 = 128 - m2.leading_zeros() as usize + e2;
    if l1 != l2 {
        return l1.cmp(&l2);
    }
    // Of the same length, the one of the larger exponent is the shorter by as much.
    if e1 >= e2 {
        (m1 << (e1 - e2)).cmp(&m2)
    } else {
        m1.cmp(&(m2 << (e2 - e1)))
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
        Some(cross(&self.num, &self.den, n, None))
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
