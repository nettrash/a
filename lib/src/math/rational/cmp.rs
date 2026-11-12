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
use super::{Rational, Repr, is_one};
use crate::math::numbers::{ByteOrder, Magnitude, Number, cmp_magnitudes};
use core::cmp::Ordering;
use core::hash::{Hash, Hasher};

impl PartialEq for Rational {
    /// Whether the rationals are the same number: whether their parts are equal. A number in
    /// words, both parts in 64 bits, is never equal to one in numbers.
    #[inline]
    fn eq(&self, other: &Rational) -> bool {
        match (self.word(), other.word()) {
            (Some(x), Some(y)) => (x.num == y.num) & (x.den == y.den) & (x.negative == y.negative),
            (None, None) => {
                let ((a, b), (c, d)) = (big(self), big(other));
                a == c && b == d
            }
            _ => false,
        }
    }
}

/// The numbers of a rational not in words.
#[inline(always)]
fn big(r: &Rational) -> (&Number, &Number) {
    r.numbers().unwrap_or_else(|| unreachable!())
}

impl Eq for Rational {}

impl Hash for Rational {
    /// Hashes the number by value, so that equal rationals hash alike whatever their byte orders:
    /// the numerator, then the denominator, as [`Number`] hashes them.
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.word() {
            Some(x) => {
                hash_word(state, x.negative, x.num);
                hash_word(state, false, x.den);
            }
            None => {
                let (a, b) = big(self);
                a.hash(state);
                b.hash(state);
            }
        }
    }
}

/// Hashes the magnitude `m` with the sign `negative` as [`Number`] hashes a number of that value:
/// the sign, the significant length, then the significant bytes, the least significant first.
#[inline(always)]
fn hash_word<H: Hasher>(state: &mut H, negative: bool, m: u64) {
    let len = 8 - (m.leading_zeros() / 8) as usize;
    state.write_u8(u8::from(negative));
    state.write_usize(len);
    if len > 0 {
        state.write(&m.to_le_bytes()[..len]);
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
        match (&self.repr, &other.repr) {
            (Repr::Big(x), Repr::Big(y)) => cross(
                Magnitude::of(&x.num),
                Magnitude::of(&x.den),
                Magnitude::of(&y.num),
                Some(Magnitude::of(&y.den)),
            ),
            _ => match (Small::of(self), Small::of(other)) {
                (Some(x), Some(y)) => small::cmp(x, y),
                _ => mixed(self, other),
            },
        }
    }
}

/// The order of a rational in words and one in numbers, either way round.
#[inline(never)]
fn mixed(x: &Rational, y: &Rational) -> Ordering {
    let ((a, b), (c, d)) = (x.magnitudes(), y.magnitudes());
    cross(a, b, c, Some(d))
}

/// The order of `a/b` and `c/d`, given by the magnitudes of the parts with the numerators' signs,
/// for positive `b` and `d`, `d` given as `None` for 1. Inlined, as the bounds of
/// [`cross_magnitudes`] decide in a few dozen cycles, in which a call and the magnitudes passed on
/// the stack took a fifth more.
#[inline(always)]
fn cross(a: Magnitude, b: Magnitude, c: Magnitude, d: Option<Magnitude>) -> Ordering {
    let (x, y) = (a, c);
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
        (false, false) => cross_magnitudes(b, d, x, y),
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
#[inline(always)]
fn cross_magnitudes(b: Magnitude, d: Option<Magnitude>, x: Magnitude, y: Magnitude) -> Ordering {
    let (ha, hb, hc, hd) = (
        Head::of(x),
        Head::of(b),
        Head::of(y),
        d.map_or(Head::ONE, Head::of),
    );
    let (lo1, hi1, e1) = ha.times(hd);
    let (lo2, hi2, e2) = hc.times(hb);
    if scaled_cmp(lo1, e1, hi2, e2) == Ordering::Greater {
        return Ordering::Greater;
    }
    if scaled_cmp(hi1, e1, lo2, e2) == Ordering::Less {
        return Ordering::Less;
    }
    exact(x, b, y, d)
}

/// The order of `|a|·d` and `|c|·b` by the products, out of line: the bounds decide nearly
/// every comparison. Equal fractions are told by their parts, with no product.
#[cold]
#[inline(never)]
fn exact(a: Magnitude, b: Magnitude, c: Magnitude, d: Option<Magnitude>) -> Ordering {
    let one = |m: Magnitude| m.bytes == [1];
    let d = d.filter(|&d| !one(d));
    let same = |x: Magnitude, y: Magnitude| cmp_magnitudes(x, y) == Ordering::Equal;
    if same(a, c) && d.map_or(one(b), |d| same(b, d)) {
        return Ordering::Equal;
    }
    let left = match d {
        None => positive(a),
        Some(d) => &positive(a) * &positive(d),
    };
    let right = if one(b) {
        positive(c)
    } else {
        &positive(c) * &positive(b)
    };
    left.cmp(&right)
}

/// The magnitude `m` as a positive number.
fn positive(m: Magnitude) -> Number {
    Number {
        value: m.bytes.to_vec(),
        negative: false,
        byte_order: if m.big_endian {
            ByteOrder::BigEndian
        } else {
            ByteOrder::LittleEndian
        },
    }
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
        match self.word() {
            Some(x) => {
                x.den == 1
                    && Small::integer(n).is_some_and(|y| y.num == x.num && y.negative == x.negative)
            }
            None => {
                let (a, b) = big(self);
                is_one(b) && a == n
            }
        }
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
        if let (Some(x), Some(y)) = (Small::of(self), Small::integer(n)) {
            return Some(small::cmp(x, y));
        }
        if let Some((a, b)) = self.numbers()
            && is_one(b)
        {
            return Some(a.cmp(n));
        }
        let (a, b) = self.magnitudes();
        Some(cross(a, b, Magnitude::of(n), None))
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
                let (negative, m) = v.parts();
                match self.word() {
                    Some(x) => (x.den == 1) & (u128::from(x.num) == m) & (x.negative == (negative && m != 0)),
                    // A number in numbers has a part of more than 64 bits.
                    None => {
                        let (a, b) = big(self);
                        is_one(b) && *a == *v
                    }
                }
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
