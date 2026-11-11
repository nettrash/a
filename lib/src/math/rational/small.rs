//! The register paths of the arithmetic, for rationals whose parts all fit in 64 bits: the parts
//! are read as `u64`s with their signs, combined in `u128`s, and the results written back in the
//! byte order of the left operand, in its vectors when it is assigned to.

use super::Rational;
use crate::math::numbers::{ByteOrder, Magnitude, Number, fraction};

/// A rational with both parts of at most 64 bits: the sign, the numerator's magnitude and the
/// denominator.
#[derive(Clone, Copy)]
pub(super) struct Small {
    pub(super) negative: bool,
    pub(super) num: u64,
    pub(super) den: u64,
}

/// The magnitude of `n`, a part of a rational and so in the shortest form, when it fits in 64 bits:
/// its vector's length tells, with no search for the significant bytes.
///
/// One to three bytes are read as the first, the middle and the last, which overlap below three,
/// and four to eight as two overlapping groups of four: two paths for the lengths of short parts,
/// where one per length would take a branch that random lengths mispredict. Measured on an Apple
/// M4 Max, the comparison of two rationals of parts below `2^16` took 37 cycles with the reads of
/// numbers and 18 so.
#[inline(always)]
fn word(n: &Number) -> Option<u64> {
    // Zero may be `[0]`, or empty as `Rational::default` leaves it.
    debug_assert!(n.value.len() <= 1 || n.value.len() == Magnitude::of(n).bytes.len());
    let v = n.value.as_slice();
    let len = v.len();
    if len > 8 {
        return None;
    }
    if len == 0 {
        return Some(0);
    }
    let be = n.byte_order == ByteOrder::BigEndian;
    // Byte `i` from the least significant one.
    let at = |i: usize| u64::from(if be { v[len - 1 - i] } else { v[i] });
    if len < 4 {
        let (mid, top) = (len / 2, len - 1);
        return Some(at(0) | (at(mid) << (8 * mid)) | (at(top) << (8 * top)));
    }
    let (lo, hi) = if be {
        (
            u32::from_be_bytes(v[len - 4..].try_into().unwrap()),
            u32::from_be_bytes(v[..4].try_into().unwrap()),
        )
    } else {
        (
            u32::from_le_bytes(v[..4].try_into().unwrap()),
            u32::from_le_bytes(v[len - 4..].try_into().unwrap()),
        )
    };
    // `hi` holds bytes `len - 4..len`, which agree with `lo` where they overlap.
    Some(u64::from(lo) | (u64::from(hi) << (8 * (len - 4))))
}

/// The magnitude of any number when it fits in 64 bits.
#[inline(always)]
fn any_word(n: &Number) -> Option<u64> {
    let m = Magnitude::of(n);
    if m.bytes.len() <= 8 {
        Some(m.small() as u64)
    } else {
        None
    }
}

impl Small {
    /// The parts of `r`, when both fit in 64 bits.
    #[inline(always)]
    pub(super) fn of(r: &Rational) -> Option<Small> {
        let num = word(&r.num)?;
        let den = word(&r.den)?;
        Some(Small {
            negative: r.num.negative && num != 0,
            num,
            den,
        })
    }

    /// The integer `n` as a fraction over 1, when it fits in 64 bits.
    #[inline(always)]
    pub(super) fn integer(n: &Number) -> Option<Small> {
        let num = any_word(n)?;
        Some(Small {
            negative: n.negative && num != 0,
            num,
            den: 1,
        })
    }
}

/// A result of the register paths: the sign, the numerator and the denominator, in lowest terms,
/// the denominator positive; zero is `0/1` and not negative. The numerator of a sum can have 129
/// bits: `top` is its bit 128.
#[derive(Clone, Copy, Debug)]
pub(super) struct Wide {
    pub(super) negative: bool,
    pub(super) num: u128,
    pub(super) top: bool,
    pub(super) den: u128,
}

/// The greatest common divisor of two `u64`s, `gcd(x, 0) = x`: [`Number::gcd`]'s method for one
/// limb, the binary method after a remainder when one is longer by 16 bits or more.
#[inline(always)]
pub(super) fn gcd(x: u64, y: u64) -> u64 {
    fraction::gcd_words(x, y)
}

/// `gcd(x, g)` for a `u128` and a non-zero `u64`.
#[inline]
fn gcd_wide(x: u128, g: u64) -> u64 {
    if g == 1 {
        return 1;
    }
    let r = if x >> 64 == 0 {
        x as u64 % g
    } else {
        (x % u128::from(g)) as u64
    };
    gcd(r, g)
}

/// `x / d` for a `u128` and a non-zero `u64`.
#[inline]
fn div_wide(x: u128, d: u64) -> u128 {
    if d == 1 {
        x
    } else if x >> 64 == 0 {
        u128::from(x as u64 / d)
    } else {
        x / u128::from(d)
    }
}

/// `a + b`, `b` taken with the sign `negative_b`.
#[inline]
pub(super) fn add(a: Small, b: Small, negative_b: bool) -> Wide {
    let nb = b.negative ^ negative_b && b.num != 0;
    if a.den == b.den {
        // `(a ± b)/d`, reduced by the divisors of `d`; the sum of two limbs has at most 65 bits.
        let (negative, num, _) = signed_sum(a.negative, u128::from(a.num), nb, u128::from(b.num));
        if num == 0 {
            return ZERO;
        }
        let g = gcd_wide(num, a.den);
        return Wide {
            negative,
            num: div_wide(num, g),
            top: false,
            den: u128::from(a.den / g),
        };
    }
    let g = gcd(a.den, b.den);
    let (s, t) = (a.den / g, b.den / g);
    let x = u128::from(a.num) * u128::from(t);
    let y = u128::from(b.num) * u128::from(s);
    let (negative, num, carry) = signed_sum(a.negative, x, nb, y);
    if carry {
        return carried(negative, num, g, s, b.den);
    }
    if num == 0 {
        return ZERO;
    }
    let g2 = gcd_wide(num, g);
    Wide {
        negative,
        num: div_wide(num, g2),
        top: false,
        den: u128::from(s) * u128::from(b.den / g2),
    }
}

/// The sum of [`add`] whose numerator `2^128 + lo` passed 128 bits: reduced by `g2`, its greatest
/// common divisor with `g`, by Hensel's exact division modulo `2^128` (the quotient by `g2 >= 2`
/// is below `2^128`), over `s·(d/g2)`.
#[cold]
#[inline(never)]
fn carried(negative: bool, lo: u128, g: u64, s: u64, d: u64) -> Wide {
    let g2 = if g == 1 {
        1
    } else {
        // `(2^128 + lo) mod g`: `2^128 mod g` is `(2^128 - 1) mod g + 1`, at most `g`.
        let g128 = u128::from(g);
        let r = ((u128::MAX % g128) + 1 + lo % g128) % g128;
        gcd(r as u64, g)
    };
    if g2 == 1 {
        return Wide {
            negative,
            num: lo,
            top: true,
            den: u128::from(s) * u128::from(d),
        };
    }
    let z = g2.trailing_zeros();
    let odd = g2 >> z;
    // The numerator shifted right by the zero bits of `g2`, which it has: `2^(128 - z) + lo >> z`
    // for `z > 0`, and `lo` modulo `2^128` for `z = 0`.
    let shifted = if z == 0 {
        lo
    } else {
        (lo >> z) | (1u128 << (128 - z))
    };
    // The inverse of `odd` modulo `2^128`: modulo `2^64`, then one Newton step.
    let v = u128::from(fraction::inverse_word(odd));
    let v = v.wrapping_mul(2u128.wrapping_sub(u128::from(odd).wrapping_mul(v)));
    Wide {
        negative,
        num: shifted.wrapping_mul(v),
        top: false,
        den: u128::from(s) * u128::from(d / g2),
    }
}

/// `a · b`, or `a / b` when `divide` is set (`b` not zero).
#[inline]
pub(super) fn mul(a: Small, b: Small, divide: bool) -> Wide {
    let (bn, bd) = if divide {
        (b.den, b.num)
    } else {
        (b.num, b.den)
    };
    if a.num == 0 || bn == 0 {
        return ZERO;
    }
    let g1 = gcd(a.num, bd);
    let g2 = gcd(bn, a.den);
    Wide {
        negative: a.negative ^ b.negative,
        num: u128::from(a.num / g1) * u128::from(bn / g2),
        top: false,
        den: u128::from(a.den / g2) * u128::from(bd / g1),
    }
}

/// The order of `a` and `b` by value.
#[inline]
pub(super) fn cmp(a: Small, b: Small) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    match (a.negative, b.negative) {
        (false, true) => Ordering::Greater,
        (true, false) => Ordering::Less,
        (negative, _) => {
            let x = u128::from(a.num) * u128::from(b.den);
            let y = u128::from(b.num) * u128::from(a.den);
            if negative { y.cmp(&x) } else { x.cmp(&y) }
        }
    }
}

/// Zero, `0/1`.
const ZERO: Wide = Wide {
    negative: false,
    num: 0,
    top: false,
    den: 1,
};

/// `±x ± y` as a sign, a magnitude modulo `2^128` and whether it carried out of 128 bits; zero is
/// not negative.
#[inline(always)]
fn signed_sum(nx: bool, x: u128, ny: bool, y: u128) -> (bool, u128, bool) {
    if nx == ny {
        let (m, carry) = x.overflowing_add(y);
        (nx, m, carry)
    } else if x >= y {
        (nx && x != y, x - y, false)
    } else {
        (ny, y - x, false)
    }
}

/// Writes the magnitude `m` with the sign `negative` over `n`, in the byte order `order`, in the
/// shortest form, in `n`'s vector when it has room.
#[inline(always)]
pub(super) fn put(n: &mut Number, order: ByteOrder, negative: bool, m: u128) {
    let len = 16 - (m | 1).leading_zeros() as usize / 8;
    let v = &mut n.value;
    if v.capacity() < len {
        // A new vector with room for any result of the register paths, rather than a reallocation
        // of one that copies the bytes about to be replaced.
        *v = Vec::with_capacity(16);
    }
    v.clear();
    match order {
        ByteOrder::BigEndian => v.extend_from_slice(&m.to_be_bytes()[16 - len..]),
        ByteOrder::LittleEndian => v.extend_from_slice(&m.to_le_bytes()[..len]),
    }
    n.negative = negative && m != 0;
    n.byte_order = order;
}

/// A new number of the magnitude `m` with the sign `negative`, in the byte order `order`, in the
/// shortest form.
#[inline(always)]
pub(super) fn number(order: ByteOrder, negative: bool, m: u128) -> Number {
    let mut n = Number {
        value: Vec::with_capacity(16),
        negative: false,
        byte_order: order,
    };
    put(&mut n, order, negative, m);
    n
}

/// Writes the magnitude `2^128 + m` with the sign `negative` over `n`, in the byte order `order`,
/// 17 bytes, in `n`'s vector when it has room.
#[cold]
#[inline(never)]
fn put_top(n: &mut Number, order: ByteOrder, negative: bool, m: u128) {
    let v = &mut n.value;
    if v.capacity() < 17 {
        *v = Vec::with_capacity(24);
    }
    v.clear();
    match order {
        ByteOrder::BigEndian => {
            v.push(1);
            v.extend_from_slice(&m.to_be_bytes());
        }
        ByteOrder::LittleEndian => {
            v.extend_from_slice(&m.to_le_bytes());
            v.push(1);
        }
    }
    n.negative = negative;
    n.byte_order = order;
}

impl Wide {
    /// Writes the result over `r`, in the byte order `order`, in its vectors when they have room.
    #[inline(always)]
    pub(super) fn write(self, r: &mut Rational, order: ByteOrder) {
        if self.top {
            put_top(&mut r.num, order, self.negative, self.num);
        } else {
            put(&mut r.num, order, self.negative, self.num);
        }
        put(&mut r.den, order, false, self.den);
    }

    /// The result as a new rational in the byte order `order`.
    #[inline(always)]
    pub(super) fn rational(self, order: ByteOrder) -> Rational {
        let mut num = number(order, self.negative, self.num);
        if self.top {
            put_top(&mut num, order, self.negative, self.num);
        }
        Rational {
            num,
            den: number(order, false, self.den),
        }
    }
}
