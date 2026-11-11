//! The register paths of the arithmetic, for rationals whose parts all fit in 64 bits: the parts
//! are read as `u64`s with their signs, combined in `u128`s, and the results written back in the
//! byte order of the left operand, in its vectors when it is assigned to.

use super::Rational;
use crate::math::numbers::{ByteOrder, Magnitude, Number};

/// A rational with both parts of at most 64 bits: the sign, the numerator's magnitude and the
/// denominator.
#[derive(Clone, Copy)]
pub(super) struct Small {
    pub(super) negative: bool,
    pub(super) num: u64,
    pub(super) den: u64,
}

/// The magnitude of `n` when it fits in 64 bits.
#[inline(always)]
fn word(n: &Number) -> Option<u64> {
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
        let num = word(n)?;
        Some(Small {
            negative: n.negative && num != 0,
            num,
            den: 1,
        })
    }
}

/// A result of the register paths: the sign, the numerator and the denominator, in lowest terms,
/// the denominator positive; zero is `0/1` and not negative.
#[derive(Clone, Copy, Debug)]
pub(super) struct Wide {
    pub(super) negative: bool,
    pub(super) num: u128,
    pub(super) den: u128,
}

/// The greatest common divisor of two `u64`s by the binary method: `gcd(x, 0) = x`.
#[inline]
pub(super) fn gcd(mut x: u64, mut y: u64) -> u64 {
    if x == 0 {
        return y;
    }
    if y == 0 {
        return x;
    }
    let shift = (x | y).trailing_zeros();
    x >>= x.trailing_zeros();
    loop {
        y >>= y.trailing_zeros();
        if x > y {
            core::mem::swap(&mut x, &mut y);
        }
        y -= x;
        if y == 0 {
            return x << shift;
        }
    }
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

/// `a + b`, `b` taken with the sign `negative_b`, or `None` when the numerator would not fit in
/// 128 bits.
#[inline]
pub(super) fn add(a: Small, b: Small, negative_b: bool) -> Option<Wide> {
    let nb = b.negative ^ negative_b && b.num != 0;
    if a.den == b.den {
        // `(a ± b)/d`, reduced by the divisors of `d`.
        let (negative, num) = signed_sum(a.negative, u128::from(a.num), nb, u128::from(b.num))?;
        if num == 0 {
            return Some(ZERO);
        }
        let g = gcd_wide(num, a.den);
        return Some(Wide {
            negative,
            num: div_wide(num, g),
            den: u128::from(a.den / g),
        });
    }
    let g = gcd(a.den, b.den);
    let (s, t) = (a.den / g, b.den / g);
    let x = u128::from(a.num) * u128::from(t);
    let y = u128::from(b.num) * u128::from(s);
    let (negative, num) = signed_sum(a.negative, x, nb, y)?;
    if num == 0 {
        return Some(ZERO);
    }
    let g2 = gcd_wide(num, g);
    Some(Wide {
        negative,
        num: div_wide(num, g2),
        den: u128::from(s) * u128::from(b.den / g2),
    })
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
    den: 1,
};

/// `±x ± y` as a sign and a magnitude, or `None` when it does not fit in 128 bits; zero is not
/// negative.
#[inline(always)]
fn signed_sum(nx: bool, x: u128, ny: bool, y: u128) -> Option<(bool, u128)> {
    if nx == ny {
        Some((nx, x.checked_add(y)?))
    } else if x >= y {
        Some((nx && x != y, x - y))
    } else {
        Some((ny, y - x))
    }
}

/// Writes the magnitude `m` with the sign `negative` over `n`, in the byte order `order`, in the
/// shortest form, in `n`'s vector when it has room.
#[inline(always)]
pub(super) fn put(n: &mut Number, order: ByteOrder, negative: bool, m: u128) {
    let len = 16 - (m | 1).leading_zeros() as usize / 8;
    let v = &mut n.value;
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

impl Wide {
    /// Writes the result over `r`, in the byte order `order`, in its vectors when they have room.
    #[inline(always)]
    pub(super) fn write(self, r: &mut Rational, order: ByteOrder) {
        put(&mut r.num, order, self.negative, self.num);
        put(&mut r.den, order, false, self.den);
    }

    /// The result as a new rational in the byte order `order`.
    #[inline(always)]
    pub(super) fn rational(self, order: ByteOrder) -> Rational {
        Rational {
            num: number(order, self.negative, self.num),
            den: number(order, false, self.den),
        }
    }
}
