//! Powers of rationals whose parts fit in 128 bits: both parts raised together in registers, by
//! products by the base limb by limb, the two chains interleaved, and written once into the
//! vectors of the result.

use super::Rational;
use crate::math::numbers::{ByteOrder, Number};

/// The limbs a power of [`powers`] holds, with a limb of room above it: powers of up to 1024 bits.
const LIMBS: usize = 17;

/// The largest exponent [`powers`] takes: past it the squares of binary exponentiation, which
/// [`Number::pow`] takes, need fewer products than the products by the base.
// Measured on an Apple M4 Max (each way forced, parts of 8 to 128 bits, exponents 2 to 48, powers
// of 129 to 1024 bits): the products by the base are faster for parts of 32 to 128 bits to this
// exponent (by up to 2.2 times at 64 bits), slower for 16-bit parts from 14 and for 24-bit ones at
// 16 to 21, whose powers of numbers stay in registers to 512 bits (and took 0.76 to 0.90 of the
// products' time for 16-bit parts). Alternated against 16, 13 took 1.47 to 1.80 times as long for
// parts of 48 and 64 bits at 14 to 16, and 20 or 32 took 1.13 to 1.88 times as long for 16- and
// 24-bit parts from 17, where they took 0.62 to 0.90 of it for longer ones: no other value in
// 4..=32 leaves every length within 2 %.
pub(super) const MAX_EXP: u32 = 16;

/// `±x^k / y^k` for magnitudes `x` and `y` of up to 128 bits, prime to each other, `y > 0`,
/// `2 <= k <= MAX_EXP`, and powers of 128 to 1024 bits: `None` when they may be shorter, whose
/// powers in `u128`s are faster, or longer.
pub(super) fn powers(
    negative: bool,
    x: u128,
    y: u128,
    k: u32,
    order: ByteOrder,
) -> Option<Rational> {
    debug_assert!((2..=MAX_EXP).contains(&k) && x != 0 && y != 0);
    let bits = u64::from(128 - (x | y).leading_zeros()) * u64::from(k);
    if !(129..=64 * (LIMBS as u64 - 1)).contains(&bits) {
        return None;
    }
    let mut a = [0u64; LIMBS];
    let mut b = [0u64; LIMBS];
    (a[0], a[1], b[0], b[1]) = (x as u64, (x >> 64) as u64, y as u64, (y >> 64) as u64);
    let base = ([a[0], a[1]], [b[0], b[1]]);
    let mut len = if (x | y) >> 64 == 0 { 1 } else { 2 };
    if base.0[1] | base.1[1] == 0 {
        for _ in 1..k {
            len = mul_1(&mut a, &mut b, len, base.0[0], base.1[0]);
        }
    } else {
        for _ in 1..k {
            len = mul_2(&mut a, &mut b, len, base.0, base.1);
        }
    }
    if len == 1 {
        return Some(Rational::from_word(negative, a[0], b[0], order));
    }
    Some(Rational::big(
        number(&a[..len], negative, order),
        number(&b[..len], false, order),
    ))
}

/// `a·x` and `b·y` in place, of `len` limbs (the longer of the two, zeros above the shorter):
/// returns their length, the longer one's.
#[inline(always)]
fn mul_1(a: &mut [u64; LIMBS], b: &mut [u64; LIMBS], len: usize, x: u64, y: u64) -> usize {
    let (mut c, mut d) = (0u64, 0u64);
    for i in 0..len {
        let t = u128::from(a[i]) * u128::from(x) + u128::from(c);
        let u = u128::from(b[i]) * u128::from(y) + u128::from(d);
        (a[i], c) = (t as u64, (t >> 64) as u64);
        (b[i], d) = (u as u64, (u >> 64) as u64);
    }
    a[len] = c;
    b[len] = d;
    len + usize::from(c | d != 0)
}

/// `a·x` and `b·y` for bases of two limbs, of `len` limbs as for [`mul_1`]: returns their length.
#[inline(always)]
fn mul_2(
    a: &mut [u64; LIMBS],
    b: &mut [u64; LIMBS],
    len: usize,
    x: [u64; 2],
    y: [u64; 2],
) -> usize {
    let mut r = [0u64; LIMBS];
    let mut s = [0u64; LIMBS];
    for j in 0..2 {
        let (mut c, mut d) = (0u64, 0u64);
        for i in 0..len {
            let t = u128::from(a[i]) * u128::from(x[j]) + u128::from(r[i + j]) + u128::from(c);
            let u = u128::from(b[i]) * u128::from(y[j]) + u128::from(s[i + j]) + u128::from(d);
            (r[i + j], c) = (t as u64, (t >> 64) as u64);
            (s[i + j], d) = (u as u64, (u >> 64) as u64);
        }
        r[len + j] = c;
        s[len + j] = d;
    }
    (*a, *b) = (r, s);
    let mut n = len + 2;
    while a[n - 1] | b[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// The magnitude of the limbs `l` (zero limbs at the top allowed, not all zero) with the sign
/// `negative`, in the byte order `order`, in the shortest form, in one new vector.
fn number(l: &[u64], negative: bool, order: ByteOrder) -> Number {
    let mut n = l.len();
    while l[n - 1] == 0 {
        n -= 1;
    }
    let top = l[n - 1];
    let len = 8 * (n - 1) + 8 - top.leading_zeros() as usize / 8;
    let mut value = Vec::with_capacity(8 * n);
    match order {
        ByteOrder::LittleEndian => {
            for w in &l[..n] {
                value.extend_from_slice(&w.to_le_bytes());
            }
            value.truncate(len);
        }
        ByteOrder::BigEndian => {
            let skip = 8 * n - len;
            value.extend_from_slice(&top.to_be_bytes()[skip..]);
            for w in l[..n - 1].iter().rev() {
                value.extend_from_slice(&w.to_be_bytes());
            }
        }
    }
    Number {
        value,
        negative,
        byte_order: order,
    }
}
