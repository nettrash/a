//! What the tests of the curves share: numbers, a generator of random ones, and the textbook
//! formulas on affine coordinates that the curves are checked against.

use a::math::numbers::{ByteOrder, Number};

/// A point as the reference holds it: `None` at infinity.
pub(super) type Affine = Option<(Number, Number)>;

/// Scales a test down when it runs under Miri.
pub(super) fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// The number `x`.
pub(super) fn n(x: i64) -> Number {
    Number::new_from_i64(x, ByteOrder::BigEndian)
}

/// The number in hexadecimal, most significant digit first.
pub(super) fn hex(s: &str) -> Number {
    Number::from_str_radix(s, 16, ByteOrder::BigEndian).unwrap()
}

/// The number in the hexadecimal bytes `s`, the least significant byte first.
pub(super) fn hex_le(s: &str) -> Number {
    let bytes = (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    Number::new(bytes, ByteOrder::LittleEndian)
}

/// Small deterministic generator.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    /// The next 64 bits.
    pub(super) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A random number of `bytes` bytes.
    pub(super) fn number(&mut self, bytes: usize) -> Number {
        Number::new(
            (0..bytes).map(|_| self.next() as u8).collect(),
            ByteOrder::BigEndian,
        )
    }

    /// A random number below `m`.
    pub(super) fn below(&mut self, m: &Number) -> Number {
        self.number(m.value.len() + 8).rem_euclid(m)
    }
}

/// `x / y` modulo `p`.
fn div(x: &Number, y: &Number, p: &Number) -> Number {
    (x * y.rem_euclid(p).mod_inverse(p).unwrap()).rem_euclid(p)
}

/// The sum on `y^2 = x^3 + a x + b`, or, with `b_coefficient`, on the Montgomery curve
/// `B y^2 = x^3 + A x^2 + x`: the slope of the chord or the tangent.
fn chord(
    p1: &Affine,
    p2: &Affine,
    p: &Number,
    slope_of_tangent: impl Fn(&Number, &Number) -> Number,
    third: impl Fn(&Number, &Number, &Number) -> Number,
) -> Affine {
    let (Some((x1, y1)), Some((x2, y2))) = (p1, p2) else {
        return p1.clone().or(p2.clone());
    };
    let lambda = if x1 == x2 {
        if (y1 + y2).rem_euclid(p).is_zero() {
            return None;
        }
        slope_of_tangent(x1, y1)
    } else {
        div(&(y2 - y1), &(x2 - x1), p)
    };
    let x3 = third(&lambda, x1, x2).rem_euclid(p);
    let y3 = (&lambda * &(x1 - &x3) - y1).rem_euclid(p);
    Some((x3, y3))
}

/// The sum on `y^2 = x^3 + a x + b` modulo `p`.
pub(super) fn add_w(p1: &Affine, p2: &Affine, a: &Number, p: &Number) -> Affine {
    chord(
        p1,
        p2,
        p,
        |x, y| div(&(x * x * 3u8 + a), &(y * 2u8), p),
        |l, x1, x2| l * l - x1 - x2,
    )
}

/// The sum on `B y^2 = x^3 + A x^2 + x` modulo `p`.
pub(super) fn add_m(p1: &Affine, p2: &Affine, a: &Number, b: &Number, p: &Number) -> Affine {
    chord(
        p1,
        p2,
        p,
        |x, y| div(&(x * x * 3u8 + x * a * 2u8 + 1u8), &(b * y * 2u8), p),
        |l, x1, x2| b * l * l - a - x1 - x2,
    )
}

/// The sum on `a x^2 + y^2 = 1 + d x^2 y^2` modulo `p`.
pub(super) fn add_e(
    p1: &(Number, Number),
    p2: &(Number, Number),
    a: &Number,
    d: &Number,
    p: &Number,
) -> (Number, Number) {
    let (x1, y1) = p1;
    let (x2, y2) = p2;
    let t = (d * x1 * x2 * y1 * y2).rem_euclid(p);
    let x = div(&(x1 * y2 + y1 * x2), &(&t + 1u8), p);
    let y = div(&(y1 * y2 - a * x1 * x2), &(1u8 - &t), p);
    (x, y)
}

/// `k p` by doublings and additions of `add`, for a positive `k`.
pub(super) fn times<T: Clone>(p: &T, k: &Number, zero: T, add: impl Fn(&T, &T) -> T) -> T {
    let mut r = zero;
    for i in (0..k.bits()).rev() {
        r = add(&r, &r);
        if k.bit(i) {
            r = add(&r, p);
        }
    }
    r
}
