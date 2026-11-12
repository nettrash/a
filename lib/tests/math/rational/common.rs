//! What the tests of rationals share: random rationals of every size and byte order, and the
//! checks of the form every rational keeps.

use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;

/// Scales a randomised test down when it runs under Miri.
pub(super) fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Small deterministic xorshift generator.
pub(super) fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A random number below `n`.
pub(super) fn below(state: &mut u64, n: u64) -> u64 {
    xorshift(state) % n
}

/// A random magnitude of up to `max` bytes, least significant first, often with long runs of zero
/// and `0xFF` bytes; empty for 0.
pub(super) fn magnitude(state: &mut u64, max: usize) -> Vec<u8> {
    let len = below(state, max as u64 + 1) as usize;
    let mut bytes = Vec::with_capacity(len);
    while bytes.len() < len {
        let run = (1 + below(state, 12) as usize).min(len - bytes.len());
        match below(state, 4) {
            0 => bytes.resize(bytes.len() + run, 0),
            1 => bytes.resize(bytes.len() + run, 0xFF),
            _ => bytes.extend((0..run).map(|_| xorshift(state) as u8)),
        }
    }
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    bytes
}

/// The byte order for the pick `k`.
pub(super) fn order(k: u64) -> ByteOrder {
    if k & 1 == 0 {
        ByteOrder::LittleEndian
    } else {
        ByteOrder::BigEndian
    }
}

/// A number of the magnitude `m` (least significant first), the sign `negative` and the byte
/// order `order`.
pub(super) fn number(m: &[u8], negative: bool, order: ByteOrder) -> Number {
    let mut v = m.to_vec();
    if order == ByteOrder::BigEndian {
        v.reverse();
    }
    Number::new_signed(v, negative, order)
}

/// A random integer of up to `max` bytes, of a random sign and byte order.
pub(super) fn integer(state: &mut u64, max: usize) -> Number {
    let m = magnitude(state, max);
    let negative = below(state, 2) == 1;
    number(&m, negative, order(xorshift(state)))
}

/// A random rational: the parts mostly of up to 8 bytes (the register path), else of up to 16, or
/// up to `max`; a common factor in a fraction of them, so that the reduction has work to do.
pub(super) fn rational(state: &mut u64, max: usize) -> Rational {
    let limit = match below(state, 4) {
        0 | 1 => 8,
        2 => 16,
        _ => max,
    };
    let mut num = integer(state, limit);
    let mut den = Number::from(0u8);
    while den.is_zero() {
        den = integer(state, limit);
    }
    if below(state, 3) == 0 {
        let f = integer(state, limit.min(9));
        if !f.is_zero() {
            num = &num * &f;
            den = &den * &f;
        }
    }
    let den = if below(state, 2) == 0 { den } else { -den };
    let r = Rational::new(num, den);
    check(&r);
    r
}

/// Asserts the form every rational keeps: a positive denominator prime to the numerator, both in
/// the shortest form and the same byte order, and zero as `0/1`.
#[track_caller]
pub(super) fn check(r: &Rational) {
    let (n, d) = (r.numer(), r.denom());
    assert!(d.is_positive(), "denominator not positive: {r:?}");
    assert_eq!(
        n.byte_order(),
        d.byte_order(),
        "parts in different byte orders: {r:?}"
    );
    assert_eq!(n.gcd(d), 1, "not in lowest terms: {r:?}");
    for p in [n, d] {
        assert!(!p.value.is_empty(), "an empty part: {r:?}");
        let top = match p.byte_order() {
            ByteOrder::BigEndian => p.value[0],
            ByteOrder::LittleEndian => *p.value.last().unwrap(),
        };
        assert!(
            top != 0 || p.value.len() == 1,
            "a part not in the shortest form: {r:?}"
        );
        assert!(!(p.is_zero() && p.is_negative()), "a negative zero: {r:?}");
    }
    if n.is_zero() {
        assert_eq!(*d, 1, "zero over {d}");
    }
}

/// The fraction `a/b` reduced by `Rational::new` from its unreduced parts, the reference the
/// arithmetic is checked against.
pub(super) fn fraction(a: Number, b: Number) -> Rational {
    Rational::new(a, b)
}
