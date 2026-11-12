//! The conversions of rationals from and to floats and integers: floats exactly, both ways round
//! trips, the special values, and the rounding of rationals checked against the two floats around
//! the result, compared exactly as rationals.

use super::common::{below, check, integer, number, rational, size, xorshift};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::{Rational, TryFromFloatError, TryFromRationalError};

/// `2^k` as a rational, for any `k`.
fn two_to(k: i64) -> Rational {
    let p = Rational::from(Number::from(1) << k.unsigned_abs() as usize);
    if k < 0 { p.recip() } else { p }
}

/// `2^k` as an `f64`, for `k` from -1074 to 1023.
fn pow2(k: i64) -> f64 {
    if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else {
        f64::from_bits(1 << (k + 1074))
    }
}

/// `2^k` as an `f32`, for `k` from -149 to 127.
fn pow2_32(k: i64) -> f32 {
    if k >= -126 {
        f32::from_bits(((k + 127) as u32) << 23)
    } else {
        f32::from_bits(1 << (k + 149))
    }
}

/// The exact value of a finite `f64`.
fn exact(x: f64) -> Rational {
    let r = Rational::try_from(x).unwrap();
    check(&r);
    r
}

/// The exact value of a finite `f32`.
fn exact32(x: f32) -> Rational {
    let r = Rational::try_from(x).unwrap();
    check(&r);
    r
}

/// Asserts that `y` is the `f64` nearest to `r`, ties to even, with the sign of `r` for zero
/// and infinity: compared exactly with the floats on either side of it.
#[track_caller]
fn assert_nearest(r: &Rational, y: f64) {
    assert!(!y.is_nan(), "{r} gave NaN");
    if r.is_zero() {
        assert_eq!(y.to_bits(), 0, "zero gave {y:?}");
        return;
    }
    assert_eq!(
        y.is_sign_negative(),
        r.is_negative(),
        "{r} gave {y:?}: the sign"
    );
    let m = r.abs();
    let y = y.abs();
    if y.is_infinite() {
        // At least halfway from MAX to 2^1024, which is the tie that goes to the even 2^1024.
        let limit = exact(f64::MAX) + two_to(970);
        assert!(m >= limit, "{r} rounded to infinity");
        return;
    }
    let ey = exact(y);
    let up = y.next_up();
    let d = (&m - &ey).abs();
    // The float above, or 2^1024 for MAX.
    let above = if up.is_infinite() {
        two_to(1024)
    } else {
        exact(up)
    };
    let du = (&m - &above).abs();
    assert!(d <= du, "{r} gave {y:?}, but {up:?} is nearer");
    if d == du {
        assert!(y.to_bits() & 1 == 0, "{r}: a tie to the odd {y:?}");
    }
    if y > 0.0 {
        let down = y.next_down();
        let dd = (&m - &exact(down)).abs();
        assert!(d <= dd, "{r} gave {y:?}, but {down:?} is nearer");
        if d == dd {
            assert!(y.to_bits() & 1 == 0, "{r}: a tie to the odd {y:?}");
        }
    }
}

/// [`assert_nearest`] for an `f32`.
#[track_caller]
fn assert_nearest32(r: &Rational, y: f32) {
    assert!(!y.is_nan(), "{r} gave NaN");
    if r.is_zero() {
        assert_eq!(y.to_bits(), 0, "zero gave {y:?}");
        return;
    }
    assert_eq!(
        y.is_sign_negative(),
        r.is_negative(),
        "{r} gave {y:?}: the sign"
    );
    let m = r.abs();
    let y = y.abs();
    if y.is_infinite() {
        let limit = exact32(f32::MAX) + two_to(103);
        assert!(m >= limit, "{r} rounded to infinity");
        return;
    }
    let ey = exact32(y);
    let up = y.next_up();
    let d = (&m - &ey).abs();
    let above = if up.is_infinite() {
        two_to(128)
    } else {
        exact32(up)
    };
    let du = (&m - &above).abs();
    assert!(d <= du, "{r} gave {y:?}, but {up:?} is nearer");
    if d == du {
        assert!(y.to_bits() & 1 == 0, "{r}: a tie to the odd {y:?}");
    }
    if y > 0.0 {
        let down = y.next_down();
        let dd = (&m - &exact32(down)).abs();
        assert!(d <= dd, "{r} gave {y:?}, but {down:?} is nearer");
        if d == dd {
            assert!(y.to_bits() & 1 == 0, "{r}: a tie to the odd {y:?}");
        }
    }
}

/// Both conversions of `r`, checked.
#[track_caller]
fn assert_both(r: &Rational) {
    assert_nearest(r, r.to_f64());
    assert_nearest32(r, r.to_f32());
}

/// `r` in the byte order `order`.
fn in_order(r: &Rational, order: ByteOrder) -> Rational {
    let flip = |n: &Number| {
        let mut v = n.value.clone();
        if n.byte_order() != order {
            v.reverse();
        }
        Number::new_signed(v, n.is_negative(), order)
    };
    Rational::new(flip(r.numer()), flip(r.denom()))
}

#[test]
fn special_floats() {
    assert_eq!(exact(0.0), Rational::from(0));
    assert_eq!(exact(-0.0), Rational::from(0));
    assert!(!exact(-0.0).is_negative());
    assert_eq!(exact(1.0), Rational::from(1));
    assert_eq!(exact(-3.0), Rational::from(-3));
    assert_eq!(exact(0.5), Rational::from(1) / 2u32);
    assert_eq!(exact(-0.375), Rational::from(-3) / 8u32);
    assert_eq!(exact(f64::from_bits(1)), two_to(-1074));
    assert_eq!(exact(-f64::from_bits(1)), -two_to(-1074));
    assert_eq!(
        exact(f64::from_bits((1 << 52) - 1)),
        Rational::from((1u64 << 52) - 1) * two_to(-1074)
    );
    assert_eq!(exact(f64::MIN_POSITIVE), two_to(-1022));
    assert_eq!(
        exact(f64::MAX),
        Rational::from((1u64 << 53) - 1) * two_to(971)
    );
    assert_eq!(exact(f64::MIN), -exact(f64::MAX));
    assert_eq!(exact32(f32::from_bits(1)), two_to(-149));
    assert_eq!(exact32(f32::MIN_POSITIVE), two_to(-126));
    assert_eq!(
        exact32(f32::MAX),
        Rational::from((1u64 << 24) - 1) * two_to(104)
    );
    assert_eq!(exact(1e20), Rational::from(100_000_000_000_000_000_000u128));
    assert_eq!(
        exact(0.1),
        Rational::new(Number::from(3602879701896397u64), Number::from(1) << 55)
    );
    for x in [f64::NAN, -f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(Rational::try_from(x).is_err());
    }
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(Rational::try_from(x).is_err());
    }
    let e: TryFromFloatError = Rational::try_from(f64::NAN).unwrap_err();
    assert!(!e.to_string().is_empty());
    // A rational converts back to these values exactly.
    for x in [
        0.0,
        1.0,
        -1.0,
        0.5,
        f64::from_bits(1),
        f64::from_bits((1 << 52) - 1),
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        f64::EPSILON,
        1e-300,
        1e300,
    ] {
        assert_eq!(exact(x).to_f64().to_bits(), x.to_bits(), "{x:?}");
    }
    assert_eq!(Rational::from(0).to_f64().to_bits(), 0);
    assert_eq!(Rational::from(0).to_f32().to_bits(), 0);
}

#[test]
fn powers_of_two() {
    for k in -1074..=1023 {
        let p = two_to(k);
        assert_eq!(exact(pow2(k)), p, "2^{k}");
        assert_eq!(p.to_f64(), pow2(k), "2^{k}");
        assert_eq!((-&p).to_f64(), -(pow2(k)), "-2^{k}");
        assert_both(&p);
    }
    for k in -149..=127 {
        assert_eq!(two_to(k).to_f32(), pow2_32(k), "2^{k}");
    }
    // Beyond the ranges.
    assert_eq!(two_to(1024).to_f64(), f64::INFINITY);
    assert_eq!((-two_to(1024)).to_f64(), f64::NEG_INFINITY);
    assert_eq!(two_to(128).to_f32(), f32::INFINITY);
    assert_eq!(two_to(-1075).to_f64().to_bits(), 0);
    assert_eq!((-two_to(-1075)).to_f64().to_bits(), (-0.0f64).to_bits());
    assert_eq!(two_to(-150).to_f32().to_bits(), 0);
    assert_eq!(two_to(-5000).to_f64(), 0.0);
    assert_eq!(two_to(5000).to_f32(), f32::INFINITY);
}

#[test]
fn halfway_cases() {
    // Between MAX and 2^1024: below the middle to MAX, at it and above to infinity.
    let max = exact(f64::MAX);
    let middle = &max + two_to(970);
    let tiny = two_to(-2000);
    assert_eq!((&middle - &tiny).to_f64(), f64::MAX);
    assert_eq!(middle.to_f64(), f64::INFINITY);
    assert_eq!((-&middle).to_f64(), f64::NEG_INFINITY);
    let middle32 = exact32(f32::MAX) + two_to(103);
    assert_eq!((&middle32 - &tiny).to_f32(), f32::MAX);
    assert_eq!(middle32.to_f32(), f32::INFINITY);
    // Half the smallest subnormal: to zero; a little above, to it.
    assert_eq!(two_to(-1075).to_f64(), 0.0);
    assert_eq!((two_to(-1075) + &tiny).to_f64(), f64::from_bits(1));
    assert_eq!((two_to(-1075) * 3u32).to_f64(), f64::from_bits(2));
    assert_eq!((two_to(-150) + &tiny).to_f32(), f32::from_bits(1));
    // Between the largest subnormal and the smallest normal number.
    let top = exact(f64::from_bits((1 << 52) - 1));
    let middle = (&top + exact(f64::MIN_POSITIVE)) / 2u32;
    assert_eq!(middle.to_f64(), f64::MIN_POSITIVE);
    assert_eq!((&middle - &tiny).to_f64(), f64::from_bits((1 << 52) - 1));
    // Ties both ways, from every binade, with long denominators on either side.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..size(3000, 60) {
        let bits = xorshift(&mut state) & 0x7FEF_FFFF_FFFF_FFFF;
        let y = f64::from_bits(bits);
        let z = y.next_up();
        if z.is_infinite() {
            continue;
        }
        let middle = (exact(y) + exact(z)) / 2u32;
        let even = if y.to_bits() & 1 == 0 { y } else { z };
        assert_eq!(middle.to_f64(), even, "{y:?} {z:?}");
        assert_eq!((-&middle).to_f64(), -even);
        let e = Rational::from(1) / (Number::from(1) << (1100 + below(&mut state, 3000) as usize));
        let e =
            &e + Rational::from(1) / (Number::from(3) << (2000 + below(&mut state, 500) as usize));
        assert_eq!((&middle + &e).to_f64(), z, "{y:?} {z:?} above");
        assert_eq!((&middle - &e).to_f64(), y, "{y:?} {z:?} below");
        // A long odd denominator just off the middle, so that the registers cannot tell.
        let q = integer(&mut state, 64).abs() * 2u32 + 1u32;
        let off = Rational::new(Number::from(1), q.clone() * (&q << 64usize));
        let off = off * exact(y).max(two_to(-1074));
        let up = &middle + &off;
        assert_eq!(up.to_f64(), z, "{y:?} {off}");
        assert_nearest(&up, up.to_f64());
        let down = &middle - &off;
        assert_eq!(down.to_f64(), y, "{y:?} {off}");
        assert_nearest(&down, down.to_f64());
        // And the f32s.
        let y = f32::from_bits((bits >> 32) as u32 & 0x7F7F_FFFF);
        let z = y.next_up();
        if z.is_infinite() {
            continue;
        }
        let middle = (exact32(y) + exact32(z)) / 2u32;
        let even = if y.to_bits() & 1 == 0 { y } else { z };
        assert_eq!(middle.to_f32(), even, "{y:?} {z:?}");
        assert_eq!((&middle + &e).to_f32(), z, "{y:?} {z:?} above");
        assert_eq!((&middle - &e).to_f32(), y, "{y:?} {z:?} below");
        assert_eq!(middle.to_f64(), middle.to_f64(), "{y:?}");
        // Through an f64, the tie with a little above would be lost.
        let above = &middle + &tiny;
        assert_eq!(above.to_f32(), z);
    }
}

#[test]
fn float_round_trips() {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for i in 0..size(20000, 300) {
        let mut bits = xorshift(&mut state);
        // A share of subnormal numbers, and of numbers near the ends of the range.
        match i % 8 {
            0 => bits &= 0x800F_FFFF_FFFF_FFFF,
            1 => bits |= 0x7FE0_0000_0000_0000,
            _ => {}
        }
        let x = f64::from_bits(bits);
        if !x.is_finite() {
            assert!(Rational::try_from(x).is_err());
            continue;
        }
        let r = exact(x);
        if x == 0.0 {
            assert!(r.is_zero());
            continue;
        }
        assert_eq!(r.to_f64().to_bits(), x.to_bits(), "{x:?}");
        assert_eq!(r.is_negative(), x < 0.0);
        // The f32 of an f64's value is rounded once.
        assert_nearest32(&r, r.to_f32());
        // In both byte orders.
        let b = in_order(&r, ByteOrder::BigEndian);
        assert_eq!(b.to_f64().to_bits(), x.to_bits(), "{x:?} big-endian");
        let y = f32::from_bits(bits as u32);
        if y.is_finite() && y != 0.0 {
            let s = exact32(y);
            assert_eq!(s.to_f32().to_bits(), y.to_bits(), "{y:?}");
            assert_eq!(s, exact(f64::from(y)));
            assert_eq!(s.to_f64(), f64::from(y));
        }
    }
}

#[test]
fn random_rationals_round_correctly() {
    let mut state = 0x1234_5678_9ABC_DEF1u64;
    for i in 0..size(4000, 80) {
        let max = if i % 10 == 0 { 300 } else { 40 };
        let r = rational(&mut state, max);
        assert_both(&r);
        // Scaled toward the ends of the ranges.
        let k = below(&mut state, 2400) as i64 - 1200;
        let s = &r * two_to(k);
        assert_both(&s);
        let b = in_order(&s, ByteOrder::BigEndian);
        assert_eq!(b.to_f64().to_bits(), s.to_f64().to_bits());
        assert_eq!(b.to_f32().to_bits(), s.to_f32().to_bits());
    }
}

#[test]
fn register_paths_round_correctly() {
    // Parts of up to 64 bits, below and above 2^53 and 2^24.
    let mut state = 0xDEAD_BEEF_CAFE_F00Du64;
    for _ in 0..size(20000, 300) {
        let bits = |state: &mut u64| {
            let b = 1 + below(state, 64) as u32;
            xorshift(state) >> (64 - b)
        };
        let (p, q) = (bits(&mut state), bits(&mut state).max(1));
        let r = Rational::new(Number::from(p), Number::from(q));
        assert_both(&r);
        let r = -r;
        assert_both(&r);
        // Integers: as Number::to_f64 rounds them.
        let n = integer(&mut state, 40);
        let r = Rational::from(&n);
        assert_eq!(r.to_f64().to_bits(), n.to_f64().to_bits(), "{n}");
        assert_eq!(r.to_f32().to_bits(), n.to_f32().to_bits(), "{n}");
    }
}

#[test]
fn long_parts_round_correctly() {
    // Long parts on either side, of every relation of lengths, both byte orders.
    let mut state = 0x0F1E_2D3C_4B5A_6978u64;
    for _ in 0..size(600, 12) {
        let la = 1 + below(&mut state, 400) as usize;
        let lb = 1 + below(&mut state, 400) as usize;
        let m = super::common::magnitude(&mut state, la);
        let mut d = super::common::magnitude(&mut state, lb);
        if d.is_empty() {
            d.push(1);
        }
        let order = super::common::order(xorshift(&mut state));
        let a = number(&m, below(&mut state, 2) == 1, order);
        let b = number(&d, false, order);
        let r = Rational::new(a, b);
        assert_both(&r);
    }
}

#[test]
fn rationals_to_integers() {
    let r = Rational::from(-7);
    assert_eq!(Number::try_from(&r), Ok(Number::from(-7)));
    assert_eq!(Number::try_from(r.clone()), Ok(Number::from(-7)));
    assert_eq!(i8::try_from(&r), Ok(-7));
    assert_eq!(i64::try_from(r.clone()), Ok(-7));
    assert!(u8::try_from(&r).is_err());
    let half = Rational::from(1) / 2u32;
    assert!(Number::try_from(&half).is_err());
    assert!(Number::try_from(half.clone()).is_err());
    assert!(u64::try_from(&half).is_err());
    let e: TryFromRationalError = Number::try_from(&half).unwrap_err();
    assert!(!e.to_string().is_empty());
    assert_eq!(u8::try_from(&Rational::from(255)), Ok(255));
    assert!(u8::try_from(&Rational::from(256)).is_err());
    assert_eq!(i128::try_from(&Rational::from(i128::MIN)), Ok(i128::MIN));
    assert_eq!(u128::try_from(&Rational::from(u128::MAX)), Ok(u128::MAX));
    assert!(u128::try_from(&(Rational::from(u128::MAX) + 1u32)).is_err());
    assert_eq!(usize::try_from(&Rational::from(0)), Ok(0));
    let mut state = 77u64;
    for _ in 0..size(500, 20) {
        let n = integer(&mut state, 20);
        let r = Rational::from(&n);
        assert_eq!(Number::try_from(&r), Ok(n.clone()));
        assert_eq!(i64::try_from(&r).ok(), i64::try_from(&n).ok());
        assert_eq!(u32::try_from(&r).ok(), u32::try_from(&n).ok());
        let r = rational(&mut state, 20);
        if !r.is_integer() {
            assert!(Number::try_from(&r).is_err());
            assert!(i128::try_from(&r).is_err());
        }
    }
}

/// A word of a shape that the register paths find hard: just above or below a power of two, a run
/// of ones, an odd number, or random bits of a random length; never zero.
fn hard_word(state: &mut u64) -> u64 {
    let k = 1 + below(state, 64) as u32;
    let x = xorshift(state) >> (64 - k);
    let w = match below(state, 5) {
        0 => (1u64 << (k - 1))
            .wrapping_add(below(state, 5))
            .wrapping_sub(2),
        1 => (u64::MAX >> (64 - k)) ^ below(state, 4),
        2 => x | 1,
        _ => x,
    };
    w.max(1)
}

#[test]
fn word_quotients_round_correctly() {
    // Parts in words of hard shapes, both signs, and the same fractions at 65 bits and beyond.
    let mut state = 0x7777_1111_9999_3333u64;
    for _ in 0..size(20000, 300) {
        let (p, q) = (hard_word(&mut state), hard_word(&mut state));
        let q = if below(&mut state, 8) == 0 { 1 } else { q };
        let r = Rational::new(Number::from(p), Number::from(q));
        assert_both(&r);
        assert_both(&-r.clone());
        // Twice the fraction, past 64 bits when a part has 64: rounded alike but for the exponent.
        let s = &r * Rational::from(2);
        assert_eq!(s.to_f64(), 2.0 * r.to_f64(), "{r}");
        assert_eq!(s.to_f32(), 2.0 * r.to_f32(), "{r}");
    }
}

#[test]
fn word_quotients_at_halfway_points() {
    // For each format, significands at both ends of a binade and in it: the halfway points
    // `(2m + 1)/2` between two floats, scaled by powers of two (ties, to the even one), and
    // fractions with an odd denominator `b` within `1/2b` of them on either side.
    let mut state = 0x2468_ACE0_1357_9BDFu64;
    for p in [24u32, 53] {
        let low = 1u64 << (p - 1);
        let high = (1u64 << p) - 1;
        let mut significands = vec![low, low + 1, low + 2, high - 1, high];
        for _ in 0..size(30, 3) {
            significands.push(low + below(&mut state, low));
        }
        for &m in &significands {
            let odd = 2 * m + 1;
            let room = odd.leading_zeros();
            for s in 0..=room {
                // `odd·2^s / 2^t`: a tie at every scale, both parts in words.
                for t in [0u32, 1, 2, 20, 62, 63] {
                    let r = Rational::new(Number::from(odd << s), Number::from(1u64 << t));
                    assert_both(&r);
                }
            }
            for _ in 0..size(20, 2) {
                // `(odd·b ± 1)/(2b)` for an odd `b` that keeps the numerator in a word.
                let b = (2 * below(&mut state, 1 << (room.saturating_sub(2)).min(20)) + 1).max(3);
                let x = odd as u128 * b as u128;
                if x >> 64 != 0 {
                    continue;
                }
                for y in [x - 1, x + 1] {
                    let r = Rational::new(Number::from(y as u64), Number::from(2 * b));
                    assert_both(&r);
                    assert_both(&-r);
                }
            }
        }
    }
    // The largest and smallest values in words, and integers at the boundaries of the formats.
    for (p, q) in [
        (u64::MAX, 1),
        (1, u64::MAX),
        (u64::MAX, u64::MAX - 1),
        (u64::MAX - 1, u64::MAX),
        ((1 << 53) + 1, 1),
        ((1 << 53) + 3, 1),
        ((1 << 24) + 1, 1),
        ((1 << 24) + 3, 1),
        (u64::MAX, 3),
        (3, u64::MAX),
    ] {
        assert_both(&Rational::new(Number::from(p), Number::from(q)));
    }
}

#[test]
fn floats_in_words_and_beyond() {
    // Floats whose parts are just within words and just past them, both ways.
    for k in [
        -66i64, -65, -64, -63, -62, -1, 0, 1, 10, 52, 53, 62, 63, 64, 65,
    ] {
        for m in [1u64, 3, (1 << 52) + 1, (1 << 53) - 1] {
            let x = m as f64 * pow2(k);
            let r = exact(x);
            let expected = if k >= 0 {
                Rational::from(Number::from(m) << k as usize)
            } else {
                Rational::new(Number::from(m), Number::from(1) << (-k) as usize)
            };
            assert_eq!(r, expected, "{m}·2^{k}");
            assert_eq!(r.to_f64(), x);
            assert_eq!(exact(-x), -expected.clone());
            assert_eq!(r.to_string(), expected.to_string());
            if m < 1 << 24 && (-149..=104).contains(&k) {
                let y = m as f32 * pow2_32(k);
                assert_eq!(exact32(y), expected, "{m}·2^{k} as f32");
            }
        }
    }
    assert_eq!(
        exact(f64::MIN_POSITIVE / 4.0).denom(),
        &(Number::from(1) << 1024usize)
    );
    assert_eq!(
        exact(f64::MAX).numer(),
        &((Number::from(1) << 1024usize) - (Number::from(1) << 971usize))
    );
}

#[test]
fn integers_out_of_both_forms() {
    // Integers of 63, 64 and 65 bits and their negatives, into numbers and every primitive type.
    for v in [
        0i128,
        1,
        -1,
        i64::MAX as i128,
        i64::MIN as i128,
        u64::MAX as i128,
        -(u64::MAX as i128),
        u64::MAX as i128 + 1,
        -(u64::MAX as i128) - 1,
        i128::MAX,
        i128::MIN,
    ] {
        let r = Rational::from(v);
        check(&r);
        assert_eq!(Number::try_from(&r), Ok(Number::from(v)));
        assert_eq!(Number::try_from(r.clone()), Ok(Number::from(v)));
        assert_eq!(i128::try_from(&r), Ok(v));
        assert_eq!(i64::try_from(&r).ok(), i64::try_from(v).ok());
        assert_eq!(u64::try_from(&r).ok(), u64::try_from(v).ok());
        assert_eq!(u128::try_from(&r).ok(), u128::try_from(v).ok());
        assert_eq!(i8::try_from(r.clone()).ok(), i8::try_from(v).ok());
        assert_eq!(usize::try_from(&r).ok(), usize::try_from(v).ok());
        // The cached parts of a short value, asked for first, change nothing.
        let s = Rational::from(v);
        let _ = s.numer();
        assert_eq!(Number::try_from(s), Ok(Number::from(v)));
        // A fraction near it is no integer.
        let f = Rational::new(
            Number::from(v) * Number::from(2) + Number::from(1),
            Number::from(2),
        );
        assert!(Number::try_from(&f).is_err());
        assert!(Number::try_from(f.clone()).is_err());
        assert!(i128::try_from(&f).is_err());
    }
    assert_eq!(u128::try_from(&Rational::from(u128::MAX)), Ok(u128::MAX));
    assert!(i128::try_from(&Rational::from(u128::MAX)).is_err());
    assert!(i128::try_from(&(Rational::from(i128::MIN) - 1u32)).is_err());
}
