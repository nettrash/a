use a::math::numbers::{ByteOrder, Number, NumberErrorKind};

use super::common::*;

/// Every way to store the integer `x`, as in the comparison tests: both byte orders, with zero
/// bytes above, and zero also empty and negative.
fn stores(x: i128) -> Vec<Number> {
    let magnitude = x.unsigned_abs().to_le_bytes();
    let top = magnitude.iter().rposition(|&b| b != 0).map_or(0, |t| t + 1);
    let mut all = Vec::new();
    for order in ORDERS {
        for zeros in [0, 1, 9] {
            let mut m = magnitude[..top].to_vec();
            m.resize(top + zeros, 0);
            all.push(number(&m, x < 0, order));
            if x == 0 {
                all.push(number(&m, true, order));
            }
        }
    }
    all
}

/// Checks the conversions of `x` to every primitive integer type that is not `u128`, against
/// the primitives' own `TryFrom<i128>`.
fn assert_converts(x: i128) {
    macro_rules! each {
        ($($t:ty),*) => {$(
            let expected = <$t>::try_from(x).ok();
            for n in stores(x) {
                assert_eq!(<$t>::try_from(&n).ok(), expected, "{x} as {}", stringify!($t));
                assert_eq!(<$t>::try_from(n).ok(), expected);
            }
        )*};
    }
    each!(u8, u16, u32, u64, usize, i8, i16, i32, i64, i128, isize);
    let expected = u128::try_from(x).ok();
    for n in stores(x) {
        assert_eq!(u128::try_from(&n).ok(), expected, "{x} as u128");
    }
}

#[test]
fn integers_convert_at_the_edges_of_every_type() {
    let mut edges = vec![
        0,
        1,
        -1,
        2,
        -2,
        i128::MAX,
        i128::MAX - 1,
        i128::MIN,
        i128::MIN + 1,
    ];
    for bits in [8, 16, 32, 64] {
        let (max, umax) = ((1i128 << (bits - 1)) - 1, (1i128 << bits) - 1);
        edges.extend([max, max + 1, -max, -max - 1, -max - 2, umax, umax + 1]);
    }
    for &x in &edges {
        assert_converts(x);
    }
    let mut state = 0x5EED_0010;
    for _ in 0..size(2000, 40) {
        assert_converts(random_i128(&mut state) >> below(&mut state, 127));
    }
}

#[test]
fn integers_become_numbers_little_endian() {
    let n = Number::from(-0x0102_0304i32);
    assert_eq!(n.value, [4, 3, 2, 1]);
    assert!(n.is_negative() && n.byte_order() == ByteOrder::LittleEndian);
    assert_eq!(Number::from(0u8).value, [0]);
    assert_eq!(Number::from(i128::MIN).value.len(), 16);
    assert_eq!(Number::from(u128::MAX), u128::MAX);
    assert_eq!(Number::from(usize::MAX), usize::MAX);
    assert_eq!(Number::from(isize::MIN), isize::MIN);
    for n in [u128::MAX, 0, 1 << 100] {
        assert_eq!(u128::try_from(Number::from(n)), Ok(n));
    }
    let long = Number::new(vec![1; 17], ByteOrder::LittleEndian);
    assert!(u128::try_from(&long).is_err() && i128::try_from(&long).is_err());
    let e = u8::try_from(&Number::from(256)).unwrap_err();
    assert_eq!(
        e.to_string(),
        "out of range integral type conversion attempted"
    );
}

/// The nearest `f64` and `f32` to `n`, from its decimal text, which the standard library reads
/// correctly rounded.
fn nearest(n: &Number) -> (f64, f32) {
    let text = n.to_string();
    (text.parse().unwrap(), text.parse().unwrap())
}

/// A number from a magnitude given least significant byte first.
fn of(magnitude: &[u8], negative: bool, order: ByteOrder) -> Number {
    number(magnitude, negative, order)
}

#[test]
fn floats_are_the_nearest_ones() {
    let mut state = 0x5EED_0011;
    for _ in 0..size(4000, 30) {
        let len = match below(&mut state, 4) {
            0 => below(&mut state, 17),
            1 => 120 + below(&mut state, 20),
            _ => below(&mut state, 140),
        };
        let n = of(
            &random_magnitude(&mut state, len),
            below(&mut state, 2) == 0,
            ORDERS[below(&mut state, 2)],
        );
        let (f, g) = nearest(&n);
        assert_eq!(n.to_f64(), f, "{}", describe(&n));
        assert_eq!(n.to_f32(), g, "{}", describe(&n));
    }
}

#[test]
fn halfway_floats_go_to_the_even_one() {
    // m·2^k with m of 54 or 25 bits whose lowest bit is the only one below the float's: halfway,
    // then with a bit far below (sticky), and with the bits around them in every combination.
    for order in ORDERS {
        for k in [0u32, 1, 7, 8, 9, 63, 64, 65, 200, 900, 969, 970, 971] {
            for m in [
                (1u128 << 53) + 1,
                (1u128 << 53) + 3,
                (1u128 << 54) - 1,
                (1u128 << 24) + 1,
                (1u128 << 24) + 3,
                (1u128 << 25) - 1,
                (1u128 << 53) | (1 << 29) | 1,
            ] {
                for low in [0u32, 1] {
                    // m·2^k + low·(1 if k > 0)
                    let mut bytes = vec![0u8; ((128 + k) / 8 + 2) as usize];
                    for bit in 0..128 {
                        if m >> bit & 1 == 1 {
                            let b = (bit + k) as usize;
                            bytes[b / 8] |= 1 << (b % 8);
                        }
                    }
                    if k > 0 && low == 1 {
                        bytes[0] |= 1;
                    }
                    let n = of(&bytes, k % 2 == 1, order);
                    let (f, g) = nearest(&n);
                    assert_eq!(n.to_f64(), f, "m = {m:#x}, k = {k}, low = {low}");
                    assert_eq!(n.to_f32(), g, "m = {m:#x}, k = {k}, low = {low}");
                }
            }
        }
    }
}

#[test]
fn floats_too_large_are_infinite() {
    let pow2 = |k: usize| {
        let mut m = vec![0u8; k / 8 + 1];
        m[k / 8] = 1 << (k % 8);
        m
    };
    for order in ORDERS {
        // f64::MAX = (2^53 - 1)·2^971; halfway to 2^1024 is 2^1024 - 2^970, which goes to the even
        // one, infinity.
        let two_1024 = of(&pow2(1024), false, order);
        let halfway = &two_1024 - &of(&pow2(970), false, order);
        assert_eq!(halfway.to_f64(), f64::INFINITY);
        assert_eq!((&halfway - 1u8).to_f64(), f64::MAX);
        assert_eq!((-&two_1024).to_f64(), f64::NEG_INFINITY);
        let two_128 = of(&pow2(128), true, order);
        let halfway = &two_128 + &of(&pow2(103), false, order);
        assert_eq!(halfway.to_f32(), f32::NEG_INFINITY);
        assert_eq!((&halfway + 1u8).to_f32(), f32::MIN);
        assert_eq!(
            of(&[0; 3], true, order).to_f64().to_bits(),
            0.0f64.to_bits()
        );
        assert_eq!(of(&[], true, order).to_f32().to_bits(), 0.0f32.to_bits());
    }
}

#[test]
fn decimal_text_parses() {
    let n: Number = "-000123456789012345678901234567890".parse().unwrap();
    assert_eq!(n.to_string(), "-123456789012345678901234567890");
    assert_eq!(n.byte_order(), ByteOrder::LittleEndian);
    assert_eq!("+7".parse::<Number>().unwrap(), 7);
    assert_eq!("-0".parse::<Number>().unwrap(), 0);
    let empty = "".parse::<Number>().unwrap_err();
    assert_eq!(*empty.kind(), NumberErrorKind::Empty);
    let bad = "1_000".parse::<Number>().unwrap_err();
    assert_eq!(*bad.kind(), NumberErrorKind::InvalidDigit);
    assert!("-".parse::<Number>().is_err() && "0x10".parse::<Number>().is_err());
}
