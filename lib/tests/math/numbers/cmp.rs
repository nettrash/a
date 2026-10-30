use a::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::common::*;

/// Every way to store the integer `x`: both byte orders, with no, one and nine zero bytes above
/// its magnitude; zero also empty and with `negative` set.
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

fn hash_of(n: &Number) -> u64 {
    let mut h = DefaultHasher::new();
    n.hash(&mut h);
    h.finish()
}

/// The order of two integers given as magnitudes, least significant byte first and possibly with
/// zero bytes above, and signs: worked out from the bytes, independently of the implementation.
fn reference_cmp(a: &Number, b: &Number) -> Ordering {
    let (x, y) = (value_of(a), value_of(b));
    let magnitudes = || {
        x.magnitude
            .len()
            .cmp(&y.magnitude.len())
            .then_with(|| x.magnitude.iter().rev().cmp(y.magnitude.iter().rev()))
    };
    match (x.negative, y.negative) {
        (false, true) => Ordering::Greater,
        (true, false) => Ordering::Less,
        (false, false) => magnitudes(),
        (true, true) => magnitudes().reverse(),
    }
}

#[test]
fn small_numbers_compare_as_their_integers() {
    let mut state = 0x5EED_0001;
    let mut values: Vec<i128> = vec![0, 1, -1, 255, 256, -256, i128::MAX, i128::MIN + 1, 1 << 64];
    values.extend((0..size(60, 8)).map(|_| random_i128(&mut state)));
    for &x in &values {
        for &y in &values {
            for a in stores(x) {
                for b in stores(y) {
                    assert_eq!(a.cmp(&b), x.cmp(&y), "{x} vs {y}: {a:?} {b:?}");
                    assert_eq!(a.partial_cmp(&b), Some(x.cmp(&y)));
                    assert_eq!(a == b, x == y, "{x} == {y}");
                    if x == y {
                        assert_eq!(hash_of(&a), hash_of(&b), "hash of {x}");
                    }
                }
            }
        }
    }
}

#[test]
fn long_numbers_compare_by_value() {
    let mut state = 0x5EED_0002;
    for _ in 0..size(3000, 60) {
        let a = random_operand(&mut state, 80);
        let b = if below(&mut state, 3) == 0 {
            // The same integer stored another way, or one that differs in one byte.
            let mut m = le_bytes(&a);
            if below(&mut state, 2) == 0 && !m.is_empty() {
                let i = below(&mut state, m.len());
                m[i] ^= 1 << below(&mut state, 8);
            }
            m.resize(m.len() + below(&mut state, 3), 0);
            number(&m, a.negative, ORDERS[below(&mut state, 2)])
        } else {
            random_operand(&mut state, 80)
        };
        let expected = reference_cmp(&a, &b);
        assert_eq!(a.cmp(&b), expected, "{} vs {}", describe(&a), describe(&b));
        assert_eq!(b.cmp(&a), expected.reverse());
        assert_eq!(a == b, expected == Ordering::Equal);
        if expected == Ordering::Equal {
            assert_eq!(
                hash_of(&a),
                hash_of(&b),
                "{} vs {}",
                describe(&a),
                describe(&b)
            );
        }
    }
}

#[test]
fn numbers_differing_only_in_a_byte_order_are_equal() {
    for len in [1, 7, 8, 9, 15, 16, 17, 31, 64, 100] {
        let m: Vec<u8> = (1..=len as u8).collect();
        let (big, little) = (
            number(&m, true, ByteOrder::BigEndian),
            number(&m, true, ByteOrder::LittleEndian),
        );
        assert_eq!(big, little);
        assert_eq!(hash_of(&big), hash_of(&little));
        let mut other = m.clone();
        other[0] ^= 1;
        assert_ne!(big, number(&other, true, ByteOrder::LittleEndian));
        assert!(big < number(&other, true, ByteOrder::LittleEndian) || other[0] < m[0]);
    }
}

#[test]
fn numbers_compare_with_the_primitive_integers() {
    let mut state = 0x5EED_0003;
    for _ in 0..size(2000, 40) {
        let x = random_i128(&mut state);
        let y = random_i128(&mut state) >> below(&mut state, 128);
        for a in stores(x) {
            assert_eq!(a.partial_cmp(&y), Some(x.cmp(&y)));
            assert_eq!(a == y, x == y);
            if let Ok(y) = i64::try_from(y) {
                assert_eq!(a.partial_cmp(&y), Some(x.cmp(&i128::from(y))));
            }
            if let Ok(y) = u64::try_from(y) {
                assert_eq!(a.partial_cmp(&y), Some(x.cmp(&i128::from(y))));
                assert_eq!(a == y as usize, x == i128::from(y));
            }
            if let Ok(y) = u128::try_from(y) {
                assert_eq!(a == y, x == y as i128);
            }
        }
    }
    let long = Number::new(vec![1; 17], ByteOrder::BigEndian);
    assert!(long > u128::MAX && long > i128::MAX);
    assert!(-long.clone() < i128::MIN);
    let (min, max) = (Number::from(i128::MIN), Number::from(u128::MAX));
    assert!(min == i128::MIN && max == u128::MAX);
    assert!(Number::from(i8::MIN) < 0u8 && Number::from(0) == 0u64 && Number::from(0) == -0i8);
}

#[test]
fn numbers_sort_and_dedup_by_value() {
    let mut state = 0x5EED_0004;
    let mut values: Vec<i128> = (0..size(300, 30))
        .map(|_| random_i128(&mut state) >> 64)
        .collect();
    let mut numbers: Vec<Number> = values
        .iter()
        .map(|&x| {
            let all = stores(x);
            let pick = below(&mut state, all.len());
            all.into_iter().nth(pick).unwrap()
        })
        .collect();
    values.sort();
    values.dedup();
    numbers.sort();
    numbers.dedup();
    assert_eq!(numbers, values);
    let set: std::collections::HashSet<Number> = numbers.iter().cloned().collect();
    assert_eq!(set.len(), values.len());
    assert!(
        values
            .iter()
            .all(|&x| stores(x).iter().all(|n| set.contains(n)))
    );
    let (lo, hi) = (Number::from(-5), Number::from(5));
    assert_eq!(Number::from(7).clamp(lo.clone(), hi.clone()), 5);
    assert_eq!(lo.clone().max(hi.clone()), 5);
    assert_eq!(lo.min(hi), -5);
}
