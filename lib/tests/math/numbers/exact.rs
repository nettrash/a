use a::math::numbers::{ByteOrder, Number};

use super::common::*;
use super::div::{add_magnitudes, mul_magnitudes, panic_message, sub_magnitudes};

/// A divisor of `len` bytes for the exact divisions: random, odd or even; all ones; a power of
/// two; or random with whole zero limbs at the low end, which Hensel's method shifts away.
fn divisor(state: &mut u64, len: usize, kind: usize) -> Vec<u8> {
    let mut y = match kind {
        0 => vec![0xFF; len],
        1 => {
            let mut v = vec![0; len];
            v[len - 1] = 1 << below(state, 8);
            v
        }
        _ => random_magnitude(state, len),
    };
    if kind == 3 {
        let zeros = below(state, len.min(20) + 1).min(len - 1);
        y[..zeros].fill(0);
    }
    if kind == 4 {
        y[0] |= 1;
    }
    y
}

/// Lengths in bytes around the short path, the paths in registers, a divisor of one limb, the
/// general path and the limit beyond which exact divisions are left to the ordinary one.
fn lengths() -> Vec<usize> {
    [
        1, 7, 8, 9, 16, 17, 24, 33, 64, 65, 120, 200, 640, 1600, 6000,
    ]
    .into_iter()
    .filter(|&l| l <= size(usize::MAX, 200))
    .collect()
}

#[test]
fn divides_products_exactly() {
    let mut state = 0x452821E6_38D01377;
    for &lq in &lengths() {
        for &lb in &lengths() {
            if lq * lb > size(4_000_000, 20_000) {
                continue;
            }
            for kind in 0..5 {
                let q = random_magnitude(&mut state, lq);
                let y = divisor(&mut state, lb, kind);
                let x = mul_magnitudes(&q, &y);
                let (oa, ob) = ORDER_PAIRS[below(&mut state, 4)];
                let (na, nb) = SIGN_PAIRS[below(&mut state, 4)];
                let (a, b) = (number(&x, na, oa), number(&y, nb, ob));
                let result = a.div_exact(&b);
                let inputs =
                    || format!("div_exact with a = {}, b = {}", describe(&a), describe(&b));
                assert_result(&result, &expected(&q, na != nb), oa, &inputs);
                assert!(a.is_multiple_of(&b), "is_multiple_of: {}", inputs());
            }
        }
    }
}

#[test]
fn products_with_a_remainder_are_not_multiples() {
    let mut state = 0xBE5466CF_34E90C6C;
    for &lq in &lengths() {
        for &lb in &lengths() {
            if lq * lb > size(4_000_000, 20_000) {
                continue;
            }
            for kind in 0..5 {
                let q = random_magnitude(&mut state, lq);
                let y = divisor(&mut state, lb, kind);
                // A remainder of 1, of a random magnitude shorter than `y`, and of `y - 1`.
                let r = match below(&mut state, 3) {
                    0 => vec![1],
                    1 => {
                        let len = 1 + below(&mut state, lb.max(2) - 1);
                        random_magnitude(&mut state, len)
                    }
                    _ => sub_magnitudes(&y, &[1]),
                };
                if r.iter().all(|&v| v == 0) {
                    continue;
                }
                let x = add_magnitudes(&mul_magnitudes(&q, &y), &r);
                let (oa, ob) = ORDER_PAIRS[below(&mut state, 4)];
                let a = number(&x, below(&mut state, 2) == 0, oa);
                let b = number(&y, below(&mut state, 2) == 0, ob);
                let multiple = (&a % &b).value == [0];
                assert_eq!(
                    a.is_multiple_of(&b),
                    multiple,
                    "is_multiple_of with a = {}, b = {}",
                    describe(&a),
                    describe(&b)
                );
                // Not exact: some number, in the shortest form and the byte order of `a`.
                let result = a.div_exact(&b);
                assert_shortest(&result, &|| "div_exact of a non-multiple".into());
                assert!(result.byte_order == oa);
            }
        }
    }
}

#[test]
fn multiples_follow_the_rules_of_primitive_integers() {
    // Zero is a multiple of everything, only zero of zero; signs do not count; a shorter number
    // or one with fewer zero bits at the low end is not a multiple.
    let mut state = 0xC0AC29B7_C97C50DD;
    for order in ORDERS {
        let zeros_ = zeros(order);
        let three = number(&[3], true, order);
        let long = number(&random_magnitude(&mut state, 40), false, order);
        for z in &zeros_ {
            assert!(z.is_multiple_of(&three) && z.is_multiple_of(&long));
            for w in &zeros_ {
                assert!(z.is_multiple_of(w));
            }
            assert!(!three.is_multiple_of(z) && !long.is_multiple_of(z));
        }
        let short = number(&random_magnitude(&mut state, 20), true, order);
        assert!(!short.is_multiple_of(&long));
        let even = number(
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            false,
            order,
        );
        let four = number(&[4], true, order);
        assert!(!even.is_multiple_of(&number(
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            false,
            order
        )));
        assert!(even.is_multiple_of(&four));
    }
    for (x, y) in [
        (12i128, 4i128),
        (-12, 4),
        (12, -4),
        (-12, -4),
        (12, 5),
        (1 << 100, 1 << 37),
        ((1 << 100) + 2, 1 << 37),
    ] {
        let a = Number::new_from_i128(x, ByteOrder::BigEndian);
        let b = Number::new_from_i128(y, ByteOrder::LittleEndian);
        assert_eq!(a.is_multiple_of(&b), x % y == 0, "{x} by {y}");
    }
}

#[test]
fn exact_division_by_zero_panics() {
    let a = number(&[0xFF; 40], false, ByteOrder::BigEndian);
    for z in zeros(ByteOrder::LittleEndian) {
        assert_eq!(
            panic_message(|| drop(a.div_exact(&z))),
            "attempt to divide by zero"
        );
        assert_eq!(
            panic_message(|| drop(number(&[5], false, ByteOrder::BigEndian).div_exact(&z))),
            "attempt to divide by zero"
        );
    }
}

#[test]
fn long_exact_divisions_go_to_the_ordinary_division() {
    // Past the limit of Hensel's method, a divisor long enough for divide and conquer: a quotient
    // of 750 limbs by a divisor of 750.
    let mut state = 0x7B54_A41D_C25A_59B5;
    let len = size(6000, 400);
    let q = random_magnitude(&mut state, len);
    let y = random_magnitude(&mut state, len);
    let a = number(&mul_magnitudes(&q, &y), true, ByteOrder::LittleEndian);
    let b = number(&y, false, ByteOrder::BigEndian);
    assert_result(
        &a.div_exact(&b),
        &expected(&q, true),
        ByteOrder::LittleEndian,
        &|| "a long exact division".into(),
    );
    assert!(a.is_multiple_of(&b));
}
