use a::math::numbers::{ByteOrder, Number};

use super::common::*;

#[test]
fn negation_turns_the_sign_over_in_the_shortest_form() {
    let mut state = 0x5EED_0020;
    for _ in 0..size(2000, 40) {
        let a = random_operand(&mut state, 40);
        let v = value_of(&a);
        let inputs = || describe(&a);
        let expected = expected(&v.magnitude, !v.negative);
        for (what, n) in [("-&a", -&a), ("-a", -copy(&a, pick(3)))] {
            assert_eq!(value_of(&n), expected, "{what} with a = {}", inputs());
            assert_eq!(n.byte_order, a.byte_order);
            assert_shortest(&n, &inputs);
        }
        let abs = a.abs();
        assert_eq!(
            value_of(&abs),
            self::expected_abs(&v.magnitude),
            "abs of {}",
            inputs()
        );
        assert_shortest(&abs, &inputs);
        let signum = a.signum();
        let sign = if v.magnitude == [0] {
            0
        } else if v.negative {
            -1
        } else {
            1
        };
        assert_eq!(signum, sign);
        assert_eq!(signum.byte_order, a.byte_order);
        assert_eq!(a.is_zero(), sign == 0);
        assert_eq!(a.is_positive(), sign == 1);
        assert_eq!(a.is_negative(), a.negative);
    }
}

fn expected_abs(magnitude: &[u8]) -> Expected {
    expected(magnitude, false)
}

#[test]
fn zero_is_never_negative() {
    for order in ORDERS {
        for z in zeros(order) {
            assert!(z.is_zero() && !z.is_positive());
            let (a, b) = (-&z, -copy(&z, 0));
            for n in [a, b, z.abs(), z.signum()] {
                assert!(same(&n, &number(&[0], false, order)), "{}", describe(&n));
            }
        }
    }
}

#[test]
fn copies_are_exact_and_reuse_the_vector() {
    let a = number(&[1, 2, 3, 0, 0], true, ByteOrder::BigEndian);
    let b = a.clone();
    assert!(same(&a, &b));
    let mut c = Number::new(Vec::with_capacity(64), ByteOrder::LittleEndian);
    let before = c.value.as_ptr();
    c.clone_from(&a);
    assert!(same(&a, &c) && c.value.as_ptr() == before);
}

#[test]
fn the_default_is_an_empty_zero() {
    let z = Number::default();
    assert!(z.is_zero() && z.value.capacity() == 0);
    assert_eq!(z.byte_order(), ByteOrder::LittleEndian);
    assert_eq!(format!("{z:?}"), "0");
    let mut n = Number::from(12345);
    assert_eq!(std::mem::take(&mut n), 12345);
    assert!(n.is_zero());
}

#[test]
fn debug_is_decimal() {
    let n = Number::new_signed(vec![0, 0x01, 0x00], true, ByteOrder::BigEndian);
    assert_eq!(format!("{n:?}"), "-256");
    assert_eq!(format!("{:?}", Some(Number::from(7u8))), "Some(7)");
    assert_eq!(format!("{:?}", ByteOrder::BigEndian), "BigEndian");
}
