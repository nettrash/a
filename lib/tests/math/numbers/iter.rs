use a::math::numbers::{ByteOrder, Number};

use super::common::*;

#[test]
fn sums_are_the_numbers_added_in_turn() {
    let mut state = 0x5EED_0030;
    for count in [0, 1, 2, 3, 10, size(100, 12)] {
        let numbers: Vec<Number> = (0..count).map(|_| random_operand(&mut state, 40)).collect();
        let order = numbers
            .first()
            .map_or(ByteOrder::LittleEndian, |n| n.byte_order);
        let expected = numbers
            .iter()
            .fold(Number::new_from_u8(0, order), |s, x| s + x);
        let owned: Number = numbers.iter().cloned().sum();
        let borrowed: Number = numbers.iter().sum();
        for n in [&owned, &borrowed] {
            assert!(
                same(n, &expected),
                "{} vs {}",
                describe(n),
                describe(&expected)
            );
            assert_shortest(n, &|| format!("a sum of {count}"));
        }
    }
}

#[test]
fn products_are_the_numbers_multiplied_together() {
    let mut state = 0x5EED_0031;
    for count in [0, 1, 2, 3, 5, 16, 17, size(200, 20)] {
        let numbers: Vec<Number> = (0..count)
            .map(|_| {
                let len = if below(&mut state, 8) == 0 { 30 } else { 4 };
                let mut n = random_operand(&mut state, len);
                if below(&mut state, 50) != 0 && n.is_zero() {
                    n = Number::from(3);
                }
                n
            })
            .collect();
        let order = numbers
            .first()
            .map_or(ByteOrder::LittleEndian, |n| n.byte_order);
        let expected = numbers
            .iter()
            .fold(Number::new_from_u8(1, order), |p, x| p * x);
        let owned: Number = numbers.iter().cloned().product();
        let borrowed: Number = numbers.iter().product();
        for n in [&owned, &borrowed] {
            assert!(
                same(n, &expected),
                "{} vs {}",
                describe(n),
                describe(&expected)
            );
            assert_shortest(n, &|| format!("a product of {count}"));
        }
    }
}

#[test]
fn a_single_number_comes_back_in_the_shortest_form() {
    let n = number(&[5, 0, 0], true, ByteOrder::BigEndian);
    let p: Number = std::iter::once(n.clone()).product();
    let s: Number = std::iter::once(&n).sum();
    for x in [p, s] {
        assert!(
            same(&x, &number(&[5], true, ByteOrder::BigEndian)),
            "{}",
            describe(&x)
        );
    }
    let z: Number = std::iter::once(number(&[], true, ByteOrder::BigEndian)).product();
    assert!(same(&z, &number(&[0], false, ByteOrder::BigEndian)));
}
