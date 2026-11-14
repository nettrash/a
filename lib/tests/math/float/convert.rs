//! The conversions of the core: the primitive floats and integers and numbers exactly, rationals
//! rounded, and finite floats into the rationals they are.

use super::common::{below, float, mode, rational, reference, same, size, xorshift};
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;

#[test]
fn primitive_floats_are_exact() {
    let mut s = 0x5EED_F10A_7000_0005u64;
    for _ in 0..size(5000, 50) {
        let x = f64::from_bits(xorshift(&mut s));
        let b = BigFloat::from(x);
        assert_eq!(b.prec(), 53);
        if x.is_nan() {
            assert!(b.is_nan());
        } else if x.is_infinite() {
            assert!(b.is_infinite() && b.is_sign_negative() == x.is_sign_negative());
        } else {
            assert_eq!(rational(&b), Rational::try_from(x).unwrap(), "{x:e}");
            assert_eq!(b.is_sign_negative(), x.is_sign_negative());
        }
        let y = f32::from_bits(xorshift(&mut s) as u32);
        let c = BigFloat::from(y);
        assert_eq!(c.prec(), 24);
        if y.is_finite() {
            assert_eq!(rational(&c), Rational::try_from(y).unwrap(), "{y:e}");
        }
    }
    assert!(BigFloat::from(-0.0).is_sign_negative() && BigFloat::from(-0.0).is_zero());
    assert_eq!(
        BigFloat::from(f64::MIN_POSITIVE / 4.0).exponent(),
        Some(-1023)
    );
}

#[test]
fn integers_and_numbers_are_exact() {
    assert_eq!(BigFloat::from(u64::MAX).prec(), 64);
    assert_eq!(
        rational(&BigFloat::from(i128::MIN)),
        Rational::from(i128::MIN)
    );
    assert_eq!(
        rational(&BigFloat::from(u128::MAX)),
        Rational::from(u128::MAX)
    );
    assert_eq!(rational(&BigFloat::from(-7i8)), Rational::from(-7));
    assert!(BigFloat::from(0u32).is_zero() && !BigFloat::from(0i32).is_sign_negative());
    let n: Number = "-123456789012345678901234567890123456789".parse().unwrap();
    let x = BigFloat::from(&n);
    assert_eq!(x.prec(), n.bits());
    assert_eq!(rational(&x), Rational::from_integer(n.clone()));
    let (y, o) = BigFloat::from_number(&n, 20, Round::TowardZero);
    assert_eq!(o, Ordering::Greater);
    assert!(rational(&y) > Rational::from_integer(n));
}

#[test]
fn rationals_round_correctly() {
    let mut s = 0x5EED_F10A_7000_0006u64;
    for _ in 0..size(3000, 30) {
        let a = float(&mut s, 120, 60);
        let b = float(&mut s, 120, 60);
        let q = rational(&a) / rational(&b);
        let prec = 1 + below(&mut s, 300);
        let round = mode(&mut s);
        same(
            &BigFloat::from_rational(&q, prec, round),
            &reference(&q, prec, round),
            &format!("{q}"),
        );
    }
    assert!(
        BigFloat::from_rational(&Rational::from(0), 5, Round::Up)
            .0
            .is_zero()
    );
}

#[test]
fn floats_are_the_rationals_they_name() {
    assert!(Rational::try_from(&BigFloat::nan(3)).is_err());
    assert!(Rational::try_from(&BigFloat::infinity(3)).is_err());
    assert_eq!(
        Rational::try_from(&BigFloat::neg_zero(3)).unwrap(),
        Rational::from(0)
    );
    assert_eq!(rational(&BigFloat::from(-0.375)), Rational::from(-3) / 8u32);
    assert_eq!(
        rational(&BigFloat::from(1e300)),
        Rational::try_from(1e300).unwrap()
    );
    let (m, e) = BigFloat::from(10).mantissa_exponent().unwrap();
    assert_eq!((m, e), (Number::from(5) << 29usize, -28));
}
