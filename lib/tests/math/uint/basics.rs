use a::math::numbers::Number;
use a::math::uint::{Int, Uint};

/// A seeded xorshift generator.
fn next(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

#[test]
fn the_skeleton_arithmetic_agrees_with_number() {
    let mut s = 0x9e37_79b9_7f4a_7c15;
    let r = Number::from(1u64) << 256;
    for _ in 0..500 {
        let a = Uint::<4>::from_limbs([next(&mut s), next(&mut s), next(&mut s), next(&mut s)]);
        let b = Uint::<4>::from_limbs([next(&mut s), next(&mut s), 0, next(&mut s) >> 7]);
        let (na, nb) = (Number::from(a), Number::from(b));
        let (sum, carry) = a.overflowing_add(b);
        let full = &na + &nb;
        assert_eq!(Number::from(sum), full.rem_euclid(&r));
        assert_eq!(carry, full >= r);
        let (diff, borrow) = a.overflowing_sub(b);
        assert_eq!(Number::from(diff), (&na - &nb).rem_euclid(&r));
        assert_eq!(borrow, nb > na);
        let (lo, hi) = a.widening_mul(b);
        assert_eq!(Number::from(lo) + (Number::from(hi) << 256), &na * &nb);
        assert_eq!(a.cmp(&b), na.cmp(&nb));
        assert_eq!(Uint::<4>::try_from(&na), Ok(a));
    }
}

#[test]
fn signed_conversions_round_trip_at_the_ends() {
    for x in [
        Int::<3>::MIN,
        Int::MAX,
        Int::ZERO,
        Int::MINUS_ONE,
        Int::from_i64(-12345),
    ] {
        let n = Number::from(x);
        assert_eq!(Int::<3>::try_from(&n), Ok(x));
    }
    let too_big = Number::from(Int::<3>::MAX) + Number::from(1u64);
    assert!(Int::<3>::try_from(&too_big).is_err());
    let too_small = Number::from(Int::<3>::MIN) - Number::from(1u64);
    assert!(Int::<3>::try_from(&too_small).is_err());
    assert!(Uint::<2>::try_from(&Number::from(-1i64)).is_err());
}
