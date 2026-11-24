//! The operator forms: owned and borrowed operands on either side and the assignments give the
//! same as the named methods; shifts by every primitive integer type; `Sum` and `Product`, which
//! overflow as the operators do.

use super::common::{Rng, outcome, size};
use a::math::uint::{Int, Uint};
use std::hint::black_box;

/// Every form of the binary operator `$op` (and `$assign`) on `x` and `y` equals `$expected`, or
/// all of them panic when it does.
macro_rules! all_forms {
    ($x:expr, $y:expr, $op:tt, $assign:tt) => {{
        let (x, y) = ($x, $y);
        let owned = outcome(|| x $op y);
        assert_eq!(outcome(|| x $op &y), owned);
        assert_eq!(outcome(|| &x $op y), owned);
        assert_eq!(outcome(|| &x $op &y), owned);
        assert_eq!(outcome(|| { let mut z = x; z $assign y; z }), owned);
        assert_eq!(outcome(|| { let mut z = x; z $assign &y; z }), owned);
        owned
    }};
}

fn forms<const N: usize>(rng: &mut Rng) {
    let (x, y) = (rng.uint::<N>(), rng.uint::<N>());
    let checked = |r: Result<Uint<N>, String>, (v, o): (Uint<N>, bool)| {
        if o && cfg!(debug_assertions) {
            assert!(r.is_err());
        } else {
            assert_eq!(r, Ok(v));
        }
    };
    checked(all_forms!(x, y, +, +=), x.overflowing_add(y));
    checked(all_forms!(x, y, -, -=), x.overflowing_sub(y));
    checked(all_forms!(x, y, *, *=), x.overflowing_mul(y));
    assert_eq!(all_forms!(x, y, /, /=).ok(), x.checked_div(y));
    assert_eq!(all_forms!(x, y, %, %=).ok(), x.checked_rem(y));
    let and = all_forms!(x, y, &, &=).unwrap();
    let or = all_forms!(x, y, |, |=).unwrap();
    let xor = all_forms!(x, y, ^, ^=).unwrap();
    for i in 0..N {
        let (a, b) = (x.as_limbs()[i], y.as_limbs()[i]);
        assert_eq!(and.as_limbs()[i], a & b);
        assert_eq!(or.as_limbs()[i], a | b);
        assert_eq!(xor.as_limbs()[i], a ^ b);
        assert_eq!((!x).as_limbs()[i], !a);
        assert_eq!((!&x).as_limbs()[i], !a);
    }

    let (p, q) = (rng.int::<N>(), rng.int::<N>());
    let checked = |r: Result<Int<N>, String>, (v, o): (Int<N>, bool)| {
        if o && cfg!(debug_assertions) {
            assert!(r.is_err());
        } else {
            assert_eq!(r, Ok(v));
        }
    };
    checked(all_forms!(p, q, +, +=), p.overflowing_add(q));
    checked(all_forms!(p, q, -, -=), p.overflowing_sub(q));
    checked(all_forms!(p, q, *, *=), p.overflowing_mul(q));
    assert_eq!(all_forms!(p, q, /, /=).ok(), p.checked_div(q));
    assert_eq!(all_forms!(p, q, %, %=).ok(), p.checked_rem(q));
    assert_eq!(
        all_forms!(p, q, &, &=),
        Ok((p.cast_unsigned() & q.cast_unsigned()).cast_signed())
    );
    assert_eq!(
        all_forms!(p, q, |, |=),
        Ok((p.cast_unsigned() | q.cast_unsigned()).cast_signed())
    );
    assert_eq!(
        all_forms!(p, q, ^, ^=),
        Ok((p.cast_unsigned() ^ q.cast_unsigned()).cast_signed())
    );
    assert_eq!(!p, (!p.cast_unsigned()).cast_signed());
    assert_eq!(!&p, !p);
    checked(outcome(|| -p), p.overflowing_neg());
    assert_eq!(outcome(|| -&p), outcome(|| -p));
}

#[test]
fn the_borrowed_and_assigning_forms_agree_with_the_owned_ones() {
    let mut rng = Rng(0x0f05);
    for _ in 0..size(300, 10) {
        forms::<1>(&mut rng);
        forms::<2>(&mut rng);
        forms::<3>(&mut rng);
        forms::<5>(&mut rng);
        forms::<8>(&mut rng);
    }
}

/// Every form of a shift by an amount of the primitive type `$t`.
macro_rules! shifts_by {
    ($x:expr, $k:expr, $expected_left:expr, $expected_right:expr; $($t:ty),*) => {$(
        let x = $x;
        // An amount that the type cannot hold (above 127 for i8) is skipped for it.
        if u128::try_from($k).is_ok_and(|v| v <= <$t>::MAX as u128) {
            let k = $k as $t;
            let left = outcome(|| x << k);
            assert_eq!(left, $expected_left, "<< {} as {}", $k, stringify!($t));
            assert_eq!(outcome(|| x << &k), left);
            assert_eq!(outcome(|| &x << k), left);
            assert_eq!(outcome(|| &x << &k), left);
            assert_eq!(outcome(|| { let mut z = x; z <<= k; z }), left);
            assert_eq!(outcome(|| { let mut z = x; z <<= &k; z }), left);
            let right = outcome(|| x >> k);
            assert_eq!(right, $expected_right, ">> {} as {}", $k, stringify!($t));
            assert_eq!(outcome(|| x >> &k), right);
            assert_eq!(outcome(|| &x >> k), right);
            assert_eq!(outcome(|| &x >> &k), right);
            assert_eq!(outcome(|| { let mut z = x; z >>= k; z }), right);
            assert_eq!(outcome(|| { let mut z = x; z >>= &k; z }), right);
        }
    )*};
}

#[test]
fn shifts_take_every_primitive_integer_type() {
    let mut rng = Rng(0x5_41f7);
    let x = rng.uint::<3>();
    let y = rng.int::<3>();
    for k in [0i64, 1, 63, 64, 100, 191, 192, 193, 255, 1000] {
        // In range, or out of range without debug assertions: the shift modulo the width.
        let ok = k < 192 || !cfg!(debug_assertions);
        fn expect<T>(ok: bool, v: T, message: &str) -> Result<T, String> {
            if ok { Ok(v) } else { Err(message.to_string()) }
        }
        let left = expect(
            ok,
            x.wrapping_shl(k as u32),
            "attempt to shift left with overflow",
        );
        let right = expect(
            ok,
            x.wrapping_shr(k as u32),
            "attempt to shift right with overflow",
        );
        shifts_by!(x, k, left, right; u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
        let left = expect(
            ok,
            y.wrapping_shl(k as u32),
            "attempt to shift left with overflow",
        );
        let right = expect(
            ok,
            y.wrapping_shr(k as u32),
            "attempt to shift right with overflow",
        );
        shifts_by!(y, k, left, right; u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
    }
    // A negative amount is out of range; without debug assertions it is taken modulo the width.
    let minus = outcome(|| x << -1i32);
    if cfg!(debug_assertions) {
        assert!(minus.is_err());
    } else {
        assert_eq!(minus, Ok(x << 191u32));
        assert_eq!(x >> -64i8, x >> 128u32);
        assert_eq!(x << i128::MIN, x << ((i128::MIN).rem_euclid(192) as u32));
    }
}

#[test]
fn sums_and_products_of_iterators_overflow_as_the_operators_do() {
    let v: Vec<Uint<3>> = (1..=40u64).map(Uint::from_u64).collect();
    let sum: Uint<3> = v.iter().sum();
    assert_eq!(sum, Uint::from_u64(820));
    assert_eq!(v.iter().copied().sum::<Uint<3>>(), sum);
    let product: Uint<3> = v.iter().product();
    // 40! has 160 bits: it fits in three limbs.
    assert_eq!(
        product.to_string(),
        "815915283247897734345611269596115894272000000000"
    );
    assert_eq!(v.into_iter().product::<Uint<3>>(), product);
    assert_eq!(Vec::<Uint<3>>::new().iter().sum::<Uint<3>>(), Uint::ZERO);
    assert_eq!(Vec::<Uint<3>>::new().iter().product::<Uint<3>>(), Uint::ONE);
    let big = [Uint::<2>::MAX, Uint::ONE];
    let overflowed = outcome(|| big.iter().sum::<Uint<2>>());
    let product = outcome(|| {
        [Uint::<2>::MAX, Uint::from_u64(2)]
            .iter()
            .product::<Uint<2>>()
    });
    if cfg!(debug_assertions) {
        assert!(overflowed.is_err() && product.is_err());
    } else {
        assert_eq!(overflowed, Ok(Uint::ZERO));
        assert_eq!(product, Ok(Uint::MAX - Uint::ONE));
    }
    let s: Vec<Int<2>> = (-20..=10i64).map(Int::from_i64).collect();
    assert_eq!(s.iter().sum::<Int<2>>(), Int::from_i64(-155));
    assert_eq!(
        s[..5].iter().product::<Int<2>>(),
        Int::from_i64(-20 * 19 * 18 * 17 * 16)
    );
    assert_eq!(s.iter().product::<Int<2>>(), Int::ZERO);
    let m = outcome(|| {
        [Int::<2>::MIN, Int::MINUS_ONE]
            .into_iter()
            .product::<Int<2>>()
    });
    if cfg!(debug_assertions) {
        assert!(m.is_err());
    } else {
        assert_eq!(m, Ok(Int::MIN));
    }
}

#[test]
fn the_panic_messages_are_the_primitives() {
    let messages = [
        (
            outcome(|| Uint::<2>::MAX + Uint::ONE),
            "attempt to add with overflow",
        ),
        (
            outcome(|| Uint::<2>::ZERO - Uint::ONE),
            "attempt to subtract with overflow",
        ),
        (
            outcome(|| Uint::<2>::MAX * Uint::MAX),
            "attempt to multiply with overflow",
        ),
        (
            outcome(|| Uint::<2>::ONE << 128u32),
            "attempt to shift left with overflow",
        ),
        (
            outcome(|| Uint::<2>::ONE >> 128u32),
            "attempt to shift right with overflow",
        ),
        (
            outcome(|| Uint::<2>::from_u64(2).pow(128)),
            "attempt to exponentiate with overflow",
        ),
        (
            outcome(|| Int::<2>::from_i64(-3).pow(81).cast_unsigned()),
            "attempt to exponentiate with overflow",
        ),
    ];
    for (outcome, message) in messages {
        if cfg!(debug_assertions) {
            assert_eq!(outcome.unwrap_err(), message);
        } else {
            assert!(outcome.is_ok());
        }
    }
    // The wording is current Rust's (1.100). The toolchain running the tests gives the same, or,
    // being older, the wording of its release: Rust 1.88 wrote "multiply" for `pow`.
    if cfg!(debug_assertions) {
        let theirs = outcome(|| black_box(2u128).pow(128)).unwrap_err();
        assert!(
            theirs == "attempt to exponentiate with overflow"
                || theirs == "attempt to multiply with overflow",
            "{theirs}"
        );
        let ours = outcome(|| Uint::<2>::from_u64(2).pow(128)).unwrap_err();
        if theirs != "attempt to multiply with overflow" {
            assert_eq!(ours, theirs);
        }
        for (ours, theirs) in [
            (
                outcome(|| Uint::<2>::MAX + Uint::ONE),
                outcome(|| black_box(u128::MAX) + 1),
            ),
            (
                outcome(|| Uint::<2>::MAX * Uint::MAX),
                outcome(|| black_box(u128::MAX) * u128::MAX),
            ),
        ] {
            assert_eq!(ours.unwrap_err(), theirs.unwrap_err());
        }
    }
    let negated = outcome(|| -Int::<2>::MIN);
    if cfg!(debug_assertions) {
        assert_eq!(negated.unwrap_err(), "attempt to negate with overflow");
    } else {
        assert_eq!(negated, Ok(Int::MIN));
    }
    // These panic whatever the profile.
    assert_eq!(
        outcome(|| Int::<2>::MIN / Int::MINUS_ONE).unwrap_err(),
        "attempt to divide with overflow"
    );
    assert_eq!(
        outcome(|| Int::<2>::MIN % Int::MINUS_ONE).unwrap_err(),
        "attempt to calculate the remainder with overflow"
    );
    assert_eq!(
        outcome(|| Int::<2>::ONE / Int::ZERO).unwrap_err(),
        "attempt to divide by zero"
    );
    assert_eq!(
        outcome(|| Int::<2>::MINUS_ONE.isqrt()).unwrap_err(),
        "argument of integer square root cannot be negative"
    );
    assert!(outcome(|| Uint::<2>::ZERO.ilog2()).is_err());
    assert!(outcome(|| Uint::<2>::from_u64(8).ilog(Uint::ONE)).is_err());
    assert!(outcome(|| Int::<2>::from_i64(8).ilog(Int::from_i64(-2))).is_err());
}
