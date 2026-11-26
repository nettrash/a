//! Every operation of `Uint<N>` and `Int<N>` against `Number`, the integers of any size, for `N`
//! from 1 to 9, 16 and 32, on seeded values biased to the edges: the true result computed
//! exactly, then each form checked against it (the value modulo the width, whether it is out of
//! range, the value held to the range), and the operators against the behaviour of the profile:
//! a panic on overflow with debug assertions, the wrapped value without.

use super::common::{
    Rng, clamp, clamp_signed, exact, exact_signed, forms, forms_signed, outcome, pow2, size, wrap,
    wrap_signed,
};
use a::math::numbers::Number;
use a::math::uint::{Int, Uint};

/// An operator's outcome against the true result `t`: the value when it is in range; otherwise a
/// panic with debug assertions and the wrapped value without.
fn operator<const N: usize>(what: &str, t: &Number, r: Result<Uint<N>, String>) {
    match exact::<N>(t) {
        Some(e) => assert_eq!(r, Ok(e), "{what}: {t}"),
        None if cfg!(debug_assertions) => assert!(r.is_err(), "{what} should panic: {t}"),
        None => assert_eq!(r, Ok(wrap::<N>(t)), "{what} should wrap: {t}"),
    }
}

/// [`operator`] for a signed result.
fn operator_signed<const N: usize>(what: &str, t: &Number, r: Result<Int<N>, String>) {
    match exact_signed::<N>(t) {
        Some(e) => assert_eq!(r, Ok(e), "{what}: {t}"),
        None if cfg!(debug_assertions) => assert!(r.is_err(), "{what} should panic: {t}"),
        None => assert_eq!(r, Ok(wrap_signed::<N>(t)), "{what} should wrap: {t}"),
    }
}

/// The shift amounts to try at `bits` bits.
fn amounts(rng: &mut Rng, bits: u32) -> Vec<u32> {
    vec![
        0,
        1,
        63,
        64,
        65,
        bits - 1,
        bits,
        bits + 1,
        rng.below(bits as u64) as u32,
        rng.below(3 * bits as u64) as u32,
        u32::MAX,
    ]
}

/// The bytes of `x` least significant first, through `write_le_bytes`.
fn le_bytes<const N: usize>(x: Uint<N>) -> Vec<u8> {
    let mut v = vec![0; 8 * N];
    x.write_le_bytes(&mut v);
    v
}

/// The largest `k` with `base^k ≤ a`, for `a ≥ 1` and `base ≥ 2`, by the powers.
fn log(a: &Number, base: &Number) -> u32 {
    let mut k = 0;
    let mut p = base.clone();
    while &p <= a {
        p = &p * base;
        k += 1;
    }
    k
}

fn unsigned<const N: usize>(rng: &mut Rng, count: usize) {
    let bits = Uint::<N>::BITS;
    let full = pow2(bits);
    let one = Number::from(1u64);
    for _ in 0..count {
        let (x, y) = (rng.uint::<N>(), rng.uint::<N>());
        let (a, b) = (Number::from(x), Number::from(y));
        assert_eq!(Uint::<N>::try_from(&a), Ok(x));
        assert_eq!(x.cmp(&y), a.cmp(&b));

        // Sums, differences and products, in every form.
        let t = &a + &b;
        forms(
            "add",
            &t,
            x.overflowing_add(y),
            x.checked_add(y),
            x.wrapping_add(y),
            Some(x.saturating_add(y)),
        );
        operator("+", &t, outcome(|| x + y));
        operator(
            "+=",
            &t,
            outcome(|| {
                let mut z = x;
                z += &y;
                z
            }),
        );
        let t = &a - &b;
        forms(
            "sub",
            &t,
            x.overflowing_sub(y),
            x.checked_sub(y),
            x.wrapping_sub(y),
            Some(x.saturating_sub(y)),
        );
        operator("-", &t, outcome(|| x - y));
        let t = &a * &b;
        forms(
            "mul",
            &t,
            x.overflowing_mul(y),
            x.checked_mul(y),
            x.wrapping_mul(y),
            Some(x.saturating_mul(y)),
        );
        operator("*", &t, outcome(|| x * y));
        let t = -&a;
        forms(
            "neg",
            &t,
            x.overflowing_neg(),
            x.checked_neg(),
            x.wrapping_neg(),
            None,
        );
        let ys = rng.int::<N>();
        let t = &a + Number::from(ys);
        forms(
            "add_signed",
            &t,
            x.overflowing_add_signed(ys),
            x.checked_add_signed(ys),
            x.wrapping_add_signed(ys),
            Some(x.saturating_add_signed(ys)),
        );

        // The limb-level helpers.
        let (lo, hi) = x.widening_mul(y);
        assert_eq!(
            Number::from(lo) + (Number::from(hi) << bits as usize),
            &a * &b
        );
        let c = rng.uint::<N>();
        let (lo, hi) = x.carrying_mul(y, c);
        assert_eq!(
            Number::from(lo) + (Number::from(hi) << bits as usize),
            &a * &b + Number::from(c)
        );
        let t = &a + &b + &one;
        assert_eq!(x.carrying_add(y, true), (wrap::<N>(&t), t >= full));
        let t = &a - &b - &one;
        assert_eq!(x.borrowing_sub(y, true), (wrap::<N>(&t), t.is_negative()));
        let m = rng.limb();
        let (lo, hi) = x.widening_mul_u64(m);
        assert_eq!(
            Number::from(lo) + (Number::from(hi) << bits as usize),
            &a * Number::from(m)
        );

        // Quotients and remainders.
        if y.is_zero() {
            assert_eq!(x.checked_div(y), None);
            assert_eq!(x.checked_rem_euclid(y), None);
            assert!(outcome(|| x / y).is_err());
        } else {
            let (q, r) = a.div_rem(&b);
            assert_eq!(x.div_rem(y), (exact(&q).unwrap(), exact(&r).unwrap()));
            assert_eq!(x.checked_div(y), exact(&q));
            assert_eq!(x.checked_rem(y), exact(&r));
            assert_eq!(x.div_euclid(y), exact(&q).unwrap());
            assert_eq!(x.rem_euclid(y), exact(&r).unwrap());
            assert_eq!(x.overflowing_div(y), (exact(&q).unwrap(), false));
            assert_eq!(
                x.div_ceil(y),
                exact(&(&q + Number::from(!r.is_zero() as u64))).unwrap()
            );
            let t = if r.is_zero() { a.clone() } else { &a + &b - &r };
            assert_eq!(x.checked_next_multiple_of(y), exact(&t));
            operator("next_multiple_of", &t, outcome(|| x.next_multiple_of(y)));
            assert_eq!(x.is_multiple_of(y), r.is_zero());
        }
        let d = rng.limb().max(1);
        let (q, r) = x.div_rem_u64(d);
        let (nq, nr) = a.div_rem(&Number::from(d));
        assert_eq!((Number::from(q), Number::from(r)), (nq, nr));
        assert_eq!(Number::from(x.abs_diff(y)), (&a - &b).abs());
        assert_eq!(Number::from(x.midpoint(y)), (&a + &b) >> 1);

        // Shifts and rotations.
        for k in amounts(rng, bits) {
            let s = (k % bits) as usize;
            assert_eq!(
                x.overflowing_shl(k),
                (wrap::<N>(&(&a << s)), k >= bits),
                "{x:?} << {k}"
            );
            assert_eq!(
                x.overflowing_shr(k),
                (exact(&(&a >> s)).unwrap(), k >= bits)
            );
            assert_eq!(x.checked_shl(k), (k < bits).then(|| wrap::<N>(&(&a << s))));
            let u = if k >= bits {
                Uint::ZERO
            } else {
                wrap::<N>(&(&a << k as usize))
            };
            assert_eq!(x.unbounded_shl(k), u);
            let u = if k >= bits {
                Uint::ZERO
            } else {
                exact(&(&a >> k as usize)).unwrap()
            };
            assert_eq!(x.unbounded_shr(k), u);
            let r = if s == 0 {
                x
            } else {
                wrap::<N>(&((&a << s) + (&a >> (bits as usize - s))))
            };
            assert_eq!(x.rotate_left(k), r);
            assert_eq!(r.rotate_right(k), x);
            let shifted = outcome(|| x << k);
            if k >= bits && cfg!(debug_assertions) {
                assert!(shifted.is_err());
            } else {
                assert_eq!(shifted, Ok(wrap::<N>(&(&a << s))));
            }
            let shifted = outcome(|| x >> (k as i64));
            if k >= bits && cfg!(debug_assertions) {
                assert!(shifted.is_err());
            } else {
                assert_eq!(shifted, Ok(exact(&(&a >> s)).unwrap()));
            }
        }

        // Powers, roots and logarithms.
        let room = 2 * bits / x.bits().max(1);
        for e in [
            0,
            1,
            2,
            3,
            rng.below(room as u64 + 1) as u32,
            room,
            room + 1,
        ] {
            let t = a.pow(e);
            forms(
                "pow",
                &t,
                x.overflowing_pow(e),
                x.checked_pow(e),
                x.wrapping_pow(e),
                Some(x.saturating_pow(e)),
            );
            operator("pow", &t, outcome(|| x.pow(e)));
        }
        if x.bits() <= 1 {
            assert_eq!(x.overflowing_pow(u32::MAX), (x, false));
        } else {
            assert_eq!(x.checked_pow(u32::MAX), None);
        }
        assert_eq!(Number::from(x.isqrt()), a.isqrt());
        if x.is_zero() {
            assert_eq!(x.checked_ilog2(), None);
            assert_eq!(x.checked_ilog10(), None);
            assert!(outcome(|| x.ilog10()).is_err());
        } else {
            assert_eq!(x.ilog2() as u64, a.bits() - 1);
            assert_eq!(x.ilog10(), log(&a, &Number::from(10u64)));
            if y.bits() > 1 {
                assert_eq!(x.ilog(y), log(&a, &b));
            }
            let small = Number::from(2 + rng.below(1000));
            assert_eq!(x.ilog(exact(&small).unwrap()), log(&a, &small));
        }

        // Powers of two, bits and bytes.
        assert_eq!(x.is_power_of_two(), a.count_ones() == 1);
        let k = if x.bits() <= 1 {
            0
        } else {
            (&a - &one).bits() as u32
        };
        let t = pow2(k);
        assert_eq!(x.checked_next_power_of_two(), exact(&t));
        operator("next_power_of_two", &t, outcome(|| x.next_power_of_two()));
        assert_eq!(x.count_ones() as u64, a.count_ones());
        assert_eq!(x.count_zeros(), bits - x.count_ones());
        assert_eq!(x.leading_zeros() as u64, bits as u64 - a.bits());
        assert_eq!(
            x.trailing_zeros() as u64,
            a.trailing_zeros().unwrap_or(bits as u64)
        );
        let inverted = &full - &one - &a;
        assert_eq!(x.leading_ones() as u64, bits as u64 - inverted.bits());
        assert_eq!(
            x.trailing_ones() as u64,
            inverted.trailing_zeros().unwrap_or(bits as u64)
        );
        assert_eq!(Number::from(!x), inverted);
        for i in [0, 1, 63, 64, bits - 1, bits, rng.below(bits as u64) as u32] {
            assert_eq!(x.bit(i), a.bit(i as u64), "bit {i} of {x:?}");
        }
        let le = le_bytes(x);
        let mut be = vec![0; 8 * N];
        x.write_be_bytes(&mut be);
        assert_eq!(be.iter().rev().copied().collect::<Vec<_>>(), le);
        assert_eq!(Uint::<N>::from_le_slice(&le), Some(x));
        assert_eq!(Uint::<N>::from_be_slice(&be), Some(x));
        let mut padded = le.clone();
        padded.extend([0, 0, 0]);
        assert_eq!(Uint::<N>::from_le_slice(&padded), Some(x));
        padded.push(1);
        assert_eq!(Uint::<N>::from_le_slice(&padded), None);
        let mut swapped = le.clone();
        swapped.reverse();
        assert_eq!(Uint::<N>::from_le_slice(&swapped), Some(x.swap_bytes()));
        let binary = format!("{:0>w$b}", x, w = bits as usize);
        let reversed: String = binary.chars().rev().collect();
        assert_eq!(
            Uint::<N>::from_str_radix(&reversed, 2),
            Ok(x.reverse_bits())
        );

        // Text against Number's.
        assert_eq!(x.to_string(), a.to_string());
        let radix = 2 + rng.below(35) as u32;
        let text = x.to_str_radix(radix);
        assert_eq!(text, a.to_str_radix(radix));
        assert_eq!(
            Uint::<N>::from_str_radix(&text.to_uppercase(), radix),
            Ok(x)
        );
        assert_eq!(format!("{x:#x}"), format!("{a:#x}"));
        assert_eq!(format!("{x:>200X}"), format!("{a:>200X}"));
        assert_eq!(format!("{x:#0100o}"), format!("{a:#0100o}"));
        assert_eq!(format!("{x:b}"), format!("{a:b}"));
        assert_eq!(format!("{x:+}"), format!("{a:+}"));

        // Conversions.
        assert_eq!(x.to_f64().to_bits(), a.to_f64().to_bits(), "{x:?}");
        assert_eq!(x.to_f32().to_bits(), a.to_f32().to_bits(), "{x:?}");
        assert_eq!(x.resize::<1>(), wrap::<1>(&a));
        assert_eq!(x.try_resize::<1>(), exact::<1>(&a));
        assert_eq!(x.try_resize::<3>(), exact::<3>(&a));
        assert_eq!(x.resize::<12>(), wrap::<12>(&a));
        assert_eq!(Int::<N>::try_from(x).ok(), exact_signed::<N>(&a));
        assert_eq!(
            Number::from(x.cast_signed()),
            Number::from(wrap_signed::<N>(&a))
        );
        assert_eq!(clamp::<N>(&a), x);
    }
}

fn signed<const N: usize>(rng: &mut Rng, count: usize) {
    let bits = Int::<N>::BITS;
    let half = pow2(bits - 1);
    for i in 0..count {
        let (x, y) = match i {
            0 => (Int::<N>::MIN, Int::MINUS_ONE),
            1 => (Int::MIN, Int::MIN),
            2 => (Int::MAX, Int::MINUS_ONE),
            3 => (Int::MIN, Int::ONE),
            _ => (rng.int::<N>(), rng.int::<N>()),
        };
        let (a, b) = (Number::from(x), Number::from(y));
        assert_eq!(Int::<N>::try_from(&a), Ok(x));
        assert_eq!(x.cmp(&y), a.cmp(&b));
        assert_eq!(x.is_negative(), a.is_negative());
        assert_eq!(x.is_positive(), a.is_positive());
        assert_eq!(Number::from(x.signum()), a.signum());

        let t = &a + &b;
        forms_signed(
            "add",
            &t,
            x.overflowing_add(y),
            x.checked_add(y),
            x.wrapping_add(y),
            Some(x.saturating_add(y)),
        );
        operator_signed("+", &t, outcome(|| x + y));
        let t = &a - &b;
        forms_signed(
            "sub",
            &t,
            x.overflowing_sub(y),
            x.checked_sub(y),
            x.wrapping_sub(y),
            Some(x.saturating_sub(y)),
        );
        operator_signed(
            "-=",
            &t,
            outcome(|| {
                let mut z = x;
                z -= y;
                z
            }),
        );
        let t = &a * &b;
        forms_signed(
            "mul",
            &t,
            x.overflowing_mul(y),
            x.checked_mul(y),
            x.wrapping_mul(y),
            Some(x.saturating_mul(y)),
        );
        operator_signed("*", &t, outcome(|| x * y));
        let t = -&a;
        forms_signed(
            "neg",
            &t,
            x.overflowing_neg(),
            x.checked_neg(),
            x.wrapping_neg(),
            Some(x.saturating_neg()),
        );
        operator_signed("-x", &t, outcome(|| -x));
        let t = a.abs();
        forms_signed(
            "abs",
            &t,
            x.overflowing_abs(),
            x.checked_abs(),
            x.wrapping_abs(),
            Some(x.saturating_abs()),
        );
        operator_signed("abs", &t, outcome(|| x.abs()));
        assert_eq!(Number::from(x.unsigned_abs()), t);
        let yu = rng.uint::<N>();
        let t = &a + Number::from(yu);
        forms_signed(
            "add_unsigned",
            &t,
            x.overflowing_add_unsigned(yu),
            x.checked_add_unsigned(yu),
            x.wrapping_add_unsigned(yu),
            Some(x.saturating_add_unsigned(yu)),
        );
        let t = &a - Number::from(yu);
        forms_signed(
            "sub_unsigned",
            &t,
            x.overflowing_sub_unsigned(yu),
            x.checked_sub_unsigned(yu),
            x.wrapping_sub_unsigned(yu),
            Some(x.saturating_sub_unsigned(yu)),
        );
        assert_eq!(Number::from(x.abs_diff(y)), (&a - &b).abs());
        assert_eq!(Number::from(x.midpoint(y)), (&a + &b) / Number::from(2u64));

        // Quotients: truncated, the remainder with the dividend's sign; and Euclid's.
        if y.is_zero() {
            assert_eq!(x.checked_div(y), None);
            assert!(outcome(|| x % y).is_err());
        } else {
            let (q, r) = a.div_rem(&b);
            let (qe, re) = (a.div_euclid(&b), a.rem_euclid(&b));
            let over = exact_signed::<N>(&q).is_none();
            assert_eq!(x.checked_div(y), exact_signed(&q));
            assert_eq!(x.overflowing_div(y), (wrap_signed(&q), over));
            assert_eq!(x.wrapping_div(y), wrap_signed(&q));
            assert_eq!(x.saturating_div(y), clamp_signed(&q));
            assert_eq!(x.checked_rem(y), (!over).then(|| exact_signed(&r).unwrap()));
            assert_eq!(x.overflowing_rem(y), (exact_signed(&r).unwrap(), over));
            assert_eq!(x.checked_div_euclid(y), exact_signed(&qe));
            assert_eq!(x.overflowing_div_euclid(y), (wrap_signed(&qe), over));
            assert_eq!(
                x.checked_rem_euclid(y),
                (!over).then(|| exact_signed(&re).unwrap())
            );
            assert_eq!(x.wrapping_rem_euclid(y), exact_signed(&re).unwrap());
            if over {
                // MIN / −1 and MIN % −1 panic whatever the profile, as i64's.
                assert!(outcome(|| x / y).is_err());
                assert!(outcome(|| x % y).is_err());
                assert!(outcome(|| x.div_euclid(y)).is_err());
                assert!(outcome(|| x.rem_euclid(y)).is_err());
                assert!(outcome(|| x.div_rem(y)).is_err());
            } else {
                assert_eq!(x / y, exact_signed(&q).unwrap());
                assert_eq!(x % y, exact_signed(&r).unwrap());
                assert_eq!(x.div_rem(y), (x / y, x % y));
                assert_eq!(x.div_euclid(y), exact_signed(&qe).unwrap());
                assert_eq!(x.rem_euclid(y), exact_signed(&re).unwrap());
            }
        }

        // Shifts: the right shift arithmetic, Number's `>>` rounding toward −∞ as it does.
        for k in amounts(rng, bits) {
            let s = (k % bits) as usize;
            assert_eq!(
                x.overflowing_shl(k),
                (wrap_signed::<N>(&(&a << s)), k >= bits)
            );
            assert_eq!(
                x.overflowing_shr(k),
                (exact_signed(&(&a >> s)).unwrap(), k >= bits)
            );
            let u = if k >= bits {
                Int::ZERO
            } else {
                wrap_signed::<N>(&(&a << k as usize))
            };
            assert_eq!(x.unbounded_shl(k), u);
            assert_eq!(
                x.unbounded_shr(k),
                exact_signed(&(&a >> k as usize)).unwrap()
            );
            let shifted = outcome(|| x >> (k as u128));
            if k >= bits && cfg!(debug_assertions) {
                assert!(shifted.is_err());
            } else {
                assert_eq!(shifted, Ok(exact_signed(&(&a >> s)).unwrap()));
            }
            assert_eq!(x.rotate_left(k).rotate_right(k), x);
            assert_eq!(
                x.rotate_left(k).cast_unsigned(),
                x.cast_unsigned().rotate_left(k)
            );
        }

        // Powers, roots and logarithms.
        let room = 2 * bits / x.unsigned_abs().bits().max(1);
        for e in [
            0,
            1,
            2,
            3,
            rng.below(room as u64 + 1) as u32,
            room,
            room + 1,
        ] {
            let t = a.pow(e);
            forms_signed(
                "pow",
                &t,
                x.overflowing_pow(e),
                x.checked_pow(e),
                x.wrapping_pow(e),
                Some(x.saturating_pow(e)),
            );
            operator_signed("pow", &t, outcome(|| x.pow(e)));
        }
        if x.is_negative() {
            assert_eq!(x.checked_isqrt(), None);
            assert!(outcome(|| x.isqrt()).is_err());
            assert_eq!(x.checked_ilog10(), None);
            assert!(outcome(|| x.ilog2()).is_err());
        } else {
            assert_eq!(Number::from(x.isqrt()), a.isqrt());
            if !x.is_zero() {
                assert_eq!(x.ilog2() as u64, a.bits() - 1);
                assert_eq!(x.ilog10(), log(&a, &Number::from(10u64)));
                if y.is_positive() && y > Int::ONE {
                    assert_eq!(x.ilog(y), log(&a, &b));
                }
            }
        }

        // Bits of the two's complement, text and conversions.
        let twos = wrap::<N>(&a);
        assert_eq!(x.cast_unsigned(), twos);
        assert_eq!(x.count_ones(), twos.count_ones());
        assert_eq!(x.leading_zeros(), twos.leading_zeros());
        assert_eq!(Number::from(!x), -&a - Number::from(1u64));
        assert_eq!(Number::from(x & y), &a & &b);
        assert_eq!(Number::from(x | y), &a | &b);
        assert_eq!(Number::from(x ^ y), &a ^ &b);
        assert_eq!(x.to_string(), a.to_string());
        let radix = 2 + rng.below(35) as u32;
        let text = x.to_str_radix(radix);
        assert_eq!(text, a.to_str_radix(radix));
        assert_eq!(Int::<N>::from_str_radix(&text, radix), Ok(x));
        assert_eq!(format!("{x:#x}"), format!("{twos:#x}"));
        assert_eq!(format!("{x:>+150}"), format!("{a:>+150}"));
        assert_eq!(format!("{x:0150}"), format!("{a:0150}"));
        assert_eq!(x.to_f64().to_bits(), a.to_f64().to_bits(), "{x:?}");
        assert_eq!(x.to_f32().to_bits(), a.to_f32().to_bits(), "{x:?}");
        assert_eq!(x.try_resize::<1>(), exact_signed::<1>(&a));
        assert_eq!(x.resize::<1>(), wrap_signed::<1>(&a));
        assert_eq!(x.resize::<12>(), wrap_signed::<12>(&a));
        assert_eq!(x.try_resize::<12>(), exact_signed::<12>(&a));
        assert_eq!(Uint::<N>::try_from(x).ok(), exact::<N>(&a));
        let mut le = vec![0; 8 * N];
        x.write_le_bytes(&mut le);
        assert_eq!(Int::<N>::from_le_slice(&le), Some(x));
        let fill = if x.is_negative() { 0xFF } else { 0 };
        le.extend([fill, fill]);
        assert_eq!(Int::<N>::from_le_slice(&le), Some(x));
        le.push(!fill);
        assert_eq!(Int::<N>::from_le_slice(&le), None);
        assert!(a < half && a >= -&half);
    }
}

macro_rules! model_tests {
    ($($test:ident $n:literal $count:literal),*) => {$(
        #[test]
        fn $test() {
            let mut rng = Rng(0x0de1_0000 + $n);
            unsigned::<$n>(&mut rng, size($count, 2));
            signed::<$n>(&mut rng, size($count, 6));
        }
    )*};
}

model_tests!(
    every_operation_agrees_with_number_at_1_limb 1 300,
    every_operation_agrees_with_number_at_2_limbs 2 300,
    every_operation_agrees_with_number_at_3_limbs 3 300,
    every_operation_agrees_with_number_at_4_limbs 4 300,
    every_operation_agrees_with_number_at_5_limbs 5 250,
    every_operation_agrees_with_number_at_6_limbs 6 250,
    every_operation_agrees_with_number_at_7_limbs 7 200,
    every_operation_agrees_with_number_at_8_limbs 8 200,
    every_operation_agrees_with_number_at_9_limbs 9 200,
    every_operation_agrees_with_number_at_16_limbs 16 80,
    every_operation_agrees_with_number_at_32_limbs 32 30
);

/// The same comparisons, twenty times as many, over more widths: too slow for the debug
/// profile of CI, run in release with `cargo test --release --test math -- --ignored uint::model`.
#[test]
#[ignore]
fn the_heavier_sweep_agrees_with_number_at_every_width() {
    macro_rules! sweep {
        ($($n:literal $count:literal),*) => {$(
            let mut rng = Rng(0x5eee_0000 + $n);
            unsigned::<$n>(&mut rng, $count);
            signed::<$n>(&mut rng, $count);
        )*};
    }
    sweep!(1 6000, 2 6000, 3 6000, 4 6000, 5 5000, 6 5000, 7 4000, 8 4000, 9 4000, 10 2000,
        11 2000, 12 2000, 13 1500, 15 1500, 16 1500, 17 1000, 24 600, 31 500, 32 500, 64 100);
}
