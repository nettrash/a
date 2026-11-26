//! `+`, `-`, `*`, `/`, the fused multiply-add and the square root against the exact results
//! rounded by integers, in every mode, at random precisions, with long runs of zero and one bits.

use super::common::{
    MODES, below, float, low_bits, mode, one, rational, reference, same, size, sqrt_mod_pow2,
    xorshift,
};
use a::math::float::{BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;
use core::cmp::Ordering;

/// Checks `got` against the exact rational `exact` rounded, or the zero `zero` when it is zero.
#[track_caller]
fn check(
    got: (BigFloat, Ordering),
    exact: &Rational,
    zero: bool,
    prec: u64,
    round: Round,
    what: &str,
) {
    if exact.is_zero() {
        let want = if zero {
            BigFloat::neg_zero(prec)
        } else {
            BigFloat::zero(prec)
        };
        same(&got, &(want, Ordering::Equal), what);
    } else {
        same(&got, &reference(exact, prec, round), what);
    }
}

/// Sums and quotients of operands of the result's limbs and of one long operand with a short one
/// (the in-place and padded kernels), the short one's exponent below, equal or above, both
/// signs and every mode, into a new number, into an existing one and in place, against the exact
/// rationals; long ones of up to 200 limbs.
#[test]
fn sums_and_quotients_of_long_and_short_operands_round_correctly() {
    let mut s = 0x5EED_F10A_7000_0077u64;
    let cases = size(1200, 16);
    let mut long_short = 0;
    for i in 0..cases {
        let prec = match i % 4 {
            0 => 129 + below(&mut s, 400),
            1 => 400 + below(&mut s, 700),
            2 if !cfg!(miri) => 2000 + below(&mut s, 11_000),
            _ => 129 + below(&mut s, 128),
        };
        // Full mantissas of `bits` bits, of either sign, `0.m·2^e` for `e` in `-4..=4` or given.
        let full = |s: &mut u64, bits: u64, e: Option<i64>| {
            let m = super::common::mantissa(s, bits);
            let m = if xorshift(s) & 1 == 1 { -m } else { m };
            let e = e.unwrap_or_else(|| below(s, 9) as i64 - 4);
            BigFloat::from_parts(&m, e - bits as i64, bits, Round::Nearest).0
        };
        let a = full(&mut s, prec, None);
        // A short operand, its exponent from far below to above; or one of the precision.
        let short = 1 + below(&mut s, 250);
        let b = match i % 3 {
            0 => full(&mut s, prec, None),
            _ => {
                // From far above to far below, and a few bits either way most often.
                let gap = match below(&mut s, 3) {
                    0 => below(&mut s, 2 * prec + 600) as i64 - prec as i64 - 300,
                    1 => below(&mut s, 40) as i64 - 20,
                    _ => below(&mut s, prec + 300) as i64 - 20,
                };
                let e = a.exponent().unwrap() - gap;
                full(&mut s, short, Some(e))
            }
        };
        let round = mode(&mut s);
        let (ra, rb) = (rational(&a), rational(&b));
        let down = round == Round::Down;
        if b.prec() < 64 * prec.div_ceil(64) - 63 {
            long_short += 1;
        }
        for (x, y, rx, ry) in [(&a, &b, &ra, &rb), (&b, &a, &rb, &ra)] {
            let what = format!("{x:?} ± {y:?} at {prec} {round:?}");
            check(
                x.add_round(y, prec, round),
                &(rx + ry),
                down,
                prec,
                round,
                &what,
            );
            check(
                x.sub_round(y, prec, round),
                &(rx - ry),
                down,
                prec,
                round,
                &what,
            );
            let mut z = BigFloat::from(7).with_prec(prec);
            let o = z.set_add(x, y, round);
            check((z, o), &(rx + ry), down, prec, round, &what);
            let mut z = x.prec_round(prec, Round::Nearest).0;
            let rz = rational(&z);
            let o = z.sub_assign_round(y, round);
            check((z, o), &(&rz - ry), down, prec, round, &what);
            if !y.is_zero() {
                check(
                    x.div_round(y, prec, round),
                    &(rx / ry),
                    false,
                    prec,
                    round,
                    &what,
                );
                let mut z = x.prec_round(prec, Round::Nearest).0;
                let rz = rational(&z);
                let o = z.div_assign_round(y, round);
                check((z, o), &(&rz / ry), false, prec, round, &what);
            }
        }
    }
    // Half natively; under Miri 6 of its 16 cases have one, so a quarter there.
    assert!(
        long_short >= size(cases / 2, cases / 4),
        "{long_short} of {cases} with a short operand"
    );
}

/// Quotients to at most 62 bits of operands of several limbs, from their top limbs (one division
/// and the bounds of the interval, or the two ends of it), with and without bits below the top
/// limbs, every mode, against the exact rationals.
#[test]
fn short_quotients_of_long_operands_round_correctly() {
    let mut s = 0x5EED_F10A_7000_0078u64;
    for i in 0..size(3000, 30) {
        let bits = |s: &mut u64| 65 + below(s, 1200);
        let (pa, pb) = (bits(&mut s), bits(&mut s));
        let full = |s: &mut u64, p: u64| {
            let mut m = super::common::mantissa(s, p);
            if i % 3 == 0 {
                // The top limb alone: the bits below it zeros.
                m = &(&m >> (p as usize - 64)) << (p as usize - 64);
            }
            let m = if xorshift(s) & 1 == 1 { -m } else { m };
            BigFloat::from_parts(&m, below(s, 9) as i64 - 4 - p as i64, p, Round::Nearest).0
        };
        let (a, b) = (full(&mut s, pa), full(&mut s, pb));
        let prec = 1 + below(&mut s, 62);
        let (ra, rb) = (rational(&a), rational(&b));
        for round in MODES {
            check(
                a.div_round(&b, prec, round),
                &(&ra / &rb),
                false,
                prec,
                round,
                &format!("{a:?} / {b:?} at {prec} {round:?}"),
            );
            let mut z = BigFloat::zero(prec);
            let o = z.set_div(&a, &b, round);
            check((z, o), &(&ra / &rb), false, prec, round, "set_div");
        }
    }
}

/// Products, squares and fused products of operands of the result's limbs (the short products
/// with guard limbs, `mul_high` and `fma_high`), from 9 to 200 limbs, every mode, into a new number
/// and into an existing one, against the exact rationals: random mantissas, and hard ones whose
/// products lie next to a number of the precision or a point halfway (all ones, a power of two
/// less one, runs of ones with a few bits apart), for which the short product must leave it to
/// the whole one; addends that nearly cancel the product.
#[test]
fn products_of_long_operands_of_the_precision_round_correctly() {
    let mut s = 0x5EED_F10A_7000_0079u64;
    let cases = size(400, 6);
    for i in 0..cases {
        let n = 9 + below(&mut s, if i % 5 == 0 { 192 } else { 60 });
        let prec = 64 * n - below(&mut s, if i % 3 == 0 { 1 } else { 64 });
        let hard = |s: &mut u64| -> Number {
            match below(s, 4) {
                // All ones: `(2^p - 1)^2 = 2^2p - 2^(p+1) + 1`.
                0 => &(&one() << prec as usize) - &one(),
                // A power of two and a few bits apart.
                1 => (&one() << (prec - 1) as usize) + (&one() << below(s, prec - 1) as usize),
                // Ones, then a zero, then ones.
                2 => {
                    let k = 1 + below(s, prec - 2);
                    &(&one() << prec as usize) - &(&one() << k as usize) - &one()
                }
                _ => super::common::mantissa(s, prec),
            }
        };
        let x = |s: &mut u64, hard_one: bool| {
            let m = if hard_one {
                hard(s)
            } else {
                super::common::mantissa(s, prec)
            };
            let m = if xorshift(s) & 1 == 1 { -m } else { m };
            BigFloat::from_parts(
                &m,
                below(s, 9) as i64 - 4 - prec as i64,
                prec,
                Round::Nearest,
            )
            .0
        };
        let h = i % 2 == 0;
        let (a, b) = (x(&mut s, h), x(&mut s, h && i % 4 == 0));
        let (ra, rb) = (rational(&a), rational(&b));
        // An addend of the fused product: of the precision, or `-a·b` rounded (a cancellation).
        let c = if i % 3 == 0 {
            -a.mul_round(&b, prec, Round::Nearest).0
        } else {
            x(&mut s, false)
        };
        let rc = rational(&c);
        for round in MODES {
            let what = format!("{n} limbs at {prec} {round:?}");
            let down = round == Round::Down;
            let fma = a.mul_add_round(&b, &c, prec, round);
            check(fma, &(&ra * &rb + &rc), down, prec, round, &what);
            let fma = a.mul_add_round(&a, &c, prec, round);
            check(fma, &(&ra * &ra + &rc), down, prec, round, &what);
            check(
                a.mul_round(&b, prec, round),
                &(&ra * &rb),
                false,
                prec,
                round,
                &what,
            );
            check(
                a.mul_round(&a, prec, round),
                &(&ra * &ra),
                false,
                prec,
                round,
                &what,
            );
            let mut z = BigFloat::zero(prec);
            let o = z.set_mul(&a, &a, round);
            check((z, o), &(&ra * &ra), false, prec, round, &what);
        }
    }
}

#[test]
fn sums_differences_products_quotients_and_fmas_round_correctly() {
    let mut s = 0x5EED_F10A_7000_0001u64;
    for i in 0..size(4000, 40) {
        let max = if i % 10 == 0 { 700 } else { 140 };
        let a = float(&mut s, max, 80);
        // Operands near each other, apart, and far apart.
        let b = match below(&mut s, 4) {
            // `-a` at another precision: the sum cancels.
            0 => {
                let p = 1 + below(&mut s, max);
                (-&a).prec_round(p, mode(&mut s)).0
            }
            1 => float(&mut s, max, 400),
            _ => float(&mut s, max, 80),
        };
        let c = float(&mut s, max, 160);
        let prec = 1 + below(&mut s, if i % 7 == 0 { 600 } else { 150 });
        let round = mode(&mut s);
        let (ra, rb, rc) = (rational(&a), rational(&b), rational(&c));
        let down = round == Round::Down;
        check(
            a.add_round(&b, prec, round),
            &(&ra + &rb),
            down,
            prec,
            round,
            &format!("{a:?} + {b:?}"),
        );
        check(
            a.sub_round(&b, prec, round),
            &(&ra - &rb),
            down,
            prec,
            round,
            &format!("{a:?} - {b:?}"),
        );
        check(
            a.mul_round(&b, prec, round),
            &(&ra * &rb),
            false,
            prec,
            round,
            &format!("{a:?} * {b:?}"),
        );
        check(
            a.div_round(&b, prec, round),
            &(&ra / &rb),
            false,
            prec,
            round,
            &format!("{a:?} / {b:?}"),
        );
        check(
            a.mul_add_round(&b, &c, prec, round),
            &(&ra * &rb + &rc),
            down,
            prec,
            round,
            &format!("{a:?} * {b:?} + {c:?}"),
        );
    }
}

#[test]
fn a_tiny_term_rounds_as_a_sticky_bit() {
    // `1 ± 2^-k` for k past the precision, in every mode: the sticky path of the sum.
    for prec in [1u64, 2, 52, 53, 54, 63, 64, 65, 127, 128, 129, 200] {
        for k in [
            prec + 1,
            prec + 2,
            prec + 3,
            prec + 4,
            prec + 10,
            prec + 200,
            10_000,
        ] {
            let tiny = BigFloat::from_parts(&one(), -(k as i64), 1, Round::Nearest).0;
            for round in MODES {
                for x in [
                    BigFloat::from(1),
                    BigFloat::from(-1),
                    BigFloat::from(3),
                    BigFloat::from(0.75),
                ] {
                    for t in [tiny.clone(), -&tiny] {
                        let exact = rational(&x) + rational(&t);
                        same(
                            &x.add_round(&t, prec, round),
                            &reference(&exact, prec, round),
                            &format!("{x:?} + {t:?} at {prec} {round:?}"),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn square_roots_round_correctly() {
    let mut s = 0x5EED_F10A_7000_0002u64;
    for i in 0..size(3000, 30) {
        let x = float(&mut s, if i % 10 == 0 { 600 } else { 130 }, 120).abs();
        let prec = 1 + below(&mut s, if i % 7 == 0 { 500 } else { 140 });
        let round = mode(&mut s);
        let (r, o) = x.sqrt_round(prec, round);
        let q = rational(&x);
        let rr = rational(&r);
        // The direction: the square of the result against `x`.
        let sq = &rr * &rr;
        assert_eq!(o, sq.cmp(&q), "sqrt({x:?}) = {r:?} at {prec} {round:?}");
        // The neighbours of the result at the precision bracket the root as the mode has it.
        let (m, e) = r.mantissa_exponent().unwrap();
        // The neighbours at the precision: the predecessor of `2^(p-1)·2^e` is half a unit
        // below, `(2^p - 1)·2^(e-1)`.
        let neighbour = |k: i64| {
            let n = &m + &Number::new_from_i64(k, ByteOrder::LittleEndian);
            if n.bits() < prec {
                let n = (&n << 1usize) + &Number::from(1);
                BigFloat::from_parts(&n, e - 1, prec + 1, Round::Nearest).0
            } else {
                BigFloat::from_parts(&n, e, prec + 1, Round::Nearest).0
            }
        };
        let (below_r, above_r) = (rational(&neighbour(-1)), rational(&neighbour(1)));
        let lower_half = (&below_r + &rr) / Rational::from(2);
        let upper_half = (&rr + &above_r) / Rational::from(2);
        match round {
            Round::Nearest => {
                // `x` between the squares of the halfway points, a tie to the even mantissa.
                let (l, u) = (&lower_half * &lower_half, &upper_half * &upper_half);
                assert!(l <= q && q <= u, "sqrt({x:?}) = {r:?} at {prec}");
                if q == l || q == u {
                    assert!(!m.bit(0), "sqrt({x:?}) = {r:?}: a tie to an odd mantissa");
                }
            }
            Round::Down | Round::TowardZero => assert!(
                sq <= q && q < &above_r * &above_r,
                "sqrt({x:?}) = {r:?} at {prec} {round:?}"
            ),
            Round::Up => assert!(
                &below_r * &below_r < q && q <= sq,
                "sqrt({x:?}) = {r:?} at {prec} {round:?}"
            ),
        }
    }
}

#[test]
fn exact_squares_have_exact_roots() {
    let mut s = 0x5EED_F10A_7000_0003u64;
    for _ in 0..size(500, 10) {
        let r = float(&mut s, 200, 100).abs();
        let x = r.mul_round(&r, 2 * r.prec(), Round::Nearest).0;
        let (got, o) = x.sqrt_round(r.prec(), mode(&mut s));
        assert_eq!(o, Ordering::Equal);
        assert_eq!(got, r);
    }
}

#[test]
fn special_values_follow_ieee_754() {
    let p = 20;
    let nan = BigFloat::nan(p);
    let (inf, ninf) = (BigFloat::infinity(p), BigFloat::neg_infinity(p));
    let (z, nz) = (BigFloat::zero(p), BigFloat::neg_zero(p));
    let one = BigFloat::from(1);
    let r = Round::Nearest;
    assert!(inf.sub_round(&inf, p, r).0.is_nan());
    assert!(inf.add_round(&ninf, p, r).0.is_nan());
    assert!(z.mul_round(&inf, p, r).0.is_nan());
    assert!(z.div_round(&z, p, r).0.is_nan());
    assert!(inf.div_round(&ninf, p, r).0.is_nan());
    assert!(one.add_round(&nan, p, r).0.is_nan());
    assert!(BigFloat::from(-1).sqrt().is_nan());
    assert!(ninf.sqrt().is_nan());
    assert_eq!(inf.add_round(&one, p, r).0, inf);
    assert_eq!(one.div_round(&z, p, r).0, inf);
    assert_eq!(one.div_round(&nz, p, r).0, ninf);
    assert_eq!(one.div_round(&inf, p, r).0, z);
    assert!(one.div_round(&ninf, p, r).0.is_sign_negative());
    // The signs of zero sums.
    for round in MODES {
        let down = round == Round::Down;
        assert_eq!(z.add_round(&nz, p, round).0.is_sign_negative(), down);
        assert!(nz.add_round(&nz, p, round).0.is_sign_negative());
        assert!(!z.add_round(&z, p, round).0.is_sign_negative());
        assert_eq!(one.sub_round(&one, p, round).0.is_sign_negative(), down);
        assert_eq!(nz.sub_round(&nz, p, round).0.is_sign_negative(), down);
        assert!(
            z.mul_round(&BigFloat::from(-2), p, round)
                .0
                .is_sign_negative()
        );
        let (fz, _) = z.mul_add_round(&BigFloat::from(-3), &z, p, round);
        assert_eq!(fz.is_sign_negative(), down);
    }
    assert!(nz.sqrt().is_sign_negative() && nz.sqrt().is_zero());
    // `x + 0` and `0 + x` round `x`.
    let x = BigFloat::from(0.1);
    let (y, o) = x.add_round(&z, 10, Round::Up);
    assert_eq!((y.clone(), o), x.prec_round(10, Round::Up));
    let (y, o) = nz.sub_round(&x, 10, Round::Up);
    let (w, ow) = (-&x).prec_round(10, Round::Up);
    assert_eq!((y, o), (w, ow));
}

#[test]
fn the_exponent_range_overflows_and_underflows_by_the_mode() {
    let big = BigFloat::from_parts(&Number::from(3), BigFloat::EXP_MAX - 2, 2, Round::Nearest).0;
    assert_eq!(big.exponent(), Some(BigFloat::EXP_MAX));
    for (round, inf) in [
        (Round::Nearest, true),
        (Round::TowardZero, false),
        (Round::Up, true),
        (Round::Down, false),
    ] {
        let (x, o) = big.mul_round(&BigFloat::from(2), 10, round);
        assert_eq!(x.is_infinite(), inf, "{round:?}");
        assert_eq!(
            o,
            if inf {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        );
        if !inf {
            assert_eq!(x.exponent(), Some(BigFloat::EXP_MAX));
            assert_eq!(x.mantissa_exponent().unwrap().0, Number::from(1023));
        }
        let (y, _) = (-&big).mul_round(&BigFloat::from(2), 10, round);
        assert_eq!(
            y.is_infinite(),
            round == Round::Nearest || round == Round::Down
        );
    }
    let small = BigFloat::from_parts(&Number::from(1), BigFloat::EXP_MIN - 1, 2, Round::Nearest).0;
    assert_eq!(small.exponent(), Some(BigFloat::EXP_MIN));
    let quarter = BigFloat::from(0.25);
    let half = BigFloat::from(0.5);
    let three_quarters = BigFloat::from(0.75);
    for round in MODES {
        // Half the smallest is a tie: to nearest it goes to zero.
        let (x, _) = small.mul_round(&half, 10, round);
        assert_eq!(x.is_zero(), round != Round::Up, "{round:?}");
        let (x, _) = small.mul_round(&three_quarters, 10, round);
        assert_eq!(
            x.is_zero(),
            round == Round::TowardZero || round == Round::Down,
            "{round:?}"
        );
        let (x, _) = small.mul_round(&quarter, 10, round);
        assert_eq!(x.is_zero(), round != Round::Up, "{round:?}");
        let (x, _) = (-&small).mul_round(&three_quarters, 10, round);
        assert_eq!(
            x.is_zero(),
            round == Round::TowardZero || round == Round::Up,
            "{round:?}"
        );
        assert!(x.is_sign_negative());
    }
}

#[test]
fn operators_round_to_nearest_at_the_larger_precision() {
    let a = BigFloat::from(1).with_prec(10);
    let b = BigFloat::from(3).with_prec(100);
    let q = &a / &b;
    assert_eq!(q.prec(), 100);
    assert_eq!(q, a.div_round(&b, 100, Round::Nearest).0);
    let mut c = a.clone();
    c += &b;
    assert_eq!(c, BigFloat::from(4));
    assert_eq!(c.prec(), 100);
    assert_eq!((-&c).abs(), c);
    assert_eq!(a.clone() * b.clone(), b);
    assert_eq!((&b - &a).prec(), 100);
    assert_eq!(a.mul_add(&b, &b).prec(), 100);
}

#[test]
fn rounding_to_a_precision_agrees_with_the_reference() {
    let mut s = 0x5EED_F10A_7000_0004u64;
    for _ in 0..size(3000, 30) {
        let x = float(&mut s, 400, 1000);
        let prec = 1 + below(&mut s, 200);
        let round = mode(&mut s);
        same(
            &x.prec_round(prec, round),
            &reference(&rational(&x), prec, round),
            &format!("{x:?} to {prec}"),
        );
        let _ = xorshift(&mut s);
    }
}

/// `±M·2^e` of `prec` bits from the integer `m` (exactly).
fn exact(m: &Number, negative: bool, e: i64, prec: u64) -> BigFloat {
    let m = if negative { -m } else { m.clone() };
    let (x, o) = BigFloat::from_parts(&m, e, prec, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    x
}

/// The limbs `n` of the products' operands and result on both sides of the short products'
/// thresholds: padded to `n + 1` limbs, short from 27 to 399 for products and from 30 to 655 for
/// squares.
const PRODUCT_EDGES: [u64; 10] = [25, 26, 28, 29, 63, 200, 398, 399, 654, 655];

/// The same for the fused products, padded to `n + 2` limbs.
const FMA_EDGES: [u64; 10] = [24, 25, 27, 28, 63, 150, 397, 398, 653, 654];

/// Products and squares of `n` limbs (on both sides of the short products' thresholds,
/// [`PRODUCT_EDGES`]) whose exact values lie `j` units of the short product's last limb from a
/// number of the precision or a point halfway, above it or below, for `j` from 0 to past the
/// short product's error band (`n` units): within it the short product leaves the rounding to the
/// whole one, past it the short product decides. At `64n` bits the bits that decide are those of
/// `x·y` from `64(n-1)` to `64n - 2`, under `2^(64n-1)`: for `x` random, `y = T·x^(-1) mod
/// 2^(64n-1)` (with its top bit set) gives `x·y ≡ T`, `T` of those bits chosen and random bits
/// below (so that the short product drops partial products that are not zero); for a square `x ≡
/// √T` (Newton's iteration on the 2-adic square root). Every mode and both signs, against the
/// exact rationals.
#[test]
fn products_next_to_a_boundary_round_correctly_on_both_sides_of_the_band() {
    let mut s = 0x5EED_B0DE_0000_0003u64;
    let one = one();
    let mut checked = 0;
    for &n in &PRODUCT_EDGES[..size(PRODUCT_EDGES.len(), 2)] {
        let (prec, k) = (64 * n, 64 * n - 1);
        let top = &one << k as usize;
        let modulus = &one << k as usize;
        let js = [
            0,
            1,
            2,
            n - 1,
            n,
            n + 1,
            n + 2,
            2 * n,
            3 * n + 3,
            below(&mut s, 64 * n),
        ];
        for &j in &js[..size(js.len(), 4)] {
            for (high, mid) in [(false, false), (false, true), (true, false), (true, true)] {
                // The last 63 bits of `⌊x·y / 2^(64(n-1))⌋`: `j` above a number of the precision
                // (a multiple of `2^63`) or a point halfway (`2^62` more), or `j` below one.
                let base = Number::from(u64::from(mid) << 62);
                let pattern = if high {
                    &(&base + &(&one << 62usize)) - &Number::from(j)
                } else {
                    &base + &Number::from(j)
                };
                let pattern = low_bits(&pattern, 63);
                let target = |s: &mut u64| {
                    let r = super::common::mantissa(s, 64 * n - 64);
                    let r = low_bits(&r, 64 * n - 64);
                    &(&pattern << (64 * n - 64) as usize) + &r
                };
                // `x·y`, of `128n - 1` bits.
                let (x, y) = loop {
                    let x =
                        &(&(&super::common::mantissa(&mut s, 64 * n) >> 1usize) << 1usize) + &one;
                    let t = target(&mut s);
                    let y = &low_bits(&(&t * &x.mod_inverse(&modulus).unwrap()), k) + &top;
                    if (&x * &y).bits() == 128 * n - 1 && !low_bits(&y, 64).is_zero() {
                        break (x, y);
                    }
                };
                // `z²`, of `128n - 1` bits.
                let z = loop {
                    let t = target(&mut s);
                    let t = &(&(&t >> 3usize) << 3usize) + &one;
                    let z = &sqrt_mod_pow2(&t, k) + &top;
                    if (&z * &z).bits() == 128 * n - 1 {
                        break z;
                    }
                };
                let negative = xorshift(&mut s) & 1 == 1;
                let e = -(64 * n as i64) + below(&mut s, 5) as i64 - 2;
                let (fx, fy, fz) = (
                    exact(&x, negative, e, prec),
                    exact(&y, false, e, prec),
                    exact(&z, negative, e, prec),
                );
                let (rx, ry, rz) = (rational(&fx), rational(&fy), rational(&fz));
                for round in MODES {
                    let what = format!("{n} limbs, j = {j} high {high} mid {mid} {round:?}");
                    check(
                        fx.mul_round(&fy, prec, round),
                        &(&rx * &ry),
                        false,
                        prec,
                        round,
                        &what,
                    );
                    check(
                        fz.mul_round(&fz, prec, round),
                        &(&rz * &rz),
                        false,
                        prec,
                        round,
                        &what,
                    );
                    let mut t = BigFloat::zero(prec);
                    let o = t.set_mul(&fx, &fy, round);
                    check((t, o), &(&rx * &ry), false, prec, round, &what);
                    let mut t = fz.clone();
                    let o = t.mul_assign_round(&fz, round);
                    check((t, o), &(&rz * &rz), false, prec, round, &what);
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, size(10 * 10 * 4 * 4, 2 * 4 * 4 * 4));
}

/// Fused products `x·y + z` and `x·x + z` of random factors of `n` limbs ([`FMA_EDGES`]) whose
/// exact values lie `k` units of `2^q` from a number of the precision or a point halfway, `q`
/// within three bits of the window's last bit `lo = 128n - prec - 67` (in units of the factors'
/// last bits squared) and `k` from -4 to 4, with and without a last bit: the window of the short
/// product decides there or leaves it to the whole one. `z = -(P mod 2^L) + k·2^q (+ 2^(L-1)) (+
/// 1)` for the exact product `P` and the last place `L` of the sum. Every mode, against the exact
/// rationals.
#[test]
fn fused_products_next_to_a_boundary_round_correctly_at_the_windows_edge() {
    let mut s = 0x5EED_F00D_0000_0005u64;
    let one = one();
    let mut checked = 0;
    for &n in &FMA_EDGES[..size(FMA_EDGES.len(), 2)] {
        let prec = 64 * n - below(&mut s, 2) * below(&mut s, 64);
        // Odd factors of `64n` bits: their last limbs are not zero.
        let full = |s: &mut u64| {
            let m = super::common::mantissa(s, 64 * n);
            &(&(&m >> 1usize) << 1usize) + &one
        };
        let (x, y) = (full(&mut s), full(&mut s));
        let e = -(64 * n as i64);
        let (fx, fy) = (exact(&x, false, e, 64 * n), exact(&y, false, e, 64 * n));
        let lo = 128 * n as i64 - prec as i64 - 67;
        for square in [false, true] {
            let p = if square { &x * &x } else { &x * &y };
            let l = p.bits() - prec;
            let low = &p - &(&(&p >> l as usize) << l as usize);
            let cases = size(7 * 9 * 2 * 2, 6);
            for c in 0..cases {
                let q = lo + (c % 7) as i64 - 3;
                let k = (c / 7 % 9) as i64 - 4;
                let tail = (c / 63 % 2) as i64;
                let mid = c / 126 % 2 == 1;
                let mut zm = -&low + &Number::from(tail);
                if mid {
                    zm = &zm + &(&one << (l - 1) as usize);
                }
                let step = &one << q as usize;
                zm = &zm + &(&Number::from(k) * &step);
                if zm.is_zero() {
                    continue;
                }
                // `z` with all its bits: `|z| < 2^(L + 1)`.
                let (fz, o) = BigFloat::from_parts(&zm, 2 * e, l + 2, Round::Nearest);
                assert_eq!(o, Ordering::Equal);
                let (rx, ry, rz) = (rational(&fx), rational(&fy), rational(&fz));
                for round in MODES {
                    let what =
                        format!("{n} limbs at {prec}, q = lo{:+}, k = {k} {round:?}", q - lo);
                    let (got, exact) = if square {
                        (fx.mul_add_round(&fx, &fz, prec, round), &(&rx * &rx) + &rz)
                    } else {
                        (fx.mul_add_round(&fy, &fz, prec, round), &(&rx * &ry) + &rz)
                    };
                    check(got, &exact, round == Round::Down, prec, round, &what);
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= size(7 * 2 * 250 * 4, 2 * 2 * 5 * 4));
}
