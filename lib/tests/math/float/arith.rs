//! `+`, `-`, `*`, `/`, the fused multiply-add and the square root against the exact results
//! rounded by integers, in every mode, at random precisions, with long runs of zero and one bits.

use super::common::{MODES, below, float, mode, one, rational, reference, same, size, xorshift};
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
