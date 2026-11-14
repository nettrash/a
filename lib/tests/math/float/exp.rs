//! `exp`, `exp2`, `exp10` and `exp_m1` against MPFR's vectors, and their exact results, ranges
//! and identities.

use super::common::{MODES, below, float, pow2, rational, reference, same, size};
use super::explog_vectors::{check, modes, near_threshold};
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use core::cmp::Ordering;

fn exps(f: &str, x: &BigFloat, p: u64, r: Round) -> (BigFloat, Ordering) {
    match f {
        "exp" => x.exp_round(p, r),
        "exp2" => x.exp2_round(p, r),
        "exp10" => x.exp10_round(p, r),
        "expm1" => x.exp_m1_round(p, r),
        _ => panic!("{f}"),
    }
}

#[test]
fn exponentials_match_mpfr() {
    check(include_str!("data/explog_exp.txt"), |f, x, _, p, r| {
        exps(f, x, p, r)
    });
}

#[test]
fn exact_and_special_values() {
    for r in MODES {
        for p in [1, 2, 53, 200] {
            let one = (BigFloat::from(1).with_prec(p), Ordering::Equal);
            same(&BigFloat::zero(8).exp_round(p, r), &one, "e^0");
            same(&BigFloat::neg_zero(8).exp2_round(p, r), &one, "2^-0");
            same(&BigFloat::zero(8).exp10_round(p, r), &one, "10^0");
            same(
                &BigFloat::neg_infinity(8).exp_round(p, r),
                &(BigFloat::zero(p), Ordering::Equal),
                "e^-inf",
            );
            same(
                &BigFloat::neg_infinity(8).exp_m1_round(p, r),
                &(BigFloat::from(-1).with_prec(p), Ordering::Equal),
                "e^-inf - 1",
            );
            same(
                &BigFloat::neg_zero(8).exp_m1_round(p, r),
                &(BigFloat::neg_zero(p), Ordering::Equal),
                "e^-0 - 1",
            );
            assert!(BigFloat::nan(8).exp_round(p, r).0.is_nan());
        }
    }
    // 2^n and 10^n exactly, rounded when 5^n is longer than the precision.
    let step = size(1, 20);
    for n in (-40i64..=40).step_by(step) {
        for r in MODES {
            for p in [3u64, 10, 24, 64] {
                let x = BigFloat::from(n);
                let want = if n >= 0 {
                    reference(&a::math::rational::Rational::from(pow2(n as u64)), p, r)
                } else {
                    reference(
                        &(a::math::rational::Rational::from(1)
                            / a::math::rational::Rational::from(pow2((-n) as u64))),
                        p,
                        r,
                    )
                };
                same(&x.exp2_round(p, r), &want, &format!("2^{n}"));
                if n >= 0 {
                    let ten = Number::from(10).pow(n as u32);
                    let want = reference(&a::math::rational::Rational::from(ten), p, r);
                    same(&x.exp10_round(p, r), &want, &format!("10^{n} at {p}"));
                }
            }
        }
    }
}

#[test]
fn far_beyond_the_range_at_once() {
    let huge = BigFloat::from_parts(&Number::from(1), 1 << 40, 8, Round::Nearest).0;
    for r in MODES {
        let (e, o) = huge.exp_round(53, r);
        match r {
            Round::Nearest | Round::Up => assert!(e.is_infinite() && o == Ordering::Greater),
            _ => assert!(e.is_normal() && o == Ordering::Less),
        }
        let (e, o) = (-&huge).exp_round(53, r);
        match r {
            Round::Up => assert!(e.is_normal() && o == Ordering::Greater),
            _ => assert!(e.is_zero() && o == Ordering::Less),
        }
        let (e, o) = (-&huge).exp_m1_round(53, r);
        assert_eq!(
            o,
            if matches!(r, Round::Up | Round::TowardZero) {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        );
        assert!(e >= -1);
        assert!(
            huge.exp10_round(24, r).0.is_infinite() || huge.exp10_round(24, r).1 == Ordering::Less
        );
        assert!(
            huge.exp2_round(24, r).0.is_infinite() || huge.exp2_round(24, r).1 == Ordering::Less
        );
    }
}

#[test]
fn exp_of_tiny_arguments_rounds_near_one() {
    let mut s = 0xE7_0001u64;
    for _ in 0..size(200, 5) {
        let p = 1 + below(&mut s, 300);
        let mut x = float(&mut s, 100, 10);
        // `|x| < 2^-(p + 3)`.
        x = x
            .mul_round(
                &BigFloat::from_parts(&Number::from(1), -(p as i64) - 20, 2, Round::Nearest).0,
                200,
                Round::Nearest,
            )
            .0;
        let positive = x.is_sign_positive();
        for r in MODES {
            let (e, o) = x.exp_round(p, r);
            let one = BigFloat::from(1);
            assert_eq!(
                o,
                if positive {
                    if matches!(r, Round::Up) {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                } else if matches!(r, Round::Down | Round::TowardZero) {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            );
            assert!(
                e == one || o == Ordering::Greater && e > one || o == Ordering::Less && e < one
            );
            // `e^x - 1` lies just above `x`.
            let (m, o) = x.exp_m1_round(p, r);
            let q = rational(&m);
            assert!(o == Ordering::Greater && q >= rational(&x) || o == Ordering::Less);
        }
    }
}

#[test]
fn modes_agree_and_ln_inverts() {
    let mut s = 0xE7_0002u64;
    for i in 0..size(300, 3) {
        let x = float(&mut s, if i % 10 == 9 { 400 } else { 90 }, 6);
        let p = 1 + below(&mut s, if i % 10 == 9 { 500 } else { 120 });
        for f in ["exp", "exp2", "exp10", "expm1"] {
            modes(|r| exps(f, &x, p, r), &format!("{f}({x:?}) at {p}"));
        }
        // `ln(e^x)` within a few units of the last place of `x` at `p + 20` bits.
        let q = x.prec() + 20;
        let e = x.exp_round(q, Round::Nearest).0;
        let back = e.ln_round(q, Round::Nearest).0;
        let d = rational(&back.sub_round(&x, q, Round::Nearest).0);
        let tol = rational(
            &BigFloat::from_parts(
                &Number::from(1),
                x.exponent().unwrap().max(1) - q as i64 + 3,
                2,
                Round::Nearest,
            )
            .0,
        );
        assert!(d.abs() <= tol, "ln(e^{x:?}) = {back:?}");
    }
}

#[test]
fn near_the_rounding_near_the_argument() {
    // `e^x - 1` near `x` and `e^x` near 1 for `e < -prec - 2`: `x²` meets the last place of a
    // long `x` from `e = -prec - 30`.
    let range = |p: u64| (-(p as i64) - 31, -(p as i64) + 1);
    near_threshold(|x, p, r| x.exp_m1_round(p, r), range, true, "expm1");
    near_threshold(|x, p, r| x.exp_round(p, r), range, true, "exp");
    near_threshold(|x, p, r| x.exp2_round(p, r), range, true, "exp2");
    near_threshold(|x, p, r| x.exp10_round(p, r), range, true, "exp10");
}
