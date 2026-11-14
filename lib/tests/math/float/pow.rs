//! `pow` against MPFR's vectors, its special values, its exact results against the exact
//! rationals rounded, and its ranges.

use super::common::{MODES, below, float, mantissa, rational, reference, same, size};
use super::explog_vectors::{check, modes};
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;

#[test]
fn powers_match_mpfr() {
    check(include_str!("data/explog_pow.txt"), |_, x, y, p, r| {
        x.pow_round(y, p, r)
    });
}

fn f(v: f64) -> BigFloat {
    BigFloat::from(v)
}

#[test]
fn special_values() {
    let nan = BigFloat::nan(5);
    let inf = BigFloat::infinity(5);
    let ninf = BigFloat::neg_infinity(5);
    let p = 20;
    let is = |got: (BigFloat, Ordering), want: BigFloat, what: &str| {
        same(&got, &(want.with_prec(p), Ordering::Equal), what);
    };
    for r in MODES {
        is(nan.pow_round(&BigFloat::zero(3), p, r), f(1.0), "nan^0");
        is(f(1.0).pow_round(&nan, p, r), f(1.0), "1^nan");
        is(f(-1.0).pow_round(&inf, p, r), f(1.0), "(-1)^inf");
        is(f(-1.0).pow_round(&ninf, p, r), f(1.0), "(-1)^-inf");
        is(
            BigFloat::neg_zero(3).pow_round(&f(-3.0), p, r),
            ninf.clone(),
            "-0^-3",
        );
        is(
            BigFloat::neg_zero(3).pow_round(&f(-2.0), p, r),
            inf.clone(),
            "-0^-2",
        );
        is(
            BigFloat::neg_zero(3).pow_round(&f(-0.5), p, r),
            inf.clone(),
            "-0^-0.5",
        );
        is(
            BigFloat::neg_zero(3).pow_round(&f(3.0), p, r),
            BigFloat::neg_zero(3),
            "-0^3",
        );
        is(
            BigFloat::neg_zero(3).pow_round(&f(2.5), p, r),
            BigFloat::zero(3),
            "-0^2.5",
        );
        is(f(0.5).pow_round(&inf, p, r), BigFloat::zero(3), "0.5^inf");
        is(f(-0.5).pow_round(&ninf, p, r), inf.clone(), "-0.5^-inf");
        is(f(2.0).pow_round(&ninf, p, r), BigFloat::zero(3), "2^-inf");
        is(
            ninf.pow_round(&f(-3.0), p, r),
            BigFloat::neg_zero(3),
            "-inf^-3",
        );
        is(ninf.pow_round(&f(-2.0), p, r), BigFloat::zero(3), "-inf^-2");
        is(ninf.pow_round(&f(3.0), p, r), ninf.clone(), "-inf^3");
        is(ninf.pow_round(&f(0.5), p, r), inf.clone(), "-inf^0.5");
        is(inf.pow_round(&f(-0.5), p, r), BigFloat::zero(3), "inf^-0.5");
        assert!(f(-2.0).pow_round(&f(0.5), p, r).0.is_nan());
        assert!(f(2.0).pow_round(&nan, p, r).0.is_nan());
    }
}

#[test]
fn exact_powers_round_as_their_rationals() {
    let mut s = 0x90_0001u64;
    for _ in 0..size(400, 8) {
        // `b^n` and `(b^(2^t))^(c / 2^t)` for a short odd `b`.
        let bits = 1 + below(&mut s, 6);
        let mut m = mantissa(&mut s, bits);
        m.set_bit(0, true);
        if below(&mut s, 2) == 0 {
            m = -m;
        }
        let e = below(&mut s, 20) as i64 - 10;
        let b = BigFloat::from_parts(&m, e, 64, Round::Nearest).0;
        let n = below(&mut s, 12) as i64 - 4;
        let p = 1 + below(&mut s, 40);
        let qb = rational(&b);
        let exact = |q: &Rational, k: i64| {
            let mut v = Rational::from(1);
            for _ in 0..k.unsigned_abs() {
                v = &v * q;
            }
            if k < 0 { Rational::from(1) / v } else { v }
        };
        let unit = m.abs() == 1;
        for r in MODES {
            if n != 0 && !unit && n < 0 {
                continue;
            }
            let want = if n == 0 {
                (BigFloat::from(1).with_prec(p), Ordering::Equal)
            } else {
                reference(&exact(&qb, n), p, r)
            };
            if n >= 0 || unit {
                same(
                    &b.pow_round(&BigFloat::from(n), p, r),
                    &want,
                    &format!("{b:?}^{n}"),
                );
            }
            // A root: `|b|^(2^t)` raised to `c / 2^t`.
            let t = 1 + below(&mut s, 3);
            let c = 2 * below(&mut s, 3) as i64 + 1;
            let base = b
                .abs()
                .pow_round(&BigFloat::from(1i64 << t), 400, Round::Nearest)
                .0;
            let y = BigFloat::from_parts(&Number::from(c), -(t as i64), 8, Round::Nearest).0;
            let want = reference(&exact(&qb.abs(), c), p, r);
            same(&base.pow_round(&y, p, r), &want, &format!("{base:?}^{y:?}"));
        }
    }
}

#[test]
fn far_beyond_the_range_at_once() {
    let y = BigFloat::from_parts(&Number::from(3), 1 << 50, 4, Round::Nearest).0;
    for r in MODES {
        let (v, o) = f(1.5).pow_round(&y, 53, r);
        assert!(v.is_infinite() && o == Ordering::Greater || v.is_normal() && o == Ordering::Less);
        let (v, o) = f(0.75).pow_round(&y, 53, r);
        assert!(v.is_zero() && o == Ordering::Less || v.is_normal() && o == Ordering::Greater);
        // An odd integer keeps the sign.
        let (v, _) = f(-2.0).pow_round(&BigFloat::from((1i64 << 62) + 1), 53, r);
        assert!(v.is_sign_negative());
    }
}

#[test]
fn modes_agree() {
    let mut s = 0x90_0002u64;
    for i in 0..size(300, 3) {
        let x = float(&mut s, 90, 8);
        let y = if i % 3 == 0 {
            BigFloat::from(below(&mut s, 30) as i64 - 15)
        } else {
            float(&mut s, 90, 6)
        };
        let p = 1 + below(&mut s, 150);
        modes(|r| x.pow_round(&y, p, r), &format!("{x:?}^{y:?} at {p}"));
    }
}
