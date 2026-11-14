//! The hyperbolic functions and their inverses against MPFR's vectors, their special values and
//! identities.

use super::common::{MODES, below, float, rational, same, size};
use super::explog_vectors::{check, modes, near_threshold};
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;

fn hyper(f: &str, x: &BigFloat, p: u64, r: Round) -> (BigFloat, Ordering) {
    match f {
        "sinh" => x.sinh_round(p, r),
        "cosh" => x.cosh_round(p, r),
        "tanh" => x.tanh_round(p, r),
        "asinh" => x.asinh_round(p, r),
        "acosh" => x.acosh_round(p, r),
        "atanh" => x.atanh_round(p, r),
        _ => panic!("{f}"),
    }
}

#[test]
fn hyperbolic_functions_match_mpfr() {
    check(include_str!("data/explog_hyper.txt"), |f, x, _, p, r| {
        hyper(f, x, p, r)
    });
}

#[test]
fn special_values() {
    for r in MODES {
        let p = 30;
        for f in ["sinh", "tanh", "asinh", "atanh"] {
            same(
                &hyper(f, &BigFloat::neg_zero(4), p, r),
                &(BigFloat::neg_zero(p), Ordering::Equal),
                f,
            );
        }
        same(
            &BigFloat::zero(4).cosh_round(p, r),
            &(BigFloat::from(1).with_prec(p), Ordering::Equal),
            "cosh 0",
        );
        same(
            &BigFloat::from(1).acosh_round(p, r),
            &(BigFloat::zero(p), Ordering::Equal),
            "acosh 1",
        );
        same(
            &BigFloat::neg_infinity(4).tanh_round(p, r),
            &(BigFloat::from(-1).with_prec(p), Ordering::Equal),
            "tanh -inf",
        );
        same(
            &BigFloat::neg_infinity(4).cosh_round(p, r),
            &(BigFloat::infinity(p), Ordering::Equal),
            "cosh -inf",
        );
        same(
            &BigFloat::from(-1).atanh_round(p, r),
            &(BigFloat::neg_infinity(p), Ordering::Equal),
            "atanh -1",
        );
        assert!(BigFloat::from(0.5).acosh_round(p, r).0.is_nan());
        assert!(BigFloat::from(1.5).atanh_round(p, r).0.is_nan());
        assert!(BigFloat::infinity(4).atanh_round(p, r).0.is_nan());
    }
}

#[test]
fn modes_agree_and_identities_hold() {
    let mut s = 0x4_7001u64;
    for i in 0..size(250, 3) {
        let x = float(&mut s, if i % 10 == 9 { 300 } else { 80 }, 6);
        let p = 1 + below(&mut s, if i % 10 == 9 { 400 } else { 120 });
        for f in ["sinh", "cosh", "tanh", "asinh", "acosh", "atanh"] {
            modes(|r| hyper(f, &x, p, r), &format!("{f}({x:?}) at {p}"));
        }
        // `cosh² - sinh² = 1` and `asinh(sinh x) = x`, within a few units at `q` bits.
        let q = 2 * x.prec() + 40;
        let sh = x.sinh_round(q, Round::Nearest).0;
        let ch = x.cosh_round(q, Round::Nearest).0;
        let one = rational(&ch) * rational(&ch) - rational(&sh) * rational(&sh);
        let big = rational(&ch) * rational(&ch);
        let tol = &big
            * rational(&BigFloat::from_parts(&Number::from(1), 4 - q as i64, 2, Round::Nearest).0);
        assert!(
            (one - Rational::from(1)).abs() <= tol,
            "cosh² - sinh² of {x:?}"
        );
        let back = sh.asinh_round(q, Round::Nearest).0;
        let d = rational(&back) - rational(&x);
        let tol = rational(&x).abs()
            * rational(&BigFloat::from_parts(&Number::from(1), 4 - q as i64, 2, Round::Nearest).0);
        assert!(d.abs() <= tol, "asinh(sinh {x:?}) = {back:?}");
    }
}

#[test]
fn near_the_rounding_near_the_argument() {
    // The odd functions round near `x` for `2e < -prec`: the window where `|x|³` and the last
    // place of a long `x` meet, `e` from `-(prec + 30)/2` to `-prec/2 + 1`.
    let range = |p: u64| (-((p as i64) + 30) / 2, -(p as i64) / 2 + 1);
    near_threshold(|x, p, r| x.sinh_round(p, r), range, true, "sinh");
    near_threshold(|x, p, r| x.tanh_round(p, r), range, true, "tanh");
    near_threshold(|x, p, r| x.asinh_round(p, r), range, true, "asinh");
    near_threshold(|x, p, r| x.atanh_round(p, r), range, true, "atanh");
    let range = |p: u64| (-(p as i64) / 2 - 3, -(p as i64) / 2 + 1);
    near_threshold(|x, p, r| x.cosh_round(p, r), range, true, "cosh");
}

/// `sinh`, `cosh` and `tanh` from the exponential in fixed point, at two to four limbs in every
/// mode, against themselves at 700 bits (more limbs than the array kernels take) rounded again.
#[test]
fn fixed_point_paths_agree_with_wide_ones() {
    let mut s = 0x4E_0001u64;
    for i in 0..size(120, 3) {
        let x = float(&mut s, 300, 0);
        let k = BigFloat::from_parts(&Number::from(1), (i % 20) as i64 - 13, 1, Round::Nearest).0;
        let x = x.mul_round(&k, 300, Round::Nearest).0;
        if !x.is_normal() {
            continue;
        }
        let p = [64u64, 100, 128, 150, 192, 240][i % 6];
        for f in ["sinh", "cosh", "tanh"] {
            let wide = hyper(f, &x, 700, Round::Nearest);
            for r in MODES {
                if let Some(want) = super::explog_vectors::reround(&wide, p, r) {
                    same(
                        &hyper(f, &x, p, r),
                        &want,
                        &format!("{f}({x:?}) at {p} {r:?}"),
                    );
                }
            }
        }
    }
}
