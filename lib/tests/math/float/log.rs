//! `ln`, `log2`, `log10` and `ln_1p` against MPFR's vectors, and their exact results and
//! identities.

use super::common::{MODES, below, float, rational, reference, same, size};
use super::explog_vectors::{check, modes, near_threshold};
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;

fn logs(f: &str, x: &BigFloat, p: u64, r: Round) -> (BigFloat, Ordering) {
    match f {
        "ln" => x.ln_round(p, r),
        "log2" => x.log2_round(p, r),
        "log10" => x.log10_round(p, r),
        "log1p" => x.ln_1p_round(p, r),
        _ => panic!("{f}"),
    }
}

#[test]
fn logarithms_match_mpfr() {
    check(include_str!("data/explog_log.txt"), |f, x, _, p, r| {
        logs(f, x, p, r)
    });
}

#[test]
fn special_values() {
    for r in MODES {
        for p in [1u64, 53] {
            for f in ["ln", "log2", "log10"] {
                same(
                    &logs(f, &BigFloat::from(1), p, r),
                    &(BigFloat::zero(p), Ordering::Equal),
                    f,
                );
                same(
                    &logs(f, &BigFloat::neg_zero(5), p, r),
                    &(BigFloat::neg_infinity(p), Ordering::Equal),
                    f,
                );
                same(
                    &logs(f, &BigFloat::infinity(5), p, r),
                    &(BigFloat::infinity(p), Ordering::Equal),
                    f,
                );
                assert!(logs(f, &BigFloat::from(-2), p, r).0.is_nan());
            }
            same(
                &BigFloat::from(-1).ln_1p_round(p, r),
                &(BigFloat::neg_infinity(p), Ordering::Equal),
                "log1p(-1)",
            );
            same(
                &BigFloat::neg_zero(3).ln_1p_round(p, r),
                &(BigFloat::neg_zero(p), Ordering::Equal),
                "log1p(-0)",
            );
            assert!(BigFloat::from(-1.5).ln_1p_round(p, r).0.is_nan());
        }
    }
}

#[test]
fn powers_of_two_and_ten_are_exact() {
    for k in (-300i64..=300).step_by(size(1, 75)) {
        let x = BigFloat::from_parts(&Number::from(1), k, 1, Round::Nearest).0;
        for r in MODES {
            for p in [2u64, 5, 64] {
                same(
                    &x.log2_round(p, r),
                    &BigFloat::from_parts(&Number::from(k), 0, p, r),
                    &format!("log2 2^{k}"),
                );
            }
        }
    }
    for k in (0..=60u32).step_by(size(1, 15)) {
        let x = BigFloat::from(&Number::from(10).pow(k));
        for r in MODES {
            for p in [3u64, 53] {
                let want = if k == 0 {
                    (BigFloat::zero(p), Ordering::Equal)
                } else {
                    reference(&Rational::from(k), p, r)
                };
                same(&x.log10_round(p, r), &want, &format!("log10 10^{k}"));
            }
        }
    }
}

#[test]
fn near_one() {
    let mut s = 0x10_6001u64;
    for _ in 0..size(200, 4) {
        let p = 1 + below(&mut s, 200);
        let d = float(&mut s, 60, 4);
        let shift = BigFloat::from_parts(
            &Number::from(1),
            -(below(&mut s, 2 * p + 40) as i64) - 3,
            2,
            Round::Nearest,
        )
        .0;
        let d = d.mul_round(&shift, 200, Round::Nearest).0;
        let x = BigFloat::from(1).add_round(&d, 600, Round::Nearest).0;
        for f in ["ln", "log2", "log10"] {
            modes(|r| logs(f, &x, p, r), &format!("{f}({x:?}) at {p}"));
        }
        modes(|r| d.ln_1p_round(p, r), &format!("log1p({d:?}) at {p}"));
        // `ln(1 + d) < d`, so rounded up it is at most `d` rounded up.
        let (l, _) = d.ln_1p_round(p, Round::Up);
        assert!(rational(&l) <= rational(&d.prec_round(p, Round::Up).0));
    }
}

#[test]
fn modes_agree_and_exp_inverts() {
    let mut s = 0x10_6002u64;
    for i in 0..size(300, 3) {
        let x = float(&mut s, if i % 10 == 9 { 400 } else { 90 }, 200).abs();
        let p = 1 + below(&mut s, if i % 10 == 9 { 500 } else { 120 });
        for f in ["ln", "log2", "log10", "log1p"] {
            modes(|r| logs(f, &x, p, r), &format!("{f}({x:?}) at {p}"));
        }
        // `e^(ln x)` within a few units of the last place of `x` at `p + 20` bits.
        let q = x.prec() + 20;
        let l = x.ln_round(q + 10, Round::Nearest).0;
        let back = l.exp_round(q, Round::Nearest).0;
        let rel = rational(&back.div_round(&x, q, Round::Nearest).0) - Rational::from(1);
        let tol =
            rational(&BigFloat::from_parts(&Number::from(1), 14 - q as i64, 2, Round::Nearest).0);
        assert!(rel.abs() <= tol, "e^ln({x:?}) = {back:?}");
    }
}

#[test]
fn near_the_rounding_near_the_argument() {
    // `ln(1 + x)` near `x` for `e < -prec - 2`, and `ln` near `x - 1`.
    let range = |p: u64| (-(p as i64) - 31, -(p as i64) + 1);
    near_threshold(|x, p, r| x.ln_1p_round(p, r), range, true, "log1p");
    near_threshold(
        |x, p, r| {
            BigFloat::from(1)
                .add_round(x, 400, Round::Nearest)
                .0
                .ln_round(p, r)
        },
        range,
        true,
        "ln(1 + x)",
    );
}
