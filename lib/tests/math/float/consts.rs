//! The constants against their decimal digits: the interval of the digits rounded at both ends
//! gives the result wherever the two ends agree.

use super::common::{MODES, reference};
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;

/// π to 100 decimals.
const PI: &str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679";
/// `ln 2` to 60 decimals.
const LN2: &str = "0.693147180559945309417232121458176568075500134360255254120680";

/// The rationals `d - 10^-k` and `d + 10^-k` about the decimals `d` of `k` places.
fn interval(digits: &str) -> (Rational, Rational) {
    let k = digits.len() - digits.find('.').unwrap() - 1;
    let d: Rational = digits.parse().unwrap();
    let u = Rational::new(Number::from(1), Number::from(10).pow(k as u32));
    (&d - &u, &d + &u)
}

/// Checks `f(prec, round)` against the digits wherever their interval decides the rounding.
fn check(digits: &str, max: u64, f: impl Fn(u64, Round) -> (BigFloat, Ordering)) {
    let (lo, hi) = interval(digits);
    let mut decided = 0;
    for prec in 1..=max {
        for round in MODES {
            let (a, b) = (reference(&lo, prec, round), reference(&hi, prec, round));
            if a.0 == b.0 {
                let got = f(prec, round);
                assert_eq!(got.0, a.0, "{prec} bits {round:?}");
                assert_eq!(got.0.prec(), prec);
                // The constant is irrational: the direction is that of both ends' result.
                let want = if a.1 == Ordering::Greater || b.1 == Ordering::Greater {
                    Ordering::Greater
                } else {
                    Ordering::Less
                };
                assert_eq!(got.1, want, "{prec} bits {round:?}");
                decided += 1;
            }
        }
    }
    assert!(decided > 4 * max as usize * 9 / 10);
}

#[test]
fn pi_is_correctly_rounded() {
    check(PI, 320, BigFloat::pi_round);
    assert_eq!(BigFloat::pi(53), BigFloat::from(core::f64::consts::PI));
    assert_eq!(BigFloat::pi(24), BigFloat::from(core::f32::consts::PI));
}

#[test]
fn ln2_is_correctly_rounded() {
    check(LN2, 190, BigFloat::ln2_round);
    assert_eq!(BigFloat::ln2(53), BigFloat::from(core::f64::consts::LN_2));
}

#[test]
fn pi_agrees_with_itself_at_large_precisions() {
    // Rounded down at 20,000 bits and again at 19,000: Ziv's rounding of a long value, the
    // series beyond the table.
    let a = BigFloat::pi_round(20_000, Round::Down).0;
    let b = BigFloat::pi_round(19_000, Round::Down).0;
    assert_eq!(a.prec_round(19_000, Round::Down).0, b);
}
