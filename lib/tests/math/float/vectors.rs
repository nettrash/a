//! Reading the vectors that MPFR 4.2 wrote for the conversions (the generator is in the work
//! directory of issue #17's convert area): floats as `<hex mantissa>p<exponent>`, modes as
//! `N`, `Z`, `U`, `D`, directions as `-1`, `0`, `1`.

use super::common::size;
use a::math::float::{BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;

/// The mode of a letter.
pub(super) fn mode(s: &str) -> Round {
    match s {
        "N" => Round::Nearest,
        "Z" => Round::TowardZero,
        "U" => Round::Up,
        "D" => Round::Down,
        _ => panic!("mode {s}"),
    }
}

/// The direction of a sign.
pub(super) fn ternary(s: &str) -> Ordering {
    match s {
        "-1" => Ordering::Less,
        "0" => Ordering::Equal,
        "1" => Ordering::Greater,
        _ => panic!("ternary {s}"),
    }
}

/// The float of the text at the precision `prec`, exactly.
pub(super) fn float(s: &str, prec: u64) -> BigFloat {
    match s {
        "nan" => BigFloat::nan(prec),
        "inf" => BigFloat::infinity(prec),
        "-inf" => BigFloat::neg_infinity(prec),
        "0" => BigFloat::zero(prec),
        "-0" => BigFloat::neg_zero(prec),
        _ => {
            let (m, e) = s.split_once('p').expect("<mantissa>p<exponent>");
            let m = Number::from_str_radix(m, 16, ByteOrder::LittleEndian).expect("hexadecimal");
            let e: i64 = e.parse().expect("an exponent");
            let (x, o) = BigFloat::from_parts(&m, e, prec, Round::Nearest);
            assert_eq!(o, Ordering::Equal, "{s} at {prec} bits");
            x
        }
    }
}

/// The lines of a file of vectors, split at the spaces: one in 40 of them under Miri.
pub(super) fn lines(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines()
        .filter(|l| !l.is_empty())
        .step_by(size(1, 40))
        .map(|l| l.split(' ').collect())
}
