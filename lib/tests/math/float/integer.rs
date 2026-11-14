//! Rounding to integers against MPFR's `mpfr_rint`, `mpfr_rint_*`, `mpfr_floor` and the others,
//! `mpfr_frac`, `mpfr_get_z`, `mpfr_get_si` and `mpfr_get_ui`, and against the integers of the
//! exact rationals; the exact conversions into integers.

use super::common::{MODES, below, float, mode, rational, reference, same, size};
use super::vectors::{self, lines, ternary};
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;

#[test]
fn rint_and_frac_match_mpfr() {
    let mut n = 0;
    for v in lines(include_str!("data/convert_rint.txt")) {
        let p: u64 = v[2].parse().unwrap();
        let q: u64 = v[3].parse().unwrap();
        let x = vectors::float(v[1], p);
        let round = vectors::mode(v[4]);
        let want = (vectors::float(v[5], q), ternary(v[6]));
        let what = format!("{} {x:?} to {q} {round:?}", v[0]);
        let got = match v[0] {
            "rint" => x.rint_round(q, round),
            "rint_floor" => x.floor_round(q, round),
            "rint_ceil" => x.ceil_round(q, round),
            "rint_trunc" => x.trunc_round(q, round),
            "rint_round" => x.round_round(q, round),
            "rint_roundeven" => x.round_ties_even_round(q, round),
            "frac" => x.fract_round(q, round),
            "floor" => x.rint_round(q, Round::Down),
            "ceil" => x.rint_round(q, Round::Up),
            "trunc" => x.rint_round(q, Round::TowardZero),
            "roundeven" => x.rint_round(q, Round::Nearest),
            "round" => {
                // MPFR's ties away from zero, at the number's own precision only.
                if q == p {
                    let r = x.round();
                    assert!(
                        r == want.0 && r.is_sign_negative() == want.0.is_sign_negative(),
                        "{what}: {r:?}"
                    );
                }
                continue;
            }
            f => panic!("{f}"),
        };
        same(&got, &want, &what);
        if q == p {
            // The forms at the number's precision are exact.
            let exact = match v[0] {
                "floor" => Some(x.floor()),
                "ceil" => Some(x.ceil()),
                "trunc" => Some(x.trunc()),
                "roundeven" => Some(x.round_ties_even()),
                "frac" => Some(x.fract()),
                _ => None,
            };
            if let Some(r) = exact {
                assert!(
                    r == want.0 && r.is_sign_negative() == want.0.is_sign_negative(),
                    "{what}: {r:?}"
                );
            }
        }
        n += 1;
    }
    assert!(n > size(500, 10));
}

#[test]
fn to_integers_match_mpfr() {
    for v in lines(include_str!("data/convert_getz.txt")) {
        let p: u64 = v[1].parse().unwrap();
        let x = vectors::float(v[0], p);
        let round = vectors::mode(v[2]);
        let z: Number = v[3].parse().unwrap();
        let t = ternary(v[4]);
        assert_eq!(
            x.to_number_round(round),
            Some((z.clone(), t)),
            "{x:?} {round:?}"
        );
        let si: i64 = v[5].parse().unwrap();
        let ui: u64 = v[6].parse().unwrap();
        assert_eq!(x.to_i64_round(round).unwrap().0, si, "{x:?} {round:?}");
        assert_eq!(x.to_u64_round(round).unwrap().0, ui, "{x:?} {round:?}");
        // The direction of a saturated value is from the number.
        let (v64, o) = x.to_i64_round(round).unwrap();
        let want = Rational::from(v64).partial_cmp(&rational(&x)).unwrap();
        assert_eq!(o, want, "{x:?} {round:?}");
        let (v128, o) = x.to_i128_round(round).unwrap();
        assert_eq!(o, Rational::from(v128).partial_cmp(&rational(&x)).unwrap());
        let (u128v, o) = x.to_u128_round(round).unwrap();
        assert_eq!(o, Rational::from(u128v).partial_cmp(&rational(&x)).unwrap());
    }
}

/// The integer that the rational `q` rounds to in the mode `round`.
fn integer(q: &Rational, round: Round) -> Number {
    match round {
        Round::Nearest => {
            // To nearest, ties to even, from the floor and the remainder.
            let f = q.floor();
            let r = q - &Rational::from_integer(f.clone());
            let half = Rational::from(1) / 2u32;
            if r > half || r == half && f.bit(0) {
                f + Number::from(1)
            } else {
                f
            }
        }
        Round::TowardZero => q.trunc(),
        Round::Up => q.ceil(),
        Round::Down => q.floor(),
    }
}

#[test]
fn integers_are_those_of_the_rationals() {
    let mut s = 0x5EED_1A7E_0000_0001u64;
    for _ in 0..size(3000, 40) {
        let x = float(&mut s, 200, 80);
        let q = rational(&x);
        let round = mode(&mut s);
        let (z, o) = x.to_number_round(round).unwrap();
        assert_eq!(z, integer(&q, round), "{x:?} {round:?}");
        assert_eq!(
            o,
            Rational::from_integer(z.clone()).partial_cmp(&q).unwrap(),
            "{x:?}"
        );
        // floor and the others as their rationals, at any precision and mode.
        let prec = 1 + below(&mut s, 120);
        let r2 = mode(&mut s);
        for (got, want) in [
            (x.floor_round(prec, r2), q.floor()),
            (x.ceil_round(prec, r2), q.ceil()),
            (x.trunc_round(prec, r2), q.trunc()),
            (x.round_round(prec, r2), q.round()),
            (
                x.round_ties_even_round(prec, r2),
                integer(&q, Round::Nearest),
            ),
        ] {
            if want.is_zero() {
                assert!(got.0.is_zero() && got.0.is_sign_negative() == x.is_sign_negative());
                assert_eq!(got.1, Ordering::Equal);
            } else {
                same(
                    &got,
                    &reference(&Rational::from_integer(want), prec, r2),
                    &format!("{x:?}"),
                );
            }
        }
        assert_eq!(x.floor(), q.floor());
        assert_eq!(x.ceil(), q.ceil());
        assert_eq!(x.trunc(), q.trunc());
        assert_eq!(x.round(), q.round());
        let f = x.fract();
        assert_eq!(rational(&f), &q - &Rational::from_integer(q.trunc()));
        assert_eq!(x.is_integer(), q.is_integer());
        assert_eq!(Number::try_from(&x).is_ok(), q.is_integer());
        if q.is_integer() {
            assert_eq!(Number::try_from(&x).unwrap(), q.trunc());
        }
        assert_eq!(
            i64::try_from(&x).ok(),
            i64::try_from(&q).ok(),
            "{x:?} into i64"
        );
        assert_eq!(u16::try_from(&x).ok(), u16::try_from(&q).ok());
        assert_eq!(i128::try_from(&x).ok(), i128::try_from(&q).ok());
    }
}

#[test]
fn special_values_and_signs() {
    for round in MODES {
        assert!(BigFloat::nan(4).rint_round(9, round).0.is_nan());
        assert!(BigFloat::nan(4).floor_round(9, round).0.is_nan());
        let inf = BigFloat::neg_infinity(4);
        assert!(inf.floor_round(9, round).0 == inf && inf.rint_round(9, round).0.prec() == 9);
        let f = inf.fract_round(9, round).0;
        assert!(f.is_zero() && f.is_sign_negative());
        assert!(BigFloat::infinity(4).to_number_round(round).is_none());
        assert_eq!(
            BigFloat::infinity(4).to_i64_round(round),
            Some((i64::MAX, Ordering::Less))
        );
        assert_eq!(
            BigFloat::neg_infinity(4).to_u64_round(round),
            Some((0, Ordering::Greater))
        );
        assert_eq!(BigFloat::nan(4).to_u128_round(round), None);
        let z = BigFloat::neg_zero(4).ceil_round(9, round).0;
        assert!(z.is_zero() && z.is_sign_negative());
    }
    // Zeros keep the sign of the number.
    let x = BigFloat::from(-0.25);
    assert!(x.ceil().is_sign_negative() && x.ceil().is_zero());
    assert!(x.trunc().is_sign_negative() && x.round().is_sign_negative());
    assert_eq!(x.floor(), -1);
    assert!(BigFloat::from(0.25).floor().is_sign_positive());
    assert!(BigFloat::from(-3.0).fract().is_sign_negative());
    let (r, o) = x.rint_round(3, Round::TowardZero);
    assert!(r.is_zero() && r.is_sign_negative() && o == Ordering::Greater);
    // Huge numbers are integers already and are not expanded.
    let huge = BigFloat::from_parts(&Number::from(3), 1 << 61, 2, Round::Nearest).0;
    assert!(huge.is_integer() && huge.floor() == huge && huge.rint_round(1, Round::Up).0 > huge);
    assert!(huge.fract().is_zero());
    assert_eq!(
        huge.to_i64_round(Round::Nearest),
        Some((i64::MAX, Ordering::Less))
    );
    assert!(i64::try_from(&huge).is_err() && u8::try_from(&BigFloat::from(-1)).is_err());
    assert!(
        i8::try_from(&BigFloat::nan(3)).is_err()
            && Number::try_from(&BigFloat::infinity(2)).is_err()
    );
    assert_eq!(i128::try_from(&BigFloat::from(i128::MIN)), Ok(i128::MIN));
    assert_eq!(u128::try_from(&BigFloat::from(u128::MAX)), Ok(u128::MAX));
    assert!(i128::try_from(&-BigFloat::from(u128::MAX)).is_err());
    assert_eq!(i8::try_from(BigFloat::neg_zero(5)), Ok(0));
    assert_eq!(
        BigFloat::from(i128::MIN).to_i128_round(Round::Up),
        Some((i128::MIN, Ordering::Equal))
    );
    let below_min = BigFloat::from(i128::MIN).with_prec(200) - BigFloat::from(0.5).with_prec(200);
    assert_eq!(
        below_min.to_i128_round(Round::Down),
        Some((i128::MIN, Ordering::Greater))
    );
    assert_eq!(
        below_min.to_i128_round(Round::Up),
        Some((i128::MIN, Ordering::Greater))
    );
}

/// The forms into an existing number, `set_rint` and its kind, against the vectors of MPFR's
/// `mpfr_rint`, `mpfr_floor`, `mpfr_ceil` and `mpfr_trunc`, into numbers of the result's
/// precision that hold a finite value of their own, a zero or NaN.
#[test]
fn rint_into_numbers_matches_mpfr() {
    let mut n = 0;
    for (i, v) in lines(include_str!("data/convert_rint.txt")).enumerate() {
        let p: u64 = v[2].parse().unwrap();
        let q: u64 = v[3].parse().unwrap();
        let x = vectors::float(v[1], p);
        let round = vectors::mode(v[4]);
        let want = (vectors::float(v[5], q), ternary(v[6]));
        let mut r = match i % 3 {
            0 => {
                BigFloat::from(-1)
                    .div_round(&BigFloat::from(3), q, Round::Nearest)
                    .0
            }
            1 => BigFloat::zero(q),
            _ => BigFloat::nan(q),
        };
        let o = match v[0] {
            "rint" | "roundeven" => r.set_rint(&x, round),
            "floor" => r.set_floor(&x),
            "ceil" => r.set_ceil(&x),
            "trunc" => r.set_trunc(&x),
            _ => continue,
        };
        same(&(r, o), &want, &format!("into: {}", v.join(" ")));
        n += 1;
    }
    assert!(n > size(200, 4));
}

/// `rint_round` and `set_rint` against the integers of the exact rationals: a number of fewer
/// bits before its point than the precision is the integer of its mode, exactly; one of more is
/// rounded to the precision; at precisions below, at and above the number's, into numbers that
/// hold a value of their own.
#[test]
fn rint_rounds_as_the_rationals() {
    let mut s = 0x5EED_1A7E_0000_0007u64;
    for i in 0..size(3000, 30) {
        let x = float(&mut s, 300, 200);
        let q = match below(&mut s, 3) {
            0 => x.prec(),
            1 => 1 + below(&mut s, x.prec()),
            _ => 1 + below(&mut s, 400),
        };
        let round = mode(&mut s);
        let got = x.rint_round(q, round);
        let mut r = if i % 2 == 0 {
            BigFloat::from(7)
                .div_round(&BigFloat::from(3), q, Round::Nearest)
                .0
        } else {
            BigFloat::infinity(q)
        };
        let o = r.set_rint(&x, round);
        same(&(r, o), &got, &format!("set_rint {x:?} to {q} {round:?}"));
        if !x.is_finite() {
            continue;
        }
        let rx = rational(&x);
        let want = match x.exponent() {
            Some(e) if e < q as i64 => {
                let z = match round {
                    Round::Down => rx.floor(),
                    Round::Up => rx.ceil(),
                    Round::TowardZero => rx.trunc(),
                    Round::Nearest => integer(&rx, Round::Nearest),
                };
                let o = Rational::from_integer(z.clone()).partial_cmp(&rx).unwrap();
                let int = if z.is_zero() {
                    if x.is_sign_negative() {
                        BigFloat::neg_zero(q)
                    } else {
                        BigFloat::zero(q)
                    }
                } else {
                    BigFloat::from_number(&z, q, Round::Nearest).0
                };
                (int, o)
            }
            _ => x.prec_round(q, round),
        };
        same(&got, &want, &format!("rint {x:?} to {q} {round:?}"));
    }
}
