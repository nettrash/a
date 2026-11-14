//! Floats to and from `f64` and `f32` in every mode, against MPFR's `mpfr_get_d`, `mpfr_get_flt`
//! and `mpfr_set_d` and against the rounding of the exact rational; and the exact comparisons
//! with the primitives, `Number`s and `Rational`s.

use super::common::{MODES, below, float, mode, rational, reference, same, size, xorshift};
use super::vectors::{self, lines, ternary};
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::rational::Rational;
use core::cmp::Ordering;

#[test]
fn to_f64_and_f32_match_mpfr() {
    let text = include_str!("data/convert_getd.txt");
    let mut n = 0;
    for v in lines(text) {
        let prec: u64 = v[2].parse().unwrap();
        let x = vectors::float(v[1], prec);
        let round = vectors::mode(v[3]);
        let want = ternary(v[5]);
        if v[0] == "d" {
            let bits = u64::from_str_radix(v[4], 16).unwrap();
            let (d, o) = x.to_f64_round(round);
            assert_eq!((d.to_bits(), o), (bits, want), "{x:?} {round:?}");
        } else {
            let bits = u32::from_str_radix(v[4], 16).unwrap();
            let (f, o) = x.to_f32_round(round);
            assert_eq!((f.to_bits(), o), (bits, want), "{x:?} {round:?}");
        }
        n += 1;
    }
    assert!(n > size(1000, 20));
}

#[test]
fn from_f64_matches_mpfr() {
    for v in lines(include_str!("data/convert_setd.txt")) {
        let bits = u64::from_str_radix(v[0], 16).unwrap();
        let prec: u64 = v[1].parse().unwrap();
        let round = vectors::mode(v[2]);
        let want = (vectors::float(v[3], prec), ternary(v[4]));
        let got = BigFloat::from_f64_round(f64::from_bits(bits), prec, round);
        same(&got, &want, &format!("{bits:x}"));
        if prec >= 24 && f64::from_bits(bits) as f32 as f64 == f64::from_bits(bits) {
            let x = f64::from_bits(bits) as f32;
            same(
                &BigFloat::from_f32_round(x, prec, round),
                &want,
                &format!("{x:e}"),
            );
        }
    }
}

/// The `f64` of a rounding to 53 bits of a float in the normal range, by the reference.
fn f64_reference(x: &BigFloat, round: Round) -> (f64, Ordering) {
    let (r, o) = reference(&rational(x), 53, round);
    let q = rational(&r);
    // The float of 53 bits is an `f64` exactly.
    let d = q.to_f64();
    assert_eq!(Rational::try_from(d).unwrap(), q);
    (d, o)
}

#[test]
fn to_f64_rounds_as_the_reference() {
    let mut s = 0x5EED_C0F0_0000_0001u64;
    for _ in 0..size(3000, 40) {
        let x = float(&mut s, 400, 1000);
        let round = mode(&mut s);
        let got = x.to_f64_round(round);
        assert_eq!(got, f64_reference(&x, round), "{x:?} {round:?}");
        let back = BigFloat::from(got.0);
        assert_eq!(
            back.partial_cmp(&x).unwrap(),
            got.1,
            "the direction of {x:?} {round:?}"
        );
    }
}

#[test]
fn special_values_and_boundaries() {
    for round in MODES {
        assert!(BigFloat::nan(9).to_f64_round(round).0.is_nan());
        assert_eq!(
            BigFloat::neg_infinity(9).to_f64_round(round),
            (f64::NEG_INFINITY, Ordering::Equal)
        );
        let z = BigFloat::neg_zero(9).to_f32_round(round);
        assert!(z.0 == 0.0 && z.0.is_sign_negative() && z.1 == Ordering::Equal);
        // 2^-1075 exactly: half the least subnormal, a tie to zero.
        let half = BigFloat::from_parts(&Number::from(1), -1075, 3, Round::Nearest).0;
        let (d, o) = half.to_f64_round(round);
        let want = match round {
            Round::Up => (f64::from_bits(1), Ordering::Greater),
            _ => (0.0, Ordering::Less),
        };
        assert_eq!((d, o), want, "{round:?}");
        // Above it, the least subnormal to nearest.
        let more = BigFloat::from_parts(&Number::from(3), -1076, 3, Round::Nearest).0;
        if round == Round::Nearest {
            assert_eq!(
                more.to_f64_round(round),
                (f64::from_bits(1), Ordering::Greater)
            );
        }
        // The largest finite f64 plus half its last place: a tie, to the infinity (the even).
        let max = BigFloat::from(f64::MAX);
        let ulp = BigFloat::from_parts(&Number::from(1), 970, 2, Round::Nearest).0;
        let tie = max.add_round(&ulp, 60, Round::Nearest).0;
        let (d, o) = tie.to_f64_round(round);
        let want = match round {
            Round::Nearest | Round::Up => (f64::INFINITY, Ordering::Greater),
            _ => (f64::MAX, Ordering::Less),
        };
        assert_eq!((d, o), want, "{round:?}");
        let (d, _) = (-&tie).to_f64_round(round);
        let want = match round {
            Round::Nearest | Round::Down => f64::NEG_INFINITY,
            _ => -f64::MAX,
        };
        assert_eq!(d, want, "{round:?}");
    }
    // The extremes of the exponent range.
    let big = BigFloat::from(1).with_prec(5);
    let big = big
        .mul_round(&BigFloat::from(2.0f64.powi(1000)), 5, Round::Nearest)
        .0;
    assert_eq!(big.to_f32_round(Round::Down), (f32::MAX, Ordering::Less));
    let tiny = BigFloat::from(f64::MIN_POSITIVE)
        .mul_round(&BigFloat::from(f64::MIN_POSITIVE), 8, Round::Nearest)
        .0;
    assert_eq!(tiny.to_f64_round(Round::Nearest), (0.0, Ordering::Less));
    assert_eq!(
        (-&tiny).to_f64_round(Round::Down),
        (-f64::from_bits(1), Ordering::Less)
    );
}

#[test]
fn f64_round_trips_and_conversions_agree() {
    let mut s = 0x5EED_C0F0_0000_0002u64;
    for _ in 0..size(5000, 50) {
        let d = f64::from_bits(xorshift(&mut s));
        if d.is_nan() {
            continue;
        }
        let x = BigFloat::from(d);
        for round in MODES {
            assert_eq!(x.to_f64_round(round), (d, Ordering::Equal));
        }
        let f = d as f32;
        assert_eq!(x.to_f32(), f, "{d:e}");
        let (_, o) = x.to_f32_round(Round::Nearest);
        assert_eq!(
            o,
            BigFloat::from(f).partial_cmp(&x).unwrap(),
            "the direction of {d:e}"
        );
        let prec = 1 + below(&mut s, 70);
        let round = mode(&mut s);
        let got = BigFloat::from_f64_round(d, prec, round);
        if d != 0.0 && d.is_finite() {
            same(
                &got,
                &reference(&rational(&x), prec, round),
                &format!("{d:e}"),
            );
        }
    }
}

#[test]
// The comparisons with borrowed numbers and rationals are what is tested.
#[allow(clippy::op_ref)]
fn comparisons_are_exact() {
    let mut s = 0x5EED_C0F0_0000_0003u64;
    for _ in 0..size(3000, 40) {
        let x = float(&mut s, 150, 140);
        let qx = rational(&x);
        // Against an i64, an u128, an f64, a Number and a Rational, near x and far.
        let i = (qx.to_f64() as i64).saturating_add(below(&mut s, 3) as i64 - 1);
        assert_eq!(
            x.partial_cmp(&i),
            qx.partial_cmp(&Rational::from(i)),
            "{x:?} {i}"
        );
        assert_eq!(x == i, qx == i);
        let u = qx.to_f64().abs() as u128;
        assert_eq!(
            x.partial_cmp(&u),
            qx.partial_cmp(&Rational::from(u)),
            "{x:?} {u}"
        );
        let d = qx.to_f64();
        let d = if below(&mut s, 2) == 0 {
            d
        } else {
            f64::from_bits(d.to_bits().wrapping_add(1))
        };
        if d.is_finite() {
            assert_eq!(
                x.partial_cmp(&d),
                qx.partial_cmp(&Rational::try_from(d).unwrap()),
                "{x:?} {d:e}"
            );
        }
        let n: Number = qx.round();
        assert_eq!(
            x.partial_cmp(&n),
            qx.partial_cmp(&Rational::from_integer(n.clone())),
            "{x:?} {n}"
        );
        assert_eq!(
            n.partial_cmp(&x),
            Rational::from_integer(n.clone()).partial_cmp(&qx)
        );
        let y = float(&mut s, 150, 140);
        let r = &qx + &(rational(&y) / Rational::from(1u64 << 40));
        let r = if below(&mut s, 3) == 0 { qx.clone() } else { r };
        assert_eq!(x.partial_cmp(&r), qx.partial_cmp(&r), "{x:?} {r}");
        assert_eq!(r.partial_cmp(&x), r.partial_cmp(&qx));
        assert_eq!(x == r, qx == r);
    }
    let n = Number::from(-3);
    let r = Rational::from(-5) / 2u32;
    assert!(BigFloat::from(-2.5) > &n && BigFloat::from(-2.5) == &r && BigFloat::from(-2.0) > &r);
    assert!(&BigFloat::from(-2.5) > &n && n < BigFloat::from(-2.5));
    let x = BigFloat::from(-2.5);
    assert!(x < -2 && x > -3i8 && x == -2.5 && x == -2.5f32 && x < 0u8);
    assert!(BigFloat::neg_zero(3) == 0 && BigFloat::zero(3) == -0.0);
    assert!(BigFloat::infinity(3) > u128::MAX && BigFloat::neg_infinity(3) < i128::MIN);
    assert!(BigFloat::infinity(3) == f64::INFINITY && BigFloat::infinity(3) > f64::MAX);
    let nan = BigFloat::nan(3);
    assert!(nan.partial_cmp(&0).is_none() && nan != 0 && nan.partial_cmp(&1.5).is_none());
    assert!(BigFloat::from(1.5).partial_cmp(&f64::NAN).is_none());
    assert!(nan.partial_cmp(&Rational::from(0)).is_none());
    assert!(BigFloat::infinity(3) > Rational::from(10).pow(1000));
    let big = BigFloat::from_parts(&Number::from(1), 1 << 50, 2, Round::Nearest).0;
    assert!(big > Rational::from(10).pow(1000) && big > Number::from(1) << 1000usize);
    assert_eq!(
        BigFloat::from(0.5).partial_cmp(&(Rational::from(1) / 2u32)),
        Some(Ordering::Equal)
    );
}

#[test]
fn comparisons_with_small_rationals() {
    // Every float of a mantissa below 16 and an exponent of -5 to 5 against every fraction of
    // parts below 16: the bounds on bit lengths decide most, near their edges.
    for m in 1..16i32 {
        for e in -5..=5i64 {
            let x = BigFloat::from_parts(&Number::from(m), e, 4, Round::Nearest).0;
            let qx = rational(&x);
            for n in -15..16i32 {
                for d in 1..16u32 {
                    let r = Rational::from(n) / d;
                    assert_eq!(x.partial_cmp(&r), qx.partial_cmp(&r), "{x:?} {r}");
                }
            }
        }
    }
    assert!(BigFloat::from(1.5) > Rational::from(4) / 3u32);
}
