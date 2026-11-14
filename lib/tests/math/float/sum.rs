//! Sums and dot products of many terms against the exact rationals rounded by integers: terms
//! near each other (the fixed-point accumulator), terms millions of bits apart (the sorted
//! accumulation that stops at a negligible rest), cancellations, and the special values and
//! zeros of MPFR's `mpfr_sum`.

use super::common::{MODES, below, float, mode, one, rational, reference, same, size};
use a::math::float::{BigFloat, Round};
use a::math::rational::Rational;
use core::cmp::Ordering;

/// Checks a sum against the exact rational, a zero of nonzero terms being `+0`, or `-0` when
/// rounding down.
#[track_caller]
fn check(got: (BigFloat, Ordering), exact: &Rational, prec: u64, round: Round, what: &str) {
    if exact.is_zero() {
        let want = if round == Round::Down {
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
fn sums_of_many_terms_round_once() {
    let mut s = 0x5EED_5000_0000_0001u64;
    for i in 0..size(1500, 15) {
        let n = 1 + below(&mut s, if i % 10 == 0 { 60 } else { 8 }) as usize;
        let exps = match i % 4 {
            0 => 10,
            1 => 200,
            2 => 3000,
            _ => 60,
        };
        let mut terms: Vec<BigFloat> = (0..n).map(|_| float(&mut s, 150, exps)).collect();
        if i % 3 == 0 && n > 1 {
            // A term that cancels another, nearly or exactly.
            let k = below(&mut s, n as u64) as usize;
            let p = 1 + below(&mut s, 160);
            terms[0] = (-&terms[k]).prec_round(p, mode(&mut s)).0;
        }
        if i % 50 == 7 && !cfg!(miri) {
            // Millions of bits apart: the sorted path.
            let e = 1 + below(&mut s, 3_000_000) as i64;
            terms.push(BigFloat::from_parts(&one(), -e, 1, Round::Nearest).0);
        }
        let prec = 1 + below(&mut s, if i % 7 == 0 { 600 } else { 140 });
        let round = mode(&mut s);
        let exact = terms
            .iter()
            .fold(Rational::from(0u8), |acc, x| acc + rational(x));
        check(
            BigFloat::sum_round(&terms, prec, round),
            &exact,
            prec,
            round,
            &format!("sum of {terms:?} at {prec} {round:?}"),
        );
    }
}

#[test]
fn dot_products_round_once() {
    let mut s = 0x5EED_5000_0000_0002u64;
    for i in 0..size(1200, 12) {
        let n = 1 + below(&mut s, if i % 10 == 0 { 40 } else { 6 }) as usize;
        let exps = if i % 3 == 0 { 2000 } else { 40 };
        let xs: Vec<BigFloat> = (0..n).map(|_| float(&mut s, 130, exps)).collect();
        let mut ys: Vec<BigFloat> = (0..n).map(|_| float(&mut s, 130, exps)).collect();
        if i % 4 == 1 && n > 1 {
            // `x0·y0 + x1·y1` with `y1 = -x0·y0/x1`, nearly: cancellation.
            let t = (&xs[0] * &ys[0]).prec_round(200, Round::Nearest).0;
            ys[1] = -(&t / &xs[1]);
        }
        let prec = 1 + below(&mut s, 300);
        let round = mode(&mut s);
        let exact = xs.iter().zip(&ys).fold(Rational::from(0u8), |acc, (x, y)| {
            acc + rational(x) * rational(y)
        });
        check(
            BigFloat::dot_round(&xs, &ys, prec, round),
            &exact,
            prec,
            round,
            &format!("dot of {xs:?} and {ys:?} at {prec} {round:?}"),
        );
    }
}

#[test]
fn a_negligible_rest_decides_by_its_sign() {
    // `x + y + t` with `t` tiny and `y` cancelling part of `t`'s effect or not: the rest after
    // `x` is far below it, its sign alone decides, and may be zero.
    let tiny = |e: i64| BigFloat::from_parts(&one(), -e, 1, Round::Nearest).0;
    // Past the span of the fixed-point accumulation, `2^16` bits and more; under Miri just past.
    let (far, farther) = if cfg!(miri) {
        (70_000, 140_000)
    } else {
        (1_000_000, 2_000_000)
    };
    let precs: &[u64] = if cfg!(miri) {
        &[53]
    } else {
        &[1, 2, 24, 53, 64, 113, 200]
    };
    let xs = if cfg!(miri) {
        vec![BigFloat::from(-3)]
    } else {
        vec![BigFloat::from(1), BigFloat::from(-3), BigFloat::from(0.75)]
    };
    let modes: &[Round] = if cfg!(miri) {
        &[Round::Nearest, Round::Down]
    } else {
        &MODES
    };
    for &prec in precs {
        for &round in modes {
            for x in &xs {
                for rest in [
                    vec![tiny(far)],
                    vec![-tiny(far)],
                    vec![tiny(far), -tiny(far)],
                    vec![tiny(far - 9), -tiny(far - 9), tiny(farther)],
                    vec![
                        tiny(far - 9),
                        -tiny(far - 9)
                            .add_round(&tiny(farther), farther as u64, Round::Nearest)
                            .0,
                    ],
                    vec![tiny(far - 7), tiny(far - 7), -tiny(far - 8)],
                ] {
                    let mut terms = vec![x.clone()];
                    terms.extend(rest.iter().cloned());
                    let exact = terms
                        .iter()
                        .fold(Rational::from(0u8), |acc, t| acc + rational(t));
                    check(
                        BigFloat::sum_round(&terms, prec, round),
                        &exact,
                        prec,
                        round,
                        &format!("{terms:?} at {prec} {round:?}"),
                    );
                }
            }
        }
    }
}

#[test]
fn special_values_and_zeros_are_mpfrs() {
    let (inf, ninf, nan) = (
        BigFloat::infinity(53),
        BigFloat::neg_infinity(53),
        BigFloat::nan(53),
    );
    let (z, nz, one) = (
        BigFloat::zero(53),
        BigFloat::neg_zero(53),
        BigFloat::from(1),
    );
    for round in MODES {
        let s = |t: &[BigFloat]| BigFloat::sum_round(t, 10, round).0;
        assert!(s(&[one.clone(), nan.clone()]).is_nan());
        assert!(s(&[inf.clone(), ninf.clone()]).is_nan());
        assert_eq!(s(&[inf.clone(), one.clone(), inf.clone()]), inf);
        assert_eq!(s(&[ninf.clone(), one.clone()]), ninf);
        // No terms: +0; zeros of one sign: that sign; of both, or cancelling: -0 only down.
        assert!(s(&[]).is_zero() && s(&[]).is_sign_positive());
        assert!(s(&[nz.clone(), nz.clone()]).is_sign_negative());
        assert!(s(&[z.clone(), z.clone()]).is_sign_positive());
        let down = round == Round::Down;
        assert_eq!(s(&[z.clone(), nz.clone()]).is_sign_negative(), down);
        assert_eq!(
            s(&[one.clone(), -&one, nz.clone()]).is_sign_negative(),
            down
        );
        let d = |x: &[BigFloat], y: &[BigFloat]| BigFloat::dot_round(x, y, 10, round).0;
        let r = core::slice::from_ref;
        assert!(d(r(&inf), r(&z)).is_nan());
        assert!(d(&[inf.clone(), one.clone()], &[nz.clone(), one.clone()]).is_nan());
        assert_eq!(d(r(&ninf), r(&ninf)), inf);
        assert!(d(r(&nz), r(&one)).is_sign_negative());
        assert_eq!(
            d(&[nz.clone(), z.clone()], &[one.clone(), one.clone()]).is_sign_negative(),
            down
        );
    }
}

#[test]
fn rests_just_above_negligible_are_added() {
    // `1 + n·2^-k` for `k` just past the precision (with a term too far to matter but past the
    // fixed span, for the sorted path): the small terms together reach the last place, and are
    // not a sticky bit.
    let far = BigFloat::from_parts(&one(), -100_000, 1, Round::Nearest).0;
    let precs: &[u64] = if cfg!(miri) { &[100] } else { &[100, 128, 200] };
    let modes: &[Round] = if cfg!(miri) {
        &[Round::Nearest, Round::Down]
    } else {
        &MODES
    };
    let counts: &[usize] = if cfg!(miri) {
        &[1, 4, 8]
    } else {
        &[1, 2, 3, 4, 5, 6, 7, 8]
    };
    for &prec in precs {
        for &round in modes {
            for k in prec + 1..=prec + if cfg!(miri) { 2 } else { 3 } {
                let small = BigFloat::from_parts(&one(), -(k as i64), 1, Round::Nearest).0;
                for &count in counts {
                    let mut terms = vec![BigFloat::from(1)];
                    terms.extend((0..count).map(|_| small.clone()));
                    terms.push(far.clone());
                    let exact = terms
                        .iter()
                        .fold(Rational::from(0u8), |acc, t| acc + rational(t));
                    check(
                        BigFloat::sum_round(&terms, prec, round),
                        &exact,
                        prec,
                        round,
                        &format!("1 + {count}·2^-{k} + 2^-100000 at {prec} {round:?}"),
                    );
                }
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn square_roots_of_sums_with_a_sticky_bit_round_correctly() {
    // `|a + bi|` with `b` tens of thousands of bits below `a`: the sum of the squares is the
    // larger with a sticky bit, and its root rounds as the exact one.
    use a::math::float::BigComplex;
    let mut s = 0x5EED_5000_0000_0003u64;
    for _ in 0..200 {
        let a = float(&mut s, 200, 10);
        let b = float(&mut s, 200, 10);
        let gap = 40_000 + below(&mut s, 1000) as i64;
        let (m, e) = b.mantissa_exponent().unwrap();
        let b = BigFloat::from_parts(&m, e - gap, b.prec(), Round::Nearest).0;
        let z = BigComplex::new(a.clone(), b.clone());
        let prec = 1 + below(&mut s, 250);
        let round = mode(&mut s);
        let n = rational(&a) * rational(&a) + rational(&b) * rational(&b);
        same(
            &z.abs_round(prec, round),
            &super::complex::sqrt_reference(&n, prec, round),
            &format!("|{z:?}| at {prec} {round:?}"),
        );
    }
}
