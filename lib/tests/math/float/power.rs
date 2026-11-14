//! Roots and powers: against the exact results rounded by integers where they are rational, and
//! otherwise by the bracket a correct rounding makes (the result on one side of the exact value
//! by its ternary, the neighbour on the other side, and to nearest the midpoint beyond), in every
//! mode; the special values of MPFR; and MPFR's own results on a file of vectors.

use super::common::{
    MODES, below, float, mantissa, mode, one, rational, reference, same, size, xorshift,
};
use a::math::float::{BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;
use core::cmp::Ordering;

/// `2^k` as a rational.
fn two_to(k: i64) -> Rational {
    let p = Number::new_from_u8(1, ByteOrder::LittleEndian) << k.unsigned_abs() as usize;
    if k >= 0 {
        Rational::from_integer(p)
    } else {
        Rational::new(Number::new_from_u8(1, ByteOrder::LittleEndian), p)
    }
}

/// `q^n` exactly, `n ≥ 0`.
pub(super) fn pow(q: &Rational, n: u64) -> Rational {
    let mut r = Rational::from(1);
    for i in (0..64 - n.leading_zeros()).rev() {
        r = &r * &r;
        if n >> i & 1 == 1 {
            r = &r * q;
        }
    }
    r
}

/// The last place of the nonzero finite `x` at the precision `prec`.
fn ulp(x: &BigFloat, prec: u64) -> Rational {
    two_to(x.exponent().unwrap() - prec as i64)
}

/// Checks that `got`, of the precision `prec`, is the correct rounding in the mode `round` of the
/// value `v` that `side(r)` orders against: `side(r)` is the order of the candidate `r` against `v`
/// (monotone in `r`), so that the result lies on the side its ternary says, the neighbour beyond
/// on the other side, and to nearest the midpoint between them beyond the result's side.
#[track_caller]
pub(super) fn bracket(
    got: &(BigFloat, Ordering),
    prec: u64,
    round: Round,
    what: &str,
    side: impl Fn(&Rational) -> Ordering,
) {
    let (x, o) = got;
    assert_eq!(x.prec(), prec, "{what}: precision");
    let r = rational(x);
    assert_eq!(side(&r), *o, "{what}: {x:?} {o:?}");
    if *o == Ordering::Equal {
        return;
    }
    // The neighbour toward the value, at the result's exponent (or one below it, for a power of
    // two stepping down).
    let mut u = ulp(x, prec);
    let inward = (*o == Ordering::Greater) != r.is_negative();
    if inward && r.abs() == two_to(x.exponent().unwrap() - 1) {
        u = &u / Rational::from(2u32);
    }
    let toward = if *o == Ordering::Less {
        &r + &u
    } else {
        &r - &u
    };
    assert_eq!(
        side(&toward),
        o.reverse(),
        "{what}: {x:?} {o:?} is not next to the value"
    );
    match round {
        Round::Nearest => {
            let mid = (&r + &toward) / Rational::from(2u32);
            let s = side(&mid);
            assert!(
                s == o.reverse()
                    || s == Ordering::Equal && {
                        // A tie goes to the even one: not possible for the irrational values here, and
                        // the rational ones are checked against the reference.
                        false
                    },
                "{what}: {x:?} {o:?} is not the nearest"
            );
        }
        Round::TowardZero => assert!(
            (r.is_negative() && *o == Ordering::Greater)
                || (!r.is_negative() && *o == Ordering::Less),
            "{what}: not toward zero"
        ),
        Round::Up => assert_eq!(*o, Ordering::Greater, "{what}: not up"),
        Round::Down => assert_eq!(*o, Ordering::Less, "{what}: not down"),
    }
}

/// A random precision with the boundaries of one and two limbs.
fn prec(s: &mut u64) -> u64 {
    const P: [u64; 16] = [
        1, 2, 3, 24, 52, 53, 63, 64, 65, 113, 127, 128, 129, 200, 256, 640,
    ];
    if below(s, 3) == 0 {
        1 + below(s, 700)
    } else {
        P[below(s, P.len() as u64) as usize]
    }
}

#[test]
fn integer_powers_round_correctly() {
    let mut s = 0x9017_0001u64;
    for i in 0..size(1500, 20) {
        let x = float(&mut s, if i % 5 == 0 { 300 } else { 70 }, 40);
        let n = below(&mut s, 25) as i64 - 12;
        let p = prec(&mut s);
        let rx = rational(&x);
        for round in MODES {
            let got = x.powi_round(n, p, round);
            let want = if n >= 0 {
                pow(&rx, n as u64)
            } else {
                Rational::from(1) / pow(&rx, n.unsigned_abs())
            };
            same(
                &got,
                &reference(&want, p, round),
                &format!("{x:?}^{n} at {p}"),
            );
            let nn = Number::from(n);
            same(
                &x.pow_number_round(&nn, p, round),
                &got,
                &format!("{x:?}^{n} as a number"),
            );
        }
    }
}

#[test]
fn powers_of_short_odd_parts_are_exact_or_halfway() {
    // `m^n` for a short odd `m`, at precisions around its bits: exact, halfway, or neither.
    let mut s = 0x9017_0002u64;
    for _ in 0..size(800, 10) {
        let m = 2 * below(&mut s, 60) + 3;
        let n = 1 + below(&mut s, 30);
        let x = BigFloat::from(m as i64)
            * BigFloat::from_parts(&one(), below(&mut s, 40) as i64 - 20, 1, Round::Nearest).0;
        let x = if xorshift(&mut s) & 1 == 1 { -x } else { x };
        let want = pow(&rational(&x), n);
        let bits = want.numer().bits();
        let p = (bits as i64 + below(&mut s, 5) as i64 - 3).max(1) as u64;
        let round = mode(&mut s);
        same(
            &x.powi_round(n as i64, p, round),
            &reference(&want, p, round),
            &format!("{x:?}^{n} at {p}"),
        );
    }
}

#[test]
fn large_powers_overflow_and_underflow_by_the_mode() {
    let two = BigFloat::from(2);
    let half = BigFloat::from(0.5);
    let three = BigFloat::from(3);
    for round in MODES {
        let max = BigFloat::EXP_MAX;
        // `2^(EXP_MAX - 1)` is the largest power of two; one more overflows.
        let (x, o) = two.powi_round(max - 1, 10, round);
        assert_eq!((x.exponent(), o), (Some(max), Ordering::Equal));
        let (x, _) = two.powi_round(max, 10, round);
        let inf = matches!(round, Round::Nearest | Round::Up);
        assert_eq!(x.is_infinite(), inf, "{round:?}");
        // 3^(2^62): far past the range; its reciprocal far below.
        let (x, o) = three.powi_round(1 << 62, 20, round);
        assert_eq!(x.is_infinite(), inf, "{round:?}");
        assert_eq!(
            o,
            if inf {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        );
        let (x, o) = three.powi_round(-(1 << 62), 20, round);
        assert_eq!(x.is_zero(), !matches!(round, Round::Up), "{round:?}");
        assert_eq!(
            o,
            if x.is_zero() {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        );
        let (x, _) = (-&three).powi_round((1 << 62) + 1, 20, round);
        assert!(x.is_sign_negative());
        let (x, _) = half.powi_round(i64::MAX, 20, round);
        assert!(x.is_zero() || x.exponent() == Some(BigFloat::EXP_MIN));
        // A huge power of a number near 1 stays in range.
        let near = BigFloat::from(1)
            .add_round(
                &BigFloat::from_parts(&one(), -70, 1, Round::Nearest).0,
                80,
                Round::Nearest,
            )
            .0;
        let big = Number::from(1u64 << 63) * Number::from(1u64 << 10);
        let (x, _) = near.pow_number_round(&big, 53, round);
        assert!(x.is_normal(), "{round:?} {x:?}");
        // (1 + 2^-70)^(2^73) = e^8 within 2^-60.
        let d = (&x - 2980.957987041728f64).abs();
        assert!(d < 1e-9, "{x:?}");
    }
}

#[test]
fn power_special_values_follow_mpfr() {
    let nan = BigFloat::nan(10);
    let zero = BigFloat::zero(10);
    let nz = BigFloat::neg_zero(10);
    let inf = BigFloat::infinity(10);
    let ninf = BigFloat::neg_infinity(10);
    assert_eq!(nan.powi(0), BigFloat::from(1));
    assert!(nan.powi(3).is_nan());
    for (x, n, want) in [
        (&zero, 3, BigFloat::zero(1)),
        (&nz, 3, BigFloat::neg_zero(1)),
        (&nz, 2, BigFloat::zero(1)),
        (&zero, -3, BigFloat::infinity(1)),
        (&nz, -3, BigFloat::neg_infinity(1)),
        (&nz, -2, BigFloat::infinity(1)),
        (&inf, 3, BigFloat::infinity(1)),
        (&ninf, 3, BigFloat::neg_infinity(1)),
        (&ninf, 2, BigFloat::infinity(1)),
        (&ninf, -3, BigFloat::neg_zero(1)),
        (&ninf, -2, BigFloat::zero(1)),
        (&inf, -1, BigFloat::zero(1)),
    ] {
        let got = x.powi_round(n, 10, Round::Nearest);
        same(
            &got,
            &(want.prec_round(10, Round::Nearest).0, Ordering::Equal),
            &format!("{x:?}^{n}"),
        );
    }
    // Roots.
    assert!(
        BigFloat::from(8)
            .nth_root_round(0, 10, Round::Nearest)
            .0
            .is_nan()
    );
    assert!(nz.nth_root(3).is_sign_negative() && nz.nth_root(3).is_zero());
    assert!(nz.nth_root(4).is_sign_positive() && nz.nth_root(4).is_zero());
    assert!(nz.nth_root(2).is_sign_positive());
    assert!(ninf.nth_root(3).is_infinite() && ninf.nth_root(3).is_sign_negative());
    assert!(ninf.nth_root(2).is_nan());
    assert!(BigFloat::from(-1).nth_root(6).is_nan());
    assert_eq!(BigFloat::from(-1).nth_root(7), BigFloat::from(-1));
    assert!(nz.cbrt().is_sign_negative());
    // The reciprocal square root.
    assert!(nz.rec_sqrt().is_infinite() && nz.rec_sqrt().is_sign_positive());
    assert!(zero.rec_sqrt().is_infinite());
    assert!(inf.rec_sqrt().is_zero() && inf.rec_sqrt().is_sign_positive());
    assert!(BigFloat::from(-4).rec_sqrt().is_nan() && ninf.rec_sqrt().is_nan());
    // Rational powers.
    let third = Rational::from(1) / 3u32;
    let half = Rational::from(1) / 2u32;
    let m2_3 = Rational::from(-2) / 3u32;
    assert!(nz.pow_rational(&third).is_sign_negative());
    assert!(nz.pow_rational(&half).is_sign_positive() && nz.pow_rational(&half).is_zero());
    assert!(nz.pow_rational(&m2_3).is_infinite() && nz.pow_rational(&m2_3).is_sign_positive());
    assert!(BigFloat::from(-8).pow_rational(&half).is_nan());
    assert!(ninf.pow_rational(&half).is_nan());
    assert!(
        ninf.pow_rational(&third).is_infinite() && ninf.pow_rational(&third).is_sign_negative()
    );
    assert!(ninf.pow_rational(&m2_3).is_zero() && ninf.pow_rational(&m2_3).is_sign_positive());
    assert!(nan.pow_rational(&Rational::from(0)).eq(&BigFloat::from(1)));
}

#[test]
fn roots_round_correctly() {
    let mut s = 0x9017_0003u64;
    for i in 0..size(1500, 15) {
        let x = float(&mut s, if i % 7 == 0 { 400 } else { 90 }, 60);
        let n = match below(&mut s, 6) {
            0 => 3,
            1 => 4,
            2 => 5,
            3 => 2 + below(&mut s, 12) as u32,
            4 => 50 + below(&mut s, 100) as u32,
            _ if i % 4 == 0 => 1000 + below(&mut s, 5000) as u32,
            _ => 7,
        };
        let x = if n % 2 == 0 { x.abs() } else { x };
        let p = prec(&mut s);
        let rx = rational(&x);
        for round in MODES {
            let got = x.nth_root_round(n, p, round);
            bracket(&got, p, round, &format!("{x:?}^(1/{n}) at {p}"), |r| {
                // `r^n` against `x`, the sign of `r` that of `x` (odd `n`).
                pow(r, u64::from(n)).cmp(&rx)
            });
        }
    }
}

#[test]
fn exact_and_halfway_roots_are_found() {
    let mut s = 0x9017_0004u64;
    for _ in 0..size(1500, 15) {
        // `y^n` for a short `y`: the root is exact at its bits and halfway one bit below.
        let n = 2 + below(&mut s, 7) as u32;
        let by = 1 + below(&mut s, 60);
        let y = mantissa(&mut s, by);
        let y = BigFloat::from_parts(&y, below(&mut s, 100) as i64 - 50, by, Round::Nearest).0;
        let y = if n % 2 == 1 && xorshift(&mut s) & 1 == 1 {
            -y
        } else {
            y
        };
        let x = y.powi_round(i64::from(n), by * u64::from(n) + 2, Round::Nearest);
        assert_eq!(x.1, Ordering::Equal);
        let x = x.0;
        let ry = rational(&y);
        let p = (by as i64 + below(&mut s, 4) as i64 - 2).max(1) as u64;
        for round in MODES {
            same(
                &x.nth_root_round(n, p, round),
                &reference(&ry, p, round),
                &format!("{x:?}^(1/{n}) at {p}"),
            );
            if n == 3 {
                same(&x.cbrt_round(p, round), &reference(&ry, p, round), "cbrt");
            }
        }
    }
}

#[test]
fn reciprocal_square_roots_round_correctly() {
    let mut s = 0x9017_0005u64;
    for i in 0..size(2000, 20) {
        let x = float(&mut s, if i % 7 == 0 { 600 } else { 140 }, 80).abs();
        let p = prec(&mut s);
        let rx = rational(&x);
        for round in MODES {
            let got = x.rec_sqrt_round(p, round);
            // `r^2·x` against 1, decreasing in... increasing in `r > 0`.
            bracket(&got, p, round, &format!("1/sqrt {x:?} at {p}"), |r| {
                (&(r * r) * &rx).cmp(&Rational::from(1))
            });
        }
    }
    // Powers of 4, and of 2 that are not.
    for k in -40i64..40 {
        let x = BigFloat::from_parts(&one(), k, 1, Round::Nearest).0;
        let got = x.rec_sqrt_round(30, Round::Nearest);
        if k % 2 == 0 {
            same(
                &got,
                &(
                    BigFloat::from_parts(&one(), -k / 2, 30, Round::Nearest).0,
                    Ordering::Equal,
                ),
                "4^j",
            );
        } else {
            assert_ne!(got.1, Ordering::Equal);
        }
    }
}

#[test]
fn rational_powers_round_correctly() {
    let mut s = 0x9017_0006u64;
    for i in 0..size(800, 10) {
        let q = 2 + below(&mut s, 6);
        let mut pn = below(&mut s, 13) as i64 - 6;
        while pn == 0 || gcd(pn.unsigned_abs(), q) != 1 {
            pn += 1;
        }
        let r = Rational::from(pn) / q as u32;
        let x = float(&mut s, if i % 5 == 0 { 200 } else { 60 }, 30);
        let x = if q.is_multiple_of(2) { x.abs() } else { x };
        let p = prec(&mut s);
        let rx = rational(&x);
        for round in MODES {
            let got = x.pow_rational_round(&r, p, round);
            let negative = x.is_sign_negative() && pn % 2 != 0;
            assert_eq!(got.0.is_sign_negative(), negative, "{x:?}^{pn}/{q}");
            // `y^q` against `x^p`: increasing in |y| for p > 0, so in y up to the sign.
            let xp = if pn > 0 {
                pow(&rx, pn as u64)
            } else {
                Rational::from(1) / pow(&rx, pn.unsigned_abs())
            };
            bracket(&got, p, round, &format!("{x:?}^({pn}/{q}) at {p}"), |y| {
                let c = pow(&y.abs(), q).cmp(&xp.abs());
                if negative { c.reverse() } else { c }
            });
        }
    }
    // Exact rational powers.
    let x = BigFloat::from(64);
    for (pn, q, want) in [
        (1i64, 2u32, 8i64),
        (2, 3, 16),
        (-1, 2, 0),
        (5, 6, 32),
        (3, 2, 512),
    ] {
        let r = Rational::from(pn) / q;
        let got = x.pow_rational_round(&r, 20, Round::Nearest);
        let w = if want == 0 {
            BigFloat::from(0.125)
        } else {
            BigFloat::from(want)
        };
        same(
            &got,
            &(w.prec_round(20, Round::Nearest).0, Ordering::Equal),
            &format!("64^({pn}/{q})"),
        );
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// A value of a line of vectors: `prec:value`.
fn parse(field: &str) -> BigFloat {
    let (p, v) = field.split_once(':').expect("prec:value");
    let p: u64 = p.parse().expect("a precision");
    match v {
        "nan" => BigFloat::nan(p),
        "inf" => BigFloat::infinity(p),
        "-inf" => BigFloat::neg_infinity(p),
        "0" => BigFloat::zero(p),
        "-0" => BigFloat::neg_zero(p),
        _ => {
            let (m, e) = v.split_once('p').expect("mantissa p exponent");
            let m = Number::from_str_radix(m, 16, ByteOrder::LittleEndian).expect("hexadecimal");
            let (x, o) = BigFloat::from_parts(&m, e.parse().expect("exponent"), p, Round::Nearest);
            assert_eq!(o, Ordering::Equal);
            x
        }
    }
}

/// Every line of MPFR's vectors: `op prec mode operands... = result ternary`.
#[test]
fn mpfr_vectors_agree() {
    let data = include_str!("data/speed_vectors.txt");
    let lines: Vec<&str> = data
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    let step = if cfg!(miri) { 97 } else { 1 };
    let mut count = 0;
    for line in lines.iter().step_by(step) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let eq = f.iter().position(|&t| t == "=").expect("=");
        let (op, p, m) = (f[0], f[1].parse::<u64>().unwrap(), f[2]);
        let round = match m {
            "N" => Round::Nearest,
            "Z" => Round::TowardZero,
            "U" => Round::Up,
            _ => Round::Down,
        };
        let xs: Vec<BigFloat> = f[3..eq].iter().map(|t| parse(t)).collect();
        let want = (parse(f[eq + 1]), f[eq + 2].parse::<i8>().unwrap().cmp(&0));
        let got = match op {
            "add" => xs[0].add_round(&xs[1], p, round),
            "sub" => xs[0].sub_round(&xs[1], p, round),
            "mul" => xs[0].mul_round(&xs[1], p, round),
            "sqr" => xs[0].mul_round(&xs[0], p, round),
            "div" => xs[0].div_round(&xs[1], p, round),
            "fma" => xs[0].mul_add_round(&xs[1], &xs[2], p, round),
            "sqrt" => xs[0].sqrt_round(p, round),
            "rec_sqrt" => xs[0].rec_sqrt_round(p, round),
            "cbrt" => xs[0].cbrt_round(p, round),
            "root5" => xs[0].nth_root_round(5, p, round),
            "root100" => xs[0].nth_root_round(100, p, round),
            "pow7" => xs[0].powi_round(7, p, round),
            "powm3" => xs[0].powi_round(-3, p, round),
            "pow1e9" => xs[0].powi_round(1_000_000_007, p, round),
            _ => panic!("op {op}"),
        };
        same(&got, &want, line);
        count += 1;
    }
    assert!(count > 0);
}
