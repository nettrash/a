//! The elementary functions of `BigComplex` against GNU MPC 1.4.1's results
//! (`data/cwave2_vectors.txt`: random arguments with zero, special, tiny, huge parts and parts
//! near ±1, precisions 1 to 200, about 1000 and 10^4 bits, every mode pair; and
//! `data/cwave2_specials.txt`: every number whose parts are NaN, ±∞, ±0, 1, -2 or 1/2, and every
//! pair for the power, in the modes NN and DD), made by the cwave2 harness's `vectors` and
//! `specials` subcommands; and their exact cases, the four modes, the roundings beside the first
//! terms of tiny arguments, the parts near ±1 and beyond the exponent range, and the documented
//! differences from MPC.

use super::common::{MODES, below, float, mode, size};
use a::math::float::{BigComplex, BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;

/// A value of the vectors at the precision `prec`.
fn value(s: &str, prec: u64) -> BigFloat {
    match s {
        "nan" => BigFloat::nan(prec),
        "inf" => BigFloat::infinity(prec),
        "-inf" => BigFloat::neg_infinity(prec),
        "0" => BigFloat::zero(prec),
        "-0" => BigFloat::neg_zero(prec),
        _ => {
            let (negative, s) = match s.strip_prefix('-') {
                Some(s) => (true, s),
                None => (false, s),
            };
            let s = s.strip_prefix("0x").expect("a hexadecimal mantissa");
            let (m, e) = s.split_once('p').expect("an exponent");
            let mut m = Number::from_str_radix(m, 16, ByteOrder::LittleEndian).expect("hex");
            if negative {
                m = -m;
            }
            let (x, o) =
                BigFloat::from_parts(&m, e.parse().expect("exponent"), prec, Round::Nearest);
            assert_eq!(o, Ordering::Equal, "{s} is not exact at {prec} bits");
            x
        }
    }
}

/// An input of the vectors, `prec:value`.
fn input(s: &str) -> BigFloat {
    let (p, v) = s.split_once(':').expect("prec:value");
    value(v, p.parse().expect("precision"))
}

fn mode_of(c: u8) -> Round {
    match c {
        b'N' => Round::Nearest,
        b'Z' => Round::TowardZero,
        b'U' => Round::Up,
        b'D' => Round::Down,
        _ => panic!("a mode"),
    }
}

fn ternary(s: &str) -> Ordering {
    match s {
        "-1" => Ordering::Less,
        "0" => Ordering::Equal,
        "1" => Ordering::Greater,
        _ => panic!("a ternary"),
    }
}

type Part = (BigFloat, Ordering);

/// Whether `got` is `want`: the same precision, value (zeros of one sign, NaN as NaN) and
/// direction.
fn agrees(got: &Part, want: &Part) -> bool {
    got.0.prec() == want.0.prec()
        && got.1 == want.1
        && (got.0.is_nan() && want.0.is_nan()
            || got.0 == want.0 && got.0.is_sign_negative() == want.0.is_sign_negative())
}

fn parts((z, o): (BigComplex, (Ordering, Ordering))) -> Vec<Part> {
    let (a, b) = z.into_parts();
    vec![(a, o.0), (b, o.1)]
}

/// The unary functions by name.
fn unary(op: &str, x: &BigComplex, p: (u64, u64), r: (Round, Round)) -> Vec<Part> {
    match op {
        "exp" => parts(x.exp_round_parts(p, r)),
        "ln" => parts(x.ln_round_parts(p, r)),
        "log10" => parts(x.log10_round_parts(p, r)),
        "proj" => parts(x.proj_round_parts(p, r)),
        "arg" => vec![x.arg_round(p.0, r.0)],
        "sin" => parts(x.sin_round_parts(p, r)),
        "cos" => parts(x.cos_round_parts(p, r)),
        "tan" => parts(x.tan_round_parts(p, r)),
        "sinh" => parts(x.sinh_round_parts(p, r)),
        "cosh" => parts(x.cosh_round_parts(p, r)),
        "tanh" => parts(x.tanh_round_parts(p, r)),
        "asin" => parts(x.asin_round_parts(p, r)),
        "acos" => parts(x.acos_round_parts(p, r)),
        "atan" => parts(x.atan_round_parts(p, r)),
        "asinh" => parts(x.asinh_round_parts(p, r)),
        "acosh" => parts(x.acosh_round_parts(p, r)),
        "atanh" => parts(x.atanh_round_parts(p, r)),
        "sin_cos" => {
            let (s, c) = x.sin_cos_round_parts(p, r, p, r);
            let mut v = parts(s);
            v.extend(parts(c));
            v
        }
        _ => panic!("op {op}"),
    }
}

/// Every line of the vectors, at most `every`-th under Miri.
fn lines(data: &str) -> impl Iterator<Item = &str> {
    let every = if cfg!(miri) { 29 } else { 1 };
    data.lines()
        .enumerate()
        .filter(move |(i, l)| {
            i % every == 0
                && !(cfg!(miri) && (l.contains(" 10000 ") || l.contains(" 9") || l.contains(" 10")))
        })
        .map(|(_, l)| l)
}

#[test]
fn every_vector_of_mpc_is_matched() {
    let data = include_str!("data/cwave2_vectors.txt");
    let mut n = 0;
    for line in lines(data) {
        let mut sides = line.split(" | ");
        let head: Vec<&str> = sides.next().expect("head").split(' ').collect();
        let args: Vec<&str> = sides.next().expect("arguments").split(' ').collect();
        let res: Vec<&str> = sides.next().expect("results").split(' ').collect();
        let op = head[0];
        let m = head[1].as_bytes();
        let r = (mode_of(m[0]), mode_of(m[1]));
        let p: (u64, u64) = (head[2].parse().unwrap(), head[3].parse().unwrap());
        let x = BigComplex::new(input(args[0]), input(args[1]));
        let got = match op {
            "pow" => {
                parts(x.pow_round_parts(&BigComplex::new(input(args[2]), input(args[3])), p, r))
            }
            "pow_fr" => parts(x.pow_real_round_parts(&input(args[2]), p, r)),
            "pow_si" => parts(x.powi_round_parts(args[2].parse().unwrap(), p, r)),
            _ => unary(op, &x, p, r),
        };
        let k = got.len();
        for (j, g) in got.iter().enumerate() {
            let prec = if op == "arg" || j % 2 == 0 { p.0 } else { p.1 };
            let want = (value(res[j], prec), ternary(res[k + j]));
            assert!(agrees(g, &want), "{line}: part {j}: {g:?}, MPC {want:?}");
        }
        n += 1;
    }
    assert!(n > 10, "{n} vectors");
}

/// A value of the special table.
fn special(c: u8) -> BigFloat {
    match c {
        b'n' => BigFloat::nan(53),
        b'I' => BigFloat::infinity(53),
        b'J' => BigFloat::neg_infinity(53),
        b'0' => BigFloat::zero(53),
        b'z' => BigFloat::neg_zero(53),
        b'1' => BigFloat::from(1).with_prec(53),
        b'm' => BigFloat::from(-2).with_prec(53),
        b'h' => BigFloat::from(0.5).with_prec(53),
        _ => panic!("a special value"),
    }
}

/// The class of a value in the table.
fn class(x: &BigFloat) -> u8 {
    if x.is_nan() {
        b'n'
    } else if x.is_infinite() {
        if x.is_sign_negative() { b'J' } else { b'I' }
    } else if x.is_zero() {
        if x.is_sign_negative() { b'z' } else { b'0' }
    } else if x.is_sign_negative() {
        b'q'
    } else {
        b'p'
    }
}

#[test]
fn special_values_are_mpcs() {
    let data = include_str!("data/cwave2_specials.txt");
    let every = if cfg!(miri) { 17 } else { 1 };
    let mut n = 0;
    for line in data.lines().step_by(every) {
        let (head, res) = line.split_once(" | ").expect("results");
        let head: Vec<&str> = head.split(' ').collect();
        let op = head[0];
        let a = head[1].as_bytes();
        let x = BigComplex::new(special(a[0]), special(a[1]));
        for (name, r) in [
            ("NN", (Round::Nearest, Round::Nearest)),
            ("DD", (Round::Down, Round::Down)),
        ] {
            let want = res
                .split(' ')
                .find_map(|s| s.strip_prefix(name).and_then(|s| s.strip_prefix(':')))
                .expect("a mode");
            let got = match op {
                "pow" => {
                    let b = head[2].as_bytes();
                    parts(x.pow_round_parts(
                        &BigComplex::new(special(b[0]), special(b[1])),
                        (53, 53),
                        r,
                    ))
                }
                "pow_si" => parts(x.powi_round_parts(head[2].parse().unwrap(), (53, 53), r)),
                _ => unary(op, &x, (53, 53), r),
            };
            let classes: Vec<u8> = got.iter().map(|g| class(&g.0)).collect();
            assert_eq!(classes, want.as_bytes(), "{line} in {name}");
        }
        n += 1;
    }
    assert!(n > 100);
}

/// The square of a number of equal parts is an exact imaginary number: its real part, a
/// cancellation, is `-0` when it rounds down, IEEE 754's sum (as the crate's `sqr`); MPC 1.4.1's
/// `mpc_sqr` (and so `mpc_pow_si` with 2) gives `+0`. The one class of the special table that
/// differs.
#[test]
fn squares_by_powi_have_ieee_zero_signs() {
    for v in [1.0, -2.0, 0.5] {
        let z = BigComplex::new(BigFloat::from(v), BigFloat::from(v));
        let (s, o) = z.powi_round(2, 53, Round::Down);
        assert!(s.re().is_zero() && s.re().is_sign_negative(), "{v}");
        assert_eq!(o, (Ordering::Equal, Ordering::Equal));
        assert_eq!(*s.im(), BigFloat::from(2.0 * v * v));
        let (s, _) = z.powi_round(2, 53, Round::Nearest);
        assert!(s.re().is_zero() && !s.re().is_sign_negative());
    }
}

#[test]
fn exact_values_are_exact() {
    let n = (Round::Nearest, Round::Nearest);
    let eq = (Ordering::Equal, Ordering::Equal);
    let c = |a: f64, b: f64| BigComplex::new(BigFloat::from(a), BigFloat::from(b));
    for &round in &MODES {
        let r = (round, round);
        // e^0 = 1, ln 1 = 0, ln(-1) = πi, ln(±i) = ±πi/2.
        let (e, o) = BigComplex::zero(53).exp_round_parts((53, 53), r);
        assert_eq!((e, o), (c(1.0, 0.0), eq));
        let (l, o) = BigComplex::from(1).ln_round_parts((53, 53), r);
        assert_eq!(o, eq);
        assert!(l.is_zero());
        let (l, o) = c(-1.0, 0.0).ln_round_parts((53, 53), r);
        assert_eq!((l.re().clone(), o.0), (BigFloat::zero(53), Ordering::Equal));
        assert_eq!(l.im().clone(), BigFloat::pi_round(53, round).0);
        // log10 of |z|² = 10^m: m/2.
        for (a, b, m) in [
            (6.0, 8.0, 2.0),
            (1.0, 3.0, 1.0),
            (-28.0, 96.0, 4.0),
            (0.0, -1000.0, 6.0),
        ] {
            let (l, o) = c(a, b).log10_round_parts((24, 53), r);
            assert_eq!(
                (l.re().clone(), o.0),
                (BigFloat::from(m / 2.0), Ordering::Equal),
                "{a} {b}"
            );
        }
        // Powers: square roots, integer powers, inverses.
        let (q, o) = c(3.0, 4.0).pow_real_round_parts(&BigFloat::from(0.5), (53, 53), r);
        assert_eq!((q, o), (c(2.0, 1.0), eq));
        let (q, o) = c(-7.0, 24.0).pow_real_round_parts(&BigFloat::from(-0.5), (53, 53), r);
        // √(-7 + 24i) = 3 + 4i, and 1/(3 + 4i) = (3 - 4i)/25 is not dyadic: inexact.
        assert_ne!(o, eq);
        let _ = q;
        let (q, o) = c(2.0, 2.0).powi_round_parts(-3, (53, 53), r);
        assert_eq!((q, o), (c(-1.0 / 32.0, -1.0 / 32.0), eq));
        let (q, o) = c(1.0, 1.0).pow_real_round_parts(&BigFloat::from(0.25), (53, 53), r);
        assert_ne!(o.0, Ordering::Equal);
        let _ = q;
        let (q, o) =
            c(-119.0, -120.0).pow_round_parts(&BigComplex::from(BigFloat::from(0.25)), (53, 53), r);
        // (2 - 3i)^4 = -119 + 120i; (3 + 2i)^4 = -119 + 120i: the principal fourth root of -119
        // - 120i is 3 - 2i.
        assert_eq!((q, o), (c(3.0, -2.0), eq), "{round:?}");
        let _ = n;
    }
    // (1 + 2i)^n against the exact product, n to 40, at a precision that holds it.
    let z = c(1.0, 2.0);
    let mut acc = BigComplex::from(1);
    for k in 1..=40i64 {
        acc = acc.mul_round(&z, 200, Round::Nearest).0;
        let (q, o) = z.powi_round(k, 200, Round::Nearest);
        assert_eq!(o, eq, "{k}");
        assert_eq!(q, acc, "{k}");
        let (q, o) = z.pow_round(&BigComplex::from(k), 200, Round::Nearest);
        assert_eq!((q, o), (acc.clone(), eq), "{k}");
    }
}

/// The four modes bracket each part: down and up neighbours, nearest one of them, toward zero
/// the smaller in magnitude; and each part rounded at `p` equals the result at `p + 64`, rounded
/// again in a directed mode (when that one is not exact).
#[test]
fn modes_bracket_and_reround() {
    let mut s = 0x1234_5678_9ABC_DEF1u64;
    let ops = [
        "exp", "ln", "log10", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan",
        "asinh", "acosh", "atanh",
    ];
    for i in 0..size(400, 20) {
        let op = ops[i % ops.len()];
        let x = BigComplex::new(float(&mut s, 130, 6), float(&mut s, 130, 6));
        let p = 1 + below(&mut s, if cfg!(miri) { 40 } else { 140 });
        let get = |r: Round, p: u64| unary(op, &x, (p, p), (r, r));
        let (dn, up, ne, tz) = (
            get(Round::Down, p),
            get(Round::Up, p),
            get(Round::Nearest, p),
            get(Round::TowardZero, p),
        );
        let hi_d = get(Round::Down, p + 64);
        for j in 0..2 {
            let (d, u, n, z) = (&dn[j], &up[j], &ne[j], &tz[j]);
            if d.1 == Ordering::Equal {
                assert!(agrees(d, u), "{op} {x:?} {p}");
                continue;
            }
            assert_eq!(
                (d.1, u.1),
                (Ordering::Less, Ordering::Greater),
                "{op} {x:?} {p}"
            );
            assert!(d.0 < u.0, "{op} {x:?}");
            assert!(agrees(n, d) || agrees(n, u), "{op} {x:?} {p}");
            let toward = if d.0.is_sign_negative() { u } else { d };
            assert!(agrees(z, toward), "{op} {x:?} {p}");
            assert_eq!(
                hi_d[j].0.prec_round(p, Round::Down).0,
                d.0,
                "{op} {x:?} {p} part {j}"
            );
        }
        let _ = mode(&mut s);
    }
}

/// Tiny arguments, far below the precision: each part rounds beside its first term at once
/// (`sin z = z - z³/6 + ...`, `e^z = 1 + z + ...`), even at exponents near the bottom of the range.
#[test]
fn tiny_arguments_round_beside_their_first_terms() {
    for e in [-200i64, -5000, -(1 << 40), -(1 << 60)] {
        let x = BigFloat::from_parts(&Number::from(3), e, 53, Round::Nearest).0; // 3·2^e
        let y = BigFloat::from_parts(&Number::from(-5), e, 53, Round::Nearest).0;
        let z = BigComplex::new(x.clone(), y.clone());
        // Re sin z = x(1 + (3y² - x²)/6 + ...) > x: up for Up, x itself and Less for Down.
        let (s, o) = z.sin_round(53, Round::Down);
        assert_eq!((s.re().clone(), o.0), (x.clone(), Ordering::Less));
        // Im sin z = y(1 + (y² - 3x²)/6 + ...): |Im| > |y|, below y: Down gives the next below.
        assert_eq!((s.im().clone(), o.1), (y.clone(), Ordering::Less));
        // asin, atan: Re asin z = x(1 + (x² - 3y²)/6) < x; Re atan z = x(1 - (x² - 3y²)/3) > x.
        let (a, o) = z.asin_round(53, Round::Up);
        assert_eq!((a.re().clone(), o.0), (x.clone(), Ordering::Greater));
        let (a, o) = z.atan_round(53, Round::Down);
        assert_eq!((a.re().clone(), o.0), (x.clone(), Ordering::Less));
        // e^z = 1 + z + ...: the real part 1 + x + ... rounds to 1 from above (x > 0), the
        // imaginary part y(1 + x + ...) beside y, larger in magnitude.
        let (w, o) = z.exp_round(53, Round::Nearest);
        assert_eq!((w.re().clone(), o.0), (BigFloat::from(1), Ordering::Less));
        assert_eq!((w.im().clone(), o.1), (y.clone(), Ordering::Greater));
        // cos z = 1 - (x² - y²)/2 + ...: above 1 (|y| > |x|); Im = -xy(...) beside -xy.
        let (c, o) = z.cos_round(53, Round::TowardZero);
        assert_eq!((c.re().clone(), o.0), (BigFloat::from(1), Ordering::Less));
        let xy = x.mul_round(&y, 53, Round::Nearest).0;
        if e > -(1 << 59) {
            assert_eq!(c.im().clone(), -xy);
        }
        // tan, tanh, sinh, asinh, atanh: beside their arguments.
        let (t, _) = z.tan_round(53, Round::Nearest);
        assert_eq!(*t.re(), x);
        let (t, _) = z.atanh_round(53, Round::Nearest);
        assert_eq!(*t.im(), y);
    }
}

/// The tangent of `x + yi` for a large `|y|`: its imaginary part lies within `e^(-2|y|)` of ±1,
/// above 1 in magnitude exactly when `cos 2x < 0` (MPC 1.4.1 takes it below 1 for `|y|` beyond
/// its exponential's range); its real part underflows with the sign of `sin 2x`.
#[test]
fn tangents_of_large_imaginary_parts() {
    for (x, cos2x_negative) in [
        (1.0f64, true),
        (0.3, false),
        (-7.5, true),
        (2.0, true),
        (-1.0, true),
        (0.7, false),
        (-0.2, false),
    ] {
        for y in [100.0f64, -3000.0, 1e19, -1e300] {
            let z = BigComplex::new(BigFloat::from(x), BigFloat::from(y));
            let (t, o) = z.tan_round(53, Round::TowardZero);
            let one = BigFloat::from(if y > 0.0 { 1 } else { -1 });
            if cos2x_negative {
                // |Im| > 1: toward zero gives ±1.
                assert_eq!(*t.im(), one, "{x} {y}");
                assert_eq!(
                    o.1,
                    if y > 0.0 {
                        Ordering::Less
                    } else {
                        Ordering::Greater
                    }
                );
            } else {
                assert!(t.im().abs() < 1, "{x} {y}");
            }
            let sin2x_negative = (2.0 * x).sin() < 0.0;
            assert_eq!(t.re().is_sign_negative(), sin2x_negative, "{x} {y}");
            let (h, _) = BigComplex::new(BigFloat::from(y), BigFloat::from(x))
                .tanh_round(53, Round::Nearest);
            let (t, _) = z.tan_round(53, Round::Nearest);
            assert_eq!(*h.re(), *t.im());
        }
    }
}

/// Parts beyond the exponent range overflow and underflow by their modes, without the cost of
/// the exponent: `e^(x + yi)` for `x` near and beyond `2^62 ln 2`, `sin(1 + yi)` for a huge `y`.
#[test]
fn parts_beyond_the_range_overflow_by_mode() {
    let huge = BigFloat::from_parts(&Number::from(1), 70, 53, Round::Nearest).0;
    let z = BigComplex::new(huge.clone(), BigFloat::from(1));
    let (e, o) = z.exp_round(53, Round::TowardZero);
    assert!(e.re().is_finite() && e.im().is_finite());
    assert_eq!(o, (Ordering::Less, Ordering::Less));
    let (e, _) = z.exp_round(53, Round::Nearest);
    assert!(e.re().is_infinite() && e.im().is_infinite());
    // e^-x cos(-1) > 0 and e^-x sin(-1) < 0 below the range: up, the least number and -0.
    let (e, o) = (-&z).exp_round(53, Round::Up);
    assert!(!e.re().is_zero() && !e.re().is_sign_negative());
    assert!(e.im().is_zero() && e.im().is_sign_negative());
    assert_eq!(o, (Ordering::Greater, Ordering::Greater));
    let s = BigComplex::new(BigFloat::from(1), huge)
        .sin_round(24, Round::Nearest)
        .0;
    assert!(s.re().is_infinite() && s.im().is_infinite());
    // e^x near the threshold: 2^62 ln 2 - 1, whose exponential is about 2^(2^62)/e, in range.
    let t = BigFloat::from_parts(&Number::from(1), 62, 200, Round::Nearest)
        .0
        .mul_round(
            &BigFloat::from(2).ln_round(200, Round::Nearest).0,
            200,
            Round::Nearest,
        )
        .0
        .sub_round(&BigFloat::from(2), 200, Round::Nearest)
        .0;
    let z = BigComplex::new(t, BigFloat::from(0.5));
    let (e, _) = z.exp_round(53, Round::Nearest);
    assert!(e.re().is_finite() && e.im().is_finite(), "{e:?}");
}

/// MPC 1.4.1's logarithm takes `|x|` as exact at its working precision in its second method
/// (`mpfr_abs` into a shorter variable), and is wrong for a `|z|` near 1 with a long part:
/// `ln|z|` of `2^-170 + (1 + 2^-62)i` is about `2^-62`, MPC gives about `2^-347`.
#[test]
fn logarithms_near_one_with_long_parts() {
    let x = BigFloat::from_parts(
        &Number::from_str_radix("42137fe0013ffffff", 16, ByteOrder::LittleEndian).unwrap(),
        -239,
        67,
        Round::Nearest,
    )
    .0;
    let y = BigFloat::from_parts(
        &Number::from_str_radix("4000000000000001", 16, ByteOrder::LittleEndian).unwrap(),
        -62,
        67,
        Round::Nearest,
    )
    .0;
    let z = BigComplex::new(x, -y);
    let (l, o) = z.ln_round(52, Round::Up);
    assert_eq!(
        *l.re(),
        BigFloat::from_parts(&Number::from(1), -62, 52, Round::Nearest).0
    );
    assert_eq!(o.0, Ordering::Greater);
    // Against the result at 300 bits more, rounded again.
    let (h, _) = z.ln_round(352, Round::Nearest);
    assert_eq!(h.re().prec_round(52, Round::Up).0, *l.re());
}

/// sin_cos is sin and cos; the identities `sinh z = -i sin(iz)`, `cosh z = cos(iz)`, `asinh z =
/// -i asin(iz)`, `exp(ln z) ≈ z`, `sin² + cos² ≈ 1`.
#[test]
fn identities() {
    let mut s = 0xFEED_FACE_CAFE_BEEFu64;
    for _ in 0..size(60, 6) {
        let z = BigComplex::new(float(&mut s, 100, 4), float(&mut s, 100, 4));
        let p = 120;
        let n = Round::Nearest;
        let ((si, _), (co, _)) = z.sin_cos_round(p, n);
        assert_eq!(si, z.sin_round(p, n).0);
        assert_eq!(co, z.cos_round(p, n).0);
        let iz = z.mul_i(false);
        let sinh = z.sinh_round(p, n).0;
        assert_eq!(sinh, iz.sin_round(p, n).0.mul_i(true));
        assert_eq!(z.cosh_round(p, n).0, iz.cos_round(p, n).0);
        assert_eq!(z.asinh_round(p, n).0, iz.asin_round(p, n).0.mul_i(true));
        let back = z.ln_round(200, n).0.exp_round(200, n).0;
        let d = back.sub_round(&z, 200, n).0.abs();
        assert!(
            d <= z
                .abs()
                .mul_round(&BigFloat::from_parts(&Number::from(1), -90, 53, n).0, 53, n)
                .0,
            "{z:?}"
        );
        let one = si.sqr().add_round(&co.sqr(), p, n).0;
        let e = one.sub_round(&BigComplex::from(1), p, n).0.abs();
        if z.re().abs() < 4 && z.im().abs() < 4 {
            assert!(
                e < BigFloat::from_parts(&Number::from(1), -80, 53, n).0,
                "{z:?} {e:?}"
            );
        }
    }
}

/// A number near the places where the functions cancel: parts near ±1 and ±i, `|z|` near 1, a
/// part tiny beside the other, or ordinary.
fn near_places(s: &mut u64) -> BigComplex {
    let p = 1 + below(s, 120);
    let t = float(s, p, 0);
    let small = BigFloat::from_parts(
        &Number::from(1 + below(s, 1000) as i64),
        -(below(s, 90) as i64) - 8,
        64,
        Round::Nearest,
    )
    .0;
    let one = BigFloat::from(if below(s, 2) == 0 { 1 } else { -1 });
    let near1 = one.add_round(&small, 200, Round::Nearest).0;
    match below(s, 5) {
        0 => BigComplex::new(near1, float(s, p, 2)),
        1 => BigComplex::new(float(s, p, 2), near1),
        2 => {
            // |z| near 1: (cos θ, sin θ) rounded to the precision.
            let (sn, cs) = t.sin_cos_round(p.max(8), Round::Nearest);
            BigComplex::new(cs.0, sn.0)
        }
        3 => BigComplex::new(t, small),
        _ => BigComplex::new(float(s, p, 3), float(s, p, 3)),
    }
}

/// Each part rounded down at `p` bits is the part rounded down at `p + 80` bits, rounded down
/// again (rounding down twice is rounding down once), for arguments near the places where the
/// formulas cancel: the bounds are tight enough there and sound.
#[test]
fn parts_near_cancellations_agree_with_longer_ones() {
    let mut s = 0x0DDB_A11C_AFE5_7A7Eu64;
    let ops = [
        "exp", "ln", "log10", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan",
        "asinh", "acosh", "atanh",
    ];
    for i in 0..size(3000, 15) {
        let op = ops[i % ops.len()];
        let x = near_places(&mut s);
        let p = 2 + below(&mut s, if cfg!(miri) { 40 } else { 200 });
        let d = (Round::Down, Round::Down);
        let lo = unary(op, &x, (p, p), d);
        let hi = unary(op, &x, (p + 80, p + 80), d);
        for j in 0..lo.len() {
            assert_eq!(
                hi[j].0.prec_round(p, Round::Down).0,
                lo[j].0,
                "{op} {x:?} {p} part {j}"
            );
        }
        if i % 5 == 0 {
            let w = BigComplex::new(float(&mut s, 60, 1), float(&mut s, 60, 1));
            let lo = parts(x.pow_round_parts(&w, (p, p), d));
            let hi = parts(x.pow_round_parts(&w, (p + 80, p + 80), d));
            for j in 0..2 {
                assert_eq!(
                    hi[j].0.prec_round(p, Round::Down).0,
                    lo[j].0,
                    "pow {x:?} {w:?} {p} part {j}"
                );
            }
        }
    }
}

/// The first term of the arcsine's series decides nothing when `x² - 3y²` is below the bound of
/// the rest (`2|z|^4`): for `x = √3·y` rounded up to 1100 bits and `y = 2^-500`, `Re asin z = x(1 +
/// (x² - 3y²)/6 - 1.2y^4 + ...)` lies below `x`, where the first term alone would put it above.
/// Against `Im ln(iz + √(1 - z²))` at 3000 bits.
#[test]
#[cfg_attr(miri, ignore = "3000-bit logarithms and roots: slow under Miri")]
fn tiny_arcsines_past_their_first_term() {
    let n = Round::Nearest;
    let y = BigFloat::from_parts(&Number::from(1), -500, 2, n).0;
    let x = BigFloat::from(3)
        .sqrt_round(1100, Round::Up)
        .0
        .mul_round(&y, 1100, n)
        .0;
    let z = BigComplex::new(x.clone(), y.clone());
    let (a, o) = z.asin_round(1100, Round::Down);
    let one = BigComplex::from(1).with_prec(3000);
    let r = one
        .sub_round(&z.sqr_round(3000, n).0, 3000, n)
        .0
        .sqrt_round(3000, n)
        .0;
    let u = r.add_round(&z.mul_i(false), 3000, n).0;
    let want = u.ln_round(3000, n).0.im().prec_round(1100, Round::Down).0;
    assert_eq!(*a.re(), want);
    assert!(*a.re() < x);
    assert_eq!(o.0, Ordering::Less);
}

/// MPC's signs of the exact zero real part of an imaginary power `(±0 + ai)^n` that is not
/// dyadic (`fix_sign`): `-0` for `|n| = 3 (mod 4)` and `+0` for `|n| = 1 (mod 4)` from `+0`, and
/// the other way from `-0`.
#[test]
fn imaginary_powers_have_mpcs_zero_signs() {
    for (n, minus) in [
        (-3i64, true),
        (-5, false),
        (-7, true),
        (-33, false),
        (3, true),
        (5, false),
    ] {
        for re_negative in [false, true] {
            let re = if re_negative {
                BigFloat::neg_zero(53)
            } else {
                BigFloat::zero(53)
            };
            for a in [3.0f64, -3.0, 0.75] {
                let z = BigComplex::new(re.clone(), BigFloat::from(a));
                let (p, o) = z.powi_round(n, 53, Round::Nearest);
                assert!(p.re().is_zero(), "{n} {a}");
                if n < 0 {
                    assert_eq!(
                        p.re().is_sign_negative(),
                        minus != re_negative,
                        "{n} {a} {re_negative}"
                    );
                }
                assert_eq!(o.0, Ordering::Equal);
            }
        }
    }
}

/// `ln(1 + ti)` for `t = 2^-(2^40)`: `ln|z| = ln(1 + t²)/2 = t²/2 - t^4/4 + ...`, rounded beside
/// `t²/2` from `d = t²` exact; `arg z = atan t`, beside `t`. Ziv's strategy alone would take about
/// `2^41` bits for each.
#[test]
fn logarithms_a_hair_from_one() {
    let n = Round::Nearest;
    let t = BigFloat::from_parts(&Number::from(1), -(1i64 << 40), 53, n).0;
    let z = BigComplex::new(BigFloat::from(1.0), t.clone());
    let (l, o) = z.ln_round(53, n);
    let half_t2 = BigFloat::from_parts(&Number::from(1), -(1i64 << 41) - 1, 53, n).0;
    assert_eq!((l.re().clone(), o.0), (half_t2, Ordering::Greater));
    assert_eq!((l.im().clone(), o.1), (t.clone(), Ordering::Greater));
    let (a, o) = z.arg_round(53, Round::Down);
    assert_eq!(o, Ordering::Less);
    assert!(a < t);
}

/// MPC's values where a part's bound is decisive (found by the mutations of the square root's
/// bound): `asin(-0x1000f·2^-2788 - 2^64 i)` at 19 bits rounded down, whose real part
/// `atan2(x, √((A - |x|)(A + |x|)))` divides by a long root.
#[test]
fn vectors_where_the_bounds_decide() {
    let x = BigFloat::from_parts(&Number::from(-0x1000f), -2788, 17, Round::Nearest).0;
    let y = BigFloat::from_parts(&Number::from(-1), 64, 17, Round::Nearest).0;
    let (a, o) = BigComplex::new(x, y).asin_round_parts((19, 53), (Round::Down, Round::TowardZero));
    let want = BigFloat::from_parts(&Number::from(-0x1000f), -2852, 19, Round::Nearest).0;
    assert_eq!((a.re().clone(), o.0), (want, Ordering::Less));
}

/// Exact powers whose exponents leave every range: `(-0 + 2^-34 i)^w` for `w = -(2^49 + 8191)·2^393`
/// (an even integer, `w/2` even) is `|z|^w` real and positive, beyond the range (MPC: `+∞`); `i^(2^(2^61))
/// = 1` at once.
#[test]
fn powers_beyond_every_range() {
    let n = Round::Nearest;
    let z = BigComplex::new(
        BigFloat::neg_zero(16),
        BigFloat::from_parts(&Number::from(1), -34, 16, n).0,
    );
    let w = BigFloat::from_parts(&Number::from(-0x2000000001fffi64), 393, 50, n).0;
    let (p, o) = z.pow_real_round_parts(&w, (8, 146), (n, n));
    assert!(p.re().is_infinite() && !p.re().is_sign_negative());
    assert_eq!(o.0, Ordering::Greater);
    assert!(p.im().is_zero());
    let big = BigFloat::from_parts(&Number::from(1), 1 << 61, 2, n).0;
    let (p, o) = BigComplex::i(53).pow_real_round(&big, 53, n);
    assert_eq!(*p.re(), 1);
    assert!(p.im().is_zero());
    assert_eq!(o, (Ordering::Equal, Ordering::Equal));
}
