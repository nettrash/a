//! The circular functions and their inverses against MPFR 4.2's results (vectors in
//! `data/trig_*.txt`: random, tiny and huge arguments, near multiples of `π/2` and near `±1`, at
//! precisions from 1 to 10,000 bits, in every mode), their special values, and identities.

use super::common::{MODES, below, float, mode, same, size, xorshift};
use a::math::float::{BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;

/// A value of the vectors: `nan`, `inf`, `-inf`, `0`, `-0` or `[-]0x<hex>p<e>` for `m·2^e`, at the
/// precision of its bits (`prec` for the specials).
fn parse(s: &str, prec: u64) -> BigFloat {
    match s {
        "nan" => return BigFloat::nan(prec),
        "inf" => return BigFloat::infinity(prec),
        "-inf" => return BigFloat::neg_infinity(prec),
        "0" => return BigFloat::zero(prec),
        "-0" => return BigFloat::neg_zero(prec),
        _ => {}
    }
    let (neg, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let rest = rest.strip_prefix("0x").expect("hex");
    let (hex, e) = rest.split_once('p').expect("exponent");
    let mut m = Number::from_str_radix(hex, 16, ByteOrder::LittleEndian).expect("digits");
    if neg {
        m = -m;
    }
    let bits = m.bits().max(1);
    let (x, o) = BigFloat::from_parts(&m, e.parse().expect("exponent"), bits, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    x
}

/// A mode of the vectors.
fn round_of(c: &str) -> Round {
    match c {
        "N" => Round::Nearest,
        "Z" => Round::TowardZero,
        "U" => Round::Up,
        "D" => Round::Down,
        _ => panic!("mode {c}"),
    }
}

/// A ternary of the vectors.
fn ternary(t: &str) -> Ordering {
    match t {
        "-1" => Ordering::Less,
        "0" => Ordering::Equal,
        "1" => Ordering::Greater,
        _ => panic!("ternary {t}"),
    }
}

/// The function of one argument the vectors name.
fn apply(name: &str, x: &BigFloat, prec: u64, round: Round) -> (BigFloat, Ordering) {
    match name {
        "sin" => x.sin_round(prec, round),
        "cos" => x.cos_round(prec, round),
        "tan" => x.tan_round(prec, round),
        "sec" => x.sec_round(prec, round),
        "csc" => x.csc_round(prec, round),
        "cot" => x.cot_round(prec, round),
        "atan" => x.atan_round(prec, round),
        "asin" => x.asin_round(prec, round),
        "acos" => x.acos_round(prec, round),
        _ => panic!("function {name}"),
    }
}

/// Checks every line of a file of vectors, every `step`-th one under Miri.
fn vectors(data: &str, step: usize) {
    let mut n = 0;
    for (i, line) in data.lines().enumerate() {
        if i % step != 0 {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let what = format!("line {}: {line}", i + 1);
        // Under Miri, the short precisions alone.
        let p: u64 = f[if f[0] == "atan2" { 3 } else { 2 }].parse().unwrap();
        if cfg!(miri) && p > 130 {
            continue;
        }
        match f[0] {
            "atan2" => {
                let prec: u64 = f[3].parse().unwrap();
                let (y, x) = (parse(f[1], prec), parse(f[2], prec));
                let got = y.atan2_round(&x, prec, round_of(f[4]));
                let want = (parse(f[5], prec).with_prec(prec), ternary(f[6]));
                same(&got, &want, &what);
            }
            "sincos" => {
                let prec: u64 = f[2].parse().unwrap();
                let x = parse(f[1], prec);
                let (s, c) = x.sin_cos_round(prec, round_of(f[3]));
                let ws = (parse(f[4], prec).with_prec(prec), ternary(f[5]));
                let wc = (parse(f[6], prec).with_prec(prec), ternary(f[7]));
                same(&s, &ws, &what);
                same(&c, &wc, &what);
            }
            name => {
                let prec: u64 = f[2].parse().unwrap();
                let x = parse(f[1], prec);
                let got = apply(name, &x, prec, round_of(f[3]));
                let want = (parse(f[4], prec).with_prec(prec), ternary(f[5]));
                same(&got, &want, &what);
            }
        }
        n += 1;
    }
    assert!(n > 0);
}

#[test]
fn agree_with_mpfr() {
    vectors(include_str!("data/trig_small.txt"), size(1, 23));
}

#[test]
fn agree_with_mpfr_on_hard_cases() {
    // Results within `2^-(p + 22)` of a breakpoint of their precision `p` (24, 30, 53): Ziv's
    // first working precision is too short for most of them.
    vectors(include_str!("data/trig_hard.txt"), size(1, 7));
}

#[test]
fn agree_with_mpfr_at_long_precisions_and_huge_arguments() {
    if cfg!(miri) {
        return;
    }
    vectors(include_str!("data/trig_large.txt"), 1);
}

/// The functions of one argument by name, for the loops.
const ONE: [&str; 9] = [
    "sin", "cos", "tan", "sec", "csc", "cot", "atan", "asin", "acos",
];

/// `±π·num/4` rounded, as the special values have it.
fn pi_part(num: u32, negative: bool, prec: u64, round: Round) -> (BigFloat, Ordering) {
    // `num·π/4` exactly from π rounded in the mode of the magnitude, `num` a power of two.
    let r = match (round, negative) {
        (Round::Up, true) => Round::Down,
        (Round::Down, true) => Round::Up,
        (r, _) => r,
    };
    let (p, o) = BigFloat::pi_round(prec, r);
    let scale = BigFloat::from(num) / BigFloat::from(4);
    let v = p.mul_round(&scale, prec, Round::Nearest).0;
    if negative { (-v, o.reverse()) } else { (v, o) }
}

#[test]
fn special_values_are_mpfrs() {
    let one = |p| BigFloat::from(1).with_prec(p);
    for round in MODES {
        let p = 17;
        for name in ONE {
            let got = apply(name, &BigFloat::nan(53), p, round);
            assert!(
                got.0.is_nan() && got.0.prec() == p && got.1 == Ordering::Equal,
                "{name}(NaN)"
            );
            for inf in [BigFloat::infinity(53), BigFloat::neg_infinity(53)] {
                let got = apply(name, &inf, p, round);
                if name == "atan" {
                    same(
                        &got,
                        &pi_part(2, inf.is_sign_negative(), p, round),
                        "atan(±∞)",
                    );
                } else {
                    assert!(
                        got.0.is_nan() && got.1 == Ordering::Equal,
                        "{name}({inf:?})"
                    );
                }
            }
            for zero in [BigFloat::zero(53), BigFloat::neg_zero(53)] {
                let neg = zero.is_sign_negative();
                let got = apply(name, &zero, p, round);
                let want = match name {
                    "sin" | "tan" | "atan" | "asin" => (zero.with_prec(p), Ordering::Equal),
                    "cos" | "sec" => (one(p), Ordering::Equal),
                    "csc" | "cot" => (
                        if neg {
                            BigFloat::neg_infinity(p)
                        } else {
                            BigFloat::infinity(p)
                        },
                        Ordering::Equal,
                    ),
                    _ => pi_part(2, false, p, round),
                };
                same(&got, &want, &format!("{name}({zero:?})"));
            }
        }
        // `asin` and `acos` at ±1 and beyond.
        same(
            &BigFloat::from(1).asin_round(p, round),
            &pi_part(2, false, p, round),
            "asin 1",
        );
        same(
            &BigFloat::from(-1).asin_round(p, round),
            &pi_part(2, true, p, round),
            "asin -1",
        );
        same(
            &BigFloat::from(1).acos_round(p, round),
            &(BigFloat::zero(p), Ordering::Equal),
            "acos 1",
        );
        same(
            &BigFloat::from(-1).acos_round(p, round),
            &pi_part(4, false, p, round),
            "acos -1",
        );
        for x in [1.0000001f64, -1.5, 2.0, -1e300] {
            assert!(BigFloat::from(x).asin_round(p, round).0.is_nan());
            assert!(BigFloat::from(x).acos_round(p, round).0.is_nan());
        }
        // `sin_cos` at the specials.
        let ((s, os), (c, oc)) = BigFloat::neg_zero(9).sin_cos_round(p, round);
        assert!(s.is_zero() && s.is_sign_negative() && os == Ordering::Equal);
        assert!(c == one(p) && oc == Ordering::Equal);
        let ((s, _), (c, _)) = BigFloat::infinity(9).sin_cos_round(p, round);
        assert!(s.is_nan() && c.is_nan());
    }
}

#[test]
fn atan2_has_the_whole_table_of_annex_f() {
    let p = 20;
    let (pz, nz) = (BigFloat::zero(10), BigFloat::neg_zero(10));
    let (pi, ni) = (BigFloat::infinity(10), BigFloat::neg_infinity(10));
    let (pf, nf) = (BigFloat::from(3), BigFloat::from(-3));
    let nan = BigFloat::nan(10);
    for round in MODES {
        let z = |neg: bool| {
            (
                if neg {
                    BigFloat::neg_zero(p)
                } else {
                    BigFloat::zero(p)
                },
                Ordering::Equal,
            )
        };
        // (y, x, want): 0 for ±0, n for ±nπ/4 with the sign of y.
        let cases: [(&BigFloat, &BigFloat, u32); 22] = [
            (&pz, &pz, 0),
            (&nz, &pz, 0),
            (&pz, &nz, 4),
            (&nz, &nz, 4),
            (&pz, &pf, 0),
            (&nz, &pf, 0),
            (&pz, &nf, 4),
            (&nz, &nf, 4),
            (&pz, &pi, 0),
            (&nz, &pi, 0),
            (&pz, &ni, 4),
            (&nz, &ni, 4),
            (&pf, &pz, 2),
            (&nf, &nz, 2),
            (&pi, &pi, 1),
            (&ni, &pi, 1),
            (&pi, &ni, 3),
            (&ni, &ni, 3),
            (&pi, &pf, 2),
            (&ni, &nf, 2),
            (&pf, &pi, 0),
            (&nf, &ni, 4),
        ];
        for (y, x, n) in cases {
            let got = y.atan2_round(x, p, round);
            let neg = y.is_sign_negative();
            let want = match n {
                0 => z(neg),
                3 => {
                    // `3π/4` by Ziv: compared with the rounding of a long `3π/4`.
                    let long = BigFloat::pi(400)
                        .mul_round(&BigFloat::from(0.75), 400, Round::Nearest)
                        .0;
                    let long = if neg { -long } else { long };
                    long.prec_round(p, round)
                }
                _ => pi_part(n, neg, p, round),
            };
            same(&got, &want, &format!("atan2({y:?}, {x:?}) {round:?}"));
        }
        for (y, x) in [(&nan, &pf), (&pf, &nan), (&nan, &nan), (&pz, &nan)] {
            assert!(y.atan2_round(x, p, round).0.is_nan());
        }
    }
}

/// The next number above `x` at its precision.
fn next_up(x: &BigFloat) -> BigFloat {
    let p = x.prec();
    let (m, e) = x.mantissa_exponent().expect("finite");
    let one = Number::from(1);
    // Toward zero from the bottom of a binade below zero, the ulp halves.
    let bottom = m.is_negative() && m.abs().trailing_zeros() == Some(p - 1);
    let (m, e) = if bottom {
        (&(&m << 1usize) + &one, e - 1)
    } else {
        (&m + &one, e)
    };
    let (y, o) = BigFloat::from_parts(&m, e, p, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    y
}

/// Checks the four roundings of one value against each other: below and above it to the
/// neighbouring numbers, as the value is never exact, nearest one of them and toward zero the
/// one of the smaller magnitude.
fn modes_agree(f: impl Fn(u64, Round) -> (BigFloat, Ordering), prec: u64, what: &str) {
    let (d, od) = f(prec, Round::Down);
    let (u, ou) = f(prec, Round::Up);
    let (n, on) = f(prec, Round::Nearest);
    let (z, oz) = f(prec, Round::TowardZero);
    if !d.is_normal() || !u.is_normal() {
        return;
    }
    assert_eq!(od, Ordering::Less, "{what}: down");
    assert_eq!(ou, Ordering::Greater, "{what}: up");
    assert_eq!(
        next_up(&d),
        u,
        "{what}: down {d:?} and up {u:?} are neighbours"
    );
    assert!(
        n == d && on == Ordering::Less || n == u && on == Ordering::Greater,
        "{what}: nearest"
    );
    let zd = !d.is_sign_negative();
    assert!(
        if zd {
            z == d && oz == od
        } else {
            z == u && oz == ou
        },
        "{what}: toward zero"
    );
}

#[test]
fn the_four_modes_bracket_the_value() {
    let mut s = 0x7216_0000_5EED_0001u64;
    for i in 0..size(400, 6) {
        let x = float(&mut s, 200, if i % 5 == 0 { 300 } else { 4 });
        let prec = 1 + below(&mut s, if i % 7 == 0 { 600 } else { 130 });
        let name = ONE[i % ONE.len()];
        let x = if matches!(name, "asin" | "acos") {
            // Into `(-1, 1)`.
            let e = x.exponent().unwrap();
            let shift = BigFloat::from_parts(&Number::from(1), -e.max(0) - 1, 1, Round::Nearest).0;
            x.mul_round(&shift, x.prec(), Round::Nearest).0
        } else {
            x
        };
        modes_agree(
            |p, r| apply(name, &x, p, r),
            prec,
            &format!("{name}({x:?}) at {prec}"),
        );
        let y = float(&mut s, 100, 6);
        modes_agree(
            |p, r| x.atan2_round(&y, p, r),
            prec,
            &format!("atan2({x:?}, {y:?})"),
        );
    }
}

#[test]
fn a_rounding_down_or_up_agrees_with_one_longer() {
    // Rounding down (up) at `p` is rounding down (up) at `p + 64` and again at `p`, for a value
    // that is never exact: the long results go through other working precisions.
    let mut s = 0x7216_0000_5EED_0002u64;
    for i in 0..size(300, 4) {
        let exps = match i % 6 {
            0 => 200,
            1 => 2,
            _ => 8,
        };
        let x = float(&mut s, 400, exps);
        let prec = 1 + below(&mut s, 300);
        let name = ONE[i % ONE.len()];
        if matches!(name, "asin" | "acos") && x.exponent().unwrap() > 0 {
            continue;
        }
        for round in [Round::Down, Round::Up] {
            let short = apply(name, &x, prec, round);
            let long = apply(name, &x, prec + 64, round);
            let again = long.0.prec_round(prec, round);
            if short.0.is_nan() {
                continue;
            }
            assert_eq!(short.0, again.0, "{name}({x:?}) at {prec} {round:?}");
            assert_eq!(short.1, long.1, "{name}({x:?}) at {prec} {round:?}");
        }
    }
}

#[test]
fn sin_cos_gives_both_as_each_alone() {
    let mut s = 0x7216_0000_5EED_0003u64;
    for i in 0..size(300, 4) {
        let x = float(&mut s, 300, if i % 4 == 0 { 400 } else { 6 });
        let prec = 1 + below(&mut s, 400);
        let round = mode(&mut s);
        let (a, b) = x.sin_cos_round(prec, round);
        same(&a, &x.sin_round(prec, round), "sin_cos's sine");
        same(&b, &x.cos_round(prec, round), "sin_cos's cosine");
    }
}

#[test]
fn odd_and_even_functions_and_identities() {
    let mut s = 0x7216_0000_5EED_0004u64;
    for i in 0..size(200, 3) {
        let x = float(&mut s, 200, if i % 3 == 0 { 100 } else { 3 });
        let prec = 2 + below(&mut s, 250);
        let round = mode(&mut s);
        let flip = match round {
            Round::Up => Round::Down,
            Round::Down => Round::Up,
            r => r,
        };
        let neg = -x.clone();
        for name in ["sin", "tan", "csc", "cot", "atan"] {
            let a = apply(name, &x, prec, round);
            let b = apply(name, &neg, prec, flip);
            same(&(-b.0, b.1.reverse()), &a, &format!("{name} is odd"));
        }
        for name in ["cos", "sec"] {
            same(
                &apply(name, &x, prec, round),
                &apply(name, &neg, prec, round),
                "even",
            );
        }
        // sin² + cos² = 1 and tan = sin/cos, within a few ulps at 2p bits.
        let w = 2 * prec + 20;
        let (sn, cs) = x.sin_cos();
        let _ = (sn, cs);
        let (sn, cs) = (
            x.sin_round(w, Round::Nearest).0,
            x.cos_round(w, Round::Nearest).0,
        );
        let one = sn.mul_add_round(&sn, &(&cs * &cs), w, Round::Nearest).0;
        let d = one.sub_round(&BigFloat::from(1), w, Round::Nearest).0;
        assert!(
            d.is_zero() || d.exponent().unwrap() < 4 - w as i64,
            "sin² + cos² at {x:?}"
        );
        let t = x.tan_round(w, Round::Nearest).0;
        let q = sn.div_round(&cs, w, Round::Nearest).0;
        let d = t.sub_round(&q, w, Round::Nearest).0;
        assert!(
            d.is_zero() || d.exponent().unwrap() < t.exponent().unwrap() + 5 - w as i64,
            "tan = sin/cos at {x:?}"
        );
        // atan2(y, x) = atan(y/x) for x > 0 with y/x exact: y = x·2^k.
        let ax = x.abs();
        let y = ax
            .mul_round(&BigFloat::from(0.125), ax.prec(), Round::Nearest)
            .0;
        same(
            &y.atan2_round(&ax, prec, round),
            &BigFloat::from(0.125).atan_round(prec, round),
            "atan2(x/8, x)",
        );
        let _ = xorshift(&mut s);
    }
}

#[test]
fn underflow_and_overflow_go_by_the_mode() {
    let p = 24;
    let tiny = BigFloat::from_parts(&Number::from(1), BigFloat::EXP_MIN - 1, p, Round::Nearest).0;
    assert_eq!(tiny.exponent(), Some(BigFloat::EXP_MIN));
    // sin x is just below the smallest number, tan x and asin x just above, atan x below.
    let (s, o) = tiny.sin_round(p, Round::Nearest);
    assert!(s == tiny && o == Ordering::Greater);
    let (s, o) = tiny.sin_round(p, Round::TowardZero);
    assert!(s.is_zero() && o == Ordering::Less);
    let (s, o) = tiny.sin_round(p, Round::Up);
    assert!(s == tiny && o == Ordering::Greater);
    let (t, o) = tiny.tan_round(p, Round::Down);
    assert!(t == tiny && o == Ordering::Less);
    let (t, o) = tiny.tan_round(p, Round::Up);
    assert!(t == next_up(&tiny) && o == Ordering::Greater);
    let (a, o) = (-tiny.clone()).atan_round(p, Round::Up);
    assert!(a.is_zero() && a.is_sign_negative() && o == Ordering::Greater);
    let (a, o) = tiny.asin_round(p, Round::Nearest);
    assert!(a == tiny && o == Ordering::Less);
    // csc of the smallest number is 2^(EXP_MAX + 1) and more: beyond the range.
    let (c, o) = tiny.csc_round(p, Round::Nearest);
    assert!(c.is_infinite() && !c.is_sign_negative() && o == Ordering::Greater);
    let (c, o) = tiny.csc_round(p, Round::TowardZero);
    assert!(c.exponent() == Some(BigFloat::EXP_MAX) && o == Ordering::Less);
    let (c, o) = (-tiny.clone()).cot_round(p, Round::Up);
    assert!(
        c.exponent() == Some(BigFloat::EXP_MAX) && c.is_sign_negative() && o == Ordering::Greater
    );
}

#[test]
fn huge_and_tiny_arguments() {
    // sin of 2^k for growing k against a much longer computation of itself: the reduction to
    // the bits that the exponent needs.
    let ks: &[i64] = if cfg!(miri) {
        &[62, 64, 200]
    } else {
        &[62, 63, 64, 200, 1000, 5000]
    };
    for &k in ks {
        let x = BigFloat::from_parts(&Number::from(1), k, 1, Round::Nearest).0;
        for round in MODES {
            for (name, p) in [("sin", 30u64), ("cos", 64), ("tan", 100)] {
                let short = apply(name, &x, p, round);
                let long = apply(name, &x, p + 200, Round::Nearest);
                // The long result is within half an ulp at p + 200 bits; its rounding at p is
                // the short one unless it lies within that of a breakpoint (not for these).
                assert_eq!(
                    short.0,
                    long.0.prec_round(p, round).0,
                    "{name}(2^{k}) {round:?}"
                );
            }
        }
    }
    // Tiny arguments: x, 1 or 1/x moved by less than half an ulp, by the side the function is on.
    let x = BigFloat::from_parts(&Number::from(3), -5000, 2, Round::Nearest).0;
    let (s, o) = x.sin_round(100, Round::Down);
    assert!(s < x && o == Ordering::Less && next_up(&s) == x.with_prec(100));
    let (s, o) = x.sin_round(100, Round::Nearest);
    assert!(s == x && o == Ordering::Greater);
    let (c, o) = x.cos_round(100, Round::Up);
    assert!(c == 1 && o == Ordering::Greater);
    let (c, o) = x.cos_round(100, Round::Down);
    assert!(c < 1 && o == Ordering::Less);
    let (t, o) = x.tan_round(100, Round::Up);
    assert!(t > x && o == Ordering::Greater);
    let (a, o) = x.asin_round(100, Round::Nearest);
    assert!(a == x && o == Ordering::Less);
    let h = BigFloat::from_parts(&Number::from(1), -3000, 1, Round::Nearest).0;
    let (c, o) = h.csc_round(64, Round::Up);
    assert!(
        c > BigFloat::from_parts(&Number::from(1), 3000, 1, Round::Nearest).0
            && o == Ordering::Greater
    );
    let (c, o) = h.cot_round(64, Round::Nearest);
    assert!(
        c == BigFloat::from_parts(&Number::from(1), 3000, 1, Round::Nearest).0
            && o == Ordering::Greater
    );
}

#[test]
fn every_kernel_agrees_with_the_others() {
    // The short kernel (to 512 bits after the point), the series with the doublings, and binary
    // splitting (from about 410,000 bits): a rounding down at one precision is the rounding down
    // of one at another, whichever kernels they took.
    let x = BigFloat::from(0.7)
        .add_round(&BigFloat::from(1e-17), 2000, Round::Nearest)
        .0;
    let pairs: &[(u64, u64)] = if cfg!(miri) {
        &[(60, 500)]
    } else if cfg!(debug_assertions) {
        &[(60, 500), (300, 2000), (1500, 9000)]
    } else {
        &[(60, 500), (300, 2000), (1500, 9000), (400_000, 420_000)]
    };
    for &(short, long) in pairs {
        for name in ["sin", "cos", "atan"] {
            for round in [Round::Down, Round::Up] {
                let a = apply(name, &x, short, round);
                let b = apply(name, &x, long, round).0.prec_round(short, round);
                assert_eq!(a.0, b.0, "{name} at {short} and {long} bits");
            }
        }
    }
}
