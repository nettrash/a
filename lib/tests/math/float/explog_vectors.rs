//! The vectors of the exponentials, logarithms, powers and hyperbolic functions: lines `f prec
//! mode x y result ternary` that MPFR 4.2 computed (`y` is `-` but for `pow`), the values in
//! hexadecimal `±0x<m>p<e>` with `m` odd, or `nan`, `inf`, `-inf`, `0`, `-0`.

use super::common::{same, size};
use a::math::float::{BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;

/// The value of a hexadecimal float of the vectors, exactly.
pub(super) fn parse(s: &str) -> BigFloat {
    match s {
        "nan" => return BigFloat::nan(1),
        "inf" => return BigFloat::infinity(1),
        "-inf" => return BigFloat::neg_infinity(1),
        "0" => return BigFloat::zero(1),
        "-0" => return BigFloat::neg_zero(1),
        _ => {}
    }
    let (neg, body) = match s.strip_prefix('-') {
        Some(b) => (true, b),
        None => (false, s),
    };
    let body = body.strip_prefix("0x").expect("0x");
    let (m, e) = body.split_once('p').expect("p");
    let mut m = Number::from_str_radix(m, 16, ByteOrder::LittleEndian).expect("hex");
    if neg {
        m = -m;
    }
    let e: i64 = e.parse().expect("exponent");
    let bits = m.bits().max(1);
    let (x, o) = BigFloat::from_parts(&m, e, bits, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    x
}

/// Checks every line of `data` with `f(x, y, prec, mode)`, every `size(1, 61)`th line under
/// Miri and none beyond `size(u64::MAX, 130)` bits there.
pub(super) fn check(
    data: &str,
    f: impl Fn(&str, &BigFloat, &BigFloat, u64, Round) -> (BigFloat, Ordering),
) {
    let step = size(1, 61);
    let max = size(usize::MAX, 130) as u64;
    let mut n = 0;
    for (i, line) in data.lines().enumerate() {
        if i % step != 0 || line.is_empty() {
            continue;
        }
        let p: Vec<&str> = line.split(' ').collect();
        assert_eq!(p.len(), 7, "{line}");
        let prec: u64 = p[1].parse().expect("prec");
        if prec > max {
            continue;
        }
        let round = match p[2] {
            "N" => Round::Nearest,
            "Z" => Round::TowardZero,
            "U" => Round::Up,
            "D" => Round::Down,
            m => panic!("mode {m}"),
        };
        let x = parse(p[3]);
        let y = if p[4] == "-" {
            BigFloat::zero(1)
        } else {
            parse(p[4])
        };
        let want = parse(p[5]).prec_round(prec, Round::Nearest).0;
        let t: i32 = p[6].parse().expect("ternary");
        let got = f(p[0], &x, &y, prec, round);
        same(&got, &(want, t.cmp(&0)), line);
        n += 1;
    }
    assert!(n > 0);
}

/// Checks that the four modes of a result agree: all exact and equal, or `Down` below with
/// `Less`, `Up` above with `Greater`, to nearest one of the two, toward zero the one nearer zero.
#[track_caller]
pub(super) fn modes(f: impl Fn(Round) -> (BigFloat, Ordering), what: &str) {
    let n = f(Round::Nearest);
    let z = f(Round::TowardZero);
    let u = f(Round::Up);
    let d = f(Round::Down);
    if n.0.is_nan() {
        assert!(
            z.0.is_nan() && u.0.is_nan() && d.0.is_nan(),
            "{what}: NaN in one mode only"
        );
        return;
    }
    if d.1 == Ordering::Equal {
        for r in [&n, &z, &u] {
            same(r, &d, what);
        }
        return;
    }
    assert_eq!(d.1, Ordering::Less, "{what}: down");
    assert_eq!(u.1, Ordering::Greater, "{what}: up");
    assert!(
        d.0 < u.0 || d.0.is_infinite() || u.0.is_infinite() || d.0 == u.0,
        "{what}: down {:?} up {:?}",
        d.0,
        u.0
    );
    assert!(same_value(&n, &d) || same_value(&n, &u), "{what}: nearest");
    // A negative value has its rounding down below zero.
    let toward = if d.0.is_sign_negative() && !d.0.is_zero() {
        &u
    } else {
        &d
    };
    assert!(same_value(&z, toward), "{what}: toward zero {:?}", z.0);
}

/// The same value and direction.
fn same_value(a: &(BigFloat, Ordering), b: &(BigFloat, Ordering)) -> bool {
    a.1 == b.1 && a.0 == b.0 && a.0.is_sign_negative() == b.0.is_sign_negative()
}

/// The rounding to `prec` bits in the mode `round` of a value known as `hi` (at many more bits)
/// with the direction `t` of its own rounding, and the direction: `None` when `hi` is a midpoint
/// of `prec` bits that was itself rounded, where rounding twice may differ from once.
pub(super) fn reround(
    hi: &(BigFloat, Ordering),
    prec: u64,
    round: Round,
) -> Option<(BigFloat, Ordering)> {
    let (r, t2) = hi.0.prec_round(prec, round);
    if t2 == Ordering::Equal {
        return Some((r, hi.1));
    }
    if hi.1 != Ordering::Equal && round == Round::Nearest {
        // A midpoint of `prec` bits has `prec + 1` significant bits.
        let (m, _) = hi.0.mantissa_exponent()?;
        let m = m.abs();
        let tz = (0..m.bits()).take_while(|&i| !m.bit(i)).count() as u64;
        if m.bits() - tz == prec + 1 {
            return None;
        }
    }
    Some((r, t2))
}

/// Checks `f` near the threshold of its rounding near the argument: arguments of `prec + 30`
/// bits with exponents `e` in `lo(prec)..=hi(prec)`, every mode, against `f` at `prec + 120`
/// bits rounded again.
pub(super) fn near_threshold(
    f: impl Fn(&BigFloat, u64, Round) -> (BigFloat, Ordering),
    range: impl Fn(u64) -> (i64, i64),
    negative_too: bool,
    what: &str,
) {
    let mut s = 0x7EA5_0001u64;
    let precs: &[u64] = if cfg!(miri) {
        &[24]
    } else {
        &[24, 53, 64, 113]
    };
    for &p in precs {
        let (lo, hi) = range(p);
        for e in (lo..=hi).step_by(size(1, 6)) {
            for k in 0..size(3, 1) {
                let mut m = super::common::mantissa(&mut s, p + 30);
                if negative_too && k % 2 == 1 {
                    m = -m;
                }
                let x = BigFloat::from_parts(&m, e - (p + 30) as i64, p + 30, Round::Nearest).0;
                let wide = f(&x, p + 120, Round::Nearest);
                for r in super::common::MODES {
                    if let Some(want) = reround(&wide, p, r) {
                        same(&f(&x, p, r), &want, &format!("{what}({x:?}) at {p} {r:?}"));
                    }
                }
            }
        }
    }
}
