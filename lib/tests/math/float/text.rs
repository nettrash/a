//! Floats as text and back: the digits against MPFR's `mpfr_get_str` and the reading against
//! `mpfr_strtofr`, in every mode and several radices, with halfway decimal strings and huge
//! exponents; `Display` and the exponent forms against `f64`'s and as the shortest text that
//! reads back; `FromStr` against `f64`'s and the exact rationals.

use super::common::{MODES, below, float, mode, rational, reference, same, size, xorshift};
use super::vectors::{self, lines, ternary};
use a::math::float::{BigFloat, BigFloatErrorKind, Round};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;
use core::cmp::Ordering;

/// The rational `0.digits·radix^exp`.
fn digits_value(digits: &str, radix: u32, exp: i64) -> Rational {
    let d = Number::from_str_radix(digits, radix, ByteOrder::LittleEndian).unwrap();
    let e = exp - digits.len() as i64;
    Rational::from_integer(d) * Rational::from(radix).pow(e as i32)
}

#[test]
fn digits_match_mpfr() {
    let mut n = 0;
    for v in lines(include_str!("data/convert_getstr.txt")) {
        let prec: u64 = v[1].parse().unwrap();
        let x = vectors::float(v[0], prec);
        let radix: u32 = v[2].parse().unwrap();
        let count: usize = v[3].parse().unwrap();
        let round = vectors::mode(v[4]);
        let want_exp: i64 = v[6].parse().unwrap();
        let digits = if count == 0 { None } else { Some(count) };
        let (d, e, o) = x.to_digits(radix, digits, round).unwrap();
        assert_eq!(
            (d.as_str(), e),
            (v[5], want_exp),
            "{x:?} radix {radix} {digits:?} {round:?}"
        );
        // The direction, from the exact value of the digits where it can be had.
        if e.abs() < 3000 {
            let got = digits_value(&d, radix, e);
            let q = rational(&x);
            let magnitude = got.partial_cmp(&q.abs()).unwrap();
            let want = if x.is_sign_negative() {
                magnitude.reverse()
            } else {
                magnitude
            };
            assert_eq!(o, want, "{x:?} radix {radix} {digits:?} {round:?}");
        }
        // The text of the same digits.
        let t = x.to_string_radix(radix, digits, round);
        let sign = if x.is_sign_negative() { "-" } else { "" };
        let tail = if d.len() > 1 {
            format!(".{}", &d[1..])
        } else {
            String::new()
        };
        let marker = if radix <= 10 { 'e' } else { '@' };
        assert_eq!(t, format!("{sign}{}{tail}{marker}{}", &d[..1], e - 1));
        n += 1;
    }
    assert!(n > size(400, 8));
}

#[test]
fn reading_matches_mpfr() {
    let mut n = 0;
    for v in lines(include_str!("data/convert_strtofr.txt")) {
        let radix: u32 = v[0].parse().unwrap();
        let prec: u64 = v[1].parse().unwrap();
        let round = vectors::mode(v[2]);
        let want = (vectors::float(v[4], prec), ternary(v[5]));
        let got = BigFloat::from_str_radix(v[3], radix, prec, round).unwrap();
        same(
            &got,
            &want,
            &format!("{} radix {radix} {prec} {round:?}", v[3]),
        );
        n += 1;
    }
    assert!(n > size(700, 15));
}

#[test]
fn reading_rounds_the_exact_rationals() {
    let mut s = 0x5EED_7E47_0000_0001u64;
    for _ in 0..size(1500, 20) {
        // A decimal of up to 60 digits and an exponent of up to 300.
        let nd = 1 + below(&mut s, 60) as usize;
        let mut text = String::new();
        if xorshift(&mut s) & 1 == 1 {
            text.push('-');
        }
        let point = below(&mut s, nd as u64 + 1) as usize;
        for i in 0..nd {
            if i == point {
                text.push('.');
            }
            text.push((b'0' + below(&mut s, 10) as u8) as char);
        }
        let e = below(&mut s, 600) as i64 - 300;
        text.push_str(&format!("e{e}"));
        let q: Rational = text.parse().unwrap();
        let prec = 1 + below(&mut s, 200);
        let round = mode(&mut s);
        let got = BigFloat::from_str_radix(&text, 10, prec, round).unwrap();
        if q.is_zero() {
            assert!(got.0.is_zero() && got.1 == Ordering::Equal);
            assert_eq!(got.0.is_sign_negative(), text.starts_with('-'));
        } else {
            same(&got, &reference(&q, prec, round), &text);
        }
        // Against f64's reading, to nearest at 53 bits, in the range of the normal numbers.
        let f: f64 = text.parse().unwrap();
        let (x, _) = BigFloat::from_str_radix(&text, 10, 53, Round::Nearest).unwrap();
        if f.is_normal() {
            assert_eq!(x, f, "{text}");
        }
    }
}

#[test]
fn display_is_the_shortest_that_reads_back() {
    let mut s = 0x5EED_7E47_0000_0002u64;
    for _ in 0..size(1500, 15) {
        let x = float(&mut s, 300, 2000);
        let p = x.prec();
        let text = x.to_string();
        let (back, _) = BigFloat::from_str_radix(&text, 10, p, Round::Nearest).unwrap();
        assert_eq!(back, x, "{x:?} as {text}");
        let e = format!("{x:e}");
        let (back, _) = BigFloat::from_str_radix(&e, 10, p, Round::Nearest).unwrap();
        assert_eq!(back, x, "{x:?} as {e}");
        // No decimal of one digit fewer reads back.
        let mantissa = e.split('e').next().unwrap().trim_start_matches('-');
        let n = mantissa.chars().filter(char::is_ascii_digit).count();
        if n > 1 {
            for round in [Round::Down, Round::Up] {
                let t = x.to_string_radix(10, Some(n - 1), round);
                let (y, _) = BigFloat::from_str_radix(&t, 10, p, Round::Nearest).unwrap();
                assert!(y != x, "{x:?}: {t} is shorter than {e}");
            }
        }
    }
}

#[test]
fn display_matches_f64() {
    let mut s = 0x5EED_7E47_0000_0003u64;
    for _ in 0..size(4000, 40) {
        let d = f64::from_bits(xorshift(&mut s));
        if !d.is_finite() || d.abs() < f64::MIN_POSITIVE * 2.0 {
            continue;
        }
        let x = BigFloat::from(d);
        assert_eq!(format!("{x:e}"), format!("{d:e}"));
        assert_eq!(format!("{x:E}"), format!("{d:E}"));
        let e = d.abs().log10().floor();
        if (-5.0..21.0).contains(&e) {
            assert_eq!(x.to_string(), d.to_string());
        }
        let frac = below(&mut s, 25) as usize;
        assert_eq!(format!("{x:.frac$e}"), format!("{d:.frac$e}"));
        if e < 21.0 {
            assert_eq!(format!("{x:.frac$}"), format!("{d:.frac$}"), "{d:e}");
        }
        let width = below(&mut s, 30) as usize;
        if e < 21.0 {
            assert_eq!(
                format!("{x:+0width$.3}"),
                format!("{d:+0width$.3}"),
                "{d:e}"
            );
        }
        assert_eq!(format!("{x:>width$e}"), format!("{d:>width$e}"));
        if (-5.0..21.0).contains(&e) {
            assert_eq!(format!("{x:*^width$}"), format!("{d:*^width$}"));
        }
    }
    for d in [
        0.0,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1.0,
        0.1,
        100.0,
        1e20,
        1.5e-5,
    ] {
        let x = BigFloat::from(d);
        for t in [
            (format!("{x}"), format!("{d}")),
            (format!("{x:e}"), format!("{d:e}")),
            (format!("{x:+08.2}"), format!("{d:+08.2}")),
            (format!("{x:<7}|"), format!("{d:<7}|")),
            (format!("{x:.0}"), format!("{d:.0}")),
        ] {
            assert_eq!(t.0, t.1, "{d:?}");
        }
    }
    // Scientific beyond 10^21 and below 10^-5, positional with the digits there are.
    assert_eq!(BigFloat::from(1e21).to_string(), "1e21");
    assert_eq!(BigFloat::from(1.25e-6).to_string(), "1.25e-6");
    assert_eq!(BigFloat::from(0.00001).to_string(), "0.00001");
    let big = BigFloat::from(u128::MAX);
    assert_eq!(big.to_string(), u128::MAX.to_string());
    let third = BigFloat::from(1)
        .div_round(&BigFloat::from(3), 100, Round::Nearest)
        .0;
    assert_eq!(third.to_string(), "0.3333333333333333333333333333335");
    assert_eq!(
        format!("{third:.40}"),
        "0.3333333333333333333333333333334648101509"
    );
}

#[test]
fn text_round_trips_in_every_radix() {
    let mut s = 0x5EED_7E47_0000_0004u64;
    for _ in 0..size(1500, 15) {
        let x = float(&mut s, 400, 3000);
        let radix = 2 + below(&mut s, 35) as u32;
        let t = x.to_string_radix(radix, None, Round::Nearest);
        let (y, _) = BigFloat::from_str_radix(&t, radix, x.prec(), Round::Nearest).unwrap();
        assert_eq!(y, x, "{x:?} radix {radix}: {t}");
        // Any mode: the digits bracket the number as the direction says.
        let round = mode(&mut s);
        let n = 1 + below(&mut s, 30) as usize;
        let (d, e, o) = x.to_digits(radix, Some(n), round).unwrap();
        let v = digits_value(&d, radix, e);
        let v = if x.is_sign_negative() { -v } else { v };
        assert_eq!(
            v.partial_cmp(&rational(&x)),
            Some(o),
            "{x:?} {radix} {n} {round:?}"
        );
        let (lo, hi) = match round {
            Round::Nearest => (o, o),
            Round::TowardZero if x.is_sign_negative() => (Ordering::Greater, Ordering::Equal),
            Round::TowardZero | Round::Down => (Ordering::Less, Ordering::Equal),
            Round::Up => (Ordering::Greater, Ordering::Equal),
        };
        assert!(o == lo || o == hi, "{x:?} {round:?} {o:?}");
    }
    for round in MODES {
        assert_eq!(BigFloat::nan(5).to_string_radix(10, None, round), "nan");
        assert_eq!(
            BigFloat::neg_infinity(5).to_string_radix(36, None, round),
            "-@inf@"
        );
        assert_eq!(
            BigFloat::neg_zero(5).to_string_radix(7, Some(3), round),
            "-0"
        );
        assert_eq!(BigFloat::nan(5).to_digits(10, None, round), None);
    }
    for (t, radix) in [
        ("nan", 16),
        ("-inf", 16),
        ("-0", 36),
        ("@nan@", 36),
        ("-@inf@", 36),
    ] {
        let (x, o) = BigFloat::from_str_radix(t, radix, 7, Round::Up).unwrap();
        assert_eq!(o, Ordering::Equal);
        assert_eq!(x.to_string_radix(radix, None, Round::Up), t);
    }
    // In radix 36, `inf` is a number.
    let (x, _) = BigFloat::from_str_radix("-inf", 36, 20, Round::Up).unwrap();
    assert_eq!(x, -(18 * 36 * 36 + 23 * 36 + 15));
}

#[test]
fn huge_exponents_and_long_texts() {
    for round in MODES {
        let (x, o) = BigFloat::from_str_radix("1e1000000000000000000000", 10, 64, round).unwrap();
        let overflow = matches!(round, Round::Nearest | Round::Up);
        assert_eq!(x.is_infinite(), overflow, "{round:?}");
        assert_eq!(
            o,
            if overflow {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        );
        let (x, o) = BigFloat::from_str_radix("-1e-1000000000000000000000", 10, 64, round).unwrap();
        assert_eq!(x.is_zero(), round != Round::Down, "{round:?}");
        assert!(x.is_sign_negative());
        assert_eq!(
            o,
            if round == Round::Down {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        );
        // Just inside the range: 2^(EXP_MAX - 1) and its neighbours in decimal.
        let (x, _) = BigFloat::from_str_radix("1e1388255822130839000", 10, 30, round).unwrap();
        let e = x.exponent().unwrap();
        assert!(e < BigFloat::EXP_MAX && e > BigFloat::EXP_MAX - 1000, "{e}");
        let (y, o) = BigFloat::from_str_radix("1e1388255822130840000", 10, 30, round).unwrap();
        assert_eq!(y.is_infinite(), overflow, "{round:?}");
        assert_eq!(
            o,
            if overflow {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        );
        // Digits beyond any precision: a long string with a digit far away.
        let mut t = String::from("0.1");
        t.extend(core::iter::repeat_n('0', size(20000, 200)));
        t.push('1');
        let (x, o) = BigFloat::from_str_radix(&t, 10, 53, round).unwrap();
        let (y, _) = BigFloat::from_str_radix("0.1", 10, 53, round).unwrap();
        // 0.1 to 53 bits rounds up to nearest; the tail moves no rounding but the direction's.
        assert_eq!(x, y, "{round:?}");
        assert_eq!(
            o,
            if matches!(round, Round::Nearest | Round::Up) {
                Ordering::Greater
            } else {
                Ordering::Less
            }
        );
    }
    // A halfway decimal of 1 + 2^-53 and the digits after it decide.
    let half = "1.00000000000000011102230246251565404236316680908203125";
    let (x, o) = BigFloat::from_str_radix(half, 10, 53, Round::Nearest).unwrap();
    assert_eq!((x, o), (BigFloat::from(1.0), Ordering::Less));
    let (x, o) = BigFloat::from_str_radix(
        &format!("{half}{}1", "0".repeat(size(5000, 100))),
        10,
        53,
        Round::Nearest,
    )
    .unwrap();
    assert_eq!(
        (x, o),
        (BigFloat::from(1.0 + f64::EPSILON), Ordering::Greater)
    );
    let (x, o) = BigFloat::from_str_radix(&format!("-{half}e0"), 10, 53, Round::Nearest).unwrap();
    assert_eq!((x, o), (BigFloat::from(-1.0), Ordering::Greater));
    // Many digits at a large precision, and back.
    let mut s = 0x5EED_7E47_0000_0005u64;
    let digits: String = (0..size(20000, 300))
        .map(|_| (b'0' + below(&mut s, 10) as u8) as char)
        .collect();
    let text = format!("0.{digits}");
    let x: BigFloat = text.parse().unwrap();
    assert!(x.prec() > 3 * digits.len() as u64);
    let back = x.to_string();
    let trimmed = text.trim_end_matches('0');
    assert_eq!(back, trimmed);
}

#[test]
fn from_str_precision_and_errors() {
    for (t, prec) in [
        ("0.1", 53),
        ("12345678901234567", 53),
        ("123456789012345678", 60),
        ("1.00000000000000000000", 70),
        ("0.000000000000000000001", 53),
        ("0x1.8p3", 53),
        ("0x123456789abcdef0123", 76),
        ("0b101", 53),
        ("-0", 53),
        ("inf", 53),
        ("nan(x_1)", 53),
    ] {
        let x: BigFloat = t.parse().unwrap();
        assert_eq!(x.prec(), prec, "{t}");
    }
    assert_eq!("0x1.8p3".parse::<BigFloat>().unwrap(), 12);
    assert_eq!(
        "0b-1".parse::<BigFloat>().err().map(|e| *e.kind()),
        Some(BigFloatErrorKind::InvalidDigit)
    );
    assert!("-0".parse::<BigFloat>().unwrap().is_sign_negative());
    assert!("-Infinity".parse::<BigFloat>().unwrap() == f64::NEG_INFINITY);
    assert!("NaN".parse::<BigFloat>().unwrap().is_nan());
    assert_eq!(
        "".parse::<BigFloat>().unwrap_err().kind(),
        &BigFloatErrorKind::Empty
    );
    for bad in [
        "-", "+", ".", "1e", "1e+", "e5", "1.2.3", "1_000", " 1", "1 ", "0x", "0xg", "1p3", "nan(",
        "nan(a b)", "infx", "@inf", "--1", "1e5.5", "0x1e3p", "١",
    ] {
        assert_eq!(
            bad.parse::<BigFloat>().unwrap_err().kind(),
            &BigFloatErrorKind::InvalidDigit,
            "{bad:?}"
        );
    }
    // Radix-specific forms.
    let r = |t: &str, radix: u32| BigFloat::from_str_radix(t, radix, 64, Round::Nearest);
    assert_eq!(r("z.8", 36).unwrap().0.to_f64(), 35.22222222222222);
    assert!(r("1e3", 16).unwrap().0 == 0x1e3);
    assert!(r("1e3", 15).unwrap().0 == 15 * 15 + 14 * 15 + 3);
    assert!(r("1e3", 11).is_err());
    assert_eq!(r("10@2", 7).unwrap().0, 343);
    assert_eq!(r("1p-2", 2).unwrap().0, 0.25);
    assert_eq!(r("0b1p-2", 2).unwrap().0, 0.25);
    assert_eq!(r("0x10", 16).unwrap().0, 16);
    assert!(r("0x10", 10).is_err());
    assert!(r("inf", 17).is_err() && r("@InF@", 17).unwrap().0.is_infinite());
    assert!(r("nan", 24).unwrap().0 == 23 * 24 * 24 + 10 * 24 + 23);
    assert!(r("@nan@(x)", 30).unwrap().0.is_nan());
}

#[test]
#[should_panic(expected = "radix 37")]
fn radix_out_of_range_panics() {
    let _ = BigFloat::from(1).to_string_radix(37, None, Round::Nearest);
}

#[test]
#[should_panic(expected = "digits of zero")]
fn zero_digits_panic() {
    let _ = BigFloat::from(1).to_digits(10, Some(0), Round::Nearest);
}

#[test]
fn exact_values_at_large_precisions() {
    // Values whose digits or binary value are exact, where the work would not be small: the
    // exact path must take them, as Ziv's loop never ends on them.
    let one = BigFloat::from(1).with_prec(size(100_000, 2000) as u64);
    assert_eq!(
        one.to_digits(10, Some(5), Round::Up),
        Some(("10000".to_string(), 1, Ordering::Equal))
    );
    let tie = BigFloat::from(2.5).with_prec(size(100_000, 2000) as u64);
    for (round, d, o) in [
        (Round::Nearest, "2", Ordering::Less),
        (Round::Up, "3", Ordering::Greater),
        (Round::TowardZero, "2", Ordering::Less),
    ] {
        assert_eq!(
            tie.to_digits(10, Some(1), round),
            Some((d.to_string(), 1, o))
        );
    }
    let tie = -BigFloat::from(3.5).with_prec(size(100_000, 2000) as u64);
    assert_eq!(
        tie.to_digits(10, Some(1), Round::Nearest),
        Some(("4".to_string(), 1, Ordering::Less))
    );
    assert_eq!(tie.to_string(), "-3.5");
    // 5^1000/10^1000 = 2^-1000, from 699 digits read to 53 bits.
    let five = Number::from(5).pow(1000).to_string();
    let text = format!("{five}e-1000");
    let (x, o) = BigFloat::from_str_radix(&text, 10, 53, Round::Nearest).unwrap();
    assert_eq!(o, Ordering::Equal);
    assert_eq!(
        x,
        BigFloat::from_parts(&Number::from(1), -1000, 1, Round::Nearest).0
    );
    // And 2^1000 in decimal, read to one bit.
    let two = (Number::from(1) << 1000usize).to_string();
    let (x, o) = BigFloat::from_str_radix(&two, 10, 1, Round::Up).unwrap();
    assert_eq!((x.exponent(), o), (Some(1001), Ordering::Equal));
}

#[test]
fn the_page_example() {
    // The example of the page's section, kept true here.
    let x: BigFloat = "0.1".parse().unwrap();
    assert_eq!(x, 0.1);
    assert_eq!(x.to_string(), "0.1");
    assert_eq!(format!("{x:.20}"), "0.10000000000000000555");
    let (d, e, o) = x.to_digits(10, Some(3), Round::Up).unwrap();
    assert_eq!((d.as_str(), e, o), ("101", 0, Ordering::Greater));
    let (y, o) = BigFloat::from_str_radix("0.1", 10, 200, Round::Down).unwrap();
    assert_eq!(o, Ordering::Less);
    assert_eq!(y.to_f64_round(Round::Nearest), (0.1, Ordering::Greater));
    assert_eq!(BigFloat::from(-2.5).round_ties_even(), -2);
    assert_eq!(BigFloat::from(10.5).rint_round(2, Round::Nearest).0, 12);
}

#[test]
fn ziv_decides_near_boundaries_at_huge_exponents() {
    // A number within 2^-1900 of a half-integer at the scale of its 20 digits, with a decimal
    // exponent of 10^15: Ziv's loop must go on until its interval leaves the half.
    let prec = size(2000, 400) as u64;
    let text = "12345678901234567890.5e1000000000000000";
    for (round, last, o) in [
        (Round::Up, "1", Ordering::Greater),
        (Round::Down, "0", Ordering::Less),
    ] {
        let (x, _) = BigFloat::from_str_radix(text, 10, prec, round).unwrap();
        let (d, e, got) = x.to_digits(10, Some(20), Round::Nearest).unwrap();
        assert_eq!(d, format!("1234567890123456789{last}"), "{round:?}");
        assert_eq!((e, got), (1_000_000_000_000_020, o), "{round:?}");
    }
    // The same below one: a decimal exponent of -10^15.
    let text = "12345678901234567890.5e-1000000000000000";
    for (round, last, o) in [
        (Round::Up, "1", Ordering::Greater),
        (Round::Down, "0", Ordering::Less),
    ] {
        let (x, _) = BigFloat::from_str_radix(text, 10, prec, round).unwrap();
        let (d, e, got) = x.to_digits(10, Some(20), Round::Nearest).unwrap();
        assert_eq!(d, format!("1234567890123456789{last}"), "{round:?}");
        assert_eq!((e, got), (-999_999_999_999_980, o), "{round:?}");
    }
    // A decimal within 10^-400 of the midpoint of two 53-bit numbers, both sides.
    let (x, _) = BigFloat::from_str_radix("1.5e1000000000000000", 10, 53, Round::Nearest).unwrap();
    let ulp = BigFloat::from_parts(
        &Number::from(1),
        x.exponent().unwrap() - 54,
        1,
        Round::Nearest,
    )
    .0;
    let (mid, o) = x.add_round(&ulp, 54, Round::Nearest);
    assert_eq!(o, Ordering::Equal);
    let next = x.add_round(&ulp, 53, Round::Up).0;
    let digits = size(400, 120);
    for (round, want, o) in [
        (Round::Up, &next, Ordering::Greater),
        (Round::Down, &x, Ordering::Less),
    ] {
        let t = mid.to_string_radix(10, Some(digits), round);
        let got = BigFloat::from_str_radix(&t, 10, 53, Round::Nearest).unwrap();
        assert_eq!(got.0, *want, "{t}");
        assert_eq!(got.1, o, "{t}");
    }
}

#[test]
fn display_of_every_short_float() {
    // Every float of up to 7 bits and exponents of -30 to 30: the shortest that reads back,
    // where the ends of the interval are often decimals of few digits.
    let (max_prec, max_exp) = if cfg!(miri) { (3, 4) } else { (7, 30) };
    for prec in 1..=max_prec {
        for m in (1u32 << (prec - 1))..(1 << prec) {
            for e in -max_exp..=max_exp {
                let x = BigFloat::from_parts(&Number::from(m), e, prec, Round::Nearest).0;
                let text = x.to_string();
                let (back, _) = BigFloat::from_str_radix(&text, 10, prec, Round::Nearest).unwrap();
                assert_eq!(back, x, "{x:?} as {text}");
                let sci = format!("{x:e}");
                let mantissa = sci.split('e').next().unwrap();
                let n = mantissa.chars().filter(char::is_ascii_digit).count();
                if n > 1 {
                    for round in [Round::Down, Round::Up] {
                        let t = x.to_string_radix(10, Some(n - 1), round);
                        let (y, _) =
                            BigFloat::from_str_radix(&t, 10, prec, Round::Nearest).unwrap();
                        assert!(y != x, "{x:?}: {t} is shorter than {sci}");
                    }
                }
            }
        }
    }
    assert_eq!(
        BigFloat::from_parts(&Number::from(3), 2, 2, Round::Nearest)
            .0
            .to_string(),
        "12"
    );
}

#[test]
fn short_decimals_near_midpoints() {
    // The decimals of up to 19 digits nearest to midpoints of 53-bit numbers, which the
    // register path reads: often within 2^-63 of the midpoint, where only the remainder's sticky
    // bit tells the side.
    let mut s = 0x5EED_7E47_0000_0006u64;
    for _ in 0..size(400, 10) {
        let m = (1u64 << 53) | (xorshift(&mut s) >> 11) | 1;
        let e = below(&mut s, 80) as i64 - 40;
        let mid = BigFloat::from_parts(&Number::from(m), e - 54, 54, Round::Nearest).0;
        for n in [17usize, 18, 19] {
            for dir in [Round::Nearest, Round::Up, Round::Down] {
                let t = mid.to_string_radix(10, Some(n), dir);
                let q: Rational = t.parse().unwrap();
                for round in MODES {
                    let prec = [53, 54, 60][below(&mut s, 3) as usize];
                    same(
                        &BigFloat::from_str_radix(&t, 10, prec, round).unwrap(),
                        &reference(&q, prec, round),
                        &t,
                    );
                }
            }
        }
    }
}

#[test]
fn held_exponents_with_many_digits() {
    // Exponents past the bound at which they are held, with digits that the reading truncates.
    let digits = "1234567890123456789012345678901234567890";
    for round in MODES {
        let big = format!("{digits}e99999999999999999999999999");
        let (x, o) = BigFloat::from_str_radix(&big, 10, 64, round).unwrap();
        let overflow = matches!(round, Round::Nearest | Round::Up);
        assert_eq!(
            (x.is_infinite(), o),
            (
                overflow,
                if overflow {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            )
        );
        let small = format!("-0.{digits}e-99999999999999999999999999");
        let (x, o) = BigFloat::from_str_radix(&small, 10, 64, round).unwrap();
        assert_eq!(x.is_zero(), round != Round::Down, "{round:?}");
        assert_eq!(
            o,
            if round == Round::Down {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        );
    }
    assert_eq!(BigFloat::zero(3).to_digits(10, Some(2), Round::Up), None);
}

#[test]
#[should_panic(expected = "digits of zero")]
fn zero_digits_panic_for_zero_too() {
    let _ = BigFloat::zero(5).to_string_radix(10, Some(0), Round::Nearest);
}
