//! Rationals as text: every form `FromStr` accepts and every error, decimals checked exactly
//! against fractions built from their parts, round trips through `Display` and through every
//! radix, and long text.

use super::common::{below, check, rational, size};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::{ParseRationalError, Rational, RationalErrorKind};

/// The rational of the text, checked for its form.
#[track_caller]
fn parse(s: &str) -> Rational {
    let r: Rational = s.parse().unwrap_or_else(|e| panic!("{s:?}: {e}"));
    check(&r);
    assert_eq!(r.byte_order(), ByteOrder::LittleEndian);
    r
}

/// The kind of the error the text gives.
#[track_caller]
fn kind(s: &str) -> RationalErrorKind {
    match s.parse::<Rational>() {
        Ok(r) => panic!("{s:?} read as {r}"),
        Err(e) => *e.kind(),
    }
}

/// `p/q`, reduced.
fn q(p: i128, q: i128) -> Rational {
    Rational::new(Number::from(p), Number::from(q))
}

/// `10^k`.
fn ten(k: u32) -> Number {
    Number::from(10).pow(k)
}

#[test]
fn accepted_forms() {
    let cases: &[(&str, Rational)] = &[
        ("0", q(0, 1)),
        ("-0", q(0, 1)),
        ("+0", q(0, 1)),
        ("7", q(7, 1)),
        ("+7", q(7, 1)),
        ("-7", q(-7, 1)),
        ("007", q(7, 1)),
        ("3/4", q(3, 4)),
        ("-3/4", q(-3, 4)),
        ("+3/4", q(3, 4)),
        ("6/4", q(3, 2)),
        ("-6/0004", q(-3, 2)),
        ("0/5", q(0, 1)),
        ("-0/5", q(0, 1)),
        ("10/5", q(2, 1)),
        ("0.1", q(1, 10)),
        ("1.5", q(3, 2)),
        ("-1.5", q(-3, 2)),
        (".5", q(1, 2)),
        ("-.5", q(-1, 2)),
        ("5.", q(5, 1)),
        ("+5.", q(5, 1)),
        ("0.0", q(0, 1)),
        ("-0.000", q(0, 1)),
        ("00.00", q(0, 1)),
        ("3.14159", q(314159, 100000)),
        ("2.50", q(5, 2)),
        ("0.125", q(1, 8)),
        ("0.2", q(1, 5)),
        ("-0.75", q(-3, 4)),
        ("1e3", q(1000, 1)),
        ("1E3", q(1000, 1)),
        ("2e10", q(20_000_000_000, 1)),
        ("2e+10", q(20_000_000_000, 1)),
        ("1e-3", q(1, 1000)),
        ("-1.5e-3", q(-3, 2000)),
        ("1.5E+2", q(150, 1)),
        ("1500e-2", q(15, 1)),
        ("120e-1", q(12, 1)),
        ("1.e2", q(100, 1)),
        (".5e1", q(5, 1)),
        ("0e5", q(0, 1)),
        ("0e-5", q(0, 1)),
        ("25e-2", q(1, 4)),
        ("-625e-4", q(-1, 16)),
        ("8e-3", q(1, 125)),
        ("1e0", q(1, 1)),
        ("1e-0", q(1, 1)),
        ("1e00000000000000000000000003", q(1000, 1)),
        ("0.0e1000000", q(0, 1)),
        (
            "12345678901234567890",
            Rational::from(12345678901234567890u64),
        ),
        (
            "-170141183460469231731687303715884105728",
            Rational::from(i128::MIN),
        ),
        (
            "1/18446744073709551616",
            Rational::from(1) / (Number::from(1) << 64usize),
        ),
        (
            "9999999999999999999/10000000000000000000",
            Rational::new(Number::from(9999999999999999999u64), ten(19)),
        ),
        ("1e19", Rational::from(ten(19))),
        ("1e-19", Rational::from(1) / ten(19)),
        ("1e20", Rational::from(ten(20))),
        ("1e-20", Rational::from(1) / ten(20)),
        (
            "5e-30",
            Rational::new(Number::from(1), Number::from(2) * ten(29)),
        ),
        (
            "0.00000000000000000000000000000000000000005",
            Rational::new(Number::from(1), Number::from(2) * ten(40)),
        ),
        (
            "-3.000000000000000000000000000000000000000000000000001",
            Rational::new(Number::from(-3) * ten(51) - 1u32, ten(51)),
        ),
    ];
    for (s, expected) in cases {
        assert_eq!(&parse(s), expected, "{s:?}");
    }
    // The largest exponents.
    assert_eq!(parse("1e1000"), Rational::from(ten(1000)));
    assert_eq!(
        parse("-7e-1000"),
        Rational::new(Number::from(-7), ten(1000))
    );
    let r = parse(if cfg!(miri) { "3e300" } else { "3e100000" });
    assert!(!r.is_negative() && r.is_integer());
}

#[test]
fn error_kinds() {
    use RationalErrorKind::*;
    let cases: &[(&str, RationalErrorKind)] = &[
        ("", Empty),
        ("+", InvalidDigit),
        ("-", InvalidDigit),
        ("--1", InvalidDigit),
        ("+-1", InvalidDigit),
        ("-+1", InvalidDigit),
        (" 1", InvalidDigit),
        ("1 ", InvalidDigit),
        ("1_000", InvalidDigit),
        ("12a", InvalidDigit),
        ("0x10", InvalidDigit),
        ("/", InvalidDigit),
        ("1/", InvalidDigit),
        ("/2", InvalidDigit),
        ("-/2", InvalidDigit),
        ("1/-2", InvalidDigit),
        ("1/+2", InvalidDigit),
        ("1//2", InvalidDigit),
        ("1/2/3", InvalidDigit),
        ("1/2.5", InvalidDigit),
        ("1.5/2", InvalidDigit),
        ("1/2e3", InvalidDigit),
        ("1e3/2", InvalidDigit),
        ("1/0", ZeroDenominator),
        ("-5/000", ZeroDenominator),
        ("123456789012345678901234567890/0", ZeroDenominator),
        (".", InvalidDigit),
        ("-.", InvalidDigit),
        ("..5", InvalidDigit),
        ("1..5", InvalidDigit),
        ("1.5.", InvalidDigit),
        ("1.2.3", InvalidDigit),
        (".-5", InvalidDigit),
        ("-.-5", InvalidDigit),
        ("1.-5", InvalidDigit),
        ("1.+5", InvalidDigit),
        ("+-.5", InvalidDigit),
        ("+-0.0", InvalidDigit),
        ("e5", InvalidDigit),
        (".e5", InvalidDigit),
        ("-e5", InvalidDigit),
        ("1e", InvalidDigit),
        ("1e+", InvalidDigit),
        ("1e-", InvalidDigit),
        ("1e--5", InvalidDigit),
        ("1e5.0", InvalidDigit),
        ("1e5e5", InvalidDigit),
        ("1ee5", InvalidDigit),
        ("1.5e 3", InvalidDigit),
        ("1.5f3", InvalidDigit),
        ("inf", InvalidDigit),
        ("NaN", InvalidDigit),
        ("١", InvalidDigit),
        ("1e1000001", ExponentOutOfRange),
        ("1e-1000001", ExponentOutOfRange),
        ("0e1000001", ExponentOutOfRange),
        ("1.5e99999999999999999999999999", ExponentOutOfRange),
        ("x.5e99999999999999999999999999", InvalidDigit),
        ("1.x5e99999999999999999999999999", InvalidDigit),
        ("-.-5e99999999999999999999999999", InvalidDigit),
        ("1e1000000x", InvalidDigit),
        ("-6/-4", InvalidDigit),
    ];
    for &(s, k) in cases {
        assert_eq!(kind(s), k, "{s:?}");
    }
    // Messages, and the error of the radices.
    for s in ["", "x", "1/0", "1e9999999"] {
        let e: ParseRationalError = s.parse::<Rational>().unwrap_err();
        assert!(!e.to_string().is_empty());
    }
    let e = Rational::from_str_radix("12", 2).unwrap_err();
    assert_eq!(*e.kind(), InvalidDigit);
    assert_eq!(*Rational::from_str_radix("", 7).unwrap_err().kind(), Empty);
    assert_eq!(
        *Rational::from_str_radix("-", 7).unwrap_err().kind(),
        InvalidDigit
    );
    assert_eq!(
        *Rational::from_str_radix("1/0", 7).unwrap_err().kind(),
        ZeroDenominator
    );
    assert_eq!(
        *Rational::from_str_radix("1.5", 10).unwrap_err().kind(),
        InvalidDigit
    );
    assert_eq!(
        *Rational::from_str_radix("1e5", 10).unwrap_err().kind(),
        InvalidDigit
    );
    assert_eq!(
        *Rational::from_str_radix("1/-1", 16).unwrap_err().kind(),
        InvalidDigit
    );
}

#[test]
#[should_panic(expected = "must lie in the range")]
fn radix_too_large() {
    let _ = Rational::from_str_radix("1", 37);
}

#[test]
#[should_panic(expected = "must lie in the range")]
fn radix_too_small_to_write() {
    let _ = Rational::from(1).to_str_radix(1);
}

#[test]
fn radices() {
    assert_eq!(Rational::from_str_radix("-FF/1e", 16).unwrap(), q(-255, 30));
    assert_eq!(Rational::from_str_radix("+101/11", 2).unwrap(), q(5, 3));
    assert_eq!(Rational::from_str_radix("zz", 36).unwrap(), q(1295, 1));
    assert_eq!(Rational::from_str_radix("1e", 15).unwrap(), q(29, 1));
    assert_eq!(Rational::from_str_radix("-0/7", 8).unwrap(), q(0, 1));
    assert_eq!(Rational::from_str_radix("10/4", 10).unwrap(), q(5, 2));
    assert_eq!(q(-255, 16).to_str_radix(16), "-ff/10");
    assert_eq!(q(5, 3).to_str_radix(2), "101/11");
    assert_eq!(q(-7, 1).to_str_radix(7), "-10");
    assert_eq!(q(0, 1).to_str_radix(36), "0");
    let mut state = 0xA5A5_5A5A_1234_4321u64;
    for _ in 0..size(2000, 40) {
        let r = rational(&mut state, 60);
        let radix = 2 + below(&mut state, 35) as u32;
        let s = r.to_str_radix(radix);
        let back = Rational::from_str_radix(&s, radix).unwrap();
        check(&back);
        assert_eq!(back, r, "{s} in radix {radix}");
        // Upper case reads the same.
        assert_eq!(
            Rational::from_str_radix(&s.to_uppercase(), radix).unwrap(),
            r
        );
        // The parts as the numbers write them.
        let expected = if r.is_integer() {
            r.numer().to_str_radix(radix)
        } else {
            format!(
                "{}/{}",
                r.numer().to_str_radix(radix),
                r.denom().to_str_radix(radix)
            )
        };
        assert_eq!(s, expected);
        if radix == 10 {
            assert_eq!(s, r.to_string());
        }
    }
}

#[test]
fn display_round_trips() {
    let mut state = 0x0123_4567_89AB_CDEFu64;
    for i in 0..size(3000, 60) {
        let max = if i % 20 == 0 { 400 } else { 40 };
        let r = rational(&mut state, max);
        let s = r.to_string();
        assert_eq!(parse(&s), r, "{s}");
        assert_eq!(format!("{r:?}"), s);
        // Width, fill and sign apply to the whole.
        let padded = format!("{r:>width$}", width = s.len() + 3);
        assert_eq!(padded, format!("   {s}"));
        let plus = format!("{r:+}");
        if r.is_negative() {
            assert_eq!(plus, s);
        } else {
            assert_eq!(plus, format!("+{s}"));
        }
        assert_eq!(format!("{r:0>w$}", w = s.len() + 1), format!("0{s}"));
        let zero_padded = format!("{r:0w$}", w = s.len() + 2);
        if r.is_negative() {
            assert_eq!(zero_padded, format!("-00{}", &s[1..]));
        } else {
            assert_eq!(zero_padded, format!("00{s}"));
        }
        // With a + in front, and big-endian parts, the same.
        let p = format!("+{}", r.abs());
        assert_eq!(parse(&p), r.abs());
    }
    assert_eq!(format!("{:<6}|", q(-1, 2)), "-1/2  |");
    assert_eq!(format!("{:^7}|", q(1, 2)), "  1/2  |");
    assert_eq!(format!("{:+}", q(0, 1)), "+0");
}

/// A random string of `n` decimal digits, the first not zero when `lead` is set.
fn digits(state: &mut u64, n: usize, lead: bool) -> String {
    (0..n)
        .map(|i| {
            let d = if i == 0 && lead {
                1 + below(state, 9)
            } else {
                below(state, 10)
            };
            char::from(b'0' + d as u8)
        })
        .collect()
}

#[test]
fn decimals_are_exact() {
    let mut state = 0xFEED_FACE_DEAD_BEEFu64;
    for i in 0..size(3000, 60) {
        let long = i % 25 == 0;
        let ni = below(&mut state, if long { 120 } else { 12 }) as usize;
        let nf = below(&mut state, if long { 120 } else { 12 }) as usize;
        if ni + nf == 0 {
            continue;
        }
        let int = digits(&mut state, ni, false);
        let mut frac = digits(&mut state, nf, false);
        // Trailing zeros, and fractions that end in a 5, now and then.
        match below(&mut state, 5) {
            0 => frac.push_str("000"),
            1 => frac.push('5'),
            _ => {}
        }
        let nf = frac.len();
        let negative = below(&mut state, 2) == 1;
        let (exponent, x): (String, i64) = match below(&mut state, 3) {
            0 => (String::new(), 0),
            _ => {
                let x = below(&mut state, 60) as i64 - 30;
                let e = if below(&mut state, 2) == 0 { 'e' } else { 'E' };
                let sign = if x >= 0 && below(&mut state, 2) == 0 {
                    "+"
                } else {
                    ""
                };
                (format!("{e}{sign}{x}"), x)
            }
        };
        let text = format!(
            "{}{int}{}{frac}{exponent}",
            if negative { "-" } else { "" },
            if nf > 0 || below(&mut state, 2) == 0 {
                "."
            } else {
                ""
            }
        );
        // The reference: the digits as an integer, times 10^(x - nf), reduced by `new`.
        let all = format!("{int}{frac}");
        let n: Number = all.parse().unwrap();
        let n = if negative { -n } else { n };
        let k = x - nf as i64;
        let expected = if k >= 0 {
            Rational::new(&n * &ten(k as u32), Number::from(1))
        } else {
            Rational::new(n, ten((-k) as u32))
        };
        assert_eq!(parse(&text), expected, "{text}");
    }
    // Fractions with many fives and twos, past the remainder by 5^27.
    for k in [1u32, 26, 27, 28, 40, 100] {
        let five = Number::from(5).pow(k);
        let s = format!("{}e-{}", five, k + 3);
        assert_eq!(parse(&s), Rational::new(five.clone(), ten(k + 3)), "{s}");
        let s = format!("{}e-{}", five, k - 1);
        assert_eq!(parse(&s), Rational::new(five, ten(k - 1)), "{s}");
        let two = Number::from(2).pow(k);
        let s = format!("{two}e-{}", k + 2);
        assert_eq!(parse(&s), Rational::new(two.clone(), ten(k + 2)), "{s}");
        let s = format!("{two}e-{}", k / 2);
        assert_eq!(parse(&s), Rational::new(two, ten(k / 2)), "{s}");
    }
    // 2^-k written out in decimal is exactly 2^-k.
    for k in [1u32, 10, 64, 100, 300] {
        let digits = Number::from(5).pow(k).to_string();
        let s = format!("0.{}{digits}", "0".repeat(k as usize - digits.len()));
        assert_eq!(
            parse(&s),
            Rational::from(1) / (Number::from(1) << k as usize),
            "{s}"
        );
    }
    // The decimal of a float is the float.
    for x in [
        0.1,
        1.0 / 3.0,
        6.02214076e23,
        1e-300,
        f64::MAX,
        f64::MIN_POSITIVE,
    ] {
        let s = format!("{x:e}");
        let r = parse(&s);
        assert_eq!(r.to_f64(), x, "{s}");
    }
}

#[test]
fn long_text() {
    let mut state = 0x5555_AAAA_3333_CCCCu64;
    for &n in &[50usize, 400, 3000, 20000] {
        if cfg!(miri) && n > 400 {
            continue;
        }
        let a = digits(&mut state, n, true);
        let b = digits(&mut state, n / 2 + 1, true);
        let na: Number = a.parse().unwrap();
        let nb: Number = b.parse().unwrap();
        let s = format!("-{a}/{b}");
        assert_eq!(parse(&s), Rational::new(-&na, nb.clone()), "{n}");
        assert_eq!(parse(&a), Rational::from(&na));
        let r = Rational::new(na.clone(), nb.clone());
        assert_eq!(parse(&r.to_string()), r);
        // A long decimal.
        let s = format!("{a}.{b}");
        let expected = Rational::new(&(&na * &ten(b.len() as u32)) + &nb, ten(b.len() as u32));
        assert_eq!(parse(&s), expected);
        // Radices.
        let radix = 2 + below(&mut state, 35) as u32;
        let t = r.to_str_radix(radix);
        assert_eq!(Rational::from_str_radix(&t, radix).unwrap(), r);
    }
    // A long integer with a few thousand zeros, as an exponent and written out.
    let zeros = size(5000, 50);
    let written = format!("7{}", "0".repeat(zeros));
    assert_eq!(parse(&format!("7e{zeros}")), parse(&written));
    assert_eq!(parse(&format!("{written}e-{zeros}")), q(7, 1));
    assert_eq!(parse(&format!("{written}.000e-{}", zeros + 1)), q(7, 10));
}
