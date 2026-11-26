//! Text: reading against the primitives' `from_str_radix` (the values, and the kind of error on
//! random text with bad characters, signs and overflows, in every radix), the formatting flags
//! against `u128`'s and `i128`'s, and round trips in every radix at every width.

use super::common::{Rng, outcome, size};
use a::math::numbers::{ByteOrder, Number};
use a::math::uint::{Int, Uint};
use core::num::IntErrorKind;

/// Random text for a radix: digits of the radix, the most often; sometimes a sign, a bad
/// character, a digit beyond the radix, a letter in the other case, nothing at all, or more digits
/// than the type holds.
fn text(rng: &mut Rng, radix: u32, max_digits: usize) -> String {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let mut s = String::new();
    match rng.below(8) {
        0 => s.push('+'),
        1 | 2 => s.push('-'),
        _ => {}
    }
    let len = match rng.below(6) {
        0 => rng.below(3) as usize,
        1 => max_digits + rng.below(3) as usize,
        2 => max_digits - 1,
        _ => 1 + rng.below(max_digits as u64 + 4) as usize,
    };
    // Leading zeros now and then.
    let zeros = if rng.below(5) == 0 {
        rng.below(30) as usize
    } else {
        0
    };
    for _ in 0..zeros {
        s.push('0');
    }
    for i in 0..len {
        let d = DIGITS[rng.below(radix as u64) as usize] as char;
        // The first digit at its largest, to reach the overflow edge.
        let d = if i == 0 && rng.below(3) == 0 {
            DIGITS[radix as usize - 1] as char
        } else {
            d
        };
        s.push(if rng.below(4) == 0 {
            d.to_ascii_uppercase()
        } else {
            d
        });
    }
    if rng.below(5) == 0 {
        let bad = [
            '_',
            ' ',
            '-',
            '+',
            'é',
            '/',
            ':',
            '@',
            '[',
            '`',
            '{',
            DIGITS[(radix as usize).min(35)] as char,
        ];
        let c = bad[rng.below(bad.len() as u64) as usize];
        // A digit beyond the radix is bad only below radix 36.
        if !(c.is_ascii_alphanumeric() && radix == 36) {
            let at = rng.below(s.len() as u64 + 1) as usize;
            let at = (0..=at).rev().find(|&i| s.is_char_boundary(i)).unwrap_or(0);
            s.insert(at, c);
        }
    }
    s
}

/// The number of digits of `u128::MAX` in `radix`.
fn digits_of_max(radix: u32, bits: u32) -> usize {
    (bits as f64 / (radix as f64).log2()).ceil() as usize
}

macro_rules! reading_against_primitive {
    ($test:ident, $n:literal, $p:ty, $s:ty, $seed:literal) => {
        #[test]
        fn $test() {
            let mut rng = Rng($seed);
            // How often each outcome came up, for the unsigned and the signed type: Ok, Empty,
            // InvalidDigit, PosOverflow, NegOverflow.
            let mut seen = [[0usize; 5]; 2];
            let slot = |r: &Result<_, a::math::uint::ParseIntError>| match r
                .as_ref()
                .map_err(|e| e.kind())
            {
                Ok(_) => 0,
                Err(IntErrorKind::Empty) => 1,
                Err(IntErrorKind::InvalidDigit) => 2,
                Err(IntErrorKind::PosOverflow) => 3,
                Err(_) => 4,
            };
            for i in 0..size(30_000, 300) {
                let radix = 2 + (i % 35) as u32;
                let t = text(&mut rng, radix, digits_of_max(radix, 64 * $n));
                let ours = Uint::<$n>::from_str_radix(&t, radix).map(|x| x.to_le_bytes());
                let theirs = <$p>::from_str_radix(&t, radix).map(|x| x.to_le_bytes());
                seen[0][slot(&ours)] += 1;
                match (&ours, &theirs) {
                    (Ok(a), Ok(b)) => assert_eq!(a, b, "{t:?} in radix {radix}"),
                    (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind(), "{t:?} in radix {radix}"),
                    _ => panic!("{t:?} in radix {radix}: {ours:?} against {theirs:?}"),
                }
                let ours = Int::<$n>::from_str_radix(&t, radix).map(|x| x.to_le_bytes());
                let theirs = <$s>::from_str_radix(&t, radix).map(|x| x.to_le_bytes());
                seen[1][slot(&ours)] += 1;
                match (&ours, &theirs) {
                    (Ok(a), Ok(b)) => assert_eq!(a, b, "{t:?} in radix {radix}"),
                    (Err(a), Err(b)) => assert_eq!(a.kind(), b.kind(), "{t:?} in radix {radix}"),
                    _ => panic!("{t:?} in radix {radix}: {ours:?} against {theirs:?}"),
                }
                if radix == 10 {
                    let ours = t.parse::<Uint<$n>>().map(|x| x.to_le_bytes());
                    let theirs = t.parse::<$p>().map(|x| x.to_le_bytes());
                    assert_eq!(ours.ok(), theirs.ok(), "{t:?}");
                    let ours = t.parse::<Int<$n>>().map(|x| x.to_le_bytes());
                    let theirs = t.parse::<$s>().map(|x| x.to_le_bytes());
                    assert_eq!(ours.ok(), theirs.ok(), "{t:?}");
                }
            }
            // Every outcome came up, each often (no negative overflow for the unsigned type).
            if !cfg!(miri) {
                assert!(seen[0][..4].iter().all(|&c| c > 50), "{seen:?}");
                assert!(seen[1].iter().all(|&c| c > 50), "{seen:?}");
            }
        }
    };
}

reading_against_primitive!(
    text_is_read_as_u64_and_i64_read_it,
    1,
    u64,
    i64,
    0x7e47_0001
);
reading_against_primitive!(
    text_is_read_as_u128_and_i128_read_it,
    2,
    u128,
    i128,
    0x7e47_0002
);

#[test]
fn the_errors_and_their_messages_are_the_primitives() {
    let cases: [(&str, IntErrorKind); 9] = [
        ("", IntErrorKind::Empty),
        ("+", IntErrorKind::InvalidDigit),
        ("-", IntErrorKind::InvalidDigit),
        ("-0", IntErrorKind::InvalidDigit),
        ("1_000", IntErrorKind::InvalidDigit),
        (" 1", IntErrorKind::InvalidDigit),
        (
            "340282366920938463463374607431768211456",
            IntErrorKind::PosOverflow,
        ),
        // The overflow comes before the bad character, so it is what is reported.
        (
            "340282366920938463463374607431768211456x",
            IntErrorKind::PosOverflow,
        ),
        (
            "34028236692093846346337460743176821145x6",
            IntErrorKind::InvalidDigit,
        ),
    ];
    for (t, kind) in cases {
        let ours = Uint::<2>::from_str_radix(t, 10).unwrap_err();
        let theirs = t.parse::<u128>().unwrap_err();
        assert_eq!(*ours.kind(), kind, "{t:?}");
        assert_eq!(ours.kind(), theirs.kind(), "{t:?}");
        assert_eq!(ours.to_string(), theirs.to_string());
    }
    let ours = Int::<2>::from_str_radix("-170141183460469231731687303715884105729", 10);
    let theirs = "-170141183460469231731687303715884105729".parse::<i128>();
    assert_eq!(
        ours.as_ref().unwrap_err().kind(),
        &IntErrorKind::NegOverflow
    );
    assert_eq!(
        ours.unwrap_err().to_string(),
        theirs.unwrap_err().to_string()
    );
    assert_eq!(Int::<2>::from_str_radix("-0", 10), Ok(Int::ZERO));
    assert!(std::error::Error::source(&Uint::<2>::from_str_radix("", 10).unwrap_err()).is_none());
}

#[test]
fn a_radix_outside_2_to_36_panics() {
    for radix in [0, 1, 37, 100] {
        assert!(outcome(|| Uint::<2>::from_str_radix("1", radix)).is_err());
        assert!(outcome(|| Int::<2>::from_str_radix("1", radix)).is_err());
        assert!(outcome(|| Uint::<2>::ONE.to_str_radix(radix)).is_err());
        assert!(outcome(|| Int::<2>::ONE.to_str_radix(radix)).is_err());
    }
}

/// Formats each value with each flag against the primitive's text.
macro_rules! formats {
    ($ours:expr, $theirs:expr; $($spec:literal),*) => {$(
        assert_eq!(format!($spec, $ours), format!($spec, $theirs), "{}", $spec);
    )*};
}

#[test]
fn the_formatting_flags_do_what_they_do_for_u128_and_i128() {
    let mut rng = Rng(0xf1a6);
    let mut values: Vec<u128> = vec![
        0,
        1,
        7,
        0xbeef,
        u64::MAX as u128,
        u128::MAX,
        1 << 127,
        10u128.pow(38),
    ];
    for _ in 0..size(200, 5) {
        values.push((rng.next() as u128) << 64 | rng.next() as u128 >> rng.below(64));
    }
    for &a in &values {
        let x = Uint::<2>::from_le_bytes(a.to_le_bytes());
        formats!(x, a; "{}", "{:>45}", "{:<45}|", "{:^45}", "{:*^46}", "{:+}", "{:045}", "{:+045}",
            "{:5}", "{:x}", "{:#x}", "{:#066x}", "{:#X}", "{:040X}", "{:o}", "{:#o}", "{:>50o}",
            "{:b}", "{:#b}", "{:#0130b}", "{:-^140b}", "{:.3}", "{:#10x}");
        for s in [
            a as i128,
            (a as i128).wrapping_neg(),
            i128::MIN,
            i128::MAX,
            -1,
        ] {
            let y = Int::<2>::from_le_bytes(s.to_le_bytes());
            formats!(y, s; "{}", "{:>45}", "{:<45}|", "{:^45}", "{:*^46}", "{:+}", "{:045}",
                "{:+045}", "{:5}", "{:x}", "{:#x}", "{:#066x}", "{:#X}", "{:040X}", "{:o}",
                "{:#o}", "{:b}", "{:#b}", "{:#0130b}", "{:.3}");
        }
    }
    // A width of 4 limbs, as the issue's example writes it.
    let x = Uint::<4>::MAX >> 4u32;
    assert_eq!(format!("{x:#066x}"), format!("0x0{}", "f".repeat(63)));
    assert_eq!(
        format!("{:#066x}", Uint::<4>::ONE),
        format!("0x{}1", "0".repeat(63))
    );
}

#[test]
fn text_round_trips_in_every_radix_at_every_width() {
    fn round<const N: usize>(rng: &mut Rng, count: usize) {
        for _ in 0..count {
            let x = rng.uint::<N>();
            let y = rng.int::<N>();
            let (a, b) = (Number::from(x), Number::from(y));
            for radix in 2..=36 {
                let t = x.to_str_radix(radix);
                assert_eq!(t, a.to_str_radix(radix));
                assert_eq!(Uint::<N>::from_str_radix(&t, radix), Ok(x));
                assert_eq!(Uint::<N>::from_str_radix(&format!("+{t}"), radix), Ok(x));
                assert_eq!(
                    Number::from_str_radix(&t, radix, ByteOrder::BigEndian).unwrap(),
                    a
                );
                let t = y.to_str_radix(radix);
                assert_eq!(t, b.to_str_radix(radix));
                assert_eq!(Int::<N>::from_str_radix(&t.to_uppercase(), radix), Ok(y));
            }
            assert_eq!(x.to_string().parse::<Uint<N>>(), Ok(x));
            assert_eq!(y.to_string().parse::<Int<N>>(), Ok(y));
            // One more than MAX and one less than MIN overflow.
            let over = (Number::from(Uint::<N>::MAX) + Number::from(1u64)).to_string();
            assert_eq!(
                over.parse::<Uint<N>>().unwrap_err().kind(),
                &IntErrorKind::PosOverflow
            );
            let under = (Number::from(Int::<N>::MIN) - Number::from(1u64)).to_string();
            assert_eq!(
                under.parse::<Int<N>>().unwrap_err().kind(),
                &IntErrorKind::NegOverflow
            );
            assert_eq!(
                Number::from(Int::<N>::MIN).to_string().parse(),
                Ok(Int::<N>::MIN)
            );
        }
    }
    let mut rng = Rng(0x7041);
    let n = size(40, 2);
    round::<1>(&mut rng, n);
    round::<2>(&mut rng, n);
    round::<3>(&mut rng, n);
    round::<4>(&mut rng, n);
    round::<5>(&mut rng, n);
    round::<6>(&mut rng, n);
    round::<7>(&mut rng, n);
    round::<8>(&mut rng, n);
    round::<9>(&mut rng, n);
    round::<16>(&mut rng, n / 2);
    round::<32>(&mut rng, n / 4);
}

#[test]
fn text_can_be_read_in_a_constant() {
    // The prime of P-256, read at compile time.
    const P: Uint<4> = match Uint::from_str_radix(
        "115792089210356248762697446949407573530086143415290314195533631308867097853951",
        10,
    ) {
        Ok(p) => p,
        Err(_) => panic!("not a number"),
    };
    let p = Number::from(2u64).pow(256) - Number::from(2u64).pow(224)
        + Number::from(2u64).pow(192)
        + Number::from(2u64).pow(96)
        - Number::from(1u64);
    assert_eq!(Number::from(P), p);
    const M: Int<2> = match Int::from_str_radix("-7fffffffffffffffffffffffffffffff", 16) {
        Ok(m) => m,
        Err(_) => panic!("not a number"),
    };
    assert_eq!(M, Int::MIN + Int::ONE);
}

/// Formats `$ours` with each `Debug` spec and `$theirs` with the matching `{:#x}` spec.
macro_rules! debug_as_hex {
    ($ours:expr, $theirs:expr; $($debug:literal $hex:literal),*) => {$(
        assert_eq!(format!($debug, $ours), format!($hex, $theirs), "{} against {}", $debug, $hex);
    )*};
}

#[test]
fn debug_takes_the_flags_as_alternate_hexadecimal_does() {
    let mut rng = Rng(0xdeb6);
    for i in 0..size(300, 10) {
        let x = if i < 4 {
            [Uint::<2>::ZERO, Uint::ONE, Uint::MAX, Uint::from_u64(255)][i]
        } else {
            rng.uint::<2>()
        };
        let a = u128::from_le_bytes(x.to_le_bytes());
        let n = Number::from(x);
        debug_as_hex!(x, a; "{:?}" "{:#x}", "{:>20?}" "{:>#20x}", "{:<20?}|" "{:<#20x}|",
            "{:^21?}" "{:^#21x}", "{:*^22?}" "{:*^#22x}", "{:020?}" "{:#020x}", "{:+?}" "{:+#x}",
            "{:+025?}" "{:+#025x}", "{:3?}" "{:#3x}", "{:-<40?}" "{:-<#40x}", "{:x?}" "{:#x}",
            "{:>10.3?}" "{:>#10.3x}", "{:#?}" "{:#x}");
        debug_as_hex!(x, n; "{:>50?}" "{:>#50x}", "{:050?}" "{:#050x}");
        // A signed number: its sign and the hexadecimal of its absolute value, as Number's
        // `{:#x}` writes them.
        for y in [
            x.cast_signed(),
            x.cast_signed().wrapping_neg(),
            Int::MIN,
            Int::MINUS_ONE,
        ] {
            let m = Number::from(y);
            debug_as_hex!(y, m; "{:?}" "{:#x}", "{:>45?}" "{:>#45x}", "{:<45?}|" "{:<#45x}|",
                "{:^46?}" "{:^#46x}", "{:_^47?}" "{:_^#47x}", "{:045?}" "{:#045x}",
                "{:+?}" "{:+#x}", "{:+040?}" "{:+#040x}", "{:5?}" "{:#5x}");
        }
        let z = rng.int::<3>();
        debug_as_hex!(z, Number::from(z); "{:>60?}" "{:>#60x}", "{:+060?}" "{:+#060x}");
    }
}

#[test]
fn debug_writes_hexadecimal() {
    assert_eq!(format!("{:?}", Uint::<3>::ZERO), "0x0");
    assert_eq!(format!("{:?}", Uint::<3>::from_u64(255)), "0xff");
    assert_eq!(
        format!("{:?}", Uint::<3>::from_limbs([1, 2, 0])),
        "0x20000000000000001"
    );
    assert_eq!(format!("{:?}", Int::<2>::from_i64(-5)), "-0x5");
    assert_eq!(
        format!("{:?}", Int::<2>::MIN),
        "-0x80000000000000000000000000000000"
    );
    assert_eq!(format!("{:?}", Uint::<0>::ZERO), "0x0");
}
