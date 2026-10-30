use a::math::numbers::{ByteOrder, Number, NumberErrorKind};

use super::common::*;
use super::div::panic_message;

const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// The digits of a magnitude given least significant byte first in `radix`, worked out
/// independently of the implementation: one digit at a time, the remainder of a short division of
/// 32-bit limbs by the radix. `0` for zero.
fn reference_digits(le: &[u8], radix: u32) -> String {
    let mut limbs: Vec<u32> = le
        .chunks(4)
        .map(|c| {
            let mut b = [0; 4];
            b[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(b)
        })
        .collect();
    let mut out = Vec::new();
    loop {
        while limbs.last() == Some(&0) {
            limbs.pop();
        }
        if limbs.is_empty() {
            break;
        }
        let mut r = 0u64;
        for l in limbs.iter_mut().rev() {
            let t = (r << 32) | u64::from(*l);
            *l = (t / u64::from(radix)) as u32;
            r = t % u64::from(radix);
        }
        out.push(DIGITS[r as usize]);
    }
    if out.is_empty() {
        out.push(b'0');
    }
    out.reverse();
    String::from_utf8(out).unwrap()
}

/// The text `n` should have in `radix`: its sign and the reference digits of its magnitude.
fn reference(n: &Number, radix: u32) -> String {
    let digits = reference_digits(&le_bytes(n), radix);
    if n.negative && digits != "0" {
        format!("-{digits}")
    } else {
        digits
    }
}

/// Lengths in bytes through the short paths and divide and conquer, both ways.
fn lengths() -> Vec<usize> {
    [
        0, 1, 7, 8, 9, 16, 17, 100, 159, 160, 161, 400, 1000, 5200, 10_000,
    ]
    .into_iter()
    .filter(|&l| l <= size(usize::MAX, 170))
    .collect()
}

#[test]
fn writes_small_values_as_i128_does() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..size(2000, 40) {
        let x = random_i128(&mut state);
        let n = Number::new_from_i128(x, ORDERS[below(&mut state, 2)]);
        assert_eq!(n.to_string(), x.to_string());
        let m = x.unsigned_abs();
        let sign = if x < 0 { "-" } else { "" };
        assert_eq!(format!("{n:x}"), format!("{sign}{m:x}"));
        assert_eq!(format!("{n:X}"), format!("{sign}{m:X}"));
        assert_eq!(format!("{n:o}"), format!("{sign}{m:o}"));
        assert_eq!(format!("{n:b}"), format!("{sign}{m:b}"));
        for radix in 2..=36 {
            assert_eq!(
                n.to_str_radix(radix),
                reference(&n, radix),
                "{x} in radix {radix}"
            );
        }
    }
}

#[test]
fn reads_small_values_as_i128_does() {
    let mut state = 0x3C6E_F372_FE94_F82B;
    for _ in 0..size(2000, 40) {
        let x = random_i128(&mut state);
        for radix in [2, 3, 8, 10, 16, 36, 7, 32] {
            let text = Number::new_from_i128(x, ByteOrder::BigEndian).to_str_radix(radix);
            for order in ORDERS {
                let n = Number::from_str_radix(&text, radix, order).unwrap();
                assert_result(&n, &expected_i128(x), order, &|| {
                    format!("{text} in radix {radix}")
                });
            }
            assert_eq!(i128::from_str_radix(&text, radix).unwrap(), x);
        }
    }
}

#[test]
fn writes_long_values_as_the_reference() {
    let mut state = 0xBB67_AE85_84CA_A73B;
    for &len in &lengths() {
        let radices: &[u32] = if len > 2000 {
            &[10, 7]
        } else {
            &[2, 3, 7, 8, 10, 16, 32, 36]
        };
        for &radix in radices {
            let n = number(
                &random_magnitude(&mut state, len),
                below(&mut state, 2) == 0,
                ORDERS[below(&mut state, 2)],
            );
            assert_eq!(
                n.to_str_radix(radix),
                reference(&n, radix),
                "{len} bytes in radix {radix}"
            );
        }
    }
}

#[test]
fn long_text_reads_back_as_written() {
    let mut state = 0xA54F_F53A_5F1D_36F1;
    for &len in &lengths() {
        for radix in [10, 3, 16, 36] {
            // Runs of zeros and of all ones, so that blocks of zero digits and of the radix's
            // last digit occur, at the splits of divide and conquer too.
            let n = number(
                &random_bytes(&mut state, len),
                below(&mut state, 2) == 0,
                ORDERS[below(&mut state, 2)],
            );
            let text = n.to_str_radix(radix);
            for order in ORDERS {
                let back = Number::from_str_radix(&text, radix, order).unwrap();
                assert_result(&back, &value_of(&n), order, &|| {
                    format!("{len} bytes in radix {radix}")
                });
            }
        }
    }
}

#[test]
fn reads_text_as_i128_does() {
    // Signs, zeros in front, either case, and every error.
    let cases = [
        "0", "-0", "+0", "000123", "-000", "+42", "-42", "zz", "ZZ", "Zz", "", "+", "-", " 1",
        "1 ", "1_000", "--1", "+-1", "12a", "١", "1e5", "0x10", "9", "z",
    ];
    for text in cases {
        for radix in [2, 10, 16, 36] {
            let ours = Number::from_str_radix(text, radix, ByteOrder::LittleEndian);
            match i128::from_str_radix(text, radix) {
                Ok(x) => assert_result(
                    &ours.unwrap(),
                    &expected_i128(x),
                    ByteOrder::LittleEndian,
                    &|| format!("{text:?} in radix {radix}"),
                ),
                Err(e) => {
                    let ours = ours.err().unwrap();
                    let kind = match e.kind() {
                        std::num::IntErrorKind::Empty => NumberErrorKind::Empty,
                        _ => NumberErrorKind::InvalidDigit,
                    };
                    assert_eq!(*ours.kind(), kind, "{text:?} in radix {radix}");
                    assert_eq!(ours.to_string(), e.to_string());
                }
            }
        }
    }
    let long = format!("-{}", "9".repeat(size(30_000, 300)));
    let n = Number::from_str_radix(&long, 10, ByteOrder::BigEndian).unwrap();
    assert_eq!(n.to_string(), long);
}

#[test]
fn formatting_flags_apply_as_to_primitive_integers() {
    for x in [0i128, 7, -7, 255, -255, 1 << 100, -(1 << 100)] {
        let n = Number::new_from_i128(x, ByteOrder::LittleEndian);
        let m = x.unsigned_abs();
        let sign = if x < 0 { "-" } else { "" };
        assert_eq!(format!("{n:>40}"), format!("{x:>40}"));
        assert_eq!(format!("{n:<40}|"), format!("{x:<40}|"));
        assert_eq!(format!("{n:^41}"), format!("{x:^41}"));
        assert_eq!(format!("{n:*>40}"), format!("{x:*>40}"));
        assert_eq!(format!("{n:+}"), format!("{x:+}"));
        assert_eq!(format!("{n:040}"), format!("{x:040}"));
        assert_eq!(format!("{n:#x}"), format!("{sign}{m:#x}"));
        assert_eq!(format!("{n:#X}"), format!("{sign}{m:#X}"));
        assert_eq!(format!("{n:#o}"), format!("{sign}{m:#o}"));
        assert_eq!(format!("{n:#b}"), format!("{sign}{m:#b}"));
        assert_eq!(
            format!("{n:#040x}"),
            format!("{sign}{m:#0w$x}", w = 40 - sign.len())
        );
    }
}

#[test]
fn zero_in_every_form_is_written_0() {
    for order in ORDERS {
        for z in zeros(order) {
            assert_eq!(z.to_string(), "0");
            assert_eq!(format!("{z:+}"), "+0");
            assert_eq!(z.to_str_radix(2), "0");
            assert_eq!(format!("{z:#x}"), "0x0");
        }
    }
    let padded = number(
        &[5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        true,
        ByteOrder::BigEndian,
    );
    assert_eq!(padded.to_string(), "-5");
}

#[test]
fn radices_outside_2_to_36_panic() {
    let n = Number::new_from_u8(5, ByteOrder::BigEndian);
    for radix in [0, 1, 37, 100] {
        let message = panic_message(|| drop(n.to_str_radix(radix)));
        assert!(message.contains("[2, 36]"), "{message}");
        let message =
            panic_message(|| drop(Number::from_str_radix("1", radix, ByteOrder::BigEndian)));
        assert!(message.contains("[2, 36]"), "{message}");
    }
}

// The paths of the implementation: blocks, thresholds and edges

/// The magnitude, least significant byte first, of the digits `text` in `radix`, worked out
/// independently of the implementation: one digit at a time into 32-bit limbs. Empty for zero.
fn reference_value(text: &str, radix: u32) -> Vec<u8> {
    let mut limbs: Vec<u32> = Vec::new();
    for c in text.chars() {
        let mut carry = u64::from(c.to_digit(radix).unwrap());
        for l in limbs.iter_mut() {
            let t = u64::from(*l) * u64::from(radix) + carry;
            *l = t as u32;
            carry = t >> 32;
        }
        if carry != 0 {
            limbs.push(carry as u32);
        }
    }
    let mut bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    bytes
}

/// `radix^d - 1`, `radix^d` and `radix^d + 1` as digit strings: the largest number of `d` digits
/// and the first ones of `d + 1`.
fn around_power(radix: u32, d: usize) -> [String; 3] {
    let last = char::from(DIGITS[radix as usize - 1]);
    let one = |tail: char| format!("1{}{tail}", "0".repeat(d - 1));
    [last.to_string().repeat(d), one('0'), one('1')]
}

/// `len` random digits in `radix`, in runs of zeros, of the radix's last digit and of random ones,
/// the first not zero.
fn random_digits(state: &mut u64, radix: u32, len: usize) -> String {
    let mut text = String::with_capacity(len);
    while text.len() < len {
        let run = (1 + below(state, 30)).min(len - text.len());
        let kind = below(state, 4);
        for _ in 0..run {
            let d = match kind {
                0 => 0,
                1 => radix as usize - 1,
                _ => below(state, radix as usize),
            };
            text.push(char::from(DIGITS[d]));
        }
    }
    if text.starts_with('0') {
        text.replace_range(0..1, "1");
    }
    text
}

/// Checks that `text` in `radix` reads as its reference value in both byte orders, and writes back
/// as itself, without the zeros in front.
fn assert_round_trip(text: &str, radix: u32) {
    let magnitude = reference_value(text, radix);
    let expected = expected(&magnitude, false);
    for order in ORDERS {
        let n = Number::from_str_radix(text, radix, order).unwrap();
        assert_result(&n, &expected, order, &|| {
            format!("{} digits in radix {radix}", text.len())
        });
    }
    let n = number(&magnitude, false, ORDERS[text.len() % 2]);
    let written = n.to_str_radix(radix);
    assert_eq!(
        written,
        text.trim_start_matches('0'),
        "{} digits in radix {radix}",
        text.len()
    );
}

#[test]
fn every_decimal_length_and_the_powers_of_ten_round_trip() {
    // Up to 120 digits, every length: one block and two in registers, the head of every length
    // in front of whole blocks, and the powers of ten around each length, whose blocks are all
    // nines or all zeros.
    let mut state = 0x243F_6A88_85A3_08D3;
    for d in 1..=size(120, 45) {
        if cfg!(miri) && below(&mut state, 3) != 0 {
            continue;
        }
        assert_round_trip(&random_digits(&mut state, 10, d), 10);
        for text in around_power(10, d) {
            assert_round_trip(&text, 10);
        }
    }
}

#[test]
fn writes_the_edges_of_two_limbs_as_u128_does() {
    // Up to 39 digits in registers: a limb, a quotient by 10^19 of one limb or of 65 bits, whose
    // first digit is 1, 2 or 3.
    let ten19 = 10_000_000_000_000_000_000u128;
    let mut values = vec![
        0,
        1,
        ten19 - 1,
        ten19,
        u128::from(u64::MAX),
        u128::from(u64::MAX) + 1,
        ten19 * ten19 - 1,
        ten19 * ten19,
        (ten19 << 64) - 1,
        ten19 << 64,
        u128::MAX,
    ];
    for t in 1..=3 {
        let x = t * ten19 * ten19;
        values.extend([x - 1, x, x + 1]);
    }
    for v in values {
        for order in ORDERS {
            let n = Number::new_from_u128(v, order);
            assert_eq!(n.to_string(), v.to_string());
            assert_eq!(n.to_str_radix(10), v.to_string());
            let back = Number::from_str_radix(&v.to_string(), 10, order).unwrap();
            assert_eq!(back.value, n.value, "{v}");
        }
    }
}

#[test]
fn every_radix_round_trips_at_the_block_edges() {
    // In every radix, lengths of one block up to a few, the powers of the radix around each, and
    // text long enough for the groups of blocks.
    let mut state = 0x1319_8A2E_0370_7344;
    for radix in 2..=36 {
        let (_, k) = block_of(radix);
        let mut lengths: Vec<usize> = (1..=3 * k + 1).collect();
        lengths.extend([8 * k - 1, 8 * k, 8 * k + 1, 20 * k + 3]);
        for d in lengths {
            if cfg!(miri) && below(&mut state, 12) != 0 {
                continue;
            }
            assert_round_trip(&random_digits(&mut state, radix, d), radix);
            for text in around_power(radix, d) {
                assert_round_trip(&text, radix);
            }
        }
    }
}

/// The largest power of `radix` in a limb and its exponent, as the reference counts them.
fn block_of(radix: u32) -> (u64, usize) {
    let (mut b, mut k) = (u64::from(radix), 1);
    while let Some(next) = b.checked_mul(u64::from(radix)) {
        b = next;
        k += 1;
    }
    (b, k)
}

#[test]
fn long_text_round_trips_around_the_thresholds() {
    // Text around the limbs read on the stack (64 blocks of 19 digits) and around divide and
    // conquer both ways (220 blocks read, 80 limbs written), in decimal and in radices 3 and 36.
    let mut state = 0x4528_21E6_38D0_1377;
    let decimal = [
        1215, 1216, 1217, 1235, 1236, 4179, 4180, 4181, 4199, 4200, 1541, 1542, 1560,
    ];
    for d in decimal {
        if cfg!(miri) && below(&mut state, 6) != 0 {
            continue;
        }
        assert_round_trip(&random_digits(&mut state, 10, d), 10);
    }
    for radix in [3, 36] {
        let (_, k) = block_of(radix);
        for blocks in [64, 65, 219, 220, 221] {
            if cfg!(miri) && below(&mut state, 6) != 0 {
                continue;
            }
            assert_round_trip(&random_digits(&mut state, radix, blocks * k), radix);
        }
    }
}

#[test]
fn writes_every_limb_count_through_the_groups_of_divisions() {
    // Magnitudes of every length up to 90 limbs, whose divisions run in groups of 1, 2, 4 and 8,
    // up to divide and conquer, with runs of zeros and of ones, in decimal, in radix 7, and in
    // radix 36, whose divisions correct their quotients by selects.
    let mut state = 0xBE54_66CF_34E9_0C6C;
    for limbs in 1..=size(90, 12) {
        for extra in [0, 5] {
            let len = 8 * limbs - extra;
            let n = number(
                &random_magnitude(&mut state, len),
                below(&mut state, 2) == 0,
                ORDERS[below(&mut state, 2)],
            );
            for radix in [10, 7, 36] {
                if cfg!(miri) && below(&mut state, 4) != 0 {
                    continue;
                }
                assert_eq!(
                    n.to_str_radix(radix),
                    reference(&n, radix),
                    "{len} bytes in radix {radix}"
                );
            }
        }
    }
}

#[test]
fn a_character_that_is_not_a_digit_is_found_at_every_place() {
    // In every place of text from 1 to 45 digits, whose decimal blocks are checked eight bytes at
    // a time, characters next to the digits in ASCII, letters, signs, spaces, a byte of 0x7F and
    // a character of two bytes; and the radix's first digit too large in other radices.
    let mut state = 0x9216_D5D9_8979_FB1B;
    let wrong = [
        "/", ":", "a", "A", " ", "_", "+", "-", "\u{7F}", "é", "\u{0}",
    ];
    for len in 1..=size(45, 22) {
        let digits = random_digits(&mut state, 10, len);
        for at in 0..len {
            for w in wrong {
                if cfg!(miri) && below(&mut state, 40) != 0 {
                    continue;
                }
                let mut text = digits.clone();
                text.replace_range(at..=at, w);
                let error = Number::from_str_radix(&text, 10, ORDERS[at % 2]).err();
                // A sign in front reads; anywhere else it is a wrong digit.
                if at == 0 && (w == "+" || w == "-") && len > 1 {
                    assert!(error.is_none(), "{text:?}");
                } else {
                    assert_eq!(
                        error.map(|e| *e.kind()),
                        Some(NumberErrorKind::InvalidDigit),
                        "{text:?}"
                    );
                }
            }
        }
    }
    for radix in [2, 3, 8, 16, 32, 36] {
        for len in [1, 2, 9, 30, 70] {
            let digits = random_digits(&mut state, radix, len);
            for at in [0, len / 2, len - 1] {
                let mut text = digits.clone();
                let w = if radix < 36 {
                    char::from(DIGITS[radix as usize]).to_string()
                } else {
                    "@".to_string()
                };
                text.replace_range(at..=at, &w);
                let error = Number::from_str_radix(&text, radix, ORDERS[at % 2]).err();
                assert_eq!(
                    error.map(|e| *e.kind()),
                    Some(NumberErrorKind::InvalidDigit),
                    "{text:?} in radix {radix}"
                );
            }
        }
    }
}

#[test]
fn powers_of_two_read_with_zeros_in_front_and_either_case() {
    // Every length up to 40 bytes and some longer, zeros in front, upper-case letters, and a top
    // digit of each bit length, in the radices of 1 to 5 bits.
    let mut state = 0xC0AC_29B7_C97C_50DD;
    for radix in [2, 4, 8, 16, 32] {
        for len in (1..=size(40, 10)).chain([100, 257]) {
            if cfg!(miri) && below(&mut state, 3) != 0 {
                continue;
            }
            let n = number(
                &random_magnitude(&mut state, len),
                false,
                ByteOrder::BigEndian,
            );
            let text = n.to_str_radix(radix);
            assert_eq!(text, reference(&n, radix));
            for top in 1..radix {
                let mut text = text.clone();
                text.replace_range(0..1, &char::from(DIGITS[top as usize]).to_string());
                let padded = format!("{}{}", "0".repeat(below(&mut state, 20)), text);
                assert_round_trip(&padded, radix);
                let upper = padded.to_ascii_uppercase();
                for order in ORDERS {
                    let a = Number::from_str_radix(&padded, radix, order).unwrap();
                    let b = Number::from_str_radix(&upper, radix, order).unwrap();
                    assert_eq!(a.value, b.value);
                }
            }
        }
    }
}

#[test]
fn formatting_long_and_short_numbers_agrees_with_to_str_radix() {
    // Text up to 160 digits is formatted from a buffer on the stack, longer text from a vector:
    // lengths around it in every radix of the formatting traits, with flags.
    let mut state = 0x6A09_E667_BB67_AE85;
    for len in [1, 8, 17, 60, 64, 65, 66, 67, 68, 80, 100] {
        let n = number(
            &random_magnitude(&mut state, len),
            below(&mut state, 2) == 0,
            ORDERS[below(&mut state, 2)],
        );
        let sign = if n.is_negative() { "-" } else { "" };
        let digits = |radix| n.to_str_radix(radix).trim_start_matches('-').to_string();
        assert_eq!(n.to_string(), n.to_str_radix(10));
        assert_eq!(format!("{n:x}"), format!("{sign}{}", digits(16)));
        assert_eq!(
            format!("{n:X}"),
            format!("{sign}{}", digits(16).to_ascii_uppercase())
        );
        assert_eq!(format!("{n:o}"), format!("{sign}{}", digits(8)));
        assert_eq!(format!("{n:b}"), format!("{sign}{}", digits(2)));
        let wide = n.to_string().len() + 3;
        assert_eq!(
            format!("{n:>w$}", w = wide),
            format!("{:>w$}", n.to_string(), w = wide)
        );
        assert_eq!(n.to_str_radix(10), reference(&n, 10));
    }
}
