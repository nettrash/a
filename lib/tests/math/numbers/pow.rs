use a::math::numbers::{ByteOrder, Number};

use super::common::*;
use super::div::{mul_magnitudes, panic_message, quotient_and_remainder};
use crate::allocations::count;

// Expected powers

/// The magnitude `x^e`, given least significant byte first, worked out independently of the
/// implementation: `e` products by the reference schoolbook multiplication on 32-bit limbs, the
/// operand and each product cut to their significant bytes.
fn pow_magnitude(x: &[u8], e: u32) -> Vec<u8> {
    let x = significant(x);
    let mut p = vec![1u8];
    for _ in 0..e {
        p = significant(&mul_magnitudes(&p, &x));
    }
    p
}

/// `x` without the zero bytes above its most significant one: empty for zero.
fn significant(x: &[u8]) -> Vec<u8> {
    let top = x.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    x[..top].to_vec()
}

/// The remainder of the magnitude `x` by the magnitude `m`, by the reference division.
fn rem_magnitude(x: &[u8], m: &[u8]) -> Vec<u8> {
    let (_, r) = quotient_and_remainder(
        &number(x, false, ByteOrder::LittleEndian),
        &number(m, false, ByteOrder::LittleEndian),
    );
    significant(&r.magnitude)
}

/// `b^e mod m` for magnitudes, least significant byte first, `m` not zero: binary exponentiation
/// from the top bit of `e` with the reference product and division.
fn pow_mod_magnitude(b: &[u8], e: &[u8], m: &[u8]) -> Vec<u8> {
    let b = rem_magnitude(b, m);
    let mut r = rem_magnitude(&[1], m);
    for i in (0..8 * e.len()).rev() {
        r = rem_magnitude(&mul_magnitudes(&r, &r), m);
        if (e[i / 8] >> (i % 8)) & 1 == 1 {
            r = rem_magnitude(&mul_magnitudes(&r, &b), m);
        }
    }
    r
}

/// What `pow_mod` must give for `b`, `e` and `m` as numbers: `|b|^e mod |m|`, and `|m|` less that
/// for a negative `b` and an odd `e`, unless it is zero.
fn expected_pow_mod(b: &Number, e: &Number, m: &Number) -> Expected {
    let (bm, em, mm) = (le_bytes(b), le_bytes(e), le_bytes(m));
    let r = pow_mod_magnitude(&bm, &em, &mm);
    let odd = em.first().is_some_and(|&x| x & 1 == 1);
    if b.negative && odd && !r.is_empty() {
        return expected(&sub_le(&significant(&mm), &r), false);
    }
    expected(&r, false)
}

/// `x - y` for magnitudes least significant byte first, `x >= y`.
fn sub_le(x: &[u8], y: &[u8]) -> Vec<u8> {
    let mut borrow = 0i16;
    x.iter()
        .enumerate()
        .map(|(i, &a)| {
            let t = i16::from(a) - i16::from(*y.get(i).unwrap_or(&0)) - borrow;
            borrow = i16::from(t < 0);
            t.rem_euclid(256) as u8
        })
        .collect()
}

/// A copy of `n` stored as it is, high zero bytes and sign of zero included.
fn same_copy(n: &Number) -> Number {
    Number {
        value: n.value.clone(),
        negative: n.negative,
        byte_order: n.byte_order,
    }
}

// Powers

#[test]
fn powers_of_small_numbers_follow_i128() {
    let mut state = 0x243F6A88_85A308D3;
    for x in -40i128..=40 {
        for e in 0..size(40, 6) as u32 {
            let p = x.checked_pow(e);
            let Some(p) = p.filter(|p| p.unsigned_abs() < 1 << 126) else {
                continue;
            };
            let order = ORDERS[below(&mut state, 2)];
            let n = number(&x.unsigned_abs().to_le_bytes(), x < 0, order);
            let result = n.pow(e);
            let inputs = || format!("{} pow {e}", describe(&n));
            assert_result(&result, &expected_i128(p), order, &inputs);
        }
    }
    // 0^0 = 1, as for `i32`, and so is every other number to the power 0.
    for order in ORDERS {
        for z in zeros(order) {
            assert_result(&z.pow(0), &expected_i128(1), order, &|| "0^0".into());
            assert_result(&z.pow(7), &expected_i128(0), order, &|| "0^7".into());
        }
    }
}

#[test]
fn powers_match_repeated_products() {
    let mut state = 0x13198A2E_03707344;
    // Bases in registers and on limbs, exponents from the register path to long results, odd and
    // even bases, all ones, powers of two and numbers with zero limbs at the low end.
    let lengths = [1, 2, 7, 8, 9, 15, 16, 17, 24, 33, 64, 100, 300];
    for &len in lengths.iter().filter(|&&l| l <= size(300, 9)) {
        for kind in 0..5 {
            let mut x = match kind {
                0 => vec![0xFF; len],
                1 => {
                    let mut v = vec![0; len];
                    v[len - 1] = 1 << below(&mut state, 8);
                    v
                }
                _ => random_magnitude(&mut state, len),
            };
            if kind == 3 {
                let zeros = below(&mut state, len);
                x[..zeros].fill(0);
            }
            let max_e = (size(4000, 40) / len).max(2);
            for e in [0, 1, 2, 3, below(&mut state, max_e) as u32, max_e as u32] {
                let order = ORDERS[below(&mut state, 2)];
                let negative = below(&mut state, 2) == 0;
                let mut stored_x = x.clone();
                if below(&mut state, 3) == 0 {
                    stored_x.resize(len + 1 + below(&mut state, 9), 0);
                }
                let n = number(&stored_x, negative, order);
                let result = n.pow(e);
                let inputs = || format!("{} pow {e}", describe(&n));
                let p = pow_magnitude(&x, e);
                assert_result(
                    &result,
                    &expected(&p, negative && e % 2 == 1),
                    order,
                    &inputs,
                );
                assert!(
                    same(&n, &number(&stored_x, negative, order)),
                    "{}: the operand changed",
                    inputs()
                );
            }
        }
    }
}

#[test]
fn powers_of_one_and_two_and_long_exponents() {
    for order in ORDERS {
        let one = number(&[1, 0, 0], false, order);
        let minus_one = number(&[1], true, order);
        for e in [0, 1, 2, 3, 1000, 1001, u32::MAX] {
            assert_result(&one.pow(e), &expected_i128(1), order, &|| "1^e".into());
            let sign = if e % 2 == 1 { -1 } else { 1 };
            assert_result(&minus_one.pow(e), &expected_i128(sign), order, &|| {
                "(-1)^e".into()
            });
        }
        // 2^k and -2^k: a one followed by zero bits, through the shift that puts back the base's
        // zero bits.
        for k in [
            1,
            7,
            8,
            63,
            64,
            65,
            127,
            128,
            129,
            1000,
            size(100_000, 2000),
        ] {
            let two = number(&[2], true, order);
            let mut m = vec![0u8; k / 8 + 1];
            m[k / 8] = 1 << (k % 8);
            let inputs = || format!("(-2)^{k}");
            assert_result(
                &two.pow(k as u32),
                &expected(&m, k % 2 == 1),
                order,
                &inputs,
            );
        }
        // Powers of `±2^z` and of 1, however long, allocate their result and nothing else.
        for x in [number(&[1], true, order), number(&[0, 0, 4], false, order)] {
            let (p, allocations) = count(|| x.pow(size(1_000_000, 10_000) as u32));
            assert_eq!(
                allocations,
                1,
                "{} pow: {} bytes",
                describe(&x),
                p.value.len()
            );
        }
        // 6^e = 2^e 3^e: an odd part and zero bits.
        let six = number(&[6], false, order);
        let e = size(500, 60) as u32;
        assert_result(
            &six.pow(e),
            &expected(&pow_magnitude(&[6], e), false),
            order,
            &|| "6^e".into(),
        );
    }
}

#[test]
fn powers_in_registers_and_just_beyond() {
    // Results around 16 bytes, the register path's limit: 2^126 .. 2^130 and their odd cousins.
    let mut state = 0xA4093822_299F31D0;
    for bits in 1..=size(128, 40) {
        for e in 1..=size(140, 20) as u32 {
            // Up to 512 bits in registers, and just beyond, on limbs.
            if bits as u64 * u64::from(e) > size(600, 300) as u64 {
                continue;
            }
            let mut x = vec![0u8; bits.div_ceil(8)];
            for (i, byte) in x.iter_mut().enumerate() {
                *byte = xorshift(&mut state) as u8 | 1;
                if 8 * i + 8 > bits {
                    *byte &= (1u16 << (bits - 8 * i)).wrapping_sub(1) as u8;
                    *byte |= 1 << ((bits - 1) % 8);
                }
            }
            let order = ORDERS[below(&mut state, 2)];
            let n = number(&x, false, order);
            let p = pow_magnitude(&x, e);
            assert_result(&n.pow(e), &expected(&p, false), order, &|| {
                format!("{} pow {e}", describe(&n))
            });
        }
    }
}

#[test]
fn powers_too_long_to_allocate_panic() {
    // 2^19 bits to the power u32::MAX: 2^48 bytes, more than the address space; the power panics
    // instead of aborting on the failed allocation. Under Miri, which would try to run the
    // allocation, the panic is not checked.
    if cfg!(miri) {
        return;
    }
    let x = number(&vec![0xFF; 1 << 16], false, ByteOrder::BigEndian);
    let message = panic_message(|| drop(x.pow(u32::MAX)));
    assert!(message.contains("too long"), "{message}");
}

// Powers modulo a number

#[test]
fn powers_modulo_small_numbers_follow_u128() {
    let mut state = 0x082EFA98_EC4E6C89;
    let mut cases = 0;
    while cases < size(30_000, 40) {
        let m = match below(&mut state, 4) {
            0 => 1 + below(&mut state, 300) as u128,
            1 => u128::from(xorshift(&mut state) >> below(&mut state, 64)),
            _ => {
                (u128::from(xorshift(&mut state)) << 64 | u128::from(xorshift(&mut state)))
                    >> below(&mut state, 128)
            }
        };
        if m == 0 {
            continue;
        }
        cases += 1;
        let b = (u128::from(xorshift(&mut state)) << 64 | u128::from(xorshift(&mut state)))
            >> below(&mut state, 128);
        let e = (u128::from(xorshift(&mut state)) << 64 | u128::from(xorshift(&mut state)))
            >> below(&mut state, 128);
        let (ob, oe, om) = (
            ORDERS[below(&mut state, 2)],
            ORDERS[below(&mut state, 2)],
            ORDERS[below(&mut state, 2)],
        );
        let (nb, nm) = (below(&mut state, 2) == 0, below(&mut state, 2) == 0);
        let bn = number(&b.to_le_bytes(), nb, ob);
        let en = number(&e.to_le_bytes(), false, oe);
        let mn = number(&m.to_le_bytes(), nm, om);
        let result = bn.pow_mod(&en, &mn);
        let inputs = || {
            format!(
                "{} pow_mod({}, {})",
                describe(&bn),
                describe(&en),
                describe(&mn)
            )
        };
        assert_result(&result, &expected_pow_mod(&bn, &en, &mn), ob, &inputs);
    }
}

/// A modulus of `len` bytes: odd, even, a power of two, all ones, or with zero limbs at its low
/// end.
fn modulus(state: &mut u64, len: usize, kind: usize) -> Vec<u8> {
    let mut m = match kind {
        0 => vec![0xFF; len],
        1 => {
            let mut v = vec![0; len];
            v[len - 1] = 1 << below(state, 8);
            v
        }
        _ => random_magnitude(state, len),
    };
    match kind {
        2 => m[0] |= 1,
        3 => m[0] &= !1,
        4 => {
            let zeros = below(state, len);
            m[..zeros].fill(0);
            m[zeros] |= 1;
        }
        _ => {}
    }
    m
}

#[test]
fn powers_modulo_long_numbers_match_the_reference() {
    let mut state = 0x452821E6_38D01377;
    // One and two limbs in registers, three to seven, the limbs from eight, and REDC by products
    // from 100 limbs (800 bytes); exponents across the window sizes' limits.
    let lengths = [
        1, 8, 9, 16, 17, 24, 31, 56, 57, 64, 65, 100, 200, 792, 793, 799, 800, 801, 900,
    ];
    for &ml in lengths.iter().filter(|&&l| l <= size(900, 24)) {
        for kind in 0..5 {
            let m = modulus(&mut state, ml, kind);
            let el = match below(&mut state, 4) {
                0 => 1 + below(&mut state, 2),
                1 => 1 + below(&mut state, 4),
                _ => 1 + below(&mut state, (size(3000, 40) / ml).clamp(1, 40)),
            };
            let bl = match below(&mut state, 3) {
                0 => ml + 1 + below(&mut state, ml),
                1 => 1 + below(&mut state, 8),
                _ => ml,
            };
            let b = random_magnitude(&mut state, bl);
            let e = random_magnitude(&mut state, el);
            let (ob, oe, om) = (
                ORDERS[below(&mut state, 2)],
                ORDERS[below(&mut state, 2)],
                ORDERS[below(&mut state, 2)],
            );
            let (nb, nm) = (below(&mut state, 2) == 0, below(&mut state, 2) == 0);
            let bn = number(&b, nb, ob);
            let en = number(&e, false, oe);
            let mn = number(&m, nm, om);
            let result = bn.pow_mod(&en, &mn);
            let inputs = || {
                format!(
                    "{} pow_mod({}, {})",
                    describe(&bn),
                    describe(&en),
                    describe(&mn)
                )
            };
            assert_result(&result, &expected_pow_mod(&bn, &en, &mn), ob, &inputs);
        }
    }
}

#[test]
fn window_sizes_and_long_exponents() {
    let mut state = 0xBE5466CF_34E90C6C;
    // Exponents of 1 to 3500 bits, around every window size's limit, modulo numbers of one limb
    // to nine.
    let bits = [
        1, 2, 7, 8, 25, 26, 81, 82, 241, 242, 673, 674, 1793, 1794, 3500,
    ];
    for &eb in bits.iter().filter(|&&b| b <= size(3500, 30)) {
        for ml in [8, 16, 24, 72] {
            let m = modulus(&mut state, ml, 2);
            let b = random_magnitude(&mut state, ml);
            let mut e = random_magnitude(&mut state, eb.div_ceil(8));
            let top = e.len() - 1;
            let top_bits = eb - 8 * top;
            e[top] = (e[top] & ((1u16 << top_bits) - 1) as u8) | (1 << (top_bits - 1));
            let (bn, en, mn) = (
                number(&b, false, ByteOrder::BigEndian),
                number(&e, false, ByteOrder::LittleEndian),
                number(&m, false, ByteOrder::LittleEndian),
            );
            let result = bn.pow_mod(&en, &mn);
            let inputs = || format!("{eb}-bit exponent modulo {ml} bytes");
            assert_result(
                &result,
                &expected_pow_mod(&bn, &en, &mn),
                ByteOrder::BigEndian,
                &inputs,
            );
        }
    }
}

#[test]
fn powers_modulo_one_zero_and_small_cases() {
    for order in ORDERS {
        let five = number(&[5], false, order);
        let one = number(&[1], true, order);
        // x^0 mod 1 = 0, and x^0 mod m = 1 for m > 1, x = 0 included.
        for z in zeros(order) {
            assert_result(&z.pow_mod(&z, &one), &expected_i128(0), order, &|| {
                "0^0 mod 1".into()
            });
            assert_result(&z.pow_mod(&z, &five), &expected_i128(1), order, &|| {
                "0^0 mod 5".into()
            });
            assert_result(&five.pow_mod(&z, &five), &expected_i128(1), order, &|| {
                "5^0 mod 5".into()
            });
            assert_result(
                &z.pow_mod(&number(&[3], false, order), &five),
                &expected_i128(0),
                order,
                &|| "0^3 mod 5".into(),
            );
        }
        // Long operands, zero exponent, modulus 1.
        let long = number(&[0xAB; 300], true, order);
        let long_one = number(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], false, order);
        let long_e = number(&[0xCD; 300], false, order);
        assert_result(
            &long.pow_mod(&long_e, &long_one),
            &expected_i128(0),
            order,
            &|| "long mod 1".into(),
        );
        assert_result(
            &long.pow_mod(&number(&[], true, order), &long),
            &expected_i128(1),
            order,
            &|| "long^0 mod long".into(),
        );
        // A negative base to an odd power, a multiple of the modulus, and a base above it.
        let m = number(&[7, 3, 0xF1], true, order);
        for (b, e) in [
            (vec![2u8], vec![1u8]),
            (vec![2], vec![2]),
            (vec![1, 0xFF], vec![3]),
        ] {
            for negative in [false, true] {
                let bn = number(&b, negative, order);
                let en = number(&e, false, order);
                assert_result(
                    &bn.pow_mod(&en, &m),
                    &expected_pow_mod(&bn, &en, &m),
                    order,
                    &|| format!("{} pow_mod {e:?}", describe(&bn)),
                );
            }
        }
        let multiple = number(&mul_magnitudes(&[7, 3, 0xF1], &[9, 9]), true, order);
        assert_result(
            &multiple.pow_mod(&number(&[5], false, order), &m),
            &expected_i128(0),
            order,
            &|| "multiple".into(),
        );
    }
}

#[test]
fn powers_modulo_powers_of_two_and_even_numbers() {
    let mut state = 0xC0AC29B7_C97C50DD;
    // `2^z` alone, in registers up to two limbs and to eight and in limbs beyond, and `m' 2^z` with
    // odd parts of one limb to many. Even bases whose zero bits fill `2^z`, and those whose do not.
    let zs = [
        1, 2, 63, 64, 65, 127, 128, 129, 191, 192, 256, 300, 383, 448, 512, 513, 700, 2560, 2561,
    ];
    for z in zs.into_iter().filter(|&z| z <= size(2600, 260)) {
        for odd_len in [0, 1, 8, 9, 30] {
            for even_base in [false, true] {
                let mut m = vec![0u8; z / 8 + 1];
                m[z / 8] = 1 << (z % 8);
                if odd_len > 0 {
                    let mut odd = random_magnitude(&mut state, odd_len);
                    odd[0] |= 1;
                    m = mul_magnitudes(&m, &odd);
                }
                let bl = 1 + below(&mut state, 40);
                let mut b = random_magnitude(&mut state, bl);
                if even_base {
                    b[0] &= !1;
                    b[0] |= 2;
                } else {
                    b[0] |= 1;
                }
                let el = 1 + below(&mut state, size(12, 2));
                let e = random_magnitude(&mut state, el);
                let order = ORDERS[below(&mut state, 2)];
                let (bn, en, mn) = (
                    number(&b, false, order),
                    number(&e, false, order),
                    number(&m, false, order),
                );
                assert_result(
                    &bn.pow_mod(&en, &mn),
                    &expected_pow_mod(&bn, &en, &mn),
                    order,
                    &|| format!("z = {z}, odd part {odd_len} bytes, even base {even_base}"),
                );
            }
        }
    }
}

#[test]
fn powers_modulo_keep_the_operands_and_take_the_base_s_order() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    let b = number(
        &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 0,
        ],
        true,
        be,
    );
    let e = number(&[0x11, 0x22, 0, 0], false, le);
    let m = number(
        &[
            0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
            0x11, 0x01, 0x07, 0x0B,
        ],
        true,
        le,
    );
    let (b0, e0, m0) = (same_copy(&b), same_copy(&e), same_copy(&m));
    let r = b.pow_mod(&e, &m);
    assert!(r.byte_order == be);
    assert!(!r.negative);
    assert_result(&r, &expected_pow_mod(&b, &e, &m), be, &|| "orders".into());
    assert!(
        same(&b, &b0) && same(&e, &e0) && same(&m, &m0),
        "an operand changed"
    );
}

#[test]
fn powers_modulo_zero_panic() {
    let b = number(&[3], false, ByteOrder::BigEndian);
    let e = number(&[3], false, ByteOrder::BigEndian);
    for order in ORDERS {
        for z in zeros(order) {
            assert_eq!(
                panic_message(|| drop(b.pow_mod(&e, &z))),
                "attempt to calculate the remainder with a divisor of zero"
            );
            let long = number(&[7; 40], false, order);
            assert_eq!(
                panic_message(|| drop(long.pow_mod(&long, &z))),
                "attempt to calculate the remainder with a divisor of zero"
            );
        }
    }
}

#[test]
fn negative_exponents_panic() {
    let b = number(&[3], false, ByteOrder::BigEndian);
    let m = number(&[7], false, ByteOrder::LittleEndian);
    for e in [
        number(&[1], true, ByteOrder::BigEndian),
        number(&[5; 40], true, ByteOrder::LittleEndian),
    ] {
        assert_eq!(
            panic_message(|| drop(b.pow_mod(&e, &m))),
            "negative exponent"
        );
    }
    // A negative zero is zero, not negative.
    let z = number(&[0, 0], true, ByteOrder::BigEndian);
    assert_result(
        &b.pow_mod(&z, &m),
        &expected_i128(1),
        ByteOrder::BigEndian,
        &|| "3^-0 mod 7".into(),
    );
}
