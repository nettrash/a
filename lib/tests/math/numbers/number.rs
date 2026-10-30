use a::math::numbers::{ByteOrder, Number};

const ORDERS: [ByteOrder; 2] = [ByteOrder::BigEndian, ByteOrder::LittleEndian];

/// Small deterministic xorshift generator, so the tests need no extra dependencies.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A random `u128` of a random bit length from 0 to 128, so every byte length occurs.
fn random_u128(state: &mut u64) -> u128 {
    let full = ((xorshift(state) as u128) << 64) | xorshift(state) as u128;
    let bits = (xorshift(state) % 129) as u32;
    full.checked_shr(128 - bits).unwrap_or(0)
}

/// A random `i128` of a random bit length and sign; negative values reach one further than
/// positive ones, as `MIN` does.
fn random_i128(state: &mut u64) -> i128 {
    let magnitude = random_u128(state) >> 1;
    match xorshift(state) % 3 {
        0 => magnitude as i128,
        1 => -(magnitude as i128),
        _ => -(magnitude as i128) - 1,
    }
}

/// Up to 11 random bytes, mostly zeros, so that runs of zeros occur at both ends and inside.
fn random_bytes(state: &mut u64) -> Vec<u8> {
    let len = (xorshift(state) % 12) as usize;
    (0..len)
        .map(|_| match xorshift(state) % 3 {
            0 => xorshift(state) as u8,
            _ => 0,
        })
        .collect()
}

/// The big-endian `bytes` without their leading zeros, keeping one byte for zero: the value
/// a big-endian `Number` should hold, worked out independently of `Number::new`.
fn trimmed(bytes: &[u8]) -> Vec<u8> {
    let start = bytes
        .iter()
        .position(|&b| b != 0)
        .unwrap_or(bytes.len() - 1);
    bytes[start..].to_vec()
}

fn reversed(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().rev().copied().collect()
}

// `new`

#[test]
fn new_keeps_bytes_without_zeros() {
    for order in ORDERS {
        let n = Number::new(vec![1, 2, 3], order);
        assert_eq!(n.value, vec![1, 2, 3]);
    }
}

#[test]
fn new_big_endian_drops_leading_zeros() {
    let n = Number::new(vec![0, 0, 1, 2], ByteOrder::BigEndian);
    assert_eq!(n.value, vec![1, 2]);
}

#[test]
fn new_little_endian_drops_trailing_zeros() {
    let n = Number::new(vec![1, 2, 0, 0], ByteOrder::LittleEndian);
    assert_eq!(n.value, vec![1, 2]);
}

#[test]
fn new_keeps_zeros_at_the_least_significant_end() {
    let big = Number::new(vec![1, 0, 0], ByteOrder::BigEndian);
    assert_eq!(big.value, vec![1, 0, 0]);
    let little = Number::new(vec![0, 0, 1], ByteOrder::LittleEndian);
    assert_eq!(little.value, vec![0, 0, 1]);
}

#[test]
fn new_keeps_inner_zeros() {
    let big = Number::new(vec![0, 1, 0, 2], ByteOrder::BigEndian);
    assert_eq!(big.value, vec![1, 0, 2]);
    let little = Number::new(vec![2, 0, 1, 0], ByteOrder::LittleEndian);
    assert_eq!(little.value, vec![2, 0, 1]);
}

#[test]
fn new_keeps_one_byte_of_zero() {
    for order in ORDERS {
        assert_eq!(Number::new(vec![0], order).value, vec![0]);
        assert_eq!(Number::new(vec![0, 0, 0], order).value, vec![0]);
    }
}

#[test]
fn new_keeps_empty_vector_empty() {
    for order in ORDERS {
        assert!(Number::new(vec![], order).value.is_empty());
    }
}

#[test]
fn new_stores_byte_order() {
    for order in ORDERS {
        let n = Number::new(vec![1], order);
        assert!(n.byte_order() == order);
        assert!(n.byte_order == order);
    }
}

#[test]
fn new_reuses_allocation() {
    // Byte 32 of 64 is the only non-zero one: big-endian keeps it and the 31 bytes after it,
    // little-endian keeps it and the 32 bytes before it.
    for (order, len) in [(ByteOrder::BigEndian, 32), (ByteOrder::LittleEndian, 33)] {
        let mut bytes = vec![0; 64];
        bytes[32] = 1;
        let ptr = bytes.as_ptr();
        let n = Number::new(bytes, order);
        assert_eq!(n.value.len(), len);
        assert_eq!(n.value.as_ptr(), ptr);
    }
}

#[test]
fn new_drops_long_runs_of_zeros() {
    let mut bytes = vec![0; 100_000];
    bytes.push(7);
    assert_eq!(
        Number::new(bytes.clone(), ByteOrder::BigEndian).value,
        vec![7]
    );
    bytes.reverse();
    assert_eq!(Number::new(bytes, ByteOrder::LittleEndian).value, vec![7]);
}

#[test]
fn new_matches_reference_on_random_bytes() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..2000 {
        let bytes = random_bytes(&mut state);
        let expected = if bytes.is_empty() {
            vec![]
        } else {
            trimmed(&bytes)
        };
        let big = Number::new(bytes.clone(), ByteOrder::BigEndian);
        assert_eq!(big.value, expected, "big-endian {bytes:?}");
        assert!(!big.is_negative(), "big-endian {bytes:?}");
        // The same number written the other way round.
        let little = Number::new(reversed(&bytes), ByteOrder::LittleEndian);
        assert_eq!(little.value, reversed(&expected), "little-endian {bytes:?}");
        assert!(!little.is_negative(), "little-endian {bytes:?}");
    }
}

// `new_signed`

#[test]
fn new_signed_keeps_sign_of_non_zero_numbers() {
    let big = Number::new_signed(vec![0, 1, 0], true, ByteOrder::BigEndian);
    assert_eq!(big.value, vec![1, 0]);
    assert!(big.is_negative());
    let little = Number::new_signed(vec![0, 1, 0], true, ByteOrder::LittleEndian);
    assert_eq!(little.value, vec![0, 1]);
    assert!(little.is_negative());
    for order in ORDERS {
        let n = Number::new_signed(vec![5], false, order);
        assert_eq!(n.value, vec![5]);
        assert!(!n.is_negative());
    }
}

#[test]
fn new_signed_zero_is_never_negative() {
    for order in ORDERS {
        for bytes in [vec![], vec![0], vec![0, 0, 0]] {
            let n = Number::new_signed(bytes.clone(), true, order);
            assert!(!n.is_negative(), "{bytes:?}");
            assert_eq!(n.value, Number::new(bytes, order).value);
        }
    }
}

#[test]
fn new_signed_reuses_allocation() {
    for order in ORDERS {
        let mut bytes = vec![0; 64];
        bytes[32] = 1;
        let ptr = bytes.as_ptr();
        let n = Number::new_signed(bytes, true, order);
        assert!(n.is_negative());
        assert_eq!(n.value.as_ptr(), ptr);
    }
}

#[test]
fn new_signed_trims_as_new_on_random_bytes() {
    let mut state = 0x3C6E_F372_FE94_F82B;
    for _ in 0..2000 {
        let bytes = random_bytes(&mut state);
        let is_zero = bytes.iter().all(|&b| b == 0);
        for order in ORDERS {
            let unsigned = Number::new(bytes.clone(), order);
            let positive = Number::new_signed(bytes.clone(), false, order);
            assert_eq!(positive.value, unsigned.value, "{bytes:?}");
            assert!(!positive.is_negative(), "{bytes:?}");
            let negative = Number::new_signed(bytes.clone(), true, order);
            assert_eq!(negative.value, unsigned.value, "{bytes:?}");
            assert_eq!(negative.is_negative(), !is_zero, "{bytes:?}");
        }
    }
}

// The fields

#[test]
fn fields_build_number_as_given() {
    // Neither the zero bytes nor the sign of zero are corrected.
    let n = Number {
        value: vec![0, 0],
        negative: true,
        byte_order: ByteOrder::BigEndian,
    };
    assert_eq!(n.value, vec![0, 0]);
    assert!(n.is_negative());
    assert!(n.byte_order() == ByteOrder::BigEndian);
}

#[test]
fn is_negative_reads_the_field() {
    for negative in [false, true] {
        let n = Number {
            value: vec![1],
            negative,
            byte_order: ByteOrder::LittleEndian,
        };
        assert_eq!(n.is_negative(), negative);
    }
}

// Unsigned integers

#[test]
fn new_from_u8_is_its_byte() {
    for order in ORDERS {
        for b in 0..=u8::MAX {
            assert_eq!(Number::new_from_u8(b, order).value, vec![b]);
        }
    }
}

#[test]
fn new_from_unsigned_is_never_negative() {
    for order in ORDERS {
        for n in [
            Number::new_from_u8(u8::MAX, order),
            Number::new_from_u16(u16::MAX, order),
            Number::new_from_u32(u32::MAX, order),
            Number::new_from_u64(u64::MAX, order),
            Number::new_from_u128(u128::MAX, order),
            Number::new_from_usize(usize::MAX, order),
            Number::new_from_u8(0, order),
            Number::new_from_u128(0, order),
        ] {
            assert!(!n.is_negative());
        }
    }
}

#[test]
fn new_from_unsigned_writes_bytes_in_order() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    assert_eq!(Number::new_from_u16(0x0102, be).value, vec![1, 2]);
    assert_eq!(Number::new_from_u16(0x0102, le).value, vec![2, 1]);
    assert_eq!(
        Number::new_from_u32(0x0102_0304, be).value,
        vec![1, 2, 3, 4]
    );
    assert_eq!(
        Number::new_from_u32(0x0102_0304, le).value,
        vec![4, 3, 2, 1]
    );
    let u64_value = 0x0102_0304_0506_0708;
    let u64_bytes = vec![1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(Number::new_from_u64(u64_value, be).value, u64_bytes);
    assert_eq!(
        Number::new_from_u64(u64_value, le).value,
        reversed(&u64_bytes)
    );
    let u128_value = 0x0102_0304_0506_0708_090A_0B0C_0D0E_0F10;
    let u128_bytes: Vec<u8> = (1..=16).collect();
    assert_eq!(Number::new_from_u128(u128_value, be).value, u128_bytes);
    assert_eq!(
        Number::new_from_u128(u128_value, le).value,
        reversed(&u128_bytes)
    );
    assert_eq!(Number::new_from_usize(0x0102, be).value, vec![1, 2]);
    assert_eq!(Number::new_from_usize(0x0102, le).value, vec![2, 1]);
}

#[test]
fn new_from_unsigned_drops_high_zero_bytes() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    for order in ORDERS {
        assert_eq!(Number::new_from_u16(1, order).value, vec![1]);
        assert_eq!(Number::new_from_u32(1, order).value, vec![1]);
        assert_eq!(Number::new_from_u64(1, order).value, vec![1]);
        assert_eq!(Number::new_from_u128(1, order).value, vec![1]);
        assert_eq!(Number::new_from_usize(1, order).value, vec![1]);
    }
    assert_eq!(Number::new_from_u32(0x0100, be).value, vec![1, 0]);
    assert_eq!(Number::new_from_u32(0x0100, le).value, vec![0, 1]);
    assert_eq!(
        Number::new_from_u64(0x00FF_0000, be).value,
        vec![0xFF, 0, 0]
    );
    assert_eq!(
        Number::new_from_u64(0x00FF_0000, le).value,
        vec![0, 0, 0xFF]
    );
    let mut bytes = vec![0; 9];
    bytes[0] = 1;
    assert_eq!(Number::new_from_u128(1 << 64, be).value, bytes);
    assert_eq!(Number::new_from_u128(1 << 64, le).value, reversed(&bytes));
}

#[test]
fn new_from_unsigned_zero_is_one_byte() {
    for order in ORDERS {
        assert_eq!(Number::new_from_u8(0, order).value, vec![0]);
        assert_eq!(Number::new_from_u16(0, order).value, vec![0]);
        assert_eq!(Number::new_from_u32(0, order).value, vec![0]);
        assert_eq!(Number::new_from_u64(0, order).value, vec![0]);
        assert_eq!(Number::new_from_u128(0, order).value, vec![0]);
        assert_eq!(Number::new_from_usize(0, order).value, vec![0]);
    }
}

#[test]
fn new_from_unsigned_max_keeps_full_width() {
    for order in ORDERS {
        assert_eq!(Number::new_from_u8(u8::MAX, order).value, vec![0xFF]);
        assert_eq!(Number::new_from_u16(u16::MAX, order).value, vec![0xFF; 2]);
        assert_eq!(Number::new_from_u32(u32::MAX, order).value, vec![0xFF; 4]);
        assert_eq!(Number::new_from_u64(u64::MAX, order).value, vec![0xFF; 8]);
        assert_eq!(
            Number::new_from_u128(u128::MAX, order).value,
            vec![0xFF; 16]
        );
        assert_eq!(
            Number::new_from_usize(usize::MAX, order).value,
            vec![0xFF; size_of::<usize>()]
        );
    }
}

#[test]
fn new_from_unsigned_matches_reference_on_random_values() {
    let mut state = 0x2545_F491_4F6C_DD1D;
    for _ in 0..2000 {
        let x = random_u128(&mut state);
        let expected = trimmed(&x.to_be_bytes());
        let big = Number::new_from_u128(x, ByteOrder::BigEndian);
        assert_eq!(big.value, expected, "{x:#x}");
        let little = Number::new_from_u128(x, ByteOrder::LittleEndian);
        assert_eq!(little.value, reversed(&expected), "{x:#x}");
    }
}

#[test]
fn new_from_unsigned_does_not_depend_on_width() {
    let mut state = 0xD1B5_4A32_D192_ED03;
    for _ in 0..2000 {
        let x = random_u128(&mut state);
        for order in ORDERS {
            let wide = Number::new_from_u128(x, order).value;
            if let Ok(x) = u64::try_from(x) {
                assert_eq!(Number::new_from_u64(x, order).value, wide, "{x:#x}");
            }
            if let Ok(x) = usize::try_from(x) {
                assert_eq!(Number::new_from_usize(x, order).value, wide, "{x:#x}");
            }
            if let Ok(x) = u32::try_from(x) {
                assert_eq!(Number::new_from_u32(x, order).value, wide, "{x:#x}");
            }
            if let Ok(x) = u16::try_from(x) {
                assert_eq!(Number::new_from_u16(x, order).value, wide, "{x:#x}");
            }
            if let Ok(x) = u8::try_from(x) {
                assert_eq!(Number::new_from_u8(x, order).value, wide, "{x:#x}");
            }
        }
    }
}

// Signed integers

#[test]
fn new_from_i8_is_sign_and_magnitude() {
    for order in ORDERS {
        for x in i8::MIN..=i8::MAX {
            let n = Number::new_from_i8(x, order);
            assert_eq!(n.value, vec![x.unsigned_abs()], "{x}");
            assert_eq!(n.is_negative(), x < 0, "{x}");
        }
    }
}

#[test]
fn new_from_signed_minus_one_is_negative_one() {
    for order in ORDERS {
        for n in [
            Number::new_from_i8(-1, order),
            Number::new_from_i16(-1, order),
            Number::new_from_i32(-1, order),
            Number::new_from_i64(-1, order),
            Number::new_from_i128(-1, order),
            Number::new_from_isize(-1, order),
        ] {
            assert_eq!(n.value, vec![1]);
            assert!(n.is_negative());
        }
    }
}

#[test]
fn new_from_signed_min_has_exact_magnitude() {
    // The absolute value of `MIN` is one more than `MAX`: a one bit followed by zero bits.
    let min = |width: usize| {
        let mut bytes = vec![0; width];
        bytes[0] = 0x80;
        bytes
    };
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    let cases = [
        (
            Number::new_from_i8(i8::MIN, be),
            Number::new_from_i8(i8::MIN, le),
            1,
        ),
        (
            Number::new_from_i16(i16::MIN, be),
            Number::new_from_i16(i16::MIN, le),
            2,
        ),
        (
            Number::new_from_i32(i32::MIN, be),
            Number::new_from_i32(i32::MIN, le),
            4,
        ),
        (
            Number::new_from_i64(i64::MIN, be),
            Number::new_from_i64(i64::MIN, le),
            8,
        ),
        (
            Number::new_from_i128(i128::MIN, be),
            Number::new_from_i128(i128::MIN, le),
            16,
        ),
        (
            Number::new_from_isize(isize::MIN, be),
            Number::new_from_isize(isize::MIN, le),
            size_of::<isize>(),
        ),
    ];
    for (big, little, width) in cases {
        assert_eq!(big.value, min(width));
        assert_eq!(little.value, reversed(&min(width)));
        assert!(big.is_negative());
        assert!(little.is_negative());
    }
}

#[test]
fn new_from_signed_writes_bytes_in_order() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    for x in [0x0102, -0x0102] {
        let numbers = [
            (Number::new_from_i16(x, be), Number::new_from_i16(x, le)),
            (
                Number::new_from_i32(x.into(), be),
                Number::new_from_i32(x.into(), le),
            ),
            (
                Number::new_from_i64(x.into(), be),
                Number::new_from_i64(x.into(), le),
            ),
            (
                Number::new_from_i128(x.into(), be),
                Number::new_from_i128(x.into(), le),
            ),
            (
                Number::new_from_isize(x.into(), be),
                Number::new_from_isize(x.into(), le),
            ),
        ];
        for (big, little) in numbers {
            assert_eq!(big.value, vec![1, 2], "{x}");
            assert_eq!(little.value, vec![2, 1], "{x}");
            assert_eq!(big.is_negative(), x < 0, "{x}");
            assert_eq!(little.is_negative(), x < 0, "{x}");
        }
    }
    let mut max = vec![0xFF; 8];
    max[0] = 0x7F;
    assert_eq!(Number::new_from_i64(i64::MAX, be).value, max);
    assert_eq!(Number::new_from_i64(i64::MAX, le).value, reversed(&max));
    assert_eq!(Number::new_from_i64(-i64::MAX, be).value, max);
    assert_eq!(Number::new_from_i64(-i64::MAX, le).value, reversed(&max));
}

#[test]
fn new_from_signed_drops_high_zero_bytes() {
    for order in ORDERS {
        for x in [1i16, -1, 255, -255] {
            let expected = vec![x.unsigned_abs() as u8];
            for n in [
                Number::new_from_i16(x, order),
                Number::new_from_i32(x.into(), order),
                Number::new_from_i64(x.into(), order),
                Number::new_from_i128(x.into(), order),
                Number::new_from_isize(x.into(), order),
            ] {
                assert_eq!(n.value, expected, "{x}");
                assert_eq!(n.is_negative(), x < 0, "{x}");
            }
        }
    }
}

#[test]
fn new_from_signed_zero_is_one_byte_and_not_negative() {
    for order in ORDERS {
        for n in [
            Number::new_from_i8(0, order),
            Number::new_from_i16(0, order),
            Number::new_from_i32(0, order),
            Number::new_from_i64(0, order),
            Number::new_from_i128(0, order),
            Number::new_from_isize(0, order),
        ] {
            assert_eq!(n.value, vec![0]);
            assert!(!n.is_negative());
        }
    }
}

#[test]
fn new_from_signed_keeps_the_sign_apart_from_the_bytes() {
    for order in ORDERS {
        // -1 and 1 share their magnitude and differ in sign only.
        let minus_one = Number::new_from_i32(-1, order);
        let one = Number::new_from_i32(1, order);
        assert_eq!(minus_one.value, one.value);
        assert!(minus_one.is_negative() && !one.is_negative());
        // -1 is no longer the bit pattern of u32::MAX.
        assert_eq!(minus_one.value, vec![1]);
        assert_eq!(Number::new_from_u32(u32::MAX, order).value, vec![0xFF; 4]);
        // 255 and -1 differ in both.
        let n = Number::new_from_i16(255, order);
        assert_eq!(n.value, vec![0xFF]);
        assert!(!n.is_negative());
    }
}

#[test]
fn new_from_signed_is_sign_and_magnitude_on_random_values() {
    let mut state = 0x6A09_E667_F3BC_C908;
    for _ in 0..2000 {
        let bits = random_u128(&mut state);
        for order in ORDERS {
            let x = bits as i128;
            let n = Number::new_from_i128(x, order);
            let magnitude = Number::new_from_u128(x.unsigned_abs(), order);
            assert_eq!(n.value, magnitude.value, "{x}");
            assert_eq!(n.is_negative(), x < 0, "{x}");
            let x = bits as i64;
            let n = Number::new_from_i64(x, order);
            let magnitude = Number::new_from_u64(x.unsigned_abs(), order);
            assert_eq!(n.value, magnitude.value, "{x}");
            assert_eq!(n.is_negative(), x < 0, "{x}");
            let x = bits as isize;
            let n = Number::new_from_isize(x, order);
            let magnitude = Number::new_from_usize(x.unsigned_abs(), order);
            assert_eq!(n.value, magnitude.value, "{x}");
            assert_eq!(n.is_negative(), x < 0, "{x}");
            let x = bits as i32;
            let n = Number::new_from_i32(x, order);
            let magnitude = Number::new_from_u32(x.unsigned_abs(), order);
            assert_eq!(n.value, magnitude.value, "{x}");
            assert_eq!(n.is_negative(), x < 0, "{x}");
            let x = bits as i16;
            let n = Number::new_from_i16(x, order);
            let magnitude = Number::new_from_u16(x.unsigned_abs(), order);
            assert_eq!(n.value, magnitude.value, "{x}");
            assert_eq!(n.is_negative(), x < 0, "{x}");
        }
    }
}

#[test]
fn new_from_signed_does_not_depend_on_width() {
    let mut state = 0x510E_527F_ADE6_82D1;
    for _ in 0..2000 {
        let x = random_i128(&mut state);
        for order in ORDERS {
            let wide = Number::new_from_i128(x, order);
            let same = |n: Number| n.value == wide.value && n.is_negative() == wide.is_negative();
            if let Ok(x) = i64::try_from(x) {
                assert!(same(Number::new_from_i64(x, order)), "{x}");
            }
            if let Ok(x) = isize::try_from(x) {
                assert!(same(Number::new_from_isize(x, order)), "{x}");
            }
            if let Ok(x) = i32::try_from(x) {
                assert!(same(Number::new_from_i32(x, order)), "{x}");
            }
            if let Ok(x) = i16::try_from(x) {
                assert!(same(Number::new_from_i16(x, order)), "{x}");
            }
            if let Ok(x) = i8::try_from(x) {
                assert!(same(Number::new_from_i8(x, order)), "{x}");
            }
            // A non-negative value is the same number from an unsigned type.
            if let Ok(x) = u128::try_from(x) {
                assert!(same(Number::new_from_u128(x, order)), "{x}");
            }
        }
    }
}

// `new_from_i8_vec`

#[test]
fn new_from_i8_vec_keeps_bits_of_each_byte() {
    for order in ORDERS {
        let n = Number::new_from_i8_vec(vec![-1, -128, 127, 1], order);
        assert_eq!(n.value, vec![0xFF, 0x80, 0x7F, 0x01]);
    }
}

#[test]
fn new_from_i8_vec_is_never_negative() {
    for order in ORDERS {
        let n = Number::new_from_i8_vec(vec![-1], order);
        assert_eq!(n.value, vec![0xFF]);
        assert!(!n.is_negative());
    }
}

#[test]
fn new_from_i8_vec_drops_high_zero_bytes() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    assert_eq!(
        Number::new_from_i8_vec(vec![0, 0, -1], be).value,
        vec![0xFF]
    );
    assert_eq!(
        Number::new_from_i8_vec(vec![-1, 0, 0], le).value,
        vec![0xFF]
    );
    assert_eq!(
        Number::new_from_i8_vec(vec![0, 0, -1], le).value,
        vec![0, 0, 0xFF]
    );
    for order in ORDERS {
        assert_eq!(Number::new_from_i8_vec(vec![0, 0], order).value, vec![0]);
        assert!(Number::new_from_i8_vec(vec![], order).value.is_empty());
    }
}

#[test]
fn new_from_i8_vec_matches_new_on_random_bytes() {
    let mut state = 0xBB67_AE85_84CA_A73B;
    for _ in 0..1000 {
        let bytes = random_bytes(&mut state);
        let signed: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        for order in ORDERS {
            let n = Number::new_from_i8_vec(signed.clone(), order);
            assert_eq!(
                n.value,
                Number::new(bytes.clone(), order).value,
                "{signed:?}"
            );
            assert!(!n.is_negative(), "{signed:?}");
        }
    }
}

// `new_from_str`

#[test]
fn new_from_str_takes_utf8_bytes_unparsed() {
    for order in ORDERS {
        let n = Number::new_from_str("123", order);
        assert_eq!(n.value, vec![0x31, 0x32, 0x33]);
    }
}

#[test]
fn new_from_str_minus_is_a_byte_not_a_sign() {
    for order in ORDERS {
        let n = Number::new_from_str("-1", order);
        assert_eq!(n.value, vec![0x2D, 0x31]);
        assert!(!n.is_negative());
    }
}

#[test]
fn new_from_str_takes_every_byte_of_non_ascii_characters() {
    for order in ORDERS {
        assert_eq!(Number::new_from_str("é", order).value, vec![0xC3, 0xA9]);
        assert_eq!(
            Number::new_from_str("€1", order).value,
            vec![0xE2, 0x82, 0xAC, 0x31]
        );
    }
}

#[test]
fn new_from_str_drops_nul_characters_at_the_most_significant_end() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    assert_eq!(Number::new_from_str("\0\0a", be).value, vec![0x61]);
    assert_eq!(Number::new_from_str("\0\0a", le).value, vec![0, 0, 0x61]);
    assert_eq!(Number::new_from_str("a\0\0", le).value, vec![0x61]);
    assert_eq!(Number::new_from_str("a\0\0", be).value, vec![0x61, 0, 0]);
    for order in ORDERS {
        assert_eq!(Number::new_from_str("\0\0", order).value, vec![0]);
    }
}

#[test]
fn new_from_str_empty_is_empty() {
    for order in ORDERS {
        assert!(Number::new_from_str("", order).value.is_empty());
    }
}

// `byte_order`

#[test]
fn every_constructor_stores_byte_order() {
    for order in ORDERS {
        let numbers = [
            Number::new(vec![1, 2], order),
            Number::new_signed(vec![1, 2], true, order),
            Number::new_from_u8(1, order),
            Number::new_from_u16(1, order),
            Number::new_from_u32(1, order),
            Number::new_from_u64(1, order),
            Number::new_from_u128(1, order),
            Number::new_from_usize(1, order),
            Number::new_from_i8_vec(vec![1, 2], order),
            Number::new_from_i8(-1, order),
            Number::new_from_i16(-1, order),
            Number::new_from_i32(-1, order),
            Number::new_from_i64(-1, order),
            Number::new_from_i128(-1, order),
            Number::new_from_isize(-1, order),
            Number::new_from_str("1", order),
        ];
        for n in numbers {
            assert!(n.byte_order() == order);
            assert!(n.byte_order == order);
        }
    }
}
