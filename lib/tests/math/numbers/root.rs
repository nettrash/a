use a::math::numbers::{ByteOrder, Number};
use std::cmp::Ordering;

use super::common::*;
use super::div::{add_magnitudes, mul_magnitudes, panic_message, sub_magnitudes};

// References

/// The magnitude of `n` least significant byte first, without zero bytes at the top: empty for
/// zero.
fn magnitude(n: &Number) -> Vec<u8> {
    let mut v = le_bytes(n);
    while v.last() == Some(&0) {
        v.pop();
    }
    v
}

/// Compares the magnitudes `x` and `y`, given least significant byte first.
fn compare(x: &[u8], y: &[u8]) -> Ordering {
    let trim = |v: &[u8]| v.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
    let (lx, ly) = (trim(x), trim(y));
    lx.cmp(&ly)
        .then_with(|| x[..lx].iter().rev().cmp(y[..ly].iter().rev()))
}

/// `x^n` for a magnitude given least significant byte first: squares and products by the
/// multiplication of `Number`, which its own tests check.
fn power(x: &[u8], n: u32) -> Vec<u8> {
    let le = ByteOrder::LittleEndian;
    let base = Number::new(x.to_vec(), le);
    let mut r = Number::new(vec![1], le);
    for i in (0..32 - n.leading_zeros()).rev() {
        r = &r * &r;
        if (n >> i) & 1 == 1 {
            r = &r * &base;
        }
    }
    magnitude(&r)
}

/// `x + 1` for a magnitude given least significant byte first.
fn plus_one(x: &[u8]) -> Vec<u8> {
    add_magnitudes(x, &[1])
}

/// The root of degree `n` of `x`, rounded down, by bisection on `u128`s.
fn root_u128(x: u128, n: u32) -> u128 {
    if n == 1 {
        return x;
    }
    let (mut lo, mut hi) = (0u128, 1u128 << (128 / n + 1).min(127));
    // lo^n <= x < hi^n.
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match mid.checked_pow(n) {
            Some(p) if p <= x => lo = mid,
            _ => hi = mid,
        }
    }
    lo
}

/// Checks that `root` is the root of degree `n` of the magnitude `x`, truncated, with the sign
/// `negative`, in the byte order `order` and the shortest form: `root^n <= x < (root + 1)^n`.
fn assert_root(root: &Number, x: &[u8], n: u32, negative: bool, order: ByteOrder, what: &str) {
    let inputs = || format!("{what}: root of degree {n} of {}", hex(x));
    assert_shortest(root, &inputs);
    assert!(root.byte_order == order, "{}: wrong byte order", inputs());
    let s = magnitude(root);
    assert!(
        root.negative == (negative && !s.is_empty()),
        "{}: wrong sign",
        inputs()
    );
    assert!(
        compare(&power(&s, n), x) != Ordering::Greater,
        "{}: the root {} is too large",
        inputs(),
        describe(root)
    );
    assert!(
        compare(&power(&plus_one(&s), n), x) == Ordering::Greater,
        "{}: the root {} is too small",
        inputs(),
        describe(root)
    );
}

/// Checks a square root and its remainder: `s^2 + r = x` and `r <= 2s`, in the byte order `order`
/// and the shortest form, by the independent multiplication of the division's tests.
fn assert_sqrt_rem(s: &Number, r: &Number, x: &[u8], order: ByteOrder) {
    let inputs = || format!("isqrt_rem of {}", hex(x));
    assert_shortest(s, &inputs);
    assert_shortest(r, &inputs);
    assert!(
        s.byte_order == order && r.byte_order == order,
        "{}: wrong byte order",
        inputs()
    );
    assert!(
        !s.negative && !r.negative,
        "{}: a negative result",
        inputs()
    );
    let (sm, rm) = (magnitude(s), magnitude(r));
    let square = mul_magnitudes(&sm, &sm);
    assert!(
        compare(&add_magnitudes(&square, &rm), x) == Ordering::Equal,
        "{}: s^2 + r is not x for s = {}, r = {}",
        inputs(),
        describe(s),
        describe(r)
    );
    assert!(
        compare(&rm, &add_magnitudes(&sm, &sm)) != Ordering::Greater,
        "{}: the remainder {} is above 2s",
        inputs(),
        describe(r)
    );
}

/// The bytes of `x`, least significant first, without the zero bytes above the top one.
fn trimmed(x: u128) -> Vec<u8> {
    let mut v = x.to_le_bytes().to_vec();
    while v.last() == Some(&0) {
        v.pop();
    }
    v
}

/// A number with the magnitude `x`, least significant byte first, the sign `negative` and the
/// byte order `order`, with up to 9 zero bytes above its top byte one time in three.
fn stored_as(state: &mut u64, x: &[u8], negative: bool, order: ByteOrder) -> Number {
    let mut v = x.to_vec();
    if below(state, 3) == 0 {
        v.resize(v.len() + 1 + below(state, 9), 0);
    }
    number(&v, negative, order)
}

/// Lengths in bytes of the long magnitudes: around the paths in registers, roots of 2 to 13
/// limbs, around the limbs of root from which a root alone is computed with a limb more (12),
/// from which the halves of the Karatsuba square root are divided by divide and conquer (80, with
/// halves of 40) and from which their scratch is counted (64 limbs of root), and longer.
fn lengths() -> Vec<usize> {
    let mut lengths: Vec<usize> = (17..=34).collect();
    for limbs in [5, 6, 7, 8, 9, 15, 16, 17, 23, 24, 25, 26, 31, 32, 33] {
        lengths.extend([8 * limbs - 3, 8 * limbs]);
    }
    for root_limbs in [
        11, 12, 13, 39, 40, 63, 64, 65, 79, 80, 81, 112, 113, 160, 250,
    ] {
        lengths.extend([16 * root_limbs - 9, 16 * root_limbs - 8, 16 * root_limbs]);
    }
    lengths.extend([5000, 9000]);
    lengths
        .into_iter()
        .filter(|&l| l <= size(usize::MAX, 48))
        .collect()
}

// Square roots

#[test]
fn square_roots_of_small_values_are_those_of_u128() {
    let mut state = 0x243F6A88_85A308D3;
    let mut values: Vec<u128> = (0..size(70_000, 600) as u128).collect();
    for _ in 0..size(100_000, 400) {
        values.push(random_i128(&mut state).unsigned_abs() << below(&mut state, 3));
    }
    for m in [
        1u128,
        2,
        3,
        0xFFFF,
        1 << 32,
        u64::MAX as u128,
        (1 << 63) + 1,
    ] {
        values.extend([m * m, m * m - 1, m * m + 1]);
    }
    values.extend([u128::MAX, u128::MAX - 1, 1 << 127, (1 << 126) - 1]);
    for x in values {
        let s = x.isqrt();
        for order in ORDERS {
            let a = stored_as(&mut state, &trimmed(x), false, order);
            let what = || format!("square root of {x} in {}", describe(&a));
            assert_result(&a.isqrt(), &expected_i128(s as i128), order, &what);
            let (root, r) = a.isqrt_rem();
            assert_result(&root, &expected_i128(s as i128), order, &what);
            assert_result(
                &r,
                &expected(&(x - s * s).to_le_bytes(), false),
                order,
                &what,
            );
            assert_result(&a.nth_root(2), &expected_i128(s as i128), order, &what);
        }
    }
}

#[test]
fn square_roots_of_long_values_are_exact() {
    let mut state = 0x13198A2E_03707344;
    for len in lengths() {
        let tries = if len > 1000 { 2 } else { 6 };
        for t in 0..tries {
            let x = match t {
                0 => vec![0xFF; len],
                1 => {
                    let mut x = vec![0; len];
                    x[len - 1] = 1 << below(&mut state, 8);
                    x
                }
                _ => random_magnitude(&mut state, len),
            };
            for order in ORDERS {
                let a = stored_as(&mut state, &x, false, order);
                let (s, r) = a.isqrt_rem();
                assert_sqrt_rem(&s, &r, &x, order);
                let alone = a.isqrt();
                assert!(
                    same(&alone, &s),
                    "isqrt of {} is not that of isqrt_rem",
                    hex(&x)
                );
                assert!(same(&a.nth_root(2), &s), "nth_root(2) of {}", hex(&x));
            }
        }
    }
}

#[test]
fn square_roots_of_every_top_limb_shift() {
    // The top limb's leading zeros set the shift of the magnitude, `2c` bits, and whether the bits
    // shifted out of a root alone spare its last square, for an even and an odd number of limbs.
    let mut state = 0xA4093822_299F31D0;
    for limbs in [3, 4, 7, 8, 24, 25, 26, size(160, 26)] {
        for zeros in 0..64 {
            let mut x = random_magnitude(&mut state, 8 * limbs);
            let top = (u64::MAX >> zeros) & !(u64::MAX >> zeros >> 1);
            let random = xorshift(&mut state) & (u64::MAX >> zeros);
            x[8 * limbs - 8..].copy_from_slice(&(top | random).to_le_bytes());
            let order = ORDERS[zeros % 2];
            let a = number(&x, false, order);
            let (s, r) = a.isqrt_rem();
            assert_sqrt_rem(&s, &r, &x, order);
            assert!(same(&a.isqrt(), &s), "isqrt of {}", hex(&x));
        }
    }
}

#[test]
fn square_roots_of_squares_and_their_neighbours() {
    let mut state = 0x082EFA98_EC4E6C89;
    for len in lengths() {
        let half = len.div_ceil(2);
        let m = match below(&mut state, 3) {
            0 => vec![0xFF; half],
            _ => random_magnitude(&mut state, half),
        };
        let square = mul_magnitudes(&m, &m);
        let below_square = sub_magnitudes(&square, &[1]);
        for (x, root) in [
            (square.clone(), m.clone()),
            (add_magnitudes(&square, &[1]), m.clone()),
            (below_square, sub_magnitudes(&m, &[1])),
        ] {
            let order = ORDERS[below(&mut state, 2)];
            let a = stored_as(&mut state, &x, false, order);
            let (s, r) = a.isqrt_rem();
            assert_sqrt_rem(&s, &r, &x, order);
            assert!(
                compare(&magnitude(&s), &root) == Ordering::Equal,
                "the root of {} is not {}",
                hex(&x),
                hex(&root)
            );
            assert!(same(&a.isqrt(), &s), "isqrt of {}", hex(&x));
        }
    }
}

#[test]
fn square_roots_of_zero_in_every_form() {
    for order in ORDERS {
        for z in zeros(order) {
            let root = z.isqrt();
            assert_result(&root, &zero(), order, &|| {
                format!("isqrt of {}", describe(&z))
            });
            let (s, r) = z.isqrt_rem();
            assert_result(&s, &zero(), order, &|| "isqrt_rem".into());
            assert_result(&r, &zero(), order, &|| "isqrt_rem".into());
            let long = number(&[0; 40], true, order);
            assert_result(&long.isqrt(), &zero(), order, &|| {
                "isqrt of 40 zeros".into()
            });
        }
    }
}

#[test]
fn square_roots_of_negative_numbers_panic_as_for_i64() {
    let message = "argument of integer square root cannot be negative";
    for order in ORDERS {
        for x in [vec![1], vec![0xFF; 16], vec![5; 17], vec![0xAB; 300]] {
            let a = number(&x, true, order);
            assert_eq!(panic_message(|| drop(a.isqrt())), message);
            assert_eq!(panic_message(|| drop(a.isqrt_rem())), message);
        }
    }
    assert_eq!(panic_message(|| _ = (-1i64).isqrt()), message);
}

// Roots of other degrees

#[test]
fn roots_of_small_values_are_those_of_u128() {
    let mut state = 0x452821E6_38D01377;
    for n in 1..=size(40, 8) as u32 {
        let mut values: Vec<u128> = (0..size(3000, 100) as u128).collect();
        for _ in 0..size(3000, 60) {
            values.push(random_i128(&mut state).unsigned_abs() << below(&mut state, 3));
        }
        for m in [2u128, 3, 7, 255, 65_535, 1 << 20, 3_000_000_000] {
            if let Some(p) = m.checked_pow(n) {
                values.extend([p - 1, p, p + 1]);
            }
        }
        values.extend([u128::MAX, 1 << 127]);
        for x in values {
            let root = root_u128(x, n);
            for order in ORDERS {
                let negative = n % 2 == 1 && below(&mut state, 2) == 0;
                let a = stored_as(&mut state, &trimmed(x), negative, order);
                let got = a.nth_root(n);
                let want = expected(&root.to_le_bytes(), negative);
                assert_result(&got, &want, order, &|| {
                    format!("root of degree {n} of {}", describe(&a))
                });
            }
        }
    }
}

/// Degrees for a magnitude of `bits` bits: small ones, odd and even, ones whose roots have a
/// few bits, around the magnitude's bits and beyond them.
fn degrees(bits: u32) -> Vec<u32> {
    let mut d = vec![1, 3, 4, 5, 7, 17, 100];
    for div in [2, 3, 4, 10, 29, 31, 33, 64] {
        d.push((bits / div).max(1));
    }
    d.extend([bits.saturating_sub(1).max(1), bits, bits + 1, u32::MAX]);
    d
}

#[test]
fn roots_of_long_values_are_exact() {
    let mut state = 0xC0AC29B7_C97C50DD;
    for len in lengths().into_iter().filter(|&l| l <= size(2100, 40)) {
        let x = random_magnitude(&mut state, len);
        let bits = 8 * len as u32 - x[len - 1].leading_zeros();
        for n in degrees(bits) {
            let order = ORDERS[below(&mut state, 2)];
            let negative = n % 2 == 1 && below(&mut state, 2) == 0;
            let a = stored_as(&mut state, &x, negative, order);
            let root = a.nth_root(n);
            if n == u32::MAX {
                assert_result(&root, &expected(&[1], negative), order, &|| {
                    "u32::MAX".into()
                });
                continue;
            }
            assert_root(&root, &x, n, negative, order, "random");
        }
    }
}

#[test]
fn roots_of_perfect_powers_and_their_neighbours() {
    // Perfect powers end the root of a long magnitude computed with 64 extra bits in the check
    // that is otherwise rare: their extra bits are zero.
    let mut state = 0x3F84D5B5_B5470917;
    let cases: [(usize, u32); 14] = [
        (5, 3),
        (11, 3),
        (40, 3),
        (200, 3),
        (1500, 3),
        (16, 5),
        (90, 5),
        (400, 7),
        (40, 17),
        (300, 17),
        (8, 100),
        (60, 100),
        (1, 300),
        (3, 1000),
    ];
    for (len, n) in cases {
        if len * n as usize > size(usize::MAX, 200) {
            continue;
        }
        for kind in 0..3 {
            let m = match kind {
                0 => vec![0xFF; len],
                1 => {
                    let mut m = vec![0; len];
                    m[len - 1] = 1;
                    m
                }
                _ => random_magnitude(&mut state, len),
            };
            let p = power(&m, n);
            for (x, root) in [
                (p.clone(), m.clone()),
                (plus_one(&p), m.clone()),
                (sub_magnitudes(&p, &[1]), sub_magnitudes(&m, &[1])),
            ] {
                let order = ORDERS[below(&mut state, 2)];
                let a = stored_as(&mut state, &x, false, order);
                let got = a.nth_root(n);
                assert!(
                    compare(&magnitude(&got), &root) == Ordering::Equal,
                    "root of degree {n} of {}: {} instead of {}",
                    hex(&x),
                    describe(&got),
                    hex(&root)
                );
                assert_root(&got, &x, n, false, order, "perfect power");
            }
        }
    }
}

#[test]
fn roots_of_degree_one_and_two() {
    let mut state = 0x9216D5D9_8979FB1B;
    for len in [0, 1, 16, 17, 100] {
        let x = random_magnitude(&mut state, len);
        for order in ORDERS {
            for negative in [false, true] {
                let a = stored_as(&mut state, &x, negative, order);
                let what = || format!("root of degree 1 of {}", describe(&a));
                assert_result(&a.nth_root(1), &expected(&x, negative), order, &what);
                if !negative || len == 0 {
                    assert!(same(&a.nth_root(2), &a.isqrt()), "degree 2: {}", what());
                }
            }
        }
    }
}

#[test]
fn roots_of_zero_and_one() {
    for order in ORDERS {
        for n in [1, 2, 3, 64, 1000, u32::MAX] {
            for z in zeros(order) {
                assert_result(&z.nth_root(n), &zero(), order, &|| "zero".into());
            }
            for negative in [false, true] {
                let one = number(&[1, 0, 0], negative, order);
                let want = expected(&[1], negative && n % 2 == 1);
                if negative && n % 2 == 0 {
                    continue;
                }
                assert_result(&one.nth_root(n), &want, order, &|| format!("one, n = {n}"));
            }
        }
    }
}

#[test]
fn roots_of_degree_zero_and_even_roots_of_negative_numbers_panic() {
    for order in ORDERS {
        for x in [vec![], vec![1], vec![7; 17], vec![0xF0; 300]] {
            let a = number(&x, false, order);
            assert_eq!(
                panic_message(|| drop(a.nth_root(0))),
                "root degree n must be at least 1"
            );
            if x.is_empty() {
                continue;
            }
            let b = number(&x, true, order);
            for n in [2, 4, 100] {
                assert_eq!(
                    panic_message(|| drop(b.nth_root(n))),
                    "even roots of a negative are imaginary"
                );
            }
        }
    }
}

#[test]
fn roots_through_every_path_even_under_miri() {
    // Sizes that the scaled-down tests leave out under Miri: square roots whose halves divide by
    // the schoolbook method on three limbs and more, and roots alone with a limb more (from 12
    // limbs of root); cube roots with 64 extra bits (from 10 limbs), of a perfect power too.
    let mut state = 0xD1310BA6_98DFB5AC;
    for len in [72, 80, 184, 200] {
        let x = random_magnitude(&mut state, len);
        let order = ORDERS[len % 3 % 2];
        let a = number(&x, false, order);
        let (s, r) = a.isqrt_rem();
        assert_sqrt_rem(&s, &r, &x, order);
        assert!(same(&a.isqrt(), &s), "isqrt of {}", hex(&x));
    }
    let x = random_magnitude(&mut state, 88);
    assert_root(
        &number(&x, true, ByteOrder::BigEndian).nth_root(3),
        &x,
        3,
        true,
        ByteOrder::BigEndian,
        "cube root with extra bits",
    );
    let m = random_magnitude(&mut state, 30);
    let cube = power(&m, 3);
    let root = number(&cube, false, ByteOrder::LittleEndian).nth_root(3);
    assert!(
        compare(&magnitude(&root), &m) == Ordering::Equal,
        "the cube root of a perfect cube"
    );
}
