use a::math::numbers::{ByteOrder, Number};

use super::common::*;
use crate::allocations::count;

// Expected products

/// The 32-bit limbs of a magnitude given least significant byte first.
fn limbs32(magnitude: &[u8]) -> Vec<u32> {
    magnitude
        .chunks(4)
        .map(|c| {
            let mut b = [0; 4];
            b[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(b)
        })
        .collect()
}

/// The product of the integers `a` and `b` hold, worked out independently of the implementation:
/// the schoolbook method on 32-bit limbs of the magnitudes, and the sign of the product of the
/// signs. The tests on small values check it against `i128` arithmetic.
fn product(a: &Number, b: &Number) -> Expected {
    let (x, y) = (limbs32(&le_bytes(a)), limbs32(&le_bytes(b)));
    let mut r = vec![0u32; x.len() + y.len() + 1];
    for (i, &xi) in x.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &yj) in y.iter().enumerate() {
            let t = u64::from(xi) * u64::from(yj) + u64::from(r[i + j]) + carry;
            r[i + j] = t as u32;
            carry = t >> 32;
        }
        let mut k = i + y.len();
        while carry != 0 {
            let t = u64::from(r[k]) + carry;
            r[k] = t as u32;
            carry = t >> 32;
            k += 1;
        }
    }
    let magnitude: Vec<u8> = r.iter().flat_map(|l| l.to_le_bytes()).collect();
    // A zero product is not negative: `expected` sees to it.
    expected(&magnitude, a.negative != b.negative)
}

/// Primes below 2^62 for [`residues`].
const PRIMES: [u64; 3] = [(1 << 61) - 1, 0x3FFF_FFFF_FFFF_FFC3, 0x1FFF_FFFF_FFFF_FFE7];

/// The magnitude given least significant byte first, modulo each of [`PRIMES`].
fn residues(magnitude: &[u8]) -> [u64; 3] {
    PRIMES.map(|p| {
        magnitude.iter().rev().fold(0u64, |r, &byte| {
            ((u128::from(r) * 256 + u128::from(byte)) % u128::from(p)) as u64
        })
    })
}

/// Checks a product too long for the schoolbook reference: in the shortest form and the byte
/// order of `a`, with the sign of the product of the signs, the length the operands allow, and
/// the product of their residues modulo three primes.
fn assert_product_residues(result: &Number, a: &Number, b: &Number, inputs: &dyn Fn() -> String) {
    assert_shortest(result, inputs);
    assert!(
        result.byte_order == a.byte_order,
        "{}: byte order",
        inputs()
    );
    let (x, y, z) = (le_bytes(a), le_bytes(b), le_bytes(result));
    let (rx, ry, rz) = (residues(&x), residues(&y), residues(&z));
    for i in 0..3 {
        let p = u128::from(PRIMES[i]);
        let want = (u128::from(rx[i]) * u128::from(ry[i]) % p) as u64;
        assert_eq!(rz[i], want, "{}: the product modulo {p}", inputs());
    }
    let significant = |m: &[u8]| m.iter().rposition(|&v| v != 0).map_or(0, |t| t + 1);
    let (la, lb) = (significant(&x), significant(&y));
    let zero = la == 0 || lb == 0;
    let len = significant(&z);
    if zero {
        assert_eq!(len, 0, "{}: a product with zero is zero", inputs());
    } else {
        assert!(
            len == la + lb || len + 1 == la + lb,
            "{}: length {len}",
            inputs()
        );
    }
    let negative = !zero && a.negative != b.negative;
    assert_eq!(result.negative, negative, "{}: sign", inputs());
}

// The forms of multiplication

/// The six ways to write a multiplication. Every form must give the same product, so the tests
/// run them all.
#[derive(Clone, Copy)]
enum Form {
    ValueValue,
    ValueRef,
    RefValue,
    RefRef,
    AssignValue,
    AssignRef,
}

const FORMS: [Form; 6] = [
    Form::ValueValue,
    Form::ValueRef,
    Form::RefValue,
    Form::RefRef,
    Form::AssignValue,
    Form::AssignRef,
];

impl Form {
    /// The form as it is written.
    fn name(self) -> &'static str {
        match self {
            Form::ValueValue => "a * b",
            Form::ValueRef => "a * &b",
            Form::RefValue => "&a * b",
            Form::RefRef => "&a * &b",
            Form::AssignValue => "a *= b",
            Form::AssignRef => "a *= &b",
        }
    }

    /// Multiplies `a` by `b` in this form; an operand the form only borrows is dropped
    /// afterwards.
    fn apply(self, a: Number, b: Number) -> Number {
        match self {
            Form::ValueValue => a * b,
            Form::ValueRef => a * &b,
            Form::RefValue => &a * b,
            Form::RefRef => &a * &b,
            Form::AssignValue => {
                let mut a = a;
                a *= b;
                a
            }
            Form::AssignRef => {
                let mut a = a;
                a *= &b;
                a
            }
        }
    }
}

/// Every form, or under Miri one picked at random: for the tests that check many products.
fn forms_to_run() -> &'static [Form] {
    if cfg!(miri) {
        let k = pick(FORMS.len());
        &FORMS[k..=k]
    } else {
        &FORMS
    }
}

/// Checks `a * b` against `expected` in every form, as [`assert_multiplies_in`] does.
fn assert_multiplies(a: &Number, b: &Number, expected: &Expected) {
    assert_multiplies_in(&FORMS, a, b, expected);
}

/// Checks `a * b` against `expected` in each of `forms`: with the vectors of the operands the
/// form takes by value as tight as their values, with one byte to spare, and with room for any
/// product, and for `a * b` and `a *= b` also with room in `b` only, so that each form meets each
/// way it can place the product. Under Miri each form runs with one of these setups, picked at
/// random.
fn assert_multiplies_in(forms: &[Form], a: &Number, b: &Number, expected: &Expected) {
    for &form in forms {
        let spares = spares(form, a, b);
        let spares = if cfg!(miri) {
            let k = pick(spares.len());
            &spares[k..=k]
        } else {
            &spares[..]
        };
        for &(spare_a, spare_b) in spares {
            assert_multiplies_with(form, spare_a, spare_b, a, b, expected);
        }
    }
}

/// The capacity setups [`assert_multiplies_in`] runs `form` with, as the bytes to spare in the
/// vectors of `a` and `b`.
fn spares(form: Form, a: &Number, b: &Number) -> Vec<(usize, usize)> {
    // A product is at most as long as both operands together.
    let room = a.value.len() + b.value.len();
    match form {
        Form::RefRef => vec![(0, 0)],
        Form::ValueValue | Form::AssignValue => vec![(0, 0), (1, 1), (room, room), (0, room)],
        Form::ValueRef | Form::RefValue | Form::AssignRef => vec![(0, 0), (1, 1), (room, room)],
    }
}

/// Checks `a * b` against `expected` in `form`, with `spare_a` and `spare_b` bytes to spare in
/// the vectors of the operands the form takes by value.
fn assert_multiplies_with(
    form: Form,
    spare_a: usize,
    spare_b: usize,
    a: &Number,
    b: &Number,
    expected: &Expected,
) {
    let result = form.apply(copy(a, spare_a), copy(b, spare_b));
    let inputs = || {
        format!(
            "{} with a = {}, b = {}, {spare_a} and {spare_b} bytes to spare",
            form.name(),
            describe(a),
            describe(b)
        )
    };
    assert_result(&result, expected, a.byte_order, &inputs);
}

/// `a * b` in `form`, checked to be in the shortest form and in the byte order of `a`.
fn multiply(form: Form, a: &Number, b: &Number) -> Number {
    let result = form.apply(copy(a, 0), copy(b, 0));
    let inputs = || {
        format!(
            "{} with a = {}, b = {}",
            form.name(),
            describe(a),
            describe(b)
        )
    };
    assert_shortest(&result, &inputs);
    assert!(
        result.byte_order == a.byte_order,
        "{}: not in the byte order of a",
        inputs()
    );
    result
}

/// A random `i128` of at most 63 bits of magnitude, so that every product of two fits.
fn random_i64(state: &mut u64) -> i128 {
    random_i128(state) >> 63
}

// Small values against `i128` arithmetic

#[test]
fn products_by_one_limb_are_built_in_one_pass() {
    // A long operand times one of at most eight bytes is multiplied in place, limb by limb, its top
    // bytes then taking the last carry: every count of bytes above the whole limbs, all ones (the
    // longest carries, and a product a byte longer) or random, by one limb of every length, all
    // ones, 1 and 0, either way round, in every form and byte order.
    let mut state = 0x5EED_1111;
    let limbs: [u64; 8] = [0, 1, 2, 10, 0xFF, 1 << 63, u64::MAX, 0x0123_4567_89AB_CDEF];
    for len in (17..=size(90, 40)).chain([size(300, 60), size(1000, 70)]) {
        for kind in 0..2 {
            let x = if kind == 0 {
                vec![0xFF; len]
            } else {
                random_magnitude(&mut state, len)
            };
            let m = limbs[below(&mut state, limbs.len())] >> below(&mut state, 64);
            let (oa, ob) = ORDER_PAIRS[below(&mut state, 4)];
            let (na, nb) = SIGN_PAIRS[below(&mut state, 4)];
            let a = number(&x, na, oa);
            let b = number(&m.to_le_bytes(), nb, ob);
            let expected = product(&a, &b);
            assert_multiplies_in(forms_to_run(), &a, &b, &expected);
            assert_multiplies_in(forms_to_run(), &b, &a, &expected);
        }
    }
}

#[test]
fn multiplies_small_values_as_i128() {
    let mut state = 0x8F1B_BCDC_CA62_C1D6;
    for _ in 0..size(1000, 40) {
        let x = random_i64(&mut state);
        let y = random_i64(&mut state);
        let expected = expected_i128(x * y);
        for (left, right) in ORDER_PAIRS {
            let a = Number::new_from_i128(x, left);
            let b = Number::new_from_i128(y, right);
            // The reference of the tests on big numbers agrees with `i128` arithmetic.
            assert_eq!(product(&a, &b), expected, "reference for {x} * {y}");
            assert_multiplies_in(forms_to_run(), &a, &b, &expected);
        }
    }
}

#[test]
fn multiplies_every_pair_of_small_integers() {
    // From -300 to 300: every carry out of the first byte of a product and into its third, in
    // every combination of signs. Each pair runs in one form, pair of byte orders and spare
    // capacity, taken in turn.
    let mut turn = 0;
    for x in (-300..=300).step_by(size(1, 23)) {
        for y in (-300..=300).step_by(size(1, 23)) {
            let (left, right) = ORDER_PAIRS[turn % 4];
            let form = FORMS[turn / 4 % 6];
            let spare = turn / 24 % 3;
            turn += 1;
            let a = Number::new_from_i32(x, left);
            let b = Number::new_from_i32(y, right);
            let result = form.apply(copy(&a, spare), copy(&b, spare));
            let inputs = || format!("{x} * {y} as {} with {spare} bytes to spare", form.name());
            assert_result(&result, &expected_i128((x * y).into()), left, &inputs);
        }
    }
}

#[test]
fn multiplies_at_the_limits_of_the_short_path() {
    // Operands of up to 16 bytes, the longest the short path takes, with every bit set, as powers
    // of two, and random: products of up to 32 bytes, in every pair of lengths.
    let mut state = 0xB8E1_AFED_6A26_7E96;
    for len_a in 0..=17 {
        for len_b in 0..=17 {
            let mut magnitudes = vec![vec![0xFF; len_a], random_magnitude(&mut state, len_a)];
            if len_a > 0 {
                let mut power = vec![0; len_a - 1];
                power.push(0x80);
                magnitudes.push(power);
            }
            for x in &magnitudes {
                let y = random_magnitude(&mut state, len_b);
                let k = below(&mut state, 16);
                let (left, right) = ORDER_PAIRS[k % 4];
                let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
                let a = number(x, negative_a, left);
                let b = number(&y, negative_b, right);
                assert_multiplies_in(forms_to_run(), &a, &b, &product(&a, &b));
                let b = number(&vec![0xFF; len_b], negative_b, right);
                assert_multiplies_in(forms_to_run(), &a, &b, &product(&a, &b));
            }
        }
    }
}

// Big values against the reference

#[test]
fn multiplies_random_big_numbers_as_the_reference() {
    let mut state = 0xA4D0_B5C2_1F3E_9D87;
    for _ in 0..size(300, 12) {
        let a = random_operand(&mut state, 600);
        let b = random_operand(&mut state, 600);
        assert_multiplies_in(forms_to_run(), &a, &b, &product(&a, &b));
    }
}

#[test]
fn multiplies_every_pair_of_lengths_up_to_50_bytes() {
    // Up to six 8-byte limbs and a bit more, with every remainder after whole limbs: the paths in
    // registers up to 16, 32 and 48 bytes, and the limbs in memory past them. Every pair of
    // lengths runs in every combination of byte orders and signs, or under Miri in one picked at
    // random.
    let mut state = 0x5851_F42D_4C95_7F2D;
    for len_a in 0..=50 {
        for len_b in 0..=50 {
            for k in 0..size(16, 1) {
                let k = if cfg!(miri) { below(&mut state, 16) } else { k };
                let (left, right) = ORDER_PAIRS[k % 4];
                let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
                let a = number(&random_magnitude(&mut state, len_a), negative_a, left);
                let b = number(&random_magnitude(&mut state, len_b), negative_b, right);
                assert_multiplies_in(forms_to_run(), &a, &b, &product(&a, &b));
            }
        }
    }
}

/// Numbers of limbs around the sizes at which a product changes method: the schoolbook method,
/// Karatsuba's from 24 limbs in the shorter operand (56 for a square), Toom-3 from 120 (168) and
/// Toom-4 from 224, a shorter operand of at most half the longer one, which multiplies it in
/// pieces, and the levels of the recursions. Under Miri, fewer.
fn product_limbs() -> Vec<usize> {
    let all = [
        1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 47, 48, 49, 55, 56, 57, 63, 64,
        65, 95, 96, 97, 119, 120, 121, 127, 128, 129, 167, 168, 169, 191, 192, 193, 223, 224, 225,
        255, 256, 257,
    ];
    let miri = [1, 3, 8, 31, 32, 33, 48, 49, 65];
    if cfg!(miri) {
        miri.to_vec()
    } else {
        all.to_vec()
    }
}

#[test]
fn multiplies_every_shape_of_long_operands_as_the_reference() {
    // The shorter operand of each of the limb counts above, the longer one as long, a limb
    // longer, about one and a half, two and three times as long, with a few bytes above the whole
    // limbs, in runs of zeros, ones and random limbs.
    let mut state = 0x7A2B_E3C1_D0F9_4B65;
    for ny in product_limbs() {
        for nx in [
            ny,
            ny + 1,
            ny + ny / 2 + 1,
            2 * ny - 1,
            2 * ny,
            2 * ny + 1,
            3 * ny + 5,
        ] {
            let ex = EXTRA_BYTES[below(&mut state, 3)];
            let ey = EXTRA_BYTES[below(&mut state, 3)];
            let x = limb_runs(&mut state, nx, ex);
            let y = limb_runs(&mut state, ny, ey);
            let k = below(&mut state, 16);
            let (left, right) = ORDER_PAIRS[k % 4];
            let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
            let a = number(&x, negative_a, left);
            let b = number(&y, negative_b, right);
            let expected = product(&a, &b);
            let form = FORMS[below(&mut state, 6)];
            assert_multiplies_in(&[form], &a, &b, &expected);
            assert_multiplies_in(&[form], &b, &a, &product(&b, &a));
        }
    }
}

#[test]
fn multiplies_numbers_of_hundreds_of_kilobytes() {
    // Too long for the schoolbook reference in a debug build: checked by their residues.
    let mut state = 0x1E37_79B9_7F4A_7C15;
    let len = size(200_000, 3_000);
    for (len_a, len_b) in [(len, len), (len, len / 3 + 5), (len / 2 + 7, len)] {
        let x = random_magnitude(&mut state, len_a);
        let y = random_magnitude(&mut state, len_b);
        let k = below(&mut state, 16);
        let (left, right) = ORDER_PAIRS[k % 4];
        let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
        let a = number(&x, negative_a, left);
        let b = number(&y, negative_b, right);
        for form in [Form::RefRef, Form::AssignRef, Form::ValueValue] {
            let result = form.apply(copy(&a, 0), copy(&b, 0));
            let inputs = || format!("{} with {len_a} and {len_b} bytes", form.name());
            assert_product_residues(&result, &a, &b, &inputs);
        }
    }
}

#[test]
fn products_of_long_runs_agree_with_their_residues() {
    // Long operands of runs of zeros, ones and random limbs, past the lengths of the reference:
    // the levels of Karatsuba's recursion on uneven halves, and pieces of a shorter operand.
    let mut state = 0x3C6E_F372_A54F_F53A;
    let counts = if cfg!(miri) {
        vec![70, 100]
    } else {
        vec![300, 511, 512, 513, 777, 1024, 1500, 2049]
    };
    for &nx in &counts {
        for ny in [nx, nx - 1, nx / 2 + 1, nx / 2, nx / 3, 33] {
            let extra = below(&mut state, 8);
            let x = limb_runs(&mut state, nx, extra);
            let extra = below(&mut state, 8);
            let y = limb_runs(&mut state, ny, extra);
            let a = number(&x, below(&mut state, 2) == 0, ORDERS[below(&mut state, 2)]);
            let b = number(&y, below(&mut state, 2) == 0, ORDERS[below(&mut state, 2)]);
            let form = FORMS[below(&mut state, 6)];
            let result = form.apply(copy(&a, 0), copy(&b, 0));
            let inputs = || format!("{} with {nx} and {ny} limbs", form.name());
            assert_product_residues(&result, &a, &b, &inputs);
        }
    }
}

/// The limbs of the shorter operand around the sizes from which Toom-3 (120) and Toom-4 (224)
/// multiply, and a few past them, where their parts recurse into Karatsuba's method and Toom-3.
/// Under Miri, one past each threshold.
fn toom_limbs() -> Vec<usize> {
    if cfg!(miri) {
        vec![121, 225]
    } else {
        vec![
            119, 120, 121, 122, 123, 128, 129, 223, 224, 225, 226, 227, 256, 300, 365, 400,
        ]
    }
}

#[test]
fn multiplies_operands_of_every_shape_by_toom_cook() {
    // From the thresholds up, the longer operand as long as the shorter one or a few limbs
    // longer (Toom-4 and Toom-3), about 1.4 times as long (Toom-3 only), 1.5 to 4 times (Toom-4
    // cutting the longer one in four parts and the shorter one in two), and longer (pieces of
    // twice the shorter one, each multiplied by Toom-4x2, and a last one of what is left).
    let mut state = 0x2F8E_44D1_9AC3_57B0;
    for ny in toom_limbs() {
        let shapes = if cfg!(miri) {
            vec![ny, 2 * ny - 1]
        } else {
            vec![
                ny,
                ny + 1,
                ny + 3,
                ny * 4 / 3 + 1,
                ny * 7 / 5,
                ny * 3 / 2,
                ny * 3 / 2 + 1,
                ny * 7 / 4,
                2 * ny - 1,
                2 * ny,
                2 * ny + 1,
                ny * 5 / 2,
                3 * ny + 2,
                4 * ny - 1,
                4 * ny + 1,
                ny * 9 / 2 + 3,
                7 * ny + 5,
            ]
        };
        for nx in shapes {
            let ex = EXTRA_BYTES[below(&mut state, 3)];
            let ey = EXTRA_BYTES[below(&mut state, 3)];
            let x = if below(&mut state, 2) == 0 {
                limb_runs(&mut state, nx, ex)
            } else {
                random_magnitude(&mut state, 8 * nx + ex)
            };
            let y = limb_runs(&mut state, ny, ey);
            let k = below(&mut state, 16);
            let (left, right) = ORDER_PAIRS[k % 4];
            let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
            let a = number(&x, negative_a, left);
            let b = number(&y, negative_b, right);
            let form = FORMS[below(&mut state, 6)];
            if nx <= 4 * ny + 1 {
                assert_multiplies_in(&[form], &a, &b, &product(&a, &b));
                assert_multiplies_in(&[form], &b, &a, &product(&b, &a));
            } else {
                let result = form.apply(copy(&a, 0), copy(&b, 0));
                let inputs = || format!("{} with {nx} and {ny} limbs", form.name());
                assert_product_residues(&result, &a, &b, &inputs);
            }
        }
    }
}

/// A magnitude of `n` limbs cut as Toom's methods cut it, into parts of `k` limbs (the top one
/// shorter), each part one of: a random block, the block plus or minus one, all ones, all zeros,
/// or the sum of the parts below it and the one above it, which makes the value at -1 zero.
fn toom_parts(state: &mut u64, n: usize, k: usize) -> Vec<u8> {
    let block = random_magnitude(state, 8 * k);
    let mut limbs: Vec<u64> = Vec::with_capacity(n);
    let block_limbs: Vec<u64> = block
        .chunks(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    while limbs.len() < n {
        let len = (n - limbs.len()).min(k);
        let mut part: Vec<u64> = block_limbs[..len].to_vec();
        match below(state, 6) {
            0 => {
                // Plus one, with the carry, unless it would carry out.
                if let Some(i) = part.iter().position(|&l| l != u64::MAX) {
                    part[..i].iter_mut().for_each(|l| *l = 0);
                    part[i] += 1;
                }
            }
            1 => {
                if let Some(i) = part.iter().position(|&l| l != 0) {
                    part[..i].iter_mut().for_each(|l| *l = u64::MAX);
                    part[i] -= 1;
                }
            }
            2 => part.iter_mut().for_each(|l| *l = u64::MAX),
            3 => part.iter_mut().for_each(|l| *l = 0),
            _ => {}
        }
        limbs.extend(part);
    }
    // One time in three, for three parts, the middle one the sum of the others (halved, so that
    // it fits): x0 - x1 + x2 = 0.
    if below(state, 3) == 0 && n > 2 * k {
        let s = n - 2 * k;
        for i in 0..k {
            limbs[i] >>= 1;
            if i < s {
                limbs[2 * k + i] >>= 1;
            }
        }
        let mut carry = 0u64;
        for i in 0..k {
            let high = if i < s { limbs[2 * k + i] } else { 0 };
            let (v, c1) = limbs[i].overflowing_add(high);
            let (v, c2) = v.overflowing_add(carry);
            limbs[k + i] = v;
            carry = u64::from(c1 | c2);
        }
        assert_eq!(carry, 0);
    }
    let mut bytes: Vec<u8> = limbs.iter().flat_map(|l| l.to_le_bytes()).collect();
    if let Some(top @ 0) = bytes.last_mut() {
        *top = 1;
    }
    bytes
}

#[test]
fn toom_cook_takes_values_that_are_zero_negative_or_all_ones() {
    // Operands whose parts are equal, a unit apart, all ones, zero, or sum to zero at -1, for the
    // parts of Toom-3 and Toom-4 on both operands: the values at -1 and -2 zero, negative and
    // positive in every combination, and the values at 1, 2 and 1/2 at their largest, in
    // products and squares.
    let mut state = 0xD1B5_4A32_D192_ED03;
    let (count, sizes): (usize, Vec<usize>) = if cfg!(miri) {
        (2, vec![121, 225])
    } else {
        (12, vec![120, 121, 128, 170, 224, 225, 240, 300, 401])
    };
    for &n in &sizes {
        for _ in 0..count {
            let parts = [n.div_ceil(3), n.div_ceil(4)];
            let kx = parts[below(&mut state, 2)];
            let ky = parts[below(&mut state, 2)];
            let ny = n - below(&mut state, 3);
            let x = toom_parts(&mut state, n, kx);
            let y = toom_parts(&mut state, ny, ky);
            let k = below(&mut state, 16);
            let (left, right) = ORDER_PAIRS[k % 4];
            let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
            let a = number(&x, negative_a, left);
            let b = number(&y, negative_b, right);
            let form = FORMS[below(&mut state, 6)];
            assert_multiplies_in(&[form], &a, &b, &product(&a, &b));
            #[allow(clippy::eq_op)]
            let square = &a * &a;
            let inputs = || format!("&a * &a with a = {}", describe(&a));
            assert_result(&square, &product(&a, &a), a.byte_order, &inputs);
        }
    }
}

#[test]
fn toom_3_carries_out_of_its_low_coefficients() {
    // With X = B^k, x = 2^(64k - 1) + X + (X - 1) X^2 and y = 2^(64k - 1) + (X - 1) X + (X - 1) X^2,
    // the coefficient of X^2 of the product is X^2 - 1 and the sum of the coefficients below it
    // carries out of limb 4k - 1, into the top limb of that coefficient: a carry that random
    // operands meet about once in B^k products.
    let ks: Vec<usize> = if cfg!(miri) {
        vec![41]
    } else {
        vec![40, 41, 50, 63, 74]
    };
    for k in ks {
        let half = {
            let mut part = vec![0u8; 8 * k];
            part[8 * k - 1] = 0x80;
            part
        };
        let mut one = vec![0u8; 8 * k];
        one[0] = 1;
        let ones = vec![0xFF; 8 * k];
        let x = [half.as_slice(), &one, &ones].concat();
        let y = [half.as_slice(), &ones, &ones].concat();
        for (left, right) in ORDER_PAIRS {
            let a = number(&x, false, left);
            let b = number(&y, true, right);
            let form = FORMS[k % 6];
            assert_multiplies_in(&[form], &a, &b, &product(&a, &b));
        }
    }
}

// Number-theoretic transforms

/// Limbs of the shorter operand at and around the sizes from which products and squares are
/// computed by number-theoretic transforms: 180 for a longer operand of at least twice as many
/// limbs, 360 for a product and 380 for a square on aarch64 with NEON; 8192 and 16,384 on the
/// other targets; 125 and 250 under Miri.
fn transform_limbs() -> Vec<usize> {
    if cfg!(miri) {
        vec![125, 250]
    } else if cfg!(target_arch = "aarch64") {
        vec![179, 180, 181, 359, 360, 361, 379, 380, 381]
    } else {
        vec![8191, 8192, 16_383, 16_384]
    }
}

/// Checks `a * b` in `form` against the schoolbook reference when the operands are short enough
/// for it, and against the residues of the operands otherwise.
fn assert_long_product(form: Form, a: &Number, b: &Number) {
    let limbs = |n: &Number| n.value.len().div_ceil(8);
    if limbs(a) * limbs(b) <= size(1_000_000, 80_000) {
        assert_multiplies_in(&[form], a, b, &product(a, b));
    } else {
        let result = form.apply(copy(a, 0), copy(b, 0));
        let inputs = || format!("{} with {} and {} limbs", form.name(), limbs(a), limbs(b));
        assert_product_residues(&result, a, b, &inputs);
    }
}

#[test]
fn multiplies_by_transforms_around_their_thresholds() {
    // The shorter operand at and around the thresholds, the longer one as long, a limb longer,
    // just below and from twice as long (where the threshold for unbalanced operands applies),
    // and longer, with a few bytes above the whole limbs, in runs of zeros, ones and random
    // limbs; products in both orders, and squares.
    let mut state = 0x4B1D_2C9E_83F7_A065;
    for ny in transform_limbs() {
        let shapes = if cfg!(miri) {
            vec![ny, 2 * ny]
        } else {
            vec![ny, ny + 1, 2 * ny - 1, 2 * ny, 2 * ny + 1, 3 * ny + 5]
        };
        for nx in shapes {
            let ex = EXTRA_BYTES[below(&mut state, 3)];
            let ey = EXTRA_BYTES[below(&mut state, 3)];
            let x = limb_runs(&mut state, nx, ex);
            let y = limb_runs(&mut state, ny, ey);
            let k = below(&mut state, 16);
            let (left, right) = ORDER_PAIRS[k % 4];
            let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
            let a = number(&x, negative_a, left);
            let b = number(&y, negative_b, right);
            let form = FORMS[below(&mut state, 6)];
            assert_long_product(form, &a, &b);
            assert_long_product(form, &b, &a);
            if nx <= ny + 1 && !cfg!(miri) {
                #[allow(clippy::eq_op)]
                let square = &b * &b;
                let inputs = || format!("&b * &b with b = {}", describe(&b));
                assert_product_residues(&square, &b, &b, &inputs);
            }
        }
    }
}

/// Every pair of byte orders, or under Miri one picked at random.
fn order_pairs_to_run() -> Vec<(ByteOrder, ByteOrder)> {
    if cfg!(miri) {
        vec![ORDER_PAIRS[pick(4)]]
    } else {
        ORDER_PAIRS.to_vec()
    }
}

/// The magnitude `(256^n - 1)(256^m - 1) = 256^(n+m) - 256^n - 256^m + 1` for `n >= m >= 1`,
/// least significant byte first: a one, `m - 1` zeros, `n - m` bytes of `0xFF`, `0xFE` and
/// `m - 1` bytes of `0xFF`.
fn all_ones_product(n: usize, m: usize) -> Vec<u8> {
    let mut bytes = vec![1];
    bytes.resize(m, 0);
    bytes.resize(n, 0xFF);
    bytes.push(0xFE);
    bytes.resize(n + m, 0xFF);
    bytes
}

/// Limbs of operands for which the transforms take each of their widths of coefficients and
/// numbers of primes on aarch64: 4 primes and 48 bits for 1500 and 3000 limbs, 5 and 56 for
/// 36,000 by 18,000, 5 and 64 for 500, 8192 and 16,384, 6 and 72 for 32,768, 7 and 88 for 700 and
/// 5000, 8 and 104 for 400 and for 6000 by 181. At 16,384 limbs each, the 2^14 coefficients of
/// 64 bits are the most that 5 primes take: all ones give the largest coefficient they are exact
/// for. Under Miri, and on the other targets where the transforms start at 16,384 limbs, a few
/// that reach them.
fn transform_shapes() -> Vec<(usize, usize)> {
    if cfg!(miri) {
        vec![(250, 250), (260, 130)]
    } else if cfg!(target_arch = "aarch64") {
        vec![
            (400, 400),
            (500, 500),
            (700, 700),
            (1500, 1500),
            (3000, 3000),
            (5000, 5000),
            (8192, 8192),
            (16_384, 16_384),
            (32_768, 32_768),
            (36_000, 18_000),
            (6000, 181),
        ]
    } else {
        vec![(16_384, 16_384), (20_000, 20_000), (40_000, 20_000)]
    }
}

#[test]
fn transforms_are_exact_for_the_largest_coefficients() {
    // Operands of all ones give every coefficient of the product its largest value,
    // m (2^w - 1)^2 for a shorter operand of m coefficients, which the primes must exceed twice:
    // products and squares for each width of coefficients, in both byte orders, against the
    // product's closed form.
    let extras: &[usize] = if cfg!(miri) { &[5] } else { &[0, 5] };
    for (nx, ny) in transform_shapes() {
        for &extra in extras {
            let (n, m) = (8 * nx + extra, 8 * ny);
            let want = expected(&all_ones_product(n, m), true);
            for (left, right) in order_pairs_to_run() {
                let a = number(&vec![0xFF; n], false, left);
                let b = number(&vec![0xFF; m], true, right);
                let result = &a * &b;
                let inputs = || format!("&a * &b of all ones, {n} and {m} bytes");
                assert_result(&result, &want, left, &inputs);
            }
            let a = number(&vec![0xFF; n], true, ORDERS[extra % 2]);
            #[allow(clippy::eq_op)]
            let square = &a * &a;
            let inputs = || format!("&a * &a of {n} bytes of 0xFF");
            assert_result(
                &square,
                &expected(&all_ones_product(n, n), false),
                a.byte_order,
                &inputs,
            );
        }
    }
}

#[test]
fn transforms_take_sparse_operands_and_powers_of_two() {
    // A power of two times a number is the number shifted, and (256^n + 1)(256^n - 1) is
    // 256^(2n) - 1: operands whose coefficients are mostly zero, with their nonzero ones at the
    // ends, where the transforms fold the upper half of an operand onto the lower one.
    let mut state = 0x7F4A_7C15_9E37_79B9;
    for (nx, ny) in transform_shapes() {
        let x = random_magnitude(&mut state, 8 * nx);
        let shift = 64 * ny - 1 - below(&mut state, 8);
        let mut power = vec![0u8; shift / 8];
        power.push(1 << (shift % 8));
        for (left, right) in order_pairs_to_run() {
            let a = number(&x, false, left);
            let b = number(&power, true, right);
            let result = &a * &b;
            let shifted = &number(&x, true, left) << shift;
            let inputs = || format!("&a * 2^{shift} with {nx} limbs");
            assert_result(&result, &value_of(&shifted), left, &inputs);
        }
        let n = 8 * ny;
        let mut plus = vec![1u8];
        plus.resize(n, 0);
        plus.push(1);
        let a = number(&plus, false, ORDERS[nx % 2]);
        let b = number(&vec![0xFF; n], false, ORDERS[ny % 2]);
        let result = &a * &b;
        let inputs = || format!("(256^{n} + 1)(256^{n} - 1)");
        assert_result(
            &result,
            &expected(&vec![0xFF; 2 * n], false),
            a.byte_order,
            &inputs,
        );
    }
}

#[test]
fn squares_by_transforms_agree_with_products() {
    // `&a * &a` takes one transform of `a` where `&a * &b` takes two: both must give the same
    // number, and its residues the square of `a`'s.
    let mut state = 0x1F83_D9AB_FB41_BD6B;
    for (n, _) in transform_shapes() {
        let x = if below(&mut state, 2) == 0 {
            limb_runs(&mut state, n, 3)
        } else {
            random_magnitude(&mut state, 8 * n)
        };
        let a = number(&x, below(&mut state, 2) == 0, ORDERS[below(&mut state, 2)]);
        let b = copy(&a, 0);
        #[allow(clippy::eq_op)]
        let square = &a * &a;
        let inputs = || format!("&a * &a with {n} limbs");
        assert_result(&square, &value_of(&(&a * &b)), a.byte_order, &inputs);
        assert_product_residues(&square, &a, &a, &inputs);
    }
}

// Squares

#[test]
fn squares_a_borrowed_number_multiplied_by_itself() {
    // `&a * &a` takes the square, which computes each product of two different limbs once.
    let mut state = 0x6C07_8965_D5F1_9B2E;
    let mut lens: Vec<usize> = (0..=40).collect();
    for limbs in product_limbs() {
        lens.push(8 * limbs);
        lens.push(8 * limbs + 3);
    }
    for len in lens {
        for x in [
            random_magnitude(&mut state, len),
            vec![0xFF; len],
            limb_runs(&mut state, len / 8, len % 8),
        ] {
            let a = number(&x, below(&mut state, 2) == 0, ORDERS[below(&mut state, 2)]);
            #[allow(clippy::eq_op)]
            let result = &a * &a;
            let inputs = || format!("&a * &a with a = {}", describe(&a));
            assert_result(&result, &product(&a, &a), a.byte_order, &inputs);
        }
    }
    if !cfg!(miri) {
        for len in [1800, 4000, 8191, 12_000, 40_000] {
            let a = number(&random_magnitude(&mut state, len), true, ORDERS[len % 2]);
            #[allow(clippy::eq_op)]
            let result = &a * &a;
            let inputs = || format!("&a * &a with {len} bytes");
            assert_product_residues(&result, &a, &a, &inputs);
        }
    }
}

// Carries

#[test]
fn products_of_all_ones_carry_through_every_limb() {
    // (256^n - 1)(256^m - 1) = 256^(n+m) - 256^n - 256^m + 1: the largest products of their
    // lengths, whose columns all carry.
    for n in (1..=size(100, 20)).chain([size(300, 40), size(1000, 70), size(2000, 80)]) {
        for m in [1, 2, 7, 8, 9, n / 2 + 1, n] {
            for (left, right) in ORDER_PAIRS {
                let a = number(&vec![0xFF; n], false, left);
                let b = number(&vec![0xFF; m], true, right);
                let expected = product(&a, &b);
                for form in forms_to_run() {
                    assert_multiplies_in(&[*form], &a, &b, &expected);
                }
            }
        }
    }
}

#[test]
fn powers_of_two_shift_the_other_operand() {
    // 2^k times x is x shifted up by k bits: for every bit of a few limbs.
    let mut state = 0x9216_D5D9_8979_FB1B;
    for len in [1, 9, 33, 300] {
        let x = random_magnitude(&mut state, len);
        for k in (0..size(140, 20)).step_by(size(1, 7)) {
            let mut power = vec![0; k / 8];
            power.push(1 << (k % 8));
            for (left, right) in ORDER_PAIRS {
                let a = number(&x, false, left);
                let b = number(&power, false, right);
                assert_multiplies_in(forms_to_run(), &a, &b, &product(&a, &b));
                assert_multiplies_in(forms_to_run(), &b, &a, &product(&b, &a));
            }
        }
    }
}

// Zero and one

#[test]
fn a_product_with_zero_is_one_zero_byte() {
    // Every way of writing zero, times zero or a number of any length, either side, is `[0]` and
    // not negative.
    let mut state = 0xF7B4_2C15_7EE6_38AD;
    for (left, right) in ORDER_PAIRS {
        for z in zeros(left) {
            for other in zeros(right) {
                assert_multiplies(&z, &other, &zero());
            }
            for len in [1, 8, 16, 17, 100] {
                for negative in [false, true] {
                    let x = number(&random_magnitude(&mut state, len), negative, right);
                    assert_multiplies_in(forms_to_run(), &z, &x, &zero());
                    assert_multiplies_in(forms_to_run(), &x, &z, &zero());
                }
            }
        }
    }
}

#[test]
fn multiplying_by_one_or_minus_one_gives_the_number_or_its_negation() {
    // x * 1 = x and x * -1 = -x in the byte order of x, also when x has zero bytes above its most
    // significant byte, and 1 * x the same in the byte order of the one.
    let mut state = 0x2FE1_66C5_4C04_8A9B;
    for len in (1..=size(40, 4)).chain([size(500, 40)]) {
        let magnitude = random_magnitude(&mut state, len);
        let mut padded = magnitude.clone();
        padded.resize(len + 3, 0);
        for (left, right) in ORDER_PAIRS {
            for negative in [false, true] {
                for x in [
                    number(&magnitude, negative, left),
                    number(&padded, negative, left),
                ] {
                    for one_negative in [false, true] {
                        let one = number(&[1], one_negative, right);
                        let result = expected(&magnitude, negative != one_negative);
                        assert_multiplies_in(forms_to_run(), &x, &one, &result);
                        assert_multiplies_in(forms_to_run(), &one, &x, &result);
                    }
                }
            }
        }
    }
}

// Signs and byte orders

#[test]
fn multiplies_every_combination_of_signs() {
    let magnitudes: [(i128, i128); 8] = [
        (5, 3),
        (0xFF, 0xFF),
        (0x1_0000, 0xFFFF),
        ((1 << 63) - 1, (1 << 63) - 1),
        (1 << 62, 3),
        (0x0123_4567_89AB_CDEF, 0x0FED_CBA9_8765_4321),
        (1, (1 << 63) - 1),
        (0x100, 0x100),
    ];
    for (x, y) in magnitudes {
        for (negative_a, negative_b) in SIGN_PAIRS {
            let x = if negative_a { -x } else { x };
            let y = if negative_b { -y } else { y };
            for (left, right) in ORDER_PAIRS {
                let a = Number::new_from_i128(x, left);
                let b = Number::new_from_i128(y, right);
                assert_multiplies(&a, &b, &expected_i128(x * y));
            }
        }
    }
    // The same on long numbers, against the reference.
    let mut state = 0x0B4E_5AC7_2D1F_9E63;
    let long = random_magnitude(&mut state, size(300, 40));
    let short = random_magnitude(&mut state, size(29, 9));
    for (x, y) in [(&long, &short), (&short, &long), (&long, &long)] {
        for (negative_a, negative_b) in SIGN_PAIRS {
            for (left, right) in ORDER_PAIRS {
                let a = number(x, negative_a, left);
                let b = number(y, negative_b, right);
                assert_multiplies(&a, &b, &product(&a, &b));
            }
        }
    }
}

#[test]
fn mixed_byte_orders_in_every_form() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    // (a, b, the product as stored in the byte order of a, and its sign)
    let cases = [
        // 0x01_02 * 0x03 = 0x03_06.
        (
            plus(&[0x01, 0x02], be, 2),
            plus(&[0x03], le, 1),
            vec![0x03, 0x06],
            false,
        ),
        (
            plus(&[0x03], le, 1),
            plus(&[0x01, 0x02], be, 2),
            vec![0x06, 0x03],
            false,
        ),
        // 0x01_02_03 * -0x20_10 = -0x20_50_80_30.
        (
            plus(&[0x01, 0x02, 0x03], be, 3),
            minus(&[0x10, 0x20], le, 2),
            vec![0x20, 0x50, 0x80, 0x30],
            true,
        ),
        (
            minus(&[0x10, 0x20], le, 2),
            plus(&[0x01, 0x02, 0x03], be, 3),
            vec![0x30, 0x80, 0x50, 0x20],
            true,
        ),
        // Seventeen bytes each, past the short path: (2^136 - 1)^2 = 2^272 - 2^137 + 1.
        (
            plus(&[0xFF; 17], be, 17),
            plus(&[0xFF; 17], le, 17),
            [vec![0xFF; 16], vec![0xFE], vec![0; 16], vec![1]].concat(),
            false,
        ),
    ];
    for (a, b, result, negative) in cases {
        let magnitude = le_bytes(&Number {
            value: result,
            negative,
            byte_order: a.byte_order,
        });
        assert_multiplies(&a, &b, &expected(&magnitude, negative));
    }
}

#[test]
fn ref_times_value_reverses_into_the_left_byte_order() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    let mut state = 0x4D1C_2E79_A3B8_6F05;
    // Short and long, written as `a` writes them, in the vector of `b`, which has room.
    for (len_a, len_b) in [(2, 1), (16, 16), (20, 30), (300, 200)] {
        for (left, right) in [(be, le), (le, be)] {
            let a = number(&random_magnitude(&mut state, len_a), false, left);
            let b = stored(
                &in_order(&random_magnitude(&mut state, len_b), right),
                true,
                right,
                len_a + len_b,
            );
            let expected = product(&a, &b);
            let pointer = b.value.as_ptr();
            let message = format!("&a * b with a = {}, b = {}", describe(&a), describe(&b));
            let (result, allocations) = count(|| &a * b);
            assert_result(&result, &expected, left, &|| message.clone());
            assert_eq!(result.value.as_ptr(), pointer, "{message}");
            assert_eq!(allocations, 0, "{message}");
        }
    }
}

// Properties

#[test]
fn multiplication_is_commutative() {
    let mut state = 0xBB67_AE85_84CA_A73B;
    for _ in 0..size(300, 8) {
        let a = random_operand(&mut state, 300);
        let b = random_operand(&mut state, 300);
        let form = FORMS[below(&mut state, 6)];
        let (ab, ba) = (multiply(form, &a, &b), multiply(form, &b, &a));
        assert_eq!(
            value_of(&ab),
            value_of(&ba),
            "{}: a = {}, b = {}",
            form.name(),
            describe(&a),
            describe(&b)
        );
    }
}

#[test]
fn multiplication_is_associative() {
    let mut state = 0x3C6E_F372_FE94_F82B;
    for _ in 0..size(200, 6) {
        let a = random_operand(&mut state, 200);
        let b = random_operand(&mut state, 200);
        let c = random_operand(&mut state, 200);
        let form = FORMS[below(&mut state, 6)];
        let ab_c = multiply(form, &multiply(form, &a, &b), &c);
        let a_bc = multiply(form, &a, &multiply(form, &b, &c));
        assert_eq!(
            value_of(&ab_c),
            value_of(&a_bc),
            "{}: a = {}, b = {}, c = {}",
            form.name(),
            describe(&a),
            describe(&b),
            describe(&c)
        );
    }
}

#[test]
fn multiplication_distributes_over_addition_and_subtraction() {
    // a (b + c) = a b + a c and a (b - c) = a b - a c.
    let mut state = 0xA54F_F53A_5F1D_36F1;
    for _ in 0..size(200, 6) {
        let a = random_operand(&mut state, 200);
        let b = random_operand(&mut state, 200);
        let c = random_operand(&mut state, 200);
        let message = format!(
            "a = {}, b = {}, c = {}",
            describe(&a),
            describe(&b),
            describe(&c)
        );
        let (ab, ac) = (&a * &b, &a * &c);
        assert_eq!(
            value_of(&(&a * &(&b + &c))),
            value_of(&(&ab + &ac)),
            "a (b + c), {message}"
        );
        assert_eq!(
            value_of(&(&a * &(&b - &c))),
            value_of(&(&ab - &ac)),
            "a (b - c), {message}"
        );
    }
}

#[test]
fn borrowed_operands_are_left_as_they_were() {
    let mut state = 0x1F83_D9AB_FB41_BD6B;
    for _ in 0..size(200, 8) {
        let a = random_operand(&mut state, 100);
        let b = random_operand(&mut state, 100);
        let (a_before, b_before) = (copy(&a, 0), copy(&b, 0));
        let message = format!("a = {}, b = {}", describe(&a), describe(&b));
        let _ = &a * &b;
        assert!(
            same(&a, &a_before) && same(&b, &b_before),
            "&a * &b: {message}"
        );
        #[allow(clippy::eq_op)]
        let _ = &a * &a;
        assert!(same(&a, &a_before), "&a * &a: {message}");
        let _ = copy(&a, 1) * &b;
        assert!(same(&b, &b_before), "a * &b: {message}");
        let _ = &a * copy(&b, 1);
        assert!(same(&a, &a_before), "&a * b: {message}");
        let mut result = copy(&a, 1);
        result *= &b;
        assert!(same(&b, &b_before), "a *= &b: {message}");
    }
}

// Allocation

/// Checks where `a * b` in `form` puts the product, with the operands in vectors of the
/// capacities of `a` and `b`, and that nothing else is allocated. The forms that take `a` by
/// value build it in the vector of `a` whenever that has room for it, and `&a * b` in that of `b`
/// whenever that has room; when only `b` has room, `a * b` and `a *= b` build it there too.
/// Otherwise the product gets one new vector, as it always does from `&a * &b`.
fn assert_placement(form: Form, a: &Number, b: &Number) {
    let expected = product(a, b);
    let len = expected.magnitude.len();
    let left = copy_with_capacity(a);
    let right = copy_with_capacity(b);
    let left_has_room = left.value.capacity() >= len;
    let right_has_room = right.value.capacity() >= len;
    let (left_pointer, right_pointer) = (left.value.as_ptr(), right.value.as_ptr());
    let (result, allocations) = count(|| form.apply(left, right));
    let inputs = || {
        format!(
            "{} with a = {} (capacity {}), b = {} (capacity {})",
            form.name(),
            describe(a),
            a.value.capacity(),
            describe(b),
            b.value.capacity()
        )
    };
    assert_result(&result, &expected, a.byte_order, &inputs);
    let in_left = result.value.as_ptr() == left_pointer;
    let in_right = result.value.as_ptr() == right_pointer;
    match placement(form, left_has_room, right_has_room) {
        Placement::Left => {
            assert!(in_left, "{}: not in the vector of a", inputs());
            assert_eq!(allocations, 0, "{}: allocations", inputs());
        }
        Placement::Right => {
            assert!(in_right, "{}: not in the vector of b", inputs());
            assert_eq!(allocations, 0, "{}: allocations", inputs());
        }
        Placement::New => assert_eq!(allocations, 1, "{}: allocations", inputs()),
    }
}

/// Where the allocation promises put the product of `form`, given which operands have room for
/// it.
fn placement(form: Form, left_has_room: bool, right_has_room: bool) -> Placement {
    match form {
        Form::RefRef => Placement::New,
        Form::RefValue if right_has_room => Placement::Right,
        Form::RefValue => Placement::New,
        _ if left_has_room => Placement::Left,
        Form::ValueValue | Form::AssignValue if right_has_room => Placement::Right,
        _ => Placement::New,
    }
}

#[test]
fn placement_follows_the_capacities_of_random_operands() {
    // Capacities around the length of the product, where a vector just has room for it or just
    // lacks it, for short operands and for long ones that are computed on the stack.
    let mut state = 0x2545_F491_4F6C_DD1D;
    for round in 0..size(1500, 40) {
        let max = if round % 3 == 0 { 300 } else { 24 };
        let a = random_operand(&mut state, max);
        let b = random_operand(&mut state, max);
        if a.value.len() + b.value.len() > 1000 {
            continue;
        }
        let len = product(&a, &b).magnitude.len();
        let mut capacity = |n: &Number| {
            let capacity = match below(&mut state, 5) {
                0 => len.saturating_sub(1),
                1 => len,
                2 => len + 1,
                3 => 0,
                _ => len + 9,
            };
            stored(
                &n.value,
                n.negative,
                n.byte_order,
                capacity.max(n.value.len()),
            )
        };
        let a = capacity(&a);
        let b = capacity(&b);
        for form in FORMS {
            assert_placement(form, &a, &b);
        }
    }
}

#[test]
fn mul_assign_grows_a_number_in_its_own_vector() {
    // 0xFF * 0xFF = 0xFE_01 and (2^136 - 1)^2 in their operand's vector: in big-endian order the
    // old bytes move back to make room.
    for order in ORDERS {
        for len in [1, 17, 100] {
            for form in [
                Form::AssignRef,
                Form::AssignValue,
                Form::ValueRef,
                Form::ValueValue,
            ] {
                let a = plus(&vec![0xFF; len], order, 2 * len);
                let b = number(&vec![0xFF; len], false, ByteOrder::LittleEndian);
                let expected = product(&a, &b);
                let pointer = a.value.as_ptr();
                let result = form.apply(a, b);
                let inputs = || format!("{} with {len} bytes of 0xFF", form.name());
                assert_result(&result, &expected, order, &inputs);
                assert_eq!(result.value.as_ptr(), pointer, "{}", inputs());
            }
        }
    }
}

#[test]
fn long_products_allocate_one_temporary_buffer_at_most() {
    // Past the working memory on the stack, one buffer is allocated for the computation and freed
    // before the product is returned: a product built in an operand's vector makes one
    // allocation, one built in a new vector two. The operands are below the transforms on every
    // target.
    let mut state = 0x5BE0_CD19_137E_2179;
    let len = size(2_800, 1_600);
    let a = number(
        &random_magnitude(&mut state, len),
        false,
        ByteOrder::LittleEndian,
    );
    let b = number(
        &random_magnitude(&mut state, len),
        true,
        ByteOrder::BigEndian,
    );
    let (result, allocations) = count(|| &a * &b);
    assert_product_residues(&result, &a, &b, &|| "&a * &b".into());
    assert_eq!(allocations, 2, "&a * &b");
    let roomy = copy(&a, len);
    let pointer = roomy.value.as_ptr();
    let (result, allocations) = count(|| roomy * &b);
    assert_product_residues(&result, &a, &b, &|| "a * &b".into());
    assert_eq!(result.value.as_ptr(), pointer, "a * &b");
    assert_eq!(allocations, 1, "a * &b");
}

#[test]
fn products_by_transforms_allocate_one_buffer_more() {
    // The transforms allocate their own buffer besides the working memory of the operands and
    // the product, both freed before the product is returned. The operands are long enough for
    // the transforms on every target, and for the working memory of a square, the operand and
    // the product, to be past the stack's 1024 limbs.
    let mut state = 0x6A09_E667_F3BC_C908;
    let len = size(140_000, 2_800);
    let a = number(
        &random_magnitude(&mut state, len),
        true,
        ByteOrder::BigEndian,
    );
    let b = number(
        &random_magnitude(&mut state, len),
        false,
        ByteOrder::LittleEndian,
    );
    let (result, allocations) = count(|| &a * &b);
    assert_product_residues(&result, &a, &b, &|| "&a * &b".into());
    assert_eq!(allocations, 3, "&a * &b");
    let roomy = copy(&a, len);
    let pointer = roomy.value.as_ptr();
    let (result, allocations) = count(|| roomy * &b);
    assert_product_residues(&result, &a, &b, &|| "a * &b".into());
    assert_eq!(result.value.as_ptr(), pointer, "a * &b");
    assert_eq!(allocations, 2, "a * &b");
    #[allow(clippy::eq_op)]
    let (result, allocations) = count(|| &a * &a);
    assert_product_residues(&result, &a, &a, &|| "&a * &a".into());
    assert_eq!(allocations, 3, "&a * &a");
}
