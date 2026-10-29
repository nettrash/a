use a::math::numbers::{ByteOrder, Number};

use super::common::*;
use crate::allocations::count;

// Expected differences

/// The difference of the integers `a` and `b` hold, worked out independently of the
/// implementation: the reference sum of `a` and the negation of `b`, in two's complement, where
/// no magnitudes are compared. The tests on small values check it against `i128` arithmetic.
fn difference(a: &Number, b: &Number) -> Expected {
    reference(a, &negated(b))
}

/// A copy of `n` with the opposite sign and the capacity of `n`: the right operand whose
/// subtraction is the addition of `n`, for the tables of sums the tests of placement share
/// with the addition.
fn turned(n: &Number) -> Number {
    Number {
        negative: !n.negative,
        ..copy_with_capacity(n)
    }
}

// The forms of subtraction

/// The six ways to write a subtraction. Every form must give the same difference, so the tests
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
            Form::ValueValue => "a - b",
            Form::ValueRef => "a - &b",
            Form::RefValue => "&a - b",
            Form::RefRef => "&a - &b",
            Form::AssignValue => "a -= b",
            Form::AssignRef => "a -= &b",
        }
    }

    /// Subtracts `b` from `a` in this form; an operand the form only borrows is dropped
    /// afterwards.
    fn apply(self, a: Number, b: Number) -> Number {
        match self {
            Form::ValueValue => a - b,
            Form::ValueRef => a - &b,
            Form::RefValue => &a - b,
            Form::RefRef => &a - &b,
            Form::AssignValue => {
                let mut a = a;
                a -= b;
                a
            }
            Form::AssignRef => {
                let mut a = a;
                a -= &b;
                a
            }
        }
    }

    /// The addition written the same way: which operands it takes by value and which it
    /// borrows.
    fn add(self, a: Number, b: Number) -> Number {
        match self {
            Form::ValueValue => a + b,
            Form::ValueRef => a + &b,
            Form::RefValue => &a + b,
            Form::RefRef => &a + &b,
            Form::AssignValue => {
                let mut a = a;
                a += b;
                a
            }
            Form::AssignRef => {
                let mut a = a;
                a += &b;
                a
            }
        }
    }
}

/// Every form, or under Miri one picked at random: for the tests that check many differences.
fn forms_to_run() -> &'static [Form] {
    if cfg!(miri) {
        let k = pick(FORMS.len());
        &FORMS[k..=k]
    } else {
        &FORMS
    }
}

/// Checks `a - b` against `expected` in every form, as [`assert_subtracts_in`] does.
fn assert_subtracts(a: &Number, b: &Number, expected: &Expected) {
    assert_subtracts_in(&FORMS, a, b, expected);
}

/// Checks `a - b` against `expected` in each of `forms`: with the vectors of the operands the
/// form takes by value as tight as their values, with one byte to spare, and with room for any
/// difference, and for `a - b` and `a -= b` also with room in `b` only, so that each form meets
/// each way it can place the difference. Under Miri each form runs with one of these setups,
/// picked at random.
fn assert_subtracts_in(forms: &[Form], a: &Number, b: &Number, expected: &Expected) {
    for &form in forms {
        let spares = spares(form, a, b);
        let spares = if cfg!(miri) {
            let k = pick(spares.len());
            &spares[k..=k]
        } else {
            &spares[..]
        };
        for &(spare_a, spare_b) in spares {
            assert_subtracts_with(form, spare_a, spare_b, a, b, expected);
        }
    }
}

/// The capacity setups [`assert_subtracts_in`] runs `form` with, as the bytes to spare in the
/// vectors of `a` and `b`.
fn spares(form: Form, a: &Number, b: &Number) -> Vec<(usize, usize)> {
    // A difference is at most one byte longer than the longer operand.
    let room = a.value.len().max(b.value.len()) + 1;
    match form {
        Form::RefRef => vec![(0, 0)],
        Form::ValueValue | Form::AssignValue => vec![(0, 0), (1, 1), (room, room), (0, room)],
        Form::ValueRef | Form::RefValue | Form::AssignRef => vec![(0, 0), (1, 1), (room, room)],
    }
}

/// Checks `a - b` against `expected` in `form`, with `spare_a` and `spare_b` bytes to spare in
/// the vectors of the operands the form takes by value.
fn assert_subtracts_with(
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

/// Checks `a - b` against `expected` in one of the forms and capacity setups
/// [`assert_subtracts_in`] runs: the one at `turn` among all of them, after which `turn` counts
/// on. For the longest operands, which take a while in every one of them in a debug build: taken
/// in turn, every setup still meets operands of every size.
fn assert_subtracts_in_turn(turn: &mut usize, a: &Number, b: &Number, expected: &Expected) {
    let setups: Vec<(Form, (usize, usize))> = FORMS
        .into_iter()
        .flat_map(|form| {
            spares(form, a, b)
                .into_iter()
                .map(move |spare| (form, spare))
        })
        .collect();
    let (form, (spare_a, spare_b)) = setups[*turn % setups.len()];
    *turn += 1;
    assert_subtracts_with(form, spare_a, spare_b, a, b, expected);
}

/// `a - b` in `form`, checked to be in the shortest form and in the byte order of `a`.
fn subtract(form: Form, a: &Number, b: &Number) -> Number {
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

// Small values against `i128` arithmetic

#[test]
fn subtracts_small_values_as_i128() {
    let mut state = 0x7137_449C_A5DB_E3B1;
    for _ in 0..size(1000, 40) {
        let x = random_i128(&mut state);
        let y = random_i128(&mut state);
        let expected = expected_i128(x - y);
        for (left, right) in ORDER_PAIRS {
            let a = Number::new_from_i128(x, left);
            let b = Number::new_from_i128(y, right);
            // The reference of the tests on big numbers agrees with `i128` arithmetic.
            assert_eq!(difference(&a, &b), expected, "reference for {x} - {y}");
            assert_subtracts_in(forms_to_run(), &a, &b, &expected);
        }
    }
}

#[test]
fn subtracts_every_pair_of_small_integers() {
    // From -300 to 300: every borrow from the second byte and every carry into it, in every
    // combination of signs. Each pair runs in one form, pair of byte orders and spare capacity,
    // taken in turn, so that every one of them meets values of every kind.
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
            let inputs = || format!("{x} - {y} as {} with {spare} bytes to spare", form.name());
            assert_result(&result, &expected_i128((x - y).into()), left, &inputs);
        }
    }
}

// Big values against the reference

#[test]
fn subtracts_random_big_numbers_as_the_reference() {
    let mut state = 0xB5C0_FBCF_E9B5_DBA5;
    for _ in 0..size(500, 16) {
        let a = random_operand(&mut state, 600);
        let b = random_operand(&mut state, 600);
        assert_subtracts_in(forms_to_run(), &a, &b, &difference(&a, &b));
    }
}

#[test]
fn subtracts_every_pair_of_lengths_up_to_40_bytes() {
    // Up to five 8-byte words, with every remainder after whole words and after groups of
    // them. Every pair of lengths runs in every combination of byte orders and signs, or under
    // Miri in one picked at random.
    let mut state = 0x3956_C25B_F348_B538;
    for len_a in 0..=40 {
        for len_b in 0..=40 {
            for k in 0..size(16, 1) {
                let k = if cfg!(miri) { below(&mut state, 16) } else { k };
                let (left, right) = ORDER_PAIRS[k % 4];
                let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
                let a = number(&random_magnitude(&mut state, len_a), negative_a, left);
                let b = number(&random_magnitude(&mut state, len_b), negative_b, right);
                assert_subtracts_in(forms_to_run(), &a, &b, &difference(&a, &b));
            }
        }
    }
}

#[test]
fn subtracts_numbers_of_millions_of_bytes() {
    let mut state = 0x59F1_11F1_923F_82A4;
    let len = size(2_000_000, 1_000);
    let x = random_magnitude(&mut state, len);
    let y = random_magnitude(&mut state, len - 5);
    for ((left, right), (negative_a, negative_b)) in ORDER_PAIRS.into_iter().zip(SIGN_PAIRS) {
        let a = number(&x, negative_a, left);
        let b = number(&y, negative_b, right);
        // Both ways round, so that the difference is built from the larger magnitude and from
        // the smaller one; a few forms only, as each takes a while in a debug build: in a new
        // vector, in place with room in the vector of `a`, and in that of `b`.
        for (a, b) in [(&a, &b), (&b, &a)] {
            let expected = difference(a, b);
            for (form, spare_a, spare_b) in [
                (Form::RefRef, 0, 0),
                (Form::AssignRef, 1, 0),
                (Form::RefValue, 0, 6),
            ] {
                let result = form.apply(copy(a, spare_a), copy(b, spare_b));
                let inputs = || {
                    format!(
                        "{} with a = {}, b = {}",
                        form.name(),
                        describe(a),
                        describe(b)
                    )
                };
                assert_result(&result, &expected, a.byte_order, &inputs);
            }
        }
    }
}

// Borrow and carry ripples

#[test]
fn borrows_ripple_through_every_byte() {
    // 256^n - 1 = [0xFF; n], one byte shorter than the longer operand, and 1 - 256^n its
    // negation; the same with both operands negated.
    for n in (1..=size(100, 40)).chain([size(10_000, 300), size(100_000, 1_000)]) {
        let all_ones = vec![0xFF; n];
        let mut power = vec![0; n];
        power.push(1);
        for (left, right) in ORDER_PAIRS {
            for negative in [false, true] {
                let power = number(&power, negative, left);
                let one = number(&[1], negative, right);
                assert_subtracts_in(forms_to_run(), &power, &one, &expected(&all_ones, negative));
                let power = number(&le_bytes(&power), negative, right);
                let one = number(&[1], negative, left);
                assert_subtracts_in(
                    forms_to_run(),
                    &one,
                    &power,
                    &expected(&all_ones, !negative),
                );
            }
        }
    }
}

#[test]
fn carries_ripple_through_every_byte() {
    // [0xFF; n] - -1 = 256^n, one byte longer than the longer operand, and -1 - [0xFF; n] its
    // negation.
    for n in (1..=size(100, 40)).chain([size(10_000, 300), size(100_000, 1_000)]) {
        let all_ones = vec![0xFF; n];
        let mut power = vec![0; n];
        power.push(1);
        for (left, right) in ORDER_PAIRS {
            for negative in [false, true] {
                let ones = number(&all_ones, negative, left);
                let one = number(&[1], !negative, right);
                assert_subtracts_in(forms_to_run(), &ones, &one, &expected(&power, negative));
                let one = number(&[1], negative, left);
                let ones = number(&all_ones, !negative, right);
                assert_subtracts_in(forms_to_run(), &one, &ones, &expected(&power, negative));
            }
        }
    }
}

#[test]
fn borrows_and_carries_go_on_or_stop_above_the_shorter_operand() {
    // The borrow (carry) out of the shorter operand goes into the bytes of the longer one above
    // it: on through all zeros (ones) to the top, or stopping in the first byte above, with up to
    // two limbs of bytes above the shorter operand, and the shorter operand below, at and above
    // a whole limb.
    let mut state = 0x923F_82A4_AB1C_5ED5;
    for ly in [1, 7, 8, 9, 15, 16, 17, 24] {
        for above in 1..=17 {
            let lx = ly + above;
            let y = random_magnitude(&mut state, ly);
            // 256^(lx - 1): a borrow out of `y` goes on to the top, and takes its 1.
            let mut power = vec![0; lx - 1];
            power.push(1);
            // All ones: a carry out of `y` goes on to the top, and the result is a byte longer.
            let ones = vec![0xFF; lx];
            // A 1 in the byte above `y`, where a borrow stops.
            let mut stop = power.clone();
            stop[ly] = 1;
            for x in [&power, &ones, &stop] {
                for k in 0..size(16, 1) {
                    let k = if cfg!(miri) { below(&mut state, 16) } else { k };
                    let (left, right) = ORDER_PAIRS[k % 4];
                    let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
                    let a = number(x, negative_a, left);
                    let b = number(&y, negative_b, right);
                    assert_subtracts_in(forms_to_run(), &a, &b, &difference(&a, &b));
                    assert_subtracts_in(forms_to_run(), &b, &a, &difference(&b, &a));
                }
            }
        }
    }
}

// Cancellation

#[test]
fn number_minus_itself_is_zero() {
    let mut state = 0xD807_AA98_1283_5B01;
    for len in (0..=size(64, 17)).chain([size(1000, 100), size(5000, 300)]) {
        let magnitude = random_magnitude(&mut state, len);
        // The same number also with zero bytes above its most significant byte.
        let mut padded = magnitude.clone();
        padded.resize(len + 5, 0);
        for (left, right) in ORDER_PAIRS {
            for negative in [false, true] {
                let x = number(&magnitude, negative, left);
                let same_x = number(&magnitude, negative, right);
                let padded = number(&padded, negative, right);
                assert_subtracts_in(forms_to_run(), &x, &same_x, &zero());
                assert_subtracts_in(forms_to_run(), &x, &padded, &zero());
                assert_subtracts_in(forms_to_run(), &padded, &x, &zero());
            }
        }
    }
}

#[test]
fn subtracts_a_number_from_itself() {
    // Both operands of &a - &a are the same number.
    let mut state = 0x2431_85BE_550C_7DC3;
    for _ in 0..size(200, 8) {
        let a = random_operand(&mut state, 300);
        #[allow(clippy::eq_op)]
        let result = &a - &a;
        let inputs = || format!("&a - &a with a = {}", describe(&a));
        assert_result(&result, &zero(), a.byte_order, &inputs);
    }
}

#[test]
fn nearly_equal_numbers_leave_the_shortest_form() {
    // x - (x + d) = -d and x - (x - d) = d: the long run of high bytes the operands share
    // cancels, and only the low bytes are left, at the front of a big-endian vector.
    let mut state = 0x72BE_5D74_F27B_896F;
    for round in 0..size(200, 8) {
        let len = if round % 20 == 0 {
            size(5000, 300)
        } else {
            9 + below(&mut state, size(200, 40))
        };
        let x = random_magnitude(&mut state, len);
        let d_len = 1 + below(&mut state, 8);
        let d = random_magnitude(&mut state, d_len);
        for d_negative in [false, true] {
            // y = x + d or x - d; x is longer than d, so y is positive.
            let le = ByteOrder::LittleEndian;
            let y = reference(&number(&x, false, le), &number(&d, d_negative, le)).magnitude;
            for (left, right) in ORDER_PAIRS {
                for negative in [false, true] {
                    // a = x and b = y, or a = -x and b = -y: the difference is x - y or y - x,
                    // so -d or d, negative when a is positive and y = x + d, or a is negative
                    // and y = x - d; and b - a is its negation.
                    let a = number(&x, negative, left);
                    let b = number(&y, negative, right);
                    let result = expected(&d, negative == d_negative);
                    assert_subtracts_in(forms_to_run(), &a, &b, &result);
                    let result = expected(&d, negative != d_negative);
                    assert_subtracts_in(forms_to_run(), &b, &a, &result);
                }
            }
        }
    }
}

// Zero

#[test]
fn zero_minus_zero_is_one_zero_byte() {
    // Empty vectors, one zero byte or several, and negative zeros give `[0]`, not negative.
    for (left, right) in ORDER_PAIRS {
        for a in zeros(left) {
            for b in zeros(right) {
                assert_subtracts(&a, &b, &zero());
            }
        }
    }
}

#[test]
fn subtracting_zero_gives_the_number_and_zero_minus_a_number_its_negation() {
    // x - 0 = x in the byte order of x, and 0 - x = -x in the byte order of the zero, also when x
    // has zero bytes above its most significant byte, for every way of writing zero.
    let mut state = 0x80DE_B1FE_9BDC_06A7;
    for len in (1..=size(20, 3)).chain([size(500, 40)]) {
        let magnitude = random_magnitude(&mut state, len);
        let mut padded = magnitude.clone();
        padded.resize(len + 3, 0);
        for (left, right) in ORDER_PAIRS {
            for negative in [false, true] {
                for x in [
                    number(&magnitude, negative, left),
                    number(&padded, negative, left),
                ] {
                    for zero in zeros(right) {
                        let x_back = expected(&magnitude, negative);
                        assert_subtracts_in(forms_to_run(), &x, &zero, &x_back);
                        let minus_x = expected(&magnitude, !negative);
                        assert_subtracts_in(forms_to_run(), &zero, &x, &minus_x);
                    }
                }
            }
        }
    }
}

// Signs

#[test]
fn subtracts_every_combination_of_signs() {
    // |a| > |b|, |a| < |b| and |a| = |b|, within a byte, across bytes, across 8-byte words, and
    // differing in the top byte only or in the bottom byte only.
    let magnitudes: [(i128, i128); 15] = [
        (5, 3),
        (3, 5),
        (5, 5),
        (0x100, 0xFF),
        (0xFF, 0x100),
        (0x1234, 0x1234),
        (1 << 64, (1 << 64) - 1),
        ((1 << 64) - 1, 1 << 64),
        (1 << 64, 1 << 64),
        (
            0x0123_4567_89AB_CDEF_0123_4567_89AB_CDEF,
            0x0123_4567_89AB_CDEF,
        ),
        (
            0x0123_4567_89AB_CDEF,
            0x0123_4567_89AB_CDEF_0123_4567_89AB_CDEF,
        ),
        (0x0123_4567_89AB_CDEF_0123, 0x0123_4567_89AB_CDEF_0123),
        (0x0200_0000_0000_0000_0001, 0x0100_0000_0000_0000_0001),
        (0x0100_0000_0000_0000_0002, 0x0100_0000_0000_0000_0001),
        (0x0100_0000_0000_0000_0001, 0x0100_0000_0000_0000_0002),
    ];
    for (x, y) in magnitudes {
        for (negative_a, negative_b) in SIGN_PAIRS {
            let x = if negative_a { -x } else { x };
            let y = if negative_b { -y } else { y };
            for (left, right) in ORDER_PAIRS {
                let a = Number::new_from_i128(x, left);
                let b = Number::new_from_i128(y, right);
                assert_subtracts(&a, &b, &expected_i128(x - y));
            }
        }
    }
    // The same on long numbers, against the reference.
    let mut state = 0xE49B_69C1_EFBE_4786;
    let long = random_magnitude(&mut state, size(300, 40));
    let short = random_magnitude(&mut state, size(290, 35));
    for (x, y) in [(&long, &short), (&short, &long), (&long, &long)] {
        for (negative_a, negative_b) in SIGN_PAIRS {
            for (left, right) in ORDER_PAIRS {
                let a = number(x, negative_a, left);
                let b = number(y, negative_b, right);
                assert_subtracts(&a, &b, &difference(&a, &b));
            }
        }
    }
}

// Byte orders

#[test]
fn mixed_byte_orders_in_every_form() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    // (a, b, the difference as stored in the byte order of a, and its sign)
    let cases = [
        // 0x01_02_03 - 0x20_10 = 0xE1_F3.
        (
            plus(&[0x01, 0x02, 0x03], be, 3),
            plus(&[0x10, 0x20], le, 2),
            vec![0xE1, 0xF3],
            false,
        ),
        // 0x20_10 - 0x01_02_03 = -0xE1_F3.
        (
            plus(&[0x10, 0x20], le, 2),
            plus(&[0x01, 0x02, 0x03], be, 3),
            vec![0xF3, 0xE1],
            true,
        ),
        // 0x01_02_03 - -0x20_10 = 0x01_22_13, and -0x20_10 - 0x01_02_03 its negation.
        (
            plus(&[0x01, 0x02, 0x03], be, 3),
            minus(&[0x10, 0x20], le, 2),
            vec![0x01, 0x22, 0x13],
            false,
        ),
        (
            minus(&[0x10, 0x20], le, 2),
            plus(&[0x01, 0x02, 0x03], be, 3),
            vec![0x13, 0x22, 0x01],
            true,
        ),
        // -0xFF_FF - 1 = -0x01_00_00, the carry out of the longer operand.
        (
            minus(&[0xFF, 0xFF], be, 2),
            plus(&[0x01], le, 1),
            vec![0x01, 0x00, 0x00],
            true,
        ),
        // 1 - 0x01_00_00 = -0xFF_FF, the borrow through the longer operand.
        (
            plus(&[0x01], le, 1),
            plus(&[0x00, 0x00, 0x01], le, 3),
            vec![0xFF, 0xFF],
            true,
        ),
        (
            plus(&[0x01], be, 1),
            plus(&[0x00, 0x00, 0x01], le, 3),
            vec![0xFF, 0xFF],
            true,
        ),
        // Nine bytes each, across an 8-byte word: 0x0A_0A_..._0A - 0x09_08_..._01 =
        // 0x01_02_..._09.
        (
            plus(&[0x0A; 9], be, 9),
            plus(&[1, 2, 3, 4, 5, 6, 7, 8, 9], le, 9),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
            false,
        ),
    ];
    for (a, b, result, negative) in cases {
        let magnitude = le_bytes(&Number {
            value: result,
            negative,
            byte_order: a.byte_order,
        });
        assert_subtracts(&a, &b, &expected(&magnitude, negative));
    }
}

#[test]
fn ref_minus_value_reverses_the_right_vector_into_the_left_byte_order() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    // 0x01_02_03 - 0x20_10 = 0xE1_F3, written as `a` writes it, in the vector of `b`.
    for (a, b, result) in [
        (
            plus(&[0x01, 0x02, 0x03], be, 3),
            plus(&[0x10, 0x20], le, 3),
            [0xE1, 0xF3],
        ),
        (
            plus(&[0x03, 0x02, 0x01], le, 3),
            plus(&[0x20, 0x10], be, 3),
            [0xF3, 0xE1],
        ),
    ] {
        let pointer = b.value.as_ptr();
        let message = format!("&a - b with a = {}, b = {}", describe(&a), describe(&b));
        let (difference, allocations) = count(|| &a - b);
        assert_shortest(&difference, &|| message.clone());
        assert_eq!(difference.value, result, "{message}");
        assert!(difference.byte_order == a.byte_order, "{message}");
        assert!(!difference.is_negative(), "{message}");
        assert_eq!(difference.value.as_ptr(), pointer, "{message}");
        assert_eq!(allocations, 0, "{message}");
    }
}

#[test]
fn sub_assign_keeps_the_byte_order_of_the_left_operand() {
    for (left, right) in ORDER_PAIRS {
        // 0x01_00 - 0x02_FF = -0x01_FF.
        let mut a = Number::new_from_u16(0x0100, left);
        a -= Number::new_from_u16(0x02FF, right);
        assert_shortest(&a, &|| "0x0100 - 0x02FF".to_string());
        assert!(a.byte_order == left);
        assert_eq!(a.value, in_order(&[0xFF, 0x01], left));
        assert!(a.is_negative());
        // 0x01_00 - -0x02_FF = 0x03_FF.
        let mut a = Number::new_from_u16(0x0100, left);
        a -= &Number::new_from_i16(-0x02FF, right);
        assert_shortest(&a, &|| "0x0100 - -0x02FF".to_string());
        assert!(a.byte_order == left);
        assert_eq!(a.value, in_order(&[0xFF, 0x03], left));
        assert!(!a.is_negative());
    }
}

// Properties

#[test]
fn subtraction_is_the_addition_of_the_negation() {
    // a - b and a + -b, in the same form and with the same capacities: the same bytes, sign and
    // byte order, built in the same vector, with as many allocations.
    let mut state = 0x9BDC_06A7_C19B_F174;
    for _ in 0..size(500, 16) {
        let a = random_operand(&mut state, 300);
        let b = random_operand(&mut state, 300);
        let form = FORMS[below(&mut state, 6)];
        let (spare_a, spare_b) = (below(&mut state, 3), below(&mut state, 3));
        let message = format!(
            "{} with a = {}, b = {}, {spare_a} and {spare_b} bytes to spare",
            form.name(),
            describe(&a),
            describe(&b)
        );
        let (left, right) = (copy(&a, spare_a), copy(&b, spare_b));
        let pointers = (left.value.as_ptr(), right.value.as_ptr());
        let (difference, allocations) = count(|| form.apply(left, right));
        let (left, right) = (copy(&a, spare_a), turned(&copy(&b, spare_b)));
        let sum_pointers = (left.value.as_ptr(), right.value.as_ptr());
        let (sum, sum_allocations) = count(|| form.add(left, right));
        assert!(
            same(&difference, &sum),
            "{message}: {}",
            describe(&difference)
        );
        let place = |n: &Number, (left, right): (*const u8, *const u8)| {
            let p = n.value.as_ptr();
            (p == left, p == right)
        };
        assert_eq!(
            place(&difference, pointers),
            place(&sum, sum_pointers),
            "{message}: not built in the same vector"
        );
        assert_eq!(allocations, sum_allocations, "{message}: allocations");
    }
}

#[test]
fn swapping_the_operands_negates_the_difference() {
    // b - a = -(a - b).
    let mut state = 0xE9B5_DBA5_3956_C25B;
    for _ in 0..size(400, 8) {
        let a = random_operand(&mut state, 300);
        let b = random_operand(&mut state, 300);
        let form = FORMS[below(&mut state, 6)];
        let ab = subtract(form, &a, &b);
        let ba = subtract(form, &b, &a);
        assert_eq!(
            value_of(&ab),
            value_of(&negated(&ba)),
            "{}: a = {}, b = {}",
            form.name(),
            describe(&a),
            describe(&b)
        );
    }
}

#[test]
fn subtracting_and_adding_a_number_gives_the_first_back() {
    // (a - b) + b = a, and (a + b) - b = a.
    let mut state = 0x243F_6A88_85A3_08D3;
    for _ in 0..size(400, 8) {
        let a = random_operand(&mut state, 300);
        let b = random_operand(&mut state, 300);
        let form = FORMS[below(&mut state, 6)];
        let message = format!(
            "{}: a = {}, b = {}",
            form.name(),
            describe(&a),
            describe(&b)
        );
        let back = form.add(subtract(form, &a, &b), copy(&b, 0));
        assert_eq!(value_of(&back), value_of(&a), "(a - b) + b, {message}");
        let back = subtract(form, &form.add(copy(&a, 0), copy(&b, 0)), &b);
        assert_eq!(value_of(&back), value_of(&a), "(a + b) - b, {message}");
    }
}

#[test]
fn borrowed_operands_are_left_as_they_were() {
    // The sign of a borrowed `b` is turned over for the addition without being written.
    let mut state = 0x1319_8A2E_0370_7344;
    for _ in 0..size(200, 8) {
        let a = random_operand(&mut state, 100);
        let b = random_operand(&mut state, 100);
        let (a_before, b_before) = (copy(&a, 0), copy(&b, 0));
        let message = format!("a = {}, b = {}", describe(&a), describe(&b));
        let _ = &a - &b;
        assert!(
            same(&a, &a_before) && same(&b, &b_before),
            "&a - &b: {message}"
        );
        let _ = copy(&a, 1) - &b;
        assert!(same(&b, &b_before), "a - &b: {message}");
        let _ = &a - copy(&b, 1);
        assert!(same(&a, &a_before), "&a - b: {message}");
        let mut difference = copy(&a, 1);
        difference -= &b;
        assert!(same(&b, &b_before), "a -= &b: {message}");
    }
}

// Allocation

/// Checks where `a - b` in `form` puts the difference, with the operands in vectors of the
/// capacities of `a` and `b`. The forms that take `a` by value build it in the vector of `a`
/// whenever that has room for it, and `&a - b` in that of `b` whenever that has room; reusing a
/// vector allocates nothing. When only `b` has room, `a - b` and `a -= b` build the difference
/// in it too. Otherwise the difference gets one new vector, as it always does from `&a - &b`.
fn assert_placement(form: Form, a: &Number, b: &Number) {
    let expected = difference(a, b);
    let len = expected.magnitude.len();
    let left = copy_with_capacity(a);
    let right = copy_with_capacity(b);
    let left_has_room = left.value.capacity() >= len;
    let right_has_room = right.value.capacity() >= len;
    let left_pointer = left.value.as_ptr();
    let right_pointer = right.value.as_ptr();
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
    // A vector with room for the difference has a capacity of at least one, so its pointer is
    // its own and not the dangling one every empty vector shares.
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

/// Where the allocation promises put the difference of `form`, given which operands have room
/// for it.
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

/// The pairs `(roomy, other)` of a table of sums, as differences: `roomy - turned(other)` is
/// the sum of the pair, so it has the room the table says.
fn differences(sums: Vec<(Number, Number)>) -> Vec<(Number, Number)> {
    sums.into_iter()
        .map(|(roomy, other)| {
            let other = turned(&other);
            (roomy, other)
        })
        .collect()
}

#[test]
fn sub_assign_grows_a_number_in_its_own_vector() {
    // 0xFF - -1 = 0x01_00: in big-endian order the old byte moves back to make room.
    for (order, result) in [
        (ByteOrder::BigEndian, [1, 0]),
        (ByteOrder::LittleEndian, [0, 1]),
    ] {
        for form in [Form::AssignRef, Form::AssignValue] {
            let a = plus(&[0xFF], order, 4);
            let pointer = a.value.as_ptr();
            let minus_one = Number::new_from_i8(-1, ByteOrder::LittleEndian);
            let difference = form.apply(a, minus_one);
            let inputs = || format!("{} with a = 0xFF, b = -1", form.name());
            assert_shortest(&difference, &inputs);
            assert_eq!(difference.value, result, "{}", inputs());
            assert_eq!(difference.value.as_ptr(), pointer, "{}", inputs());
        }
    }
}

#[test]
fn sub_assign_shrinks_a_number_in_its_own_vector() {
    let be = ByteOrder::BigEndian;
    // 0x01_00 - 1 = 0xFF, 0x09_08_07_06_05 - 0x09_08_07_06_00 = 5, and 5 - 7 = -2: in big-endian
    // order the difference moves to the front of the vector.
    for (a, b, result, negative) in [
        (
            plus(&[1, 0], be, 2),
            Number::new_from_u8(1, be),
            vec![0xFF],
            false,
        ),
        (
            plus(&[9, 8, 7, 6, 5], be, 5),
            Number::new(vec![9, 8, 7, 6, 0], be),
            vec![5],
            false,
        ),
        (plus(&[5], be, 1), Number::new_from_u8(7, be), vec![2], true),
    ] {
        for form in [Form::AssignRef, Form::AssignValue] {
            let a = copy_with_capacity(&a);
            let b = copy(&b, 0);
            let pointer = a.value.as_ptr();
            let message = format!(
                "{} with a = {}, b = {}",
                form.name(),
                describe(&a),
                describe(&b)
            );
            let difference = form.apply(a, b);
            assert_shortest(&difference, &|| message.clone());
            assert_eq!(difference.value, result, "{message}");
            assert_eq!(difference.is_negative(), negative, "{message}");
            assert_eq!(difference.value.as_ptr(), pointer, "{message}");
        }
    }
}

#[test]
fn value_forms_build_the_difference_in_the_left_vector_when_it_has_room() {
    for (roomy, other) in differences(sums_with_room())
        .into_iter()
        .chain(differences(sums_that_just_fit()))
    {
        for form in [
            Form::ValueValue,
            Form::ValueRef,
            Form::AssignValue,
            Form::AssignRef,
        ] {
            assert_placement(form, &roomy, &other);
        }
    }
}

#[test]
fn ref_minus_value_builds_the_difference_in_the_right_vector_when_it_has_room() {
    // other - turned(roomy) is other + roomy, built in the vector of roomy.
    for (roomy, other) in sums_with_room().into_iter().chain(sums_that_just_fit()) {
        assert_placement(Form::RefValue, &other, &turned(&roomy));
    }
}

#[test]
fn value_minus_value_falls_back_on_the_right_vector() {
    // The left operand with no room to spare, the right one with room for the difference: the
    // difference is in the vector of the left operand if it fits there, and otherwise in that
    // of the right one.
    for (roomy, other) in sums_with_room().into_iter().chain(sums_that_just_fit()) {
        let tight = copy(&other, 0);
        for form in [Form::ValueValue, Form::AssignValue] {
            assert_placement(form, &tight, &turned(&roomy));
        }
    }
}

#[test]
fn ref_minus_ref_allocates_once() {
    let pairs = sums_with_room()
        .into_iter()
        .chain(sums_that_just_fit())
        .chain(sums_without_room());
    for (a, b) in pairs {
        assert_placement(Form::RefRef, &a, &turned(&b));
        assert_placement(Form::RefRef, &b, &turned(&a));
    }
}

#[test]
fn every_form_allocates_one_vector_when_no_vector_has_room() {
    for (a, b) in sums_without_room() {
        let len = reference(&a, &b).magnitude.len();
        assert!(a.value.capacity() < len && b.value.capacity() < len);
        for form in FORMS {
            assert_placement(form, &a, &turned(&b));
            assert_placement(form, &b, &turned(&a));
        }
    }
}

#[test]
fn placement_follows_the_capacities_of_random_operands() {
    // Capacities around the length of the difference, where a vector just has room for it or
    // just lacks it.
    let mut state = 0xA409_3822_299F_31D0;
    for _ in 0..size(1000, 30) {
        let a = random_operand(&mut state, 40);
        let b = random_operand(&mut state, 40);
        let len = difference(&a, &b).magnitude.len();
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

// Long operands, through every path of the kernels

/// Checks `a - b`, of operands of `limbs` limbs, against `expected` in every form, as
/// [`assert_subtracts_in`] does, or from 1000 limbs on in the form and capacity setup at `turn`,
/// as [`assert_subtracts_in_turn`] does: a debug build takes a while over each of them.
fn assert_long_subtracts(
    turn: &mut usize,
    limbs: usize,
    a: &Number,
    b: &Number,
    expected: &Expected,
) {
    if limbs < 1000 {
        assert_subtracts_in(forms_to_run(), a, b, expected);
    } else {
        assert_subtracts_in_turn(turn, a, b, expected);
    }
}

#[test]
fn borrows_ripple_through_every_limb() {
    // x = 2 * 256^(len - 1) and y = 1 + 256^(len - 1), so x - y = 256^(len - 1) - 1, which is
    // [0xFF; len - 1]: the borrow from the bottom byte goes through every limb, and cancels
    // the top byte. With several chains, every part above the first is subtracted with no
    // borrow into it and comes out zero, so on its fix-up the borrow out of the part below
    // ripples through every limb of it.
    let mut turn = 0;
    for limbs in limb_counts() {
        for extra in EXTRA_BYTES {
            let len = 8 * limbs + extra;
            let mut x = vec![0; len];
            x[len - 1] = 2;
            let mut y = vec![0; len];
            y[0] = 1;
            y[len - 1] = 1;
            let result = vec![0xFF; len - 1];
            for (left, right) in ORDER_PAIRS {
                for negative in [false, true] {
                    // x - y and y - x, and both negated: the difference is built in the vector
                    // of either operand, which shrinks, the larger or the smaller magnitude.
                    let a = number(&x, negative, left);
                    let b = number(&y, negative, right);
                    assert_long_subtracts(&mut turn, limbs, &a, &b, &expected(&result, negative));
                    let a = number(&y, negative, left);
                    let b = number(&x, negative, right);
                    assert_long_subtracts(&mut turn, limbs, &a, &b, &expected(&result, !negative));
                }
            }
        }
    }
}

#[test]
fn carries_ripple_through_every_limb() {
    // x = [0xFF; len] and y = 1 + 256^(len - 1), so x - -y = 256^len + 256^(len - 1): the carry
    // from the bottom byte goes through every limb, and on the fix-up of every part above the
    // first, through every limb of it.
    let mut turn = 0;
    for limbs in limb_counts() {
        for extra in EXTRA_BYTES {
            let len = 8 * limbs + extra;
            let x = vec![0xFF; len];
            let mut y = vec![0; len];
            y[0] = 1;
            y[len - 1] = 1;
            let mut result = vec![0; len + 1];
            result[len - 1] = 1;
            result[len] = 1;
            for (left, right) in ORDER_PAIRS {
                for negative in [false, true] {
                    let result = expected(&result, negative);
                    for (a, b) in [(&x, &y), (&y, &x)] {
                        let a = number(a, negative, left);
                        let b = number(b, !negative, right);
                        assert_long_subtracts(&mut turn, limbs, &a, &b, &result);
                    }
                }
            }
        }
    }
}

#[test]
fn subtracts_long_runs_of_whole_limbs_as_the_reference() {
    // Operands of the limb counts above, both long, in runs of whole limbs: the second as long
    // as the first, a few limbs apart from it, of another of the counts, or the first with new
    // low limbs, so that the difference nearly cancels.
    let mut state = 0x4528_21E6_38D0_1377;
    let counts = limb_counts();
    for &limbs in &counts {
        for _ in 0..if limbs < 1000 { 8 } else { 2 } {
            let extra = below(&mut state, 8);
            let x = limb_runs(&mut state, limbs, extra);
            let extra = below(&mut state, 8);
            let y = match below(&mut state, 4) {
                0 => limb_runs(&mut state, limbs, extra),
                1 => {
                    let near = (limbs + below(&mut state, 5)).saturating_sub(2).max(1);
                    limb_runs(&mut state, near, extra)
                }
                2 => {
                    let other = counts[below(&mut state, counts.len())];
                    limb_runs(&mut state, other, extra)
                }
                _ => {
                    let mut y = x.clone();
                    let low = below(&mut state, limbs + 1);
                    y[..8 * low].copy_from_slice(&limb_runs(&mut state, low, 0));
                    y
                }
            };
            let a = number(&x, below(&mut state, 2) == 0, ORDERS[below(&mut state, 2)]);
            let b = number(&y, below(&mut state, 2) == 0, ORDERS[below(&mut state, 2)]);
            assert_subtracts_in(forms_to_run(), &a, &b, &difference(&a, &b));
        }
    }
}
