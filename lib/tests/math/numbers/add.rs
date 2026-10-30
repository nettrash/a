use a::math::numbers::{ByteOrder, Number};

use super::common::*;
use crate::allocations::count;

// The forms of addition

/// The six ways to write an addition. Every form must give the same sum, so the tests run
/// them all.
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
            Form::ValueValue => "a + b",
            Form::ValueRef => "a + &b",
            Form::RefValue => "&a + b",
            Form::RefRef => "&a + &b",
            Form::AssignValue => "a += b",
            Form::AssignRef => "a += &b",
        }
    }

    /// Adds `b` to `a` in this form; an operand the form only borrows is dropped afterwards.
    fn apply(self, a: Number, b: Number) -> Number {
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

/// Every form, or under Miri one picked at random: for the tests that check many sums.
fn forms_to_run() -> &'static [Form] {
    if cfg!(miri) {
        let k = pick(FORMS.len());
        &FORMS[k..=k]
    } else {
        &FORMS
    }
}

/// Checks `a + b` against `expected` in every form, as [`assert_adds_in`] does.
fn assert_adds(a: &Number, b: &Number, expected: &Expected) {
    assert_adds_in(&FORMS, a, b, expected);
}

/// Checks `a + b` against `expected` in each of `forms`: with the vectors of the operands the
/// form takes by value as tight as their values, with one byte to spare, and with room for any
/// sum, and for `a + b` and `a += b` also with room in `b` only, so that each form meets each
/// way it can place the sum. Under Miri each form runs with one of these setups, picked at
/// random.
fn assert_adds_in(forms: &[Form], a: &Number, b: &Number, expected: &Expected) {
    for &form in forms {
        let spares = spares(form, a, b);
        let spares = if cfg!(miri) {
            let k = pick(spares.len());
            &spares[k..=k]
        } else {
            &spares[..]
        };
        for &(spare_a, spare_b) in spares {
            assert_adds_with(form, spare_a, spare_b, a, b, expected);
        }
    }
}

/// The capacity setups [`assert_adds_in`] runs `form` with, as the bytes to spare in the
/// vectors of `a` and `b`.
fn spares(form: Form, a: &Number, b: &Number) -> Vec<(usize, usize)> {
    // A sum is at most one byte longer than the longer operand.
    let room = a.value.len().max(b.value.len()) + 1;
    match form {
        Form::RefRef => vec![(0, 0)],
        Form::ValueValue | Form::AssignValue => vec![(0, 0), (1, 1), (room, room), (0, room)],
        Form::ValueRef | Form::RefValue | Form::AssignRef => vec![(0, 0), (1, 1), (room, room)],
    }
}

/// Checks `a + b` against `expected` in `form`, with `spare_a` and `spare_b` bytes to spare in
/// the vectors of the operands the form takes by value.
fn assert_adds_with(
    form: Form,
    spare_a: usize,
    spare_b: usize,
    a: &Number,
    b: &Number,
    expected: &Expected,
) {
    let sum = form.apply(copy(a, spare_a), copy(b, spare_b));
    let inputs = || {
        format!(
            "{} with a = {}, b = {}, {spare_a} and {spare_b} bytes to spare",
            form.name(),
            describe(a),
            describe(b)
        )
    };
    assert_result(&sum, expected, a.byte_order, &inputs);
}

/// Checks `a + b` against `expected` in one of the forms and capacity setups [`assert_adds_in`]
/// runs: the one at `turn` among all of them, after which `turn` counts on. For the longest
/// operands, which take a while in every one of them in a debug build: taken in turn, every
/// setup still meets operands of every size.
fn assert_adds_in_turn(turn: &mut usize, a: &Number, b: &Number, expected: &Expected) {
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
    assert_adds_with(form, spare_a, spare_b, a, b, expected);
}

/// `a + b` in `form`, checked to be in the shortest form and in the byte order of `a`.
fn add(form: Form, a: &Number, b: &Number) -> Number {
    let sum = form.apply(copy(a, 0), copy(b, 0));
    let inputs = || {
        format!(
            "{} with a = {}, b = {}",
            form.name(),
            describe(a),
            describe(b)
        )
    };
    assert_shortest(&sum, &inputs);
    assert!(
        sum.byte_order == a.byte_order,
        "{}: not in the byte order of a",
        inputs()
    );
    sum
}

// Small values against `i128` arithmetic

#[test]
fn adds_small_values_as_i128() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..size(1000, 40) {
        let x = random_i128(&mut state);
        let y = random_i128(&mut state);
        let sum = expected_i128(x + y);
        for (left, right) in ORDER_PAIRS {
            let a = Number::new_from_i128(x, left);
            let b = Number::new_from_i128(y, right);
            // The reference of the tests on big numbers agrees with `i128` arithmetic.
            assert_eq!(reference(&a, &b), sum, "reference for {x} + {y}");
            assert_adds_in(forms_to_run(), &a, &b, &sum);
        }
    }
}

#[test]
fn adds_every_pair_of_small_integers() {
    // From -300 to 300: every carry into the second byte and every borrow from it, in every
    // combination of signs. Each pair runs in one form, pair of byte orders and spare
    // capacity, taken in turn, so that every one of them meets values of every kind.
    let mut turn = 0;
    for x in (-300..=300).step_by(size(1, 23)) {
        for y in (-300..=300).step_by(size(1, 23)) {
            let (left, right) = ORDER_PAIRS[turn % 4];
            let form = FORMS[turn / 4 % 6];
            let spare = turn / 24 % 3;
            turn += 1;
            let a = Number::new_from_i32(x, left);
            let b = Number::new_from_i32(y, right);
            let sum = form.apply(copy(&a, spare), copy(&b, spare));
            let inputs = || format!("{x} + {y} as {} with {spare} bytes to spare", form.name());
            assert_result(&sum, &expected_i128((x + y).into()), left, &inputs);
        }
    }
}

// Big values against the reference

#[test]
fn adds_random_big_numbers_as_the_reference() {
    let mut state = 0x3C6E_F372_FE94_F82B;
    for _ in 0..size(500, 16) {
        let a = random_operand(&mut state, 600);
        let b = random_operand(&mut state, 600);
        assert_adds_in(forms_to_run(), &a, &b, &reference(&a, &b));
    }
}

#[test]
fn adds_every_pair_of_lengths_up_to_40_bytes() {
    // Up to five 8-byte words, with every remainder after whole words and after groups of
    // them. Every pair of lengths runs in every combination of byte orders and signs, or under
    // Miri in one picked at random.
    let mut state = 0xA54F_F53A_5F1D_36F1;
    for len_a in 0..=40 {
        for len_b in 0..=40 {
            for k in 0..size(16, 1) {
                let k = if cfg!(miri) { below(&mut state, 16) } else { k };
                let (left, right) = ORDER_PAIRS[k % 4];
                let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
                let a = number(&random_magnitude(&mut state, len_a), negative_a, left);
                let b = number(&random_magnitude(&mut state, len_b), negative_b, right);
                assert_adds_in(forms_to_run(), &a, &b, &reference(&a, &b));
            }
        }
    }
}

#[test]
fn adds_numbers_of_millions_of_bytes() {
    let mut state = 0x510E_527F_ADE6_82D1;
    let len = size(2_000_000, 1_000);
    let x = random_magnitude(&mut state, len);
    let y = random_magnitude(&mut state, len - 5);
    for ((left, right), (negative_a, negative_b)) in ORDER_PAIRS.into_iter().zip(SIGN_PAIRS) {
        let a = number(&x, negative_a, left);
        let b = number(&y, negative_b, right);
        let expected = reference(&a, &b);
        // A few forms only, as each sum takes a while in a debug build: in a new vector, in
        // place with room in the vector of `a`, and in that of `b`.
        for (form, spare_a, spare_b) in [
            (Form::RefRef, 0, 0),
            (Form::AssignRef, 1, 0),
            (Form::RefValue, 0, 6),
        ] {
            let sum = form.apply(copy(&a, spare_a), copy(&b, spare_b));
            let inputs = || {
                format!(
                    "{} with a = {}, b = {}",
                    form.name(),
                    describe(&a),
                    describe(&b)
                )
            };
            assert_result(&sum, &expected, left, &inputs);
        }
    }
}

// The paths in registers: up to 16 bytes, and up to 64

/// Vector lengths at the limits of the paths in registers: around every whole number of limbs
/// up to nine, and around 16 and 64 bytes, where the short and the medium path end.
const REGISTER_LENGTHS: [usize; 29] = [
    0, 1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 40, 41, 47, 48, 49, 55, 56, 57, 63,
    64, 65, 72,
];

#[test]
fn adds_every_pair_of_lengths_at_the_limits_of_the_paths_in_registers() {
    // Every pair runs in every combination of byte orders and signs, or under Miri in one picked
    // at random, and over the limbs of each length a random sum carries out of the top and
    // a random difference borrows, a coin toss each.
    let mut state = 0x1F83_D9AB_FB41_BD6B;
    for len_a in REGISTER_LENGTHS {
        for len_b in REGISTER_LENGTHS {
            if cfg!(miri) && below(&mut state, 4) != 0 {
                continue;
            }
            for k in 0..size(16, 1) {
                let k = if cfg!(miri) { below(&mut state, 16) } else { k };
                let (left, right) = ORDER_PAIRS[k % 4];
                let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
                let a = number(&random_magnitude(&mut state, len_a), negative_a, left);
                let b = number(&random_magnitude(&mut state, len_b), negative_b, right);
                assert_adds_in(forms_to_run(), &a, &b, &reference(&a, &b));
            }
        }
    }
}

#[test]
fn small_sums_one_byte_longer_or_much_shorter_than_the_operands() {
    // Up to 16 bytes the stores are chosen by the length of the longer operand's vector, `k`:
    // a sum of `k` or `k + 1` bytes takes the same stores, and a shorter one the stores of its
    // length. Here sums with and without a carry out of the top byte, and differences that leave
    // one byte to `k - 1` of them, for every `k`.
    let mut state = 0x6A09_E667_F3BC_C909;
    for k in 1..=16 {
        let carries = {
            let mut x = random_magnitude(&mut state, k);
            x[k - 1] |= 0x80;
            x
        };
        let stays = {
            let mut x = random_magnitude(&mut state, k);
            x[k - 1] = 1;
            x
        };
        for short in (1..k).chain([0]) {
            for (left, right) in ORDER_PAIRS {
                for negative in [false, true] {
                    for x in [&carries, &stays] {
                        // x + x: a carry out of the top byte, or none.
                        let a = number(x, negative, left);
                        let b = number(x, negative, right);
                        assert_adds_in(forms_to_run(), &a, &b, &reference(&a, &b));
                        // x - y with `y` equal to `x` above its lowest `short` bytes: at most
                        // `short` bytes are left, or none.
                        let mut y = x.clone();
                        y[..short].copy_from_slice(&random_bytes(&mut state, short));
                        let b = number(&y, !negative, right);
                        assert_adds_in(forms_to_run(), &a, &b, &reference(&a, &b));
                    }
                }
            }
        }
    }
}

#[test]
fn medium_sums_much_shorter_than_their_vectors() {
    // Up to 64 bytes, a sum of operands of up to `L` limbs that is shorter than `L - 1` limbs,
    // from zero bytes above the most significant ones or from operands that nearly cancel,
    // takes the slower path that writes it from a buffer: with the vector it is built in chosen
    // as the operators promise, which every capacity setup checks.
    let mut state = 0xBB67_AE85_84CA_A73B;
    let vectors = [17, 24, 25, 33, 40, 48, 56, 63, 64];
    let significant = [0, 1, 7, 8, 9, 15, 16, 17, 24, 31, 40, 55];
    for len_a in vectors {
        for len_b in [0, 3, 16, 17, 24, 40, 64] {
            for sig_a in significant.into_iter().filter(|&s| s <= len_a) {
                let sig_b = significant[below(&mut state, significant.len())].min(len_b);
                let mut x = random_magnitude(&mut state, sig_a);
                x.resize(len_a, 0);
                let mut y = random_magnitude(&mut state, sig_b);
                y.resize(len_b, 0);
                let k = below(&mut state, 16);
                let (left, right) = ORDER_PAIRS[k % 4];
                let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
                let a = number(&x, negative_a, left);
                let b = number(&y, negative_b, right);
                assert_adds_in(forms_to_run(), &a, &b, &reference(&a, &b));
                assert_adds_in(forms_to_run(), &b, &a, &reference(&b, &a));
            }
        }
    }
    // x + -(x + d) for small d: a few bytes are left of up to 64.
    for len in [17, 24, 25, 32, 40, 47, 56, 64] {
        for d in [1u8, 2, 0xFF] {
            let x = random_magnitude(&mut state, len);
            let a = number(&x, false, ByteOrder::LittleEndian);
            let mut y = x.clone();
            let carry = y.iter_mut().try_for_each(|byte| {
                let (sum, over) = byte.overflowing_add(d);
                *byte = sum;
                if over { Some(()) } else { None }
            });
            if carry.is_some() {
                y.push(1);
            }
            for (left, right) in ORDER_PAIRS {
                for negative in [false, true] {
                    let a = number(&le_bytes(&a), negative, left);
                    let b = number(&y, !negative, right);
                    assert_adds_in(forms_to_run(), &a, &b, &reference(&a, &b));
                    assert_adds_in(forms_to_run(), &b, &a, &reference(&b, &a));
                }
            }
        }
    }
}

#[test]
fn placement_follows_the_capacities_of_operands_in_registers() {
    // As `placement_follows_the_capacities_of_random_operands`, for operands of up to 72 bytes,
    // one time in four with zero bytes above their most significant ones.
    let mut state = 0x3C6E_F372_FE94_F82B;
    for _ in 0..size(3000, 40) {
        let operand = |state: &mut u64| {
            let len = REGISTER_LENGTHS[below(state, REGISTER_LENGTHS.len())];
            let mut magnitude = random_magnitude(state, len);
            if below(state, 4) == 0 {
                let high = below(state, 3 * len + 2);
                magnitude.resize(len + high, 0);
            }
            number(&magnitude, below(state, 2) == 0, ORDERS[below(state, 2)])
        };
        let (a, b) = (operand(&mut state), operand(&mut state));
        let len = reference(&a, &b).magnitude.len();
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
fn ref_plus_ref_builds_a_vector_of_exactly_the_sum() {
    // The one new vector of `&a + &b` has the length of the sum as its capacity, on every path.
    let mut state = 0xA54F_F53A_5F1D_36F1;
    for len in REGISTER_LENGTHS.into_iter().chain([100, 300]) {
        for k in 0..size(16, 2) {
            let (left, right) = ORDER_PAIRS[k % 4];
            let (negative_a, negative_b) = SIGN_PAIRS[(k / 4) % 4];
            let a = number(&random_magnitude(&mut state, len), negative_a, left);
            let b = number(&random_magnitude(&mut state, len), negative_b, right);
            let sum = &a + &b;
            assert_eq!(
                sum.value.capacity(),
                sum.value.len(),
                "&a + &b with a = {}, b = {}",
                describe(&a),
                describe(&b)
            );
        }
    }
}

// Carry and borrow ripples

#[test]
fn carries_ripple_through_every_byte() {
    // [0xFF; n] + 1 = 256^n, one byte longer than the longer operand, and the same negated.
    for n in (1..=size(100, 40)).chain([size(10_000, 300), size(100_000, 1_000)]) {
        let all_ones = vec![0xFF; n];
        let mut power = vec![0; n];
        power.push(1);
        for (left, right) in ORDER_PAIRS {
            for negative in [false, true] {
                let ones = number(&all_ones, negative, left);
                let one = number(&[1], negative, right);
                let sum = expected(&power, negative);
                assert_adds_in(forms_to_run(), &ones, &one, &sum);
                assert_adds_in(forms_to_run(), &one, &ones, &sum);
            }
        }
    }
}

#[test]
fn borrows_ripple_through_every_byte() {
    // 256^n + -1 = [0xFF; n], one byte shorter than the longer operand, and the same negated.
    for n in (1..=size(100, 40)).chain([size(10_000, 300), size(100_000, 1_000)]) {
        let all_ones = vec![0xFF; n];
        let mut power = vec![0; n];
        power.push(1);
        for (left, right) in ORDER_PAIRS {
            for negative in [false, true] {
                let power = number(&power, negative, left);
                let one = number(&[1], !negative, right);
                let sum = expected(&all_ones, negative);
                assert_adds_in(forms_to_run(), &power, &one, &sum);
                assert_adds_in(forms_to_run(), &one, &power, &sum);
            }
        }
    }
}

#[test]
fn carries_and_borrows_go_on_or_stop_above_the_shorter_operand() {
    // The carry (borrow) out of the shorter operand goes into the bytes of the longer one
    // above it without a branch: into a whole limb, or into fewer than eight bytes as one
    // partial limb. Here it goes on through all ones (zeros) to the top, or stops in the first
    // byte above, with up to two limbs of bytes above the shorter operand, and the shorter
    // operand below, at and above a whole limb.
    let mut state = 0x9B05_688C_2B3E_6C1F;
    for ly in [1, 7, 8, 9, 15, 16, 17, 24] {
        for above in 1..=17 {
            let lx = ly + above;
            let y = random_magnitude(&mut state, ly);
            // All ones: a carry out of `y` goes on to the top, and the sum is a byte longer.
            let ones = vec![0xFF; lx];
            // 256^(lx - 1): a borrow out of `y` goes on to the top, and takes its 1.
            let mut power = vec![0; lx - 1];
            power.push(1);
            // All ones but the byte above `y`, where a carry stops.
            let mut stop = ones.clone();
            stop[ly] = 0x7F;
            for x in [&ones, &power, &stop] {
                for k in 0..size(16, 1) {
                    let k = if cfg!(miri) { below(&mut state, 16) } else { k };
                    let (left, right) = ORDER_PAIRS[k % 4];
                    let (negative_a, negative_b) = SIGN_PAIRS[k / 4];
                    let a = number(x, negative_a, left);
                    let b = number(&y, negative_b, right);
                    assert_adds_in(forms_to_run(), &a, &b, &reference(&a, &b));
                    assert_adds_in(forms_to_run(), &b, &a, &reference(&b, &a));
                }
            }
        }
    }
}

// Cancellation

#[test]
fn number_plus_its_negation_is_zero() {
    let mut state = 0x1F83_D9AB_FB41_BD6B;
    for len in (0..=size(64, 17)).chain([size(1000, 100), size(5000, 300)]) {
        let magnitude = random_magnitude(&mut state, len);
        // The negation also with zero bytes above its most significant byte.
        let mut padded = magnitude.clone();
        padded.resize(len + 5, 0);
        for (left, right) in ORDER_PAIRS {
            for negative in [false, true] {
                let x = number(&magnitude, negative, left);
                let minus_x = number(&magnitude, !negative, right);
                let padded = number(&padded, !negative, right);
                assert_adds_in(forms_to_run(), &x, &minus_x, &zero());
                assert_adds_in(forms_to_run(), &x, &padded, &zero());
                assert_adds_in(forms_to_run(), &padded, &x, &zero());
            }
        }
    }
}

#[test]
fn nearly_cancelling_sums_shrink_to_the_shortest_form() {
    // x + -(x + d) = -d and x + -(x - d) = d: the long run of high bytes the operands share
    // cancels, and only the low bytes are left, at the front of a big-endian vector.
    let mut state = 0x5BE0_CD19_137E_2179;
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
                    // a = x and b = -y, or a = -x and b = y: the sum is x - y or y - x, so d
                    // or -d, negative when a is positive and y = x + d, or a is negative and
                    // y = x - d.
                    let a = number(&x, negative, left);
                    let b = number(&y, !negative, right);
                    let sum = expected(&d, negative == d_negative);
                    assert_adds_in(forms_to_run(), &a, &b, &sum);
                    assert_adds_in(forms_to_run(), &b, &a, &sum);
                }
            }
        }
    }
}

// Zero

#[test]
fn zero_plus_zero_is_one_zero_byte() {
    // Empty vectors, one zero byte or several, and negative zeros give `[0]`, not negative.
    for (left, right) in ORDER_PAIRS {
        for a in zeros(left) {
            for b in zeros(right) {
                assert_adds(&a, &b, &zero());
            }
        }
    }
}

#[test]
fn adding_zero_gives_the_number_in_the_left_byte_order() {
    // x + 0 = x in the byte order of x, and 0 + x = x in the byte order of the zero, also when x
    // has zero bytes above its most significant byte, for every way of writing zero.
    let mut state = 0xCBBB_9D5D_C105_9ED8;
    for len in (1..=size(20, 3)).chain([size(500, 40)]) {
        let magnitude = random_magnitude(&mut state, len);
        let mut padded = magnitude.clone();
        padded.resize(len + 3, 0);
        for (left, right) in ORDER_PAIRS {
            for negative in [false, true] {
                let sum = expected(&magnitude, negative);
                for x in [
                    number(&magnitude, negative, left),
                    number(&padded, negative, left),
                ] {
                    for zero in zeros(right) {
                        assert_adds_in(forms_to_run(), &x, &zero, &sum);
                        assert_adds_in(forms_to_run(), &zero, &x, &sum);
                    }
                }
            }
        }
    }
}

// Signs

#[test]
fn adds_every_combination_of_signs() {
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
                assert_adds(&a, &b, &expected_i128(x + y));
            }
        }
    }
    // The same on long numbers, against the reference.
    let mut state = 0x6A09_E667_F3BC_C908;
    let long = random_magnitude(&mut state, size(300, 40));
    let short = random_magnitude(&mut state, size(290, 35));
    for (x, y) in [(&long, &short), (&short, &long), (&long, &long)] {
        for (negative_a, negative_b) in SIGN_PAIRS {
            for (left, right) in ORDER_PAIRS {
                let a = number(x, negative_a, left);
                let b = number(y, negative_b, right);
                assert_adds(&a, &b, &reference(&a, &b));
            }
        }
    }
}

// Byte orders

#[test]
fn mixed_byte_orders_in_every_form() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    // (a, b, the sum as stored in the byte order of a, and its sign)
    let cases = [
        // 0x01_02_03 + 0x20_10 = 0x01_22_13.
        (
            plus(&[0x01, 0x02, 0x03], be, 3),
            plus(&[0x10, 0x20], le, 2),
            vec![0x01, 0x22, 0x13],
            false,
        ),
        (
            plus(&[0x10, 0x20], le, 2),
            plus(&[0x01, 0x02, 0x03], be, 3),
            vec![0x13, 0x22, 0x01],
            false,
        ),
        // 0x01_02_03 - 0x20_10 = 0xE1_F3, and -0x20_10 + 0x01_02_03 the same.
        (
            plus(&[0x01, 0x02, 0x03], be, 3),
            minus(&[0x10, 0x20], le, 2),
            vec![0xE1, 0xF3],
            false,
        ),
        (
            minus(&[0x10, 0x20], le, 2),
            plus(&[0x01, 0x02, 0x03], be, 3),
            vec![0xF3, 0xE1],
            false,
        ),
        // 0x20_10 - 0x01_02_03 = -0xE1_F3.
        (
            plus(&[0x10, 0x20], le, 2),
            minus(&[0x01, 0x02, 0x03], be, 3),
            vec![0xF3, 0xE1],
            true,
        ),
        // -0xFF_FF + -0x01 = -0x01_00_00, the carry out of the longer operand.
        (
            minus(&[0xFF, 0xFF], be, 2),
            minus(&[0x01], le, 1),
            vec![0x01, 0x00, 0x00],
            true,
        ),
        (
            minus(&[0x01], le, 1),
            minus(&[0xFF, 0xFF], be, 2),
            vec![0x00, 0x00, 0x01],
            true,
        ),
        // Nine bytes each, across an 8-byte word: 0x01_02_..._09 + 0x09_08_..._01 = 0x0A_0A_..._0A.
        (
            plus(&[1, 2, 3, 4, 5, 6, 7, 8, 9], be, 9),
            plus(&[1, 2, 3, 4, 5, 6, 7, 8, 9], le, 9),
            vec![0x0A; 9],
            false,
        ),
    ];
    for (a, b, sum, negative) in cases {
        let magnitude = le_bytes(&Number {
            value: sum,
            negative,
            byte_order: a.byte_order,
        });
        assert_adds(&a, &b, &expected(&magnitude, negative));
    }
}

#[test]
fn ref_plus_value_reverses_the_right_vector_into_the_left_byte_order() {
    let be = ByteOrder::BigEndian;
    let le = ByteOrder::LittleEndian;
    // 0x01_02_03 + 0x20_10 = 0x01_22_13, written as `a` writes it, in the vector of `b`.
    for (a, b, sum) in [
        (
            plus(&[0x01, 0x02, 0x03], be, 3),
            plus(&[0x10, 0x20], le, 3),
            [0x01, 0x22, 0x13],
        ),
        (
            plus(&[0x03, 0x02, 0x01], le, 3),
            plus(&[0x20, 0x10], be, 3),
            [0x13, 0x22, 0x01],
        ),
    ] {
        let pointer = b.value.as_ptr();
        let message = format!("&a + b with a = {}, b = {}", describe(&a), describe(&b));
        let (result, allocations) = count(|| &a + b);
        assert_shortest(&result, &|| message.clone());
        assert_eq!(result.value, sum, "{message}");
        assert!(result.byte_order == a.byte_order, "{message}");
        assert!(!result.is_negative(), "{message}");
        assert_eq!(result.value.as_ptr(), pointer, "{message}");
        assert_eq!(allocations, 0, "{message}");
    }
}

#[test]
fn add_assign_keeps_the_byte_order_of_the_left_operand() {
    for (left, right) in ORDER_PAIRS {
        // 0x01_00 + 0x02_FF = 0x03_FF.
        let mut a = Number::new_from_u16(0x0100, left);
        a += Number::new_from_u16(0x02FF, right);
        assert_shortest(&a, &|| "0x0100 + 0x02FF".to_string());
        assert!(a.byte_order == left);
        assert_eq!(a.value, in_order(&[0xFF, 0x03], left));
        let mut a = Number::new_from_u16(0x0100, left);
        a += &Number::new_from_i16(-0x02FF, right);
        assert_shortest(&a, &|| "0x0100 - 0x02FF".to_string());
        assert!(a.byte_order == left);
        assert_eq!(a.value, in_order(&[0xFF, 0x01], left));
        assert!(a.is_negative());
    }
}

// Properties

#[test]
fn addition_is_commutative() {
    let mut state = 0xBB67_AE85_84CA_A73B;
    for _ in 0..size(400, 8) {
        let a = random_operand(&mut state, 300);
        let b = random_operand(&mut state, 300);
        let form = FORMS[below(&mut state, 6)];
        let ab = add(form, &a, &b);
        let ba = add(form, &b, &a);
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
fn addition_is_associative() {
    let mut state = 0x3C6E_F372_FE94_F82B;
    for _ in 0..size(400, 8) {
        let a = random_operand(&mut state, 300);
        let b = random_operand(&mut state, 300);
        let c = random_operand(&mut state, 300);
        let form = FORMS[below(&mut state, 6)];
        let ab_c = add(form, &add(form, &a, &b), &c);
        let a_bc = add(form, &a, &add(form, &b, &c));
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
fn adding_and_subtracting_a_number_gives_the_first_back() {
    // (a + b) + -b = a.
    let mut state = 0xA54F_F53A_5F1D_36F1;
    for _ in 0..size(400, 8) {
        let a = random_operand(&mut state, 300);
        let b = random_operand(&mut state, 300);
        let form = FORMS[below(&mut state, 6)];
        let back = add(form, &add(form, &a, &b), &negated(&b));
        assert_eq!(
            value_of(&back),
            value_of(&a),
            "{}: a = {}, b = {}",
            form.name(),
            describe(&a),
            describe(&b)
        );
    }
}

#[test]
fn adds_a_number_to_itself() {
    // Both operands of &a + &a are the same number.
    let mut state = 0x9B05_688C_2B3E_6C1F;
    for _ in 0..size(200, 8) {
        let a = random_operand(&mut state, 300);
        let sum = &a + &a;
        let inputs = || format!("&a + &a with a = {}", describe(&a));
        assert_result(&sum, &reference(&a, &a), a.byte_order, &inputs);
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
        let _ = &a + &b;
        assert!(
            same(&a, &a_before) && same(&b, &b_before),
            "&a + &b: {message}"
        );
        let _ = copy(&a, 1) + &b;
        assert!(same(&b, &b_before), "a + &b: {message}");
        let _ = &a + copy(&b, 1);
        assert!(same(&a, &a_before), "&a + b: {message}");
        let mut sum = copy(&a, 1);
        sum += &b;
        assert!(same(&b, &b_before), "a += &b: {message}");
    }
}

// Allocation

/// Checks where `a + b` in `form` puts the sum, with the operands in vectors of the capacities
/// of `a` and `b`. The forms that take `a` by value build it in the vector of `a` whenever that
/// has room for it, and `&a + b` in that of `b` whenever that has room; reusing a vector
/// allocates nothing. When only `b` has room, `a + b` and `a += b` build the sum in it too.
/// Otherwise the sum gets one new vector, as it always does from `&a + &b`.
fn assert_placement(form: Form, a: &Number, b: &Number) {
    let expected = reference(a, b);
    let len = expected.magnitude.len();
    let left = copy_with_capacity(a);
    let right = copy_with_capacity(b);
    let left_has_room = left.value.capacity() >= len;
    let right_has_room = right.value.capacity() >= len;
    let left_pointer = left.value.as_ptr();
    let right_pointer = right.value.as_ptr();
    let (sum, allocations) = count(|| form.apply(left, right));
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
    assert_result(&sum, &expected, a.byte_order, &inputs);
    // A vector with room for the sum has a capacity of at least one, so its pointer is its own
    // and not the dangling one every empty vector shares.
    let in_left = sum.value.as_ptr() == left_pointer;
    let in_right = sum.value.as_ptr() == right_pointer;
    match placement(form, left_has_room, right_has_room) {
        Placement::Left => {
            assert!(in_left, "{}: the sum is not in the vector of a", inputs());
            assert_eq!(allocations, 0, "{}: allocations", inputs());
        }
        Placement::Right => {
            assert!(in_right, "{}: the sum is not in the vector of b", inputs());
            assert_eq!(allocations, 0, "{}: allocations", inputs());
        }
        Placement::New => assert_eq!(allocations, 1, "{}: allocations", inputs()),
    }
}

/// Where the allocation promises put the sum of `form`, given which operands have room for it.
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

/// Checks that the sum of `(roomy, other)` has a length within the capacity of `roomy`, and
/// exactly that capacity when `exactly` is set, so that a table above says what it claims.
fn assert_room(roomy: &Number, other: &Number, exactly: bool) {
    let len = reference(roomy, other).magnitude.len();
    let capacity = roomy.value.capacity();
    let inputs = || format!("roomy = {}, other = {}", describe(roomy), describe(other));
    assert!(capacity >= len, "{}: no room for the sum", inputs());
    assert!(!exactly || capacity == len, "{}: room to spare", inputs());
}

#[test]
fn add_assign_grows_a_number_in_its_own_vector() {
    // 0xFF + 1 = 0x01_00: in big-endian order the old byte moves back to make room.
    for (order, sum) in [
        (ByteOrder::BigEndian, [1, 0]),
        (ByteOrder::LittleEndian, [0, 1]),
    ] {
        for form in [Form::AssignRef, Form::AssignValue] {
            let a = plus(&[0xFF], order, 4);
            let pointer = a.value.as_ptr();
            let one = Number::new_from_u8(1, ByteOrder::LittleEndian);
            let result = form.apply(a, one);
            let inputs = || format!("{} with a = 0xFF, b = 1", form.name());
            assert_shortest(&result, &inputs);
            assert_eq!(result.value, sum, "{}", inputs());
            assert_eq!(result.value.as_ptr(), pointer, "{}", inputs());
        }
    }
}

#[test]
fn add_assign_shrinks_a_number_in_its_own_vector() {
    let be = ByteOrder::BigEndian;
    // 0x01_00 - 1 = 0xFF, and 0x09_08_07_06_05 - 0x09_08_07_06_00 = 5: in big-endian order the
    // sum moves to the front of the vector.
    for (a, b, sum) in [
        (
            plus(&[1, 0], be, 2),
            Number::new_from_i8(-1, be),
            vec![0xFF],
        ),
        (
            plus(&[9, 8, 7, 6, 5], be, 5),
            Number::new_signed(vec![9, 8, 7, 6, 0], true, be),
            vec![5],
        ),
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
            let result = form.apply(a, b);
            assert_shortest(&result, &|| message.clone());
            assert_eq!(result.value, sum, "{message}");
            assert!(!result.is_negative(), "{message}");
            assert_eq!(result.value.as_ptr(), pointer, "{message}");
        }
    }
}

#[test]
fn value_forms_build_the_sum_in_the_left_vector_when_it_has_room() {
    for (roomy, other) in sums_with_room() {
        assert_room(&roomy, &other, false);
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
fn value_forms_build_the_sum_in_the_left_vector_when_it_just_fits() {
    for (roomy, other) in sums_that_just_fit() {
        assert_room(&roomy, &other, true);
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
fn ref_plus_value_builds_the_sum_in_the_right_vector_when_it_has_room() {
    for (roomy, other) in sums_with_room() {
        assert_room(&roomy, &other, false);
        assert_placement(Form::RefValue, &other, &roomy);
    }
}

#[test]
fn ref_plus_value_builds_the_sum_in_the_right_vector_when_it_just_fits() {
    for (roomy, other) in sums_that_just_fit() {
        assert_room(&roomy, &other, true);
        assert_placement(Form::RefValue, &other, &roomy);
    }
}

#[test]
fn value_plus_value_falls_back_on_the_right_vector() {
    // The left operand with no room to spare, the right one with room for the sum: the sum is
    // in the vector of the left operand if it fits there, and otherwise in that of the right
    // one.
    for (roomy, other) in sums_with_room().into_iter().chain(sums_that_just_fit()) {
        let tight = copy(&other, 0);
        for form in [Form::ValueValue, Form::AssignValue] {
            assert_placement(form, &tight, &roomy);
        }
    }
}

#[test]
fn ref_plus_ref_allocates_once() {
    let pairs = sums_with_room()
        .into_iter()
        .chain(sums_that_just_fit())
        .chain(sums_without_room());
    for (a, b) in pairs {
        assert_placement(Form::RefRef, &a, &b);
        assert_placement(Form::RefRef, &b, &a);
    }
}

#[test]
fn every_form_allocates_one_vector_when_no_vector_has_room() {
    for (a, b) in sums_without_room() {
        let len = reference(&a, &b).magnitude.len();
        assert!(a.value.capacity() < len && b.value.capacity() < len);
        for form in FORMS {
            assert_placement(form, &a, &b);
            assert_placement(form, &b, &a);
        }
    }
}

#[test]
fn placement_follows_the_capacities_of_random_operands() {
    // Capacities around the length of the sum, where a vector just has room for it or just
    // lacks it.
    let mut state = 0x2545_F491_4F6C_DD1D;
    for _ in 0..size(1000, 30) {
        let a = random_operand(&mut state, 40);
        let b = random_operand(&mut state, 40);
        let len = reference(&a, &b).magnitude.len();
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

/// Checks `a + b`, of operands of `limbs` limbs, against `expected` in every form, as
/// [`assert_adds_in`] does, or from 1000 limbs on in the form and capacity setup at `turn`, as
/// [`assert_adds_in_turn`] does: a debug build takes a while over each of those sums.
fn assert_long_adds(turn: &mut usize, limbs: usize, a: &Number, b: &Number, expected: &Expected) {
    if limbs < 1000 {
        assert_adds_in(forms_to_run(), a, b, expected);
    } else {
        assert_adds_in_turn(turn, a, b, expected);
    }
}

#[test]
fn carries_ripple_through_every_limb() {
    // x = [0xFF; len] and y = 1 + 256^(len - 1), so x + y = 256^len + 256^(len - 1). Both
    // operands are long, so the kernels get every whole limb of them, and the carry from the
    // bottom byte goes through all of them. With several chains, every part above the first
    // is added with no carry into it and comes out all ones, so on its fix-up the carry out
    // of the part below ripples through every limb of it.
    let mut turn = 0;
    for limbs in limb_counts() {
        for extra in EXTRA_BYTES {
            let len = 8 * limbs + extra;
            let x = vec![0xFF; len];
            let mut y = vec![0; len];
            y[0] = 1;
            y[len - 1] = 1;
            let mut sum = vec![0; len + 1];
            sum[len - 1] = 1;
            sum[len] = 1;
            for (left, right) in ORDER_PAIRS {
                for negative in [false, true] {
                    let sum = expected(&sum, negative);
                    // The sum built in the vector of either operand, which grows: big-endian
                    // it ends a byte further on, above the operand.
                    for (a, b) in [(&x, &y), (&y, &x)] {
                        let a = number(a, negative, left);
                        let b = number(b, negative, right);
                        assert_long_adds(&mut turn, limbs, &a, &b, &sum);
                    }
                }
            }
        }
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
            let difference = vec![0xFF; len - 1];
            for (left, right) in ORDER_PAIRS {
                for negative in [false, true] {
                    // x + -y and -y + x, and both negated: the sum is built in the vector of
                    // either operand, which shrinks, the larger or the smaller magnitude.
                    let sum = expected(&difference, negative);
                    let a = number(&x, negative, left);
                    let b = number(&y, !negative, right);
                    assert_long_adds(&mut turn, limbs, &a, &b, &sum);
                    let a = number(&y, !negative, left);
                    let b = number(&x, negative, right);
                    assert_long_adds(&mut turn, limbs, &a, &b, &sum);
                }
            }
        }
    }
}

#[test]
fn adds_long_runs_of_whole_limbs_as_the_reference() {
    // Operands of the limb counts above, both long, in runs of whole limbs: the second as long
    // as the first, a few limbs apart from it, of another of the counts, or the first with
    // new low limbs, so that the sum nearly cancels or nearly doubles.
    let mut state = 0x8C3F_AE2D_1B5E_9F47;
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
            assert_adds_in(forms_to_run(), &a, &b, &reference(&a, &b));
        }
    }
}
