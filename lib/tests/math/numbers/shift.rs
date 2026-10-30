use a::math::numbers::{ByteOrder, Number};
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::common::*;
use crate::allocations::count;

// Expected results

/// The bits of a magnitude given least significant byte first, least significant first.
fn bits(magnitude: &[u8]) -> Vec<bool> {
    magnitude
        .iter()
        .flat_map(|&byte| (0..8).map(move |i| byte >> i & 1 == 1))
        .collect()
}

/// The magnitude whose bits are `bits`, least significant first, as bytes least significant
/// first.
fn from_bits(bits: &[bool]) -> Vec<u8> {
    bits.chunks(8)
        .map(|c| {
            c.iter()
                .rev()
                .fold(0, |byte, &bit| byte << 1 | u8::from(bit))
        })
        .collect()
}

/// `a << k`, worked out independently of the implementation: `k` zero bits put below the bits
/// of the magnitude, and the sign kept. The tests on small values check it against `i128`
/// arithmetic.
fn shifted_left(a: &Number, k: usize) -> Expected {
    let mut b = vec![false; k];
    b.extend(bits(&le_bytes(a)));
    // A zero result is not negative: `expected` sees to it.
    expected(&from_bits(&b), a.negative)
}

/// `a >> k` rounded towards negative infinity, worked out independently of the implementation:
/// the bits of the magnitude from bit `k` up, plus one when the number is negative and a bit
/// below `k` is set.
fn shifted_right(a: &Number, k: usize) -> Expected {
    let b = bits(&le_bytes(a));
    let negative = a.negative && b.contains(&true);
    let mut quotient = b.get(k..).unwrap_or_default().to_vec();
    if negative && b.iter().take(k).any(|&bit| bit) {
        // Plus one: ones turn to zeros up to the first zero, which turns to one.
        match quotient.iter().position(|&bit| !bit) {
            Some(i) => {
                quotient[..i].fill(false);
                quotient[i] = true;
            }
            None => {
                quotient.fill(false);
                quotient.push(true);
            }
        }
    }
    expected(&from_bits(&quotient), negative)
}

// The forms of the shifts

/// The three ways to write a shift. Every form must give the same result, so the tests run them
/// all.
#[derive(Clone, Copy, PartialEq)]
enum Form {
    Value,
    Ref,
    Assign,
}

const FORMS: [Form; 3] = [Form::Value, Form::Ref, Form::Assign];

/// The direction of a shift.
#[derive(Clone, Copy, PartialEq)]
enum Direction {
    Left,
    Right,
}

impl Form {
    /// The form as it is written.
    fn name(self, direction: Direction) -> &'static str {
        match (direction, self) {
            (Direction::Left, Form::Value) => "a << k",
            (Direction::Left, Form::Ref) => "&a << k",
            (Direction::Left, Form::Assign) => "a <<= k",
            (Direction::Right, Form::Value) => "a >> k",
            (Direction::Right, Form::Ref) => "&a >> k",
            (Direction::Right, Form::Assign) => "a >>= k",
        }
    }

    /// Shifts `a` by `k` in this form and direction; an operand the form only borrows is
    /// dropped afterwards.
    fn apply(self, direction: Direction, a: Number, k: usize) -> Number {
        match (direction, self) {
            (Direction::Left, Form::Value) => a << k,
            (Direction::Left, Form::Ref) => &a << k,
            (Direction::Left, Form::Assign) => {
                let mut a = a;
                a <<= k;
                a
            }
            (Direction::Right, Form::Value) => a >> k,
            (Direction::Right, Form::Ref) => &a >> k,
            (Direction::Right, Form::Assign) => {
                let mut a = a;
                a >>= k;
                a
            }
        }
    }
}

/// Every form, or under Miri one picked at random: for the tests that check many shifts.
fn forms_to_run() -> &'static [Form] {
    if cfg!(miri) {
        let k = pick(FORMS.len());
        &FORMS[k..=k]
    } else {
        &FORMS
    }
}

/// The expected result of shifting `a` by `k` in `direction`.
fn reference(direction: Direction, a: &Number, k: usize) -> Expected {
    match direction {
        Direction::Left => shifted_left(a, k),
        Direction::Right => shifted_right(a, k),
    }
}

/// Checks `a << k` and `a >> k` against the reference in each form, as [`assert_shifts_in`]
/// does.
fn assert_shifts(a: &Number, k: usize) {
    for direction in [Direction::Left, Direction::Right] {
        let expected = reference(direction, a, k);
        assert_shifts_in(forms_to_run(), direction, a, k, &expected);
    }
}

/// Checks the shift of `a` by `k` in `direction` against `expected` in each of `forms`: with the
/// vector of an owned operand as tight as its value, with one byte to spare, with exactly the
/// room the result needs (when it is longer) and with room to spare, so that each form meets
/// each way it can place the result, and checks where the result is built and that nothing else
/// is allocated. Under Miri each form runs with one of these setups, picked at random.
fn assert_shifts_in(
    forms: &[Form],
    direction: Direction,
    a: &Number,
    k: usize,
    expected: &Expected,
) {
    let len = expected.magnitude.len();
    for &form in forms {
        let mut spares = vec![0];
        if form != Form::Ref {
            spares.extend([1, len.saturating_sub(a.value.len()), len + 9]);
        }
        let spares = if cfg!(miri) {
            let k = pick(spares.len());
            &spares[k..=k]
        } else {
            &spares[..]
        };
        for &spare in spares {
            assert_shift_with(form, direction, spare, a, k, expected);
        }
    }
}

/// Checks the shift of `a` by `k` in `form` and `direction` against `expected`, with `spare`
/// bytes to spare in the vector of `a` when the form takes it by value, and where the result is
/// built: in the vector of an owned `a` when it has room for the result, and otherwise in one new
/// vector of exactly its length, the only allocation.
fn assert_shift_with(
    form: Form,
    direction: Direction,
    spare: usize,
    a: &Number,
    k: usize,
    expected: &Expected,
) {
    let operand = copy(a, spare);
    let has_room = form != Form::Ref && operand.value.capacity() >= expected.magnitude.len();
    let pointer = operand.value.as_ptr();
    let (result, allocations) = count(|| form.apply(direction, operand, k));
    let inputs = || {
        format!(
            "{} with a = {}, k = {k}, {spare} bytes to spare",
            form.name(direction),
            describe(a)
        )
    };
    assert_result(&result, expected, a.byte_order, &inputs);
    if has_room {
        assert_eq!(result.value.as_ptr(), pointer, "{}: not in place", inputs());
        assert_eq!(allocations, 0, "{}: allocations", inputs());
    } else {
        assert_eq!(allocations, 1, "{}: allocations", inputs());
        assert_eq!(
            result.value.capacity(),
            result.value.len(),
            "{}: the new vector is not of the result's length",
            inputs()
        );
    }
}

/// `a` shifted by `k` in `form` and `direction`, checked to be in the shortest form and in the
/// byte order of `a`.
fn shift(form: Form, direction: Direction, a: &Number, k: usize) -> Number {
    let result = form.apply(direction, copy(a, 0), k);
    let inputs = || format!("{} with a = {}, k = {k}", form.name(direction), describe(a));
    assert_shortest(&result, &inputs);
    assert!(
        result.byte_order == a.byte_order,
        "{}: not in the byte order of a",
        inputs()
    );
    result
}

/// The shifts around the byte, group and register boundaries for an operand of `len` bytes:
/// every amount up to 17 bits, around 64 and 128 bits, around the length of the operand in bits
/// and in bytes, and a few more.
fn amounts(len: usize) -> Vec<usize> {
    let mut k: Vec<usize> = (0..=17).collect();
    for edge in [56, 64, 72, 120, 128, 136, 8 * len, 64 * len.div_ceil(8)] {
        k.extend(edge.saturating_sub(9)..=edge + 9);
    }
    k.extend([200, 1000, 4 * len + 3]);
    k.sort_unstable();
    k.dedup();
    if cfg!(miri) {
        // Under Miri, a few of them, picked at random.
        let mut state = 0x3C6E_F372_FE94_F82B ^ len as u64;
        k.retain(|_| below(&mut state, 8) == 0);
    }
    k
}

// Small values against `i128` arithmetic

#[test]
fn reference_agrees_with_i128() {
    let mut state = 0x6A09_E667_F3BC_C908;
    for _ in 0..size(2000, 50) {
        let x = random_i128(&mut state) >> 64;
        let k = below(&mut state, 64);
        for order in ORDERS {
            let a = Number::new_from_i128(x, order);
            assert_eq!(shifted_left(&a, k), expected_i128(x << k), "{x} << {k}");
            assert_eq!(shifted_right(&a, k), expected_i128(x >> k), "{x} >> {k}");
        }
    }
}

#[test]
fn shifts_small_values_as_i128() {
    // `x << k` for |x| below 2^62 and k up to 64 fits; `x >> k` for k from 128 is 0 or -1.
    let mut state = 0xBB67_AE85_84CA_A73B;
    for _ in 0..size(1500, 30) {
        let x = random_i128(&mut state) >> 64;
        for order in ORDERS {
            let a = Number::new_from_i128(x, order);
            let k = below(&mut state, 65);
            assert_shifts_in(
                forms_to_run(),
                Direction::Left,
                &a,
                k,
                &expected_i128(x << k),
            );
            let k = below(&mut state, 140);
            let quotient = if k < 128 { x >> k } else { x >> 127 };
            assert_shifts_in(
                forms_to_run(),
                Direction::Right,
                &a,
                k,
                &expected_i128(quotient),
            );
        }
    }
}

#[test]
fn shifts_every_small_integer_by_every_small_amount() {
    let range = size(300, 20) as i128;
    for x in -range..=range {
        for k in 0..=size(20, 10) {
            for order in ORDERS {
                let a = Number::new_from_i128(x, order);
                for form in forms_to_run() {
                    let left = shift(*form, Direction::Left, &a, k);
                    assert_eq!(value_of(&left), expected_i128(x << k), "{x} << {k}");
                    let right = shift(*form, Direction::Right, &a, k);
                    assert_eq!(value_of(&right), expected_i128(x >> k), "{x} >> {k}");
                }
            }
        }
    }
}

#[test]
fn shifts_right_as_i32_does() {
    // The rules of the primitive integers: rounding towards negative infinity.
    for (x, k, quotient) in [
        (-7, 1, -4),
        (7, 1, 3),
        (-1, 1, -1),
        (-1, 1000, -1),
        (7, 1000, 0),
        (-7, 1000, -1),
        (-8, 3, -1),
        (-9, 3, -2),
        (-0x1FF, 1, -0x100),
        (-0x100, 8, -1),
        (-0x101, 8, -2),
        (0, 1000, 0),
    ] {
        for order in ORDERS {
            let a = Number::new_from_i32(x, order);
            for form in FORMS {
                let result = shift(form, Direction::Right, &a, k);
                assert_eq!(
                    value_of(&result),
                    expected_i128(i128::from(quotient)),
                    "{x} >> {k}"
                );
            }
        }
    }
}

// Lengths, amounts and the forms of the operand

#[test]
fn shifts_every_length_by_amounts_around_the_boundaries() {
    // Every length up to 72 bytes and a few longer ones around multiples of 16 (where the kernels
    // change their number of blocks), with random bytes and with runs of 0xFF and of zeros, each
    // sign and byte order, by amounts around 8, 64 and 128 bits and around the operand's own
    // length.
    let mut state = 0x3C6E_F372_FE94_F82B;
    let lengths = (1..=size(72, 20)).chain([95, 96, 97, 127, 128, 129, 255, 256, 257]);
    for len in lengths {
        let mut all_ones = vec![0xFF; len];
        all_ones[0] = 0xFE;
        let magnitudes = [
            random_magnitude(&mut state, len),
            vec![0xFF; len],
            all_ones,
            [vec![0; len - 1], vec![1]].concat(),
        ];
        for magnitude in &magnitudes {
            if cfg!(miri) && below(&mut state, 4) != 0 {
                continue;
            }
            for k in amounts(len) {
                let negative = below(&mut state, 2) == 0;
                let order = ORDERS[below(&mut state, 2)];
                assert_shifts(&number(magnitude, negative, order), k);
            }
        }
    }
}

#[test]
fn shifts_operands_not_in_the_shortest_form() {
    // Zero bytes above the most significant one, up to more than a group of eight, and
    // big-endian, where they come first, the vector of the result starts where they did.
    let mut state = 0xA54F_F53A_5F1D_36F1;
    for len in [1, 5, 8, 16, 17, 24, 40, 100] {
        let magnitude = random_magnitude(&mut state, len);
        for padding in [1, 7, 8, 9, 16, 30] {
            let mut padded = magnitude.clone();
            padded.resize(len + padding, 0);
            for order in ORDERS {
                for negative in [false, true] {
                    let a = number(&padded, negative, order);
                    for k in [0, 1, 7, 8, 13, 64, 100, 8 * len - 1, 8 * len + 3] {
                        if cfg!(miri) && below(&mut state, 6) != 0 {
                            continue;
                        }
                        assert_shifts(&a, k);
                    }
                }
            }
        }
    }
}

#[test]
fn zero_in_every_form_stays_zero() {
    // An empty vector, zero bytes, and a negative zero: zero shifted either way is `[0]`, never
    // negative, for any amount.
    for order in ORDERS {
        for z in zeros(order) {
            for k in [0, 1, 8, 100, 1 << 20, usize::MAX] {
                for direction in [Direction::Left, Direction::Right] {
                    assert_shifts_in(&FORMS, direction, &z, k, &zero());
                }
            }
        }
    }
}

#[test]
fn shifts_long_random_operands_as_the_reference() {
    let mut state = 0x510E_527F_ADE6_82D1;
    for _ in 0..size(300, 6) {
        let len = below(&mut state, size(3000, 120)) + 1;
        let mut magnitude = random_magnitude(&mut state, len);
        if below(&mut state, 4) == 0 {
            magnitude.resize(len + below(&mut state, 20), 0);
        }
        let a = number(
            &magnitude,
            below(&mut state, 2) == 0,
            ORDERS[below(&mut state, 2)],
        );
        let k = match below(&mut state, 4) {
            0 => below(&mut state, 130),
            1 => 8 * below(&mut state, len + 2),
            _ => below(&mut state, 8 * len + 40),
        };
        assert_shifts(&a, k);
    }
}

#[test]
fn shifts_long_runs_through_every_group() {
    // Operands of whole groups of eight bytes and a few more, in runs of zeros, 0xFF and random
    // bytes, so that the bits carried between groups and the rounding of negative quotients
    // meet every pattern, by amounts with and without whole bytes.
    let mut state = 0x9B05_688C_2B3E_6C1F;
    for limbs in limb_counts().into_iter().filter(|&n| n <= size(1025, 33)) {
        for extra in EXTRA_BYTES {
            let magnitude = limb_runs(&mut state, limbs, extra);
            let len = magnitude.len();
            for k in [1, 7, 8, 9, 63, 64, 65, 8 * len / 2 + 5] {
                if cfg!(miri) && below(&mut state, 4) != 0 {
                    continue;
                }
                let a = number(
                    &magnitude,
                    below(&mut state, 2) == 0,
                    ORDERS[below(&mut state, 2)],
                );
                assert_shifts(&a, k);
            }
        }
    }
}

// Rounding towards negative infinity

#[test]
fn rounding_down_carries_through_every_byte_of_the_quotient() {
    // -(2^(8n) - 1) · 2^r - 1 >> r: the quotient before rounding is n bytes of 0xFF, and rounded
    // down it is 256^n, one byte longer. The same with whole bytes below it, and the quotient
    // just one short of all ones, which does not grow.
    let mut state = 0x1F83_D9AB_FB41_BD6B;
    for n in (1..=size(40, 20)).chain([63, 64, 65, 200]) {
        for r in 0..8 {
            for below_bytes in [0, 1, 9] {
                let k = 8 * below_bytes + r;
                if k == 0 || (cfg!(miri) && below(&mut state, 4) != 0) {
                    continue;
                }
                // The bits below the quotient: a single one at the bottom.
                let mut b = vec![true];
                b.resize(k, false);
                b.extend(vec![true; 8 * n]);
                let all_ones = from_bits(&b);
                let mut short_of = b.clone();
                short_of[k] = false;
                let short_of = from_bits(&short_of);
                // Also with zero bytes above the most significant one, which big-endian in place
                // are moved out of the way first.
                let mut padded = all_ones.clone();
                padded.resize(all_ones.len() + 9, 0);
                for order in ORDERS {
                    for magnitude in [&all_ones, &short_of, &padded] {
                        let a = number(magnitude, true, order);
                        let expected = shifted_right(&a, k);
                        assert_shifts_in(forms_to_run(), Direction::Right, &a, k, &expected);
                    }
                }
                let quotient = shifted_right(&number(&all_ones, true, ORDERS[0]), k);
                assert_eq!(quotient.magnitude.len(), n + 1);
                assert_eq!(quotient.magnitude[n], 1);
            }
        }
    }
}

#[test]
fn a_negative_number_shifted_right_by_its_length_or_more_is_minus_one() {
    let mut state = 0x5BE0_CD19_137E_2179;
    for len in [1, 2, 8, 16, 17, 40, 300] {
        let magnitude = random_magnitude(&mut state, len);
        let bits = 8 * len;
        for order in ORDERS {
            let minus = number(&magnitude, true, order);
            let plus = number(&magnitude, false, order);
            for k in [
                bits,
                bits + 1,
                bits + 8,
                10 * bits,
                usize::MAX / 2,
                usize::MAX,
            ] {
                assert_shifts_in(
                    forms_to_run(),
                    Direction::Right,
                    &minus,
                    k,
                    &expected_i128(-1),
                );
                assert_shifts_in(forms_to_run(), Direction::Right, &plus, k, &zero());
            }
        }
    }
}

// Placement

#[test]
fn shl_assign_grows_a_number_in_its_own_vector() {
    // 0xFF << 9 = 0x01_FE_00 and longer numbers, in their own vectors: in big-endian order the
    // bytes stay at the start and the zeros come at the end.
    for order in ORDERS {
        for len in [1, 16, 17, 100] {
            for k in [1, 9, 64, 200] {
                for form in [Form::Assign, Form::Value] {
                    let a = plus(&vec![0xFF; len], order, len + k / 8 + 1);
                    let expected = shifted_left(&a, k);
                    let pointer = a.value.as_ptr();
                    let (result, allocations) = count(|| form.apply(Direction::Left, a, k));
                    let inputs = || {
                        format!(
                            "{} with {len} bytes of 0xFF, k = {k}",
                            form.name(Direction::Left)
                        )
                    };
                    assert_result(&result, &expected, order, &inputs);
                    assert_eq!(result.value.as_ptr(), pointer, "{}", inputs());
                    assert_eq!(allocations, 0, "{}", inputs());
                }
            }
        }
    }
}

#[test]
fn shr_never_allocates_for_an_owned_operand() {
    // The quotient is never longer than its operand, so it is built in the operand's vector,
    // whose capacity is exactly the operand's length here, rounding carries and all; only zero
    // in a vector without capacity needs one.
    let mut state = 0x6A09_E667_F3BC_C908;
    for _ in 0..size(500, 20) {
        let a = random_operand(&mut state, 200);
        let k = below(&mut state, 8 * a.value.len() + 20);
        let expected = shifted_right(&a, k);
        for form in [Form::Value, Form::Assign] {
            let operand = copy(&a, 0);
            let pointer = operand.value.as_ptr();
            let has_capacity = operand.value.capacity() > 0;
            let (result, allocations) = count(|| form.apply(Direction::Right, operand, k));
            let inputs = || {
                format!(
                    "{} with a = {}, k = {k}",
                    form.name(Direction::Right),
                    describe(&a)
                )
            };
            assert_result(&result, &expected, a.byte_order, &inputs);
            if has_capacity {
                assert_eq!(result.value.as_ptr(), pointer, "{}", inputs());
                assert_eq!(allocations, 0, "{}", inputs());
            } else {
                assert_eq!(allocations, 1, "{}", inputs());
                assert_eq!(result.value.capacity(), 1, "{}", inputs());
            }
        }
    }
}

#[test]
fn placement_follows_the_capacity_of_random_operands() {
    // Capacities around the length of the result, where the vector just has room for it or just
    // lacks it.
    let mut state = 0x2545_F491_4F6C_DD1D;
    for _ in 0..size(1500, 40) {
        let a = random_operand(&mut state, 60);
        let k = below(&mut state, 8 * a.value.len() + 150);
        for direction in [Direction::Left, Direction::Right] {
            let expected = reference(direction, &a, k);
            let len = expected.magnitude.len();
            let capacity = match below(&mut state, 5) {
                0 => len.saturating_sub(1),
                1 => len,
                2 => len + 1,
                3 => 0,
                _ => len + 9,
            };
            let a = stored(
                &a.value,
                a.negative,
                a.byte_order,
                capacity.max(a.value.len()),
            );
            for form in FORMS {
                assert_shift_with(form, direction, 0, &copy_with_capacity(&a), k, &expected);
            }
        }
    }
}

#[test]
fn borrowed_operands_are_left_as_they_were() {
    let mut state = 0xCBBB_9D5D_C105_9ED8;
    for _ in 0..size(200, 10) {
        let a = random_operand(&mut state, 100);
        let before = copy(&a, 0);
        let k = below(&mut state, 8 * a.value.len() + 20);
        let _ = &a << k;
        assert!(
            same(&a, &before),
            "&a << {k} changed a = {}",
            describe(&before)
        );
        let _ = &a >> k;
        assert!(
            same(&a, &before),
            "&a >> {k} changed a = {}",
            describe(&before)
        );
    }
}

// Huge amounts

#[test]
fn one_shifted_far_is_a_power_of_two() {
    // 2^k for k up to 100,000 bits, mostly zero bytes below a single bit, from a one-byte operand
    // and from 17 bytes holding one with zeros above it.
    for order in ORDERS {
        for k in [0, 7, 8, 127, 128, 129, 1000, size(100_000, 5_000)] {
            for one in [
                number(&[1], false, order),
                number(&[[1].as_slice(), &[0; 16]].concat(), true, order),
            ] {
                for form in FORMS {
                    let result = shift(form, Direction::Left, &one, k);
                    let mut magnitude = vec![0; k / 8];
                    magnitude.push(1 << (k % 8));
                    assert_eq!(
                        value_of(&result),
                        expected(&magnitude, one.negative),
                        "{} with k = {k}",
                        form.name(Direction::Left)
                    );
                }
            }
        }
    }
}

#[test]
fn a_left_shift_too_long_to_allocate_panics() {
    // Zero is zero whatever the amount; anything else shifted by usize::MAX bits would need more
    // memory than there is, and the shift panics instead of wrapping its length around. Under
    // Miri, which would try to run the allocation, the panic is not checked.
    for order in ORDERS {
        for form in FORMS {
            for z in zeros(order) {
                let result = form.apply(Direction::Left, copy(&z, 0), usize::MAX);
                assert_eq!(value_of(&result), zero());
            }
            if cfg!(miri) {
                continue;
            }
            for len in [1, 16, 17, 100] {
                let a = number(&vec![0x5A; len], true, order);
                for k in [usize::MAX, usize::MAX - 7, usize::MAX / 2] {
                    let operand = copy(&a, 0);
                    let result =
                        catch_unwind(AssertUnwindSafe(|| form.apply(Direction::Left, operand, k)));
                    assert!(
                        result.is_err(),
                        "{} with {len} bytes, k = {k}: no panic",
                        form.name(Direction::Left)
                    );
                }
            }
        }
    }
}

// Properties

#[test]
fn a_left_shift_is_a_product_with_a_power_of_two() {
    let mut state = 0x428A_2F98_D728_AE22;
    for _ in 0..size(300, 10) {
        let a = random_operand(&mut state, 200);
        let k = below(&mut state, 300);
        let mut power = vec![0; k / 8];
        power.push(1 << (k % 8));
        let power = number(&power, false, ORDERS[below(&mut state, 2)]);
        let form = FORMS[below(&mut state, 3)];
        let shifted = shift(form, Direction::Left, &a, k);
        assert_eq!(
            value_of(&shifted),
            value_of(&(&a * &power)),
            "{}: a = {}, k = {k}",
            form.name(Direction::Left),
            describe(&a)
        );
    }
}

#[test]
fn a_right_shift_undoes_a_left_shift() {
    let mut state = 0x7137_4491_23EF_65CD;
    for _ in 0..size(300, 10) {
        let a = random_operand(&mut state, 200);
        let k = below(&mut state, 300);
        let form = FORMS[below(&mut state, 3)];
        let there = shift(form, Direction::Left, &a, k);
        let back = shift(form, Direction::Right, &there, k);
        assert_eq!(
            value_of(&back),
            value_of(&a),
            "{}: a = {}, k = {k}",
            form.name(Direction::Right),
            describe(&a)
        );
    }
}

#[test]
fn shifting_twice_adds_the_amounts() {
    // (a >> j) >> k = a >> (j + k) holds for rounding towards negative infinity, as it does for
    // the primitive integers, and (a << j) << k = a << (j + k).
    let mut state = 0xB5C0_FBCF_EC4D_3B2F;
    for _ in 0..size(300, 10) {
        let a = random_operand(&mut state, 150);
        let (j, k) = (below(&mut state, 200), below(&mut state, 200));
        let form = FORMS[below(&mut state, 3)];
        for direction in [Direction::Left, Direction::Right] {
            let twice = shift(form, direction, &shift(form, direction, &a, j), k);
            let once = shift(form, direction, &a, j + k);
            assert_eq!(
                value_of(&twice),
                value_of(&once),
                "{}: a = {}, j = {j}, k = {k}",
                form.name(direction),
                describe(&a)
            );
        }
    }
}

#[test]
fn byte_orders_give_the_same_values() {
    let mut state = 0xE9B5_DBA5_8189_DBBC;
    for _ in 0..size(300, 10) {
        let a = random_operand(&mut state, 120);
        let k = below(&mut state, 8 * a.value.len() + 20);
        let other = number(
            &le_bytes(&a),
            a.negative,
            if a.byte_order == ByteOrder::BigEndian {
                ByteOrder::LittleEndian
            } else {
                ByteOrder::BigEndian
            },
        );
        for direction in [Direction::Left, Direction::Right] {
            let form = FORMS[below(&mut state, 3)];
            assert_eq!(
                value_of(&shift(form, direction, &a, k)),
                value_of(&shift(form, direction, &other, k)),
                "{}: a = {}, k = {k}",
                form.name(direction),
                describe(&a)
            );
        }
    }
}

// The paths of the implementation

#[test]
fn zero_vectors_of_every_length_stay_zero() {
    // Zero bytes of every length up to 40, of either sign, through the registers up to 16 bytes,
    // the four limbs of a left shift by fewer than 64 bits up to 32, and the kernels beyond.
    for order in ORDERS {
        for len in 0..=size(40, 34) {
            for negative in [false, true] {
                let z = number(&vec![0; len], negative, order);
                for k in [0, 1, 13, 63, 64, 100] {
                    if cfg!(miri) && pick(4) != 0 {
                        continue;
                    }
                    for direction in [Direction::Left, Direction::Right] {
                        assert_shifts_in(forms_to_run(), direction, &z, k, &zero());
                    }
                }
            }
        }
    }
}

#[test]
fn kernels_store_at_every_offset() {
    // From 64 bytes, the kernels store their blocks at addresses on 16 bytes, with one more
    // block at an end: every whole-byte offset of the result modulo 16, with bits, both ways and
    // in both byte orders, and lengths around the multiples of 16.
    let mut state = 0x7137_4491_B5C0_FBCF;
    for len in [63, 64, 65, 79, 80, 81, 95, 96, 100, 127, 129, 200] {
        let magnitude = random_magnitude(&mut state, len);
        for q in 0..=33 {
            for r in [1, 4, 7] {
                if cfg!(miri) && below(&mut state, 30) != 0 {
                    continue;
                }
                let a = number(
                    &magnitude,
                    below(&mut state, 2) == 0,
                    ORDERS[below(&mut state, 2)],
                );
                assert_shifts(&a, 8 * q + r);
            }
        }
    }
}

#[test]
fn left_shifts_in_registers_up_to_32_bytes() {
    // 17 to 32 bytes shifted left by fewer than 64 bits are computed in registers: every length
    // and every amount, results from 17 to 40 bytes, with the top byte of each size.
    let mut state = 0xE9B5_DBA5_3956_C25B;
    for len in 17..=32 {
        for top in [1, 0x7F, 0x80, 0xFF] {
            let mut magnitude = random_magnitude(&mut state, len);
            magnitude[len - 1] = top;
            for k in 0..64 {
                if cfg!(miri) && below(&mut state, 40) != 0 {
                    continue;
                }
                let a = number(
                    &magnitude,
                    below(&mut state, 2) == 0,
                    ORDERS[below(&mut state, 2)],
                );
                let expected = shifted_left(&a, k);
                assert_shifts_in(forms_to_run(), Direction::Left, &a, k, &expected);
            }
        }
    }
}
