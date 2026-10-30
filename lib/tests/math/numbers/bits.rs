use a::math::numbers::{ByteOrder, Number};
use std::panic::{AssertUnwindSafe, catch_unwind};

use super::common::*;
use crate::allocations::count;

// Expected results

/// The three bitwise operators on two numbers.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Op {
    And,
    Or,
    Xor,
}

const OPS: [Op; 3] = [Op::And, Op::Or, Op::Xor];

impl Op {
    /// The operator as it is written.
    fn symbol(self) -> &'static str {
        match self {
            Op::And => "&",
            Op::Or => "|",
            Op::Xor => "^",
        }
    }

    /// The operator on two limbs.
    fn on_u32(self, x: u32, y: u32) -> u32 {
        match self {
            Op::And => x & y,
            Op::Or => x | y,
            Op::Xor => x ^ y,
        }
    }

    /// The operator on two `i128`s.
    fn on_i128(self, x: i128, y: i128) -> i128 {
        match self {
            Op::And => x & y,
            Op::Or => x | y,
            Op::Xor => x ^ y,
        }
    }
}

/// The number whose two's complement is `limbs`, least significant first, its top bit the sign,
/// as an expected result.
fn from_complement(mut limbs: Vec<u32>) -> Expected {
    let negative = limbs.last().is_some_and(|top| top >> 31 == 1);
    if negative {
        negate(&mut limbs);
    }
    let magnitude: Vec<u8> = limbs.iter().flat_map(|limb| limb.to_le_bytes()).collect();
    expected(&magnitude, negative)
}

/// `a op b`, worked out independently of the implementation: in two's complement on 32-bit
/// limbs, one limb more than the longer vector fills, so that the top one holds the signs. The
/// tests on small values check it against `i128` arithmetic.
fn reference(op: Op, a: &Number, b: &Number) -> Expected {
    let limbs = a.value.len().max(b.value.len()) / 4 + 2;
    let x = twos_complement(a, limbs);
    let y = twos_complement(b, limbs);
    from_complement(x.iter().zip(&y).map(|(&x, &y)| op.on_u32(x, y)).collect())
}

/// `!a`, worked out independently: every limb of the two's complement inverted.
fn complemented(a: &Number) -> Expected {
    let limbs = a.value.len() / 4 + 2;
    from_complement(twos_complement(a, limbs).iter().map(|&x| !x).collect())
}

/// The two's complement of `a` with room for bit `i` and a sign limb above it.
fn complement_with(a: &Number, i: u64) -> Vec<u32> {
    twos_complement(a, (a.value.len() / 4).max(i as usize / 32) + 2)
}

/// Bit `i` of `a` in two's complement, worked out independently.
fn bit_of(a: &Number, i: u64) -> bool {
    complement_with(a, i)[i as usize / 32] >> (i % 32) & 1 == 1
}

/// `a` with bit `i` set to `value` in two's complement, worked out independently.
fn with_bit(a: &Number, i: u64, value: bool) -> Expected {
    let mut x = complement_with(a, i);
    let mask = 1 << (i % 32);
    if value {
        x[i as usize / 32] |= mask;
    } else {
        x[i as usize / 32] &= !mask;
    }
    from_complement(x)
}

// The forms of the binary operators

/// The six ways to write a bitwise operation. Every form must give the same result, so the
/// tests run them all.
#[derive(Clone, Copy, PartialEq)]
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
    /// The form as it is written, for the operator `op`.
    fn name(self, op: Op) -> String {
        let s = op.symbol();
        match self {
            Form::ValueValue => format!("a {s} b"),
            Form::ValueRef => format!("a {s} &b"),
            Form::RefValue => format!("&a {s} b"),
            Form::RefRef => format!("&a {s} &b"),
            Form::AssignValue => format!("a {s}= b"),
            Form::AssignRef => format!("a {s}= &b"),
        }
    }

    /// `a op b` in this form; an operand the form only borrows is dropped afterwards.
    fn apply(self, op: Op, a: Number, b: Number) -> Number {
        let mut a = a;
        match (op, self) {
            (Op::And, Form::ValueValue) => a & b,
            (Op::And, Form::ValueRef) => a & &b,
            (Op::And, Form::RefValue) => &a & b,
            (Op::And, Form::RefRef) => &a & &b,
            (Op::And, Form::AssignValue) => {
                a &= b;
                a
            }
            (Op::And, Form::AssignRef) => {
                a &= &b;
                a
            }
            (Op::Or, Form::ValueValue) => a | b,
            (Op::Or, Form::ValueRef) => a | &b,
            (Op::Or, Form::RefValue) => &a | b,
            (Op::Or, Form::RefRef) => &a | &b,
            (Op::Or, Form::AssignValue) => {
                a |= b;
                a
            }
            (Op::Or, Form::AssignRef) => {
                a |= &b;
                a
            }
            (Op::Xor, Form::ValueValue) => a ^ b,
            (Op::Xor, Form::ValueRef) => a ^ &b,
            (Op::Xor, Form::RefValue) => &a ^ b,
            (Op::Xor, Form::RefRef) => &a ^ &b,
            (Op::Xor, Form::AssignValue) => {
                a ^= b;
                a
            }
            (Op::Xor, Form::AssignRef) => {
                a ^= &b;
                a
            }
        }
    }

    /// Whether the form takes `a` by value.
    fn owns_a(self) -> bool {
        matches!(
            self,
            Form::ValueValue | Form::ValueRef | Form::AssignValue | Form::AssignRef
        )
    }

    /// Whether the form takes `b` by value.
    fn owns_b(self) -> bool {
        matches!(self, Form::ValueValue | Form::RefValue | Form::AssignValue)
    }
}

/// Every form, or under Miri one picked at random: for the tests that check many results.
fn forms_to_run() -> &'static [Form] {
    if cfg!(miri) {
        let k = pick(FORMS.len());
        &FORMS[k..=k]
    } else {
        &FORMS
    }
}

/// Every operator, or under Miri one picked at random.
fn ops_to_run() -> &'static [Op] {
    if cfg!(miri) {
        let k = pick(OPS.len());
        &OPS[k..=k]
    } else {
        &OPS
    }
}

/// `items`, or under Miri `keep` of them picked at random: for the tests that run many shapes.
fn sample<T: Copy>(items: &[T], keep: usize) -> Vec<T> {
    if cfg!(miri) {
        (0..keep.min(items.len()))
            .map(|_| items[pick(items.len())])
            .collect()
    } else {
        items.to_vec()
    }
}

/// Checks `a op b` against `expected` in every form, as [`assert_bitwise_in`] does.
fn assert_bitwise(op: Op, a: &Number, b: &Number, expected: &Expected) {
    assert_bitwise_in(&FORMS, op, a, b, expected);
}

/// Checks `a op b` against `expected` in each of `forms`: with the vectors of the operands the
/// form takes by value as tight as their values, with one byte to spare, with exactly the room
/// the result needs, and with room to spare, and for `a op b` and `a op= b` also with room in
/// `b` only, so that each form meets each way it can place the result. Under Miri each form runs
/// with one of these setups, picked at random.
fn assert_bitwise_in(forms: &[Form], op: Op, a: &Number, b: &Number, expected: &Expected) {
    let len = expected.magnitude.len();
    for &form in forms {
        let fit = |n: &Number| len.saturating_sub(n.value.len());
        let mut spares = vec![(0, 0)];
        if form != Form::RefRef {
            spares.extend([(1, 1), (fit(a), fit(b)), (len + 9, len + 9)]);
        }
        if form.owns_a() && form.owns_b() {
            spares.extend([(0, fit(b)), (0, len + 9)]);
        }
        let spares = if cfg!(miri) {
            let k = pick(spares.len());
            &spares[k..=k]
        } else {
            &spares[..]
        };
        for &(spare_a, spare_b) in spares {
            assert_bitwise_with(form, op, spare_a, spare_b, a, b, expected);
        }
    }
}

/// Checks `a op b` against `expected` in `form`, with `spare_a` and `spare_b` bytes to spare in
/// the vectors of the operands, and where the result is built: in the vector of an owned `a`
/// when it has room, or else of an owned `b` when that has room, allocating nothing, and
/// otherwise in one new vector of exactly its length, the only allocation.
fn assert_bitwise_with(
    form: Form,
    op: Op,
    spare_a: usize,
    spare_b: usize,
    a: &Number,
    b: &Number,
    expected: &Expected,
) {
    let (x, y) = (copy(a, spare_a), copy(b, spare_b));
    let (pa, pb) = (x.value.as_ptr(), y.value.as_ptr());
    let len = expected.magnitude.len();
    let room_a = form.owns_a() && x.value.capacity() >= len;
    let room_b = form.owns_b() && y.value.capacity() >= len;
    let (result, allocations) = count(|| form.apply(op, x, y));
    let inputs = || {
        format!(
            "{} with a = {}, b = {}, {spare_a} and {spare_b} bytes to spare",
            form.name(op),
            describe(a),
            describe(b)
        )
    };
    assert_result(&result, expected, a.byte_order, &inputs);
    if room_a {
        assert_eq!(result.value.as_ptr(), pa, "{}: not in a", inputs());
        assert_eq!(allocations, 0, "{}: allocations", inputs());
    } else if room_b {
        assert_eq!(result.value.as_ptr(), pb, "{}: not in b", inputs());
        assert_eq!(allocations, 0, "{}: allocations", inputs());
    } else {
        assert_eq!(allocations, 1, "{}: allocations", inputs());
        assert_eq!(
            result.value.capacity(),
            len,
            "{}: the new vector is not of the result's length",
            inputs()
        );
    }
}

/// `a op b` in `form`, checked against the reference.
fn check(form: Form, op: Op, a: &Number, b: &Number) {
    let expected = reference(op, a, b);
    let result = form.apply(op, copy(a, 0), copy(b, 0));
    let inputs = || {
        format!(
            "{} with a = {}, b = {}",
            form.name(op),
            describe(a),
            describe(b)
        )
    };
    assert_result(&result, &expected, a.byte_order, &inputs);
}

/// A random number of a magnitude of `len` bytes up to its top one (runs of zeros, `0xFF` and
/// random bytes, as [`random_magnitude`] makes), of a random sign.
fn random_number(state: &mut u64, len: usize, order: ByteOrder) -> Number {
    number(&random_magnitude(state, len), below(state, 2) == 0, order)
}

// Small values against `i128` arithmetic

#[test]
fn reference_agrees_with_i128() {
    let mut state = 0x510E_527F_ADE6_82D1;
    for _ in 0..size(3000, 60) {
        let (x, y) = (random_i128(&mut state), random_i128(&mut state));
        for (left, right) in ORDER_PAIRS {
            let (a, b) = (
                Number::new_from_i128(x, left),
                Number::new_from_i128(y, right),
            );
            for op in OPS {
                assert_eq!(
                    reference(op, &a, &b),
                    expected_i128(op.on_i128(x, y)),
                    "{x} {} {y}",
                    op.symbol()
                );
            }
            assert_eq!(complemented(&a), expected_i128(!x), "!{x}");
            for i in [0, 1, 7, 8, 63, 64, 100, 126, 127] {
                assert_eq!(bit_of(&a, i), x >> i & 1 == 1, "bit {i} of {x}");
            }
        }
    }
}

#[test]
fn bitwise_small_values_as_i128() {
    let mut state = 0x9B05_688C_2B3E_6C1F;
    for _ in 0..size(700, 20) {
        let (x, y) = (random_i128(&mut state), random_i128(&mut state));
        for (left, right) in ORDER_PAIRS {
            let (a, b) = (
                Number::new_from_i128(x, left),
                Number::new_from_i128(y, right),
            );
            for &op in ops_to_run() {
                let expected = expected_i128(op.on_i128(x, y));
                assert_bitwise_in(forms_to_run(), op, &a, &b, &expected);
            }
        }
    }
}

#[test]
fn bitwise_every_pair_of_small_integers() {
    let range = size(130, 5) as i128;
    for x in -range..=range {
        for y in -range..=range {
            let order = |k: i128| ORDERS[(k & 1) as usize];
            let (a, b) = (
                Number::new_from_i128(x, order(x)),
                Number::new_from_i128(y, order(y + x / 3)),
            );
            for op in OPS {
                let result = match op {
                    Op::And => &a & &b,
                    Op::Or => &a | &b,
                    Op::Xor => &a ^ &b,
                };
                let inputs = || format!("{x} {} {y}", op.symbol());
                assert_result(
                    &result,
                    &expected_i128(op.on_i128(x, y)),
                    a.byte_order,
                    &inputs,
                );
            }
        }
    }
}

#[test]
fn bitwise_as_i32_does() {
    for (x, y) in [
        (12, -10),
        (-12, 10),
        (-12, -10),
        (-0x80, -0x81),
        (0xFF, -1),
        (-0x100, 0xFF),
        (-1, -1),
        (0, -1),
        (-0x8000, 0x7FFF),
        (i32::MIN, -1),
        (i32::MIN, i32::MAX),
    ] {
        for (left, right) in ORDER_PAIRS {
            let (a, b) = (
                Number::new_from_i32(x, left),
                Number::new_from_i32(y, right),
            );
            for op in OPS {
                let expected = expected_i128(op.on_i128(x.into(), y.into()));
                assert_bitwise_in(forms_to_run(), op, &a, &b, &expected);
            }
        }
    }
}

// Every length, every path

#[test]
fn bitwise_at_every_pair_of_lengths_up_to_80_bytes() {
    // The short path (both up to 16 bytes), the path in registers (up to 64) and the general one,
    // with every sign and both byte orders.
    let mut state = 0x1F83_D9AB_FB41_BD6B;
    let step = size(1, 9);
    for la in (0..=80).step_by(step) {
        for lb in (0..=80).step_by(step) {
            for (left, right) in ORDER_PAIRS {
                let a = random_number(&mut state, la, left);
                let b = random_number(&mut state, lb, right);
                for &op in ops_to_run() {
                    for &form in forms_to_run() {
                        check(form, op, &a, &b);
                    }
                }
            }
        }
    }
}

#[test]
fn bitwise_random_operands_as_the_reference() {
    let mut state = 0x5BE0_CD19_137E_2179;
    for _ in 0..size(1500, 25) {
        let a = random_operand(&mut state, 300);
        let b = random_operand(&mut state, 300);
        for &op in ops_to_run() {
            let expected = reference(op, &a, &b);
            assert_bitwise_in(forms_to_run(), op, &a, &b, &expected);
        }
    }
}

#[test]
fn bitwise_long_runs_of_limbs_as_the_reference() {
    // Runs of zero, all-ones and random limbs, so that the borrows of negative operands and the
    // carry of a negative result run through whole runs, the head included, at the lengths at
    // which the kernels change shape.
    let mut state = 0xCBBB_9D5D_C105_9ED8;
    let counts = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 200];
    for la in sample(&counts[..size(17, 9)], 4) {
        for lb in sample(&counts[..size(17, 9)], 4) {
            for extra in sample(&EXTRA_BYTES, 1) {
                for (left, right) in sample(&ORDER_PAIRS, 1) {
                    let a = number(
                        &limb_runs(&mut state, la, extra),
                        below(&mut state, 2) == 0,
                        left,
                    );
                    let b = number(
                        &limb_runs(&mut state, lb, 0),
                        below(&mut state, 2) == 0,
                        right,
                    );
                    for &op in ops_to_run() {
                        let expected = reference(op, &a, &b);
                        assert_bitwise_in(forms_to_run(), op, &a, &b, &expected);
                    }
                }
            }
        }
    }
}

#[test]
fn borrows_and_carries_cross_whole_magnitudes() {
    // Negative operands whose magnitudes are a power of two times something, so that their borrow
    // runs through all the zero bytes below; and results whose carry runs through all of them:
    // `-2^(8k-1) & -(2^(8k-1) + 1)` is `-2^(8k)`, one byte longer than either operand, and
    // `(2^(8k) - 1) ^ -1` is `-2^(8k)` too.
    for k in sample(
        &[1, 2, 8, 9, 16, 17, 24, 63, 64, 65, 100, size(1000, 70)],
        4,
    ) {
        let mut top = vec![0; k];
        top[k - 1] = 0x80;
        let mut top_plus_one = top.clone();
        top_plus_one[0] |= 1;
        let ones = vec![0xFF; k];
        let mut power = vec![0; k + 1];
        power[k] = 1;
        for (left, right) in sample(&ORDER_PAIRS, 1) {
            let a = number(&top, true, left);
            let b = number(&top_plus_one, true, right);
            assert_bitwise(Op::And, &a, &b, &expected(&power, true));
            let a = number(&ones, false, left);
            let minus_one = number(&[1], true, right);
            assert_bitwise(Op::Xor, &a, &minus_one, &expected(&power, true));
            assert_bitwise(Op::Or, &a, &minus_one, &expected(&[1], true));
            assert_bitwise(Op::And, &a, &minus_one, &expected(&ones, false));
            // A power of two and its negation: `x & -x` is the lowest one of `x`.
            let p = number(&power, false, left);
            let minus_p = number(&power, true, right);
            assert_bitwise(Op::And, &p, &minus_p, &expected(&power, false));
            assert_bitwise(Op::Or, &p, &minus_p, &expected(&power, true));
            for &op in ops_to_run() {
                for (x, y) in [(&a, &b), (&b, &a), (&p, &minus_one), (&minus_p, &a)] {
                    assert_bitwise_in(forms_to_run(), op, x, y, &reference(op, x, y));
                }
            }
        }
    }
}

#[test]
fn long_heads_and_results_that_move() {
    // Negative operands with tens of zero limbs at the low end, whose heads are longer than the
    // limbs the in-place path holds back, combined with operands of other lengths, so that a
    // big-endian result is longer or shorter than the vector it is built in.
    let mut state = 0x6295_C4F0_3B0E_8A37;
    for zeros in sample(&[0, 7, 8, 9, 31, 32, 33, 40, 41, 100], 3) {
        for la in sample(&[zeros + 1, zeros + 8, zeros + 70], 1) {
            for lb in sample(&[1, 9, 40, zeros + 2, zeros + 90], 2) {
                let mut x = vec![0; zeros];
                x.extend(random_magnitude(&mut state, la - zeros));
                let y = random_magnitude(&mut state, lb);
                for (left, right) in sample(&ORDER_PAIRS, 2) {
                    for (sa, sb) in sample(&SIGN_PAIRS, 2) {
                        let a = number(&x, sa, left);
                        let b = number(&y, sb, right);
                        for &op in ops_to_run() {
                            for (p, q) in [(&a, &b), (&b, &a)] {
                                assert_bitwise_in(forms_to_run(), op, p, q, &reference(op, p, q));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn operands_not_in_the_shortest_form() {
    // High zero bytes, empty vectors and negative zeros, in both byte orders and in every path.
    let mut state = 0x2F6C_5E1B_7A3D_9C04;
    for order in ORDERS {
        let mut operands = zeros(order);
        for len in [1, 5, 16, 17, 40, 64, 65, 130] {
            let m = random_magnitude(&mut state, len);
            for pad in [1, 9, 30] {
                let mut padded = m.clone();
                padded.resize(len + pad, 0);
                operands.push(number(&padded, below(&mut state, 2) == 0, order));
            }
        }
        for a in &operands {
            for b in &operands {
                if cfg!(miri) && pick(16) != 0 {
                    continue;
                }
                for &op in ops_to_run() {
                    assert_bitwise_in(forms_to_run(), op, a, b, &reference(op, a, b));
                }
            }
        }
    }
}

#[test]
fn zero_with_anything() {
    let mut state = 0x7137_4491_B5C0_FBCF;
    for len in sample(&[1, 16, 17, 64, 65, 100], 3) {
        for (left, right) in sample(&ORDER_PAIRS, 2) {
            for zero in zeros(left) {
                let b = random_number(&mut state, len, right);
                let value = value_of(&b);
                let forms = forms_to_run();
                assert_bitwise_in(forms, Op::And, &zero, &b, &expected(&[], false));
                assert_bitwise_in(forms, Op::Or, &zero, &b, &value);
                assert_bitwise_in(forms, Op::Xor, &zero, &b, &value);
                assert_bitwise_in(forms, Op::And, &b, &zero, &expected(&[], false));
                assert_bitwise_in(forms, Op::Or, &b, &zero, &value);
            }
        }
    }
}

// Laws

#[test]
fn identities_of_two_complement() {
    let mut state = 0xE9B5_DBA5_3956_C25B;
    for _ in 0..size(400, 10) {
        let a = random_operand(&mut state, 200);
        let b = random_operand(&mut state, 200);
        let not_a = !&a;
        // `x & !x` is 0, `x | !x` is -1, `x ^ !x` is -1, `x ^ x` is 0.
        assert_eq!(
            value_of(&(&a & &not_a)),
            expected(&[], false),
            "{}",
            describe(&a)
        );
        assert_eq!(
            value_of(&(&a | &not_a)),
            expected(&[1], true),
            "{}",
            describe(&a)
        );
        assert_eq!(
            value_of(&(&a ^ &not_a)),
            expected(&[1], true),
            "{}",
            describe(&a)
        );
        assert_eq!(
            value_of(&(&a ^ &a)),
            expected(&[], false),
            "{}",
            describe(&a)
        );
        // De Morgan: `!(a & b) == !a | !b`, `!(a | b) == !a & !b`.
        let not_b = !&b;
        assert_eq!(value_of(&!(&a & &b)), value_of(&(&not_a | &not_b)));
        assert_eq!(value_of(&!(&a | &b)), value_of(&(&not_a & &not_b)));
        // `a ^ b == (a | b) & !(a & b)`, and the operators are commutative.
        assert_eq!(value_of(&(&a ^ &b)), value_of(&((&a | &b) & !(&a & &b))));
        for op in OPS {
            let (ab, ba) = match op {
                Op::And => (&a & &b, &b & &a),
                Op::Or => (&a | &b, &b | &a),
                Op::Xor => (&a ^ &b, &b ^ &a),
            };
            assert_eq!(value_of(&ab), value_of(&ba));
        }
    }
}

#[test]
fn borrowed_operands_are_left_as_they_were() {
    let mut state = 0x3956_C25B_59F1_11F1;
    for _ in 0..size(200, 5) {
        let a = random_operand(&mut state, 150);
        let b = random_operand(&mut state, 150);
        let (keep_a, keep_b) = (copy(&a, 0), copy(&b, 0));
        for op in OPS {
            for form in FORMS {
                // Each form gets copies of the operands it takes by value, and borrows `a` and `b`
                // themselves otherwise.
                let _ = match (form.owns_a(), form.owns_b(), op) {
                    (false, false, Op::And) => &a & &b,
                    (false, false, Op::Or) => &a | &b,
                    (false, false, Op::Xor) => &a ^ &b,
                    (true, false, Op::And) => copy(&a, 4) & &b,
                    (true, false, Op::Or) => copy(&a, 4) | &b,
                    (true, false, Op::Xor) => copy(&a, 4) ^ &b,
                    (false, true, Op::And) => &a & copy(&b, 4),
                    (false, true, Op::Or) => &a | copy(&b, 4),
                    (false, true, Op::Xor) => &a ^ copy(&b, 4),
                    (true, true, _) => form.apply(op, copy(&a, 4), copy(&b, 4)),
                };
            }
        }
        let _ = !&a;
        assert!(same(&a, &keep_a), "a changed: {}", describe(&a));
        assert!(same(&b, &keep_b), "b changed: {}", describe(&b));
    }
}

#[test]
fn mixed_byte_orders_build_in_the_left_order() {
    // `&a op b` with `b` in the other order and room in its vector reverses `b` in place.
    let mut state = 0x923F_82A4_AB1C_5ED5;
    for len in sample(&[3, 16, 17, 40, 64, 65, 150], 3) {
        for (left, right) in [
            (ByteOrder::BigEndian, ByteOrder::LittleEndian),
            (ByteOrder::LittleEndian, ByteOrder::BigEndian),
        ] {
            for (sa, sb) in sample(&SIGN_PAIRS, 2) {
                let a = number(&random_magnitude(&mut state, len), sa, left);
                let b = number(&random_magnitude(&mut state, len + 3), sb, right);
                for &op in ops_to_run() {
                    assert_bitwise_in(forms_to_run(), op, &a, &b, &reference(op, &a, &b));
                    assert_bitwise_in(forms_to_run(), op, &b, &a, &reference(op, &b, &a));
                }
            }
        }
    }
}

// Not

/// Checks `!a` against `expected` in both forms: `!&a` in one new vector of exactly its length,
/// and `!a` with the vector of `a` as tight as its value, with a byte to spare and with room to
/// spare, in it when it has room, and otherwise in one new vector.
fn assert_not(a: &Number, expected: &Expected) {
    let inputs = || format!("!{}", describe(a));
    let (result, allocations) = count(|| !a);
    assert_result(&result, expected, a.byte_order, &inputs);
    assert_eq!(allocations, 1, "{}: allocations of !&a", inputs());
    assert_eq!(
        result.value.capacity(),
        result.value.len(),
        "{}: !&a",
        inputs()
    );
    let len = expected.magnitude.len();
    let spares = [0, 1, len + 3];
    let spares = if cfg!(miri) {
        let k = pick(spares.len());
        &spares[k..=k]
    } else {
        &spares[..]
    };
    for &spare in spares {
        let x = copy(a, spare);
        let (pointer, room) = (x.value.as_ptr(), x.value.capacity() >= len);
        let (result, allocations) = count(|| !x);
        let inputs = || format!("!{} with {spare} bytes to spare", describe(a));
        assert_result(&result, expected, a.byte_order, &inputs);
        if room {
            assert_eq!(result.value.as_ptr(), pointer, "{}: not in place", inputs());
            assert_eq!(allocations, 0, "{}: allocations", inputs());
        } else {
            assert_eq!(allocations, 1, "{}: allocations", inputs());
            assert_eq!(result.value.capacity(), len, "{}: new vector", inputs());
        }
    }
}

#[test]
fn not_small_values_as_i128() {
    let mut state = 0xD807_AA98_1283_5B01;
    for _ in 0..size(2000, 40) {
        let x = random_i128(&mut state);
        for order in ORDERS {
            assert_not(&Number::new_from_i128(x, order), &expected_i128(!x));
        }
    }
    let range = size(600, 20) as i128;
    for x in -range..=range {
        for order in ORDERS {
            assert_not(&Number::new_from_i128(x, order), &expected_i128(!x));
        }
    }
}

#[test]
fn not_runs_carries_and_borrows_through_whole_magnitudes() {
    // `!0xFF…FF` is `-0x01_00…00`, one byte longer; `!-0x01_00…00` is `0xFF…FF`, one byte
    // shorter; and the runs end anywhere inside a magnitude too.
    let mut state = 0x2431_85BE_550C_7DC3;
    for len in [
        1,
        2,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        64,
        65,
        size(1000, 80),
    ] {
        for order in ORDERS {
            let ones = number(&vec![0xFF; len], false, order);
            let mut power = vec![0; len];
            power.push(1);
            assert_not(&ones, &expected(&power, true));
            assert_not(
                &number(&power, true, order),
                &expected(&vec![0xFF; len], false),
            );
            for run in [0, 1, len / 2, len - 1] {
                let mut m = vec![0xFF; run];
                m.extend(random_magnitude(&mut state, len - run));
                let a = number(&m, false, order);
                assert_not(&a, &complemented(&a));
                let mut m = vec![0; run];
                m.extend(random_magnitude(&mut state, len - run));
                let a = number(&m, true, order);
                assert_not(&a, &complemented(&a));
            }
            let a = random_number(&mut state, len, order);
            assert_not(&a, &complemented(&a));
        }
    }
}

#[test]
fn not_of_unnormalized_numbers() {
    let mut state = 0x72BE_5D74_80DE_B1FE;
    for order in ORDERS {
        for zero in zeros(order) {
            assert_not(&zero, &expected(&[1], true));
        }
        for len in [1, 5, 16, 17, 40] {
            for pad in [1, 9, 20] {
                for negative in [false, true] {
                    let mut m = random_magnitude(&mut state, len);
                    m.resize(len + pad, 0);
                    let a = number(&m, negative, order);
                    assert_not(&a, &complemented(&a));
                    let mut m = vec![0xFF; len];
                    m.resize(len + pad, 0);
                    let a = number(&m, negative, order);
                    assert_not(&a, &complemented(&a));
                }
            }
        }
    }
}

#[test]
fn not_twice_is_the_number() {
    let mut state = 0x9BDC_06A7_C19B_F174;
    for _ in 0..size(500, 10) {
        let a = random_operand(&mut state, 300);
        assert_eq!(value_of(&!!copy(&a, 0)), value_of(&a), "{}", describe(&a));
        assert_eq!(value_of(&!&!&a), value_of(&a), "{}", describe(&a));
    }
}

// The bits of a number

/// The number of bits of the magnitude of `a`, worked out independently.
fn bit_length(a: &Number) -> u64 {
    let m = le_bytes(a);
    match m.iter().rposition(|&b| b != 0) {
        Some(top) => 8 * top as u64 + u64::from(8 - m[top].leading_zeros()),
        None => 0,
    }
}

#[test]
fn bits_of_small_values_as_i128() {
    let mut state = 0xE49B_69C1_EFBE_4786;
    for _ in 0..size(3000, 100) {
        let x = random_i128(&mut state);
        for order in ORDERS {
            let a = Number::new_from_i128(x, order);
            assert_eq!(
                a.bits(),
                u64::from(128 - x.unsigned_abs().leading_zeros()),
                "{x}"
            );
        }
    }
    assert_eq!(Number::new_from_u8(1, ByteOrder::BigEndian).bits(), 1);
    assert_eq!(
        Number::new_from_u128(u128::MAX, ByteOrder::LittleEndian).bits(),
        128
    );
}

#[test]
fn bits_of_long_and_unnormalized_numbers() {
    let mut state = 0x0FC1_9DC6_8B8C_D5B5;
    for order in ORDERS {
        for zero in zeros(order) {
            assert_eq!(zero.bits(), 0, "{}", describe(&zero));
        }
        for len in [1, 7, 8, 9, 100, 1000] {
            for pad in [0, 1, 17] {
                let mut m = random_magnitude(&mut state, len);
                m.resize(len + pad, 0);
                let a = number(&m, below(&mut state, 2) == 0, order);
                assert_eq!(a.bits(), bit_length(&a), "{}", describe(&a));
            }
        }
    }
}

#[test]
fn count_ones_counts_the_magnitude() {
    // Every length up to 300 bytes, and around the blocks of 64 bytes the NEON kernel adds up in
    // sixteen-bit lanes before it widens them, 4095 of them.
    let mut state = 0x240C_A1CC_77AC_9C65;
    let mut lengths: Vec<usize> = (0..=size(300, 40)).collect();
    if !cfg!(miri) {
        lengths.extend([
            4095 * 64 - 1,
            4095 * 64,
            4095 * 64 + 1,
            4095 * 64 + 63,
            2 * 4095 * 64 + 17,
        ]);
    }
    for len in lengths {
        let bytes = random_bytes(&mut state, len);
        let ones: u64 = bytes.iter().map(|b| u64::from(b.count_ones())).sum();
        for order in ORDERS {
            for negative in [false, true] {
                let a = number(&bytes, negative, order);
                assert_eq!(a.count_ones(), ones, "{len} bytes, {}", describe(&a));
            }
        }
        if len > 0 && len <= 64 {
            let all = number(&vec![0xFF; len], true, ByteOrder::LittleEndian);
            assert_eq!(all.count_ones(), 8 * len as u64);
        }
    }
    let big = number(&vec![0xFF; size(300_000, 300)], false, ByteOrder::BigEndian);
    assert_eq!(big.count_ones(), 8 * big.value.len() as u64);
}

#[test]
fn trailing_zeros_of_x_and_minus_x() {
    let mut state = 0x4A74_84AA_5CB0_A9DC;
    for _ in 0..size(2000, 50) {
        let x = random_i128(&mut state);
        for order in ORDERS {
            let a = Number::new_from_i128(x, order);
            let expected = (x != 0).then(|| u64::from(x.trailing_zeros()));
            assert_eq!(a.trailing_zeros(), expected, "{x}");
        }
    }
    for order in ORDERS {
        for zero in zeros(order) {
            assert_eq!(zero.trailing_zeros(), None, "{}", describe(&zero));
        }
        for zeros_below in [0, 1, 7, 8, 9, 63, 64, 65, 1000] {
            for pad in [0, 3] {
                let mut m = vec![0; zeros_below];
                m.push(0b1010_0000);
                m.extend(random_magnitude(&mut state, 5));
                m.resize(m.len() + pad, 0);
                for negative in [false, true] {
                    let a = number(&m, negative, order);
                    assert_eq!(a.trailing_zeros(), Some(8 * zeros_below as u64 + 5));
                }
            }
        }
    }
}

#[test]
fn bit_of_small_values_as_i128() {
    let mut state = 0x76F9_88DA_983E_5152;
    for _ in 0..size(500, 10) {
        let x = random_i128(&mut state);
        for order in ORDERS {
            let a = Number::new_from_i128(x, order);
            for i in 0..size(140, 20) as u64 {
                let expected = if i < 128 { x >> i & 1 == 1 } else { x < 0 };
                assert_eq!(a.bit(i), expected, "bit {i} of {x}");
            }
            assert_eq!(a.bit(u64::MAX), x < 0, "bit MAX of {x}");
        }
    }
}

#[test]
fn bit_of_long_and_unnormalized_numbers() {
    let mut state = 0xA831_C66D_B003_27C8;
    for order in ORDERS {
        for zero in zeros(order) {
            assert!(!zero.bit(0) && !zero.bit(100) && !zero.bit(u64::MAX));
        }
        for len in sample(&[1, 9, 40, 200], 2) {
            for low_zeros in sample(&[0, 1, 8, 20], 2) {
                for pad in sample(&[0, 2], 1) {
                    let mut m = vec![0; low_zeros];
                    m.extend(random_magnitude(&mut state, len));
                    m.resize(m.len() + pad, 0);
                    let a = number(&m, below(&mut state, 2) == 0, order);
                    let top = 8 * m.len() as u64 + 20;
                    let step = size(1, 13);
                    for i in (0..top).step_by(step) {
                        assert_eq!(a.bit(i), bit_of(&a, i), "bit {i} of {}", describe(&a));
                    }
                }
            }
        }
    }
}

/// Checks `a.set_bit(i, value)` against the reference: the value, the byte order and the
/// shortest form.
fn assert_set_bit(a: &Number, i: u64, value: bool) {
    let expected = with_bit(a, i, value);
    let mut n = copy(a, 0);
    n.set_bit(i, value);
    let inputs = || format!("{}.set_bit({i}, {value})", describe(a));
    assert_result(&n, &expected, a.byte_order, &inputs);
}

#[test]
fn set_bit_of_small_values_as_i128() {
    let mut state = 0xC6E0_0BF3_3DA8_8FC2;
    for _ in 0..size(400, 8) {
        // Below 2^125, so that setting a bit up to 125 keeps it in range.
        let x = random_i128(&mut state) >> 1;
        for order in ORDERS {
            let a = Number::new_from_i128(x, order);
            for i in 0..size(126, 12) as u64 {
                let set = x | (1 << i);
                let clear = x & !(1 << i);
                let mut n = copy(&a, 0);
                n.set_bit(i, true);
                assert_result(&n, &expected_i128(set), order, &|| format!("{x} set {i}"));
                let mut n = copy(&a, 0);
                n.set_bit(i, false);
                assert_result(&n, &expected_i128(clear), order, &|| {
                    format!("{x} clear {i}")
                });
            }
        }
    }
}

#[test]
fn set_bit_of_long_and_unnormalized_numbers() {
    // Around the lowest one of negative numbers (where setting and clearing borrow and carry),
    // around the top (where the number grows or shrinks, and big-endian moves), and far above.
    let mut state = 0xD5A7_9147_06CA_6351;
    for order in ORDERS {
        for zero in zeros(order) {
            for i in [0, 7, 8, 100] {
                assert_set_bit(&zero, i, true);
                assert_set_bit(&zero, i, false);
            }
        }
        for len in sample(&[1, 2, 9, 33, 100], 3) {
            for low_zeros in sample(&[0, 1, 9], 2) {
                for pad in sample(&[0, 3], 1) {
                    for shape in sample(&[0, 1, 2], 2) {
                        let mut m = vec![0; low_zeros];
                        m.extend(match shape {
                            0 => random_magnitude(&mut state, len),
                            1 => vec![0xFF; len],
                            _ => {
                                let mut p = vec![0; len];
                                p[len - 1] = 1;
                                p
                            }
                        });
                        m.resize(m.len() + pad, 0);
                        for negative in [false, true] {
                            let a = number(&m, negative, order);
                            let t = 8 * low_zeros as u64;
                            let top = 8 * m.len() as u64;
                            let mut indices = vec![0, 1, top, top + 1, top + 9, top + 100];
                            for edge in [t, top - 8, top - 1] {
                                indices.extend(edge.saturating_sub(2)..=edge + 2);
                            }
                            if cfg!(miri) {
                                indices.retain(|_| below(&mut state, 4) == 0);
                            }
                            for &i in &indices {
                                assert_set_bit(&a, i, true);
                                assert_set_bit(&a, i, false);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn set_bit_keeps_the_vector_when_it_can() {
    // Setting a bit inside a positive number writes one byte in place.
    let mut n = stored(&[1, 2, 3, 4], false, ByteOrder::BigEndian, 8);
    let pointer = n.value.as_ptr();
    let ((), allocations) = count(|| n.set_bit(4, true));
    assert_eq!(n.value, [1, 2, 3, 0x14]);
    assert_eq!(n.value.as_ptr(), pointer);
    assert_eq!(allocations, 0);
    // Above the top, the vector grows, here in its capacity; big-endian, the bytes move.
    let ((), allocations) = count(|| n.set_bit(32, true));
    assert_eq!(n.value, [1, 1, 2, 3, 0x14]);
    assert_eq!(n.value.as_ptr(), pointer);
    assert_eq!(allocations, 0);
    let ((), allocations) = count(|| n.set_bit(32, false));
    assert_eq!(n.value, [1, 2, 3, 0x14]);
    assert_eq!(allocations, 0);
    let mut n = stored(&[4, 3, 2, 1], false, ByteOrder::LittleEndian, 8);
    n.set_bit(63, true);
    assert_eq!(n.value, [4, 3, 2, 1, 0, 0, 0, 0x80]);
}

#[test]
fn set_bit_panics_beyond_any_vector() {
    for order in ORDERS {
        for negative in [false, true] {
            // Setting bit u64::MAX of a positive number, or clearing it in a negative one, asks for
            // a vector of 2^61 bytes. Under Miri, which would try to run the allocation, the panic
            // is not checked.
            if !cfg!(miri) {
                let mut n = Number::new_from_i32(if negative { -5 } else { 5 }, order);
                let result = catch_unwind(AssertUnwindSafe(|| n.set_bit(u64::MAX, !negative)));
                let message = result.expect_err("a vector of 2^61 bytes");
                let message = message
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| message.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                assert!(message.contains("cannot be allocated"), "{message}");
            }
            // The other way round there is nothing to do: the bit is already so.
            let mut n = Number::new_from_i32(if negative { -5 } else { 5 }, order);
            n.set_bit(u64::MAX, negative);
            assert_eq!(value_of(&n), expected_i128(if negative { -5 } else { 5 }));
        }
    }
}
