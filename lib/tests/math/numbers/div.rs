use a::math::numbers::{ByteOrder, Number};
use std::panic::{self, AssertUnwindSafe};

use super::common::*;
use crate::allocations::count;

// Expected quotients and remainders

/// The 32-bit limbs of a magnitude given least significant byte first, without the zero limbs
/// above the most significant one.
fn limbs32(magnitude: &[u8]) -> Vec<u32> {
    let mut limbs: Vec<u32> = magnitude
        .chunks(4)
        .map(|c| {
            let mut b = [0; 4];
            b[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(b)
        })
        .collect();
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
    limbs
}

/// The bytes of 32-bit limbs, least significant first.
fn bytes32(limbs: &[u32]) -> Vec<u8> {
    limbs.iter().flat_map(|l| l.to_le_bytes()).collect()
}

/// The quotient and the remainder of the magnitudes `x` and `y`, given least significant byte
/// first, `y` not zero, worked out independently of the implementation: Knuth's Algorithm D on
/// 32-bit limbs with `u64` arithmetic, as Hacker's Delight gives it, and the short division for
/// a divisor of one limb. The tests on small values check it against `i128` arithmetic.
fn divide_magnitudes(x: &[u8], y: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let (u, v) = (limbs32(x), limbs32(y));
    assert!(!v.is_empty(), "the reference divides by zero");
    if u.len() < v.len() {
        return (vec![], bytes32(&u));
    }
    if v.len() == 1 {
        let d = u64::from(v[0]);
        let mut q = vec![0; u.len()];
        let mut r = 0u64;
        for i in (0..u.len()).rev() {
            let t = (r << 32) | u64::from(u[i]);
            q[i] = (t / d) as u32;
            r = t % d;
        }
        return (bytes32(&q), bytes32(&[r as u32]));
    }
    let (m, n) = (u.len(), v.len());
    let s = v[n - 1].leading_zeros();
    let shl = |w: &[u32], extra: bool| {
        let mut out: Vec<u32> = (0..w.len())
            .map(|i| {
                let low = if i > 0 && s > 0 {
                    w[i - 1] >> (32 - s)
                } else {
                    0
                };
                (w[i] << s) | low
            })
            .collect();
        if extra {
            out.push(if s > 0 { w[w.len() - 1] >> (32 - s) } else { 0 });
        }
        out
    };
    let vn = shl(&v, false);
    let mut un = shl(&u, true);
    let mut q = vec![0u32; m - n + 1];
    let b = 1u64 << 32;
    for j in (0..=m - n).rev() {
        let num = (u64::from(un[j + n]) << 32) | u64::from(un[j + n - 1]);
        let mut qhat = num / u64::from(vn[n - 1]);
        let mut rhat = num % u64::from(vn[n - 1]);
        while qhat >= b || qhat * u64::from(vn[n - 2]) > (rhat << 32) + u64::from(un[j + n - 2]) {
            qhat -= 1;
            rhat += u64::from(vn[n - 1]);
            if rhat >= b {
                break;
            }
        }
        let mut borrow = 0i64;
        let mut carry = 0u64;
        for i in 0..n {
            let p = qhat * u64::from(vn[i]) + carry;
            carry = p >> 32;
            let t = i64::from(un[i + j]) - borrow - (p & 0xFFFF_FFFF) as i64;
            un[i + j] = t as u32;
            borrow = i64::from(t < 0);
        }
        let t = i64::from(un[j + n]) - borrow - carry as i64;
        un[j + n] = t as u32;
        if t < 0 {
            qhat -= 1;
            let mut carry = 0u64;
            for i in 0..n {
                let sum = u64::from(un[i + j]) + u64::from(vn[i]) + carry;
                un[i + j] = sum as u32;
                carry = sum >> 32;
            }
            un[j + n] = un[j + n].wrapping_add(carry as u32);
        }
        q[j] = qhat as u32;
    }
    let r: Vec<u32> = (0..n)
        .map(|i| {
            let high = if i + 1 < n && s > 0 {
                un[i + 1] << (32 - s)
            } else {
                0
            };
            (un[i] >> s) | high
        })
        .collect();
    (bytes32(&q), bytes32(&r))
}

/// `x - y` for magnitudes given least significant byte first, `x >= y`.
pub(super) fn sub_magnitudes(x: &[u8], y: &[u8]) -> Vec<u8> {
    let mut borrow = 0i16;
    let mut out: Vec<u8> = x
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            let t = i16::from(b) - i16::from(*y.get(i).unwrap_or(&0)) - borrow;
            borrow = i16::from(t < 0);
            t as u8
        })
        .collect();
    assert_eq!(borrow, 0, "the reference subtracted a larger magnitude");
    out.push(0);
    out
}

/// `x + 1` for a magnitude given least significant byte first.
fn plus_one(x: &[u8]) -> Vec<u8> {
    let mut out = x.to_vec();
    out.push(0);
    for b in &mut out {
        let (v, carry) = b.overflowing_add(1);
        *b = v;
        if !carry {
            break;
        }
    }
    out
}

/// The truncated quotient and remainder of `a` by `b`, as the operators must give them: the
/// quotient negative when exactly one operand is, the remainder with the sign of `a`.
pub(super) fn quotient_and_remainder(a: &Number, b: &Number) -> (Expected, Expected) {
    let (q, r) = divide_magnitudes(&le_bytes(a), &le_bytes(b));
    // A zero result is not negative: `expected` sees to it.
    (
        expected(&q, a.negative != b.negative),
        expected(&r, a.negative),
    )
}

/// Euclid's quotient and remainder of `a` by `b`: when the truncated remainder is below zero, the
/// quotient is one further from zero and the remainder `|b|` more.
pub(super) fn euclid(a: &Number, b: &Number) -> (Expected, Expected) {
    let (q, r) = quotient_and_remainder(a, b);
    if !r.negative {
        return (q, r);
    }
    let q = expected(&plus_one(&q.magnitude), a.negative != b.negative);
    let r = expected(&sub_magnitudes(&le_bytes(b), &r.magnitude), false);
    (q, r)
}

/// The product of the magnitudes `x` and `y`, given least significant byte first, for building
/// dividends: the schoolbook method on 32-bit limbs.
pub(super) fn mul_magnitudes(x: &[u8], y: &[u8]) -> Vec<u8> {
    let (x, y) = (limbs32(x), limbs32(y));
    let mut r = vec![0u32; x.len() + y.len() + 1];
    for (i, &xi) in x.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &yj) in y.iter().enumerate() {
            let t = u64::from(xi) * u64::from(yj) + u64::from(r[i + j]) + carry;
            r[i + j] = t as u32;
            carry = t >> 32;
        }
        r[i + y.len()] = carry as u32;
    }
    bytes32(&r)
}

/// `x + y` for magnitudes given least significant byte first.
pub(super) fn add_magnitudes(x: &[u8], y: &[u8]) -> Vec<u8> {
    let mut carry = 0u16;
    let mut out: Vec<u8> = (0..x.len().max(y.len()))
        .map(|i| {
            let t = u16::from(*x.get(i).unwrap_or(&0)) + u16::from(*y.get(i).unwrap_or(&0)) + carry;
            carry = t >> 8;
            t as u8
        })
        .collect();
    out.push(carry as u8);
    out
}

// The forms of division

/// The six ways to write a division or a remainder. Every form must give the same result, so the
/// tests run them all.
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

/// The two operations the forms write: the quotient with `/` and the remainder with `%`.
#[derive(Clone, Copy, PartialEq)]
enum Op {
    Div,
    Rem,
}

const OPS: [Op; 2] = [Op::Div, Op::Rem];

impl Form {
    /// The form as it is written, for `op`.
    fn name(self, op: Op) -> String {
        let o = if op == Op::Div { "/" } else { "%" };
        match self {
            Form::ValueValue => format!("a {o} b"),
            Form::ValueRef => format!("a {o} &b"),
            Form::RefValue => format!("&a {o} b"),
            Form::RefRef => format!("&a {o} &b"),
            Form::AssignValue => format!("a {o}= b"),
            Form::AssignRef => format!("a {o}= &b"),
        }
    }

    /// `a / b` or `a % b` in this form; an operand the form only borrows is dropped afterwards.
    fn apply(self, op: Op, a: Number, b: Number) -> Number {
        match (op, self) {
            (Op::Div, Form::ValueValue) => a / b,
            (Op::Div, Form::ValueRef) => a / &b,
            (Op::Div, Form::RefValue) => &a / b,
            (Op::Div, Form::RefRef) => &a / &b,
            (Op::Div, Form::AssignValue) => {
                let mut a = a;
                a /= b;
                a
            }
            (Op::Div, Form::AssignRef) => {
                let mut a = a;
                a /= &b;
                a
            }
            (Op::Rem, Form::ValueValue) => a % b,
            (Op::Rem, Form::ValueRef) => a % &b,
            (Op::Rem, Form::RefValue) => &a % b,
            (Op::Rem, Form::RefRef) => &a % &b,
            (Op::Rem, Form::AssignValue) => {
                let mut a = a;
                a %= b;
                a
            }
            (Op::Rem, Form::AssignRef) => {
                let mut a = a;
                a %= &b;
                a
            }
        }
    }
}

/// Checks every way of dividing `a` by `b` against the reference: both operations in every form,
/// with the vectors of the operands taken by value as tight as their values, with one byte to
/// spare and with plenty, and `div_rem`, `div_euclid` and `rem_euclid`. Under Miri, one form and
/// one setup picked at random for each operation.
fn assert_divides(a: &Number, b: &Number) {
    let (q, r) = quotient_and_remainder(a, b);
    let (eq, er) = euclid(a, b);
    let room = a.value.len() + b.value.len() + 1;
    let spares = [(0, 0), (1, 1), (room, room), (0, room)];
    for op in OPS {
        let expected = if op == Op::Div { &q } else { &r };
        let forms: &[Form] = if cfg!(miri) {
            let k = pick(FORMS.len());
            &FORMS[k..=k]
        } else {
            &FORMS
        };
        for &form in forms {
            let spares = if cfg!(miri) {
                let k = pick(spares.len());
                &spares[k..=k]
            } else {
                &spares[..]
            };
            for &(spare_a, spare_b) in spares {
                let result = form.apply(op, copy(a, spare_a), copy(b, spare_b));
                let inputs = || {
                    format!(
                        "{} with a = {}, b = {}, {spare_a} and {spare_b} bytes to spare",
                        form.name(op),
                        describe(a),
                        describe(b)
                    )
                };
                assert_result(&result, expected, a.byte_order, &inputs);
            }
        }
    }
    let inputs = |what: &str| format!("{what} with a = {}, b = {}", describe(a), describe(b));
    let (quotient, remainder) = a.div_rem(b);
    assert_result(&quotient, &q, a.byte_order, &|| {
        inputs("div_rem's quotient")
    });
    assert_result(&remainder, &r, a.byte_order, &|| {
        inputs("div_rem's remainder")
    });
    assert_result(&a.div_euclid(b), &eq, a.byte_order, &|| {
        inputs("div_euclid")
    });
    assert_result(&a.rem_euclid(b), &er, a.byte_order, &|| {
        inputs("rem_euclid")
    });
}

/// Checks `a / b` and `a % b` against the reference in one form each, picked at random, and the
/// Euclidean pair: for the tests that check many divisions.
fn assert_divides_quickly(a: &Number, b: &Number) {
    let (q, r) = quotient_and_remainder(a, b);
    let (eq, er) = euclid(a, b);
    for (op, expected) in [(Op::Div, &q), (Op::Rem, &r)] {
        let form = FORMS[pick(FORMS.len())];
        let result = form.apply(op, copy(a, pick(3)), copy(b, pick(3)));
        let inputs = || {
            format!(
                "{} with a = {}, b = {}",
                form.name(op),
                describe(a),
                describe(b)
            )
        };
        assert_result(&result, expected, a.byte_order, &inputs);
    }
    let inputs = |what: &str| format!("{what} with a = {}, b = {}", describe(a), describe(b));
    assert_result(&a.div_euclid(b), &eq, a.byte_order, &|| {
        inputs("div_euclid")
    });
    assert_result(&a.rem_euclid(b), &er, a.byte_order, &|| {
        inputs("rem_euclid")
    });
}

/// A random `i128` of a random bit length below 127, with every bit set, a power of two, or
/// random bits, and a random sign; never zero.
fn random_nonzero_i128(state: &mut u64) -> i128 {
    loop {
        let x = random_i128(state);
        if x != 0 {
            return x;
        }
    }
}

// Small values against `i128` arithmetic

#[test]
fn the_reference_divides_as_i128() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..size(20_000, 300) {
        let (x, y) = (random_i128(&mut state), random_nonzero_i128(&mut state));
        let a = number(
            &x.unsigned_abs().to_le_bytes(),
            x < 0,
            ByteOrder::LittleEndian,
        );
        let b = number(&y.unsigned_abs().to_le_bytes(), y < 0, ByteOrder::BigEndian);
        let (q, r) = quotient_and_remainder(&a, &b);
        assert_eq!(q, expected_i128(x / y), "{x} / {y}");
        assert_eq!(r, expected_i128(x % y), "{x} % {y}");
        let (q, r) = euclid(&a, &b);
        assert_eq!(q, expected_i128(x.div_euclid(y)), "{x} div_euclid {y}");
        assert_eq!(r, expected_i128(x.rem_euclid(y)), "{x} rem_euclid {y}");
    }
}

#[test]
fn divides_small_values_as_i128() {
    let mut state = 0x3C6E_F372_FE94_F82B;
    for _ in 0..size(3000, 40) {
        let (x, y) = (random_i128(&mut state), random_nonzero_i128(&mut state));
        let (order_a, order_b) = ORDER_PAIRS[below(&mut state, 4)];
        let a = number(&x.unsigned_abs().to_le_bytes(), x < 0, order_a);
        let b = number(&y.unsigned_abs().to_le_bytes(), y < 0, order_b);
        let a = Number::new_signed(a.value, a.negative, order_a);
        let b = Number::new_signed(b.value, b.negative, order_b);
        assert_divides(&a, &b);
    }
}

#[test]
fn divides_every_pair_of_small_integers() {
    // Every dividend from -300 to 300 by every divisor from -40 to 40 but zero, and the values
    // around the byte boundaries, in every pair of byte orders.
    let mut values: Vec<i128> = (-300..=300).collect();
    for k in [8, 16, 24, 32, 56, 63, 64, 65, 96, 120, 125] {
        values.extend(
            [1 << k, (1 << k) - 1, (1 << k) + 1]
                .map(|v: i128| [v, -v])
                .concat(),
        );
    }
    let divisors: Vec<i128> = (-40..=40)
        .filter(|&d| d != 0)
        .chain(values.iter().copied().filter(|&v| v.unsigned_abs() > 40))
        .collect();
    let step = size(1, 37);
    for (i, &x) in values.iter().enumerate().step_by(step) {
        for (j, &y) in divisors.iter().enumerate().step_by(step) {
            let (oa, ob) = ORDER_PAIRS[(i + j) % 4];
            let a = Number::new_signed(
                x.unsigned_abs().to_le_bytes().to_vec(),
                x < 0,
                ByteOrder::LittleEndian,
            );
            let a = number(&le_bytes(&a), a.negative, oa);
            let b = Number::new_signed(
                y.unsigned_abs().to_le_bytes().to_vec(),
                y < 0,
                ByteOrder::LittleEndian,
            );
            let b = number(&le_bytes(&b), b.negative, ob);
            let inputs = || format!("{x} by {y}");
            for op in OPS {
                let form = FORMS[(i + 3 * j + op as usize) % FORMS.len()];
                let result = form.apply(op, copy(&a, 0), copy(&b, 0));
                let want = if op == Op::Div { x / y } else { x % y };
                assert_result(&result, &expected_i128(want), oa, &|| {
                    format!("{}: {}", form.name(op), inputs())
                });
            }
            assert_result(
                &a.div_euclid(&b),
                &expected_i128(x.div_euclid(y)),
                oa,
                &|| format!("div_euclid: {}", inputs()),
            );
            assert_result(
                &a.rem_euclid(&b),
                &expected_i128(x.rem_euclid(y)),
                oa,
                &|| format!("rem_euclid: {}", inputs()),
            );
        }
    }
}

#[test]
fn divides_at_the_limits_of_the_register_paths() {
    // Dividends around 16, 32 and 64 bytes and divisors around 8 and 16, where the short path in
    // registers, the paths of four and eight limbs and the general path meet: all ones, a top bit
    // alone, and random bytes.
    let mut state = 0xBB67_AE85_84CA_A73B;
    let dividends = [
        1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 40, 48, 63, 64, 65, 72,
    ];
    let divisors = [1, 2, 7, 8, 9, 15, 16, 17, 24, 32, 33];
    for &la in &dividends {
        for &lb in &divisors {
            for kind in 0..3 {
                let make = |state: &mut u64, len: usize| match kind {
                    0 => vec![0xFF; len],
                    1 => {
                        let mut v = vec![0; len];
                        v[len - 1] = 0x80;
                        v
                    }
                    _ => random_magnitude(state, len),
                };
                let (x, y) = (make(&mut state, la), make(&mut state, lb));
                let (oa, ob) = ORDER_PAIRS[below(&mut state, 4)];
                let (na, nb) = SIGN_PAIRS[below(&mut state, 4)];
                let a = number(&x, na, oa);
                let b = number(&y, nb, ob);
                if cfg!(miri) {
                    assert_divides_quickly(&a, &b);
                } else {
                    assert_divides(&a, &b);
                }
            }
        }
    }
}

// Long values against the reference

#[test]
fn divides_random_numbers_as_the_reference() {
    let mut state = 0xA54F_F53A_5F1D_36F1;
    for _ in 0..size(2000, 30) {
        let a = random_operand(&mut state, size(600, 80));
        let mut b = random_operand(&mut state, size(300, 40));
        if le_bytes(&b).iter().all(|&x| x == 0) {
            b = number(&[7], b.negative, b.byte_order);
        }
        assert_divides_quickly(&a, &b);
    }
}

#[test]
fn divides_every_pair_of_lengths_up_to_80_bytes() {
    let mut state = 0x510E_527F_ADE6_82D1;
    let step = size(1, 9);
    for la in (0..=80).step_by(step) {
        for lb in (1..=40).step_by(step) {
            let a = number(
                &random_magnitude(&mut state, la),
                below(&mut state, 2) == 0,
                ORDERS[below(&mut state, 2)],
            );
            let b = number(
                &random_magnitude(&mut state, lb),
                below(&mut state, 2) == 0,
                ORDERS[below(&mut state, 2)],
            );
            assert_divides_quickly(&a, &b);
        }
    }
}

/// Shapes of long divisions, as limbs of the dividend and of the divisor: around one limb and the
/// 20 from which a division by one limb folds the limbs in; two limbs; the schoolbook method
/// from three; and divide and conquer from a divisor of 40 limbs, with quotients shorter than the
/// divisor, as long, and longer, whose first block is of one limb, two, or more. Under Miri,
/// which runs the portable kernel, the smaller ones only.
fn division_shapes() -> Vec<(usize, usize)> {
    let mut shapes = vec![];
    for n in [1, 2, 3, 4, 7, 8, 9, 31, 32, 33] {
        for m in [n, n + 1, n + 2, n + 3, 2 * n, 3 * n + 5] {
            shapes.push((m, n));
        }
    }
    for m in [18, 19, 20, 21, 40, 100] {
        shapes.push((m, 1));
    }
    for n in [39, 40, 41, 64, 100, 151] {
        for m in [
            n + 2,
            n + 3,
            n + 4,
            n + n / 2,
            2 * n - 1,
            2 * n,
            2 * n + 1,
            2 * n + 2,
            3 * n,
            4 * n + 7,
        ] {
            shapes.push((m, n));
        }
    }
    for n in [255, 256, 300] {
        for m in [n + 3, 2 * n, 3 * n] {
            shapes.push((m, n));
        }
    }
    shapes
        .into_iter()
        .filter(|&(m, _)| m <= size(usize::MAX, 70))
        .collect()
}

#[test]
fn divides_every_shape_of_long_operands_as_the_reference() {
    let mut state = 0x9B05_688C_2B3E_6C1F;
    for (m, n) in division_shapes() {
        for extra in [0, 5] {
            let x = limb_runs(&mut state, m, extra);
            let y = limb_runs(&mut state, n, if extra == 0 { 0 } else { 3 });
            let (oa, ob) = ORDER_PAIRS[below(&mut state, 4)];
            let (na, nb) = SIGN_PAIRS[below(&mut state, 4)];
            assert_divides_quickly(&number(&x, na, oa), &number(&y, nb, ob));
        }
    }
}

#[test]
fn quotients_that_need_correcting_come_out_right() {
    // Dividends built as `q b + r` with remainders of 0, 1, `b - 1` and about half `b`, by
    // divisors of all ones, of a top bit alone, and random, and quotients of all ones, which make
    // the estimates of the quotient limbs too large, the top limbs of a partial remainder equal
    // to those of the divisor, and the corrections of divide and conquer.
    let mut state = 0x1F83_D9AB_FB41_BD6B;
    let lengths = [
        1, 2, 3, 7, 8, 9, 16, 17, 24, 25, 100, 160, 399, 400, 401, 800,
    ];
    for &lb in lengths.iter().filter(|&&l| l <= size(usize::MAX, 100)) {
        for &lq in lengths.iter().filter(|&&l| l <= size(usize::MAX, 30)) {
            for kind in 0..3 {
                let y = match kind {
                    0 => vec![0xFF; lb],
                    1 => {
                        let mut v = vec![0; lb];
                        v[lb - 1] = 0x80;
                        v
                    }
                    _ => random_magnitude(&mut state, lb),
                };
                let q = if below(&mut state, 2) == 0 {
                    vec![0xFF; lq]
                } else {
                    random_magnitude(&mut state, lq)
                };
                let half: Vec<u8> = y[1..].to_vec();
                for r in [vec![], vec![1], sub_magnitudes(&y, &[1]), half] {
                    let x = add_magnitudes(&mul_magnitudes(&q, &y), &r);
                    let (oa, ob) = ORDER_PAIRS[below(&mut state, 4)];
                    let (na, nb) = SIGN_PAIRS[below(&mut state, 4)];
                    let (a, b) = (number(&x, na, oa), number(&y, nb, ob));
                    assert_divides_quickly(&a, &b);
                }
            }
        }
    }
}

#[test]
fn quotients_from_a_shrinking_divisor_come_out_right() {
    // The last limbs of a quotient alone come from a divisor that drops a low limb at each step:
    // exact multiples and the dividends just below or above them make the dropped products
    // decide the quotient, to be checked and made one smaller; `B^(n - 2) (q d - 1) + z`, for a
    // divisor of `n` limbs, leaves a partial remainder too large for the shortened divisor, and
    // the quotient is then computed in full. Divisors of 3 to 13 limbs, random, with their low
    // limbs all ones or zeros, normalized or not, by quotients of 1 to `n + 2` limbs; and of 40
    // to 100 limbs, whose quotient's top limbs come from divide and conquer.
    let mut state = 0x5DEE_CE66_D1CE_4E5B;
    let ones = |n: usize| vec![0xFF; n];
    let long = if cfg!(miri) {
        &[][..]
    } else {
        &[40, 41, 100][..]
    };
    for n in (3..=size(13, 5)).chain(long.iter().copied()) {
        let quotients: Vec<usize> = if n <= 13 {
            (1..=n + 2).collect()
        } else {
            vec![1, 2, 4, 9, n - 1, n + 2]
        };
        for lq in quotients {
            for kind in 0..5 {
                let mut y = random_magnitude(&mut state, 8 * n);
                match kind {
                    1 => y[..8 * (n - 2)].fill(0xFF),
                    2 => y[..8 * (n - 2)].fill(0),
                    3 => y[..8 * (n - 1)].fill(0xFF),
                    _ => {}
                }
                if kind != 4 {
                    y[8 * n - 1] |= 0x80;
                }
                let q = if lq % 2 == 0 {
                    ones(8 * lq)
                } else {
                    random_magnitude(&mut state, 8 * lq)
                };
                let qy = mul_magnitudes(&q, &y);
                let below_qy = sub_magnitudes(&qy, &[1]);
                let mut shifted = vec![0; 8 * (n - 2)];
                shifted.extend_from_slice(&below_qy);
                let z = random_magnitude(&mut state, 8 * (n - 2));
                for x in [
                    qy.clone(),
                    below_qy.clone(),
                    add_magnitudes(&qy, &sub_magnitudes(&y, &[1])),
                    sub_magnitudes(&qy, &y),
                    shifted.clone(),
                    add_magnitudes(&shifted, &z),
                ] {
                    let (oa, ob) = ORDER_PAIRS[below(&mut state, 4)];
                    let (na, nb) = SIGN_PAIRS[below(&mut state, 4)];
                    assert_divides_quickly(&number(&x, na, oa), &number(&y, nb, ob));
                }
            }
        }
    }
}

#[test]
fn short_quotients_from_the_top_limbs_come_out_right() {
    // A quotient of `k` limbs alone by a divisor of `n >= 2k + 6` limbs comes from the top `2k + 2`
    // limbs of the dividend, with a zero limb above them, by the top `k + 1` of the divisor `t`,
    // with a shrinking divisor: top limbs `w = B^(k - 1) (q t - 1) + z` for a limb `q` make it
    // give up, and they are divided in full; `w = B^(k - 1) q t` and random limbs below make the
    // dropped products decide the quotient.
    let mut state = 0x2F1C_5B7E_A3D9_0461;
    for k in 2..=size(9, 3) {
        for n in [2 * k + 6, 2 * k + 9] {
            for kind in 0..3 {
                let mut y = random_magnitude(&mut state, 8 * n);
                y[8 * n - 1] |= 0x80;
                let top = &y[8 * (n - k - 1)..];
                let q = random_magnitude(&mut state, 8);
                let qt = mul_magnitudes(&q, top);
                let w = match kind {
                    0 => sub_magnitudes(&qt, &[1]),
                    _ => qt,
                };
                let mut x = random_bytes(&mut state, 8 * (n - k - 2));
                x.extend_from_slice(&[0; 8].repeat(k - 1));
                if kind != 2 {
                    let z = random_bytes(&mut state, 8 * (k - 1));
                    let len = x.len();
                    x[len - 8 * (k - 1)..].copy_from_slice(&z);
                }
                x.extend_from_slice(&w);
                let (oa, ob) = ORDER_PAIRS[below(&mut state, 4)];
                let (na, nb) = SIGN_PAIRS[below(&mut state, 4)];
                assert_divides_quickly(&number(&x, na, oa), &number(&y, nb, ob));
            }
        }
    }
}

#[test]
fn divides_by_one_limb_through_the_folded_steps() {
    // Divisors of one limb, from 1 to all ones, and around the powers of two at which the
    // normalization shift changes, by dividends long enough to fold their limbs in, whose sums
    // carry out of two limbs as often as all ones make them.
    let mut state = 0x5BE0_CD19_137E_2179;
    let divisors: [u64; 10] = [
        1,
        2,
        3,
        10_000_000_000_000_000_000,
        (1 << 63) - 1,
        1 << 63,
        (1 << 63) + 1,
        u64::MAX - 1,
        u64::MAX,
        0x8000_0000_0000_0001,
    ];
    for d in divisors {
        for len in [19 * 8, 20 * 8, 20 * 8 + 3, size(400, 24 * 8)] {
            for kind in 0..2 {
                let x = if kind == 0 {
                    vec![0xFF; len]
                } else {
                    random_magnitude(&mut state, len)
                };
                let (oa, ob) = ORDER_PAIRS[below(&mut state, 4)];
                let a = number(&x, below(&mut state, 2) == 0, oa);
                let b = Number::new(d.to_le_bytes().to_vec(), ByteOrder::LittleEndian);
                let b = number(&le_bytes(&b), below(&mut state, 2) == 0, ob);
                assert_divides_quickly(&a, &b);
            }
        }
    }
}

// Zero, non-shortest operands and division by zero

#[test]
fn zero_divided_by_anything_is_zero() {
    for order in ORDERS {
        for a in zeros(order) {
            for b in [
                number(&[3], true, ByteOrder::BigEndian),
                number(&[0xFF; 40], false, ByteOrder::LittleEndian),
            ] {
                assert_divides(&a, &b);
            }
        }
    }
}

#[test]
fn operands_with_zero_bytes_above_their_magnitude_divide_as_their_values() {
    let mut state = 0x6A09_E667_F3BC_C908;
    for _ in 0..size(300, 10) {
        let la = below(&mut state, 90);
        let lb = 1 + below(&mut state, 40);
        let mut x = random_magnitude(&mut state, la);
        let mut y = random_magnitude(&mut state, lb);
        x.resize(la + below(&mut state, 20), 0);
        y.resize(lb + below(&mut state, 20), 0);
        let (oa, ob) = ORDER_PAIRS[below(&mut state, 4)];
        let (na, nb) = SIGN_PAIRS[below(&mut state, 4)];
        assert_divides_quickly(&number(&x, na, oa), &number(&y, nb, ob));
    }
}

/// The panic message of `f`, which must panic.
pub(super) fn panic_message(f: impl FnOnce()) -> String {
    let payload = panic::catch_unwind(AssertUnwindSafe(f)).expect_err("no panic");
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        String::new()
    }
}

#[test]
fn dividing_by_zero_panics_as_for_primitive_integers() {
    // In every form, by every way of holding zero, short and long dividends, with the messages of
    // `i32`.
    let dividends = [
        number(&[7], false, ByteOrder::BigEndian),
        number(&[0xFF; 20], true, ByteOrder::LittleEndian),
        number(&[0xFF; 100], false, ByteOrder::BigEndian),
    ];
    let mut zeros_: Vec<Number> = zeros(ByteOrder::BigEndian);
    zeros_.extend(zeros(ByteOrder::LittleEndian));
    zeros_.push(number(&[0; 40], true, ByteOrder::BigEndian));
    for a in &dividends {
        for z in &zeros_ {
            let forms: &[Form] = if cfg!(miri) { &FORMS[..1] } else { &FORMS };
            for &form in forms {
                for op in OPS {
                    let message = panic_message(|| {
                        form.apply(op, copy(a, 0), copy(z, 0));
                    });
                    let want = if op == Op::Div {
                        "attempt to divide by zero"
                    } else {
                        "attempt to calculate the remainder with a divisor of zero"
                    };
                    assert_eq!(message, want, "{} by {}", form.name(op), describe(z));
                }
            }
            assert_eq!(
                panic_message(|| drop(a.div_rem(z))),
                "attempt to divide by zero"
            );
            assert_eq!(
                panic_message(|| drop(a.div_euclid(z))),
                "attempt to divide by zero"
            );
            assert_eq!(
                panic_message(|| drop(a.rem_euclid(z))),
                "attempt to calculate the remainder with a divisor of zero"
            );
        }
    }
}

// Signs, byte orders and Euclid's division

#[test]
fn signs_follow_the_rules_of_primitive_integers() {
    // ±7 by ±2 and ±0x0123…(20 bytes) by ±0x45…(9 bytes): truncated toward zero, the remainder
    // with the sign of the dividend, Euclid's remainder never negative.
    for (x, y) in [(7i128, 2i128), (6, 3), (1, 7)] {
        for (na, nb) in SIGN_PAIRS {
            let (x, y) = (if na { -x } else { x }, if nb { -y } else { y });
            let a = Number::new_from_i128(x, ByteOrder::BigEndian);
            let b = Number::new_from_i128(y, ByteOrder::LittleEndian);
            let o = ByteOrder::BigEndian;
            assert_result(&(&a / &b), &expected_i128(x / y), o, &|| {
                format!("{x} / {y}")
            });
            assert_result(&(&a % &b), &expected_i128(x % y), o, &|| {
                format!("{x} % {y}")
            });
            assert_result(
                &a.div_euclid(&b),
                &expected_i128(x.div_euclid(y)),
                o,
                &|| format!("{x} div_euclid {y}"),
            );
            assert_result(
                &a.rem_euclid(&b),
                &expected_i128(x.rem_euclid(y)),
                o,
                &|| format!("{x} rem_euclid {y}"),
            );
        }
    }
    let x: Vec<u8> = (1..=20).collect();
    let y: Vec<u8> = (0x45..0x4E).collect();
    for (na, nb) in SIGN_PAIRS {
        for (oa, ob) in ORDER_PAIRS {
            assert_divides(&number(&x, na, oa), &number(&y, nb, ob));
        }
    }
}

#[test]
fn euclid_s_division_puts_the_dividend_between_multiples_of_the_divisor() {
    // `a = b q + r` with `0 <= r < |b|`, for long operands, checked with the crate's own
    // multiplication and addition.
    let mut state = 0xCBBB_9D5D_C105_9ED8;
    for _ in 0..size(200, 8) {
        let a = random_operand(&mut state, size(500, 60));
        let mut b = random_operand(&mut state, size(200, 30));
        if le_bytes(&b).iter().all(|&x| x == 0) {
            b = number(&[1, 2, 3], true, b.byte_order);
        }
        let (q, r) = (a.div_euclid(&b), a.rem_euclid(&b));
        assert!(!r.is_negative(), "a negative Euclidean remainder");
        let back = &(&b * &q) + &r;
        assert_eq!(value_of(&back), value_of(&a), "b q + r is not a");
        let below = &r - &Number::new(le_bytes(&b), ByteOrder::LittleEndian);
        assert!(
            below.is_negative(),
            "the remainder is not below |b| for a = {}, b = {}",
            describe(&a),
            describe(&b)
        );
    }
}

#[test]
fn mixed_byte_orders_in_every_form() {
    let x: Vec<u8> = (1..=70).map(|i| i * 3).collect();
    for lb in [5, 12, 30] {
        let y: Vec<u8> = (0..lb).map(|i| 0xF0 - i as u8).collect();
        for (oa, ob) in ORDER_PAIRS {
            assert_divides(&number(&x, false, oa), &number(&y, true, ob));
        }
    }
}

#[test]
fn borrowed_operands_are_left_as_they_were() {
    let a = stored(
        &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 0,
        ],
        true,
        ByteOrder::BigEndian,
        30,
    );
    let b = stored(&[0, 0, 9, 8, 7], false, ByteOrder::LittleEndian, 9);
    let (a0, b0) = (copy_with_capacity(&a), copy_with_capacity(&b));
    let _ = &a / &b;
    let _ = &a % &b;
    let _ = a.div_rem(&b);
    let _ = a.div_euclid(&b);
    let _ = a.rem_euclid(&b);
    let _ = copy(&a, 0) / &b;
    let _ = &a % copy(&b, 0);
    assert!(same(&a, &a0) && a.value.capacity() == a0.value.capacity());
    assert!(same(&b, &b0) && b.value.capacity() == b0.value.capacity());
}

// Allocation

/// Checks where `op` in `form` puts its result, with the operands in vectors of the capacities of
/// `a` and `b`, and that nothing else is allocated: as for the other operators, the forms that
/// take `a` by value build it in the vector of `a` whenever that has room, and `&a / b` in that of
/// `b` whenever that has room; when only `b` has room, `a / b` and `a /= b` build it there too.
/// Otherwise the result gets one new vector, as it always does from `&a / &b`.
fn assert_placement(form: Form, op: Op, a: &Number, b: &Number) {
    let (q, r) = quotient_and_remainder(a, b);
    let expected = if op == Op::Div { q } else { r };
    let len = expected.magnitude.len();
    let left = copy_with_capacity(a);
    let right = copy_with_capacity(b);
    let left_has_room = left.value.capacity() >= len;
    let right_has_room = right.value.capacity() >= len;
    let (left_pointer, right_pointer) = (left.value.as_ptr(), right.value.as_ptr());
    let (result, allocations) = count(|| form.apply(op, left, right));
    let inputs = || {
        format!(
            "{} with a = {} (capacity {}), b = {} (capacity {})",
            form.name(op),
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

/// Where the allocation promises put the result of `form`, given which operands have room for it.
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
    let mut state = 0x2545_F491_4F6C_DD1D;
    for round in 0..size(1000, 30) {
        let max = if round % 3 == 0 { 300 } else { 40 };
        let a = random_operand(&mut state, max);
        let b = random_operand(&mut state, max / 2);
        if le_bytes(&b).iter().all(|&x| x == 0) || a.value.len() + b.value.len() > 1000 {
            continue;
        }
        let (q, r) = quotient_and_remainder(&a, &b);
        let len = q.magnitude.len().max(r.magnitude.len());
        let mut capacity = |n: &Number| {
            let capacity = match below(&mut state, 4) {
                0 => len.saturating_sub(1),
                1 => len,
                2 => 0,
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
            for op in OPS {
                assert_placement(form, op, &a, &b);
            }
        }
    }
}

#[test]
fn the_dividend_s_vector_always_has_room_but_when_empty() {
    // A quotient or a remainder is never longer than the dividend, so `a /= b` and `a %= b` keep
    // `a`'s vector; an empty vector, which is zero, has no room for the zero byte of the result.
    for order in ORDERS {
        for len in [1, 9, 17, 40, 100] {
            for op in OPS {
                for form in [
                    Form::AssignRef,
                    Form::AssignValue,
                    Form::ValueRef,
                    Form::ValueValue,
                ] {
                    let a = plus(&vec![0xAB; len], order, len);
                    let b = number(&[3], true, ByteOrder::LittleEndian);
                    let pointer = a.value.as_ptr();
                    let (result, allocations) = count(|| form.apply(op, a, b));
                    let inputs = || format!("{} with {len} bytes", form.name(op));
                    assert_eq!(result.value.as_ptr(), pointer, "{}", inputs());
                    assert_eq!(allocations, 0, "{}", inputs());
                }
            }
        }
        let empty = plus(&[], order, 0);
        let five = number(&[5], false, order);
        let (result, allocations) = count(|| empty / &five);
        assert_eq!(result.value, [0]);
        assert_eq!(allocations, 1);
    }
}

#[test]
fn long_divisions_allocate_one_temporary_buffer_at_most() {
    // Past the working memory on the stack, one buffer is allocated for the computation and freed
    // before the result is returned.
    let mut state = 0x6A09_E667_BB67_AE85;
    // Working memory of more than 1024 limbs: under Miri, 525 limbs by 100.
    let (la, lb) = (size(12_000, 4_200), size(5_000, 800));
    let a = number(
        &random_magnitude(&mut state, la),
        false,
        ByteOrder::LittleEndian,
    );
    let b = number(
        &random_magnitude(&mut state, lb),
        true,
        ByteOrder::BigEndian,
    );
    let (q, r) = quotient_and_remainder(&a, &b);
    let (result, allocations) = count(|| &a / &b);
    assert_result(&result, &q, a.byte_order, &|| "&a / &b".into());
    assert_eq!(allocations, 2, "&a / &b");
    let roomy = copy(&a, 0);
    let pointer = roomy.value.as_ptr();
    let (result, allocations) = count(|| roomy % &b);
    assert_result(&result, &r, a.byte_order, &|| "a % &b".into());
    assert_eq!(result.value.as_ptr(), pointer, "a % &b");
    assert_eq!(allocations, 1, "a % &b");
    let ((quotient, remainder), allocations) = count(|| a.div_rem(&b));
    assert_result(&quotient, &q, a.byte_order, &|| "div_rem".into());
    assert_result(&remainder, &r, a.byte_order, &|| "div_rem".into());
    assert_eq!(allocations, 3, "div_rem");
}
