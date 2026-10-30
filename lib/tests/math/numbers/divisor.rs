use a::math::numbers::{ByteOrder, Divisor, Number};

use super::common::*;
use super::div::{
    MU_PREPARED, euclid, mul_magnitudes, newton_divisor, panic_message, quotient_and_remainder,
    sub_magnitudes,
};
use crate::allocations::count;

/// A form of division by a divisor that takes its dividend by value.
type Run = fn(Number, &Divisor) -> Number;

/// Checks every way of dividing `a` by the divisor `d` made from `b` against the reference: `/`,
/// `/=`, `%` and `%=` with `a` owned and borrowed, `div_rem`, `div_euclid` and `rem_euclid`; with
/// `a`'s vector tight or with room to spare. Under Miri, one form of each operation picked at
/// random.
fn assert_divides_by(a: &Number, b: &Number, d: &Divisor) {
    let (q, r) = quotient_and_remainder(a, b);
    let (eq, er) = euclid(a, b);
    let order = a.byte_order;
    let inputs = |what: &str| format!("{what} with a = {}, d = {}", describe(a), describe(b));
    let runs: [(&str, &dyn Fn() -> Number, &Expected); 6] = [
        ("&a / &d", &|| &copy(a, 0) / d, &q),
        ("a / &d", &|| copy(a, pick(3)) / d, &q),
        (
            "a /= &d",
            &|| {
                let mut n = copy(a, pick(3));
                n /= d;
                n
            },
            &q,
        ),
        ("&a % &d", &|| &copy(a, 0) % d, &r),
        ("a % &d", &|| copy(a, pick(3)) % d, &r),
        (
            "a %= &d",
            &|| {
                let mut n = copy(a, pick(3));
                n %= d;
                n
            },
            &r,
        ),
    ];
    for (k, (name, run, expected)) in runs.iter().enumerate() {
        if cfg!(miri) && k % 3 != pick(3) {
            continue;
        }
        assert_result(&run(), expected, order, &|| inputs(name));
    }
    let (quotient, remainder) = d.div_rem(a);
    assert_result(&quotient, &q, order, &|| inputs("div_rem's quotient"));
    assert_result(&remainder, &r, order, &|| inputs("div_rem's remainder"));
    assert_result(&d.div_euclid(a), &eq, order, &|| inputs("div_euclid"));
    assert_result(&d.rem_euclid(a), &er, order, &|| inputs("rem_euclid"));
}

#[test]
fn divides_as_the_number_it_was_made_from() {
    let mut state = 0x243F_6A88_85A3_08D3;
    for _ in 0..size(1500, 25) {
        let a = random_operand(&mut state, size(500, 70));
        let mut b = random_operand(&mut state, size(250, 35));
        if le_bytes(&b).iter().all(|&x| x == 0) {
            b = number(&[9], b.negative, b.byte_order);
        }
        assert_divides_by(&a, &b, &Divisor::new(&b));
    }
}

#[test]
fn one_divisor_divides_many_numbers() {
    // Divisors of one limb, two, three, four, a few and many, each for dividends of every length
    // up to a few hundred bytes: the paths in registers, the general one, divide and conquer and
    // the quotient from the top limbs.
    let mut state = 0x1319_8A2E_0370_7344;
    for lb in [1, 3, 8, 9, 15, 16, 17, 24, 25, 32, 40, 100, 330, 400] {
        let b = number(
            &random_magnitude(&mut state, lb),
            below(&mut state, 2) == 0,
            ORDERS[below(&mut state, 2)],
        );
        let d = Divisor::new(&b);
        let step = size(1, 23);
        for la in (0..size(900, 120)).step_by(step) {
            let a = number(
                &random_magnitude(&mut state, la),
                below(&mut state, 2) == 0,
                ORDERS[below(&mut state, 2)],
            );
            assert_divides_by(&a, &b, &d);
        }
    }
}

#[test]
fn divisors_of_every_shift_and_form() {
    // A top byte of every bit length, so every normalizing shift; divisors not in the shortest
    // form and negative; a clone of the divisor dividing as the original does.
    let mut state = 0xA409_3822_299F_31D0;
    for lb in [1, 2, 7, 8, 9, 16, 17, 24, 25, 32, 33] {
        for top_bits in 1..=8 {
            let mut y = random_magnitude(&mut state, lb);
            y[lb - 1] =
                (1u16 << (top_bits - 1)) as u8 | (y[lb - 1] & ((1 << (top_bits - 1)) - 1) as u8);
            let mut padded = y.clone();
            padded.resize(lb + below(&mut state, 9), 0);
            let b = number(
                &padded,
                below(&mut state, 2) == 0,
                ORDERS[below(&mut state, 2)],
            );
            let d = Divisor::new(&b).clone();
            for la in [0, 1, 8, 16, 17, 24, 32, 40, 48, 64, 65, size(200, 80)] {
                let a = number(
                    &random_magnitude(&mut state, la),
                    below(&mut state, 2) == 0,
                    ORDERS[below(&mut state, 2)],
                );
                assert_divides_by(&a, &b, &d);
            }
        }
    }
}

#[test]
fn a_zero_divisor_panics_when_it_is_made() {
    let mut zeros_ = zeros(ByteOrder::BigEndian);
    zeros_.extend(zeros(ByteOrder::LittleEndian));
    for z in &zeros_ {
        assert_eq!(
            panic_message(|| drop(Divisor::new(z))),
            "attempt to divide by zero",
            "{}",
            describe(z)
        );
    }
}

#[test]
fn owned_dividends_keep_their_vector() {
    // `a / &d`, `a /= &d`, `a % &d` and `a %= &d` build the result in `a`'s vector and allocate
    // nothing, in registers and on the general path; `&a / &d` allocates one vector.
    let mut state = 0xBE54_66CF_34E9_0C6C;
    for (la, lb) in [(10, 3), (40, 12), (64, 16), (100, 8), (300, 40), (300, 200)] {
        let b = number(
            &random_magnitude(&mut state, lb),
            true,
            ByteOrder::BigEndian,
        );
        let d = Divisor::new(&b);
        for order in ORDERS {
            let a = number(&random_magnitude(&mut state, la), false, order);
            let (q, r) = quotient_and_remainder(&a, &b);
            let runs: [(&str, Run, &Expected); 4] = [
                ("a / &d", |a, d| a / d, &q),
                (
                    "a /= &d",
                    |mut a, d| {
                        a /= d;
                        a
                    },
                    &q,
                ),
                ("a % &d", |a, d| a % d, &r),
                (
                    "a %= &d",
                    |mut a, d| {
                        a %= d;
                        a
                    },
                    &r,
                ),
            ];
            for (name, run, expected) in runs {
                let owned = copy(&a, 0);
                let pointer = owned.value.as_ptr();
                let (result, allocations) = count(|| run(owned, &d));
                let inputs = || format!("{name} with {la} bytes by {lb}");
                assert_result(&result, expected, order, &inputs);
                assert_eq!(result.value.as_ptr(), pointer, "{}", inputs());
                assert_eq!(allocations, 0, "{}", inputs());
            }
            let (result, allocations) = count(|| &a / &d);
            assert_result(&result, &q, order, &|| "&a / &d".into());
            assert_eq!(allocations, 1, "&a / &d");
        }
    }
}

#[test]
fn a_prepared_divisor_divides_numbers_of_every_length_by_newton_s_method() {
    // From its first division by Newton's method, a divisor keeps the inverse of all its limbs and
    // the transforms: quotients of whole blocks of its length; of a block and one limb, two, or
    // up to a quarter of a block, whose last limbs the schoolbook method or divide and conquer
    // computes; of a block and a quarter or more, a last block of its own; shorter than two
    // thirds of a block, from the top limbs of the inverse. Exact multiples and their neighbours
    // make the guard limb of a quotient alone ask for the remainder.
    let mut state = 0x1F83_D9AB_5BE0_CD19;
    let n = if cfg!(miri) { MU_PREPARED + 1 } else { 400 };
    for kind in 0..size(3, 1) {
        let y = newton_divisor(&mut state, n, kind);
        let b = number(&y, kind == 1, ByteOrder::BigEndian);
        let d = Divisor::new(&b);
        let quotients = if cfg!(miri) {
            vec![MU_PREPARED, n + 1, 2 * n]
        } else {
            vec![
                MU_PREPARED,
                2 * n / 3 - 1,
                2 * n / 3 + 1,
                n,
                n + 1,
                n + 2,
                n + n / 4 - 1,
                n + n / 4,
                3 * n + 7,
            ]
        };
        for (i, qn) in quotients.into_iter().enumerate() {
            let q = random_magnitude(&mut state, 8 * qn);
            let qy = mul_magnitudes(&q, &y);
            let order = ORDERS[(kind + i) % 2];
            let x = random_magnitude(&mut state, 8 * (n + qn));
            for x in [x, qy.clone(), sub_magnitudes(&qy, &[1])] {
                let a = number(&x, i % 2 == 1, order);
                assert_divides_by(&a, &b, &d);
            }
        }
    }
}
