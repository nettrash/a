use a::math::numbers::Number;

use super::common::*;
use super::div::panic_message;

/// Checks every operator of `a` with the integer `p` against the same operator with `p` made a
/// number, `b`: the results must be stored alike, byte order included.
macro_rules! assert_like_numbers {
    ($a:expr, $p:expr, $t:ty) => {{
        let a: &Number = $a;
        let p: $t = $p;
        let b = Number::from(p);
        let zero = b.is_zero();
        let inputs = || format!("a = {}, p = {p}{}", describe(a), stringify!($t));
        let check = |what: &str, got: Number, expected: Number| {
            assert!(
                same(&got, &expected),
                "{what} with {}: {} vs {}",
                inputs(),
                describe(&got),
                describe(&expected)
            );
        };
        check("&a + p", a + p, a + &b);
        check("a + p", copy(a, pick(3)) + p, a + &b);
        check("p + &a", p + a, a + &b);
        check("p + a", p + copy(a, 0), a + &b);
        check("&a - p", a - p, a - &b);
        check("a - p", copy(a, pick(3)) - p, a - &b);
        check("p - &a", p - a, -(a - &b));
        check("p - a", p - copy(a, 0), -(a - &b));
        check("&a * p", a * p, a * &b);
        check("a * p", copy(a, pick(3)) * p, a * &b);
        check("p * &a", p * a, a * &b);
        let mut n = copy(a, pick(3));
        n += p;
        check("a += p", n, a + &b);
        let mut n = copy(a, pick(3));
        n -= p;
        check("a -= p", n, a - &b);
        let mut n = copy(a, pick(3));
        n *= p;
        check("a *= p", n, a * &b);
        if zero {
            for (what, f) in [
                ("&a / p", Box::new(|| drop(a / p)) as Box<dyn Fn()>),
                ("a / p", Box::new(|| drop(copy(a, 0) / p))),
                ("a /= p", Box::new(|| copy(a, 0).div_assign_for_tests(p))),
            ] {
                assert_eq!(panic_message(f), "attempt to divide by zero", "{what}");
            }
            let message = panic_message(|| drop(a % p));
            assert_eq!(
                message,
                "attempt to calculate the remainder with a divisor of zero"
            );
        } else {
            check("&a / p", a / p, a / &b);
            check("a / p", copy(a, pick(3)) / p, a / &b);
            check("&a % p", a % p, a % &b);
            check("a % p", copy(a, pick(3)) % p, a % &b);
            let mut n = copy(a, pick(3));
            n /= p;
            check("a /= p", n, a / &b);
            let mut n = copy(a, pick(3));
            n %= p;
            check("a %= p", n, a % &b);
        }
        if !a.is_zero() {
            // The integer on the left, as a number in `a`'s byte order.
            let left = Number::new_signed(
                in_order(&le_bytes(&b), a.byte_order),
                b.is_negative(),
                a.byte_order,
            );
            check("p / &a", p / a, &left / a);
            check("p / a", p / copy(a, pick(3)), &left / a);
            check("p % &a", p % a, &left % a);
            check("p % a", p % copy(a, pick(3)), &left % a);
        }
    }};
}

/// `n /= p`, as a method, so that a closure can run it.
trait DivAssignForTests<T> {
    fn div_assign_for_tests(self, p: T);
}

impl<T> DivAssignForTests<T> for Number
where
    Number: core::ops::DivAssign<T>,
{
    fn div_assign_for_tests(mut self, p: T) {
        self /= p;
    }
}

#[test]
fn operators_with_integers_are_those_with_numbers() {
    let mut state = 0x5EED_0040;
    for _ in 0..size(1500, 25) {
        let a = random_operand(&mut state, 72);
        let x = random_i128(&mut state) >> below(&mut state, 128);
        let wide = (u128::from(xorshift(&mut state)) << 64) | u128::from(xorshift(&mut state));
        let wide = wide >> below(&mut state, 128);
        assert_like_numbers!(&a, x as i8, i8);
        assert_like_numbers!(&a, x as u8, u8);
        assert_like_numbers!(&a, x as i16, i16);
        assert_like_numbers!(&a, x as u16, u16);
        assert_like_numbers!(&a, x as i32, i32);
        assert_like_numbers!(&a, x as u32, u32);
        assert_like_numbers!(&a, x as i64, i64);
        assert_like_numbers!(&a, x as u64, u64);
        assert_like_numbers!(&a, x as isize, isize);
        assert_like_numbers!(&a, x as usize, usize);
        assert_like_numbers!(&a, x, i128);
        assert_like_numbers!(&a, wide, u128);
    }
}

#[test]
fn integers_at_the_edges_operate_as_numbers() {
    let mut state = 0x5EED_0041;
    let operands = [
        Number::from(0),
        Number::from(1),
        Number::from(-1),
        Number::from(u128::MAX),
        Number::from(i128::MIN),
        -Number::from(u128::MAX),
        random_operand(&mut state, 300),
    ];
    for a in &operands {
        for p in [0, 1, -1, i128::MAX, i128::MIN, i128::MIN + 1] {
            assert_like_numbers!(a, p, i128);
        }
        for p in [0, 1, u128::MAX, 1 << 127, 1 << 64] {
            assert_like_numbers!(a, p, u128);
        }
        for p in [i64::MIN, i64::MAX, -1, 0] {
            assert_like_numbers!(a, p, i64);
        }
    }
}
