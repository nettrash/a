//! The arithmetic, comparisons, rounding, sums and products of rationals, against fractions
//! formed without reduction and reduced by `Rational::new`, and against `i128` arithmetic.

use super::common::{below, check, fraction, integer, rational, size, xorshift};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;
use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// `a op b` by the unreduced formulas.
fn reference(a: &Rational, b: &Rational, op: char) -> Rational {
    let (an, ad) = (a.numer(), a.denom());
    let (bn, bd) = (b.numer(), b.denom());
    match op {
        '+' => fraction(&(an * bd) + &(bn * ad), ad * bd),
        '-' => fraction(&(an * bd) - &(bn * ad), ad * bd),
        '*' => fraction(an * bn, ad * bd),
        '/' => fraction(an * bd, ad * bn),
        _ => unreachable!(),
    }
}

/// Every form of `a op b` (borrowed and owned operands, the assignments), checked against the
/// reference and against each other, with the byte order of the left operand.
fn every_form(a: &Rational, b: &Rational, op: char) {
    let expected = reference(a, b, op);
    let forms: [Rational; 6] = match op {
        '+' => [
            a + b,
            a.clone() + b,
            a + b.clone(),
            a.clone() + b.clone(),
            {
                let mut x = a.clone();
                x += b;
                x
            },
            {
                let mut x = a.clone();
                x += b.clone();
                x
            },
        ],
        '-' => [
            a - b,
            a.clone() - b,
            a - b.clone(),
            a.clone() - b.clone(),
            {
                let mut x = a.clone();
                x -= b;
                x
            },
            {
                let mut x = a.clone();
                x -= b.clone();
                x
            },
        ],
        '*' => [
            a * b,
            a.clone() * b,
            a * b.clone(),
            a.clone() * b.clone(),
            {
                let mut x = a.clone();
                x *= b;
                x
            },
            {
                let mut x = a.clone();
                x *= b.clone();
                x
            },
        ],
        '/' => [
            a / b,
            a.clone() / b,
            a / b.clone(),
            a.clone() / b.clone(),
            {
                let mut x = a.clone();
                x /= b;
                x
            },
            {
                let mut x = a.clone();
                x /= b.clone();
                x
            },
        ],
        _ => unreachable!(),
    };
    for (k, r) in forms.iter().enumerate() {
        check(r);
        assert_eq!(*r, expected, "{a:?} {op} {b:?}, form {k}");
        assert!(
            r.byte_order() == a.byte_order(),
            "{a:?} {op} {b:?}, form {k}: byte order"
        );
    }
}

#[test]
fn arithmetic_against_unreduced_fractions() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..size(4000, 60) {
        let a = rational(&mut s, 40);
        let b = rational(&mut s, 40);
        for op in ['+', '-', '*'] {
            every_form(&a, &b, op);
        }
        if !b.is_zero() {
            every_form(&a, &b, '/');
        }
    }
}

#[test]
fn arithmetic_on_the_register_path_against_i128() {
    // Parts up to 2^62 keep every unreduced numerator and denominator inside `i128`.
    let mut s = 7u64;
    let reduce = |n: i128, d: i128| {
        let (n, d) = if d < 0 { (-n, -d) } else { (n, d) };
        let g = gcd(n.unsigned_abs(), d.unsigned_abs()) as i128;
        (n / g, d / g)
    };
    for _ in 0..size(20000, 200) {
        let bits = 1 + below(&mut s, 62) as u32;
        let part = |s: &mut u64| (xorshift(s) >> (64 - bits)) as i64;
        let (an, mut ad) = (
            part(&mut s) as i128 * if below(&mut s, 2) == 0 { 1 } else { -1 },
            part(&mut s) as i128,
        );
        let (bn, mut bd) = (
            part(&mut s) as i128 * if below(&mut s, 2) == 0 { 1 } else { -1 },
            part(&mut s) as i128,
        );
        if ad == 0 {
            ad = 1;
        }
        if bd == 0 {
            bd = 3;
        }
        let a = Rational::new(Number::from(an), Number::from(ad));
        let b = Rational::new(Number::from(bn), Number::from(bd));
        let cases = [
            ('+', reduce(an * bd + bn * ad, ad * bd)),
            ('-', reduce(an * bd - bn * ad, ad * bd)),
            ('*', reduce(an * bn, ad * bd)),
        ];
        for (op, (n, d)) in cases {
            let r = match op {
                '+' => &a + &b,
                '-' => &a - &b,
                _ => &a * &b,
            };
            check(&r);
            assert_eq!(
                (r.numer().clone(), r.denom().clone()),
                (Number::from(n), Number::from(d)),
                "{a} {op} {b}"
            );
        }
        if bn != 0 {
            let r = &a / &b;
            check(&r);
            let (n, d) = reduce(an * bd, ad * bn);
            assert_eq!(
                (r.numer().clone(), r.denom().clone()),
                (Number::from(n), Number::from(d)),
                "{a} / {b}"
            );
        }
        let order = (an * bd).cmp(&(bn * ad));
        assert_eq!(a.cmp(&b), order, "{a} cmp {b}");
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

#[test]
fn sums_whose_numerators_pass_128_bits() {
    // Parts of 64 bits whose cross products sum beyond 2^128 leave the register path.
    let m = Number::from(u64::MAX);
    let a = Rational::new(m.clone(), Number::from(u64::MAX - 1));
    let b = Rational::new(m.clone(), Number::from(u64::MAX - 2));
    every_form(&a, &b, '+');
    every_form(&-&a, &-&b, '+');
    every_form(&a, &-&b, '-');
}

#[test]
fn arithmetic_with_numbers_and_primitives() {
    let mut s = 0xDEAD_BEEFu64;
    for _ in 0..size(1500, 30) {
        let a = rational(&mut s, 30);
        let len = if below(&mut s, 2) == 0 { 8 } else { 30 };
        let n = integer(&mut s, len);
        let r = Rational::from(&n);
        check(&r);
        let checks = [
            (&a + &n, &a + &r),
            (&a - &n, &a - &r),
            (&a * &n, &a * &r),
            (&n + &a, &r + &a),
            (&n - &a, &r - &a),
            (&n * &a, &r * &a),
            (a.clone() + n.clone(), &a + &r),
            (n.clone() - a.clone(), &r - &a),
        ];
        for (k, (x, y)) in checks.iter().enumerate() {
            check(x);
            assert_eq!(x, y, "{a} and {n}, case {k}");
        }
        assert!((&n + &a).byte_order() == n.byte_order());
        if !n.is_zero() {
            check(&(&a / &n));
            assert_eq!(&a / &n, &a / &r);
            let mut x = a.clone();
            x /= &n;
            assert_eq!(x, &a / &r);
        }
        if !a.is_zero() {
            assert_eq!(&n / &a, &r / &a);
            assert_eq!(n.clone() / a.clone(), &r / &a);
        }
        let shift = below(&mut s, 64);
        let v = xorshift(&mut s) as i64 >> shift;
        let rv = Rational::from(v);
        assert_eq!(&a + v, &a + &rv);
        assert_eq!(v - &a, &rv - &a);
        assert_eq!(a.clone() * v, &a * &rv);
        if v != 0 {
            assert_eq!(&a / v, &a / &rv);
        }
        let w = (xorshift(&mut s) as u128) << 64 | xorshift(&mut s) as u128;
        let rw = Rational::from(w);
        assert_eq!(&a + w, &a + &rw);
        assert_eq!(w * a.clone(), &rw * &a);
        let mut x = a.clone();
        x -= w;
        assert_eq!(x, &a - &rw);
        assert_eq!(a.partial_cmp(&v), a.partial_cmp(&rv));
        assert_eq!(a.partial_cmp(&w), a.partial_cmp(&rw));
        assert_eq!(a.partial_cmp(&n), a.partial_cmp(&r));
        assert_eq!(n.partial_cmp(&a), r.partial_cmp(&a));
        assert_eq!(a == n, a == r);
    }
}

#[test]
fn comparisons_against_cross_products() {
    let mut s = 99u64;
    for _ in 0..size(3000, 60) {
        let a = rational(&mut s, 40);
        let b = if below(&mut s, 8) == 0 {
            a.clone()
        } else {
            rational(&mut s, 40)
        };
        let left = a.numer() * b.denom();
        let right = b.numer() * a.denom();
        assert_eq!(a.cmp(&b), left.cmp(&right), "{a} cmp {b}");
        assert_eq!(a == b, left == right);
        if a == b {
            let (mut h1, mut h2) = (DefaultHasher::new(), DefaultHasher::new());
            a.hash(&mut h1);
            b.hash(&mut h2);
            assert_eq!(h1.finish(), h2.finish());
        }
    }
}

#[test]
fn equal_whatever_the_byte_order() {
    let mut s = 5u64;
    for _ in 0..size(500, 20) {
        let a = rational(&mut s, 40);
        let flip = |n: &Number| {
            let order = match n.byte_order() {
                ByteOrder::BigEndian => ByteOrder::LittleEndian,
                ByteOrder::LittleEndian => ByteOrder::BigEndian,
            };
            let mut v = n.value.clone();
            v.reverse();
            Number::new_signed(v, n.is_negative(), order)
        };
        let b = Rational::new(flip(a.numer()), flip(a.denom()));
        assert!(a.byte_order() != b.byte_order());
        assert_eq!(a, b);
        assert_eq!(a.cmp(&b), Ordering::Equal);
        let (mut h1, mut h2) = (DefaultHasher::new(), DefaultHasher::new());
        a.hash(&mut h1);
        b.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
        let sum = &a + &b;
        check(&sum);
        assert!(sum.byte_order() == a.byte_order());
        assert_eq!(sum, &a * 2u8);
    }
}

#[test]
fn rounding_against_the_definitions() {
    let mut s = 31u64;
    for _ in 0..size(3000, 60) {
        let r = rational(&mut s, 30);
        let (n, d) = (r.numer(), r.denom());
        let floor = n.div_floor(d);
        assert_eq!(r.floor(), floor, "floor {r}");
        let ceil = if (&floor * d) == *n {
            floor.clone()
        } else {
            &floor + 1u8
        };
        assert_eq!(r.ceil(), ceil, "ceil {r}");
        assert_eq!(r.trunc(), n / d, "trunc {r}");
        let fract = r.fract();
        check(&fract);
        assert_eq!(&fract + &Rational::from(r.trunc()), r, "fract {r}");
        assert!(fract.abs() < 1u8);
        // The nearest integers: no integer is nearer, and halves go the stated way.
        for (even, rounded) in [(false, r.round()), (true, r.round_ties_even())] {
            let x = Rational::from(&rounded);
            let distance = (&r - &x).abs();
            let half = Rational::from(1u8) / 2u8;
            assert!(distance <= half, "round {r} -> {rounded}");
            if distance == half {
                if even {
                    assert!(!rounded.bit(0), "round_ties_even {r} -> {rounded}");
                } else {
                    assert!(rounded.abs() > r.abs(), "round {r} -> {rounded}");
                }
            }
        }
        for x in [r.floor(), r.ceil(), r.round(), r.trunc()] {
            assert!(x.byte_order() == r.byte_order());
        }
    }
}

#[test]
fn halves_round_as_floats_do() {
    for k in -12i64..=12 {
        let r = Rational::from(k) / 2u8;
        let f = k as f64 / 2.0;
        assert_eq!(r.round(), Number::from(f.round() as i64), "{r}");
        assert_eq!(
            r.round_ties_even(),
            Number::from(f.round_ties_even() as i64),
            "{r}"
        );
        assert_eq!(r.floor(), Number::from(f.floor() as i64), "{r}");
        assert_eq!(r.ceil(), Number::from(f.ceil() as i64), "{r}");
        assert_eq!(r.trunc(), Number::from(f.trunc() as i64), "{r}");
    }
}

#[test]
fn powers_reciprocals_and_signs() {
    let mut s = 77u64;
    for _ in 0..size(400, 20) {
        let r = rational(&mut s, 20);
        let k = below(&mut s, 7) as i32;
        let mut p = Rational::from(1u8);
        for _ in 0..k {
            p = &p * &r;
        }
        assert_eq!(r.pow(k), p, "{r}^{k}");
        check(&r.pow(k));
        if !r.is_zero() {
            assert_eq!(r.pow(-k), p.recip(), "{r}^-{k}");
            assert_eq!(&r * &r.recip(), 1u8);
            check(&r.recip());
            assert!(r.recip().is_negative() == r.is_negative());
        }
        assert_eq!(
            r.abs().cmp(&Rational::default()),
            if r.is_zero() {
                Ordering::Equal
            } else {
                Ordering::Greater
            }
        );
        assert_eq!(&r.signum() * &r.abs(), r);
        assert_eq!(-(-&r), r);
        check(&-&r);
    }
    assert_eq!(Rational::default().pow(0), 1u8);
}

#[test]
#[should_panic(expected = "divide by zero")]
fn division_by_zero_panics() {
    let _ = Rational::from(1u8) / Rational::default();
}

#[test]
#[should_panic(expected = "divide by zero")]
fn reciprocal_of_zero_panics() {
    let _ = Rational::default().recip();
}

#[test]
#[should_panic(expected = "denominator")]
fn zero_denominator_panics() {
    let _ = Rational::new(Number::from(1u8), Number::from(0u8));
}

#[test]
fn sums_and_products_of_iterators() {
    let h: Rational = (1..=30u32).map(|k| Rational::from(1u8) / k).sum();
    let mut expected = Rational::default();
    for k in 1..=30u32 {
        expected = &expected + &(Rational::from(1u8) / k);
    }
    assert_eq!(h, expected);
    assert_eq!(h.to_string(), "9304682830147/2329089562800");
    let v: Vec<Rational> = (2..=9u32).map(|k| Rational::from(k - 1) / k).collect();
    assert_eq!(v.iter().product::<Rational>(), Rational::from(1u8) / 9u8);
    assert_eq!(
        v.iter().sum::<Rational>(),
        v.clone().into_iter().sum::<Rational>()
    );
    assert_eq!(
        Vec::<Rational>::new().into_iter().product::<Rational>(),
        1u8
    );
    assert!(Vec::<Rational>::new().iter().sum::<Rational>().is_zero());
}

#[test]
fn text_and_formatting() {
    assert_eq!(
        Rational::new(Number::from(-6), Number::from(4)).to_string(),
        "-3/2"
    );
    assert_eq!(Rational::from(0u8).to_string(), "0");
    assert_eq!(format!("{:>6}", Rational::from(-1) / 3), "  -1/3");
    assert_eq!(format!("{:<6}|", Rational::from(1) / 3), "1/3   |");
    assert_eq!(format!("{:+}", Rational::from(2) / 3), "+2/3");
    assert_eq!(format!("{:?}", Rational::from(-2) / 6), "-1/3");
}
