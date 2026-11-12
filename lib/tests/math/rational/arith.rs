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
fn the_default_is_zero_in_the_form() {
    let z = Rational::default();
    check(&z);
    assert!(z.is_zero());
    assert_eq!(z, Rational::from(0u8));
    assert_eq!(z.to_string(), "0");
    let mut s = z.clone();
    s += Rational::from(3) / 4u32;
    check(&s);
    assert_eq!(s.to_string(), "3/4");
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

/// A number of exactly `bits` bits (at least 1), random below the top bit, in the byte order
/// `order`.
fn of_bits(s: &mut u64, bits: usize, order: ByteOrder) -> Number {
    let bits = bits.max(1);
    let mut m = Vec::with_capacity(bits.div_ceil(8));
    while m.len() < bits.div_ceil(8) {
        m.push(xorshift(s) as u8);
    }
    let top = bits - 8 * (m.len() - 1);
    let last = m.last_mut().unwrap();
    *last &= ((1u16 << top) - 1) as u8;
    *last |= 1 << (top - 1);
    if order == ByteOrder::BigEndian {
        m.reverse();
    }
    Number::new(m, order)
}

/// Bit lengths around the boundaries of the register paths and of the top bits the comparisons
/// read: 1 to 3, about 63, 64, 65, 127, 128, 129 and 192, and random up to 700.
fn boundary_bits(s: &mut u64) -> usize {
    const EDGES: [usize; 14] = [1, 2, 3, 62, 63, 64, 65, 71, 72, 127, 128, 129, 192, 193];
    if below(s, 3) == 0 {
        1 + below(s, 700) as usize
    } else {
        EDGES[below(s, EDGES.len() as u64) as usize]
    }
}

/// A number of [`boundary_bits`] bits in the byte order `order`.
fn boundary(s: &mut u64, order: ByteOrder) -> Number {
    let bits = boundary_bits(s);
    of_bits(s, bits, order)
}

#[test]
fn comparisons_of_close_numbers() {
    // `a/b` against `(a·k + e)/(b·k)` for `e` of -1, 0 and 1: numbers that agree to about the
    // length of `b·k`, which the bounds from the top bits cannot tell apart, and equal ones in
    // other forms; with parts at the lengths where the reading of the top bits changes.
    let mut s = 0x0C10_5E00u64;
    for _ in 0..size(3000, 60) {
        let (o1, o2) = (
            super::common::order(xorshift(&mut s)),
            super::common::order(xorshift(&mut s)),
        );
        let mut an = boundary(&mut s, o1);
        if below(&mut s, 2) == 0 {
            an = -an;
        }
        let ad = boundary(&mut s, o1);
        let a = Rational::new(an, ad);
        let k = boundary(&mut s, o2);
        let e = below(&mut s, 3) as i32 - 1;
        let bn = &(a.numer() * &k) + &Number::from(e);
        let bd = a.denom() * &k;
        let b = Rational::new(bn, bd);
        let left = a.numer() * b.denom();
        let right = b.numer() * a.denom();
        assert_eq!(a.cmp(&b), left.cmp(&right), "{a} cmp {b}");
        assert_eq!(b.cmp(&a), right.cmp(&left), "{b} cmp {a}");
        assert_eq!(
            a.partial_cmp(b.numer()),
            a.numer().partial_cmp(&(b.numer() * a.denom()))
        );
        // Random numbers of the boundary lengths, for the bounds themselves.
        let c = Rational::new(boundary(&mut s, o1), boundary(&mut s, o1));
        let d = Rational::new(boundary(&mut s, o2), boundary(&mut s, o2));
        let left = c.numer() * d.denom();
        let right = d.numer() * c.denom();
        assert_eq!(c.cmp(&d), left.cmp(&right), "{c} cmp {d}");
        assert_eq!((-&c).cmp(&-&d), right.cmp(&left), "-{c} cmp -{d}");
    }
}

/// Checks `r = a op b` without `Rational::new`: `r` in lowest terms, in the byte order of `a`, and
/// `r.num · D = N · r.den` for the unreduced `N/D` of the operation.
#[track_caller]
fn verify(a: &Rational, b: &Rational, op: char, r: &Rational) {
    check(r);
    assert!(
        r.byte_order() == a.byte_order(),
        "{a:?} {op} {b:?}: byte order"
    );
    let (an, ad) = (a.numer(), a.denom());
    let (bn, bd) = (b.numer(), b.denom());
    let (n, d) = match op {
        '+' => (&(an * bd) + &(bn * ad), ad * bd),
        '-' => (&(an * bd) - &(bn * ad), ad * bd),
        '*' => (an * bn, ad * bd),
        _ => (an * bd, ad * bn),
    };
    assert_eq!(
        r.numer() * &d,
        &n * r.denom(),
        "{a:?} {op} {b:?} gave {r:?}"
    );
}

/// Every form of `a op b`, each verified, all equal.
fn every_form_verified(a: &Rational, b: &Rational, op: char) {
    let results = [
        match op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            _ => a / b,
        },
        match op {
            '+' => a.clone() + b,
            '-' => a.clone() - b,
            '*' => a.clone() * b,
            _ => a.clone() / b,
        },
        match op {
            '+' => a + b.clone(),
            '-' => a - b.clone(),
            '*' => a * b.clone(),
            _ => a / b.clone(),
        },
        {
            let mut x = a.clone();
            match op {
                '+' => x += b,
                '-' => x -= b,
                '*' => x *= b,
                _ => x /= b,
            }
            x
        },
    ];
    verify(a, b, op, &results[0]);
    for (k, r) in results.iter().enumerate() {
        check(r);
        assert!(r.byte_order() == a.byte_order(), "form {k}: byte order");
        assert_eq!(*r, results[0], "{a:?} {op} {b:?}, form {k}");
    }
}

/// A random rational whose parts have `num` and `den` bytes (at least 1), times `f` above and
/// below when given (so that it shares nothing after the reduction but what `f` brings to other
/// operands), in the byte order `order`, of a random sign.
fn sized(s: &mut u64, num: usize, den: usize, order: ByteOrder) -> Rational {
    let bits = 8 * num - below(s, 8) as usize;
    let mut n = of_bits(s, bits, order);
    if below(s, 2) == 0 {
        n = -n;
    }
    let bits = 8 * den - below(s, 8) as usize;
    let d = of_bits(s, bits, order);
    Rational::new(n, d)
}

/// Lengths in bytes across the paths: the register path, two limbs, a few, the schoolbook and
/// Karatsuba products, and past the half gcd and the working memory on the stack.
///
/// Under Miri, where the half gcd starts from 6 limbs, the longest are of up to 160 bytes.
fn length(s: &mut u64) -> usize {
    match below(s, 10) {
        0..=2 => 1 + below(s, 8) as usize,
        3 => 9 + below(s, 8) as usize,
        4 | 5 => 17 + below(s, 48) as usize,
        _ if cfg!(miri) => 65 + below(s, 96) as usize,
        6 | 7 => 65 + below(s, 300) as usize,
        8 => 400 + below(s, 1600) as usize,
        _ => 6400 + below(s, 3000) as usize,
    }
}

#[test]
fn long_operands_against_cross_products() {
    let mut s = 0x10A6_0000_0001u64;
    for i in 0..size(300, 6) {
        let (o1, o2) = (
            super::common::order(xorshift(&mut s)),
            super::common::order(xorshift(&mut s)),
        );
        let (l1, l2, l3, l4) = (
            length(&mut s),
            length(&mut s),
            length(&mut s),
            length(&mut s),
        );
        let mut a = sized(&mut s, l1, l2, o1);
        let mut b = sized(&mut s, l3, l4, o2);
        if i % 3 == 1 {
            // Denominators with a common factor, and numerators sharing factors with the other
            // operand's denominator, for the gcds of the sums and of the products.
            let (lf, lh) = (8 * length(&mut s).min(200), 8 * length(&mut s).min(200));
            let f = Rational::new(of_bits(&mut s, lf, o1), Number::from(1u8));
            let h = of_bits(&mut s, lh, o2);
            a = &(&a / &f) * &Rational::from(&h);
            b = &(&b / &f) / &Rational::from(&h);
        }
        for op in ['+', '-', '*', '/'] {
            every_form_verified(&a, &b, op);
            every_form_verified(&b, &a, op);
        }
        // Equal denominators, and an operand against itself.
        let c = Rational::new(
            &a.numer().clone() + &Number::from(xorshift(&mut s)),
            a.denom().clone(),
        );
        every_form_verified(&a, &c, '+');
        every_form_verified(&a, &c, '-');
        every_form_verified(&a, &a.clone(), '-');
        every_form_verified(&a, &a.clone(), '/');
    }
}

#[test]
fn long_operands_against_short_ones() {
    // One operand of long parts, the other of parts of one limb or an integer, as in sums of
    // series: the remainders, exact divisions and products by one limb.
    let mut s = 0x5_4A11u64;
    for _ in 0..size(400, 8) {
        let o1 = super::common::order(xorshift(&mut s));
        let o2 = super::common::order(xorshift(&mut s));
        let (l1, l2) = (length(&mut s).max(9), length(&mut s).max(9));
        let a = sized(&mut s, l1, l2, o1);
        let (l3, l4) = (1 + below(&mut s, 8) as usize, 1 + below(&mut s, 8) as usize);
        let b = sized(&mut s, l3, l4, o2);
        // A short operand that divides the long denominator, or shares a factor with it.
        let g = Number::from(1 + below(&mut s, 1 << 20));
        let c = Rational::new(
            Number::from(xorshift(&mut s) as i64),
            &g * &Number::from(3u8),
        );
        let a2 = &a / &Rational::from(&g);
        for x in [&a, &a2] {
            for y in [&b, &c] {
                for op in ['+', '-', '*', '/'] {
                    every_form_verified(x, y, op);
                    every_form_verified(y, x, op);
                }
            }
            let n = Number::from(xorshift(&mut s) >> below(&mut s, 64));
            let rn = Rational::from(&n);
            assert_eq!(x + &n, x + &rn);
            assert_eq!(&n - x, &rn - x);
            assert_eq!(x * &n, x * &rn);
            if !n.is_zero() {
                assert_eq!(x / &n, x / &rn);
            }
            assert_eq!(&n / x, &rn / x);
            let mut t = x.clone();
            t -= &n;
            assert_eq!(t, x - &rn);
        }
    }
}

#[test]
fn sums_of_many_terms() {
    // Harmonic sums and sums of random small fractions by `+=`, against the same sums of the
    // unreduced numerators over the product of the denominators, reduced once.
    let mut s = 0x15u64;
    for &(n, bits) in &[(200u64, 0u32), (60, 16), (40, 40), (25, 64)] {
        let n = if cfg!(miri) { n.min(12) } else { n };
        let mut h = Rational::from(0u8);
        let mut big_num = Number::from(0u8);
        let mut big_den = Number::from(1u8);
        for k in 1..=n {
            let (p, q) = if bits == 0 {
                (Number::from(1u8), Number::from(k))
            } else {
                let p = (xorshift(&mut s) >> (64 - bits)) as i64 * if k % 2 == 0 { 1 } else { -1 };
                let q = (xorshift(&mut s) >> (64 - bits)).max(1);
                (Number::from(p), Number::from(q))
            };
            let t = Rational::new(p.clone(), q.clone());
            h += &t;
            check(&h);
            big_num = &(&big_num * &q) + &(&p * &big_den);
            big_den = &big_den * &q;
        }
        assert_eq!(
            h,
            Rational::new(big_num, big_den),
            "{n} terms of {bits} bits"
        );
    }
}

#[test]
fn numbers_with_zero_bytes_stored_as_operands() {
    // Integer operands whose vectors hold zero bytes above the most significant one, or none.
    let mut s = 0x2E40u64;
    for _ in 0..size(300, 20) {
        let (l1, l2) = (length(&mut s).min(100), length(&mut s).min(100));
        let o = super::common::order(xorshift(&mut s));
        let a = sized(&mut s, l1, l2, o);
        let m = super::common::magnitude(&mut s, 40);
        let pad = below(&mut s, 12) as usize;
        let negative = below(&mut s, 2) == 0 && !m.is_empty();
        let order = super::common::order(xorshift(&mut s));
        let mut v = m.clone();
        v.resize(m.len() + pad, 0);
        if order == ByteOrder::BigEndian {
            v.reverse();
        }
        let padded = Number {
            value: v,
            negative,
            byte_order: order,
        };
        let r = Rational::from(super::common::number(&m, negative, order));
        assert_eq!(&a + &padded, &a + &r);
        assert_eq!(&a - &padded, &a - &r);
        assert_eq!(&a * &padded, &a * &r);
        assert_eq!(&padded - &a, &r - &a);
        assert_eq!(&padded * &a, &r * &a);
        if !r.is_zero() {
            assert_eq!(&a / &padded, &a / &r);
        }
        if !a.is_zero() {
            assert_eq!(&padded / &a, &r / &a);
        }
        let mut t = a.clone();
        t += &padded;
        check(&t);
        assert_eq!(t, &a + &r);
    }
}

#[test]
fn new_reduces_against_numbers() {
    // `Rational::new` of parts with common factors of every size, signs, byte orders and stored
    // zero bytes, against the division by the greatest common divisor of numbers.
    let mut s = 0x4E57u64;
    for i in 0..size(2000, 40) {
        let (o1, o2) = (
            super::common::order(xorshift(&mut s)),
            super::common::order(xorshift(&mut s)),
        );
        let lengths = |s: &mut u64| {
            if i % 4 == 3 {
                length(s)
            } else {
                boundary_bits(s).div_ceil(8)
            }
        };
        let (lp, lq, lg) = (lengths(&mut s), lengths(&mut s), lengths(&mut s));
        let g = if below(&mut s, 3) == 0 {
            Number::from(1u8)
        } else {
            of_bits(&mut s, 8 * lg, o2)
        };
        let (bp, bq) = (
            8 * lp - below(&mut s, 8) as usize,
            8 * lq - below(&mut s, 8) as usize,
        );
        let mut p = &of_bits(&mut s, bp, o1) * &g;
        let mut q = &of_bits(&mut s, bq, o2) * &g;
        if below(&mut s, 2) == 0 {
            p = -p;
        }
        if below(&mut s, 2) == 0 {
            q = -q;
        }
        if below(&mut s, 16) == 0 {
            p = Number::new(vec![0], o1);
        }
        // Zero bytes stored above the most significant ones.
        let pad = |n: &Number, k: usize| {
            let mut v = n.value.clone();
            match n.byte_order() {
                ByteOrder::BigEndian => {
                    v.splice(0..0, core::iter::repeat_n(0, k));
                }
                ByteOrder::LittleEndian => v.resize(v.len() + k, 0),
            }
            Number {
                value: v,
                negative: n.is_negative(),
                byte_order: n.byte_order(),
            }
        };
        let (kp, kq) = (below(&mut s, 3) as usize * 5, below(&mut s, 3) as usize * 5);
        let (pp, qq) = (pad(&p, kp), pad(&q, kq));
        let r = Rational::new(pp, qq);
        check(&r);
        assert!(r.byte_order() == o1);
        let d = p.gcd(&q);
        let (mut en, mut ed) = (p.div_exact(&d), q.div_exact(&d));
        if ed.is_negative() {
            en = -en;
            ed = -ed;
        }
        if en.is_zero() {
            ed = Number::from(1u8);
        }
        assert_eq!((r.numer(), r.denom()), (&en, &ed), "{p}/{q}");
    }
}

/// Whether `n` is in the shortest form: no zero bytes above the most significant one, zero as
/// `[0]`, never negative.
fn shortest(n: &Number) -> bool {
    let top = match n.byte_order() {
        ByteOrder::BigEndian => n.value[0],
        ByteOrder::LittleEndian => *n.value.last().unwrap(),
    };
    (top != 0 || n.value.len() == 1) && !(n.is_zero() && n.is_negative())
}

#[test]
fn rounding_of_long_and_wide_parts() {
    // Parts of every length, among them up to 16 bytes (the paths on `u128`s) and halves of long
    // numbers, against the definitions on numbers.
    let mut s = 0x2047u64;
    for i in 0..size(1500, 30) {
        let o = super::common::order(xorshift(&mut s));
        let r = if i % 5 == 4 {
            // `(2m + 1)/2`, a half, for the ties.
            let bits = 8 * length(&mut s).min(300);
            let mut m = of_bits(&mut s, bits, o);
            if below(&mut s, 2) == 0 {
                m = -m;
            }
            Rational::new(
                &(&m * &Number::from(2u8)) + &Number::from(1u8),
                Number::from(2u8),
            )
        } else {
            let (l1, l2) = if i % 2 == 0 {
                let a = boundary_bits(&mut s).div_ceil(8);
                (a, boundary_bits(&mut s).div_ceil(8))
            } else {
                let a = length(&mut s).min(500);
                (a, length(&mut s).min(500))
            };
            sized(&mut s, l1, l2, o)
        };
        let (n, d) = (r.numer(), r.denom());
        let floor = n.div_floor(d);
        let ceil = if (&floor * d) == *n {
            floor.clone()
        } else {
            &floor + 1u8
        };
        assert_eq!(r.floor(), floor, "floor {r}");
        assert_eq!(r.ceil(), ceil, "ceil {r}");
        assert_eq!(r.trunc(), n / d, "trunc {r}");
        let twice = &(n * &Number::from(2u8)) + d;
        let half_up = twice.div_floor(&(d * &Number::from(2u8)));
        // Halves away from zero: `floor(x + 1/2)` for `x >= 0`, `-floor(-x + 1/2)` below.
        let away = if n.is_negative() {
            -(&(-n * &Number::from(2u8)) + d).div_floor(&(d * &Number::from(2u8)))
        } else {
            half_up.clone()
        };
        assert_eq!(r.round(), away, "round {r}");
        let even = r.round_ties_even();
        let x = Rational::from(&even);
        let distance = (&r - &x).abs();
        let half = Rational::from(1u8) / 2u8;
        assert!(
            distance < half || (distance == half && !even.bit(0)),
            "round_ties_even {r} -> {even}"
        );
        for x in [r.floor(), r.ceil(), r.round(), r.trunc(), even] {
            assert!(x.byte_order() == r.byte_order());
            assert!(shortest(&x), "{x:?} of {r}");
        }
    }
}

#[test]
fn two_limb_parts_against_cross_products() {
    // Parts of up to 128 bits (the paths on two limbs), with common factors of up to 128 bits
    // between the denominators and across numerators and denominators, so that the gcds and the
    // second gcd of a sum (`g₂`) are taken, and a factor above 64 bits leaves for the limbs.
    let mut s = 0x2_11B5u64;
    for i in 0..size(4000, 60) {
        let (o1, o2) = (
            super::common::order(xorshift(&mut s)),
            super::common::order(xorshift(&mut s)),
        );
        let bits = |s: &mut u64| 1 + below(s, 128) as usize;
        let (b1, b2, b3, b4) = (bits(&mut s), bits(&mut s), bits(&mut s), bits(&mut s));
        let mut a = Rational::new(of_bits(&mut s, b1, o1), of_bits(&mut s, b2, o1));
        let mut b = Rational::new(of_bits(&mut s, b3, o2), of_bits(&mut s, b4, o2));
        if below(&mut s, 2) == 0 {
            a = -a;
        }
        if i % 3 != 0 {
            // A common factor of the denominators, numerators, or across, then reduced: parts
            // stay within 128 bits about two thirds of the time.
            let (gb, hb) = (
                1 + below(&mut s, 70) as usize,
                1 + below(&mut s, 40) as usize,
            );
            let g = of_bits(&mut s, gb, o1);
            let h = of_bits(&mut s, hb, o2);
            let (p, q) = (a.numer().clone(), a.denom().clone());
            a = Rational::new(&p * &h, &q * &g);
            let (p, q) = (b.numer().clone(), b.denom().clone());
            b = match i % 3 {
                1 => Rational::new(p, &q * &g),
                _ => Rational::new(&p * &g, &q * &h),
            };
        }
        for op in ['+', '-', '*', '/'] {
            if op == '/' && b.is_zero() {
                continue;
            }
            every_form_verified(&a, &b, op);
            every_form_verified(&b, &a, op);
        }
        let x = Rational::from(a.numer());
        every_form_verified(&b, &x, '+');
        every_form_verified(&x, &b, '*');
    }
}

/// A part of up to 64 bits made of factors that other parts share: powers of 2, small odd
/// primes, primes near `2^16` and `2^32`, and a random cofactor, so that the gcds of the register
/// paths meet every case: 1, a power of 2, an odd divisor, one past `2^32`, parts apart by 16 bits
/// or more, and equal parts.
fn factored(s: &mut u64) -> u64 {
    const FACTORS: [u64; 12] = [
        2,
        4,
        1 << 20,
        3,
        9,
        5,
        7,
        65_521,
        65_537,
        4_294_967_291,
        4_294_967_311,
        1 << 33,
    ];
    let mut x: u64 = 1;
    for _ in 0..below(s, 4) {
        let f = FACTORS[below(s, FACTORS.len() as u64) as usize];
        match x.checked_mul(f) {
            Some(y) => x = y,
            None => break,
        }
    }
    match below(s, 4) {
        0 => x,
        1 => x.saturating_mul(1 + below(s, 1000)),
        _ => {
            let room = 64 - x.leading_zeros().min(63);
            let c = xorshift(s) >> (room + below(s, 64 - room as u64 + 1) as u32).min(63);
            x.checked_mul(c.max(1)).unwrap_or(x)
        }
    }
}

#[test]
fn register_path_with_shared_factors() {
    let mut s = 0x5EED_0027_u64;
    for i in 0..size(20000, 300) {
        let o = super::common::order(i as u64);
        let part = |s: &mut u64| {
            let m = factored(s).to_le_bytes();
            let mut v = m.to_vec();
            while v.last() == Some(&0) {
                v.pop();
            }
            v
        };
        let (an, ad, bn, bd) = (part(&mut s), part(&mut s), part(&mut s), part(&mut s));
        let (sa, sb) = (below(&mut s, 2) == 1, below(&mut s, 2) == 1);
        let a = Rational::new(
            super::common::number(&an, sa, o),
            super::common::number(&ad, false, o),
        );
        let b = if below(&mut s, 8) == 0 {
            // Equal denominators, and now and then the same number.
            let n = if below(&mut s, 2) == 0 {
                a.numer().clone()
            } else {
                super::common::number(&bn, sb, o)
            };
            Rational::new(n, a.denom().clone())
        } else {
            Rational::new(
                super::common::number(&bn, sb, o),
                super::common::number(&bd, false, o),
            )
        };
        for op in ['+', '-', '*'] {
            every_form_verified(&a, &b, op);
        }
        if !b.is_zero() {
            every_form_verified(&a, &b, '/');
        }
        let (x, y) = (a.numer() * b.denom(), b.numer() * a.denom());
        assert_eq!(a.cmp(&b), x.cmp(&y), "{a:?} cmp {b:?}");
        assert_eq!(a == b, x == y, "{a:?} == {b:?}");
    }
}

#[test]
fn powers_of_words_across_the_boundary() {
    // Parts of every length to 64 bits, raised so that the powers end on either side of 64 bits
    // and of 512, in either byte order and sign; against the powers of the parts as numbers.
    let mut s = 0x0909_2727_u64;
    for i in 0..size(3000, 60) {
        let o = super::common::order(i as u64);
        let bits = |s: &mut u64| 1 + below(s, 64) as u32;
        let part = |s: &mut u64, b: u32| (xorshift(s) >> (64 - b)) | (1u64 << (b - 1));
        let (bn, bd) = (bits(&mut s), bits(&mut s));
        let (n, d) = (part(&mut s, bn), part(&mut s, bd));
        let negative = below(&mut s, 2) == 1;
        let p = super::common::number(&n.to_le_bytes(), negative, o);
        let q = super::common::number(&d.to_le_bytes(), false, o);
        let r = Rational::new(p, q);
        let k = match below(&mut s, 3) {
            0 => below(&mut s, 4) as i32,
            1 => (64 / bn.max(bd)) as i32 + below(&mut s, 2) as i32,
            _ => 1 + below(&mut s, 12) as i32,
        };
        for e in [k, -k] {
            if e < 0 && r.is_zero() {
                continue;
            }
            let got = r.pow(e);
            check(&got);
            let (rn, rd) = (
                r.numer().pow(e.unsigned_abs()),
                r.denom().pow(e.unsigned_abs()),
            );
            let want = if e < 0 {
                Rational::new(rd, rn)
            } else {
                Rational::new(rn, rd)
            };
            assert_eq!(got, want, "({r})^{e}");
            assert!(got.byte_order() == r.byte_order(), "({r})^{e}: byte order");
        }
    }
    assert_eq!(Rational::from(0u8).pow(0), 1u8);
    assert_eq!((Rational::from(-2) / 3u8).pow(3).to_string(), "-8/27");
}
