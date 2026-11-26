//! Rationals in their two forms, words for parts of up to 64 bits and numbers beyond: every
//! operation across both and across the boundary, against unreduced fractions on numbers, and
//! the one form a value has whatever made it (a value left in the wrong form would not be equal to
//! itself made another way, nor hash alike).

use super::common::{below, check, number, order, rational, size, xorshift};
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// The hash of `x` under the standard hasher.
fn hash_of<T: Hash>(x: &T) -> u64 {
    let mut h = DefaultHasher::new();
    x.hash(&mut h);
    h.finish()
}

/// `2^e` exactly, from its bits (Miri perturbs `powi`).
fn pow2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// Asserts that `a` and `b` are the same number, equal and hashing alike, in every comparison.
#[track_caller]
fn same(a: &Rational, b: &Rational) {
    check(a);
    check(b);
    assert_eq!(a, b);
    assert_eq!(a.cmp(b), core::cmp::Ordering::Equal);
    assert_eq!(hash_of(a), hash_of(b), "{a} hashes apart from itself");
}

/// A magnitude near `2^64`: `2^k + d` for `k` from 56 to 72 and `d` from -2 to 2, at least 1.
fn near_boundary(state: &mut u64) -> Number {
    let k = 56 + below(state, 17) as usize;
    let d = below(state, 5) as i64 - 2;
    let n = (Number::from(1u8) << k) + Number::from(d);
    if n.is_zero() { Number::from(1u8) } else { n }
}

/// A random rational of parts near `2^64`, of either sign and byte order.
fn boundary_rational(state: &mut u64) -> Rational {
    let mut num = near_boundary(state);
    if below(state, 2) == 1 {
        num = -num;
    }
    let num = in_order(num, order(xorshift(state)));
    Rational::new(num, near_boundary(state))
}

/// `x` in the byte order `o`.
fn in_order(x: Number, o: ByteOrder) -> Number {
    let mut m = x.abs().to_le_bytes_vec();
    while m.last() == Some(&0) {
        m.pop();
    }
    number(&m, x.is_negative(), o)
}

/// Little-endian magnitude bytes, through the text of the number.
trait LeBytes {
    fn to_le_bytes_vec(&self) -> Vec<u8>;
}

impl LeBytes for Number {
    fn to_le_bytes_vec(&self) -> Vec<u8> {
        let mut v = self.value.clone();
        if self.byte_order() == ByteOrder::BigEndian {
            v.reverse();
        }
        v
    }
}

/// The unreduced `x op y` on numbers: `(an·bd ± bn·ad) / (ad·bd)`, `(an·bn) / (ad·bd)` and
/// `(an·bd) / (ad·bn)`.
fn unreduced(x: &Rational, y: &Rational, op: u64) -> (Number, Number) {
    let (an, ad, bn, bd) = (x.numer(), x.denom(), y.numer(), y.denom());
    match op {
        0 => (&(an * bd) + &(bn * ad), ad * bd),
        1 => (&(an * bd) - &(bn * ad), ad * bd),
        2 => (an * bn, ad * bd),
        _ => (an * bd, ad * bn),
    }
}

/// Asserts that `r` is the fraction `p/q`: `r.numer()·q = p·r.denom()`, and that it keeps the form.
#[track_caller]
fn is_fraction(r: &Rational, p: &Number, q: &Number) {
    check(r);
    assert_eq!(r.numer() * q, p * r.denom(), "{r} is not {p}/{q}");
}

#[test]
fn every_operator_across_the_forms() {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..size(3000, 40) {
        let pick = |state: &mut u64| match below(state, 3) {
            0 => boundary_rational(state),
            1 => rational(state, 24),
            _ => rational(state, 8),
        };
        let (x, y) = (pick(&mut state), pick(&mut state));
        for op in 0..4 {
            if op == 3 && y.is_zero() {
                continue;
            }
            let (p, q) = unreduced(&x, &y, op);
            let by_ref = match op {
                0 => &x + &y,
                1 => &x - &y,
                2 => &x * &y,
                _ => &x / &y,
            };
            is_fraction(&by_ref, &p, &q);
            let left = match op {
                0 => x.clone() + &y,
                1 => x.clone() - &y,
                2 => x.clone() * &y,
                _ => x.clone() / &y,
            };
            let right = match op {
                0 => &x + y.clone(),
                1 => &x - y.clone(),
                2 => &x * y.clone(),
                _ => &x / y.clone(),
            };
            let mut assigned = x.clone();
            match op {
                0 => assigned += &y,
                1 => assigned -= &y,
                2 => assigned *= &y,
                _ => assigned /= &y,
            }
            for r in [&left, &right, &assigned] {
                same(r, &by_ref);
            }
            // The same from parts that are not reduced, by `new`.
            if !q.is_zero() {
                same(&Rational::new(p.clone(), q.clone()), &by_ref);
            }
        }
        // The order, by the sign of `an·bd - bn·ad`.
        let (p, _) = unreduced(&x, &y, 1);
        let expected = if p.is_zero() {
            core::cmp::Ordering::Equal
        } else if p.is_negative() {
            core::cmp::Ordering::Less
        } else {
            core::cmp::Ordering::Greater
        };
        assert_eq!(x.cmp(&y), expected, "{x} against {y}");
        assert_eq!(x == y, p.is_zero());
    }
}

#[test]
fn operators_with_numbers_and_primitives_across_the_forms() {
    let mut state = 0x1234_5678_9ABC_DEF1u64;
    for _ in 0..size(2000, 30) {
        let x = if below(&mut state, 2) == 0 {
            boundary_rational(&mut state)
        } else {
            rational(&mut state, 16)
        };
        let n = match below(&mut state, 3) {
            0 => near_boundary(&mut state),
            1 => Number::from(xorshift(&mut state) as i64),
            _ => Number::from(below(&mut state, 5) as i64 - 2),
        };
        let y = Rational::from(n.clone());
        for op in 0..4 {
            if op == 3 && n.is_zero() {
                continue;
            }
            let expected = match op {
                0 => &x + &y,
                1 => &x - &y,
                2 => &x * &y,
                _ => &x / &y,
            };
            let got = match op {
                0 => &x + &n,
                1 => &x - &n,
                2 => &x * &n,
                _ => &x / &n,
            };
            same(&got, &expected);
            let mut assigned = x.clone();
            match op {
                0 => assigned += &n,
                1 => assigned -= &n,
                2 => assigned *= &n,
                _ => assigned /= &n,
            }
            same(&assigned, &expected);
            if !x.is_zero() || op != 3 {
                let back = match op {
                    0 => &n + &x,
                    1 => &n - &x,
                    2 => &n * &x,
                    _ => &n / &x,
                };
                let expected = match op {
                    0 => &y + &x,
                    1 => &y - &x,
                    2 => &y * &x,
                    _ => &y / &x,
                };
                same(&back, &expected);
            }
        }
        // A primitive on the right.
        let v = xorshift(&mut state) as i64 >> below(&mut state, 64);
        let yv = Rational::from(v);
        same(&(&x + v), &(&x + &yv));
        same(&(&x * v), &(&x * &yv));
        same(&(v - &x), &(&yv - &x));
        assert_eq!(x == v, x == yv);
        assert_eq!(x.partial_cmp(&v), Some(x.cmp(&yv)));
        assert_eq!(x == n, x == y);
        assert_eq!(x.partial_cmp(&n), Some(x.cmp(&y)));
    }
}

#[test]
fn one_form_whatever_made_the_value() {
    let mut state = 0x0F0F_1234_5555_AAAAu64;
    for _ in 0..size(2000, 30) {
        let x = boundary_rational(&mut state);
        // Through a long value and back: `(x + L) - L` and `(x · L) / L` for a long `L`.
        let long = Rational::new(
            Number::from(1u8) << (70 + below(&mut state, 200) as usize),
            near_boundary(&mut state),
        );
        same(&(&(&x + &long) - &long), &x);
        if !long.is_zero() {
            same(&(&(&x * &long) / &long), &x);
            let mut y = x.clone();
            y += &long;
            y -= &long;
            same(&y, &x);
        }
        // From parts with a long common factor.
        let f = Number::from(3u8).pow(40 + below(&mut state, 40) as u32);
        same(&Rational::new(x.numer() * &f, x.denom() * &f), &x);
        // From its parts, its text, the reciprocal twice, a power and back, a negation twice.
        let (p, q) = x.clone().into_parts();
        same(&Rational::new(p, q), &x);
        same(&x.to_string().parse::<Rational>().unwrap(), &x);
        if !x.is_zero() {
            same(&x.recip().recip(), &x);
            same(&x.clone().into_recip().into_recip(), &x);
        }
        same(&-(-x.clone()), &x);
        same(&x.pow(1), &x);
        same(&x.pow(0), &Rational::from(1u8));
        same(&(x.fract() + Rational::from(x.trunc())), &x);
    }
    // Integers of every primitive type and as numbers, in both byte orders.
    for v in [0i64, 1, -1, i64::MAX, i64::MIN, 1 << 40] {
        let r = Rational::from(v);
        same(&r, &Rational::from(Number::from(v)));
        same(
            &r,
            &Rational::from_integer(in_order(Number::from(v), ByteOrder::BigEndian)),
        );
        same(&r, &Rational::from(v as i128));
    }
    same(
        &Rational::from(u64::MAX),
        &Rational::from(Number::from(u64::MAX)),
    );
    same(
        &Rational::from(u128::MAX),
        &Rational::from(Number::from(u128::MAX)),
    );
    same(&Rational::default(), &Rational::from(0u8));
    same(
        &Rational::try_from(0.75f64).unwrap(),
        &(Rational::from(3) / 4u32),
    );
    same(
        &Rational::try_from(pow2(63)).unwrap(),
        &Rational::from(1u64 << 63),
    );
    same(
        &Rational::try_from(pow2(64)).unwrap(),
        &Rational::from(1u128 << 64),
    );
    same(
        &Rational::try_from(pow2(-63)).unwrap(),
        &(Rational::from(1) / (1u64 << 63)),
    );
    same(
        &Rational::try_from(pow2(-64)).unwrap(),
        &(Rational::from(1) / (1u128 << 64)),
    );
}

#[test]
fn the_parts_follow_every_change_in_place() {
    let mut state = 0xDEAD_BEEF_0BAD_F00Du64;
    for _ in 0..size(1000, 20) {
        let mut x = rational(&mut state, 8);
        let y = rational(&mut state, 8);
        // The parts made once are the same ones until the value changes.
        let first = x.numer() as *const Number;
        assert_eq!(first, x.numer() as *const Number);
        let (n, d) = (x.numer().clone(), x.denom().clone());
        let c = x.clone();
        assert_eq!((c.numer(), c.denom()), (&n, &d));
        x = -x;
        assert_eq!(x.numer(), &-n.clone());
        assert_eq!(x.denom(), &d);
        x += &y;
        let (p, q) = unreduced(&-c.clone(), &y, 0);
        is_fraction(&x, &p, &q);
        if !x.is_zero() {
            let _ = x.numer();
            x = x.into_recip();
            assert_eq!(x.numer() * &p, q.clone() * x.denom(), "{x} is not {q}/{p}");
            check(&x);
        }
        // Taken out after they were made, and before.
        let z = rational(&mut state, 8);
        let (zn, zd) = (z.numer().clone(), z.denom().clone());
        assert_eq!(z.clone().into_parts(), (zn.clone(), zd.clone()));
        let _ = z.numer();
        assert_eq!(z.into_parts(), (zn, zd));
    }
}

#[test]
fn rounding_and_conversions_across_the_forms() {
    let mut state = 0xABCD_EF01_2345_6789u64;
    for _ in 0..size(2000, 30) {
        let x = if below(&mut state, 2) == 0 {
            boundary_rational(&mut state)
        } else {
            rational(&mut state, 20)
        };
        let (n, d) = (x.numer(), x.denom());
        assert_eq!(x.floor(), n.div_floor(d), "floor of {x}");
        assert_eq!(x.ceil(), -(-n).div_floor(d), "ceil of {x}");
        assert_eq!(x.trunc(), n / d, "trunc of {x}");
        is_fraction(&x.fract(), &(n % d), d);
        assert_eq!(x.is_integer(), d == &Number::from(1u8));
        assert_eq!(x.is_zero(), n.is_zero());
        assert_eq!(x.is_negative(), n.is_negative());
        same(&x.abs(), &Rational::new(n.abs(), d.clone()));
        same(&x.signum(), &Rational::from(n.signum()));
        // Display and the radices agree with the parts.
        let text = if x.is_integer() {
            n.to_string()
        } else {
            format!("{n}/{d}")
        };
        assert_eq!(x.to_string(), text);
        assert_eq!(x.to_str_radix(10), text);
        assert_eq!(format!("{x:>50}"), format!("{text:>50}"));
        let hex = if x.is_integer() {
            n.to_str_radix(16)
        } else {
            format!("{}/{}", n.to_str_radix(16), d.to_str_radix(16))
        };
        assert_eq!(x.to_str_radix(16), hex);
        // To integers.
        assert_eq!(Number::try_from(&x).ok(), x.is_integer().then(|| n.clone()));
        assert_eq!(
            i64::try_from(&x).ok(),
            if x.is_integer() {
                i64::try_from(n).ok()
            } else {
                None
            }
        );
        assert_eq!(
            u64::try_from(&x).ok(),
            if x.is_integer() {
                u64::try_from(n).ok()
            } else {
                None
            }
        );
        assert_eq!(
            i128::try_from(&x).ok(),
            if x.is_integer() {
                i128::try_from(n).ok()
            } else {
                None
            }
        );
        // Floats both ways: the float of `x` is the float of the same fraction made from numbers.
        let y = Rational::new(n.clone() * Number::from(1u8), d.clone());
        assert_eq!(x.to_f64().to_bits(), y.to_f64().to_bits());
        let f = f64::from_bits(xorshift(&mut state));
        if f.is_finite() {
            let r = Rational::try_from(f).unwrap();
            check(&r);
            assert_eq!(
                r.to_f64().to_bits(),
                if f == 0.0 {
                    0.0f64.to_bits()
                } else {
                    f.to_bits()
                }
            );
        }
    }
}

#[test]
fn hashes_agree_with_the_parts_and_rationals_are_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Rational>();
    let mut state = 0x5151_5151_7777_0001u64;
    for _ in 0..size(500, 20) {
        let x = rational(&mut state, 16);
        // Equal values hash alike, whatever byte order and whatever made them: the same value from
        // its parts in the other byte order, and through numbers and back.
        let other = match x.byte_order() {
            ByteOrder::LittleEndian => ByteOrder::BigEndian,
            ByteOrder::BigEndian => ByteOrder::LittleEndian,
        };
        let flip = |n: &Number| {
            let mut v = n.value.clone();
            v.reverse();
            Number::new_signed(v, n.is_negative(), other)
        };
        let y = Rational::new(flip(x.numer()), flip(x.denom()));
        assert_eq!(x, y);
        assert_eq!(hash_of(&x), hash_of(&y), "{x}");
        let z = (&x + &Rational::from(1)) - Rational::from(1);
        assert_eq!(hash_of(&x), hash_of(&z), "{x}");
        // A value hashes as its two parts do, whatever its form and byte order.
        let mut h = DefaultHasher::new();
        x.numer().hash(&mut h);
        x.denom().hash(&mut h);
        assert_eq!(hash_of(&x), h.finish(), "{x}");
    }
}

/// A hasher that keeps every call made to it, the method and its bytes, so that two hashings
/// compare call by call, as any hasher may tell them apart.
#[derive(Default, PartialEq, Debug)]
struct Calls(Vec<(&'static str, Vec<u8>)>);

impl Hasher for Calls {
    fn finish(&self) -> u64 {
        0
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0.push(("write", bytes.to_vec()));
    }
    fn write_u8(&mut self, i: u8) {
        self.0.push(("u8", vec![i]));
    }
    fn write_u16(&mut self, i: u16) {
        self.0.push(("u16", i.to_le_bytes().to_vec()));
    }
    fn write_u32(&mut self, i: u32) {
        self.0.push(("u32", i.to_le_bytes().to_vec()));
    }
    fn write_u64(&mut self, i: u64) {
        self.0.push(("u64", i.to_le_bytes().to_vec()));
    }
    fn write_u128(&mut self, i: u128) {
        self.0.push(("u128", i.to_le_bytes().to_vec()));
    }
    fn write_usize(&mut self, i: usize) {
        self.0.push(("usize", i.to_le_bytes().to_vec()));
    }
    fn write_i8(&mut self, i: i8) {
        self.0.push(("i8", i.to_le_bytes().to_vec()));
    }
    fn write_i16(&mut self, i: i16) {
        self.0.push(("i16", i.to_le_bytes().to_vec()));
    }
    fn write_i32(&mut self, i: i32) {
        self.0.push(("i32", i.to_le_bytes().to_vec()));
    }
    fn write_i64(&mut self, i: i64) {
        self.0.push(("i64", i.to_le_bytes().to_vec()));
    }
    fn write_i128(&mut self, i: i128) {
        self.0.push(("i128", i.to_le_bytes().to_vec()));
    }
    fn write_isize(&mut self, i: isize) {
        self.0.push(("isize", i.to_le_bytes().to_vec()));
    }
}

#[test]
fn a_rational_hashes_as_its_parts_call_for_call() {
    // RATIONAL-2 of the v0.2 speed pass's audit: `hash(r)` is the hash of `(numer, denom)` as
    // `Number` hashes them, with any hasher, so the very calls made to it are the same, for
    // rationals in words (both parts below 2^64) and in numbers, of either sign and byte order,
    // at 0, 1, 2^63, 2^64 and their neighbours, and at random.
    let edges: Vec<Number> = [
        0u128,
        1,
        2,
        255,
        256,
        (1 << 63) - 1,
        1 << 63,
        (1 << 63) + 1,
        u64::MAX as u128 - 1,
        u64::MAX as u128,
        1 << 64,
        (1 << 64) + 1,
        u128::MAX,
    ]
    .iter()
    .map(|&m| Number::from(m))
    .chain([Number::from(1u8) << 200usize])
    .collect();
    let mut values = Vec::new();
    for n in &edges {
        for d in edges.iter().filter(|d| !d.is_zero()) {
            for negative in [false, true] {
                let n = if negative { -n.clone() } else { n.clone() };
                values.push(Rational::new(n, d.clone()));
            }
        }
    }
    let mut state = 0x4A54_4841_5348_0001u64;
    for _ in 0..size(2000, 40) {
        values.push(rational(&mut state, 24));
    }
    let (mut words, mut longs) = (0, 0);
    for x in &values {
        for o in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
            let flip = |p: &Number| {
                let mut v = p.value.clone();
                if p.byte_order() != o {
                    v.reverse();
                }
                Number::new_signed(v, p.is_negative(), o)
            };
            let y = Rational::new(flip(x.numer()), flip(x.denom()));
            let (mut ours, mut parts) = (Calls::default(), Calls::default());
            y.hash(&mut ours);
            (y.numer(), y.denom()).hash(&mut parts);
            assert_eq!(ours, parts, "{y}");
            let mut h = DefaultHasher::new();
            y.numer().hash(&mut h);
            y.denom().hash(&mut h);
            assert_eq!(hash_of(&y), h.finish(), "{y}");
            if y.numer().bits() <= 64 && y.denom().bits() <= 64 {
                words += 1;
            } else {
                longs += 1;
            }
        }
    }
    assert!(words > 100 && longs > 100, "{words} {longs}");
    // And the sign is not a bit of the numerator: -k and 2^63 + k hash apart.
    for k in [0u64, 1, 5, (1 << 63) - 1] {
        let a = Rational::from(-Number::from(k));
        let b = Rational::from(Number::from(k | 1 << 63));
        let (mut ha, mut hb) = (Calls::default(), Calls::default());
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_ne!(ha, hb, "{a} {b}");
    }
}

/// A number of the magnitude `m` (least significant first) with `pad` zero bytes above it, the
/// sign `negative` (kept as given, for zero too) and the byte order `o`.
fn padded(m: &[u8], pad: usize, negative: bool, o: ByteOrder) -> Number {
    let mut v = m.to_vec();
    v.resize(m.len() + pad, 0);
    if o == ByteOrder::BigEndian {
        v.reverse();
    }
    Number {
        value: v,
        negative,
        byte_order: o,
    }
}

#[test]
fn rationals_made_from_short_vectors() {
    // `Rational::new`, `from_integer` and the operators take vectors of up to 8 bytes straight:
    // zero bytes above the top one, an empty vector, a negative zero, the parts in different
    // byte orders, and vectors of 9 to 16 bytes with values of a word, all against the same parts
    // trimmed.
    let mut state = 0x5407_7000_u64;
    for _ in 0..size(4000, 80) {
        let (ln, ld) = (
            below(&mut state, 9) as usize,
            1 + below(&mut state, 8) as usize,
        );
        let part = |len: usize, state: &mut u64| {
            let mut m: Vec<u8> = (0..len).map(|_| xorshift(state) as u8).collect();
            while m.last() == Some(&0) {
                m.pop();
            }
            m
        };
        let (mn, md) = (part(ln, &mut state), part(ld, &mut state));
        if md.is_empty() {
            continue;
        }
        let (pn, pd) = (
            below(&mut state, 10) as usize,
            below(&mut state, 10) as usize,
        );
        let (sn, sd) = (below(&mut state, 2) == 1, below(&mut state, 2) == 1);
        let (on, od) = (order(xorshift(&mut state)), order(xorshift(&mut state)));
        let r = Rational::new(padded(&mn, pn, sn, on), padded(&md, pd, sd, od));
        let want = Rational::new(number(&mn, sn, on), number(&md, sd, on));
        same(&r, &want);
        is_fraction(&r, &number(&mn, sn != sd, on), &number(&md, false, on));
        assert!(r.byte_order() == on, "{r}: byte order");
        let i = Rational::from_integer(padded(&mn, pn, sn, on));
        same(&i, &Rational::new(number(&mn, sn, on), Number::from(1u8)));
        same(&Rational::from(padded(&mn, pn, sn, on)), &i);
        // As operands, on either side.
        let n = padded(&mn, pn, sn, on);
        same(&(&r + &n), &(&r + &i));
        same(&(&n * &r), &(&i * &r));
        assert_eq!(r == n, r == i);
        assert_eq!(r.partial_cmp(&n), r.partial_cmp(&i));
    }
}

#[test]
#[should_panic(expected = "the denominator of a rational is zero")]
fn a_zero_denominator_of_zero_bytes_panics() {
    let _ = Rational::new(
        Number::from(3u8),
        padded(&[], 3, false, ByteOrder::BigEndian),
    );
}

#[test]
fn sums_and_products_from_words_into_the_tree() {
    // Terms added one by one while the sum is in words, then in the tree from the first that makes
    // it long: against the sum and the product term by term, for every length to 40 and terms that
    // leave the words early, late or never, and come back to them.
    let mut state = 0x5A55_0027_u64;
    for i in 0..size(400, 12) {
        let len = below(&mut state, 41) as usize;
        let terms: Vec<Rational> = (0..len)
            .map(|_| match below(&mut state, 4) {
                0 => boundary_rational(&mut state),
                1 => rational(&mut state, 24),
                _ => Rational::new(
                    Number::from(below(&mut state, 1 << (1 + i % 20)) as i64 - 3),
                    Number::from(1 + below(&mut state, 1 << (1 + i % 20))),
                ),
            })
            .collect();
        let (mut sum, mut product) = (Rational::default(), Rational::from(1u8));
        for t in &terms {
            sum += t;
            product *= t;
        }
        let s1: Rational = terms.iter().sum();
        let s2: Rational = terms.iter().cloned().sum();
        let p1: Rational = terms.iter().product();
        let p2: Rational = terms.iter().cloned().product();
        for (got, want) in [(&s1, &sum), (&s2, &sum), (&p1, &product), (&p2, &product)] {
            same(got, want);
        }
        if let Some(first) = terms.first() {
            assert!(s1.byte_order() == first.byte_order(), "sum: byte order");
            assert!(p2.byte_order() == first.byte_order(), "product: byte order");
        }
    }
}
