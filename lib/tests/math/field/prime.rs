//! Prime fields: `a::math::field::{PrimeField, Fp}` against the arithmetic of `Number`, in every
//! form the fields choose, and against PARI/GP's Legendre symbols and square roots
//! (`data/prime_reference.txt`, written by `data/prime_reference.gp`).

use crate::allocations::count;
use a::math::ec::{Curve, named};
use a::math::field::{Fp, PrimeField};
use a::math::numbers::{ByteOrder, Number};
use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};

/// A deterministic generator of pseudo-random words (SplitMix64).
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number of up to `bits` random bits.
    pub fn number(&mut self, bits: u64) -> Number {
        let mut x = Number::default();
        let mut left = bits;
        while left > 0 {
            let take = left.min(64);
            let w = if take == 64 {
                self.next()
            } else {
                self.next() >> (64 - take)
            };
            x = (x << take as usize) + Number::from(w);
            left -= take;
        }
        x
    }
}

fn fields() -> Vec<&'static PrimeField> {
    vec![
        PrimeField::bn254_base(),
        PrimeField::bn254_scalar(),
        PrimeField::bls12_381_base(),
        PrimeField::bls12_381_scalar(),
    ]
}

#[test]
fn the_operations_agree_with_numbers_modulo_p() {
    let mut rng = Rng(24);
    let small = PrimeField::new(&Number::from(1_000_000_007u64));
    let two = PrimeField::new(&Number::from(2u8));
    let mut all = fields();
    all.push(&small);
    all.push(&two);
    for f in all {
        let p = f.modulus();
        for _ in 0..50 {
            let (x, y) = (rng.number(f.bits() + 8), rng.number(f.bits() + 8));
            let (a, b) = (f.element(&x), f.element(&y));
            let m = |v: Number| v.rem_euclid(p);
            assert_eq!((&a + &b).value(), m(&x + &y));
            assert_eq!((&a - &b).value(), m(&x - &y));
            assert_eq!((&a * &b).value(), m(&x * &y));
            assert_eq!((-&a).value(), m(-&x));
            if !b.is_zero() {
                assert_eq!(&(&a / &b) * &b, a);
                assert_eq!(&b.inverse().unwrap() * &b, f.one());
            }
            assert_eq!(a.square(), &a * &a);
            assert_eq!(a.double(), &a + &a);
            assert_eq!(a.pow_u64(5), &(&a * &a.square()) * &a.square());
        }
    }
}

#[test]
fn square_roots_and_the_legendre_symbol() {
    let mut rng = Rng(7);
    for f in fields() {
        let half = (f.modulus() - &Number::from(1u8)) >> 1usize;
        for _ in 0..30 {
            let a = f.element(&rng.number(f.bits()));
            let s = a.square();
            let r = s.sqrt().expect("a square has a root");
            assert_eq!(r.square(), s);
            assert!(r.value() <= half, "the smaller root");
            assert_eq!(s.legendre(), i32::from(!s.is_zero()));
            match a.legendre() {
                -1 => assert!(a.sqrt().is_none()),
                1 => assert!(a.sqrt().is_some()),
                _ => assert!(a.is_zero()),
            }
        }
    }
    let f = PrimeField::new(&Number::from(2u8));
    assert_eq!(f.one().sqrt(), Some(f.one()));
    assert_eq!(f.one().legendre(), 1);
}

#[test]
fn batch_inversion_leaves_zeros() {
    let f = PrimeField::bls12_381_scalar();
    let mut rng = Rng(3);
    let mut xs: Vec<_> = (0..20).map(|_| f.element(&rng.number(255))).collect();
    xs[0] = f.zero();
    xs[7] = f.zero();
    let inv = f.batch_invert(&xs);
    for (x, i) in xs.iter().zip(&inv) {
        if x.is_zero() {
            assert!(i.is_zero());
        } else {
            assert_eq!(x * i, f.one());
        }
    }
    assert!(f.batch_invert(&[]).is_empty());
}

#[test]
fn the_named_fields_are_their_primes() {
    let bits: Vec<u64> = fields().iter().map(|f| f.bits()).collect();
    assert_eq!(bits, [254, 254, 381, 255]);
}

#[test]
#[should_panic(expected = "is not a prime")]
fn a_composite_modulus_panics() {
    let _ = PrimeField::new(&Number::from(91u8));
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn elements_of_different_fields_do_not_mix() {
    let (f, g) = (PrimeField::bn254_base(), PrimeField::bn254_scalar());
    let _ = f.one() + g.one();
}

#[test]
#[should_panic(expected = "division by zero")]
fn division_by_zero_panics() {
    let f = PrimeField::bn254_base();
    let _ = f.one() / f.zero();
}

// Every form, at every length, and every method of square roots.

/// The number with the hexadecimal digits `s`.
fn hex(s: &str) -> Number {
    Number::from_str_radix(s, 16, ByteOrder::LittleEndian).unwrap()
}

/// `2^k`.
fn pow2(k: usize) -> Number {
    Number::from(1u8) << k
}

/// `x - c`, `x + c` for a word `c`.
fn minus(x: Number, c: u64) -> Number {
    x - Number::from(c)
}

fn plus(x: Number, c: u64) -> Number {
    x + Number::from(c)
}

/// `2^(bits - 1) + d 2^s + 1`, a prime with `p - 1 = q 2^s` found by `prime_reference.gp`.
fn two_adic(bits: usize, s: usize, d: u64) -> Number {
    pow2(bits - 1) + (Number::from(d) << s) + Number::from(1u8)
}

/// A prime, the form its field must choose and the method of its square roots.
struct Case {
    p: Number,
    form: &'static str,
    sqrt: &'static str,
}

fn case(p: Number, form: &'static str, sqrt: &'static str) -> Case {
    Case { p, form, sqrt }
}

/// Primes of every form: words, the largest prime below each limb boundary and the smallest above
/// it, the shapes `2^k - c` and the edges of their choice, the named primes, P-521, and numbers
/// of 577, 600 and 1500 bits.
fn cases() -> Vec<Case> {
    let m = |n: usize| -> &'static str {
        [
            "",
            "Montgomery, 1 limb",
            "Montgomery, 2 limbs",
            "Montgomery, 3 limbs",
            "Montgomery, 4 limbs",
            "Montgomery, 5 limbs",
            "Montgomery, 6 limbs",
            "Montgomery, 7 limbs",
            "Montgomery, 8 limbs",
            "Montgomery, 9 limbs",
        ][n]
    };
    let (three, atkin, ts, cip) = ("p = 3 mod 4", "Atkin", "Tonelli-Shanks", "Cipolla");
    let below = [59, 159, 237, 189, 197, 317, 203, 569, 789];
    let above = [13, 51, 133, 297, 27, 231, 211, 75, 243];
    let mut v = vec![
        case(Number::from(2u8), "GF(2)", "GF(2)"),
        case(Number::from(3u8), m(1), three),
        case(Number::from(5u8), m(1), atkin),
        case(Number::from(41u8), m(1), ts),
        case(Number::from(65_537u32), m(1), cip),
        case(Number::from(15u64 * (1 << 27) + 1), m(1), cip),
        case(Number::from(3u64 * (1 << 30) + 1), m(1), cip),
        case(minus(pow2(61), 1), m(1), three),
        case(Number::from(u64::MAX - (1 << 32) + 2), m(1), cip),
    ];
    let forms_below = [
        (m(1), atkin),
        (m(2), ts),
        (m(3), three),
        ("2^k - c, 4 limbs", three),
        (m(5), three),
        ("2^k - c, 6 limbs", three),
        (m(7), atkin),
        (m(8), three),
        ("2^k - c, 9 limbs", three),
    ];
    for (k, &(form, sqrt)) in forms_below.iter().enumerate() {
        v.push(case(minus(pow2(64 * (k + 1)), below[k]), form, sqrt));
    }
    let forms_above = [
        (m(2), atkin),
        (m(3), three),
        (m(4), atkin),
        (m(5), ts),
        (m(6), three),
        (m(7), three),
        (m(8), three),
        (m(9), three),
        ("Montgomery, more than 9 limbs", three),
    ];
    for (k, &(form, sqrt)) in forms_above.iter().enumerate() {
        v.push(case(plus(pow2(64 * (k + 1)), above[k]), form, sqrt));
    }
    v.extend([
        case(minus(pow2(383), 31), "2^k - c, 6 limbs", ts),
        case(minus(pow2(570), 261), "2^k - c, 9 limbs", three),
        case(two_adic(256, 3, 155), m(4), ts),
        case(two_adic(256, 50, 127), m(4), ts),
        case(two_adic(256, 56, 79), m(4), cip),
        case(two_adic(256, 64, 137), m(4), cip),
        case(two_adic(384, 4, 23), m(6), ts),
        case(two_adic(512, 5, 367), m(8), ts),
        case(two_adic(600, 20, 637), "Montgomery, more than 9 limbs", ts),
        case(two_adic(600, 90, 197), "Montgomery, more than 9 limbs", cip),
        case(
            plus(Number::from(3u8) << 598usize, 485),
            "Montgomery, more than 9 limbs",
            atkin,
        ),
        case(
            plus(Number::from(5u8) << 1497usize, 81),
            "Montgomery, more than 9 limbs",
            ts,
        ),
    ]);
    // The edges of ec's choice of the form 2^k - c, which takes c < 2^(62 - s) for s = 64 n - k:
    // for 4, 6 and 9 limbs and s = 0, 13, 32 and 55 (50 for six limbs, where no prime 2^329 - c
    // has c < 2^7), the largest c below the bound, whose folds by f = c 2^s come nearest 2^62, and
    // the smallest from it up, which takes Montgomery's form; at s = 0 that one fills the top limb,
    // so that Montgomery's reductions carry out of it. And 2^201 - 55 and 2^529 - 31, the least c
    // at s = 55 and s = 47. Found by PARI/GP (`isprime`), with the method of square roots by the
    // rules of the docs.
    let (p4, p6, p9) = ("2^k - c, 4 limbs", "2^k - c, 6 limbs", "2^k - c, 9 limbs");
    let edges: [(usize, u64, &str, &str); 26] = [
        (256, 4_611_686_018_427_387_443, p4, atkin),
        (256, 4_611_686_018_427_388_187, m(4), atkin),
        (243, 562_949_953_421_169, p4, three),
        (243, 562_949_953_421_629, m(4), three),
        (224, 1_073_741_493, p4, three),
        (224, 1_073_741_885, m(4), three),
        (201, 111, p4, ts),
        (201, 313, m(4), three),
        (201, 55, p4, ts),
        (384, 4_611_686_018_427_387_833, p6, three),
        (384, 4_611_686_018_427_387_923, m(6), atkin),
        (371, 562_949_953_421_065, p6, three),
        (371, 562_949_953_421_479, m(6), ts),
        (352, 1_073_741_819, p6, atkin),
        (352, 1_073_741_843, m(6), atkin),
        (334, 3735, p6, ts),
        (334, 4151, m(6), ts),
        (576, 4_611_686_018_427_387_833, p9, three),
        (576, 4_611_686_018_427_388_289, m(9), three),
        (563, 562_949_953_421_019, p9, atkin),
        (563, 562_949_953_421_365, m(9), three),
        (544, 1_073_741_739, p9, atkin),
        (544, 1_073_741_925, m(9), three),
        (521, 115, p9, atkin),
        (521, 271, m(9), ts),
        (529, 31, p9, ts),
    ];
    for (k, c, form, sqrt) in edges {
        v.push(case(minus(pow2(k), c), form, sqrt));
    }
    for (f, form, sqrt) in [
        (PrimeField::bn254_base(), m(4), three),
        (PrimeField::bn254_scalar(), m(4), ts),
        (PrimeField::bls12_381_base(), m(6), three),
        (PrimeField::bls12_381_scalar(), m(4), ts),
        (PrimeField::secp256k1_base(), "2^k - c, 4 limbs", three),
        (PrimeField::secp256k1_scalar(), m(4), ts),
        (PrimeField::p256_base(), "P-256", three),
        (PrimeField::p256_scalar(), m(4), ts),
        (PrimeField::p384_base(), "P-384", three),
        (PrimeField::p384_scalar(), m(6), three),
        (PrimeField::p521_base(), "P-521, limbs of 58 bits", three),
        (PrimeField::p521_scalar(), m(9), ts),
        (PrimeField::curve25519_base(), "2^k - c, 4 limbs", atkin),
        (PrimeField::curve25519_scalar(), m(4), atkin),
    ] {
        v.push(case(f.modulus().clone(), form, sqrt));
    }
    v
}

#[test]
fn each_prime_gets_its_form_and_its_method_of_square_roots() {
    let mut forms = HashSet::new();
    let mut methods = HashSet::new();
    for c in cases() {
        let f = PrimeField::new(&c.p);
        assert_eq!(f.form(), c.form, "the form of {}", c.p);
        assert_eq!(f.sqrt_method(), c.sqrt, "the square roots of {}", c.p);
        forms.insert(c.form);
        methods.insert(c.sqrt);
    }
    // Montgomery's nine lengths, three of 2^k - c, P-256, P-384, P-521, Number and GF(2).
    assert_eq!(forms.len(), 17);
    assert_eq!(methods.len(), 5);
}

/// The number `x` stands for, as the tests compute it.
fn m(x: &Number, p: &Number) -> Number {
    x.rem_euclid(p)
}

/// The smaller of `r` and `p - r`.
fn smaller(r: &Number, p: &Number) -> Number {
    let other = m(&(p - r), p);
    if other < *r { other } else { r.clone() }
}

/// The Legendre symbol by `Number`, which knows nothing of the fields.
fn jacobi(x: &Number, p: &Number) -> i32 {
    if *p == 2u8 {
        i32::from(!m(x, p).is_zero())
    } else {
        x.jacobi(p)
    }
}

/// How many random elements to try in a field of `bits` bits, in a debug build.
fn rounds(bits: u64) -> usize {
    match bits {
        0..=300 => 40,
        301..=600 => 16,
        _ => 4,
    }
}

/// Checks every operation of `f` on `n` random pairs against `Number` modulo `p`.
fn check_operations(f: &PrimeField, rng: &mut Rng, n: usize) {
    let p = f.modulus();
    let one = Number::from(1u8);
    let half = (p - &one) >> 1usize;
    for round in 0..n {
        // Inputs of either sign and up to 70 bits longer than p, and some at the edges.
        let mut x = rng.number(f.bits() + 70);
        let extra = rng.next() % 70;
        let mut y = rng.number(f.bits() + extra);
        if rng.next() & 1 == 1 {
            x = -x;
        }
        match round {
            0 => y = p - &one,
            1 => x = p.clone(),
            2 => y = Number::from(1u8),
            _ => {}
        }
        let (a, b) = (f.element(&x), f.element(&y));
        let (xm, ym) = (m(&x, p), m(&y, p));
        assert_eq!(a.value(), xm, "element of {x} mod {p}");
        assert_eq!((&a + &b).value(), m(&(&x + &y), p));
        assert_eq!((&a - &b).value(), m(&(&x - &y), p));
        assert_eq!((&b - &a).value(), m(&(&y - &x), p));
        assert_eq!((&a * &b).value(), m(&(&x * &y), p));
        assert_eq!((-&a).value(), m(&-&x, p));
        assert_eq!(a.square().value(), m(&(&x * &x), p));
        assert_eq!(a.double().value(), m(&(&x + &x), p));
        assert_eq!(a.is_zero(), xm.is_zero());
        assert_eq!(a.is_one(), xm == one);
        assert_eq!(a == b, xm == ym);
        match ym.mod_inverse(p) {
            Some(inv) => {
                assert_eq!(b.inverse().unwrap().value(), inv);
                assert_eq!((&a / &b).value(), m(&(&xm * &inv), p));
            }
            None => {
                assert!(ym.is_zero());
                assert!(b.inverse().is_none());
            }
        }
        let w = rng.next();
        assert_eq!(a.pow_u64(w).value(), xm.pow_mod(&Number::from(w), p));
        assert_eq!(f.from_u64(w).value(), m(&Number::from(w), p));
        let e = rng.number(f.bits() + 3);
        assert_eq!(a.pow(&e).value(), xm.pow_mod(&e, p));
        // The Legendre symbol, squares and roots.
        let l = jacobi(&xm, p);
        assert_eq!(a.legendre(), l, "the symbol of {xm} mod {p}");
        assert_eq!(a.is_square(), l >= 0);
        match a.sqrt() {
            Some(r) => {
                assert!(l >= 0);
                assert_eq!(m(&(&r.value() * &r.value()), p), xm);
                assert!(r.value() <= half || *p == 2u8, "the smaller root");
            }
            None => assert_eq!(l, -1, "{xm} has a root mod {p}"),
        }
        let s = a.square();
        assert_eq!(s.sqrt().unwrap().value(), smaller(&xm, p));
        assert_eq!(a.sgn0(), xm.bit(0));
    }
}

#[test]
fn every_form_agrees_with_numbers_modulo_p() {
    let mut rng = Rng(2024);
    for c in cases() {
        let f = PrimeField::new(&c.p);
        check_operations(&f, &mut rng, rounds(f.bits()));
    }
}

/// The same as `every_form_agrees_with_numbers_modulo_p`, with 25 times as many inputs, about 25
/// seconds: `cargo test --test math -- --ignored field::prime`.
#[test]
#[ignore]
fn every_form_agrees_with_numbers_modulo_p_on_many_inputs() {
    let mut rng = Rng(99);
    for c in cases() {
        let f = PrimeField::new(&c.p);
        check_operations(&f, &mut rng, 25 * rounds(f.bits()));
    }
}

#[test]
fn primes_given_in_either_byte_order_make_the_same_field() {
    let mut rng = Rng(15);
    for c in cases().into_iter().step_by(2) {
        let mut be = c.p.clone();
        if be.byte_order == ByteOrder::LittleEndian {
            be.value.reverse();
            be.byte_order = ByteOrder::BigEndian;
        }
        let (f, g) = (PrimeField::new(&c.p), PrimeField::new(&be));
        assert_eq!(g.form(), c.form);
        let x = rng.number(f.bits() + 10);
        let y = rng.number(f.bits());
        let (a, b) = (g.element(&x), g.element(&y));
        assert_eq!((&a * &b).value(), (f.element(&x) * f.element(&y)).value());
        assert_eq!((&a - &b).value(), (f.element(&x) - f.element(&y)).value());
        assert_eq!(a.to_bytes_be(), f.element(&x).to_bytes_be());
        assert_eq!(
            a.sqrt().map(|r| r.value()),
            f.element(&x).sqrt().map(|r| r.value())
        );
        assert_eq!(a.legendre(), f.element(&x).legendre());
        if !b.is_zero() {
            assert_eq!(
                b.inverse().unwrap().value(),
                f.element(&y).inverse().unwrap().value()
            );
        }
        // Elements of the two fields are of the same field.
        assert_eq!(&a + &f.element(&y), f.element(&(&x + &y)));
        check_operations(&g, &mut rng, 2);
    }
}

/// The elements of the fields of up to nine limbs hold their limbs inline: their arithmetic, the
/// comparisons, the hashes and the conversions from words and bytes allocate nothing. (Powers and
/// square roots keep their tables of powers in vectors, inversions and the Legendre symbol work
/// on a number, and `value` and `to_bytes_*` return one.)
#[test]
fn the_arithmetic_of_up_to_nine_limbs_does_not_allocate() {
    let mut rng = Rng(66);
    for c in cases() {
        let f = PrimeField::new(&c.p);
        let a = f.element(&rng.number(f.bits()));
        let b = f.element(&rng.number(f.bits()));
        let bytes = a.to_bytes_be();
        let (_, n) = count(|| {
            let mut x = &a + &b;
            x = &x - &a;
            x = &x * &b;
            x *= &a;
            x += &b;
            x -= &a;
            x = -x;
            x = x.square().double();
            let y = f.from_u64(12345) + f.one() + f.zero();
            let z = f.from_bytes_be(&bytes).unwrap();
            let w = f.from_bytes_le(&bytes);
            let eq = x == y && z == a && w.is_some() && x.is_zero() && y.is_one() && z.sgn0();
            let mut h = DefaultHasher::new();
            x.hash(&mut h);
            (eq, h.finish(), x.clone())
        });
        if f.bits() > 576 {
            assert!(n > 0, "a long field's elements are numbers");
        } else {
            assert_eq!(n, 0, "allocations in {}", f.form());
        }
    }
}

#[test]
fn gf2_and_gf3_have_their_tables() {
    for p in [2u64, 3] {
        let f = PrimeField::new(&Number::from(p));
        for x in 0..p {
            let a = f.from_u64(x);
            assert_eq!(a.value(), Number::from(x));
            assert_eq!((-&a).value(), Number::from((p - x) % p));
            assert_eq!(
                a.inverse().map(|i| i.value()),
                (x != 0).then(|| Number::from(x))
            );
            // The squares are 0 and 1; 2 is not one modulo 3.
            let l = [0, 1, -1][x as usize];
            assert_eq!(a.legendre(), l);
            assert_eq!(
                a.sqrt().map(|r| r.value()),
                (l >= 0).then(|| Number::from(x))
            );
            for y in 0..p {
                let b = f.from_u64(y);
                assert_eq!((&a + &b).value(), Number::from((x + y) % p));
                assert_eq!((&a - &b).value(), Number::from((x + p - y) % p));
                assert_eq!((&a * &b).value(), Number::from(x * y % p));
                if y != 0 {
                    // Every nonzero element of GF(2) and GF(3) is its own inverse.
                    assert_eq!((&a / &b).value(), Number::from(x * y % p));
                }
            }
            for e in 0..5 {
                assert_eq!(a.pow_u64(e).value(), Number::from(x.pow(e as u32) % p));
            }
        }
        assert_eq!(f.from_u64(u64::MAX).value(), Number::from(u64::MAX % p));
        assert_eq!(
            f.element(&Number::from(-7i8)).value(),
            Number::from(7 * (p - 1) % p)
        );
    }
}

#[test]
fn small_primes_have_the_roots_and_symbols_of_their_squares() {
    // Every class of p modulo 8, and s from 1 to 12, and 16 (Cipolla for 65537).
    let primes = [
        3u64, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 73, 97, 113, 193, 257, 641, 769, 7681,
        12289, 13313, 18433,
    ];
    for &p in primes.iter().chain(&[65_537]) {
        let f = PrimeField::new(&Number::from(p));
        let mut root = vec![u64::MAX; p as usize];
        for r in (0..p).rev() {
            root[(r * r % p) as usize] = r;
        }
        // The exhaustive check, sampled for the longest prime.
        let step = if p > 20_000 { 97 } else { 1 };
        for x in (0..p).step_by(step) {
            let a = f.from_u64(x);
            let expected = root[x as usize];
            match a.sqrt() {
                Some(r) => assert_eq!(r.value(), Number::from(expected), "sqrt {x} mod {p}"),
                None => assert_eq!(expected, u64::MAX, "{x} is a square mod {p}"),
            }
            let l = if x == 0 {
                0
            } else if expected == u64::MAX {
                -1
            } else {
                1
            };
            assert_eq!(a.legendre(), l, "({x} / {p})");
        }
    }
}

#[test]
fn square_roots_by_every_method_at_every_length() {
    let mut rng = Rng(31);
    for c in cases() {
        let f = PrimeField::new(&c.p);
        let p = f.modulus();
        let n = if f.bits() > 600 { 2 } else { 6 };
        for _ in 0..n {
            let x = m(&rng.number(f.bits()), p);
            let a = f.element(&x);
            let s = a.square();
            assert_eq!(
                s.sqrt().unwrap().value(),
                smaller(&x, p),
                "{} mod {p}",
                c.sqrt
            );
            // A non-square: a times a fixed non-square, when there is one.
            if *p != 2u8 && !x.is_zero() {
                let mut z = Number::from(2u8);
                while z.jacobi(p) != -1 {
                    z += Number::from(1u8);
                }
                let ns = &s * &f.element(&z);
                assert!(ns.sqrt().is_none());
                assert_eq!(ns.legendre(), -1);
            }
        }
        // -1 is a square exactly when p = 1 (mod 4).
        let minus_one = -f.one();
        assert_eq!(
            minus_one.sqrt().is_some(),
            *p == 2u8 || p.bit(0) && !p.bit(1)
        );
    }
}

#[test]
fn the_reference_values_of_pari() {
    let text = include_str!("data/prime_reference.txt");
    let mut lines = text.lines();
    let (mut primes, mut symbols_read, mut squares) = (0, 0, 0);
    while let Some(head) = lines.next() {
        let parts: Vec<&str> = head.split(' ').collect();
        assert_eq!(parts[0], "p");
        let p = hex(parts[1]);
        let n: u64 = parts[2].parse().unwrap();
        let symbols = lines.next().unwrap().strip_prefix("l ").unwrap().as_bytes();
        symbols_read += symbols.len();
        let f = PrimeField::new(&p);
        let (third, seventh) = (&p / &Number::from(3u8), &p / &Number::from(7u8));
        for k in 0..n {
            let i = Number::from(k);
            let cube = &(&i * &i) * &i;
            let x = m(
                &(&(&third * &cube) + &(&(&seventh * &i) + &Number::from(1u8))),
                &p,
            );
            let a = f.element(&x);
            let l = match symbols[k as usize] {
                b'+' => 1,
                b'-' => -1,
                _ => 0,
            };
            assert_eq!(a.legendre(), l, "({x} / {p})");
            assert_eq!(x.jacobi(&p), l);
            let root = lines.next().unwrap();
            if root == "-" {
                assert!(a.sqrt().is_none());
            } else {
                assert_eq!(a.sqrt().unwrap().value(), hex(root), "sqrt {x} mod {p}");
                squares += 1;
            }
        }
        primes += 1;
    }
    assert_eq!((primes, symbols_read, squares), (40, 896, 489));
}

#[test]
fn powers_of_every_length_and_sign() {
    let mut rng = Rng(5);
    for c in cases().into_iter().step_by(3) {
        let f = PrimeField::new(&c.p);
        let p = f.modulus();
        let one = Number::from(1u8);
        let x = m(&rng.number(f.bits()), p);
        let a = f.element(&x);
        // Exponents at the edges, and longer than p, which are reduced modulo p - 1.
        let long = rng.number(3 * f.bits() + 100);
        for e in [
            Number::from(0u8),
            one.clone(),
            Number::from(2u8),
            p - &one,
            p - &Number::from(2u8),
            p.clone(),
            p + &one,
            long.clone(),
            &long * &(p - &one),
        ] {
            assert_eq!(a.pow(&e).value(), x.pow_mod(&e, p), "{x}^{e} mod {p}");
            assert_eq!(f.zero().pow(&e).value(), Number::from(0u8).pow_mod(&e, p));
            if !x.is_zero() {
                let inv = x.mod_inverse(p).unwrap();
                assert_eq!(a.pow(&-&e).value(), inv.pow_mod(&e, p), "{x}^-{e} mod {p}");
            }
        }
        assert_eq!(f.zero().pow_u64(0), f.one());
        assert_eq!(f.zero().pow_u64(3), f.zero());
    }
}

#[test]
#[should_panic(expected = "0 has no inverse")]
fn a_negative_power_of_zero_panics() {
    let f = PrimeField::p256_base();
    let _ = f.zero().pow(&Number::from(-1i8));
}

#[test]
fn the_operators_take_every_combination_of_operands() {
    let mut rng = Rng(8);
    for c in cases().into_iter().step_by(4) {
        let f = PrimeField::new(&c.p);
        let a = f.element(&rng.number(f.bits()));
        let mut b = f.element(&rng.number(f.bits()));
        if b.is_zero() {
            b = f.one();
        }
        let sum = &a + &b;
        assert_eq!(a.clone() + b.clone(), sum);
        assert_eq!(&a + b.clone(), sum);
        assert_eq!(a.clone() + &b, sum);
        let mut t = a.clone();
        t += &b;
        assert_eq!(t, sum);
        let mut t = a.clone();
        t += b.clone();
        assert_eq!(t, sum);
        let diff = &a - &b;
        assert_eq!(a.clone() - b.clone(), diff);
        assert_eq!(&a - b.clone(), diff);
        assert_eq!(a.clone() - &b, diff);
        let mut t = a.clone();
        t -= &b;
        assert_eq!(t, diff);
        let mut t = a.clone();
        t -= b.clone();
        assert_eq!(t, diff);
        let prod = &a * &b;
        assert_eq!(a.clone() * b.clone(), prod);
        assert_eq!(&a * b.clone(), prod);
        assert_eq!(a.clone() * &b, prod);
        let mut t = a.clone();
        t *= &b;
        assert_eq!(t, prod);
        let mut t = a.clone();
        t *= b.clone();
        assert_eq!(t, prod);
        let quot = &a / &b;
        assert_eq!(&quot * &b, a);
        assert_eq!(a.clone() / b.clone(), quot);
        assert_eq!(&a / b.clone(), quot);
        assert_eq!(a.clone() / &b, quot);
        let mut t = a.clone();
        t /= &b;
        assert_eq!(t, quot);
        let mut t = a.clone();
        t /= b.clone();
        assert_eq!(t, quot);
        assert_eq!(-a.clone(), -&a);
        assert_eq!(&(-&a) + &a, f.zero());
    }
}

/// The hash of `x`.
fn hash_of(x: &Fp<'_>) -> u64 {
    let mut h = DefaultHasher::new();
    x.hash(&mut h);
    h.finish()
}

// Clippy sees a cell behind an element, the lazy part of the `Divisor` in the field of a long
// prime; the hash is of the value alone.
#[allow(clippy::mutable_key_type)]
#[test]
fn equal_elements_hash_alike_whatever_their_representation() {
    let mut rng = Rng(12);
    for c in cases() {
        let f = PrimeField::new(&c.p);
        let p = f.modulus();
        let x = m(&rng.number(f.bits()), p);
        let a = f.element(&x);
        // Reached by different paths: the forms 2^k - c and P-521 keep elements that are not
        // reduced, which must still compare and hash as the value.
        let b = f.element(&(&x + p));
        let c2 = &(&a + &f.element(&(p - &Number::from(1u8)))) + &f.one();
        let d = &(&a * &f.from_u64(3)) - &(&a + &a);
        let e = -&(-&a);
        for y in [&b, &c2, &d, &e] {
            assert_eq!(*y, a);
            assert_eq!(hash_of(y), hash_of(&a));
            assert_eq!(y.value(), x);
        }
        let zero = &(&a - &b) * &f.from_u64(5);
        assert!(zero.is_zero());
        assert_eq!(zero, f.zero());
        assert_eq!(hash_of(&zero), hash_of(&f.zero()));
        let one = &(&a + &f.one()) - &a;
        assert!(one.is_one());
        let set: HashSet<Fp<'_>> = [a.clone(), b, c2, d, e, f.zero(), zero]
            .into_iter()
            .collect();
        assert_eq!(set.len(), if x.is_zero() { 1 } else { 2 });
    }
}

/// Sums that reach `p` or pass it without a carry out of the limbs: the forms `2^k - c` and
/// P-521's keep such values unreduced (`(p - 1) + 1` is held as `p`), and every reading of an
/// element must see the value.
/// What allocates, as the docs say: an inversion (`inverse`, `/`, `/=`) twice, the value as a
/// `Number` and the inverse `Number::mod_inverse` returns; the Legendre symbol once, the value, up
/// to six limbs, where `Number::jacobi` works on the stack (beyond, it may take working memory);
/// a reduced decoding three times, the bytes as a number, its remainder and the remainder's limbs,
/// and an empty one not at all. `GF(2)` allocates in none of them.
#[test]
fn inversions_symbols_and_reduced_decodings_allocate_as_documented() {
    let mut rng = Rng(67);
    for c in cases() {
        let f = PrimeField::new(&c.p);
        if f.bits() > 576 {
            continue;
        }
        let a = f.element(&rng.number(f.bits()));
        let b = f.element(&(rng.number(f.bits()) + Number::from(1u8)));
        let b = if b.is_zero() { f.one() } else { b };
        let exact = a.to_bytes_be();
        let long: Vec<u8> = (0..2 * f.byte_len() + 7)
            .map(|_| rng.next() as u8)
            .collect();
        let two = *f.modulus() == 2u8;
        let (inv, n) = count(|| b.inverse());
        assert_eq!(n, if two { 0 } else { 2 }, "inverse in {}", f.form());
        assert_eq!(inv.unwrap() * &b, f.one());
        let (_, n) = count(|| &a / &b);
        assert_eq!(n, if two { 0 } else { 2 }, "/ in {}", f.form());
        let (_, n) = count(|| {
            let mut x = a.clone();
            x /= &b;
            x
        });
        assert_eq!(n, if two { 0 } else { 2 }, "/= in {}", f.form());
        for (_, n) in [count(|| a.legendre()), count(|| a.is_square() as i32)] {
            if two {
                assert_eq!(n, 0);
            } else if f.bits() <= 384 {
                assert_eq!(n, 1, "the Legendre symbol in {}", f.form());
            } else {
                assert!(n >= 1, "the Legendre symbol in {}", f.form());
            }
        }
        for bytes in [&exact[..], &long[..]] {
            let (_, n) = count(|| f.from_bytes_be_reduced(bytes));
            assert_eq!(n, if two { 1 } else { 3 }, "reduced in {}", f.form());
            let (_, n) = count(|| f.from_bytes_le_reduced(bytes));
            assert_eq!(n, if two { 1 } else { 3 }, "reduced in {}", f.form());
        }
        let (z, n) = count(|| f.from_bytes_be_reduced(&[]));
        assert!(z.is_zero());
        assert_eq!(n, 0);
    }
}

#[test]
fn unreduced_representations_read_as_their_values() {
    for c in cases() {
        let f = PrimeField::new(&c.p);
        let p = f.modulus();
        let top = f.element(&(p - &Number::from(1u8)));
        let len = f.byte_len();
        let zeros = [
            &top + &f.one(),
            &(&top + &top) + &f.from_u64(2),
            &(&top + &f.from_u64(2)) - &f.one(),
            -&(&top + &f.one()),
            &(&top + &f.one()) * &top,
        ];
        let zero = f.zero();
        for z in &zeros {
            assert!(z.is_zero(), "{} in {}", z, f.form());
            assert_eq!(*z, zero);
            assert_eq!(hash_of(z), hash_of(&zero));
            assert_eq!(z.value(), Number::from(0u8));
            assert!(!z.sgn0());
            assert_eq!(z.to_bytes_le(), vec![0; len]);
            assert_eq!(z.legendre(), 0);
            assert_eq!(z.sqrt(), Some(f.zero()));
            assert!(z.inverse().is_none());
        }
        let one = &top + &f.from_u64(2);
        assert!(one.is_one(), "{}", f.form());
        assert_eq!(one, f.one());
        assert_eq!(hash_of(&one), hash_of(&f.one()));
        assert!(one.sgn0());
        assert_eq!(one.inverse(), Some(f.one()));
        let minus_one = &f.zero() - &f.one();
        assert_eq!(minus_one, top);
        assert_eq!(minus_one.value(), p - &Number::from(1u8));
        assert_eq!(f.from_bytes_be(&minus_one.to_bytes_be()), Some(top.clone()));
        // `2 (p - 1) = p - 2`, which reaches past p in the forms with room above it.
        let twice = top.double();
        assert_eq!(twice.value(), m(&(p - &Number::from(2u8)), p));
        assert_eq!(twice.sgn0(), m(&(p - &Number::from(2u8)), p).bit(0));
        assert_eq!(&twice * &twice, f.from_u64(4));
        assert_eq!(
            twice.square().sqrt().unwrap().value(),
            smaller(&twice.value(), p)
        );
    }
}

#[test]
fn fields_of_the_same_prime_mix() {
    let p = PrimeField::bn254_base().modulus().clone();
    let (f, g) = (PrimeField::new(&p), PrimeField::new(&p));
    let x = f.from_u64(5) + g.from_u64(6);
    assert_eq!(x, g.from_u64(11));
    assert_eq!(x, PrimeField::bn254_base().from_u64(11));
    let h = f.clone();
    assert_eq!(h.from_u64(2) * f.from_u64(3), g.from_u64(6));
    // Elements of different primes are never equal.
    assert_ne!(PrimeField::bn254_scalar().one(), f.one());
}

#[test]
fn bytes_round_trip_in_both_orders() {
    let mut rng = Rng(77);
    for c in cases() {
        let f = PrimeField::new(&c.p);
        let p = f.modulus();
        let len = f.byte_len();
        assert_eq!(len, p.bits().div_ceil(8) as usize);
        for _ in 0..4 {
            let x = m(&rng.number(f.bits()), p);
            let a = f.element(&x);
            let be = a.to_bytes_be();
            let le = a.to_bytes_le();
            assert_eq!(be.len(), len);
            assert_eq!(le.len(), len);
            assert_eq!(Number::new(be.clone(), ByteOrder::BigEndian), x);
            assert_eq!(Number::new(le.clone(), ByteOrder::LittleEndian), x);
            let mut rev = be.clone();
            rev.reverse();
            assert_eq!(rev, le);
            assert_eq!(f.from_bytes_be(&be), Some(a.clone()));
            assert_eq!(f.from_bytes_le(&le), Some(a.clone()));
            // Any length, reduced.
            for n in [0, 1, len - 1, len, len + 1, 2 * len, 64] {
                let bytes: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
                let be = f.from_bytes_be_reduced(&bytes);
                let le = f.from_bytes_le_reduced(&bytes);
                if n == 0 {
                    assert!(be.is_zero() && le.is_zero());
                    continue;
                }
                assert_eq!(
                    be.value(),
                    m(&Number::new(bytes.clone(), ByteOrder::BigEndian), p)
                );
                assert_eq!(
                    le.value(),
                    m(&Number::new(bytes, ByteOrder::LittleEndian), p)
                );
            }
        }
        // The edges: p - 1 is an element, p and the largest value of the length are not, and
        // neither is a shorter or longer encoding.
        let top = f.element(&(p - &Number::from(1u8)));
        assert_eq!(f.from_bytes_be(&top.to_bytes_be()), Some(top.clone()));
        let mut p_be = p.value.clone();
        if p.byte_order == ByteOrder::LittleEndian {
            p_be.reverse();
        }
        while p_be.len() < len {
            p_be.insert(0, 0);
        }
        assert_eq!(Number::new(p_be.clone(), ByteOrder::BigEndian), *p);
        let mut p_le = p_be.clone();
        p_le.reverse();
        assert!(f.from_bytes_be(&p_be).is_none(), "p itself");
        assert!(f.from_bytes_le(&p_le).is_none(), "p itself");
        assert!(f.from_bytes_be(&vec![0xff; len]).is_none());
        assert!(f.from_bytes_be(&vec![0; len - 1]).is_none());
        assert!(f.from_bytes_be(&vec![0; len + 1]).is_none());
        assert!(f.from_bytes_le(&[]).is_none());
        assert_eq!(f.from_bytes_le(&vec![0; len]), Some(f.zero()));
        assert_eq!(f.zero().to_bytes_be(), vec![0; len]);
    }
}

#[test]
fn batch_inversion_in_every_form() {
    let mut rng = Rng(41);
    for c in cases() {
        let f = PrimeField::new(&c.p);
        let n = if f.bits() > 600 { 3 } else { 9 };
        let mut xs: Vec<Fp<'_>> = (0..n).map(|_| f.element(&rng.number(f.bits()))).collect();
        xs[1] = f.zero();
        xs.push(f.zero());
        let inv = f.batch_invert(&xs);
        assert_eq!(inv.len(), xs.len());
        for (x, i) in xs.iter().zip(&inv) {
            match x.inverse() {
                Some(y) => assert_eq!(*i, y),
                None => assert!(i.is_zero()),
            }
        }
        let zeros = f.batch_invert(&[f.zero(), f.zero()]);
        assert!(zeros.iter().all(Fp::is_zero));
        let single = f.batch_invert(&[f.from_u64(1)]);
        assert_eq!(single, [f.one()]);
    }
}

#[test]
fn the_named_fields_are_those_of_the_curves() {
    let checks: [(&PrimeField, &PrimeField, &Number, &Number); 4] = [
        (
            PrimeField::secp256k1_base(),
            PrimeField::secp256k1_scalar(),
            named::secp256k1().modulus(),
            named::secp256k1().order(),
        ),
        (
            PrimeField::p256_base(),
            PrimeField::p256_scalar(),
            named::p256().modulus(),
            named::p256().order(),
        ),
        (
            PrimeField::p384_base(),
            PrimeField::p384_scalar(),
            named::p384().modulus(),
            named::p384().order(),
        ),
        (
            PrimeField::p521_base(),
            PrimeField::p521_scalar(),
            named::p521().modulus(),
            named::p521().order(),
        ),
    ];
    for (base, scalar, p, n) in checks {
        assert_eq!(base.modulus(), p);
        assert_eq!(scalar.modulus(), n);
    }
    assert_eq!(
        PrimeField::curve25519_base().modulus(),
        named::curve25519().modulus()
    );
    assert_eq!(
        PrimeField::curve25519_scalar().modulus(),
        named::curve25519().order()
    );
    assert_eq!(
        PrimeField::curve25519_base().modulus(),
        named::ed25519().modulus()
    );
    assert_eq!(
        PrimeField::curve25519_scalar().modulus(),
        named::ed25519().order()
    );
    let bytes: Vec<usize> = [
        PrimeField::secp256k1_base(),
        PrimeField::p256_base(),
        PrimeField::p384_base(),
        PrimeField::p521_base(),
        PrimeField::curve25519_base(),
        PrimeField::bn254_base(),
        PrimeField::bls12_381_base(),
    ]
    .iter()
    .map(|f| f.byte_len())
    .collect();
    assert_eq!(bytes, [32, 32, 48, 66, 32, 32, 48]);
    // BN254's and BLS12-381's from their definitions: 36u^4 + 36u^3 + 24u^2 + 6u + 1 (and the
    // order with 18u^2 for 24u^2) for u = 4965661367192848881; and (x - 1)^2 (x^4 - x^2 + 1) / 3
    // + x with the order x^4 - x^2 + 1 for x = -0xd201000000010000.
    let u = Number::from(4_965_661_367_192_848_881u64);
    let u2 = &u * &u;
    let u3 = &u2 * &u;
    let u4 = &u3 * &u;
    let bn = |c: u64| {
        &(&(&(&Number::from(36u8) * &u4) + &(&Number::from(36u8) * &u3))
            + &(&Number::from(c) * &u2))
            + &(&(&Number::from(6u8) * &u) + &Number::from(1u8))
    };
    assert_eq!(PrimeField::bn254_base().modulus(), &bn(24));
    assert_eq!(PrimeField::bn254_scalar().modulus(), &bn(18));
    let x = -Number::from(0xd201_0000_0001_0000u64);
    let x2 = &x * &x;
    let r = &(&(&x2 * &x2) - &x2) + &Number::from(1u8);
    let xm1 = &x - &Number::from(1u8);
    let q = &(&(&(&xm1 * &xm1) * &r) / &Number::from(3u8)) + &x;
    assert_eq!(PrimeField::bls12_381_base().modulus(), &q);
    assert_eq!(PrimeField::bls12_381_scalar().modulus(), &r);
}

#[test]
fn the_field_reports_its_size() {
    let f = PrimeField::new(&Number::from(65_521u32));
    assert_eq!(f.characteristic(), &Number::from(65_521u32));
    assert_eq!(f.order(), f.modulus());
    assert_eq!(f.bits(), 16);
    assert_eq!(f.byte_len(), 2);
    let two = PrimeField::new(&Number::from(2u8));
    assert_eq!((two.bits(), two.byte_len()), (2, 1));
    assert_eq!(format!("{:?}", f.from_u64(70_000)), "4479 (mod 65521)");
    assert_eq!(format!("{}", f.from_u64(70_000)), "4479");
    let debug = format!("{f:?}");
    assert!(
        debug.contains("65521") && debug.contains("Montgomery, 1 limb"),
        "{debug}"
    );
    // An element and its field.
    let x = f.from_u64(3);
    assert!(core::ptr::eq(x.field(), &f));
}

#[test]
fn signs_of_rfc_9380() {
    // sgn0 is the parity of the value: 0 and p - 1 are even, 1 and (p + 1) / 2 odd (for these p).
    for f in [
        PrimeField::p256_base(),
        PrimeField::bls12_381_base(),
        PrimeField::p521_base(),
    ] {
        assert!(!f.zero().sgn0());
        assert!(f.one().sgn0());
        assert!(!(-f.one()).sgn0());
        let half = (f.modulus() + &Number::from(1u8)) >> 1usize;
        assert_eq!(f.element(&half).sgn0(), half.bit(0));
        for k in 1..20u64 {
            let x = f.from_u64(k);
            assert_ne!(x.sgn0(), (-&x).sgn0(), "x and -x have opposite signs");
        }
    }
}

#[test]
#[should_panic(expected = "is not a prime")]
fn one_is_not_a_prime_modulus() {
    let _ = PrimeField::new(&Number::from(1u8));
}

#[test]
#[should_panic(expected = "is not a prime")]
fn zero_is_not_a_prime_modulus() {
    let _ = PrimeField::new(&Number::from(0u8));
}

#[test]
#[should_panic(expected = "is not a prime")]
fn a_negative_modulus_panics() {
    let _ = PrimeField::new(&Number::from(-7i8));
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn a_product_of_different_fields_panics() {
    let _ = PrimeField::p256_base().one() * PrimeField::p256_scalar().one();
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn a_difference_of_different_fields_panics() {
    let _ = PrimeField::p521_base().one() - PrimeField::p521_scalar().one();
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn a_quotient_of_different_fields_panics() {
    let _ = PrimeField::secp256k1_base().one() / PrimeField::curve25519_base().one();
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn an_assignment_from_a_different_field_panics() {
    let mut x = PrimeField::bls12_381_base().one();
    x *= PrimeField::bls12_381_scalar().one();
}

#[test]
#[should_panic(expected = "elements of different fields")]
fn a_batch_with_an_element_of_a_different_field_panics() {
    let f = PrimeField::bn254_base();
    let _ = f.batch_invert(&[f.one(), PrimeField::bn254_scalar().one()]);
}

#[test]
fn zero_has_no_inverse_in_any_form() {
    for c in cases() {
        let f = PrimeField::new(&c.p);
        assert!(f.zero().inverse().is_none());
        assert!(f.element(&c.p).inverse().is_none());
    }
}

#[test]
#[should_panic(expected = "division by zero")]
fn a_division_by_zero_panics_in_a_long_field() {
    let f = PrimeField::new(&plus(Number::from(3u8) << 598usize, 485));
    let _ = f.from_u64(3) / f.zero();
}

#[test]
#[should_panic(expected = "division by zero")]
fn a_division_assignment_by_zero_panics() {
    let f = PrimeField::curve25519_base();
    let mut x = f.from_u64(3);
    x /= f.element(f.modulus());
}

#[test]
fn the_in_place_operations_equal_the_others_in_every_form() {
    // Every form: GF(2), Montgomery of 1 to 9 limbs, 2^k - c, the NIST primes, P-521, and the wide
    // Montgomery form beyond nine limbs.
    let mut fields: Vec<PrimeField> = [
        "2",
        "3",
        "65537",
        "18446744069414584321",
        "340282366920938463463374607431768211297",
    ]
    .iter()
    .map(|d| PrimeField::new(&Number::from_str_radix(d, 10, ByteOrder::LittleEndian).unwrap()))
    .collect();
    for f in [
        PrimeField::bn254_base(),
        PrimeField::bls12_381_base(),
        PrimeField::secp256k1_base(),
        PrimeField::p256_base(),
        PrimeField::p384_base(),
        PrimeField::p521_base(),
        PrimeField::curve25519_base(),
    ] {
        fields.push(f.clone());
    }
    fields.push(PrimeField::new(
        &(Number::from(1u8) << 607usize).next_prime(),
    ));
    fields.push(PrimeField::new(
        &(Number::from(1u8) << 1279usize).next_prime(),
    ));
    for f in &fields {
        let p = f.modulus();
        for seed in 0..40u64 {
            let x = f.element(
                &(&Number::from(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                    * &(p - &Number::from(seed + 1))),
            );
            let (mut s, mut d, mut n) = (x.clone(), x.clone(), x.clone());
            s.square_in_place();
            d.double_in_place();
            n.neg_in_place();
            assert_eq!(s, x.square(), "{}", f.form());
            assert_eq!(d, x.double(), "{}", f.form());
            assert_eq!(n, -&x, "{}", f.form());
            assert_eq!(s.value(), (&x.value() * &x.value()).rem_euclid(p));
        }
    }
}

#[test]
fn the_in_place_operations_do_not_allocate_up_to_nine_limbs() {
    let f = PrimeField::bls12_381_base();
    let mut x = f.from_u64(1234567);
    let ((), n) = count(|| {
        x.square_in_place();
        x.double_in_place();
        x.neg_in_place();
    });
    assert_eq!(n, 0);
}

#[test]
fn long_fields_keep_decoded_values_until_an_operation() {
    // Beyond nine limbs an element made from its value keeps it: the encodings, the value and the
    // hash agree with an element made by arithmetic, and every operation on it (as either operand,
    // by reference and in place) agrees with Number.
    let one = Number::from(1u8);
    let primes: Vec<Number> = if cfg!(miri) {
        vec![&(&one << 607usize) - &one] // a Mersenne prime of ten limbs
    } else {
        [600usize, 1024, 2048]
            .iter()
            .map(|&b| (&one << b).next_prime())
            .collect()
    };
    for p in primes {
        let f = PrimeField::new(&p);
        let len = f.byte_len();
        for s in 1..6u64 {
            let x = (&Number::from(s.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                * &(&p - &Number::from(s)))
                .rem_euclid(&p);
            let y = (&(&x * &x) + &Number::from(s)).rem_euclid(&p);
            let mut bytes = x.value.clone();
            if x.byte_order == ByteOrder::BigEndian {
                bytes.reverse();
            }
            bytes.resize(len.max(bytes.len()), 0);
            assert!(bytes[len..].iter().all(|&b| b == 0));
            bytes.truncate(len);
            bytes.reverse();
            let a = f.from_bytes_be(&bytes).unwrap();
            assert_eq!(a.to_bytes_be(), bytes);
            assert_eq!(a.value(), x);
            let made = &(&f.element(&x) * &f.one()) + &f.zero();
            assert_eq!(a, made);
            let hash = |v: &Fp| {
                let mut h = DefaultHasher::new();
                v.hash(&mut h);
                h.finish()
            };
            assert_eq!(hash(&a), hash(&made));
            assert_eq!(made.to_bytes_be(), bytes);
            let c = f.element(&y);
            assert_eq!((&a * &c).value(), (&x * &y).rem_euclid(&p));
            assert_eq!((&c * &a).value(), (&x * &y).rem_euclid(&p));
            assert_eq!((&a + &c).value(), (&x + &y).rem_euclid(&p));
            assert_eq!((&c - &a).value(), (&y - &x).rem_euclid(&p));
            let mut d = f.from_bytes_be(&bytes).unwrap();
            d *= &c;
            assert_eq!(d.value(), (&x * &y).rem_euclid(&p));
            let mut d = f.from_bytes_be(&bytes).unwrap();
            d += &c;
            d -= &a;
            assert_eq!(d.value(), y);
            let mut d = f.from_u64(s);
            d.square_in_place();
            assert_eq!(d.value(), Number::from(s * s));
            assert_eq!(a.inverse().unwrap().value(), x.mod_inverse(&p).unwrap());
            assert_eq!(a.square().value(), (&x * &x).rem_euclid(&p));
        }
    }
}
