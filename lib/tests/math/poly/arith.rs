//! The products of `ModPolynomial`s modulo a word: the schoolbook method, Kronecker substitution
//! and the transforms, across their crossovers, against the oracle of the integers reduced, and
//! long ones by evaluation at random points and by cutting an operand.

use super::common::*;
use a::math::numbers::Number;
use a::math::poly::ModPolynomial;

/// Word moduli at the edges of the methods: tiny, even and composite, at the bounds of the sums
/// (2^32, 2^63) and at the top of a word.
pub(super) fn word_moduli() -> Vec<u64> {
    if cfg!(miri) {
        return vec![2, 6, 65_521, (1 << 32) + 15, (1 << 63) - 25, u64::MAX];
    }
    vec![
        2,
        3,
        6,
        7,
        1024,
        65_521,
        (1 << 28) - 57,
        (1 << 31) - 1,
        (1 << 32) - 5,
        (1 << 32) + 15,
        1_000_000_007,
        (1 << 50) - 27,
        (1 << 62) - 57,
        (1 << 63) - 25,
        (1 << 63) + 29,
        u64::MAX - 58,
        u64::MAX,
    ]
}

/// Random residues modulo `n`, `len` of them, the top one not zero, with runs of 0 and of
/// `n - 1`.
pub(super) fn residues(s: &mut u64, len: usize, n: u64) -> Vec<u64> {
    let mut v: Vec<u64> = (0..len)
        .map(|_| match below(s, 8) {
            0 => 0,
            1 => n - 1,
            _ => below(s, n),
        })
        .collect();
    if let Some(t) = v.last_mut()
        && *t == 0
    {
        *t = 1;
    }
    v
}

/// The polynomial of the residues.
pub(super) fn wpoly(c: &[u64], n: u64) -> ModPolynomial {
    ModPolynomial::from_u64(c, n)
}

/// The oracle's product of `a` and `b` modulo `n`.
fn oracle(a: &ModPolynomial, b: &ModPolynomial) -> Vec<Number> {
    oracle_mod(&oracle_mul(&a.coeffs(), &b.coeffs()), a.modulus())
}

#[test]
fn word_products_agree_with_the_oracle() {
    let mut s = 0x9E37_79B9_7F4A_7C15;
    let lens = [1, 2, 3, 5, 8, 9, 12, 13, 16, 31, 40, 64, 100];
    for n in word_moduli() {
        for &la in &lens {
            for &lb in &lens {
                if la * lb > size(4000, 64) {
                    continue;
                }
                let a = wpoly(&residues(&mut s, la, n), n);
                let b = wpoly(&residues(&mut s, lb, n), n);
                let p = &a * &b;
                assert_eq!(p.coeffs(), oracle(&a, &b), "n = {n}, {la} by {lb}");
                let q = &a * &a.clone();
                assert_eq!(q.coeffs(), oracle(&a, &a), "n = {n}, square of {la}");
            }
        }
    }
}

#[test]
fn word_products_of_all_maximal_residues() {
    // Every coefficient n - 1: each sum of the product at its bound.
    for n in word_moduli() {
        for len in [1, 7, 8, 9, size(33, 12), size(300, 20), size(2500, 24)] {
            let a = wpoly(&vec![n - 1; len], n);
            let p = &a * &a;
            assert_eq!(p.coeffs(), oracle(&a, &a), "n = {n}, {len}");
            let b = wpoly(&vec![n - 1; len / 2 + 1], n);
            assert_eq!((&a * &b).coeffs(), oracle(&a, &b), "n = {n}, {len}");
        }
    }
}

/// `a·b` checked at random points against `a(x) b(x)` and against the sum of the products of
/// the halves of `a` by `b`.
pub(super) fn check_long(a: &ModPolynomial, b: &ModPolynomial, s: &mut u64) {
    let n = a.modulus().clone();
    let p = a * b;
    let lead = (&a.leading().unwrap() * &b.leading().unwrap()).rem_euclid(&n);
    if !lead.is_zero() {
        assert_eq!(p.length(), a.length() + b.length() - 1, "the leading terms");
    }
    for _ in 0..3 {
        let x = Number::from(xorshift(s)).rem_euclid(&n);
        let want = (&a.eval(&x) * &b.eval(&x)).rem_euclid(&n);
        assert_eq!(p.eval(&x), want, "at a point, n = {n}");
    }
    // Cut a at h: a = lo + x^h hi.
    let h = a.length() / 3 + 1;
    let ca = a.coeffs();
    let lo = a.with_coeffs(&ca[..h.min(ca.len())]);
    let hi = a.with_coeffs(ca.get(h..).unwrap_or(&[]));
    let mut shifted = vec![Number::from(0); h];
    shifted.extend((&hi * b).coeffs());
    let want = &(&lo * b) + &a.with_coeffs(&shifted);
    assert_eq!(p, want, "by halves, n = {n}");
}

#[test]
fn long_word_products_through_the_transforms() {
    let mut s = 0x0123_4567_89AB_CDEF;
    let lens = [
        size(1000, 24),
        size(2048, 30),
        size(2049, 36),
        size(4000, 48),
        size(9000, 64),
    ];
    for n in word_moduli() {
        for &la in &lens {
            for &lb in &[la, la / 3 + 1, 17] {
                let a = wpoly(&residues(&mut s, la, n), n);
                let b = wpoly(&residues(&mut s, lb, n), n);
                check_long(&a, &b, &mut s);
            }
            // Squares against products: (a + b)^2 = a^2 + 2ab + b^2.
            let a = wpoly(&residues(&mut s, la, n), n);
            let b = wpoly(&residues(&mut s, la - 1, n), n);
            let sum = &a + &b;
            let two = a.with_coeffs(&[Number::from(2)]);
            let want = &(&(&a * &a) + &(&two * &(&a * &b))) + &(&b * &b);
            assert_eq!(&sum * &sum, want, "squares, n = {n}, {la}");
        }
    }
}

#[test]
fn products_of_the_zero_polynomial_and_constants() {
    for n in word_moduli() {
        let z = ModPolynomial::from_u64(&[], n);
        let a = wpoly(&[1, 2, 3], n);
        assert!((&z * &a).is_zero());
        assert!((&a * &z).is_zero());
        assert!((&z * &z).is_zero());
        let c = wpoly(&[n - 1], n);
        assert_eq!((&c * &a).coeffs(), oracle(&c, &a));
    }
    // Zero divisors: modulo 6, (2x + 2)(3x + 3) = 0.
    let p = &wpoly(&[2, 2], 6) * &wpoly(&[3, 3], 6);
    assert!(p.is_zero());
    // Modulo 2^10, the top coefficient vanishes: (32x + 1)(32x + 1) = 64x + 1.
    let q = wpoly(&[1, 32], 1024);
    assert_eq!((&q * &q).word_coeffs(), Some(&[1, 64][..]));
}

#[test]
fn products_longer_than_the_transforms_take() {
    // Past 2^21 coefficients of the product: the longer operand cut into pieces for a short
    // one, Kronecker substitution when both are long; checked at points, in release builds
    // only (the transforms of 2^21 points are slow unoptimized).
    if cfg!(debug_assertions) || cfg!(miri) {
        return;
    }
    let mut s = 0x2222_4444_6666_8888;
    let n = (1u64 << 63) - 25;
    let a = wpoly(&residues(&mut s, 1_200_000, n), n);
    let b = wpoly(&residues(&mut s, 1_000_000, n), n);
    let p = &a * &b;
    assert_eq!(p.length(), 2_199_999);
    // A long operand by a short one: the long one in pieces through the transforms.
    let long = wpoly(&residues(&mut s, 2_100_000, n), n);
    let short = wpoly(&residues(&mut s, 100_000, n), n);
    let ls = &long * &short;
    for _ in 0..2 {
        let x = Number::from(xorshift(&mut s) % n);
        let want = (&long.eval(&x) * &short.eval(&x)).rem_euclid(a.modulus());
        assert_eq!(ls.eval(&x), want);
    }
    // Both long (by Kronecker substitution), and a square of 1.1 million coefficients.
    let c = wpoly(&residues(&mut s, 1_100_000, n), n);
    let q = &a * &c;
    let sq = &c * &c;
    for _ in 0..2 {
        let x = Number::from(xorshift(&mut s) % n);
        let (ax, cx) = (a.eval(&x), c.eval(&x));
        assert_eq!(q.eval(&x), (&ax * &cx).rem_euclid(a.modulus()));
        assert_eq!(sq.eval(&x), (&cx * &cx).rem_euclid(a.modulus()));
    }
    for _ in 0..2 {
        let x = Number::from(xorshift(&mut s) % n);
        let want = (&a.eval(&x) * &b.eval(&x)).rem_euclid(a.modulus());
        assert_eq!(p.eval(&x), want);
    }
}
