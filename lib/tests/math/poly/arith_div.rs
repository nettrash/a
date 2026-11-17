//! Division of `ModPolynomial`s by Newton's method, powers and compositions modulo a
//! polynomial, and evaluation, checked by their identities across the crossovers.

use super::common::*;
use a::math::numbers::Number;
use a::math::poly::ModPolynomial;

/// The moduli of these tests: the word and multi-limb primes, and composites whose units the
/// polynomials are made to lead with.
fn all_moduli() -> Vec<Number> {
    let mut v = moduli();
    if cfg!(miri) {
        // 3, 65537, 2^64 - 59, 2^64 + 13, 2^127 - 1, 2^128.
        v = [1, 3, 5, 6, 7, 10].iter().map(|&i| v[i].clone()).collect();
    }
    v.push(Number::from((1u64 << 63) - 25));
    v.push(Number::from(65_521));
    v
}

/// A random polynomial of exactly `len` coefficients modulo `n` whose constant (`at_top` false)
/// or leading coefficient is 1.
fn unit_poly(s: &mut u64, len: usize, n: &Number, at_top: bool) -> ModPolynomial {
    let bytes = n.to_str_radix(16).len() / 2 + 2;
    let mut c = coeffs(s, len, bytes);
    if len > 0 {
        let i = if at_top { len - 1 } else { 0 };
        c[i] = Number::from(1);
    }
    ModPolynomial::new(&c, n)
}

/// `x^k` modulo `n`.
fn xk(k: usize, n: &Number) -> ModPolynomial {
    let mut c = vec![Number::from(0); k + 1];
    c[k] = Number::from(1);
    ModPolynomial::new(&c, n)
}

#[test]
fn power_series_inverses() {
    let mut s = 0x1111_2222_3333_4444;
    for n in all_moduli() {
        for len in [1, 2, 5, size(97, 9), size(300, 12)] {
            for prec in [
                1,
                2,
                3,
                17,
                size(96, 10),
                size(97, 11),
                size(250, 20),
                size(1000, 30),
            ] {
                let f = unit_poly(&mut s, len, &n, false);
                let g = f.inv_series(prec);
                assert!(g.length() <= prec);
                let e = &f * &g;
                let mut low = e.coeffs();
                low.truncate(prec);
                let want: Vec<Number> =
                    trim((0..prec).map(|i| Number::from(u8::from(i == 0))).collect());
                assert_eq!(trim(low), want, "n = {n}, {len} to {prec}");
            }
        }
        assert!(ModPolynomial::one(&n).inv_series(0).is_zero());
    }
}

#[test]
#[should_panic(expected = "the constant coefficient is not a unit modulo n")]
fn a_series_without_a_unit_constant_panics() {
    let _ = ModPolynomial::from_u64(&[2, 1], 6).inv_series(5);
}

#[test]
#[should_panic(expected = "the leading coefficient is not a unit modulo n")]
fn division_by_a_non_unit_leading_coefficient_panics() {
    let _ = ModPolynomial::from_u64(&[1, 1, 1, 1], 6).div_rem(&ModPolynomial::from_u64(&[1, 3], 6));
}

#[test]
fn division_with_a_remainder() {
    let mut s = 0x5555_6666_7777_8888;
    let lens = [
        1,
        2,
        3,
        10,
        size(64, 8),
        size(65, 9),
        size(130, 14),
        size(700, 25),
    ];
    for n in all_moduli() {
        for &ld in &lens {
            for &la in &lens {
                let la = la + ld / 2;
                let a = mpoly(&mut s, la, &n);
                let d = unit_poly(&mut s, ld, &n, true);
                let (q, r) = a.div_rem(&d);
                assert!(r.length() < d.length(), "n = {n}, {la} by {ld}");
                assert_eq!(&(&q * &d) + &r, a, "n = {n}, {la} by {ld}");
                assert_eq!(&a / &d, q);
                assert_eq!(&a % &d, r);
            }
        }
    }
}

#[test]
fn products_and_powers_modulo_a_polynomial() {
    let mut s = 0x9999_AAAA_BBBB_CCCC;
    for n in all_moduli().into_iter().filter(|n| n.bits() <= 64) {
        products_and_powers_modulo(&mut s, &n, usize::MAX);
    }
}

#[test]
fn products_and_powers_modulo_a_polynomial_of_several_limbs() {
    // Apart from the word moduli, so that Miri on 32 bits, which gives every small value of the
    // transforms' portable lanes an address of its own, runs each in a process of its own.
    let mut s = 0x9999_AAAA_BBBB_CCCD;
    // On 32 bits Miri runs out of addresses from 3 coefficients in f: the products go by the
    // transforms on pieces, whose portable lanes make many small values, each given an address
    // of its own. The 64-bit targets take every length, and arith_limbs takes the same paths
    // (the pieces, the ready transforms of f and its inverse) on 32 bits.
    let max = if cfg!(all(miri, target_pointer_width = "32")) {
        2
    } else {
        usize::MAX
    };
    for n in all_moduli().into_iter().filter(|n| n.bits() > 64) {
        products_and_powers_modulo(&mut s, &n, max);
    }
}

/// `mul_mod` against the remainder of the product, `pow_mod` against repeated products, `e1 + e2`,
/// `x^(2e)` by shifts against `(x^2)^e` by windows, and `x^(e^3)`, modulo `n`, for `f` of up to
/// `max` coefficients.
fn products_and_powers_modulo(s: &mut u64, n: &Number, max: usize) {
    for lf in [1, 2, 3, size(40, 6), size(65, 9), size(200, 15)] {
        if lf > max {
            continue;
        }
        let f = unit_poly(s, lf, n, true);
        let a = mpoly(s, lf + 3, n);
        let b = mpoly(s, 2 * lf, n);
        let r = a.mul_mod(&b, &f);
        assert!(r.length() < f.length().max(1));
        assert_eq!(r, &(&a * &b) % &f, "n = {n}, {lf}");
        // a^e mod f against repeated products, and e1 + e2.
        let a = &a % &f;
        let mut want = &ModPolynomial::one(n) % &f;
        for e in 0..6u64 {
            assert_eq!(a.pow_mod(&Number::from(e), &f), want, "n = {n}, {lf}, ^{e}");
            want = want.mul_mod(&a, &f);
        }
        // Shorter exponents under Miri, whose 32-bit address space the products would fill.
        let cut = size(0, 40) as u32;
        let e1 = Number::from(xorshift(s) >> cut);
        let e2 = Number::from(xorshift(s) >> (3 + cut));
        let lhs = a.pow_mod(&(&e1 + &e2), &f);
        let rhs = a.pow_mod(&e1, &f).mul_mod(&a.pow_mod(&e2, &f), &f);
        assert_eq!(lhs, rhs, "n = {n}, {lf}, exponents");
        // x^e by shifts against the general powers.
        let x = xk(1, n);
        let e = Number::from(xorshift(s) >> cut);
        // (x^2)^e by windows against x^(2e) by shifts.
        let general = (&x * &x).pow_mod(&e, &f);
        assert_eq!(x.pow_mod(&(&e + &e), &f), general, "n = {n}, {lf}, x^2e");
        let big = &(&e * &e) * &e;
        assert_eq!(
            x.pow_mod(&big, &f),
            x.pow_mod(&e, &f).pow_mod(&(&e * &e), &f),
            "n = {n}, {lf}, x^(e^3)"
        );
    }
}

#[test]
fn frobenius_on_irreducible_polynomials() {
    // x^2 + 1 is irreducible modulo 7, x^3 + x + 1 modulo 2: x^(p^d) ≡ x.
    for (c, p, d) in [(&[1u64, 0, 1][..], 7u64, 2u32), (&[1, 1, 0, 1][..], 2, 3)] {
        let f = ModPolynomial::from_u64(c, p);
        let x = ModPolynomial::from_u64(&[0, 1], p);
        assert_eq!(x.pow_mod(&Number::from(p.pow(d)), &f), x);
    }
}

#[test]
fn compositions() {
    let mut s = 0x0DDB_A11C_AFE0_F00D;
    for n in all_moduli() {
        for (lp, lg) in [
            (1, 3),
            (3, 1),
            (5, 4),
            (size(40, 9), 3),
            (size(90, 12), size(20, 5)),
        ] {
            let p = mpoly(&mut s, lp, &n);
            let g = mpoly(&mut s, lg, &n);
            let c = p.compose(&g);
            for _ in 0..3 {
                let x = Number::from(xorshift(&mut s)).rem_euclid(&n);
                assert_eq!(c.eval(&x), p.eval(&g.eval(&x)), "n = {n}, {lp} of {lg}");
            }
            for lf in [1, 2, 7, size(50, 8)] {
                let f = unit_poly(&mut s, lf, &n, true);
                assert_eq!(
                    p.compose_mod(&g, &f),
                    &c % &f,
                    "n = {n}, {lp} of {lg} mod {lf}"
                );
            }
        }
    }
}

#[test]
fn evaluation_agrees_with_the_sum_of_terms() {
    let mut s = 0xFEED_FACE_CAFE_BEEF;
    for n in all_moduli() {
        for len in [0, 1, 2, 7, 8, 9, 33] {
            let p = mpoly(&mut s, len, &n);
            let x = Number::from(xorshift(&mut s)).rem_euclid(&n);
            assert_eq!(p.eval(&x), oracle_eval(&p.coeffs(), &x).rem_euclid(&n));
        }
    }
}

#[test]
fn powers_modulo_short_polynomials() {
    // Every length of f up to past the one-pass products on the stack (16 coefficients), each
    // power against repeated products by the division.
    let mut s = 0x6C8E_9CF5_70D8_2B4A;
    for n in [
        Number::from(7),
        Number::from(65_521),
        Number::from(u64::MAX - 58),
        Number::from((1u64 << 63) - 25),
    ] {
        for lf in 2..=size(18, 7) {
            let f = unit_poly(&mut s, lf, &n, true);
            let a = &mpoly(&mut s, lf + 2, &n) % &f;
            let mut want = &ModPolynomial::one(&n) % &f;
            for e in 0..size(12, 5) as u64 {
                assert_eq!(a.pow_mod(&Number::from(e), &f), want, "n = {n}, {lf}, ^{e}");
                want = &(&want * &a) % &f;
            }
            let x = xk(1, &n);
            let e = Number::from(xorshift(&mut s) >> 40);
            let mut slow = &ModPolynomial::one(&n) % &f;
            for i in (0..e.bits()).rev() {
                slow = &(&slow * &slow) % &f;
                if e.bit(i) {
                    slow = &(&slow * &x) % &f;
                }
            }
            assert_eq!(x.pow_mod(&e, &f), slow, "n = {n}, {lf}, x^e");
        }
    }
}
