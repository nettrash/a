//! Products of polynomials over the integers (Kronecker substitution with signs) and modulo
//! integers of several limbs, against the schoolbook oracle.

use super::common::*;
use a::math::numbers::{ByteOrder, Number};
use a::math::poly::{ModPolynomial, Polynomial};

/// A random polynomial of exactly `len` coefficients of up to `bytes` bytes, the top one not zero.
pub(super) fn exact_poly(s: &mut u64, len: usize, bytes: usize) -> Polynomial<Number> {
    let mut c = coeffs(s, len, bytes);
    if let Some(t) = c.last_mut()
        && t.is_zero()
    {
        *t = if xorshift(s) & 1 == 0 {
            Number::from(1)
        } else {
            Number::from(-1)
        };
    }
    Polynomial::new(c)
}

#[test]
fn integer_products_agree_with_the_oracle() {
    let mut s = 0xDEAD_BEEF_0BAD_F00D;
    let lens = [1, 2, 3, 4, 7, 16, 33];
    for bytes in [1, 4, 8, 9, 16, 33, 130] {
        for &la in &lens {
            for &lb in &lens {
                if la * lb * bytes > size(20_000, 300) {
                    continue;
                }
                let a = exact_poly(&mut s, la, bytes);
                let b = exact_poly(&mut s, lb, bytes / 2 + 1);
                let p = &a * &b;
                assert_eq!(
                    p.coeffs(),
                    oracle_mul(a.coeffs(), b.coeffs()),
                    "{la} by {lb}, {bytes} bytes"
                );
                let q = &a * &a.clone();
                assert_eq!(
                    q.coeffs(),
                    oracle_mul(a.coeffs(), a.coeffs()),
                    "square of {la}"
                );
                assert_eq!(a.sqr(), q);
            }
        }
    }
}

#[test]
fn integer_products_with_borrows_across_fields() {
    // Coefficients of alternating signs and at the bound of the field, whose fields borrow.
    let big = Number::from_str_radix(&"f".repeat(64), 16, ByteOrder::BigEndian).unwrap();
    for len in [2, 3, 5, 17, size(64, 8)] {
        let c: Vec<Number> = (0..len)
            .map(|i| if i % 2 == 0 { big.clone() } else { -&big })
            .collect();
        let a = Polynomial::new(c.clone());
        let b = Polynomial::new(c.iter().rev().cloned().collect());
        assert_eq!((&a * &b).coeffs(), oracle_mul(a.coeffs(), b.coeffs()));
        assert_eq!(a.sqr().coeffs(), oracle_mul(a.coeffs(), a.coeffs()));
        let ones = Polynomial::new(vec![Number::from(-1); len]);
        assert_eq!((&ones * &a).coeffs(), oracle_mul(ones.coeffs(), a.coeffs()));
        // A product whose middle coefficients vanish: (x^n - 1)(x^n + 1).
        let mut m = vec![Number::from(0); len + 1];
        m[0] = Number::from(-1);
        m[len] = Number::from(1);
        let mut p = m.clone();
        p[0] = Number::from(1);
        let prod = &Polynomial::new(m) * &Polynomial::new(p);
        let mut want = vec![Number::from(0); 2 * len + 1];
        want[0] = Number::from(-1);
        want[2 * len] = Number::from(1);
        assert_eq!(prod.coeffs(), want);
    }
}

#[test]
fn integer_products_in_both_byte_orders_and_with_zeros() {
    let mut s = 0x1357_9BDF_2468_ACE0;
    for _ in 0..size(30, 4) {
        let a = exact_poly(&mut s, 9, 12);
        let b = exact_poly(&mut s, 6, 20);
        // The same coefficients big-endian.
        let be = |p: &Polynomial<Number>| {
            p.map(|c| {
                let mut x =
                    Number::from_str_radix(&c.abs().to_str_radix(16), 16, ByteOrder::BigEndian)
                        .unwrap();
                if c.is_negative() {
                    x = -x;
                }
                x
            })
        };
        let want = &a * &b;
        assert_eq!(&be(&a) * &be(&b), want);
        assert_eq!(&be(&a) * &b, want);
        // Zero coefficients inside.
        let mut c = a.coeffs().to_vec();
        for i in (0..c.len() - 1).step_by(2) {
            c[i] = Number::from(0);
        }
        let z = Polynomial::new(c);
        assert_eq!((&z * &b).coeffs(), oracle_mul(z.coeffs(), b.coeffs()));
    }
}

#[test]
fn long_integer_products_by_evaluation() {
    let mut s = 0x0F0F_F0F0_1234_4321;
    for (len, bytes) in [
        (size(2000, 40), 8),
        (size(500, 20), 130),
        (size(60, 5), 1300),
    ] {
        let a = exact_poly(&mut s, len, bytes);
        let b = exact_poly(&mut s, len / 2 + 3, bytes);
        let p = &a * &b;
        assert_eq!(p.length(), a.length() + b.length() - 1);
        for x in [
            Number::from(1),
            Number::from(-1),
            Number::from(2),
            Number::from(-3),
        ] {
            assert_eq!(p.eval(&x), &a.eval(&x) * &b.eval(&x), "{len} at {x}");
        }
        assert_eq!(a.sqr().eval(&Number::from(-2)), {
            let v = a.eval(&Number::from(-2));
            &v * &v
        });
    }
}

#[test]
fn multi_limb_products_agree_with_the_oracle() {
    let mut s = 0xA5A5_5A5A_C3C3_3C3C;
    for n in moduli().into_iter().filter(|n| n.bits() > 64) {
        for (la, lb) in [(1, 1), (2, 3), (5, 5), (6, 7), (7, 7), (12, 30), (40, 41)] {
            let a = mpoly(&mut s, la, &n);
            let b = mpoly(&mut s, lb, &n);
            let want = oracle_mod(&oracle_mul(&a.coeffs(), &b.coeffs()), &n);
            assert_eq!((&a * &b).coeffs(), want, "n = {n}, {la} by {lb}");
            let sq = oracle_mod(&oracle_mul(&a.coeffs(), &a.coeffs()), &n);
            assert_eq!((&a * &a.clone()).coeffs(), sq, "n = {n}, square of {la}");
        }
        // All coefficients n - 1: every sum at its bound.
        let top = &n - &Number::from(1);
        for len in [3, 6, 7, 30] {
            let a = ModPolynomial::new(&vec![top.clone(); len], &n);
            let want = oracle_mod(&oracle_mul(&a.coeffs(), &a.coeffs()), &n);
            assert_eq!((&a * &a).coeffs(), want, "n = {n}, {len}");
        }
    }
}

#[test]
fn long_multi_limb_products_by_evaluation() {
    let mut s = 0x7777_1111_9999_3333;
    for n in moduli().into_iter().filter(|n| n.bits() > 64) {
        let len = size(600, 30);
        let a = mpoly(&mut s, len, &n);
        let b = mpoly(&mut s, len, &n);
        if a.is_zero() || b.is_zero() {
            continue;
        }
        super::arith::check_long(&a, &b, &mut s);
    }
}

#[test]
fn integer_products_by_the_transforms_agree_with_the_oracle() {
    // From 48 coefficients and up to 112 bits a coefficient, with sums below 2^224: the
    // transforms on the coefficients, the signed theorem; past those bounds Kronecker again.
    let mut s = 0x0123_3210_4567_7654;
    for bytes in [1, 4, 8, 9, 12, 13, 14, 15] {
        for (la, lb) in [
            (size(48, 5), size(48, 5)),
            (size(70, 6), size(50, 5)),
            (size(64, 6), size(130, 7)),
        ] {
            let a = exact_poly(&mut s, la, bytes);
            let b = exact_poly(&mut s, lb, bytes);
            assert_eq!(
                (&a * &b).coeffs(),
                oracle_mul(a.coeffs(), b.coeffs()),
                "{la} by {lb}, {bytes} bytes"
            );
            assert_eq!(
                a.sqr().coeffs(),
                oracle_mul(a.coeffs(), a.coeffs()),
                "square, {bytes} bytes"
            );
        }
    }
    // Every coefficient at the top of its size, all of one sign or alternating: the sums at
    // their bound, either sign.
    for bits in [31u32, 63, 64, 100, 111, 112] {
        let top = &Number::from(2).pow(bits) - &Number::from(1);
        for alternate in [false, true] {
            let c: Vec<Number> = (0..size(64, 6))
                .map(|i| {
                    if alternate && i % 2 == 1 {
                        -&top
                    } else {
                        top.clone()
                    }
                })
                .collect();
            let a = Polynomial::new(c);
            let neg = -&a;
            assert_eq!(
                (&a * &neg).coeffs(),
                oracle_mul(a.coeffs(), neg.coeffs()),
                "{bits} bits"
            );
            assert_eq!(
                a.sqr().coeffs(),
                oracle_mul(a.coeffs(), a.coeffs()),
                "{bits} bits"
            );
        }
    }
}
