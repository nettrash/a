//! The core of `ModPolynomial`: residues of one word and of several limbs, against the oracle of
//! the polynomials over the integers reduced modulo `n`.

use super::common::*;
use a::math::numbers::Number;
use a::math::poly::{ModPolynomial, Polynomial};

#[test]
fn coefficients_are_reduced_residues() {
    let n = Number::from(7);
    let p = ModPolynomial::new(&[Number::from(-1), Number::from(15), Number::from(14)], &n);
    assert_eq!(p.coeffs(), [Number::from(6), Number::from(1)]);
    assert_eq!(p.word_coeffs(), Some(&[6, 1][..]));
    assert_eq!(p.degree(), Some(1));
    assert_eq!(p.modulus(), &n);
    assert_eq!(p.coeff(5), Number::from(0));
    assert_eq!(p.leading(), Some(Number::from(1)));
    assert!(p.is_monic());
    assert_eq!(p.to_symmetric(), ints(&[-1, 1]));
    assert!(ModPolynomial::zero(&n).is_zero());
    assert!(ModPolynomial::one(&n).is_one());
    assert_eq!(ModPolynomial::x(&n), ModPolynomial::from_u64(&[0, 1], 7));
    assert_eq!(
        ModPolynomial::from_u64(&[7, 14], 7),
        ModPolynomial::zero(&n)
    );
    let big = Number::from_str_radix(
        "18446744073709551629",
        10,
        a::math::numbers::ByteOrder::BigEndian,
    )
    .unwrap();
    let q = ModPolynomial::new(&[Number::from(-1)], &big);
    assert_eq!(q.coeffs(), [&big - &Number::from(1)]);
    assert_eq!(q.word_coeffs(), None);
    assert_eq!(
        q.to_string(),
        "18446744073709551628 (mod 18446744073709551629)"
    );
    assert_eq!(
        ModPolynomial::from_polynomial(&ints(&[3, -2]), &n).to_string(),
        "5*x + 3 (mod 7)"
    );
}

#[test]
#[should_panic(expected = "the modulus must be at least 2")]
fn a_modulus_of_one_panics() {
    let _ = ModPolynomial::from_u64(&[1], 1);
}

#[test]
#[should_panic(expected = "the polynomials have different moduli")]
fn different_moduli_panic() {
    let _ = &ModPolynomial::from_u64(&[1], 5) + &ModPolynomial::from_u64(&[1], 7);
}

#[test]
fn arithmetic_agrees_with_the_integers_reduced() {
    let mut s = 0x2468_1357_ACE0_BDF9;
    for n in moduli() {
        for _ in 0..size(40, 3) {
            let a = mpoly(&mut s, 24, &n);
            let b = mpoly(&mut s, 24, &n);
            let (za, zb) = (a.to_polynomial(), b.to_polynomial());
            let want = oracle_mod(oracle_mul(za.coeffs(), zb.coeffs()).as_slice(), &n);
            let c = &a * &b;
            assert_eq!(c.coeffs(), want, "mod {n:?}");
            assert_eq!(a.clone() * b.clone(), c);
            let mut t = a.clone();
            t *= &b;
            assert_eq!(t, c);
            assert_eq!((&a + &b).coeffs(), oracle_mod((&za + &zb).coeffs(), &n));
            assert_eq!((&a - &b).coeffs(), oracle_mod((&za - &zb).coeffs(), &n));
            assert_eq!((-&a).coeffs(), oracle_mod((-&za).coeffs(), &n));
            assert!((&a - &a).is_zero());
            assert_eq!(&(-&a) + &a, ModPolynomial::zero(&n));
            let x = number(&mut s, 20);
            assert_eq!(a.eval(&x), oracle_eval(za.coeffs(), &x).rem_euclid(&n));
            assert_eq!(
                a.derivative().coeffs(),
                oracle_mod(za.derivative().coeffs(), &n)
            );
            let k = Number::from(-3);
            assert_eq!(a.scale(&k).coeffs(), oracle_mod(za.scale(&k).coeffs(), &n));
            assert_eq!(a.pow(3), &(&a * &a) * &a);
            assert_eq!(a.to_symmetric().coeffs().len(), a.length());
        }
    }
}

#[test]
fn compositions_agree_with_evaluation() {
    let mut s = 0x8642_9753_1357_2468;
    for n in moduli() {
        for _ in 0..size(10, 2) {
            let a = mpoly(&mut s, 6, &n);
            let g = mpoly(&mut s, 5, &n);
            let x = number(&mut s, 10);
            assert_eq!(a.compose(&g).eval(&x), a.eval(&g.eval(&x)));
        }
    }
}

#[test]
fn division_and_greatest_common_divisors_modulo_primes() {
    let mut s = 0x1111_2222_3333_4444;
    for p in primes() {
        for _ in 0..size(30, 3) {
            let a = mpoly(&mut s, 20, &p);
            let mut d = mpoly(&mut s, 10, &p);
            while d.is_zero() {
                d = mpoly(&mut s, 10, &p);
            }
            let (q, r) = a.div_rem(&d);
            assert!(r.length() < d.length());
            assert_eq!(&(&q * &d) + &r, a, "mod {p:?}");
            assert_eq!(&a / &d, q);
            assert_eq!(&a % &d, r);
            let mut t = a.clone();
            t %= &d;
            assert_eq!(t, r);
            assert!(d.to_monic().is_monic());
            // gcd(a·g, b·g) is the monic g times the monic gcd of a and b.
            let b = mpoly(&mut s, 8, &p);
            let g = d.to_monic();
            let (x, y) = (&a * &g, &b * &g);
            let h = x.gcd(&y);
            if !x.is_zero() || !y.is_zero() {
                assert!(h.is_monic());
                assert!((&x % &h).is_zero() && (&y % &h).is_zero());
                assert!((&h % &g).is_zero());
                let (h2, sx, ty) = x.xgcd(&y);
                assert_eq!(h2, h);
                assert_eq!(&(&sx * &x) + &(&ty * &y), h);
            }
        }
    }
}

#[test]
fn powers_modulo_a_polynomial() {
    let mut s = 0xABCD_EF01_2345_6789;
    for p in primes() {
        for _ in 0..size(10, 2) {
            let a = mpoly(&mut s, 8, &p);
            let mut f = mpoly(&mut s, 6, &p);
            while f.is_zero() {
                f = mpoly(&mut s, 6, &p);
            }
            for e in [0u64, 1, 2, 5, 17] {
                assert_eq!(a.pow_mod(&Number::from(e), &f), &a.pow(e) % &f);
            }
        }
    }
    // x^(p^2) ≡ x and x^p ≡ -x modulo the irreducible x² + 1 over F_p for p ≡ 3 (mod 4), of one
    // word and of two limbs (2^127 - 1).
    let mersenne = Number::from(1) << 127usize;
    let mersenne = &mersenne - &Number::from(1);
    for p in [
        Number::from(3),
        Number::from(7),
        Number::from(1_000_000_007),
        mersenne,
    ] {
        let f = ModPolynomial::new(&[Number::from(1), Number::from(0), Number::from(1)], &p);
        let x = ModPolynomial::x(&p);
        assert_eq!(x.pow_mod(&(&p * &p), &f), x);
        assert_eq!(x.pow_mod(&p, &f), -&x);
    }
}

#[test]
fn the_polynomial_over_the_integers_round_trips() {
    let mut s = 0x4242_4242_1313_1313;
    for n in moduli() {
        let a = mpoly(&mut s, 12, &n);
        assert_eq!(ModPolynomial::from_polynomial(&a.to_polynomial(), &n), a);
        assert_eq!(ModPolynomial::from_polynomial(&a.to_symmetric(), &n), a);
        assert_eq!(a.with_coeffs(&a.coeffs()), a);
        let z: Polynomial<Number> = a.to_polynomial();
        assert!(z.coeffs().iter().all(|c| !c.is_negative() && c < &n));
    }
}
