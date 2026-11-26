//! The pseudo-division past the Kronecker crossover by the schoolbook method with the fewest
//! factors of the leading coefficient (the fuzz of the merged tree: exact divisions, a monomial
//! divisor, a common factor), against its definition `l^k a = q d + r` and the rationals' quotient;
//! the general case falling back to the substitution after a few factors.

use super::arith_z::exact_poly;
use super::common::*;
use a::math::numbers::Number;
use a::math::poly::Polynomial;
use a::math::rational::Rational;

/// Checks the pseudo-division of `a` by `d` against its definition and the rationals.
fn check(a: &Polynomial<Number>, d: &Polynomial<Number>) {
    let (q, r) = a.pseudo_div_rem(d);
    let k = (a.length() - d.length() + 1) as u32;
    let lk = Polynomial::constant(d.leading().unwrap().pow(k));
    assert_eq!(&(&q * d) + &r, &lk * a);
    assert!(r.length() < d.length());
    let (qq, rr) = Polynomial::<Rational>::from(a).div_rem(&Polynomial::<Rational>::from(d));
    let lkq = Polynomial::<Rational>::from(&lk);
    assert_eq!(Polynomial::<Rational>::from(&q), &lkq * &qq);
    assert_eq!(Polynomial::<Rational>::from(&r), &lkq * &rr);
}

/// `2^bits + 1`, a leading coefficient that divides nothing by chance.
fn wide(bits: u32) -> Number {
    &Number::from(2).pow(bits) + &Number::from(1)
}

#[test]
fn exact_and_nearly_exact_pseudo_divisions_past_the_substitution() {
    let mut s = 0x0123_4567_89AB_CDEF;
    // (lq, ld, bytes of d): 384 coefficients in the quotient, or 32 by a 1,000-bit divisor.
    for (lq, ld, bytes) in [(size(400, 5), 6, 2), (size(40, 4), size(12, 3), 130)] {
        let mut d = exact_poly(&mut s, ld, bytes);
        let mut c = d.coeffs().to_vec();
        *c.last_mut().unwrap() = if bytes > 8 { -wide(1030) } else { wide(40) };
        d = Polynomial::new(c);
        let q = exact_poly(&mut s, lq, 1);
        let a = &d * &q;
        // Exact: the quotient l^k q, no remainder.
        let (pq, pr) = a.pseudo_div_rem(&d);
        let lk = Polynomial::constant(d.leading().unwrap().pow(lq as u32));
        assert_eq!(pq, &lk * &q);
        assert!(pr.is_zero());
        check(&a, &d);
        // A few low coefficients off: one or two steps need a factor of l.
        let mut c = a.coeffs().to_vec();
        c[0] = &c[0] + &Number::from(3);
        c[1] = &c[1] - &Number::from(1);
        check(&Polynomial::new(c), &d);
        // A random dividend: the general case.
        let b = exact_poly(&mut s, a.length(), 3);
        check(&b, &d);
    }
}

#[test]
fn pseudo_division_by_a_monomial_and_by_a_common_factor() {
    let mut s = 0x5A5A_A5A5_0F0F_F0F0;
    // a of small coefficients by l x^4 (the fuzz case 3004203): one factor of l in all.
    let a = exact_poly(&mut s, size(1279, 12), 6);
    let mut c = vec![Number::default(); 5];
    c[4] = wide(size(612, 70) as u32);
    let d = Polynomial::new(c);
    check(&a, &d);
    // a = g u, d = g v with wide coefficients: not divisible, every step from the top tried.
    let g = exact_poly(&mut s, size(40, 4), size(60, 9));
    let u = exact_poly(&mut s, size(400, 8), 2);
    let v = exact_poly(&mut s, size(12, 3), 2);
    check(&(&g * &u), &(&g * &v));
    check(&(&g * &u), &-&(&g * &v));
}

#[test]
fn rational_powers_modulo_a_polynomial_through_the_integers() {
    let mut s = 0x7777_1111_3333_9999;
    for (lb, lf) in [
        (1, 2),
        (3, 2),
        (5, 4),
        (size(12, 4), size(12, 4)),
        (size(20, 5), 3),
    ] {
        let b = rpoly(&mut s, lb, 6);
        let mut f = rpoly(&mut s, lf, 6);
        while f.length() < 2 {
            f = rpoly(&mut s, lf, 6);
        }
        for e in [0u64, 1, 2, 3, 7, size(39, 9) as u64] {
            // Against products and remainders one at a time.
            let mut want = &Polynomial::<Rational>::one() % &f;
            for _ in 0..e {
                want = &(&want * &b) % &f;
            }
            assert_eq!(
                b.pow_mod(&Number::from(e), &f),
                want,
                "lb {lb}, lf {lf}, e {e}"
            );
            assert_eq!(b.pow_mod(&Number::from(e), &-&f), want);
        }
    }
}
