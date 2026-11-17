//! The schoolbook pseudo-division and exact division over the integers on limbs: the identity
//! `l^k a = q d + r` and the quotient against the rationals' for every sign of the leading
//! coefficient and parity of `k`, monic and multi-limb divisors, zeros inside; exact quotients
//! found and refused.

use super::arith_z::exact_poly;
use super::common::*;
use a::math::numbers::Number;
use a::math::poly::Polynomial;
use a::math::rational::Rational;

/// `d` with its leading coefficient made `l`.
fn with_leading(d: &Polynomial<Number>, l: Number) -> Polynomial<Number> {
    let mut c = d.coeffs().to_vec();
    *c.last_mut().unwrap() = l;
    Polynomial::new(c)
}

/// Checks the pseudo-division of `a` by `d` against its definition and the rationals.
fn check_pseudo(a: &Polynomial<Number>, d: &Polynomial<Number>) {
    let (q, r) = a.pseudo_div_rem(d);
    let k = (a.length() - d.length() + 1) as u32;
    let lk = Polynomial::constant(d.leading().unwrap().pow(k));
    assert_eq!(&(&q * d) + &r, &lk * a, "{a} by {d}");
    assert!(r.length() < d.length());
    let (qq, rr) = Polynomial::<Rational>::from(a).div_rem(&Polynomial::<Rational>::from(d));
    let lkq = Polynomial::<Rational>::from(&lk);
    assert_eq!(Polynomial::<Rational>::from(&q), &lkq * &qq);
    assert_eq!(Polynomial::<Rational>::from(&r), &lkq * &rr);
}

#[test]
fn pseudo_division_on_limbs_agrees_with_the_rationals() {
    let mut s = 0x3C6E_F372_FE94_F82B;
    let shapes = [
        (3, 2),
        (4, 2),
        (5, 3),
        (8, 4),
        (9, 2),
        (16, 8),
        (17, 9),
        (24, 3),
        (33, 16),
    ];
    for &(la, ld) in &shapes {
        for bytes in [1, 4, 8, 9, 17, 40] {
            if la * bytes > size(800, 80) {
                continue;
            }
            for _ in 0..size(3, 1) {
                let a = exact_poly(&mut s, la, bytes);
                let d = exact_poly(&mut s, ld, bytes.div_ceil(2));
                check_pseudo(&a, &d);
                // Negative, monic, minus one, multi-limb leading coefficients.
                for l in [
                    -d.leading().unwrap().clone(),
                    Number::from(1),
                    Number::from(-1),
                    number(&mut s, 24),
                ] {
                    if l.is_zero() {
                        continue;
                    }
                    check_pseudo(&a, &with_leading(&d, l));
                }
            }
        }
    }
    // Zeros inside the dividend and the divisor, and a dividend of a monomial.
    let a = ints(&[0, 0, 5, 0, 0, 0, -7, 0, 3]);
    let d = ints(&[0, 2, 0, -3]);
    check_pseudo(&a, &d);
    check_pseudo(&ints(&[0, 0, 0, 0, 0, 0, 1]), &ints(&[1, 0, -4]));
}

#[test]
fn exact_division_on_limbs() {
    let mut s = 0xA409_3822_299F_31D0;
    for (lq, ld) in [(3, 2), (4, 3), (6, 2), (8, 5), (11, 4), (11, 11), (20, 6)] {
        for bytes in [1, 8, 9, 33, 70] {
            if lq * bytes > size(800, 60) {
                continue;
            }
            let q = exact_poly(&mut s, lq, bytes);
            let d = exact_poly(&mut s, ld, bytes);
            let a = &q * &d;
            assert_eq!(a.div_exact(&d), Some(q.clone()), "{lq} by {ld}");
            assert_eq!(a.div_exact(&-&d), Some(-&q));
            // The top divides but the rest does not; a constant added.
            let mut c = a.coeffs().to_vec();
            c[0] = &c[0] + &Number::from(1);
            assert_eq!(Polynomial::new(c).div_exact(&d), None);
            assert_eq!((&a + &Polynomial::one()).div_exact(&d), None);
        }
    }
    // Quotients far larger than the dividend: (x + 1)^n / (x + 1), and a cofactor that grows
    // past every bound before the remainder shows it is not one: (x^9 + 1) / (2x - 1)... not
    // divisible at the top already, and x^10 / (x - 3) whose remainder is 3^10.
    let x1 = ints(&[1, 1]);
    assert_eq!(x1.pow(9).div_exact(&x1), Some(x1.pow(8)));
    assert_eq!(
        ints(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]).div_exact(&ints(&[-1, 2])),
        None
    );
    assert_eq!(
        ints(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).div_exact(&ints(&[-3, 1])),
        None
    );
    assert_eq!(
        Polynomial::<Number>::zero().div_exact(&x1),
        Some(Polynomial::zero())
    );
}
