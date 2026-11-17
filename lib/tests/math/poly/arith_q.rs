//! Products and division over the rationals and the floats, the pseudo-division and exact
//! division over the integers by Kronecker substitution, and compositions, powers and products
//! modulo a polynomial over a field.

use super::arith_z::exact_poly;
use super::common::*;
use a::math::complex::Complex;
use a::math::numbers::Number;
use a::math::poly::Polynomial;
use a::math::rational::Rational;

/// The schoolbook product of rationals, the oracle.
fn rat_oracle(a: &[Rational], b: &[Rational]) -> Vec<Rational> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut c = vec![Rational::from(0); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            c[i + j] = &c[i + j] + &(x * y);
        }
    }
    while c.last().is_some_and(Rational::is_zero) {
        c.pop();
    }
    c
}

#[test]
fn rational_products_agree_with_the_oracle() {
    let mut s = 0x2545_F491_4F6C_DD1D;
    for bytes in [1, 3, 9, 20] {
        for (la, lb) in [
            (1, 1),
            (1, 5),
            (2, 2),
            (3, 7),
            (8, 8),
            (size(30, 6), size(17, 4)),
        ] {
            let a = rpoly(&mut s, la, bytes);
            let b = rpoly(&mut s, lb, bytes);
            assert_eq!((&a * &b).coeffs(), rat_oracle(a.coeffs(), b.coeffs()));
            assert_eq!(a.sqr().coeffs(), rat_oracle(a.coeffs(), a.coeffs()));
        }
    }
}

/// `Σ |a_i|²` square-rooted.
fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn float_products_by_the_transform_within_their_bound() {
    let mut s = 0x6A09_E667_F3BC_C908;
    let mut f = || (xorshift(&mut s) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
    for len in [size(400, 15), size(512, 16), size(700, 20), size(3000, 33)] {
        let a: Vec<f64> = (0..len).map(|_| f()).collect();
        let b: Vec<f64> = (0..len / 2 + 7).map(|_| f()).collect();
        let fast = &Polynomial::new(a.clone()) * &Polynomial::new(b.clone());
        // The schoolbook sums in the order of the core.
        let mut slow = vec![0.0; a.len() + b.len() - 1];
        let mut abs = vec![0.0; a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                slow[i + j] += x * y;
                abs[i + j] += (x * y).abs();
            }
        }
        let n = (a.len() + b.len() - 1).next_power_of_two() as f64;
        let eps = f64::EPSILON / 2.0;
        let bound = 11.0 * n.log2() * eps * norm(&a) * norm(&b);
        for k in 0..slow.len() {
            let gamma = (b.len() as f64) * eps / (1.0 - b.len() as f64 * eps);
            let err = (fast.coeff(k) - slow[k]).abs();
            assert!(err <= bound + gamma * abs[k], "{len}: {k} off by {err}");
        }
        let sq = Polynomial::new(a.clone()).sqr();
        let p = &Polynomial::new(a.clone()) * &Polynomial::new(a.clone());
        let sb = 11.0 * n.log2() * eps * norm(&a) * norm(&a) * 2.0;
        for k in 0..sq.length() {
            assert!((sq.coeff(k) - p.coeff(k)).abs() <= sb);
        }
    }
    // Small integers are exact through the transform: every error below 1/2.
    let a: Vec<f64> = (0..size(1500, 20))
        .map(|i| ((i * 7) % 13) as f64 - 6.0)
        .collect();
    let b: Vec<f64> = (0..size(1200, 18))
        .map(|i| ((i * 5) % 11) as f64 - 5.0)
        .collect();
    let p = &Polynomial::new(a.clone()) * &Polynomial::new(b.clone());
    let ai: Vec<Number> = a.iter().map(|&x| Number::from(x as i64)).collect();
    let bi: Vec<Number> = b.iter().map(|&x| Number::from(x as i64)).collect();
    let exact = oracle_mul(&ai, &bi);
    for (k, c) in exact.iter().enumerate() {
        assert_eq!(Number::from(p.coeff(k).round() as i64), *c);
    }
}

#[test]
fn complex_products_by_the_transform() {
    let mut s = 0xBB67_AE85_84CA_A73B;
    let mut f = || (xorshift(&mut s) % 21) as f64 - 10.0;
    for len in [size(256, 16), size(900, 20)] {
        let a: Vec<Complex<f64>> = (0..len).map(|_| Complex::new(f(), f())).collect();
        let b: Vec<Complex<f64>> = (0..len - 3).map(|_| Complex::new(f(), f())).collect();
        let p = &Polynomial::new(a.clone()) * &Polynomial::new(b.clone());
        let q = Polynomial::new(a.clone()).sqr();
        for k in 0..a.len() + b.len() - 1 {
            let mut c = Complex::new(0.0, 0.0);
            let mut d = Complex::new(0.0, 0.0);
            for i in 0..a.len() {
                if k >= i && k - i < b.len() {
                    c += a[i] * b[k - i];
                }
                if k >= i && k - i < a.len() {
                    d += a[i] * a[k - i];
                }
            }
            // Integers: exact after rounding.
            assert_eq!(p.coeff(k).re.round(), c.re);
            assert_eq!(p.coeff(k).im.round(), c.im);
            assert_eq!(q.coeff(k).re.round(), d.re);
            assert_eq!(q.coeff(k).im.round(), d.im);
        }
    }
}

#[test]
fn series_inverses_over_fields() {
    let mut s = 0x3C6E_F372_FE94_F82B;
    for len in [1, 2, 5, size(40, 7)] {
        for prec in [1, 3, 32, 33, size(100, 12)] {
            let mut f = rpoly(&mut s, len, 4);
            if f.coeff(0).is_zero() {
                f = &f + &Polynomial::one();
            }
            let g = f.inv_series(prec);
            let e = (&f * &g).truncate(prec);
            assert!(e.is_one(), "{len} to {prec}");
        }
    }
    let p = Polynomial::new(vec![2.0, 1.0]);
    let g = p.inv_series(size(600, 40));
    let e = (&p * &g).truncate(size(600, 40));
    assert!((e.coeff(0) - 1.0).abs() < 1e-12);
    assert!(e.coeffs()[1..].iter().all(|c| c.abs() < 1e-12));
}

#[test]
#[should_panic(expected = "the constant coefficient is 0")]
fn a_series_with_a_zero_constant_panics() {
    let _ = Polynomial::new(vec![0.0, 1.0]).inv_series(3);
}

#[test]
fn float_division_by_newton() {
    // a = q0 d + r0 with a monic, well-conditioned d: recovered to rounding.
    let n = size(1500, 30);
    let q0: Vec<f64> = (0..n).map(|i| ((i * 3) % 7) as f64 - 3.0).collect();
    // Small lower coefficients: the reversed divisor's inverse decays, well conditioned.
    let mut d: Vec<f64> = (0..n - 2)
        .map(|i| (((i * 5) % 9) as f64 - 4.0) / (16.0 * n as f64))
        .collect();
    d.push(1.0);
    let r0: Vec<f64> = (0..n - 3).map(|i| ((i * 11) % 5) as f64 - 2.0).collect();
    let (qp, dp, rp) = (Polynomial::new(q0), Polynomial::new(d), Polynomial::new(r0));
    let a = &(&qp * &dp) + &rp;
    let (q, r) = a.div_rem(&dp);
    let back = &(&q * &dp) + &r;
    let scale = a.coeffs().iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for k in 0..a.length() {
        assert!((back.coeff(k) - a.coeff(k)).abs() <= 1e-6 * scale, "{k}");
    }
    assert!(r.length() < dp.length());
}

#[test]
fn pseudo_division_by_kronecker_substitution() {
    let mut s = 0xA54F_F53A_5F1D_36F1;
    for (la, ld, bytes) in [
        (10, 3, 4),
        (20, 2, 8),
        (size(60, 12), size(25, 4), 9),
        (size(200, 14), 9, 20),
    ] {
        let a = exact_poly(&mut s, la, bytes);
        let d = exact_poly(&mut s, ld, bytes / 2 + 1);
        let (q, r) = a.pseudo_div_rem(&d);
        let l = d.leading().unwrap().clone();
        let k = (la - ld + 1) as u32;
        let lk = Polynomial::constant(l.pow(k));
        assert_eq!(&(&q * &d) + &r, &lk * &a, "{la} by {ld}");
        assert!(r.length() < d.length());
        // The pseudo-quotient is l^k times the rational quotient.
        let (qq, _) = Polynomial::<Rational>::from(&a).div_rem(&Polynomial::<Rational>::from(&d));
        let want: Polynomial<Rational> = &Polynomial::<Rational>::from(&lk) * &qq;
        assert_eq!(Polynomial::<Rational>::from(&q), want);
    }
}

#[test]
fn long_pseudo_divisions_by_kronecker_substitution() {
    // Past the crossovers: 384 coefficients in the quotient, or 32 by a divisor of 1,000 bits.
    let mut s = 0x7F4A_7C15_9E37_79B9;
    for (la, ld, bytes) in [(size(400, 10), 5, 2), (size(50, 9), size(8, 3), 130)] {
        let a = exact_poly(&mut s, la, bytes);
        let d = exact_poly(&mut s, ld, bytes);
        let (q, r) = a.pseudo_div_rem(&d);
        let k = (la - ld + 1) as u32;
        let lk = Polynomial::constant(d.leading().unwrap().pow(k));
        assert!(r.length() < d.length());
        assert_eq!(&(&q * &d) + &r, &lk * &a, "{la} by {ld}");
    }
}

#[test]
fn exact_division_by_kronecker_substitution() {
    let mut s = 0x510E_527F_ADE6_82D1;
    for (lq, ld, bytes) in [
        (9, 2, 1),
        (12, 5, 8),
        (size(80, 10), size(30, 4), 4),
        (size(40, 9), 3, 60),
    ] {
        let q = exact_poly(&mut s, lq, bytes);
        let d = exact_poly(&mut s, ld, bytes);
        let a = &q * &d;
        assert_eq!(a.div_exact(&d), Some(q.clone()), "{lq} by {ld}");
        // Not divisible: one coefficient changed, or a constant added.
        let mut c = a.coeffs().to_vec();
        c[lq / 2] = &c[lq / 2] + &Number::from(1);
        assert_eq!(Polynomial::new(c).div_exact(&d), None);
        assert_eq!((&a + &Polynomial::one()).div_exact(&d), None);
    }
    // Quotients with coefficients far larger than the dividend's: (x - 1)^n / (x - 1).
    let x1 = ints(&[-1, 1]);
    let p = x1.pow(size(60, 12) as u32);
    let want = x1.pow(size(59, 11) as u32);
    assert_eq!(p.div_exact(&x1), Some(want));
    // (x^n - 1) / (x - 1) = 1 + x + ... + x^(n-1).
    let n = size(50, 10);
    let mut c = vec![Number::from(0); n + 1];
    c[0] = Number::from(-1);
    c[n] = Number::from(1);
    assert_eq!(
        Polynomial::new(c).div_exact(&x1),
        Some(Polynomial::new(vec![Number::from(1); n]))
    );
}

#[test]
fn compositions_and_powers_over_fields() {
    let mut s = 0x9B05_688C_2B3E_6C1F;
    for (lp, lg) in [(1, 4), (4, 1), (9, 3), (size(40, 10), 4)] {
        let p = exact_poly(&mut s, lp, 3);
        let g = exact_poly(&mut s, lg, 3);
        let c = p.compose(&g);
        for x in [-2i64, 0, 1, 3] {
            let x = Number::from(x);
            assert_eq!(c.eval(&x), p.eval(&g.eval(&x)));
        }
        let (pq, gq) = (
            Polynomial::<Rational>::from(&p),
            Polynomial::<Rational>::from(&g),
        );
        let f = Polynomial::<Rational>::from(&exact_poly(&mut s, 6, 2));
        assert_eq!(pq.compose_mod(&gq, &f), &pq.compose(&gq) % &f);
        assert_eq!(pq.mul_mod(&gq, &f), &(&pq * &gq) % &f);
        for e in [0u32, 1, 2, 5] {
            assert_eq!(pq.pow_mod(&Number::from(e), &f), &pq.pow(e) % &f);
        }
    }
}

#[test]
fn rational_division_through_the_integers() {
    let mut s = 0x1F83_D9AB_FB41_BD6B;
    for (la, ld, bytes) in [
        (9, 2, 2),
        (12, 4, 3),
        (size(40, 12), size(9, 3), 4),
        (size(60, 14), size(30, 5), 2),
    ] {
        let mut a = rpoly(&mut s, la, bytes);
        while a.length() < la / 2 + ld {
            a = &a + &Polynomial::monomial(Rational::from(1), la);
        }
        let mut d = rpoly(&mut s, ld, bytes);
        while d.length() < 2 {
            d = &d + &Polynomial::monomial(Rational::new((-3).into(), 7.into()), ld);
        }
        let (q, r) = a.div_rem(&d);
        assert!(r.length() < d.length(), "{la} by {ld}");
        assert_eq!(&(&q * &d) + &r, a, "{la} by {ld}");
    }
}
