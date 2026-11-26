//! The core of `Polynomial<T>`: construction, coefficients, arithmetic, evaluation, the
//! derivative, composition, division and the simple greatest common divisors, against the oracle
//! of plain vectors.

use super::common::*;
use a::math::complex::Complex;
use a::math::numbers::Number;
use a::math::poly::Polynomial;
use a::math::rational::Rational;

#[test]
fn construction_drops_zeros_at_the_top() {
    let p = Polynomial::new(vec![Number::from(1), Number::from(0), Number::from(0)]);
    assert_eq!(p.coeffs(), [Number::from(1)]);
    assert_eq!(p.degree(), Some(0));
    assert_eq!(p.length(), 1);
    let z = Polynomial::new(vec![Number::from(0); 4]);
    assert!(z.is_zero());
    assert_eq!(z.degree(), None);
    assert_eq!(z, Polynomial::zero());
    assert_eq!(Polynomial::<Number>::default(), Polynomial::zero());
    assert!(Polynomial::<f64>::one().is_one());
    assert_eq!(Polynomial::<f64>::x().coeffs(), [0.0, 1.0]);
    assert_eq!(Polynomial::monomial(3.0, 2).coeffs(), [0.0, 0.0, 3.0]);
    assert!(Polynomial::monomial(0.0, 5).is_zero());
    assert!(Polynomial::constant(0.0).is_zero());
    assert_eq!(p.coeff(7), Number::from(0));
    assert_eq!(ints(&[1, 2]).leading(), Some(&Number::from(2)));
    assert!(ints(&[5, 1]).is_monic());
    assert!(!ints(&[5, 2]).is_monic());
    assert!(ints(&[5]).is_constant() && Polynomial::<Number>::zero().is_constant());
    // -0.0 is a zero at the top too.
    assert_eq!(Polynomial::new(vec![1.0, -0.0]).degree(), Some(0));
}

#[test]
fn products_agree_with_the_schoolbook_oracle() {
    let mut s = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..size(400, 20) {
        let a = poly(&mut s, 40, 40);
        let b = poly(&mut s, 40, 40);
        let want = oracle_mul(a.coeffs(), b.coeffs());
        let c = &a * &b;
        assert_eq!(c.coeffs(), want, "{a:?} * {b:?}");
        assert_eq!(&b * &a, c);
        assert_eq!(a.clone() * b.clone(), c);
        assert_eq!(a.clone() * &b, c);
        assert_eq!(&a * b.clone(), c);
        let mut d = a.clone();
        d *= &b;
        assert_eq!(d, c);
        assert_eq!(a.sqr().coeffs(), oracle_mul(a.coeffs(), a.coeffs()));
        assert_eq!(&a * &a, a.sqr());
    }
}

#[test]
fn sums_and_differences_in_every_form() {
    let mut s = 0x1234_5678_9ABC_DEF1;
    for _ in 0..size(300, 20) {
        let a = poly(&mut s, 20, 24);
        let b = poly(&mut s, 20, 24);
        let n = a.length().max(b.length());
        let want: Vec<Number> = (0..n).map(|i| &a.coeff(i) + &b.coeff(i)).collect();
        let sum = &a + &b;
        assert_eq!(sum.coeffs(), trim(want));
        assert_eq!(a.clone() + &b, sum);
        assert_eq!(&a + b.clone(), sum);
        assert_eq!(a.clone() + b.clone(), sum);
        let mut t = a.clone();
        t += b.clone();
        assert_eq!(t, sum);
        let diff = &a - &b;
        let want: Vec<Number> = (0..n).map(|i| &a.coeff(i) - &b.coeff(i)).collect();
        assert_eq!(diff.coeffs(), trim(want));
        assert_eq!(a.clone() - &b, diff);
        assert_eq!(&a - b.clone(), diff);
        assert_eq!(a.clone() - b.clone(), diff);
        let mut t = a.clone();
        t -= &b;
        assert_eq!(t, diff);
        assert!((&a - &a).is_zero());
        assert_eq!(&(&a + &b) - &b, a);
        assert_eq!(-&diff, &b - &a);
        assert_eq!(-diff.clone(), &b - &a);
    }
}

#[test]
fn horner_evaluation_agrees_with_the_sum_of_terms() {
    let mut s = 0x0F0F_1234_5555_AAAA;
    for _ in 0..size(300, 20) {
        let a = poly(&mut s, 30, 20);
        let b = poly(&mut s, 30, 20);
        let x = number(&mut s, 12);
        assert_eq!(a.eval(&x), oracle_eval(a.coeffs(), &x));
        assert_eq!((&a * &b).eval(&x), &a.eval(&x) * &b.eval(&x));
    }
    assert_eq!(
        Polynomial::<Number>::zero().eval(&Number::from(5)),
        Number::from(0)
    );
}

#[test]
fn derivatives_follow_the_product_rule() {
    let mut s = 0xDEAD_BEEF_0BAD_F00D;
    for _ in 0..size(200, 10) {
        let a = poly(&mut s, 20, 16);
        let b = poly(&mut s, 20, 16);
        let lhs = (&a * &b).derivative();
        let rhs = &(&a.derivative() * &b) + &(&a * &b.derivative());
        assert_eq!(lhs, rhs);
    }
    assert_eq!(ints(&[5, 3, 0, 2]).derivative(), ints(&[3, 0, 6]));
    assert!(ints(&[5]).derivative().is_zero());
}

#[test]
fn compositions_agree_with_evaluation() {
    let mut s = 0x5151_7272_9393_B4B4;
    for _ in 0..size(100, 6) {
        let a = poly(&mut s, 8, 8);
        let g = poly(&mut s, 6, 8);
        let x = number(&mut s, 6);
        assert_eq!(a.compose(&g).eval(&x), a.eval(&g.eval(&x)));
    }
}

#[test]
fn powers_are_repeated_products() {
    let mut s = 0xAAAA_5555_1111_2222;
    for _ in 0..size(60, 6) {
        let a = poly(&mut s, 6, 6);
        let mut want = Polynomial::one();
        for e in 0..9u32 {
            assert_eq!(a.pow(e), want, "{a:?}^{e}");
            want = &want * &a;
        }
    }
}

#[test]
fn shifts_truncations_and_reversals() {
    let p = ints(&[1, 2, 3, 4]);
    assert_eq!(p.shift_left(2), ints(&[0, 0, 1, 2, 3, 4]));
    assert_eq!(p.shift_right(1), ints(&[2, 3, 4]));
    assert!(p.shift_right(9).is_zero());
    assert_eq!(p.truncate(2), ints(&[1, 2]));
    assert_eq!(p.truncate(9), p);
    assert_eq!(p.reverse(4), ints(&[4, 3, 2, 1]));
    assert_eq!(p.reverse(6), ints(&[0, 0, 4, 3, 2, 1]));
    assert_eq!(ints(&[0, 0, 5]).reverse(3), ints(&[5]));
    assert_eq!(p.scale(&Number::from(-2)), ints(&[-2, -4, -6, -8]));
    assert!(p.scale(&Number::from(0)).is_zero());
    let r: Polynomial<Rational> = p.map(|c| Rational::new(c.clone(), Number::from(2)));
    assert_eq!(
        r,
        Polynomial::<Rational>::from(&p).scale(&Rational::new(1.into(), 2.into()))
    );
}

#[test]
fn text_as_pari_writes_it() {
    assert_eq!(ints(&[]).to_string(), "0");
    assert_eq!(ints(&[1]).to_string(), "1");
    assert_eq!(ints(&[-1]).to_string(), "-1");
    assert_eq!(ints(&[0, 1]).to_string(), "x");
    assert_eq!(ints(&[0, -1]).to_string(), "-x");
    assert_eq!(ints(&[1, -1, 3]).to_string(), "3*x^2 - x + 1");
    assert_eq!(ints(&[-7, 0, 0, -2]).to_string(), "-2*x^3 - 7");
    assert_eq!(ints(&[0, 0, 1]).to_string(), "x^2");
    let r = Polynomial::new(vec![
        Rational::new((-1).into(), 2.into()),
        Rational::from(1),
    ]);
    assert_eq!(r.to_string(), "x - 1/2");
    assert_eq!(Polynomial::new(vec![0.5, -1.5]).to_string(), "-1.5*x + 0.5");
    let c = Polynomial::new(vec![Complex::new(1.0, -2.0), Complex::new(1.0, 0.0)]);
    assert_eq!(c.to_string(), "x + (1-2i)");
    assert_eq!(
        format!("{:.1}", Polynomial::new(vec![0.25, 2.0])),
        "2.0*x + 0.2"
    );
}

#[test]
fn division_with_a_remainder_over_the_rationals() {
    let mut s = 0x1357_9BDF_2468_ACE0;
    for _ in 0..size(200, 10) {
        let a = rpoly(&mut s, 16, 6);
        let mut d = rpoly(&mut s, 10, 6);
        while d.is_zero() {
            d = rpoly(&mut s, 10, 6);
        }
        let (q, r) = a.div_rem(&d);
        assert!(r.length() < d.length());
        assert_eq!(&(&q * &d) + &r, a);
        assert_eq!(&a / &d, q);
        assert_eq!(&a % &d, r);
        assert_eq!(a.clone() / d.clone(), q);
        assert_eq!(a.clone() % &d, r);
        let mut t = a.clone();
        t /= &d;
        assert_eq!(t, q);
        let mut t = a.clone();
        t %= d.clone();
        assert_eq!(t, r);
        assert!(d.to_monic().is_monic());
    }
}

#[test]
#[should_panic(expected = "division by the zero polynomial")]
fn division_by_zero_panics() {
    let _ = rats(&[1, 2]).div_rem(&Polynomial::zero());
}

#[test]
fn division_of_floats_and_complex_numbers() {
    let (q, r) = Polynomial::new(vec![-4.0, 0.0, 1.0]).div_rem(&Polynomial::new(vec![-2.0, 1.0]));
    assert_eq!((q.coeffs(), r.coeffs()), (&[2.0, 1.0][..], &[][..]));
    let i = Complex::new(0.0, 1.0);
    let one = Complex::new(1.0, 0.0);
    // x² + 1 = (x - i)(x + i)
    let p = Polynomial::new(vec![one, Complex::new(0.0, 0.0), one]);
    let (q, r) = p.div_rem(&Polynomial::new(vec![-i, one]));
    assert_eq!(q.coeffs(), [i, one]);
    assert!(r.is_zero());
}

#[test]
fn pseudo_division_over_the_integers() {
    let mut s = 0xFEDC_BA98_7654_3210;
    for _ in 0..size(200, 10) {
        let a = poly(&mut s, 14, 10);
        let mut d = poly(&mut s, 8, 10);
        while d.is_zero() {
            d = poly(&mut s, 8, 10);
        }
        let (q, r) = a.pseudo_div_rem(&d);
        assert!(r.length() < d.length());
        let k = (a.length() + 1).saturating_sub(d.length()) as u32;
        let l = d.leading().unwrap().pow(k);
        assert_eq!(a.scale(&l), &(&q * &d) + &r, "{a:?} {d:?}");
        // An exact quotient, and none when there is a remainder.
        let p = &a * &d;
        if !a.is_zero() {
            assert_eq!(p.div_exact(&d), Some(a.clone()));
            let mut off = p.clone();
            off += &Polynomial::one();
            if d.length() > 1 {
                assert_eq!(off.div_exact(&d), None);
            }
        }
    }
    assert_eq!(ints(&[1, 0, 1]).div_exact(&ints(&[0, 2])), None);
    assert_eq!(ints(&[2]).div_exact(&ints(&[1, 1])), None);
    assert_eq!(
        Polynomial::zero().div_exact(&ints(&[1, 1])),
        Some(Polynomial::zero())
    );
}

#[test]
fn contents_and_primitive_parts() {
    let mut s = 0x0123_4567_89AB_CDEF;
    for _ in 0..size(200, 10) {
        let p = poly(&mut s, 10, 10);
        let c = p.content();
        assert!(!c.is_negative());
        let pp = p.primitive_part();
        if p.is_zero() {
            assert!(c.is_zero() && pp.is_zero());
            continue;
        }
        assert_eq!(pp.content(), Number::from(1));
        assert!(pp.leading().unwrap().is_positive());
        let sign = if p.leading().unwrap().is_negative() {
            -&c
        } else {
            c.clone()
        };
        assert_eq!(pp.scale(&sign), p);
        let q: Polynomial<Rational> = p.map(|x| Rational::new(x.clone(), Number::from(6)));
        let (k, prim) = q.to_primitive();
        assert_eq!(prim, pp);
        assert_eq!(Polynomial::<Rational>::from(&prim).scale(&k), q);
    }
    assert_eq!(
        Polynomial::<Rational>::zero().to_primitive(),
        (Rational::default(), Polynomial::zero())
    );
}

#[test]
fn greatest_common_divisors_of_products() {
    let mut s = 0x7777_3333_9999_1111;
    for _ in 0..size(120, 6) {
        let a = poly(&mut s, 6, 4);
        let b = poly(&mut s, 6, 4);
        let g = poly(&mut s, 5, 4);
        let (x, y) = (&a * &g, &b * &g);
        let h = x.gcd(&y);
        if x.is_zero() && y.is_zero() {
            assert!(h.is_zero());
            continue;
        }
        assert!(h.leading().unwrap().is_positive());
        assert!(x.div_exact(&h).is_some() && y.div_exact(&h).is_some());
        if !g.is_zero() {
            assert!(
                h.div_exact(&g.primitive_part()).is_some(),
                "{x:?} {y:?} {h:?}"
            );
        }
        // Over the rationals: the monic gcd, and Bézout's identity.
        let (xq, yq) = (
            Polynomial::<Rational>::from(&x),
            Polynomial::<Rational>::from(&y),
        );
        let hq = xq.gcd(&yq);
        assert_eq!(hq, Polynomial::<Rational>::from(&h).to_monic());
        let (g2, sx, ty) = xq.xgcd(&yq);
        assert_eq!(g2, hq);
        assert_eq!(&(&sx * &xq) + &(&ty * &yq), hq);
    }
    assert!(ints(&[]).gcd(&ints(&[])).is_zero());
    assert_eq!(
        ints(&[]).gcd(&ints(&[-4, -2])),
        ints(&[2, 1]).scale(&Number::from(2))
    );
    assert_eq!(ints(&[6, 3]).gcd(&ints(&[])), ints(&[6, 3]));
}
