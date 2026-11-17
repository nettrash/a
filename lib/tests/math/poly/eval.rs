//! Evaluation at many points, products of linear factors and interpolation of `Polynomial`s.

use super::common::*;
use a::math::complex::Complex;
use a::math::numbers::Number;
use a::math::poly::Polynomial;
use a::math::rational::Rational;

#[test]
fn integers_eval_many_matches_eval() {
    let mut s = 0x1234_5678_9abc_def1u64;
    for &(len, count, cb, xb) in &[
        (0, 4, 8, 8),
        (1, 3, 8, 8),
        (4, 0, 8, 8),
        (5, 9, 8, 3),
        (60, 70, 8, 2),
        (60, 200, 40, 9),
        (130, 140, 3, 1),
        (200, 90, 2, 30),
        (70, 500, 1, 1),
    ] {
        let (len, count) = (size(len, len.min(12)), size(count, count.min(14)));
        let p = Polynomial::new(coeffs(&mut s, len, cb));
        let xs: Vec<Number> = (0..count).map(|_| number(&mut s, xb)).collect();
        let v = p.eval_many(&xs);
        assert_eq!(v.len(), xs.len());
        for (x, y) in xs.iter().zip(&v) {
            assert_eq!(
                &oracle_eval(p.coeffs(), x),
                y,
                "len {}, x = {x}",
                p.length()
            );
        }
    }
}

#[test]
fn integers_from_roots() {
    let mut s = 99u64;
    for &(count, b) in &[(0, 4), (1, 4), (2, 9), (17, 3), (40, 20), (100, 2), (65, 1)] {
        let count = size(count, count.min(10));
        let r: Vec<Number> = (0..count).map(|_| number(&mut s, b)).collect();
        let p = Polynomial::from_roots(&r);
        let mut q = vec![Number::from(1)];
        for x in &r {
            q = oracle_mul(&q, &[-x, Number::from(1)]);
        }
        assert_eq!(p.coeffs(), &q[..]);
        assert!(p.is_monic());
        assert!(p.eval_many(&r).iter().all(Number::is_zero));
    }
}

/// `count` distinct rationals.
fn distinct_rats(s: &mut u64, count: usize, b: usize, integral: bool) -> Vec<Rational> {
    let mut v: Vec<Rational> = Vec::new();
    while v.len() < count {
        let x = if integral {
            Rational::from(number(s, b))
        } else {
            rational(s, b)
        };
        if !v.contains(&x) {
            v.push(x);
        }
    }
    v
}

#[test]
fn rationals_eval_many_and_roots() {
    let mut s = 5u64;
    for &(len, count, integral) in &[
        (0, 3, true),
        (1, 3, false),
        (6, 9, false),
        (60, 70, true),
        (60, 70, false),
        (90, 120, false),
    ] {
        let (len, count) = (size(len, len.min(8)), size(count, count.min(9)));
        let p = rpoly(&mut s, len, 3);
        let xs = distinct_rats(&mut s, count, 3, integral);
        let v = p.eval_many(&xs);
        for (x, y) in xs.iter().zip(&v) {
            assert_eq!(&p.eval(x), y);
        }
        let r = Polynomial::from_roots(&xs);
        assert!(r.is_monic() && r.degree() == Some(xs.len()));
        assert!(r.eval_many(&xs).iter().all(Rational::is_zero));
    }
}

#[test]
fn rationals_interpolate_round_trips() {
    let mut s = 77u64;
    for &(count, integral) in &[
        (0, true),
        (1, true),
        (2, false),
        (3, true),
        (12, false),
        (60, true),
        (80, false),
        (150, true),
    ] {
        let count = size(count, count.min(9));
        let xs = distinct_rats(&mut s, count, 2, integral);
        let ys: Vec<Rational> = (0..count).map(|_| rational(&mut s, 3)).collect();
        let p = Polynomial::interpolate(&xs, &ys);
        assert!(p.length() <= count);
        assert_eq!(p.eval_many(&xs), ys);
        if count <= 80 {
            assert_eq!(Polynomial::interpolate_newton(&xs, &ys), p);
            assert_eq!(Polynomial::interpolate_lagrange(&xs, &ys), p);
        }
        // A polynomial of lower degree comes back.
        if count >= 2 {
            let q = rpoly(&mut s, count - 1, 2);
            let v = q.eval_many(&xs);
            assert_eq!(Polynomial::interpolate(&xs, &v), q);
        }
    }
}

#[test]
fn floats() {
    let p = Polynomial::new(vec![1.0, -3.0, 0.5, 2.0]);
    let xs: Vec<f64> = (0..21).map(|i| i as f64 * 0.37 - 3.0).collect();
    let v = p.eval_many(&xs);
    for (x, y) in xs.iter().zip(&v) {
        assert_eq!(p.eval(x).to_bits(), y.to_bits());
    }
    let q = Polynomial::interpolate(&xs[..4], &v[..4]);
    for (a, b) in q.coeffs().iter().zip(p.coeffs()) {
        assert!((a - b).abs() < 1e-9, "{a} {b}");
    }
    let l = Polynomial::interpolate_lagrange(&xs[..4], &v[..4]);
    for (a, b) in l.coeffs().iter().zip(p.coeffs()) {
        assert!((a - b).abs() < 1e-9, "{a} {b}");
    }
    let r = Polynomial::from_roots(&[1.0, 2.0, -0.5]);
    assert_eq!(r.coeffs(), [1.0, 0.5, -2.5, 1.0]);
    let c = Polynomial::new(vec![Complex::new(1.0, 1.0), Complex::new(0.0, 2.0)]);
    let z = [Complex::new(0.5, -1.0), Complex::new(2.0, 0.0)];
    let w = c.eval_many(&z);
    assert_eq!(w, [c.eval(&z[0]), c.eval(&z[1])]);
    let back = Polynomial::interpolate(&z, &w);
    for (a, b) in back.coeffs().iter().zip(c.coeffs()) {
        assert!((*a - *b).norm_sqr() < 1e-24);
    }
}

#[test]
#[should_panic(expected = "repeated")]
fn repeated_rational_point() {
    let xs = [Rational::from(1), Rational::from(2), Rational::from(1)];
    let ys = [Rational::from(0), Rational::from(1), Rational::from(2)];
    let _ = Polynomial::interpolate(&xs, &ys);
}

#[test]
#[should_panic(expected = "repeated")]
fn repeated_float_point() {
    let _ = Polynomial::interpolate(&[1.0, 2.0, 1.0], &[0.0, 1.0, 2.0]);
}

#[test]
#[should_panic(expected = "as many values")]
fn lengths_differ() {
    let _ = Polynomial::interpolate(&[1.0, 2.0], &[0.0]);
}

#[test]
fn edge_cases() {
    let none: [Number; 0] = [];
    assert!(Polynomial::from_roots(&none).is_one());
    assert!(
        Polynomial::<Number>::zero()
            .eval_many(&[Number::from(3)])
            .iter()
            .all(Number::is_zero)
    );
    assert!(ints(&[1, 2]).eval_many(&none).is_empty());
    let nr: [Rational; 0] = [];
    assert!(Polynomial::interpolate(&nr, &nr).is_zero());
    assert!(Polynomial::from_roots(&nr).is_one());
    let p = Polynomial::interpolate(&[Rational::from(5)], &[Rational::new(3.into(), 4.into())]);
    assert_eq!(p.to_string(), "3/4");
    // 0, ±1 and large points, in either byte order.
    let p = ints(&[3, -1, 4, -1, 5, -9, 2, 6]);
    let xs: Vec<Number> = vec![
        Number::from(0),
        Number::from(1),
        Number::from(-1),
        Number::from(i64::MIN),
        Number::from_str_radix(
            "-123456789012345678901234567890",
            10,
            a::math::numbers::ByteOrder::BigEndian,
        )
        .unwrap(),
    ];
    for (x, y) in xs.iter().zip(p.eval_many(&xs)) {
        assert_eq!(oracle_eval(p.coeffs(), x), y);
    }
}

#[test]
fn integers_long_and_divide_and_conquer() {
    let mut s = 31u64;
    // Word points with blocks inside divide and conquer; long points.
    for &(len, count, cb, xb) in &[
        (700, 6, 8, 3),
        (300, 4, 8, 8),
        (64, 5, 200, 100),
        (1200, 3, 1, 2),
    ] {
        let len = size(len, 30);
        let p = Polynomial::new(coeffs(&mut s, len, cb));
        let xs: Vec<Number> = (0..count).map(|_| number(&mut s, xb)).collect();
        for (x, y) in xs.iter().zip(p.eval_many(&xs)) {
            assert_eq!(p.eval(x), y);
        }
    }
}

#[test]
fn rationals_with_small_common_denominator() {
    let mut s = 41u64;
    let p = rpoly(&mut s, size(40, 6), 4);
    // Halves, quarters and eighths: the common denominator 8, by the integer points 8x.
    let xs: Vec<Rational> = (0..size(60, 8) as i64)
        .map(|i| Rational::new(Number::from(i * 3 - 70), Number::from(1 << (i % 4))))
        .collect();
    let v = p.eval_many(&xs);
    for (x, y) in xs.iter().zip(&v) {
        assert_eq!(&p.eval(x), y);
    }
    let ys: Vec<Rational> = (0..xs.len()).map(|_| rational(&mut s, 2)).collect();
    let mut seen = Vec::new();
    let xs: Vec<Rational> = xs
        .into_iter()
        .filter(|x| {
            let new = !seen.contains(x);
            seen.push(x.clone());
            new
        })
        .collect();
    let ys = &ys[..xs.len()];
    let q = Polynomial::interpolate(&xs, ys);
    assert_eq!(q.eval_many(&xs), ys);
}

#[test]
fn floats_from_roots_tree() {
    let r: Vec<f64> = (0..size(40, 20)).map(|i| (i as f64 * 0.7).sin()).collect();
    let p = Polynomial::from_roots(&r);
    assert_eq!(p.degree(), Some(r.len()));
    for x in &r {
        assert!(p.eval(x).abs() < 1e-9);
    }
    let z: Vec<Complex<f64>> = (0..30)
        .map(|i| Complex::new((i as f64).cos(), (i as f64).sin()))
        .collect();
    let q = Polynomial::from_roots(&z);
    for x in &z {
        assert!(q.eval(x).norm_sqr() < 1e-18);
    }
    let none: [f64; 0] = [];
    assert!(Polynomial::interpolate(&none, &none).is_zero());
    assert_eq!(Polynomial::interpolate(&[2.0], &[5.0]).coeffs(), [5.0]);
}
