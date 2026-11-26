//! Certified complex roots of polynomials over the integers and the rationals: every disc
//! checked against exact values where they are known, the discs pairwise disjoint (by a test of
//! directed roundings here), the power sums of the roots against those of the coefficients
//! (Newton's identities, exactly), the real ones against `real_roots`, the multiplicities against
//! the square-free factorisation by gcds.

use super::common::*;
use a::math::float::{BigComplex, BigFloat, Round};
use a::math::numbers::Number;
use a::math::poly::{ComplexRoot, Polynomial};
use a::math::rational::Rational;

const B: u64 = 64;

fn up(x: &BigFloat, y: &BigFloat) -> BigFloat {
    x.add_round(y, B, Round::Up).0
}

/// A lower bound of `|z − w|`.
fn dist_down(z: &BigComplex, w: &BigComplex) -> BigFloat {
    let part = |x: &BigFloat, y: &BigFloat| {
        let a = x.sub_round(y, B, Round::Up).0;
        let b = x.sub_round(y, B, Round::Down).0;
        if a.is_sign_negative() != b.is_sign_negative() || a.is_zero() || b.is_zero() {
            BigFloat::zero(B)
        } else if a.abs() <= b.abs() {
            a.abs()
        } else {
            b.abs()
        }
    };
    BigComplex::new(part(z.re(), w.re()), part(z.im(), w.im()))
        .abs_round(B, Round::Down)
        .0
}

/// An upper bound of `|z − w|`.
fn dist_up(z: &BigComplex, w: &BigComplex) -> BigFloat {
    let part = |x: &BigFloat, y: &BigFloat| {
        let a = x.sub_round(y, B, Round::Up).0.abs();
        let b = x.sub_round(y, B, Round::Down).0.abs();
        if a >= b { a } else { b }
    };
    BigComplex::new(part(z.re(), w.re()), part(z.im(), w.im()))
        .abs_round(B, Round::Up)
        .0
}

/// The checks every result passes: the count with multiplicities is the degree, the discs are
/// disjoint, the radii small enough, the order and the exact conjugates.
fn check_shape(p: &Polynomial<Number>, r: &[ComplexRoot], prec: u64) {
    let deg = p.degree().unwrap();
    assert_eq!(
        r.iter().map(|x| x.multiplicity() as usize).sum::<usize>(),
        deg,
        "{p}"
    );
    for (i, x) in r.iter().enumerate() {
        assert!(
            x.value().re().prec() >= prec
                && x.value().im().prec() >= prec.min(x.value().im().prec())
        );
        // radius ≤ 2^(1−prec)·|value|, or 0 for the exact root 0.
        let m = x.value().abs_round(B, Round::Up).0;
        let lim = m
            .mul_round(
                &BigFloat::from_parts(&Number::from(1), 1 - prec as i64, B, Round::Up).0,
                B,
                Round::Up,
            )
            .0;
        assert!(
            *x.radius() <= lim,
            "{p}: radius {} for {}",
            x.radius(),
            x.value()
        );
        for y in &r[i + 1..] {
            assert!(
                dist_down(x.value(), y.value()) > up(x.radius(), y.radius()),
                "{p}: discs meet"
            );
        }
    }
    // The real roots first, increasing, then each with a positive imaginary part before its
    // conjugate.
    let reals: Vec<&ComplexRoot> = r.iter().take_while(|x| x.value().im().is_zero()).collect();
    assert!(r[reals.len()..].iter().all(|x| !x.value().im().is_zero()));
    for w in reals.windows(2) {
        assert!(w[0].value().re() < w[1].value().re());
    }
    let rest = &r[reals.len()..];
    assert_eq!(rest.len() % 2, 0);
    for pair in rest.chunks(2) {
        assert!(pair[0].value().im().is_sign_positive());
        assert_eq!(*pair[1].value(), pair[0].value().conj());
        assert_eq!(pair[0].multiplicity(), pair[1].multiplicity());
    }
    assert_eq!(reals.len(), p.count_real_roots(), "{p}");
}

/// The power sums `Σ m·zᵏ` of the roots against those of the coefficients, `k = 1..=3`, by
/// Newton's identities over the rationals, within the bound the radii give:
/// `|Σ m·(vᵏ − zᵏ)| ≤ Σ m·((|v| + r)ᵏ − |v|ᵏ)`.
fn check_power_sums(p: &Polynomial<Number>, r: &[ComplexRoot]) {
    let n = p.degree().unwrap();
    let c: Vec<Rational> = p.coeffs().iter().map(Rational::from).collect();
    let an = &c[n];
    // e_k: the elementary symmetric functions, (−1)^k a_{n−k}/a_n.
    let e = |k: usize| -> Rational {
        if k > n {
            return Rational::default();
        }
        let v = &c[n - k] / an;
        if k % 2 == 1 { -v } else { v }
    };
    let mut s: Vec<Rational> = vec![Rational::from(n as i64)];
    for k in 1..=3usize {
        // s_k = Σ_{i=1}^{k−1} (−1)^(i−1) e_i s_{k−i} + (−1)^(k−1) k e_k
        let mut v = Rational::default();
        for i in 1..k {
            let t = &e(i) * &s[k - i];
            v = if i % 2 == 1 { &v + &t } else { &v - &t };
        }
        let t = &e(k) * &Rational::from(k as i64);
        v = if k % 2 == 1 { &v + &t } else { &v - &t };
        s.push(v);
    }
    let w = 4 * r.iter().map(|x| x.value().re().prec()).max().unwrap() + 64;
    for (k, sk) in s.iter().enumerate().skip(1) {
        let mut sum = BigComplex::zero(w);
        let mut bound = BigFloat::zero(B);
        for x in r {
            let v = x.value().prec_round(w, Round::Nearest).0;
            let mut pw = BigComplex::from(BigFloat::from(1));
            for _ in 0..k {
                pw = pw.mul_round(&v, w, Round::Nearest).0;
            }
            let m = BigFloat::from(x.multiplicity());
            sum = sum
                .add_round(
                    &pw.mul_real_round(&m, w, Round::Nearest).0,
                    w,
                    Round::Nearest,
                )
                .0;
            let a = x.value().abs_round(B, Round::Up).0;
            let ar = up(&a, x.radius());
            let hi = ar.powi_round(k as i64, B, Round::Up).0;
            let lo = x
                .value()
                .abs_round(B, Round::Down)
                .0
                .powi_round(k as i64, B, Round::Down)
                .0;
            let d = hi.sub_round(&lo, B, Round::Up).0;
            bound = up(&bound, &d.mul_round(&m, B, Round::Up).0);
        }
        // Room for the rounding of the sum at w bits.
        let slack = sum
            .abs_round(B, Round::Up)
            .0
            .mul_round(
                &BigFloat::from_parts(&Number::from(1), 8 - w as i64, B, Round::Up).0,
                B,
                Round::Up,
            )
            .0;
        let exact = BigComplex::from(BigFloat::from_rational(sk, w, Round::Nearest).0);
        let err = dist_up(&sum, &exact);
        assert!(
            err <= up(&bound, &slack),
            "{p}: power sum {k}: {err} > {bound}"
        );
    }
}

fn check(p: &Polynomial<Number>, prec: u64) -> Vec<ComplexRoot> {
    let r = p.complex_roots(prec);
    check_shape(p, &r, prec);
    check_power_sums(p, &r);
    r
}

/// Whether the disc of `x` holds `z`.
fn holds(x: &ComplexRoot, z: &BigComplex) -> bool {
    dist_up(x.value(), z) <= *x.radius()
}

#[test]
fn known_roots() {
    // ±i
    let r = check(&ints(&[1, 0, 1]), 64);
    let i = BigComplex::new(BigFloat::from(0), BigFloat::from(1));
    assert!(holds(&r[0], &i) && holds(&r[1], &i.conj()));
    // (x − 2)²(x² + 2): 2 twice, ±i√2.
    let precs: &[u64] = if cfg!(miri) {
        &[1, 53]
    } else {
        &[1, 2, 53, 100, 1000]
    };
    for &prec in precs {
        let r = check(&ints(&[8, -8, 6, -4, 1]), prec);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].multiplicity(), 2);
        assert!(holds(&r[0], &BigComplex::from(BigFloat::from(2))));
        let s = BigFloat::from(2).sqrt_round(prec + 64, Round::Nearest).0;
        assert!(holds(&r[1], &BigComplex::new(BigFloat::from(0), s)));
    }
    // x^n − 2: 2^(1/n)·e^(2πik/n).
    let ns: &[i64] = if cfg!(miri) {
        &[1, 2, 3]
    } else {
        &[1, 2, 3, 7, 16]
    };
    for &n in ns {
        let mut c = vec![0i64; n as usize + 1];
        c[0] = -2;
        c[n as usize] = 1;
        let prec = if cfg!(miri) { 64 } else { 200 };
        let r = check(&ints(&c), prec);
        let w = prec + 64;
        let m = BigFloat::from(2)
            .nth_root_round(n as u32, w, Round::Nearest)
            .0;
        let pi = BigFloat::pi(w);
        for k in 0..n {
            let th = pi
                .mul_round(&BigFloat::from(2 * k), w, Round::Nearest)
                .0
                .div_round(&BigFloat::from(n), w, Round::Nearest)
                .0;
            let (s, c) = th.sin_cos();
            let z = BigComplex::new(
                c.mul_round(&m, w, Round::Nearest).0,
                s.mul_round(&m, w, Round::Nearest).0,
            );
            assert_eq!(
                r.iter().filter(|x| holds(x, &z)).count(),
                1,
                "x^{n} − 2, k {k}"
            );
        }
    }
    // Constants and roots at 0.
    assert!(ints(&[5]).complex_roots(10).is_empty());
    let r = check(&ints(&[0, 0, 0, 1, 1]), 30);
    assert_eq!(r[1].multiplicity(), 3);
    assert!(r[1].value().is_zero() && r[1].radius().is_zero());
}

#[test]
fn random_polynomials() {
    let mut s = 0xFACE_0001;
    for t in 0..size(80, 2) {
        let n = 1 + below(
            &mut s,
            if t % 10 == 0 {
                size(60, 6)
            } else {
                size(15, 6)
            } as u64,
        ) as usize;
        let mut p = Polynomial::new(coeffs(&mut s, n + 1, 1 + t % 12));
        if p.degree().unwrap_or(0) == 0 {
            continue;
        }
        if t % 6 == 5 {
            let f = Polynomial::new(coeffs(&mut s, 3, 2));
            if f.degree().unwrap_or(0) > 0 {
                p = &p * &f.pow(3);
            }
        }
        let prec = [8u64, 53, 128, 400][t % 4];
        let r = check(&p, prec);
        // The real roots agree with real_roots.
        let reals = p.real_roots(prec + 8);
        let rr: Vec<&ComplexRoot> = r.iter().filter(|x| x.value().im().is_zero()).collect();
        assert_eq!(rr.len(), reals.len());
        for (x, (y, m)) in rr.iter().zip(&reals) {
            assert_eq!(x.multiplicity(), *m);
            let d = dist_up(x.value(), &BigComplex::from(y.clone()));
            let ulp = y
                .abs()
                .mul_round(
                    &BigFloat::from_parts(&Number::from(1), -(prec as i64 + 8), B, Round::Up).0,
                    B,
                    Round::Up,
                )
                .0;
            assert!(d <= up(x.radius(), &ulp), "{p}");
        }
    }
}

#[test]
fn close_and_large_roots() {
    // Mignotte: x^n − 2(a·x − 1)², two real roots about a^(−n/2) apart near 1/a.
    let cases: &[(usize, i64)] = if cfg!(miri) {
        &[(7, 10)]
    } else {
        &[(7, 10), (17, 1000), (30, 1 << 30)]
    };
    for &(n, a) in cases {
        let l = ints(&[-1, a]);
        let p = &Polynomial::monomial(Number::from(1), n) - &(&l * &l).scale(&Number::from(2));
        check(&p, 53);
    }
    // Conjugate roots near the real axis: 2^100·((x − 1)² + 2^−100), roots 1 ± 2^−50·i, and
    // (x² + 2^−80)(x − 3): ±2^−40·i near the real root of x and the real 3.
    let t = Number::from(1) << 100usize;
    let p = Polynomial::new(vec![&t + &Number::from(1), -(&t << 1usize), t.clone()]);
    for prec in [8u64, 53, 200] {
        let r = check(&p, prec);
        assert_eq!(r.len(), 2);
        assert!(!r[0].value().im().is_zero());
    }
    let q = Polynomial::new(vec![
        Number::from(1),
        Number::from(0),
        Number::from(1) << 80usize,
    ]);
    let r = check(&(&q * &ints(&[-3, 1])), 30);
    assert_eq!(r.iter().filter(|x| x.value().im().is_zero()).count(), 1);
    // Coefficients beyond the range of f64: (2^3000·x − 1)(x − 2^3000)(x² + 1).
    let big = Number::from(1) << if cfg!(miri) { 1100usize } else { 3000 };
    let f = Polynomial::new(vec![Number::from(-1), big.clone()]);
    let g = Polynomial::new(vec![-big.clone(), Number::from(1)]);
    let p = &(&f * &g) * &ints(&[1, 0, 1]);
    let r = check(&p, 80);
    assert!(holds(&r[1], &BigComplex::from(BigFloat::from(&big))));
    // Wilkinson's ∏ (x − i): the integers, certified.
    let w = (1..=if cfg!(miri) { 6 } else { 20 }).fold(ints(&[1]), |p, i| &p * &ints(&[-i, 1]));
    let r = check(&w, 64);
    for (k, x) in r.iter().enumerate() {
        assert!(holds(x, &BigComplex::from(BigFloat::from(k as i64 + 1))));
    }
}

#[test]
fn rational_polynomials() {
    // x² − 1/4: ±1/2; and the rational polynomial's roots are its primitive part's.
    let p = Polynomial::new(vec![
        Rational::new((-1).into(), 4.into()),
        0.into(),
        1.into(),
    ]);
    let r = p.complex_roots(53);
    assert_eq!(r.len(), 2);
    assert!(holds(&r[0], &BigComplex::from(BigFloat::from(-0.5))));
    assert!(holds(&r[1], &BigComplex::from(BigFloat::from(0.5))));
    let q = Polynomial::new(vec![
        Rational::new(3.into(), 7.into()),
        Rational::from(2),
        Rational::new(5.into(), 3.into()),
    ]);
    let (_, prim) = q.to_primitive();
    let a = q.complex_roots(100);
    let b = prim.complex_roots(100);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.value(), y.value());
    }
}

#[test]
#[should_panic(expected = "the zero polynomial has every number as a root")]
fn roots_of_zero_panic() {
    Polynomial::<Number>::zero().complex_roots(53);
}

#[test]
#[should_panic(expected = "precision 0 is not in 1..=2^40")]
fn precision_0_panics() {
    ints(&[1, 1]).complex_roots(0);
}

#[test]
fn fixed_point_evaluations() {
    // The iteration and the discs evaluate in fixed point where the parts of a point are of
    // like sizes: roots of mixed sizes (2^k·x − 3, real), complex ones off the axis on both
    // sides of the unit circle ((x² + 2^40)(4x² + 1)(x² + x + 1)), Wilkinson's, at precisions
    // below and above a few limbs.
    let w = (1..=if cfg!(miri) { 5 } else { 15 }).fold(ints(&[1]), |p, i| &p * &ints(&[-i, 1]));
    let mixed = (0..if cfg!(miri) { 4 } else { 10 }).fold(ints(&[1]), |p, k| {
        &p * &Polynomial::new(vec![Number::from(-3), Number::from(1) << (4 * k) as usize])
    });
    let c = &(&Polynomial::new(vec![
        Number::from(1) << 40usize,
        Number::from(0),
        Number::from(1),
    ]) * &ints(&[1, 0, 4]))
        * &ints(&[1, 1, 1]);
    let precs: &[u64] = if cfg!(miri) { &[53] } else { &[53, 300, 700] };
    for p in [&w, &mixed, &c] {
        for &prec in precs {
            let r = check(p, prec);
            assert_eq!(r.len(), p.degree().unwrap());
        }
    }
    for (k, x) in check(&w, 100).iter().enumerate() {
        assert!(holds(x, &BigComplex::from(BigFloat::from(k as i64 + 1))));
    }
}

#[test]
fn coefficients_over_wide_ranges() {
    // (2^e·x − 1)(x − 2^e)(x² + 1): roots 2^−e, ±i, 2^e. Up to e = 1000 the coefficients fit
    // double precision after a scaling, and the iteration there must keep every root (a span
    // past 2^1000 underflowed, losing two) or, where its range spoils a start, be replaced by the
    // Newton polygon's (from e ≈ 700 the start in double precision did not converge).
    let es: &[usize] = if cfg!(miri) {
        &[800]
    } else {
        &[300, 700, 800, 900, 1000, 1100, 1500]
    };
    for &e in es {
        let big = Number::from(1) << e;
        let f = Polynomial::new(vec![Number::from(-1), big.clone()]);
        let g = Polynomial::new(vec![-big.clone(), Number::from(1)]);
        let p = &(&f * &g) * &ints(&[1, 0, 1]);
        let r = check(&p, 60);
        assert_eq!(r.len(), 4, "e = {e}");
        assert!(
            holds(&r[1], &BigComplex::from(BigFloat::from(&big))),
            "e = {e}"
        );
    }
}

#[test]
fn low_precisions() {
    // Fuzz finding F6: at 1 and 2 bits the rounded discs of close roots meet, and each meeting
    // doubled the precision once more (x^128 − 1 at 1 bit: values of 1,146,878 bits, 70 s).
    let ns: &[usize] = if cfg!(miri) { &[16] } else { &[32, 128, 256] };
    for &n in ns {
        let mut c = vec![Number::from(0); n + 1];
        c[0] = Number::from(-1);
        c[n] = Number::from(1);
        let p = Polynomial::new(c);
        for prec in [1u64, 2] {
            let r = check(&p, prec);
            assert_eq!(r.len(), n);
            // A few bits more than the distance of neighbours needs, not thousands.
            assert!(
                r.iter().all(|x| x.value().re().prec() <= 32),
                "x^{n} − 1 at {prec}"
            );
        }
    }
    let w = (1..=if cfg!(miri) { 6 } else { 40 }).fold(ints(&[1]), |p, i| &p * &ints(&[-i, 1]));
    for prec in [1u64, 2, 3] {
        check(&w, prec);
    }
}

#[test]
fn coefficients_thousands_of_bits_apart() {
    // Fuzz finding F7: x^4 + (2^2443 − 1)x + 2^2896 and x^8 + 2^2971·x^3 + 2^2259·x^2 + 2^739
    // hung: the sum of the Aberth step dropped the terms whose squares overflowed double
    // precision (|z| ≈ 2^814), and approximations on one another stayed together. (Under Miri
    // the exponents are a fifth.)
    let k = |e: usize| if cfg!(miri) { e / 5 } else { e };
    let two = |e: usize| Number::from(1) << k(e);
    let mut c = vec![Number::from(0); 5];
    c[4] = Number::from(1);
    c[1] = two(2443) - Number::from(1);
    c[0] = two(2896);
    let p = Polynomial::new(c);
    assert_eq!(check(&p, 53).len(), 4);
    let mut c = vec![Number::from(0); 9];
    c[8] = Number::from(1);
    c[3] = two(2971);
    c[2] = two(2259);
    c[0] = two(739);
    let p = Polynomial::new(c);
    assert_eq!(check(&p, 53).len(), 8);
}
