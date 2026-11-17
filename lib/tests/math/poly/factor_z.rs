//! Factorisation over the integers and the rationals: the square-free decomposition and the
//! complete factorisation, against products of polynomials irreducible by Eisenstein's
//! criterion, the cyclotomic polynomials (`xⁿ - 1 = Π_{d | n} Φ_d`), Swinnerton-Dyer
//! polynomials (irreducible, yet split into factors of degree at most 2 modulo every prime),
//! products of many linear factors, huge coefficients, and PARI's `factor`.

use super::common::*;
use a::math::numbers::{ByteOrder, Number};
use a::math::poly::Polynomial;
use a::math::rational::Rational;

type Z = Polynomial<Number>;

/// `c·Π gᵢ^eᵢ`.
fn product(c: &Number, fs: &[(Z, u32)]) -> Z {
    let mut r = Z::constant(c.clone());
    for (g, e) in fs {
        r = &r * &g.pow(*e);
    }
    r
}

/// Sorts factors as the factorisation does: by degree, then the coefficients from the leading
/// one down, then the multiplicity.
fn sorted(mut v: Vec<(Z, u32)>) -> Vec<(Z, u32)> {
    v.sort_by(|a, b| {
        a.0.length()
            .cmp(&b.0.length())
            .then_with(|| a.0.coeffs().iter().rev().cmp(b.0.coeffs().iter().rev()))
            .then(a.1.cmp(&b.1))
    });
    v
}

/// Checks the shape of a factorisation of `f`: the product, primitive factors of positive
/// leading coefficients, sorted, distinct, the content with the sign of `f`.
fn check_shape(f: &Z, c: &Number, fs: &[(Z, u32)]) {
    assert_eq!(&product(c, fs), f, "the product of the factors of {f}");
    if let Some(l) = f.leading() {
        assert_eq!(c.is_negative(), l.is_negative());
    }
    assert_eq!(sorted(fs.to_vec()), fs.to_vec());
    for (g, e) in fs {
        assert!(*e >= 1 && g.degree() >= Some(1));
        assert!(g.content() == 1, "{g} primitive");
        assert!(!g.leading().unwrap().is_negative());
    }
    for w in fs.windows(2) {
        assert_ne!(w[0].0, w[1].0);
    }
}

/// A random polynomial irreducible by Eisenstein's criterion at 2 or 3, of degree `d` and
/// coefficients of up to `bytes` bytes: the leading coefficient prime to q, the others multiples
/// of q, the constant not a multiple of q².
fn eisenstein(s: &mut u64, d: usize, bytes: usize) -> Z {
    let q = Number::from(if below(s, 2) == 0 { 2 } else { 3 });
    let q2 = &q * &q;
    let mut c = Vec::with_capacity(d + 1);
    let mut c0 = &number(s, bytes) * &q;
    while c0.rem_euclid(&q2).is_zero() {
        c0 = &c0 + &q;
    }
    c.push(c0);
    for _ in 1..d {
        c.push(&number(s, bytes) * &q);
    }
    let mut l = number(s, bytes);
    while l.rem_euclid(&q).is_zero() {
        l = &l + &Number::from(1);
    }
    c.push(l);
    Polynomial::new(c).primitive_part()
}

/// The n-th cyclotomic polynomial, `(xⁿ - 1) / Π_{d | n, d < n} Φ_d`.
fn cyclotomic(n: usize) -> Z {
    let mut c = vec![0i64; n + 1];
    c[0] = -1;
    c[n] = 1;
    let mut f = ints(&c);
    for d in 1..n {
        if n.is_multiple_of(d) {
            f = f.div_exact(&cyclotomic(d)).unwrap();
        }
    }
    f
}

/// The Swinnerton-Dyer polynomial of the first k primes, `Π (x ± √2 ± √3 ± …)`, by `S_k(x) =
/// A(x)² - p·B(x)²` with `S_(k-1)(x + √p) = A(x) + √p·B(x)`.
fn swinnerton_dyer(k: usize) -> Z {
    let primes = [2i64, 3, 5, 7, 11];
    let mut f = ints(&[0, 1]);
    for &p in &primes[..k] {
        let n = f.degree().unwrap();
        let (mut a, mut b) = (Z::zero(), Z::zero());
        let mut d = f.clone();
        let mut fact = Number::from(1);
        let mut pp = Number::from(1);
        for j in 0..=n {
            if j > 0 {
                d = d.derivative();
                fact = &fact * &Number::from(j as i64);
            }
            let term = Polynomial::new(d.coeffs().iter().map(|c| &(c * &pp) / &fact).collect());
            if j % 2 == 0 {
                a = &a + &term;
            } else {
                b = &b + &term;
                pp = &pp * &Number::from(p);
            }
        }
        f = &(&a * &a) - &(&(&b * &b) * &ints(&[p]));
    }
    f
}

#[test]
fn edge_cases() {
    assert_eq!(Z::zero().factor(), (Number::from(0), vec![]));
    assert_eq!(Z::zero().squarefree(), (Number::from(0), vec![]));
    assert!(!Z::zero().is_irreducible());
    assert_eq!(ints(&[-6]).factor(), (Number::from(-6), vec![]));
    assert_eq!(ints(&[-6]).squarefree(), (Number::from(-6), vec![]));
    assert!(!ints(&[5]).is_irreducible());
    assert_eq!(
        ints(&[4, -6]).factor(),
        (Number::from(-2), vec![(ints(&[-2, 3]), 1)])
    );
    assert!(ints(&[4, -6]).is_irreducible());
    assert!(ints(&[2, 2]).is_irreducible());
    assert!(!ints(&[0, 0, 1]).is_irreducible());
    assert!(!ints(&[1, 2, 1]).is_irreducible());
    // Powers of x and their products.
    assert_eq!(
        ints(&[0, 0, 0, -5]).factor(),
        (Number::from(-5), vec![(ints(&[0, 1]), 3)])
    );
    assert_eq!(
        ints(&[0, 0, 1, 1]).factor(),
        (
            Number::from(1),
            vec![(ints(&[0, 1]), 2), (ints(&[1, 1]), 1)]
        )
    );
    assert_eq!(
        ints(&[0, 0, 1, 1]).squarefree(),
        (
            Number::from(1),
            vec![(ints(&[0, 1]), 2), (ints(&[1, 1]), 1)]
        )
    );
    // x⁴ + 1, reducible modulo every prime, irreducible.
    assert_eq!(
        ints(&[1, 0, 0, 0, 1]).factor().1,
        vec![(ints(&[1, 0, 0, 0, 1]), 1)]
    );
    // x⁴ + 4 = (x² - 2x + 2)(x² + 2x + 2).
    assert_eq!(
        ints(&[4, 0, 0, 0, 1]).factor().1,
        vec![(ints(&[2, -2, 1]), 1), (ints(&[2, 2, 1]), 1)]
    );
}

#[test]
fn products_of_eisenstein_polynomials() {
    let mut s = 0x3141_5926_5358_9793;
    for _ in 0..size(30, 3) {
        let mut want: Vec<(Z, u32)> = Vec::new();
        for _ in 0..1 + below(&mut s, size(4, 2) as u64) {
            let d = 1 + below(&mut s, size(7, 3) as u64) as usize;
            let bytes = 1 + below(&mut s, 12) as usize;
            let g = eisenstein(&mut s, d, bytes);
            let e = 1 + below(&mut s, 3) as u32;
            match want.iter_mut().find(|(h, _)| *h == g) {
                Some(x) => x.1 += e,
                None => want.push((g, e)),
            }
        }
        let mut c = number(&mut s, 6);
        if c.is_zero() {
            c = Number::from(-7);
        }
        let f = product(&c, &want);
        let (c2, fs) = f.factor();
        check_shape(&f, &c2, &fs);
        assert_eq!(fs, sorted(want.clone()), "{f}");
        assert_eq!(c2, c);
        for (g, _) in &fs {
            assert!(g.is_irreducible(), "{g}");
        }
        // The square-free decomposition groups them by multiplicity.
        let (c3, sq) = f.squarefree();
        assert_eq!(product(&c3, &sq), f);
        for (g, e) in &sq {
            let want_g = want
                .iter()
                .filter(|(_, e2)| e2 == e)
                .fold(ints(&[1]), |a, (h, _)| &a * h);
            assert_eq!(*g, want_g);
        }
        // Over the rationals, the same factors.
        let q = f.map(|x| Rational::new(x.clone(), Number::from(6)));
        let (cq, fq) = q.factor();
        assert_eq!(fq, fs);
        assert_eq!(cq, Rational::new(c2.clone(), Number::from(6)));
        assert!(q.is_irreducible() == (fs.len() == 1 && fs[0].1 == 1));
        let (cs, sqq) = q.squarefree();
        assert_eq!(sqq, sq);
        assert_eq!(cs, Rational::new(c3, Number::from(6)));
    }
}

#[test]
fn cyclotomic_polynomials_and_x_to_the_n_minus_one() {
    for n in 1..=size(60, 6) {
        let phi = cyclotomic(n);
        assert!(phi.is_irreducible(), "Φ_{n}");
        let mut c = vec![0i64; n + 1];
        c[0] = -1;
        c[n] = 1;
        let f = ints(&c);
        let (k, fs) = f.factor();
        check_shape(&f, &k, &fs);
        let want: Vec<(Z, u32)> = (1..=n)
            .filter(|d| n % d == 0)
            .map(|d| (cyclotomic(d), 1))
            .collect();
        assert_eq!(fs, sorted(want), "x^{n} - 1");
    }
    if !cfg!(miri) {
        for &n in &[105usize, 120] {
            let mut c = vec![0i64; n + 1];
            c[0] = -1;
            c[n] = 1;
            let fs = ints(&c).factor().1;
            assert_eq!(fs.len(), (1..=n).filter(|d| n % d == 0).count());
            assert!(fs.contains(&(cyclotomic(n), 1)));
        }
    }
}

#[test]
fn swinnerton_dyer_polynomials_are_irreducible() {
    for k in 1..=size(4, 2) {
        let f = swinnerton_dyer(k);
        assert_eq!(f.degree(), Some(1 << k));
        assert!(f.is_irreducible(), "S_{k}");
        assert_eq!(f.factor(), (Number::from(1), vec![(f.clone(), 1)]));
        // A product of two of them, and a square.
        if k >= 2 {
            let g = swinnerton_dyer(k - 1);
            let h = &(&f * &g) * &g;
            assert_eq!(h.factor().1, vec![(g.clone(), 2), (f.clone(), 1)]);
        }
    }
}

#[test]
fn swinnerton_dyer_of_five_primes() {
    // 16 factors modulo every prime, 2^15 subsets: a millisecond in release.
    if cfg!(miri) || cfg!(debug_assertions) {
        return;
    }
    let f = swinnerton_dyer(5);
    assert!(f.is_irreducible());
}

#[test]
fn products_of_many_linear_factors() {
    for n in [1i64, 2, 5, size(12, 3) as i64] {
        let mut f = ints(&[1]);
        let mut want = Vec::new();
        for i in -n..=n {
            // (k x - i) for k = 1, 2.
            let k = 1 + (i.rem_euclid(2));
            let g = Polynomial::new(vec![Number::from(-i), Number::from(k)]).primitive_part();
            f = &f * &g;
            want.push((g, 1));
        }
        let f = &f * &ints(&[-3]);
        let (c, fs) = f.factor();
        check_shape(&f, &c, &fs);
        assert_eq!(fs, sorted(want));
    }
}

#[test]
fn huge_coefficients() {
    let mut s = 0x2718_2818_2845_9045;
    for _ in 0..size(4, 1) {
        let g = eisenstein(&mut s, 3, size(120, 12));
        let h = eisenstein(&mut s, 4, size(120, 12));
        let k = eisenstein(&mut s, 2, size(60, 8));
        let f = &(&g * &h) * &(&k * &k);
        let (c, fs) = f.factor();
        check_shape(&f, &c, &fs);
        assert_eq!(fs, sorted(vec![(g, 1), (h, 1), (k, 2)]));
    }
}

#[test]
fn random_polynomials_round_trip() {
    let mut s = 0x0123_4567_89AB_CDEF;
    for _ in 0..size(40, 4) {
        let f = &poly(&mut s, size(8, 4), 3) * &poly(&mut s, size(8, 3), 9);
        let (c, fs) = f.factor();
        if f.is_zero() {
            assert!(fs.is_empty() && c.is_zero());
            continue;
        }
        check_shape(&f, &c, &fs);
        for (g, _) in &fs {
            assert!(g.is_irreducible(), "{g}");
        }
    }
}

/// The integers of the decimal list `s`.
fn parse_list(s: &str) -> Vec<Number> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',')
        .map(|x| Number::from_str_radix(x, 10, ByteOrder::BigEndian).unwrap())
        .collect()
}

#[test]
fn pari_factor() {
    let data = include_str!("data/factor.txt");
    let mut cases = 0;
    for line in data.lines() {
        let f: Vec<&str> = line.split(';').collect();
        if f[0] != "z" {
            continue;
        }
        if cfg!(miri) && cases > 3 {
            break;
        }
        let poly = Polynomial::new(parse_list(f[1]));
        if cfg!(debug_assertions) && poly.degree() > Some(60) {
            continue;
        }
        let content = &parse_list(f[2])[0];
        let want: Vec<(Z, u32)> = f[3]
            .split('|')
            .filter(|x| !x.is_empty())
            .map(|t| {
                let (e, c) = t.split_once(':').unwrap();
                (Polynomial::new(parse_list(c)), e.parse().unwrap())
            })
            .collect();
        let (c, fs) = poly.factor();
        assert_eq!((&c, &fs), (content, &sorted(want)), "{line}");
        cases += 1;
    }
    assert!(cases > 3);
}
