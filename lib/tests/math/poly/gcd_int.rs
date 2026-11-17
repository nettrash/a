//! Greatest common divisors, least common multiples, resultants and discriminants over the
//! integers and the rationals, and over the floats: against the gcd modulo a prime by the core's
//! division (whose degree the gcd over the integers must have for a prime that divides neither
//! leading coefficient), exact division, the Sylvester determinant by Bareiss's elimination, and
//! the products over known roots.

use super::common::*;
use a::math::complex::Complex;
use a::math::numbers::Number;
use a::math::poly::{ModPolynomial, Polynomial};
use a::math::rational::Rational;

/// The prime 2^61 - 1.
fn prime() -> Number {
    Number::from((1u64 << 61) - 1)
}

/// The degree of the monic gcd modulo `p` by Euclid's algorithm on the core's division.
fn degree_mod(a: &Polynomial<Number>, b: &Polynomial<Number>, p: &Number) -> Option<usize> {
    let (mut x, mut y) = (
        ModPolynomial::from_polynomial(a, p),
        ModPolynomial::from_polynomial(b, p),
    );
    while !y.is_zero() {
        let r = x.div_rem(&y).1;
        x = y;
        y = r;
    }
    x.degree()
}

/// Checks `g = gcd(a, b)`: positive leading coefficient, divides both, the content right, and its
/// degree that of the gcd modulo a prime dividing neither leading coefficient.
fn check_gcd(a: &Polynomial<Number>, b: &Polynomial<Number>, g: &Polynomial<Number>) {
    if a.is_zero() && b.is_zero() {
        assert!(g.is_zero());
        return;
    }
    assert!(
        !g.leading().unwrap().is_negative(),
        "sign of gcd({a}, {b}) = {g}"
    );
    assert!(a.div_exact(g).is_some(), "{g} divides {a}");
    assert!(b.div_exact(g).is_some(), "{g} divides {b}");
    assert_eq!(g.content(), a.content().gcd(&b.content()), "content");
    let p = prime();
    if a.is_zero() || b.is_zero() {
        let n = if a.is_zero() { b } else { a };
        assert_eq!(g.degree(), n.degree());
        return;
    }
    let lc = a.leading().unwrap() * b.leading().unwrap();
    if !lc.rem_euclid(&p).is_zero() {
        assert_eq!(g.degree(), degree_mod(a, b, &p), "degree of gcd({a}, {b})");
    }
}

/// The Sylvester determinant of `a` and `b` by Bareiss's fraction-free elimination.
fn sylvester(a: &Polynomial<Number>, b: &Polynomial<Number>) -> Number {
    let (Some(m), Some(n)) = (a.degree(), b.degree()) else {
        return Number::from(0);
    };
    if m + n == 0 {
        return Number::from(1);
    }
    let size = m + n;
    let mut mat = vec![vec![Number::from(0); size]; size];
    for i in 0..n {
        for (j, c) in a.coeffs().iter().rev().enumerate() {
            mat[i][i + j] = c.clone();
        }
    }
    for i in 0..m {
        for (j, c) in b.coeffs().iter().rev().enumerate() {
            mat[n + i][i + j] = c.clone();
        }
    }
    let mut sign = false;
    let mut prev = Number::from(1);
    for k in 0..size {
        let Some(piv) = (k..size).find(|&r| !mat[r][k].is_zero()) else {
            return Number::from(0);
        };
        if piv != k {
            mat.swap(piv, k);
            sign = !sign;
        }
        for i in k + 1..size {
            for j in k + 1..size {
                let t = &(&mat[i][j] * &mat[k][k]) - &(&mat[i][k] * &mat[k][j]);
                mat[i][j] = t.div_exact(&prev);
            }
            mat[i][k] = Number::from(0);
        }
        prev = mat[k][k].clone();
    }
    let d = mat[size - 1][size - 1].clone();
    if sign { -d } else { d }
}

#[test]
fn gcd_edge_cases() {
    let z = Polynomial::<Number>::zero();
    assert_eq!(z.gcd(&z), z);
    assert_eq!(ints(&[-4, -6]).gcd(&z), ints(&[4, 6]));
    assert_eq!(z.gcd(&ints(&[-4, -6])), ints(&[4, 6]));
    assert_eq!(ints(&[6]).gcd(&ints(&[4, 8])), ints(&[2]));
    assert_eq!(ints(&[1, 2, 1]).gcd(&ints(&[-1, -1])), ints(&[1, 1]));
    assert_eq!(ints(&[-12, -6, 6]).gcd(&ints(&[4, 4])), ints(&[2, 2]));
    // Coprime.
    assert_eq!(ints(&[1, 0, 1]).gcd(&ints(&[1, 1])), ints(&[1]));
    // Equal up to the content and the sign.
    let a = ints(&[3, -5, 0, 7, 2]);
    assert_eq!(
        (&a * &ints(&[-3])).gcd(&(&a * &ints(&[6]))),
        &a * &ints(&[3])
    );
    assert_eq!(z.lcm(&a), z);
    assert_eq!(ints(&[2, 2]).lcm(&ints(&[-3, 0, 3])), ints(&[-6, 0, 6]));
}

#[test]
fn gcd_of_random_polynomials() {
    let mut st = 0xC0FF_EE12_3456_789A;
    for _ in 0..size(120, 6) {
        let g = poly(&mut st, size(7, 4), size(12, 4));
        let a = &poly(&mut st, size(9, 5), size(12, 4)) * &g;
        let b = &poly(&mut st, size(9, 5), size(12, 4)) * &g;
        let h = a.gcd(&b);
        check_gcd(&a, &b, &h);
        assert_eq!(b.gcd(&a), h);
        // gcd · lcm = |a b| up to the sign of the leading coefficient.
        if !a.is_zero() && !b.is_zero() {
            let l = a.lcm(&b);
            assert!(!l.leading().unwrap().is_negative());
            let ab = &a * &b;
            let gl = &h * &l;
            assert!(gl == ab || gl == -&ab, "gcd·lcm of {a}, {b}");
        }
    }
}

#[test]
fn gcd_of_large_coefficients_and_degrees() {
    // Short polynomials of long coefficients take the heuristic gcd; long ones the multimodular
    // algorithm; both with a common factor and without.
    let mut st = 0x1357_9BDF_2468_ACE0;
    let cases: &[(usize, usize)] = if cfg!(miri) {
        &[(4, 40), (12, 9)]
    } else {
        &[(4, 400), (12, 150), (40, 40), (300, 8), (900, 9)]
    };
    for &(len, bytes) in cases {
        for _ in 0..size(2, 1) {
            let lg = 1 + below(&mut st, len as u64 / 2 + 1) as usize;
            let g = Polynomial::new(coeffs(&mut st, lg, bytes));
            let a = &Polynomial::new(coeffs(&mut st, len, bytes)) * &g;
            let b = &Polynomial::new(coeffs(&mut st, len - 1, bytes)) * &g;
            let h = a.gcd(&b);
            check_gcd(&a, &b, &h);
            let c = Polynomial::new(coeffs(&mut st, len, bytes));
            let d = Polynomial::new(coeffs(&mut st, len + 1, bytes));
            check_gcd(&c, &d, &c.gcd(&d));
        }
    }
}

#[test]
fn gcd_with_many_primes() {
    if cfg!(miri) {
        return;
    }
    // A gcd with coefficients of several words, which the images modulo primes below 2^63 need
    // several primes for, and leading coefficients that share factors.
    let mut st = 0xFEED_FACE_CAFE_BEEF;
    for _ in 0..size(3, 1) {
        let g = Polynomial::new(coeffs(&mut st, 30, 40));
        let mut u = Polynomial::new(coeffs(&mut st, 200, 2));
        let mut v = Polynomial::new(coeffs(&mut st, 199, 2));
        u = &u * &ints(&[6]);
        v = &v * &ints(&[10]);
        let (a, b) = (&g * &u, &g * &v);
        let h = a.gcd(&b);
        check_gcd(&a, &b, &h);
    }
}

#[test]
fn resultants_agree_with_sylvester() {
    let mut st = 0x0BAD_5EED_1234_5678;
    for _ in 0..size(60, 4) {
        let a = poly(&mut st, size(7, 5), size(6, 3));
        let b = poly(&mut st, size(7, 5), size(6, 3));
        assert_eq!(a.resultant(&b), sylvester(&a, &b), "res({a}, {b})");
    }
    assert_eq!(ints(&[2]).resultant(&ints(&[3])), Number::from(1));
    assert_eq!(ints(&[2]).resultant(&ints(&[1, 0, 1])), Number::from(4));
    assert_eq!(ints(&[1, 0, 1]).resultant(&ints(&[2])), Number::from(4));
    assert_eq!(
        Polynomial::<Number>::zero().resultant(&ints(&[1, 1])),
        Number::from(0)
    );
}

#[test]
fn resultants_over_known_roots() {
    // res(Π (x - rᵢ), b) = Π b(rᵢ), and disc(Π (x - rᵢ)) = Π_{i<j} (rᵢ - rⱼ)².
    let mut st = 0x7357_1357_2468_0000;
    for _ in 0..size(6, 1) {
        let k = 2 + below(&mut st, size(30, 4) as u64) as usize;
        let roots: Vec<i64> = (0..k).map(|_| below(&mut st, 2001) as i64 - 1000).collect();
        let mut f = ints(&[1]);
        for &r in &roots {
            f = &f * &ints(&[-r, 1]);
        }
        let lb = 1 + below(&mut st, size(40, 5) as u64) as usize;
        let b = Polynomial::new(coeffs(&mut st, lb, size(20, 4)));
        let mut want = Number::from(1);
        for &r in &roots {
            want = &want * &b.eval(&Number::from(r));
        }
        if b.is_zero() {
            want = Number::from(0);
        }
        assert_eq!(f.resultant(&b), want);
        let mut disc = Number::from(1);
        for i in 0..k {
            for j in i + 1..k {
                let d = Number::from(roots[i] - roots[j]);
                disc = &disc * &(&d * &d);
            }
        }
        assert_eq!(f.discriminant(), disc);
        // res(a, b) = (-1)^(deg a deg b) res(b, a).
        if let (Some(da), Some(db)) = (f.degree(), b.degree()) {
            let r = b.resultant(&f);
            let flip = if da * db % 2 == 1 { -r } else { r };
            assert_eq!(f.resultant(&b), flip);
        }
    }
}

#[test]
fn resultant_multiplicativity_at_size() {
    if cfg!(miri) {
        return;
    }
    let mut st = 0x1111_2222_3333_4444;
    for _ in 0..size(2, 1) {
        let a = Polynomial::new(coeffs(&mut st, 60, 12));
        let b = Polynomial::new(coeffs(&mut st, 40, 12));
        let c = Polynomial::new(coeffs(&mut st, 30, 12));
        assert_eq!(a.resultant(&(&b * &c)), &a.resultant(&b) * &a.resultant(&c));
        assert!((&a * &c).resultant(&(&b * &c)).is_zero());
    }
}

#[test]
fn discriminant_edge_cases() {
    assert_eq!(Polynomial::<Number>::zero().discriminant(), Number::from(0));
    assert_eq!(ints(&[5]).discriminant(), Number::from(0));
    assert_eq!(ints(&[1, 2]).discriminant(), Number::from(1));
    assert_eq!(ints(&[1, 0, 0, 3]).discriminant(), Number::from(-243));
    assert_eq!(ints(&[2, 3, 1]).discriminant(), Number::from(1));
}

#[test]
fn rational_gcd_resultant_and_xgcd() {
    let mut st = 0x2222_7777_ABCD_0001;
    for _ in 0..size(30, 3) {
        let g = rpoly(&mut st, size(5, 3), size(4, 2));
        let a = &rpoly(&mut st, size(6, 3), size(4, 2)) * &g;
        let b = &rpoly(&mut st, size(6, 3), size(4, 2)) * &g;
        let h = a.gcd(&b);
        // That of the primitive parts, monic.
        let (_, pa) = a.to_primitive();
        let (_, pb) = b.to_primitive();
        let want: Polynomial<Rational> = pa.gcd(&pb).into();
        assert_eq!(h, want.to_monic());
        let (h2, s, t) = a.xgcd(&b);
        assert_eq!(h2, h);
        assert_eq!(&(&s * &a) + &(&t * &b), h);
        // The resultant: the Sylvester determinant of the integer multiples.
        let (ca, qa) = a.to_primitive();
        let (cb, qb) = b.to_primitive();
        if let (Some(da), Some(db)) = (qa.degree(), qb.degree()) {
            let want =
                &(&Rational::from(sylvester(&qa, &qb)) * &ca.pow(db as i32)) * &cb.pow(da as i32);
            assert_eq!(a.resultant(&b), want);
        } else {
            assert!(
                a.resultant(&b).is_zero()
                    || a.is_zero()
                    || b.is_zero()
                    || a.degree() == Some(0)
                    || b.degree() == Some(0)
            );
        }
        let l = a.lcm(&b);
        if !a.is_zero() && !b.is_zero() {
            assert!((&l % &a).is_zero() && (&l % &b).is_zero() && l.is_monic());
        }
    }
    // disc(x² - 1/4) = 1
    let f = Polynomial::new(vec![
        Rational::new((-1).into(), 4.into()),
        0.into(),
        1.into(),
    ]);
    assert_eq!(f.discriminant(), Rational::from(1));
}

#[test]
fn float_gcd_resultant_and_discriminant() {
    // (x - 1)(x - 2) and (x - 1)(x - 3): exact in binary.
    let a = Polynomial::new(vec![2.0, -3.0, 1.0]);
    let b = Polynomial::new(vec![3.0, -4.0, 1.0]);
    assert_eq!(a.gcd(&b).coeffs(), [-1.0, 1.0]);
    let (g, s, t) = a.xgcd(&b);
    assert_eq!(g.coeffs(), [-1.0, 1.0]);
    assert_eq!((&(&s * &a) + &(&t * &b)).coeffs(), [-1.0, 1.0]);
    // res((x-1)(x-2), x - 4) = (4-1)(4-2) = 6; disc(x² - 3x + 2) = 1.
    assert_eq!(a.resultant(&Polynomial::new(vec![-4.0, 1.0])), 6.0);
    assert_eq!(a.discriminant(), 1.0);
    assert_eq!(a.lcm(&b).coeffs(), [-6.0, 11.0, -6.0, 1.0]);
    // Complex: (x - i)(x + i) = x² + 1 and x - i.
    let c = |re: f64, im: f64| Complex::new(re, im);
    let p = Polynomial::new(vec![c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]);
    let q = Polynomial::new(vec![c(0.0, -1.0), c(1.0, 0.0)]);
    assert_eq!(p.gcd(&q), q);
    assert_eq!(p.resultant(&q), c(0.0, 0.0));
    assert_eq!(p.discriminant(), c(-4.0, 0.0));
}

/// The subresultant of index `j` of `p` and `q` (degrees `m ≥ n > j`) by its definition: the
/// determinants, by Bareiss's elimination, of the rows `x^(n-j-1)p … p, x^(m-j-1)q … q` cut to the
/// top `m + n - 2j - 1` columns and the column of `xⁱ`.
fn subresultant(p: &Polynomial<Number>, q: &Polynomial<Number>, j: usize) -> Polynomial<Number> {
    let (m, n) = (p.degree().unwrap(), q.degree().unwrap());
    let rows = m + n - 2 * j;
    let width = m + n - j;
    // Row r as coefficients of x^(width-1) down to x^0.
    let mut base: Vec<Vec<Number>> = Vec::new();
    for k in (0..n - j).rev() {
        let mut row = vec![Number::from(0); width];
        for (i, c) in p.coeffs().iter().enumerate() {
            row[width - 1 - (i + k)] = c.clone();
        }
        base.push(row);
    }
    for k in (0..m - j).rev() {
        let mut row = vec![Number::from(0); width];
        for (i, c) in q.coeffs().iter().enumerate() {
            row[width - 1 - (i + k)] = c.clone();
        }
        base.push(row);
    }
    let mut out = Vec::new();
    for i in 0..=j {
        let cols: Vec<usize> = (0..rows - 1).chain([width - 1 - i]).collect();
        let mat: Vec<Vec<Number>> = base
            .iter()
            .map(|r| cols.iter().map(|&c| r[c].clone()).collect())
            .collect();
        out.push(det(mat));
    }
    Polynomial::new(out)
}

/// The determinant by Bareiss's elimination.
fn det(mut mat: Vec<Vec<Number>>) -> Number {
    let size = mat.len();
    let mut sign = false;
    let mut prev = Number::from(1);
    for k in 0..size {
        let Some(piv) = (k..size).find(|&r| !mat[r][k].is_zero()) else {
            return Number::from(0);
        };
        if piv != k {
            mat.swap(piv, k);
            sign = !sign;
        }
        for i in k + 1..size {
            for j in k + 1..size {
                let t = &(&mat[i][j] * &mat[k][k]) - &(&mat[i][k] * &mat[k][j]);
                mat[i][j] = t.div_exact(&prev);
            }
            mat[i][k] = Number::from(0);
        }
        prev = mat[k][k].clone();
    }
    let d = mat[size - 1][size - 1].clone();
    if sign { -d } else { d }
}

#[test]
fn subresultants_agree_with_their_definition() {
    let mut st = 0x5AB5_E5D1_7EAD_0042;
    for _ in 0..size(40, 3) {
        let m = 1 + below(&mut st, size(7, 4) as u64) as usize;
        let n = 1 + below(&mut st, m as u64) as usize;
        let mut p = Polynomial::new(coeffs(&mut st, m + 1, 3));
        let mut q = Polynomial::new(coeffs(&mut st, n + 1, 3));
        // Defective sequences: a common factor, or sparse inputs.
        if below(&mut st, 3) == 0 && n >= 2 {
            let g = Polynomial::new(coeffs(&mut st, 2, 2));
            p = &p.truncate(m) * &g;
            q = &q.truncate(n) * &g;
        }
        if below(&mut st, 3) == 0 {
            p = p.map(|c| {
                if c.bits() % 2 == 0 {
                    Number::from(0)
                } else {
                    c.clone()
                }
            });
        }
        let (Some(dm), Some(dn)) = (p.degree(), q.degree()) else {
            continue;
        };
        if dn == 0 || dm < dn {
            continue;
        }
        let s = p.subresultants(&q);
        let want: Vec<Polynomial<Number>> = (0..dn)
            .map(|j| subresultant(&p, &q, j))
            .filter(|s| !s.is_zero())
            .collect();
        assert_eq!(s, want, "subresultants of {p} and {q}");
        if let Some(s0) = s.first()
            && s0.degree() == Some(0)
        {
            assert_eq!(s0.coeffs()[0], p.resultant(&q));
        }
    }
}

/// The oracle's extended gcd over the rationals: Euclid's algorithm on the core's division.
fn euclid_xgcd(
    a: &Polynomial<Rational>,
    b: &Polynomial<Rational>,
) -> (
    Polynomial<Rational>,
    Polynomial<Rational>,
    Polynomial<Rational>,
) {
    let (mut r0, mut r1) = (a.clone(), b.clone());
    let (mut s0, mut s1) = (Polynomial::one(), Polynomial::zero());
    let (mut t0, mut t1) = (Polynomial::zero(), Polynomial::one());
    while !r1.is_zero() {
        let (q, r) = r0.div_rem(&r1);
        let s = &s0 - &(&q * &s1);
        let t = &t0 - &(&q * &t1);
        (r0, r1) = (r1, r);
        (s0, s1) = (s1, s);
        (t0, t1) = (t1, t);
    }
    match r0.leading().cloned() {
        None => (r0, Polynomial::zero(), Polynomial::zero()),
        Some(l) => {
            let inv = l.recip();
            (r0.scale(&inv), s0.scale(&inv), t0.scale(&inv))
        }
    }
}

#[test]
fn rational_xgcd_agrees_with_euclid() {
    // The multimodular cofactors are the unique ones of the least degrees, Euclid's.
    let mut st = 0x4242_1717_9090_3131;
    for _ in 0..size(25, 2) {
        let lg = below(&mut st, size(6, 3) as u64) as usize;
        let g = Polynomial::new(rats_of(&mut st, lg));
        let la = below(&mut st, size(20, 4) as u64) as usize;
        let lb = below(&mut st, size(20, 4) as u64) as usize;
        let a = &Polynomial::new(rats_of(&mut st, la)) * &g;
        let b = &Polynomial::new(rats_of(&mut st, lb)) * &g;
        assert_eq!(a.xgcd(&b), euclid_xgcd(&a, &b), "xgcd of {a} and {b}");
    }
}

/// `len` random rationals of short numerators and denominators.
fn rats_of(st: &mut u64, len: usize) -> Vec<Rational> {
    (0..len).map(|_| rational(st, 3)).collect()
}

#[test]
fn subresultants_across_long_gaps() {
    // q of degree 6 and p = (x + 2)q + r with r of degree 2, 1 or 0: the first step falls 4, 5
    // or 6 degrees, and Lazard's powering takes its multiplying steps (δ - 1 = 3, 4, 5).
    let q = ints(&[1, 3, 0, -2, 0, 0, 1]);
    for r in [ints(&[7, 2, 5]), ints(&[4, -3]), ints(&[9])] {
        let p = &(&ints(&[2, 1]) * &q) + &r;
        let want: Vec<Polynomial<Number>> = (0..6)
            .map(|j| subresultant(&p, &q, j))
            .filter(|s| !s.is_zero())
            .collect();
        assert_eq!(p.subresultants(&q), want);
        // With a content in the gap's leading coefficient.
        let q3 = &q * &ints(&[3]);
        let p3 = &(&ints(&[2, 1]) * &q3) + &(&r * &ints(&[5]));
        let want: Vec<Polynomial<Number>> = (0..6)
            .map(|j| subresultant(&p3, &q3, j))
            .filter(|s| !s.is_zero())
            .collect();
        assert_eq!(p3.subresultants(&q3), want);
    }
}

#[test]
fn many_primes_for_long_coefficients() {
    // Regression: the primes past the first window of the sieve (about 760 below 2^62; a dozen
    // under Miri). Coefficients of 1000 bits at degrees 5 to 30 need up to a thousand primes for
    // the resultant: against the Sylvester determinant and the subresultant chain's S₀; the gcd
    // and the rational xgcd of such polynomials too.
    let mut st = 0x1000_B175_DEAD_0005;
    let (bytes, degrees): (usize, &[usize]) = if cfg!(miri) {
        (12, &[3, 5])
    } else {
        (125, &[5, 10, 30])
    };
    for &d in degrees {
        let a = Polynomial::new(coeffs(&mut st, d + 1, bytes));
        let b = Polynomial::new(coeffs(&mut st, d, bytes));
        let r = a.resultant(&b);
        if d <= 10 {
            assert_eq!(r, sylvester(&a, &b), "res at degree {d}");
        }
        let s0 = a.subresultants(&b);
        let want = match s0.first() {
            Some(s) if s.degree() == Some(0) => s.coeffs()[0].clone(),
            _ => Number::from(0),
        };
        assert_eq!(r, want, "res against the chain at degree {d}");
        let g = Polynomial::new(coeffs(&mut st, d / 2 + 1, bytes));
        let (ga, gb) = (&a * &g, &b * &g);
        check_gcd(&ga, &gb, &ga.gcd(&gb));
        let qa: Polynomial<Rational> = (&ga).into();
        let qb: Polynomial<Rational> = (&gb).into();
        let (h, s, t) = qa.xgcd(&qb);
        assert_eq!(&(&s * &qa) + &(&t * &qb), h);
        assert_eq!(h, qa.gcd(&qb));
    }
}
