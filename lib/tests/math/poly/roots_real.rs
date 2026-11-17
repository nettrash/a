//! Real roots of polynomials over the integers and the rationals: the isolating intervals by
//! Descartes's rule and by continued fractions, the counts, and the correctly rounded roots,
//! against an independent oracle of Sturm sequences over the rationals and the gcds of the
//! derivatives for the multiplicities.

use super::common::*;
use a::math::float::{BigFloat, Round};
use a::math::numbers::Number;
use a::math::poly::Polynomial;
use a::math::rational::Rational;

type Q = Polynomial<Rational>;

/// The sign of `p(x)`.
fn sign(p: &Q, x: &Rational) -> i32 {
    let v = p.eval(x);
    if v.is_zero() {
        0
    } else if v.is_negative() {
        -1
    } else {
        1
    }
}

/// The Sturm sequence of `p`.
fn sturm(p: &Q) -> Vec<Q> {
    let mut s = vec![p.clone(), p.derivative()];
    while !s[s.len() - 1].is_zero() {
        let n = s.len();
        let r = -(&s[n - 2] % &s[n - 1]);
        s.push(r);
    }
    s.pop();
    s
}

/// The sign variations of the sequence at `x` (zeros skipped).
fn var_at(s: &[Q], x: &Rational) -> usize {
    let signs: Vec<i32> = s.iter().map(|q| sign(q, x)).filter(|&v| v != 0).collect();
    signs.windows(2).filter(|w| w[0] != w[1]).count()
}

/// The sign variations at ±∞.
fn var_inf(s: &[Q], neg: bool) -> usize {
    let signs: Vec<i32> = s
        .iter()
        .map(|q| {
            let l = q.leading().unwrap();
            let mut v = if l.is_negative() { -1 } else { 1 };
            if neg && q.degree().unwrap() % 2 == 1 {
                v = -v;
            }
            v
        })
        .collect();
    signs.windows(2).filter(|w| w[0] != w[1]).count()
}

/// The square-free part of `p`, over the rationals.
fn squarefree(p: &Q) -> Q {
    let g = p.gcd(&p.derivative());
    (p / &g).to_monic()
}

/// The number of distinct roots of `p` in `(a, b]` by Sturm's theorem (`p` not zero, `a < b`).
fn count_in(p: &Q, a: &Rational, b: &Rational) -> usize {
    let s = sturm(&squarefree(p));
    var_at(&s, a) - var_at(&s, b)
}

/// The number of distinct real roots of `p`.
fn count_all(p: &Q) -> usize {
    if p.degree() == Some(0) {
        return 0;
    }
    let s = sturm(&squarefree(p));
    var_inf(&s, true) - var_inf(&s, false)
}

/// The multiplicity of the root of `p` isolated in `[a, b]` (that of the only root of the
/// square-free part there): the largest `m` for which `gcd(p, p', …, p^(m−1))` has a root in it.
fn multiplicity(p: &Q, a: &Rational, b: &Rational) -> u32 {
    let mut g = p.clone();
    let mut d = p.clone();
    let mut m = 0;
    loop {
        let here = if a == b {
            g.eval(a).is_zero()
        } else {
            g.degree().unwrap_or(0) > 0 && count_in(&g, a, b) - usize::from(g.eval(b).is_zero()) > 0
        };
        if !here {
            return m;
        }
        m += 1;
        d = d.derivative();
        g = g.gcd(&d);
        if g.degree().unwrap_or(0) == 0 {
            return m;
        }
    }
}

/// Checks the intervals of `p` against the oracle.
fn check(p: &Polynomial<Number>, iv: &[(Rational, Rational, u32)]) {
    let q: Q = p.into();
    assert_eq!(iv.len(), count_all(&q), "{p}: {iv:?}");
    for (k, (a, b, m)) in iv.iter().enumerate() {
        assert!(a <= b);
        if k > 0 {
            // Increasing and disjoint: an open interval may end where the next begins.
            let (_, pb, _) = &iv[k - 1];
            assert!(pb <= a, "{p}: {iv:?}");
            if pb == a {
                assert!(iv[k - 1].0 != *pb || a != b, "{p}: two roots at one point");
            }
        }
        if a == b {
            assert!(q.eval(a).is_zero(), "{p}: {a} is not a root");
        } else {
            // One root in the open interval, none at its ends.
            assert!(
                !q.eval(a).is_zero() && !q.eval(b).is_zero(),
                "{p}: an end is a root"
            );
            assert_eq!(count_in(&q, a, b), 1, "{p}: ({a}, {b})");
        }
        assert_eq!(multiplicity(&q, a, b), *m, "{p}: ({a}, {b})");
    }
}

/// Chebyshev's T_n.
fn chebyshev(n: usize) -> Polynomial<Number> {
    let mut a = ints(&[1]);
    let mut b = ints(&[0, 1]);
    for _ in 1..n {
        let c = &(&b * &ints(&[0, 2])) - &a;
        a = b;
        b = c;
    }
    if n == 0 { a } else { b }
}

/// ∏ (x − i), i = 1..=n.
fn wilkinson(n: i64) -> Polynomial<Number> {
    (1..=n).fold(ints(&[1]), |p, i| &p * &ints(&[-i, 1]))
}

/// Mignotte's x^n − 2(a·x − 1)²: two real roots within about a^(−n/2) of each other near 1/a.
fn mignotte(n: usize, a: i64) -> Polynomial<Number> {
    let l = ints(&[-1, a]);
    &Polynomial::monomial(Number::from(1), n) - &(&l * &l).scale(&Number::from(2))
}

#[test]
fn known_intervals() {
    // x² − 2
    let iv = ints(&[-2, 0, 1]).real_root_intervals();
    assert_eq!(iv.len(), 2);
    check(&ints(&[-2, 0, 1]), &iv);
    // x³ − x: −1, 0, 1, exact points.
    let iv = ints(&[0, -1, 0, 1]).real_root_intervals();
    assert_eq!(
        iv,
        [-1, 0, 1]
            .map(|x| (Rational::from(x), Rational::from(x), 1))
            .to_vec()
    );
    // (x − 1)²(x² − 2) and x⁵(x + 3)³(4x − 1)
    let p = ints(&[-2, 4, -1, -2, 1]);
    check(&p, &p.real_root_intervals());
    let p = &(&Polynomial::monomial(Number::from(1), 5) * &ints(&[3, 1]).pow(3)) * &ints(&[-1, 4]);
    let iv = p.real_root_intervals();
    assert_eq!(iv.iter().map(|x| x.2).collect::<Vec<_>>(), [3, 5, 1]);
    check(&p, &iv);
    // No real roots; constants; degree 1 with a huge root.
    assert!(ints(&[1, 0, 1]).real_root_intervals().is_empty());
    assert!(ints(&[7]).real_root_intervals().is_empty());
    let big = Number::from(1) << 300usize;
    let p = Polynomial::new(vec![-(&big + &Number::from(1)), Number::from(3)]);
    check(&p, &p.real_root_intervals());
    check(&p, &p.real_root_intervals_cf());
}

#[test]
fn both_methods_agree_with_the_oracle() {
    let mut s = 0x5EED_0001;
    for t in 0..size(300, 4) {
        let n = 1 + below(&mut s, size(14, 6) as u64) as usize;
        let mut p = Polynomial::new(coeffs(&mut s, n + 1, 1 + (t % 9)));
        if p.degree().unwrap_or(0) == 0 {
            continue;
        }
        // Repeated factors and roots at 0 now and then.
        if t % 5 == 2 {
            let f = Polynomial::new(coeffs(&mut s, 3, 2));
            if !f.is_zero() {
                p = &p * &f.pow(2);
            }
        }
        if t % 7 == 3 {
            p = p.shift_left(2);
        }
        let d = p.real_root_intervals_descartes();
        check(&p, &d);
        let c = p.real_root_intervals_cf();
        check(&p, &c);
        assert_eq!(p.real_root_intervals().len(), d.len());
        assert_eq!(p.count_real_roots(), d.len());
    }
}

#[test]
fn many_real_roots() {
    let ns: &[usize] = if cfg!(miri) {
        &[1, 2, 7]
    } else {
        &[1, 2, 7, 20, 41, 64]
    };
    for &n in ns {
        let p = chebyshev(n);
        for iv in [
            p.real_root_intervals_descartes(),
            p.real_root_intervals_cf(),
        ] {
            assert_eq!(iv.len(), n);
            if n <= 20 {
                check(&p, &iv);
            }
        }
        let r = p.real_roots(30);
        assert!(r.iter().all(|(x, _)| x.abs() < 1));
        assert_eq!(p.count_real_roots(), n);
    }
    let ws: &[i64] = if cfg!(miri) { &[1, 5] } else { &[1, 5, 13, 30] };
    for &n in ws {
        let p = wilkinson(n);
        let iv = p.real_root_intervals();
        assert_eq!(iv.len(), n as usize);
        for (k, (a, b, _)) in iv.iter().enumerate() {
            let r = Rational::from(k as i64 + 1);
            assert!(a <= &r && &r <= b);
        }
        assert_eq!(p.real_root_intervals_cf().len(), n as usize);
    }
}

#[test]
fn close_roots() {
    // Mignotte's polynomials: two roots about a^(−n/2) apart.
    let cases: &[(usize, i64)] = if cfg!(miri) {
        &[(5, 10)]
    } else {
        &[(5, 10), (11, 100), (21, 1000), (40, 1 << 20)]
    };
    for &(n, a) in cases {
        let p = mignotte(n, a);
        let d = p.real_root_intervals_descartes();
        let c = p.real_root_intervals_cf();
        assert_eq!(d.len(), c.len());
        assert_eq!(d.len(), p.count_real_roots());
        if n <= 11 {
            check(&p, &d);
            check(&p, &c);
        }
        // The pair near 1/a is apart.
        let inv = Rational::new(1.into(), a.into());
        let eps = Rational::new(1.into(), Number::from(a) * Number::from(a));
        let r = p.real_roots(64);
        let near = r
            .iter()
            .filter(|(x, _)| (&Rational::try_from(x).unwrap() - &inv).abs() < eps)
            .count();
        assert_eq!(near, 2, "{n} {a}: {d:?}");
    }
}

#[test]
fn rational_polynomials() {
    // (x − 1/3)(x + 7/2)², by both methods.
    let third = Rational::new(1.into(), 3.into());
    let p = Polynomial::new(vec![-third.clone(), Rational::from(1)]);
    let q = Polynomial::new(vec![Rational::new(7.into(), 2.into()), Rational::from(1)]);
    let f = &p * &q.pow(2);
    let iv = f.real_root_intervals();
    assert_eq!(iv.len(), 2);
    assert_eq!(
        iv[0],
        (
            Rational::new((-7).into(), 2.into()),
            Rational::new((-7).into(), 2.into()),
            2
        )
    );
    assert!(iv[1].0 <= third && third <= iv[1].1 && iv[1].2 == 1);
    assert_eq!(f.real_root_intervals_descartes().len(), 2);
    assert_eq!(f.real_root_intervals_cf().len(), 2);
    assert_eq!(f.count_real_roots(), 2);
    let r = f.real_roots(64);
    assert_eq!(r[0], (BigFloat::from(-3.5), 2));
    assert_eq!(
        r[1].0,
        BigFloat::from_rational(&third, 64, Round::Nearest).0
    );
}

/// The interval of the reals that round to nearest to `x` at its precision, as rationals.
fn cell(x: &BigFloat) -> (Rational, Rational) {
    let p = x.prec();
    let (m, e) = x.mantissa_exponent().unwrap();
    let two = |k: i64| {
        if k >= 0 {
            Rational::from(Number::from(1) << k as usize)
        } else {
            Rational::new(1.into(), Number::from(1) << (-k) as usize)
        }
    };
    let am = m.abs();
    let v = Rational::from(am.clone()) * two(e);
    let half = two(e - 1);
    let up = &v + &half;
    let pow2 = am == (Number::from(1) << (p - 1) as usize);
    let down = if pow2 { &v - &two(e - 2) } else { &v - &half };
    if m.is_negative() {
        (-up, -down)
    } else {
        (down, up)
    }
}

#[test]
fn real_roots_are_correctly_rounded() {
    let mut s = 0xABCD_EF01;
    for t in 0..size(120, 2) {
        let n = 1 + below(&mut s, size(10, 5) as u64) as usize;
        let p = Polynomial::new(coeffs(&mut s, n + 1, 1 + (t % 6)));
        if p.degree().unwrap_or(0) == 0 {
            continue;
        }
        let q: Q = (&p).into();
        let sq = squarefree(&q);
        let prec = [1u64, 2, 24, 53, 113, 300][t % 6];
        let r = p.real_roots(prec);
        assert_eq!(r.len(), count_all(&q));
        let iv = p.real_root_intervals();
        for ((x, m), (a, b, mm)) in r.iter().zip(&iv) {
            assert_eq!(m, mm);
            assert_eq!(x.prec(), prec);
            if a == b {
                assert_eq!(*x, BigFloat::from_rational(a, prec, Round::Nearest).0);
                continue;
            }
            // The root is in the cell of x: the square-free part changes sign over it (or the
            // root is an end of it, which then rounds to x).
            let (lo, hi) = cell(x);
            let (lo, hi) = (lo.max(a.clone()), hi.min(b.clone()));
            let (sl, sh) = (sign(&sq, &lo), sign(&sq, &hi));
            if sl == 0 || sh == 0 {
                let e = if sl == 0 { &lo } else { &hi };
                assert_eq!(*x, BigFloat::from_rational(e, prec, Round::Nearest).0);
            } else {
                assert!(sl != sh, "{p} prec {prec}: {x} [{lo}, {hi}]");
            }
        }
    }
}

#[test]
fn real_roots_of_known_values() {
    let precs: &[u64] = if cfg!(miri) {
        &[1, 2, 53]
    } else {
        &[1, 2, 3, 53, 64, 200, 1000, 5000]
    };
    for &prec in precs {
        for c in [2i64, 3, 5, 1 << 40] {
            let r = ints(&[-c, 0, 1]).real_roots(prec);
            let e = BigFloat::from(c).sqrt_round(prec, Round::Nearest).0;
            assert_eq!(r, vec![(-e.clone(), 1), (e, 1)], "{c} {prec}");
        }
        // The cube root of 10, and 2^(1/7).
        let r = ints(&[-10, 0, 0, 1]).real_roots(prec);
        assert_eq!(
            r[0].0,
            BigFloat::from(10).cbrt_round(prec, Round::Nearest).0
        );
        let r = ints(&[-2, 0, 0, 0, 0, 0, 0, 1]).real_roots(prec);
        assert_eq!(
            r[0].0,
            BigFloat::from(2).nth_root_round(7, prec, Round::Nearest).0
        );
    }
    // Roots within 2^−(2·prec) of a midpoint: √(M² ± 1) for the midpoint M = 2^prec + 1 of the
    // floats 2^prec and 2^prec + 2, which Newton's method at prec + 32 bits cannot tell apart.
    for prec in [8u64, 20, 53, 100] {
        let m = (Number::from(1) << prec as usize) + Number::from(1);
        for d in [-1i64, 1] {
            let c = &(&m * &m) + &Number::from(d);
            let p = Polynomial::new(vec![-c.clone(), Number::from(0), Number::from(1)]);
            let r = p.real_roots(prec);
            let e = BigFloat::from(&c).sqrt_round(prec, Round::Nearest).0;
            assert_eq!(r[1].0, e, "prec {prec} d {d}");
        }
    }
    // Ties: 5 at 2 bits is a midpoint of 4 and 6, and rounds to the even mantissa.
    let r = ints(&[-5, 1]).real_roots(2);
    assert_eq!(
        r[0].0,
        BigFloat::from_rational(&Rational::from(5), 2, Round::Nearest).0
    );
    let r = ints(&[-5, 2]).real_roots(2);
    assert_eq!(
        r[0].0,
        BigFloat::from_rational(&Rational::new(5.into(), 2.into()), 2, Round::Nearest).0
    );
    // Exact roots of Wilkinson's polynomial at 1 bit and 60.
    for prec in [1u64, 60] {
        let r = wilkinson(12).real_roots(prec);
        for (k, (x, m)) in r.iter().enumerate() {
            assert_eq!(*m, 1);
            assert_eq!(
                *x,
                BigFloat::from_rational(&Rational::from(k as i64 + 1), prec, Round::Nearest).0
            );
        }
    }
}

#[test]
fn counts() {
    assert_eq!(ints(&[-1, 0, 1]).count_real_roots(), 2);
    assert_eq!(ints(&[1, 0, 1]).count_real_roots(), 0);
    assert_eq!(ints(&[0, 0, 0, 1]).count_real_roots(), 1);
    assert_eq!(ints(&[5]).count_real_roots(), 0);
    let (c, w) = if cfg!(miri) { (8, 6) } else { (100, 40) };
    assert_eq!(chebyshev(c).count_real_roots(), c);
    assert_eq!(wilkinson(w as i64).count_real_roots(), w);
    let mut s = 0x77;
    for _ in 0..size(100, 2) {
        let p = poly(&mut s, size(25, 7), 6);
        if p.is_zero() {
            continue;
        }
        let q: Q = (&p).into();
        assert_eq!(p.count_real_roots(), count_all(&q), "{p}");
    }
}

#[test]
fn large_degrees() {
    // Random dense polynomials of degree a few hundred: the two methods agree and every
    // interval holds a sign change of the polynomial.
    let mut s = 0x1111;
    for &n in &[100usize, 300] {
        if cfg!(miri) {
            break;
        }
        let p = Polynomial::new(coeffs(&mut s, n + 1, 8));
        let d = p.real_root_intervals_descartes();
        let c = p.real_root_intervals_cf();
        assert_eq!(d.len(), c.len());
        assert_eq!(p.count_real_roots(), d.len());
        let q: Q = (&p).into();
        for (a, b, _) in d.iter().chain(&c) {
            if a != b {
                assert!(sign(&q, a) * sign(&q, b) < 0);
            }
        }
    }
}

#[test]
#[should_panic(expected = "the zero polynomial has every number as a root")]
fn intervals_of_zero_panic() {
    Polynomial::<Number>::zero().real_root_intervals();
}

#[test]
#[should_panic(expected = "the zero polynomial has every number as a root")]
fn count_of_zero_panics() {
    Polynomial::<Rational>::zero().count_real_roots();
}

#[test]
#[should_panic(expected = "precision 0 is not in 1..=2^40")]
fn real_roots_at_precision_0_panic() {
    ints(&[-2, 0, 1]).real_roots(0);
}

/// PARI 2.19's `polsturm` and `polrootsreal` (40 digits) for 60 random polynomials, some with
/// repeated factors and roots at 0 (`data/roots_pari.txt`, written by `data/gen_pari.gp`).
#[test]
fn against_pari() {
    let text = include_str!("data/roots_pari.txt");
    let mut lines = text.lines();
    let mut cases = 0;
    while let (Some(p), Some(n), Some(r)) = (lines.next(), lines.next(), lines.next()) {
        cases += 1;
        if cfg!(miri) && cases > 1 {
            break;
        }
        let c: Vec<Number> = p[2..].split(' ').map(|x| x.parse().unwrap()).collect();
        let p = Polynomial::new(c);
        let n: usize = n[2..].parse().unwrap();
        assert_eq!(p.count_real_roots(), n, "{p}");
        let pari: Vec<BigFloat> = r[2..]
            .split_whitespace()
            .map(|x| x.parse::<BigFloat>().unwrap())
            .collect();
        let ours: Vec<BigFloat> = p
            .real_roots(128)
            .into_iter()
            .flat_map(|(x, m)| core::iter::repeat_n(x, m as usize))
            .collect();
        assert_eq!(ours.len(), pari.len(), "{p}");
        for (x, y) in ours.iter().zip(&pari) {
            // 40 digits: within 10^−38 of the larger magnitude, or of 1 near 0.
            let d = (x - y).abs();
            let scale = if y.abs() > 1 {
                y.abs()
            } else {
                BigFloat::from(1)
            };
            assert!(d < scale * BigFloat::from(1e-38), "{p}: {x} against {y}");
        }
    }
    assert!(cfg!(miri) || cases == 60);
}

/// Checks that each root of `p` at `prec` bits is correctly rounded, by the signs of the
/// square-free part at the ends of its rounding cell.
fn check_rounding(p: &Polynomial<Number>, prec: u64) {
    let q: Q = p.into();
    let sq = squarefree(&q);
    let r = p.real_roots(prec);
    let iv = p.real_root_intervals();
    assert_eq!(r.len(), iv.len());
    for ((x, _), (a, b, _)) in r.iter().zip(&iv) {
        if a == b {
            assert_eq!(*x, BigFloat::from_rational(a, prec, Round::Nearest).0);
            continue;
        }
        let (lo, hi) = cell(x);
        let (lo, hi) = (lo.max(a.clone()), hi.min(b.clone()));
        let (sl, sh) = (sign(&sq, &lo), sign(&sq, &hi));
        if sl == 0 || sh == 0 {
            let e = if sl == 0 { &lo } else { &hi };
            assert_eq!(*x, BigFloat::from_rational(e, prec, Round::Nearest).0);
        } else {
            assert!(sl != sh, "{p} prec {prec}: {x}");
        }
    }
}

#[test]
fn wide_evaluations_round_correctly() {
    // From eight limbs and the degree 8 the refinement evaluates by rectangular splitting:
    // random polynomials (roots inside and outside the unit circle, of either sign), Chebyshev's
    // (ill conditioned near ±1), Wilkinson's (roots up to 12) and roots of mixed sizes, the
    // products of 2^k·x − 3.
    let precs: &[u64] = if cfg!(miri) {
        &[600]
    } else {
        &[400, 600, 1100, 2000]
    };
    let mut s = 0x0D1E_5EED;
    let mut ps = vec![
        chebyshev(if cfg!(miri) { 8 } else { 24 }),
        wilkinson(if cfg!(miri) { 8 } else { 12 }),
    ];
    ps.push(
        (0..if cfg!(miri) { 8 } else { 14 }).fold(ints(&[1]), |p, k| {
            &p * &Polynomial::new(vec![Number::from(-3), Number::from(1) << (3 * k) as usize])
        }),
    );
    for t in 0..size(6, 1) {
        let n = 8 + below(&mut s, size(30, 2) as u64) as usize;
        let p = Polynomial::new(coeffs(&mut s, n + 1, 2 + (t % 4)));
        if p.degree().unwrap_or(0) >= 8 {
            ps.push(p);
        }
    }
    for p in &ps {
        for &prec in precs {
            check_rounding(p, prec);
        }
    }
}

#[test]
fn long_taylor_shifts() {
    // Past 256 coefficients (512 for coefficients of more than 96 bits) the Taylor shifts are
    // divide and conquer: (x^260 − 2)(x^260 − 3) has the 4 real roots ±2^(1/260), ±3^(1/260);
    // a random polynomial of the degree 600 against continued fractions and sign changes.
    // (Under Miri the divide and conquer runs from 12 coefficients: a polynomial of the degree
    // 20 against the oracle.)
    if cfg!(miri) {
        let mut s = 0x20;
        let p = Polynomial::new(coeffs(&mut s, 14, 2));
        let q: Q = (&p).into();
        for (a, b, _) in &p.real_root_intervals_descartes() {
            assert!(a == b || sign(&q, a) * sign(&q, b) < 0);
        }
        return;
    }
    let mut a = vec![Number::from(0); 261];
    a[0] = Number::from(-2);
    a[260] = Number::from(1);
    let mut b = a.clone();
    b[0] = Number::from(-3);
    let p = &Polynomial::new(a) * &Polynomial::new(b);
    let iv = p.real_root_intervals_descartes();
    assert_eq!(iv.len(), 4);
    let q: Q = (&p).into();
    for (a, b, m) in &iv {
        assert_eq!(*m, 1);
        assert!(sign(&q, a) * sign(&q, b) < 0);
    }
    assert_eq!(p.count_real_roots(), 4);
    let r = p.real_roots(80);
    let two = BigFloat::from(2).nth_root_round(260, 80, Round::Nearest).0;
    assert_eq!(r[2].0, two);
    let mut s = 0x600;
    let p = Polynomial::new(coeffs(&mut s, 601, 4));
    let d = p.real_root_intervals_descartes();
    assert_eq!(d.len(), p.real_root_intervals_cf().len());
    assert_eq!(d.len(), p.count_real_roots());
    let q: Q = (&p).into();
    for (a, b, _) in &d {
        if a != b {
            assert!(sign(&q, a) * sign(&q, b) < 0);
        }
    }
}

#[test]
fn collapsing_sturm_sequences() {
    // xⁿ plus a polynomial of a low degree: the count takes Sturm's sequence, whose first
    // remainder has that low degree; against the oracle (and the isolations).
    let cases: &[(usize, i64)] = if cfg!(miri) {
        &[(13, 3)]
    } else {
        &[
            (13, 3),
            (16, 1000),
            (24, 1 << 20),
            (31, 7),
            (40, 1 << 30),
            (60, 1 << 40),
        ]
    };
    for &(n, a) in cases {
        let p = mignotte(n, a);
        let c = p.count_real_roots();
        assert_eq!(c, p.real_root_intervals_cf().len(), "{n} {a}");
        if n <= 24 {
            assert_eq!(c, count_all(&(&p).into()), "{n} {a}");
        }
    }
    let mut s = 0x57;
    for _ in 0..size(40, 1) {
        let n = 13 + below(&mut s, size(40, 1) as u64) as usize;
        let mut c = coeffs(&mut s, 4, size(30, 4));
        c.resize(n, Number::from(0));
        c.push(
            Number::from(1 + below(&mut s, 3) as i64)
                * Number::from(if below(&mut s, 2) == 0 { 1 } else { -1 }),
        );
        let p = Polynomial::new(c);
        if p.degree().unwrap_or(0) == 0 {
            continue;
        }
        let q: Q = (&p).into();
        assert_eq!(p.count_real_roots(), count_all(&q), "{p}");
    }
}

#[test]
fn the_default_method_by_input() {
    // Roots far apart in a long interval (Wilkinson's) take continued fractions, very close ones
    // (Mignotte's) Descartes's rule down to a depth and continued fractions from there: the same
    // intervals as the oracle's, and the same counts as both methods.
    let ws: &[i64] = if cfg!(miri) { &[9] } else { &[9, 20, 35] };
    for &n in ws {
        let p = wilkinson(n);
        let iv = p.real_root_intervals();
        if n <= 20 {
            check(&p, &iv);
        }
        assert_eq!(iv.len(), n as usize);
    }
    let ms: &[(usize, i64)] = if cfg!(miri) {
        &[(8, 1 << 20)]
    } else {
        &[
            (8, 1 << 20),
            (11, 100),
            (21, 1000),
            (21, 1 << 12),
            (40, 1 << 30),
        ]
    };
    for &(n, a) in ms {
        let p = mignotte(n, a);
        let iv = p.real_root_intervals();
        assert_eq!(iv.len(), p.real_root_intervals_descartes().len());
        if n <= 21 {
            check(&p, &iv);
        }
        // Every interval (those of the deep nodes too) holds a sign change.
        let q: Q = (&p).into();
        for (lo, hi, _) in &iv {
            assert!(lo < hi || q.eval(lo).is_zero(), "{n} {a}: ({lo}, {hi})");
            if lo < hi {
                assert!(sign(&q, lo) * sign(&q, hi) < 0, "{n} {a}: ({lo}, {hi})");
            }
        }
        let r = p.real_roots(64);
        assert_eq!(r.len(), iv.len());
    }
}

/// `2^k`.
fn two(k: usize) -> Number {
    Number::from(1) << k
}

#[test]
fn tiny_roots_among_huge_coefficients() {
    // Fuzz finding F5: x^21 + 2^1528·x^20 − 2^1992·x^8 + 2^1217·x^3 − 2^1347·x + 1 panicked in
    // every real_roots: a Newton iterate exactly 0 (the root 2^−1348) took zero limbs of
    // fraction with no magnitude under them. (Under Miri the same shape at the degree 6 and a
    // twentieth of the exponents.)
    let p = if cfg!(miri) {
        let mut c = vec![Number::from(0); 7];
        c[6] = Number::from(1);
        c[5] = two(76);
        c[3] = -two(99);
        c[2] = two(60);
        c[1] = -two(67);
        c[0] = Number::from(1);
        Polynomial::new(c)
    } else {
        let mut c = vec![Number::from(0); 22];
        c[21] = Number::from(1);
        c[20] = two(1528);
        c[8] = -two(1992);
        c[3] = two(1217);
        c[1] = -two(1347);
        c[0] = Number::from(1);
        Polynomial::new(c)
    };
    let precs: &[u64] = if cfg!(miri) { &[8] } else { &[1, 53, 200] };
    for &prec in precs {
        assert_eq!(p.real_roots(prec).len(), p.real_root_intervals().len());
    }
    if !cfg!(miri) {
        check_rounding(&p, 53);
    }
}

#[test]
fn close_pairs_at_high_precision() {
    // Fuzz finding F8: Newton's method converges only linearly towards Mignotte's close pair;
    // n = 60, a = 243 at 2000 bits took 12.7 s (now 20 ms), n = 113, a = 3 at 1000 bits 49 s.
    let cases: &[(usize, i64, u64)] = if cfg!(miri) {
        &[(8, 50, 300)]
    } else {
        &[
            (60, 243, 2000),
            (113, 3, 1000),
            (80, 100_000, 3000),
            (30, 1000, 600),
        ]
    };
    for &(n, a, prec) in cases {
        let p = mignotte(n, a);
        let r = p.real_roots(prec);
        assert_eq!(r.len(), p.count_real_roots());
        if n <= 30 {
            check_rounding(&p, prec);
        }
        // The pair near 1/a: two values, apart.
        let inv = BigFloat::from_rational(&Rational::new(1.into(), a.into()), 64, Round::Nearest).0;
        let near: Vec<&BigFloat> = r
            .iter()
            .map(|(x, _)| x)
            .filter(|x| {
                let d = x.sub_round(&inv, 64, Round::Nearest).0.abs();
                d < inv.mul_round(&BigFloat::from(0.01), 64, Round::Up).0
            })
            .collect();
        assert_eq!(near.len(), 2, "{n} {a}");
        assert!(near[0] != near[1]);
    }
}

#[test]
fn roots_next_to_powers_of_two() {
    // Fuzz finding F9: x^3 − 2^1066·x^2 − 2^1318·x − 2^1437 (roots near −2^252, −2^119 and
    // 2^1066) and x^5 − (2^2485 + 1)x^4 − 2^2623·x^3 − (2^2467 − 1)x + 1 took minutes: their
    // isolating intervals span hundreds of binades, which Newton's method from the middle
    // crossed a third at a time. (Under Miri the exponents are a twentieth, one precision.)
    let k = |e: usize| if cfg!(miri) { e / 20 } else { e };
    let p = Polynomial::new(vec![
        -two(k(1437)),
        -two(k(1318)),
        -two(k(1066)),
        Number::from(1),
    ]);
    let r = p.real_roots(if cfg!(miri) { 8 } else { 53 });
    assert_eq!(r.len(), 3);
    if !cfg!(miri) {
        check_rounding(&p, 53);
    }
    let q = Polynomial::new(vec![
        Number::from(1),
        -(two(k(2467)) - Number::from(1)),
        Number::from(0),
        -two(k(2623)),
        -(two(k(2485)) + Number::from(1)),
        Number::from(1),
    ]);
    let precs: &[u64] = if cfg!(miri) { &[8] } else { &[8, 53] };
    for &prec in precs {
        assert_eq!(q.real_roots(prec).len(), q.real_root_intervals().len());
    }
    if !cfg!(miri) {
        check_rounding(&q, 8);
    }
}
