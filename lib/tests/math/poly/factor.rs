//! Factorisation modulo primes: the square-free decomposition, the distinct-degree and
//! equal-degree factorisations, the complete factorisation by both methods, Rabin's test and the
//! roots, against independent oracles — products, the exhaustive count of irreducible
//! polynomials over small fields, trial division by every polynomial of lower degree, the
//! factorisation of `x^(p^n) - x`, values at every residue — and PARI's `factormod` and
//! `polrootsmod`.

use super::common::*;
use a::math::numbers::Number;
use a::math::poly::ModPolynomial;

/// The primes of the randomised tests: those of [`primes`], under Miri one of a few bits, one
/// at the top of a word and one of two limbs.
fn test_primes() -> Vec<Number> {
    let all = primes();
    if cfg!(miri) {
        vec![all[1].clone(), all[5].clone(), all[6].clone()]
    } else {
        all
    }
}

/// The polynomial of the words `c` modulo `p`.
fn wp(c: &[u64], p: u64) -> ModPolynomial {
    ModPolynomial::from_u64(c, p)
}

/// `l·Π gᵢ^eᵢ`.
fn product(f: &ModPolynomial, l: &Number, fs: &[(ModPolynomial, u32)]) -> ModPolynomial {
    let mut r = ModPolynomial::one(f.modulus()).scale(l);
    for (g, e) in fs {
        r = &r * &g.pow(u64::from(*e));
    }
    r
}

/// Whether `g` is irreducible by Rabin's test on the core's own arithmetic (schoolbook products
/// and `pow_mod`), independent of the factorisation's kernels: `x^(p^n) ≡ x mod g` and
/// `gcd(x^(p^(n/q)) - x, g) = 1` for each prime `q` dividing `n`.
fn irreducible_by_core_rabin(g: &ModPolynomial) -> bool {
    let n = g.degree().unwrap();
    if n == 0 {
        return false;
    }
    let p = g.modulus().clone();
    let x = &ModPolynomial::x(&p) % g;
    let mut pows = vec![x.clone()];
    for i in 0..n {
        let next = pows[i].pow_mod(&p, g);
        pows.push(next);
    }
    if pows[n] != x {
        return false;
    }
    let mut m = n;
    let mut q = 2;
    while m > 1 {
        if m.is_multiple_of(q) {
            while m.is_multiple_of(q) {
                m /= q;
            }
            if !g.gcd(&(&pows[n / q] - &x)).is_one() {
                return false;
            }
        }
        q += 1;
    }
    true
}

/// Checks a complete factorisation of `f`: the product, monic factors sorted by degree and
/// coefficients, each irreducible by Ben-Or's test and, up to degree 40 (8 under Miri), by Rabin's
/// on the core's arithmetic, and distinct.
fn check_factorisation(f: &ModPolynomial, l: &Number, fs: &[(ModPolynomial, u32)]) {
    assert_eq!(&product(f, l, fs), f, "the product of the factors of {f}");
    if !f.is_zero() {
        assert_eq!(Some(l.clone()), f.leading());
    }
    for w in fs.windows(2) {
        let (a, b) = (&w[0].0, &w[1].0);
        assert!(a.degree() <= b.degree(), "sorted by degree: {a}, {b}");
        assert_ne!(a, b, "distinct factors");
    }
    for (g, e) in fs {
        assert!(*e >= 1);
        assert!(g.is_monic() && g.degree() >= Some(1), "{g}");
        assert!(g.is_irreducible(), "{g} irreducible");
        if g.degree() <= Some(size(40, 8)) && g.modulus().bits() <= 130 {
            assert!(irreducible_by_core_rabin(g), "{g} irreducible by the core");
        }
    }
}

/// Every monic polynomial of degree `n` modulo the small `p`.
fn all_monic(p: u64, n: usize) -> Vec<ModPolynomial> {
    let count = p.pow(n as u32);
    (0..count)
        .map(|mut i| {
            let mut c = Vec::with_capacity(n + 1);
            for _ in 0..n {
                c.push(i % p);
                i /= p;
            }
            c.push(1);
            wp(&c, p)
        })
        .collect()
}

/// Whether `f` is irreducible, by trial division by every monic polynomial of degree 1 to
/// `deg f / 2`.
fn irreducible_by_trial(f: &ModPolynomial, p: u64) -> bool {
    let Some(n) = f.degree() else { return false };
    if n == 0 {
        return false;
    }
    for d in 1..=n / 2 {
        for g in all_monic(p, d) {
            if (f % &g).is_zero() {
                return false;
            }
        }
    }
    true
}

/// The Möbius function.
fn mobius(mut n: u64) -> i64 {
    let mut r = 1;
    let mut q = 2;
    while q * q <= n {
        if n.is_multiple_of(q) {
            n /= q;
            if n.is_multiple_of(q) {
                return 0;
            }
            r = -r;
        }
        q += 1;
    }
    if n > 1 { -r } else { r }
}

/// The number of monic irreducible polynomials of degree `n` over `F_p`, `(1/n) Σ μ(d) p^(n/d)`.
fn irreducible_count(p: u64, n: u64) -> u64 {
    let s: i64 = (1..=n)
        .filter(|d| n.is_multiple_of(*d))
        .map(|d| mobius(d) * p.pow((n / d) as u32) as i64)
        .sum();
    s as u64 / n
}

#[test]
fn edge_cases() {
    let n = Number::from(7);
    let z = ModPolynomial::zero(&n);
    assert_eq!(z.factor(), (Number::from(0), vec![]));
    assert_eq!(z.factor_berlekamp(), (Number::from(0), vec![]));
    assert_eq!(z.squarefree(), (Number::from(0), vec![]));
    assert!(!z.is_irreducible());
    let c = wp(&[5], 7);
    assert_eq!(c.factor(), (Number::from(5), vec![]));
    assert_eq!(c.squarefree(), (Number::from(5), vec![]));
    assert!(!c.is_irreducible());
    assert!(c.roots().is_empty());
    assert!(c.distinct_degree_factor().is_empty());
    assert!(c.equal_degree_factor(3).is_empty());
    // Degree 1: 3x + 2 = 3(x + 3).
    let l = wp(&[2, 3], 7);
    assert_eq!(l.factor(), (Number::from(3), vec![(wp(&[3, 1], 7), 1)]));
    assert!(l.is_irreducible());
    assert_eq!(l.roots(), vec![(Number::from(4), 1)]);
    assert_eq!(l.distinct_degree_factor(), vec![(wp(&[3, 1], 7), 1)]);
    assert_eq!(l.equal_degree_factor(1), vec![wp(&[3, 1], 7)]);
    // x itself and its powers.
    assert_eq!(
        wp(&[0, 0, 0, 2], 3).factor(),
        (Number::from(2), vec![(wp(&[0, 1], 3), 3)])
    );
    assert_eq!(wp(&[0, 0, 0, 1], 3).roots(), vec![(Number::from(0), 3)]);
    // Modulo 2, every quadratic.
    assert_eq!(wp(&[0, 0, 1], 2).factor().1, vec![(wp(&[0, 1], 2), 2)]);
    assert_eq!(wp(&[1, 0, 1], 2).factor().1, vec![(wp(&[1, 1], 2), 2)]);
    assert_eq!(
        wp(&[0, 1, 1], 2).factor().1,
        vec![(wp(&[0, 1], 2), 1), (wp(&[1, 1], 2), 1)]
    );
    assert_eq!(wp(&[1, 1, 1], 2).factor().1, vec![(wp(&[1, 1, 1], 2), 1)]);
}

#[test]
#[should_panic(expected = "the modulus is not prime")]
fn a_composite_modulus_panics() {
    let _ = wp(&[1, 0, 1], 15).factor();
}

#[test]
#[should_panic(expected = "the modulus is not prime")]
fn a_composite_modulus_of_two_limbs_panics() {
    let n = &Number::from(u64::MAX) * &Number::from(3);
    let _ = ModPolynomial::new(&[Number::from(1), Number::from(1)], &n).is_irreducible();
}

#[test]
#[should_panic(expected = "the modulus is not prime")]
fn roots_need_a_prime() {
    let _ = wp(&[1, 0, 1], 4).roots();
}

#[test]
#[should_panic(expected = "the roots of the zero polynomial")]
fn roots_of_zero_panic() {
    let _ = ModPolynomial::zero(&Number::from(5)).roots();
}

#[test]
#[should_panic(expected = "the polynomial is not square-free")]
fn distinct_degree_needs_square_free() {
    let _ = wp(&[1, 2, 1], 5).distinct_degree_factor();
}

#[test]
#[should_panic(expected = "not a product of distinct irreducible factors")]
fn equal_degree_of_the_wrong_degree_panics() {
    // x² + 1 is irreducible modulo 7: not a product of factors of degree 1.
    let _ = wp(&[1, 0, 1], 7).equal_degree_factor(1);
}

#[test]
#[should_panic(expected = "not a product of distinct irreducible factors")]
fn equal_degree_of_a_degree_not_dividing_panics() {
    let _ = wp(&[1, 0, 0, 1, 1], 7).equal_degree_factor(3);
}

#[test]
fn rabin_agrees_with_trial_division_exhaustively() {
    // Every monic polynomial of degree up to 7 modulo 2, up to 4 modulo 3 and up to 3 modulo 5,
    // and the counts of irreducible ones.
    for &(p, top) in &[(2u64, size(7, 3)), (3, size(4, 2)), (5, size(3, 2))] {
        for n in 1..=top {
            let mut count = 0;
            for f in all_monic(p, n) {
                let irr = f.is_irreducible();
                assert_eq!(irr, irreducible_by_trial(&f, p), "{f}");
                if irr {
                    count += 1;
                    let (_, fs) = f.factor();
                    assert_eq!(fs, vec![(f.clone(), 1)], "{f}");
                    let (_, fb) = f.factor_berlekamp();
                    assert_eq!(fb, fs);
                }
                count += 0;
            }
            assert_eq!(count, irreducible_count(p, n as u64), "p = {p}, n = {n}");
        }
    }
}

#[test]
fn x_to_the_p_to_the_n_minus_x() {
    // The product of every monic irreducible polynomial of a degree dividing n.
    for &(p, n) in &[(2u64, size(8, 4)), (3, size(4, 2)), (5, size(3, 2)), (7, 2)] {
        let q = p.pow(n as u32) as usize;
        let mut c = vec![0u64; q + 1];
        c[q] = 1;
        c[1] = p - 1;
        let f = wp(&c, p);
        let (l, fs) = f.factor();
        check_factorisation(&f, &l, &fs);
        let expected: u64 = (1..=n as u64)
            .filter(|d| (n as u64).is_multiple_of(*d))
            .map(|d| irreducible_count(p, d))
            .sum();
        assert_eq!(fs.len() as u64, expected, "p = {p}, n = {n}");
        assert!(fs.iter().all(|(_, e)| *e == 1));
        let ddf = f.distinct_degree_factor();
        assert_eq!(
            ddf.iter().map(|(_, d)| *d).collect::<Vec<_>>(),
            (1..=n as u32)
                .filter(|d| (n as u32).is_multiple_of(*d))
                .collect::<Vec<_>>()
        );
        for (g, d) in &ddf {
            let parts = g.equal_degree_factor(*d);
            assert_eq!(parts.len() as u64, irreducible_count(p, u64::from(*d)));
            assert!(parts.iter().all(|h| h.degree() == Some(*d as usize)));
        }
        if q <= 64 {
            assert_eq!(f.factor_berlekamp(), (l, fs));
        }
        // Every residue is a root, once.
        let r = f.roots();
        assert_eq!(r.len() as u64, p);
        assert!(
            r.iter()
                .enumerate()
                .all(|(i, (x, e))| u64::try_from(x) == Ok(i as u64) && *e == 1)
        );
    }
}

#[test]
fn random_factorisations_round_trip() {
    let mut s = 0x0F0E_0D0C_0B0A_0908;
    for n in test_primes() {
        let top = if n.bits() > 64 {
            size(14, 3)
        } else {
            size(40, 5)
        };
        for _ in 0..size(8, 2) {
            let f = mpoly(&mut s, top, &n);
            let (l, fs) = f.factor();
            check_factorisation(&f, &l, &fs);
            assert_eq!(f.factor_berlekamp(), (l.clone(), fs.clone()), "{f}");
            // The square-free decomposition is the factorisation grouped by multiplicity.
            let (l2, sq) = f.squarefree();
            assert_eq!(product(&f, &l2, &sq), f);
            for (g, e) in &sq {
                let want = fs
                    .iter()
                    .filter(|(_, e2)| e2 == e)
                    .fold(ModPolynomial::one(&n), |acc, (h, _)| &acc * h);
                assert_eq!(&want, g, "the part of multiplicity {e} of {f}");
            }
            // The roots are the factors of degree 1.
            if !f.is_zero() {
                let want: Vec<(Number, u32)> = {
                    let mut v: Vec<(Number, u32)> = fs
                        .iter()
                        .filter(|(g, _)| g.degree() == Some(1))
                        .map(|(g, e)| ((&n - &g.coeff(0)).rem_euclid(&n), *e))
                        .collect();
                    v.sort();
                    v
                };
                assert_eq!(f.roots(), want, "{f}");
            }
        }
    }
}

#[test]
fn products_of_known_irreducibles() {
    // Products of random irreducible polynomials, found by Rabin's test and checked by trial
    // division modulo small primes: the factorisation finds them all with their multiplicities.
    let mut s = 0x1357_9BDF_2468_ACE0;
    for &p in &[2u64, 3, 5, 65537, 18_446_744_073_709_551_557] {
        let n = Number::from(p);
        for _ in 0..size(6, 1) {
            let mut want: Vec<(ModPolynomial, u32)> = Vec::new();
            for _ in 0..size(4, 2) {
                let d = 1 + below(&mut s, size(6, 3) as u64) as usize;
                let g = loop {
                    let mut c: Vec<u64> = (0..d).map(|_| below(&mut s, p)).collect();
                    c.push(1);
                    let g = wp(&c, p);
                    if g.is_irreducible() {
                        break g;
                    }
                };
                if p < 10 {
                    assert!(irreducible_by_trial(&g, p), "{g}");
                }
                let e = 1 + below(&mut s, 3) as u32;
                match want.iter_mut().find(|(h, _)| *h == g) {
                    Some(x) => x.1 += e,
                    None => want.push((g, e)),
                }
            }
            let lc = Number::from(1 + below(&mut s, p - 1));
            let f = product(&ModPolynomial::one(&n), &lc, &want);
            let (l, fs) = f.factor();
            check_factorisation(&f, &l, &fs);
            want.sort_by(|a, b| {
                a.0.degree()
                    .cmp(&b.0.degree())
                    .then_with(|| a.0.coeffs().iter().rev().cmp(b.0.coeffs().iter().rev()))
            });
            assert_eq!(fs, want, "{f}");
            assert_eq!(f.factor_berlekamp().1, want);
        }
    }
}

#[test]
fn equal_degree_splits_products_of_one_degree() {
    let mut s = 0xDEAD_BEEF_1234_5678;
    for n in test_primes() {
        for d in 1..=size(4, 2) {
            let mut gs: Vec<ModPolynomial> = Vec::new();
            // Modulo a small prime there are only so many.
            let count = match u64::try_from(&n) {
                Ok(p) if p < 100 => irreducible_count(p, d as u64) as usize,
                _ => usize::MAX,
            };
            while gs.len() < size(4, 2).min(count) {
                let mut c = coeffs(&mut s, d, 20);
                c.push(Number::from(1));
                let g = ModPolynomial::new(&c, &n);
                if g.degree() == Some(d) && g.is_irreducible() && !gs.contains(&g) {
                    gs.push(g);
                }
            }
            let f = gs.iter().fold(ModPolynomial::one(&n), |a, g| &a * g);
            let mut got = f.equal_degree_factor(d as u32);
            gs.sort_by(|a, b| a.coeffs().iter().rev().cmp(b.coeffs().iter().rev()));
            got.sort_by(|a, b| a.coeffs().iter().rev().cmp(b.coeffs().iter().rev()));
            assert_eq!(got, gs, "modulo {n}, degree {d}");
            let ddf = f.distinct_degree_factor();
            assert_eq!(ddf, vec![(f.clone(), d as u32)]);
        }
    }
}

#[test]
fn distinct_degree_parts_multiply_back() {
    let mut s = 0x7777_3333_1111_5555;
    for n in test_primes() {
        let top = if n.bits() > 64 {
            size(16, 3)
        } else {
            size(60, 5)
        };
        for _ in 0..size(6, 1) {
            let f = mpoly(&mut s, top, &n);
            if f.is_zero() || f.degree() == Some(0) {
                continue;
            }
            let (_, sq) = f.squarefree();
            for (g, _) in sq {
                let parts = g.distinct_degree_factor();
                let prod = parts
                    .iter()
                    .fold(ModPolynomial::one(&n), |a, (h, _)| &a * h);
                assert_eq!(prod, g.to_monic());
                for (h, d) in &parts {
                    let (_, fs) = h.factor();
                    assert!(
                        fs.iter()
                            .all(|(q, e)| q.degree() == Some(*d as usize) && *e == 1)
                    );
                }
                for w in parts.windows(2) {
                    assert!(w[0].1 < w[1].1);
                }
            }
        }
    }
}

#[test]
fn long_polynomials() {
    // Past the crossovers of the products, the remainders and the baby-step giant-step method.
    if cfg!(miri) {
        return;
    }
    let mut s = 0xABCD_EF01_2345_6789;
    for &(p, len) in &[
        (2u64, 300usize),
        (3, 250),
        (65537, 200),
        (18_446_744_073_709_551_557, 180),
        (1_000_000_007, 400),
    ] {
        let c: Vec<u64> = (0..len).map(|_| below(&mut s, p)).collect();
        let mut c2: Vec<u64> = (0..len / 3).map(|_| below(&mut s, p)).collect();
        c2.push(1);
        // A random polynomial times the square of another.
        let g = wp(&c2, p);
        let f = &(&wp(&c, p) * &g) * &g;
        let (l, fs) = f.factor();
        check_factorisation(&f, &l, &fs);
        let ddf_total: usize = f
            .squarefree()
            .1
            .iter()
            .map(|(h, _)| {
                h.distinct_degree_factor()
                    .iter()
                    .map(|(q, _)| q.degree().unwrap())
                    .sum::<usize>()
            })
            .sum();
        assert_eq!(
            ddf_total,
            f.squarefree()
                .1
                .iter()
                .map(|(h, _)| h.degree().unwrap())
                .sum::<usize>()
        );
        if len <= 250 {
            assert_eq!(f.factor_berlekamp(), (l, fs));
        }
    }
    // Modulo a prime of two limbs past the crossover of the Newton division, the quotients in
    // chunks: a product of three polynomials, one squared.
    let n = a::math::numbers::Number::from_str_radix(
        "170141183460469231731687303715884105727",
        10,
        a::math::numbers::ByteOrder::LittleEndian,
    )
    .unwrap();
    let g = ModPolynomial::new(&coeffs(&mut s, 31, 16), &n);
    let h = ModPolynomial::new(&coeffs(&mut s, 40, 16), &n);
    let k = ModPolynomial::new(&coeffs(&mut s, 3, 16), &n);
    let f = &(&g * &h) * &(&k * &k);
    let (l, fs) = f.factor();
    check_factorisation(&f, &l, &fs);
    let (q, r) = f.div_rem(&h);
    assert!(r.is_zero() && &q * &h == f);
}

#[test]
fn p_th_powers_in_small_characteristic() {
    let mut s = 0x2222_4444_6666_8888;
    for &p in &[2u64, 3, 5] {
        for _ in 0..size(10, 2) {
            let g = mpoly(&mut s, 6, &Number::from(p));
            let h = mpoly(&mut s, 5, &Number::from(p));
            if g.degree().unwrap_or(0) == 0 || h.is_zero() {
                continue;
            }
            // g(x^p)·h, g(x^p) = g̃^p with g̃ = g.
            let gp = g.compose(&wp(
                &{
                    let mut c = vec![0; p as usize + 1];
                    c[p as usize] = 1;
                    c
                },
                p,
            ));
            let f = &gp.pow(1 + below(&mut s, 2)) * &h;
            let (l, fs) = f.factor();
            check_factorisation(&f, &l, &fs);
            assert_eq!(f.factor_berlekamp(), (l.clone(), fs.clone()));
            let (l2, sq) = f.squarefree();
            assert_eq!(product(&f, &l2, &sq), f);
            for w in sq.windows(2) {
                assert!(w[0].0.gcd(&w[1].0).is_one());
            }
            for (q, _) in &sq {
                assert!(q.gcd(&q.derivative()).is_one(), "{q} square-free");
            }
        }
    }
}

#[test]
fn roots_by_values_at_every_residue() {
    let mut s = 0x1111_2222_3333_4444;
    for &p in &[2u64, 3, 5, 7, 13, 101, 257] {
        let n = Number::from(p);
        for _ in 0..size(12, 2) {
            let f = mpoly(&mut s, 12, &n);
            if f.is_zero() {
                continue;
            }
            let got = f.roots();
            let mut want = Vec::new();
            for x in 0..p {
                let mut g = f.clone();
                let r = wp(&[(p - x) % p, 1], p);
                let mut e = 0;
                while !g.is_zero() && g.eval(&Number::from(x)).is_zero() {
                    g = &g / &r;
                    e += 1;
                }
                if e > 0 {
                    want.push((Number::from(x), e));
                }
            }
            assert_eq!(got, want, "{f}");
        }
    }
}

#[test]
fn roots_of_products_of_linear_factors_modulo_large_primes() {
    let mut s = 0x5A5A_A5A5_0F0F_F0F0;
    for n in test_primes().into_iter().filter(|n| n.bits() > 20) {
        for _ in 0..size(4, 1) {
            let mut want: Vec<(Number, u32)> = Vec::new();
            let mut f = ModPolynomial::one(&n);
            for _ in 0..size(6, 2) {
                let r = number(&mut s, 20).rem_euclid(&n);
                let e = 1 + below(&mut s, 3) as u32;
                f = &f * &ModPolynomial::new(&[-&r, Number::from(1)], &n).pow(u64::from(e));
                match want.iter_mut().find(|(x, _)| *x == r) {
                    Some(x) => x.1 += e,
                    None => want.push((r, e)),
                }
            }
            // And a factor without roots: an irreducible quadratic.
            let q = loop {
                let c = vec![number(&mut s, 20), number(&mut s, 20), Number::from(1)];
                let q = ModPolynomial::new(&c, &n);
                if q.is_irreducible() {
                    break q;
                }
            };
            f = &f * &q;
            want.sort();
            assert_eq!(f.roots(), want, "modulo {n}");
        }
    }
}

/// The integers of the decimal list `s`.
fn parse_list(s: &str) -> Vec<Number> {
    if s.is_empty() {
        return Vec::new();
    }
    s.split(',')
        .map(|x| Number::from_str_radix(x, 10, a::math::numbers::ByteOrder::LittleEndian).unwrap())
        .collect()
}

#[test]
fn pari_factormod_and_polrootsmod() {
    let data = include_str!("data/factor.txt");
    let mut cases = 0;
    for line in data.lines() {
        let f: Vec<&str> = line.split(';').collect();
        match f[0] {
            "fp" => {
                if cfg!(miri) && cases > 8 {
                    continue;
                }
                let p = &parse_list(f[1])[0];
                let poly = ModPolynomial::new(&parse_list(f[2]), p);
                let lc = &parse_list(f[3])[0];
                let mut want: Vec<(ModPolynomial, u32)> = f[4]
                    .split('|')
                    .filter(|x| !x.is_empty())
                    .map(|t| {
                        let (e, c) = t.split_once(':').unwrap();
                        (ModPolynomial::new(&parse_list(c), p), e.parse().unwrap())
                    })
                    .collect();
                want.sort_by(|a, b| {
                    a.0.degree()
                        .cmp(&b.0.degree())
                        .then_with(|| a.0.coeffs().iter().rev().cmp(b.0.coeffs().iter().rev()))
                });
                let (l, fs) = poly.factor();
                assert_eq!((&l, &fs), (lc, &want), "{line}");
                if poly.degree() <= Some(16) {
                    assert_eq!(poly.factor_berlekamp(), (l, fs), "{line}");
                }
                cases += 1;
            }
            "rt" => {
                if cfg!(miri) && cases > 12 {
                    continue;
                }
                let p = &parse_list(f[1])[0];
                let poly = ModPolynomial::new(&parse_list(f[2]), p);
                let want: Vec<(Number, u32)> = f[3]
                    .split('|')
                    .filter(|x| !x.is_empty())
                    .map(|t| {
                        let (r, e) = t.split_once(':').unwrap();
                        (parse_list(r)[0].clone(), e.parse().unwrap())
                    })
                    .collect();
                assert_eq!(poly.roots(), want, "{line}");
                cases += 1;
            }
            _ => {}
        }
    }
    assert!(cases > 10);
}
