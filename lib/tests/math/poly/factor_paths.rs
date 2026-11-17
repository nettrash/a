//! The paths of the factorisation modulo primes that the choice by size takes: Cantor and
//! Zassenhaus's method against Berlekamp's in small characteristic (where `factor` takes
//! Berlekamp's), Berlekamp's splitting by minimal polynomials and their roots (with values that
//! collide modulo a small prime), the products and remainders modulo `f` across their crossovers
//! (one pass for a short `f`, the schoolbook division, Newton's), and the divisions modulo a
//! prime of two limbs past the Newton crossover.

use super::common::*;
use a::math::numbers::{ByteOrder, Number};
use a::math::poly::ModPolynomial;

/// The polynomial of the words `c` modulo `p`.
fn wp(c: &[u64], p: u64) -> ModPolynomial {
    ModPolynomial::from_u64(c, p)
}

/// A random monic polynomial of degree `n` modulo the word `p`.
fn monic(s: &mut u64, p: u64, n: usize) -> ModPolynomial {
    let mut c: Vec<u64> = (0..n).map(|_| below(s, p)).collect();
    c.push(1);
    wp(&c, p)
}

/// `l·Π gᵢ^eᵢ`.
fn product(n: &Number, l: &Number, fs: &[(ModPolynomial, u32)]) -> ModPolynomial {
    let mut r = ModPolynomial::one(n).scale(l);
    for (g, e) in fs {
        r = &r * &g.pow(u64::from(*e));
    }
    r
}

/// The factorisation by the public steps of Cantor and Zassenhaus's method: the square-free
/// parts, their distinct-degree factorisation, each part split by equal degree; sorted as
/// `factor` sorts.
fn by_steps(f: &ModPolynomial) -> (Number, Vec<(ModPolynomial, u32)>) {
    let (l, parts) = f.squarefree();
    let mut out = Vec::new();
    for (g, e) in parts {
        for (h, d) in g.distinct_degree_factor() {
            for q in h.equal_degree_factor(d) {
                out.push((q, e));
            }
        }
    }
    out.sort_by(|a, b| {
        a.0.degree().cmp(&b.0.degree()).then_with(|| {
            let (x, y) = (a.0.coeffs(), b.0.coeffs());
            x.iter().rev().cmp(y.iter().rev())
        })
    });
    (l, out)
}

#[test]
fn cantor_zassenhaus_agrees_with_berlekamp_in_small_characteristic() {
    let mut s = 0x1357_9BDF_2468_ACE0;
    for &p in &[2u64, 3, 5, 7] {
        for &n in &[1usize, 2, 5, 12, 30, 61, 120] {
            if cfg!(miri) && n > 12 {
                continue;
            }
            for _ in 0..size(3, 1) {
                let f = &monic(&mut s, p, n) * &monic(&mut s, p, n / 3 + 1);
                let (l, fs) = f.factor();
                assert_eq!(product(f.modulus(), &l, &fs), f, "{f}");
                assert_eq!(by_steps(&f), (l.clone(), fs.clone()), "{f}");
                assert_eq!(f.factor_berlekamp(), (l, fs), "{f}");
            }
        }
    }
}

#[test]
fn berlekamp_by_minimal_polynomials() {
    // Products of distinct linear factors (the kernel's dimension is their number) and random
    // factors: modulo 13 and 17 the values of a random kernel element collide often, so the
    // splitting takes several rounds; modulo large primes one round.
    let mut s = 0x2468_ACE0_1357_9BDF;
    let big = [
        Number::from(13u64),
        Number::from(17u64),
        Number::from(101u64),
        Number::from(65_537u64),
        Number::from(18_446_744_073_709_551_557u64),
        Number::from_str_radix("18446744073709551629", 10, ByteOrder::LittleEndian).unwrap(),
        Number::from_str_radix(
            "170141183460469231731687303715884105727",
            10,
            ByteOrder::LittleEndian,
        )
        .unwrap(),
    ];
    for n in &big {
        let roots = size(12, 3);
        let mut f = ModPolynomial::one(n);
        let mut used: Vec<Number> = Vec::new();
        while used.len() < roots {
            let c = Number::from(below(&mut s, 1 << 40)).rem_euclid(n);
            if used.contains(&c) {
                continue;
            }
            f = &f * &ModPolynomial::new(&[-&c, Number::from(1u8)], n);
            used.push(c);
        }
        for _ in 0..size(3, 1) {
            let g = &f * &mpoly(&mut s, size(20, 4), n);
            if g.is_zero() {
                continue;
            }
            let (l, fs) = g.factor();
            assert_eq!(product(n, &l, &fs), g, "{g}");
            assert_eq!(g.factor_berlekamp(), (l, fs.clone()), "{g}");
            used.sort();
            for c in &used {
                assert!(
                    g.roots().iter().any(|(r, _)| r == c),
                    "{c} a root of {g} modulo {n}"
                );
            }
            for (h, _) in &fs {
                assert!(h.is_irreducible());
            }
        }
    }
}

#[test]
fn products_modulo_f_across_the_crossovers() {
    // Degrees around the one-pass products (32 coefficients of f for 32-bit primes, 48 for
    // squares and 80 for products above), the schoolbook division (150) and the Newton division
    // beyond, at moduli of 30, 50, 63 and 64 bits and a small one (the private Newton path).
    let mut s = 0x0BAD_F00D_DEAD_BEEF;
    let ps = [
        1_073_741_789u64,
        1_125_899_906_842_597,
        9_223_372_036_854_775_783,
        18_446_744_073_709_551_557,
        3,
        251,
    ];
    let ns: &[usize] = if cfg!(miri) {
        &[2, 5, 6, 7, 9]
    } else {
        &[
            2, 15, 31, 32, 33, 47, 48, 49, 79, 80, 81, 149, 150, 151, 170,
        ]
    };
    for &p in &ps {
        for &n in ns {
            let f = monic(&mut s, p, n);
            // x^p mod f by the factorisation's kernels against the core's pow_mod.
            let x = ModPolynomial::x(&Number::from(p));
            let want = x.pow_mod(&Number::from(p), &f);
            let roots = f.roots();
            for (r, _) in &roots {
                assert!(f.eval(r).is_zero());
            }
            // The roots are those of gcd(f, x^p - x), whose degree is their number when f is
            // square-free.
            let (_, sq) = f.squarefree();
            if sq.len() == 1 && sq[0].1 == 1 {
                let g = f.gcd(&(&want - &x));
                assert_eq!(g.degree(), Some(roots.len()), "{f}");
            }
            let (l, fs) = f.factor();
            assert_eq!(product(f.modulus(), &l, &fs), f);
            for (h, _) in &fs {
                assert!(h.is_irreducible(), "{h}");
            }
            assert_eq!(
                fs.iter().all(|(h, _)| h.degree() == Some(n)),
                f.is_irreducible()
            );
        }
    }
}

#[test]
fn divisions_modulo_two_limbs_past_the_newton_crossover() {
    // f = g·h² modulo 2^127 - 1 with g and h of 30 to 45 coefficients: the square-free
    // decomposition divides by its greatest common divisors, quotients and divisors past 24
    // coefficients, by Newton's method.
    if cfg!(miri) {
        return;
    }
    let n = Number::from_str_radix(
        "170141183460469231731687303715884105727",
        10,
        ByteOrder::LittleEndian,
    )
    .unwrap();
    let mut s = 0x1111_2222_3333_4444;
    for _ in 0..3 {
        let g = ModPolynomial::new(&coeffs(&mut s, 45, 16), &n).to_monic();
        let h = ModPolynomial::new(&coeffs(&mut s, 30, 16), &n).to_monic();
        if g.degree() < Some(30) || h.degree() < Some(25) {
            continue;
        }
        let f = &g * &(&h * &h);
        let (l, sq) = f.squarefree();
        assert_eq!(product(&n, &l, &sq), f);
        for (a, _) in &sq {
            assert!(a.gcd(&a.derivative()).is_one(), "{a} square-free");
        }
        let (l2, fs) = f.factor();
        assert_eq!(product(&n, &l2, &fs), f);
    }
}

#[test]
fn roots_with_multiplicities_by_synthetic_division() {
    // Modulo primes up to 64, and up to the degree, every residue is tried and each root divided
    // out as often as it divides: products of powers of linear factors and a random factor,
    // against the factors of degree 1 of the factorisation.
    let mut s = 0x5A5A_A5A5_0F0F_F0F0;
    for &p in &[2u64, 3, 5, 61, 101] {
        for _ in 0..size(6, 2) {
            let d = (below(&mut s, 8) as usize).min(size(8, 3));
            let mut f = monic(&mut s, p, d);
            for _ in 0..size(6, 2) {
                let c = below(&mut s, p);
                let e = 1 + below(&mut s, 4);
                f = &f * &wp(&[(p - c) % p, 1], p).pow(e);
            }
            if p == 101 && !cfg!(miri) {
                // Past the degree p.
                f = &f * &monic(&mut s, p, 100);
            }
            let (_, fs) = f.factor();
            let mut want: Vec<(Number, u32)> = fs
                .iter()
                .filter(|(g, _)| g.degree() == Some(1))
                .map(|(g, e)| {
                    (
                        (&Number::from(p) - &g.coeff(0)).rem_euclid(&Number::from(p)),
                        *e,
                    )
                })
                .collect();
            want.sort();
            assert_eq!(f.roots(), want, "{f}");
        }
    }
}
